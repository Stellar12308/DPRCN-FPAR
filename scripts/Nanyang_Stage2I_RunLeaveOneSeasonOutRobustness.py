from __future__ import annotations

import importlib.util
import json
import logging
import platform
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn
from torch.utils.data import DataLoader


SCRIPT_PATH = Path(__file__).resolve()
STAGE2_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(STAGE2_ROOT))

from dprcn.losses import build_loss  # noqa: E402
from dprcn.model import build_model  # noqa: E402
from dprcn.reproducibility import seed_everything, sha256_file  # noqa: E402
from dprcn.season_fold_dataset import SeasonFoldPatchDataset  # noqa: E402


CONFIG_PATH = STAGE2_ROOT / "configs" / "Nanyang_Stage2I_LeaveOneSeasonOut_Config_v1.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法载入模块：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def season_from_doy(doy: int) -> str:
    d = int(doy)
    if d < 60 or d >= 340:
        return "Winter"
    if d < 150:
        return "Spring"
    if d < 245:
        return "Summer"
    return "Autumn"


class FixedA3InputMask(nn.Module):
    def __init__(self, model: nn.Module, active_indices: list[int]) -> None:
        super().__init__()
        self.model = model
        mask = torch.zeros(1, 9, 1, 1, dtype=torch.float32)
        mask[:, active_indices] = 1.0
        self.register_buffer("input_mask", mask)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(inputs * self.input_mask)


def worker_seed(worker_id: int) -> None:
    value = torch.initial_seed() % (2**32)
    np.random.seed(value)
    random.seed(value)


def fit_normalization(stage2b, manifest: pd.DataFrame, held_out: str, config: dict, fold_dir: Path) -> Path:
    training = manifest.loc[
        manifest["season_en"].ne(held_out) & manifest["eligible_for_use"].eq(1)
    ].copy()
    states = {
        channel: {"count": 0, "mean": 0.0, "m2": 0.0, "minimum": np.inf, "maximum": -np.inf}
        for channel in stage2b.CHANNELS
    }
    audits = []
    stage2g = load_json(resolve(Path(config["project_root"]), config["stage2g_config"]))
    for (date, doy), rows in training.groupby(["date", "stored_DOY"], sort=True):
        arrays, mask, metadata = stage2b.scene_arrays(
            rows,
            [float(value) for value in config["allowed_weights"]],
            float(config["weight_tolerance"]),
            stage2g["terrain"],
        )
        if not mask.any():
            raise RuntimeError(f"{held_out}折训练日期{date}无有效归一化像元。")
        record = {"date": date, "stored_DOY": int(doy), "season": season_from_doy(doy), **metadata}
        for channel in stage2b.CHANNELS:
            values = stage2b.transform_channel(channel, arrays[channel], stage2g["normalization"])[mask]
            stage2b.update_moments(states[channel], values)
            record[f"{channel}_mean_transformed"] = float(np.mean(values))
        audits.append(record)
    parameters = {}
    counts = set()
    for channel, state in states.items():
        count = int(state["count"])
        std = float(np.sqrt(max(float(state["m2"]) / count, 0.0)))
        if count < 2 or not np.isfinite(std) or std <= float(config["minimum_standard_deviation"]):
            raise RuntimeError(f"{held_out}折{channel}统计无效。")
        counts.add(count)
        parameters[channel] = {
            "source_transform": stage2g["normalization"][channel],
            "normalization": "z_score",
            "count": count,
            "mean": float(state["mean"]),
            "std_population": std,
            "minimum": float(state["minimum"]),
            "maximum": float(state["maximum"]),
        }
    if len(counts) != 1:
        raise RuntimeError(f"{held_out}折各通道统计像元数不一致。")
    audit_path = fold_dir / "NormalizationSceneAudit.csv"
    pd.DataFrame(audits).sort_values("stored_DOY").to_csv(audit_path, index=False, encoding="utf-8-sig")
    document = {
        "schema_version": "Stage2I_SeasonFoldNormalization_v1",
        "held_out_season": held_out,
        "fit_role": "all_dates_outside_held_out_season",
        "fit_scene_count": len(audits),
        "fit_patch_count": len(training),
        "fit_unique_pixel_count": next(iter(counts)),
        "held_out_used_for_fit": False,
        "overlapping_patch_duplicates_counted": False,
        "terrain_policy": stage2g["terrain"],
        "channel_order": stage2b.CHANNELS,
        "parameters": parameters,
    }
    path = fold_dir / "Normalization.json"
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def train_fold(dataset, normalization: dict, stage2g: dict, config: dict, fold_dir: Path, held_out: str):
    seed = int(config["random_seed"])
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["loader"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["loader"]["num_workers"]),
        pin_memory=bool(config["loader"]["pin_memory"]),
        worker_init_fn=worker_seed,
        generator=generator,
        drop_last=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FixedA3InputMask(
        build_model(stage2g, normalization), config["active_channel_indices"]
    ).to(device)
    criterion = build_loss(stage2g)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(stage2g["optimizer"]["learning_rate"]),
        weight_decay=float(stage2g["optimizer"]["weight_decay"]),
    )
    history = []
    for epoch in range(1, int(config["fixed_epochs"]) + 1):
        model.train()
        totals = defaultdict(float)
        batches = 0
        for raw in loader:
            inputs = raw["input"].to(device)
            target = raw["target"].to(device)
            mask = raw["valid_mask"].to(device)
            weight = raw["weight"].to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = criterion(model(inputs)["prediction"], target, mask, weight)
            losses["total"].backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(stage2g["optimizer"]["gradient_clip_norm"]))
            if not torch.isfinite(grad):
                raise RuntimeError(f"{held_out}折出现非有限梯度。")
            optimizer.step()
            for name in ["total", "pixel", "gradient"]:
                totals[name] += float(losses[name].detach().cpu())
            batches += 1
        record = {
            "epoch": epoch,
            "train_total_loss": totals["total"] / batches,
            "train_pixel_loss": totals["pixel"] / batches,
            "train_gradient_loss": totals["gradient"] / batches,
            "batch_count": batches,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(fold_dir / "TrainingHistory.csv", index=False, encoding="utf-8-sig")
        logging.info("fold=%s epoch=%d/%d loss=%.6f", held_out, epoch, config["fixed_epochs"], record["train_total_loss"])
    checkpoint = fold_dir / "Model.pt"
    torch.save({
        "schema_version": "Stage2I_SeasonFoldCheckpoint_v1",
        "held_out_season": held_out,
        "model_state_dict": model.state_dict(),
        "active_channel_indices": config["active_channel_indices"],
        "fixed_epochs": config["fixed_epochs"],
        "random_seed": seed,
    }, checkpoint)
    return model, device, history, checkpoint


def evaluate_fold(dataset, model, device, evaluator, config: dict, fold_dir: Path):
    loader = DataLoader(dataset, batch_size=int(config["loader"]["batch_size"]), shuffle=False, num_workers=0)
    dimensions = {}
    for date, rows in dataset.table.groupby("date"):
        dimensions[str(date)] = (
            int((rows["row_off"] + rows["height"]).max()),
            int((rows["col_off"] + rows["width"]).max()),
        )
    mosaics = {
        date: {"p": np.zeros(shape), "y": np.zeros(shape), "n": np.zeros(shape, dtype=np.uint16)}
        for date, shape in dimensions.items()
    }
    model.eval()
    with torch.no_grad():
        for raw in loader:
            prediction = model(raw["input"].to(device))["prediction"].cpu().numpy()
            target = raw["target"].numpy()
            mask = raw["valid_mask"].numpy() > 0.5
            for i in range(prediction.shape[0]):
                date = str(raw["date"][i]); y0 = int(raw["row_off"][i]); x0 = int(raw["col_off"][i])
                h, w = prediction.shape[-2:]; ys = slice(y0, y0+h); xs = slice(x0, x0+w); local = mask[i, 0]
                mosaics[date]["p"][ys, xs] += prediction[i, 0] * local
                mosaics[date]["y"][ys, xs] += target[i, 0] * local
                mosaics[date]["n"][ys, xs] += local.astype(np.uint16)
    raster_dir = fold_dir / "HeldOutRasters"; raster_dir.mkdir(exist_ok=True)
    rows = []
    for date in sorted(mosaics):
        item = mosaics[date]; valid = item["n"] > 0
        p = np.divide(item["p"], item["n"], out=np.zeros_like(item["p"]), where=valid)
        y = np.divide(item["y"], item["n"], out=np.zeros_like(item["y"]), where=valid)
        result = evaluator.metrics(p[valid], y[valid])
        policy = {"window_size": 11, "minimum_valid_fraction": 0.8, "data_range": 1.0, "k1": 0.01, "k2": 0.03}
        ssim, n_windows = evaluator.masked_ssim(p, y, valid, policy)
        source = Path(dataset.table.loc[dataset.table["date"].eq(date), "label_path"].iloc[0])
        with rasterio.open(source) as src:
            profile = src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")
        prediction_out = np.where(valid, p, -9999.0).astype(np.float32)
        residual_out = np.where(valid, p-y, -9999.0).astype(np.float32)
        with rasterio.open(raster_dir / f"Prediction_{date}.tif", "w", **profile) as dst:
            dst.write(prediction_out, 1)
        with rasterio.open(raster_dir / f"Residual_{date}.tif", "w", **profile) as dst:
            dst.write(residual_out, 1)
        doy = int(dataset.table.loc[dataset.table["date"].eq(date), "stored_DOY"].iloc[0])
        rows.append({"date": date, "stored_DOY": doy, "season": season_from_doy(doy), **result, "SSIM": ssim, "N_SSIM_windows": n_windows})
    return pd.DataFrame(rows).sort_values("stored_DOY")


def main() -> None:
    config = load_json(CONFIG_PATH)
    root = Path(config["project_root"])
    output = resolve(root, config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(output / "00_Stage2I_RunLog.txt", mode="w", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    if config["selected_model"] != "A3_PRIOR_MODIS" or config["active_channel_indices"] != [0,1,2,3]:
        raise ValueError("Stage2I必须使用冻结A3。")
    if config["fixed_epochs"] != 32 or config["early_stopping"] is not False:
        raise ValueError("Stage2I必须固定32轮且无早停。")
    original_manifest = resolve(root, config["patch_manifest"])
    if sha256_file(original_manifest) != config["patch_manifest_sha256"]:
        raise ValueError("Stage2A清单哈希不一致。")
    manifest = pd.read_csv(original_manifest, encoding="utf-8-sig", dtype={"date": str})
    manifest["date"] = manifest["date"].astype(str).str.zfill(8)
    manifest["season_en"] = manifest["stored_DOY"].map(season_from_doy)
    augmented_manifest = output / "Nanyang_Stage2I_SeasonAnnotatedPatchManifest.csv"
    manifest.to_csv(augmented_manifest, index=False, encoding="utf-8-sig")
    stage2g = load_json(resolve(root, config["stage2g_config"]))
    stage2b = load_module("stage2b", STAGE2_ROOT / "scripts" / "Nanyang_Stage2B_FitTrainOnlyNormalization.py")
    evaluator = load_module("stage2h_metrics", STAGE2_ROOT / "scripts" / "Nanyang_Stage2H_EvaluateFinalA3_OneTimeTest.py")
    fold_rows, all_dates = [], []
    started = time.perf_counter()
    for held_out in config["fold_order"]:
        fold_started = time.perf_counter()
        fold_dir = output / f"Fold_{held_out}"; fold_dir.mkdir(exist_ok=True)
        normalization_path = fit_normalization(stage2b, manifest, held_out, config, fold_dir)
        normalization = load_json(normalization_path)
        kwargs = {
            "patch_manifest_path": augmented_manifest,
            "normalization_path": normalization_path,
            "held_out_season": held_out,
            "allowed_weights": config["allowed_weights"],
            "weight_tolerance": config["weight_tolerance"],
            "scene_cache_size": config["loader"]["scene_cache_size"],
        }
        train = SeasonFoldPatchDataset(role="train", **kwargs)
        held = SeasonFoldPatchDataset(role="held_out", **kwargs)
        train_dates = set(train.table["date"]); held_dates = set(held.table["date"])
        if train_dates & held_dates:
            raise RuntimeError(f"{held_out}折日期泄漏。")
        model, device, history, checkpoint = train_fold(train, normalization, stage2g, config, fold_dir, held_out)
        date_table = evaluate_fold(held, model, device, evaluator, config, fold_dir)
        date_table.insert(0, "held_out_season", held_out)
        date_table.to_csv(fold_dir / "DateMetrics.csv", index=False, encoding="utf-8-sig")
        all_dates.append(date_table)
        fold_rows.append({
            "held_out_season": held_out,
            "train_date_count": len(train_dates), "held_out_date_count": len(held_dates),
            "train_patch_count": len(train), "held_out_patch_count": len(held),
            "normalization_fit_pixels": normalization["fit_unique_pixel_count"],
            "initial_loss": history[0]["train_total_loss"], "final_loss": history[-1]["train_total_loss"],
            "date_macro_RMSE": float(date_table["RMSE"].mean()),
            "date_macro_MAE": float(date_table["MAE"].mean()),
            "date_macro_absolute_Bias": float(date_table["Bias"].abs().mean()),
            "date_macro_Pearson_R": float(date_table["Pearson_R"].mean()),
            "date_macro_R2": float(date_table["R2"].mean()),
            "date_macro_ubRMSE": float(date_table["ubRMSE"].mean()),
            "date_macro_SSIM": float(date_table["SSIM"].mean()),
            "normalization_sha256": sha256_file(normalization_path),
            "checkpoint_sha256": sha256_file(checkpoint),
            "elapsed_seconds": time.perf_counter()-fold_started,
            "held_out_used_for_training_or_normalization": 0,
            "status": "PASS",
        })
    fold_table = pd.DataFrame(fold_rows)
    date_table = pd.concat(all_dates, ignore_index=True).sort_values("stored_DOY")
    fold_table.to_csv(output / "Nanyang_Stage2I_FoldSummary.csv", index=False, encoding="utf-8-sig")
    date_table.to_csv(output / "Nanyang_Stage2I_AllDateMetrics.csv", index=False, encoding="utf-8-sig")
    audit = {
        "schema_version": "Stage2I_FinalAudit_v1",
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "fold_count": len(fold_table), "evaluated_date_count": len(date_table),
        "each_date_evaluated_exactly_once": bool(date_table["date"].nunique() == 26 and len(date_table) == 26),
        "held_out_leakage_count": int(fold_table["held_out_used_for_training_or_normalization"].sum()),
        "configuration_sha256": sha256_file(CONFIG_PATH),
        "source_patch_manifest_sha256": sha256_file(original_manifest),
        "augmented_manifest_sha256": sha256_file(augmented_manifest),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "dataset_adapter_sha256": sha256_file(STAGE2_ROOT / "dprcn" / "season_fold_dataset.py"),
        "elapsed_seconds": time.perf_counter()-started,
        "overall_status": "PASS" if len(fold_table) == 4 and len(date_table) == 26 else "FAIL",
        "interpretation": config["interpretation"],
        "fold_results": fold_table.to_dict(orient="records"),
    }
    (output / "00_Stage2I_FinalAudit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("STAGE2I=%s folds=%d dates=%d", audit["overall_status"], len(fold_table), len(date_table))
    print("STAGE2I_LEAVE_ONE_SEASON_OUT=" + audit["overall_status"])
    print(fold_table.to_string(index=False))


if __name__ == "__main__":
    main()

