from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn


SCRIPT_PATH = Path(__file__).resolve()
STAGE3_ROOT = SCRIPT_PATH.parents[1]
CONFIG_PATH = STAGE3_ROOT / "configs" / "Nanyang_Stage3F_CommonMaskComparison_Config_v1.json"
STAGE2_ROOT = STAGE3_ROOT.parent / "03_Stage2_DPRCN"
sys.path.insert(0, str(STAGE2_ROOT))

from dprcn.dataset import build_dataloader  # noqa: E402
from dprcn.final_dataset import FinalTrainValPatchDataset  # noqa: E402
from dprcn.model import build_model  # noqa: E402
from dprcn.reproducibility import seed_everything, sha256_file  # noqa: E402


class FixedA3InputMask(nn.Module):
    def __init__(self, model: nn.Module, active_indices: list[int]) -> None:
        super().__init__()
        self.model = model
        mask = torch.zeros(1, 9, 1, 1, dtype=torch.float32)
        mask[:, active_indices] = 1.0
        self.register_buffer("input_mask", mask)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(inputs * self.input_mask)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stage2h_module():
    path = STAGE2_ROOT / "scripts" / "Nanyang_Stage2H_EvaluateFinalA3_OneTimeTest.py"
    spec = importlib.util.spec_from_file_location("stage2h_metrics", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def raster(path: Path):
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float64)
        valid = np.isfinite(array) & (array >= 0.0) & (array <= 1.0)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= array != float(src.nodata)
        return array, valid, src.profile.copy()


def same_grid(a: dict, b: dict) -> bool:
    return a["width"] == b["width"] and a["height"] == b["height"] and a["crs"] == b["crs"] and a["transform"] == b["transform"]


def save_raster(path: Path, array: np.ndarray, valid: np.ndarray, profile: dict):
    metadata = profile.copy()
    metadata.update(count=1, dtype="float32", nodata=-9999.0, compress="deflate")
    with rasterio.open(path, "w", **metadata) as dst:
        dst.write(np.where(valid, array, -9999.0).astype(np.float32), 1)


def reconstruct_a3(root: Path, cfg: dict, output: Path, manifest: pd.DataFrame, log):
    frozen = read_json(root / cfg["stage2h_config"])
    checkpoint_path = resolve(root, frozen["checkpoint"])
    normalization_path = resolve(root, frozen["normalization"])
    patch_manifest_path = resolve(root, frozen["patch_manifest"])
    for path, expected in [
        (checkpoint_path, frozen["checkpoint_sha256"]),
        (normalization_path, frozen["normalization_sha256"]),
        (patch_manifest_path, frozen["patch_manifest_sha256"]),
    ]:
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"冻结文件哈希不一致：{path}")
    seed_everything(int(frozen["random_seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalization = read_json(normalization_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("test_split_accessed") is not False:
        raise RuntimeError("Stage2G检查点测试封存标记异常。")
    model = FixedA3InputMask(build_model(checkpoint["config"], normalization), frozen["active_channel_indices"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    dataset = FinalTrainValPatchDataset(patch_manifest_path, normalization_path, "test", scene_cache_size=int(frozen["scene_cache_size"]))
    loader = build_dataloader(dataset, int(frozen["batch_size"]), False, int(frozen["random_seed"]), 0, False)
    dimensions = {}
    for _, row in manifest.iterrows():
        with rasterio.open(Path(row["target_label_evaluation_only"])) as label_source:
            dimensions[str(row["target_date"])] = (label_source.height, label_source.width)
    mosaics = {date: {"sum": np.zeros(shape, np.float64), "count": np.zeros(shape, np.uint16)} for date, shape in dimensions.items()}
    with torch.no_grad():
        for raw in loader:
            prediction = model(raw["input"].to(device))["prediction"].cpu().numpy()
            mask = raw["valid_mask"].numpy() > 0.5
            for index in range(prediction.shape[0]):
                date = str(raw["date"][index])
                y0, x0 = int(raw["row_off"][index]), int(raw["col_off"][index])
                height, width = prediction.shape[-2:]
                ys, xs = slice(y0, y0 + height), slice(x0, x0 + width)
                local = mask[index, 0]
                mosaics[date]["sum"][ys, xs] += prediction[index, 0] * local
                mosaics[date]["count"][ys, xs] += local.astype(np.uint16)
    paths = {}
    for _, row in manifest.iterrows():
        date = str(row["target_date"])
        item = mosaics[date]
        valid = item["count"] > 0
        prediction = np.divide(item["sum"], item["count"], out=np.zeros_like(item["sum"]), where=valid)
        prediction = np.clip(prediction, 0.0, 1.0)
        _, _, label_profile = raster(Path(row["target_label_evaluation_only"]))
        if prediction.shape != (label_profile["height"], label_profile["width"]):
            raise ValueError(f"A3完整画布与标签不一致：{date}")
        path = output / f"Nanyang_{date}_DPRCN_A3_FinalPrediction.tif"
        save_raster(path, prediction, valid, label_profile)
        paths[date] = path
        log.info("A3 mosaic date=%s valid=%d", date, int(valid.sum()))
    return paths, str(device), frozen


def create_figures(date_table: pd.DataFrame, macro_table: pd.DataFrame, output: Path):
    colors = {"DPRCN-A3": "#1f77b4", "ESTARFM": "#ff7f0e", "PK-STEM (adapted)": "#2ca02c"}
    dates = sorted(date_table["date"].unique())
    x = np.arange(len(dates))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for model in colors:
        part = date_table[date_table["model"].eq(model)].set_index("date").loc[dates]
        axes[0].plot(x, part["RMSE"], marker="o", linewidth=2, label=model, color=colors[model])
        axes[1].plot(x, part["SSIM"], marker="o", linewidth=2, label=model, color=colors[model])
    axes[0].set_ylabel("RMSE")
    axes[1].set_ylabel("SSIM")
    axes[1].set_xlabel("Frozen test date")
    axes[1].set_xticks(x, dates, rotation=30, ha="right")
    axes[0].grid(alpha=0.25); axes[1].grid(alpha=0.25)
    axes[0].legend(ncol=3, frameon=False, loc="upper center")
    fig.suptitle("Common-mask performance on five frozen test dates", fontsize=14)
    fig.savefig(output / "Fig_Stage3F_PerDate_RMSE_SSIM.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    metrics = ["RMSE", "MAE", "Absolute_Bias", "Pearson_R", "R2", "ubRMSE", "SSIM"]
    models = list(colors)
    raw = macro_table.set_index("model").loc[models, metrics]
    score = raw.copy()
    for metric in metrics:
        values = raw[metric].to_numpy(float)
        lo, hi = np.nanmin(values), np.nanmax(values)
        normalized = np.ones_like(values) if math.isclose(lo, hi) else (values - lo) / (hi - lo)
        if metric in {"RMSE", "MAE", "Absolute_Bias", "ubRMSE"}:
            normalized = 1.0 - normalized
        score[metric] = normalized
    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    image = ax.imshow(score.to_numpy(float), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)), ["RMSE", "MAE", "|Bias|", "R", "$R^2$", "ubRMSE", "SSIM"])
    ax.set_yticks(np.arange(len(models)), models)
    for i in range(len(models)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{raw.iloc[i, j]:.3f}", ha="center", va="center", color="white" if score.iloc[i, j] > 0.55 else "black", fontsize=9)
    ax.set_title("Five-date macro performance (color normalized within each metric)")
    fig.colorbar(image, ax=ax, label="Relative performance (higher is better)", fraction=0.025, pad=0.03)
    fig.savefig(output / "Fig_Stage3F_MacroMetricPerformanceHeatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    started = time.perf_counter()
    cfg = read_json(CONFIG_PATH)
    root = Path(cfg["project_root"])
    output = root / cfg["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.FileHandler(output / "00_Stage3F_CommonMaskComparison_RunLog.txt", mode="w", encoding="utf-8"), logging.StreamHandler()], force=True)
    log = logging.getLogger("stage3f")
    metric_module = load_stage2h_module()
    manifest_path = root / cfg["reference_manifest"]
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", dtype=str)
    if len(manifest) != int(cfg["expected_test_dates"]) or not manifest["test_label_used_as_input"].astype(int).eq(0).all():
        raise RuntimeError("Stage3A冻结测试清单异常。")
    a3_paths, device, frozen = reconstruct_a3(root, cfg, output, manifest, log)
    model_paths = {
        "DPRCN-A3": a3_paths,
        "ESTARFM": {str(r["target_date"]): root / cfg["estarfm_dir"] / f"Nanyang_{r['target_date']}_ESTARFM_FPAR.tif" for _, r in manifest.iterrows()},
        "PK-STEM (adapted)": {str(r["target_date"]): root / cfg["pkstem_dir"] / f"Nanyang_{r['target_date']}_AdaptedPKSTEM_ModeIII_FPAR.tif" for _, r in manifest.iterrows()},
    }
    rows, pooled = [], {model: {"p": [], "y": []} for model in cfg["models"]}
    mask_rows = []
    for _, item in manifest.iterrows():
        date = str(item["target_date"])
        target, target_valid, target_profile = raster(Path(item["target_label_evaluation_only"]))
        arrays, valids = {}, []
        for model in cfg["models"]:
            array, valid, profile = raster(model_paths[model][date])
            if not same_grid(profile, target_profile):
                raise ValueError(f"{date} {model}网格不一致。")
            arrays[model] = array; valids.append(valid)
        common = target_valid.copy()
        for valid in valids:
            common &= valid
        if not common.any():
            raise RuntimeError(f"{date}无三模型共同有效像元。")
        mask_path = output / f"Nanyang_{date}_ThreeModelCommonValidMask.tif"
        save_raster(mask_path, common.astype(np.float32), np.ones_like(common, bool), target_profile)
        mask_rows.append({"date": date, "stored_DOY": int(item["target_DOY"]), "N_grid_pixels": int(common.size), "N_common_valid": int(common.sum()), "common_coverage_pct": float(common.mean() * 100.0), "mask_path": str(mask_path), "mask_sha256": sha256(mask_path)})
        for model in cfg["models"]:
            result = metric_module.metrics(arrays[model][common], target[common])
            ssim, n_windows = metric_module.masked_ssim(arrays[model], target, common, cfg["ssim"])
            rows.append({"date": date, "stored_DOY": int(item["target_DOY"]), "model": model, **result, "Absolute_Bias": abs(result["Bias"]), "SSIM": ssim, "N_SSIM_windows": n_windows})
            pooled[model]["p"].append(arrays[model][common]); pooled[model]["y"].append(target[common])
            log.info("date=%s model=%s N=%d RMSE=%.6f SSIM=%.6f", date, model, int(common.sum()), result["RMSE"], ssim)
    date_table = pd.DataFrame(rows).sort_values(["stored_DOY", "model"]).reset_index(drop=True)
    macro_rows = []
    for model in cfg["models"]:
        part = date_table[date_table["model"].eq(model)]
        global_result = metric_module.metrics(np.concatenate(pooled[model]["p"]), np.concatenate(pooled[model]["y"]))
        macro_rows.append({"model": model, "N_common_valid_global": global_result["N_valid_pixels"], "global_RMSE": global_result["RMSE"], "global_MAE": global_result["MAE"], "global_Bias": global_result["Bias"], "global_Pearson_R": global_result["Pearson_R"], "global_R2": global_result["R2"], "global_ubRMSE": global_result["ubRMSE"], "RMSE": float(part["RMSE"].mean()), "MAE": float(part["MAE"].mean()), "Bias": float(part["Bias"].mean()), "Absolute_Bias": float(part["Absolute_Bias"].mean()), "Pearson_R": float(part["Pearson_R"].mean()), "R2": float(part["R2"].mean()), "ubRMSE": float(part["ubRMSE"].mean()), "SSIM": float(part["SSIM"].mean())})
    macro_table = pd.DataFrame(macro_rows)
    metrics = cfg["reported_metrics"]
    wide = date_table.pivot(index="date", columns="model", values=metrics)
    wide = wide.swaplevel(0, 1, axis=1).reindex(columns=pd.MultiIndex.from_product([cfg["models"], metrics]))
    macro_series = {(model, metric): float(macro_table.loc[macro_table["model"].eq(model), metric].iloc[0]) for model in cfg["models"] for metric in metrics}
    wide.loc["Macro average"] = pd.Series(macro_series)
    wide.index.name = "Test date"
    date_table.to_csv(output / "Nanyang_Stage3F_CommonMask_DateMetrics_Long.csv", index=False, encoding="utf-8-sig")
    macro_table.to_csv(output / "Nanyang_Stage3F_CommonMask_MacroAndGlobalMetrics.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(output / "Nanyang_Stage3F_CommonMask_ModelMetricTable_DoubleHeader.csv", encoding="utf-8-sig")
    pd.DataFrame(mask_rows).to_csv(output / "Nanyang_Stage3F_CommonMask_Audit_5Dates.csv", index=False, encoding="utf-8-sig")
    winner = macro_table.sort_values("RMSE").iloc[0]
    evidence = {"run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "device": device, "common_mask_policy": cfg["common_mask_policy"], "winner_by_primary_metric": winner["model"], "best_date_macro_RMSE": float(winner["RMSE"]), "test_labels_used_for_prediction": False, "test_labels_used_for_final_evaluation_only": True, "stage2h_checkpoint_sha256": frozen["checkpoint_sha256"], "config_sha256": sha256(CONFIG_PATH), "script_sha256": sha256(SCRIPT_PATH), "manifest_sha256": sha256(manifest_path), "elapsed_seconds": time.perf_counter() - started, "status": "PASS"}
    with (output / "00_Stage3F_CommonMaskComparison_Evidence.json").open("w", encoding="utf-8") as file:
        json.dump(evidence, file, ensure_ascii=False, indent=2)
    log.info("winner=%s macro_RMSE=%.6f", winner["model"], float(winner["RMSE"]))
    log.info("STAGE3F_COMMON_MASK_COMPARISON=PASS")
    print("STAGE3F_COMMON_MASK_COMPARISON=PASS")
    print(macro_table.to_string(index=False))


if __name__ == "__main__":
    main()

