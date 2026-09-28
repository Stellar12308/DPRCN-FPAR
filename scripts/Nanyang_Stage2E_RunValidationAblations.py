from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


SCRIPT_PATH = Path(__file__).resolve()
STAGE2_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(STAGE2_ROOT))

from dprcn.dataset import NanyangFPARPatchDataset, build_dataloader  # noqa: E402
from dprcn.losses import build_loss  # noqa: E402
from dprcn.model import build_model  # noqa: E402
from dprcn.reproducibility import seed_everything, sha256_file  # noqa: E402


ABLATION_CONFIG_PATH = (
    STAGE2_ROOT / "configs" / "Nanyang_Stage2E_ValidationAblation_Config_v1.json"
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_stage2d_module():
    path = STAGE2_ROOT / "scripts" / "Nanyang_Stage2D_TrainDPRCN.py"
    spec = importlib.util.spec_from_file_location("stage2d_training", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法载入Stage2D训练模块。")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixedInputMaskModel(nn.Module):
    """Keep architecture fixed and replace excluded standardized channels by zero."""

    def __init__(self, model: nn.Module, active_indices: list[int]) -> None:
        super().__init__()
        self.model = model
        mask = torch.zeros(1, 9, 1, 1, dtype=torch.float32)
        mask[:, active_indices] = 1.0
        self.register_buffer("input_mask", mask)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(inputs * self.input_mask)


def build_data(base: dict, root: Path):
    patch_path = resolve(root, base["stage2a_patch_manifest"])
    normalization_path = resolve(root, base["stage2b_parameters"])
    kwargs = {
        "patch_manifest_path": patch_path,
        "normalization_path": normalization_path,
        "allowed_weights": (0.5, 0.8, 1.0),
        "weight_tolerance": 1e-6,
        "scene_cache_size": int(base["loader"]["scene_cache_size"]),
    }
    train_dataset = NanyangFPARPatchDataset(split="train", **kwargs)
    val_dataset = NanyangFPARPatchDataset(split="val", **kwargs)
    train_loader = build_dataloader(
        train_dataset,
        batch_size=int(base["loader"]["batch_size"]),
        shuffle=True,
        seed=int(base["random_seed"]),
        num_workers=int(base["loader"]["num_workers"]),
        pin_memory=bool(base["loader"]["pin_memory"]),
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=int(base["loader"]["batch_size"]),
        shuffle=False,
        seed=int(base["random_seed"]),
        num_workers=int(base["loader"]["num_workers"]),
        pin_memory=bool(base["loader"]["pin_memory"]),
    )
    return train_dataset, val_dataset, train_loader, val_loader, normalization_path


def evaluate_raw_channel(loader, dataset, channel_index, mean, std, stage2d):
    dimensions = {}
    for date, rows in dataset.table.groupby("date"):
        dimensions[str(date)] = (
            int((rows["row_off"] + rows["height"]).max()),
            int((rows["col_off"] + rows["width"]).max()),
        )
    mosaics = {
        date: {
            "prediction_sum": np.zeros(shape, dtype=np.float64),
            "target_sum": np.zeros(shape, dtype=np.float64),
            "count": np.zeros(shape, dtype=np.uint16),
        }
        for date, shape in dimensions.items()
    }
    for batch in loader:
        inputs = batch["input"].numpy()
        targets = batch["target"].numpy()
        masks = batch["valid_mask"].numpy() > 0.5
        prediction = np.clip(inputs[:, channel_index:channel_index + 1] * std + mean, 0, 1)
        for index in range(prediction.shape[0]):
            date = str(batch["date"][index])
            row = int(batch["row_off"][index])
            column = int(batch["col_off"][index])
            height, width = prediction.shape[-2:]
            local_mask = masks[index, 0]
            ys, xs = slice(row, row + height), slice(column, column + width)
            mosaics[date]["prediction_sum"][ys, xs] += prediction[index, 0] * local_mask
            mosaics[date]["target_sum"][ys, xs] += targets[index, 0] * local_mask
            mosaics[date]["count"][ys, xs] += local_mask.astype(np.uint16)
    records, all_prediction, all_target = [], [], []
    for date in sorted(mosaics):
        item = mosaics[date]
        valid = item["count"] > 0
        prediction = item["prediction_sum"][valid] / item["count"][valid]
        target = item["target_sum"][valid] / item["count"][valid]
        metrics = stage2d.regression_metrics(prediction, target)
        doy = int(dataset.table.loc[dataset.table["date"] == date, "stored_DOY"].iloc[0])
        records.append({"date": date, "stored_DOY": doy, **metrics})
        all_prediction.append(prediction)
        all_target.append(target)
    date_table = pd.DataFrame(records).sort_values("stored_DOY")
    global_metrics = stage2d.regression_metrics(
        np.concatenate(all_prediction), np.concatenate(all_target)
    )
    summary = {
        **{f"global_{key}": value for key, value in global_metrics.items()},
        "date_macro_RMSE": float(date_table["RMSE"].mean()),
        "date_macro_MAE": float(date_table["MAE"].mean()),
        "date_macro_absolute_Bias": float(date_table["Bias"].abs().mean()),
        "date_macro_R": float(date_table["R"].mean()),
        "date_macro_R2": float(date_table["R2"].mean()),
    }
    return summary, date_table


def train_variant(variant, base, normalization, datasets, loaders, device, output_dir, stage2d):
    train_dataset, val_dataset = datasets
    train_loader, val_loader = loaders
    variant_dir = output_dir / variant["id"]
    variant_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(base["random_seed"]))
    core = build_model(base, normalization)
    if variant.get("reuse_frozen_stage2d_checkpoint"):
        raise RuntimeError("完整模型必须通过专用复用分支处理。")
    model = FixedInputMaskModel(core, variant["active_channel_indices"]).to(device)
    criterion = build_loss(base).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(base["optimizer"]["learning_rate"]),
        weight_decay=float(base["optimizer"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(base["scheduler"]["factor"]),
        patience=int(base["scheduler"]["patience"]),
        min_lr=float(base["scheduler"]["minimum_learning_rate"]),
    )
    maximum_epochs = int(base["formal_training"]["maximum_epochs"])
    early_patience = int(base["formal_training"]["early_stopping_patience"])
    minimum_delta = float(base["formal_training"]["minimum_delta"])
    best_metric, best_epoch, stale = math.inf, 0, 0
    history = []
    checkpoint_path = variant_dir / "best_model.pt"
    started = time.perf_counter()
    for epoch in range(1, maximum_epochs + 1):
        train_metrics = stage2d.train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            float(base["optimizer"]["gradient_clip_norm"]), None,
        )
        validation, dates = stage2d.evaluate_validation(
            model, val_loader, val_dataset, device, None
        )
        metric = float(validation["date_macro_RMSE"])
        scheduler.step(metric)
        improved = metric < best_metric - minimum_delta
        if improved:
            best_metric, best_epoch, stale = metric, epoch, 0
            torch.save({
                "schema_version": "Stage2E_AblationCheckpoint_v1",
                "variant": variant,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "validation": validation,
                "base_training_config_sha256": sha256_file(
                    STAGE2_ROOT / "configs" / "Nanyang_Stage2D_Training_Config_v1.json"
                ),
            }, checkpoint_path)
            dates.to_csv(
                variant_dir / "best_validation_date_metrics.csv",
                index=False, encoding="utf-8-sig",
            )
        else:
            stale += 1
        history.append({
            "epoch": epoch,
            "train_total_loss": train_metrics["total"],
            **{f"val_{key}": value for key, value in validation.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "is_best": int(improved),
        })
        pd.DataFrame(history).to_csv(
            variant_dir / "training_history.csv", index=False, encoding="utf-8-sig"
        )
        print(f'{variant["id"]} epoch={epoch} val_macro_RMSE={metric:.6f} best={best_metric:.6f}', flush=True)
        if stale >= early_patience:
            break
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = checkpoint["validation"]
    return {
        "variant_id": variant["id"],
        "label": variant["label"],
        "variant_type": "trained_ablation",
        "active_channel_count": len(variant["active_channel_indices"]),
        "active_channel_indices": json.dumps(variant["active_channel_indices"]),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="all")
    args = parser.parse_args()
    ablation = load_json(ABLATION_CONFIG_PATH)
    if ablation["test_split_access_allowed"] is not False:
        raise RuntimeError("Stage2E禁止测试集访问。")
    root = Path(ablation["project_root"])
    base_path = resolve(root, ablation["base_training_config"])
    base = load_json(base_path)
    if set(base["allowed_splits"]) != {"train", "val"}:
        raise RuntimeError("基础配置不是train/val限定配置。")
    output_dir = resolve(root, ablation["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stage2d = load_stage2d_module()
    train_dataset, val_dataset, train_loader, val_loader, normalization_path = build_data(base, root)
    normalization = load_json(normalization_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_path = output_dir / "Nanyang_Stage2E_AblationValidationSummary.csv"
    existing = pd.read_csv(summary_path, encoding="utf-8-sig") if summary_path.exists() else pd.DataFrame()
    records = existing.to_dict("records")
    completed = set(existing.get("variant_id", pd.Series(dtype=str)).astype(str))

    if args.variant in {"all", "baselines"}:
        channel_lookup = {name: index for index, name in enumerate(ablation["channel_order"])}
        for baseline in ablation["analytical_baselines"]:
            if baseline["id"] in completed:
                continue
            channel = baseline["prediction_source"]
            parameter = normalization["parameters"][channel]
            metrics, dates = evaluate_raw_channel(
                val_loader, val_dataset, channel_lookup[channel],
                float(parameter["mean"]), float(parameter["std_population"]), stage2d,
            )
            dates.to_csv(
                output_dir / f'{baseline["id"]}_validation_date_metrics.csv',
                index=False, encoding="utf-8-sig",
            )
            records.append({
                "variant_id": baseline["id"], "label": baseline["label"],
                "variant_type": "analytical_baseline", "active_channel_count": 1,
                "active_channel_indices": "", "best_epoch": 0,
                "epochs_completed": 0, "elapsed_seconds": 0.0,
                "checkpoint_sha256": "", **metrics,
            })
            completed.add(baseline["id"])

    variants = ablation["trainable_variants"]
    requested = [item for item in variants if args.variant in {"all", item["id"]}]
    for variant in requested:
        if variant["id"] in completed:
            continue
        if variant.get("reuse_frozen_stage2d_checkpoint"):
            checkpoint_path = resolve(root, ablation["stage2d_checkpoint"])
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            seed_everything(int(base["random_seed"]))
            model = build_model(base, normalization).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            metrics, dates = stage2d.evaluate_validation(model, val_loader, val_dataset, device, None)
            dates.to_csv(
                output_dir / f'{variant["id"]}_validation_date_metrics.csv',
                index=False, encoding="utf-8-sig",
            )
            record = {
                "variant_id": variant["id"], "label": variant["label"],
                "variant_type": "frozen_stage2d_model",
                "active_channel_count": len(variant["active_channel_indices"]),
                "active_channel_indices": json.dumps(variant["active_channel_indices"]),
                "best_epoch": int(checkpoint["epoch"]), "epochs_completed": 59,
                "elapsed_seconds": 0.0, "checkpoint_sha256": sha256_file(checkpoint_path),
                **metrics,
            }
        else:
            record = train_variant(
                variant, base, normalization,
                (train_dataset, val_dataset), (train_loader, val_loader),
                device, output_dir, stage2d,
            )
        records.append(record)
        completed.add(variant["id"])
        pd.DataFrame(records).to_csv(summary_path, index=False, encoding="utf-8-sig")

    table = pd.DataFrame(records).drop_duplicates("variant_id", keep="last")
    table = table.sort_values("date_macro_RMSE").reset_index(drop=True)
    table["validation_rank"] = np.arange(1, len(table) + 1)
    best = float(table["date_macro_RMSE"].min())
    tolerance = float(ablation["model_selection"]["complexity_tolerance_rmse"])
    eligible = table.loc[
        table["variant_type"].isin(["trained_ablation", "frozen_stage2d_model"])
        & table["date_macro_RMSE"].le(best + tolerance)
    ].sort_values(["active_channel_count", "date_macro_RMSE"])
    selected = str(eligible.iloc[0]["variant_id"])
    table["selected_by_frozen_rule"] = table["variant_id"].eq(selected).astype(int)
    table["test_accessed"] = 0
    table["overall_status"] = "PASS"
    table.to_csv(summary_path, index=False, encoding="utf-8-sig")
    audit = {
        "schema_version": "Stage2E_ValidationAblationAudit_v1",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "selection_split": "val",
        "test_split_accessed": False,
        "selected_variant": selected,
        "selection_rule": ablation["model_selection"]["rule"],
        "configuration_sha256": sha256_file(ABLATION_CONFIG_PATH),
        "summary_sha256": sha256_file(summary_path),
        "overall_status": "PASS",
    }
    with (output_dir / "00_Stage2E_ValidationAblationAudit.json").open("w", encoding="utf-8") as file:
        json.dump(audit, file, ensure_ascii=False, indent=2)
    print(table[["validation_rank", "variant_id", "active_channel_count", "date_macro_RMSE", "date_macro_R2", "selected_by_frozen_rule"]].to_string(index=False))
    print("STAGE2E_VALIDATION_ABLATION=PASS")
    print("SELECTED_VARIANT=" + selected)
    print("TEST_ACCESSED=0")


if __name__ == "__main__":
    main()

