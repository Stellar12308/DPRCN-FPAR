from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import platform
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_NAME = SCRIPT_PATH.name
STAGE2_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(STAGE2_ROOT))

from dprcn.dataset import NanyangFPARPatchDataset, build_dataloader  # noqa: E402
from dprcn.losses import build_loss  # noqa: E402
from dprcn.model import build_model  # noqa: E402
from dprcn.reproducibility import seed_everything, sha256_file  # noqa: E402


CONFIG_PATH = STAGE2_ROOT / "configs" / "Nanyang_Stage2D_Training_Config_v1.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["smoke", "formal"],
        default="smoke",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        **batch,
        "input": batch["input"].to(device, non_blocking=True),
        "target": batch["target"].to(device, non_blocking=True),
        "valid_mask": batch["valid_mask"].to(device, non_blocking=True),
        "weight": batch["weight"].to(device, non_blocking=True),
    }


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    prediction = prediction.astype(np.float64, copy=False)
    target = target.astype(np.float64, copy=False)
    error = prediction - target
    n = int(target.size)
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    target_mean = float(np.mean(target))
    target_std = float(np.std(target, ddof=0))
    denominator = float(np.sum((target - target_mean) ** 2))
    r2 = float(1.0 - np.sum(error ** 2) / denominator) if denominator > 0 else math.nan
    pred_std = float(np.std(prediction, ddof=0))
    if target_std > 0 and pred_std > 0:
        covariance = float(np.mean(
            (prediction - float(np.mean(prediction)))
            * (target - target_mean),
            dtype=np.float64,
        ))
        correlation = covariance / (pred_std * target_std)
    else:
        correlation = math.nan
    return {
        "N": n,
        "RMSE": rmse,
        "MAE": mae,
        "Bias": bias,
        "R": correlation,
        "R2": r2,
        "target_mean": target_mean,
        "target_std": target_std,
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": pred_std,
        "prediction_le_0p01_pct": float(np.mean(prediction <= 0.01) * 100.0),
        "prediction_ge_0p99_pct": float(np.mean(prediction >= 0.99) * 100.0),
    }


def evaluate_validation(
    model: torch.nn.Module,
    loader,
    dataset: NanyangFPARPatchDataset,
    device: torch.device,
    maximum_batches: int | None,
) -> tuple[dict, pd.DataFrame]:
    dimensions = {}
    for date, rows in dataset.table.groupby("date"):
        height = int((rows["row_off"] + rows["height"]).max())
        width = int((rows["col_off"] + rows["width"]).max())
        dimensions[str(date)] = (height, width)
    mosaics = {
        date: {
            "prediction_sum": np.zeros(shape, dtype=np.float64),
            "target_sum": np.zeros(shape, dtype=np.float64),
            "count": np.zeros(shape, dtype=np.uint16),
        }
        for date, shape in dimensions.items()
    }
    model.eval()
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if maximum_batches is not None and batch_index >= maximum_batches:
                break
            batch = move_batch(raw_batch, device)
            prediction = model(batch["input"])["prediction"].cpu().numpy()
            target = batch["target"].cpu().numpy()
            mask = batch["valid_mask"].cpu().numpy() > 0.5
            batch_size = prediction.shape[0]
            for index in range(batch_size):
                date = str(raw_batch["date"][index])
                y = int(raw_batch["row_off"][index])
                x = int(raw_batch["col_off"][index])
                height, width = prediction.shape[-2:]
                local_mask = mask[index, 0]
                destination = mosaics[date]
                ys = slice(y, y + height)
                xs = slice(x, x + width)
                destination["prediction_sum"][ys, xs] += prediction[index, 0] * local_mask
                destination["target_sum"][ys, xs] += target[index, 0] * local_mask
                destination["count"][ys, xs] += local_mask.astype(np.uint16)

    date_records = []
    all_predictions = []
    all_targets = []
    for date in sorted(mosaics):
        mosaic = mosaics[date]
        valid = mosaic["count"] > 0
        if not valid.any():
            raise RuntimeError(f"验证日期{date}没有拼接后的有效像元。")
        prediction = mosaic["prediction_sum"][valid] / mosaic["count"][valid]
        target = mosaic["target_sum"][valid] / mosaic["count"][valid]
        metrics = regression_metrics(prediction, target)
        doy = int(dataset.table.loc[dataset.table["date"] == date, "stored_DOY"].iloc[0])
        date_records.append({"date": date, "stored_DOY": doy, **metrics})
        all_predictions.append(prediction)
        all_targets.append(target)
    date_table = pd.DataFrame(date_records).sort_values("stored_DOY")
    global_metrics = regression_metrics(
        np.concatenate(all_predictions),
        np.concatenate(all_targets),
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


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    gradient_clip_norm: float,
    maximum_batches: int | None,
) -> dict:
    model.train()
    totals = defaultdict(float)
    batches = 0
    for batch_index, raw_batch in enumerate(loader):
        if maximum_batches is not None and batch_index >= maximum_batches:
            break
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["input"])
        losses = criterion(
            output["prediction"],
            batch["target"],
            batch["valid_mask"],
            batch["weight"],
        )
        losses["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_norm,
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("出现非有限梯度。")
        optimizer.step()
        for name in ["total", "pixel", "gradient"]:
            totals[name] += float(losses[name].detach().cpu())
        totals["gradient_norm"] += float(gradient_norm.detach().cpu())
        batches += 1
    if batches == 0:
        raise RuntimeError("训练Epoch没有Batch。")
    return {key: value / batches for key, value in totals.items()} | {"batch_count": batches}


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    validation: dict,
    config: dict,
    source_hashes: dict,
) -> None:
    torch.save({
        "schema_version": "Stage2D_DPRCN_Checkpoint_v1",
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "validation": validation,
        "config": config,
        "source_hashes": source_hashes,
    }, path)


def main() -> None:
    arguments = parse_arguments()
    config = load_json(CONFIG_PATH)
    if config.get("schema_version") != "Stage2D_Training_Config_v1":
        raise ValueError("Stage2D配置版本错误。")
    if config.get("test_split_access_allowed") is not False:
        raise RuntimeError("Stage2D必须禁止测试集访问。")
    if set(config.get("allowed_splits", [])) != {"train", "val"}:
        raise RuntimeError("Stage2D只允许train和val。")
    seed_everything(int(config["random_seed"]))
    root = Path(config["project_root"])
    patch_path = resolve(root, config["stage2a_patch_manifest"])
    normalization_path = resolve(root, config["stage2b_parameters"])
    stage2c_summary_path = resolve(root, config["stage2c_summary"])
    base_output = resolve(root, config["output_dir"])
    run_name = "SmokeTest_Seed42" if arguments.mode == "smoke" else "Formal_MainModel_Seed42"
    output_dir = base_output / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "00_Stage2D_Training_RunLog.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    log = logging.getLogger("Stage2D")
    log.info("Stage2D开始，mode=%s", arguments.mode)

    for path in [patch_path, normalization_path, stage2c_summary_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    stage2c_summary = pd.read_csv(stage2c_summary_path, encoding="utf-8-sig")
    if stage2c_summary.iloc[0]["overall_status"] != "PASS":
        raise RuntimeError("Stage2C不是PASS。")
    if sha256_file(patch_path) != str(stage2c_summary.iloc[0]["stage2a_patch_manifest_sha256"]):
        raise RuntimeError("Patch清单与Stage2C记录不一致。")
    if sha256_file(normalization_path) != str(stage2c_summary.iloc[0]["stage2b_normalization_sha256"]):
        raise RuntimeError("归一化参数与Stage2C记录不一致。")

    normalization = load_json(normalization_path)
    loader_config = config["loader"]
    dataset_kwargs = {
        "patch_manifest_path": patch_path,
        "normalization_path": normalization_path,
        "allowed_weights": (0.5, 0.8, 1.0),
        "weight_tolerance": 1e-6,
        "scene_cache_size": int(loader_config["scene_cache_size"]),
    }
    train_dataset = NanyangFPARPatchDataset(split="train", **dataset_kwargs)
    val_dataset = NanyangFPARPatchDataset(split="val", **dataset_kwargs)
    train_loader = build_dataloader(
        train_dataset,
        batch_size=int(loader_config["batch_size"]),
        shuffle=True,
        seed=int(config["random_seed"]),
        num_workers=int(loader_config["num_workers"]),
        pin_memory=bool(loader_config["pin_memory"]),
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=int(loader_config["batch_size"]),
        shuffle=False,
        seed=int(config["random_seed"]),
        num_workers=int(loader_config["num_workers"]),
        pin_memory=bool(loader_config["pin_memory"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, normalization).to(device)
    criterion = build_loss(config).to(device)
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler_config = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_config["factor"]),
        patience=int(scheduler_config["patience"]),
        min_lr=float(scheduler_config["minimum_learning_rate"]),
    )

    mode_config = config["smoke_test"] if arguments.mode == "smoke" else config["formal_training"]
    maximum_epochs = int(mode_config["maximum_epochs"])
    early_patience = int(mode_config["early_stopping_patience"])
    maximum_train_batches = mode_config.get("maximum_train_batches_per_epoch")
    maximum_val_batches = mode_config.get("maximum_val_batches")
    if maximum_train_batches is not None:
        maximum_train_batches = int(maximum_train_batches)
    if maximum_val_batches is not None:
        maximum_val_batches = int(maximum_val_batches)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    source_hashes = {
        "training_config_sha256": sha256_file(CONFIG_PATH),
        "patch_manifest_sha256": sha256_file(patch_path),
        "normalization_sha256": sha256_file(normalization_path),
        "dataset_module_sha256": sha256_file(STAGE2_ROOT / "dprcn" / "dataset.py"),
        "model_module_sha256": sha256_file(STAGE2_ROOT / "dprcn" / "model.py"),
        "loss_module_sha256": sha256_file(STAGE2_ROOT / "dprcn" / "losses.py"),
    }
    shutil.copy2(CONFIG_PATH, output_dir / CONFIG_PATH.name.replace(".json", "_Frozen.json"))
    log.info("device=%s", device)
    if device.type == "cuda":
        log.info("GPU=%s", torch.cuda.get_device_name(0))
    log.info("参数量=%d", parameter_count)
    log.info("train/val Patch=%d/%d", len(train_dataset), len(val_dataset))
    log.info("测试集访问=禁止")

    history_records = []
    best_metric = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.perf_counter()
    best_path = output_dir / "best_model.pt"
    for epoch in range(1, maximum_epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            float(optimizer_config["gradient_clip_norm"]),
            maximum_train_batches,
        )
        validation, date_table = evaluate_validation(
            model,
            val_loader,
            val_dataset,
            device,
            maximum_val_batches,
        )
        selection_metric = float(validation["date_macro_RMSE"])
        scheduler.step(selection_metric)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        improved = selection_metric < best_metric - float(
            config["formal_training"]["minimum_delta"]
        )
        if improved:
            best_metric = selection_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                validation,
                config,
                source_hashes,
            )
            date_table.to_csv(
                output_dir / "best_epoch_validation_date_metrics.csv",
                index=False,
                encoding="utf-8-sig",
            )
        else:
            epochs_without_improvement += 1
        record = {
            "epoch": epoch,
            "train_total_loss": train_metrics["total"],
            "train_pixel_loss": train_metrics["pixel"],
            "train_gradient_loss": train_metrics["gradient"],
            "train_gradient_norm": train_metrics["gradient_norm"],
            "train_batch_count": train_metrics["batch_count"],
            **{f"val_{key}": value for key, value in validation.items()},
            "learning_rate": learning_rate,
            "is_best": int(improved),
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history_records.append(record)
        pd.DataFrame(history_records).to_csv(
            output_dir / "training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )
        log.info(
            "Epoch=%d train_loss=%.6f val_macro_RMSE=%.6f val_global_RMSE=%.6f best=%d lr=%.2e",
            epoch,
            train_metrics["total"],
            selection_metric,
            validation["global_RMSE"],
            int(improved),
            learning_rate,
        )
        if epochs_without_improvement >= early_patience:
            log.info("Early stopping at epoch=%d", epoch)
            break

    if not best_path.exists() or best_epoch <= 0:
        raise RuntimeError("没有生成最佳模型。")
    elapsed = time.perf_counter() - start_time
    if arguments.mode == "smoke":
        losses = pd.Series([row["train_total_loss"] for row in history_records])
        smoke_passed = (
            len(history_records) == maximum_epochs
            and np.isfinite(losses).all()
            and np.isfinite(best_metric)
            and best_metric > 0
        )
        if not smoke_passed:
            raise RuntimeError("Smoke Test未通过。")

    summary_path = output_dir / "Nanyang_Stage2D_TrainingSummary.csv"
    pd.DataFrame([{
        "script_name": SCRIPT_NAME,
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": arguments.mode,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "random_seed": config["random_seed"],
        "parameter_count": parameter_count,
        "train_patch_count": len(train_dataset),
        "val_patch_count": len(val_dataset),
        "test_accessed": 0,
        "epochs_completed": len(history_records),
        "best_epoch": best_epoch,
        "best_validation_date_macro_RMSE": best_metric,
        "elapsed_seconds": elapsed,
        "best_checkpoint_sha256": sha256_file(best_path),
        **source_hashes,
        "overall_status": "PASS",
    }]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    log.info("最佳Epoch=%d，验证日期宏平均RMSE=%.6f", best_epoch, best_metric)
    log.info("耗时=%.2f秒", elapsed)
    log.info("总状态=PASS")


if __name__ == "__main__":
    main()

