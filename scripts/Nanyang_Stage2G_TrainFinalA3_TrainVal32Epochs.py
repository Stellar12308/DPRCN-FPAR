from __future__ import annotations

import json
import logging
import platform
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader


SCRIPT_PATH = Path(__file__).resolve()
STAGE2_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(STAGE2_ROOT))

from dprcn.final_dataset import FinalTrainValPatchDataset  # noqa: E402
from dprcn.losses import build_loss  # noqa: E402
from dprcn.model import build_model  # noqa: E402
from dprcn.reproducibility import seed_everything, sha256_file  # noqa: E402


CONFIG_PATH = STAGE2_ROOT / "configs" / "Nanyang_Stage2G_FinalTrainingProtocol_Config_v1.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def worker_seed(worker_id: int) -> None:
    value = torch.initial_seed() % (2**32)
    np.random.seed(value)
    random.seed(value)


class FixedA3InputMask(nn.Module):
    def __init__(self, model: nn.Module, active_indices: list[int]) -> None:
        super().__init__()
        self.model = model
        mask = torch.zeros(1, 9, 1, 1, dtype=torch.float32)
        mask[:, active_indices] = 1.0
        self.register_buffer("input_mask", mask)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(inputs * self.input_mask)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "input": batch["input"].to(device, non_blocking=True),
        "target": batch["target"].to(device, non_blocking=True),
        "valid_mask": batch["valid_mask"].to(device, non_blocking=True),
        "weight": batch["weight"].to(device, non_blocking=True),
    }


def train_epoch(model, loader, criterion, optimizer, device, clip_norm: float) -> dict:
    model.train()
    totals = defaultdict(float)
    batches = 0
    for raw in loader:
        batch = move_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["input"])
        losses = criterion(
            output["prediction"], batch["target"], batch["valid_mask"], batch["weight"]
        )
        losses["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("最终训练出现非有限梯度。")
        optimizer.step()
        for name in ["total", "pixel", "gradient"]:
            totals[name] += float(losses[name].detach().cpu())
        totals["gradient_norm"] += float(gradient_norm.detach().cpu())
        batches += 1
    if batches == 0:
        raise RuntimeError("最终训练没有Batch。")
    return {key: value / batches for key, value in totals.items()} | {"batch_count": batches}


def main() -> None:
    config = load_json(CONFIG_PATH)
    if config["schema_version"] != "Stage2G_FinalTrainingProtocol_Config_v1":
        raise ValueError("Stage2G配置版本错误。")
    if config["selected_model"] != "A3_PRIOR_MODIS":
        raise RuntimeError("最终结构不是冻结的A3。")
    if config["active_channel_indices"] != [0, 1, 2, 3]:
        raise RuntimeError("A3活动通道不正确。")
    if config["test_split_access_allowed"] is not False:
        raise RuntimeError("Stage2G禁止测试集访问。")
    if config["early_stopping_allowed"] is not False:
        raise RuntimeError("最终Train+Val拟合禁止早停。")
    if int(config["fixed_training_epochs"]) != 32:
        raise RuntimeError("最终固定Epoch不是32。")
    seed = int(config["random_seed"])
    seed_everything(seed)
    root = Path(config["project_root"])
    patch_path = resolve(root, config["stage2a_patch_manifest"])
    normalization_path = (
        resolve(root, config["normalization_output_dir"])
        / "Nanyang_Stage2G_TrainValNormalizationParameters.json"
    )
    normalization_summary_path = (
        resolve(root, config["normalization_output_dir"])
        / "Nanyang_Stage2G_TrainValNormalizationSummary.csv"
    )
    stage2f_path = resolve(root, config["stage2f_audit"])
    output_dir = resolve(root, config["training_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in [patch_path, normalization_path, normalization_summary_path, stage2f_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    stage2f = load_json(stage2f_path)
    if stage2f["selected_candidate"] != "A3_PRIOR_MODIS" or stage2f["test_split_accessed"] is not False:
        raise RuntimeError("Stage2F候选或测试隔离状态错误。")
    normalization_summary = pd.read_csv(normalization_summary_path, encoding="utf-8-sig")
    if len(normalization_summary) != 1 or normalization_summary.iloc[0]["overall_status"] != "PASS":
        raise RuntimeError("Stage2G Train+Val归一化不是PASS。")
    if int(normalization_summary.iloc[0]["test_used_for_fit"]) != 0:
        raise RuntimeError("归一化使用了测试集。")
    normalization = load_json(normalization_path)

    log_path = output_dir / "00_Stage2G_FinalA3Training_RunLog.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    log = logging.getLogger("Stage2G-Training")
    dataset_kwargs = {
        "patch_manifest_path": patch_path,
        "normalization_path": normalization_path,
        "allowed_weights": tuple(config["allowed_training_weights"]),
        "weight_tolerance": float(config["weight_tolerance"]),
        "scene_cache_size": int(config["loader"]["scene_cache_size"]),
    }
    train_dataset = FinalTrainValPatchDataset(split="train", **dataset_kwargs)
    val_dataset = FinalTrainValPatchDataset(split="val", **dataset_kwargs)
    if len(train_dataset) + len(val_dataset) != int(config["expected_fit_patch_count"]):
        raise RuntimeError("Train+Val Dataset Patch总数错误。")
    combined = ConcatDataset([train_dataset, val_dataset])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        combined,
        batch_size=int(config["loader"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["loader"]["num_workers"]),
        pin_memory=bool(config["loader"]["pin_memory"]),
        worker_init_fn=worker_seed,
        generator=generator,
        persistent_workers=bool(int(config["loader"]["num_workers"]) > 0),
        drop_last=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FixedA3InputMask(
        build_model(config, normalization), config["active_channel_indices"]
    ).to(device)
    criterion = build_loss(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    log.info("Stage2G final A3 Train+Val fit starts")
    log.info("device=%s parameter_count=%d", device, parameter_count)
    log.info("train_patches=%d val_patches=%d total=%d", len(train_dataset), len(val_dataset), len(combined))
    log.info("fixed_epochs=32 early_stopping=disabled test_access=0")
    history = []
    started = time.perf_counter()
    for epoch in range(1, int(config["fixed_training_epochs"]) + 1):
        epoch_started = time.perf_counter()
        metrics = train_epoch(
            model, loader, criterion, optimizer, device,
            float(config["optimizer"]["gradient_clip_norm"]),
        )
        record = {
            "epoch": epoch,
            "train_total_loss": metrics["total"],
            "train_pixel_loss": metrics["pixel"],
            "train_gradient_loss": metrics["gradient"],
            "train_gradient_norm": metrics["gradient_norm"],
            "batch_count": metrics["batch_count"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
            "test_accessed": 0,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(
            output_dir / "Nanyang_Stage2G_FinalA3_TrainingHistory.csv",
            index=False, encoding="utf-8-sig",
        )
        log.info("epoch=%d/%d loss=%.6f", epoch, config["fixed_training_epochs"], metrics["total"])

    checkpoint_path = output_dir / "Nanyang_Stage2G_FinalA3_TrainVal_Seed42_Epoch32.pt"
    source_hashes = {
        "stage2g_config_sha256": sha256_file(CONFIG_PATH),
        "patch_manifest_sha256": sha256_file(patch_path),
        "train_val_normalization_sha256": sha256_file(normalization_path),
        "stage2f_final_audit_sha256": sha256_file(stage2f_path),
        "model_module_sha256": sha256_file(STAGE2_ROOT / "dprcn" / "model.py"),
        "loss_module_sha256": sha256_file(STAGE2_ROOT / "dprcn" / "losses.py"),
        "base_dataset_module_sha256": sha256_file(STAGE2_ROOT / "dprcn" / "dataset.py"),
        "final_dataset_module_sha256": sha256_file(STAGE2_ROOT / "dprcn" / "final_dataset.py"),
        "training_script_sha256": sha256_file(SCRIPT_PATH),
    }
    torch.save({
        "schema_version": "Stage2G_FinalA3_TrainValCheckpoint_v1",
        "selected_model": "A3_PRIOR_MODIS",
        "active_channel_indices": config["active_channel_indices"],
        "random_seed": seed,
        "fixed_epochs": int(config["fixed_training_epochs"]),
        "model_state_dict": model.state_dict(),
        "normalization": normalization,
        "config": config,
        "source_hashes": source_hashes,
        "test_split_accessed": False,
    }, checkpoint_path)
    elapsed = time.perf_counter() - started
    shutil.copy2(CONFIG_PATH, output_dir / CONFIG_PATH.name.replace(".json", "_Frozen.json"))
    summary_path = output_dir / "Nanyang_Stage2G_FinalA3_TrainingSummary.csv"
    pd.DataFrame([{
        "script_name": SCRIPT_PATH.name,
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "selected_model": "A3_PRIOR_MODIS",
        "active_channel_count": 4,
        "random_seed": seed,
        "fixed_epochs": int(config["fixed_training_epochs"]),
        "early_stopping_used": 0,
        "train_patch_count": len(train_dataset),
        "validation_patch_count_used_for_final_fit": len(val_dataset),
        "total_fit_patch_count": len(combined),
        "test_patch_count_accessed": 0,
        "parameter_count": parameter_count,
        "initial_loss": history[0]["train_total_loss"],
        "final_loss": history[-1]["train_total_loss"],
        "elapsed_seconds": elapsed,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        **source_hashes,
        "overall_status": "PASS",
    }]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    log.info("checkpoint_sha256=%s", sha256_file(checkpoint_path))
    log.info("elapsed_seconds=%.2f overall_status=PASS", elapsed)
    print("STAGE2G_FINAL_A3_TRAINING=PASS")
    print("TEST_ACCESSED=0")
    print("CHECKPOINT_SHA256=" + sha256_file(checkpoint_path))


if __name__ == "__main__":
    main()

