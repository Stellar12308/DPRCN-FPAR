from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_NAME = SCRIPT_PATH.name
STAGE2_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(STAGE2_ROOT))

from dprcn.dataset import (  # noqa: E402
    CHANNELS,
    NanyangFPARPatchDataset,
    build_dataloader,
)
from dprcn.reproducibility import seed_everything, sha256_file  # noqa: E402


CONFIG_PATH = STAGE2_ROOT / "configs" / "Nanyang_Stage2C_Dataset_Config_v1.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def tensor_sha256(sample: dict) -> str:
    digest = hashlib.sha256()
    for key in ["input", "target", "valid_mask", "weight"]:
        array = sample[key].detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def audit_sample(
    sample: dict,
    allowed_weights: list[float],
    tolerance: float,
) -> dict:
    inputs = sample["input"]
    target = sample["target"]
    mask = sample["valid_mask"]
    weight = sample["weight"]
    expected_hw = (128, 128)
    if tuple(inputs.shape) != (len(CHANNELS), *expected_hw):
        raise ValueError(f"{sample['patch_id']}输入形状错误：{tuple(inputs.shape)}")
    for name, tensor in {
        "target": target,
        "valid_mask": mask,
        "weight": weight,
    }.items():
        if tuple(tensor.shape) != (1, *expected_hw):
            raise ValueError(f"{sample['patch_id']}的{name}形状错误。")
    if any(tensor.dtype != torch.float32 for tensor in [inputs, target, mask, weight]):
        raise TypeError(f"{sample['patch_id']}存在非float32张量。")
    if not all(torch.isfinite(tensor).all().item() for tensor in [inputs, target, mask, weight]):
        raise ValueError(f"{sample['patch_id']}存在NaN或Inf。")
    valid_mask = mask > 0.5
    if not valid_mask.any():
        raise ValueError(f"{sample['patch_id']}没有有效监督像元。")
    target_valid = target[valid_mask]
    weight_valid = weight[valid_mask]
    if target_valid.min().item() < 0.0 or target_valid.max().item() > 1.0:
        raise ValueError(f"{sample['patch_id']}有效标签超出[0,1]。")
    allowed = torch.zeros_like(weight_valid, dtype=torch.bool)
    for value in allowed_weights:
        allowed |= torch.isclose(
            weight_valid,
            torch.tensor(float(value), dtype=weight_valid.dtype),
            rtol=0.0,
            atol=float(tolerance),
        )
    if not allowed.all().item():
        raise ValueError(f"{sample['patch_id']}存在非法训练权重。")
    invalid = ~valid_mask
    if not torch.all(target[invalid] == 0).item():
        raise ValueError(f"{sample['patch_id']}无效位置target未归零。")
    if not torch.all(weight[invalid] == 0).item():
        raise ValueError(f"{sample['patch_id']}无效位置weight未归零。")
    return {
        "patch_id": sample["patch_id"],
        "date": sample["date"],
        "stored_DOY": sample["stored_DOY"],
        "split": sample["split"],
        "N_pixels": int(mask.numel()),
        "N_valid_pixels": int(valid_mask.sum().item()),
        "valid_fraction": float(valid_mask.float().mean().item()),
        "target_min": float(target_valid.min().item()),
        "target_max": float(target_valid.max().item()),
        "target_mean": float(target_valid.mean().item()),
        "weight_mean": float(weight_valid.mean().item()),
        "input_min": float(inputs.min().item()),
        "input_max": float(inputs.max().item()),
        "tensor_sha256": tensor_sha256(sample),
        "status": "PASS",
    }


def main() -> None:
    config = load_json(CONFIG_PATH)
    if config.get("schema_version") != "Stage2C_Dataset_Config_v1":
        raise ValueError("Stage2C配置版本错误。")
    seed_everything(int(config["random_seed"]))
    root = Path(config["project_root"])
    base_config_path = resolve(root, config["stage2_base_config"])
    patch_path = resolve(root, config["stage2a_patch_manifest"])
    stage2a_summary_path = resolve(root, config["stage2a_summary"])
    normalization_path = resolve(root, config["stage2b_parameters"])
    stage2b_summary_path = resolve(root, config["stage2b_summary"])
    output_dir = resolve(root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "00_Stage2C_DatasetLoaderAudit_RunLog.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    log = logging.getLogger("Stage2C")
    log.info("Stage2C正式Dataset/DataLoader审计开始")
    required = [
        base_config_path,
        patch_path,
        stage2a_summary_path,
        normalization_path,
        stage2b_summary_path,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    summary_a = pd.read_csv(stage2a_summary_path, encoding="utf-8-sig")
    summary_b = pd.read_csv(stage2b_summary_path, encoding="utf-8-sig")
    if summary_a.iloc[0]["overall_status"] != "PASS":
        raise RuntimeError("Stage2A不是PASS。")
    if summary_b.iloc[0]["overall_status"] != "PASS":
        raise RuntimeError("Stage2B不是PASS。")
    if sha256_file(patch_path) != str(summary_a.iloc[0]["patch_manifest_sha256"]):
        raise RuntimeError("Stage2A Patch清单哈希不一致。")
    if sha256_file(normalization_path) != str(summary_b.iloc[0]["normalization_parameters_sha256"]):
        raise RuntimeError("Stage2B归一化参数哈希不一致。")

    datasets = {}
    audit_records = []
    split_summary = []
    patch_sets = {}
    date_sets = {}
    for split in ["train", "val", "test"]:
        dataset = NanyangFPARPatchDataset(
            patch_manifest_path=patch_path,
            normalization_path=normalization_path,
            split=split,
            allowed_weights=config["allowed_training_weights"],
            weight_tolerance=config["weight_tolerance"],
            scene_cache_size=config["scene_cache_size"],
            input_fill=config["nodata_input_fill_after_normalization"],
            target_fill=config["invalid_target_fill"],
            weight_fill=config["invalid_weight_fill"],
        )
        datasets[split] = dataset
        expected_patch_count = int(config["expected_eligible_patch_counts"][split])
        expected_scene_count = int(config["expected_scene_counts"][split])
        if len(dataset) != expected_patch_count:
            raise RuntimeError(f"{split} Patch数={len(dataset)}，预期={expected_patch_count}。")
        if dataset.table["date"].nunique() != expected_scene_count:
            raise RuntimeError(f"{split}景数不符合预期。")
        patch_sets[split] = set(dataset.table["patch_id"])
        date_sets[split] = set(dataset.table["date"])

        n_valid = 0
        for index in range(len(dataset)):
            record = audit_sample(
                dataset[index],
                [float(value) for value in config["allowed_training_weights"]],
                float(config["weight_tolerance"]),
            )
            audit_records.append(record)
            n_valid += int(record["N_valid_pixels"])

        probe_indices = sorted(set([0, len(dataset) // 2, len(dataset) - 1]))
        for index in probe_indices:
            if tensor_sha256(dataset[index]) != tensor_sha256(dataset[index]):
                raise RuntimeError(f"{split}数据重复读取不一致。")

        loader_cfg = config["audit_loader"]
        loader = build_dataloader(
            dataset,
            batch_size=int(loader_cfg["batch_size"]),
            shuffle=(split == "train"),
            seed=int(config["random_seed"]),
            num_workers=int(loader_cfg["num_workers"]),
            pin_memory=bool(loader_cfg["pin_memory"]),
        )
        first_batch = next(iter(loader))
        if first_batch["input"].ndim != 4 or first_batch["input"].shape[1] != len(CHANNELS):
            raise RuntimeError(f"{split} DataLoader批次形状错误。")
        split_summary.append({
            "split": split,
            "scene_count": dataset.table["date"].nunique(),
            "patch_count": len(dataset),
            "patch_valid_pixel_count_with_overlap": n_valid,
            "first_batch_size": int(first_batch["input"].shape[0]),
            "input_shape_per_patch": f"{len(CHANNELS)}x128x128",
            "target_shape_per_patch": "1x128x128",
            "status": "PASS",
        })
        log.info(
            "%s：景=%d，Patch=%d，重叠计数有效像元=%d",
            split,
            dataset.table["date"].nunique(),
            len(dataset),
            n_valid,
        )

    splits = ["train", "val", "test"]
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            if patch_sets[left] & patch_sets[right]:
                raise RuntimeError(f"{left}/{right}存在Patch泄漏。")
            if date_sets[left] & date_sets[right]:
                raise RuntimeError(f"{left}/{right}存在日期泄漏。")

    audit_table = pd.DataFrame(audit_records)
    if len(audit_table) != sum(config["expected_eligible_patch_counts"].values()):
        raise RuntimeError("Stage2C审计记录总数错误。")
    if not audit_table["status"].eq("PASS").all():
        raise RuntimeError("存在未通过Patch。")
    audit_path = output_dir / "Nanyang_Stage2C_PatchTensorAudit.csv"
    split_path = output_dir / "Nanyang_Stage2C_DatasetSplitSummary.csv"
    audit_table.to_csv(audit_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(split_summary).to_csv(split_path, index=False, encoding="utf-8-sig")
    shutil.copy2(CONFIG_PATH, output_dir / CONFIG_PATH.name.replace(".json", "_Frozen.json"))

    summary_path = output_dir / "Nanyang_Stage2C_DatasetLoaderAuditSummary.csv"
    pd.DataFrame([{
        "script_name": SCRIPT_NAME,
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "rasterio_version": rasterio.__version__,
        "torch_version": torch.__version__,
        "cuda_available": int(torch.cuda.is_available()),
        "channel_count": len(CHANNELS),
        "train_scene_count": len(date_sets["train"]),
        "val_scene_count": len(date_sets["val"]),
        "test_scene_count": len(date_sets["test"]),
        "train_patch_count": len(patch_sets["train"]),
        "val_patch_count": len(patch_sets["val"]),
        "test_patch_count": len(patch_sets["test"]),
        "date_overlap_count": sum(
            len(date_sets[left] & date_sets[right])
            for left, right in [("train", "val"), ("train", "test"), ("val", "test")]
        ),
        "patch_overlap_count": sum(
            len(patch_sets[left] & patch_sets[right])
            for left, right in [("train", "val"), ("train", "test"), ("val", "test")]
        ),
        "normalization_fit_split": "train",
        "stage2a_patch_manifest_sha256": sha256_file(patch_path),
        "stage2b_normalization_sha256": sha256_file(normalization_path),
        "patch_tensor_audit_sha256": sha256_file(audit_path),
        "split_summary_sha256": sha256_file(split_path),
        "overall_status": "PASS",
    }]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    log.info("日期交叉=0，Patch交叉=0")
    log.info("总审计Patch=%d", len(audit_table))
    log.info("总状态=PASS")


if __name__ == "__main__":
    main()

