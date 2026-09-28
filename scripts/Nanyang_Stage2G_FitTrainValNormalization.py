from __future__ import annotations

import importlib.util
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


SCRIPT_PATH = Path(__file__).resolve()
STAGE2_ROOT = SCRIPT_PATH.parents[1]
CONFIG_PATH = STAGE2_ROOT / "configs" / "Nanyang_Stage2G_FinalTrainingProtocol_Config_v1.json"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("stage2b_normalization", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法载入：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    config = load_json(CONFIG_PATH)
    if config["schema_version"] != "Stage2G_FinalTrainingProtocol_Config_v1":
        raise ValueError("Stage2G配置版本错误。")
    if config["test_split_access_allowed"] is not False:
        raise RuntimeError("Stage2G必须禁止测试集访问。")
    fit_splits = set(config["normalization_fit_splits"])
    if fit_splits != {"train", "val"}:
        raise RuntimeError("Stage2G归一化拟合集必须恰好为train+val。")
    stage2b = load_module(
        STAGE2_ROOT / "scripts" / "Nanyang_Stage2B_FitTrainOnlyNormalization.py"
    )
    root = Path(config["project_root"])
    patch_path = resolve(root, config["stage2a_patch_manifest"])
    summary_path = resolve(root, config["stage2a_summary"])
    stage2f_path = resolve(root, config["stage2f_audit"])
    output_dir = resolve(root, config["normalization_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "00_Stage2G_TrainValNormalization_RunLog.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    log = logging.getLogger("Stage2G-Normalization")
    for path in [patch_path, summary_path, stage2f_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    stage2f = load_json(stage2f_path)
    if stage2f["overall_status"] != "PASS" or stage2f["selected_candidate"] != "A3_PRIOR_MODIS":
        raise RuntimeError("Stage2F尚未PASS或候选模型不是A3。")
    if stage2f["test_split_accessed"] is not False:
        raise RuntimeError("Stage2F测试集隔离失败。")
    stage2a_summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    if len(stage2a_summary) != 1 or stage2a_summary.iloc[0]["overall_status"] != "PASS":
        raise RuntimeError("Stage2A不是PASS。")
    patch_hash = stage2b.sha256_file(patch_path)
    if patch_hash != str(stage2a_summary.iloc[0]["patch_manifest_sha256"]):
        raise RuntimeError("Patch清单哈希不一致。")
    patches = pd.read_csv(patch_path, encoding="utf-8-sig", dtype={"date": str})
    patches["date"] = patches["date"].astype(str).str.zfill(8)
    fit = patches.loc[
        patches["split"].isin(fit_splits) & patches["eligible_for_use"].eq(1)
    ].copy()
    if len(fit) != int(config["expected_fit_patch_count"]):
        raise RuntimeError(f"Train+Val Patch数={len(fit)}，预期={config['expected_fit_patch_count']}。")
    if fit["date"].nunique() != int(config["expected_fit_scene_count"]):
        raise RuntimeError("Train+Val景数不符合冻结配置。")
    if patches.loc[patches["split"] == "test", "date"].isin(fit["date"]).any():
        raise RuntimeError("测试日期进入了最终拟合集。")

    states = {
        channel: {"count": 0, "mean": 0.0, "m2": 0.0, "minimum": np.inf, "maximum": -np.inf}
        for channel in stage2b.CHANNELS
    }
    audit_records = []
    for (date, doy, split), rows in fit.groupby(["date", "stored_DOY", "split"], sort=True):
        arrays, mask, metadata = stage2b.scene_arrays(
            rows,
            [float(value) for value in config["allowed_training_weights"]],
            float(config["weight_tolerance"]),
            config["terrain"],
        )
        if not mask.any():
            raise RuntimeError(f"{date}没有最终归一化拟合像元。")
        record = {
            "date": date, "stored_DOY": int(doy), "original_split": split,
            **metadata, "status": "PASS",
        }
        for channel in stage2b.CHANNELS:
            values = stage2b.transform_channel(
                channel, arrays[channel], config["normalization"]
            )[mask]
            stage2b.update_moments(states[channel], values)
            record[f"{channel}_mean_transformed"] = float(np.mean(values))
            record[f"{channel}_std_transformed"] = float(np.std(values, ddof=0))
        audit_records.append(record)
        log.info("date=%s split=%s fit_pixels=%d", date, split, int(mask.sum()))

    parameters = {}
    counts = set()
    for channel, state in states.items():
        count = int(state["count"])
        std = float(np.sqrt(max(float(state["m2"]) / count, 0.0)))
        if count < 2 or not np.isfinite(std) or std <= float(config["minimum_standard_deviation"]):
            raise RuntimeError(f"{channel}归一化统计无效。")
        counts.add(count)
        parameters[channel] = {
            "source_transform": config["normalization"][channel],
            "normalization": "z_score",
            "count": count,
            "mean": float(state["mean"]),
            "std_population": std,
            "minimum": float(state["minimum"]),
            "maximum": float(state["maximum"]),
        }
    if len(counts) != 1 or len(audit_records) != int(config["expected_fit_scene_count"]):
        raise RuntimeError("归一化通道像元数或景数不一致。")
    audit_path = output_dir / "Nanyang_Stage2G_TrainValNormalization_SceneAudit.csv"
    pd.DataFrame(audit_records).sort_values("stored_DOY").to_csv(
        audit_path, index=False, encoding="utf-8-sig"
    )
    document = {
        "schema_version": "Stage2G_TrainValNormalization_v1",
        "region_name": "Nanyang",
        "fit_split": "train_val",
        "fit_splits": ["train", "val"],
        "fit_scene_count": len(audit_records),
        "fit_patch_count": len(fit),
        "fit_pixel_count_unique_per_scene": next(iter(counts)),
        "leakage_guard": {
            "validation_used_for_final_fit": True,
            "test_used_for_fit": False,
            "overlapping_patch_duplicates_counted": False,
            "policy": config["normalization"]["pixel_counting_policy"],
        },
        "terrain_policy": config["terrain"],
        "channel_order": stage2b.CHANNELS,
        "parameters": parameters,
        "source_hashes": {
            "stage2g_config_sha256": stage2b.sha256_file(CONFIG_PATH),
            "stage2a_patch_manifest_sha256": patch_hash,
            "stage2f_final_audit_sha256": stage2b.sha256_file(stage2f_path),
            "normalization_source_script_sha256": stage2b.sha256_file(SCRIPT_PATH),
        },
    }
    parameter_path = output_dir / "Nanyang_Stage2G_TrainValNormalizationParameters.json"
    parameter_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(CONFIG_PATH, output_dir / CONFIG_PATH.name.replace(".json", "_Frozen.json"))
    pd.DataFrame([{
        "script_name": SCRIPT_PATH.name,
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "rasterio_version": rasterio.__version__,
        "fit_split": "train_val",
        "fit_scene_count": len(audit_records),
        "fit_patch_count": len(fit),
        "fit_pixel_count": next(iter(counts)),
        "validation_used_for_final_fit": 1,
        "test_used_for_fit": 0,
        "patch_overlap_duplicates_counted": 0,
        "channel_count": len(stage2b.CHANNELS),
        "normalization_parameters_sha256": stage2b.sha256_file(parameter_path),
        "scene_audit_sha256": stage2b.sha256_file(audit_path),
        "overall_status": "PASS",
    }]).to_csv(
        output_dir / "Nanyang_Stage2G_TrainValNormalizationSummary.csv",
        index=False, encoding="utf-8-sig",
    )
    log.info("fit_scene_count=%d", len(audit_records))
    log.info("fit_patch_count=%d", len(fit))
    log.info("fit_pixel_count=%d", next(iter(counts)))
    log.info("validation_used_for_final_fit=1, test_used_for_fit=0")
    log.info("overall_status=PASS")
    print("STAGE2G_TRAINVAL_NORMALIZATION=PASS")
    print("TEST_USED_FOR_FIT=0")
    print("PARAMETERS_SHA256=" + stage2b.sha256_file(parameter_path))


if __name__ == "__main__":
    main()

