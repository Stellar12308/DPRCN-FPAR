from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


SCRIPT_PATH = Path(__file__).resolve()
STAGE3_ROOT = SCRIPT_PATH.parents[1]
CONFIG_PATH = STAGE3_ROOT / "configs" / "Nanyang_Stage3D_EvaluateESTARFM_Config_v1.json"
STAGE2H_SCRIPT = (
    STAGE3_ROOT.parent
    / "03_Stage2_DPRCN"
    / "scripts"
    / "Nanyang_Stage2H_EvaluateFinalA3_OneTimeTest.py"
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stage2h_metrics():
    spec = importlib.util.spec_from_file_location("stage2h_frozen_metrics", STAGE2H_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载Stage2H指标实现：{STAGE2H_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.metrics, module.masked_ssim


def read_raster(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        valid = np.isfinite(array)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= array != float(src.nodata)
    return array, valid, profile


def same_grid(left: dict, right: dict) -> bool:
    return (
        left["width"] == right["width"]
        and left["height"] == right["height"]
        and left["crs"] == right["crs"]
        and left["transform"] == right["transform"]
    )


def main() -> None:
    started = time.perf_counter()
    cfg = read_json(CONFIG_PATH)
    if cfg.get("schema_version") != "Stage3D_EvaluateESTARFM_Config_v1":
        raise ValueError("Stage3D配置版本错误。")
    root = Path(cfg["project_root"])
    manifest_path = root / cfg["reference_manifest"]
    prediction_dir = root / cfg["prediction_dir"]
    audit_path = root / cfg["stage3c_audit"]
    output = root / cfg["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(output / "00_Stage3D_ESTARFM_Evaluation_RunLog.txt", mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    log = logging.getLogger("stage3d")
    metrics, masked_ssim = load_stage2h_metrics()
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", dtype=str)
    audit = pd.read_csv(audit_path, encoding="utf-8-sig")
    expected = int(cfg["expected_test_date_count"])
    if len(manifest) != expected or len(audit) != expected:
        raise ValueError("Stage3A或Stage3C记录数不是5。")
    if not manifest["test_label_used_as_input"].astype(int).eq(0).all():
        raise RuntimeError("检测到测试标签输入泄漏。")
    if not audit["status"].eq("PASS").all():
        raise RuntimeError("Stage3C存在未通过场景。")

    low, high = map(float, cfg["valid_range"])
    rows, all_predictions, all_targets = [], [], []
    prediction_hashes = {}
    for _, row in manifest.iterrows():
        date = str(row["target_date"])
        prediction_path = prediction_dir / cfg["prediction_pattern"].format(date=date)
        target_path = Path(row["target_label_evaluation_only"])
        if not prediction_path.exists() or not target_path.exists():
            raise FileNotFoundError(f"缺少预测或标签：{date}")
        prediction, p_valid, p_profile = read_raster(prediction_path)
        target, y_valid, y_profile = read_raster(target_path)
        if not same_grid(p_profile, y_profile):
            raise ValueError(f"预测与标签网格不一致：{date}")
        valid = p_valid & y_valid
        valid &= (prediction >= low) & (prediction <= high)
        valid &= (target >= low) & (target <= high)
        if not valid.any():
            raise ValueError(f"没有共同有效像元：{date}")
        result = metrics(prediction[valid], target[valid])
        ssim, n_windows = masked_ssim(prediction, target, valid, cfg["ssim"])
        rows.append({
            "date": date,
            "stored_DOY": int(row["target_DOY"]),
            **result,
            "SSIM": ssim,
            "N_SSIM_windows": n_windows,
            "evaluation_mask_policy": cfg["evaluation_mask_policy"],
        })
        all_predictions.append(prediction[valid])
        all_targets.append(target[valid])
        prediction_hashes[date] = sha256(prediction_path)
        log.info("date=%s N=%d RMSE=%.6f R=%.6f SSIM=%.6f", date, result["N_valid_pixels"], result["RMSE"], result["Pearson_R"], ssim)

    date_table = pd.DataFrame(rows).sort_values("stored_DOY").reset_index(drop=True)
    global_result = metrics(np.concatenate(all_predictions), np.concatenate(all_targets))
    summary = {
        "script_name": SCRIPT_PATH.name,
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "model": "ESTARFM",
        "test_date_count": len(date_table),
        **{f"global_{key}": value for key, value in global_result.items()},
        "date_macro_RMSE": float(date_table["RMSE"].mean()),
        "date_macro_MAE": float(date_table["MAE"].mean()),
        "date_macro_absolute_Bias": float(date_table["Bias"].abs().mean()),
        "date_macro_Bias": float(date_table["Bias"].mean()),
        "date_macro_Pearson_R": float(date_table["Pearson_R"].mean()),
        "date_macro_R2": float(date_table["R2"].mean()),
        "date_macro_ubRMSE": float(date_table["ubRMSE"].mean()),
        "date_macro_SSIM": float(date_table["SSIM"].mean()),
        "evaluation_mask_policy": cfg["evaluation_mask_policy"],
        "strict_common_mask_comparison_completed": 0,
        "config_sha256": sha256(CONFIG_PATH),
        "script_sha256": sha256(SCRIPT_PATH),
        "manifest_sha256": sha256(manifest_path),
        "stage3c_audit_sha256": sha256(audit_path),
        "elapsed_seconds": time.perf_counter() - started,
        "overall_status": "PASS",
    }
    date_path = output / "Nanyang_Stage3D_ESTARFM_TestDateMetrics.csv"
    overall_path = output / "Nanyang_Stage3D_ESTARFM_TestOverallMetrics.csv"
    date_table.to_csv(date_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(overall_path, index=False, encoding="utf-8-sig")
    evidence = {
        "config": cfg,
        "summary": summary,
        "prediction_sha256": prediction_hashes,
        "date_metrics": rows,
        "important_interpretation": cfg["comparison_note"],
    }
    with (output / "00_Stage3D_ESTARFM_Evaluation_Evidence.json").open("w", encoding="utf-8") as file:
        json.dump(evidence, file, ensure_ascii=False, indent=2)
    log.info("date_macro_RMSE=%.6f global_RMSE=%.6f macro_SSIM=%.6f", summary["date_macro_RMSE"], summary["global_RMSE"], summary["date_macro_SSIM"])
    log.info("STAGE3D_ESTARFM_EVALUATION=PASS")
    print("STAGE3D_ESTARFM_EVALUATION=PASS")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()

