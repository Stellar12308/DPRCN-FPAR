from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


SCRIPT_PATH = Path(__file__).resolve()
ROOT3 = SCRIPT_PATH.parents[1]
CONFIG_PATH = ROOT3 / "configs" / "Nanyang_Stage3C_ESTARFM_Config_v1.json"
CORE_DIR = ROOT3 / "estarfm"
sys.path.insert(0, str(CORE_DIR))
from estarfm_legacy_adapted import run_estarfm_core  # noqa: E402


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fine(path: Path, nodata_out: float) -> tuple[np.ndarray, dict, np.ndarray]:
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        valid = np.isfinite(array)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= array != src.nodata
        valid &= (array >= 0.0) & (array <= 1.0)
    return np.where(valid, array, nodata_out), profile, valid


def read_modis(path: Path, cfg: dict, reference_profile: dict) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        if src.count < int(cfg["modis_valid_mask_band"]):
            raise ValueError(f"MODIS波段不足：{path}")
        if src.crs != reference_profile["crs"] or src.transform != reference_profile["transform"]:
            raise ValueError(f"MODIS网格不一致：{path}")
        if src.width != reference_profile["width"] or src.height != reference_profile["height"]:
            raise ValueError(f"MODIS形状不一致：{path}")
        fpar = src.read(int(cfg["modis_fpar_band"])).astype(np.float32)
        mask = src.read(int(cfg["modis_valid_mask_band"]))
        valid = np.isfinite(fpar) & (fpar >= 0.0) & (fpar <= 1.0)
        valid &= mask == int(cfg["required_modis_mask_value"])
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= fpar != src.nodata
    return np.where(valid, fpar, float(cfg["nodata"])), valid


def write_single(path: Path, array: np.ndarray, profile: dict, nodata: float) -> None:
    metadata = profile.copy()
    metadata.update(count=1, dtype="float32", nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **metadata) as dst:
        dst.write(array.astype(np.float32), 1)


def main() -> None:
    cfg = read_json(CONFIG_PATH)
    root = Path(cfg["project_root"])
    manifest_path = root / cfg["reference_manifest"]
    output = root / cfg["output_dir"]
    temporary = root / cfg["temporary_dir"]
    output.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(output / "00_Stage3C_ESTARFM_RunLog.txt", mode="w", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    table = pd.read_csv(manifest_path, encoding="utf-8-sig", dtype=str)
    if len(table) != int(cfg["expected_scene_count"]):
        raise ValueError("ESTARFM测试场景数不是5。")
    if not table["test_label_used_as_input"].astype(int).eq(0).all():
        raise RuntimeError("Stage3A存在测试标签输入泄漏。")
    records = []
    for _, row in table.iterrows():
        started = time.perf_counter()
        date = str(row["target_date"])
        scene_temp = temporary / date
        if scene_temp.exists():
            shutil.rmtree(scene_temp)
        scene_temp.mkdir(parents=True)
        fine1, profile, fine1_valid = read_fine(Path(row["previous_fine_label_input"]), float(cfg["nodata"]))
        fine2, profile2, fine2_valid = read_fine(Path(row["next_fine_label_input"]), float(cfg["nodata"]))
        if profile2["crs"] != profile["crs"] or profile2["transform"] != profile["transform"] or fine2.shape != fine1.shape:
            raise ValueError(f"{date}两幅精细参考网格不一致。")
        coarse1, coarse1_valid = read_modis(Path(row["previous_modis_input"]), cfg, profile)
        coarse2, coarse2_valid = read_modis(Path(row["next_modis_input"]), cfg, profile)
        coarse0, coarse0_valid = read_modis(Path(row["target_modis_input"]), cfg, profile)
        joint = fine1_valid & fine2_valid & coarse1_valid & coarse2_valid & coarse0_valid
        arrays = {"F1": fine1, "C1": coarse1, "F2": fine2, "C2": coarse2, "C0": coarse0}
        paths = {}
        for name, array in arrays.items():
            prepared = np.where(joint, array, float(cfg["nodata"]))
            paths[name] = scene_temp / f"{name}.tif"
            write_single(paths[name], prepared, profile, float(cfg["nodata"]))
        engine_dir = scene_temp / "engine"
        result_path = Path(run_estarfm_core(
            str(paths["F1"]), str(paths["C1"]), str(paths["F2"]),
            str(paths["C2"]), str(paths["C0"]), str(engine_dir)
        ))
        with rasterio.open(result_path) as src:
            raw = src.read(1).astype(np.float32)
        valid_output = joint & np.isfinite(raw) & (raw != float(cfg["nodata"]))
        raw_output = np.where(valid_output, raw, float(cfg["nodata"]))
        clipped = np.where(valid_output, np.clip(raw, 0.0, 1.0), float(cfg["nodata"]))
        raw_path = output / f"Nanyang_{date}_ESTARFM_raw.tif"
        clipped_path = output / f"Nanyang_{date}_ESTARFM_FPAR.tif"
        write_single(raw_path, raw_output, profile, float(cfg["nodata"]))
        write_single(clipped_path, clipped, profile, float(cfg["nodata"]))
        raw_valid = raw[valid_output]
        records.append({
            "target_date": date,
            "previous_date": row["previous_date"], "next_date": row["next_date"],
            "previous_gap_days": int(row["previous_gap_days"]), "next_gap_days": int(row["next_gap_days"]),
            "N_grid_pixels": int(raw.size), "N_joint_input_valid": int(joint.sum()),
            "N_output_valid": int(valid_output.sum()), "output_coverage_pct": float(valid_output.mean() * 100.0),
            "raw_below_0_pct": float(np.mean(raw_valid < 0.0) * 100.0) if raw_valid.size else np.nan,
            "raw_above_1_pct": float(np.mean(raw_valid > 1.0) * 100.0) if raw_valid.size else np.nan,
            "clipped_le_0p01_pct": float(np.mean(np.clip(raw_valid, 0, 1) <= 0.01) * 100.0) if raw_valid.size else np.nan,
            "clipped_ge_0p99_pct": float(np.mean(np.clip(raw_valid, 0, 1) >= 0.99) * 100.0) if raw_valid.size else np.nan,
            "raw_prediction_path": str(raw_path), "prediction_path": str(clipped_path),
            "raw_sha256": sha256(raw_path), "prediction_sha256": sha256(clipped_path),
            "elapsed_seconds": time.perf_counter() - started, "status": "PASS",
        })
        logging.info("date=%s valid=%d coverage=%.2f%%", date, int(valid_output.sum()), valid_output.mean() * 100.0)
        shutil.rmtree(scene_temp)
    audit = pd.DataFrame(records)
    audit.to_csv(output / "Nanyang_Stage3C_ESTARFM_SceneAudit.csv", index=False, encoding="utf-8-sig")
    evidence = {
        "schema_version": "Stage3C_ESTARFM_Evidence_v1",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scene_count": len(audit), "test_label_used_as_input": False,
        "manifest_sha256": sha256(manifest_path), "config_sha256": sha256(CONFIG_PATH),
        "driver_sha256": sha256(SCRIPT_PATH),
        "core_sha256": sha256(CORE_DIR / "estarfm_legacy_adapted.py"),
        "overall_status": "PASS" if len(audit) == 5 and audit["status"].eq("PASS").all() else "FAIL",
    }
    (output / "00_Stage3C_ESTARFM_Evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("STAGE3C=%s", evidence["overall_status"])
    print("STAGE3C_ESTARFM=" + evidence["overall_status"])


if __name__ == "__main__":
    main()

