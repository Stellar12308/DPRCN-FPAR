from __future__ import annotations

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
sys.path.insert(0, str(STAGE2_ROOT))

from dprcn.reproducibility import (  # noqa: E402
    canonical_json_bytes,
    seed_everything,
    sha256_bytes,
    sha256_file,
)


SCRIPT_NAME = SCRIPT_PATH.name
CONFIG_PATH = STAGE2_ROOT / "configs" / "Nanyang_Stage2_Config_v1.json"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if config.get("schema_version") != "Stage2_Config_v1":
        raise ValueError("配置版本不是Stage2_Config_v1。")
    return config


def resolve_from_root(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def axis_starts(length: int, patch: int, stride: int) -> list[int]:
    if patch <= 0 or stride <= 0:
        raise ValueError("patch和stride必须为正整数。")
    if patch > length:
        raise ValueError(f"patch={patch}大于影像轴长度={length}。")
    starts = list(range(0, length - patch + 1, stride))
    last = length - patch
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def valid(array: np.ndarray, nodata) -> np.ndarray:
    mask = np.isfinite(array)
    if nodata is not None and np.isfinite(nodata):
        mask &= array != nodata
    return mask


def open_inputs(row: pd.Series):
    keys = ["label_path", "label_qc_path", "label_weight_path", "prior_path", "modis_path", "dem_path"]
    paths = {key: Path(str(row[key])) for key in keys}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Stage1C清单存在缺失文件：\n" + "\n".join(missing))
    datasets = {key: rasterio.open(path) for key, path in paths.items()}
    reference = datasets["label_path"]
    for key, dataset in datasets.items():
        if (
            dataset.width != reference.width
            or dataset.height != reference.height
            or dataset.transform != reference.transform
            or dataset.crs != reference.crs
        ):
            raise ValueError(f"DOY{int(row['stored_DOY']):03d}的{key}网格不一致。")
    return paths, datasets


def joint_mask(datasets: dict) -> np.ndarray:
    label_ds = datasets["label_path"]
    qc_ds = datasets["label_qc_path"]
    weight_ds = datasets["label_weight_path"]
    prior_ds = datasets["prior_path"]
    modis_ds = datasets["modis_path"]
    dem_ds = datasets["dem_path"]

    label = label_ds.read(1).astype(np.float32)
    qc = qc_ds.read(1).astype(np.float32)
    weight = weight_ds.read(1).astype(np.float32)
    prior_median = prior_ds.read(1).astype(np.float32)
    prior_n = prior_ds.read(2).astype(np.float32)
    prior_sigma = prior_ds.read(3).astype(np.float32)
    modis_fpar = modis_ds.read(1).astype(np.float32)
    modis_sd = modis_ds.read(2).astype(np.float32)
    modis_qc50 = modis_ds.read(6).astype(np.float32)
    dem = dem_ds.read(1).astype(np.float32)

    return (
        valid(label, label_ds.nodata) & (label >= 0) & (label <= 1)
        & valid(qc, qc_ds.nodata) & (qc == 1)
        & valid(weight, weight_ds.nodata) & (weight > 0) & (weight <= 1)
        & valid(prior_median, prior_ds.nodata) & (prior_median >= 0) & (prior_median <= 1)
        & valid(prior_n, prior_ds.nodata) & (prior_n > 0)
        & valid(prior_sigma, prior_ds.nodata) & (prior_sigma >= 0)
        & valid(modis_fpar, modis_ds.nodata) & (modis_fpar >= 0) & (modis_fpar <= 1)
        & valid(modis_sd, modis_ds.nodata) & (modis_sd >= 0)
        & valid(modis_qc50, modis_ds.nodata) & (modis_qc50 == 1)
        & valid(dem, dem_ds.nodata)
    )


def build_tables(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    project_root = Path(config["project_root"])
    source_manifest = resolve_from_root(project_root, config["stage1c_manifest"])
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)
    scenes = pd.read_csv(source_manifest, encoding="utf-8-sig", dtype={"date": str})
    scenes["date"] = scenes["date"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(8)
    scenes["stored_DOY"] = pd.to_numeric(scenes["stored_DOY"], errors="raise").astype(int)
    expected = config["frozen_split_counts"]
    if len(scenes) != 26 or scenes["split"].value_counts().to_dict() != expected:
        raise ValueError("Stage1C清单不符合冻结的16/5/5划分。")
    if not scenes.groupby(["date", "stored_DOY"])["split"].nunique().eq(1).all():
        raise ValueError("同一日期存在多个split。")

    patch_cfg = config["patch"]
    ph, pw = int(patch_cfg["height"]), int(patch_cfg["width"])
    sy, sx = int(patch_cfg["stride_y"]), int(patch_cfg["stride_x"])
    train_min_fraction = float(patch_cfg["train_min_joint_valid_fraction"])
    eval_min_pixels = int(patch_cfg["evaluation_min_joint_valid_pixels"])
    patch_records, scene_records = [], []

    for _, row in scenes.sort_values(["split", "stored_DOY"]).iterrows():
        doy, split = int(row["stored_DOY"]), str(row["split"])
        paths, datasets = open_inputs(row)
        try:
            reference = datasets["label_path"]
            mask = joint_mask(datasets)
            y_starts = axis_starts(reference.height, ph, sy)
            x_starts = axis_starts(reference.width, pw, sx)
            covered_all = np.zeros_like(mask, dtype=bool)
            covered_eligible = np.zeros_like(mask, dtype=bool)
            scene_patch_count = 0
            eligible_count = 0

            for y in y_starts:
                for x in x_starts:
                    patch_mask = mask[y:y + ph, x:x + pw]
                    n_valid = int(patch_mask.sum())
                    n_total = ph * pw
                    fraction = n_valid / n_total
                    eligible = fraction >= train_min_fraction if split == "train" else n_valid >= eval_min_pixels
                    patch_id = f"{config['region_name']}_{split}_DOY{doy:03d}_Y{y:03d}_X{x:03d}_H{ph}_W{pw}"
                    patch_records.append({
                        "patch_id": patch_id,
                        "region": config["region_name"],
                        "date": row["date"],
                        "stored_DOY": doy,
                        "season": row["season"],
                        "split": split,
                        "in_cv_pool": row.get("in_cv_pool", ""),
                        "cv_fold_test": row.get("cv_fold_test", ""),
                        "row_off": y,
                        "col_off": x,
                        "height": ph,
                        "width": pw,
                        "N_patch_pixels": n_total,
                        "N_joint_valid_pixels": n_valid,
                        "joint_valid_fraction": round(fraction, 8),
                        "eligible_for_use": int(eligible),
                        "usage_policy": "training" if split == "train" else "evaluation_only",
                        **{key: str(path) for key, path in paths.items()},
                    })
                    scene_patch_count += 1
                    eligible_count += int(eligible)
                    covered_all[y:y + ph, x:x + pw] = True
                    if eligible:
                        covered_eligible[y:y + ph, x:x + pw] = True

            n_joint = int(mask.sum())
            n_covered = int((mask & covered_eligible).sum())
            coverage = n_covered / n_joint * 100.0 if n_joint else 0.0
            scene_records.append({
                "date": row["date"],
                "stored_DOY": doy,
                "season": row["season"],
                "split": split,
                "image_height": reference.height,
                "image_width": reference.width,
                "patch_count_all": scene_patch_count,
                "patch_count_eligible": eligible_count,
                "N_joint_scene_pixels": n_joint,
                "N_joint_pixels_covered_by_eligible_patches": n_covered,
                "joint_pixel_coverage_pct": round(coverage, 8),
                "all_image_pixels_covered_by_patch_grid": int(covered_all.all()),
                "status": "PASS" if eligible_count > 0 and coverage > 0 else "FAIL",
            })
        finally:
            for dataset in datasets.values():
                dataset.close()

    patch_table = pd.DataFrame(patch_records).sort_values(["split", "stored_DOY", "row_off", "col_off"]).reset_index(drop=True)
    scene_table = pd.DataFrame(scene_records).sort_values(["split", "stored_DOY"]).reset_index(drop=True)
    if patch_table["patch_id"].duplicated().any():
        raise ValueError("patch_id不唯一。")
    return patch_table, scene_table


def main() -> None:
    config = load_config()
    seed_everything(int(config["random_seed"]))
    project_root = Path(config["project_root"])
    output_dir = resolve_from_root(project_root, config["stage2a_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "00_Stage2A_PatchManifest_RunLog.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    log = logging.getLogger("Stage2A")
    log.info("Stage2A确定性Patch清单开始")
    log.info("脚本=%s", SCRIPT_PATH)
    log.info("配置=%s", CONFIG_PATH)

    first_patch, first_scene = build_tables(config)
    second_patch, second_scene = build_tables(config)
    if not first_patch.equals(second_patch) or not first_scene.equals(second_scene):
        raise RuntimeError("同次运行重复构建结果不同，确定性检查失败。")

    patch_path = output_dir / "Nanyang_Stage2A_DeterministicPatchManifest.csv"
    scene_path = output_dir / "Nanyang_Stage2A_ScenePatchCoverageAudit.csv"
    first_patch.to_csv(patch_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    first_scene.to_csv(scene_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    shutil.copy2(CONFIG_PATH, output_dir / "Nanyang_Stage2_Config_v1_Frozen.json")

    split_scene_counts = first_scene["split"].value_counts().to_dict()
    eligible = first_patch.loc[first_patch["eligible_for_use"] == 1]
    passed = (
        len(first_scene) == 26
        and split_scene_counts == config["frozen_split_counts"]
        and first_scene["status"].eq("PASS").all()
        and first_scene["all_image_pixels_covered_by_patch_grid"].eq(1).all()
        and first_scene.loc[first_scene["split"].isin(["val", "test"]), "joint_pixel_coverage_pct"].ge(99.999999).all()
        and not first_patch["patch_id"].duplicated().any()
    )
    hashes = {
        "config_sha256": sha256_bytes(canonical_json_bytes(config)),
        "source_stage1c_manifest_sha256": sha256_file(resolve_from_root(project_root, config["stage1c_manifest"])),
        "patch_manifest_sha256": sha256_file(patch_path),
        "scene_audit_sha256": sha256_file(scene_path),
    }
    summary = pd.DataFrame([{
        "script_name": SCRIPT_NAME,
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "rasterio_version": rasterio.__version__,
        "random_seed": config["random_seed"],
        "patch_height": config["patch"]["height"],
        "patch_width": config["patch"]["width"],
        "stride_y": config["patch"]["stride_y"],
        "stride_x": config["patch"]["stride_x"],
        "scene_count": len(first_scene),
        "patch_count_all": len(first_patch),
        "patch_count_eligible": len(eligible),
        "train_patch_count_eligible": int((eligible["split"] == "train").sum()),
        "val_patch_count_eligible": int((eligible["split"] == "val").sum()),
        "test_patch_count_eligible": int((eligible["split"] == "test").sum()),
        "minimum_eval_joint_coverage_pct": float(first_scene.loc[first_scene["split"].isin(["val", "test"]), "joint_pixel_coverage_pct"].min()),
        **hashes,
        "overall_status": "PASS" if passed else "FAIL",
    }])
    summary_path = output_dir / "Nanyang_Stage2A_PatchManifestSummary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    with (output_dir / "Nanyang_Stage2A_SHA256.json").open("w", encoding="utf-8") as file:
        json.dump(hashes, file, ensure_ascii=False, indent=2)
    log.info("场景=%d，全部Patch=%d，可用Patch=%d", len(first_scene), len(first_patch), len(eligible))
    log.info("train/val/test可用Patch=%d/%d/%d", int((eligible["split"] == "train").sum()), int((eligible["split"] == "val").sum()), int((eligible["split"] == "test").sum()))
    log.info("Patch manifest SHA256=%s", hashes["patch_manifest_sha256"])
    log.info("总状态=%s", "PASS" if passed else "FAIL")
    if not passed:
        raise RuntimeError("Stage2A未通过，禁止进入Stage2B。")


if __name__ == "__main__":
    main()

