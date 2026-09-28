from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


SCRIPT_PATH = Path(__file__).resolve()
STAGE3_ROOT = SCRIPT_PATH.parents[1]
CONFIG_PATH = STAGE3_ROOT / "configs" / "Nanyang_Stage3E_AdaptedPKSTEM_Config_v1.json"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_array(src: rasterio.DatasetReader, band: int, low: float, high: float):
    array = src.read(band).astype(np.float32)
    valid = np.isfinite(array) & (array >= low) & (array <= high)
    if src.nodata is not None and np.isfinite(src.nodata):
        valid &= array != float(src.nodata)
    return array, valid


def read_prior(path: Path, cfg: dict):
    with rasterio.open(path) as src:
        mean, valid = valid_array(src, int(cfg["prior_mean_band"]), 0.0, 1.0)
        uncertainty = src.read(int(cfg["prior_uncertainty_band"])).astype(np.float32)
        uncertainty = np.where(np.isfinite(uncertainty) & (uncertainty >= 0), uncertainty, np.nan)
        profile = src.profile.copy()
    return mean, uncertainty, valid, profile


def read_label(path: Path, reference: dict):
    with rasterio.open(path) as src:
        if not same_grid(src.profile, reference):
            raise ValueError(f"标签网格不一致：{path}")
        return valid_array(src, 1, 0.0, 1.0)[:2]


def read_modis_30m(path: Path, cfg: dict, reference: dict):
    with rasterio.open(path) as src:
        if not same_grid(src.profile, reference):
            raise ValueError(f"MODIS 30m网格不一致：{path}")
        fpar, valid = valid_array(src, int(cfg["modis_fpar_band_30m"]), 0.0, 1.0)
        valid &= src.read(int(cfg["modis_valid_mask_band_30m"])) == 1
    return fpar, valid


def same_grid(left: dict, right: dict) -> bool:
    return (
        left["width"] == right["width"]
        and left["height"] == right["height"]
        and left["crs"] == right["crs"]
        and left["transform"] == right["transform"]
    )


def native_modis_correction(prior_path: Path, native_path: Path, cfg: dict):
    with rasterio.open(prior_path) as fine, rasterio.open(native_path) as coarse:
        prior = fine.read(int(cfg["prior_mean_band"])).astype(np.float32)
        aggregated = np.full((coarse.height, coarse.width), np.nan, dtype=np.float32)
        reproject(
            source=prior,
            destination=aggregated,
            src_transform=fine.transform,
            src_crs=fine.crs,
            src_nodata=fine.nodata,
            dst_transform=coarse.transform,
            dst_crs=coarse.crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
        modis, valid = valid_array(coarse, int(cfg["modis_fpar_band_native"]), 0.0, 1.0)
        valid &= coarse.read(int(cfg["modis_valid_mask_band_native"])) == 1
        valid &= np.isfinite(aggregated) & (aggregated >= 0.0) & (aggregated <= 1.0)
    n = int(valid.sum())
    if n < int(cfg["minimum_native_regression_cells"]):
        return 1.0, 0.0, n, np.nan
    x = aggregated[valid].astype(np.float64)
    y = modis[valid].astype(np.float64)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    slope = float(np.dot(x_centered, y - y.mean()) / denominator) if denominator > 1e-12 else 1.0
    intercept = float(y.mean() - slope * x.mean())
    slope = float(np.clip(slope, *map(float, cfg["slope_bounds"])))
    intercept = float(np.clip(intercept, *map(float, cfg["intercept_bounds"])))
    residual = y - (slope * x + intercept)
    return slope, intercept, n, float(np.sqrt(np.mean(residual * residual)))


def robust_sigma(values: list[np.ndarray], fallback: float, low: float, high: float) -> float:
    if not values:
        return fallback
    sample = np.concatenate(values)
    if sample.size > 2_000_000:
        sample = sample[:: int(np.ceil(sample.size / 2_000_000))]
    median = float(np.median(sample))
    sigma = 1.4826 * float(np.median(np.abs(sample - median)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = fallback
    return float(np.clip(sigma, low, high))


def run_filter(records: list[dict], cfg: dict, reverse: bool, seed: int, observation_sigma: float):
    order = list(range(len(records)))
    if reverse:
        order.reverse()
    rng = np.random.default_rng(seed)
    ensemble = None
    previous_center = None
    outputs = {}
    n_ensemble = int(cfg["ensemble_size"])
    low, high = float(cfg["clip_min"]), float(cfg["clip_max"])
    for index in order:
        item = records[index]
        center = item["corrected_prior"]
        valid = item["state_valid"]
        q = np.clip(
            np.nan_to_num(item["prior_uncertainty"], nan=observation_sigma)
            * float(cfg["prior_uncertainty_scale"]),
            float(cfg["minimum_process_sigma"]),
            float(cfg["maximum_process_sigma"]),
        ).astype(np.float32)
        if ensemble is None:
            ensemble = center[None] + rng.normal(size=(n_ensemble,) + center.shape).astype(np.float32) * q[None]
        else:
            anomaly = ensemble - previous_center[None]
            denominator = np.maximum(np.abs(previous_center), float(cfg["epsilon"]))
            ratio = np.clip(center / denominator, 0.2, 5.0)
            ensemble = center[None] + anomaly * ratio[None]
            ensemble += rng.normal(size=ensemble.shape).astype(np.float32) * q[None]
        ensemble = np.clip(ensemble, low, high)
        ensemble[:, ~valid] = np.nan
        if item["is_assimilation"]:
            observation = item["label"]
            obs_valid = item["label_valid"] & valid
            forecast_variance = np.full(center.shape, np.nan, dtype=np.float32)
            forecast_variance[valid] = np.var(ensemble[:, valid], axis=0, ddof=1)
            gain = forecast_variance / (forecast_variance + observation_sigma**2)
            noise = rng.normal(0.0, observation_sigma, size=ensemble.shape).astype(np.float32)
            innovation = observation[None] + noise - ensemble
            ensemble[:, obs_valid] += gain[None, obs_valid] * innovation[:, obs_valid]
            ensemble = np.clip(ensemble, low, high)
        filtered_mean = np.full(center.shape, np.nan, dtype=np.float32)
        filtered_mean[valid] = np.mean(ensemble[:, valid], axis=0)
        outputs[index] = filtered_mean
        previous_center = center
    return outputs


def write_prediction(path: Path, array: np.ndarray, valid: np.ndarray, profile: dict, cfg: dict):
    nodata = float(cfg["nodata"])
    metadata = profile.copy()
    metadata.update(count=1, dtype=cfg["output_dtype"], nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **metadata) as dst:
        dst.write(np.where(valid, array, nodata).astype(np.float32), 1)


def main() -> None:
    started = time.perf_counter()
    cfg = read_json(CONFIG_PATH)
    if cfg.get("schema_version") != "Stage3E_AdaptedPKSTEM_Config_v1":
        raise ValueError("Stage3E配置版本错误。")
    root = Path(cfg["project_root"])
    split_path = root / cfg["split_manifest"]
    output = root / cfg["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(output / "00_Stage3E_AdaptedPKSTEM_RunLog.txt", mode="w", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    log = logging.getLogger("stage3e")
    table = pd.read_csv(split_path, encoding="utf-8-sig")
    table["stored_DOY"] = table["stored_DOY"].astype(int)
    table = table.sort_values("stored_DOY").reset_index(drop=True)
    if len(table) != int(cfg["expected_date_count"]):
        raise ValueError("2023年日期数不是26。")
    if int(table["split"].isin(["train", "val"]).sum()) != int(cfg["expected_assimilation_count"]):
        raise ValueError("同化日期数不是21。")
    if int(table["split"].eq("test").sum()) != int(cfg["expected_test_count"]):
        raise ValueError("测试日期数不是5。")

    records, correction_rows, residuals = [], [], []
    reference_profile = None
    for _, row in table.iterrows():
        date, doy, split = str(row["date"]), int(row["stored_DOY"]), str(row["split"])
        prior_path = root / cfg["prior_dir"] / f"Prior_30m_DOY{doy:03d}.tif"
        modis30_path = root / cfg["modis_30m_dir"] / f"Nanyang_{date}_DOY{doy:03d}_MODIS_30m_Stage1B.tif"
        native_path = root / cfg["modis_native_dir"] / f"Nanyang_{date}_DOY{doy:03d}_MODIS_Native500m_Stage1B.tif"
        label_path = root / cfg["label_dir"] / f"Label_DOY{doy:03d}.tif"
        for required in [prior_path, modis30_path, native_path]:
            if not required.exists():
                raise FileNotFoundError(required)
        prior, uncertainty, prior_valid, profile = read_prior(prior_path, cfg)
        if reference_profile is None:
            reference_profile = profile
        elif not same_grid(profile, reference_profile):
            raise ValueError(f"先验网格不一致：{date}")
        modis30, modis_valid = read_modis_30m(modis30_path, cfg, profile)
        slope, intercept, n_cells, regression_rmse = native_modis_correction(prior_path, native_path, cfg)
        corrected = np.clip(slope * prior + intercept, float(cfg["clip_min"]), float(cfg["clip_max"])).astype(np.float32)
        state_valid = prior_valid & modis_valid & np.isfinite(corrected)
        item = {
            "date": date, "doy": doy, "split": split,
            "corrected_prior": corrected, "prior_uncertainty": uncertainty,
            "state_valid": state_valid, "is_assimilation": split in {"train", "val"},
            "label": None, "label_valid": None, "profile": profile,
        }
        if item["is_assimilation"]:
            if not label_path.exists():
                raise FileNotFoundError(label_path)
            label, label_valid = read_label(label_path, profile)
            item["label"], item["label_valid"] = label, label_valid
            common = state_valid & label_valid
            if common.any():
                residuals.append((label[common] - corrected[common]).astype(np.float32))
        elif split == "test":
            # 防泄漏：预测阶段不检查、不打开测试标签。
            item["label"], item["label_valid"] = None, None
        records.append(item)
        correction_rows.append({
            "date": date, "stored_DOY": doy, "split": split,
            "native_regression_cells": n_cells, "modis_correction_slope": slope,
            "modis_correction_intercept": intercept, "native_regression_RMSE": regression_rmse,
            "N_state_valid": int(state_valid.sum()), "test_label_opened": 0,
        })

    observation_sigma = robust_sigma(
        residuals,
        float(cfg["fallback_observation_sigma"]),
        float(cfg["minimum_observation_sigma"]),
        float(cfg["maximum_observation_sigma"]),
    )
    log.info("observation_sigma_from_train_val=%.6f", observation_sigma)
    forward = run_filter(records, cfg, False, int(cfg["random_seed"]), observation_sigma)
    backward = run_filter(records, cfg, True, int(cfg["random_seed"]) + 1000, observation_sigma)
    assimilation_doys = sorted(int(r["doy"]) for r in records if r["is_assimilation"])
    scene_rows, output_hashes = [], {}
    for index, item in enumerate(records):
        if item["split"] != "test":
            continue
        doy, date = int(item["doy"]), item["date"]
        previous = max(x for x in assimilation_doys if x < doy)
        following = min(x for x in assimilation_doys if x > doy)
        previous_gap, next_gap = doy - previous, following - doy
        forward_weight = next_gap / (previous_gap + next_gap)
        backward_weight = previous_gap / (previous_gap + next_gap)
        prediction = forward_weight * forward[index] + backward_weight * backward[index]
        valid = item["state_valid"] & np.isfinite(prediction)
        prediction = np.clip(prediction, float(cfg["clip_min"]), float(cfg["clip_max"]))
        path = output / f"Nanyang_{date}_AdaptedPKSTEM_ModeIII_FPAR.tif"
        write_prediction(path, prediction, valid, item["profile"], cfg)
        output_hashes[date] = sha256(path)
        scene_rows.append({
            "target_date": date, "stored_DOY": doy,
            "previous_assimilation_DOY": previous, "next_assimilation_DOY": following,
            "previous_gap_days": previous_gap, "next_gap_days": next_gap,
            "forward_weight": forward_weight, "backward_weight": backward_weight,
            "N_output_valid": int(valid.sum()), "output_coverage_pct": float(valid.mean() * 100.0),
            "prediction_mean": float(np.mean(prediction[valid])), "prediction_std": float(np.std(prediction[valid])),
            "prediction_le_0p01_pct": float(np.mean(prediction[valid] <= 0.01) * 100.0),
            "prediction_ge_0p99_pct": float(np.mean(prediction[valid] >= 0.99) * 100.0),
            "test_label_opened": 0, "prediction_path": str(path), "prediction_sha256": output_hashes[date], "status": "PASS",
        })
        log.info("date=%s valid=%d coverage=%.2f%%", date, int(valid.sum()), float(valid.mean() * 100.0))

    correction_table = pd.DataFrame(correction_rows)
    scene_table = pd.DataFrame(scene_rows)
    if len(scene_table) != int(cfg["expected_test_count"]) or not scene_table["status"].eq("PASS").all():
        raise RuntimeError("PK-STEM测试预测不完整。")
    correction_path = output / "Nanyang_Stage3E_PKSTEM_MODISCorrectionAudit_26Dates.csv"
    scene_path = output / "Nanyang_Stage3E_AdaptedPKSTEM_SceneAudit_5TestDates.csv"
    correction_table.to_csv(correction_path, index=False, encoding="utf-8-sig")
    scene_table.to_csv(scene_path, index=False, encoding="utf-8-sig")
    evidence = {
        "method": cfg["method_name"],
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "observation_sigma_estimated_from_train_val_only": observation_sigma,
        "test_labels_opened_during_prediction": False,
        "assimilation_date_count": int(sum(r["is_assimilation"] for r in records)),
        "test_date_count": len(scene_table),
        "output_sha256": output_hashes,
        "config_sha256": sha256(CONFIG_PATH),
        "script_sha256": sha256(SCRIPT_PATH),
        "split_manifest_sha256": sha256(split_path),
        "elapsed_seconds": time.perf_counter() - started,
        "status": "PASS",
        "implementation_note": "Adapted reproducible Mode III: historical prior trend, native-500m MODIS linear correction, bidirectional 100-member ensemble Kalman filtering, and time-weighted smoothing. Not an exact reproduction of the authors' unpublished source code.",
    }
    with (output / "00_Stage3E_AdaptedPKSTEM_Evidence.json").open("w", encoding="utf-8") as file:
        json.dump(evidence, file, ensure_ascii=False, indent=2)
    log.info("STAGE3E_ADAPTED_PKSTEM=PASS")
    print("STAGE3E_ADAPTED_PKSTEM=PASS")
    print(scene_table.to_string(index=False))


if __name__ == "__main__":
    main()

