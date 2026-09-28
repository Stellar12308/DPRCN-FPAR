from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_NAME = SCRIPT_PATH.name
STAGE2_ROOT = SCRIPT_PATH.parents[1]
CONFIG_PATH = STAGE2_ROOT / "configs" / "Nanyang_Stage2B_Normalization_Config_v1.json"

CHANNELS = [
    "Prior_median",
    "Prior_N_obs",
    "Prior_sigma",
    "MODIS_FPAR",
    "MODIS_FPAR_SD",
    "DEM_normalized",
    "Slope_normalized",
    "Aspect_sin",
    "Aspect_cos",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: dict) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def valid(array: np.ndarray, nodata) -> np.ndarray:
    mask = np.isfinite(array)
    if nodata is not None and np.isfinite(nodata):
        mask &= array != nodata
    return mask


def allowed_weight_mask(
    weight: np.ndarray,
    allowed: list[float],
    tolerance: float,
) -> np.ndarray:
    result = np.zeros(weight.shape, dtype=bool)
    for value in allowed:
        result |= np.isclose(weight, value, rtol=0.0, atol=tolerance)
    return result


def terrain_channels(
    dem: np.ndarray,
    dem_valid: np.ndarray,
    transform,
    edge_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """由DEM确定性计算坡度及坡向正余弦。

    坡向定义为下坡方向、从正北起顺时针。所有需要相邻像元但
    邻域中存在NoData的位置都会被排除，绝不把NoData当作零高程。
    """
    work = dem.astype(np.float64, copy=True)
    work[~dem_valid] = np.nan
    pixel_x = abs(float(transform.a))
    pixel_y = abs(float(transform.e))
    if pixel_x <= 0 or pixel_y <= 0:
        raise ValueError("DEM像元大小无效。")

    dz_drow, dz_dx = np.gradient(
        work,
        pixel_y,
        pixel_x,
        edge_order=edge_order,
    )
    dz_dnorth = -dz_drow
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dnorth)))
    east_down = -dz_dx
    north_down = -dz_dnorth
    aspect = np.mod(np.arctan2(east_down, north_down), 2.0 * np.pi)
    aspect_sin = np.sin(aspect)
    aspect_cos = np.cos(aspect)
    terrain_valid = (
        dem_valid
        & np.isfinite(slope)
        & np.isfinite(aspect_sin)
        & np.isfinite(aspect_cos)
    )
    return slope, aspect_sin, aspect_cos, terrain_valid


def union_patch_mask(rows: pd.DataFrame, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for row in rows.itertuples(index=False):
        y = int(row.row_off)
        x = int(row.col_off)
        height = int(row.height)
        width = int(row.width)
        mask[y:y + height, x:x + width] = True
    return mask


def update_moments(state: dict, values: np.ndarray) -> None:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return
    count = int(data.size)
    mean = float(data.mean(dtype=np.float64))
    centered = data - mean
    # 使用逐元素平方求和，避免在部分Windows Conda环境中为短向量
    # 调用BLAS时出现无Python异常栈的原生进程退出。
    m2 = float(np.sum(centered * centered, dtype=np.float64))
    if state["count"] == 0:
        state.update(
            count=count,
            mean=mean,
            m2=m2,
            minimum=float(data.min()),
            maximum=float(data.max()),
        )
        return
    old_count = int(state["count"])
    total = old_count + count
    delta = mean - float(state["mean"])
    state["mean"] += delta * count / total
    state["m2"] += m2 + delta * delta * old_count * count / total
    state["count"] = total
    state["minimum"] = min(float(state["minimum"]), float(data.min()))
    state["maximum"] = max(float(state["maximum"]), float(data.max()))


def transform_channel(name: str, array: np.ndarray, policies: dict) -> np.ndarray:
    policy = policies[name]
    if policy == "identity":
        return array.astype(np.float64, copy=False)
    if policy == "log1p":
        if np.any(array[np.isfinite(array)] < 0):
            raise ValueError(f"{name}包含负值，不能执行log1p。")
        return np.log1p(array.astype(np.float64, copy=False))
    raise ValueError(f"不支持的变换：{name}={policy}")


def scene_arrays(
    scene_rows: pd.DataFrame,
    allowed_weights: list[float],
    tolerance: float,
    terrain_config: dict,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    first = scene_rows.iloc[0]
    paths = {
        key: Path(str(first[key]))
        for key in [
            "label_path",
            "label_qc_path",
            "label_weight_path",
            "prior_path",
            "modis_path",
            "dem_path",
        ]
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少输入文件：\n" + "\n".join(missing))

    datasets = {key: rasterio.open(path) for key, path in paths.items()}
    try:
        reference = datasets["label_path"]
        for name, dataset in datasets.items():
            if (
                dataset.shape != reference.shape
                or dataset.transform != reference.transform
                or dataset.crs != reference.crs
            ):
                raise ValueError(f"{name}与标签网格不一致。")

        label = reference.read(1).astype(np.float64)
        qc = datasets["label_qc_path"].read(1).astype(np.float64)
        weight = datasets["label_weight_path"].read(1).astype(np.float64)
        prior = datasets["prior_path"].read([1, 2, 3]).astype(np.float64)
        modis = datasets["modis_path"].read([1, 2, 6]).astype(np.float64)
        dem = datasets["dem_path"].read(1).astype(np.float64)

        slope, aspect_sin, aspect_cos, terrain_valid = terrain_channels(
            dem,
            valid(dem, datasets["dem_path"].nodata),
            reference.transform,
            int(terrain_config["edge_order"]),
        )
        patch_union = union_patch_mask(scene_rows, reference.shape)
        mask = (
            patch_union
            & valid(label, reference.nodata)
            & (label >= 0.0)
            & (label <= 1.0)
            & valid(qc, datasets["label_qc_path"].nodata)
            & (qc == 1.0)
            & valid(weight, datasets["label_weight_path"].nodata)
            & allowed_weight_mask(weight, allowed_weights, tolerance)
            & valid(prior[0], datasets["prior_path"].nodata)
            & (prior[0] >= 0.0)
            & (prior[0] <= 1.0)
            & valid(prior[1], datasets["prior_path"].nodata)
            & (prior[1] > 0.0)
            & valid(prior[2], datasets["prior_path"].nodata)
            & (prior[2] >= 0.0)
            & valid(modis[0], datasets["modis_path"].nodata)
            & (modis[0] >= 0.0)
            & (modis[0] <= 1.0)
            & valid(modis[1], datasets["modis_path"].nodata)
            & (modis[1] >= 0.0)
            & valid(modis[2], datasets["modis_path"].nodata)
            & (modis[2] == 1.0)
            & terrain_valid
        )
        arrays = {
            "Prior_median": prior[0],
            "Prior_N_obs": prior[1],
            "Prior_sigma": prior[2],
            "MODIS_FPAR": modis[0],
            "MODIS_FPAR_SD": modis[1],
            "DEM_normalized": dem,
            "Slope_normalized": slope,
            "Aspect_sin": aspect_sin,
            "Aspect_cos": aspect_cos,
        }
        metadata = {
            "height": reference.height,
            "width": reference.width,
            "crs": reference.crs.to_string() if reference.crs else None,
            "transform": list(reference.transform)[:6],
            "N_patch_union_pixels": int(patch_union.sum()),
            "N_fit_pixels": int(mask.sum()),
        }
        return arrays, mask, metadata
    finally:
        for dataset in datasets.values():
            dataset.close()


def main() -> None:
    config = load_json(CONFIG_PATH)
    if config.get("schema_version") != "Stage2B_Normalization_Config_v1":
        raise ValueError("Stage2B配置版本错误。")
    project_root = Path(config["project_root"])
    base_config_path = resolve(project_root, config["stage2_base_config"])
    patch_path = resolve(project_root, config["stage2a_patch_manifest"])
    stage2a_summary_path = resolve(project_root, config["stage2a_summary"])
    output_dir = resolve(project_root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "00_Stage2B_TrainOnlyNormalization_RunLog.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    log = logging.getLogger("Stage2B")
    log.info("Stage2B训练集专属归一化开始")
    log.info("脚本=%s", SCRIPT_PATH)
    log.info("配置=%s", CONFIG_PATH)

    for required in [base_config_path, patch_path, stage2a_summary_path]:
        if not required.exists():
            raise FileNotFoundError(required)
    log.info("必需输入文件存在性检查通过")
    base_config = load_json(base_config_path)
    stage2a_summary = pd.read_csv(stage2a_summary_path, encoding="utf-8-sig")
    log.info("Stage2A汇总表读取完成")
    if len(stage2a_summary) != 1 or stage2a_summary.iloc[0]["overall_status"] != "PASS":
        raise RuntimeError("Stage2A未通过，禁止拟合Stage2B。")
    recorded_patch_hash = str(stage2a_summary.iloc[0]["patch_manifest_sha256"])
    current_patch_hash = sha256_file(patch_path)
    if current_patch_hash != recorded_patch_hash:
        raise RuntimeError("Stage2A Patch清单哈希与冻结记录不一致。")
    log.info("Stage2A Patch清单哈希核验通过")

    patches = pd.read_csv(patch_path, encoding="utf-8-sig", dtype={"date": str})
    log.info("Stage2A Patch清单读取完成，记录数=%d", len(patches))
    patches["date"] = patches["date"].astype(str).str.zfill(8)
    fit = patches.loc[
        (patches["split"] == config["fit_split"])
        & (patches["eligible_for_use"] == 1)
    ].copy()
    if fit.empty:
        raise RuntimeError("没有可用于归一化的训练Patch。")
    scene_keys = fit[["date", "stored_DOY"]].drop_duplicates()
    if len(scene_keys) != int(config["expected_fit_scene_count"]):
        raise RuntimeError(
            f"训练景数为{len(scene_keys)}，预期为{config['expected_fit_scene_count']}。"
        )
    if set(patches.loc[patches["split"] != "train", "date"]) & set(fit["date"]):
        raise RuntimeError("发现日期级split交叉。")
    log.info("训练专属选择完成，训练景数=%d，训练Patch数=%d", len(scene_keys), len(fit))

    policies = config["normalization"]
    states = {
        channel: {
            "count": 0,
            "mean": 0.0,
            "m2": 0.0,
            "minimum": np.inf,
            "maximum": -np.inf,
        }
        for channel in CHANNELS
    }
    audit_records = []
    for (date, doy), rows in fit.groupby(["date", "stored_DOY"], sort=True):
        log.info("开始读取DOY%03d", int(doy))
        arrays, fit_mask, metadata = scene_arrays(
            rows,
            [float(value) for value in config["allowed_training_weights"]],
            float(config["weight_tolerance"]),
            config["terrain"],
        )
        log.info("DOY%03d栅格和联合掩膜读取完成", int(doy))
        if not fit_mask.any():
            raise RuntimeError(f"DOY{int(doy):03d}没有归一化拟合像元。")
        record = {
            "date": date,
            "stored_DOY": int(doy),
            "split": "train",
            **metadata,
            "status": "PASS",
        }
        for channel in CHANNELS:
            log.info("DOY%03d统计通道=%s", int(doy), channel)
            values = transform_channel(channel, arrays[channel], policies)[fit_mask]
            update_moments(states[channel], values)
            record[f"{channel}_mean_transformed"] = float(np.mean(values))
            record[f"{channel}_std_transformed"] = float(np.std(values, ddof=0))
            record[f"{channel}_min_transformed"] = float(np.min(values))
            record[f"{channel}_max_transformed"] = float(np.max(values))
        audit_records.append(record)
        log.info("DOY%03d拟合像元=%d", int(doy), int(fit_mask.sum()))

    parameters = {}
    min_std = float(config["minimum_standard_deviation"])
    for channel, state in states.items():
        count = int(state["count"])
        if count < 2:
            raise RuntimeError(f"{channel}有效训练像元不足。")
        variance = float(state["m2"]) / count
        std = float(np.sqrt(max(variance, 0.0)))
        if not np.isfinite(std) or std <= min_std:
            raise RuntimeError(f"{channel}训练标准差无效：{std}")
        parameters[channel] = {
            "source_transform": policies[channel],
            "normalization": "z_score",
            "count": count,
            "mean": float(state["mean"]),
            "std_population": std,
            "minimum": float(state["minimum"]),
            "maximum": float(state["maximum"]),
        }

    counts = {value["count"] for value in parameters.values()}
    passed = (
        len(audit_records) == int(config["expected_fit_scene_count"])
        and len(counts) == 1
        and all(record["status"] == "PASS" for record in audit_records)
        and not (fit["split"] != "train").any()
    )
    if not passed:
        raise RuntimeError("Stage2B审计失败。")

    audit_path = output_dir / "Nanyang_Stage2B_TrainSceneNormalizationAudit.csv"
    pd.DataFrame(audit_records).sort_values("stored_DOY").to_csv(
        audit_path,
        index=False,
        encoding="utf-8-sig",
    )
    parameter_document = {
        "schema_version": "Stage2B_TrainOnlyNormalization_v1",
        "region_name": config["region_name"],
        "fit_split": "train",
        "fit_scene_count": len(audit_records),
        "fit_pixel_count_unique_per_scene": next(iter(counts)),
        "leakage_guard": {
            "validation_used_for_fit": False,
            "test_used_for_fit": False,
            "overlapping_patch_duplicates_counted": False,
            "policy": policies["pixel_counting_policy"],
        },
        "terrain_policy": config["terrain"],
        "channel_order": CHANNELS,
        "parameters": parameters,
        "source_hashes": {
            "stage2b_config_sha256": canonical_json_sha256(config),
            "stage2_base_config_sha256": sha256_file(base_config_path),
            "stage2a_patch_manifest_sha256": current_patch_hash,
            "stage2a_summary_sha256": sha256_file(stage2a_summary_path),
        },
    }
    parameter_path = output_dir / "Nanyang_Stage2B_TrainOnlyNormalizationParameters.json"
    with parameter_path.open("w", encoding="utf-8") as file:
        json.dump(parameter_document, file, ensure_ascii=False, indent=2)
    shutil.copy2(CONFIG_PATH, output_dir / CONFIG_PATH.name.replace(".json", "_Frozen.json"))

    summary_path = output_dir / "Nanyang_Stage2B_NormalizationSummary.csv"
    pd.DataFrame([{
        "script_name": SCRIPT_NAME,
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "rasterio_version": rasterio.__version__,
        "fit_split": "train",
        "fit_scene_count": len(audit_records),
        "fit_pixel_count": next(iter(counts)),
        "validation_used_for_fit": 0,
        "test_used_for_fit": 0,
        "patch_overlap_duplicates_counted": 0,
        "channel_count": len(CHANNELS),
        "stage2a_patch_manifest_sha256": current_patch_hash,
        "normalization_parameters_sha256": sha256_file(parameter_path),
        "scene_audit_sha256": sha256_file(audit_path),
        "overall_status": "PASS",
    }]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    log.info("训练景数=%d", len(audit_records))
    log.info("每个通道拟合像元数=%d", next(iter(counts)))
    log.info("验证/测试参与拟合=否/否")
    log.info("参数文件SHA256=%s", sha256_file(parameter_path))
    log.info("总状态=PASS")


if __name__ == "__main__":
    main()

