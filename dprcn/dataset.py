from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset


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


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def valid(array: np.ndarray, nodata) -> np.ndarray:
    result = np.isfinite(array)
    if nodata is not None and np.isfinite(nodata):
        result &= array != nodata
    return result


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
        edge_order=int(edge_order),
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


def source_transform(name: str, array: np.ndarray, policy: str) -> np.ndarray:
    values = array.astype(np.float64, copy=False)
    if policy == "identity":
        return values
    if policy == "log1p":
        finite = values[np.isfinite(values)]
        if finite.size and np.any(finite < 0):
            raise ValueError(f"{name}存在负值，不能执行log1p。")
        return np.log1p(values)
    raise ValueError(f"不支持的输入变换：{name}={policy}")


def _worker_seed(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class NanyangFPARPatchDataset(Dataset):
    """Stage2正式Patch数据集。

    标签仅用于target、valid_mask和weight，不会进入九通道输入。
    NoData输入在z-score之后填0，即训练均值对应的中性值；损失仍由
    valid_mask严格排除，因此NoData不会作为合法FPAR零值参与训练。
    """

    def __init__(
        self,
        patch_manifest_path: str | Path,
        normalization_path: str | Path,
        split: str,
        allowed_weights: list[float] | tuple[float, ...] = (0.5, 0.8, 1.0),
        weight_tolerance: float = 1e-6,
        scene_cache_size: int = 4,
        input_fill: float = 0.0,
        target_fill: float = 0.0,
        weight_fill: float = 0.0,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"无效split：{split}")
        self.patch_manifest_path = Path(patch_manifest_path)
        self.normalization_path = Path(normalization_path)
        self.split = split
        self.allowed_weights = [float(value) for value in allowed_weights]
        self.weight_tolerance = float(weight_tolerance)
        self.scene_cache_size = max(1, int(scene_cache_size))
        self.input_fill = float(input_fill)
        self.target_fill = float(target_fill)
        self.weight_fill = float(weight_fill)

        table = pd.read_csv(
            self.patch_manifest_path,
            encoding="utf-8-sig",
            dtype={"date": str},
        )
        table["date"] = table["date"].astype(str).str.zfill(8)
        self.table = (
            table.loc[
                (table["split"] == split)
                & (table["eligible_for_use"] == 1)
            ]
            .sort_values(["stored_DOY", "row_off", "col_off", "patch_id"])
            .reset_index(drop=True)
        )
        if self.table.empty:
            raise ValueError(f"{split}没有可用Patch。")
        if self.table["patch_id"].duplicated().any():
            raise ValueError(f"{split}的patch_id不唯一。")

        document = load_json(self.normalization_path)
        if document.get("schema_version") != "Stage2B_TrainOnlyNormalization_v1":
            raise ValueError("归一化参数版本错误。")
        if document.get("fit_split") != "train":
            raise ValueError("归一化参数不是仅由训练集拟合。")
        if document.get("channel_order") != CHANNELS:
            raise ValueError("归一化通道顺序不一致。")
        leakage = document.get("leakage_guard", {})
        if leakage.get("validation_used_for_fit") or leakage.get("test_used_for_fit"):
            raise ValueError("归一化参数存在验证集或测试集泄漏。")
        self.normalization = document
        self.parameters = document["parameters"]
        self.terrain_policy = document["terrain_policy"]
        self._scene_cache: OrderedDict[tuple, dict] = OrderedDict()

    def __len__(self) -> int:
        return len(self.table)

    def _scene_key(self, row: pd.Series) -> tuple:
        return (
            str(row["date"]),
            int(row["stored_DOY"]),
            str(row["label_path"]),
            str(row["prior_path"]),
            str(row["modis_path"]),
        )

    def _load_scene(self, row: pd.Series) -> dict:
        key = self._scene_key(row)
        if key in self._scene_cache:
            value = self._scene_cache.pop(key)
            self._scene_cache[key] = value
            return value

        path_keys = [
            "label_path",
            "label_qc_path",
            "label_weight_path",
            "prior_path",
            "modis_path",
            "dem_path",
        ]
        paths = {name: Path(str(row[name])) for name in path_keys}
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("缺少数据集输入：\n" + "\n".join(missing))
        datasets = {name: rasterio.open(path) for name, path in paths.items()}
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
                int(self.terrain_policy["edge_order"]),
            )
            raw_channels = {
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
            channel_valid = {
                "Prior_median": valid(prior[0], datasets["prior_path"].nodata)
                & (prior[0] >= 0.0) & (prior[0] <= 1.0),
                "Prior_N_obs": valid(prior[1], datasets["prior_path"].nodata)
                & (prior[1] > 0.0),
                "Prior_sigma": valid(prior[2], datasets["prior_path"].nodata)
                & (prior[2] >= 0.0),
            }
            channel_valid.update({
                "MODIS_FPAR": valid(modis[0], datasets["modis_path"].nodata)
                & (modis[0] >= 0.0) & (modis[0] <= 1.0),
                "MODIS_FPAR_SD": valid(modis[1], datasets["modis_path"].nodata)
                & (modis[1] >= 0.0),
                "DEM_normalized": terrain_valid,
                "Slope_normalized": terrain_valid,
                "Aspect_sin": terrain_valid,
                "Aspect_cos": terrain_valid,
            })
            normalized = []
            for name in CHANNELS:
                parameter = self.parameters[name]
                transformed = source_transform(
                    name,
                    raw_channels[name],
                    parameter["source_transform"],
                )
                values = (
                    transformed - float(parameter["mean"])
                ) / float(parameter["std_population"])
                values = np.where(channel_valid[name], values, self.input_fill)
                normalized.append(values.astype(np.float32))
            inputs = np.stack(normalized, axis=0)

            joint = (
                valid(label, reference.nodata)
                & (label >= 0.0) & (label <= 1.0)
                & valid(qc, datasets["label_qc_path"].nodata)
                & (qc == 1.0)
                & valid(weight, datasets["label_weight_path"].nodata)
                & allowed_weight_mask(
                    weight,
                    self.allowed_weights,
                    self.weight_tolerance,
                )
                & valid(modis[2], datasets["modis_path"].nodata)
                & (modis[2] == 1.0)
            )
            for name in CHANNELS:
                joint &= channel_valid[name]
            scene = {
                "input": inputs,
                "target": np.where(joint, label, self.target_fill).astype(np.float32),
                "valid_mask": joint.astype(np.float32),
                "weight": np.where(joint, weight, self.weight_fill).astype(np.float32),
            }
        finally:
            for dataset in datasets.values():
                dataset.close()
        self._scene_cache[key] = scene
        while len(self._scene_cache) > self.scene_cache_size:
            self._scene_cache.popitem(last=False)
        return scene

    def __getitem__(self, index: int) -> dict:
        row = self.table.iloc[int(index)]
        scene = self._load_scene(row)
        y = int(row["row_off"])
        x = int(row["col_off"])
        height = int(row["height"])
        width = int(row["width"])
        slices = (slice(y, y + height), slice(x, x + width))
        return {
            "input": torch.from_numpy(scene["input"][:, slices[0], slices[1]].copy()),
            "target": torch.from_numpy(scene["target"][slices][None, ...].copy()),
            "valid_mask": torch.from_numpy(scene["valid_mask"][slices][None, ...].copy()),
            "weight": torch.from_numpy(scene["weight"][slices][None, ...].copy()),
            "patch_id": str(row["patch_id"]),
            "date": str(row["date"]),
            "stored_DOY": int(row["stored_DOY"]),
            "split": self.split,
            "row_off": y,
            "col_off": x,
        }


def build_dataloader(
    dataset: NanyangFPARPatchDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        worker_init_fn=_worker_seed,
        generator=generator,
        persistent_workers=bool(num_workers > 0),
        drop_last=False,
    )

