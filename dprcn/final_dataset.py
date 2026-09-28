from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pandas as pd

from .dataset import CHANNELS, NanyangFPARPatchDataset, load_json


class FinalTrainValPatchDataset(NanyangFPARPatchDataset):
    """Stage2G dataset using normalization fitted on train+validation.

    Raster reading, masks, terrain derivation and tensor construction are
    inherited unchanged from the audited Stage2C dataset. Only initialization
    differs, because final fitting legitimately uses validation after model
    selection, while the test split must remain excluded.
    """

    def __init__(
        self,
        patch_manifest_path: str | Path,
        normalization_path: str | Path,
        split: str,
        allowed_weights=(0.5, 0.8, 1.0),
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
                table["split"].eq(split)
                & table["eligible_for_use"].eq(1)
            ]
            .sort_values(["stored_DOY", "row_off", "col_off", "patch_id"])
            .reset_index(drop=True)
        )
        if self.table.empty:
            raise ValueError(f"{split}没有可用Patch。")
        if self.table["patch_id"].duplicated().any():
            raise ValueError(f"{split}的patch_id不唯一。")

        document = load_json(self.normalization_path)
        if document.get("schema_version") != "Stage2G_TrainValNormalization_v1":
            raise ValueError("不是Stage2G Train+Validation归一化参数。")
        if document.get("fit_split") != "train_val":
            raise ValueError("Stage2G归一化拟合集不是train_val。")
        if set(document.get("fit_splits", [])) != {"train", "val"}:
            raise ValueError("Stage2G归一化拟合集字段不完整。")
        if document.get("channel_order") != CHANNELS:
            raise ValueError("归一化通道顺序不一致。")
        leakage = document.get("leakage_guard", {})
        if leakage.get("validation_used_for_final_fit") is not True:
            raise ValueError("未明确记录验证集用于最终拟合。")
        if leakage.get("test_used_for_fit") is not False:
            raise ValueError("最终归一化存在测试集泄漏。")
        self.normalization = document
        self.parameters = document["parameters"]
        self.terrain_policy = document["terrain_policy"]
        self._scene_cache = OrderedDict()


