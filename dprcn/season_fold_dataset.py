from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pandas as pd

from .dataset import CHANNELS, NanyangFPARPatchDataset, load_json


class SeasonFoldPatchDataset(NanyangFPARPatchDataset):
    """Patch dataset for a frozen outer seasonal fold."""

    def __init__(
        self,
        patch_manifest_path: str | Path,
        normalization_path: str | Path,
        role: str,
        held_out_season: str,
        allowed_weights=(0.5, 0.8, 1.0),
        weight_tolerance: float = 1e-6,
        scene_cache_size: int = 4,
        input_fill: float = 0.0,
        target_fill: float = 0.0,
        weight_fill: float = 0.0,
    ) -> None:
        if role not in {"train", "held_out"}:
            raise ValueError(f"无效role：{role}")
        self.patch_manifest_path = Path(patch_manifest_path)
        self.normalization_path = Path(normalization_path)
        self.split = role
        self.allowed_weights = [float(value) for value in allowed_weights]
        self.weight_tolerance = float(weight_tolerance)
        self.scene_cache_size = max(1, int(scene_cache_size))
        self.input_fill = float(input_fill)
        self.target_fill = float(target_fill)
        self.weight_fill = float(weight_fill)
        table = pd.read_csv(self.patch_manifest_path, encoding="utf-8-sig", dtype={"date": str})
        table["date"] = table["date"].astype(str).str.zfill(8)
        selected = table["season_en"].eq(held_out_season)
        if role == "train":
            selected = ~selected
        self.table = (
            table.loc[selected & table["eligible_for_use"].eq(1)]
            .sort_values(["stored_DOY", "row_off", "col_off", "patch_id"])
            .reset_index(drop=True)
        )
        if self.table.empty or self.table["patch_id"].duplicated().any():
            raise ValueError(f"{held_out_season}/{role} Patch表无效。")
        document = load_json(self.normalization_path)
        if document.get("schema_version") != "Stage2I_SeasonFoldNormalization_v1":
            raise ValueError("不是Stage2I逐折归一化。")
        if document.get("held_out_season") != held_out_season:
            raise ValueError("归一化与留出季节不一致。")
        if document.get("held_out_used_for_fit") is not False:
            raise ValueError("留出季节进入归一化。")
        if document.get("channel_order") != CHANNELS:
            raise ValueError("通道顺序不一致。")
        self.normalization = document
        self.parameters = document["parameters"]
        self.terrain_policy = document["terrain_policy"]
        self._scene_cache = OrderedDict()

