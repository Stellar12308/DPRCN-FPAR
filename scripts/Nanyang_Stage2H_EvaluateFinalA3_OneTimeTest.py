from __future__ import annotations

import json
import logging
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


SCRIPT_PATH = Path(__file__).resolve()
STAGE2_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(STAGE2_ROOT))

from dprcn.dataset import build_dataloader  # noqa: E402
from dprcn.final_dataset import FinalTrainValPatchDataset  # noqa: E402
from dprcn.model import build_model  # noqa: E402
from dprcn.reproducibility import seed_everything, sha256_file  # noqa: E402


CONFIG_PATH = STAGE2_ROOT / "configs" / "Nanyang_Stage2H_OneTimeTestEvaluation_Config_v1.json"


class FixedA3InputMask(nn.Module):
    """Exact inference wrapper used by the frozen Stage2G final fit."""

    def __init__(self, model: nn.Module, active_indices: list[int]) -> None:
        super().__init__()
        self.model = model
        mask = torch.zeros(1, 9, 1, 1, dtype=torch.float32)
        mask[:, active_indices] = 1.0
        self.register_buffer("input_mask", mask)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(inputs * self.input_mask)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    p = prediction.astype(np.float64, copy=False)
    y = target.astype(np.float64, copy=False)
    error = p - y
    n = int(y.size)
    bias = float(np.mean(error))
    rmse = float(np.sqrt(np.mean(error * error)))
    y_mean = float(np.mean(y))
    p_mean = float(np.mean(p))
    y_centered = y - y_mean
    p_centered = p - p_mean
    y_var = float(np.mean(y_centered * y_centered))
    p_var = float(np.mean(p_centered * p_centered))
    covariance = float(np.mean(y_centered * p_centered))
    r = covariance / math.sqrt(y_var * p_var) if y_var > 0 and p_var > 0 else math.nan
    denominator = float(np.sum(y_centered * y_centered))
    r2 = 1.0 - float(np.sum(error * error)) / denominator if denominator > 0 else math.nan
    return {
        "N_valid_pixels": n,
        "RMSE": rmse,
        "MAE": float(np.mean(np.abs(error))),
        "Bias": bias,
        "Pearson_R": r,
        "R2": r2,
        "ubRMSE": float(np.sqrt(max(0.0, rmse * rmse - bias * bias))),
        "target_mean": y_mean,
        "target_std": math.sqrt(y_var),
        "prediction_mean": p_mean,
        "prediction_std": math.sqrt(p_var),
        "prediction_le_0p01_pct": float(np.mean(p <= 0.01) * 100.0),
        "prediction_ge_0p99_pct": float(np.mean(p >= 0.99) * 100.0),
    }


def masked_ssim(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray, policy: dict) -> tuple[float, int]:
    window = int(policy["window_size"])
    if window % 2 != 1:
        raise ValueError("SSIM窗口必须为奇数。")
    p = torch.from_numpy(np.where(valid, prediction, 0.0).astype(np.float64))[None, None]
    y = torch.from_numpy(np.where(valid, target, 0.0).astype(np.float64))[None, None]
    m = torch.from_numpy(valid.astype(np.float64))[None, None]
    kernel = torch.ones((1, 1, window, window), dtype=torch.float64)
    pad = window // 2
    count = F.conv2d(m, kernel, padding=pad)
    enough = count >= float(policy["minimum_valid_fraction"]) * window * window
    safe = count.clamp_min(1.0)
    mu_p = F.conv2d(p, kernel, padding=pad) / safe
    mu_y = F.conv2d(y, kernel, padding=pad) / safe
    var_p = (F.conv2d(p * p, kernel, padding=pad) / safe - mu_p * mu_p).clamp_min(0.0)
    var_y = (F.conv2d(y * y, kernel, padding=pad) / safe - mu_y * mu_y).clamp_min(0.0)
    cov = F.conv2d(p * y, kernel, padding=pad) / safe - mu_p * mu_y
    data_range = float(policy["data_range"])
    c1 = (float(policy["k1"]) * data_range) ** 2
    c2 = (float(policy["k2"]) * data_range) ** 2
    numerator = (2.0 * mu_p * mu_y + c1) * (2.0 * cov + c2)
    denominator = (mu_p * mu_p + mu_y * mu_y + c1) * (var_p + var_y + c2)
    values = numerator / denominator.clamp_min(np.finfo(np.float64).eps)
    selected = values[enough]
    if selected.numel() == 0:
        return math.nan, 0
    return float(selected.mean()), int(selected.numel())


def main() -> None:
    config = read_json(CONFIG_PATH)
    root = Path(config["project_root"])
    output_dir = resolve(root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "00_Stage2H_OneTimeTestEvaluation_RunLog.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    log = logging.getLogger("stage2h")
    started = time.perf_counter()
    if config.get("schema_version") != "Stage2H_OneTimeTestEvaluation_Config_v1":
        raise ValueError("Stage2H配置版本错误。")
    checkpoint_path = resolve(root, config["checkpoint"])
    normalization_path = resolve(root, config["normalization"])
    manifest_path = resolve(root, config["patch_manifest"])
    for path, expected in [
        (checkpoint_path, config["checkpoint_sha256"]),
        (normalization_path, config["normalization_sha256"]),
        (manifest_path, config["patch_manifest_sha256"]),
    ]:
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"冻结文件哈希不一致：{path}\nexpected={expected}\nactual={actual}")

    seed_everything(int(config["random_seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalization = read_json(normalization_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("test_split_accessed") is not False:
        raise ValueError("Stage2G检查点的测试封存标记不正确。")
    if checkpoint.get("active_channel_indices") != config["active_channel_indices"]:
        raise ValueError("A3活动通道不一致。")
    model = FixedA3InputMask(
        build_model(checkpoint["config"], normalization),
        config["active_channel_indices"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # 本行是冻结协议后首次且唯一的测试集实例化位置。
    dataset = FinalTrainValPatchDataset(
        manifest_path, normalization_path, "test",
        scene_cache_size=int(config["scene_cache_size"]),
    )
    if len(dataset) != int(config["expected_test_patch_count"]):
        raise ValueError(f"测试Patch数不是{config['expected_test_patch_count']}。")
    dates = sorted(dataset.table["date"].unique().tolist())
    if len(dates) != int(config["expected_test_date_count"]):
        raise ValueError(f"测试日期数不是{config['expected_test_date_count']}。")
    loader = build_dataloader(dataset, int(config["batch_size"]), False, int(config["random_seed"]), 0, False)
    dimensions = {}
    for date, rows in dataset.table.groupby("date"):
        dimensions[str(date)] = (
            int((rows["row_off"] + rows["height"]).max()),
            int((rows["col_off"] + rows["width"]).max()),
        )
    mosaics = {
        date: {
            "prediction_sum": np.zeros(shape, dtype=np.float64),
            "target_sum": np.zeros(shape, dtype=np.float64),
            "count": np.zeros(shape, dtype=np.uint16),
        }
        for date, shape in dimensions.items()
    }
    log.info("ONE-TIME TEST OPENED: patches=%d dates=%d device=%s", len(dataset), len(dates), device)
    with torch.no_grad():
        for raw in loader:
            inputs = raw["input"].to(device)
            prediction = model(inputs)["prediction"].cpu().numpy()
            target = raw["target"].numpy()
            mask = raw["valid_mask"].numpy() > 0.5
            for index in range(prediction.shape[0]):
                date = str(raw["date"][index])
                y0 = int(raw["row_off"][index])
                x0 = int(raw["col_off"][index])
                height, width = prediction.shape[-2:]
                local = mask[index, 0]
                ys, xs = slice(y0, y0 + height), slice(x0, x0 + width)
                mosaics[date]["prediction_sum"][ys, xs] += prediction[index, 0] * local
                mosaics[date]["target_sum"][ys, xs] += target[index, 0] * local
                mosaics[date]["count"][ys, xs] += local.astype(np.uint16)

    date_rows, all_p, all_y = [], [], []
    for date in dates:
        item = mosaics[date]
        valid = item["count"] > 0
        prediction_map = np.divide(item["prediction_sum"], item["count"], out=np.zeros_like(item["prediction_sum"]), where=valid)
        target_map = np.divide(item["target_sum"], item["count"], out=np.zeros_like(item["target_sum"]), where=valid)
        result = metrics(prediction_map[valid], target_map[valid])
        ssim, n_windows = masked_ssim(prediction_map, target_map, valid, config["ssim"])
        doy = int(dataset.table.loc[dataset.table["date"].eq(date), "stored_DOY"].iloc[0])
        date_rows.append({"date": date, "stored_DOY": doy, **result, "SSIM": ssim, "N_SSIM_windows": n_windows})
        all_p.append(prediction_map[valid])
        all_y.append(target_map[valid])
    date_table = pd.DataFrame(date_rows).sort_values("stored_DOY").reset_index(drop=True)
    global_result = metrics(np.concatenate(all_p), np.concatenate(all_y))
    summary = {
        "script_name": SCRIPT_PATH.name,
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device),
        "model": config["model_name"],
        "test_opened_once": 1,
        "test_patch_count": len(dataset),
        "test_date_count": len(dates),
        **{f"global_{key}": value for key, value in global_result.items()},
        "date_macro_RMSE": float(date_table["RMSE"].mean()),
        "date_macro_MAE": float(date_table["MAE"].mean()),
        "date_macro_absolute_Bias": float(date_table["Bias"].abs().mean()),
        "date_macro_Bias": float(date_table["Bias"].mean()),
        "date_macro_Pearson_R": float(date_table["Pearson_R"].mean()),
        "date_macro_R2": float(date_table["R2"].mean()),
        "date_macro_ubRMSE": float(date_table["ubRMSE"].mean()),
        "date_macro_SSIM": float(date_table["SSIM"].mean()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "normalization_sha256": sha256_file(normalization_path),
        "patch_manifest_sha256": sha256_file(manifest_path),
        "evaluation_config_sha256": sha256_file(CONFIG_PATH),
        "evaluation_script_sha256": sha256_file(SCRIPT_PATH),
        "elapsed_seconds": time.perf_counter() - started,
        "overall_status": "PASS",
    }
    date_table.to_csv(output_dir / "Nanyang_Stage2H_TestDateMetrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(output_dir / "Nanyang_Stage2H_TestOverallMetrics.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "00_Stage2H_OneTimeTestEvidence.json").open("w", encoding="utf-8") as file:
        json.dump({"config": config, "summary": summary, "date_metrics": date_rows}, file, ensure_ascii=False, indent=2)
    log.info("date_macro_RMSE=%.6f global_RMSE=%.6f macro_SSIM=%.6f", summary["date_macro_RMSE"], summary["global_RMSE"], summary["date_macro_SSIM"])
    log.info("STAGE2H_ONE_TIME_TEST=PASS")
    print("STAGE2H_ONE_TIME_TEST=PASS")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()

