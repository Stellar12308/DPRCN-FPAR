from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SCRIPT_PATH = Path(__file__).resolve()
STAGE2_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(STAGE2_ROOT))

from dprcn.reproducibility import sha256_file  # noqa: E402


CONFIG_PATH = (
    STAGE2_ROOT
    / "configs"
    / "Nanyang_Stage2F_MultiSeedSeasonRobustness_Config_v1.json"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法载入模块：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def season_from_doy(doy: int, definitions: dict) -> str:
    for name, limits in definitions.items():
        if int(limits[0]) <= doy <= int(limits[1]):
            return name
    if doy <= 59:
        return "Winter"
    raise ValueError(f"DOY无法分配季节：{doy}")


def seed42_record(model_id: str, stage2e_table: pd.DataFrame, stage2e_dir: Path) -> tuple[dict, Path, Path]:
    row = stage2e_table.loc[stage2e_table["variant_id"] == model_id]
    if len(row) != 1:
        raise RuntimeError(f"Stage2E缺少唯一{model_id}记录。")
    item = row.iloc[0]
    if model_id == "A3_PRIOR_MODIS":
        checkpoint = stage2e_dir / model_id / "best_model.pt"
        date_metrics = stage2e_dir / model_id / "best_validation_date_metrics.csv"
    elif model_id == "A6_FULL_9CH":
        checkpoint = (
            stage2e_dir.parent
            / "Stage2D_DPRCNTraining"
            / "Formal_MainModel_Seed42"
            / "best_model.pt"
        )
        date_metrics = stage2e_dir / "A6_FULL_9CH_validation_date_metrics.csv"
    else:
        raise ValueError(model_id)
    for path in [checkpoint, date_metrics]:
        if not path.exists():
            raise FileNotFoundError(path)
    record = {
        "model_id": model_id,
        "seed": 42,
        "source": "reused_frozen_result",
        "active_channel_count": int(item["active_channel_count"]),
        "best_epoch": int(item["best_epoch"]),
        "epochs_completed": int(item["epochs_completed"]),
        "elapsed_seconds": float(item["elapsed_seconds"]),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    for column in [
        "global_RMSE", "global_MAE", "global_Bias", "global_R", "global_R2",
        "date_macro_RMSE", "date_macro_MAE", "date_macro_absolute_Bias",
        "date_macro_R", "date_macro_R2",
    ]:
        record[column] = float(item[column])
    return record, checkpoint, date_metrics


def main() -> None:
    config = load_json(CONFIG_PATH)
    if config["schema_version"] != "Stage2F_MultiSeedSeasonRobustness_Config_v1":
        raise ValueError("Stage2F配置版本错误。")
    if config["test_split_access_allowed"] is not False:
        raise RuntimeError("Stage2F禁止测试集访问。")
    root = Path(config["project_root"])
    base_path = resolve(root, config["base_training_config"])
    base_template = load_json(base_path)
    if set(base_template["allowed_splits"]) != {"train", "val"}:
        raise RuntimeError("基础训练配置未限定train/val。")
    stage2e_path = resolve(root, config["stage2e_summary"])
    stage2e_dir = resolve(root, config["stage2e_result_dir"])
    output_dir = resolve(root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stage2e_table = pd.read_csv(stage2e_path, encoding="utf-8-sig")
    stage2e = load_module(
        "stage2e_ablation",
        STAGE2_ROOT / "scripts" / "Nanyang_Stage2E_RunValidationAblations.py",
    )
    stage2d = stage2e.load_stage2d_module()
    summary_path = output_dir / "Nanyang_Stage2F_MultiSeed_ModelMetrics.csv"
    records = (
        pd.read_csv(summary_path, encoding="utf-8-sig").to_dict("records")
        if summary_path.exists() else []
    )
    completed = {(str(row["model_id"]), int(row["seed"])) for row in records}
    date_records = []
    date_master_path = output_dir / "Nanyang_Stage2F_MultiSeed_DateMetrics.csv"
    if date_master_path.exists():
        date_records = pd.read_csv(date_master_path, encoding="utf-8-sig").to_dict("records")

    for seed in [int(value) for value in config["random_seeds"]]:
        base = json.loads(json.dumps(base_template))
        base["random_seed"] = seed
        train_dataset, val_dataset, train_loader, val_loader, normalization_path = stage2e.build_data(base, root)
        normalization = load_json(normalization_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model_definition in config["models"]:
            model_id = str(model_definition["id"])
            key = (model_id, seed)
            seed_dir = output_dir / f"Seed{seed}"
            if key in completed:
                continue
            if seed == 42 and bool(config["reuse_seed42"]):
                record, checkpoint, date_source = seed42_record(
                    model_id, stage2e_table, stage2e_dir
                )
                destination = seed_dir / model_id
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(date_source, destination / "best_validation_date_metrics.csv")
                (destination / "reused_checkpoint_reference.txt").write_text(
                    str(checkpoint) + "\n" + sha256_file(checkpoint) + "\n",
                    encoding="utf-8",
                )
            else:
                variant = {
                    "id": model_id,
                    "label": model_id,
                    "active_channel_indices": model_definition["active_channel_indices"],
                }
                trained = stage2e.train_variant(
                    variant,
                    base,
                    normalization,
                    (train_dataset, val_dataset),
                    (train_loader, val_loader),
                    device,
                    seed_dir,
                    stage2d,
                )
                record = {
                    "model_id": model_id,
                    "seed": seed,
                    "source": "independent_training",
                    **{
                        key_name: value
                        for key_name, value in trained.items()
                        if key_name not in {"variant_id", "label", "variant_type", "active_channel_indices"}
                    },
                }
            date_path = seed_dir / model_id / "best_validation_date_metrics.csv"
            dates = pd.read_csv(date_path, encoding="utf-8-sig")
            dates.insert(0, "seed", seed)
            dates.insert(0, "model_id", model_id)
            dates["season"] = dates["stored_DOY"].map(
                lambda value: season_from_doy(int(value), config["season_definition"])
            )
            date_records = [
                row for row in date_records
                if not (str(row["model_id"]) == model_id and int(row["seed"]) == seed)
            ]
            date_records.extend(dates.to_dict("records"))
            records.append(record)
            completed.add(key)
            pd.DataFrame(records).to_csv(summary_path, index=False, encoding="utf-8-sig")
            pd.DataFrame(date_records).to_csv(date_master_path, index=False, encoding="utf-8-sig")
            print(
                f'STAGE2F_COMPLETED model={model_id} seed={seed} '
                f'RMSE={float(record["date_macro_RMSE"]):.6f}',
                flush=True,
            )

    model_table = pd.DataFrame(records).sort_values(["model_id", "seed"])
    expected_pairs = {
        (str(model["id"]), int(seed))
        for model in config["models"] for seed in config["random_seeds"]
    }
    actual_pairs = set(zip(model_table["model_id"].astype(str), model_table["seed"].astype(int)))
    if actual_pairs != expected_pairs:
        raise RuntimeError("模型×随机种子组合不完整。")
    if "test_accessed" not in model_table:
        model_table["test_accessed"] = 0
    model_table["test_accessed"] = 0
    model_table.to_csv(summary_path, index=False, encoding="utf-8-sig")

    aggregation = model_table.groupby("model_id").agg(
        active_channel_count=("active_channel_count", "first"),
        completed_seeds=("seed", "nunique"),
        mean_date_macro_RMSE=("date_macro_RMSE", "mean"),
        std_date_macro_RMSE=("date_macro_RMSE", "std"),
        min_date_macro_RMSE=("date_macro_RMSE", "min"),
        max_date_macro_RMSE=("date_macro_RMSE", "max"),
        mean_date_macro_MAE=("date_macro_MAE", "mean"),
        mean_date_macro_absolute_Bias=("date_macro_absolute_Bias", "mean"),
        mean_date_macro_R=("date_macro_R", "mean"),
        mean_date_macro_R2=("date_macro_R2", "mean"),
    ).reset_index()
    minimum_seeds = int(config["robustness_selection_rule"]["minimum_completed_seeds_per_model"])
    if not aggregation["completed_seeds"].eq(minimum_seeds).all():
        raise RuntimeError("随机种子完成数不足。")
    best_mean = float(aggregation["mean_date_macro_RMSE"].min())
    tolerance = float(config["robustness_selection_rule"]["equivalence_tolerance_rmse"])
    eligible = aggregation.loc[
        aggregation["mean_date_macro_RMSE"] <= best_mean + tolerance
    ].sort_values(["active_channel_count", "mean_date_macro_RMSE"])
    selected = str(eligible.iloc[0]["model_id"])
    aggregation["mean_RMSE_difference_from_best"] = (
        aggregation["mean_date_macro_RMSE"] - best_mean
    )
    aggregation["selected_by_frozen_rule"] = aggregation["model_id"].eq(selected).astype(int)
    aggregation["test_accessed"] = 0
    aggregation.to_csv(
        output_dir / "Nanyang_Stage2F_MultiSeed_AggregatedMetrics.csv",
        index=False, encoding="utf-8-sig",
    )

    dates = pd.DataFrame(date_records)
    seasonal = dates.groupby(["model_id", "seed", "season"]).agg(
        N_dates=("date", "nunique"),
        season_mean_RMSE=("RMSE", "mean"),
        season_mean_MAE=("MAE", "mean"),
        season_mean_absolute_Bias=("Bias", lambda values: values.abs().mean()),
        season_mean_R=("R", "mean"),
        season_mean_R2=("R2", "mean"),
    ).reset_index()
    seasonal.to_csv(
        output_dir / "Nanyang_Stage2F_SeasonMetrics_BySeed.csv",
        index=False, encoding="utf-8-sig",
    )
    seasonal_aggregate = seasonal.groupby(["model_id", "season"]).agg(
        completed_seeds=("seed", "nunique"),
        mean_RMSE=("season_mean_RMSE", "mean"),
        std_RMSE=("season_mean_RMSE", "std"),
        mean_MAE=("season_mean_MAE", "mean"),
        mean_absolute_Bias=("season_mean_absolute_Bias", "mean"),
        mean_R=("season_mean_R", "mean"),
        mean_R2=("season_mean_R2", "mean"),
    ).reset_index()
    seasonal_aggregate.to_csv(
        output_dir / "Nanyang_Stage2F_SeasonMetrics_Aggregated.csv",
        index=False, encoding="utf-8-sig",
    )

    paired = model_table.pivot(index="seed", columns="model_id", values="date_macro_RMSE").reset_index()
    paired["A3_minus_A6_RMSE"] = paired["A3_PRIOR_MODIS"] - paired["A6_FULL_9CH"]
    paired["winner"] = np.where(
        paired["A3_minus_A6_RMSE"] < 0, "A3_PRIOR_MODIS", "A6_FULL_9CH"
    )
    paired.to_csv(
        output_dir / "Nanyang_Stage2F_PairedSeedComparison.csv",
        index=False, encoding="utf-8-sig",
    )
    audit = {
        "schema_version": "Stage2F_MultiSeedSeasonRobustness_Audit_v1",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [model["id"] for model in config["models"]],
        "random_seeds": config["random_seeds"],
        "selection_split": "val",
        "test_split_accessed": False,
        "selected_candidate": selected,
        "selection_rule": config["robustness_selection_rule"],
        "configuration_sha256": sha256_file(CONFIG_PATH),
        "model_metrics_sha256": sha256_file(summary_path),
        "aggregated_metrics_sha256": sha256_file(
            output_dir / "Nanyang_Stage2F_MultiSeed_AggregatedMetrics.csv"
        ),
        "season_metrics_sha256": sha256_file(
            output_dir / "Nanyang_Stage2F_SeasonMetrics_Aggregated.csv"
        ),
        "paired_comparison_sha256": sha256_file(
            output_dir / "Nanyang_Stage2F_PairedSeedComparison.csv"
        ),
        "overall_status": "PASS",
    }
    with (output_dir / "00_Stage2F_MultiSeedSeasonRobustness_Audit.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(audit, file, ensure_ascii=False, indent=2)
    print(aggregation.to_string(index=False))
    print(paired.to_string(index=False))
    print("STAGE2F_FINAL_AUDIT=PASS")
    print("SELECTED_CANDIDATE=" + selected)
    print("TEST_ACCESSED=0")


if __name__ == "__main__":
    main()

