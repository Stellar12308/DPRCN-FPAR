from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SCRIPT_PATH = Path(__file__).resolve()
STAGE2_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(STAGE2_ROOT))

from dprcn.reproducibility import sha256_file  # noqa: E402


def render_figure(table: pd.DataFrame, output_path: Path) -> None:
    width, height = 2200, 1250
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=42)
    font = ImageFont.load_default(size=30)
    small = ImageFont.load_default(size=25)
    draw.text((100, 45), "Stage2E validation-only controlled ablation", fill="#111111", font=title_font)
    draw.text(
        (100, 100),
        "Primary metric: date-macro RMSE after full-scene mosaicking; test split sealed",
        fill="#444444",
        font=small,
    )
    plot_left, plot_right = 780, 2050
    plot_top, row_height = 190, 115
    minimum, maximum = 0.095, 0.145
    colors = {
        "analytical_baseline": "#9e9e9e",
        "trained_ablation": "#4c78a8",
        "frozen_stage2d_model": "#f58518",
    }
    for tick in [0.10, 0.11, 0.12, 0.13, 0.14]:
        x = plot_left + (tick - minimum) / (maximum - minimum) * (plot_right - plot_left)
        draw.line((x, plot_top - 25, x, plot_top + row_height * len(table)), fill="#dddddd", width=2)
        draw.text((x - 35, plot_top - 70), f"{tick:.2f}", fill="#555555", font=small)
    for index, row in table.reset_index(drop=True).iterrows():
        y = plot_top + index * row_height
        label = f'{int(row["validation_rank"])}. {row["variant_id"]}'
        draw.text((100, y + 20), label, fill="#111111", font=font)
        value = float(row["date_macro_RMSE"])
        x = plot_left + (value - minimum) / (maximum - minimum) * (plot_right - plot_left)
        color = colors[str(row["variant_type"])]
        draw.rectangle((plot_left, y + 20, x, y + 72), fill=color)
        if int(row["selected_by_frozen_rule"]) == 1:
            draw.rectangle((plot_left - 8, y + 12, x + 8, y + 80), outline="#c00000", width=7)
            suffix = " selected"
        else:
            suffix = ""
        draw.text((x + 18, y + 28), f"{value:.5f}{suffix}", fill="#111111", font=small)
    draw.text(
        (100, 1130),
        "Selection rule: least complex trainable variant within 0.002 RMSE of the numerical best.",
        fill="#333333",
        font=small,
    )
    image.save(output_path, format="PNG", optimize=True)


def main() -> None:
    root = Path(os.environ.get("DPRCN_PROJECT_ROOT", ".")).resolve()
    result_dir = root / "04_Results" / "Stage2E_ValidationAblationAndModelSelection"
    summary_path = result_dir / "Nanyang_Stage2E_AblationValidationSummary.csv"
    audit_path = result_dir / "00_Stage2E_ValidationAblationAudit.json"
    config_path = STAGE2_ROOT / "configs" / "Nanyang_Stage2E_ValidationAblation_Config_v1.json"
    table = pd.read_csv(summary_path, encoding="utf-8-sig")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = {
        "B0_PRIOR_RAW", "B1_MODIS_RAW", "A1_PRIOR", "A2_PRIOR_RELIABILITY",
        "A3_PRIOR_MODIS", "A4_PRIOR_MODIS_QC", "A5_PLUS_DEM", "A6_FULL_9CH",
    }
    if set(table["variant_id"]) != expected or len(table) != 8:
        raise RuntimeError("消融结果不是预期的8项。")
    if not table["test_accessed"].eq(0).all():
        raise RuntimeError("检测到测试集访问。")
    if audit["overall_status"] != "PASS" or audit["selected_variant"] != "A3_PRIOR_MODIS":
        raise RuntimeError("选择审计状态不正确。")
    prior_rmse = float(table.loc[table["variant_id"] == "B0_PRIOR_RAW", "date_macro_RMSE"].iloc[0])
    full_rmse = float(table.loc[table["variant_id"] == "A6_FULL_9CH", "date_macro_RMSE"].iloc[0])
    table["RMSE_improvement_vs_raw_prior"] = prior_rmse - table["date_macro_RMSE"]
    table["RMSE_relative_improvement_vs_raw_prior_pct"] = (
        table["RMSE_improvement_vs_raw_prior"] / prior_rmse * 100.0
    )
    table["RMSE_difference_vs_full9"] = table["date_macro_RMSE"] - full_rmse
    paper_path = result_dir / "Nanyang_Stage2E_PaperReady_AblationTable.csv"
    table.to_csv(paper_path, index=False, encoding="utf-8-sig")
    figure_path = result_dir / "Nanyang_Stage2E_ValidationAblation_RMSE.png"
    render_figure(table.sort_values("date_macro_RMSE"), figure_path)

    selected_source = result_dir / "A3_PRIOR_MODIS" / "best_model.pt"
    bundle_dir = result_dir / "SelectedCandidate_A3_PRIOR_MODIS"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    selected_copy = bundle_dir / "Nanyang_Stage2E_SelectedCandidate_A3_PRIOR_MODIS_Seed42.pt"
    shutil.copy2(selected_source, selected_copy)
    shutil.copy2(config_path, bundle_dir / config_path.name.replace(".json", "_Frozen.json"))
    selection = {
        "schema_version": "Stage2E_SelectedCandidate_v1",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_status": "VALIDATION_SELECTED_PENDING_ROBUSTNESS_TESTS",
        "selected_variant": "A3_PRIOR_MODIS",
        "active_channels": [
            "Prior_median", "Prior_N_obs_log1p", "Prior_sigma", "MODIS_FPAR"
        ],
        "best_epoch": int(table.loc[table["variant_id"] == "A3_PRIOR_MODIS", "best_epoch"].iloc[0]),
        "validation_date_macro_RMSE": float(
            table.loc[table["variant_id"] == "A3_PRIOR_MODIS", "date_macro_RMSE"].iloc[0]
        ),
        "difference_from_full9_RMSE": float(
            table.loc[table["variant_id"] == "A3_PRIOR_MODIS", "RMSE_difference_vs_full9"].iloc[0]
        ),
        "test_split_accessed": False,
        "checkpoint_sha256": sha256_file(selected_copy),
        "ablation_summary_sha256": sha256_file(summary_path),
        "paper_table_sha256": sha256_file(paper_path),
        "figure_sha256": sha256_file(figure_path),
        "configuration_sha256": sha256_file(config_path),
        "next_required_step": "Multi-seed or seasonal robustness validation before one-time test evaluation.",
        "overall_status": "PASS",
    }
    selection_path = bundle_dir / "00_Stage2E_SelectedCandidate_Audit.json"
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STAGE2E_FINAL_AUDIT=PASS")
    print("SELECTED_CANDIDATE=A3_PRIOR_MODIS")
    print("CANDIDATE_STATUS=VALIDATION_SELECTED_PENDING_ROBUSTNESS_TESTS")
    print("TEST_ACCESSED=0")
    print("CHECKPOINT_SHA256=" + selection["checkpoint_sha256"])


if __name__ == "__main__":
    main()

