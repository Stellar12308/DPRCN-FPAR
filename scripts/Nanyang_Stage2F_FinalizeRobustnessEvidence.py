from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SCRIPT_PATH = Path(__file__).resolve()
STAGE2_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(STAGE2_ROOT))

from dprcn.reproducibility import sha256_file  # noqa: E402


def main() -> None:
    root = Path(os.environ.get("DPRCN_PROJECT_ROOT", ".")).resolve()
    result_dir = root / "04_Results" / "Stage2F_MultiSeedSeasonRobustness"
    model_path = result_dir / "Nanyang_Stage2F_MultiSeed_ModelMetrics.csv"
    aggregate_path = result_dir / "Nanyang_Stage2F_MultiSeed_AggregatedMetrics.csv"
    season_path = result_dir / "Nanyang_Stage2F_SeasonMetrics_Aggregated.csv"
    paired_path = result_dir / "Nanyang_Stage2F_PairedSeedComparison.csv"
    audit_path = result_dir / "00_Stage2F_MultiSeedSeasonRobustness_Audit.json"
    models = pd.read_csv(model_path, encoding="utf-8-sig")
    aggregate = pd.read_csv(aggregate_path, encoding="utf-8-sig")
    seasons = pd.read_csv(season_path, encoding="utf-8-sig")
    paired = pd.read_csv(paired_path, encoding="utf-8-sig")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if len(models) != 6 or not models["test_accessed"].eq(0).all():
        raise RuntimeError("Stage2F模型记录数或测试集隔离状态错误。")
    if audit["overall_status"] != "PASS" or audit["selected_candidate"] != "A3_PRIOR_MODIS":
        raise RuntimeError("Stage2F审计或候选模型不正确。")

    paper_models = aggregate[[
        "model_id", "active_channel_count", "completed_seeds",
        "mean_date_macro_RMSE", "std_date_macro_RMSE",
        "mean_date_macro_MAE", "mean_date_macro_absolute_Bias",
        "mean_date_macro_R", "mean_date_macro_R2",
        "mean_RMSE_difference_from_best", "selected_by_frozen_rule",
    ]].copy()
    paper_models.to_csv(
        result_dir / "Nanyang_Stage2F_PaperReady_MultiSeedTable.csv",
        index=False, encoding="utf-8-sig",
    )
    seasons.to_csv(
        result_dir / "Nanyang_Stage2F_PaperReady_SeasonTable.csv",
        index=False, encoding="utf-8-sig",
    )

    width, height = 2300, 1350
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = ImageFont.load_default(size=40)
    font = ImageFont.load_default(size=29)
    small = ImageFont.load_default(size=24)
    draw.text((90, 40), "Multi-seed and seasonal robustness on the sealed validation split", fill="#111", font=title)

    # Panel A: paired seed RMSE.
    x0, x1, y0, y1 = 160, 1050, 210, 690
    draw.text((x0, 135), "(a) Paired random-seed comparison", fill="#111", font=font)
    ymin, ymax = 0.100, 0.103
    for tick in [0.100, 0.101, 0.102, 0.103]:
        y = y1 - (tick - ymin) / (ymax - ymin) * (y1 - y0)
        draw.line((x0, y, x1, y), fill="#dddddd", width=2)
        draw.text((70, y - 14), f"{tick:.3f}", fill="#555", font=small)
    x_positions = {13: 300, 42: 600, 2026: 900}
    colors = {"A3_PRIOR_MODIS": "#4c78a8", "A6_FULL_9CH": "#f58518"}
    for seed in [13, 42, 2026]:
        seed_rows = models.loc[models["seed"] == seed]
        points = {}
        for _, row in seed_rows.iterrows():
            model = str(row["model_id"])
            x = x_positions[seed] + (-24 if model == "A3_PRIOR_MODIS" else 24)
            value = float(row["date_macro_RMSE"])
            y = y1 - (value - ymin) / (ymax - ymin) * (y1 - y0)
            points[model] = (x, y)
            draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=colors[model])
        draw.line((*points["A3_PRIOR_MODIS"], *points["A6_FULL_9CH"]), fill="#888", width=3)
        draw.text((x_positions[seed] - 45, y1 + 25), str(seed), fill="#222", font=small)
    draw.text((385, y1 + 70), "Random seed", fill="#333", font=small)
    draw.rectangle((210, 780, 245, 815), fill=colors["A3_PRIOR_MODIS"])
    draw.text((260, 780), "A3 (4 channels)", fill="#222", font=small)
    draw.rectangle((570, 780, 605, 815), fill=colors["A6_FULL_9CH"])
    draw.text((620, 780), "A6 (9 channels)", fill="#222", font=small)

    # Panel B: seasonal mean plus/minus standard deviation.
    bx0, bx1, by0, by1 = 1280, 2180, 210, 940
    draw.text((bx0, 135), "(b) Seasonal RMSE across three seeds", fill="#111", font=font)
    smin, smax = 0.05, 0.15
    for tick in [0.06, 0.08, 0.10, 0.12, 0.14]:
        y = by1 - (tick - smin) / (smax - smin) * (by1 - by0)
        draw.line((bx0, y, bx1, y), fill="#dddddd", width=2)
        draw.text((bx0 - 85, y - 14), f"{tick:.2f}", fill="#555", font=small)
    season_order = ["Spring", "Summer", "Autumn", "Winter"]
    for index, season in enumerate(season_order):
        center = bx0 + 120 + index * 215
        for model, offset in [("A3_PRIOR_MODIS", -35), ("A6_FULL_9CH", 35)]:
            row = seasons.loc[(seasons["season"] == season) & (seasons["model_id"] == model)].iloc[0]
            mean, std = float(row["mean_RMSE"]), float(row["std_RMSE"])
            y = by1 - (mean - smin) / (smax - smin) * (by1 - by0)
            y_low = by1 - (mean - std - smin) / (smax - smin) * (by1 - by0)
            y_high = by1 - (mean + std - smin) / (smax - smin) * (by1 - by0)
            x = center + offset
            draw.line((x, y_high, x, y_low), fill=colors[model], width=5)
            draw.line((x - 12, y_high, x + 12, y_high), fill=colors[model], width=4)
            draw.line((x - 12, y_low, x + 12, y_low), fill=colors[model], width=4)
            draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill=colors[model])
        draw.text((center - 55, by1 + 25), season, fill="#222", font=small)

    draw.text(
        (90, 1110),
        "A3: 0.101452 +/- 0.000463; A6: 0.101394 +/- 0.000575. Difference = 0.000058.",
        fill="#222", font=font,
    )
    draw.text(
        (90, 1170),
        "Seasonal validation dates: Spring=1, Summer=1, Autumn=2, Winter=1; seasonal inference is descriptive.",
        fill="#7a3e00", font=small,
    )
    draw.text(
        (90, 1220),
        "The test split was not accessed. A3 remains selected by the frozen 0.002-RMSE equivalence rule.",
        fill="#222", font=small,
    )
    figure_path = result_dir / "Nanyang_Stage2F_MultiSeedSeasonRobustness.png"
    image.save(figure_path, format="PNG", optimize=True)

    report = """# Nanyang Stage2F：A3与A6多随机种子及季节稳健性验证

## 冻结设计

- 比较模型：A3（4通道）与A6（完整9通道）。
- 随机种子：13、42、2026。
- Seed 42复用Stage2D/Stage2E冻结结果，其余组合独立训练。
- 模型选择和季节分析仅使用验证集，测试集访问次数为0。
- 主指标为整景拼接后、按日期等权平均的RMSE。
- 若模型与多种子数值最优模型的RMSE均值差不超过0.002，则选择通道更少的模型。

## 多随机种子结论

- A3：RMSE = 0.101452 ± 0.000463。
- A6：RMSE = 0.101394 ± 0.000575。
- 二者均值差仅0.000058，远小于冻结的0.002等效容差。
- A3的种子间标准差略低，且仅使用4个通道，因此继续选择A3。

## 季节结论

- 春季：A3=0.083439，A6=0.081851。
- 夏季：A3=0.145575，A6=0.146594。
- 秋季：A3=0.109976，A6=0.108566。
- 冬季：A3=0.058297，A6=0.061394。
- 两模型季节变化趋势一致；夏季误差最高，冬季最低。
- 验证集只有5个日期，其中春、夏、冬各1景，秋季2景，因此季节结果只能作为描述性稳健性证据，不能声称具有充分统计推断能力。

## 冻结结论

A3通过多随机种子稳健性验证，保持为最终结构候选。下一步在打开测试集之前，应明确是否按固定Epoch在train+val上重训练最终A3，或冻结Seed 42检查点直接进行一次性测试。二者只能预先选择一种，不能根据测试结果反向切换。
"""
    report_path = result_dir / "Nanyang_Stage2F_RobustnessInterpretation.md"
    report_path.write_text(report, encoding="utf-8")
    final = {
        "schema_version": "Stage2F_FinalEvidenceAudit_v1",
        "selected_candidate": "A3_PRIOR_MODIS",
        "test_split_accessed": False,
        "model_table_sha256": sha256_file(result_dir / "Nanyang_Stage2F_PaperReady_MultiSeedTable.csv"),
        "season_table_sha256": sha256_file(result_dir / "Nanyang_Stage2F_PaperReady_SeasonTable.csv"),
        "figure_sha256": sha256_file(figure_path),
        "report_sha256": sha256_file(report_path),
        "overall_status": "PASS",
    }
    final_path = result_dir / "00_Stage2F_FinalEvidenceAudit.json"
    final_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STAGE2F_FINAL_EVIDENCE=PASS")
    print("SELECTED_CANDIDATE=A3_PRIOR_MODIS")
    print("TEST_ACCESSED=0")


if __name__ == "__main__":
    main()

