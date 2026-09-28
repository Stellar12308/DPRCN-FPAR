from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(os.environ.get("DPRCN_PROJECT_ROOT", ".")).resolve()
OUTPUT_DIR = PROJECT_ROOT / "04_Results" / "Stage3G_NanyangFinalExperimentFreeze"
SCRIPT_PATH = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def artifact(path: Path, role: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        "relative_path": rel(path),
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stage2g = PROJECT_ROOT / "04_Results" / "Stage2G_FinalA3TrainValFit"
    stage2h = PROJECT_ROOT / "04_Results" / "Stage2H_OneTimeTestEvaluation"
    stage2i = PROJECT_ROOT / "04_Results" / "Stage2I_LeaveOneSeasonOutRobustness"
    stage3a = PROJECT_ROOT / "04_Results" / "Stage3A_BaselineReferenceManifest"
    stage3c = PROJECT_ROOT / "04_Results" / "Stage3C_ESTARFM_FrozenTest"
    stage3d = PROJECT_ROOT / "04_Results" / "Stage3D_ESTARFM_TestEvaluation"
    stage3e = PROJECT_ROOT / "04_Results" / "Stage3E_AdaptedPKSTEM_FrozenTest"
    stage3f = PROJECT_ROOT / "04_Results" / "Stage3F_FinalCommonMaskComparison"
    stage3code = PROJECT_ROOT / "02_Code" / "04_Stage3_Baselines"

    required = [
        (stage2g / "01_TrainValNormalization" / "Nanyang_Stage2G_FinalTrainingProtocol_Config_v1_Frozen.json", "final_training_protocol"),
        (stage2g / "01_TrainValNormalization" / "Nanyang_Stage2G_TrainValNormalizationParameters.json", "final_normalization"),
        (stage2g / "02_FinalModel" / "Nanyang_Stage2G_FinalA3_TrainVal_Seed42_Epoch32.pt", "final_model_checkpoint"),
        (stage2h / "00_Stage2H_OneTimeTestEvidence.json", "one_time_test_evidence"),
        (stage2h / "Nanyang_Stage2H_TestDateMetrics.csv", "a3_original_test_metrics"),
        (stage2h / "Nanyang_Stage2H_TestOverallMetrics.csv", "a3_original_test_summary"),
        (stage2i / "00_Stage2I_FinalAudit.json", "leave_one_season_out_evidence"),
        (stage2i / "Nanyang_Stage2I_FoldSummary.csv", "leave_one_season_out_summary"),
        (stage3a / "Nanyang_Stage3A_BaselineReferenceManifest_5TestDates.csv", "baseline_reference_manifest"),
        (stage3c / "00_Stage3C_ESTARFM_Evidence.json", "estarfm_prediction_evidence"),
        (stage3c / "Nanyang_Stage3C_ESTARFM_SceneAudit.csv", "estarfm_prediction_audit"),
        (stage3d / "Nanyang_Stage3D_ESTARFM_TestOverallMetrics.csv", "estarfm_independent_summary"),
        (stage3e / "00_Stage3E_AdaptedPKSTEM_Evidence.json", "adapted_pkstem_prediction_evidence"),
        (stage3e / "Nanyang_Stage3E_AdaptedPKSTEM_SceneAudit_5TestDates.csv", "adapted_pkstem_prediction_audit"),
        (stage3f / "00_Stage3F_CommonMaskComparison_Evidence.json", "final_comparison_evidence"),
        (stage3f / "Nanyang_Stage3F_CommonMask_DateMetrics_Long.csv", "final_date_metrics"),
        (stage3f / "Nanyang_Stage3F_CommonMask_MacroAndGlobalMetrics.csv", "final_macro_global_metrics"),
        (stage3f / "Nanyang_Stage3F_CommonMask_ModelMetricTable_DoubleHeader.csv", "final_publication_table_csv"),
        (stage3f / "Nanyang_Stage3F_FinalThreeModel_CommonMaskComparison.xlsx", "final_review_workbook"),
    ]
    for date in ["20230130", "20230514", "20230623", "20230818", "20231021"]:
        required.extend([
            (stage3f / f"Nanyang_{date}_DPRCN_A3_FinalPrediction.tif", "final_a3_prediction"),
            (stage3c / f"Nanyang_{date}_ESTARFM_FPAR.tif", "final_estarfm_prediction"),
            (stage3e / f"Nanyang_{date}_AdaptedPKSTEM_ModeIII_FPAR.tif", "final_adapted_pkstem_prediction"),
            (stage3f / f"Nanyang_{date}_ThreeModelCommonValidMask.tif", "final_common_mask"),
        ])
    code_files = sorted(
        path for path in stage3code.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".json"} and "__pycache__" not in path.parts
    )
    records = [artifact(path, role) for path, role in required]
    records.extend(artifact(path, "stage3_code_or_config") for path in code_files)
    seen, unique_records = set(), []
    for item in records:
        if item["relative_path"] not in seen:
            unique_records.append(item)
            seen.add(item["relative_path"])

    final_metrics = pd.read_csv(stage3f / "Nanyang_Stage3F_CommonMask_MacroAndGlobalMetrics.csv", encoding="utf-8-sig")
    season_metrics = pd.read_csv(stage2i / "Nanyang_Stage2I_FoldSummary.csv", encoding="utf-8-sig")
    mask_audit = pd.read_csv(stage3f / "Nanyang_Stage3F_CommonMask_Audit_5Dates.csv", encoding="utf-8-sig")
    winner = final_metrics.sort_values("RMSE").iloc[0]
    a3 = final_metrics.loc[final_metrics["model"].eq("DPRCN-A3")].iloc[0]
    estarfm = final_metrics.loc[final_metrics["model"].eq("ESTARFM")].iloc[0]
    pkstem = final_metrics.loc[final_metrics["model"].eq("PK-STEM (adapted)")].iloc[0]

    protocol = {
        "schema_version": "Stage3G_NanyangFinalExperimentProtocol_v1_FROZEN",
        "frozen_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S %z"),
        "region": "Nanyang",
        "development_model": "DPRCN-A3 / A3_PRIOR_MODIS",
        "active_channels": ["historical_prior_FPAR", "target_MODIS_FPAR", "DEM", "MODIS_QC_or_validity_channel"],
        "final_fit": {"data": "train+val only", "epochs": 32, "seed": 42, "test_early_stopping": False},
        "test_dates": mask_audit["date"].astype(str).str.zfill(8).tolist(),
        "test_status": "OPENED_ONCE_FOR_STAGE2H_AND_CLOSED_FOR_MODEL_SELECTION",
        "post_test_rule": "No architecture, channel, epoch, loss, normalization, or hyperparameter may be selected using these five test labels.",
        "baseline_policy": {
            "ESTARFM": "two neighboring fine/coarse pairs; target coarse image; test target label evaluation-only",
            "PK-STEM": "adapted Mode III; 110-scene historical prior; native-500m MODIS correction; 100-member bidirectional ensemble filtering; train+val Landsat assimilation; test labels excluded",
        },
        "final_comparison_mask": "per-date intersection of target label and DPRCN-A3, ESTARFM, and adapted PK-STEM valid pixels",
        "primary_metric": "five-date macro RMSE",
        "reported_metrics": ["RMSE", "MAE", "Bias", "absolute Bias", "Pearson R", "R2", "ubRMSE", "SSIM"],
        "final_primary_winner": str(winner["model"]),
        "ansai_external_validation_rule": "Nanyang decisions remain frozen before any Ansai label is inspected for model selection. First report zero-shot transfer; any fine-tuning is a separate experiment.",
    }
    protocol_path = OUTPUT_DIR / "Nanyang_Stage3G_FinalExperimentProtocol_v1_FROZEN.json"
    with protocol_path.open("w", encoding="utf-8") as file:
        json.dump(protocol, file, ensure_ascii=False, indent=2)

    manifest = {
        "schema_version": "Stage3G_FreezeManifest_v1",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "project_root": str(PROJECT_ROOT),
        "artifact_count": len(unique_records),
        "artifacts": unique_records,
        "protocol_relative_path": rel(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "verification_policy": "Recompute SHA-256 and require exact match before reuse or Ansai migration.",
    }
    manifest_path = OUTPUT_DIR / "00_Stage3G_FreezeManifest_v1.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    def pct_reduction(reference: float, candidate: float) -> float:
        return (reference - candidate) / reference * 100.0

    season_lines = "\n".join(
        f"| {row.held_out_season} | {row.date_macro_RMSE:.4f} | {row.date_macro_MAE:.4f} | {row.date_macro_Pearson_R:.4f} | {row.date_macro_R2:.4f} | {row.date_macro_SSIM:.4f} |"
        for row in season_metrics.itertuples()
    )
    model_lines = "\n".join(
        f"| {row.model} | {row.global_RMSE:.4f} | {row.global_MAE:.4f} | {row.global_Pearson_R:.4f} | {row.global_R2:.4f} | {row.RMSE:.4f} | {row.SSIM:.4f} |"
        for row in final_metrics.itertuples()
    )
    document = f"""# Nanyang Stage3G Final Experiment Freeze

## 1. Freeze status

- Status: **FROZEN / PASS**
- Final development model: **DPRCN-A3**
- Fixed training protocol: train+val, seed 42, 32 epochs, no test-based early stopping.
- The five Nanyang test labels have already been opened for the one-time final evaluation. They are now closed for all model selection.
- Any later correction of a software defect must be documented as a protocol deviation and must not be chosen because it improves test performance.

## 2. Scientific question closed by the Nanyang experiment

The Nanyang experiment tests whether a learnable historical-prior-guided model can reconstruct 30-m FPAR more accurately than classical fusion and prior-assimilation baselines without requiring fine-resolution observations at the target date.

## 3. Final common-mask comparison

All three models were evaluated on exactly the same valid pixels for each date. The pooled common-mask sample size is 131,928 pixels.

| Model | Global RMSE | Global MAE | Global R | Global R2 | Date-macro RMSE | Date-macro SSIM |
|---|---:|---:|---:|---:|---:|---:|
{model_lines}

DPRCN-A3 reduces global RMSE by {pct_reduction(float(estarfm.global_RMSE), float(a3.global_RMSE)):.2f}% relative to ESTARFM and by {pct_reduction(float(pkstem.global_RMSE), float(a3.global_RMSE)):.2f}% relative to adapted PK-STEM. Its date-macro RMSE is also the lowest. ESTARFM has the highest macro SSIM and the smallest absolute macro bias, so the result does not support a claim that DPRCN is superior for every metric or every date.

## 4. Difficult date

The date 2023-08-18 has only 7,833 common valid pixels (15.22% coverage), and all methods have weak or negative R2. It must be retained and discussed as a difficult case rather than removed post hoc.

## 5. Leave-one-season-out robustness

| Held-out season | RMSE | MAE | R | R2 | SSIM |
|---|---:|---:|---:|---:|---:|
{season_lines}

Seasonal transfer is strongest in winter and autumn and weakest in summer. These experiments support robustness analysis but also show that summer generalization remains a limitation.

## 6. Baseline interpretation

- ESTARFM is a classical two-pair spatial-temporal fusion baseline and uses neighboring fine-resolution observations.
- PK-STEM is explicitly an **adapted reproducible Mode III implementation**, not a claim of source-code-identical reproduction. It uses the 110-scene historical prior, MODIS correction, 100-member bidirectional ensemble filtering, and train+val Landsat assimilation.
- DPRCN-A3 does not consume neighboring test-date fine-resolution FPAR observations during inference. This difference in operational input requirements must accompany the accuracy comparison.

## 7. Test-set governance

The Nanyang test set may be used only for frozen reporting, error analysis, and figure generation. It may not be used to change architecture, channels, loss, normalization, epochs, random seed, threshold, or baseline parameters. New hypotheses must be assessed in Ansai or in a separately declared future dataset.

## 8. Ansai transition

1. Audit Ansai data with the frozen Stage0-Stage1 rules.
2. Preserve the A3 channel definition and Nanyang normalization for the first zero-shot experiment.
3. Do not use Ansai labels for model selection before reporting zero-shot performance.
4. If fine-tuning is studied, create a new split and report it separately from zero-shot transfer.
5. Recreate ESTARFM and adapted PK-STEM with equivalent region-specific inputs and use a common-mask comparison.

## 9. Reproducibility evidence

`00_Stage3G_FreezeManifest_v1.json` records SHA-256, size, role, and relative path for every frozen artifact. Before reuse or migration, recompute hashes and require exact agreement.
"""
    closure_path = OUTPUT_DIR / "Nanyang_Stage3G_NanyangExperimentClosure_v1.md"
    closure_path.write_text(document, encoding="utf-8")

    previous_workflow_path = (
        PROJECT_ROOT
        / "04_Results"
        / "00_WorkflowDocumentation"
        / "Nanyang_Stage0_to_Stage2H_CompleteReproducibleWorkflow_and_AnsaiMigration_v1.md"
    )
    if not previous_workflow_path.exists():
        raise FileNotFoundError(previous_workflow_path)
    workflow_appendix = f"""

---

# Stage3A–Stage3G：基线对比、稳健性检验与南阳实验冻结

## Stage3A：五个冻结测试日期的基线输入清单

- 只使用 Stage2G 冻结后声明的五个测试日期。
- 为 DPRCN-A3、ESTARFM 与 adapted PK-STEM 建立同一日期索引。
- 测试标签只允许用于最终评价，不允许用于参数选择。

## Stage3B–Stage3D：ESTARFM 基线

- 以目标日期前后相邻的两组精细/粗分辨率影像和目标日期粗分辨率影像进行预测。
- 生成五景 30 m FPAR，并独立审计空间参考、网格、掩膜和数值范围。
- 固定实现后才在五个测试日期上评价；不得因测试结果调整窗口或阈值。

## Stage3E：adapted PK-STEM 基线

- 明确标记为 adapted reproducible Mode III，而非原作者源码逐行复现。
- 使用 2013–2022 年110景历史先验、原生500 m MODIS校正、100成员双向集合滤波。
- 只同化 train+val Landsat；测试标签不进入状态更新或参数选择。

## Stage3F：三模型共同有效像元公平比较

逐日期共同掩膜定义为：

$$
M_t=M_t^{{label}}\cap M_t^{{A3}}\cap M_t^{{ESTARFM}}\cap M_t^{{PKSTEM}}.
$$

所有模型均在同一个 $M_t$ 上计算 RMSE、MAE、Bias、绝对 Bias、Pearson $R$、$R^2$、ubRMSE 与 SSIM。最终结果为：

| Model | Global RMSE | Global MAE | Global R | Global R2 | Date-macro RMSE | Date-macro SSIM |
|---|---:|---:|---:|---:|---:|---:|
{model_lines}

DPRCN-A3 的全局 RMSE 相对 ESTARFM 降低 {pct_reduction(float(estarfm.global_RMSE), float(a3.global_RMSE)):.2f}%，相对 adapted PK-STEM 降低 {pct_reduction(float(pkstem.global_RMSE), float(a3.global_RMSE)):.2f}%。2023-08-18 是保留的困难日期，不得事后删除。

## Stage2I/Stage3G：季节稳健性与最终冻结

留一季节法完全排除被留出季节的训练样本，结果如下：

| Held-out season | RMSE | MAE | R | R2 | SSIM |
|---|---:|---:|---:|---:|---:|
{season_lines}

南阳最终协议冻结为 A3、train+val、Seed 42、32 epochs、无测试早停。五个测试日期已完成一次性评价，此后不得再用于架构、通道、损失、归一化、epoch、随机种子、阈值或基线参数的选择。

## 安塞迁移的强制顺序

1. 先按冻结的 Stage0–Stage1 SOP完成安塞数据审计。
2. 保持 A3 通道语义、模型参数与南阳归一化不变，先做 zero-shot。
3. 在 zero-shot 结果报告前，不得使用安塞标签选择模型。
4. 如需微调，必须另建训练/验证/测试划分，并作为独立实验报告。
5. ESTARFM 和 adapted PK-STEM 按等价输入条件重建，再在逐日期共同有效掩膜上比较。

## 可复现性入口

- 最终协议：`04_Results/Stage3G_NanyangFinalExperimentFreeze/Nanyang_Stage3G_FinalExperimentProtocol_v1_FROZEN.json`
- 哈希清单：`04_Results/Stage3G_NanyangFinalExperimentFreeze/00_Stage3G_FreezeManifest_v1.json`
- 南阳结项：`04_Results/Stage3G_NanyangFinalExperimentFreeze/Nanyang_Stage3G_NanyangExperimentClosure_v1.md`
"""
    master_workflow_path = (
        PROJECT_ROOT
        / "04_Results"
        / "00_WorkflowDocumentation"
        / "Nanyang_Stage0_to_Stage3G_CompleteReproducibleWorkflow_and_AnsaiMigration_v1.md"
    )
    master_workflow_path.write_text(
        previous_workflow_path.read_text(encoding="utf-8") + workflow_appendix,
        encoding="utf-8",
    )

    # Stage3G自身产生的冻结协议与说明文档也必须进入清单。
    for generated_path, generated_role in [
        (protocol_path, "frozen_final_experiment_protocol"),
        (closure_path, "nanyang_experiment_closure"),
        (master_workflow_path, "stage0_to_stage3g_master_workflow"),
    ]:
        relative_path = rel(generated_path)
        if relative_path not in seen:
            unique_records.append(artifact(generated_path, generated_role))
            seen.add(relative_path)
    manifest["artifact_count"] = len(unique_records)
    manifest["artifacts"] = unique_records
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    runlog = {
        "status": "STAGE3G_NANYANG_FINAL_FREEZE=PASS",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "artifact_count": len(unique_records),
        "final_primary_winner": str(winner["model"]),
        "date_macro_RMSE": float(winner["RMSE"]),
        "protocol_sha256": sha256(protocol_path),
        "freeze_manifest_sha256": sha256(manifest_path),
        "closure_document_sha256": sha256(closure_path),
        "master_workflow_sha256": sha256(master_workflow_path),
    }
    (OUTPUT_DIR / "00_Stage3G_RunLog.json").write_text(json.dumps(runlog, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STAGE3G_NANYANG_FINAL_FREEZE=PASS")
    print(json.dumps(runlog, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

