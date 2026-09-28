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
    project_root = Path(os.environ.get("DPRCN_PROJECT_ROOT", ".")).resolve()
    output_dir = (
        project_root
        / "04_Results"
        / "Stage2D_DPRCNTraining"
        / "Formal_MainModel_Seed42"
    )
    summary_path = output_dir / "Nanyang_Stage2D_TrainingSummary.csv"
    history_path = output_dir / "training_history.csv"
    checkpoint_path = output_dir / "best_model.pt"
    date_metrics_path = output_dir / "best_epoch_validation_date_metrics.csv"
    for path in [summary_path, history_path, checkpoint_path, date_metrics_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    if len(summary) != 1 or summary.iloc[0]["overall_status"] != "PASS":
        raise RuntimeError("Stage2D正式训练不是PASS。")
    if int(summary.iloc[0]["test_accessed"]) != 0:
        raise RuntimeError("Stage2D错误访问了测试集。")
    history = pd.read_csv(history_path, encoding="utf-8-sig")
    best_epoch = int(summary.iloc[0]["best_epoch"])
    if int(history.loc[history["is_best"] == 1, "epoch"].max()) != best_epoch:
        raise RuntimeError("训练历史与最佳Epoch不一致。")

    figure_path = output_dir / "Nanyang_Stage2D_TrainingCurves.png"
    best_value = float(summary.iloc[0]["best_validation_date_macro_RMSE"])
    # Render with Pillow rather than Matplotlib. The project's Windows NumPy
    # build cannot load its external BLAS DLL for Matplotlib transforms.
    canvas = Image.new("RGB", (2400, 900), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=28)

    def draw_panel(x0, y0, x1, y1, series, title, y_label):
        draw.rectangle((x0, y0, x1, y1), outline="#222222", width=3)
        draw.text((x0, y0 - 48), title, fill="#111111", font=font)
        draw.text((x0, y1 + 15), "Epoch", fill="#111111", font=font)
        draw.text((x0 + 8, y0 + 8), y_label, fill="#555555", font=font)
        epochs = [float(value) for value in history["epoch"]]
        values = [float(value) for _, column, _ in series for value in history[column]]
        minimum = min(values)
        maximum = max(values)
        padding = max((maximum - minimum) * 0.08, 1e-6)
        minimum -= padding
        maximum += padding
        def point(epoch, value):
            px = x0 + (epoch - min(epochs)) / (max(epochs) - min(epochs)) * (x1 - x0)
            py = y1 - (value - minimum) / (maximum - minimum) * (y1 - y0)
            return (round(px), round(py))
        for label, column, color in series:
            points = [point(epoch, value) for epoch, value in zip(epochs, history[column])]
            draw.line(points, fill=color, width=5)
        return point

    draw_panel(
        120, 130, 1120, 760,
        [("Training loss", "train_total_loss", "#1f77b4")],
        "Training loss", "Quality-weighted loss",
    )
    validation_point = draw_panel(
        1300, 130, 2300, 760,
        [
            ("Date-macro RMSE", "val_date_macro_RMSE", "#d62728"),
            ("Global RMSE", "val_global_RMSE", "#ff7f0e"),
        ],
        "Validation performance", "RMSE",
    )
    bx, by = validation_point(best_epoch, best_value)
    draw.ellipse((bx - 9, by - 9, bx + 9, by + 9), fill="black")
    draw.text((1320, 790), "red: date-macro RMSE | orange: global RMSE | black: selected", fill="#222222", font=font)
    canvas.save(figure_path, format="PNG", optimize=True)

    source_paths = {
        "training_config": STAGE2_ROOT / "configs" / "Nanyang_Stage2D_Training_Config_v1.json",
        "dataset_module": STAGE2_ROOT / "dprcn" / "dataset.py",
        "model_module": STAGE2_ROOT / "dprcn" / "model.py",
        "loss_module": STAGE2_ROOT / "dprcn" / "losses.py",
        "training_script": STAGE2_ROOT / "scripts" / "Nanyang_Stage2D_TrainDPRCN.py",
        "model_loss_test": STAGE2_ROOT / "tests" / "test_model_and_loss.py",
        "real_batch_test": STAGE2_ROOT / "tests" / "test_real_batch_overfit.py",
    }
    result_paths = {
        "best_checkpoint": checkpoint_path,
        "training_history": history_path,
        "training_summary": summary_path,
        "best_validation_date_metrics": date_metrics_path,
        "training_curves": figure_path,
    }
    document = {
        "schema_version": "Stage2D_TrainingAudit_SHA256_v1",
        "test_split_accessed": False,
        "best_epoch": best_epoch,
        "best_validation_date_macro_RMSE": best_value,
        "source_sha256": {
            name: sha256_file(path)
            for name, path in source_paths.items()
        },
        "result_sha256": {
            name: sha256_file(path)
            for name, path in result_paths.items()
        },
        "overall_status": "PASS",
    }
    manifest_path = output_dir / "Nanyang_Stage2D_SourceAndResult_SHA256.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(document, file, ensure_ascii=False, indent=2)
    print("STAGE2D_FINAL_AUDIT=PASS")
    print("best_epoch", best_epoch)
    print("best_validation_date_macro_RMSE", best_value)
    print("checkpoint_sha256", document["result_sha256"]["best_checkpoint"])
    print("test_split_accessed", 0)


if __name__ == "__main__":
    main()

