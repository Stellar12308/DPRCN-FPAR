from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


TEST_PATH = Path(__file__).resolve()
STAGE2_ROOT = TEST_PATH.parents[1]
sys.path.insert(0, str(STAGE2_ROOT))

from dprcn.losses import build_loss  # noqa: E402
from dprcn.model import build_model  # noqa: E402


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    config = load_json(
        STAGE2_ROOT / "configs" / "Nanyang_Stage2D_Training_Config_v1.json"
    )
    root = Path(config["project_root"])
    normalization = load_json(root / config["stage2b_parameters"])
    model = build_model(config, normalization)
    criterion = build_loss(config)

    x = torch.randn(2, 9, 128, 128)
    target = torch.rand(2, 1, 128, 128)
    mask = (torch.rand(2, 1, 128, 128) > 0.2).float()
    choices = torch.tensor([0.5, 0.8, 1.0])
    weight = choices[torch.randint(0, 3, mask.shape)] * mask
    output = model(x)
    prediction = output["prediction"]
    if prediction.shape != target.shape:
        raise AssertionError("模型输出形状错误。")
    if not torch.isfinite(prediction).all():
        raise AssertionError("模型输出存在NaN或Inf。")
    if not ((prediction > 0.0) & (prediction < 1.0)).all():
        raise AssertionError("模型输出不在开放区间(0,1)。")
    losses = criterion(prediction, target, mask, weight)
    losses["total"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise AssertionError("反向传播梯度无效。")

    prediction_2 = prediction.detach().clone().requires_grad_(True)
    changed_target = target.clone()
    changed_target[mask == 0] = 999.0
    loss_1 = criterion(prediction_2, target, mask, weight)["total"]
    loss_2 = criterion(prediction_2, changed_target, mask, weight)["total"]
    if not torch.allclose(loss_1, loss_2, rtol=0.0, atol=1e-7):
        raise AssertionError("无效像元错误地影响了损失。")
    print("MODEL_AND_LOSS_TEST=PASS")
    print(
        "prediction_range",
        float(prediction.detach().min()),
        float(prediction.detach().max()),
    )
    print("parameter_count", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()

