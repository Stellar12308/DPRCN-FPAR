from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    return (values * weights).sum() / (weights.sum() + epsilon)


class QualityWeightedMaskedLoss(nn.Module):
    def __init__(
        self,
        smooth_l1_beta: float = 0.05,
        gradient_loss_weight: float = 0.05,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.gradient_loss_weight = float(gradient_loss_weight)
        self.epsilon = float(epsilon)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        quality_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not (
            prediction.shape
            == target.shape
            == valid_mask.shape
            == quality_weight.shape
        ):
            raise ValueError("prediction/target/mask/weight形状必须一致。")
        mask = (valid_mask > 0.5).to(prediction.dtype)
        weights = mask * quality_weight.clamp_min(0.0)
        if weights.sum().detach().item() <= 0:
            raise ValueError("当前Batch没有有效监督权重。")
        pixel_map = F.smooth_l1_loss(
            prediction,
            target,
            reduction="none",
            beta=self.smooth_l1_beta,
        )
        pixel_loss = weighted_mean(pixel_map, weights, self.epsilon)

        pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        pair_x = mask[..., :, 1:] * mask[..., :, :-1]
        pair_weight_x = 0.5 * (
            quality_weight[..., :, 1:] + quality_weight[..., :, :-1]
        ) * pair_x
        grad_x = weighted_mean(
            torch.abs(pred_dx - target_dx),
            pair_weight_x,
            self.epsilon,
        )

        pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        pair_y = mask[..., 1:, :] * mask[..., :-1, :]
        pair_weight_y = 0.5 * (
            quality_weight[..., 1:, :] + quality_weight[..., :-1, :]
        ) * pair_y
        grad_y = weighted_mean(
            torch.abs(pred_dy - target_dy),
            pair_weight_y,
            self.epsilon,
        )
        gradient_loss = grad_x + grad_y
        total = pixel_loss + self.gradient_loss_weight * gradient_loss
        return {
            "total": total,
            "pixel": pixel_loss,
            "gradient": gradient_loss,
            "valid_weight_sum": weights.sum(),
        }


def build_loss(config: dict) -> QualityWeightedMaskedLoss:
    loss_config = config["loss"]
    return QualityWeightedMaskedLoss(
        smooth_l1_beta=float(loss_config["smooth_l1_beta"]),
        gradient_loss_weight=float(loss_config["gradient_loss_weight"]),
        epsilon=float(loss_config["epsilon"]),
    )

