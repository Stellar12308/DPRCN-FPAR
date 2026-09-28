from __future__ import annotations

import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(self.pool(x))


class RCAB(nn.Module):
    def __init__(self, features: int, reduction: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(features, features, 3, padding=1),
            ChannelAttention(features, reduction),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ResidualGroup(nn.Module):
    def __init__(self, features: int, blocks: int, reduction: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[
            RCAB(features, reduction)
            for _ in range(blocks)
        ])
        self.tail = nn.Conv2d(features, features, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.tail(self.blocks(x))


class TopographyConditionedDPRCN(nn.Module):
    """Reliability-aware, topography-conditioned prior residual network."""

    def __init__(
        self,
        prior_mean: float,
        prior_std: float,
        features: int = 48,
        residual_groups: int = 3,
        blocks_per_group: int = 4,
        attention_reduction: int = 8,
        prior_epsilon: float = 0.01,
        maximum_logit_correction: float = 8.0,
        film_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 < prior_epsilon < 0.5:
            raise ValueError("prior_epsilon必须位于(0,0.5)。")
        self.prior_epsilon = float(prior_epsilon)
        self.maximum_logit_correction = float(maximum_logit_correction)
        self.film_scale = float(film_scale)
        self.register_buffer("prior_mean", torch.tensor(float(prior_mean)))
        self.register_buffer("prior_std", torch.tensor(float(prior_std)))

        self.dynamic_encoder = nn.Sequential(
            nn.Conv2d(5, features, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(features, features, 3, padding=1),
        )
        self.topography_encoder = nn.Sequential(
            nn.Conv2d(4, features // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(features // 2, features, 3, padding=1),
            nn.GELU(),
        )
        self.film = nn.Conv2d(features, features * 2, 1)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.trunk = nn.Sequential(*[
            ResidualGroup(
                features,
                blocks_per_group,
                attention_reduction,
            )
            for _ in range(residual_groups)
        ])
        self.trunk_tail = nn.Conv2d(features, features, 3, padding=1)
        self.correction_head = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(features, 1, 3, padding=1),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 9:
            raise ValueError(f"DPRCN要求[B,9,H,W]输入，实际为{tuple(x.shape)}。")
        dynamic = x[:, :5]
        topography = x[:, 5:9]
        dynamic_features = self.dynamic_encoder(dynamic)
        topography_features = self.topography_encoder(topography)
        gamma, beta = self.film(topography_features).chunk(2, dim=1)
        conditioned = (
            dynamic_features * (1.0 + self.film_scale * torch.tanh(gamma))
            + self.film_scale * beta
        )
        residual_features = self.trunk(conditioned)
        features = conditioned + self.trunk_tail(residual_features)
        raw_correction = self.correction_head(features)
        logit_correction = (
            self.maximum_logit_correction * torch.tanh(raw_correction)
        )

        prior = x[:, 0:1] * self.prior_std + self.prior_mean
        prior = prior.clamp(self.prior_epsilon, 1.0 - self.prior_epsilon)
        prior_logit = torch.logit(prior)
        prediction = torch.sigmoid(prior_logit + logit_correction)
        return {
            "prediction": prediction,
            "logit_correction": logit_correction,
            "raw_correction": raw_correction,
            "prior": prior,
        }


def build_model(config: dict, normalization: dict) -> TopographyConditionedDPRCN:
    model_config = config["model"]
    prior_parameters = normalization["parameters"]["Prior_median"]
    return TopographyConditionedDPRCN(
        prior_mean=float(prior_parameters["mean"]),
        prior_std=float(prior_parameters["std_population"]),
        features=int(model_config["features"]),
        residual_groups=int(model_config["residual_groups"]),
        blocks_per_group=int(model_config["blocks_per_group"]),
        attention_reduction=int(model_config["attention_reduction"]),
        prior_epsilon=float(model_config["prior_epsilon"]),
        maximum_logit_correction=float(model_config["maximum_logit_correction"]),
        film_scale=float(model_config["film_scale"]),
    )

