from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


RISK_FEATURE_NAMES = (
    "log_u_ens",
    "log_r_phys",
    "one_minus_c_flow",
    "m_nadv",
    "predicted_intensity",
    "input_intensity",
    "lead_time",
    "predicted_gradient",
    "predicted_object_size",
)
RISK_TARGET_NAMES = (
    "continuous_error",
    "strong_echo_miss",
    "strong_echo_false_alarm",
    "low_patch_csi",
)


def _pad_leads(value: torch.Tensor, frames: int) -> torch.Tensor:
    if value.ndim == 4:
        value = value.unsqueeze(2)
    if value.ndim != 5 or value.shape[1] not in {frames - 1, frames}:
        raise ValueError(f"risk lead tensor must have {frames - 1} or {frames} leads")
    if value.shape[1] == frames:
        return value
    return torch.cat((value, value[:, -1:]), dim=1)


def _patch_mean(value: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, frames, channels, height, width = value.shape
    if channels != 1 or height % patch_size or width % patch_size:
        raise ValueError("risk inputs must have one channel and divisible spatial dimensions")
    pooled = F.avg_pool2d(
        value.reshape(batch * frames, 1, height, width), patch_size, stride=patch_size
    )
    return pooled.reshape(batch, frames, -1)


def _patch_max(value: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, frames, channels, height, width = value.shape
    pooled = F.max_pool2d(
        value.reshape(batch * frames, channels, height, width), patch_size, stride=patch_size
    )
    return pooled.reshape(batch, frames, -1)


def build_risk_batch(
    ensemble: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    history: torch.Tensor,
    r_phys: torch.Tensor,
    c_flow: torch.Tensor,
    m_nadv: torch.Tensor,
    *,
    patch_size: int = 16,
    vil_scale: float = 255.0,
    error_threshold: float = 32.0,
    strong_threshold: float = 219.0,
    low_csi_threshold: float = 0.3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if ensemble.ndim != 6 or prediction.ndim != 5 or target.shape != prediction.shape:
        raise ValueError("risk tensors have incompatible shapes")
    if ensemble.shape[0] != prediction.shape[0] or ensemble.shape[2:] != prediction.shape[1:]:
        raise ValueError("ensemble and prediction dimensions do not match")
    batch, frames = prediction.shape[:2]
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    spread = ensemble.float().std(dim=1, unbiased=False).mean(dim=2, keepdim=True)
    history_last = history[:, -1:].expand(-1, frames, -1, -1, -1)
    gradient_x = prediction[:, :, :, :, 1:] - prediction[:, :, :, :, :-1]
    gradient_y = prediction[:, :, :, 1:, :] - prediction[:, :, :, :-1, :]
    gradient = prediction.new_zeros(prediction.shape)
    gradient[:, :, :, :, 1:] += gradient_x.abs()
    gradient[:, :, :, 1:, :] += gradient_y.abs()
    predicted_vil = prediction * vil_scale
    target_vil = target * vil_scale
    patch_spread = _patch_mean(spread, patch_size).clamp_min(1e-6)
    patch_r_phys = _patch_mean(_pad_leads(r_phys, frames), patch_size).clamp_min(1e-6)
    patch_c_flow = _patch_mean(_pad_leads(c_flow, frames), patch_size).clamp(0.0, 1.0)
    patch_m_nadv = _patch_mean(_pad_leads(m_nadv, frames), patch_size).clamp(0.0, 1.0)
    patch_prediction = _patch_mean(predicted_vil, patch_size)
    patch_input = _patch_mean(history_last * vil_scale, patch_size)
    patch_lead = prediction.new_tensor(
        torch.arange(1, frames + 1, device=prediction.device, dtype=prediction.dtype)
    ).reshape(1, frames, 1).expand(batch, -1, patch_prediction.shape[-1])
    patch_gradient = _patch_mean(gradient, patch_size)
    patch_object_size = _patch_mean(
        (predicted_vil >= strong_threshold).to(prediction.dtype), patch_size
    )
    features = torch.stack(
        (
            patch_spread.log(),
            patch_r_phys.log(),
            (1.0 - patch_c_flow).clamp_min(0.0),
            patch_m_nadv,
            patch_prediction / vil_scale,
            patch_input / vil_scale,
            patch_lead / frames,
            patch_gradient,
            patch_object_size,
        ),
        dim=-1,
    ).reshape(-1, len(RISK_FEATURE_NAMES))
    prediction_max = _patch_max(predicted_vil, patch_size)
    target_max = _patch_max(target_vil, patch_size)
    error = _patch_mean((predicted_vil - target_vil).abs(), patch_size)
    target_event = target_max >= strong_threshold
    prediction_event = prediction_max >= strong_threshold
    overlap = target_event & prediction_event
    union = target_event | prediction_event
    patch_csi = overlap.to(prediction.dtype) / union.to(prediction.dtype).clamp_min(1.0)
    targets = {
        "continuous_error": (error >= error_threshold).reshape(-1).float(),
        "strong_echo_miss": (target_event & ~prediction_event).reshape(-1).float(),
        "strong_echo_false_alarm": (~target_event & prediction_event).reshape(-1).float(),
        "low_patch_csi": (patch_csi < low_csi_threshold).reshape(-1).float(),
    }
    return features, targets


class LogisticRiskCalibrator:
    def __init__(self, feature_names: tuple[str, ...] = RISK_FEATURE_NAMES) -> None:
        self.feature_names = tuple(feature_names)
        self.mean: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self.weight: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        steps: int = 300,
        learning_rate: float = 0.05,
        weight_decay: float = 1e-4,
    ) -> "LogisticRiskCalibrator":
        if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
            raise ValueError("features must be [N,D] and labels must be [N]")
        if features.shape[1] != len(self.feature_names):
            raise ValueError("feature count does not match feature_names")
        x = features.float()
        y = labels.float().clamp(0.0, 1.0)
        self.mean = x.mean(dim=0)
        self.scale = x.std(dim=0, unbiased=False).clamp_min(1e-6)
        normalized = (x - self.mean) / self.scale
        model = nn.Linear(normalized.shape[1], 1)
        positive = y.sum().clamp_min(1.0)
        negative = (y.numel() - y.sum()).clamp_min(1.0)
        positive_weight = (negative / positive).to(normalized)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for _ in range(steps):
            logits = model(normalized).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=positive_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self.weight = model.weight.detach().squeeze(0)
        self.bias = model.bias.detach().squeeze(0)
        return self

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.scale is None or self.weight is None or self.bias is None:
            raise RuntimeError("calibrator must be fitted before prediction")
        normalized = (features.float() - self.mean) / self.scale
        return torch.sigmoid(normalized @ self.weight + self.bias)

    def state_dict(self) -> dict[str, object]:
        if self.mean is None or self.scale is None or self.weight is None or self.bias is None:
            raise RuntimeError("calibrator must be fitted before serialization")
        return {
            "feature_names": self.feature_names,
            "mean": self.mean,
            "scale": self.scale,
            "weight": self.weight,
            "bias": self.bias,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "LogisticRiskCalibrator":
        calibrator = cls(tuple(state["feature_names"]))
        calibrator.mean = torch.as_tensor(state["mean"]).float()
        calibrator.scale = torch.as_tensor(state["scale"]).float()
        calibrator.weight = torch.as_tensor(state["weight"]).float()
        calibrator.bias = torch.as_tensor(state["bias"]).float()
        return calibrator
