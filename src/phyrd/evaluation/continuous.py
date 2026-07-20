from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    return (prediction - target).abs().mean()


def _gaussian_window(
    size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    coordinates = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    gaussian = torch.exp(-(coordinates**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    window = gaussian[:, None] * gaussian[None, :]
    return window.reshape(1, 1, size, size).repeat(channels, 1, 1, 1)


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Mean frame-wise SSIM using a local Gaussian window."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("SSIM expects matching [B,T,C,H,W] tensors")
    batch, frames, channels, height, width = prediction.shape
    if min(height, width) < window_size:
        raise ValueError("spatial dimensions are smaller than the SSIM window")
    pred = prediction.reshape(batch * frames, channels, height, width)
    truth = target.reshape(batch * frames, channels, height, width)
    window = _gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)
    padding = window_size // 2
    mu_pred = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_truth = F.conv2d(truth, window, padding=padding, groups=channels)
    mu_pred_sq = mu_pred.square()
    mu_truth_sq = mu_truth.square()
    mu_product = mu_pred * mu_truth
    variance_pred = F.conv2d(pred.square(), window, padding=padding, groups=channels) - mu_pred_sq
    variance_truth = (
        F.conv2d(truth.square(), window, padding=padding, groups=channels) - mu_truth_sq
    )
    covariance = F.conv2d(pred * truth, window, padding=padding, groups=channels) - mu_product
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_product + c1) * (2 * covariance + c2)
    denominator = (mu_pred_sq + mu_truth_sq + c1) * (
        variance_pred + variance_truth + c2
    )
    return (numerator / denominator.clamp_min(torch.finfo(pred.dtype).eps)).mean()


_LPIPS_MODELS: dict[tuple[str, str], Any] = {}


def _lpips_model(device: torch.device, net: str) -> Any:
    key = (str(device), net)
    if key not in _LPIPS_MODELS:
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError("LPIPS is required but is not installed") from exc
        model = lpips.LPIPS(net=net, verbose=False).eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _LPIPS_MODELS[key] = model
    return _LPIPS_MODELS[key]


@torch.no_grad()
def lpips_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    net: str = "alex",
    chunk_size: int = 4,
) -> torch.Tensor:
    """Frame-wise pretrained LPIPS; inputs must be in `[0,1]`."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("LPIPS expects matching [B,T,C,H,W] tensors")
    if prediction.shape[2] not in (1, 3):
        raise ValueError("LPIPS supports one or three channels")
    batch, frames, channels, height, width = prediction.shape
    pred = prediction.clamp(0.0, 1.0).reshape(batch * frames, channels, height, width)
    truth = target.clamp(0.0, 1.0).reshape(batch * frames, channels, height, width)
    if channels == 1:
        pred = pred.repeat(1, 3, 1, 1)
        truth = truth.repeat(1, 3, 1, 1)
    pred = pred * 2.0 - 1.0
    truth = truth * 2.0 - 1.0
    model = _lpips_model(pred.device, net)
    values = []
    for start in range(0, pred.shape[0], chunk_size):
        values.append(model(pred[start : start + chunk_size], truth[start : start + chunk_size]))
    distance = torch.cat(values, dim=0).mean()
    if not math.isfinite(float(distance.item())):
        raise RuntimeError("LPIPS produced a non-finite value")
    return distance

