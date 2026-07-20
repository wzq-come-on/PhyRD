from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from phyrd.physics.warp import warp_image


@dataclass
class MotionFields:
    flow: torch.Tensor
    c_flow: torch.Tensor
    m_nadv: torch.Tensor
    forward_flow: torch.Tensor
    backward_flow: torch.Tensor


def estimate_farneback_pair(
    previous: torch.Tensor, current: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate forward/backward optical flow on CPU, returning `(dx,dy)` on input device."""
    if previous.shape != current.shape or previous.ndim != 4 or previous.shape[1] != 1:
        raise ValueError("previous/current must share [B,1,H,W]")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for Farneback motion") from exc
    device = previous.device
    dtype = previous.dtype
    forward: list[torch.Tensor] = []
    backward: list[torch.Tensor] = []
    for prev_item, curr_item in zip(previous.detach().cpu(), current.detach().cpu()):
        prev_array = np.ascontiguousarray(prev_item[0].numpy().astype(np.float32) * 255.0)
        curr_array = np.ascontiguousarray(curr_item[0].numpy().astype(np.float32) * 255.0)
        arguments = dict(
            pyr_scale=0.5,
            levels=4,
            winsize=21,
            iterations=4,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )
        fwd = cv2.calcOpticalFlowFarneback(prev_array, curr_array, None, **arguments)
        bwd = cv2.calcOpticalFlowFarneback(curr_array, prev_array, None, **arguments)
        forward.append(torch.from_numpy(fwd).permute(2, 0, 1))
        backward.append(torch.from_numpy(bwd).permute(2, 0, 1))
    return (
        torch.stack(forward).to(device=device, dtype=dtype),
        torch.stack(backward).to(device=device, dtype=dtype),
    )


def _robust_positive_scale(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(1)
    median = flat.median(dim=1).values
    deviation = (flat - median[:, None]).abs().median(dim=1).values
    return (median + 3.0 * deviation).clamp_min(1e-4).reshape(-1, 1, 1, 1)


def _texture_penalty(image: torch.Tensor) -> torch.Tensor:
    sobel_x = image.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    sobel_y = sobel_x.t()
    gx = F.conv2d(image, sobel_x.reshape(1, 1, 3, 3), padding=1)
    gy = F.conv2d(image, sobel_y.reshape(1, 1, 3, 3), padding=1)
    gradient = (gx.square() + gy.square() + 1e-8).sqrt()
    scale = _robust_positive_scale(gradient)
    return torch.exp(-gradient / scale)


def build_motion_fields(
    history: torch.Tensor,
    output_frames: int = 12,
    *,
    lead_decay: float = 8.0,
) -> MotionFields:
    if history.ndim != 5 or history.shape[2] != 1 or history.shape[1] < 2:
        raise ValueError("history must have [B,T>=2,1,H,W]")
    previous = history[:, -2]
    current = history[:, -1]
    forward, backward = estimate_farneback_pair(previous, current)
    warped_backward, valid = warp_image(backward, forward, return_valid=True)
    fb_error = (forward + warped_backward).square().sum(dim=1, keepdim=True).sqrt()
    fb_norm = fb_error / _robust_positive_scale(fb_error)
    texture = _texture_penalty(current)
    confidence_observed = valid * torch.exp(-fb_norm - texture)

    transitions = output_frames - 1
    leads = torch.arange(
        1, transitions + 1, device=history.device, dtype=history.dtype
    ).reshape(1, transitions, 1, 1)
    decay = torch.exp(-leads / lead_decay)
    c_flow = confidence_observed[:, 0, None] * decay

    advected_previous = warp_image(previous, forward)
    intensity_residual = (current - advected_previous).abs()
    residual_scale = _robust_positive_scale(intensity_residual)
    historical_trend = (current - previous).abs()
    trend_scale = _robust_positive_scale(historical_trend)
    evidence = 0.5 * (intensity_residual / residual_scale) + 0.5 * (
        historical_trend / trend_scale
    )
    m_observed = (1.0 - torch.exp(-evidence)).clamp(0.0, 1.0)
    m_nadv = m_observed[:, 0, None].repeat(1, transitions, 1, 1)
    flow = forward[:, None].repeat(1, transitions, 1, 1, 1)
    return MotionFields(flow, c_flow, m_nadv, forward, backward)
