from __future__ import annotations

import torch
from torch.nn import functional as F


def warp_image(
    source: torch.Tensor,
    flow: torch.Tensor,
    *,
    padding_mode: str = "border",
    return_valid: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Forward-advection warp using flow `(dx,dy)` in pixels per frame.

    The output at target coordinate `p` samples `source[p-flow(p)]`.
    """
    if source.ndim != 4:
        raise ValueError(f"source must have [N,C,H,W], got {tuple(source.shape)}")
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"flow must have [N,2,H,W], got {tuple(flow.shape)}")
    if source.shape[0] != flow.shape[0] or source.shape[-2:] != flow.shape[-2:]:
        raise ValueError("source and flow batch/spatial shapes must match")
    batch, _, height, width = source.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=source.device, dtype=source.dtype),
        torch.arange(width, device=source.device, dtype=source.dtype),
        indexing="ij",
    )
    source_x = x[None] - flow[:, 0]
    source_y = y[None] - flow[:, 1]
    if width > 1:
        grid_x = 2.0 * source_x / (width - 1) - 1.0
    else:
        grid_x = torch.zeros_like(source_x)
    if height > 1:
        grid_y = 2.0 * source_y / (height - 1) - 1.0
    else:
        grid_y = torch.zeros_like(source_y)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    warped = F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
    valid = (
        (source_x >= 0)
        & (source_x <= width - 1)
        & (source_y >= 0)
        & (source_y <= height - 1)
    ).to(source.dtype)
    if return_valid:
        return warped, valid[:, None]
    return warped

