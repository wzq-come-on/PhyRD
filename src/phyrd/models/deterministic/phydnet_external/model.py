"""Thin adapter for a separately maintained PhyDNet checkout.

The source remains outside PhyRD because it is an external baseline.  This
adapter only normalizes its inference contract for a frozen backbone pool.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


class ExternalPhyDNetForecast(nn.Module):
    def __init__(
        self,
        input_frames: int,
        output_frames: int,
        *,
        root: str,
        model_resolution: int = 128,
    ) -> None:
        super().__init__()
        external_root = Path(root)
        if not external_root.is_dir():
            raise FileNotFoundError(f"PhyDNet root not found: {external_root}")
        if str(external_root) not in sys.path:
            sys.path.insert(0, str(external_root))
        from models.phydnet_sevir import get_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_model(
            in_shape=(1, int(model_resolution), int(model_resolution)),
            T_in=input_frames,
            T_out=output_frames,
            device=device,
            lucc_embed_dim=0,
            lucc_mask=0,
        )

    def load_external_checkpoint(self, checkpoint: str | Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model", payload.get("state_dict", payload))
        self.model.load_state_dict(state, strict=True)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.model.inference(history).clamp(0.0, 1.0)
