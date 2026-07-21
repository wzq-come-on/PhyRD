"""PhyRD adapter around the unmodified official SDIR implementation.

The upstream model code lives under ``third_party/sdir_official``. This module
only translates PhyRD's ``[B,T,1,H,W]`` contract and exposes the common
deterministic-backbone interface.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from ..base import DeterministicLossOutput


_OFFICIAL_ROOT = Path(__file__).resolve().parents[5] / "third_party" / "sdir_official"
if str(_OFFICIAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_OFFICIAL_ROOT))

from model.model import Network  # type: ignore[import-not-found]  # noqa: E402
from model.psdloss import PSDLoss  # type: ignore[import-not-found]  # noqa: E402
from utils import get_coarse_condition  # type: ignore[import-not-found]  # noqa: E402


class OfficialSDIRForecast(nn.Module):
    """Official SDIR Network with only a PhyRD data/interface adapter."""

    def __init__(
        self,
        input_frames: int,
        output_frames: int,
        *,
        patch_size: int = 4,
        hidden_size: int = 512,
        num_heads: int = 4,
        depth: int = 8,
        frequency_stride: int = 16,
        curriculum_alpha: float = 1.0,
        curriculum_beta: float = 3.0,
        pcpsd_weight: float = 0.01,
        model_resolution: int = 128,
    ) -> None:
        super().__init__()
        self.input_frames = int(input_frames)
        self.output_frames = int(output_frames)
        self.model_resolution = int(model_resolution)
        self.frequency_stride = int(frequency_stride)
        self.curriculum_alpha = float(curriculum_alpha)
        self.curriculum_beta = float(curriculum_beta)
        self.pcpsd_weight = float(pcpsd_weight)
        self.configs = SimpleNamespace(
            img_channel=1,
            input_length=self.input_frames,
            output_length=self.output_frames,
            img_size=self.model_resolution,
            patch_size=int(patch_size),
            depth=int(depth),
            batch_size=1,
        )
        self.network = Network(
            self.configs,
            self.model_resolution // int(patch_size),
            hidden_size=int(hidden_size),
            num_heads=int(num_heads),
            depth=int(depth),
        )
        self.mae = nn.L1Loss(reduction="mean")
        self.pcpsd = PSDLoss(
            nbins=64,
            log_spectrum=True,
            normalize="shape",
            high_freq_boost=1.0,
            apply_hann_window=True,
            apply_log1p=True,
            log1p_scale=50.0,
        )

    def _check_inputs(self, history: torch.Tensor, target: torch.Tensor | None = None) -> None:
        if history.ndim != 5 or history.shape[1] != self.input_frames or history.shape[2] != 1:
            raise ValueError(f"history must be [B,{self.input_frames},1,H,W], got {tuple(history.shape)}")
        if history.shape[-2:] != (self.model_resolution, self.model_resolution):
            raise ValueError(f"expected {self.model_resolution}x{self.model_resolution} inputs")
        if target is not None and tuple(target.shape) != (
            history.shape[0], self.output_frames, 1, self.model_resolution, self.model_resolution
        ):
            raise ValueError(f"target has incompatible shape {tuple(target.shape)}")

    def _network_forward(
        self, history: torch.Tensor, condition: torch.Tensor, scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.configs.batch_size = int(history.shape[0])
        return self.network(history, condition, scale)

    def training_loss(self, history: torch.Tensor, target: torch.Tensor) -> DeterministicLossOutput:
        self._check_inputs(history, target)
        condition, scale = get_coarse_condition(
            target,
            alpha=self.curriculum_alpha,
            beta=self.curriculum_beta,
        )
        pred1, pred2, pred3 = self._network_forward(history, condition, scale)
        loss_skeleton = self.mae(pred1, target)
        loss_residual = self.mae(pred2, target - pred1)
        loss_pcpsd_per_frame = self.pcpsd(pred3.squeeze(2), target.squeeze(2), scale)
        alpha = 0.01 * (scale / self.model_resolution).square().repeat_interleave(
            self.output_frames
        )
        loss_pcpsd = (alpha * loss_pcpsd_per_frame).mean()
        loss = loss_skeleton + loss_residual + self.pcpsd_weight * loss_pcpsd
        return DeterministicLossOutput(
            loss=loss,
            prediction=pred3,
            metrics={
                "loss_skeleton": loss_skeleton,
                "loss_residual": loss_residual,
                "loss_pcpsd": loss_pcpsd,
                "retained_scale": scale,
            },
        )

    @torch.no_grad()
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        self._check_inputs(history)
        condition = history.new_zeros(
            history.shape[0], self.output_frames, 1, self.model_resolution, self.model_resolution
        )
        schedule = list(range(0, self.model_resolution, self.frequency_stride))
        prediction = condition
        for index, retained_size in enumerate(schedule):
            scale = history.new_full((history.shape[0],), float(retained_size))
            _, _, prediction = self._network_forward(history, condition, scale)
            if index + 1 < len(schedule):
                next_size = schedule[index + 1]
                resized = F.interpolate(
                    prediction.flatten(0, 1), size=(next_size, next_size), mode="bicubic"
                )
                condition = F.interpolate(
                    resized,
                    size=(self.model_resolution, self.model_resolution),
                    mode="bicubic",
                ).reshape_as(prediction)
        return prediction.clamp(0.0, 1.0)
