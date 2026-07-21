from __future__ import annotations

import torch
from pathlib import Path
from torch import nn

from phyrd.models import available_probabilistic_models, build_composite_from_config
from phyrd.models.composer import ForecastComposer
from phyrd.models.deterministic import register_backbone
from phyrd.models.probabilistic import build_probabilistic


class TinyDeterministic(nn.Module):
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return history[:, -1:].repeat(1, 2, 1, 1, 1)


register_backbone("tiny_test", TinyDeterministic)


class TinyProbabilistic(nn.Module):
    def training_loss(self, history, target, trend):
        return {"loss_gen": (trend - target).square().mean(), "clean_prediction": trend - trend}

    @torch.no_grad()
    def sample(self, history, trend, *, ensemble_size=1, sampling_steps=20, guidance_factory=None):
        return trend.unsqueeze(1).repeat(1, ensemble_size, 1, 1, 1, 1)


def test_composer_passes_trend_between_components() -> None:
    composer = ForecastComposer(TinyDeterministic(), TinyProbabilistic())
    history = torch.zeros(2, 3, 1, 4, 4)
    target = torch.ones(2, 2, 1, 4, 4)
    result = composer.training_loss(history, target)
    assert result["trend"].shape == target.shape
    assert composer.sample(history, ensemble_size=3).shape == (2, 3, 2, 1, 4, 4)


def test_residual_diffusion_is_config_selectable() -> None:
    assert "residual_diffusion" in available_probabilistic_models()
    model = build_probabilistic(
        "residual_diffusion",
        input_frames=2,
        output_frames=2,
        params={"base_channels": 8, "diffusion_steps": 8},
    )
    assert model.diffusion.diffusion_steps == 8


def test_universal_residual_diffusion_uses_fixed_residual_coordinates() -> None:
    model = build_probabilistic(
        "universal_residual_diffusion",
        input_frames=2,
        output_frames=2,
        params={"base_channels": 8, "diffusion_steps": 8},
    )
    assert torch.allclose(model.diffusion.residual_center, torch.zeros_like(model.diffusion.residual_center))
    assert torch.allclose(model.diffusion.residual_scale, torch.ones_like(model.diffusion.residual_scale))


def test_composite_factory_accepts_legacy_diffusion_config() -> None:
    config = {
        "model": {
            "base_channels": 8,
            "diffusion_steps": 8,
            "freeze_deterministic": True,
            "deterministic": {"name": "tiny_test", "params": {}},
            "diffusion": {"prediction_type": "v"},
        }
    }
    model = build_composite_from_config(config, input_frames=2, output_frames=2)
    assert model.deterministic_name == "tiny_test"
    assert model.diffusion.prediction_type == "v"


def test_composite_factory_selects_a_frozen_backbone_pool(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tiny.pt"
    source = TinyDeterministic(2, 2)
    torch.save({"deterministic": source.state_dict()}, checkpoint)
    config = {
        "model": {
            "base_channels": 8,
            "diffusion_steps": 8,
            "freeze_deterministic": True,
            "deterministic_pool": {
                "selection": "round_robin",
                "members": [
                    {"name": "tiny_test", "params": {}, "checkpoint": str(checkpoint)},
                ],
            },
            "probabilistic": {"name": "universal_residual_diffusion", "params": {}},
        }
    }
    model = build_composite_from_config(config, input_frames=2, output_frames=2)
    assert model.select_backbone_for_step(0, 42) == "tiny_test"
    assert all(not parameter.requires_grad for parameter in model.deterministic.parameters())
