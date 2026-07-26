from __future__ import annotations

import csv

import torch
from torch import nn

from phyrd.evaluation.results_registry import append_full_test_result
from phyrd.models.composer import ForecastComposer
from phyrd.models.deterministic.base import DeterministicLossOutput
from phyrd.models.probabilistic.trajres_diffusion import (
    TrajectoryResidualDiffusionModel,
)


class TinyDeterministic(nn.Module):
    def __init__(self, output_frames: int) -> None:
        super().__init__()
        self.output_frames = output_frames
        self.scale = nn.Parameter(torch.tensor(0.9))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.scale * history[:, -1:].expand(
            -1, self.output_frames, -1, -1, -1
        )

    def training_loss(
        self, history: torch.Tensor, target: torch.Tensor
    ) -> DeterministicLossOutput:
        prediction = self(history)
        return DeterministicLossOutput(
            loss=(prediction - target).abs().mean(),
            prediction=prediction,
        )


def tiny_probability() -> TrajectoryResidualDiffusionModel:
    return TrajectoryResidualDiffusionModel(
        2,
        4,
        segment_frames=2,
        base_channels=8,
        attention_heads=2,
        diffusion_steps=8,
    )


def test_trajres_training_and_sequential_sampling_shapes() -> None:
    model = tiny_probability()
    history = torch.rand(1, 2, 1, 16, 16)
    trend = torch.rand(1, 4, 1, 16, 16)
    target = torch.rand_like(trend)
    result = model.training_loss(history, target, trend)
    assert torch.isfinite(result["loss_gen"])
    assert result["prediction_x0"].shape == target.shape
    result["loss_gen"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    ensemble = model.sample(history, trend, ensemble_size=2, sampling_steps=2)
    assert ensemble.shape == (1, 2, 4, 1, 16, 16)
    assert ensemble.min() >= 0 and ensemble.max() <= 1


def test_joint_stage_backpropagates_into_both_components() -> None:
    composer = ForecastComposer(
        TinyDeterministic(4),
        tiny_probability(),
        freeze_deterministic=False,
        deterministic_name="tiny",
    )
    history = torch.rand(1, 2, 1, 16, 16)
    target = torch.rand(1, 4, 1, 16, 16)
    result = composer(history, target, stage="joint_residual")
    total = 0.5 * result["loss_det"] + 0.5 * result["loss_diff"]
    total.backward()
    assert composer.deterministic.scale.grad is not None
    assert any(parameter.grad is not None for parameter in composer.diffusion.parameters())
    state = composer.diffusion.state_dict()
    assert "sqrt_alphas_cumprod" in state
    assert any(name.startswith("denoiser.") for name in state)


def test_result_registry_is_idempotent_by_evidence(tmp_path) -> None:
    registry = tmp_path / "results.csv"
    metrics = {
        "split": "report_test",
        "samples": 5600,
        "ensemble_size": 10,
        "world_size": 8,
        "epoch": 12,
        "CRPS": 1.0,
        "CSI": 0.2,
        "HSS": 0.3,
        "MAE": 0.04,
        "MSE": 0.01,
        "SSIM": 0.7,
        "status": "completed",
    }
    first = append_full_test_result(
        registry,
        metrics=metrics,
        experiment="test",
        deterministic_backbone="tiny",
        probabilistic_module="trajres",
        evidence="metrics/full.json",
    )
    second = append_full_test_result(
        registry,
        metrics=metrics,
        experiment="test",
        deterministic_backbone="tiny",
        probabilistic_module="trajres",
        evidence="metrics/full.json",
    )
    assert first == second == "R-000"
    with registry.open("r", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 1
