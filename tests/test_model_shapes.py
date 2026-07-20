from __future__ import annotations

import torch

from phyrd.models import PhyRDModel, available_backbones


TINY_SDIR = {
    "patch_size": 8,
    "hidden_size": 32,
    "num_heads": 4,
    "depth": 1,
    "frequency_stride": 16,
    "model_resolution": 32,
}


def deterministic_config() -> dict[str, object]:
    return {"name": "sdir_official", "params": TINY_SDIR}


def test_model_training_shapes_and_gradients() -> None:
    model = PhyRDModel(
        base_channels=8,
        diffusion_steps=20,
        freeze_deterministic=True,
        deterministic=deterministic_config(),
    )
    history = torch.rand(1, 13, 1, 32, 32)
    target = torch.rand(1, 12, 1, 32, 32)
    result = model.diffusion_loss(history, target)
    assert result["trend"].shape == target.shape
    assert result["prediction_x0"].shape == target.shape
    result["loss_gen"].backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.diffusion.parameters()
    )
    assert all(parameter.grad is None for parameter in model.deterministic.parameters())


def test_deterministic_model_supports_5_to_20() -> None:
    model = PhyRDModel(
        input_frames=5,
        output_frames=20,
        base_channels=8,
        diffusion_steps=20,
        freeze_deterministic=False,
        deterministic=deterministic_config(),
    )
    prediction = model(torch.rand(2, 5, 1, 32, 32), stage="deterministic")
    assert prediction.shape == (2, 20, 1, 32, 32)


def test_sdir_official_training_objective_is_finite_and_differentiable() -> None:
    model = PhyRDModel(
        input_frames=5,
        output_frames=20,
        base_channels=8,
        diffusion_steps=20,
        freeze_deterministic=False,
        deterministic=deterministic_config(),
    )
    history = torch.rand(2, 5, 1, 32, 32)
    target = torch.rand(2, 20, 1, 32, 32)
    result = model(history, target, stage="deterministic")
    assert set(("loss_gen", "loss_skeleton", "loss_residual", "loss_pcpsd")) <= result.keys()
    assert result["trend"].shape == target.shape
    assert result["retained_scale"].shape == (2,)
    loss_names = ("loss_gen", "loss_skeleton", "loss_residual", "loss_pcpsd")
    assert all(torch.isfinite(result[name]) for name in loss_names)
    result["loss_gen"].backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.deterministic.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.deterministic.network.sfg_former.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.deterministic.network.fr_refiner.parameters()
    )


def test_registry_exposes_only_the_official_sdir() -> None:
    assert "sdir_official" in available_backbones()
    assert "sdir" not in available_backbones()
    assert "unet2d" not in available_backbones()


def test_unknown_deterministic_backbone_is_rejected() -> None:
    try:
        PhyRDModel(deterministic={"name": "unet2d", "params": {}})
    except ValueError as error:
        assert "available backbones" in str(error)
    else:
        raise AssertionError("unknown deterministic backbone was silently accepted")
