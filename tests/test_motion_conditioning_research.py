from __future__ import annotations

import torch

from phyrd.research.motion_conditioning import build_motion_probe


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.rand(2, 5, 1, 32, 32), torch.rand(2, 20, 1, 32, 32)


def test_motion_probe_shapes() -> None:
    history, trend = _inputs()
    for variant in ("baseline", "globalnet", "tora_adaln"):
        model = build_motion_probe(variant, hidden_size=16, patch_size=4)
        assert model(history, trend).shape == trend.shape


def test_zero_initialized_candidates_equal_baseline() -> None:
    history, trend = _inputs()
    torch.manual_seed(7)
    baseline = build_motion_probe("baseline", hidden_size=16, patch_size=4)
    reference = baseline(history, trend)
    for variant in ("globalnet", "tora_adaln"):
        torch.manual_seed(7)
        candidate = build_motion_probe(variant, hidden_size=16, patch_size=4)
        output = candidate(history, trend)
        assert torch.equal(reference, output)


def test_motion_branches_receive_gradients_after_zero_projection_opens() -> None:
    history, trend = _inputs()
    for variant in ("globalnet", "tora_adaln"):
        model = build_motion_probe(variant, hidden_size=16, patch_size=4)
        projection = (
            model.context_projection
            if variant == "globalnet"
            else model.to_scale_shift
        )
        torch.nn.init.normal_(projection.weight, std=1e-3)
        model(history, trend).square().mean().backward()
        motion_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if not name.startswith("stem.") and parameter.grad is not None
        ]
        assert motion_gradients
        assert any(torch.count_nonzero(gradient).item() for gradient in motion_gradients)
