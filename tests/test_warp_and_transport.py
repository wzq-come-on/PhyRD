from __future__ import annotations

import torch

from phyrd.physics import transport_residual, warp_image, weak_transport_loss


def test_warp_constant_translation() -> None:
    source = torch.zeros(1, 1, 16, 16)
    source[0, 0, 8, 5] = 1.0
    flow = torch.zeros(1, 2, 16, 16)
    flow[:, 0] = 2.0
    warped = warp_image(source, flow)
    assert warped[0, 0, 8, 7] > 0.999


def test_correct_flow_has_lower_transport_residual() -> None:
    first = torch.zeros(1, 1, 16, 16)
    first[0, 0, 8, 5] = 1.0
    flow = torch.zeros(1, 2, 16, 16)
    flow[:, 0] = 2.0
    second = warp_image(first, flow)
    sequence = torch.stack((first, second), dim=1)
    correct, _, _ = transport_residual(sequence, flow[:, None])
    wrong, _, _ = transport_residual(sequence, torch.zeros_like(flow[:, None]))
    assert correct.abs().mean() < wrong.abs().mean()


def test_physics_loss_reaches_prediction_gradient() -> None:
    prediction = torch.rand(1, 4, 1, 16, 16, requires_grad=True)
    flow = torch.zeros(1, 3, 2, 16, 16)
    confidence = torch.ones(1, 3, 16, 16)
    nonadvective = torch.zeros_like(confidence)
    loss, _ = weak_transport_loss(
        prediction,
        flow,
        confidence,
        nonadvective,
        robust_scale=0.1,
        tolerance=0.0,
        pool_sizes=(4, 8),
    )
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad.abs().sum() > 0

