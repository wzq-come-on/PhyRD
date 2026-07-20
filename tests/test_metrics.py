from __future__ import annotations

import pytest
import torch

from phyrd.evaluation import categorical_scores, crps_ensemble, mae, ssim


def test_perfect_categorical_scores_and_pooling() -> None:
    target = torch.zeros(1, 2, 1, 32, 32)
    target[..., ::2, :] = 255.0
    scores = categorical_scores(target, target)
    assert scores["CSI"] == pytest.approx(1.0)
    assert scores["CSI_pool4"] == pytest.approx(1.0)
    assert scores["CSI_pool16"] == pytest.approx(1.0)
    assert scores["HSS"] == pytest.approx(1.0)


def test_mae_and_ssim_identity() -> None:
    target = torch.rand(1, 2, 1, 32, 32)
    assert mae(target, target).item() == 0
    assert ssim(target, target).item() == pytest.approx(1.0, abs=1e-5)


def test_empirical_crps_reference() -> None:
    ensemble = torch.tensor([-1.0, 1.0]).reshape(1, 2, 1, 1, 1, 1)
    target = torch.zeros(1, 1, 1, 1, 1)
    assert crps_ensemble(ensemble, target).item() == pytest.approx(0.5)
