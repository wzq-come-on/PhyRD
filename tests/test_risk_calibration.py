from __future__ import annotations

import torch

from phyrd.evaluation.risk import LogisticRiskCalibrator, build_risk_batch


def test_risk_batch_has_deployment_features_only() -> None:
    ensemble = torch.rand(2, 3, 4, 1, 16, 16)
    prediction = ensemble.mean(dim=1)
    target = torch.rand_like(prediction)
    history = torch.rand(2, 3, 1, 16, 16)
    residual = torch.rand(2, 3, 16, 16)
    confidence = torch.rand(2, 3, 16, 16)
    nonadvective = torch.rand(2, 3, 16, 16)
    features, targets = build_risk_batch(
        ensemble,
        prediction,
        target,
        history,
        residual,
        confidence,
        nonadvective,
        patch_size=8,
    )
    assert features.shape == (2 * 4 * 4, 9)
    assert set(targets) == {
        "continuous_error",
        "strong_echo_miss",
        "strong_echo_false_alarm",
        "low_patch_csi",
    }
    assert all(values.shape == (features.shape[0],) for values in targets.values())


def test_logistic_risk_calibrator_serializes() -> None:
    torch.manual_seed(0)
    features = torch.randn(64, 9)
    labels = (features[:, 0] > 0).float()
    calibrator = LogisticRiskCalibrator().fit(features, labels, steps=20)
    probabilities = calibrator.predict_proba(features)
    restored = LogisticRiskCalibrator.from_state_dict(calibrator.state_dict())
    assert torch.allclose(probabilities, restored.predict_proba(features))
