from __future__ import annotations

from collections.abc import Sequence

import torch

from .categorical import SEVIR_THRESHOLDS, categorical_scores
from .continuous import lpips_distance, mae, ssim
from .probabilistic import crps_ensemble


@torch.no_grad()
def evaluate_forecasts(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    ensemble: torch.Tensor | None = None,
    thresholds: Sequence[float] = SEVIR_THRESHOLDS,
    vil_scale: float = 255.0,
    include_lpips: bool = True,
    lpips_net: str = "alex",
) -> dict[str, float | str]:
    """Evaluate registered metrics with explicit encoded-VIL and perceptual domains."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    prediction_unit = prediction.clamp(0.0, 1.0)
    target_unit = target.clamp(0.0, 1.0)
    prediction_vil = prediction_unit * vil_scale
    target_vil = target_unit * vil_scale
    result: dict[str, float | str] = categorical_scores(
        prediction_vil, target_vil, thresholds=thresholds
    )
    result["MAE"] = float(mae(prediction_vil, target_vil).item())
    result["SSIM"] = float(ssim(prediction_unit, target_unit).item())
    if include_lpips:
        result["LPIPS"] = float(
            lpips_distance(prediction_unit, target_unit, net=lpips_net).item()
        )
    else:
        result["LPIPS"] = "SKIPPED"
    if ensemble is None:
        ensemble = prediction_unit[:, None]
        result["CRPS_note"] = "K=1 deterministic sanity check; CRPS equals MAE"
    ensemble_vil = ensemble.clamp(0.0, 1.0) * vil_scale
    result["CRPS"] = float(crps_ensemble(ensemble_vil, target_vil).item())
    result["metric_domain"] = "CSI/HSS/MAE/CRPS: encoded VIL [0,255]; LPIPS/SSIM: [0,1]"
    result["ensemble_size"] = float(ensemble.shape[1])
    return result

