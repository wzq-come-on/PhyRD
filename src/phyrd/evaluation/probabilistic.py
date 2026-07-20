from __future__ import annotations

import torch


def crps_ensemble(ensemble: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Empirical ensemble CRPS averaged over batch, lead, channel, and space.

    `ensemble` is `[B,K,T,C,H,W]`; values remain in the caller's declared domain.
    """
    if ensemble.ndim != 6 or target.ndim != 5:
        raise ValueError("CRPS expects ensemble [B,K,T,C,H,W] and target [B,T,C,H,W]")
    if ensemble.shape[0] != target.shape[0] or ensemble.shape[2:] != target.shape[1:]:
        raise ValueError("ensemble and target dimensions do not match")
    members = ensemble.shape[1]
    if members <= 0:
        raise ValueError("ensemble must contain at least one member")
    observation_term = (ensemble - target[:, None]).abs().mean()
    pairwise = ensemble.new_zeros(())
    for first in range(members):
        for second in range(members):
            pairwise = pairwise + (ensemble[:, first] - ensemble[:, second]).abs().mean()
    pairwise = pairwise / (members * members)
    return observation_term - 0.5 * pairwise

