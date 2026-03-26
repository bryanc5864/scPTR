"""Negative binomial distribution utilities for DeepPTR."""

from __future__ import annotations

import torch
from torch import Tensor


def log_nb_positive(
    x: Tensor,
    mu: Tensor,
    theta: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Log-likelihood of the negative binomial distribution (mean-dispersion).

    Parameterized by mean ``mu`` and inverse dispersion ``theta`` so that
    Var = mu + mu^2 / theta.  Larger theta → less overdispersion.

    Parameters
    ----------
    x : Tensor
        Observed counts (non-negative integers), shape ``(N, G)``.
    mu : Tensor
        Predicted mean, shape ``(N, G)``.
    theta : Tensor
        Inverse dispersion, shape ``(G,)`` or ``(N, G)``.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    Tensor
        Log-probability, same shape as *x*.
    """
    mu = mu.clamp(min=eps)
    theta = theta.clamp(min=eps)

    log_theta_mu = torch.log(theta + mu + eps)

    return (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta + eps) - log_theta_mu)
        + x * (torch.log(mu + eps) - log_theta_mu)
    )
