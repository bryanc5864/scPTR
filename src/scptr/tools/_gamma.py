"""Per-cell degradation rate (gamma) estimation."""

from __future__ import annotations

import logging

import numpy as np
from anndata import AnnData

from .._constants import (
    SMOOTHED_UNSPLICED,
    SMOOTHED_SPLICED,
    BETA,
    GAMMA,
    DEFAULT_CLIP_QUANTILE,
)
from .._utils import get_layer, require_layers, require_var, log_params
from .._numba_kernels import _compute_gamma_kernel

logger = logging.getLogger("scptr")


def estimate_gamma(
    adata: AnnData,
    clip_quantile: float = DEFAULT_CLIP_QUANTILE,
    mode: str = "steady_state",
    velocity_layer: str | None = None,
    min_spliced: float = 0.01,
) -> None:
    """Compute per-cell, per-gene mRNA degradation rate gamma.

    Parameters
    ----------
    adata
        Annotated data matrix with smoothed layers and ``var['beta']``.
    clip_quantile
        Upper quantile for per-gene clipping (default 0.99).
    mode
        ``'steady_state'`` (default): ``gamma = beta * u / s``.
        ``'dynamic'``: ``gamma = (beta * u - ds/dt) / s``, using the
        full ODE instead of the steady-state assumption.
    velocity_layer
        Layer name containing the ``ds/dt`` estimate, required when
        ``mode='dynamic'``.
    min_spliced
        Minimum smoothed spliced count for reliable gamma estimation.
        Cells with ``Ms < min_spliced`` for a given gene get gamma=0.
    """
    require_layers(adata, SMOOTHED_UNSPLICED, SMOOTHED_SPLICED)
    require_var(adata, BETA)

    u_smooth = get_layer(adata, SMOOTHED_UNSPLICED).astype(np.float32)
    s_smooth = get_layer(adata, SMOOTHED_SPLICED).astype(np.float32)
    beta = adata.var[BETA].values.astype(np.float32)

    if mode == "steady_state":
        logger.info("Estimating gamma in steady-state mode.")
        raw_gamma = _steady_state_gamma(u_smooth, s_smooth, beta, min_spliced)
    elif mode == "dynamic":
        if velocity_layer is None:
            raise ValueError(
                "velocity_layer must be provided when mode='dynamic'."
            )
        require_layers(adata, velocity_layer)
        ds_dt = get_layer(adata, velocity_layer).astype(np.float32)
        logger.info("Estimating gamma in dynamic mode (velocity_layer=%r).", velocity_layer)
        raw_gamma = _dynamic_gamma(u_smooth, s_smooth, beta, ds_dt, min_spliced)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'steady_state' or 'dynamic'.")

    # Per-gene clip at upper quantile
    clip_vals = np.quantile(raw_gamma, clip_quantile, axis=0).astype(np.float32)
    clip_vals = np.maximum(clip_vals, 1e-6)
    out = np.minimum(raw_gamma, clip_vals[np.newaxis, :])

    # Global clip: use per-gene median distribution to set a reasonable cap.
    # Cap at 10x the 99th percentile of per-gene medians — this removes
    # extreme gene-level outliers while preserving meaningful variation.
    gene_medians = np.median(out, axis=0)
    positive_medians = gene_medians[gene_medians > 0]
    if len(positive_medians) > 0:
        global_cap = 10.0 * np.percentile(positive_medians, 99)
        out = np.minimum(out, global_cap)

    adata.layers[GAMMA] = out

    log_params(adata, "estimate_gamma", {
        "clip_quantile": clip_quantile,
        "mode": mode,
        "velocity_layer": velocity_layer,
        "min_spliced": min_spliced,
    })


def _steady_state_gamma(
    u_smooth: np.ndarray,
    s_smooth: np.ndarray,
    beta: np.ndarray,
    min_spliced: float = 0.01,
) -> np.ndarray:
    """Steady-state gamma: beta * u / s.

    Sets gamma=0 where spliced counts are below threshold (unreliable ratio).
    """
    # Mask: only compute gamma where spliced signal is meaningful
    reliable = s_smooth >= min_spliced
    s_safe = np.where(reliable, s_smooth, 1.0)  # placeholder where unreliable
    raw_gamma = beta[np.newaxis, :] * u_smooth / s_safe
    raw_gamma = np.clip(raw_gamma, 0.0, None)
    # Zero out unreliable entries
    raw_gamma[~reliable] = 0.0
    return raw_gamma.astype(np.float32)


def _dynamic_gamma(
    u_smooth: np.ndarray,
    s_smooth: np.ndarray,
    beta: np.ndarray,
    ds_dt: np.ndarray,
    min_spliced: float = 0.01,
) -> np.ndarray:
    """Dynamic gamma: (beta * u - ds/dt) / s from the full ODE."""
    reliable = s_smooth >= min_spliced
    s_safe = np.where(reliable, s_smooth, 1.0)
    raw_gamma = (beta[np.newaxis, :] * u_smooth - ds_dt) / s_safe
    raw_gamma = np.clip(raw_gamma, 0.0, None)
    raw_gamma[~reliable] = 0.0
    return raw_gamma.astype(np.float32)
