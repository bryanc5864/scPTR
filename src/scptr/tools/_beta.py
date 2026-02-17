"""Beta (splicing rate) estimation from phase portrait."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from .._constants import (
    SMOOTHED_UNSPLICED,
    SMOOTHED_SPLICED,
    BETA,
    DEFAULT_BETA_QUANTILE,
)
from .._utils import get_layer, require_layers, require_obs, log_params


def estimate_beta(
    adata: AnnData,
    method: str = "quantile",
    quantile: float = DEFAULT_BETA_QUANTILE,
    groupby: str | None = None,
) -> None:
    """Estimate per-gene splicing rate beta from the u/s phase portrait.

    Default fast mode uses the upper quantile of the u/s ratio per gene
    as the slope estimate.

    Parameters
    ----------
    adata
        Annotated data matrix with smoothed layers ``Mu`` and ``Ms``.
    method
        ``'quantile'`` (default) — fast quantile-based estimation.
    quantile
        Quantile of u/s ratio to use (default 0.95).
    groupby
        Optional obs column for per-group estimation. When set, estimates
        beta separately for each group and stores per-group values in
        ``adata.varm['beta_groups']``. The consensus beta (median across
        groups) is stored in ``adata.var['beta']``.
    """
    require_layers(adata, SMOOTHED_UNSPLICED, SMOOTHED_SPLICED)

    u = get_layer(adata, SMOOTHED_UNSPLICED)
    s = get_layer(adata, SMOOTHED_SPLICED)

    if method != "quantile":
        raise ValueError(f"Unknown method: {method!r}. Use 'quantile'.")

    if groupby is not None:
        require_obs(adata, groupby)
        groups = adata.obs[groupby].astype(str)
        unique_groups = sorted(groups.unique())
        beta_groups = pd.DataFrame(
            index=adata.var_names, columns=unique_groups, dtype=np.float32,
        )
        for grp in unique_groups:
            mask = (groups == grp).values
            beta_g = _quantile_beta(u[mask], s[mask], quantile)
            beta_groups[grp] = beta_g.astype(np.float32)

        adata.varm["beta_groups"] = beta_groups
        # Consensus: median across groups
        beta = np.nanmedian(beta_groups.values, axis=1)
    else:
        beta = _quantile_beta(u, s, quantile)

    # Global clip: cap extreme beta values at the 99th percentile of
    # positive betas.  Genes with very sparse unspliced counts can produce
    # wildly large u/s quantiles that propagate into gamma.
    positive_beta = beta[beta > 0]
    if len(positive_beta) > 0:
        beta_cap = np.percentile(positive_beta, 99)
        beta = np.clip(beta, 0, beta_cap)

    adata.var[BETA] = beta.astype(np.float32)

    log_params(adata, "estimate_beta", {
        "method": method,
        "quantile": quantile,
        "groupby": groupby,
    })


def _quantile_beta(
    u: np.ndarray, s: np.ndarray, quantile: float
) -> np.ndarray:
    """Estimate beta as the upper quantile of u/s ratio per gene.

    Parameters
    ----------
    u : (n_obs, n_vars)
    s : (n_obs, n_vars)
    quantile : float

    Returns
    -------
    beta : (n_vars,)
    """
    # Avoid division by zero
    s_safe = np.clip(s, 1e-6, None)
    ratio = u / s_safe

    # Use only cells with sufficient expression
    mask = (u > 0) & (s > 1e-6)
    n_vars = u.shape[1]
    beta = np.zeros(n_vars, dtype=np.float64)

    for g in range(n_vars):
        valid = ratio[mask[:, g], g]
        if len(valid) > 0:
            beta[g] = np.quantile(valid, quantile)
        else:
            beta[g] = 0.0

    return beta
