"""Variance decomposition into transcriptional and post-transcriptional fractions."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from .._constants import (
    SMOOTHED_UNSPLICED,
    GAMMA,
    TF_SCORE,
    PTF_SCORE,
)
from .._utils import get_layer, require_layers, log_params


def variance_decomposition(adata: AnnData) -> None:
    """Decompose gene expression variance into TF and PTF components.

    In log space:
    - ``TF_g = Var(log(u_g)) / (Var(log(u_g)) + Var(log(gamma_g)))``
    - ``PTF_g = 1 - TF_g``

    Results are stored in ``adata.var['tf_score']`` and ``adata.var['ptf_score']``.

    Parameters
    ----------
    adata
        Annotated data matrix with smoothed unspliced layer and gamma.
    """
    require_layers(adata, SMOOTHED_UNSPLICED, GAMMA)

    u = get_layer(adata, SMOOTHED_UNSPLICED)
    gamma = get_layer(adata, GAMMA)

    # Log-transform with pseudocount
    log_u = np.log1p(u)
    log_gamma = np.log1p(gamma)

    var_u = np.var(log_u, axis=0)
    var_gamma = np.var(log_gamma, axis=0)

    total_var = var_u + var_gamma
    # Avoid division by zero
    total_var = np.clip(total_var, 1e-10, None)

    tf_score = var_u / total_var
    ptf_score = var_gamma / total_var

    adata.var[TF_SCORE] = tf_score.astype(np.float32)
    adata.var[PTF_SCORE] = ptf_score.astype(np.float32)

    log_params(adata, "variance_decomposition", {})
