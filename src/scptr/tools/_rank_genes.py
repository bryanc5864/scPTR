"""Rank genes by differential gamma across PT states."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from .._constants import GAMMA, PT_STATE
from .._utils import get_layer, require_layers, require_obs, log_params


def rank_pt_genes(
    adata: AnnData,
    groupby: str = PT_STATE,
    method: str = "t-test",
    n_genes: int = 100,
) -> pd.DataFrame:
    """Rank genes by differential degradation rate across groups.

    Uses scanpy's ``rank_genes_groups`` on the gamma matrix.

    Parameters
    ----------
    adata
        Annotated data matrix with ``gamma`` layer and group labels.
    groupby
        Column in ``adata.obs`` defining groups.
    method
        Statistical test: ``'t-test'``, ``'wilcoxon'``, etc.
    n_genes
        Number of top genes to return per group.

    Returns
    -------
    DataFrame with ranked genes per group.
    """
    import scanpy as sc

    require_layers(adata, GAMMA)
    require_obs(adata, groupby)

    gamma = get_layer(adata, GAMMA)

    # Temporary AnnData for ranking
    gamma_adata = AnnData(X=gamma.copy())
    gamma_adata.obs_names = adata.obs_names.copy()
    gamma_adata.var_names = adata.var_names.copy()
    gamma_adata.obs[groupby] = adata.obs[groupby].values

    sc.tl.rank_genes_groups(
        gamma_adata,
        groupby=groupby,
        method=method,
        n_genes=n_genes,
    )

    # Store results back
    adata.uns["rank_pt_genes"] = gamma_adata.uns["rank_genes_groups"]

    # Build a summary DataFrame
    result = sc.get.rank_genes_groups_df(gamma_adata, group=None)

    log_params(adata, "rank_pt_genes", {
        "groupby": groupby,
        "method": method,
        "n_genes": n_genes,
    })

    return result
