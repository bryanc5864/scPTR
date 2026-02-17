"""Correlation of estimated gamma with published mRNA half-lives."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import stats

from .._constants import GAMMA
from .._utils import get_layer, require_layers


def correlate_with_halflives(
    adata: AnnData,
    halflives_df: pd.DataFrame,
    gene_col: str = "gene_symbol",
    halflife_col: str = "half_life_hours",
    min_gamma_fraction: float = 0.1,
    case_insensitive: bool = True,
) -> dict:
    """Correlate per-gene median gamma with published mRNA half-lives.

    Expects a negative correlation: high gamma (fast degradation) should
    correspond to short half-lives.

    Parameters
    ----------
    adata
        Annotated data matrix with ``gamma`` layer.
    halflives_df
        DataFrame with gene symbols and half-life measurements.
    gene_col
        Column name for gene symbols in ``halflives_df``.
    halflife_col
        Column name for half-life values in ``halflives_df``.
    min_gamma_fraction
        Minimum fraction of cells with gamma > 0 for a gene to be
        included in the correlation (default 0.1). Genes with too
        few unspliced reads produce unreliable gamma estimates.
    case_insensitive
        Match gene symbols case-insensitively (default True). Useful
        for cross-species comparisons (mouse Titlecase vs human UPPER).

    Returns
    -------
    dict with keys: ``spearman_r``, ``spearman_p``, ``pearson_r``,
    ``pearson_p``, ``n_genes``, ``n_genes_unfiltered``, ``matched_genes``.
    """
    require_layers(adata, GAMMA)

    gamma = get_layer(adata, GAMMA)

    # Filter genes: require minimum fraction of cells with non-zero gamma
    nonzero_frac = (gamma > 0).mean(axis=0)
    gene_mask = nonzero_frac >= min_gamma_fraction

    median_gamma = np.median(gamma, axis=0)

    gene_names = adata.var_names.tolist()
    gamma_series = pd.Series(median_gamma, index=gene_names)
    mask_series = pd.Series(gene_mask, index=gene_names)

    hl_series = halflives_df.set_index(gene_col)[halflife_col]

    if case_insensitive:
        # Build uppercase-to-original mapping, match via uppercase
        gamma_upper = {g.upper(): g for g in gene_names}
        hl_upper = {}
        for g in hl_series.index:
            if isinstance(g, str):
                hl_upper[g.upper()] = g
        shared_upper = set(gamma_upper.keys()) & set(hl_upper.keys())
        # Map back to original names
        shared_all = pd.Index([gamma_upper[u] for u in shared_upper])
        # Rebuild hl_series indexed by adata gene names
        hl_remap = {gamma_upper[u]: hl_series[hl_upper[u]] for u in shared_upper}
        hl_series = pd.Series(hl_remap)
    else:
        shared_all = gamma_series.index.intersection(hl_series.index)

    # Apply gene quality filter
    shared = shared_all[mask_series[shared_all].values]

    n_unfiltered = len(shared_all)

    if len(shared) < 3:
        return {
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "n_genes": len(shared),
            "n_genes_unfiltered": n_unfiltered,
            "matched_genes": shared.tolist(),
        }

    g = gamma_series[shared].values.astype(float)
    h = hl_series[shared].values.astype(float)

    # Remove NaN/Inf
    valid = np.isfinite(g) & np.isfinite(h) & (g > 0) & (h > 0)
    g, h = g[valid], h[valid]

    if len(g) < 3:
        return {
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "n_genes": 0,
            "n_genes_unfiltered": n_unfiltered,
            "matched_genes": [],
        }

    sp_r, sp_p = stats.spearmanr(g, h)
    pe_r, pe_p = stats.pearsonr(np.log1p(g), np.log1p(h))

    return {
        "spearman_r": float(sp_r),
        "spearman_p": float(sp_p),
        "pearson_r": float(pe_r),
        "pearson_p": float(pe_p),
        "n_genes": int(valid.sum()),
        "n_genes_unfiltered": n_unfiltered,
        "matched_genes": shared[valid].tolist(),
    }
