"""Benchmark visualization plots."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData

from .._constants import GAMMA
from .._utils import get_layer


def halflife_scatter(
    adata: AnnData,
    halflives_df: pd.DataFrame,
    gene_col: str = "gene_symbol",
    halflife_col: str = "half_life_hours",
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Figure:
    """Scatter plot of per-gene median gamma vs published half-lives.

    Parameters
    ----------
    adata
        Annotated data with ``gamma`` layer.
    halflives_df
        DataFrame with gene symbols and half-life measurements.
    gene_col
        Column name for gene symbols.
    halflife_col
        Column name for half-life values.
    ax
        Optional matplotlib Axes.

    Returns
    -------
    matplotlib Figure.
    """
    gamma = get_layer(adata, GAMMA)
    median_gamma = pd.Series(np.median(gamma, axis=0), index=adata.var_names)
    hl = halflives_df.set_index(gene_col)[halflife_col]
    shared = median_gamma.index.intersection(hl.index)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    g = median_gamma[shared].values
    h = hl[shared].values

    ax.scatter(h, g, alpha=0.6, s=20, **kwargs)
    ax.set_xlabel("Half-life (hours)")
    ax.set_ylabel("Median gamma")
    ax.set_title(f"Gamma vs Half-life (n={len(shared)} genes)")

    return fig


def enrichment_barplot(
    results: list[dict],
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Figure:
    """Bar plot comparing gamma distributions for enrichment results.

    Parameters
    ----------
    results
        List of dicts from ``are_enrichment`` / ``nmd_enrichment``.
    ax
        Optional matplotlib Axes.

    Returns
    -------
    matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    labels = [r["label"] for r in results]
    in_set = [r.get("median_gamma_in_set", 0) for r in results]
    background = [r.get("median_gamma_background", 0) for r in results]
    p_values = [r.get("p_value", 1.0) for r in results]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, in_set, width, label="Gene set", **kwargs)
    ax.bar(x + width / 2, background, width, label="Background", alpha=0.7)

    # Annotate p-values
    for i, p in enumerate(p_values):
        if np.isfinite(p):
            ax.text(x[i], max(in_set[i], background[i]) * 1.05,
                    f"p={p:.2e}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median gamma")
    ax.set_title("Enrichment Analysis")
    ax.legend()

    return fig
