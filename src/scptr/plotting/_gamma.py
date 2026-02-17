"""Gamma visualization: heatmap and violin plots."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from anndata import AnnData

from .._constants import GAMMA, PT_STATE
from .._utils import get_layer, require_layers
from ._utils import setup_axes, save_or_show


def gamma_heatmap(
    adata: AnnData,
    groupby: str = PT_STATE,
    n_genes: int = 50,
    figsize: tuple[float, float] = (12, 8),
    cmap: str = "viridis",
    save: str | None = None,
    show: bool = True,
) -> plt.Figure | None:
    """Plot heatmap of mean gamma per group.

    Parameters
    ----------
    adata
        Annotated data matrix with ``gamma`` layer and group labels.
    groupby
        Obs column for grouping cells.
    n_genes
        Number of top variable genes to display.
    figsize
        Figure size.
    cmap
        Colormap.
    save
        Path to save figure.
    show
        Whether to display.
    """
    require_layers(adata, GAMMA)

    gamma = get_layer(adata, GAMMA)
    groups = adata.obs[groupby].values

    unique_groups = sorted(set(groups))
    mean_gamma = np.zeros((len(unique_groups), adata.n_vars))

    for i, g in enumerate(unique_groups):
        mask = groups == g
        mean_gamma[i] = gamma[mask].mean(axis=0)

    # Select top variable genes (use log-space variance for robustness)
    log_mean = np.log1p(mean_gamma)
    gene_var = np.var(log_mean, axis=0)
    top_idx = np.argsort(gene_var)[::-1][:n_genes]

    # Log-transform for visualization
    plot_data = np.log1p(mean_gamma[:, top_idx])

    # Z-score per gene for clearer cross-gene comparison
    gene_means = plot_data.mean(axis=0, keepdims=True)
    gene_stds = plot_data.std(axis=0, keepdims=True)
    gene_stds = np.clip(gene_stds, 1e-10, None)
    plot_data_z = (plot_data - gene_means) / gene_stds

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        plot_data_z,
        aspect="auto", cmap=cmap, interpolation="nearest",
        vmin=-2, vmax=2,
    )
    ax.set_yticks(range(len(unique_groups)))
    ax.set_yticklabels(unique_groups)
    ax.set_xlabel("Genes (top variable)")
    ax.set_ylabel(groupby)
    ax.set_title(f"Mean gamma per group (z-scored log scale, top {n_genes} genes)")
    plt.colorbar(im, ax=ax, label="z-score of log(1+gamma)")
    fig.tight_layout()

    save_or_show(fig, save, show)
    return fig if not show else None


def gamma_violin(
    adata: AnnData,
    genes: str | Sequence[str],
    groupby: str = PT_STATE,
    figsize_per: tuple[float, float] = (6, 4),
    save: str | None = None,
    show: bool = True,
) -> plt.Figure | None:
    """Violin plot of gamma values per group for selected genes.

    Parameters
    ----------
    adata
        Annotated data matrix.
    genes
        Gene name(s) to plot.
    groupby
        Obs column for grouping.
    figsize_per
        Size per subplot.
    save
        Path to save.
    show
        Whether to display.
    """
    require_layers(adata, GAMMA)

    if isinstance(genes, str):
        genes = [genes]

    gamma = get_layer(adata, GAMMA)
    gene_names = adata.var_names.tolist()
    groups = adata.obs[groupby].values

    n_genes = len(genes)
    fig, axes = plt.subplots(
        1, n_genes,
        figsize=(figsize_per[0] * n_genes, figsize_per[1]),
        squeeze=False,
    )

    for i, gene in enumerate(genes):
        if gene not in gene_names:
            continue
        gi = gene_names.index(gene)
        df = pd.DataFrame({"gamma": gamma[:, gi], groupby: groups})
        sns.violinplot(data=df, x=groupby, y="gamma", ax=axes[0, i])
        axes[0, i].set_title(gene)

    fig.tight_layout()
    save_or_show(fig, save, show)
    return fig if not show else None
