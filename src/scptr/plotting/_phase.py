"""Phase portrait plotting (unspliced vs spliced colored by gamma)."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

from .._constants import SMOOTHED_UNSPLICED, SMOOTHED_SPLICED, GAMMA
from .._utils import get_layer, require_layers
from ._utils import setup_axes, save_or_show


def phase_portrait(
    adata: AnnData,
    genes: str | Sequence[str],
    color_by: str = GAMMA,
    ncols: int = 3,
    figsize_per: tuple[float, float] = (4, 3.5),
    cmap: str = "viridis",
    save: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot phase portrait (unspliced vs spliced) colored by gamma.

    Parameters
    ----------
    adata
        Annotated data matrix.
    genes
        Gene name(s) to plot.
    color_by
        Layer to use for coloring. Default ``'gamma'``.
    ncols
        Number of columns in multi-gene grid.
    figsize_per
        Size per subplot panel.
    cmap
        Colormap name.
    save
        Path to save figure.
    show
        Whether to display the figure.
    ax
        Pre-existing axes (only for single gene).
    """
    require_layers(adata, SMOOTHED_UNSPLICED, SMOOTHED_SPLICED)

    if isinstance(genes, str):
        genes = [genes]

    u = get_layer(adata, SMOOTHED_UNSPLICED)
    s = get_layer(adata, SMOOTHED_SPLICED)

    if color_by in adata.layers:
        colors = get_layer(adata, color_by)
    else:
        colors = None

    n_genes = len(genes)

    if n_genes == 1 and ax is not None:
        fig, ax = setup_axes(ax)
        axes_list = [ax]
    else:
        nrows = (n_genes + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
            squeeze=False,
        )
        axes_list = axes.ravel().tolist()

    gene_names = adata.var_names.tolist()

    for i, gene in enumerate(genes):
        if gene not in gene_names:
            continue
        gi = gene_names.index(gene)
        ax_i = axes_list[i]

        c = colors[:, gi] if colors is not None else None
        sc = ax_i.scatter(
            s[:, gi], u[:, gi],
            c=c, cmap=cmap, s=3, alpha=0.6, rasterized=True,
        )
        ax_i.set_xlabel("Spliced (Ms)")
        ax_i.set_ylabel("Unspliced (Mu)")
        ax_i.set_title(gene)
        if c is not None:
            plt.colorbar(sc, ax=ax_i, label=color_by)

    # Hide unused axes
    for j in range(n_genes, len(axes_list)):
        axes_list[j].set_visible(False)

    fig.tight_layout()
    save_or_show(fig, save, show)
    return fig if not show else None
