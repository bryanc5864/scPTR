"""PT state UMAP and comparison plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

from .._constants import PT_STATE
from .._utils import require_obs
from ._utils import setup_axes, save_or_show


def pt_umap(
    adata: AnnData,
    color: str = PT_STATE,
    basis: str = "X_gamma_umap",
    figsize: tuple[float, float] = (6, 5),
    cmap: str = "tab20",
    save: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """UMAP of cells in gamma space, colored by PT state.

    Parameters
    ----------
    adata
        Annotated data matrix with gamma UMAP embedding.
    color
        Obs column for coloring.
    basis
        Key in ``adata.obsm`` for the embedding coordinates.
    figsize
        Figure size.
    cmap
        Colormap for categorical labels.
    save
        Path to save.
    show
        Whether to display.
    ax
        Pre-existing axes.
    """
    if basis not in adata.obsm:
        raise KeyError(
            f"Embedding {basis!r} not found. Run scptr.tl.pt_states() first."
        )

    coords = adata.obsm[basis]
    fig, ax = setup_axes(ax, figsize=figsize)

    if color in adata.obs.columns:
        labels = adata.obs[color].values
        unique_labels = sorted(set(labels))
        colormap = plt.colormaps.get_cmap(cmap).resampled(len(unique_labels))
        label_to_int = {l: i for i, l in enumerate(unique_labels)}
        c = [label_to_int[l] for l in labels]

        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=c, cmap=colormap, s=5, alpha=0.7, rasterized=True,
        )
        # Legend
        for i, label in enumerate(unique_labels):
            ax.scatter([], [], c=[colormap(i)], label=label, s=20)
        ax.legend(
            title=color, bbox_to_anchor=(1.05, 1), loc="upper left",
            markerscale=2, frameon=False,
        )
    else:
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=5, alpha=0.7, rasterized=True,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Gamma UMAP — {color}")
    fig.tight_layout()

    save_or_show(fig, save, show)
    return fig if not show else None


def pt_comparison(
    adata: AnnData,
    figsize: tuple[float, float] = (12, 5),
    save: str | None = None,
    show: bool = True,
) -> plt.Figure | None:
    """Side-by-side UMAP comparing expression and gamma space.

    Parameters
    ----------
    adata
        Annotated data matrix with both ``X_umap`` and ``X_gamma_umap``.
    figsize
        Figure size.
    save
        Path to save.
    show
        Whether to display.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Expression UMAP
    if "X_umap" in adata.obsm:
        coords_expr = adata.obsm["X_umap"]
        labels = adata.obs.get(PT_STATE)
        if labels is not None:
            unique_labels = sorted(set(labels))
            cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_labels))
            label_to_int = {l: i for i, l in enumerate(unique_labels)}
            c = [label_to_int[l] for l in labels]
            ax1.scatter(
                coords_expr[:, 0], coords_expr[:, 1],
                c=c, cmap=cmap, s=5, alpha=0.7, rasterized=True,
            )
        else:
            ax1.scatter(
                coords_expr[:, 0], coords_expr[:, 1],
                s=5, alpha=0.7, rasterized=True,
            )
        ax1.set_title("Expression UMAP")
        ax1.set_xlabel("UMAP 1")
        ax1.set_ylabel("UMAP 2")
    else:
        ax1.text(0.5, 0.5, "X_umap not found", ha="center", va="center",
                 transform=ax1.transAxes)

    # Gamma UMAP
    if "X_gamma_umap" in adata.obsm:
        coords_gamma = adata.obsm["X_gamma_umap"]
        labels = adata.obs.get(PT_STATE)
        if labels is not None:
            unique_labels = sorted(set(labels))
            cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_labels))
            label_to_int = {l: i for i, l in enumerate(unique_labels)}
            c = [label_to_int[l] for l in labels]
            ax2.scatter(
                coords_gamma[:, 0], coords_gamma[:, 1],
                c=c, cmap=cmap, s=5, alpha=0.7, rasterized=True,
            )
        else:
            ax2.scatter(
                coords_gamma[:, 0], coords_gamma[:, 1],
                s=5, alpha=0.7, rasterized=True,
            )
        ax2.set_title("Gamma UMAP")
        ax2.set_xlabel("UMAP 1")
        ax2.set_ylabel("UMAP 2")
    else:
        ax2.text(0.5, 0.5, "X_gamma_umap not found", ha="center",
                 va="center", transform=ax2.transAxes)

    fig.tight_layout()
    save_or_show(fig, save, show)
    return fig if not show else None
