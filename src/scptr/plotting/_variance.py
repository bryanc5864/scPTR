"""TF/PTF variance decomposition plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

from .._constants import TF_SCORE, PTF_SCORE
from .._utils import require_var
from ._utils import setup_axes, save_or_show


def tf_ptf_scatter(
    adata: AnnData,
    label_top: int = 10,
    figsize: tuple[float, float] = (10, 5),
    save: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Two-panel variance decomposition plot.

    Left: histogram of TF scores showing distribution of transcriptional
    vs post-transcriptional regulation. Right: ranked TF scores with
    top genes labeled.

    Parameters
    ----------
    adata
        Annotated data matrix with ``var['tf_score']`` and ``var['ptf_score']``.
    label_top
        Number of extreme genes to label in the ranked plot.
    figsize
        Figure size.
    save
        Path to save.
    show
        Whether to display.
    ax
        Pre-existing axes (used for left panel only if provided).
    """
    require_var(adata, TF_SCORE, PTF_SCORE)

    tf = adata.var[TF_SCORE].values
    ptf = adata.var[PTF_SCORE].values
    genes = adata.var_names.tolist()

    if ax is not None:
        # Single-axis mode: just histogram
        fig = ax.figure
        ax.hist(tf, bins=50, color="steelblue", alpha=0.8, edgecolor="white")
        ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="TF=PTF")
        ax.set_xlabel("Transcriptional Fraction (TF)")
        ax.set_ylabel("Number of genes")
        ax.set_title("Variance Decomposition")
        ax.legend()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left: TF score histogram
        ax1.hist(tf, bins=50, color="steelblue", alpha=0.8, edgecolor="white")
        ax1.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="TF=PTF")
        median_tf = np.median(tf)
        ax1.axvline(median_tf, color="orange", linestyle="-", alpha=0.7,
                     label=f"Median={median_tf:.3f}")
        n_tf_dom = (tf > 0.5).sum()
        n_ptf_dom = (tf <= 0.5).sum()
        ax1.set_xlabel("Transcriptional Fraction (TF)")
        ax1.set_ylabel("Number of genes")
        ax1.set_title(f"TF > 0.5: {n_tf_dom} genes | PTF > 0.5: {n_ptf_dom} genes")
        ax1.legend(fontsize=8)

        # Right: ranked TF score with labels
        sorted_idx = np.argsort(tf)
        ranks = np.arange(len(tf))
        ax2.scatter(ranks, tf[sorted_idx], s=3, alpha=0.5, c="steelblue",
                    rasterized=True)
        ax2.axhline(0.5, color="red", linestyle="--", alpha=0.3)
        ax2.set_xlabel("Gene rank")
        ax2.set_ylabel("TF score")
        ax2.set_title("Genes ranked by TF score")

        # Label top TF genes (right side)
        if label_top > 0:
            top_tf_idx = sorted_idx[-label_top:]
            for idx in top_tf_idx:
                rank = np.searchsorted(sorted_idx, idx)
                ax2.annotate(
                    genes[idx],
                    (rank, tf[idx]),
                    fontsize=6, alpha=0.8,
                    xytext=(5, 0), textcoords="offset points",
                )
            # Label top PTF genes (left side)
            top_ptf_idx = sorted_idx[:label_top]
            for idx in top_ptf_idx:
                rank = np.searchsorted(sorted_idx, idx)
                ax2.annotate(
                    genes[idx],
                    (rank, tf[idx]),
                    fontsize=6, alpha=0.8,
                    xytext=(5, 0), textcoords="offset points",
                )

        fig.suptitle("Variance Decomposition: Transcriptional vs Post-Transcriptional",
                     fontsize=12, y=1.02)
        fig.tight_layout()

    save_or_show(fig, save, show)
    return fig if not show else None
