"""Network graph visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData

from ._utils import setup_axes, save_or_show


def network_graph(
    adata: AnnData,
    n_edges: int = 50,
    figsize: tuple[float, float] = (8, 8),
    save: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot the inferred regulatory network as a graph.

    Requires ``adata.uns['pt_network']`` (from ``scptr.tl.infer_network``).

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_edges
        Number of top edges to display.
    figsize
        Figure size.
    save
        Path to save.
    show
        Whether to display.
    ax
        Pre-existing axes.
    """
    if "pt_network" not in adata.uns:
        raise KeyError("Run scptr.tl.infer_network() first.")

    edges_df = adata.uns["pt_network"]
    if len(edges_df) == 0:
        fig, ax = setup_axes(ax, figsize=figsize)
        ax.text(0.5, 0.5, "No edges found", ha="center", va="center",
                transform=ax.transAxes)
        save_or_show(fig, save, show)
        return fig if not show else None

    edges_df = edges_df.head(n_edges)

    # Collect unique nodes
    nodes = list(set(edges_df["regulator"].tolist() + edges_df["target"].tolist()))
    node_idx = {n: i for i, n in enumerate(nodes)}

    # Simple circular layout
    n_nodes = len(nodes)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    fig, ax = setup_axes(ax, figsize=figsize)

    # Draw edges
    max_weight = edges_df["weight"].abs().max()
    for _, row in edges_df.iterrows():
        i = node_idx[row["regulator"]]
        j = node_idx[row["target"]]
        w = abs(row["weight"]) / (max_weight + 1e-10)
        color = "red" if row["weight"] > 0 else "blue"
        ax.annotate(
            "",
            xy=(x[j], y[j]),
            xytext=(x[i], y[i]),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                alpha=0.3 + 0.7 * w,
                linewidth=0.5 + 2.0 * w,
            ),
        )

    # Draw nodes
    ax.scatter(x, y, s=100, c="lightgray", edgecolors="black", zorder=5)
    for node, idx in node_idx.items():
        ax.annotate(
            node,
            (x[idx], y[idx]),
            fontsize=7,
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("PT Regulatory Network")
    fig.tight_layout()

    save_or_show(fig, save, show)
    return fig if not show else None
