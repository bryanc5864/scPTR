#!/usr/bin/env python
"""Generate PT velocity streamline comparison figures for pancreas and dentate gyrus.

For each dataset:
  - Left panel: PT velocity streamlines (cell types colored underneath)
  - Right panel: RNA velocity (scVelo) quiver from existing gap_analysis output
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

sys.path.insert(0, str(Path(__file__).parent))
from _common import set_figure_style

import scptr

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "velocity_comparison"
GAP_DIR = Path(__file__).parent.parent / "output" / "gap_analysis"


def save_fig(fig, name, subdir="figures"):
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def run_pipeline(adata, name):
    """Run standard scPTR pipeline."""
    print(f"\n--- Pipeline: {name} ---")
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)
    scptr.tl.variance_decomposition(adata)
    scptr.tl.pt_states(adata)
    scptr.tl.pt_velocity(adata)
    print(f"  Done: {adata.shape}")
    return adata


def generate_streamline_figure(adata, name):
    """Generate 2-panel figure: PT velocity streamlines + RNA velocity quiver."""
    print(f"\n  Generating streamline figure for {name}...")

    cluster_col = "clusters"
    basis = "X_gamma_umap"

    if basis not in adata.obsm:
        print(f"  No {basis}, computing UMAP on gamma PCA...")
        from sklearn.decomposition import PCA
        gamma = adata.layers["gamma"]
        nonzero_frac = (gamma > 0).mean(axis=0)
        good = nonzero_frac >= 0.1
        n_pcs = min(30, gamma.shape[0] - 1, good.sum() - 1)
        pca = PCA(n_components=n_pcs, random_state=42)
        gamma_pcs = pca.fit_transform(gamma[:, good])
        adata.obsm["X_gamma_pca"] = gamma_pcs
        sc.pp.neighbors(adata, use_rep="X_gamma_pca", key_added="gamma")
        sc.tl.umap(adata, neighbors_key="gamma")
        adata.obsm[basis] = adata.obsm["X_umap"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left panel: PT velocity streamlines with cell types
    coords = adata.obsm[basis]
    clusters = adata.obs[cluster_col]

    for ci, cat in enumerate(clusters.unique()):
        mask = (clusters == cat).values
        axes[0].scatter(coords[mask, 0], coords[mask, 1],
                        s=3, alpha=0.2, label=cat,
                        c=[plt.cm.tab20(ci / 20)],
                        rasterized=True)

    # Project velocity to 2D and build streamlines
    from scptr.plotting._velocity import _project_velocity_to_2d
    from scipy.ndimage import gaussian_filter
    from scipy.stats import binned_statistic_2d

    v_emb = _project_velocity_to_2d(adata, basis)
    grid_size = 50

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05
    x_edges = np.linspace(x_min - pad_x, x_max + pad_x, grid_size + 1)
    y_edges = np.linspace(y_min - pad_y, y_max + pad_y, grid_size + 1)

    U, _, _, _ = binned_statistic_2d(
        coords[:, 0], coords[:, 1], v_emb[:, 0],
        statistic="mean", bins=[x_edges, y_edges])
    V, _, _, _ = binned_statistic_2d(
        coords[:, 0], coords[:, 1], v_emb[:, 1],
        statistic="mean", bins=[x_edges, y_edges])

    U = gaussian_filter(np.nan_to_num(U, nan=0.0), sigma=1.5)
    V = gaussian_filter(np.nan_to_num(V, nan=0.0), sigma=1.5)

    gx = 0.5 * (x_edges[:-1] + x_edges[1:])
    gy = 0.5 * (y_edges[:-1] + y_edges[1:])
    speed = np.sqrt(U**2 + V**2)

    axes[0].streamplot(gx, gy, U.T, V.T,
                       color=speed.T, cmap="coolwarm",
                       density=1.0, linewidth=0.8, arrowsize=1.2)
    axes[0].set_title(f"PT Velocity Streamlines: {name}")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].legend(fontsize=5, markerscale=3, loc="best", ncol=2)

    # Right panel: PT velocity quiver (discrete arrows for comparison)
    for ci, cat in enumerate(clusters.unique()):
        mask = (clusters == cat).values
        axes[1].scatter(coords[mask, 0], coords[mask, 1],
                        s=3, alpha=0.2,
                        c=[plt.cm.tab20(ci / 20)],
                        rasterized=True)

    n_show = min(500, adata.n_obs)
    idx = np.random.choice(adata.n_obs, n_show, replace=False)
    norms = np.linalg.norm(v_emb, axis=1)
    cap = np.percentile(norms[norms > 0], 95) if (norms > 0).any() else 1.0
    v_scaled = v_emb / max(cap, 1e-10)
    arrow_mask = norms[idx] > 0.01 * cap

    axes[1].quiver(coords[idx[arrow_mask], 0], coords[idx[arrow_mask], 1],
                   v_scaled[idx[arrow_mask], 0], v_scaled[idx[arrow_mask], 1],
                   color="black", alpha=0.5, scale=20, width=0.003,
                   headwidth=4, headlength=5)
    axes[1].set_title(f"PT Velocity Quiver: {name}")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")

    fig.suptitle(f"PT Velocity Visualization: {name}", fontsize=14)
    fig.tight_layout()
    save_fig(fig, f"streamlines_{name}")


def main():
    set_figure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pancreas
    print("=" * 60)
    print("PANCREAS")
    print("=" * 60)
    adata_pan = scptr.datasets.pancreas()
    adata_pan = run_pipeline(adata_pan, "pancreas")
    generate_streamline_figure(adata_pan, "pancreas")

    # Dentate Gyrus
    print("\n" + "=" * 60)
    print("DENTATE GYRUS")
    print("=" * 60)
    adata_dg = scptr.datasets.dentate_gyrus()
    adata_dg = run_pipeline(adata_dg, "dentate_gyrus")
    generate_streamline_figure(adata_dg, "dentate_gyrus")

    print(f"\n{'='*60}")
    print("VELOCITY COMPARISON COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
