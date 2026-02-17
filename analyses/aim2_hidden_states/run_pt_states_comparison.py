#!/usr/bin/env python
"""Aim 2: Compare expression-based vs gamma-based clustering.

Evaluates whether gamma-space clustering reveals distinct states
not captured by expression-space analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

sys.path.insert(0, str(Path(__file__).parent.parent))
from _common import set_figure_style, save_figure, setup_output_dirs

import scptr


def main(args: argparse.Namespace) -> None:
    set_figure_style()
    fig_dir, res_dir = setup_output_dirs(
        "figures/aim2", "results/aim2"
    )

    # Load and process
    print("Loading pancreas dataset...")
    adata = scptr.datasets.pancreas()

    # Run scPTR pipeline
    print("Running scPTR pipeline...")
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)

    # Expression-space clustering
    print("Running expression-space clustering...")
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=30, use_rep="X_pca", key_added="expression")
    sc.tl.leiden(adata, resolution=args.resolution, neighbors_key="expression",
                 key_added="expr_cluster")
    sc.tl.umap(adata, neighbors_key="expression")

    # Gamma-space clustering
    print("Running gamma-space clustering...")
    scptr.tl.pt_states(adata, resolution=args.resolution)

    # Comparison plot
    fig = scptr.pl.pt_comparison(adata)
    save_figure(fig, "expression_vs_gamma_clustering", "figures/aim2")

    # Compute overlap statistics (Adjusted Rand Index)
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(
        adata.obs["expr_cluster"].astype(str),
        adata.obs["pt_state"].astype(str),
    )
    print(f"Adjusted Rand Index (expression vs gamma clustering): {ari:.3f}")

    # Save ARI
    with open(res_dir / "clustering_comparison.txt", "w") as f:
        f.write(f"ARI: {ari:.4f}\n")
        f.write(f"n_expression_clusters: {adata.obs['expr_cluster'].nunique()}\n")
        f.write(f"n_pt_states: {adata.obs['pt_state'].nunique()}\n")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution", type=float, default=1.0,
        help="Leiden clustering resolution (default: 1.0)",
    )
    main(parser.parse_args())
