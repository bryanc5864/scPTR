#!/usr/bin/env python
"""Aim 2: Differential degradation rate analysis across cell types.

Identifies genes with significantly different gamma values between
PT states, revealing post-transcriptional regulation programs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

    print("Running pipeline...")
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)

    # Per-cell-type beta estimation (if cell type annotation exists)
    groupby = args.groupby
    if groupby and groupby in adata.obs.columns:
        print(f"Estimating beta per cell type ({groupby})...")
        scptr.tl.estimate_beta(adata, groupby=groupby)
    else:
        print("Estimating global beta...")
        scptr.tl.estimate_beta(adata)

    scptr.tl.estimate_gamma(adata)
    scptr.tl.pt_states(adata, resolution=args.resolution)

    # Rank genes by differential gamma across PT states
    print("Ranking genes by differential gamma...")
    rank_df = scptr.tl.rank_pt_genes(adata, n_genes=args.n_genes)
    rank_df.to_csv(res_dir / "differential_gamma_genes.csv", index=False)

    # Plot top genes
    top_genes = rank_df.head(args.n_plot)["names"].unique().tolist()
    for gene in top_genes[:min(5, len(top_genes))]:
        if gene in adata.var_names:
            fig = scptr.pl.gamma_violin(adata, gene)
            save_figure(fig, f"gamma_violin_{gene}", "figures/aim2")

    # Gamma heatmap of top variable genes
    fig = scptr.pl.gamma_heatmap(adata)
    save_figure(fig, "gamma_heatmap_top", "figures/aim2")

    print(f"Found {len(rank_df)} differentially degraded genes")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution", type=float, default=1.0,
        help="Leiden clustering resolution (default: 1.0)",
    )
    parser.add_argument(
        "--groupby", type=str, default=None,
        help="Obs column for per-cell-type beta estimation",
    )
    parser.add_argument(
        "--n-genes", type=int, default=100,
        help="Number of top genes to report (default: 100)",
    )
    parser.add_argument(
        "--n-plot", type=int, default=10,
        help="Number of top genes to plot (default: 10)",
    )
    main(parser.parse_args())
