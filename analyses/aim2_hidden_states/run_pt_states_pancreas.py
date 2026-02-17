#!/usr/bin/env python
"""Aim 2: Post-transcriptional state discovery on pancreas data.

Discovers PT states via gamma-space clustering and compares
to expression-based cell type annotations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

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
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)
    scptr.tl.variance_decomposition(adata)

    # PT state discovery
    print("Discovering PT states...")
    scptr.tl.pt_states(adata, resolution=args.resolution)

    n_states = adata.obs["pt_state"].nunique()
    print(f"Found {n_states} PT states")

    # Plot PT UMAP
    fig = scptr.pl.pt_umap(adata)
    save_figure(fig, "pt_states_pancreas", "figures/aim2")

    # State composition summary
    state_counts = adata.obs["pt_state"].value_counts()
    state_counts.to_csv(res_dir / "pt_state_counts.csv")

    # Rank genes by differential gamma
    print("Ranking genes by differential gamma...")
    rank_df = scptr.tl.rank_pt_genes(adata)
    rank_df.to_csv(res_dir / "ranked_pt_genes.csv", index=False)

    # TF vs PTF scatter
    fig = scptr.pl.tf_ptf_scatter(adata)
    save_figure(fig, "tf_ptf_scatter", "figures/aim2")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution", type=float, default=1.0,
        help="Leiden clustering resolution (default: 1.0)",
    )
    main(parser.parse_args())
