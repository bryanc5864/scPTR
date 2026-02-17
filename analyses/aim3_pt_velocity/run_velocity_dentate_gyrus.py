#!/usr/bin/env python
"""Aim 3: Post-transcriptional velocity on dentate gyrus trajectory.

Computes PT velocity on the dentate gyrus neurogenesis dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))
from _common import set_figure_style, save_figure, setup_output_dirs

import scptr


def main(args: argparse.Namespace) -> None:
    set_figure_style()
    fig_dir, res_dir = setup_output_dirs(
        "figures/aim3", "results/aim3"
    )

    # Load and process
    print("Loading dentate gyrus dataset...")
    adata = scptr.datasets.dentate_gyrus()

    print("Running preprocessing...")
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)

    print("Running analysis...")
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)
    scptr.tl.pt_states(adata)

    # PT velocity
    print("Computing PT velocity...")
    scptr.tl.pt_velocity(adata, use_graph=args.graph)

    # Velocity embedding plot
    fig = scptr.pl.pt_velocity_embedding(adata)
    save_figure(fig, "pt_velocity_dentate_gyrus", "figures/aim3")

    # PT UMAP colored by state
    fig = scptr.pl.pt_umap(adata)
    save_figure(fig, "pt_states_dentate_gyrus", "figures/aim3")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", default="gamma",
        choices=["gamma", "expression"],
        help="Neighbor graph to use for velocity (default: gamma)",
    )
    main(parser.parse_args())
