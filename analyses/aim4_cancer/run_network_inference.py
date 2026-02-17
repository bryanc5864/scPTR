#!/usr/bin/env python
"""Aim 4: RBP-target regulatory network inference on cancer data.

Infers post-transcriptional regulatory networks using known RBPs
as regulators and optionally incorporating motif-guided priors.
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
        "figures/aim4", "results/aim4"
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    adata = scptr.readwrite.read_h5ad(args.dataset)

    # Pipeline
    print("Running preprocessing...")
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)

    print("Running analysis...")
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)

    # Get known RBPs that are present in this dataset
    known_rbps = scptr.tl.list_known_rbps(organism=args.organism)
    available_rbps = [g for g in known_rbps if g in adata.var_names]
    print(f"Found {len(available_rbps)}/{len(known_rbps)} known RBPs in dataset")

    if len(available_rbps) < 5:
        print("Warning: Very few known RBPs found. Using all genes as regulators.")
        available_rbps = None

    # Load priors if provided
    prior_network = None
    if args.prior_csv:
        print(f"Loading priors from {args.prior_csv}")
        prior_network = scptr.tl.load_motif_priors(args.prior_csv)
        print(f"  Loaded {len(prior_network)} prior edges")

    # Infer network
    print("Inferring regulatory network...")
    network_df = scptr.tl.infer_network(
        adata,
        regulators=available_rbps,
        alpha=args.alpha,
        n_top=args.n_top,
        prior_network=prior_network,
    )

    print(f"Inferred {len(network_df)} edges")

    # Save network
    out_path = res_dir / "regulatory_network.csv"
    network_df.to_csv(out_path, index=False)
    print(f"Network saved to {out_path}")

    # Plot network
    if len(network_df) > 0:
        fig = scptr.pl.network_graph(adata)
        save_figure(fig, "regulatory_network", "figures/aim4")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True,
        help="Path to h5ad file with cancer scRNA-seq data",
    )
    parser.add_argument(
        "--organism", default="human",
        choices=["human", "mouse"],
        help="Organism for RBP list (default: human)",
    )
    parser.add_argument(
        "--prior-csv", default=None,
        help="Path to prior network CSV (regulator, target, weight)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Elastic net mixing parameter (default: 0.5)",
    )
    parser.add_argument(
        "--n-top", type=int, default=50,
        help="Top edges per target (default: 50)",
    )
    main(parser.parse_args())
