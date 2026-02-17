#!/usr/bin/env python
"""Aim 1: Validate gamma estimates against published mRNA half-lives.

Correlates per-gene median gamma from scPTR with SLAM-seq (Herzog 2017)
and TimeLapse-seq (Schofield 2018) measurements.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from _common import set_figure_style, save_figure, setup_output_dirs

import scptr


def main(args: argparse.Namespace) -> None:
    set_figure_style()
    fig_dir, res_dir = setup_output_dirs(
        "figures/aim1", "results/aim1"
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "pancreas":
        adata = scptr.datasets.pancreas()
    else:
        adata = scptr.readwrite.read_h5ad(args.dataset)

    # Run scPTR pipeline
    print("Running preprocessing...")
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=args.n_neighbors)
    scptr.pp.smooth_layers(adata)

    print("Estimating beta and gamma...")
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)

    # Load half-life references
    print("Loading half-life references...")
    herzog = scptr.datasets.herzog2017_halflives()
    schofield = scptr.datasets.schofield2018_halflives()

    # Correlate
    results = {}
    for name, hl_df in [("herzog2017", herzog), ("schofield2018", schofield)]:
        print(f"Correlating with {name}...")
        corr = scptr.benchmark.correlate_with_halflives(adata, hl_df)
        results[name] = corr
        print(f"  Spearman r={corr['spearman_r']:.3f}, p={corr['spearman_p']:.2e}")
        print(f"  Pearson  r={corr['pearson_r']:.3f}, p={corr['pearson_p']:.2e}")
        print(f"  n_genes={corr['n_genes']}")

        # Plot
        fig = scptr.pl.halflife_scatter(adata, hl_df)
        save_figure(fig, f"halflife_scatter_{name}", "figures/aim1")

    # Save results
    out_path = res_dir / "halflife_correlations.json"
    # Convert to serializable format
    serializable = {}
    for k, v in results.items():
        serializable[k] = {kk: vv for kk, vv in v.items() if kk != "matched_genes"}
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="pancreas",
        help="Dataset name or path to h5ad file (default: pancreas)",
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=30,
        help="Number of neighbors for kNN graph (default: 30)",
    )
    main(parser.parse_args())
