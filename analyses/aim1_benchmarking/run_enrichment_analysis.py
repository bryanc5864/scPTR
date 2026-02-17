#!/usr/bin/env python
"""Aim 1: ARE and NMD enrichment analysis.

Tests whether AU-rich element (ARE) genes and nonsense-mediated decay (NMD)
targets show higher degradation rates than background genes.
"""

from __future__ import annotations

import argparse
import json
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
        "figures/aim1", "results/aim1"
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "pancreas":
        adata = scptr.datasets.pancreas()
    else:
        adata = scptr.readwrite.read_h5ad(args.dataset)

    # Pipeline
    print("Running pipeline...")
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)

    # Enrichment tests
    print("Running ARE enrichment...")
    are_result = scptr.benchmark.are_enrichment(adata)
    print(f"  ARE: U={are_result['U_statistic']}, p={are_result['p_value']}")

    print("Running NMD enrichment...")
    nmd_result = scptr.benchmark.nmd_enrichment(adata)
    print(f"  NMD: U={nmd_result['U_statistic']}, p={nmd_result['p_value']}")

    # Plot
    fig = scptr.pl.enrichment_barplot([are_result, nmd_result])
    save_figure(fig, "enrichment_barplot", "figures/aim1")

    # Save results
    out_path = res_dir / "enrichment_results.json"
    with open(out_path, "w") as f:
        json.dump({"ARE": are_result, "NMD": nmd_result}, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="pancreas",
        help="Dataset name or path to h5ad file (default: pancreas)",
    )
    main(parser.parse_args())
