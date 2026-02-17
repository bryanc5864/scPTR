#!/usr/bin/env python
"""Aim 1: Subsampling robustness and cross-platform consistency.

Tests how stable gamma estimates are across cell subsamples and
across different datasets.
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


def run_pipeline(adata):
    """Run the standard scPTR pipeline on an AnnData."""
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)
    return adata


def main(args: argparse.Namespace) -> None:
    set_figure_style()
    fig_dir, res_dir = setup_output_dirs(
        "figures/aim1", "results/aim1"
    )

    # Load and process dataset
    print("Loading pancreas dataset...")
    adata = scptr.datasets.pancreas()
    run_pipeline(adata)

    # Subsampling robustness
    print("Running subsampling robustness...")
    fractions = [0.2, 0.4, 0.6, 0.8, 0.9]
    robust_df = scptr.benchmark.subsampling_robustness(
        adata, fractions=fractions, n_repeats=args.n_repeats
    )
    robust_df.to_csv(res_dir / "subsampling_robustness.csv", index=False)

    # Plot robustness
    fig, ax = plt.subplots(figsize=(6, 4))
    for frac in fractions:
        sub = robust_df[robust_df["fraction"] == frac]
        ax.scatter(
            [frac] * len(sub), sub["spearman_r"],
            color="steelblue", alpha=0.7, s=30,
        )
    means = robust_df.groupby("fraction")["spearman_r"].mean()
    ax.plot(means.index, means.values, "o-", color="darkblue", linewidth=2)
    ax.set_xlabel("Fraction of cells")
    ax.set_ylabel("Spearman r (vs full data)")
    ax.set_title("Subsampling Robustness")
    ax.set_ylim(0, 1.05)
    save_figure(fig, "subsampling_robustness", "figures/aim1")

    # Cross-dataset consistency (if dentate gyrus also available)
    if not args.skip_cross_dataset:
        print("Loading dentate gyrus for cross-dataset consistency...")
        try:
            dg = scptr.datasets.dentate_gyrus()
            run_pipeline(dg)

            consistency_df = scptr.benchmark.cross_dataset_consistency({
                "pancreas": adata,
                "dentate_gyrus": dg,
            })
            consistency_df.to_csv(res_dir / "cross_dataset_consistency.csv", index=False)
            print("Cross-dataset consistency:")
            print(consistency_df.to_string(index=False))
        except Exception as e:
            print(f"Skipping cross-dataset: {e}")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-repeats", type=int, default=5,
        help="Number of repeats per fraction (default: 5)",
    )
    parser.add_argument(
        "--skip-cross-dataset", action="store_true",
        help="Skip cross-dataset consistency analysis",
    )
    main(parser.parse_args())
