#!/usr/bin/env python
"""miRNA-target analysis: test whether miRNA-targeted genes have higher gamma.

Uses TargetScan 8.0 predictions to identify miRNA-target relationships,
then tests whether predicted targets have systematically higher degradation
rates (gamma) than non-targets using Mann-Whitney U tests.

This addresses Aim 4 of the research plan: post-transcriptional regulatory networks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from _common import set_figure_style

import scptr

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "mirna_analysis"


def save_fig(fig, name, subdir="figures"):
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def run_mirna_analysis(adata, dataset_name, mirna_targets):
    """Run miRNA-gamma correlation analysis on a dataset."""
    print(f"\n{'='*60}")
    print(f"miRNA ANALYSIS: {dataset_name}")
    print(f"{'='*60}")

    # Run scPTR pipeline
    import copy
    adata = copy.deepcopy(adata)
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)

    # Run miRNA-gamma correlation
    print(f"  Running miRNA-gamma correlation...")
    result_df = scptr.tl.mirna_gamma_correlation(
        adata, mirna_targets, n_top_targets=200, min_cells_expressing=50
    )

    if len(result_df) == 0:
        print(f"  No miRNA families with sufficient targets found.")
        return None

    # Summary statistics
    n_tested = len(result_df)
    n_sig = (result_df["fdr"] < 0.05).sum()
    n_sig_10 = (result_df["fdr"] < 0.10).sum()
    n_enriched = (result_df["fold_enrichment"] > 1.0).sum()

    print(f"\n  Results:")
    print(f"    miRNA families tested: {n_tested}")
    print(f"    Significant (FDR < 0.05): {n_sig} ({100*n_sig/n_tested:.1f}%)")
    print(f"    Significant (FDR < 0.10): {n_sig_10} ({100*n_sig_10/n_tested:.1f}%)")
    print(f"    Enriched (fold > 1.0): {n_enriched} ({100*n_enriched/n_tested:.1f}%)")
    print(f"    Median fold enrichment: {result_df['fold_enrichment'].median():.3f}")

    # Top significant miRNAs
    top_sig = result_df[result_df["fdr"] < 0.10].head(20)
    if len(top_sig) > 0:
        print(f"\n  Top significant miRNAs (FDR < 0.10):")
        for _, row in top_sig.iterrows():
            print(f"    {row['representative_mirna']:>25s}  "
                  f"n_targets={row['n_targets_in_data']:3d}  "
                  f"fold={row['fold_enrichment']:.2f}  "
                  f"p={row['mannwhitney_p']:.2e}  "
                  f"FDR={row['fdr']:.3f}")

    # Top miRNAs by effect size regardless of significance
    top_effect = result_df.nlargest(10, "fold_enrichment")
    print(f"\n  Top miRNAs by fold enrichment:")
    for _, row in top_effect.iterrows():
        print(f"    {row['representative_mirna']:>25s}  "
              f"fold={row['fold_enrichment']:.2f}  "
              f"FDR={row['fdr']:.3f}")

    # Aggregate test: all miRNA targets vs non-targets
    gamma = np.median(adata.layers["gamma"], axis=0)
    gene_names_upper = [g.upper() for g in adata.var_names]

    all_target_genes = set()
    for _, row in mirna_targets.iterrows():
        all_target_genes.add(str(row["gene_symbol"]).upper())

    informative = (adata.layers["gamma"] > 0).mean(axis=0) >= 0.1
    target_gamma = []
    nontarget_gamma = []
    for i, g in enumerate(gene_names_upper):
        if not informative[i]:
            continue
        if g in all_target_genes:
            target_gamma.append(gamma[i])
        else:
            nontarget_gamma.append(gamma[i])

    if len(target_gamma) >= 10 and len(nontarget_gamma) >= 10:
        u, p = stats.mannwhitneyu(target_gamma, nontarget_gamma, alternative="greater")
        print(f"\n  Aggregate test (all targets vs non-targets):")
        print(f"    Target genes in data: {len(target_gamma)}")
        print(f"    Non-target genes: {len(nontarget_gamma)}")
        print(f"    Target median gamma: {np.median(target_gamma):.6f}")
        print(f"    Non-target median gamma: {np.median(nontarget_gamma):.6f}")
        print(f"    Fold: {np.median(target_gamma) / (np.median(nontarget_gamma) + 1e-8):.3f}")
        print(f"    Mann-Whitney p: {p:.2e}")

    # Save results
    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(res_dir / f"mirna_gamma_{dataset_name}.csv", index=False)

    # Figures
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Volcano plot (fold enrichment vs -log10 p)
    neg_log_p = -np.log10(result_df["mannwhitney_p"].clip(lower=1e-50))
    sig_mask = result_df["fdr"] < 0.05
    axes[0].scatter(result_df["fold_enrichment"][~sig_mask], neg_log_p[~sig_mask],
                    s=10, alpha=0.3, color="gray", label="NS")
    axes[0].scatter(result_df["fold_enrichment"][sig_mask], neg_log_p[sig_mask],
                    s=20, alpha=0.7, color="red", label=f"FDR<0.05 (n={sig_mask.sum()})")
    axes[0].axhline(y=-np.log10(0.05), color="blue", linestyle="--", alpha=0.5)
    axes[0].axvline(x=1.0, color="black", linestyle="--", alpha=0.3)
    axes[0].set_xlabel("Fold enrichment (target/non-target gamma)")
    axes[0].set_ylabel("-log10(p)")
    axes[0].set_title(f"miRNA Target Enrichment ({dataset_name})")
    axes[0].legend()

    # Panel 2: Distribution of fold enrichments
    axes[1].hist(result_df["fold_enrichment"], bins=30, color="steelblue",
                 edgecolor="black", linewidth=0.5)
    axes[1].axvline(x=1.0, color="red", linestyle="--", label="No enrichment")
    axes[1].axvline(x=result_df["fold_enrichment"].median(), color="green",
                    linestyle="--", label=f"Median={result_df['fold_enrichment'].median():.2f}")
    axes[1].set_xlabel("Fold enrichment")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of Fold Enrichments")
    axes[1].legend()

    # Panel 3: Aggregate target vs non-target boxplot
    if len(target_gamma) >= 10:
        box_data = [target_gamma, nontarget_gamma]
        bp = axes[2].boxplot(box_data, labels=["miRNA\ntargets", "Non-\ntargets"],
                             patch_artist=True)
        bp["boxes"][0].set_facecolor("coral")
        bp["boxes"][1].set_facecolor("lightblue")
        axes[2].set_ylabel("Median gamma per gene")
        axes[2].set_title(f"Aggregate: targets vs non-targets\np={p:.2e}")
        axes[2].set_yscale("symlog", linthresh=0.001)

    fig.suptitle(f"miRNA-Gamma Analysis: {dataset_name}", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, f"mirna_analysis_{dataset_name}")

    return {
        "n_families_tested": n_tested,
        "n_significant_005": int(n_sig),
        "n_significant_010": int(n_sig_10),
        "n_enriched": int(n_enriched),
        "median_fold_enrichment": float(result_df["fold_enrichment"].median()),
        "aggregate_p": float(p) if len(target_gamma) >= 10 else None,
    }


def main():
    set_figure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load TargetScan predictions
    print("=" * 60)
    print("LOADING TARGETSCAN PREDICTIONS")
    print("=" * 60)

    cache_dir = Path(__file__).parent.parent / ".cache" / "targetscan"
    try:
        mirna_targets = scptr.tl.load_targetscan_predictions(
            species_id=9606,  # Human
            min_context_score=-0.2,
            cache_dir=cache_dir,
        )
        print(f"  Loaded {len(mirna_targets)} human miRNA-target predictions")
        print(f"  miRNA families: {mirna_targets['mirna_family'].nunique()}")
        print(f"  Target genes: {mirna_targets['gene_symbol'].nunique()}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Please download TargetScan data first.")
        sys.exit(1)

    # Also load mouse predictions for mouse datasets
    try:
        mirna_targets_mouse = scptr.tl.load_targetscan_predictions(
            species_id=10090,  # Mouse
            min_context_score=-0.2,
            cache_dir=cache_dir,
        )
        print(f"  Loaded {len(mirna_targets_mouse)} mouse miRNA-target predictions")
        print(f"  miRNA families: {mirna_targets_mouse['mirna_family'].nunique()}")
        print(f"  Target genes: {mirna_targets_mouse['gene_symbol'].nunique()}")
    except Exception as e:
        print(f"  Mouse predictions not available: {e}")
        mirna_targets_mouse = mirna_targets  # Fallback: use human

    # Load datasets
    print("\n" + "=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    adata_pan = scptr.datasets.pancreas()
    adata_dg = scptr.datasets.dentate_gyrus()

    # Try to load sci-fate
    try:
        adata_sci = scptr.datasets.sci_fate()
    except Exception:
        adata_sci = None

    # Run analysis on each dataset
    all_results = {}

    # Pancreas (mouse) - use mouse predictions
    all_results["pancreas"] = run_mirna_analysis(
        adata_pan, "pancreas", mirna_targets_mouse
    )

    # Dentate Gyrus (mouse) - use mouse predictions
    all_results["dentate_gyrus"] = run_mirna_analysis(
        adata_dg, "dentate_gyrus", mirna_targets_mouse
    )

    # sci-fate (human A549) - use human predictions
    if adata_sci is not None:
        all_results["sci_fate"] = run_mirna_analysis(
            adata_sci, "sci_fate", mirna_targets
        )

    # Save summary
    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "mirna_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'='*60}")
    print("miRNA ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':>15s}  {'Tested':>7s}  {'Sig(5%)':>7s}  {'Sig(10%)':>8s}  "
          f"{'Enriched':>8s}  {'Med.Fold':>8s}  {'Agg.p':>10s}")
    for name, res in all_results.items():
        if res is None:
            continue
        print(f"{name:>15s}  {res['n_families_tested']:>7d}  "
              f"{res['n_significant_005']:>7d}  {res['n_significant_010']:>8d}  "
              f"{res['n_enriched']:>8d}  {res['median_fold_enrichment']:>8.3f}  "
              f"{res['aggregate_p']:>10.2e}" if res['aggregate_p'] else "")

    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
