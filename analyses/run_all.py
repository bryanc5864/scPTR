#!/usr/bin/env python
"""Run the full scPTR analysis pipeline on pancreas data and produce results.

This script runs Aims 1-3 end-to-end on the pancreas dataset:
- Aim 1: Benchmark gamma estimates against published half-lives, ARE/NMD enrichment
- Aim 2: PT state discovery and differential gamma analysis
- Aim 3: PT velocity computation

Results are saved to output/ directory.
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from _common import set_figure_style, setup_output_dirs

import scptr

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def save_fig(fig, name, subdir="figures"):
    """Save a matplotlib figure to output dir."""
    if fig is None:
        print(f"  [WARNING] {name}: plot returned None, skipping save")
        return
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    set_figure_style()
    OUTPUT_DIR.mkdir(exist_ok=True)

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("=" * 60)
    print("LOADING PANCREAS DATASET")
    print("=" * 60)
    adata = scptr.datasets.pancreas()
    print(f"  Shape: {adata.shape}")
    print(f"  Layers: {list(adata.layers.keys())}")
    print(f"  Cell types: {adata.obs['clusters'].value_counts().to_dict()}")

    # =========================================================================
    # PREPROCESSING
    # =========================================================================
    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    scptr.pp.filter_genes(adata)
    print(f"  After filtering: {adata.shape}")

    scptr.pp.normalize_layers(adata)
    print("  Normalized layers")

    scptr.pp.neighbors(adata, n_neighbors=30)
    print("  Built kNN graph (k=30)")

    scptr.pp.smooth_layers(adata)
    print("  Smoothed layers (Mu, Ms)")

    # =========================================================================
    # CORE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 60)
    print("CORE ANALYSIS")
    print("=" * 60)

    # Beta estimation (global + per-cell-type)
    scptr.tl.estimate_beta(adata)
    beta = adata.var['beta'].values
    print(f"  Beta: median={np.median(beta):.4f}, max={np.max(beta):.4f}, "
          f"nonzero={np.sum(beta > 0)}/{len(beta)}")

    if "clusters" in adata.obs.columns:
        scptr.tl.estimate_beta(adata, groupby="clusters")
        print(f"  Beta (per-cluster): {adata.varm['beta_groups'].shape}")

    # Gamma estimation
    scptr.tl.estimate_gamma(adata)
    gamma_vals = adata.layers["gamma"]
    gamma_med = np.median(gamma_vals, axis=0)
    print(f"  Gamma: shape={gamma_vals.shape}")
    print(f"    Median per-gene: median={np.median(gamma_med):.4f}, "
          f"max={np.max(gamma_med):.4f}")
    print(f"    Global: max={np.max(gamma_vals):.4f}, "
          f"99.5th pctl={np.percentile(gamma_vals[gamma_vals>0], 99.5):.4f}")
    print(f"    Genes with >0 median gamma: {np.sum(gamma_med > 0)}/{len(gamma_med)}")

    # Variance decomposition
    scptr.tl.variance_decomposition(adata)
    tf = adata.var['tf_score'].values
    ptf = adata.var['ptf_score'].values
    print(f"  TF score:  median={np.median(tf):.4f}, mean={np.mean(tf):.4f}")
    print(f"  PTF score: median={np.median(ptf):.4f}, mean={np.mean(ptf):.4f}")
    print(f"  Genes with TF > 0.5: {np.sum(tf > 0.5)}/{len(tf)}")

    # PT states
    scptr.tl.pt_states(adata)
    n_states = adata.obs["pt_state"].nunique()
    print(f"  PT states found: {n_states}")

    # PT velocity
    scptr.tl.pt_velocity(adata)
    print("  PT velocity computed")

    # =========================================================================
    # AIM 1: BENCHMARKING
    # =========================================================================
    print("\n" + "=" * 60)
    print("AIM 1: BENCHMARKING")
    print("=" * 60)
    fig_dir, res_dir = setup_output_dirs("figures/aim1", "results/aim1")

    # 1a. Half-life correlation (mouse reference)
    print("\n--- Half-life correlation (mouse reference) ---")
    hl_mouse = scptr.datasets.herzog2017_halflives()
    corr = scptr.benchmark.correlate_with_halflives(adata, hl_mouse)
    print(f"  n_genes matched: {corr['n_genes']} (unfiltered: {corr['n_genes_unfiltered']})")
    print(f"  Spearman r = {corr['spearman_r']:.4f} (p = {corr['spearman_p']:.2e})")
    print(f"  Pearson  r = {corr['pearson_r']:.4f} (p = {corr['pearson_p']:.2e})")

    # Also try human reference for cross-species comparison
    print("\n--- Half-life correlation (human reference) ---")
    hl_human = scptr.datasets.schofield2018_halflives()
    corr_human = scptr.benchmark.correlate_with_halflives(adata, hl_human)
    print(f"  n_genes matched: {corr_human['n_genes']} (unfiltered: {corr_human['n_genes_unfiltered']})")
    print(f"  Spearman r = {corr_human['spearman_r']:.4f} (p = {corr_human['spearman_p']:.2e})")

    # Save both correlation results
    corr_save = {k: v for k, v in corr.items() if k != "matched_genes"}
    corr_human_save = {k: v for k, v in corr_human.items() if k != "matched_genes"}
    with open(res_dir / "halflife_correlation.json", "w") as f:
        json.dump({"mouse_reference": corr_save, "human_reference": corr_human_save}, f, indent=2)

    # Half-life scatter plot (log-log scale, filtered genes only)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    gamma_med = np.median(adata.layers["gamma"], axis=0)
    gamma_s = pd.Series(gamma_med, index=adata.var_names)
    hl_s = hl_mouse.set_index("gene_symbol")["half_life_hours"]
    shared = gamma_s.index.intersection(hl_s.index)
    g = gamma_s[shared].values
    h = hl_s[shared].values

    # Left: all genes
    axes[0].scatter(h, g, alpha=0.1, s=5, c="steelblue")
    axes[0].set_xlabel("Published half-life (hours)")
    axes[0].set_ylabel("scPTR median gamma")
    axes[0].set_title(f"All genes (n={len(shared)})")

    # Right: filtered genes (gamma > 0), log-log
    mask = (g > 0) & (h > 0) & np.isfinite(g) & np.isfinite(h)
    axes[1].scatter(h[mask], g[mask], alpha=0.15, s=8, c="steelblue")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Published half-life (hours)")
    axes[1].set_ylabel("scPTR median gamma")
    axes[1].set_title(
        f"Filtered genes (Spearman r={corr['spearman_r']:.3f}, "
        f"p={corr['spearman_p']:.1e}, n={corr['n_genes']})"
    )
    fig.suptitle("Gamma vs Published mRNA Half-lives", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "halflife_scatter", "figures/aim1")

    # 1b. ARE / NMD enrichment
    print("\n--- ARE / NMD enrichment ---")
    are_result = scptr.benchmark.are_enrichment(adata)
    nmd_result = scptr.benchmark.nmd_enrichment(adata)
    print(f"  ARE: n_in={are_result['n_genes_in_set']}, "
          f"median_gamma_in={are_result.get('median_gamma_in_set', 'N/A'):.4f}, "
          f"median_gamma_bg={are_result.get('median_gamma_background', 'N/A'):.4f}, "
          f"p={are_result['p_value']:.4f}")
    print(f"  NMD: n_in={nmd_result['n_genes_in_set']}, "
          f"median_gamma_in={nmd_result.get('median_gamma_in_set', 'N/A'):.4f}, "
          f"median_gamma_bg={nmd_result.get('median_gamma_background', 'N/A'):.4f}, "
          f"p={nmd_result['p_value']:.4f}")

    with open(res_dir / "enrichment_results.json", "w") as f:
        json.dump({"ARE": are_result, "NMD": nmd_result}, f, indent=2)

    fig = scptr.pl.enrichment_barplot([are_result, nmd_result])
    save_fig(fig, "enrichment_barplot", "figures/aim1")

    # 1c. Subsampling robustness
    print("\n--- Subsampling robustness ---")
    fractions = [0.2, 0.4, 0.6, 0.8, 0.9]
    robust_df = scptr.benchmark.subsampling_robustness(
        adata, fractions=fractions, n_repeats=5
    )
    robust_df.to_csv(res_dir / "subsampling_robustness.csv", index=False)

    for frac in fractions:
        sub = robust_df[robust_df["fraction"] == frac]
        mean_r = sub["spearman_r"].mean()
        print(f"  fraction={frac:.1f}: mean Spearman r = {mean_r:.4f}")

    # Robustness plot
    fig, ax = plt.subplots(figsize=(6, 4))
    for frac in fractions:
        sub = robust_df[robust_df["fraction"] == frac]
        ax.scatter([frac] * len(sub), sub["spearman_r"],
                   color="steelblue", alpha=0.6, s=25)
    means = robust_df.groupby("fraction")["spearman_r"].mean()
    ax.plot(means.index, means.values, "o-", color="darkblue", linewidth=2, markersize=6)
    ax.set_xlabel("Fraction of cells")
    ax.set_ylabel("Spearman r (vs full data)")
    ax.set_title("Subsampling Robustness")
    ax.set_ylim(0.5, 1.02)
    save_fig(fig, "subsampling_robustness", "figures/aim1")

    # =========================================================================
    # AIM 2: HIDDEN PT STATES
    # =========================================================================
    print("\n" + "=" * 60)
    print("AIM 2: PT STATE DISCOVERY")
    print("=" * 60)
    fig_dir, res_dir = setup_output_dirs("figures/aim2", "results/aim2")

    # State composition
    state_counts = adata.obs["pt_state"].value_counts()
    state_counts.to_csv(res_dir / "pt_state_counts.csv")
    print(f"  PT states: {dict(state_counts)}")

    # PT UMAP (use show=False to get fig back)
    fig = scptr.pl.pt_umap(adata, show=False)
    save_fig(fig, "pt_umap", "figures/aim2")

    # TF vs PTF scatter
    fig = scptr.pl.tf_ptf_scatter(adata, show=False)
    save_fig(fig, "tf_ptf_scatter", "figures/aim2")

    # Cross-tabulate PT states vs expression clusters
    if "clusters" in adata.obs.columns:
        ct = pd.crosstab(adata.obs["pt_state"], adata.obs["clusters"])
        ct.to_csv(res_dir / "pt_state_vs_clusters.csv")
        print(f"\n  PT state vs expression cluster crosstab:")
        print(ct.to_string())

    # Rank genes by differential gamma
    rank_df = scptr.tl.rank_pt_genes(adata, n_genes=50)
    rank_df.to_csv(res_dir / "ranked_pt_genes.csv", index=False)
    print(f"\n  Top differentially degraded genes: {len(rank_df)} entries")
    print(f"  Top 10 gene names: {rank_df.head(10)['names'].tolist()}")

    # Gamma heatmap
    fig = scptr.pl.gamma_heatmap(adata, show=False)
    save_fig(fig, "gamma_heatmap", "figures/aim2")

    # =========================================================================
    # AIM 3: PT VELOCITY
    # =========================================================================
    print("\n" + "=" * 60)
    print("AIM 3: PT VELOCITY")
    print("=" * 60)
    fig_dir, res_dir = setup_output_dirs("figures/aim3", "results/aim3")

    # Velocity embedding (show 30% of cells for cleaner arrows)
    fig = scptr.pl.pt_velocity_embedding(adata, density=0.3, arrow_size=1.5, show=False)
    save_fig(fig, "pt_velocity_embedding", "figures/aim3")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Dataset: pancreas ({adata.n_obs} cells, {adata.n_vars} genes)")
    print(f"  Beta: median={np.median(adata.var['beta']):.4f}, max={np.max(adata.var['beta']):.4f}")
    print(f"  Gamma max: {np.max(adata.layers['gamma']):.4f}")
    print(f"  PT states discovered: {n_states}")
    print(f"  TF score: median={np.median(adata.var['tf_score']):.4f}")
    print(f"  Half-life Spearman r (mouse): {corr['spearman_r']:.4f} (n={corr['n_genes']} genes)")
    print(f"  Half-life Spearman r (human): {corr_human['spearman_r']:.4f} (n={corr_human['n_genes']} genes)")
    print(f"  ARE enrichment p: {are_result['p_value']:.4f}")
    print(f"  NMD enrichment p: {nmd_result['p_value']:.4f}")
    print(f"  Robustness (90% cells): {robust_df[robust_df['fraction']==0.9]['spearman_r'].mean():.4f}")
    print(f"\nAll results saved to: {OUTPUT_DIR.resolve()}")
    print("Done!")


if __name__ == "__main__":
    main()
