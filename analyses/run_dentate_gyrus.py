#!/usr/bin/env python
"""Run the full scPTR analysis pipeline on dentate gyrus data.

This script mirrors run_all.py but on the dentate gyrus neurogenesis dataset.
Results are saved to output/dentate_gyrus/ directory.
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

sys.path.insert(0, str(Path(__file__).parent))
from _common import set_figure_style

import scptr

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "dentate_gyrus"


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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("=" * 60)
    print("LOADING DENTATE GYRUS DATASET")
    print("=" * 60)
    adata = scptr.datasets.dentate_gyrus()
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

    scptr.tl.estimate_beta(adata)
    beta = adata.var['beta'].values
    print(f"  Beta: median={np.median(beta):.4f}, max={np.max(beta):.4f}, "
          f"nonzero={np.sum(beta > 0)}/{len(beta)}")

    scptr.tl.estimate_beta(adata, groupby="clusters")
    print(f"  Beta (per-cluster): {adata.varm['beta_groups'].shape}")

    scptr.tl.estimate_gamma(adata)
    gamma_vals = adata.layers["gamma"]
    gamma_med = np.median(gamma_vals, axis=0)
    print(f"  Gamma: shape={gamma_vals.shape}")
    print(f"    Median per-gene: median={np.median(gamma_med):.4f}, "
          f"max={np.max(gamma_med):.4f}")
    print(f"    Global max={np.max(gamma_vals):.4f}")
    print(f"    Genes with >0 median gamma: {np.sum(gamma_med > 0)}/{len(gamma_med)}")

    scptr.tl.variance_decomposition(adata)
    tf = adata.var['tf_score'].values
    print(f"  TF score:  median={np.median(tf):.4f}, mean={np.mean(tf):.4f}")
    print(f"  Genes with TF > 0.5: {np.sum(tf > 0.5)}/{len(tf)}")

    scptr.tl.pt_states(adata)
    n_states = adata.obs["pt_state"].nunique()
    print(f"  PT states found: {n_states}")

    scptr.tl.pt_velocity(adata)
    print("  PT velocity computed")

    # =========================================================================
    # BENCHMARKING
    # =========================================================================
    print("\n" + "=" * 60)
    print("BENCHMARKING")
    print("=" * 60)
    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Half-life correlation (mouse reference — dentate gyrus is mouse data)
    print("\n--- Half-life correlation (mouse reference) ---")
    hl_mouse = scptr.datasets.herzog2017_halflives()
    corr = scptr.benchmark.correlate_with_halflives(adata, hl_mouse)
    print(f"  n_genes matched: {corr['n_genes']} (unfiltered: {corr['n_genes_unfiltered']})")
    print(f"  Spearman r = {corr['spearman_r']:.4f} (p = {corr['spearman_p']:.2e})")
    print(f"  Pearson  r = {corr['pearson_r']:.4f} (p = {corr['pearson_p']:.2e})")

    # Also human reference
    print("\n--- Half-life correlation (human reference) ---")
    hl_human = scptr.datasets.schofield2018_halflives()
    corr_human = scptr.benchmark.correlate_with_halflives(adata, hl_human)
    print(f"  n_genes matched: {corr_human['n_genes']} (unfiltered: {corr_human['n_genes_unfiltered']})")
    print(f"  Spearman r = {corr_human['spearman_r']:.4f} (p = {corr_human['spearman_p']:.2e})")

    corr_save = {k: v for k, v in corr.items() if k != "matched_genes"}
    corr_human_save = {k: v for k, v in corr_human.items() if k != "matched_genes"}
    with open(res_dir / "halflife_correlation.json", "w") as f:
        json.dump({"mouse_reference": corr_save, "human_reference": corr_human_save}, f, indent=2)

    # Half-life scatter
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    gamma_med_s = pd.Series(gamma_med, index=adata.var_names)
    hl_s = hl_mouse.set_index("gene_symbol")["half_life_hours"]
    shared = gamma_med_s.index.intersection(hl_s.index)
    g = gamma_med_s[shared].values
    h = hl_s[shared].values

    axes[0].scatter(h, g, alpha=0.1, s=5, c="steelblue")
    axes[0].set_xlabel("Published half-life (hours)")
    axes[0].set_ylabel("scPTR median gamma")
    axes[0].set_title(f"All genes (n={len(shared)})")

    mask = (g > 0) & (h > 0) & np.isfinite(g) & np.isfinite(h)
    axes[1].scatter(h[mask], g[mask], alpha=0.15, s=8, c="steelblue")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Published half-life (hours)")
    axes[1].set_ylabel("scPTR median gamma")
    axes[1].set_title(
        f"Filtered (Spearman r={corr['spearman_r']:.3f}, "
        f"p={corr['spearman_p']:.1e}, n={corr['n_genes']})"
    )
    fig.suptitle("Dentate Gyrus: Gamma vs Published Half-lives", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "halflife_scatter")

    # ARE/NMD enrichment
    print("\n--- ARE / NMD enrichment ---")
    are_result = scptr.benchmark.are_enrichment(adata)
    nmd_result = scptr.benchmark.nmd_enrichment(adata)
    print(f"  ARE: n_in={are_result['n_genes_in_set']}, p={are_result['p_value']:.4f}")
    print(f"  NMD: n_in={nmd_result['n_genes_in_set']}, p={nmd_result['p_value']:.4f}")

    with open(res_dir / "enrichment_results.json", "w") as f:
        json.dump({"ARE": are_result, "NMD": nmd_result}, f, indent=2)

    fig = scptr.pl.enrichment_barplot([are_result, nmd_result])
    save_fig(fig, "enrichment_barplot")

    # Subsampling robustness
    print("\n--- Subsampling robustness ---")
    fractions = [0.2, 0.4, 0.6, 0.8, 0.9]
    robust_df = scptr.benchmark.subsampling_robustness(
        adata, fractions=fractions, n_repeats=5
    )
    robust_df.to_csv(res_dir / "subsampling_robustness.csv", index=False)
    for frac in fractions:
        sub = robust_df[robust_df["fraction"] == frac]
        print(f"  fraction={frac:.1f}: mean Spearman r = {sub['spearman_r'].mean():.4f}")

    # =========================================================================
    # PT STATES
    # =========================================================================
    print("\n" + "=" * 60)
    print("PT STATE DISCOVERY")
    print("=" * 60)

    state_counts = adata.obs["pt_state"].value_counts()
    state_counts.to_csv(res_dir / "pt_state_counts.csv")
    print(f"  PT states: {dict(state_counts)}")

    fig = scptr.pl.pt_umap(adata, show=False)
    save_fig(fig, "pt_umap")

    fig = scptr.pl.tf_ptf_scatter(adata, show=False)
    save_fig(fig, "tf_ptf_scatter")

    ct = pd.crosstab(adata.obs["pt_state"], adata.obs["clusters"])
    ct.to_csv(res_dir / "pt_state_vs_clusters.csv")
    print(f"\n  PT state vs expression cluster crosstab:")
    print(ct.to_string())

    rank_df = scptr.tl.rank_pt_genes(adata, n_genes=50)
    rank_df.to_csv(res_dir / "ranked_pt_genes.csv", index=False)
    print(f"\n  Top differentially degraded genes: {len(rank_df)} entries")
    print(f"  Top 10: {rank_df.head(10)['names'].tolist()}")

    fig = scptr.pl.gamma_heatmap(adata, show=False)
    save_fig(fig, "gamma_heatmap")

    # =========================================================================
    # PT VELOCITY
    # =========================================================================
    print("\n" + "=" * 60)
    print("PT VELOCITY")
    print("=" * 60)

    fig = scptr.pl.pt_velocity_embedding(adata, density=0.3, arrow_size=1.5, show=False)
    save_fig(fig, "pt_velocity_embedding")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Dataset: dentate_gyrus ({adata.n_obs} cells, {adata.n_vars} genes)")
    print(f"  Beta: median={np.median(adata.var['beta']):.4f}, max={np.max(adata.var['beta']):.4f}")
    print(f"  Gamma max: {np.max(adata.layers['gamma']):.4f}")
    print(f"  PT states discovered: {n_states}")
    print(f"  TF score: median={np.median(adata.var['tf_score']):.4f}")
    print(f"  Half-life Spearman r (mouse): {corr['spearman_r']:.4f} (n={corr['n_genes']})")
    print(f"  Half-life Spearman r (human): {corr_human['spearman_r']:.4f} (n={corr_human['n_genes']})")
    print(f"  Robustness (90%): {robust_df[robust_df['fraction']==0.9]['spearman_r'].mean():.4f}")
    print(f"\nAll results saved to: {OUTPUT_DIR.resolve()}")

    # Return adata for cross-dataset use
    return adata


if __name__ == "__main__":
    main()
