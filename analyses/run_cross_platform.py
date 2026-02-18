#!/usr/bin/env python
"""Cross-platform benchmarking: compare scPTR gamma estimates across
different sequencing platforms and datasets.

Compares gamma estimates between:
1. 10x Chromium datasets (pancreas, dentate gyrus)
2. sci (combinatorial indexing) dataset (sci-fate A549)
3. Assesses whether gene-level gamma rankings are consistent across platforms

This addresses the cross-platform benchmarking component of the research plan.
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

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "cross_platform"


def save_fig(fig, name, subdir="figures"):
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def run_pipeline(adata, name):
    """Run full scPTR pipeline and return per-gene median gamma."""
    import copy
    adata = copy.deepcopy(adata)
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)

    gamma = np.median(adata.layers["gamma"], axis=0)
    expr = np.mean(adata.layers["spliced"], axis=0) if "spliced" in adata.layers else np.mean(adata.X, axis=0)
    if hasattr(expr, 'A1'):
        expr = np.asarray(expr).flatten()

    return pd.DataFrame({
        "gene": adata.var_names,
        "gamma": gamma,
        "expression": expr,
        "nonzero_frac": (adata.layers["gamma"] > 0).mean(axis=0),
    }).set_index("gene"), adata


def compare_datasets(df_a, df_b, name_a, name_b):
    """Compare gamma estimates between two datasets."""
    print(f"\n  {name_a} vs {name_b}:")

    # Find shared genes (case-insensitive)
    genes_a = {g.upper(): g for g in df_a.index}
    genes_b = {g.upper(): g for g in df_b.index}
    shared = set(genes_a.keys()) & set(genes_b.keys())
    print(f"    Shared genes: {len(shared)}")

    if len(shared) < 50:
        print(f"    Too few shared genes for comparison.")
        return None

    gamma_a = np.array([df_a.loc[genes_a[g], "gamma"] for g in shared])
    gamma_b = np.array([df_b.loc[genes_b[g], "gamma"] for g in shared])
    expr_a = np.array([df_a.loc[genes_a[g], "expression"] for g in shared])
    expr_b = np.array([df_b.loc[genes_b[g], "expression"] for g in shared])
    nonzero_a = np.array([df_a.loc[genes_a[g], "nonzero_frac"] for g in shared])
    nonzero_b = np.array([df_b.loc[genes_b[g], "nonzero_frac"] for g in shared])

    # Overall correlation
    valid = (gamma_a > 0) & (gamma_b > 0)
    if valid.sum() < 20:
        print(f"    Too few valid genes (both gamma>0): {valid.sum()}")
        return None

    r_gamma, p_gamma = stats.spearmanr(gamma_a[valid], gamma_b[valid])
    r_expr, p_expr = stats.spearmanr(expr_a[valid], expr_b[valid])

    print(f"    Gamma Spearman r = {r_gamma:.4f} (n={valid.sum()})")
    print(f"    Expression Spearman r = {r_expr:.4f}")

    # Stratify by expression level
    expr_combined = expr_a + expr_b
    quartiles = np.percentile(expr_combined[valid], [25, 50, 75])
    labels = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    bounds = [(-np.inf, quartiles[0]), (quartiles[0], quartiles[1]),
              (quartiles[1], quartiles[2]), (quartiles[2], np.inf)]

    print(f"\n    Stratified by expression level:")
    stratified = []
    for label, (lo, hi) in zip(labels, bounds):
        mask = valid & (expr_combined >= lo) & (expr_combined < hi)
        if mask.sum() < 10:
            continue
        r_q, p_q = stats.spearmanr(gamma_a[mask], gamma_b[mask])
        r_e, _ = stats.spearmanr(expr_a[mask], expr_b[mask])
        print(f"      {label}: gamma r={r_q:.3f}, expr r={r_e:.3f} (n={mask.sum()})")
        stratified.append({
            "quartile": label,
            "gamma_r": float(r_q),
            "expr_r": float(r_e),
            "n_genes": int(mask.sum()),
        })

    # Informative genes only (>10% nonzero in both)
    informative = valid & (nonzero_a >= 0.1) & (nonzero_b >= 0.1)
    if informative.sum() >= 20:
        r_inf, _ = stats.spearmanr(gamma_a[informative], gamma_b[informative])
        print(f"\n    Informative genes only (>10% nonzero both): "
              f"r={r_inf:.4f} (n={informative.sum()})")

    return {
        "dataset_a": name_a,
        "dataset_b": name_b,
        "shared_genes": len(shared),
        "valid_genes": int(valid.sum()),
        "gamma_r": float(r_gamma),
        "expr_r": float(r_expr),
        "informative_gamma_r": float(r_inf) if informative.sum() >= 20 else None,
        "stratified": stratified,
    }


def main():
    set_figure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("=" * 60)
    print("CROSS-PLATFORM BENCHMARKING")
    print("=" * 60)

    datasets = {}

    # 10x Chromium datasets
    print("\nLoading pancreas (10x Chromium)...")
    df_pan, adata_pan = run_pipeline(scptr.datasets.pancreas(), "pancreas")
    datasets["pancreas_10x"] = df_pan
    print(f"  {len(df_pan)} genes, {(df_pan['gamma'] > 0).sum()} with gamma>0")

    print("\nLoading dentate gyrus (10x Chromium)...")
    df_dg, adata_dg = run_pipeline(scptr.datasets.dentate_gyrus(), "dentate_gyrus")
    datasets["dg_10x"] = df_dg
    print(f"  {len(df_dg)} genes, {(df_dg['gamma'] > 0).sum()} with gamma>0")

    # sci-fate (combinatorial indexing)
    try:
        print("\nLoading sci-fate (sci)...")
        df_sci, adata_sci = run_pipeline(scptr.datasets.sci_fate(), "sci_fate")
        datasets["scifate_sci"] = df_sci
        print(f"  {len(df_sci)} genes, {(df_sci['gamma'] > 0).sum()} with gamma>0")
    except Exception as e:
        print(f"  sci-fate not available: {e}")

    # Pairwise comparisons
    print("\n" + "=" * 60)
    print("PAIRWISE COMPARISONS")
    print("=" * 60)

    pairs = []
    dataset_names = list(datasets.keys())
    all_comparisons = []

    for i in range(len(dataset_names)):
        for j in range(i + 1, len(dataset_names)):
            name_a, name_b = dataset_names[i], dataset_names[j]
            result = compare_datasets(datasets[name_a], datasets[name_b],
                                      name_a, name_b)
            if result:
                all_comparisons.append(result)

    # Platform comparison summary
    print(f"\n{'='*60}")
    print("PLATFORM COMPARISON SUMMARY")
    print(f"{'='*60}")

    # Categorize comparisons
    same_platform = []
    cross_platform = []
    for comp in all_comparisons:
        a, b = comp["dataset_a"], comp["dataset_b"]
        a_platform = "10x" if "10x" in a else "sci" if "sci" in a else "other"
        b_platform = "10x" if "10x" in b else "sci" if "sci" in b else "other"

        if a_platform == b_platform:
            same_platform.append(comp)
        else:
            cross_platform.append(comp)

    print(f"\n  Same platform comparisons:")
    for comp in same_platform:
        print(f"    {comp['dataset_a']} vs {comp['dataset_b']}: "
              f"gamma r={comp['gamma_r']:.3f}")

    print(f"\n  Cross-platform comparisons:")
    for comp in cross_platform:
        print(f"    {comp['dataset_a']} vs {comp['dataset_b']}: "
              f"gamma r={comp['gamma_r']:.3f}")

    # Half-life validation per platform
    print(f"\n  Half-life validation per platform:")
    halflife_dir = Path(__file__).parent.parent / "src" / "scptr" / "datasets" / "data"
    for hl_file, hl_name in [("schofield2018_halflives.csv", "Schofield 2018")]:
        hl_path = halflife_dir / hl_file
        if not hl_path.exists():
            continue

        hl = pd.read_csv(hl_path)
        for ds_name, df in datasets.items():
            gene_map = {g.upper(): g for g in df.index}
            gamma_vals, hl_vals = [], []
            for _, row in hl.iterrows():
                g = str(row.iloc[0]).upper()
                if g in gene_map and df.loc[gene_map[g], "gamma"] > 0:
                    gamma_vals.append(df.loc[gene_map[g], "gamma"])
                    hl_vals.append(float(row.iloc[1]))
            if len(gamma_vals) >= 20:
                r, p = stats.spearmanr(gamma_vals, hl_vals)
                print(f"    {ds_name}: r={r:.4f}, n={len(gamma_vals)} ({hl_name})")

    # Save results
    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "cross_platform_results.json", "w") as f:
        json.dump(all_comparisons, f, indent=2)

    # Figure
    n_comps = len(all_comparisons)
    fig, axes = plt.subplots(1, max(n_comps, 1), figsize=(6 * max(n_comps, 1), 5))
    if n_comps == 1:
        axes = [axes]

    for idx, comp in enumerate(all_comparisons):
        name_a, name_b = comp["dataset_a"], comp["dataset_b"]
        df_a, df_b = datasets[name_a], datasets[name_b]

        genes_a = {g.upper(): g for g in df_a.index}
        genes_b = {g.upper(): g for g in df_b.index}
        shared = set(genes_a.keys()) & set(genes_b.keys())

        ga = np.array([df_a.loc[genes_a[g], "gamma"] for g in shared])
        gb = np.array([df_b.loc[genes_b[g], "gamma"] for g in shared])
        valid = (ga > 0) & (gb > 0)

        axes[idx].scatter(ga[valid], gb[valid], s=2, alpha=0.3, color="steelblue")
        axes[idx].set_xlabel(f"Gamma ({name_a})")
        axes[idx].set_ylabel(f"Gamma ({name_b})")
        axes[idx].set_title(f"r={comp['gamma_r']:.3f} (n={comp['valid_genes']})")
        lim = max(ga[valid].max(), gb[valid].max()) * 1.1
        axes[idx].plot([0, lim], [0, lim], "r--", alpha=0.5)

    fig.suptitle("Cross-Platform Gamma Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "cross_platform_gamma")

    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
