#!/usr/bin/env python
"""Half-life ablation: compare scPTR gamma vs naive methods for biological accuracy.

For each dataset, compute per-gene median values using four methods:
  1. scPTR gamma (full kinetic model)
  2. Raw u/s ratio (no beta normalization)
  3. Unspliced only (raw unspliced counts)
  4. Expression (spliced counts, negative control)

Then correlate each with published mRNA half-lives. scPTR gamma should produce
the strongest negative correlation because the kinetic model (beta normalization,
smoothing, clipping) produces biologically meaningful degradation rates.
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

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "halflife_ablation"
DATASETS_DIR = Path(__file__).parent.parent / "src" / "scptr" / "datasets" / "data"


def save_fig(fig, name, subdir="figures"):
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def run_pipeline(adata, name):
    print(f"\n--- Pipeline: {name} ---")
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)
    print(f"  Done: {adata.shape}")
    return adata


def halflife_ablation(adata, name):
    """Compare half-life correlations across methods."""
    print(f"\n{'='*60}")
    print(f"HALF-LIFE ABLATION: {name}")
    print(f"{'='*60}")

    gamma = adata.layers["gamma"]
    u_layer = adata.layers.get("Mu", adata.layers.get("unspliced"))
    s_layer = adata.layers.get("Ms", adata.layers.get("spliced"))
    u = u_layer.toarray() if hasattr(u_layer, 'toarray') else np.asarray(u_layer)
    s = s_layer.toarray() if hasattr(s_layer, 'toarray') else np.asarray(s_layer)
    expr = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)

    # Raw u/s ratio
    s_safe = np.where(s > 0.01, s, 1.0)
    raw_ratio = u / s_safe
    raw_ratio[s < 0.01] = 0

    # Per-gene medians for each method
    methods = {
        "scPTR gamma": np.median(gamma, axis=0),
        "Raw u/s ratio": np.median(raw_ratio, axis=0),
        "Unspliced only": np.median(u, axis=0),
        "Expression": np.median(expr, axis=0),
    }

    # Filter to gamma-informative genes
    nonzero_frac = (gamma > 0).mean(axis=0)
    informative = nonzero_frac >= 0.1

    # Load half-life references
    hl_files = [
        ("Mouse (Herzog)", DATASETS_DIR / "herzog2017_halflives.csv"),
        ("Human (Schofield)", DATASETS_DIR / "schofield2018_halflives.csv"),
    ]

    results = []

    for hl_label, hl_path in hl_files:
        if not hl_path.exists():
            continue

        hl_df = pd.read_csv(hl_path)
        hl_df = hl_df[["gene_symbol", "half_life_hours"]].dropna()
        hl_dict = dict(zip(hl_df["gene_symbol"].str.upper(), hl_df["half_life_hours"]))

        print(f"\n  Reference: {hl_label}")

        for method_name, medians in methods.items():
            matched_vals = []
            matched_hl = []
            for i, gene in enumerate(adata.var_names):
                g_upper = gene.upper()
                if g_upper in hl_dict and informative[i]:
                    matched_vals.append(medians[i])
                    matched_hl.append(hl_dict[g_upper])

            if len(matched_vals) < 50:
                continue

            r, p = stats.spearmanr(matched_vals, matched_hl)
            print(f"    {method_name:<20s}: r = {r:.4f}  (p = {p:.2e}, n = {len(matched_vals)})")

            results.append({
                "dataset": name,
                "reference": hl_label,
                "method": method_name,
                "spearman_r": float(r),
                "p_value": float(p),
                "n_genes": len(matched_vals),
            })

    return results


def main():
    set_figure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    all_results = []

    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    adata_pan = scptr.datasets.pancreas()
    adata_pan = run_pipeline(adata_pan, "pancreas")
    all_results.extend(halflife_ablation(adata_pan, "pancreas"))

    adata_dg = scptr.datasets.dentate_gyrus()
    adata_dg = run_pipeline(adata_dg, "dentate_gyrus")
    all_results.extend(halflife_ablation(adata_dg, "dentate_gyrus"))

    # sci-fate
    from run_scifate import load_scifate_data, prepare_for_scptr
    adata_sf_raw = load_scifate_data()
    adata_sf = prepare_for_scptr(adata_sf_raw)
    adata_sf = run_pipeline(adata_sf, "scifate")
    all_results.extend(halflife_ablation(adata_sf, "scifate"))

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(res_dir / "halflife_ablation.csv", index=False)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Use Human (Schofield) as primary reference
    human_results = results_df[results_df["reference"] == "Human (Schofield)"]
    if len(human_results) > 0:
        pivot = human_results.pivot_table(
            index="method", columns="dataset", values="spearman_r", aggfunc="first"
        )
        print("\n  Spearman r with Human (Schofield) half-lives:")
        print(pivot.to_string())

    # Figure: grouped bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (hl_label, hl_sub) in enumerate(results_df.groupby("reference")):
        ax = axes[ax_idx]
        datasets = hl_sub["dataset"].unique()
        methods_order = ["scPTR gamma", "Raw u/s ratio", "Unspliced only", "Expression"]
        colors = ["steelblue", "orange", "lightblue", "gray"]
        x = np.arange(len(datasets))
        width = 0.18

        for mi, (method, color) in enumerate(zip(methods_order, colors)):
            vals = []
            for ds in datasets:
                sub = hl_sub[(hl_sub["method"] == method) & (hl_sub["dataset"] == ds)]
                vals.append(sub["spearman_r"].values[0] if len(sub) > 0 else 0)
            bars = ax.bar(x + mi * width, vals, width, label=method, color=color,
                         edgecolor="black", linewidth=0.5)
            for bi, v in enumerate(vals):
                ax.text(x[bi] + mi * width, v - 0.02, f"{v:.3f}",
                       ha="center", va="top", fontsize=7, rotation=90)

        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(datasets, fontsize=9)
        ax.set_ylabel("Spearman r with half-life")
        ax.set_title(f"Half-life Correlation: {hl_label}")
        ax.legend(fontsize=7, loc="lower left")
        ax.axhline(y=0, color="black", linewidth=0.5)

    fig.suptitle("Ablation: Which Method Best Predicts mRNA Half-Life?", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "halflife_ablation")

    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
