#!/usr/bin/env python
"""Cross-dataset validation summary: consolidate results from all datasets.

Runs the full scPTR pipeline on all 3 datasets and produces:
1. Cross-dataset consistency (pairwise gamma correlation)
2. Half-life validation across all datasets
3. ARE/NMD enrichment across datasets
4. Subsampling robustness across datasets
5. Summary table and comparison figures
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
from run_scifate import load_scifate_data, prepare_for_scptr

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "summary"


def save_fig(fig, name, subdir="figures"):
    if fig is None:
        return
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def run_pipeline(adata, name, groupby=None):
    """Run standard scPTR pipeline on a dataset."""
    print(f"\n--- Running pipeline on {name} ---")
    print(f"  Input: {adata.shape}")

    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)

    scptr.tl.estimate_beta(adata)
    if groupby:
        scptr.tl.estimate_beta(adata, groupby=groupby)
    scptr.tl.estimate_gamma(adata)
    scptr.tl.variance_decomposition(adata)
    scptr.tl.pt_states(adata)
    scptr.tl.pt_velocity(adata)

    gamma = adata.layers["gamma"]
    gamma_med = np.median(gamma, axis=0)
    n_states = adata.obs["pt_state"].nunique()
    print(f"  After pipeline: {adata.shape}")
    print(f"  Gamma: median={np.median(gamma_med):.4f}, max={np.max(gamma):.2f}")
    print(f"  PT states: {n_states}")
    return adata


def main():
    set_figure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # load all datasets
    # =========================================================================
    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    print("\n--- Pancreas ---")
    adata_pan = scptr.datasets.pancreas()
    print(f"  Shape: {adata_pan.shape}")

    print("\n--- Dentate Gyrus ---")
    adata_dg = scptr.datasets.dentate_gyrus()
    print(f"  Shape: {adata_dg.shape}")

    print("\n--- sci-fate A549 ---")
    adata_sf_raw = load_scifate_data()
    adata_sf = prepare_for_scptr(adata_sf_raw)
    print(f"  Shape: {adata_sf.shape}")

    # =========================================================================
    # run pipelines
    # =========================================================================
    print("\n" + "=" * 60)
    print("RUNNING PIPELINES")
    print("=" * 60)

    adata_pan = run_pipeline(adata_pan, "pancreas", groupby="clusters")
    adata_dg = run_pipeline(adata_dg, "dentate_gyrus", groupby="clusters")
    adata_sf = run_pipeline(adata_sf, "scifate")

    datasets = {
        "pancreas": adata_pan,
        "dentate_gyrus": adata_dg,
        "scifate": adata_sf,
    }

    # =========================================================================
    # 1. cross-dataset consistency
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. CROSS-DATASET CONSISTENCY")
    print("=" * 60)

    consistency = scptr.benchmark.cross_dataset_consistency(datasets)
    consistency.to_csv(res_dir / "cross_dataset_consistency.csv", index=False)
    print(consistency.to_string(index=False))

    # =========================================================================
    # 2. half-life validation
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. HALF-LIFE VALIDATION")
    print("=" * 60)

    hl_mouse = scptr.datasets.herzog2017_halflives()
    hl_human = scptr.datasets.schofield2018_halflives()

    hl_results = []
    for name, adata in datasets.items():
        for hl_name, hl_df in [("mouse_Herzog2017", hl_mouse),
                                ("human_Schofield2018", hl_human)]:
            corr = scptr.benchmark.correlate_with_halflives(adata, hl_df)
            hl_results.append({
                "dataset": name,
                "reference": hl_name,
                "spearman_r": corr["spearman_r"],
                "spearman_p": corr["spearman_p"],
                "pearson_r": corr["pearson_r"],
                "n_genes": corr["n_genes"],
            })
            print(f"  {name} vs {hl_name}: Spearman r = {corr['spearman_r']:.4f} "
                  f"(n={corr['n_genes']})")

    hl_df_out = pd.DataFrame(hl_results)
    hl_df_out.to_csv(res_dir / "halflife_correlations.csv", index=False)

    # =========================================================================
    # 3. are/nmd enrichment
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. ARE/NMD ENRICHMENT")
    print("=" * 60)

    enrichment_results = []
    for name, adata in datasets.items():
        are = scptr.benchmark.are_enrichment(adata)
        nmd = scptr.benchmark.nmd_enrichment(adata)
        enrichment_results.append({
            "dataset": name,
            "test": "ARE",
            "n_in_set": are["n_genes_in_set"],
            "U_statistic": are["U_statistic"],
            "p_value": are["p_value"],
        })
        enrichment_results.append({
            "dataset": name,
            "test": "NMD",
            "n_in_set": nmd["n_genes_in_set"],
            "U_statistic": nmd["U_statistic"],
            "p_value": nmd["p_value"],
        })
        print(f"  {name}: ARE p={are['p_value']:.4f} (n={are['n_genes_in_set']}), "
              f"NMD p={nmd['p_value']:.4f} (n={nmd['n_genes_in_set']})")

    enr_df = pd.DataFrame(enrichment_results)
    enr_df.to_csv(res_dir / "enrichment_results.csv", index=False)

    # =========================================================================
    # 4. subsampling robustness
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. SUBSAMPLING ROBUSTNESS")
    print("=" * 60)

    fractions = [0.2, 0.4, 0.6, 0.8, 0.9]
    robustness_results = []
    for name, adata in datasets.items():
        print(f"\n  {name}:")
        robust = scptr.benchmark.subsampling_robustness(
            adata, fractions=fractions, n_repeats=3
        )
        robust["dataset"] = name
        robustness_results.append(robust)
        for frac in fractions:
            sub = robust[robust["fraction"] == frac]
            print(f"    {frac:.0%}: mean Spearman r = {sub['spearman_r'].mean():.4f}")

    robust_all = pd.concat(robustness_results, ignore_index=True)
    robust_all.to_csv(res_dir / "subsampling_robustness.csv", index=False)

    # =========================================================================
    # 5. dataset statistics
    # =========================================================================
    print("\n" + "=" * 60)
    print("5. DATASET STATISTICS")
    print("=" * 60)

    dataset_stats = []
    for name, adata in datasets.items():
        gamma = adata.layers["gamma"]
        gamma_med = np.median(gamma, axis=0)
        n_states = adata.obs["pt_state"].nunique()
        tf_scores = adata.var["tf_score"].values

        dataset_stats.append({
            "dataset": name,
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "beta_median": float(np.median(adata.var["beta"])),
            "gamma_median_of_medians": float(np.median(gamma_med)),
            "gamma_max": float(np.max(gamma)),
            "n_pt_states": n_states,
            "tf_score_median": float(np.median(tf_scores)),
            "tf_score_gt_0.5": int(np.sum(tf_scores > 0.5)),
        })
        print(f"  {name}: {adata.n_obs} cells, {adata.n_vars} genes, "
              f"{n_states} PT states")

    stats_df = pd.DataFrame(dataset_stats)
    stats_df.to_csv(res_dir / "dataset_statistics.csv", index=False)

    # =========================================================================
    # FIGURES
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY FIGURES")
    print("=" * 60)

    # Figure 1: Half-life correlation comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    hl_pivot = hl_df_out.pivot(index="dataset", columns="reference",
                               values="spearman_r")
    x = np.arange(len(hl_pivot))
    width = 0.35
    bars1 = ax.bar(x - width/2, hl_pivot["mouse_Herzog2017"].values,
                   width, label="Mouse (Herzog 2017)", color="steelblue")
    bars2 = ax.bar(x + width/2, hl_pivot["human_Schofield2018"].values,
                   width, label="Human (Schofield 2018)", color="darkorange")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Spearman correlation with half-lives")
    ax.set_title("Half-life Validation Across Datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(hl_pivot.index)
    ax.legend()
    ax.axhline(y=0, color="gray", linewidth=0.5)
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h,
                   f"{h:.3f}", ha="center", va="bottom" if h > 0 else "top",
                   fontsize=8)
    fig.tight_layout()
    save_fig(fig, "halflife_comparison")

    # Figure 2: Robustness curves
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"pancreas": "steelblue", "dentate_gyrus": "darkorange",
              "scifate": "forestgreen"}
    for name in datasets:
        sub = robust_all[robust_all["dataset"] == name]
        means = sub.groupby("fraction")["spearman_r"].mean()
        stds = sub.groupby("fraction")["spearman_r"].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   marker="o", label=name, color=colors.get(name, "gray"),
                   capsize=3)
    ax.set_xlabel("Subsampling fraction")
    ax.set_ylabel("Spearman r with full-data gamma")
    ax.set_title("Subsampling Robustness Across Datasets")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_fig(fig, "robustness_curves")

    # Figure 3: Cross-dataset consistency heatmap
    ds_names = sorted(datasets.keys())
    mat = np.eye(len(ds_names))
    for _, row in consistency.iterrows():
        i = ds_names.index(row["dataset_a"])
        j = ds_names.index(row["dataset_b"])
        mat[i, j] = mat[j, i] = row["spearman_r"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(ds_names)))
    ax.set_yticks(range(len(ds_names)))
    ax.set_xticklabels(ds_names, rotation=45, ha="right")
    ax.set_yticklabels(ds_names)
    for i in range(len(ds_names)):
        for j in range(len(ds_names)):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                   fontsize=10, fontweight="bold" if i != j else "normal")
    plt.colorbar(im, ax=ax, label="Spearman r")
    ax.set_title("Cross-Dataset Gamma Consistency")
    fig.tight_layout()
    save_fig(fig, "cross_dataset_heatmap")

    # Figure 4: Enrichment comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, test in enumerate(["ARE", "NMD"]):
        sub = enr_df[enr_df["test"] == test]
        x = np.arange(len(sub))
        pvals = sub["p_value"].values
        neg_log_p = [-np.log10(max(p, 1e-300)) for p in pvals]
        bars = axes[idx].bar(x, neg_log_p,
                            color=["steelblue", "darkorange", "forestgreen"])
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(sub["dataset"].values, rotation=45, ha="right")
        axes[idx].set_ylabel("-log10(p-value)")
        axes[idx].set_title(f"{test} Enrichment")
        axes[idx].axhline(y=-np.log10(0.05), color="red", linestyle="--",
                         alpha=0.5, label="p=0.05")
        axes[idx].legend()
    fig.suptitle("ARE/NMD Enrichment Across Datasets", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "enrichment_comparison")

    # =========================================================================
    # summary table
    # =========================================================================
    print("\n" + "=" * 60)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 60)

    summary = {}
    for name in datasets:
        s = stats_df[stats_df["dataset"] == name].iloc[0]
        hl_sub = hl_df_out[hl_df_out["dataset"] == name]
        rob_90 = robust_all[(robust_all["dataset"] == name) &
                           (robust_all["fraction"] == 0.9)]
        are_sub = enr_df[(enr_df["dataset"] == name) & (enr_df["test"] == "ARE")]
        nmd_sub = enr_df[(enr_df["dataset"] == name) & (enr_df["test"] == "NMD")]

        summary[name] = {
            "cells": int(s["n_cells"]),
            "genes": int(s["n_genes"]),
            "pt_states": int(s["n_pt_states"]),
            "hl_mouse_r": float(hl_sub[hl_sub["reference"] == "mouse_Herzog2017"]["spearman_r"].values[0]),
            "hl_human_r": float(hl_sub[hl_sub["reference"] == "human_Schofield2018"]["spearman_r"].values[0]),
            "robustness_90pct": float(rob_90["spearman_r"].mean()),
            "are_p": float(are_sub["p_value"].values[0]),
            "nmd_p": float(nmd_sub["p_value"].values[0]),
        }

    with open(res_dir / "comprehensive_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print formatted summary
    print(f"\n{'Dataset':<15} {'Cells':>6} {'Genes':>6} {'States':>6} "
          f"{'HL(m)':>8} {'HL(h)':>8} {'Rob90':>7} {'ARE_p':>8} {'NMD_p':>8}")
    print("-" * 85)
    for name, s in summary.items():
        print(f"{name:<15} {s['cells']:>6} {s['genes']:>6} {s['pt_states']:>6} "
              f"{s['hl_mouse_r']:>8.4f} {s['hl_human_r']:>8.4f} "
              f"{s['robustness_90pct']:>7.4f} "
              f"{s['are_p']:>8.4f} {s['nmd_p']:>8.4f}")

    print(f"\nAll results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
