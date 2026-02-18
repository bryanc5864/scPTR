#!/usr/bin/env python
"""Validate remaining package features: dynamic mode, groupby beta, scalability.

These are features that were implemented but never validated on real data.
"""

from __future__ import annotations

import json
import sys
import time
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

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "remaining_validation"


def save_fig(fig, name, subdir="figures"):
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =========================================================================
# 1. Per-cell-type beta estimation (groupby)
# =========================================================================
def validate_groupby_beta(adata, name, cluster_col="clusters"):
    """Validate per-cell-type beta estimation vs global beta."""
    print(f"\n{'='*60}")
    print(f"GROUPBY BETA VALIDATION ({name})")
    print(f"{'='*60}")

    # First: run global beta (standard)
    import copy
    adata_global = copy.deepcopy(adata)
    scptr.pp.filter_genes(adata_global)
    scptr.pp.normalize_layers(adata_global)
    scptr.pp.neighbors(adata_global, n_neighbors=30)
    scptr.pp.smooth_layers(adata_global)
    scptr.tl.estimate_beta(adata_global)
    global_beta = adata_global.var["beta"].values.copy()

    # Second: run groupby beta
    adata_group = copy.deepcopy(adata)
    scptr.pp.filter_genes(adata_group)
    scptr.pp.normalize_layers(adata_group)
    scptr.pp.neighbors(adata_group, n_neighbors=30)
    scptr.pp.smooth_layers(adata_group)

    if cluster_col not in adata_group.obs.columns:
        # Run clustering first
        import scanpy as sc
        sc.tl.leiden(adata_group, key_added=cluster_col)

    n_types = adata_group.obs[cluster_col].nunique()
    print(f"  Cell types: {n_types}")
    print(f"  Type sizes: {adata_group.obs[cluster_col].value_counts().to_dict()}")

    scptr.tl.estimate_beta(adata_group, groupby=cluster_col)
    consensus_beta = adata_group.var["beta"].values.copy()

    # Compare global vs consensus
    valid = (global_beta > 0) & (consensus_beta > 0)
    r, p = stats.spearmanr(global_beta[valid], consensus_beta[valid])
    print(f"\n  Global vs consensus beta:")
    print(f"    Spearman r = {r:.4f}, p = {p:.2e}")
    print(f"    Valid genes: {valid.sum()}")

    # Check per-group variation
    if "beta_groups" in adata_group.varm:
        beta_groups = adata_group.varm["beta_groups"]
        print(f"\n  Per-group beta variation:")
        print(f"    Groups: {list(beta_groups.columns)}")

        # CV of beta across groups
        group_vals = beta_groups.values.astype(float)
        group_means = np.nanmean(group_vals, axis=1)
        group_stds = np.nanstd(group_vals, axis=1)
        cvs = group_stds / (group_means + 1e-8)
        valid_cv = group_means > 0
        print(f"    Median CV across groups: {np.median(cvs[valid_cv]):.4f}")
        print(f"    Genes with CV > 0.5 (high variation): "
              f"{(cvs[valid_cv] > 0.5).sum()}/{valid_cv.sum()}")

        # Do different cell types have different beta distributions?
        print(f"\n  Per-cell-type beta medians:")
        for col in beta_groups.columns:
            med = np.nanmedian(beta_groups[col].values.astype(float))
            print(f"    {col}: median beta = {med:.4f}")

    # Now compare gamma with groupby beta vs global beta
    scptr.tl.estimate_gamma(adata_global)
    scptr.tl.estimate_gamma(adata_group)

    gamma_global = np.median(adata_global.layers["gamma"], axis=0)
    gamma_group = np.median(adata_group.layers["gamma"], axis=0)
    valid_g = (gamma_global > 0) & (gamma_group > 0)
    r_g, p_g = stats.spearmanr(gamma_global[valid_g], gamma_group[valid_g])
    print(f"\n  Gamma comparison (global vs groupby beta):")
    print(f"    Spearman r = {r_g:.4f}, p = {p_g:.2e}")
    print(f"    Valid genes: {valid_g.sum()}")

    # Half-life correlation comparison
    halflife_dir = Path(__file__).parent.parent / "src" / "scptr" / "datasets" / "data"
    for hl_name, hl_file in [("Herzog 2017", "herzog2017_halflives.csv"),
                              ("Schofield 2018", "schofield2018_halflives.csv")]:
        hl_path = halflife_dir / hl_file
        if not hl_path.exists():
            continue
        hl = pd.read_csv(hl_path)
        gene_map = {g.upper(): i for i, g in enumerate(adata_global.var_names)}
        hl_gamma_global, hl_gamma_group, hl_vals = [], [], []
        for _, row in hl.iterrows():
            raw_g = row["gene_symbol"] if "gene_symbol" in hl.columns else row.iloc[0]
            if pd.isna(raw_g) or str(raw_g).strip() == "":
                continue
            g = str(raw_g).upper()
            if g in gene_map:
                gi = gene_map[g]
                gg = gamma_global[gi]
                ggrp = gamma_group[gi]
                if gg > 0 and ggrp > 0:
                    hl_gamma_global.append(gg)
                    hl_gamma_group.append(ggrp)
                    hl_val = row["half_life_hours"] if "half_life_hours" in hl.columns else row.iloc[1]
                    hl_vals.append(float(hl_val))

        if len(hl_vals) >= 20:
            r_hl_g, _ = stats.spearmanr(hl_gamma_global, hl_vals)
            r_hl_grp, _ = stats.spearmanr(hl_gamma_group, hl_vals)
            print(f"\n  Half-life correlation ({hl_name}):")
            print(f"    Global beta: r = {r_hl_g:.4f}")
            print(f"    Groupby beta: r = {r_hl_grp:.4f}")
            print(f"    {'Groupby BETTER' if abs(r_hl_grp) > abs(r_hl_g) else 'Global BETTER'}")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Global vs consensus beta
    axes[0].scatter(global_beta[valid], consensus_beta[valid], s=2, alpha=0.3)
    axes[0].set_xlabel("Global beta")
    axes[0].set_ylabel("Consensus beta (groupby)")
    axes[0].set_title(f"Beta: Global vs Per-Cell-Type\nr={r:.3f}")
    lim = max(global_beta[valid].max(), consensus_beta[valid].max()) * 1.1
    axes[0].plot([0, lim], [0, lim], "r--", alpha=0.5)

    # Panel 2: Gamma comparison
    if valid_g.sum() > 0:
        axes[1].scatter(gamma_global[valid_g], gamma_group[valid_g], s=2, alpha=0.3)
        axes[1].set_xlabel("Gamma (global beta)")
        axes[1].set_ylabel("Gamma (groupby beta)")
        axes[1].set_title(f"Gamma: Global vs Groupby\nr={r_g:.3f}")
        lim_g = max(gamma_global[valid_g].max(), gamma_group[valid_g].max()) * 1.1
        axes[1].plot([0, lim_g], [0, lim_g], "r--", alpha=0.5)

    # Panel 3: Beta CV histogram
    if "beta_groups" in adata_group.varm:
        axes[2].hist(cvs[valid_cv], bins=50, color="steelblue", edgecolor="black",
                     linewidth=0.5)
        axes[2].axvline(x=np.median(cvs[valid_cv]), color="red", linestyle="--",
                        label=f"median={np.median(cvs[valid_cv]):.2f}")
        axes[2].set_xlabel("CV of beta across cell types")
        axes[2].set_ylabel("Number of genes")
        axes[2].set_title("Beta Variation Across Cell Types")
        axes[2].legend()

    fig.suptitle(f"Per-Cell-Type Beta Validation: {name}", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, f"groupby_beta_{name}")

    return {
        "global_vs_consensus_r": float(r),
        "gamma_r": float(r_g),
        "n_cell_types": int(n_types),
        "median_cv": float(np.median(cvs[valid_cv])) if "beta_groups" in adata_group.varm else None,
    }


# =========================================================================
# 2. Dynamic mode validation
# =========================================================================
def validate_dynamic_mode(adata, name):
    """Compare steady-state vs dynamic gamma estimation.

    Dynamic mode uses the full ODE: gamma = (beta*u - ds/dt) / s
    This requires a velocity layer (ds/dt estimate).
    """
    print(f"\n{'='*60}")
    print(f"DYNAMIC MODE VALIDATION ({name})")
    print(f"{'='*60}")

    import copy
    adata_ss = copy.deepcopy(adata)
    scptr.pp.filter_genes(adata_ss)
    scptr.pp.normalize_layers(adata_ss)
    scptr.pp.neighbors(adata_ss, n_neighbors=30)
    scptr.pp.smooth_layers(adata_ss)
    scptr.tl.estimate_beta(adata_ss)

    # Steady-state gamma
    scptr.tl.estimate_gamma(adata_ss, mode="steady_state")
    gamma_ss = adata_ss.layers["gamma"].copy()

    # For dynamic mode, we need ds/dt. Estimate it as the difference
    # between a cell's spliced count and its neighbors' mean.
    # This is a simple approximation of the time derivative.
    s_smooth = adata_ss.layers["Ms"].copy()
    import scanpy as sc

    # Compute diffusion pseudotime for temporal ordering
    sc.tl.diffmap(adata_ss)

    # Approximate ds/dt using the spliced expression trend along the manifold
    # Use the velocity estimation approach: ds/dt ≈ beta*u - gamma_ss*s
    # (rearranging the ODE at non-steady-state)
    # Actually, let's use a simpler approach: finite differences along kNN graph
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=30)

    # Use PCA space for neighbors
    if "X_pca" in adata_ss.obsm:
        nn.fit(adata_ss.obsm["X_pca"][:, :30])
    else:
        sc.tl.pca(adata_ss)
        nn.fit(adata_ss.obsm["X_pca"][:, :30])

    _, indices = nn.kneighbors()

    # ds/dt ≈ mean(s_neighbors) - s_cell (displacement on manifold)
    n_cells, n_genes = s_smooth.shape
    ds_dt = np.zeros_like(s_smooth)
    for i in range(n_cells):
        nbr_mean = s_smooth[indices[i]].mean(axis=0)
        ds_dt[i] = nbr_mean - s_smooth[i]

    # Store as a layer
    adata_ss.layers["ds_dt"] = ds_dt.astype(np.float32)

    # Dynamic gamma
    adata_dyn = copy.deepcopy(adata_ss)
    adata_dyn.layers["gamma"] = gamma_ss  # will be overwritten
    scptr.tl.estimate_gamma(adata_dyn, mode="dynamic", velocity_layer="ds_dt")
    gamma_dyn = adata_dyn.layers["gamma"].copy()

    # Compare
    med_ss = np.median(gamma_ss, axis=0)
    med_dyn = np.median(gamma_dyn, axis=0)
    valid = (med_ss > 0) & (med_dyn > 0)
    r, p = stats.spearmanr(med_ss[valid], med_dyn[valid])

    print(f"  Steady-state gamma genes > 0: {(med_ss > 0).sum()}")
    print(f"  Dynamic gamma genes > 0: {(med_dyn > 0).sum()}")
    print(f"  Correlation (shared): r = {r:.4f}, p = {p:.2e}, n = {valid.sum()}")

    # Genes that differ most between modes
    ratio = np.zeros_like(med_ss)
    ratio[valid] = med_dyn[valid] / med_ss[valid]
    most_different = np.argsort(np.abs(np.log(ratio[valid] + 1e-8)))[::-1][:10]
    print(f"\n  Most different genes (dynamic/steady-state ratio):")
    valid_genes = adata_ss.var_names[valid]
    for idx in most_different:
        g = valid_genes[idx]
        r_val = ratio[valid][idx]
        print(f"    {g}: dynamic/ss = {r_val:.2f}")

    # Half-life correlation comparison
    halflife_dir = Path(__file__).parent.parent / "src" / "scptr" / "datasets" / "data"
    for hl_name, hl_file in [("Herzog 2017", "herzog2017_halflives.csv"),
                              ("Schofield 2018", "schofield2018_halflives.csv")]:
        hl_path = halflife_dir / hl_file
        if not hl_path.exists():
            continue
        hl = pd.read_csv(hl_path)
        gene_map = {g.upper(): i for i, g in enumerate(adata_ss.var_names)}
        hl_ss, hl_dyn, hl_vals = [], [], []
        for _, row in hl.iterrows():
            raw_g = row["gene_symbol"] if "gene_symbol" in hl.columns else row.iloc[0]
            if pd.isna(raw_g) or str(raw_g).strip() == "":
                continue
            g = str(raw_g).upper()
            if g in gene_map:
                gi = gene_map[g]
                if med_ss[gi] > 0 and med_dyn[gi] > 0:
                    hl_ss.append(med_ss[gi])
                    hl_dyn.append(med_dyn[gi])
                    hl_val = row["half_life_hours"] if "half_life_hours" in hl.columns else row.iloc[1]
                    hl_vals.append(float(hl_val))

        if len(hl_vals) >= 20:
            r_ss, _ = stats.spearmanr(hl_ss, hl_vals)
            r_dyn, _ = stats.spearmanr(hl_dyn, hl_vals)
            print(f"\n  Half-life correlation ({hl_name}):")
            print(f"    Steady-state: r = {r_ss:.4f}")
            print(f"    Dynamic: r = {r_dyn:.4f}")
            print(f"    {'Dynamic BETTER' if abs(r_dyn) > abs(r_ss) else 'Steady-state BETTER'}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(med_ss[valid], med_dyn[valid], s=2, alpha=0.3, color="steelblue")
    axes[0].set_xlabel("Median gamma (steady-state)")
    axes[0].set_ylabel("Median gamma (dynamic)")
    axes[0].set_title(f"Steady-State vs Dynamic Gamma ({name})\nr={r:.3f}")
    lim = max(med_ss[valid].max(), med_dyn[valid].max()) * 1.1
    axes[0].plot([0, lim], [0, lim], "r--", alpha=0.5)

    # Panel 2: ratio distribution
    log_ratio = np.log2(ratio[valid] + 1e-8)
    log_ratio = log_ratio[np.isfinite(log_ratio)]
    axes[1].hist(log_ratio, bins=50, color="steelblue", edgecolor="black", linewidth=0.5)
    axes[1].axvline(x=0, color="red", linestyle="--", label="Equal")
    axes[1].set_xlabel("log2(dynamic / steady-state)")
    axes[1].set_ylabel("Number of genes")
    axes[1].set_title("Dynamic vs Steady-State Ratio")
    axes[1].legend()

    fig.tight_layout()
    save_fig(fig, f"dynamic_mode_{name}")

    return {
        "ss_vs_dynamic_r": float(r),
        "n_genes_both": int(valid.sum()),
    }


# =========================================================================
# 3. Scalability profiling
# =========================================================================
def profile_scalability(adata, name):
    """Profile scPTR runtime and memory on increasing cell counts."""
    print(f"\n{'='*60}")
    print(f"SCALABILITY PROFILING ({name})")
    print(f"{'='*60}")

    import copy
    import tracemalloc

    # Prepare full dataset
    adata_full = copy.deepcopy(adata)
    scptr.pp.filter_genes(adata_full)
    scptr.pp.normalize_layers(adata_full)

    n_total = adata_full.n_obs
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    results = []

    for frac in fractions:
        n_cells = int(n_total * frac)
        if n_cells < 100:
            continue

        # Subsample
        rng = np.random.RandomState(42)
        idx = rng.choice(n_total, n_cells, replace=False)
        adata_sub = adata_full[idx].copy()

        print(f"\n  {frac:.0%} ({n_cells} cells, {adata_sub.n_vars} genes):")

        tracemalloc.start()
        t0 = time.time()

        scptr.pp.neighbors(adata_sub, n_neighbors=min(30, n_cells - 1))
        scptr.pp.smooth_layers(adata_sub)
        scptr.tl.estimate_beta(adata_sub)
        scptr.tl.estimate_gamma(adata_sub)

        t1 = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed = t1 - t0
        peak_mb = peak / 1024 / 1024

        print(f"    Time: {elapsed:.1f}s")
        print(f"    Peak memory: {peak_mb:.0f} MB")

        results.append({
            "fraction": frac,
            "n_cells": n_cells,
            "n_genes": adata_sub.n_vars,
            "time_seconds": elapsed,
            "peak_memory_mb": peak_mb,
        })

    # Extrapolate to 100K cells
    if len(results) >= 3:
        times = [r["time_seconds"] for r in results]
        cells = [r["n_cells"] for r in results]
        # Linear fit in log space for scaling behavior
        log_cells = np.log(cells)
        log_times = np.log(times)
        slope, intercept = np.polyfit(log_cells, log_times, 1)
        estimated_100k = np.exp(intercept) * (100000 ** slope)
        print(f"\n  Scaling exponent: {slope:.2f} (1.0=linear, 2.0=quadratic)")
        print(f"  Estimated time for 100K cells: {estimated_100k:.0f}s ({estimated_100k/60:.1f} min)")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cells = [r["n_cells"] for r in results]
    times = [r["time_seconds"] for r in results]
    mems = [r["peak_memory_mb"] for r in results]

    axes[0].plot(cells, times, "o-", color="steelblue", linewidth=2, markersize=8)
    axes[0].set_xlabel("Number of cells")
    axes[0].set_ylabel("Runtime (seconds)")
    axes[0].set_title(f"scPTR Runtime Scaling ({name})")

    axes[1].plot(cells, mems, "o-", color="#E53935", linewidth=2, markersize=8)
    axes[1].set_xlabel("Number of cells")
    axes[1].set_ylabel("Peak memory (MB)")
    axes[1].set_title(f"scPTR Memory Scaling ({name})")

    fig.tight_layout()
    save_fig(fig, f"scalability_{name}")

    return results


# =========================================================================
# MAIN
# =========================================================================
def main():
    set_figure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    adata_pan = scptr.datasets.pancreas()
    adata_dg = scptr.datasets.dentate_gyrus()

    # 1. Groupby beta validation
    print("\n" + "#" * 60)
    print("# GROUPBY BETA VALIDATION")
    print("#" * 60)

    groupby_results = {}
    groupby_results["pancreas"] = validate_groupby_beta(adata_pan, "pancreas")
    groupby_results["dentate_gyrus"] = validate_groupby_beta(adata_dg, "dentate_gyrus")

    with open(res_dir / "groupby_beta.json", "w") as f:
        json.dump(groupby_results, f, indent=2)

    # 2. Dynamic mode validation
    print("\n" + "#" * 60)
    print("# DYNAMIC MODE VALIDATION")
    print("#" * 60)

    dynamic_results = {}
    dynamic_results["pancreas"] = validate_dynamic_mode(adata_pan, "pancreas")
    dynamic_results["dentate_gyrus"] = validate_dynamic_mode(adata_dg, "dentate_gyrus")

    with open(res_dir / "dynamic_mode.json", "w") as f:
        json.dump(dynamic_results, f, indent=2)

    # 3. Scalability profiling
    print("\n" + "#" * 60)
    print("# SCALABILITY PROFILING")
    print("#" * 60)

    scale_results = {}
    scale_results["pancreas"] = profile_scalability(adata_pan, "pancreas")
    scale_results["dentate_gyrus"] = profile_scalability(adata_dg, "dentate_gyrus")

    with open(res_dir / "scalability.json", "w") as f:
        json.dump(scale_results, f, indent=2)

    print(f"\n{'='*60}")
    print("REMAINING VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
