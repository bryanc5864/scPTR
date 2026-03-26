#!/usr/bin/env python
"""Comprehensive benchmark: DeepPTR vs analytical scPTR on synthetic + real data.

Runs:
1. Synthetic recovery: gamma correlation, CI coverage, latent CCA
2. Real datasets (pancreas, dentate gyrus): analytical vs DeepPTR
   - Half-life correlation (mouse + human references)
   - ARE/NMD enrichment
   - Subsampling robustness
   - Analytical vs DeepPTR gamma agreement
3. sci-fate metabolic labeling: ground-truth validation for both methods

All results saved to output/deep_benchmark/.
"""

from __future__ import annotations

# Thread control — MUST be set before any numpy/torch import
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

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

import torch
torch.set_num_threads(4)

sys.path.insert(0, str(Path(__file__).parent))
from _common import set_figure_style

import scptr

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "deep_benchmark"


def save_fig(fig, name, subdir="figures"):
    if fig is None:
        print(f"  [WARNING] {name}: plot returned None, skipping save")
        return
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def ensure_dirs():
    for sub in ("figures", "results"):
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)


# ============================================================================
# 1. SYNTHETIC RECOVERY
# ============================================================================

def run_synthetic_benchmark():
    """End-to-end DeepPTR on synthetic kinetic data with known ground truth."""
    from scptr.deep.synthetic import (
        generate_kinetic_data,
        gamma_recovery,
        ci_coverage,
        latent_recovery,
    )

    print("=" * 60)
    print("1. SYNTHETIC RECOVERY BENCHMARK")
    print("=" * 60)

    adata, truth = generate_kinetic_data(
        n_cells=1500, n_genes=100, n_cell_types=5,
        dispersion=10.0, sparsity=0.3, seed=0,
    )
    print(f"  Generated: {adata.shape}, {truth['gamma'].shape}")

    # Fit DeepPTR (compact model for CPU)
    torch.set_num_threads(4)
    t0 = time.time()
    model, history = scptr.deep.fit_deepptr(
        adata,
        d_T=8, d_PT=8, d_hidden=48, n_enc_layers=2,
        batch_size=256, max_epochs=150, kl_warmup_epochs=20,
        patience=15, n_posterior_samples=20,
        device="cpu", seed=0, verbose=True,
    )
    elapsed = time.time() - t0
    print(f"  Training: {len(history.train_loss)} epochs in {elapsed:.1f}s")

    # Evaluate
    gamma_r = gamma_recovery(truth["gamma"], adata.layers["gamma"], per_gene=True)
    gamma_r_global = gamma_recovery(truth["gamma"], adata.layers["gamma"], per_gene=False)
    ci_cov = ci_coverage(truth["gamma"], adata.layers["gamma"], adata.layers["gamma_var"])
    z_T_r = latent_recovery(truth["z_T"], adata.obsm["X_z_T"])
    z_PT_r = latent_recovery(truth["z_PT"], adata.obsm["X_z_PT"])

    results = {
        "gamma_recovery_per_gene": gamma_r,
        "gamma_recovery_global": gamma_r_global,
        "ci_coverage_95": ci_cov,
        "latent_recovery_T": z_T_r,
        "latent_recovery_PT": z_PT_r,
        "n_epochs": len(history.train_loss),
        "final_train_loss": history.train_loss[-1],
        "final_val_loss": history.val_loss[-1],
        "training_time_s": elapsed,
    }

    print(f"\n  Gamma recovery (per-gene median Spearman r): {gamma_r:.4f}")
    print(f"  Gamma recovery (global Spearman r):           {gamma_r_global:.4f}")
    print(f"  95% CI coverage:                              {ci_cov:.4f}")
    print(f"  Latent recovery z_T (mean CCA):               {z_T_r:.4f}")
    print(f"  Latent recovery z_PT (mean CCA):              {z_PT_r:.4f}")

    with open(OUTPUT_DIR / "results" / "synthetic_recovery.json", "w") as f:
        json.dump(results, f, indent=2)

    # Training curve plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history.train_loss) + 1)
    axes[0].plot(epochs, history.train_loss, label="train")
    axes[0].plot(epochs, history.val_loss, label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total Loss")
    axes[0].legend()

    axes[1].plot(epochs, history.train_recon, label="train")
    axes[1].plot(epochs, history.val_recon, label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Reconstruction Loss")
    axes[1].set_title("Reconstruction")
    axes[1].legend()

    axes[2].plot(epochs, history.kl_weight, "k-")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("KL Weight")
    axes[2].set_title("KL Annealing")

    fig.suptitle(f"DeepPTR Training (synthetic, gamma r={gamma_r:.3f})", y=1.02)
    fig.tight_layout()
    save_fig(fig, "synthetic_training_curves")

    return results


# ============================================================================
# 2. REAL DATA: PANCREAS + DENTATE GYRUS
# ============================================================================

def preprocess_for_analytical(adata, cluster_key="clusters"):
    """Standard scPTR preprocessing + analytical gamma."""
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)
    return adata


def select_top_genes(adata, n_top=500):
    """Select top genes by unspliced signal for DeepPTR (reduces dim for CPU speed).

    Uses total unspliced counts × fraction of cells expressing as the ranking.
    Returns a view of adata with only the selected genes.
    """
    from scipy.sparse import issparse

    u = adata.layers["unspliced"]
    if issparse(u):
        u = np.asarray(u.todense())
    u = np.asarray(u, dtype=np.float32)

    # Rank by: total counts * fraction nonzero (rewards both signal and breadth)
    total_counts = u.sum(axis=0)
    frac_nonzero = (u > 0).mean(axis=0)
    score = total_counts * frac_nonzero

    top_idx = np.argsort(score)[::-1][:n_top]
    top_idx = np.sort(top_idx)  # keep original order

    gene_names = adata.var_names[top_idx]
    print(f"  Selected top {len(gene_names)} genes for DeepPTR (from {adata.n_vars})")
    adata_sub = adata[:, gene_names].copy()

    # Ensure dense layers for efficient DataLoader conversion
    from scipy.sparse import issparse as _issparse
    for key in ("spliced", "unspliced"):
        if key in adata_sub.layers and _issparse(adata_sub.layers[key]):
            adata_sub.layers[key] = np.asarray(adata_sub.layers[key].todense())
    return adata_sub


def run_halflife_comparison(adata, adata_deep, dataset_name):
    """Compare half-life correlations: analytical vs DeepPTR."""
    hl_mouse = scptr.datasets.herzog2017_halflives()
    hl_human = scptr.datasets.schofield2018_halflives()

    results = {}
    for ref_name, hl_df in [("mouse_herzog", hl_mouse), ("human_schofield", hl_human)]:
        # Analytical
        corr_an = scptr.benchmark.correlate_with_halflives(adata, hl_df)
        # DeepPTR
        corr_dp = scptr.benchmark.correlate_with_halflives(adata_deep, hl_df)

        results[ref_name] = {
            "analytical": {
                "spearman_r": corr_an["spearman_r"],
                "pearson_r": corr_an["pearson_r"],
                "n_genes": corr_an["n_genes"],
            },
            "deepptr": {
                "spearman_r": corr_dp["spearman_r"],
                "pearson_r": corr_dp["pearson_r"],
                "n_genes": corr_dp["n_genes"],
            },
        }
        print(f"  {ref_name}:")
        print(f"    Analytical: Spearman r = {corr_an['spearman_r']:.4f} (n={corr_an['n_genes']})")
        print(f"    DeepPTR:    Spearman r = {corr_dp['spearman_r']:.4f} (n={corr_dp['n_genes']})")

    return results


def run_enrichment_comparison(adata, adata_deep, dataset_name):
    """Compare ARE/NMD enrichment: analytical vs DeepPTR."""
    results = {}
    for test_name, test_fn in [("ARE", scptr.benchmark.are_enrichment),
                                ("NMD", scptr.benchmark.nmd_enrichment)]:
        res_an = test_fn(adata)
        res_dp = test_fn(adata_deep)

        results[test_name] = {
            "analytical": {
                "U_statistic": float(res_an.get("U_statistic", np.nan)),
                "p_value": float(res_an.get("p_value", np.nan)),
                "n_genes_in_set": int(res_an.get("n_genes_in_set", 0)),
            },
            "deepptr": {
                "U_statistic": float(res_dp.get("U_statistic", np.nan)),
                "p_value": float(res_dp.get("p_value", np.nan)),
                "n_genes_in_set": int(res_dp.get("n_genes_in_set", 0)),
            },
        }
        p_an = res_an.get("p_value", np.nan)
        p_dp = res_dp.get("p_value", np.nan)
        print(f"  {test_name}: analytical p={p_an:.2e}, DeepPTR p={p_dp:.2e}")

    return results


def run_gamma_agreement(adata, adata_deep, dataset_name):
    """Correlate per-gene median gamma: analytical vs DeepPTR on shared genes."""
    gamma_an_s = pd.Series(
        np.median(adata.layers["gamma"], axis=0), index=adata.var_names
    )
    gamma_dp_s = pd.Series(
        np.median(adata_deep.layers["gamma"], axis=0), index=adata_deep.var_names
    )

    # Match on shared genes
    shared = gamma_an_s.index.intersection(gamma_dp_s.index)
    g_an = gamma_an_s[shared].values.astype(float)
    g_dp = gamma_dp_s[shared].values.astype(float)

    mask = (g_an > 0) & (g_dp > 0) & np.isfinite(g_an) & np.isfinite(g_dp)
    g_an = g_an[mask]
    g_dp = g_dp[mask]

    if len(g_an) < 3:
        print(f"  Analytical vs DeepPTR gamma: too few shared genes ({len(g_an)})")
        return {"spearman_r": np.nan, "pearson_r": np.nan, "n_genes": 0}

    sp_r, sp_p = stats.spearmanr(g_an, g_dp)
    pe_r, pe_p = stats.pearsonr(np.log1p(g_an), np.log1p(g_dp))

    result = {
        "spearman_r": float(sp_r),
        "spearman_p": float(sp_p),
        "pearson_r": float(pe_r),
        "pearson_p": float(pe_p),
        "n_genes": int(mask.sum()),
    }
    print(f"  Analytical vs DeepPTR gamma: Spearman r = {sp_r:.4f} (n={mask.sum()})")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(g_an, g_dp, alpha=0.15, s=8, c="steelblue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Analytical median gamma")
    ax.set_ylabel("DeepPTR median gamma")
    ax.set_title(f"{dataset_name}: Analytical vs DeepPTR (r={sp_r:.3f}, n={mask.sum()})")
    lims = [min(g_an.min(), g_dp.min()), max(g_an.max(), g_dp.max())]
    ax.plot(lims, lims, "k--", alpha=0.3, lw=1)
    save_fig(fig, f"{dataset_name}_analytical_vs_deepptr")

    return result


def run_real_dataset(name, adata_loader, cluster_key="clusters"):
    """Full benchmark for one real dataset."""
    print(f"\n{'=' * 60}")
    print(f"2. REAL DATA: {name.upper()}")
    print("=" * 60)

    # Load and preprocess
    print(f"\n--- Loading {name} ---")
    adata = adata_loader()
    print(f"  Shape: {adata.shape}")

    print(f"\n--- Preprocessing (analytical) ---")
    preprocess_for_analytical(adata, cluster_key=cluster_key)
    gamma_an = adata.layers["gamma"]
    gamma_med_an = np.median(gamma_an, axis=0)
    print(f"  Analytical gamma: median of medians = {np.median(gamma_med_an):.4f}")

    # DeepPTR: preprocess, select top genes, then fit
    print(f"\n--- Running DeepPTR ---")
    adata_deep = adata_loader()
    scptr.pp.filter_genes(adata_deep)
    scptr.pp.normalize_layers(adata_deep)
    scptr.pp.neighbors(adata_deep, n_neighbors=30)
    scptr.pp.smooth_layers(adata_deep)
    scptr.tl.estimate_beta(adata_deep)
    # Select top genes to keep training tractable on CPU
    adata_deep = select_top_genes(adata_deep, n_top=300)

    torch.set_num_threads(4)  # Reset after TF/scanpy imports
    t0 = time.time()
    model, history = scptr.deep.fit_deepptr(
        adata_deep,
        d_T=8, d_PT=8, d_hidden=48, n_enc_layers=2,
        batch_size=512, max_epochs=100, kl_warmup_epochs=20,
        patience=15, n_posterior_samples=15,
        device="cpu", seed=0, verbose=True,
    )
    elapsed = time.time() - t0
    n_epochs = len(history.train_loss)
    print(f"  DeepPTR: {n_epochs} epochs in {elapsed:.1f}s")

    gamma_dp = adata_deep.layers["gamma"]
    gamma_med_dp = np.median(gamma_dp, axis=0)
    print(f"  DeepPTR gamma: median of medians = {np.median(gamma_med_dp):.4f}")

    # --- Benchmarks ---
    all_results = {
        "dataset": name,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "deepptr_epochs": n_epochs,
        "deepptr_time_s": elapsed,
        "deepptr_final_val_loss": history.val_loss[-1],
    }

    # Half-life correlations
    print(f"\n--- Half-life correlations ---")
    hl_results = run_halflife_comparison(adata, adata_deep, name)
    all_results["halflife"] = hl_results

    # ARE/NMD enrichment
    print(f"\n--- ARE/NMD enrichment ---")
    try:
        enrich_results = run_enrichment_comparison(adata, adata_deep, name)
        all_results["enrichment"] = enrich_results
    except Exception as e:
        print(f"  Enrichment failed: {e}")
        all_results["enrichment"] = {"error": str(e)}

    # Analytical vs DeepPTR agreement
    print(f"\n--- Analytical vs DeepPTR agreement ---")
    agree = run_gamma_agreement(adata, adata_deep, name)
    all_results["gamma_agreement"] = agree

    # Subsampling robustness (DeepPTR only — analytical already known)
    print(f"\n--- Subsampling robustness (analytical) ---")
    try:
        rob_an = scptr.benchmark.subsampling_robustness(
            adata, fractions=[0.5, 0.8], n_repeats=2
        )
        print(f"  Analytical: median r @ 30% = {rob_an[rob_an['fraction']==0.3]['spearman_r'].median():.4f}")
        all_results["robustness_analytical"] = rob_an.to_dict(orient="records")
    except Exception as e:
        print(f"  Robustness failed: {e}")

    # Training curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, n_epochs + 1)
    axes[0].plot(epochs, history.train_loss, label="train")
    axes[0].plot(epochs, history.val_loss, label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{name}: Training Loss")
    axes[0].legend()

    axes[1].plot(epochs, history.train_recon, label="train recon")
    axes[1].plot(epochs, history.train_kl, label="train KL")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss Component")
    axes[1].set_title(f"{name}: Loss Components")
    axes[1].legend()
    fig.tight_layout()
    save_fig(fig, f"{name}_training_curves")

    # Uncertainty visualization
    gamma_var = adata_deep.layers["gamma_var"]
    mean_var = np.mean(gamma_var, axis=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(gamma_med_dp, mean_var, alpha=0.2, s=8, c="steelblue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Posterior mean gamma (median over cells)")
    ax.set_ylabel("Posterior variance (mean over cells)")
    ax.set_title(f"{name}: DeepPTR Uncertainty")
    save_fig(fig, f"{name}_uncertainty")

    # Save
    with open(OUTPUT_DIR / "results" / f"{name}_benchmark.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


# ============================================================================
# 3. SCI-FATE GROUND TRUTH VALIDATION
# ============================================================================

def run_scifate_benchmark():
    """Compare analytical vs DeepPTR on sci-fate metabolic labeling data."""
    import gzip
    from scipy.io import mmread
    from scipy.sparse import csc_matrix

    print(f"\n{'=' * 60}")
    print("3. SCI-FATE METABOLIC LABELING VALIDATION")
    print("=" * 60)

    CACHE_DIR = Path.home() / ".cache" / "scptr" / "scifate"
    if not CACHE_DIR.exists():
        print("  [SKIP] sci-fate data not cached. Run analyses/run_scifate.py first.")
        return None

    # Load raw data
    print("  Loading sci-fate data...")
    cell_ann = pd.read_csv(CACHE_DIR / "GSM3770930_A549_cell_annotate.txt.gz", compression="gzip")
    gene_ann = pd.read_csv(CACHE_DIR / "GSM3770930_A549_gene_annotate.txt.gz", compression="gzip")

    with gzip.open(CACHE_DIR / "GSM3770930_A549_gene_count.txt.gz", "rb") as f:
        total_mat = csc_matrix(mmread(f)).T
    with gzip.open(CACHE_DIR / "GSM3770930_A549_gene_count_newly_synthesised.txt.gz", "rb") as f:
        new_mat = csc_matrix(mmread(f)).T

    import anndata as ad
    adata_raw = ad.AnnData(
        X=total_mat,
        obs=cell_ann.set_index("sample"),
        var=gene_ann.set_index("gene_id"),
    )
    adata_raw.layers["new"] = new_mat
    adata_raw.var_names_make_unique()
    adata_raw.var["gene_id_full"] = adata_raw.var_names.tolist()
    adata_raw.var_names = adata_raw.var["gene_short_name"].values
    adata_raw.var_names_make_unique()
    print(f"  Shape: {adata_raw.shape}")

    # Ground truth
    total = np.asarray(adata_raw.X.toarray() if hasattr(adata_raw.X, "toarray") else adata_raw.X)
    new = np.asarray(adata_raw.layers["new"].toarray() if hasattr(adata_raw.layers["new"], "toarray") else adata_raw.layers["new"])
    old = total - new
    mean_new = new.mean(axis=0)
    mean_old = old.mean(axis=0)
    mean_total = total.mean(axis=0)
    reliable = (mean_total >= 0.5) & (mean_old > 0.1)
    gt_ratio = np.full(adata_raw.n_vars, np.nan)
    gt_ratio[reliable] = mean_new[reliable] / mean_old[reliable]
    print(f"  Ground truth: {reliable.sum()} reliable genes")

    # Prepare for scPTR (unspliced=new, spliced=old)
    keep = mean_total >= 0.5
    if "gene_type" in adata_raw.var.columns:
        is_pc = adata_raw.var["gene_type"] == "protein_coding"
        keep = keep & is_pc.values

    def make_scptr_adata():
        a = ad.AnnData(
            X=total[:, keep].astype(np.float32),
            obs=adata_raw.obs.copy(),
            var=adata_raw.var.iloc[keep].copy(),
        )
        a.layers["unspliced"] = new[:, keep].astype(np.float32)
        a.layers["spliced"] = old[:, keep].astype(np.float32)
        return a

    # --- Analytical ---
    print("\n--- Analytical pipeline ---")
    adata_an = make_scptr_adata()
    scptr.pp.filter_genes(adata_an, min_unspliced_counts=1, min_unspliced_cells=1)
    scptr.pp.normalize_layers(adata_an)
    scptr.pp.neighbors(adata_an, n_neighbors=30)
    scptr.pp.smooth_layers(adata_an)
    scptr.tl.estimate_beta(adata_an)
    scptr.tl.estimate_gamma(adata_an)
    gamma_med_an = np.median(adata_an.layers["gamma"], axis=0)
    print(f"  Analytical: {adata_an.shape}, median gamma = {np.median(gamma_med_an):.4f}")

    # --- DeepPTR ---
    print("\n--- DeepPTR ---")
    adata_dp = make_scptr_adata()
    scptr.pp.filter_genes(adata_dp, min_unspliced_counts=1, min_unspliced_cells=1)
    scptr.pp.normalize_layers(adata_dp)
    scptr.pp.neighbors(adata_dp, n_neighbors=30)
    scptr.pp.smooth_layers(adata_dp)
    scptr.tl.estimate_beta(adata_dp)
    adata_dp = select_top_genes(adata_dp, n_top=500)

    t0 = time.time()
    model, history = scptr.deep.fit_deepptr(
        adata_dp,
        d_T=8, d_PT=8, d_hidden=48, n_enc_layers=2,
        batch_size=512, max_epochs=100, kl_warmup_epochs=20,
        patience=15, n_posterior_samples=15,
        device="cpu", seed=0, verbose=True,
    )
    elapsed = time.time() - t0
    gamma_med_dp = np.median(adata_dp.layers["gamma"], axis=0)
    print(f"  DeepPTR: {len(history.train_loss)} epochs in {elapsed:.1f}s")

    # Correlate both with ground truth
    gt_s_an = pd.Series(gt_ratio, index=adata_raw.var_names)
    gamma_s_an = pd.Series(gamma_med_an, index=adata_an.var_names)
    gamma_s_dp = pd.Series(gamma_med_dp, index=adata_dp.var_names)

    shared_an = gamma_s_an.index.intersection(gt_s_an.dropna().index)
    shared_dp = gamma_s_dp.index.intersection(gt_s_an.dropna().index)

    def correlate(gamma_s, gt_s, shared):
        g = gamma_s[shared].values.astype(float)
        t = gt_s[shared].values.astype(float)
        mask = np.isfinite(g) & np.isfinite(t) & (g > 0) & (t > 0)
        if mask.sum() < 3:
            return {"spearman_r": np.nan, "n_genes": 0}
        sp_r, sp_p = stats.spearmanr(g[mask], t[mask])
        return {"spearman_r": float(sp_r), "spearman_p": float(sp_p), "n_genes": int(mask.sum())}

    corr_an = correlate(gamma_s_an, gt_s_an, shared_an)
    corr_dp = correlate(gamma_s_dp, gt_s_an, shared_dp)

    print(f"\n--- Ground truth correlation (new/old ratio) ---")
    print(f"  Analytical: Spearman r = {corr_an['spearman_r']:.4f} (n={corr_an['n_genes']})")
    print(f"  DeepPTR:    Spearman r = {corr_dp['spearman_r']:.4f} (n={corr_dp['n_genes']})")

    # Half-life correlation
    print(f"\n--- Half-life correlations ---")
    hl_human = scptr.datasets.schofield2018_halflives()
    corr_hl_an = scptr.benchmark.correlate_with_halflives(adata_an, hl_human)
    corr_hl_dp = scptr.benchmark.correlate_with_halflives(adata_dp, hl_human)
    print(f"  Analytical: Spearman r = {corr_hl_an['spearman_r']:.4f} (n={corr_hl_an['n_genes']})")
    print(f"  DeepPTR:    Spearman r = {corr_hl_dp['spearman_r']:.4f} (n={corr_hl_dp['n_genes']})")

    # Agreement
    shared_both = gamma_s_an.index.intersection(gamma_s_dp.index)
    g_an = gamma_s_an[shared_both].values
    g_dp = gamma_s_dp[shared_both].values
    mask_both = (g_an > 0) & (g_dp > 0) & np.isfinite(g_an) & np.isfinite(g_dp)
    if mask_both.sum() >= 3:
        agree_r, _ = stats.spearmanr(g_an[mask_both], g_dp[mask_both])
        print(f"\n  Analytical vs DeepPTR: Spearman r = {agree_r:.4f} (n={mask_both.sum()})")
    else:
        agree_r = np.nan

    results = {
        "dataset": "scifate",
        "n_cells": int(adata_an.n_obs),
        "n_genes_analytical": int(adata_an.n_vars),
        "n_genes_deep": int(adata_dp.n_vars),
        "ground_truth_corr": {
            "analytical": corr_an,
            "deepptr": corr_dp,
        },
        "halflife_human": {
            "analytical": {"spearman_r": corr_hl_an["spearman_r"], "n_genes": corr_hl_an["n_genes"]},
            "deepptr": {"spearman_r": corr_hl_dp["spearman_r"], "n_genes": corr_hl_dp["n_genes"]},
        },
        "gamma_agreement": {"spearman_r": float(agree_r), "n_genes": int(mask_both.sum())},
        "deepptr_epochs": len(history.train_loss),
        "deepptr_time_s": elapsed,
    }

    with open(OUTPUT_DIR / "results" / "scifate_benchmark.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Scatter: analytical vs DeepPTR vs ground truth
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: Analytical vs ground truth
    g = gamma_s_an[shared_an].values.astype(float)
    t = gt_s_an[shared_an].values.astype(float)
    m = np.isfinite(g) & np.isfinite(t) & (g > 0) & (t > 0)
    axes[0].scatter(t[m], g[m], alpha=0.1, s=5, c="steelblue")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Ground truth (new/old ratio)")
    axes[0].set_ylabel("Analytical gamma")
    axes[0].set_title(f"Analytical (r={corr_an['spearman_r']:.3f})")

    # Panel 2: DeepPTR vs ground truth
    g = gamma_s_dp[shared_dp].values.astype(float)
    t = gt_s_an[shared_dp].values.astype(float)
    m = np.isfinite(g) & np.isfinite(t) & (g > 0) & (t > 0)
    axes[1].scatter(t[m], g[m], alpha=0.1, s=5, c="darkorange")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Ground truth (new/old ratio)")
    axes[1].set_ylabel("DeepPTR gamma")
    axes[1].set_title(f"DeepPTR (r={corr_dp['spearman_r']:.3f})")

    # Panel 3: Analytical vs DeepPTR
    if mask_both.sum() >= 3:
        axes[2].scatter(g_an[mask_both], g_dp[mask_both], alpha=0.1, s=5, c="seagreen")
        axes[2].set_xscale("log")
        axes[2].set_yscale("log")
        lims = [min(g_an[mask_both].min(), g_dp[mask_both].min()),
                max(g_an[mask_both].max(), g_dp[mask_both].max())]
        axes[2].plot(lims, lims, "k--", alpha=0.3, lw=1)
    axes[2].set_xlabel("Analytical gamma")
    axes[2].set_ylabel("DeepPTR gamma")
    axes[2].set_title(f"Agreement (r={agree_r:.3f})")

    fig.suptitle("sci-fate: Analytical vs DeepPTR", y=1.02)
    fig.tight_layout()
    save_fig(fig, "scifate_comparison")

    return results


# ============================================================================
# 4. SUMMARY TABLE
# ============================================================================

def print_summary(synth, pancreas, dg, scifate):
    """Print final comparison table."""
    print(f"\n{'=' * 70}")
    print("SUMMARY: Analytical vs DeepPTR")
    print("=" * 70)

    # Header
    print(f"\n{'Metric':<40} {'Analytical':>12} {'DeepPTR':>12}")
    print("-" * 65)

    if synth:
        print(f"\n  SYNTHETIC RECOVERY")
        print(f"  {'Gamma recovery (per-gene r)':<38} {'N/A':>12} {synth['gamma_recovery_per_gene']:>12.4f}")
        print(f"  {'95% CI coverage':<38} {'N/A':>12} {synth['ci_coverage_95']:>12.4f}")
        print(f"  {'Latent recovery z_T':<38} {'N/A':>12} {synth['latent_recovery_T']:>12.4f}")
        print(f"  {'Latent recovery z_PT':<38} {'N/A':>12} {synth['latent_recovery_PT']:>12.4f}")

    for name, res in [("PANCREAS", pancreas), ("DENTATE GYRUS", dg)]:
        if res is None:
            continue
        print(f"\n  {name}")
        for ref in ("mouse_herzog", "human_schofield"):
            if ref in res.get("halflife", {}):
                hl = res["halflife"][ref]
                an_r = hl["analytical"]["spearman_r"]
                dp_r = hl["deepptr"]["spearman_r"]
                print(f"  {'Half-life ' + ref:<38} {an_r:>12.4f} {dp_r:>12.4f}")
        if "gamma_agreement" in res:
            print(f"  {'Gamma agreement (Spearman r)':<38} {'---':>12} {res['gamma_agreement']['spearman_r']:>12.4f}")

    if scifate:
        print(f"\n  SCI-FATE")
        gt = scifate.get("ground_truth_corr", {})
        if "analytical" in gt and "deepptr" in gt:
            an_r = gt["analytical"]["spearman_r"]
            dp_r = gt["deepptr"]["spearman_r"]
            print(f"  {'Ground truth (new/old ratio)':<38} {an_r:>12.4f} {dp_r:>12.4f}")
        hl = scifate.get("halflife_human", {})
        if "analytical" in hl and "deepptr" in hl:
            an_r = hl["analytical"]["spearman_r"]
            dp_r = hl["deepptr"]["spearman_r"]
            print(f"  {'Half-life (human Schofield)':<38} {an_r:>12.4f} {dp_r:>12.4f}")

    print()


def main():
    set_figure_style()
    ensure_dirs()

    # 1. Synthetic
    synth_results = run_synthetic_benchmark()

    # 2. Pancreas
    pancreas_results = run_real_dataset(
        "pancreas", scptr.datasets.pancreas, cluster_key="clusters"
    )

    # 3. Dentate Gyrus
    dg_results = run_real_dataset(
        "dentate_gyrus", scptr.datasets.dentate_gyrus, cluster_key="clusters"
    )

    # 4. sci-fate (if data available)
    scifate_results = run_scifate_benchmark()

    # 5. Summary
    print_summary(synth_results, pancreas_results, dg_results, scifate_results)

    # Save combined results
    combined = {
        "synthetic": synth_results,
        "pancreas": pancreas_results,
        "dentate_gyrus": dg_results,
        "scifate": scifate_results,
    }
    with open(OUTPUT_DIR / "results" / "combined_benchmark.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
