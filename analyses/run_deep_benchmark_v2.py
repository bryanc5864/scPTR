#!/usr/bin/env python
"""Expanded DeepPTR benchmark v2: deeper analysis beyond basic half-life correlation.

Adds to v1:
1. Enrichment on full gene set (map DeepPTR gamma onto analytical genes)
2. Per-cell-type gamma patterns (cell-type-specific agreement)
3. Uncertainty calibration on real data (variance vs prediction error)
4. DeepPTR subsampling robustness (retrain on subsets)
5. Cross-dataset consistency (DeepPTR vs analytical)
6. Latent space structure (z_T/z_PT UMAP colored by cell type)
7. Gene ranking comparison (top differentially-degraded genes)

All results saved to output/deep_benchmark_v2/.
"""

from __future__ import annotations

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

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "deep_benchmark_v2"


def save_fig(fig, name, subdir="figures"):
    if fig is None:
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


def select_top_genes(adata, n_top=300):
    """Select top genes by unspliced signal for DeepPTR."""
    from scipy.sparse import issparse
    u = adata.layers["unspliced"]
    if issparse(u):
        u = np.asarray(u.todense())
    u = np.asarray(u, dtype=np.float32)
    score = u.sum(axis=0) * (u > 0).mean(axis=0)
    top_idx = np.sort(np.argsort(score)[::-1][:n_top])
    adata_sub = adata[:, adata.var_names[top_idx]].copy()
    from scipy.sparse import issparse as _iss
    for key in ("spliced", "unspliced"):
        if key in adata_sub.layers and _iss(adata_sub.layers[key]):
            adata_sub.layers[key] = np.asarray(adata_sub.layers[key].todense())
    print(f"  Selected top {n_top} genes (from {adata.n_vars})")
    return adata_sub


def run_analytical_pipeline(adata):
    """Full analytical scPTR pipeline."""
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)
    return adata


def fit_deep(adata_deep):
    """Fit DeepPTR on preprocessed adata (with beta already estimated)."""
    torch.set_num_threads(4)
    model, history = scptr.deep.fit_deepptr(
        adata_deep,
        d_T=8, d_PT=8, d_hidden=48, n_enc_layers=2,
        batch_size=512, max_epochs=100, kl_warmup_epochs=20,
        patience=15, n_posterior_samples=15,
        device="cpu", seed=0, verbose=True,
    )
    return model, history


# ============================================================================
# 1. enrichment with full gene mapping
# ============================================================================

def run_enrichment_mapped(adata_an, adata_deep, dataset_name):
    """Map DeepPTR gamma onto full analytical gene set, then run enrichment.

    DeepPTR only models top-N genes. For enrichment, we create a hybrid:
    use DeepPTR gamma where available, analytical gamma elsewhere.
    Also test DeepPTR-only genes separately.
    """
    print(f"\n--- Enrichment (mapped) ---")
    import anndata as ad

    gamma_an_med = np.median(adata_an.layers["gamma"], axis=0)
    gamma_dp_med = np.median(adata_deep.layers["gamma"], axis=0)
    dp_genes = set(adata_deep.var_names)

    # Hybrid: prefer DeepPTR where available
    gamma_hybrid = gamma_an_med.copy()
    for i, g in enumerate(adata_an.var_names):
        if g in dp_genes:
            j = list(adata_deep.var_names).index(g)
            gamma_hybrid[i] = gamma_dp_med[j]

    # Create hybrid adata for enrichment
    adata_hybrid = adata_an.copy()
    adata_hybrid.layers["gamma"] = np.tile(gamma_hybrid, (adata_an.n_obs, 1))

    results = {}
    for test_name, test_fn in [("ARE", scptr.benchmark.are_enrichment),
                                ("NMD", scptr.benchmark.nmd_enrichment)]:
        res_an = test_fn(adata_an)
        res_hybrid = test_fn(adata_hybrid)

        results[test_name] = {
            "analytical": {
                "p_value": float(res_an.get("p_value", np.nan)),
                "n_genes_in_set": int(res_an.get("n_genes_in_set", 0)),
                "median_gamma_in": float(res_an.get("median_gamma_in_set", np.nan)),
                "median_gamma_bg": float(res_an.get("median_gamma_background", np.nan)),
            },
            "hybrid_deepptr": {
                "p_value": float(res_hybrid.get("p_value", np.nan)),
                "n_genes_in_set": int(res_hybrid.get("n_genes_in_set", 0)),
                "median_gamma_in": float(res_hybrid.get("median_gamma_in_set", np.nan)),
                "median_gamma_bg": float(res_hybrid.get("median_gamma_background", np.nan)),
            },
        }
        p_an = res_an.get("p_value", np.nan)
        p_hy = res_hybrid.get("p_value", np.nan)
        print(f"  {test_name}: analytical p={p_an:.2e}, hybrid p={p_hy:.2e}")

    return results


# ============================================================================
# 2. per-cell-type gamma agreement
# ============================================================================

def run_celltype_agreement(adata_an, adata_deep, dataset_name, cluster_key="clusters"):
    """Compare per-cell-type median gamma between analytical and DeepPTR."""
    print(f"\n--- Per-cell-type gamma agreement ---")

    if cluster_key not in adata_an.obs.columns:
        print(f"  [SKIP] No '{cluster_key}' column")
        return None

    shared_genes = adata_an.var_names.intersection(adata_deep.var_names)
    if len(shared_genes) < 10:
        print(f"  [SKIP] Too few shared genes ({len(shared_genes)})")
        return None

    an_idx = [list(adata_an.var_names).index(g) for g in shared_genes]
    dp_idx = [list(adata_deep.var_names).index(g) for g in shared_genes]

    cell_types = adata_an.obs[cluster_key].unique()
    records = []

    for ct in sorted(cell_types):
        mask_an = adata_an.obs[cluster_key] == ct
        mask_dp = adata_deep.obs[cluster_key] == ct

        if mask_an.sum() < 5 or mask_dp.sum() < 5:
            continue

        gamma_an_ct = np.median(adata_an.layers["gamma"][mask_an][:, an_idx], axis=0)
        gamma_dp_ct = np.median(adata_deep.layers["gamma"][mask_dp][:, dp_idx], axis=0)

        valid = (gamma_an_ct > 0) & (gamma_dp_ct > 0) & np.isfinite(gamma_an_ct) & np.isfinite(gamma_dp_ct)
        if valid.sum() < 5:
            continue

        sp_r, _ = stats.spearmanr(gamma_an_ct[valid], gamma_dp_ct[valid])
        records.append({
            "cell_type": str(ct),
            "n_cells_an": int(mask_an.sum()),
            "n_cells_dp": int(mask_dp.sum()),
            "n_genes": int(valid.sum()),
            "spearman_r": float(sp_r),
        })
        print(f"  {ct}: r={sp_r:.4f} (n_genes={valid.sum()}, n_cells={mask_an.sum()})")

    if not records:
        return None

    df = pd.DataFrame(records)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(df["cell_type"], df["spearman_r"], color="steelblue", alpha=0.7)
    ax.set_xlabel("Spearman r (analytical vs DeepPTR)")
    ax.set_title(f"{dataset_name}: Per-cell-type gamma agreement")
    ax.axvline(x=df["spearman_r"].median(), color="red", ls="--", alpha=0.5,
               label=f"median={df['spearman_r'].median():.3f}")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, f"{dataset_name}_celltype_agreement")

    return records


# ============================================================================
# 3. uncertainty calibration on real data
# ============================================================================

def run_uncertainty_analysis(adata_an, adata_deep, dataset_name):
    """Evaluate DeepPTR uncertainty: does high variance predict high error?"""
    print(f"\n--- Uncertainty calibration ---")

    shared_genes = adata_an.var_names.intersection(adata_deep.var_names)
    if len(shared_genes) < 10:
        print(f"  [SKIP] Too few shared genes")
        return None

    an_idx = [list(adata_an.var_names).index(g) for g in shared_genes]
    dp_idx = [list(adata_deep.var_names).index(g) for g in shared_genes]

    # Per-gene: compare variance with squared error vs analytical
    gamma_an = np.median(adata_an.layers["gamma"][:, an_idx], axis=0)
    gamma_dp = np.median(adata_deep.layers["gamma"][:, dp_idx], axis=0)
    gamma_var = np.mean(adata_deep.layers["gamma_var"][:, dp_idx], axis=0)

    # Prediction error (using analytical as reference)
    valid = (gamma_an > 0) & (gamma_dp > 0) & np.isfinite(gamma_an) & np.isfinite(gamma_dp)
    if valid.sum() < 10:
        print(f"  [SKIP] Too few valid genes")
        return None

    error = np.abs(gamma_dp[valid] - gamma_an[valid])
    var = gamma_var[valid]

    # Does high posterior variance correlate with high error?
    sp_r, sp_p = stats.spearmanr(var, error)
    print(f"  Variance-error correlation: Spearman r = {sp_r:.4f} (p={sp_p:.2e})")

    # Binned calibration: split genes into variance quintiles
    n_bins = 5
    var_ranks = np.argsort(np.argsort(var))
    bin_size = len(var) // n_bins
    bin_errors = []
    bin_vars = []
    for b in range(n_bins):
        mask = (var_ranks >= b * bin_size) & (var_ranks < (b + 1) * bin_size)
        if b == n_bins - 1:
            mask = var_ranks >= b * bin_size
        bin_errors.append(np.median(error[mask]))
        bin_vars.append(np.median(var[mask]))

    result = {
        "var_error_spearman_r": float(sp_r),
        "var_error_spearman_p": float(sp_p),
        "n_genes": int(valid.sum()),
        "bin_median_var": [float(v) for v in bin_vars],
        "bin_median_error": [float(e) for e in bin_errors],
    }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: variance vs error
    axes[0].scatter(var, error, alpha=0.2, s=8, c="steelblue")
    axes[0].set_xlabel("Mean posterior variance")
    axes[0].set_ylabel("|DeepPTR - Analytical| error")
    axes[0].set_title(f"Variance vs Error (r={sp_r:.3f})")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")

    # Binned calibration
    axes[1].bar(range(n_bins), bin_errors, color="steelblue", alpha=0.7)
    axes[1].set_xlabel("Posterior variance quintile (low → high)")
    axes[1].set_ylabel("Median absolute error")
    axes[1].set_title(f"{dataset_name}: Calibration")
    axes[1].set_xticks(range(n_bins))
    axes[1].set_xticklabels([f"Q{i+1}" for i in range(n_bins)])

    fig.tight_layout()
    save_fig(fig, f"{dataset_name}_uncertainty_calibration")

    return result


# ============================================================================
# 4. DEEPPTR SUBSAMPLING ROBUSTNESS
# ============================================================================

def run_deep_subsampling(adata_loader, dataset_name, fractions=(0.5, 0.8)):
    """Test DeepPTR robustness by retraining on subsampled cells."""
    print(f"\n--- DeepPTR subsampling robustness ---")

    # Full model
    adata_full = adata_loader()
    scptr.pp.filter_genes(adata_full)
    scptr.pp.normalize_layers(adata_full)
    scptr.pp.neighbors(adata_full, n_neighbors=30)
    scptr.pp.smooth_layers(adata_full)
    scptr.tl.estimate_beta(adata_full)
    adata_full = select_top_genes(adata_full, n_top=300)

    torch.set_num_threads(4)
    model_full, _ = scptr.deep.fit_deepptr(
        adata_full,
        d_T=8, d_PT=8, d_hidden=48, n_enc_layers=2,
        batch_size=512, max_epochs=100, kl_warmup_epochs=20,
        patience=15, n_posterior_samples=10,
        device="cpu", seed=0, verbose=False,
    )
    gamma_full = np.median(adata_full.layers["gamma"], axis=0)

    records = []
    rng = np.random.RandomState(42)

    for frac in fractions:
        n_sub = max(int(adata_full.n_obs * frac), 50)
        idx = rng.choice(adata_full.n_obs, size=n_sub, replace=False)

        adata_sub = adata_full[idx].copy()
        # Ensure dense
        from scipy.sparse import issparse
        for key in ("spliced", "unspliced"):
            if key in adata_sub.layers and issparse(adata_sub.layers[key]):
                adata_sub.layers[key] = np.asarray(adata_sub.layers[key].todense())

        torch.set_num_threads(4)
        _, _ = scptr.deep.fit_deepptr(
            adata_sub,
            d_T=8, d_PT=8, d_hidden=48, n_enc_layers=2,
            batch_size=512, max_epochs=100, kl_warmup_epochs=20,
            patience=15, n_posterior_samples=10,
            device="cpu", seed=0, verbose=False,
        )
        gamma_sub = np.median(adata_sub.layers["gamma"], axis=0)

        valid = np.isfinite(gamma_full) & np.isfinite(gamma_sub)
        sp_r, _ = stats.spearmanr(gamma_full[valid], gamma_sub[valid])

        records.append({
            "fraction": frac,
            "n_cells": n_sub,
            "spearman_r": float(sp_r),
        })
        print(f"  {frac*100:.0f}%: r={sp_r:.4f} (n_cells={n_sub})")

    return records


# ============================================================================
# 5. latent space visualization
# ============================================================================

def run_latent_analysis(adata_deep, dataset_name, cluster_key="clusters"):
    """Visualize DeepPTR latent spaces with UMAP."""
    print(f"\n--- Latent space visualization ---")
    import scanpy as sc

    if "X_z_T" not in adata_deep.obsm or "X_z_PT" not in adata_deep.obsm:
        print("  [SKIP] No latent embeddings found")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    has_ct = cluster_key in adata_deep.obs.columns

    for ax_idx, (key, title) in enumerate([
        ("X_z_T", "z_T (transcription)"),
        ("X_z_PT", "z_PT (post-transcription)"),
    ]):
        z = adata_deep.obsm[key]
        # Quick PCA+UMAP for visualization
        from sklearn.decomposition import PCA
        if z.shape[1] > 2:
            pca = PCA(n_components=2)
            z_2d = pca.fit_transform(z)
        else:
            z_2d = z

        if has_ct:
            categories = adata_deep.obs[cluster_key].astype("category")
            codes = categories.cat.codes.values
            cmap = plt.cm.get_cmap("tab20", len(categories.cat.categories))
            scatter = axes[ax_idx].scatter(z_2d[:, 0], z_2d[:, 1], c=codes,
                                            cmap=cmap, alpha=0.3, s=3)
        else:
            axes[ax_idx].scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.3, s=3, c="steelblue")
        axes[ax_idx].set_title(title)
        axes[ax_idx].set_xlabel("PC1")
        axes[ax_idx].set_ylabel("PC2")

    # Third panel: gamma PCA
    gamma = adata_deep.layers["gamma"]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    g_2d = pca.fit_transform(gamma)
    if has_ct:
        categories = adata_deep.obs[cluster_key].astype("category")
        codes = categories.cat.codes.values
        cmap = plt.cm.get_cmap("tab20", len(categories.cat.categories))
        axes[2].scatter(g_2d[:, 0], g_2d[:, 1], c=codes, cmap=cmap, alpha=0.3, s=3)
    else:
        axes[2].scatter(g_2d[:, 0], g_2d[:, 1], alpha=0.3, s=3, c="steelblue")
    axes[2].set_title("gamma (DeepPTR)")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")

    if has_ct:
        cats = categories.cat.categories.tolist()
        if len(cats) <= 15:
            handles = [plt.Line2D([0], [0], marker="o", color="w",
                                   markerfacecolor=cmap(i), markersize=6, label=c)
                       for i, c in enumerate(cats)]
            fig.legend(handles=handles, loc="center right", fontsize=7,
                      bbox_to_anchor=(1.15, 0.5))

    fig.suptitle(f"{dataset_name}: DeepPTR Latent Spaces", y=1.02)
    fig.tight_layout()
    save_fig(fig, f"{dataset_name}_latent_spaces")

    # Quantify: silhouette score of cell types in latent space
    if has_ct and len(categories.cat.categories) >= 2:
        from sklearn.metrics import silhouette_score
        codes = categories.cat.codes.values
        sil_T = silhouette_score(adata_deep.obsm["X_z_T"], codes, sample_size=min(2000, len(codes)))
        sil_PT = silhouette_score(adata_deep.obsm["X_z_PT"], codes, sample_size=min(2000, len(codes)))
        sil_gamma = silhouette_score(gamma, codes, sample_size=min(2000, len(codes)))
        print(f"  Silhouette: z_T={sil_T:.4f}, z_PT={sil_PT:.4f}, gamma={sil_gamma:.4f}")
        return {"silhouette_z_T": sil_T, "silhouette_z_PT": sil_PT, "silhouette_gamma": sil_gamma}

    return None


# ============================================================================
# 6. gene ranking comparison
# ============================================================================

def run_gene_ranking(adata_an, adata_deep, dataset_name, n_top=50):
    """Compare top differentially-degraded genes between methods."""
    print(f"\n--- Gene ranking comparison (top {n_top}) ---")

    shared_genes = adata_an.var_names.intersection(adata_deep.var_names)
    if len(shared_genes) < 20:
        print("  [SKIP] Too few shared genes")
        return None

    gamma_an = pd.Series(
        np.median(adata_an.layers["gamma"], axis=0), index=adata_an.var_names
    )
    gamma_dp = pd.Series(
        np.median(adata_deep.layers["gamma"], axis=0), index=adata_deep.var_names
    )

    # Variance of gamma across cells (identifies genes with heterogeneous degradation)
    gamma_var_an = pd.Series(
        np.var(adata_an.layers["gamma"], axis=0), index=adata_an.var_names
    )
    gamma_var_dp = pd.Series(
        np.var(adata_deep.layers["gamma"], axis=0), index=adata_deep.var_names
    )

    # Top genes by median gamma (shared)
    top_an = gamma_an[shared_genes].nlargest(n_top).index.tolist()
    top_dp = gamma_dp[shared_genes].nlargest(n_top).index.tolist()
    overlap_median = len(set(top_an) & set(top_dp))

    # Top genes by gamma variance (shared)
    top_var_an = gamma_var_an[shared_genes].nlargest(n_top).index.tolist()
    top_var_dp = gamma_var_dp[shared_genes].nlargest(n_top).index.tolist()
    overlap_var = len(set(top_var_an) & set(top_var_dp))

    # Rank correlation on shared genes
    ranks_an = gamma_an[shared_genes].rank(ascending=False)
    ranks_dp = gamma_dp[shared_genes].rank(ascending=False)
    rank_corr, _ = stats.spearmanr(ranks_an.values, ranks_dp.values)

    result = {
        "n_shared_genes": len(shared_genes),
        "top_median_overlap": overlap_median,
        "top_median_overlap_frac": overlap_median / n_top,
        "top_var_overlap": overlap_var,
        "top_var_overlap_frac": overlap_var / n_top,
        "rank_correlation": float(rank_corr),
    }
    print(f"  Top-{n_top} median gamma overlap: {overlap_median}/{n_top} ({overlap_median/n_top*100:.0f}%)")
    print(f"  Top-{n_top} var gamma overlap:    {overlap_var}/{n_top} ({overlap_var/n_top*100:.0f}%)")
    print(f"  Rank correlation (shared genes):   {rank_corr:.4f}")

    return result


# ============================================================================
# main: run on each dataset
# ============================================================================

def run_dataset(name, adata_loader, cluster_key="clusters"):
    """Run all expanded benchmarks on one dataset."""
    print(f"\n{'=' * 60}")
    print(f"DATASET: {name.upper()}")
    print("=" * 60)

    # --- Analytical ---
    print(f"\n--- Analytical pipeline ---")
    adata_an = adata_loader()
    run_analytical_pipeline(adata_an)
    print(f"  Analytical: {adata_an.shape}")

    # --- DeepPTR ---
    print(f"\n--- DeepPTR ---")
    adata_deep = adata_loader()
    scptr.pp.filter_genes(adata_deep)
    scptr.pp.normalize_layers(adata_deep)
    scptr.pp.neighbors(adata_deep, n_neighbors=30)
    scptr.pp.smooth_layers(adata_deep)
    scptr.tl.estimate_beta(adata_deep)
    adata_deep = select_top_genes(adata_deep, n_top=300)

    t0 = time.time()
    model, history = fit_deep(adata_deep)
    elapsed = time.time() - t0
    print(f"  DeepPTR: {len(history.train_loss)} epochs in {elapsed:.1f}s")

    all_results = {"dataset": name, "n_epochs": len(history.train_loss), "time_s": elapsed}

    # 1. Enrichment
    enrich = run_enrichment_mapped(adata_an, adata_deep, name)
    all_results["enrichment"] = enrich

    # 2. Per-cell-type
    ct_results = run_celltype_agreement(adata_an, adata_deep, name, cluster_key)
    all_results["celltype_agreement"] = ct_results

    # 3. Uncertainty
    unc_results = run_uncertainty_analysis(adata_an, adata_deep, name)
    all_results["uncertainty"] = unc_results

    # 4. Latent space
    lat_results = run_latent_analysis(adata_deep, name, cluster_key)
    all_results["latent_structure"] = lat_results

    # 5. Gene ranking
    rank_results = run_gene_ranking(adata_an, adata_deep, name)
    all_results["gene_ranking"] = rank_results

    # Save
    with open(OUTPUT_DIR / "results" / f"{name}_v2.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


def run_cross_dataset_consistency(datasets):
    """Compare cross-dataset consistency for analytical vs DeepPTR."""
    print(f"\n{'=' * 60}")
    print("CROSS-DATASET CONSISTENCY")
    print("=" * 60)

    # Build analytical and deep adatas
    an_dict = {}
    dp_dict = {}

    for name, loader, cluster_key in datasets:
        print(f"\n  Processing {name}...")
        adata_an = loader()
        run_analytical_pipeline(adata_an)
        an_dict[name] = adata_an

        adata_dp = loader()
        scptr.pp.filter_genes(adata_dp)
        scptr.pp.normalize_layers(adata_dp)
        scptr.pp.neighbors(adata_dp, n_neighbors=30)
        scptr.pp.smooth_layers(adata_dp)
        scptr.tl.estimate_beta(adata_dp)
        adata_dp = select_top_genes(adata_dp, n_top=300)
        torch.set_num_threads(4)
        scptr.deep.fit_deepptr(
            adata_dp,
            d_T=8, d_PT=8, d_hidden=48, n_enc_layers=2,
            batch_size=512, max_epochs=100, kl_warmup_epochs=20,
            patience=15, n_posterior_samples=10,
            device="cpu", seed=0, verbose=False,
        )
        dp_dict[name] = adata_dp

    print(f"\n--- Analytical cross-dataset ---")
    cons_an = scptr.benchmark.cross_dataset_consistency(an_dict)
    print(cons_an.to_string(index=False))

    print(f"\n--- DeepPTR cross-dataset ---")
    cons_dp = scptr.benchmark.cross_dataset_consistency(dp_dict)
    print(cons_dp.to_string(index=False))

    result = {
        "analytical": cons_an.to_dict(orient="records"),
        "deepptr": cons_dp.to_dict(orient="records"),
    }

    with open(OUTPUT_DIR / "results" / "cross_dataset_consistency.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def run_subsampling_all(datasets):
    """Run DeepPTR subsampling robustness on each dataset."""
    print(f"\n{'=' * 60}")
    print("DEEPPTR SUBSAMPLING ROBUSTNESS")
    print("=" * 60)

    all_results = {}
    for name, loader, _ in datasets:
        print(f"\n  {name}:")
        records = run_deep_subsampling(loader, name, fractions=(0.5, 0.8))
        all_results[name] = records

    with open(OUTPUT_DIR / "results" / "subsampling_robustness.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


def print_summary(results, cross_ds, subsampling):
    """Print final summary table."""
    print(f"\n{'=' * 70}")
    print("EXPANDED BENCHMARK SUMMARY")
    print("=" * 70)

    for name, res in results.items():
        print(f"\n  {name.upper()}")

        # Enrichment
        enrich = res.get("enrichment", {})
        for test in ("ARE", "NMD"):
            if test in enrich:
                p_an = enrich[test].get("analytical", {}).get("p_value", np.nan)
                p_hy = enrich[test].get("hybrid_deepptr", {}).get("p_value", np.nan)
                print(f"    {test} enrichment: analytical p={p_an:.2e}, hybrid p={p_hy:.2e}")

        # Cell-type agreement
        ct = res.get("celltype_agreement")
        if ct:
            median_r = np.median([r["spearman_r"] for r in ct])
            print(f"    Cell-type agreement: median r={median_r:.4f} ({len(ct)} types)")

        # Uncertainty
        unc = res.get("uncertainty")
        if unc:
            print(f"    Uncertainty calibration: var-error r={unc['var_error_spearman_r']:.4f}")

        # Latent
        lat = res.get("latent_structure")
        if lat:
            print(f"    Silhouette: z_T={lat['silhouette_z_T']:.4f}, z_PT={lat['silhouette_z_PT']:.4f}, gamma={lat['silhouette_gamma']:.4f}")

        # Gene ranking
        rank = res.get("gene_ranking")
        if rank:
            print(f"    Gene ranking: top-50 overlap={rank['top_median_overlap']}/50, rank r={rank['rank_correlation']:.4f}")

    # Cross-dataset
    if cross_ds:
        print(f"\n  CROSS-DATASET CONSISTENCY")
        for method in ("analytical", "deepptr"):
            entries = cross_ds.get(method, [])
            for e in entries:
                print(f"    {method}: {e['dataset_a']} vs {e['dataset_b']}: "
                      f"r={e['spearman_r']:.4f} (n={e['n_shared_genes']})")

    # Subsampling
    if subsampling:
        print(f"\n  SUBSAMPLING ROBUSTNESS (DeepPTR)")
        for ds_name, records in subsampling.items():
            for r in records:
                print(f"    {ds_name} @ {r['fraction']*100:.0f}%: r={r['spearman_r']:.4f}")


def main():
    set_figure_style()
    ensure_dirs()

    datasets = [
        ("pancreas", scptr.datasets.pancreas, "clusters"),
        ("dentate_gyrus", scptr.datasets.dentate_gyrus, "clusters"),
    ]

    # Per-dataset analysis
    results = {}
    for name, loader, cluster_key in datasets:
        results[name] = run_dataset(name, loader, cluster_key)

    # Cross-dataset consistency
    cross_ds = run_cross_dataset_consistency(datasets)

    # Subsampling robustness
    subsampling = run_subsampling_all(datasets)

    # Summary
    print_summary(results, cross_ds, subsampling)

    # Save combined
    combined = {
        "per_dataset": {k: v for k, v in results.items()},
        "cross_dataset": cross_ds,
        "subsampling": subsampling,
    }
    with open(OUTPUT_DIR / "results" / "combined_v2.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
