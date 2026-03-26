#!/usr/bin/env python
"""Demonstrate what DeepPTR can do that the analytical method cannot.

Key advantages:
1. Uncertainty-guided gene filtering improves half-life correlation
2. Cell-specific gamma resolves transition-state heterogeneity
3. Latent disentanglement discovers post-transcriptional programs
4. Posterior sampling enables statistical testing of gamma differences

All results saved to output/deep_advantages/.
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
import scanpy as sc

import torch
torch.set_num_threads(4)

sys.path.insert(0, str(Path(__file__).parent))
from _common import set_figure_style

import scptr

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "deep_advantages"


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
    return adata_sub


def prepare_both(adata_loader, n_top=300):
    """Run analytical and DeepPTR pipelines, return both adatas."""
    # Analytical
    adata_an = adata_loader()
    scptr.pp.filter_genes(adata_an)
    scptr.pp.normalize_layers(adata_an)
    scptr.pp.neighbors(adata_an, n_neighbors=30)
    scptr.pp.smooth_layers(adata_an)
    scptr.tl.estimate_beta(adata_an)
    scptr.tl.estimate_gamma(adata_an)

    # DeepPTR
    adata_dp = adata_loader()
    scptr.pp.filter_genes(adata_dp)
    scptr.pp.normalize_layers(adata_dp)
    scptr.pp.neighbors(adata_dp, n_neighbors=30)
    scptr.pp.smooth_layers(adata_dp)
    scptr.tl.estimate_beta(adata_dp)
    adata_dp = select_top_genes(adata_dp, n_top=n_top)

    torch.set_num_threads(4)
    model, history = scptr.deep.fit_deepptr(
        adata_dp,
        d_T=8, d_PT=8, d_hidden=48, n_enc_layers=2,
        batch_size=512, max_epochs=100, kl_warmup_epochs=20,
        patience=15, n_posterior_samples=30,
        device="cpu", seed=0, verbose=True,
    )
    return adata_an, adata_dp, model


# ============================================================================
# 1. UNCERTAINTY-GUIDED GENE FILTERING
# ============================================================================

def advantage_uncertainty_filtering(adata_dp, dataset_name):
    """Show that filtering genes by low posterior variance improves half-life correlation.

    The analytical method has no uncertainty estimate — all genes are treated equally.
    DeepPTR's posterior variance lets us select high-confidence genes, improving
    downstream correlations.
    """
    print(f"\n{'=' * 60}")
    print(f"ADVANTAGE 1: Uncertainty-guided gene filtering ({dataset_name})")
    print("=" * 60)

    hl_mouse = scptr.datasets.herzog2017_halflives()
    hl_human = scptr.datasets.schofield2018_halflives()

    gamma_med = np.median(adata_dp.layers["gamma"], axis=0)
    gamma_var_med = np.median(adata_dp.layers["gamma_var"], axis=0)

    # Coefficient of variation of gamma across posterior samples
    gamma_cv = np.sqrt(gamma_var_med) / (gamma_med + 1e-8)

    results = {}
    for ref_name, hl_df in [("mouse", hl_mouse), ("human", hl_human)]:
        # Match genes
        hl_s = hl_df.set_index("gene_symbol")["half_life_hours"]
        # Case-insensitive matching
        gamma_upper = {g.upper(): i for i, g in enumerate(adata_dp.var_names)}
        hl_upper = {g.upper(): g for g in hl_s.index if isinstance(g, str)}
        shared = set(gamma_upper.keys()) & set(hl_upper.keys())

        if len(shared) < 10:
            print(f"  {ref_name}: too few shared genes ({len(shared)})")
            continue

        g_idx = [gamma_upper[u] for u in shared]
        h_vals = np.array([hl_s[hl_upper[u]] for u in shared], dtype=float)
        g_vals = gamma_med[g_idx]
        cv_vals = gamma_cv[g_idx]

        valid = np.isfinite(g_vals) & np.isfinite(h_vals) & (g_vals > 0) & (h_vals > 0)
        g_vals, h_vals, cv_vals = g_vals[valid], h_vals[valid], cv_vals[valid]

        # Baseline: all genes
        sp_all, _ = stats.spearmanr(g_vals, h_vals)

        # Filter by uncertainty thresholds
        thresholds = [1.0, 0.75, 0.5, 0.3, 0.2]
        records = [{"threshold": "all", "n_genes": len(g_vals), "spearman_r": float(sp_all)}]

        for thr in thresholds:
            mask = cv_vals < thr
            if mask.sum() < 10:
                continue
            sp_r, _ = stats.spearmanr(g_vals[mask], h_vals[mask])
            records.append({
                "threshold": f"CV<{thr}",
                "n_genes": int(mask.sum()),
                "spearman_r": float(sp_r),
            })

        # Also try variance-based percentile filtering
        for pct in [75, 50, 25]:
            cutoff = np.percentile(cv_vals, pct)
            mask = cv_vals <= cutoff
            if mask.sum() < 10:
                continue
            sp_r, _ = stats.spearmanr(g_vals[mask], h_vals[mask])
            records.append({
                "threshold": f"bottom_{pct}pct_CV",
                "n_genes": int(mask.sum()),
                "spearman_r": float(sp_r),
            })

        results[ref_name] = records
        print(f"\n  {ref_name} half-life:")
        for r in records:
            print(f"    {r['threshold']:>20s}: r={r['spearman_r']:.4f} (n={r['n_genes']})")

    # Plot improvement
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, (ref_name, records) in enumerate(results.items()):
        if not records:
            continue
        labels = [r["threshold"] for r in records]
        rs = [r["spearman_r"] for r in records]
        ns = [r["n_genes"] for r in records]

        ax = axes[ax_idx]
        bars = ax.bar(range(len(labels)), [-r for r in rs], color="steelblue", alpha=0.7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("|Spearman r| with half-life")
        ax.set_title(f"{dataset_name}: {ref_name} reference")

        # Annotate with n_genes
        for i, (bar, n) in enumerate(zip(bars, ns)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"n={n}", ha="center", va="bottom", fontsize=7)

        # Highlight improvement
        if len(rs) > 1:
            best = max(range(len(rs)), key=lambda i: abs(rs[i]))
            if best > 0:
                bars[best].set_color("darkorange")

    fig.suptitle("Uncertainty-guided filtering improves half-life correlation", y=1.02)
    fig.tight_layout()
    save_fig(fig, f"{dataset_name}_uncertainty_filtering")

    return results


# ============================================================================
# 2. CELL-SPECIFIC GAMMA RESOLUTION
# ============================================================================

def advantage_cell_resolution(adata_an, adata_dp, dataset_name, cluster_key="clusters"):
    """Show DeepPTR captures per-cell gamma variation that smoothed analytical misses.

    The analytical method smoothes Mu/Ms across neighbors, collapsing per-cell variation.
    DeepPTR infers gamma per-cell from the generative model, preserving heterogeneity
    at transition states.
    """
    print(f"\n{'=' * 60}")
    print(f"ADVANTAGE 2: Cell-specific gamma resolution ({dataset_name})")
    print("=" * 60)

    if cluster_key not in adata_an.obs.columns:
        print("  [SKIP] No cluster key")
        return None

    shared = adata_an.var_names.intersection(adata_dp.var_names)
    an_idx = [list(adata_an.var_names).index(g) for g in shared]
    dp_idx = [list(adata_dp.var_names).index(g) for g in shared]

    cell_types = sorted(adata_an.obs[cluster_key].unique())

    # For each cell type: compare within-cluster gamma CV (coefficient of variation)
    # Higher CV = more heterogeneity captured
    records = []
    for ct in cell_types:
        mask_an = (adata_an.obs[cluster_key] == ct).values
        mask_dp = (adata_dp.obs[cluster_key] == ct).values

        if mask_an.sum() < 10 or mask_dp.sum() < 10:
            continue

        gamma_an_ct = adata_an.layers["gamma"][mask_an][:, an_idx]
        gamma_dp_ct = adata_dp.layers["gamma"][mask_dp][:, dp_idx]

        # Per-gene CV within this cell type
        mean_an = gamma_an_ct.mean(axis=0)
        std_an = gamma_an_ct.std(axis=0)
        cv_an = np.where(mean_an > 0.01, std_an / mean_an, 0)

        mean_dp = gamma_dp_ct.mean(axis=0)
        std_dp = gamma_dp_ct.std(axis=0)
        cv_dp = np.where(mean_dp > 0.01, std_dp / mean_dp, 0)

        # Median CV across genes
        records.append({
            "cell_type": str(ct),
            "n_cells": int(mask_an.sum()),
            "median_cv_analytical": float(np.median(cv_an)),
            "median_cv_deepptr": float(np.median(cv_dp)),
            "mean_cv_analytical": float(np.mean(cv_an)),
            "mean_cv_deepptr": float(np.mean(cv_dp)),
        })

    if not records:
        return None

    df = pd.DataFrame(records)
    print(f"\n  Within-cluster gamma CV (higher = more heterogeneity):")
    print(f"  {'Cell type':<25} {'Analytical':>12} {'DeepPTR':>12} {'Ratio':>8}")
    for _, row in df.iterrows():
        ratio = row["median_cv_deepptr"] / max(row["median_cv_analytical"], 1e-8)
        print(f"  {row['cell_type']:<25} {row['median_cv_analytical']:>12.4f} "
              f"{row['median_cv_deepptr']:>12.4f} {ratio:>8.2f}x")

    # Inter-vs-intra cluster variance ratio (a.k.a. "signal to noise")
    # If DeepPTR captures real biological variation, its inter/intra ratio
    # should be similar or better than analytical
    gamma_an_shared = adata_an.layers["gamma"][:, an_idx]
    gamma_dp_shared = adata_dp.layers["gamma"][:, dp_idx]
    labels = adata_an.obs[cluster_key].values

    # F-statistic per gene (one-way ANOVA: do cell types differ?)
    from scipy.stats import f_oneway
    n_sig_an = 0
    n_sig_dp = 0
    n_tested = 0
    f_stats_an = []
    f_stats_dp = []

    for g in range(len(shared)):
        groups_an = [gamma_an_shared[labels == ct, g] for ct in cell_types
                     if (labels == ct).sum() >= 5]
        groups_dp = [gamma_dp_shared[adata_dp.obs[cluster_key].values == ct, g]
                     for ct in cell_types
                     if (adata_dp.obs[cluster_key].values == ct).sum() >= 5]

        if len(groups_an) < 2 or len(groups_dp) < 2:
            continue

        # Only test if there's signal
        if np.std(gamma_an_shared[:, g]) < 1e-6 and np.std(gamma_dp_shared[:, g]) < 1e-6:
            continue

        n_tested += 1
        try:
            f_an, p_an = f_oneway(*groups_an)
            f_dp, p_dp = f_oneway(*groups_dp)
            f_stats_an.append(f_an)
            f_stats_dp.append(f_dp)
            if p_an < 0.05:
                n_sig_an += 1
            if p_dp < 0.05:
                n_sig_dp += 1
        except Exception:
            pass

    print(f"\n  Cell-type-specific gamma (ANOVA, {n_tested} genes):")
    print(f"    Analytical: {n_sig_an}/{n_tested} genes significant (p<0.05)")
    print(f"    DeepPTR:    {n_sig_dp}/{n_tested} genes significant (p<0.05)")
    if f_stats_an and f_stats_dp:
        print(f"    Median F-stat: analytical={np.median(f_stats_an):.2f}, "
              f"DeepPTR={np.median(f_stats_dp):.2f}")

    result = {
        "per_celltype_cv": records,
        "anova_n_tested": n_tested,
        "anova_n_sig_analytical": n_sig_an,
        "anova_n_sig_deepptr": n_sig_dp,
        "anova_median_F_analytical": float(np.median(f_stats_an)) if f_stats_an else None,
        "anova_median_F_deepptr": float(np.median(f_stats_dp)) if f_stats_dp else None,
    }

    # Plot: scatter of F-statistics
    if f_stats_an and f_stats_dp:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # F-statistic comparison
        ax = axes[0]
        min_len = min(len(f_stats_an), len(f_stats_dp))
        ax.scatter(f_stats_an[:min_len], f_stats_dp[:min_len], alpha=0.3, s=8, c="steelblue")
        lim = max(max(f_stats_an[:min_len]), max(f_stats_dp[:min_len]))
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
        ax.set_xlabel("Analytical F-statistic")
        ax.set_ylabel("DeepPTR F-statistic")
        ax.set_title(f"Cell-type discrimination per gene")
        ax.set_xscale("log")
        ax.set_yscale("log")

        # CV comparison
        ax = axes[1]
        ax.bar(range(len(df)), df["median_cv_analytical"], width=0.4,
               label="Analytical", alpha=0.7, color="steelblue")
        ax.bar([x + 0.4 for x in range(len(df))], df["median_cv_deepptr"], width=0.4,
               label="DeepPTR", alpha=0.7, color="darkorange")
        ax.set_xticks([x + 0.2 for x in range(len(df))])
        ax.set_xticklabels(df["cell_type"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Median within-cluster gamma CV")
        ax.set_title(f"Per-cell heterogeneity")
        ax.legend()

        fig.suptitle(f"{dataset_name}: Cell-specific gamma resolution", y=1.02)
        fig.tight_layout()
        save_fig(fig, f"{dataset_name}_cell_resolution")

    return result


# ============================================================================
# 3. LATENT DISENTANGLEMENT DISCOVERS PT PROGRAMS
# ============================================================================

def advantage_disentanglement(adata_dp, dataset_name, cluster_key="clusters"):
    """Show z_PT captures post-transcriptional programs invisible in expression.

    z_T captures transcriptional identity (cell type).
    z_PT captures orthogonal post-transcriptional regulation.
    Genes loading on z_PT but not z_T reveal PT-specific regulation.
    """
    print(f"\n{'=' * 60}")
    print(f"ADVANTAGE 3: Latent disentanglement ({dataset_name})")
    print("=" * 60)

    z_T = adata_dp.obsm["X_z_T"]
    z_PT = adata_dp.obsm["X_z_PT"]
    gamma = adata_dp.layers["gamma"]

    # 1. Correlation of each gene's gamma with z_T vs z_PT
    # Genes correlated with z_PT but not z_T are PT-specific
    r_T = np.zeros(adata_dp.n_vars)
    r_PT = np.zeros(adata_dp.n_vars)

    for g in range(adata_dp.n_vars):
        gv = gamma[:, g]
        if gv.std() < 1e-8:
            continue
        # Max absolute correlation with any z_T dimension
        r_T[g] = max(abs(stats.spearmanr(gv, z_T[:, d]).statistic)
                      for d in range(z_T.shape[1]))
        r_PT[g] = max(abs(stats.spearmanr(gv, z_PT[:, d]).statistic)
                       for d in range(z_PT.shape[1]))

    # Genes specifically correlated with z_PT
    pt_specific_mask = (r_PT > 0.3) & (r_PT > r_T * 1.5)
    t_specific_mask = (r_T > 0.3) & (r_T > r_PT * 1.5)

    pt_genes = adata_dp.var_names[pt_specific_mask].tolist()
    t_genes = adata_dp.var_names[t_specific_mask].tolist()

    print(f"\n  PT-specific genes (r_PT>0.3, r_PT>1.5*r_T): {len(pt_genes)}")
    if pt_genes:
        print(f"    Top PT genes: {pt_genes[:15]}")
    print(f"  T-specific genes  (r_T>0.3, r_T>1.5*r_PT):  {len(t_genes)}")
    if t_genes:
        print(f"    Top T genes:  {t_genes[:15]}")

    # 2. Cluster in z_PT space to find PT states
    from sklearn.cluster import KMeans
    n_pt_clusters = min(5, max(2, len(set(adata_dp.obs.get(cluster_key, []))) // 2))
    km = KMeans(n_clusters=n_pt_clusters, random_state=0, n_init=10)
    pt_labels = km.fit_predict(z_PT)
    adata_dp.obs["pt_cluster_deep"] = pd.Categorical([f"PT_{i}" for i in pt_labels])

    # 3. Compare: do PT clusters align with expression clusters?
    if cluster_key in adata_dp.obs.columns:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        expr_labels = adata_dp.obs[cluster_key].astype("category").cat.codes.values
        ari = adjusted_rand_score(expr_labels, pt_labels)
        nmi = normalized_mutual_info_score(expr_labels, pt_labels)
        print(f"\n  PT clusters vs expression clusters:")
        print(f"    ARI = {ari:.4f} (0=random, 1=identical)")
        print(f"    NMI = {nmi:.4f}")
        print(f"    → {'Low' if ari < 0.3 else 'Moderate' if ari < 0.6 else 'High'} "
              f"overlap: PT space captures {'different' if ari < 0.3 else 'partially overlapping'} structure")
    else:
        ari = nmi = None

    # 4. Find genes differentially degraded between PT clusters
    # (these are genes whose degradation rate differs for reasons orthogonal to expression)
    from scipy.stats import kruskal
    pt_de_genes = []
    for g in range(adata_dp.n_vars):
        groups = [gamma[pt_labels == k, g] for k in range(n_pt_clusters)]
        groups = [grp for grp in groups if len(grp) >= 5]
        if len(groups) < 2:
            continue
        try:
            h_stat, p_val = kruskal(*groups)
            if p_val < 0.01:
                effect = np.max([np.median(grp) for grp in groups]) / max(np.min([np.median(grp) for grp in groups]), 1e-8)
                pt_de_genes.append({
                    "gene": adata_dp.var_names[g],
                    "H_statistic": float(h_stat),
                    "p_value": float(p_val),
                    "fold_change": float(effect),
                })
        except Exception:
            pass

    pt_de_genes.sort(key=lambda x: x["p_value"])
    print(f"\n  Genes differentially degraded between PT clusters: {len(pt_de_genes)}")
    if pt_de_genes:
        print(f"    Top 10:")
        for g in pt_de_genes[:10]:
            print(f"      {g['gene']:<15} H={g['H_statistic']:.1f} p={g['p_value']:.2e} FC={g['fold_change']:.2f}")

    result = {
        "n_pt_specific_genes": len(pt_genes),
        "pt_specific_genes": pt_genes[:50],
        "n_t_specific_genes": len(t_genes),
        "t_specific_genes": t_genes[:50],
        "pt_vs_expr_ari": float(ari) if ari is not None else None,
        "pt_vs_expr_nmi": float(nmi) if nmi is not None else None,
        "n_pt_de_genes": len(pt_de_genes),
        "top_pt_de_genes": pt_de_genes[:20],
    }

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: r_T vs r_PT scatter
    ax = axes[0]
    ax.scatter(r_T, r_PT, alpha=0.3, s=8, c="gray")
    if pt_specific_mask.any():
        ax.scatter(r_T[pt_specific_mask], r_PT[pt_specific_mask],
                   alpha=0.7, s=15, c="darkorange", label=f"PT-specific ({len(pt_genes)})")
    if t_specific_mask.any():
        ax.scatter(r_T[t_specific_mask], r_PT[t_specific_mask],
                   alpha=0.7, s=15, c="steelblue", label=f"T-specific ({len(t_genes)})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("Max |r| with z_T")
    ax.set_ylabel("Max |r| with z_PT")
    ax.set_title("Gene regulation mode")
    ax.legend(fontsize=8)

    # Panel 2: z_PT PCA colored by PT cluster
    from sklearn.decomposition import PCA
    z_2d = PCA(n_components=2).fit_transform(z_PT)
    cmap = plt.colormaps.get_cmap("Set2")
    ax = axes[1]
    for k in range(n_pt_clusters):
        mask = pt_labels == k
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1], alpha=0.3, s=5,
                   c=[cmap(k)], label=f"PT_{k}")
    ax.set_title("z_PT space (PT clusters)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7, markerscale=3)

    # Panel 3: z_PT colored by expression cluster
    ax = axes[2]
    if cluster_key in adata_dp.obs.columns:
        cats = adata_dp.obs[cluster_key].astype("category")
        codes = cats.cat.codes.values
        n_cats = len(cats.cat.categories)
        cmap_expr = plt.colormaps.get_cmap("tab20")
        for i, cat in enumerate(cats.cat.categories):
            mask = codes == i
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1], alpha=0.3, s=5,
                       c=[cmap_expr(i / n_cats)], label=str(cat))
        ax.set_title(f"z_PT space (expression clusters)\nARI={ari:.3f}")
        if n_cats <= 12:
            ax.legend(fontsize=6, markerscale=3, ncol=2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    fig.suptitle(f"{dataset_name}: Latent disentanglement", y=1.02)
    fig.tight_layout()
    save_fig(fig, f"{dataset_name}_disentanglement")

    return result


# ============================================================================
# 4. POSTERIOR-BASED STATISTICAL TESTING
# ============================================================================

def advantage_statistical_testing(adata_dp, dataset_name, cluster_key="clusters"):
    """Demonstrate posterior-based statistical testing of gamma differences.

    With DeepPTR, we can compute credible intervals for gamma differences
    between cell types — something impossible with a point estimate.
    """
    print(f"\n{'=' * 60}")
    print(f"ADVANTAGE 4: Posterior-based statistical testing ({dataset_name})")
    print("=" * 60)

    if cluster_key not in adata_dp.obs.columns:
        print("  [SKIP] No cluster key")
        return None

    gamma = adata_dp.layers["gamma"]
    gamma_var = adata_dp.layers["gamma_var"]

    cell_types = sorted(adata_dp.obs[cluster_key].unique())
    if len(cell_types) < 2:
        return None

    # Pick two cell types to compare
    # Choose the pair with most cells
    ct_sizes = {ct: (adata_dp.obs[cluster_key] == ct).sum() for ct in cell_types}
    sorted_cts = sorted(ct_sizes.keys(), key=lambda x: ct_sizes[x], reverse=True)
    ct_a, ct_b = sorted_cts[0], sorted_cts[1]

    mask_a = (adata_dp.obs[cluster_key] == ct_a).values
    mask_b = (adata_dp.obs[cluster_key] == ct_b).values

    gamma_a = gamma[mask_a]
    gamma_b = gamma[mask_b]
    var_a = gamma_var[mask_a]
    var_b = gamma_var[mask_b]

    # Per-gene: test if mean gamma differs between cell types
    # Use posterior: mean_diff ~ N(mu_a - mu_b, var_a/n_a + var_b/n_b)
    n_a, n_b = mask_a.sum(), mask_b.sum()
    mean_a = gamma_a.mean(axis=0)
    mean_b = gamma_b.mean(axis=0)
    # Posterior variance of the mean
    var_mean_a = var_a.mean(axis=0) / n_a
    var_mean_b = var_b.mean(axis=0) / n_b

    diff = mean_a - mean_b
    diff_se = np.sqrt(var_mean_a + var_mean_b + 1e-10)
    z_score = diff / diff_se

    # Two-sided test
    p_vals = 2 * (1 - stats.norm.cdf(np.abs(z_score)))

    # Compare with simple t-test (no uncertainty info)
    from scipy.stats import ttest_ind
    p_ttest = np.zeros(adata_dp.n_vars)
    for g in range(adata_dp.n_vars):
        try:
            _, p_ttest[g] = ttest_ind(gamma_a[:, g], gamma_b[:, g])
        except Exception:
            p_ttest[g] = 1.0

    # Count significant at FDR 0.05
    from statsmodels.stats.multitest import multipletests
    _, p_adj_post, _, _ = multipletests(p_vals, method="fdr_bh")
    _, p_adj_ttest, _, _ = multipletests(p_ttest, method="fdr_bh")

    n_sig_post = (p_adj_post < 0.05).sum()
    n_sig_ttest = (p_adj_ttest < 0.05).sum()

    print(f"\n  Comparing {ct_a} ({n_a} cells) vs {ct_b} ({n_b} cells):")
    print(f"    Posterior-informed test: {n_sig_post}/{adata_dp.n_vars} genes significant (FDR<0.05)")
    print(f"    Simple t-test:          {n_sig_ttest}/{adata_dp.n_vars} genes significant (FDR<0.05)")

    # Identify genes found by posterior but not by t-test (and vice versa)
    post_only = (p_adj_post < 0.05) & (p_adj_ttest >= 0.05)
    ttest_only = (p_adj_ttest < 0.05) & (p_adj_post >= 0.05)
    both = (p_adj_post < 0.05) & (p_adj_ttest < 0.05)

    print(f"    Both:          {both.sum()}")
    print(f"    Posterior-only: {post_only.sum()}")
    print(f"    T-test-only:   {ttest_only.sum()}")

    result = {
        "ct_a": str(ct_a),
        "ct_b": str(ct_b),
        "n_cells_a": int(n_a),
        "n_cells_b": int(n_b),
        "n_sig_posterior": int(n_sig_post),
        "n_sig_ttest": int(n_sig_ttest),
        "n_both": int(both.sum()),
        "n_posterior_only": int(post_only.sum()),
        "n_ttest_only": int(ttest_only.sum()),
    }

    # If posterior finds additional genes, list them
    if post_only.any():
        post_only_genes = adata_dp.var_names[post_only].tolist()
        print(f"\n    Posterior-only genes (uncertainty-aware):")
        for g in post_only_genes[:10]:
            idx = list(adata_dp.var_names).index(g)
            print(f"      {g}: diff={diff[idx]:.4f} ± {diff_se[idx]:.4f}")
        result["posterior_only_genes"] = post_only_genes[:20]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(-np.log10(p_ttest + 1e-300), -np.log10(p_vals + 1e-300),
               alpha=0.2, s=5, c="gray")
    if post_only.any():
        ax.scatter(-np.log10(p_ttest[post_only] + 1e-300),
                   -np.log10(p_vals[post_only] + 1e-300),
                   alpha=0.7, s=15, c="darkorange", label="Posterior-only")
    if ttest_only.any():
        ax.scatter(-np.log10(p_ttest[ttest_only] + 1e-300),
                   -np.log10(p_vals[ttest_only] + 1e-300),
                   alpha=0.7, s=15, c="steelblue", label="T-test-only")
    ax.set_xlabel("-log10(p) t-test")
    ax.set_ylabel("-log10(p) posterior")
    ax.set_title(f"{ct_a} vs {ct_b}")
    ax.plot([0, 20], [0, 20], "k--", alpha=0.3)
    ax.legend(fontsize=8)

    # Volcano plot with uncertainty
    ax = axes[1]
    sig = p_adj_post < 0.05
    ax.scatter(diff[~sig], -np.log10(p_vals[~sig] + 1e-300),
               alpha=0.1, s=3, c="gray")
    ax.scatter(diff[sig], -np.log10(p_vals[sig] + 1e-300),
               alpha=0.5, s=8, c="darkorange")
    ax.set_xlabel(f"Mean gamma difference ({ct_a} - {ct_b})")
    ax.set_ylabel("-log10(p)")
    ax.set_title(f"Posterior volcano ({n_sig_post} significant)")
    ax.axhline(-np.log10(0.05), color="red", ls="--", alpha=0.3)

    fig.suptitle(f"{dataset_name}: Posterior-based differential degradation", y=1.02)
    fig.tight_layout()
    save_fig(fig, f"{dataset_name}_posterior_testing")

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    set_figure_style()
    ensure_dirs()

    datasets = [
        ("pancreas", scptr.datasets.pancreas, "clusters"),
        ("dentate_gyrus", scptr.datasets.dentate_gyrus, "clusters"),
    ]

    all_results = {}

    for name, loader, cluster_key in datasets:
        print(f"\n{'#' * 60}")
        print(f"# {name.upper()}")
        print(f"{'#' * 60}")

        adata_an, adata_dp, model = prepare_both(loader, n_top=300)

        results = {}

        # 1. Uncertainty-guided filtering
        results["uncertainty_filtering"] = advantage_uncertainty_filtering(adata_dp, name)

        # 2. Cell-specific gamma
        results["cell_resolution"] = advantage_cell_resolution(adata_an, adata_dp, name, cluster_key)

        # 3. Latent disentanglement
        results["disentanglement"] = advantage_disentanglement(adata_dp, name, cluster_key)

        # 4. Posterior testing
        results["statistical_testing"] = advantage_statistical_testing(adata_dp, name, cluster_key)

        all_results[name] = results

        with open(OUTPUT_DIR / "results" / f"{name}_advantages.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Summary
    print(f"\n{'=' * 70}")
    print("DEEPPTR UNIQUE ADVANTAGES SUMMARY")
    print("=" * 70)

    for name, results in all_results.items():
        print(f"\n  {name.upper()}")

        # Uncertainty filtering
        uf = results.get("uncertainty_filtering", {})
        for ref, records in uf.items():
            if records:
                r_all = records[0]["spearman_r"]
                r_best = min(records, key=lambda x: x["spearman_r"])  # most negative
                improvement = abs(r_best["spearman_r"]) - abs(r_all)
                print(f"    Uncertainty filtering ({ref}): {r_all:.4f} → {r_best['spearman_r']:.4f} "
                      f"(+{improvement:.4f} at {r_best['threshold']})")

        # Cell resolution
        cr = results.get("cell_resolution", {})
        if cr:
            print(f"    Cell-type ANOVA: analytical={cr['anova_n_sig_analytical']}, "
                  f"DeepPTR={cr['anova_n_sig_deepptr']} significant genes")

        # Disentanglement
        dis = results.get("disentanglement", {})
        if dis:
            print(f"    PT-specific genes: {dis['n_pt_specific_genes']}, "
                  f"T-specific: {dis['n_t_specific_genes']}")
            if dis.get("pt_vs_expr_ari") is not None:
                print(f"    PT vs expr overlap: ARI={dis['pt_vs_expr_ari']:.4f} "
                      f"({'orthogonal' if dis['pt_vs_expr_ari'] < 0.2 else 'partially overlapping'})")
            print(f"    DE genes between PT clusters: {dis['n_pt_de_genes']}")

        # Statistical testing
        st = results.get("statistical_testing", {})
        if st:
            print(f"    Posterior testing ({st['ct_a']} vs {st['ct_b']}): "
                  f"{st['n_sig_posterior']} posterior, {st['n_sig_ttest']} t-test, "
                  f"{st['n_posterior_only']} posterior-only")

    # Save combined
    with open(OUTPUT_DIR / "results" / "combined_advantages.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
