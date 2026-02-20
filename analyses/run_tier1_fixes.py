#!/usr/bin/env python
"""Address Tier 1 and Tier 2 reviewer concerns systematically.

T1-1: Functionally characterize invisible states (GSEA on differentially degraded genes)
T1-2: Fix gamma=0 reporting (filter genes with insufficient unspliced coverage)
T1-3: Investigate destabilizing bias in RBP networks
T1-4: PT velocity streamlines on UMAP
T2-1: Ablation experiments (naive u/s ratio vs full scPTR)
T2-2: Explain TF score discrepancy between datasets
T2-3: Housekeeping gene analysis for cross-dataset consistency
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
import scanpy as sc
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent))
from _common import set_figure_style

import scptr

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "tier1_fixes"


def save_fig(fig, name, subdir="figures"):
    if fig is None:
        return
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def run_pipeline(adata, name):
    """Run standard scPTR pipeline."""
    print(f"\n--- Pipeline: {name} ---")
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)
    scptr.tl.variance_decomposition(adata)
    scptr.tl.pt_states(adata)
    scptr.tl.pt_velocity(adata)
    print(f"  Done: {adata.shape}")
    return adata


# =========================================================================
# T1-2: Fix gamma=0 reporting
# =========================================================================
def fix_gamma_reporting(adata, name):
    """Report gamma statistics only on genes with reliable estimates.

    The gamma=0 median is a sparsity artifact: genes with zero unspliced
    counts get gamma=0 by definition. Report separately for:
    1. All genes (including zeros)
    2. Genes with >=10% cells having nonzero gamma ("gamma-informative")
    """
    print(f"\n{'='*60}")
    print(f"T1-2: GAMMA REPORTING FIX ({name})")
    print(f"{'='*60}")

    gamma = adata.layers["gamma"]
    n_genes = gamma.shape[1]

    # Per-gene: fraction of cells with nonzero gamma
    nonzero_frac = (gamma > 0).mean(axis=0)
    median_gamma = np.median(gamma, axis=0)

    # Thresholds for "informative"
    for thresh in [0.0, 0.05, 0.1, 0.2]:
        mask = nonzero_frac >= thresh
        n = mask.sum()
        if n > 0:
            med = np.median(median_gamma[mask])
            mean = np.mean(median_gamma[mask])
            print(f"  Genes with >=  {thresh:.0%} nonzero gamma: {n}/{n_genes} "
                  f"(median of medians = {med:.4f}, mean = {mean:.4f})")

    # Key metric: what fraction of genes have usable gamma?
    informative = nonzero_frac >= 0.1
    print(f"\n  Gamma-informative genes (>=10% nonzero): {informative.sum()}/{n_genes} "
          f"({100*informative.mean():.1f}%)")
    print(f"  These genes' median gamma: {np.median(median_gamma[informative]):.4f}")

    # Unspliced detection rate
    u = adata.layers.get("Mu", adata.layers.get("unspliced"))
    if u is not None:
        u_arr = u.toarray() if hasattr(u, 'toarray') else np.asarray(u)
        u_detection = (u_arr > 0).mean(axis=0)
        print(f"\n  Unspliced detection: mean={u_detection.mean():.3f}, "
              f"median={np.median(u_detection):.3f}")
        print(f"  Genes with >5% unspliced detection: {(u_detection > 0.05).sum()}/{n_genes}")

    return {
        "n_genes_total": int(n_genes),
        "n_informative_10pct": int(informative.sum()),
        "frac_informative": float(informative.mean()),
        "median_gamma_informative": float(np.median(median_gamma[informative])),
        "median_gamma_all": float(np.median(median_gamma)),
    }


# =========================================================================
# T1-1: Functional characterization of invisible states
# =========================================================================
def characterize_invisible_states(adata, name):
    """Find invisible states and characterize differentially degraded genes."""
    print(f"\n{'='*60}")
    print(f"T1-1: FUNCTIONAL CHARACTERIZATION ({name})")
    print(f"{'='*60}")

    res_dir = OUTPUT_DIR / "results" / "invisible_states" / name
    res_dir.mkdir(parents=True, exist_ok=True)

    gamma = adata.layers["gamma"]
    clusters = adata.obs["clusters"]

    all_results = []

    for cluster_name in clusters.unique():
        mask = (clusters == cluster_name).values
        n_cells = mask.sum()
        if n_cells < 50:
            continue

        gamma_sub = gamma[mask]
        # Filter to gamma-informative genes for this cluster
        gene_nonzero = (gamma_sub > 0).mean(axis=0)
        good_genes = gene_nonzero >= 0.1
        if good_genes.sum() < 20:
            continue

        gamma_filtered = gamma_sub[:, good_genes]
        gene_names = adata.var_names[good_genes]

        # PCA + KMeans on gamma
        n_pcs = min(15, n_cells - 1, gamma_filtered.shape[1] - 1)
        pca = PCA(n_components=n_pcs, random_state=42)
        gamma_pcs = pca.fit_transform(gamma_filtered)

        best_k, best_sil, best_labels = 1, -1, np.zeros(n_cells, dtype=int)
        for k in [2, 3]:
            if n_cells < k * 10:
                continue
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(gamma_pcs)
            if min(np.bincount(labels)) < 10:
                continue
            sil = silhouette_score(gamma_pcs, labels)
            if sil > best_sil:
                best_sil, best_k, best_labels = sil, k, labels

        if best_k <= 1:
            continue

        # Expression silhouette for same labels
        expr_sub = adata.X[mask].toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X[mask])
        n_expr_pcs = min(15, n_cells - 1, expr_sub.shape[1] - 1)
        pca_expr = PCA(n_components=n_expr_pcs, random_state=42)
        expr_pcs = pca_expr.fit_transform(expr_sub)
        sil_expr = silhouette_score(expr_pcs, best_labels)

        invisibility = best_sil - sil_expr
        is_invisible = invisibility > 0.05

        if not is_invisible:
            continue

        print(f"\n  {cluster_name}: INVISIBLE (sil_gamma={best_sil:.3f}, "
              f"sil_expr={sil_expr:.3f})")

        # Differential degradation between sub-clusters
        diff_results = []
        for gi, gene in enumerate(gene_names):
            groups = [gamma_filtered[best_labels == j, gi] for j in range(best_k)]
            if all(len(g) >= 5 for g in groups):
                if best_k == 2:
                    u_stat, p_val = stats.mannwhitneyu(groups[0], groups[1],
                                                       alternative='two-sided')
                else:
                    _, p_val = stats.kruskal(*groups)

                medians = [np.median(g) for g in groups]
                max_med = max(medians)
                min_med = min(medians)
                log_fc = np.log2((max_med + 0.01) / (min_med + 0.01))

                diff_results.append({
                    "gene": gene,
                    "p_value": p_val,
                    "log2_fc_gamma": log_fc,
                    "medians": medians,
                })

        if not diff_results:
            continue

        diff_df = pd.DataFrame(diff_results)
        # FDR correction
        from statsmodels.stats.multitest import multipletests
        _, diff_df["fdr"], _, _ = multipletests(diff_df["p_value"], method="fdr_bh")

        # Significant differentially degraded genes
        sig = diff_df[diff_df["fdr"] < 0.05].sort_values("log2_fc_gamma", ascending=False)
        print(f"    Differentially degraded genes (FDR<0.05): {len(sig)}/{len(diff_df)}")

        if len(sig) > 0:
            # Top destabilized (high gamma in one sub-cluster)
            top_destab = sig.head(10)
            print(f"    Top destabilized: {top_destab['gene'].tolist()}")

            # Top stabilized (low gamma difference but significant)
            top_stab = sig.tail(10)
            print(f"    Top stabilized: {top_stab['gene'].tolist()}")

            sig.to_csv(res_dir / f"{cluster_name}_diff_degraded.csv", index=False)

            # Run enrichment using gseapy (Enrichr API)
            try:
                import gseapy as gp

                # Use top differentially degraded genes for enrichment
                gene_list = sig["gene"].tolist()
                if len(gene_list) >= 5:
                    # Determine organism
                    # If gene names are Titlecase → mouse; UPPERCASE → human
                    sample_gene = gene_list[0]
                    organism = "mouse" if sample_gene[0].isupper() and sample_gene[1:].islower() else "human"

                    gene_sets = ["GO_Biological_Process_2023",
                                 "KEGG_2021_Human" if organism == "human" else "KEGG_2019_Mouse"]

                    enr = gp.enrichr(gene_list=gene_list,
                                     gene_sets=gene_sets,
                                     organism=organism,
                                     outdir=None,
                                     no_plot=True)

                    enr_df = enr.results
                    sig_enr = enr_df[enr_df["Adjusted P-value"] < 0.1].head(15)

                    if len(sig_enr) > 0:
                        print(f"    Enriched pathways (FDR<0.1):")
                        for _, row in sig_enr.iterrows():
                            print(f"      {row['Term'][:60]}: p={row['Adjusted P-value']:.4f}")
                        sig_enr.to_csv(res_dir / f"{cluster_name}_enrichment.csv", index=False)
                    else:
                        print(f"    No significant pathway enrichment found")
            except Exception as e:
                print(f"    [WARNING] Enrichment failed: {e}")

        all_results.append({
            "cluster": cluster_name,
            "n_cells": n_cells,
            "n_subclusters": best_k,
            "sil_gamma": best_sil,
            "sil_expr": sil_expr,
            "invisibility": invisibility,
            "n_diff_genes": len(sig) if len(sig) > 0 else 0,
        })

    return pd.DataFrame(all_results)


# =========================================================================
# T1-3: Investigate destabilizing bias in RBP networks
# =========================================================================
def investigate_destabilizing_bias(adata, name):
    """Investigate why RBP networks show predominantly destabilizing effects."""
    print(f"\n{'='*60}")
    print(f"T1-3: DESTABILIZING BIAS INVESTIGATION ({name})")
    print(f"{'='*60}")

    res_dir = OUTPUT_DIR / "results" / "network_bias"
    res_dir.mkdir(parents=True, exist_ok=True)

    gamma = adata.layers["gamma"]
    gamma_med = np.median(gamma, axis=0)

    # Load RBP list
    rbp_path = Path(__file__).parent.parent / "src" / "scptr" / "tools" / "data" / "known_rbps.csv"
    rbps = pd.read_csv(rbp_path)["gene_symbol"].tolist()

    # Find RBPs in dataset (case-insensitive)
    adata_genes_upper = {g.upper(): g for g in adata.var_names}
    rbp_in_data = []
    for r in rbps:
        if r.upper() in adata_genes_upper:
            rbp_in_data.append(adata_genes_upper[r.upper()])

    print(f"  RBPs in dataset: {len(rbp_in_data)}")

    # Get expression matrix
    if hasattr(adata.X, 'toarray'):
        expr = adata.X.toarray()
    else:
        expr = np.asarray(adata.X)

    # Select target genes: top variable gamma (filtered to informative)
    nonzero_frac = (gamma > 0).mean(axis=0)
    informative = nonzero_frac >= 0.1
    gamma_var = np.var(gamma[:, informative], axis=0)
    n_targets = min(200, informative.sum())
    top_var_idx = np.argsort(gamma_var)[-n_targets:]
    info_indices = np.where(informative)[0]
    target_indices = info_indices[top_var_idx]
    target_genes = adata.var_names[target_indices]

    # Correlation analysis
    print("\n  Correlation analysis:")
    all_edges = []

    for rbp_name in rbp_in_data:
        rbp_idx = list(adata.var_names).index(rbp_name)
        rbp_expr = expr[:, rbp_idx]

        if np.std(rbp_expr) < 1e-6:
            continue

        for ti, target_name in zip(target_indices, target_genes):
            target_gamma = gamma[:, ti]

            # Only use cells with nonzero gamma for this gene
            valid = target_gamma > 0
            if valid.sum() < 50:
                continue

            r, p = stats.spearmanr(rbp_expr[valid], target_gamma[valid])

            if p < 0.05 / (len(rbp_in_data) * n_targets):  # Bonferroni
                all_edges.append({
                    "rbp": rbp_name,
                    "target": target_name,
                    "spearman_r": r,
                    "p_value": p,
                    "direction": "destabilizing" if r > 0 else "stabilizing",
                })

    edges_df = pd.DataFrame(all_edges)
    if len(edges_df) == 0:
        print("  No significant edges found")
        return

    n_destab = (edges_df["spearman_r"] > 0).sum()
    n_stab = (edges_df["spearman_r"] < 0).sum()
    print(f"  Total significant edges: {len(edges_df)}")
    print(f"  Destabilizing (r>0): {n_destab} ({100*n_destab/len(edges_df):.1f}%)")
    print(f"  Stabilizing (r<0): {n_stab} ({100*n_stab/len(edges_df):.1f}%)")

    # Key diagnostic: is the bias in the gamma distribution itself?
    # Check: is gamma positively correlated with total expression?
    expr_mean = expr.mean(axis=0)
    gamma_mean = gamma.mean(axis=0)
    r_expr_gamma, _ = stats.spearmanr(expr_mean[informative], gamma_mean[informative])
    print(f"\n  Diagnostic: Spearman(mean_expression, mean_gamma) = {r_expr_gamma:.4f}")
    print(f"  If positive, RBP expression correlates with gamma because both")
    print(f"  correlate with overall expression level → confounding.")

    # Check: does the bias persist after regressing out total expression?
    print("\n  After controlling for total expression per cell:")
    total_expr_per_cell = expr.sum(axis=1)

    n_stab_ctrl = 0
    n_destab_ctrl = 0
    controlled_edges = []

    for rbp_name in rbp_in_data[:10]:  # Test top 10 RBPs
        rbp_idx = list(adata.var_names).index(rbp_name)
        rbp_expr = expr[:, rbp_idx]
        if np.std(rbp_expr) < 1e-6:
            continue

        # Partial correlation: regress out total expression
        # Residualize both RBP expression and gamma against total expression
        from numpy.polynomial.polynomial import polyfit, polyval
        rbp_resid = rbp_expr - np.mean(rbp_expr)
        # Simple: rank-based partial correlation
        rbp_rank = stats.rankdata(rbp_expr)
        total_rank = stats.rankdata(total_expr_per_cell)

        # Regress out total from RBP
        slope = np.cov(rbp_rank, total_rank)[0, 1] / np.var(total_rank)
        rbp_resid = rbp_rank - slope * total_rank

        for ti in target_indices[:50]:
            target_gamma = gamma[:, ti]
            valid = target_gamma > 0
            if valid.sum() < 50:
                continue

            gamma_rank = stats.rankdata(target_gamma[valid])
            total_rank_v = stats.rankdata(total_expr_per_cell[valid])
            slope_g = np.cov(gamma_rank, total_rank_v)[0, 1] / (np.var(total_rank_v) + 1e-10)
            gamma_resid = gamma_rank - slope_g * total_rank_v

            r, p = stats.spearmanr(rbp_resid[valid], gamma_resid)
            if r > 0:
                n_destab_ctrl += 1
            else:
                n_stab_ctrl += 1

    total_ctrl = n_destab_ctrl + n_stab_ctrl
    if total_ctrl > 0:
        print(f"  Destabilizing: {n_destab_ctrl}/{total_ctrl} ({100*n_destab_ctrl/total_ctrl:.1f}%)")
        print(f"  Stabilizing: {n_stab_ctrl}/{total_ctrl} ({100*n_stab_ctrl/total_ctrl:.1f}%)")

        if n_destab_ctrl / total_ctrl < 0.6:
            print(f"  → Bias is reduced after controlling for library size!")
            print(f"  → The original bias was partly a confound: RBPs with higher")
            print(f"     expression → higher overall counts → higher gamma artifacts")
        else:
            print(f"  → Bias persists even after correction")

    # Per-RBP breakdown
    print("\n  Per-RBP breakdown:")
    hub_counts = edges_df.groupby("rbp").agg(
        n_targets=("target", "count"),
        n_stab=("direction", lambda x: (x == "stabilizing").sum()),
        n_destab=("direction", lambda x: (x == "destabilizing").sum()),
        mean_r=("spearman_r", "mean"),
    ).sort_values("n_targets", ascending=False)

    for rbp_name, row in hub_counts.head(10).iterrows():
        ratio = row["n_destab"] / max(row["n_targets"], 1)
        print(f"    {rbp_name}: {int(row['n_targets'])} targets "
              f"({int(row['n_stab'])} stab, {int(row['n_destab'])} destab, "
              f"mean_r={row['mean_r']:.3f})")

    edges_df.to_csv(res_dir / f"edges_{name}.csv", index=False)
    hub_counts.to_csv(res_dir / f"hub_counts_{name}.csv")

    return {
        "n_edges": len(edges_df),
        "frac_destabilizing": float(n_destab / len(edges_df)),
        "expr_gamma_correlation": float(r_expr_gamma),
    }


# =========================================================================
# T1-4: PT velocity streamlines on UMAP
# =========================================================================
def velocity_streamlines(adata, name):
    """Generate proper streamline plots for PT velocity on UMAP."""
    print(f"\n{'='*60}")
    print(f"T1-4: VELOCITY STREAMLINES ({name})")
    print(f"{'='*60}")

    gamma = adata.layers["gamma"]
    velocity = adata.layers["pt_velocity"]

    # We need UMAP coordinates
    if "X_gamma_umap" not in adata.obsm:
        print("  No gamma UMAP, computing...")
        sc.tl.umap(adata)
        coords = adata.obsm["X_umap"]
    else:
        coords = adata.obsm["X_gamma_umap"]

    # Build transition matrix from velocity
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=30)
    nn.fit(coords)
    dists, indices = nn.kneighbors(coords)

    # For each cell, compute velocity-weighted displacement in UMAP space
    n_cells = len(coords)
    dx = np.zeros((n_cells, 2))

    for i in range(n_cells):
        neighbors = indices[i, 1:]  # exclude self
        vel_i = velocity[i]

        for j in neighbors:
            # Gamma displacement: how different is neighbor's gamma from mine?
            gamma_disp = gamma[j] - gamma[i]

            # Project: does the velocity vector point toward this neighbor?
            cos_sim = np.dot(vel_i, gamma_disp) / (
                np.linalg.norm(vel_i) * np.linalg.norm(gamma_disp) + 1e-10
            )

            if cos_sim > 0:
                # Weight by cosine similarity and UMAP displacement
                umap_disp = coords[j] - coords[i]
                dx[i] += cos_sim * umap_disp

    # Normalize
    norms = np.linalg.norm(dx, axis=1, keepdims=True)
    cap = np.percentile(norms[norms > 0], 95)
    dx = dx / (cap + 1e-10)

    # Velocity magnitude for coloring
    vel_mag = np.linalg.norm(velocity, axis=1)
    vel_mag = vel_mag / (np.percentile(vel_mag, 95) + 1e-10)

    # Create streamline-style plot using quiver at grid points
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Quiver plot colored by cluster
    clusters = adata.obs["clusters"]
    cluster_colors = {c: plt.cm.tab20(i / 20) for i, c in enumerate(clusters.unique())}

    for c in clusters.unique():
        mask = (clusters == c).values
        axes[0].scatter(coords[mask, 0], coords[mask, 1], s=3, alpha=0.3,
                       c=[cluster_colors[c]], label=c)

    # Subsample arrows for clarity
    n_arrows = min(500, n_cells)
    arrow_idx = np.random.choice(n_cells, n_arrows, replace=False)
    arrow_mask = np.linalg.norm(dx[arrow_idx], axis=1) > 0.01

    axes[0].quiver(coords[arrow_idx[arrow_mask], 0],
                   coords[arrow_idx[arrow_mask], 1],
                   dx[arrow_idx[arrow_mask], 0],
                   dx[arrow_idx[arrow_mask], 1],
                   color="black", alpha=0.6, scale=20, width=0.003,
                   headwidth=4, headlength=5)
    axes[0].set_title(f"PT Velocity Streamlines: {name}")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].legend(fontsize=6, loc="best", markerscale=3)

    # Panel 2: Velocity magnitude
    sc_plot = axes[1].scatter(coords[:, 0], coords[:, 1], s=3, alpha=0.5,
                              c=np.clip(vel_mag, 0, 1), cmap="YlOrRd")
    axes[1].quiver(coords[arrow_idx[arrow_mask], 0],
                   coords[arrow_idx[arrow_mask], 1],
                   dx[arrow_idx[arrow_mask], 0],
                   dx[arrow_idx[arrow_mask], 1],
                   color="black", alpha=0.4, scale=20, width=0.002,
                   headwidth=4, headlength=5)
    axes[1].set_title(f"PT Velocity Magnitude: {name}")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    plt.colorbar(sc_plot, ax=axes[1], label="Velocity magnitude")

    fig.tight_layout()
    save_fig(fig, f"velocity_streamlines_{name}")

    print(f"  Mean velocity magnitude: {np.mean(np.linalg.norm(velocity, axis=1)):.4f}")
    print(f"  Cells with significant displacement: {arrow_mask.sum()}/{n_arrows}")


# =========================================================================
# T2-1: Ablation experiments
# =========================================================================
def ablation_experiments(adata, name):
    """Compare full scPTR against naive alternatives using invisibility score.

    For each cluster x method:
    1. Find sub-clusters in method's space (sil_method)
    2. Evaluate SAME labels in expression PCA space (sil_expr)
    3. Invisibility = sil_method - sil_expr

    The key claim: scPTR gamma maximizes invisibility (finds sub-populations
    most invisible to expression), not raw separability.
    """
    print(f"\n{'='*60}")
    print(f"T2-1: ABLATION EXPERIMENTS ({name})")
    print(f"{'='*60}")

    res_dir = OUTPUT_DIR / "results" / "ablation"
    res_dir.mkdir(parents=True, exist_ok=True)

    gamma = adata.layers["gamma"]
    clusters = adata.obs["clusters"]

    # Get unspliced counts
    u_layer = adata.layers.get("Mu", adata.layers.get("unspliced"))
    s_layer = adata.layers.get("Ms", adata.layers.get("spliced"))
    u = u_layer.toarray() if hasattr(u_layer, 'toarray') else np.asarray(u_layer)
    s = s_layer.toarray() if hasattr(s_layer, 'toarray') else np.asarray(s_layer)

    # Expression matrix (for expression silhouette computation)
    expr_full = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)

    # Method 1: Full scPTR gamma (already computed)
    # Method 2: Raw u/s ratio (naive, no kinetic model)
    raw_ratio = np.zeros_like(gamma)
    s_safe = np.where(s > 0.01, s, 1.0)
    raw_ratio = u / s_safe
    raw_ratio[s < 0.01] = 0

    # Method 3: PCA on unspliced counts alone
    # Method 4: PCA on expression alone (baseline)

    methods = {
        "scPTR_gamma": gamma,
        "raw_u_s_ratio": raw_ratio,
        "unspliced_only": u,
        "expression": expr_full,
    }

    results = []

    for cluster_name in clusters.unique():
        mask = (clusters == cluster_name).values
        n_cells = mask.sum()
        if n_cells < 50:
            continue

        # Pre-compute expression PCA for this cluster (used for all methods)
        expr_sub = expr_full[mask]
        nonzero_expr = (expr_sub > 0).mean(axis=0)
        good_expr = nonzero_expr >= 0.05
        if good_expr.sum() < 20:
            continue
        n_expr_pcs = min(15, n_cells - 1, good_expr.sum() - 1)
        pca_expr = PCA(n_components=n_expr_pcs, random_state=42)
        expr_pcs = pca_expr.fit_transform(expr_sub[:, good_expr])

        for method_name, data in methods.items():
            data_sub = data[mask]

            # Filter to informative features
            nonzero = (data_sub > 0).mean(axis=0)
            good = nonzero >= 0.05
            if good.sum() < 20:
                continue
            data_filtered = data_sub[:, good]

            n_pcs = min(15, n_cells - 1, data_filtered.shape[1] - 1)
            pca = PCA(n_components=n_pcs, random_state=42)
            pcs = pca.fit_transform(data_filtered)

            best_sil = -1
            best_labels = None
            for k in [2, 3]:
                if n_cells < k * 10:
                    continue
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(pcs)
                if min(np.bincount(labels)) < 10:
                    continue
                sil = silhouette_score(pcs, labels)
                if sil > best_sil:
                    best_sil = sil
                    best_labels = labels

            if best_labels is None:
                continue

            # Compute silhouette of SAME labels in expression PCA space
            sil_expr = silhouette_score(expr_pcs, best_labels)
            invisibility = best_sil - sil_expr

            results.append({
                "cluster": cluster_name,
                "method": method_name,
                "n_cells": n_cells,
                "sil_method_space": best_sil,
                "sil_expr_space": sil_expr,
                "invisibility": invisibility,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(res_dir / f"ablation_{name}.csv", index=False)

    # Summary: mean invisibility by method
    print("\n  Mean invisibility score by method (higher = better):")
    summary = results_df.groupby("method")["invisibility"].agg(["mean", "std", "count"])
    for method, row in summary.sort_values("mean", ascending=False).iterrows():
        print(f"    {method:<20s}: {row['mean']:.4f} +/- {row['std']:.4f} "
              f"(n={int(row['count'])})")

    print("\n  Mean silhouette in method-space vs expression-space:")
    for method in ["scPTR_gamma", "raw_u_s_ratio", "unspliced_only", "expression"]:
        sub = results_df[results_df["method"] == method]
        if len(sub) == 0:
            continue
        print(f"    {method:<20s}: sil_method={sub['sil_method_space'].mean():.4f}, "
              f"sil_expr={sub['sil_expr_space'].mean():.4f}, "
              f"invisibility={sub['invisibility'].mean():.4f}")

    # Figure: Panel A (invisibility bars) + Panel B (per-cluster heatmap)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Mean invisibility by method
    methods_order = ["expression", "unspliced_only", "raw_u_s_ratio", "scPTR_gamma"]
    method_labels = ["Expression\n(baseline)", "Unspliced\nonly", "Raw u/s\nratio", "scPTR\ngamma"]
    colors = ["gray", "lightblue", "orange", "steelblue"]
    positions = np.arange(len(methods_order))

    means = []
    stds = []
    for m in methods_order:
        sub = results_df[results_df["method"] == m]["invisibility"]
        means.append(sub.mean() if len(sub) > 0 else 0)
        stds.append(sub.std() if len(sub) > 0 else 0)

    bars = axes[0].bar(positions, means, 0.6, yerr=stds, color=colors,
                       edgecolor="black", linewidth=0.5, capsize=3)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(method_labels, fontsize=10)
    axes[0].set_ylabel("Mean Invisibility Score\n(sil_method - sil_expr)")
    axes[0].set_title(f"A: Invisibility Score ({name})")
    axes[0].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[0].text(i, m + s + 0.005, f"{m:.3f}", ha="center", fontsize=9)

    # Panel B: Per-cluster invisibility heatmap
    pivot = results_df.pivot_table(
        index="cluster", columns="method", values="invisibility", aggfunc="mean"
    )
    if len(pivot) > 0:
        # Reorder columns
        col_order = [m for m in methods_order if m in pivot.columns]
        pivot = pivot[col_order]

        im = axes[1].imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                            vmin=-0.3, vmax=0.3)
        axes[1].set_xticks(np.arange(len(col_order)))
        axes[1].set_xticklabels([m.replace("_", "\n") for m in col_order], fontsize=8)
        axes[1].set_yticks(np.arange(len(pivot.index)))
        axes[1].set_yticklabels(pivot.index, fontsize=8)
        axes[1].set_title(f"B: Per-cluster Invisibility ({name})")

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(col_order)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    axes[1].text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=7, color="white" if abs(val) > 0.15 else "black")

        plt.colorbar(im, ax=axes[1], label="Invisibility", shrink=0.8)

    fig.suptitle(f"Ablation: Invisibility Score Analysis ({name})", fontsize=13)
    fig.tight_layout()
    save_fig(fig, f"ablation_{name}")

    return results_df


# =========================================================================
# T2-2: TF score discrepancy
# =========================================================================
def explain_tf_discrepancy(datasets):
    """Investigate why TF score varies dramatically across datasets."""
    print(f"\n{'='*60}")
    print(f"T2-2: TF SCORE DISCREPANCY")
    print(f"{'='*60}")

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    for name, adata in datasets.items():
        gamma = adata.layers["gamma"]
        u = adata.layers.get("Mu", adata.layers.get("unspliced"))
        u_arr = u.toarray() if hasattr(u, 'toarray') else np.asarray(u)

        tf = adata.var["tf_score"].values
        nonzero_frac = (gamma > 0).mean(axis=0)
        u_detection = (u_arr > 0).mean(axis=0)

        print(f"\n  {name}:")
        print(f"    TF score: median={np.median(tf):.4f}, mean={np.mean(tf):.4f}")
        print(f"    Unspliced detection rate: median={np.median(u_detection):.4f}")
        print(f"    Gamma nonzero fraction: median={np.median(nonzero_frac):.4f}")

        # Key insight: when gamma=0 for a gene, log1p(gamma)=0 → Var(log1p(gamma))=0
        # → TF = Var(log1p(u)) / (Var(log1p(u)) + 0) = 1.0
        # When gamma is nonzero, its variance dominates → TF ≈ 0
        n_zero_gamma = (np.median(gamma, axis=0) == 0).sum()
        tf_for_nonzero = tf[np.median(gamma, axis=0) > 0]
        tf_for_zero = tf[np.median(gamma, axis=0) == 0]
        print(f"    Genes with zero median gamma: {n_zero_gamma}/{len(tf)}")
        print(f"    TF score for zero-gamma genes: {np.median(tf_for_zero):.4f}")
        print(f"    TF score for nonzero-gamma genes: {np.median(tf_for_nonzero):.4f}")

        # Correlation between unspliced detection and TF score
        r, p = stats.spearmanr(u_detection, nonzero_frac)
        print(f"    Corr(u_detection, gamma_nonzero): r={r:.4f}")

    print(f"\n  EXPLANATION:")
    print(f"  The TF score discrepancy is a data sparsity artifact:")
    print(f"  - In 10x data (pancreas, DG), most genes have very sparse unspliced")
    print(f"    counts, leading to gamma=0 for most cells → Var(log1p(gamma))≈0")
    print(f"    → TF=1.0 (trivially) for those genes.")
    print(f"  - In sci-fate, the new/old mapping produces dense 'unspliced' counts")
    print(f"    → gamma is nonzero for most genes → TF reflects real biology.")
    print(f"  - FIX: Report TF scores only for gamma-informative genes (>=10% nonzero).")


# =========================================================================
# T2-3: Housekeeping gene analysis
# =========================================================================
def housekeeping_analysis(datasets):
    """Show cross-dataset consistency improves for housekeeping genes."""
    print(f"\n{'='*60}")
    print(f"T2-3: HOUSEKEEPING GENE ANALYSIS")
    print(f"{'='*60}")

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Curated list of housekeeping genes (common across species)
    hk_genes = [
        "ACTB", "GAPDH", "TUBB", "HSP90AB1", "LDHA", "PPIA", "RPL13A",
        "RPS18", "EEF1A1", "UBC", "B2M", "TUBA1B", "ENO1", "PKM",
        "YWHAZ", "HNRNPA1", "NPM1", "HSPA8", "EIF4A1", "ATP5F1B",
        "NONO", "SNRPD2", "SRSF3", "DDX5", "HNRNPC", "HNRNPU",
        "SF3B1", "RPL3", "RPL7", "RPS3", "RPS6", "RPL4", "RPL5",
        "RPS2", "RPL8", "RPS4X", "RPL11", "RPL13", "RPL18",
        "RPL27", "RPS5", "RPS7", "RPS8", "RPS14", "RPS15A",
        "RPS19", "RPS24", "RPS27A", "RPL6", "RPL9", "RPL10",
    ]
    hk_set = set(g.upper() for g in hk_genes)

    # Compute per-dataset median gamma
    medians = {}
    for name, adata in datasets.items():
        gamma = adata.layers["gamma"]
        med = pd.Series(np.median(gamma, axis=0), index=adata.var_names)
        medians[name] = med

    # Pairwise correlation: all genes vs housekeeping only
    names = sorted(datasets.keys())
    print("\n  Cross-dataset consistency:")
    print(f"  {'Pair':<30s} {'All genes':>12s} {'Housekeeping':>14s} {'Improvement':>12s}")
    print(f"  {'-'*68}")

    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            # Case-insensitive matching
            map_a = {g.upper(): g for g in medians[name_a].index if isinstance(g, str)}
            map_b = {g.upper(): g for g in medians[name_b].index if isinstance(g, str)}

            # All shared genes
            shared_upper = set(map_a.keys()) & set(map_b.keys())
            ga_all = np.array([medians[name_a][map_a[u]] for u in shared_upper])
            gb_all = np.array([medians[name_b][map_b[u]] for u in shared_upper])
            valid = np.isfinite(ga_all) & np.isfinite(gb_all)
            r_all, _ = stats.spearmanr(ga_all[valid], gb_all[valid])

            # Housekeeping genes only
            shared_hk = shared_upper & hk_set
            if len(shared_hk) >= 5:
                ga_hk = np.array([medians[name_a][map_a[u]] for u in shared_hk])
                gb_hk = np.array([medians[name_b][map_b[u]] for u in shared_hk])
                valid_hk = np.isfinite(ga_hk) & np.isfinite(gb_hk)
                if valid_hk.sum() >= 5:
                    r_hk, _ = stats.spearmanr(ga_hk[valid_hk], gb_hk[valid_hk])
                else:
                    r_hk = np.nan
            else:
                r_hk = np.nan

            pair = f"{name_a} vs {name_b}"
            improvement = r_hk - r_all if not np.isnan(r_hk) else np.nan
            print(f"  {pair:<30s} {r_all:>12.4f} {r_hk:>14.4f} "
                  f"{'':>2s}{'+' if improvement > 0 else ''}{improvement:.4f}")


# =========================================================================
# MAIN
# =========================================================================
def main():
    set_figure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and process datasets
    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    adata_pan = scptr.datasets.pancreas()
    adata_pan = run_pipeline(adata_pan, "pancreas")

    adata_dg = scptr.datasets.dentate_gyrus()
    adata_dg = run_pipeline(adata_dg, "dentate_gyrus")

    datasets = {"pancreas": adata_pan, "dentate_gyrus": adata_dg}

    # T1-2: Fix gamma reporting
    gamma_stats = {}
    for name, adata in datasets.items():
        gamma_stats[name] = fix_gamma_reporting(adata, name)

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "gamma_reporting.json", "w") as f:
        json.dump(gamma_stats, f, indent=2)

    # T1-1: Functional characterization
    for name, adata in datasets.items():
        invis_df = characterize_invisible_states(adata, name)
        if len(invis_df) > 0:
            invis_df.to_csv(res_dir / f"invisible_states_{name}.csv", index=False)

    # T1-3: Destabilizing bias
    for name, adata in datasets.items():
        investigate_destabilizing_bias(adata, name)

    # T1-4: Velocity streamlines
    for name, adata in datasets.items():
        velocity_streamlines(adata, name)

    # T2-1: Ablation
    for name, adata in datasets.items():
        ablation_experiments(adata, name)

    # T2-2: TF discrepancy
    # Also load sci-fate for comparison
    from run_scifate import load_scifate_data, prepare_for_scptr
    adata_sf_raw = load_scifate_data()
    adata_sf = prepare_for_scptr(adata_sf_raw)
    adata_sf = run_pipeline(adata_sf, "scifate")
    all_datasets = {**datasets, "scifate": adata_sf}
    explain_tf_discrepancy(all_datasets)

    # T2-3: Housekeeping genes
    housekeeping_analysis(all_datasets)

    print(f"\n{'='*60}")
    print("ALL TIER 1/2 FIXES COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
