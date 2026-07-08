#!/usr/bin/env python
"""Fill research plan gaps: expression-invisible states, RNA velocity comparison,
and network inference on real data.

Gap 1 (Aim 2): Formally demonstrate expression-invisible PT states
Gap 2 (Aim 3): Compare PT velocity with scvelo RNA velocity
Gap 3 (Aim 4): Run RBP network inference on real data
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

sys.path.insert(0, str(Path(__file__).parent))
from _common import set_figure_style

import scptr

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "gap_analysis"


def save_fig(fig, name, subdir="figures"):
    if fig is None:
        print(f"  [WARNING] {name}: None, skipping")
        return
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def process_dataset(name):
    """Load and run full preprocessing + core analysis on a dataset."""
    print(f"\nLoading {name}...")
    if name == "pancreas":
        adata = scptr.datasets.pancreas()
    else:
        adata = scptr.datasets.dentate_gyrus()

    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)
    scptr.tl.variance_decomposition(adata)
    scptr.tl.pt_states(adata)
    scptr.tl.pt_velocity(adata)
    print(f"  {name}: {adata.n_obs} cells, {adata.n_vars} genes, "
          f"{adata.obs['pt_state'].nunique()} PT states")
    return adata


# =========================================================================
# GAP 1: Expression-invisible PT states (Aim 2 central claim)
# =========================================================================
def run_invisible_states(adata, dataset_name):
    """Formally demonstrate that gamma clustering reveals sub-populations
    invisible to expression-based clustering.

    Method:
    1. For each expression cluster, extract cells
    2. Re-cluster using gamma profiles (sub-clustering)
    3. Test significance via silhouette score and ANOVA on gamma PCs
    4. Characterize differentially stabilized genes in sub-clusters
    """
    print("\n" + "=" * 60)
    print(f"GAP 1: EXPRESSION-INVISIBLE STATES ({dataset_name})")
    print("=" * 60)

    res_dir = OUTPUT_DIR / "results" / "invisible_states" / dataset_name
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_prefix = f"invisible_states/{dataset_name}"

    gamma = scptr.tools._gamma  # just for access to layer
    gamma_mat = adata.layers["gamma"]
    clusters = adata.obs["clusters"].astype(str)

    results = []

    for cluster_name in sorted(clusters.unique()):
        mask = (clusters == cluster_name).values
        n_cells = mask.sum()

        if n_cells < 50:  # need enough cells for sub-clustering
            print(f"  {cluster_name}: {n_cells} cells (too few, skipping)")
            continue

        # Extract gamma for this cluster
        gamma_sub = gamma_mat[mask]

        # PCA on gamma within this cluster
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        n_pcs = min(15, n_cells - 1, gamma_sub.shape[1] - 1)
        pca = PCA(n_components=n_pcs, random_state=42)
        gamma_pcs = pca.fit_transform(gamma_sub)

        # Try 2-4 sub-clusters, pick best silhouette
        best_k = 1
        best_sil = -1
        best_labels = np.zeros(n_cells, dtype=int)

        for k in [2, 3]:
            if n_cells < k * 10:
                continue
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(gamma_pcs)
            # Only evaluate if all clusters have >= 10 cells
            min_size = min(np.bincount(labels))
            if min_size < 10:
                continue
            sil = silhouette_score(gamma_pcs, labels)
            if sil > best_sil:
                best_sil = sil
                best_k = k
                best_labels = labels

        # Statistical test: MANOVA-like test using gamma PCs
        # Use ANOVA on first few PCs as a proxy
        if best_k > 1:
            p_values_pcs = []
            for pc in range(min(5, n_pcs)):
                groups = [gamma_pcs[best_labels == j, pc] for j in range(best_k)]
                if all(len(g) >= 2 for g in groups):
                    f_stat, p_val = stats.f_oneway(*groups)
                    p_values_pcs.append(p_val)
            # Combine p-values (Fisher's method)
            if p_values_pcs:
                # Clamp p-values to avoid log(0)
                p_clamped = [max(p, 1e-300) for p in p_values_pcs]
                combined_stat = -2 * sum(np.log(p) for p in p_clamped)
                from scipy.stats import chi2
                combined_p = 1 - chi2.cdf(combined_stat, 2 * len(p_clamped))
            else:
                combined_p = 1.0
        else:
            combined_p = 1.0

        # Now test if these sub-clusters are visible in expression space
        # Use expression PCA and compute silhouette for the SAME labels
        expr_sub = adata.X[mask] if not hasattr(adata.X, 'toarray') else adata.X[mask].toarray()
        n_expr_pcs = min(15, n_cells - 1, expr_sub.shape[1] - 1)
        pca_expr = PCA(n_components=n_expr_pcs, random_state=42)
        expr_pcs = pca_expr.fit_transform(expr_sub)

        if best_k > 1:
            sil_gamma = best_sil
            sil_expr = silhouette_score(expr_pcs, best_labels)
        else:
            sil_gamma = 0
            sil_expr = 0

        # Find differentially degraded genes between sub-clusters
        top_genes = []
        if best_k > 1:
            median_gamma_by_sub = np.zeros((best_k, gamma_sub.shape[1]))
            for j in range(best_k):
                median_gamma_by_sub[j] = np.median(gamma_sub[best_labels == j], axis=0)
            # Max fold change across sub-clusters
            max_gamma = np.max(median_gamma_by_sub, axis=0)
            min_gamma = np.minimum(np.min(median_gamma_by_sub, axis=0), 1e-6)
            fold_change = max_gamma / np.clip(min_gamma, 1e-6, None)
            # Filter to genes with nonzero gamma
            nonzero_mask = max_gamma > 0.01
            if nonzero_mask.sum() > 0:
                fc_masked = fold_change.copy()
                fc_masked[~nonzero_mask] = 0
                top_idx = np.argsort(fc_masked)[::-1][:20]
                top_genes = [adata.var_names[i] for i in top_idx if fc_masked[i] > 1.5]

        result = {
            "cluster": cluster_name,
            "n_cells": int(n_cells),
            "n_subclusters": int(best_k),
            "silhouette_gamma": float(sil_gamma),
            "silhouette_expr": float(sil_expr),
            "invisibility_score": float(sil_gamma - sil_expr),
            "combined_p": float(combined_p),
            "top_diff_genes": top_genes[:10],
        }
        results.append(result)

        status = "INVISIBLE" if sil_gamma > 0.1 and sil_expr < 0.1 else \
                 "PARTIALLY" if sil_gamma > sil_expr + 0.05 else "VISIBLE"
        print(f"  {cluster_name}: {n_cells} cells, k={best_k}, "
              f"sil_gamma={sil_gamma:.3f}, sil_expr={sil_expr:.3f}, "
              f"p={combined_p:.2e} [{status}]")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(res_dir / "invisible_states.csv", index=False)

    # Summary figure: silhouette in gamma vs expression space
    if len(results_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: paired bar chart
        x = np.arange(len(results_df))
        width = 0.35
        axes[0].bar(x - width/2, results_df["silhouette_gamma"], width,
                     label="Gamma space", color="steelblue")
        axes[0].bar(x + width/2, results_df["silhouette_expr"], width,
                     label="Expression space", color="salmon")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(results_df["cluster"], rotation=45, ha="right")
        axes[0].set_ylabel("Silhouette score")
        axes[0].set_title("Sub-cluster separation: Gamma vs Expression")
        axes[0].legend()
        axes[0].axhline(0, color="gray", linestyle="--", alpha=0.3)

        # Right: invisibility score
        colors = ["steelblue" if v > 0.05 else "gray"
                  for v in results_df["invisibility_score"]]
        axes[1].barh(results_df["cluster"], results_df["invisibility_score"],
                      color=colors)
        axes[1].set_xlabel("Invisibility score (sil_gamma - sil_expr)")
        axes[1].set_title("Expression-invisible PT sub-states")
        axes[1].axvline(0, color="gray", linestyle="--", alpha=0.3)

        fig.suptitle(f"Expression-Invisible States: {dataset_name}", fontsize=13, y=1.02)
        fig.tight_layout()
        save_fig(fig, f"invisible_states_{dataset_name}", f"figures/invisible_states")

    return results_df


# =========================================================================
# GAP 2: RNA velocity comparison (Aim 3)
# =========================================================================
def run_velocity_comparison(adata, dataset_name):
    """Compare PT velocity with scvelo RNA velocity on the same dataset.

    Shows:
    1. Side-by-side velocity embeddings
    2. Correlation of velocity magnitudes
    3. Angular agreement between velocity fields
    """
    print("\n" + "=" * 60)
    print(f"GAP 2: RNA VELOCITY COMPARISON ({dataset_name})")
    print("=" * 60)

    res_dir = OUTPUT_DIR / "results" / "velocity_comparison" / dataset_name
    res_dir.mkdir(parents=True, exist_ok=True)

    import scvelo as scv

    # Run scvelo RNA velocity
    print("  Running scvelo RNA velocity...")
    # scvelo needs its own preprocessing
    adata_scv = adata.copy()

    # scvelo pipeline
    scv.pp.filter_and_normalize(adata_scv, min_shared_counts=20)
    scv.pp.moments(adata_scv, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(adata_scv)

    # Project scvelo velocity onto the gamma UMAP for fair comparison
    # Use the gamma UMAP coordinates from scPTR
    if "X_gamma_umap" in adata.obsm:
        adata_scv.obsm["X_gamma_umap"] = adata.obsm["X_gamma_umap"]

    # Compute UMAP for scvelo data
    sc.tl.umap(adata_scv)

    # Get velocity vectors
    scv_velocity = adata_scv.layers.get("velocity")
    pt_velocity = adata.layers.get("pt_velocity")

    if scv_velocity is None:
        print("  [WARNING] scvelo velocity not computed, skipping comparison")
        return

    print(f"  scvelo velocity shape: {scv_velocity.shape}")
    print(f"  PT velocity shape: {pt_velocity.shape}")

    # Find shared genes
    shared_genes = adata.var_names.intersection(adata_scv.var_names)
    print(f"  Shared genes: {len(shared_genes)}")

    # Compare velocity magnitudes per cell
    # Use scvelo's gene set for fair comparison
    scv_genes = adata_scv.var_names
    scv_gene_idx_in_adata = [list(adata.var_names).index(g)
                              for g in scv_genes if g in adata.var_names]
    pt_vel_shared = pt_velocity[:, scv_gene_idx_in_adata]
    scv_vel_shared_genes = [g for g in scv_genes if g in adata.var_names]
    scv_vel_idx = [list(adata_scv.var_names).index(g) for g in scv_vel_shared_genes]
    scv_vel_shared = scv_velocity[:, scv_vel_idx]

    # Handle NaN in scvelo
    scv_vel_shared = np.nan_to_num(scv_vel_shared, 0)

    # Per-cell velocity magnitude
    pt_mag = np.linalg.norm(pt_vel_shared, axis=1)
    scv_mag = np.linalg.norm(scv_vel_shared, axis=1)

    # Cosine similarity per cell
    dot_product = np.sum(pt_vel_shared * scv_vel_shared, axis=1)
    norms = pt_mag * scv_mag
    norms = np.clip(norms, 1e-10, None)
    cosine_sim = dot_product / norms

    # Filter to cells with nonzero velocity in both
    valid = (pt_mag > 1e-6) & (scv_mag > 1e-6)
    print(f"  Cells with nonzero velocity in both: {valid.sum()}/{len(valid)}")

    if valid.sum() > 10:
        mag_corr, mag_p = stats.spearmanr(pt_mag[valid], scv_mag[valid])
        mean_cosine = np.mean(cosine_sim[valid])
        print(f"  Magnitude Spearman r = {mag_corr:.4f} (p={mag_p:.2e})")
        print(f"  Mean cosine similarity = {mean_cosine:.4f}")
    else:
        mag_corr = np.nan
        mean_cosine = np.nan

    # Save results
    results = {
        "n_shared_genes": len(scv_vel_shared_genes),
        "n_cells_both_nonzero": int(valid.sum()),
        "magnitude_spearman_r": float(mag_corr) if not np.isnan(mag_corr) else None,
        "mean_cosine_similarity": float(mean_cosine) if not np.isnan(mean_cosine) else None,
    }
    with open(res_dir / "velocity_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Figure: 2x2 panel
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: scvelo velocity on scvelo UMAP
    coords_scv = adata_scv.obsm.get("X_umap")
    if coords_scv is not None:
        axes[0, 0].scatter(coords_scv[:, 0], coords_scv[:, 1],
                           c=scv_mag, cmap="YlOrRd", s=3, alpha=0.5,
                           vmax=np.percentile(scv_mag, 95))
        axes[0, 0].set_title("RNA Velocity magnitude (scvelo UMAP)")
        axes[0, 0].set_xlabel("UMAP 1")
        axes[0, 0].set_ylabel("UMAP 2")

    # Top-right: PT velocity on gamma UMAP
    coords_gamma = adata.obsm.get("X_gamma_umap")
    if coords_gamma is not None:
        axes[0, 1].scatter(coords_gamma[:, 0], coords_gamma[:, 1],
                           c=pt_mag, cmap="YlOrRd", s=3, alpha=0.5,
                           vmax=np.percentile(pt_mag, 95))
        axes[0, 1].set_title("PT Velocity magnitude (gamma UMAP)")
        axes[0, 1].set_xlabel("UMAP 1")
        axes[0, 1].set_ylabel("UMAP 2")

    # Bottom-left: magnitude correlation
    if valid.sum() > 10:
        axes[1, 0].scatter(scv_mag[valid], pt_mag[valid], alpha=0.1, s=3, c="steelblue")
        axes[1, 0].set_xlabel("RNA velocity magnitude")
        axes[1, 0].set_ylabel("PT velocity magnitude")
        axes[1, 0].set_title(f"Magnitude correlation (r={mag_corr:.3f})")

    # Bottom-right: cosine similarity distribution
    if valid.sum() > 10:
        axes[1, 1].hist(cosine_sim[valid], bins=50, color="steelblue",
                        alpha=0.8, edgecolor="white")
        axes[1, 1].axvline(mean_cosine, color="red", linestyle="--",
                           label=f"Mean={mean_cosine:.3f}")
        axes[1, 1].set_xlabel("Cosine similarity (PT vel vs RNA vel)")
        axes[1, 1].set_ylabel("Number of cells")
        axes[1, 1].set_title("Directional agreement")
        axes[1, 1].legend()

    fig.suptitle(f"PT Velocity vs RNA Velocity: {dataset_name}", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, f"velocity_comparison_{dataset_name}", "figures/velocity_comparison")

    return results


# =========================================================================
# GAP 3: Network inference on real data (Aim 4)
# =========================================================================
def run_network_inference(adata, dataset_name):
    """Run RBP-target network inference on real data.

    Identifies RBPs whose expression correlates with target gene gamma shifts.
    """
    print("\n" + "=" * 60)
    print(f"GAP 3: NETWORK INFERENCE ({dataset_name})")
    print("=" * 60)

    res_dir = OUTPUT_DIR / "results" / "network" / dataset_name
    res_dir.mkdir(parents=True, exist_ok=True)

    # Get known RBPs that are expressed in this dataset
    known_rbps = scptr.tl.list_known_rbps(organism="mouse")
    rbp_genes = [g for g in known_rbps if g in adata.var_names]
    print(f"  Known RBPs in dataset: {len(rbp_genes)}/{len(known_rbps)}")

    if len(rbp_genes) < 5:
        print("  Too few RBPs, skipping network inference")
        return

    # Get top differentially degraded genes as targets
    gamma = adata.layers["gamma"]
    gamma_var = np.var(gamma, axis=0)
    # Use top 500 most variable gamma genes as targets
    top_targets_idx = np.argsort(gamma_var)[::-1][:500]
    target_genes = [adata.var_names[i] for i in top_targets_idx
                    if gamma_var[i] > 0 and adata.var_names[i] not in rbp_genes]
    target_genes = target_genes[:200]
    print(f"  Target genes (top variable gamma): {len(target_genes)}")

    # For each cell type, compute correlation between RBP expression and
    # target gene gamma
    clusters = adata.obs["clusters"].astype(str)
    all_edges = []

    for cluster_name in sorted(clusters.unique()):
        mask = (clusters == cluster_name).values
        n_cells = mask.sum()
        if n_cells < 30:
            continue

        # Get expression of RBPs in this cluster
        rbp_idx = [list(adata.var_names).index(g) for g in rbp_genes]
        if hasattr(adata.X, 'toarray'):
            rbp_expr = adata.X[mask][:, rbp_idx].toarray()
        else:
            rbp_expr = adata.X[mask][:, rbp_idx]

        # Get gamma of target genes
        target_idx = [list(adata.var_names).index(g) for g in target_genes]
        target_gamma = gamma[mask][:, target_idx]

        # Correlation: RBP expression vs target gamma
        for ri, rbp in enumerate(rbp_genes):
            rbp_x = rbp_expr[:, ri]
            if np.std(rbp_x) < 1e-6:
                continue

            for ti, target in enumerate(target_genes):
                target_g = target_gamma[:, ti]
                if np.std(target_g) < 1e-6:
                    continue

                r, p = stats.spearmanr(rbp_x, target_g)
                if abs(r) > 0.2 and p < 0.01:
                    all_edges.append({
                        "cluster": cluster_name,
                        "rbp": rbp,
                        "target": target,
                        "spearman_r": float(r),
                        "p_value": float(p),
                        "direction": "stabilizing" if r < 0 else "destabilizing",
                    })

    edges_df = pd.DataFrame(all_edges)
    if len(edges_df) > 0:
        # Multiple testing correction (Benjamini-Hochberg)
        from statsmodels.stats.multitest import multipletests
        _, edges_df["fdr"], _, _ = multipletests(edges_df["p_value"], method="fdr_bh")
        edges_df = edges_df[edges_df["fdr"] < 0.05].copy()

    edges_df.to_csv(res_dir / "network_edges.csv", index=False)
    print(f"  Significant edges (FDR<0.05): {len(edges_df)}")

    if len(edges_df) > 0:
        # Top RBP hubs
        hub_counts = edges_df.groupby("rbp").size().sort_values(ascending=False)
        print(f"\n  Top RBP hubs:")
        for rbp, count in hub_counts.head(15).items():
            n_stab = len(edges_df[(edges_df["rbp"] == rbp) & (edges_df["direction"] == "stabilizing")])
            n_dest = len(edges_df[(edges_df["rbp"] == rbp) & (edges_df["direction"] == "destabilizing")])
            print(f"    {rbp}: {count} targets ({n_stab} stabilizing, {n_dest} destabilizing)")

        hub_counts.head(30).to_csv(res_dir / "rbp_hub_counts.csv")

        # Network summary figure
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: top RBP hubs
        top_hubs = hub_counts.head(20)
        colors = ["steelblue" if h > hub_counts.median() else "lightblue"
                  for h in top_hubs.values]
        axes[0].barh(range(len(top_hubs)), top_hubs.values, color=colors)
        axes[0].set_yticks(range(len(top_hubs)))
        axes[0].set_yticklabels(top_hubs.index)
        axes[0].set_xlabel("Number of target genes")
        axes[0].set_title("Top RBP Regulators")
        axes[0].invert_yaxis()

        # Right: effect size distribution
        axes[1].hist(edges_df["spearman_r"], bins=40, color="steelblue",
                     alpha=0.8, edgecolor="white")
        axes[1].axvline(0, color="red", linestyle="--", alpha=0.5)
        n_stab = (edges_df["direction"] == "stabilizing").sum()
        n_dest = (edges_df["direction"] == "destabilizing").sum()
        axes[1].set_xlabel("Spearman correlation (RBP expr vs target gamma)")
        axes[1].set_ylabel("Number of edges")
        axes[1].set_title(f"Edge effects: {n_stab} stabilizing, {n_dest} destabilizing")

        fig.suptitle(f"RBP-Target Network: {dataset_name}", fontsize=13, y=1.02)
        fig.tight_layout()
        save_fig(fig, f"network_{dataset_name}", "figures/network")

    return edges_df


# =========================================================================
# MAIN
# =========================================================================
def main():
    set_figure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process both datasets
    panc = process_dataset("pancreas")
    dg = process_dataset("dentate_gyrus")

    # GAP 1: Expression-invisible states
    invis_panc = run_invisible_states(panc, "pancreas")
    invis_dg = run_invisible_states(dg, "dentate_gyrus")

    # GAP 2: RNA velocity comparison
    vel_panc = run_velocity_comparison(panc, "pancreas")
    vel_dg = run_velocity_comparison(dg, "dentate_gyrus")

    # GAP 3: Network inference
    net_panc = run_network_inference(panc, "pancreas")
    net_dg = run_network_inference(dg, "dentate_gyrus")

    # Summary
    print("\n" + "=" * 60)
    print("GAP ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nAll results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
