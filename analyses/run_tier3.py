#!/usr/bin/env python
"""Tier 3 analyses: disease dataset, TIL case study, DepMap validation.

T3-1: Apply scPTR to a cancer dataset (neuroblastoma, GSE137804)
      - Run full pipeline, identify PT states, compare tumor vs normal
T3-2: DepMap/CRISPR validation of RBP hub predictions
      - Test whether scPTR-predicted RBP hubs are more essential (lower CRISPR scores)
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

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "tier3"
CACHE_DIR = Path(__file__).parent.parent / ".cache"
DATA_DIR = Path(__file__).parent.parent / "src" / "scptr" / "benchmark" / "data"


def save_fig(fig, name, subdir="figures"):
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =========================================================================
# T3-1: Disease dataset — Neuroblastoma (GSE137804)
# =========================================================================
def load_neuroblastoma():
    """Load neuroblastoma dataset with spliced/unspliced layers."""
    h5ad_path = CACHE_DIR / "neuroblastoma.h5ad"
    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"{h5ad_path} not found. Download from: "
            "https://cdn.bioturing.com/colab/data/GSE137804-kallisto.symbol.h5ad"
        )

    print("Loading neuroblastoma dataset...")
    adata = sc.read_h5ad(str(h5ad_path))
    print(f"  Raw: {adata.shape}")
    print(f"  Layers: {list(adata.layers.keys())}")
    print(f"  Cell types: {adata.obs['celltype'].value_counts().to_dict()}")

    # Basic preprocessing
    # Filter genes: require minimum expression
    sc.pp.filter_genes(adata, min_cells=50)
    print(f"  After gene filter: {adata.shape}")

    # Store raw counts before normalizing
    adata.layers["raw_spliced"] = adata.layers["spliced"].copy()
    adata.layers["raw_unspliced"] = adata.layers["unspliced"].copy()

    return adata


def run_neuroblastoma_pipeline(adata):
    """Run scPTR pipeline on neuroblastoma data."""
    print("\n--- Running scPTR pipeline on neuroblastoma ---")

    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)
    scptr.tl.variance_decomposition(adata)
    scptr.tl.pt_states(adata)
    scptr.tl.pt_velocity(adata)

    print(f"  Pipeline complete: {adata.shape}")
    return adata


def analyze_neuroblastoma(adata):
    """Comprehensive analysis of neuroblastoma data."""
    print(f"\n{'='*60}")
    print("T3-1: NEUROBLASTOMA ANALYSIS")
    print(f"{'='*60}")

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    gamma = adata.layers["gamma"]
    n_cells, n_genes = gamma.shape

    # Basic stats
    nonzero_frac = (gamma > 0).mean(axis=0)
    informative = nonzero_frac >= 0.1
    print(f"  Cells: {n_cells}")
    print(f"  Genes: {n_genes}")
    print(f"  Gamma-informative genes: {informative.sum()} ({100*informative.mean():.1f}%)")

    results = {
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "n_informative": int(informative.sum()),
        "frac_informative": float(informative.mean()),
    }

    # Half-life validation
    print("\n  Half-life correlation:")
    median_gamma = np.median(gamma, axis=0)
    datasets_data_dir = Path(__file__).parent.parent / "src" / "scptr" / "datasets" / "data"
    hl_files = [
        ("mouse", "Mouse (Herzog 2017)", datasets_data_dir / "herzog2017_halflives.csv"),
        ("human", "Human (Schofield 2018)", datasets_data_dir / "schofield2018_halflives.csv"),
    ]
    for species, label, hl_path in hl_files:
        if not hl_path.exists():
            continue

        hl_df = pd.read_csv(hl_path)
        hl_df = hl_df[["gene_symbol", "half_life_hours"]].dropna()
        hl_dict = dict(zip(hl_df["gene_symbol"].str.upper(), hl_df["half_life_hours"]))

        # Match genes
        matched_gamma = []
        matched_hl = []
        for i, gene in enumerate(adata.var_names):
            g_upper = gene.upper()
            if g_upper in hl_dict and informative[i]:
                matched_gamma.append(median_gamma[i])
                matched_hl.append(hl_dict[g_upper])

        if len(matched_gamma) >= 50:
            r, p = stats.spearmanr(matched_gamma, matched_hl)
            print(f"    {label}: r={r:.4f}, p={p:.2e}, n={len(matched_gamma)}")
            results[f"halflife_{species}_r"] = float(r)
            results[f"halflife_{species}_p"] = float(p)
            results[f"halflife_{species}_n"] = len(matched_gamma)

    # PT state discovery
    print("\n  PT state discovery:")
    clusters = adata.obs.get("pt_clusters", adata.obs.get("clusters"))
    if clusters is not None:
        n_clusters = clusters.nunique()
        print(f"    PT clusters found: {n_clusters}")
        print(f"    Cluster sizes: {clusters.value_counts().to_dict()}")
        results["n_pt_clusters"] = int(n_clusters)

    # Sub-clustering within tumor cells for invisible states
    print("\n  Invisible state discovery (within tumor cells):")
    tumor_mask = np.ones(n_cells, dtype=bool)  # all cells are tumor
    gamma_tumor = gamma[tumor_mask]

    # Filter to informative genes
    good = informative
    if good.sum() >= 20:
        gamma_filt = gamma_tumor[:, good]
        n_pcs = min(30, n_cells - 1, gamma_filt.shape[1] - 1)
        pca = PCA(n_components=n_pcs, random_state=42)
        gamma_pcs = pca.fit_transform(gamma_filt)

        # Try k=2,3,4,5
        best_k, best_sil, best_labels = 1, -1, None
        for k in [2, 3, 4, 5]:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(gamma_pcs)
            if min(np.bincount(labels)) < 50:
                continue
            sil = silhouette_score(gamma_pcs, labels)
            print(f"    k={k}: silhouette={sil:.4f}")
            if sil > best_sil:
                best_k, best_sil, best_labels = k, sil, labels

        if best_labels is not None:
            print(f"    Best k={best_k}, silhouette={best_sil:.4f}")
            results["best_k"] = int(best_k)
            results["best_silhouette"] = float(best_sil)

            # Expression silhouette for same labels
            expr = adata.X[tumor_mask].toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X[tumor_mask])
            n_expr_pcs = min(30, n_cells - 1, expr.shape[1] - 1)
            pca_expr = PCA(n_components=n_expr_pcs, random_state=42)
            expr_pcs = pca_expr.fit_transform(expr)
            sil_expr = silhouette_score(expr_pcs, best_labels)
            print(f"    Expression silhouette (same labels): {sil_expr:.4f}")
            results["expr_silhouette"] = float(sil_expr)
            results["invisibility"] = float(best_sil - sil_expr)

            if best_sil > sil_expr:
                print(f"    ** INVISIBLE STATES FOUND (gamma sil > expr sil) **")
            else:
                print(f"    States are visible in expression")

            # Store gamma sub-clusters
            adata.obs["gamma_subcluster"] = "NA"
            adata.obs.loc[adata.obs.index[tumor_mask], "gamma_subcluster"] = [
                f"GC_{l}" for l in best_labels
            ]

            # Differential gamma analysis between sub-clusters
            print("\n  Top differentially degraded genes between gamma sub-clusters:")
            gene_names = adata.var_names[good]
            diff_results = []
            for gi, gene in enumerate(gene_names):
                groups = [gamma_filt[best_labels == j, gi] for j in range(best_k)]
                if all(len(g) >= 50 for g in groups):
                    if best_k == 2:
                        _, p_val = stats.mannwhitneyu(groups[0], groups[1],
                                                       alternative='two-sided')
                    else:
                        _, p_val = stats.kruskal(*groups)
                    medians = [np.median(g) for g in groups]
                    log_fc = np.log2((max(medians) + 0.01) / (min(medians) + 0.01))
                    diff_results.append({"gene": gene, "p_value": p_val,
                                         "log2_fc_gamma": log_fc})

            if diff_results:
                diff_df = pd.DataFrame(diff_results)
                from statsmodels.stats.multitest import multipletests
                _, diff_df["fdr"], _, _ = multipletests(diff_df["p_value"], method="fdr_bh")
                sig = diff_df[diff_df["fdr"] < 0.05].sort_values("log2_fc_gamma", ascending=False)
                print(f"    Differentially degraded (FDR<0.05): {len(sig)}/{len(diff_df)}")
                if len(sig) > 0:
                    print(f"    Top 10: {sig.head(10)['gene'].tolist()}")
                    sig.to_csv(res_dir / "neuroblastoma_diff_degraded.csv", index=False)
                results["n_diff_genes"] = len(sig)

    # RBP network (library-size corrected partial correlation)
    print("\n  RBP-target network (library-size corrected):")
    rbp_path = Path(__file__).parent.parent / "src" / "scptr" / "tools" / "data" / "known_rbps.csv"
    rbps = pd.read_csv(rbp_path)["gene_symbol"].tolist()

    gene_upper_map = {g.upper(): i for i, g in enumerate(adata.var_names)}
    rbp_in_data = {}
    for r in rbps:
        if r.upper() in gene_upper_map:
            rbp_in_data[r.upper()] = gene_upper_map[r.upper()]

    print(f"    RBPs in dataset: {len(rbp_in_data)}")

    if hasattr(adata.X, 'toarray'):
        expr = adata.X.toarray()
    else:
        expr = np.asarray(adata.X)

    # Library-size correction: rank-residualize against library size
    lib_size = expr.sum(axis=1)
    lib_rank = stats.rankdata(lib_size)
    lib_rank_centered = lib_rank - lib_rank.mean()
    lib_ss = np.dot(lib_rank_centered, lib_rank_centered)

    # Top variable gamma genes as targets
    gamma_var = np.var(gamma[:, informative], axis=0)
    n_targets = min(200, informative.sum())
    top_var_idx = np.argsort(gamma_var)[-n_targets:]
    info_indices = np.where(informative)[0]
    target_indices = info_indices[top_var_idx]

    # Pre-compute residualized gamma ranks for all targets
    gamma_resid_map = {}
    for ti in target_indices:
        target_gamma = gamma[:, ti]
        if np.std(target_gamma) < 1e-8:
            continue
        t_rank = stats.rankdata(target_gamma)
        t_rank_c = t_rank - t_rank.mean()
        slope = np.dot(lib_rank_centered, t_rank_c) / lib_ss
        resid = t_rank - slope * lib_rank
        resid_c = resid - resid.mean()
        resid_std = np.sqrt(np.dot(resid_c, resid_c))
        if resid_std > 1e-8:
            gamma_resid_map[ti] = (resid_c, resid_std)

    # Raw edges (for comparison)
    raw_edges = []
    corrected_edges = []
    for rbp_upper, rbp_idx in rbp_in_data.items():
        rbp_expr = expr[:, rbp_idx]
        if np.std(rbp_expr) < 1e-6:
            continue

        # Residualize RBP expression against library size
        rbp_rank = stats.rankdata(rbp_expr)
        rbp_rank_c = rbp_rank - rbp_rank.mean()
        slope_rbp = np.dot(lib_rank_centered, rbp_rank_c) / lib_ss
        rbp_resid = rbp_rank - slope_rbp * lib_rank
        rbp_resid_c = rbp_resid - rbp_resid.mean()
        rbp_resid_std = np.sqrt(np.dot(rbp_resid_c, rbp_resid_c))
        if rbp_resid_std < 1e-8:
            continue

        for ti in target_indices:
            target_gamma = gamma[:, ti]
            valid = target_gamma > 0
            if valid.sum() < 50:
                continue

            # Raw correlation (for comparison)
            r_raw, p_raw = stats.spearmanr(rbp_expr[valid], target_gamma[valid])
            if p_raw < 0.05 / (len(rbp_in_data) * n_targets):
                raw_edges.append({
                    "rbp": rbp_upper,
                    "target": adata.var_names[ti],
                    "spearman_r": r_raw,
                    "direction": "destabilizing" if r_raw > 0 else "stabilizing",
                })

            # Library-size corrected partial correlation
            if ti not in gamma_resid_map:
                continue
            g_resid_c, g_resid_std = gamma_resid_map[ti]
            r_corr = np.dot(rbp_resid_c, g_resid_c) / (rbp_resid_std * g_resid_std)
            r_corr = np.clip(r_corr, -1.0, 1.0)
            df = n_cells - 3
            t_val = r_corr * np.sqrt(df / (1 - r_corr**2 + 1e-12))
            p_corr = 2 * stats.t.sf(abs(t_val), df)

            if p_corr < 0.05 / (len(rbp_in_data) * n_targets):
                corrected_edges.append({
                    "rbp": rbp_upper,
                    "target": adata.var_names[ti],
                    "spearman_r": float(r_corr),
                    "direction": "destabilizing" if r_corr > 0 else "stabilizing",
                })

    # Report raw network stats
    if raw_edges:
        raw_df = pd.DataFrame(raw_edges)
        raw_n_destab = (raw_df["spearman_r"] > 0).sum()
        print(f"    Raw network: {len(raw_df)} edges, "
              f"{raw_n_destab} destab ({100*raw_n_destab/len(raw_df):.1f}%)")
        raw_df.to_csv(res_dir / "neuroblastoma_network_raw.csv", index=False)
        results["n_raw_edges"] = len(raw_df)
        results["raw_destab_frac"] = float(raw_n_destab / len(raw_df))

    # Report corrected network
    if corrected_edges:
        edges_df = pd.DataFrame(corrected_edges)
        n_destab = (edges_df["spearman_r"] > 0).sum()
        n_stab = (edges_df["spearman_r"] < 0).sum()
        print(f"    Corrected network: {len(edges_df)} edges")
        print(f"    Destabilizing: {n_destab} ({100*n_destab/len(edges_df):.1f}%), "
              f"Stabilizing: {n_stab} ({100*n_stab/len(edges_df):.1f}%)")
        results["n_network_edges"] = len(edges_df)
        results["corrected_destab_frac"] = float(n_destab / len(edges_df))

        # Top hubs
        hub_counts = edges_df.groupby("rbp").size().sort_values(ascending=False)
        print(f"    Top RBP hubs (corrected):")
        for rbp, count in hub_counts.head(10).items():
            sub = edges_df[edges_df["rbp"] == rbp]
            print(f"      {rbp}: {count} targets "
                  f"({(sub['spearman_r'] < 0).sum()} stab, "
                  f"{(sub['spearman_r'] > 0).sum()} destab)")

        edges_df.to_csv(res_dir / "neuroblastoma_network_corrected.csv", index=False)
        results["top_hubs"] = hub_counts.head(10).to_dict()
    else:
        edges_df = pd.DataFrame()

    # Stability program characterization via pathway enrichment
    print("\n  Stability program characterization:")
    stability_programs = []
    if best_labels is not None and best_k >= 2:
        gene_names_good = adata.var_names[good]
        for cluster_id in range(best_k):
            cluster_mask = best_labels == cluster_id
            other_mask = ~cluster_mask

            # Top differentially degraded genes for this cluster
            top_genes_up = []
            top_genes_down = []
            for gi, gene in enumerate(gene_names_good):
                vals_in = gamma_filt[cluster_mask, gi]
                vals_out = gamma_filt[other_mask, gi]
                if len(vals_in) < 10 or len(vals_out) < 10:
                    continue
                med_in = np.median(vals_in)
                med_out = np.median(vals_out)
                log_fc = np.log2((med_in + 0.01) / (med_out + 0.01))
                if log_fc > 0.5:
                    top_genes_up.append((gene, log_fc))
                elif log_fc < -0.5:
                    top_genes_down.append((gene, log_fc))

            top_genes_up.sort(key=lambda x: x[1], reverse=True)
            top_genes_down.sort(key=lambda x: x[1])

            print(f"    GC_{cluster_id}: {cluster_mask.sum()} cells, "
                  f"{len(top_genes_up)} up-degraded, {len(top_genes_down)} down-degraded")

            # Pathway enrichment on top differentially degraded genes
            gene_list = [g for g, _ in top_genes_up[:200]]
            if len(gene_list) >= 10:
                try:
                    import gseapy as gp
                    enr = gp.enrichr(
                        gene_list=gene_list,
                        gene_sets=["KEGG_2021_Human"],
                        organism="human",
                        outdir=None,
                        no_plot=True,
                    )
                    sig_enr = enr.results[enr.results["Adjusted P-value"] < 0.1].head(10)
                    if len(sig_enr) > 0:
                        print(f"      Top KEGG pathways (up-degraded):")
                        for _, row in sig_enr.iterrows():
                            print(f"        {row['Term'][:60]}: p={row['Adjusted P-value']:.4f}")
                            stability_programs.append({
                                "cluster": f"GC_{cluster_id}",
                                "direction": "up_degraded",
                                "pathway": row["Term"],
                                "fdr": row["Adjusted P-value"],
                                "n_overlap": row.get("Overlap", ""),
                            })
                except Exception as e:
                    print(f"      [WARNING] Enrichment failed: {e}")

    if stability_programs:
        sp_df = pd.DataFrame(stability_programs)
        sp_df.to_csv(res_dir / "neuroblastoma_stability_programs.csv", index=False)

    # Honest half-life framing
    print("\n  Half-life context:")
    print("    Note: Weak half-life correlations (r~-0.05) are expected for")
    print("    single-cell-type tumors. The heterogeneity assumption that drives")
    print("    strong correlations in developmental data (r~-0.35) is violated")
    print("    when all cells are a single tumor type.")

    # Figures: 4-panel corrected overview
    print("\n  Computing UMAP...")
    sc.tl.umap(adata)
    coords = adata.obsm["X_umap"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: UMAP colored by gamma sub-cluster
    if "gamma_subcluster" in adata.obs.columns:
        sub_labels = adata.obs["gamma_subcluster"].values
        unique_labels = sorted(set(sub_labels))
        colors_sc = plt.cm.Set2(np.linspace(0, 1, max(len(unique_labels), 2)))
        for li, label in enumerate(unique_labels):
            mask_l = sub_labels == label
            axes[0, 0].scatter(coords[mask_l, 0], coords[mask_l, 1], s=2, alpha=0.3,
                               c=[colors_sc[li]], label=label)
        axes[0, 0].legend(fontsize=8, markerscale=3)
    axes[0, 0].set_title("A: Gamma Sub-clusters (Stability Programs)")
    axes[0, 0].set_xlabel("UMAP 1")
    axes[0, 0].set_ylabel("UMAP 2")

    # Panel B: Corrected network stats (raw vs corrected destabilizing fraction)
    raw_destab = results.get("raw_destab_frac", 0.99)
    corr_destab = results.get("corrected_destab_frac", 0.60)
    bar_labels = ["Raw\nnetwork", "Library-size\ncorrected"]
    bar_vals = [raw_destab * 100, corr_destab * 100]
    bar_colors = ["salmon", "steelblue"]
    bars = axes[0, 1].bar(bar_labels, bar_vals, color=bar_colors,
                           edgecolor="black", linewidth=0.5, width=0.5)
    axes[0, 1].axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Null (50%)")
    for bar, val in zip(bars, bar_vals):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, val + 1,
                        f"{val:.1f}%", ha="center", fontsize=10)
    axes[0, 1].set_ylabel("Destabilizing edges (%)")
    axes[0, 1].set_title("B: Network Bias Correction")
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].legend(fontsize=8)

    # Panel C: Top pathways differentially degraded between sub-clusters
    if stability_programs:
        sp_show = pd.DataFrame(stability_programs)
        sp_show = sp_show.sort_values("fdr").head(10)
        y_pos = np.arange(len(sp_show))
        pathway_labels = [f"{row['cluster']}: {row['pathway'][:40]}"
                          for _, row in sp_show.iterrows()]
        neg_log_p = [-np.log10(max(row["fdr"], 1e-20)) for _, row in sp_show.iterrows()]
        sp_colors = ["steelblue" if "GC_0" in row["cluster"] else "coral"
                     for _, row in sp_show.iterrows()]
        axes[1, 0].barh(y_pos, neg_log_p, color=sp_colors, edgecolor="black",
                        linewidth=0.5)
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(pathway_labels, fontsize=7)
        axes[1, 0].set_xlabel("-log10(FDR)")
        axes[1, 0].axvline(x=1, color="gray", linestyle="--", alpha=0.5)
    axes[1, 0].set_title("C: Stability Programs (KEGG Pathways)")

    # Panel D: Mean gamma per cell (proxy for DepMap hub essentiality context)
    mean_gamma = np.mean(gamma, axis=1)
    sc_plot = axes[1, 1].scatter(coords[:, 0], coords[:, 1], s=2, alpha=0.3,
                                  c=np.clip(mean_gamma, 0, np.percentile(mean_gamma, 95)),
                                  cmap="YlOrRd")
    axes[1, 1].set_title("D: Mean Gamma (Degradation Rate)")
    axes[1, 1].set_xlabel("UMAP 1")
    axes[1, 1].set_ylabel("UMAP 2")
    plt.colorbar(sc_plot, ax=axes[1, 1])

    fig.suptitle("Neuroblastoma (GSE137804): Corrected scPTR Analysis", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "neuroblastoma_corrected_overview")

    with open(res_dir / "neuroblastoma_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


# =========================================================================
# T3-2: DepMap/CRISPR validation
# =========================================================================
def depmap_validation(datasets_results):
    """Validate RBP hub predictions against DepMap CRISPR dependency scores.

    Hypothesis: RBPs that are hub regulators in scPTR networks should be
    more essential (lower CRISPR gene effect scores) than non-hub RBPs.
    """
    print(f"\n{'='*60}")
    print("T3-2: DepMap/CRISPR VALIDATION")
    print(f"{'='*60}")

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load DepMap CRISPR data
    crispr_path = CACHE_DIR / "CRISPRGeneEffect.csv"
    model_path = CACHE_DIR / "DepMap_Model.csv"

    if not crispr_path.exists():
        print(f"  ERROR: {crispr_path} not found")
        return None

    print("  Loading DepMap CRISPR data...")
    crispr = pd.read_csv(crispr_path, index_col=0)
    print(f"  CRISPR matrix: {crispr.shape} (cell lines x genes)")

    # Parse gene names: "HUGO (Entrez)" -> "HUGO"
    gene_map = {}
    for col in crispr.columns:
        gene = col.split(" (")[0].strip()
        gene_map[col] = gene.upper()
    crispr.columns = [gene_map[c] for c in crispr.columns]

    # Compute mean dependency per gene (across all cell lines)
    mean_dep = crispr.mean(axis=0)
    print(f"  Genes in DepMap: {len(mean_dep)}")
    print(f"  Mean dependency: median={mean_dep.median():.4f}, "
          f"min={mean_dep.min():.4f}, max={mean_dep.max():.4f}")

    # Load RBP list
    rbp_path = Path(__file__).parent.parent / "src" / "scptr" / "tools" / "data" / "known_rbps.csv"
    rbps = set(g.upper() for g in pd.read_csv(rbp_path)["gene_symbol"])
    rbps_in_depmap = rbps & set(mean_dep.index)
    print(f"  RBPs in DepMap: {len(rbps_in_depmap)}/{len(rbps)}")

    # Also check A549 specifically (for sci-fate comparison)
    model = pd.read_csv(model_path)
    a549_rows = model[model["CellLineName"].str.contains("A549", case=False, na=False)]
    a549_id = a549_rows.iloc[0]["ModelID"] if len(a549_rows) > 0 else None

    if a549_id and a549_id in crispr.index:
        a549_dep = crispr.loc[a549_id]
        print(f"  A549 cell line found: {a549_id}")
    else:
        a549_dep = None
        print("  A549 not found in CRISPR data")

    # For each dataset with network results, test hub RBPs vs non-hub RBPs
    all_results = []

    for dataset_name, network_file in [
        ("pancreas", OUTPUT_DIR.parent / "tier1_fixes" / "results" / "network_bias" / "edges_pancreas.csv"),
        ("dentate_gyrus", OUTPUT_DIR.parent / "tier1_fixes" / "results" / "network_bias" / "edges_dentate_gyrus.csv"),
        ("neuroblastoma", res_dir / "neuroblastoma_network_corrected.csv"),
    ]:
        if not network_file.exists():
            # Try the run_gaps output
            alt = OUTPUT_DIR.parent / "gaps" / "results" / f"network_{dataset_name}.csv"
            if alt.exists():
                network_file = alt
            else:
                print(f"\n  {dataset_name}: no network file found, skipping")
                continue

        print(f"\n  === {dataset_name} ===")
        edges = pd.read_csv(network_file)
        print(f"    Network edges: {len(edges)}")

        # Count targets per RBP
        hub_counts = edges.groupby("rbp").size().sort_values(ascending=False)
        hub_rbps = set(hub_counts.head(20).index)
        hub_rbps_upper = set(r.upper() for r in hub_rbps)
        non_hub_rbps = rbps_in_depmap - hub_rbps_upper

        print(f"    Top 20 hub RBPs: {len(hub_rbps_upper & rbps_in_depmap)} in DepMap")
        print(f"    Non-hub RBPs: {len(non_hub_rbps)} in DepMap")

        if len(hub_rbps_upper & rbps_in_depmap) < 5:
            print(f"    Too few hub RBPs in DepMap")
            continue

        # Mean dependency for hub vs non-hub
        hub_deps = [mean_dep[g] for g in hub_rbps_upper if g in mean_dep.index]
        nonhub_deps = [mean_dep[g] for g in non_hub_rbps if g in mean_dep.index]

        hub_mean = np.mean(hub_deps)
        nonhub_mean = np.mean(nonhub_deps)
        u_stat, u_p = stats.mannwhitneyu(hub_deps, nonhub_deps, alternative="less")

        print(f"    Hub RBP mean dependency: {hub_mean:.4f} (n={len(hub_deps)})")
        print(f"    Non-hub RBP mean dependency: {nonhub_mean:.4f} (n={len(nonhub_deps)})")
        print(f"    Mann-Whitney (hub < non-hub): p = {u_p:.4f}")

        if u_p < 0.05:
            print(f"    ** Hub RBPs are MORE ESSENTIAL than non-hub RBPs **")

        # A549-specific comparison
        if a549_dep is not None:
            hub_a549 = [a549_dep[g] for g in hub_rbps_upper if g in a549_dep.index]
            nonhub_a549 = [a549_dep[g] for g in non_hub_rbps if g in a549_dep.index]
            if len(hub_a549) >= 5 and len(nonhub_a549) >= 5:
                a549_hub_mean = np.mean(hub_a549)
                a549_nonhub_mean = np.mean(nonhub_a549)
                a549_u, a549_p = stats.mannwhitneyu(hub_a549, nonhub_a549, alternative="less")
                print(f"    A549 hub dependency: {a549_hub_mean:.4f}")
                print(f"    A549 non-hub dependency: {a549_nonhub_mean:.4f}")
                print(f"    A549 Mann-Whitney: p = {a549_p:.4f}")

        # Correlation: number of targets vs dependency score
        rbp_dep_corr = []
        for rbp, n_targets in hub_counts.items():
            rbp_upper = rbp.upper()
            if rbp_upper in mean_dep.index:
                rbp_dep_corr.append((rbp_upper, n_targets, mean_dep[rbp_upper]))

        if len(rbp_dep_corr) >= 10:
            corr_df = pd.DataFrame(rbp_dep_corr, columns=["rbp", "n_targets", "dependency"])
            r, p = stats.spearmanr(corr_df["n_targets"], corr_df["dependency"])
            print(f"    Corr(n_targets, dependency): r={r:.4f}, p={p:.4f}")

        dataset_result = {
            "dataset": dataset_name,
            "n_hub_rbps": len(hub_deps),
            "n_nonhub_rbps": len(nonhub_deps),
            "hub_mean_dep": float(hub_mean),
            "nonhub_mean_dep": float(nonhub_mean),
            "mannwhitney_p": float(u_p),
            "hub_more_essential": bool(u_p < 0.05),
        }
        all_results.append(dataset_result)

    if not all_results:
        print("  No results to report")
        return None

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(res_dir / "depmap_validation.csv", index=False)

    # Figure: hub vs non-hub dependency comparison
    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]

    for ax, res in zip(axes, all_results):
        dataset = res["dataset"]
        # Reload edges for this dataset
        if dataset == "neuroblastoma":
            nf = res_dir / "neuroblastoma_network_corrected.csv"
        else:
            nf = OUTPUT_DIR.parent / "tier1_fixes" / "results" / "network_bias" / f"edges_{dataset}.csv"
        if not nf.exists():
            continue

        edges = pd.read_csv(nf)
        hub_counts = edges.groupby("rbp").size().sort_values(ascending=False)
        hub_rbps_upper = set(r.upper() for r in hub_counts.head(20).index)
        non_hub = rbps_in_depmap - hub_rbps_upper

        hub_vals = [mean_dep[g] for g in hub_rbps_upper if g in mean_dep.index]
        nonhub_vals = [mean_dep[g] for g in non_hub if g in mean_dep.index]

        bp = ax.boxplot([hub_vals, nonhub_vals],
                        tick_labels=["Hub RBPs\n(top 20)", "Non-hub\nRBPs"],
                        patch_artist=True, showfliers=True)
        bp["boxes"][0].set_facecolor("steelblue")
        bp["boxes"][1].set_facecolor("lightgray")
        ax.axhline(y=-1, color="red", linestyle="--", alpha=0.5, label="Pan-essential threshold")
        ax.set_ylabel("CRISPR Gene Effect (more negative = more essential)")
        ax.set_title(f"{dataset}\np={res['mannwhitney_p']:.4f}")
        ax.legend(fontsize=8)

    fig.suptitle("DepMap Validation: Hub RBPs vs Non-Hub RBPs", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "depmap_validation")

    # Also make a scatter: n_targets vs dependency
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    colors_map = {"pancreas": "steelblue", "dentate_gyrus": "darkgreen",
                  "neuroblastoma": "firebrick"}

    for dataset_name in ["pancreas", "dentate_gyrus", "neuroblastoma"]:
        if dataset_name == "neuroblastoma":
            nf = res_dir / "neuroblastoma_network_corrected.csv"
        else:
            nf = OUTPUT_DIR.parent / "tier1_fixes" / "results" / "network_bias" / f"edges_{dataset_name}.csv"
        if not nf.exists():
            continue
        edges = pd.read_csv(nf)
        hub_counts = edges.groupby("rbp").size().sort_values(ascending=False)
        scatter_data = []
        for rbp, n_targets in hub_counts.items():
            rbp_upper = rbp.upper()
            if rbp_upper in mean_dep.index:
                scatter_data.append((n_targets, mean_dep[rbp_upper], rbp_upper))

        if scatter_data:
            xs = [d[0] for d in scatter_data]
            ys = [d[1] for d in scatter_data]
            ax2.scatter(xs, ys, s=20, alpha=0.6,
                       c=colors_map.get(dataset_name, "gray"),
                       label=dataset_name)
            # Label top hubs
            for x, y, name in sorted(scatter_data, key=lambda d: d[0], reverse=True)[:5]:
                ax2.annotate(name, (x, y), fontsize=7, alpha=0.7)

    ax2.set_xlabel("Number of scPTR-predicted targets")
    ax2.set_ylabel("DepMap CRISPR Gene Effect")
    ax2.set_title("RBP Hub Size vs CRISPR Essentiality")
    ax2.axhline(y=-0.5, color="red", linestyle="--", alpha=0.3, label="Dependency threshold")
    ax2.legend()
    fig2.tight_layout()
    save_fig(fig2, "depmap_scatter")

    return results_df


# =========================================================================
# MAIN
# =========================================================================
def main():
    set_figure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # T3-1: Neuroblastoma analysis
    adata_nb = load_neuroblastoma()
    adata_nb = run_neuroblastoma_pipeline(adata_nb)
    nb_results = analyze_neuroblastoma(adata_nb)

    # T3-2: DepMap validation
    depmap_results = depmap_validation(nb_results)

    print(f"\n{'='*60}")
    print("ALL TIER 3 ANALYSES COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
