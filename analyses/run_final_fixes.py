#!/usr/bin/env python
"""Final publication fixes: address critical reviewer concerns.

1. Fix Fisher's exact test bug in hub consistency (contingency table was wrong)
2. Pathway specificity analysis (scPTR vs unspliced-only: unique vs generic pathways)
3. Per-cell sci-fate: stratify by expression level to show where scPTR advantage is largest
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

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "final_fixes"
PROJECT_ROOT = Path(__file__).parent.parent


def save_fig(fig, name, subdir="figures"):
    if fig is None:
        return
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =========================================================================
# 1. Fix Fisher's exact test for hub consistency
# =========================================================================
def fix_hub_fisher():
    """Recompute Fisher's exact using only shared RBPs as the universe."""
    print("\n" + "=" * 60)
    print("1. CORRECTED FISHER'S EXACT FOR HUB CONSISTENCY")
    print("=" * 60)

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load hub counts
    hub_files = {
        "pancreas": PROJECT_ROOT / "output" / "gap_analysis" / "results" / "network" / "pancreas" / "rbp_hub_counts.csv",
        "dentate_gyrus": PROJECT_ROOT / "output" / "gap_analysis" / "results" / "network" / "dentate_gyrus" / "rbp_hub_counts.csv",
    }

    hub_counts = {}
    for name, path in hub_files.items():
        if path.exists():
            df = pd.read_csv(path)
            count_col = [c for c in df.columns if c != "rbp"][0]
            hub_counts[name] = pd.Series(df[count_col].values, index=df["rbp"].values)

    # NB from corrected network
    nb_net_path = PROJECT_ROOT / "output" / "tier3" / "results" / "neuroblastoma_network_corrected.csv"
    if nb_net_path.exists():
        nb_net = pd.read_csv(nb_net_path)
        hub_counts["neuroblastoma"] = nb_net.groupby("rbp").size().sort_values(ascending=False)

    # Uppercase
    hub_upper = {}
    for name, series in hub_counts.items():
        hub_upper[name] = pd.Series(series.values, index=[g.upper() for g in series.index])

    names = sorted(hub_upper.keys())
    results = []

    print("\n  Corrected Fisher's exact (universe = shared RBPs only):")
    for i, name_a in enumerate(names):
        for j in range(len(names)):
            if i == j:
                continue
            name_b = names[j]
            shared = set(hub_upper[name_a].index) & set(hub_upper[name_b].index)
            n_shared = len(shared)
            if n_shared < 5:
                continue

            # Rank RBPs WITHIN the shared set only
            shared_a = hub_upper[name_a].reindex(list(shared)).dropna().sort_values(ascending=False)
            shared_b = hub_upper[name_b].reindex(list(shared)).dropna().sort_values(ascending=False)

            # Top-k from A (within shared), top-k from B (within shared)
            k_a = min(5, n_shared // 3)  # top third or 5
            k_b = min(10, n_shared // 2)  # top half or 10

            top_a = set(shared_a.index[:k_a])
            top_b = set(shared_b.index[:k_b])

            # 2x2 contingency table (universe = shared)
            a_and_b = len(top_a & top_b)
            a_not_b = len(top_a - top_b)
            b_not_a = len(top_b - top_a)
            neither = n_shared - a_and_b - a_not_b - b_not_a

            table = [[a_and_b, a_not_b], [b_not_a, neither]]
            odds_ratio, fisher_p = stats.fisher_exact(table, alternative="greater")
            print(f"  Top-{k_a} {name_a} in top-{k_b} {name_b}: "
                  f"{a_and_b}/{k_a} overlap, OR={odds_ratio:.2f}, p={fisher_p:.4f} "
                  f"(n_shared={n_shared})")

            results.append({
                "source": name_a, "target": name_b,
                "k_source": k_a, "k_target": k_b,
                "overlap": a_and_b, "n_shared": n_shared,
                "odds_ratio": float(odds_ratio), "fisher_p": float(fisher_p),
            })

    with open(res_dir / "corrected_hub_fisher.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# =========================================================================
# 2. Pathway specificity: unique tissue pathways per method
# =========================================================================
def pathway_specificity():
    """Analyze whether scPTR finds different or more specific pathways
    than unspliced-only, rather than just counting totals."""
    print("\n" + "=" * 60)
    print("2. PATHWAY SPECIFICITY ANALYSIS")
    print("=" * 60)

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load full coherence ablation results
    coherence_csv = PROJECT_ROOT / "output" / "comprehensive_fixes" / "results" / "coherence_ablation.csv"
    pathways_csv = PROJECT_ROOT / "output" / "comprehensive_fixes" / "results" / "coherence_ablation_pathways.csv"

    coherence_df = pd.read_csv(coherence_csv)
    pathways_df = pd.read_csv(pathways_csv)

    # Generic housekeeping pathways (appear in every tissue, not informative)
    generic_pathways = {
        "ribosome", "oxidative phosphorylation", "thermogenesis",
        "huntington disease", "alzheimer disease", "parkinson disease",
        "non-alcoholic fatty liver disease", "cardiac muscle contraction",
        "diabetic cardiomyopathy", "chemical carcinogenesis",
        "metabolic pathways", "carbon metabolism",
    }

    # Tissue-specific pathways (the ones we care about)
    tissue_specific = {
        "pancreas": {
            "protein processing in endoplasmic reticulum", "autophagy",
            "insulin secretion", "insulin signaling pathway",
            "pancreatic secretion", "maturity onset diabetes",
            "unfolded protein response", "protein folding",
        },
        "dentate_gyrus": {
            "synaptic vesicle cycle", "long-term potentiation",
            "glutamatergic synapse", "gabaergic synapse",
            "axon guidance", "neurotrophin signaling pathway",
            "dopaminergic synapse", "serotonergic synapse",
        },
    }

    results = {}

    for dataset in ["pancreas", "dentate_gyrus"]:
        print(f"\n--- {dataset} ---")
        ds_paths = pathways_df[pathways_df["dataset"] == dataset]
        ds_coherence = coherence_df[coherence_df["dataset"] == dataset]

        expected_set = tissue_specific.get(dataset, set())

        for method in ["scPTR_gamma", "raw_u_s_ratio", "unspliced_only"]:
            method_paths = ds_paths[ds_paths["method"] == method]
            all_terms = [t.lower() for t in method_paths["pathway"].values]

            n_total = len(all_terms)
            n_generic = sum(1 for t in all_terms
                          if any(g in t for g in generic_pathways))
            n_tissue = sum(1 for t in all_terms
                          if any(ts in t for ts in expected_set))
            n_specific = n_total - n_generic

            # Unique pathways (found by this method but not others)
            other_methods = [m for m in ["scPTR_gamma", "raw_u_s_ratio", "unspliced_only"]
                           if m != method]
            other_terms = set()
            for om in other_methods:
                om_paths = ds_paths[ds_paths["method"] == om]
                other_terms |= set(t.lower() for t in om_paths["pathway"].values)

            unique_terms = [t for t in all_terms if t not in other_terms]
            n_unique = len(unique_terms)

            # From coherence CSV: mean invisibility, mean diff genes
            mc = ds_coherence[ds_coherence["method"] == method]

            key = f"{dataset}_{method}"
            results[key] = {
                "dataset": dataset, "method": method,
                "n_total_pathways": n_total,
                "n_generic": n_generic,
                "n_tissue_specific": n_tissue,
                "n_non_generic": n_specific,
                "n_unique_to_method": n_unique,
                "generic_fraction": n_generic / max(n_total, 1),
                "mean_invisibility": float(mc["invisibility"].mean()) if len(mc) > 0 else 0,
                "mean_diff_genes": float(mc["n_diff_genes"].mean()) if len(mc) > 0 else 0,
            }

            print(f"  {method:<20s}: {n_total} total, {n_generic} generic, "
                  f"{n_tissue} tissue-specific, {n_unique} unique")

    # Cross-method comparison: per-cluster agreement
    print("\n  Per-cluster: do all methods find the same expected pathways?")
    clusters_tested = coherence_df["cluster"].unique()
    agreement_data = []

    for cluster in clusters_tested:
        cluster_data = coherence_df[coherence_df["cluster"] == cluster]
        for _, row in cluster_data.iterrows():
            agreement_data.append({
                "cluster": cluster,
                "dataset": row["dataset"],
                "method": row["method"],
                "n_expected": row["n_expected_pathways"],
                "n_sig": row["n_sig_pathways"],
                "n_diff_genes": row["n_diff_genes"],
            })

    agreement_df = pd.DataFrame(agreement_data)

    # Key metric: per cluster, which method finds the MOST expected pathways?
    print("\n  Per-cluster winner (most expected pathways):")
    winner_counts = {"scPTR_gamma": 0, "raw_u_s_ratio": 0, "unspliced_only": 0, "tie": 0}

    for cluster in clusters_tested:
        cl = agreement_df[agreement_df["cluster"] == cluster]
        if len(cl) == 0:
            continue
        max_expected = cl["n_expected"].max()
        winners = cl[cl["n_expected"] == max_expected]["method"].tolist()
        if len(winners) == 1:
            winner_counts[winners[0]] += 1
        else:
            winner_counts["tie"] += 1

    for method, count in winner_counts.items():
        print(f"    {method}: wins {count}/{len(clusters_tested)} clusters")

    results["winner_counts"] = winner_counts

    with open(res_dir / "pathway_specificity.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: stacked bar of generic vs tissue-specific vs other
    methods = ["scPTR_gamma", "raw_u_s_ratio", "unspliced_only"]
    method_labels = ["scPTR\ngamma", "Raw u/s\nratio", "Unspliced\nonly"]
    x = np.arange(len(methods))

    for di, dataset in enumerate(["pancreas", "dentate_gyrus"]):
        offset = di * 0.35 - 0.175
        generics = []
        tissues = []
        others = []
        for m in methods:
            key = f"{dataset}_{m}"
            if key in results:
                r = results[key]
                generics.append(r["n_generic"])
                tissues.append(r["n_tissue_specific"])
                others.append(r["n_non_generic"] - r["n_tissue_specific"])
            else:
                generics.append(0)
                tissues.append(0)
                others.append(0)

        color_generic = "lightgray" if di == 0 else "silver"
        color_tissue = "steelblue" if di == 0 else "darkorange"
        color_other = "lightblue" if di == 0 else "moccasin"

        axes[0].bar(x + offset, tissues, 0.3, label=f"{dataset} tissue-specific",
                    color=color_tissue, edgecolor="black", linewidth=0.3)
        axes[0].bar(x + offset, others, 0.3, bottom=tissues,
                    label=f"{dataset} other", color=color_other,
                    edgecolor="black", linewidth=0.3)
        axes[0].bar(x + offset, generics, 0.3,
                    bottom=[t + o for t, o in zip(tissues, others)],
                    label=f"{dataset} generic", color=color_generic,
                    edgecolor="black", linewidth=0.3)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(method_labels)
    axes[0].set_ylabel("Number of top pathways")
    axes[0].set_title("Pathway Composition by Method")
    axes[0].legend(fontsize=6, ncol=2)

    # Panel 2: per-cluster winner counts
    cats = list(winner_counts.keys())
    vals = [winner_counts[c] for c in cats]
    colors_bar = ["steelblue", "orange", "lightblue", "gray"]
    axes[1].bar(cats, vals, color=colors_bar, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Number of clusters won")
    axes[1].set_title("Per-Cluster: Most Expected Pathways")
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.2, str(v), ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Pathway Specificity Analysis", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "pathway_specificity")

    return results


# =========================================================================
# 3. Per-cell sci-fate stratified by expression level
# =========================================================================
def percell_stratified():
    """Show that scPTR's advantage over raw u/s increases for
    low-expression genes, where smoothing matters most."""
    print("\n" + "=" * 60)
    print("3. PER-CELL SCI-FATE STRATIFIED BY EXPRESSION LEVEL")
    print("=" * 60)

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    from run_scifate import load_scifate_data, prepare_for_scptr
    import scptr

    adata_raw = load_scifate_data()
    adata = prepare_for_scptr(adata_raw)

    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)
    print(f"  Pipeline complete: {adata.shape}")

    gamma = adata.layers["gamma"]
    u_layer = adata.layers.get("Mu", adata.layers.get("unspliced"))
    s_layer = adata.layers.get("Ms", adata.layers.get("spliced"))
    u = u_layer.toarray() if hasattr(u_layer, 'toarray') else np.asarray(u_layer)
    s = s_layer.toarray() if hasattr(s_layer, 'toarray') else np.asarray(s_layer)

    raw_ratio = np.zeros_like(gamma)
    s_safe = np.where(s > 0.01, s, 1.0)
    raw_ratio = u / s_safe
    raw_ratio[s < 0.01] = 0

    # Ground truth per cell
    total_raw = np.asarray(adata_raw.X.toarray() if hasattr(adata_raw.X, 'toarray') else adata_raw.X)
    new_raw = np.asarray(adata_raw.layers["new"].toarray() if hasattr(adata_raw.layers["new"], 'toarray') else adata_raw.layers["new"])
    old_raw = total_raw - new_raw

    raw_gene_map = {g: i for i, g in enumerate(adata_raw.var_names)}
    filtered_in_raw = [raw_gene_map[g] for g in adata.var_names if g in raw_gene_map]
    genes_in_both = [g for g in adata.var_names if g in raw_gene_map]
    gene_idx_in_filtered = [list(adata.var_names).index(g) for g in genes_in_both]

    gt_new = new_raw[:, filtered_in_raw]
    gt_old = old_raw[:, filtered_in_raw]
    gt_total = total_raw[:, filtered_in_raw]
    gt_ratio = np.zeros_like(gt_new, dtype=float)
    valid_gt = gt_old > 0.1
    gt_ratio[valid_gt] = gt_new[valid_gt] / gt_old[valid_gt]
    gt_ratio[~valid_gt] = np.nan

    gamma_matched = gamma[:, gene_idx_in_filtered]
    raw_matched = raw_ratio[:, gene_idx_in_filtered]

    # Stratify genes by expression level (mean total counts)
    gene_mean_expr = gt_total.mean(axis=0)
    terciles = np.percentile(gene_mean_expr[gene_mean_expr > 0], [33, 67])

    strata = {
        "low": gene_mean_expr <= terciles[0],
        "medium": (gene_mean_expr > terciles[0]) & (gene_mean_expr <= terciles[1]),
        "high": gene_mean_expr > terciles[1],
    }

    n_cells = adata.n_obs
    results = {}

    for stratum_name, gene_mask in strata.items():
        n_genes_stratum = gene_mask.sum()
        print(f"\n  --- {stratum_name} expression ({n_genes_stratum} genes) ---")

        gamma_corrs = []
        raw_corrs = []

        for i in range(n_cells):
            gt_i = gt_ratio[i, gene_mask]
            gamma_i = gamma_matched[i, gene_mask]
            raw_i = raw_matched[i, gene_mask]

            valid = np.isfinite(gt_i) & (gt_i > 0) & (gamma_i > 0) & (raw_i > 0)
            if valid.sum() >= 10:
                r_g, _ = stats.spearmanr(gamma_i[valid], gt_i[valid])
                r_r, _ = stats.spearmanr(raw_i[valid], gt_i[valid])
                gamma_corrs.append(r_g)
                raw_corrs.append(r_r)

        gamma_corrs = np.array(gamma_corrs)
        raw_corrs = np.array(raw_corrs)

        # Filter out NaN correlations (from constant inputs)
        finite_mask = np.isfinite(gamma_corrs) & np.isfinite(raw_corrs)
        gamma_corrs = gamma_corrs[finite_mask]
        raw_corrs = raw_corrs[finite_mask]

        if len(gamma_corrs) < 10:
            print(f"    Skipped: only {len(gamma_corrs)} valid cells after NaN filtering")
            continue

        mean_g = np.mean(gamma_corrs)
        mean_r = np.mean(raw_corrs)
        gamma_wins = (gamma_corrs > raw_corrs).sum()
        n_valid = len(gamma_corrs)
        w_stat, w_p = stats.wilcoxon(gamma_corrs, raw_corrs, alternative="greater")

        print(f"    scPTR gamma: mean r = {mean_g:.4f}")
        print(f"    Raw u/s:     mean r = {mean_r:.4f}")
        print(f"    Advantage:   {mean_g - mean_r:.4f}")
        print(f"    gamma wins:  {gamma_wins}/{n_valid} ({100*gamma_wins/n_valid:.1f}%)")
        print(f"    Wilcoxon p:  {w_p:.2e}")

        results[stratum_name] = {
            "n_genes": int(n_genes_stratum),
            "n_valid_cells": int(n_valid),
            "mean_gamma_corr": float(mean_g),
            "mean_raw_corr": float(mean_r),
            "advantage": float(mean_g - mean_r),
            "gamma_wins_frac": float(gamma_wins / n_valid),
            "wilcoxon_p": float(w_p),
        }

    with open(res_dir / "percell_stratified.json", "w") as f:
        json.dump(results, f, indent=2)

    # Figure: advantage by expression stratum
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    strata_order = ["low", "medium", "high"]
    strata_labels = ["Low\nexpr", "Medium\nexpr", "High\nexpr"]

    # Panel 1: mean correlation per stratum
    gamma_means = [results.get(s, {}).get("mean_gamma_corr", 0) for s in strata_order]
    raw_means = [results.get(s, {}).get("mean_raw_corr", 0) for s in strata_order]
    x = np.arange(len(strata_order))
    axes[0].bar(x - 0.15, gamma_means, 0.3, label="scPTR gamma",
                color="steelblue", edgecolor="black", linewidth=0.5)
    axes[0].bar(x + 0.15, raw_means, 0.3, label="Raw u/s",
                color="salmon", edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(strata_labels)
    axes[0].set_ylabel("Mean per-cell Spearman r")
    axes[0].set_title("Per-Cell Correlation by Expression Level")
    axes[0].legend()

    # Panel 2: advantage (gamma - raw) by stratum
    advantages = [results.get(s, {}).get("advantage", 0) for s in strata_order]
    p_values = [results.get(s, {}).get("wilcoxon_p", 1) for s in strata_order]
    colors = ["steelblue" if a > 0 else "salmon" for a in advantages]
    axes[1].bar(strata_labels, advantages, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("scPTR advantage (gamma r - raw r)")
    axes[1].set_title("scPTR Advantage by Expression Level")
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.3)
    for i, (a, p) in enumerate(zip(advantages, p_values)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        axes[1].text(i, a + 0.001 if a > 0 else a - 0.002,
                    f"{a:.4f}\n({sig})", ha="center", fontsize=9)

    fig.suptitle("scPTR Advantage Stratified by Gene Expression Level",
                fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "percell_stratified")

    return results


# =========================================================================
# MAIN
# =========================================================================
def main():
    set_figure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    all_results = {}

    # 1. Fix Fisher's exact
    all_results["hub_fisher"] = fix_hub_fisher()

    # 2. Pathway specificity
    all_results["pathway_specificity"] = pathway_specificity()

    # 3. Per-cell stratified
    all_results["percell_stratified"] = percell_stratified()

    with open(OUTPUT_DIR / "results" / "all_final_fixes.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("ALL FINAL FIXES COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
