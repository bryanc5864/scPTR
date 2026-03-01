#!/usr/bin/env python
"""Comprehensive improvement of scPTR weaknesses.

Fix A: Per-cell sci-fate ablation (scPTR vs raw u/s per cell)
Fix B: 3' UTR sequence validation of network direction
Fix C: Neuroblastoma-specific DepMap validation
Fix D: Cross-dataset RBP hub consistency
Fix E: Biological coherence ablation (GSEA on invisible states)
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

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "comprehensive_fixes"
PROJECT_ROOT = Path(__file__).parent.parent


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
# FIX B: 3' UTR Sequence Validation of Network Direction
# =========================================================================
def fix_b_utr_validation():
    """Validate network direction using 3' UTR sequence features.

    Destabilizing targets should have longer 3' UTRs (more regulatory elements)
    and higher AU content.
    """
    print("\n" + "=" * 60)
    print("FIX B: 3' UTR SEQUENCE VALIDATION OF NETWORK DIRECTION")
    print("=" * 60)

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load UTR features
    data_dir = PROJECT_ROOT / "src" / "scptr" / "benchmark" / "data"
    mouse_utr = pd.read_csv(data_dir / "mouse_utr_features.csv")
    human_utr = pd.read_csv(data_dir / "human_utr_features.csv")
    print(f"  Mouse UTR features: {len(mouse_utr)} genes")
    print(f"  Human UTR features: {len(human_utr)} genes")

    # Load corrected networks
    networks = {}
    net_files = {
        "pancreas": PROJECT_ROOT / "output" / "weakness_fixes" / "results" / "corrected_network_pancreas.csv",
        "dentate_gyrus": PROJECT_ROOT / "output" / "weakness_fixes" / "results" / "corrected_network_dentate_gyrus.csv",
        "neuroblastoma": PROJECT_ROOT / "output" / "tier3" / "results" / "neuroblastoma_network_corrected.csv",
    }

    for name, path in net_files.items():
        if path.exists():
            networks[name] = pd.read_csv(path)
            print(f"  {name} network: {len(networks[name])} edges")
        else:
            print(f"  [WARNING] {name} network not found at {path}")

    results = {}
    all_summaries = []

    for net_name, edges_df in networks.items():
        print(f"\n--- {net_name} ---")

        # Determine which UTR dataset to use
        # Neuroblastoma = human, pancreas/DG = mouse
        if net_name == "neuroblastoma":
            utr_df = human_utr.copy()
            # Column for correlation is spearman_r
            r_col = "spearman_r" if "spearman_r" in edges_df.columns else "r"
        else:
            utr_df = mouse_utr.copy()
            r_col = "r" if "r" in edges_df.columns else "spearman_r"

        # Build gene-level summary: mean correlation across all RBP connections
        target_stats = edges_df.groupby("target").agg(
            mean_r=(r_col, "mean"),
            n_rbps=(r_col, "count"),
        ).reset_index()

        # Classify as predominantly destabilized (mean r > 0) or stabilized (mean r < 0)
        target_stats["class"] = np.where(target_stats["mean_r"] > 0,
                                         "destabilized", "stabilized")
        n_dest = (target_stats["class"] == "destabilized").sum()
        n_stab = (target_stats["class"] == "stabilized").sum()
        print(f"  Target genes: {len(target_stats)} ({n_dest} destabilized, {n_stab} stabilized)")

        # Match target genes to UTR features (case-insensitive)
        utr_map = {g.upper(): i for i, g in enumerate(utr_df["gene"])}
        target_stats["gene_upper"] = target_stats["target"].str.upper()
        matched = target_stats[target_stats["gene_upper"].isin(utr_map)].copy()
        matched["utr_length"] = matched["gene_upper"].map(
            lambda g: utr_df.iloc[utr_map[g]]["utr_length"])
        matched["au_content"] = matched["gene_upper"].map(
            lambda g: utr_df.iloc[utr_map[g]]["au_content"])

        n_matched = len(matched)
        print(f"  Matched to UTR features: {n_matched}/{len(target_stats)}")

        if n_matched < 10:
            print(f"  Too few matched genes, skipping")
            continue

        dest = matched[matched["class"] == "destabilized"]
        stab = matched[matched["class"] == "stabilized"]

        net_results = {"dataset": net_name, "n_targets": len(target_stats),
                       "n_matched": n_matched}

        # Test 1: UTR length destabilized vs stabilized
        if len(dest) >= 5 and len(stab) >= 5:
            u_stat, p_len = stats.mannwhitneyu(
                dest["utr_length"].values, stab["utr_length"].values,
                alternative="greater")
            med_dest_len = dest["utr_length"].median()
            med_stab_len = stab["utr_length"].median()
            print(f"  UTR length: destab median={med_dest_len:.0f}, "
                  f"stab median={med_stab_len:.0f}, "
                  f"MW p={p_len:.4f} (destab > stab)")
            net_results["utr_length_destab_median"] = float(med_dest_len)
            net_results["utr_length_stab_median"] = float(med_stab_len)
            net_results["utr_length_mw_p"] = float(p_len)
        else:
            p_len = np.nan

        # Test 2: AU content destabilized vs stabilized
        if len(dest) >= 5 and len(stab) >= 5:
            u_stat, p_au = stats.mannwhitneyu(
                dest["au_content"].values, stab["au_content"].values,
                alternative="greater")
            med_dest_au = dest["au_content"].median()
            med_stab_au = stab["au_content"].median()
            print(f"  AU content: destab median={med_dest_au:.4f}, "
                  f"stab median={med_stab_au:.4f}, "
                  f"MW p={p_au:.4f} (destab > stab)")
            net_results["au_content_destab_median"] = float(med_dest_au)
            net_results["au_content_stab_median"] = float(med_stab_au)
            net_results["au_content_mw_p"] = float(p_au)
        else:
            p_au = np.nan

        # Test 3: Spearman correlation of mean_r vs UTR length
        r_vs_len, p_r_len = stats.spearmanr(
            matched["mean_r"].values, matched["utr_length"].values)
        print(f"  Spearman(mean_r, UTR length): r={r_vs_len:.4f}, p={p_r_len:.4f}")
        net_results["spearman_r_vs_utr_length"] = float(r_vs_len)
        net_results["spearman_p_vs_utr_length"] = float(p_r_len)

        # Test 4: Spearman correlation of mean_r vs AU content
        r_vs_au, p_r_au = stats.spearmanr(
            matched["mean_r"].values, matched["au_content"].values)
        print(f"  Spearman(mean_r, AU content): r={r_vs_au:.4f}, p={p_r_au:.4f}")
        net_results["spearman_r_vs_au_content"] = float(r_vs_au)
        net_results["spearman_p_vs_au_content"] = float(p_r_au)

        results[net_name] = net_results
        all_summaries.append(net_results)

    # Save results
    with open(res_dir / "utr_network_validation.json", "w") as f:
        json.dump(results, f, indent=2)

    # Figure: 2x3 panels (UTR length and AU content for each dataset)
    n_nets = len(results)
    if n_nets == 0:
        print("  No networks to plot")
        return results

    fig, axes = plt.subplots(2, n_nets, figsize=(5 * n_nets, 8))
    if n_nets == 1:
        axes = axes.reshape(2, 1)

    for col, (net_name, edges_df) in enumerate(networks.items()):
        if net_name not in results:
            continue

        r_col = "spearman_r" if "spearman_r" in edges_df.columns else "r"
        if net_name == "neuroblastoma":
            utr_df = human_utr
        else:
            utr_df = mouse_utr

        # Rebuild matched data for plotting
        target_stats = edges_df.groupby("target").agg(
            mean_r=(r_col, "mean"),
        ).reset_index()
        target_stats["gene_upper"] = target_stats["target"].str.upper()
        utr_map = {g.upper(): i for i, g in enumerate(utr_df["gene"])}
        matched = target_stats[target_stats["gene_upper"].isin(utr_map)].copy()
        matched["utr_length"] = matched["gene_upper"].map(
            lambda g: utr_df.iloc[utr_map[g]]["utr_length"])
        matched["au_content"] = matched["gene_upper"].map(
            lambda g: utr_df.iloc[utr_map[g]]["au_content"])

        # Row 0: scatter mean_r vs UTR length
        ax = axes[0, col]
        ax.scatter(matched["mean_r"], matched["utr_length"],
                   alpha=0.3, s=10, c="steelblue")
        r_val = results[net_name].get("spearman_r_vs_utr_length", np.nan)
        p_val = results[net_name].get("spearman_p_vs_utr_length", np.nan)
        ax.set_xlabel("Mean RBP-target r")
        ax.set_ylabel("3' UTR length (nt)")
        ax.set_title(f"{net_name}\nr={r_val:.3f}, p={p_val:.3f}")

        # Row 1: scatter mean_r vs AU content
        ax = axes[1, col]
        ax.scatter(matched["mean_r"], matched["au_content"],
                   alpha=0.3, s=10, c="darkorange")
        r_val = results[net_name].get("spearman_r_vs_au_content", np.nan)
        p_val = results[net_name].get("spearman_p_vs_au_content", np.nan)
        ax.set_xlabel("Mean RBP-target r")
        ax.set_ylabel("AU content")
        ax.set_title(f"{net_name}\nr={r_val:.3f}, p={p_val:.3f}")

    fig.suptitle("3' UTR Validation of Network Direction", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "utr_network_validation")

    return results


# =========================================================================
# FIX D: Cross-Dataset RBP Hub Consistency
# =========================================================================
def fix_d_hub_consistency():
    """Compare hub rankings across pancreas, DG, and neuroblastoma."""
    print("\n" + "=" * 60)
    print("FIX D: CROSS-DATASET RBP HUB CONSISTENCY")
    print("=" * 60)

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load hub counts from gap_analysis
    hub_files = {
        "pancreas": PROJECT_ROOT / "output" / "gap_analysis" / "results" / "network" / "pancreas" / "rbp_hub_counts.csv",
        "dentate_gyrus": PROJECT_ROOT / "output" / "gap_analysis" / "results" / "network" / "dentate_gyrus" / "rbp_hub_counts.csv",
    }

    hub_counts = {}

    for name, path in hub_files.items():
        if path.exists():
            df = pd.read_csv(path)
            # Format: rbp, 0 (where 0 is the count column)
            count_col = [c for c in df.columns if c != "rbp"][0]
            series = pd.Series(df[count_col].values, index=df["rbp"].values)
            hub_counts[name] = series
            print(f"  {name}: {len(series)} RBPs")
        else:
            print(f"  [WARNING] {name} hub counts not found at {path}")

    # Compute NB hub counts from corrected network
    nb_net_path = PROJECT_ROOT / "output" / "tier3" / "results" / "neuroblastoma_network_corrected.csv"
    if nb_net_path.exists():
        nb_net = pd.read_csv(nb_net_path)
        nb_hubs = nb_net.groupby("rbp").size().sort_values(ascending=False)
        hub_counts["neuroblastoma"] = nb_hubs
        print(f"  neuroblastoma: {len(nb_hubs)} RBPs")

    if len(hub_counts) < 2:
        print("  Need at least 2 datasets for comparison")
        return {}

    # Unify gene names to uppercase
    hub_upper = {}
    for name, series in hub_counts.items():
        hub_upper[name] = pd.Series(series.values, index=[g.upper() for g in series.index])

    # Pairwise Spearman on target counts across shared RBPs
    names = sorted(hub_upper.keys())
    results = {"pairwise_correlations": [], "universal_hubs": [], "dataset_hubs": {}}

    print("\n  Pairwise hub count correlations:")
    for i, name_a in enumerate(names):
        for j in range(i + 1, len(names)):
            name_b = names[j]
            shared = hub_upper[name_a].index.intersection(hub_upper[name_b].index)
            if len(shared) < 5:
                print(f"  {name_a} vs {name_b}: only {len(shared)} shared RBPs, skipping")
                continue

            va = hub_upper[name_a][shared].values.astype(float)
            vb = hub_upper[name_b][shared].values.astype(float)
            r, p = stats.spearmanr(va, vb)
            print(f"  {name_a} vs {name_b}: Spearman r={r:.4f}, p={p:.4f} (n={len(shared)})")

            results["pairwise_correlations"].append({
                "dataset_a": name_a,
                "dataset_b": name_b,
                "spearman_r": float(r),
                "spearman_p": float(p),
                "n_shared": int(len(shared)),
            })

    # Fisher's exact: are top-10 hubs in A enriched among top-20 in B?
    print("\n  Fisher's exact test (top-10 in A enriched among top-20 in B?):")
    for i, name_a in enumerate(names):
        for j in range(len(names)):
            if i == j:
                continue
            name_b = names[j]
            shared = hub_upper[name_a].index.intersection(hub_upper[name_b].index)
            if len(shared) < 5:
                continue

            top_a = set(hub_upper[name_a].nlargest(10).index)
            top_b = set(hub_upper[name_b].nlargest(20).index)

            # Contingency table
            a_in_b = len(top_a & top_b)
            a_not_b = len(top_a - top_b)
            not_a_in_b = len(top_b - top_a)
            not_a_not_b = len(shared) - a_in_b - a_not_b - not_a_in_b

            if not_a_not_b < 0:
                not_a_not_b = 0

            table = [[a_in_b, a_not_b], [not_a_in_b, not_a_not_b]]
            odds_ratio, fisher_p = stats.fisher_exact(table, alternative="greater")
            print(f"  Top-10 {name_a} in top-20 {name_b}: "
                  f"{a_in_b}/10, OR={odds_ratio:.2f}, p={fisher_p:.4f}")

    # Identify "universal" hubs (top 20 in >= 2 datasets)
    print("\n  Universal hubs (top 20 in >= 2 datasets):")
    top20_sets = {}
    for name in names:
        top20_sets[name] = set(hub_upper[name].nlargest(20).index)

    all_rbps = set()
    for s in top20_sets.values():
        all_rbps |= s

    hub_table = []
    for rbp in sorted(all_rbps):
        datasets_in_top20 = [name for name in names if rbp in top20_sets[name]]
        counts_per_dataset = {name: int(hub_upper[name].get(rbp, 0))
                              for name in names}
        hub_table.append({
            "rbp": rbp,
            "n_datasets_top20": len(datasets_in_top20),
            "datasets": ", ".join(datasets_in_top20),
            **{f"targets_{name}": counts_per_dataset[name] for name in names},
        })

    hub_df = pd.DataFrame(hub_table).sort_values("n_datasets_top20", ascending=False)

    # Save per-dataset top hubs
    for name in names:
        results["dataset_hubs"][name] = hub_upper[name].nlargest(10).to_dict()

    universal = hub_df[hub_df["n_datasets_top20"] >= 2]
    tissue_specific = hub_df[hub_df["n_datasets_top20"] == 1]
    print(f"  Universal (>=2): {len(universal)} RBPs")
    for _, row in universal.iterrows():
        print(f"    {row['rbp']}: {row['datasets']}")
    print(f"  Tissue-specific (1 only): {len(tissue_specific)} RBPs")

    results["universal_hubs"] = universal.to_dict(orient="records")
    results["n_universal"] = int(len(universal))
    results["n_tissue_specific"] = int(len(tissue_specific))

    # Save
    hub_df.to_csv(res_dir / "hub_consistency_table.csv", index=False)
    with open(res_dir / "hub_consistency.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Figure: heatmap of hub counts + bar chart of universal vs specific
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: heatmap of top RBPs across datasets
    top_rbps = hub_df.nlargest(20, "n_datasets_top20")
    target_cols = [f"targets_{n}" for n in names]
    heatmap_data = top_rbps[target_cols].values.astype(float)
    heatmap_labels = top_rbps["rbp"].values

    im = axes[0].imshow(heatmap_data, aspect="auto", cmap="YlOrRd")
    axes[0].set_yticks(np.arange(len(heatmap_labels)))
    axes[0].set_yticklabels(heatmap_labels, fontsize=8)
    axes[0].set_xticks(np.arange(len(names)))
    axes[0].set_xticklabels(names, fontsize=9, rotation=30, ha="right")
    axes[0].set_title("Hub RBP Target Counts Across Datasets")
    for i in range(len(heatmap_labels)):
        for j in range(len(names)):
            val = int(heatmap_data[i, j])
            if val > 0:
                axes[0].text(j, i, str(val), ha="center", va="center",
                            fontsize=7, color="white" if val > heatmap_data.max() * 0.6 else "black")
    plt.colorbar(im, ax=axes[0], label="Target count", shrink=0.8)

    # Panel 2: universal vs tissue-specific
    axes[1].bar(["Universal\n(>=2 datasets)", "Tissue-specific\n(1 dataset)"],
                [len(universal), len(tissue_specific)],
                color=["steelblue", "salmon"], edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Number of RBPs")
    axes[1].set_title("Hub Consistency Across Datasets")
    for i, v in enumerate([len(universal), len(tissue_specific)]):
        axes[1].text(i, v + 0.5, str(v), ha="center", fontsize=11, fontweight="bold")

    fig.suptitle("Cross-Dataset RBP Hub Consistency", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "hub_consistency")

    return results


# =========================================================================
# FIX C: Neuroblastoma-Specific DepMap
# =========================================================================
def fix_c_nb_depmap():
    """Filter DepMap CRISPR scores to NB-specific cell lines."""
    print("\n" + "=" * 60)
    print("FIX C: NEUROBLASTOMA-SPECIFIC DepMap VALIDATION")
    print("=" * 60)

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = PROJECT_ROOT / ".cache"

    # Load DepMap model metadata
    model_df = pd.read_csv(cache_dir / "DepMap_Model.csv")
    nb_models = model_df[model_df["OncotreePrimaryDisease"] == "Neuroblastoma"]
    nb_model_ids = set(nb_models["ModelID"].values)
    print(f"  Neuroblastoma cell lines in DepMap: {len(nb_model_ids)}")

    # Load CRISPR gene effect
    print("  Loading CRISPRGeneEffect.csv...")
    crispr_df = pd.read_csv(cache_dir / "CRISPRGeneEffect.csv", index_col=0)
    print(f"  CRISPR data: {crispr_df.shape[0]} cell lines, {crispr_df.shape[1]} genes")

    # Parse gene names from column headers: "GENE (ID)" -> "GENE"
    gene_names = [col.split(" (")[0] for col in crispr_df.columns]
    crispr_df.columns = gene_names

    # Filter to NB cell lines
    nb_ids_in_crispr = nb_model_ids & set(crispr_df.index)
    print(f"  NB cell lines with CRISPR data: {len(nb_ids_in_crispr)}")

    nb_crispr = crispr_df.loc[list(nb_ids_in_crispr)]
    all_crispr = crispr_df

    # Mean dependency per gene
    nb_mean_dep = nb_crispr.mean(axis=0)
    all_mean_dep = all_crispr.mean(axis=0)
    non_nb_crispr = crispr_df.loc[~crispr_df.index.isin(nb_model_ids)]
    non_nb_mean_dep = non_nb_crispr.mean(axis=0)

    # Load network hubs for each dataset
    hub_files = {
        "neuroblastoma": PROJECT_ROOT / "output" / "tier3" / "results" / "neuroblastoma_network_corrected.csv",
        "pancreas": PROJECT_ROOT / "output" / "weakness_fixes" / "results" / "corrected_network_pancreas.csv",
        "dentate_gyrus": PROJECT_ROOT / "output" / "weakness_fixes" / "results" / "corrected_network_dentate_gyrus.csv",
    }

    results = {}

    for net_name, net_path in hub_files.items():
        if not net_path.exists():
            print(f"  [WARNING] {net_name} network not found")
            continue

        net_df = pd.read_csv(net_path)
        hub_counts = net_df.groupby("rbp").size().sort_values(ascending=False)
        top_n = min(20, len(hub_counts))
        hub_rbps = set(hub_counts.index[:top_n])
        non_hub_rbps = set(hub_counts.index[top_n:])

        print(f"\n--- {net_name} ({len(hub_rbps)} hub, {len(non_hub_rbps)} non-hub RBPs) ---")

        # Match to CRISPR gene names (uppercase)
        crispr_genes_upper = {g.upper(): g for g in nb_mean_dep.index}

        hub_nb_deps = []
        hub_all_deps = []
        hub_non_nb_deps = []
        for rbp in hub_rbps:
            g_upper = rbp.upper()
            if g_upper in crispr_genes_upper:
                cg = crispr_genes_upper[g_upper]
                hub_nb_deps.append(nb_mean_dep[cg])
                hub_all_deps.append(all_mean_dep[cg])
                hub_non_nb_deps.append(non_nb_mean_dep[cg])

        nonhub_nb_deps = []
        nonhub_all_deps = []
        nonhub_non_nb_deps = []
        for rbp in non_hub_rbps:
            g_upper = rbp.upper()
            if g_upper in crispr_genes_upper:
                cg = crispr_genes_upper[g_upper]
                nonhub_nb_deps.append(nb_mean_dep[cg])
                nonhub_all_deps.append(all_mean_dep[cg])
                nonhub_non_nb_deps.append(non_nb_mean_dep[cg])

        net_results = {
            "n_hub_rbps": len(hub_rbps),
            "n_hub_matched": len(hub_nb_deps),
            "n_nonhub_matched": len(nonhub_nb_deps),
        }

        # NB-specific: hub vs non-hub
        if len(hub_nb_deps) >= 3 and len(nonhub_nb_deps) >= 3:
            u_stat, p_nb = stats.mannwhitneyu(
                hub_nb_deps, nonhub_nb_deps, alternative="less")
            print(f"  NB-specific: hub mean={np.mean(hub_nb_deps):.4f}, "
                  f"non-hub mean={np.mean(nonhub_nb_deps):.4f}, "
                  f"MW p={p_nb:.4e}")
            net_results["nb_hub_mean_dep"] = float(np.mean(hub_nb_deps))
            net_results["nb_nonhub_mean_dep"] = float(np.mean(nonhub_nb_deps))
            net_results["nb_mw_p"] = float(p_nb)

        # Pan-cancer: hub vs non-hub
        if len(hub_all_deps) >= 3 and len(nonhub_all_deps) >= 3:
            u_stat, p_all = stats.mannwhitneyu(
                hub_all_deps, nonhub_all_deps, alternative="less")
            print(f"  Pan-cancer:  hub mean={np.mean(hub_all_deps):.4f}, "
                  f"non-hub mean={np.mean(nonhub_all_deps):.4f}, "
                  f"MW p={p_all:.4e}")
            net_results["all_hub_mean_dep"] = float(np.mean(hub_all_deps))
            net_results["all_nonhub_mean_dep"] = float(np.mean(nonhub_all_deps))
            net_results["all_mw_p"] = float(p_all)

        # NB-specificity: are NB hubs MORE essential in NB vs non-NB?
        if len(hub_nb_deps) >= 3 and len(hub_non_nb_deps) >= 3:
            u_stat, p_spec = stats.mannwhitneyu(
                hub_nb_deps, hub_non_nb_deps, alternative="less")
            print(f"  NB-specificity: hub in NB={np.mean(hub_nb_deps):.4f}, "
                  f"hub in non-NB={np.mean(hub_non_nb_deps):.4f}, "
                  f"MW p={p_spec:.4e}")
            net_results["nb_specificity_p"] = float(p_spec)
            net_results["hub_nb_mean"] = float(np.mean(hub_nb_deps))
            net_results["hub_non_nb_mean"] = float(np.mean(hub_non_nb_deps))

        # Correlation: n_targets vs NB-specific dependency
        all_rbps_in_net = hub_counts.index.tolist()
        n_targets_list = []
        dep_list = []
        for rbp in all_rbps_in_net:
            g_upper = rbp.upper()
            if g_upper in crispr_genes_upper:
                cg = crispr_genes_upper[g_upper]
                n_targets_list.append(hub_counts[rbp])
                dep_list.append(nb_mean_dep[cg])

        if len(n_targets_list) >= 5:
            r_corr, p_corr = stats.spearmanr(n_targets_list, dep_list)
            print(f"  Corr(n_targets, NB dep): r={r_corr:.4f}, p={p_corr:.4f}")
            net_results["ntargets_dep_spearman_r"] = float(r_corr)
            net_results["ntargets_dep_spearman_p"] = float(p_corr)

        results[net_name] = net_results

    # Save results
    with open(res_dir / "nb_specific_depmap.json", "w") as f:
        json.dump(results, f, indent=2)

    # Figure: grouped bar chart comparing NB-specific vs pan-cancer
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Hub vs non-hub dependency by dataset and scope
    datasets = [n for n in ["neuroblastoma", "pancreas", "dentate_gyrus"] if n in results]
    x = np.arange(len(datasets))
    width = 0.2

    for offset, (scope, label, color) in enumerate([
        ("nb_hub_mean_dep", "Hub (NB)", "darkred"),
        ("nb_nonhub_mean_dep", "Non-hub (NB)", "salmon"),
        ("all_hub_mean_dep", "Hub (pan-cancer)", "darkblue"),
        ("all_nonhub_mean_dep", "Non-hub (pan-cancer)", "lightblue"),
    ]):
        vals = [results.get(d, {}).get(scope, 0) for d in datasets]
        axes[0].bar(x + (offset - 1.5) * width, vals, width, label=label,
                    color=color, edgecolor="black", linewidth=0.3)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets, fontsize=9)
    axes[0].set_ylabel("Mean CRISPR dependency\n(more negative = more essential)")
    axes[0].set_title("Hub RBP Essentiality: NB-Specific vs Pan-Cancer")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.3)

    # Panel 2: NB-specificity for NB network hubs
    if "neuroblastoma" in results:
        nb_res = results["neuroblastoma"]
        categories = []
        values = []
        colors = []
        if "hub_nb_mean" in nb_res:
            categories.append("NB hub\n(in NB lines)")
            values.append(nb_res["hub_nb_mean"])
            colors.append("darkred")
        if "hub_non_nb_mean" in nb_res:
            categories.append("NB hub\n(in non-NB)")
            values.append(nb_res["hub_non_nb_mean"])
            colors.append("lightcoral")
        if "nb_nonhub_mean_dep" in nb_res:
            categories.append("Non-hub\n(in NB lines)")
            values.append(nb_res["nb_nonhub_mean_dep"])
            colors.append("gray")

        if values:
            axes[1].bar(categories, values, color=colors, edgecolor="black", linewidth=0.5)
            axes[1].set_ylabel("Mean CRISPR dependency")
            axes[1].set_title("NB Hub RBPs: Tissue-Specific Essentiality")
            if "nb_specificity_p" in nb_res:
                axes[1].text(0.5, 0.95, f"NB vs non-NB: p={nb_res['nb_specificity_p']:.4f}",
                            transform=axes[1].transAxes, ha="center", va="top", fontsize=9)

    fig.suptitle("Neuroblastoma-Specific DepMap Validation", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "nb_specific_depmap")

    return results


# =========================================================================
# FIX A: Per-Cell sci-fate Ablation
# =========================================================================
def fix_a_per_cell_scifate():
    """Compare per-cell correlations: scPTR gamma vs raw u/s ratio."""
    print("\n" + "=" * 60)
    print("FIX A: PER-CELL SCI-FATE ABLATION")
    print("=" * 60)

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Import sci-fate loading functions
    from run_scifate import load_scifate_data, prepare_for_scptr

    # Load raw sci-fate data
    adata_raw = load_scifate_data()

    # Prepare for scPTR
    adata = prepare_for_scptr(adata_raw)

    # Run scPTR pipeline
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)
    print(f"  Pipeline complete: {adata.shape}")

    # Get gamma matrix (smoothed, beta-normalized)
    gamma = adata.layers["gamma"]  # cells x genes

    # Compute raw u/s ratio (unsmoothed)
    u_layer = adata.layers.get("Mu", adata.layers.get("unspliced"))
    s_layer = adata.layers.get("Ms", adata.layers.get("spliced"))
    u = u_layer.toarray() if hasattr(u_layer, 'toarray') else np.asarray(u_layer)
    s = s_layer.toarray() if hasattr(s_layer, 'toarray') else np.asarray(s_layer)

    # Raw u/s ratio with same safeguard as scPTR
    raw_ratio = np.zeros_like(gamma)
    s_safe = np.where(s > 0.01, s, 1.0)
    raw_ratio = u / s_safe
    raw_ratio[s < 0.01] = 0

    # Compute ground truth new/old ratio per cell
    # Map back to the genes that survived filtering
    total_raw = np.asarray(adata_raw.X.toarray() if hasattr(adata_raw.X, 'toarray') else adata_raw.X)
    new_raw = np.asarray(adata_raw.layers["new"].toarray() if hasattr(adata_raw.layers["new"], 'toarray') else adata_raw.layers["new"])
    old_raw = total_raw - new_raw

    # Match genes between adata (filtered) and adata_raw
    raw_gene_map = {g: i for i, g in enumerate(adata_raw.var_names)}
    filtered_in_raw = [raw_gene_map[g] for g in adata.var_names if g in raw_gene_map]
    genes_in_both = [g for g in adata.var_names if g in raw_gene_map]

    if len(genes_in_both) < len(adata.var_names):
        print(f"  [WARNING] {len(adata.var_names) - len(genes_in_both)} genes not matched")

    # Ground truth per cell: new/old ratio for each gene
    gt_new = new_raw[:, filtered_in_raw]
    gt_old = old_raw[:, filtered_in_raw]
    gt_ratio = np.zeros_like(gt_new, dtype=float)
    valid_gt = gt_old > 0.1
    gt_ratio[valid_gt] = gt_new[valid_gt] / gt_old[valid_gt]
    gt_ratio[~valid_gt] = np.nan

    # Get corresponding columns from gamma and raw_ratio
    gene_idx_in_filtered = [list(adata.var_names).index(g) for g in genes_in_both]
    gamma_matched = gamma[:, gene_idx_in_filtered]
    raw_matched = raw_ratio[:, gene_idx_in_filtered]

    n_cells = adata.n_obs
    print(f"  Computing per-cell correlations for {n_cells} cells...")

    # Per-cell: Spearman(gamma_vector, gt_vector) and Spearman(raw_vector, gt_vector)
    gamma_corrs = np.full(n_cells, np.nan)
    raw_corrs = np.full(n_cells, np.nan)
    gamma_cvs = np.full(n_cells, np.nan)
    raw_cvs = np.full(n_cells, np.nan)

    min_genes_per_cell = 20

    for i in range(n_cells):
        gt_i = gt_ratio[i]
        gamma_i = gamma_matched[i]
        raw_i = raw_matched[i]

        # Mask: need valid gt AND nonzero method value
        valid = np.isfinite(gt_i) & (gt_i > 0) & (gamma_i > 0) & (raw_i > 0)
        n_valid = valid.sum()

        if n_valid >= min_genes_per_cell:
            r_gamma, _ = stats.spearmanr(gamma_i[valid], gt_i[valid])
            r_raw, _ = stats.spearmanr(raw_i[valid], gt_i[valid])
            gamma_corrs[i] = r_gamma
            raw_corrs[i] = r_raw

            # CV: coefficient of variation (std/mean) — lower = less noisy
            gamma_cv = np.std(gamma_i[valid]) / (np.mean(gamma_i[valid]) + 1e-10)
            raw_cv = np.std(raw_i[valid]) / (np.mean(raw_i[valid]) + 1e-10)
            gamma_cvs[i] = gamma_cv
            raw_cvs[i] = raw_cv

    valid_cells = np.isfinite(gamma_corrs) & np.isfinite(raw_corrs)
    n_valid_cells = valid_cells.sum()
    print(f"  Valid cells: {n_valid_cells}/{n_cells}")

    if n_valid_cells < 10:
        print("  Too few valid cells, aborting Fix A")
        return {}

    # Summary statistics
    mean_gamma_corr = np.nanmean(gamma_corrs[valid_cells])
    mean_raw_corr = np.nanmean(raw_corrs[valid_cells])
    med_gamma_corr = np.nanmedian(gamma_corrs[valid_cells])
    med_raw_corr = np.nanmedian(raw_corrs[valid_cells])

    print(f"\n  Per-cell correlation with ground truth:")
    print(f"    scPTR gamma: mean={mean_gamma_corr:.4f}, median={med_gamma_corr:.4f}")
    print(f"    Raw u/s:     mean={mean_raw_corr:.4f}, median={med_raw_corr:.4f}")

    # Wilcoxon signed-rank test (paired)
    w_stat, wilcox_p = stats.wilcoxon(
        gamma_corrs[valid_cells], raw_corrs[valid_cells],
        alternative="greater")
    print(f"  Wilcoxon signed-rank (gamma > raw): p={wilcox_p:.4e}")

    # Fraction of cells where gamma beats raw
    gamma_better = (gamma_corrs[valid_cells] > raw_corrs[valid_cells]).sum()
    raw_better = (raw_corrs[valid_cells] > gamma_corrs[valid_cells]).sum()
    print(f"  gamma beats raw: {gamma_better}/{n_valid_cells} ({100*gamma_better/n_valid_cells:.1f}%)")
    print(f"  raw beats gamma: {raw_better}/{n_valid_cells} ({100*raw_better/n_valid_cells:.1f}%)")

    # CV comparison
    valid_cv = np.isfinite(gamma_cvs) & np.isfinite(raw_cvs)
    if valid_cv.sum() > 10:
        mean_gamma_cv = np.nanmean(gamma_cvs[valid_cv])
        mean_raw_cv = np.nanmean(raw_cvs[valid_cv])
        w_cv, cv_p = stats.wilcoxon(
            gamma_cvs[valid_cv], raw_cvs[valid_cv],
            alternative="less")
        print(f"\n  Coefficient of variation (noise):")
        print(f"    scPTR gamma: mean CV={mean_gamma_cv:.4f}")
        print(f"    Raw u/s:     mean CV={mean_raw_cv:.4f}")
        print(f"    Wilcoxon (gamma < raw): p={cv_p:.4e}")
    else:
        mean_gamma_cv = np.nan
        mean_raw_cv = np.nan
        cv_p = np.nan

    results = {
        "n_cells_total": int(n_cells),
        "n_cells_valid": int(n_valid_cells),
        "mean_gamma_corr": float(mean_gamma_corr),
        "mean_raw_corr": float(mean_raw_corr),
        "median_gamma_corr": float(med_gamma_corr),
        "median_raw_corr": float(med_raw_corr),
        "wilcoxon_p": float(wilcox_p),
        "gamma_better_frac": float(gamma_better / n_valid_cells),
        "raw_better_frac": float(raw_better / n_valid_cells),
        "mean_gamma_cv": float(mean_gamma_cv) if np.isfinite(mean_gamma_cv) else None,
        "mean_raw_cv": float(mean_raw_cv) if np.isfinite(mean_raw_cv) else None,
        "cv_wilcoxon_p": float(cv_p) if np.isfinite(cv_p) else None,
    }

    with open(res_dir / "per_cell_scifate.json", "w") as f:
        json.dump(results, f, indent=2)

    # Figure: paired distribution comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: histogram of per-cell correlations
    bins = np.linspace(-0.5, 1.0, 50)
    axes[0].hist(gamma_corrs[valid_cells], bins=bins, alpha=0.6,
                 label=f"scPTR gamma (mean={mean_gamma_corr:.3f})",
                 color="steelblue", edgecolor="white")
    axes[0].hist(raw_corrs[valid_cells], bins=bins, alpha=0.6,
                 label=f"Raw u/s (mean={mean_raw_corr:.3f})",
                 color="salmon", edgecolor="white")
    axes[0].set_xlabel("Per-cell Spearman r with ground truth")
    axes[0].set_ylabel("Number of cells")
    axes[0].set_title(f"Per-Cell Correlation with Ground Truth\n"
                      f"(Wilcoxon p={wilcox_p:.2e})")
    axes[0].legend(fontsize=8)

    # Panel 2: scatter gamma_corr vs raw_corr
    axes[1].scatter(raw_corrs[valid_cells], gamma_corrs[valid_cells],
                    alpha=0.1, s=3, c="steelblue")
    lims = [min(axes[1].get_xlim()[0], axes[1].get_ylim()[0]),
            max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])]
    axes[1].plot(lims, lims, "k--", alpha=0.3, lw=1)
    axes[1].set_xlabel("Raw u/s per-cell r")
    axes[1].set_ylabel("scPTR gamma per-cell r")
    axes[1].set_title(f"gamma better: {gamma_better}/{n_valid_cells} "
                      f"({100*gamma_better/n_valid_cells:.0f}%)")

    # Panel 3: difference distribution
    diff = gamma_corrs[valid_cells] - raw_corrs[valid_cells]
    axes[2].hist(diff, bins=50, color="steelblue", alpha=0.8, edgecolor="white")
    axes[2].axvline(0, color="red", linestyle="--", alpha=0.5)
    axes[2].axvline(np.mean(diff), color="black", linestyle="-", alpha=0.8,
                    label=f"Mean diff={np.mean(diff):.4f}")
    axes[2].set_xlabel("Difference (gamma r - raw r)")
    axes[2].set_ylabel("Number of cells")
    axes[2].set_title("Per-Cell Improvement")
    axes[2].legend(fontsize=8)

    fig.suptitle("Per-Cell sci-fate Ablation: scPTR gamma vs Raw u/s Ratio",
                fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "per_cell_scifate")

    return results


# =========================================================================
# FIX E: Biological Coherence Ablation
# =========================================================================
def fix_e_coherence_ablation():
    """Run GSEA on sub-clusters from each method to test biological coherence."""
    print("\n" + "=" * 60)
    print("FIX E: BIOLOGICAL COHERENCE ABLATION")
    print("=" * 60)

    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from statsmodels.stats.multitest import multipletests

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Expected tissue-appropriate pathways
    expected_pathways = {
        "pancreas": [
            "endoplasmic reticulum", "autophagy", "protein folding",
            "unfolded protein", "er stress", "insulin", "secretion",
            "pancrea", "endocrine", "exocrine",
        ],
        "dentate_gyrus": [
            "synaptic", "long-term potentiation", "spliceosome", "neuron",
            "axon", "dendrite", "glutamat", "gaba", "hippocampus",
            "neurogenesis", "myelination",
        ],
    }

    all_results = []
    pathway_details = []

    for dataset_name in ["pancreas", "dentate_gyrus"]:
        print(f"\n--- {dataset_name} ---")

        # Load dataset
        if dataset_name == "pancreas":
            adata = scptr.datasets.pancreas()
        else:
            adata = scptr.datasets.dentate_gyrus()

        adata = run_pipeline(adata, dataset_name)

        gamma = adata.layers["gamma"]
        clusters = adata.obs["clusters"]

        # Get layers for ablation methods
        u_layer = adata.layers.get("Mu", adata.layers.get("unspliced"))
        s_layer = adata.layers.get("Ms", adata.layers.get("spliced"))
        u = u_layer.toarray() if hasattr(u_layer, 'toarray') else np.asarray(u_layer)
        s = s_layer.toarray() if hasattr(s_layer, 'toarray') else np.asarray(s_layer)
        expr = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)

        # Raw u/s ratio
        raw_ratio = np.zeros_like(gamma)
        s_safe = np.where(s > 0.01, s, 1.0)
        raw_ratio = u / s_safe
        raw_ratio[s < 0.01] = 0

        methods = {
            "scPTR_gamma": gamma,
            "raw_u_s_ratio": raw_ratio,
            "unspliced_only": u,
        }

        # Load UTR features for UTR length enrichment test
        utr_df = pd.read_csv(
            PROJECT_ROOT / "src" / "scptr" / "benchmark" / "data" / "mouse_utr_features.csv")
        utr_map = {row["gene"].upper(): row for _, row in utr_df.iterrows()}

        # Determine organism for GSEA
        sample_gene = adata.var_names[0]
        organism = "mouse" if sample_gene[0].isupper() and sample_gene[1:].islower() else "human"

        for cluster_name in sorted(clusters.unique()):
            mask = (clusters == cluster_name).values
            n_cells = mask.sum()
            if n_cells < 50:
                continue

            # Pre-compute expression PCA for invisibility check
            expr_sub = expr[mask]
            nonzero_expr = (expr_sub > 0).mean(axis=0)
            good_expr = nonzero_expr >= 0.05
            if good_expr.sum() < 20:
                continue
            n_expr_pcs = min(15, n_cells - 1, good_expr.sum() - 1)
            pca_expr = PCA(n_components=n_expr_pcs, random_state=42)
            expr_pcs = pca_expr.fit_transform(expr_sub[:, good_expr])

            # Check if ANY method finds invisible sub-clusters
            any_invisible = False
            for method_name, data in methods.items():
                data_sub = data[mask]
                nonzero = (data_sub > 0).mean(axis=0)
                good = nonzero >= 0.05
                if good.sum() < 20:
                    continue
                data_filtered = data_sub[:, good]
                n_pcs = min(15, n_cells - 1, data_filtered.shape[1] - 1)
                pca = PCA(n_components=n_pcs, random_state=42)
                pcs = pca.fit_transform(data_filtered)

                for k in [2, 3]:
                    if n_cells < k * 10:
                        continue
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = km.fit_predict(pcs)
                    if min(np.bincount(labels)) < 10:
                        continue
                    sil = silhouette_score(pcs, labels)
                    sil_expr = silhouette_score(expr_pcs, labels)
                    if sil - sil_expr > 0.05:
                        any_invisible = True
                        break
                if any_invisible:
                    break

            if not any_invisible:
                continue

            print(f"\n  {cluster_name} ({n_cells} cells) — invisible in at least one method")

            for method_name, data in methods.items():
                data_sub = data[mask]
                nonzero = (data_sub > 0).mean(axis=0)
                good = nonzero >= 0.05
                if good.sum() < 20:
                    continue

                data_filtered = data_sub[:, good]
                gene_names_filtered = adata.var_names[good]
                n_pcs = min(15, n_cells - 1, data_filtered.shape[1] - 1)
                pca = PCA(n_components=n_pcs, random_state=42)
                pcs = pca.fit_transform(data_filtered)

                best_sil = -1
                best_labels = None
                best_k = 1
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
                        best_k = k

                if best_labels is None or best_k <= 1:
                    continue

                sil_expr_val = silhouette_score(expr_pcs, best_labels)
                invisibility = best_sil - sil_expr_val

                # Find differentially degraded genes between sub-clusters
                diff_results = []
                for gi, gene in enumerate(gene_names_filtered):
                    groups = [data_filtered[best_labels == j, gi] for j in range(best_k)]
                    if all(len(g) >= 5 for g in groups):
                        if best_k == 2:
                            _, p_val = stats.mannwhitneyu(groups[0], groups[1],
                                                          alternative='two-sided')
                        else:
                            _, p_val = stats.kruskal(*groups)

                        medians = [np.median(g) for g in groups]
                        max_med = max(medians)
                        min_med = min(medians)
                        log_fc = np.log2((max_med + 0.01) / (min_med + 0.01))
                        diff_results.append({"gene": gene, "p_value": p_val,
                                             "log2_fc": log_fc})

                if not diff_results:
                    continue

                diff_df = pd.DataFrame(diff_results)
                _, diff_df["fdr"], _, _ = multipletests(diff_df["p_value"], method="fdr_bh")
                sig_genes = diff_df[diff_df["fdr"] < 0.05].sort_values("log2_fc", ascending=False)

                gene_list = sig_genes["gene"].tolist()

                # UTR length enrichment: sig genes vs background
                sig_utr_lengths = []
                bg_utr_lengths = []
                for g in gene_list:
                    if g.upper() in utr_map:
                        sig_utr_lengths.append(utr_map[g.upper()]["utr_length"])
                for g in adata.var_names:
                    if g.upper() in utr_map:
                        bg_utr_lengths.append(utr_map[g.upper()]["utr_length"])

                utr_p = np.nan
                if len(sig_utr_lengths) >= 5 and len(bg_utr_lengths) >= 5:
                    _, utr_p = stats.mannwhitneyu(
                        sig_utr_lengths, bg_utr_lengths, alternative="greater")

                # Run GSEA via gseapy Enrichr API
                n_sig_pathways = 0
                n_expected_pathways = 0
                pathway_terms = []

                if len(gene_list) >= 5:
                    try:
                        import gseapy as gp
                        gene_sets = ["GO_Biological_Process_2023",
                                     "KEGG_2019_Mouse" if organism == "mouse" else "KEGG_2021_Human"]

                        enr = gp.enrichr(gene_list=gene_list,
                                         gene_sets=gene_sets,
                                         organism=organism,
                                         outdir=None,
                                         no_plot=True)

                        enr_df = enr.results
                        sig_enr = enr_df[enr_df["Adjusted P-value"] < 0.1]
                        n_sig_pathways = len(sig_enr)

                        # Check for expected tissue pathways
                        expected = expected_pathways.get(dataset_name, [])
                        for _, row in sig_enr.iterrows():
                            term_lower = row["Term"].lower()
                            pathway_terms.append(row["Term"])
                            for kw in expected:
                                if kw in term_lower:
                                    n_expected_pathways += 1
                                    break

                    except Exception as e:
                        print(f"    [WARNING] GSEA failed for {method_name}/{cluster_name}: {e}")

                result_entry = {
                    "dataset": dataset_name,
                    "cluster": cluster_name,
                    "method": method_name,
                    "n_cells": int(n_cells),
                    "n_subclusters": int(best_k),
                    "sil_method": float(best_sil),
                    "sil_expr": float(sil_expr_val),
                    "invisibility": float(invisibility),
                    "n_diff_genes": int(len(sig_genes)),
                    "n_sig_pathways": int(n_sig_pathways),
                    "n_expected_pathways": int(n_expected_pathways),
                    "mean_utr_length_sig": float(np.mean(sig_utr_lengths)) if sig_utr_lengths else None,
                    "mean_utr_length_bg": float(np.mean(bg_utr_lengths)) if bg_utr_lengths else None,
                    "utr_enrichment_p": float(utr_p) if np.isfinite(utr_p) else None,
                }
                all_results.append(result_entry)

                if pathway_terms:
                    for term in pathway_terms[:5]:
                        pathway_details.append({
                            "dataset": dataset_name,
                            "cluster": cluster_name,
                            "method": method_name,
                            "pathway": term,
                        })

                print(f"    {method_name}: sil={best_sil:.3f}, invis={invisibility:.3f}, "
                      f"diff_genes={len(sig_genes)}, sig_pathways={n_sig_pathways}, "
                      f"expected={n_expected_pathways}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(res_dir / "coherence_ablation.csv", index=False)

    if pathway_details:
        pd.DataFrame(pathway_details).to_csv(
            res_dir / "coherence_ablation_pathways.csv", index=False)

    # Summary
    if len(results_df) > 0:
        print("\n  Summary: mean metrics by method")
        summary = results_df.groupby("method").agg(
            mean_invisibility=("invisibility", "mean"),
            mean_sig_pathways=("n_sig_pathways", "mean"),
            total_sig_pathways=("n_sig_pathways", "sum"),
            mean_expected=("n_expected_pathways", "mean"),
            total_expected=("n_expected_pathways", "sum"),
            mean_diff_genes=("n_diff_genes", "mean"),
        )
        for method, row in summary.iterrows():
            print(f"    {method:<20s}: pathways={row['total_sig_pathways']:.0f} "
                  f"(expected={row['total_expected']:.0f}), "
                  f"diff_genes={row['mean_diff_genes']:.0f}, "
                  f"invis={row['mean_invisibility']:.3f}")

    # Save JSON summary
    json_results = {
        "n_clusters_tested": len(results_df["cluster"].unique()) if len(results_df) > 0 else 0,
        "summary_by_method": {},
    }
    if len(results_df) > 0:
        for method in ["scPTR_gamma", "raw_u_s_ratio", "unspliced_only"]:
            sub = results_df[results_df["method"] == method]
            if len(sub) > 0:
                json_results["summary_by_method"][method] = {
                    "n_clusters": int(len(sub)),
                    "mean_invisibility": float(sub["invisibility"].mean()),
                    "total_sig_pathways": int(sub["n_sig_pathways"].sum()),
                    "total_expected_pathways": int(sub["n_expected_pathways"].sum()),
                    "mean_diff_genes": float(sub["n_diff_genes"].mean()),
                }

    with open(res_dir / "coherence_ablation.json", "w") as f:
        json.dump(json_results, f, indent=2)

    # Figure
    if len(results_df) > 0:
        methods_order = ["unspliced_only", "raw_u_s_ratio", "scPTR_gamma"]
        method_labels = ["Unspliced\nonly", "Raw u/s\nratio", "scPTR\ngamma"]
        colors = ["lightblue", "orange", "steelblue"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: total significant GSEA pathways
        vals = []
        for m in methods_order:
            sub = results_df[results_df["method"] == m]
            vals.append(sub["n_sig_pathways"].sum() if len(sub) > 0 else 0)
        axes[0].bar(method_labels, vals, color=colors, edgecolor="black", linewidth=0.5)
        axes[0].set_ylabel("Total significant pathways (FDR<0.1)")
        axes[0].set_title("GSEA Pathway Enrichment")
        for i, v in enumerate(vals):
            axes[0].text(i, v + 0.3, str(int(v)), ha="center", fontsize=10, fontweight="bold")

        # Panel 2: expected tissue pathways
        vals_exp = []
        for m in methods_order:
            sub = results_df[results_df["method"] == m]
            vals_exp.append(sub["n_expected_pathways"].sum() if len(sub) > 0 else 0)
        axes[1].bar(method_labels, vals_exp, color=colors, edgecolor="black", linewidth=0.5)
        axes[1].set_ylabel("Tissue-appropriate pathways found")
        axes[1].set_title("Expected Pathway Hits")
        for i, v in enumerate(vals_exp):
            axes[1].text(i, v + 0.2, str(int(v)), ha="center", fontsize=10, fontweight="bold")

        # Panel 3: mean invisibility
        vals_inv = []
        for m in methods_order:
            sub = results_df[results_df["method"] == m]
            vals_inv.append(sub["invisibility"].mean() if len(sub) > 0 else 0)
        axes[2].bar(method_labels, vals_inv, color=colors, edgecolor="black", linewidth=0.5)
        axes[2].set_ylabel("Mean invisibility score")
        axes[2].set_title("Invisibility Score")
        axes[2].axhline(0, color="gray", linestyle="--", alpha=0.3)

        fig.suptitle("Biological Coherence Ablation", fontsize=13, y=1.02)
        fig.tight_layout()
        save_fig(fig, "coherence_ablation")

    return json_results


# =========================================================================
# MAIN
# =========================================================================
def main():
    set_figure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Fix B (fastest — CSV only)
    print("\n" + "#" * 60)
    print("# FIX B: 3' UTR SEQUENCE VALIDATION")
    print("#" * 60)
    all_results["fix_b_utr"] = fix_b_utr_validation()

    # Fix D (fast — CSV only)
    print("\n" + "#" * 60)
    print("# FIX D: CROSS-DATASET HUB CONSISTENCY")
    print("#" * 60)
    all_results["fix_d_hub_consistency"] = fix_d_hub_consistency()

    # Fix C (moderate — loads large CSV)
    print("\n" + "#" * 60)
    print("# FIX C: NEUROBLASTOMA-SPECIFIC DepMap")
    print("#" * 60)
    all_results["fix_c_nb_depmap"] = fix_c_nb_depmap()

    # Fix A (moderate — loads sci-fate data)
    print("\n" + "#" * 60)
    print("# FIX A: PER-CELL SCI-FATE ABLATION")
    print("#" * 60)
    all_results["fix_a_per_cell"] = fix_a_per_cell_scifate()

    # Fix E (slowest — loads 2 datasets + GSEA API)
    print("\n" + "#" * 60)
    print("# FIX E: BIOLOGICAL COHERENCE ABLATION")
    print("#" * 60)
    all_results["fix_e_coherence"] = fix_e_coherence_ablation()

    # Save combined results
    with open(OUTPUT_DIR / "results" / "all_comprehensive_fixes.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("ALL COMPREHENSIVE FIXES COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
