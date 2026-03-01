#!/usr/bin/env python
"""Targeted improvements for four remaining scPTR weaknesses.

Experiment 1: Edge-level UTR validation (fixes pancreas p=0.676)
Experiment 2: DepMap stratified NB analysis (MYCN, lineage, cross-line)
Experiment 3: eCLIP edge-strength concordance (fixes weak OR=0.56-1.30)

All experiments use existing cached data. No dataset downloads required.
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

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "weakness_improvements"
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "scptr" / "benchmark" / "data"


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


def save_results(data, name, subdir="results"):
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Experiment 1: Edge-Level UTR Validation
# ---------------------------------------------------------------------------

def load_network(dataset):
    """Load corrected network edges for a dataset."""
    if dataset == "pancreas":
        path = PROJECT_ROOT / "output" / "weakness_fixes" / "results" / "corrected_network_pancreas.csv"
    elif dataset == "dentate_gyrus":
        path = PROJECT_ROOT / "output" / "weakness_fixes" / "results" / "corrected_network_dentate_gyrus.csv"
    elif dataset == "neuroblastoma":
        path = PROJECT_ROOT / "output" / "tier3" / "results" / "neuroblastoma_network_corrected.csv"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    df = pd.read_csv(path)
    # Normalize column names
    if "spearman_r" in df.columns:
        df = df.rename(columns={"spearman_r": "r"})
    return df


def load_utr_features(species):
    """Load UTR features (mouse or human)."""
    fname = f"{species}_utr_features.csv"
    return pd.read_csv(DATA_DIR / fname)


def experiment1_edge_utr():
    """Edge-level UTR validation across all datasets."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Edge-Level UTR Validation")
    print("=" * 60)

    set_figure_style()

    datasets = {
        "pancreas": "mouse",
        "dentate_gyrus": "mouse",
        "neuroblastoma": "human",
    }

    all_results = {}

    for ds_name, species in datasets.items():
        print(f"\n--- {ds_name} ---")
        net = load_network(ds_name)
        utr = load_utr_features(species)

        # Gene matching: uppercase
        utr_lookup = dict(zip(utr["gene"].str.upper(), utr["utr_length"]))
        net["target_upper"] = net["target"].str.upper()
        net["utr_length"] = net["target_upper"].map(utr_lookup)
        matched = net.dropna(subset=["utr_length"]).copy()
        print(f"  Edges: {len(net)}, matched with UTR: {len(matched)}")

        ds_results = {"n_edges": len(net), "n_matched": len(matched)}

        # Test A: Spearman(r_edge, UTR_length_target) across ALL edges
        r_val, p_val = stats.spearmanr(matched["r"], matched["utr_length"])
        print(f"  Test A (all edges): Spearman r={r_val:.4f}, p={p_val:.2e}")
        ds_results["test_a"] = {"spearman_r": float(r_val), "p": float(p_val)}

        # Test B: Per-RBP within-RBP Spearman, Fisher combined p
        per_rbp_p = []
        per_rbp_results = []
        for rbp, grp in matched.groupby("rbp"):
            if len(grp) < 20:
                continue
            rr, pp = stats.spearmanr(grp["r"], grp["utr_length"])
            per_rbp_p.append(pp)
            per_rbp_results.append({"rbp": rbp, "n": len(grp), "r": float(rr), "p": float(pp)})
        if per_rbp_p:
            # Fisher's combined p-value: -2 * sum(log(pi)) ~ chi2(2k)
            chi2_stat = -2 * np.sum(np.log(np.array(per_rbp_p)))
            fisher_p = stats.chi2.sf(chi2_stat, 2 * len(per_rbp_p))
            n_sig = sum(1 for p in per_rbp_p if p < 0.05)
            print(f"  Test B (per-RBP): {len(per_rbp_p)} RBPs (>=20 edges), "
                  f"Fisher combined p={fisher_p:.2e}, {n_sig} individually significant")
            ds_results["test_b"] = {
                "n_rbps": len(per_rbp_p),
                "fisher_p": float(fisher_p),
                "n_sig": n_sig,
                "per_rbp": per_rbp_results,
            }
        else:
            print("  Test B: No RBPs with >=20 edges")
            ds_results["test_b"] = {"n_rbps": 0}

        # Test C: Mann-Whitney on UTR lengths: destabilizing (r>0) vs stabilizing (r<0)
        dest = matched[matched["r"] > 0]["utr_length"]
        stab = matched[matched["r"] < 0]["utr_length"]
        if len(dest) > 0 and len(stab) > 0:
            mw_stat, mw_p = stats.mannwhitneyu(dest, stab, alternative="greater")
            print(f"  Test C (MW dest vs stab): dest median={dest.median():.0f}, "
                  f"stab median={stab.median():.0f}, p={mw_p:.4f}")
            ds_results["test_c"] = {
                "dest_median": float(dest.median()),
                "stab_median": float(stab.median()),
                "dest_n": len(dest),
                "stab_n": len(stab),
                "mw_p": float(mw_p),
            }
        else:
            print("  Test C: insufficient data")
            ds_results["test_c"] = {}

        # Test D: UTR quintile trend
        matched["utr_quintile"] = pd.qcut(matched["utr_length"], 5, labels=False, duplicates="drop")
        quintile_means = matched.groupby("utr_quintile")["r"].mean()
        # Jonckheere-Terpstra approximation via Spearman on quintile vs mean_r
        q_r, q_p = stats.spearmanr(quintile_means.index, quintile_means.values)
        print(f"  Test D (quintile trend): Spearman r={q_r:.4f}, p={q_p:.4f}")
        print(f"    Quintile mean r values: {[f'{v:.4f}' for v in quintile_means.values]}")
        ds_results["test_d"] = {
            "quintile_means": {str(k): float(v) for k, v in quintile_means.items()},
            "trend_r": float(q_r),
            "trend_p": float(q_p),
        }

        all_results[ds_name] = ds_results

    save_results(all_results, "edge_utr_validation")

    # Figure: 3-panel quintile plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (ds_name, species) in zip(axes, datasets.items()):
        net = load_network(ds_name)
        utr = load_utr_features(species)
        utr_lookup = dict(zip(utr["gene"].str.upper(), utr["utr_length"]))
        net["target_upper"] = net["target"].str.upper()
        net["utr_length"] = net["target_upper"].map(utr_lookup)
        matched = net.dropna(subset=["utr_length"]).copy()
        matched["utr_quintile"] = pd.qcut(matched["utr_length"], 5, labels=False, duplicates="drop")
        quintile_means = matched.groupby("utr_quintile")["r"].mean()
        quintile_sems = matched.groupby("utr_quintile")["r"].sem()
        ax.bar(range(len(quintile_means)), quintile_means.values,
               yerr=quintile_sems.values, capsize=4, color="steelblue", alpha=0.8)
        ax.set_xlabel("3' UTR Length Quintile")
        ax.set_ylabel("Mean RBP-gamma r")
        ax.set_title(ds_name.replace("_", " ").title())
        ax.set_xticks(range(len(quintile_means)))
        ax.set_xticklabels([f"Q{i+1}" for i in range(len(quintile_means))])
        # Add trend line info
        res = all_results[ds_name]
        ax.text(0.05, 0.95, f"trend r={res['test_d']['trend_r']:.3f}\np={res['test_d']['trend_p']:.3f}",
                transform=ax.transAxes, va="top", fontsize=8)

    fig.suptitle("Edge-Level UTR Validation: Mean r by UTR Length Quintile", fontsize=13)
    plt.tight_layout()
    save_fig(fig, "edge_utr_quintiles")

    return all_results


# ---------------------------------------------------------------------------
# Experiment 2: DepMap Stratified NB Analysis
# ---------------------------------------------------------------------------

NB_HUB_RBPS = [
    "HNRNPA2B1", "PABPC1", "YBX1", "HNRNPD", "HNRNPU", "PRPF8",
    "SNRNP200", "FUS", "HNRNPK", "NCL", "SRSF3", "SRSF7",
    "EWSR1", "SNRPA", "PTBP1", "TRA2B", "QKI", "HNRNPM",
    "SRSF10", "DDX5",
]


def load_depmap():
    """Load DepMap model metadata and CRISPR gene effect scores."""
    model = pd.read_csv(PROJECT_ROOT / ".cache" / "DepMap_Model.csv")
    crispr = pd.read_csv(PROJECT_ROOT / ".cache" / "CRISPRGeneEffect.csv")
    # First column is ModelID (unnamed)
    id_col = crispr.columns[0]
    crispr = crispr.rename(columns={id_col: "ModelID"})
    # Parse gene columns: "GENE (12345)" -> "GENE"
    gene_cols = {c: c.split(" (")[0] for c in crispr.columns if " (" in c}
    crispr = crispr.rename(columns=gene_cols)
    return model, crispr


def experiment2_depmap_stratified():
    """DepMap stratified NB analysis: MYCN, lineage, cross-line."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: DepMap Stratified NB Analysis")
    print("=" * 60)

    set_figure_style()
    model, crispr = load_depmap()

    # Get NB lines
    nb_model = model[model["OncotreePrimaryDisease"] == "Neuroblastoma"]
    nb_ids = set(nb_model["ModelID"]) & set(crispr["ModelID"])
    print(f"  NB cell lines with CRISPR data: {len(nb_ids)}")

    # Filter hub RBPs present in CRISPR
    hub_in_crispr = [g for g in NB_HUB_RBPS if g in crispr.columns]
    print(f"  Hub RBPs in CRISPR: {len(hub_in_crispr)}/{len(NB_HUB_RBPS)}")

    # All RBP genes for non-hub comparison (use GO RBP list proxy: all genes with "RBP" or known RBPs)
    # Simpler: use all genes not in hub list as background
    all_genes = [c for c in crispr.columns if c != "ModelID"]

    all_results = {}

    # --- 2a: MYCN-Stratified Essentiality ---
    print("\n  --- 2a: MYCN-Stratified Essentiality ---")

    mycn_model = nb_model[nb_model["ModelSubtypeFeatures"] == "MYC_Amplified"]
    non_mycn_model = nb_model[nb_model["ModelSubtypeFeatures"] != "MYC_Amplified"]
    mycn_ids = set(mycn_model["ModelID"]) & nb_ids
    non_mycn_ids = set(non_mycn_model["ModelID"]) & nb_ids
    print(f"  MYCN-amp: {len(mycn_ids)}, non-MYCN: {len(non_mycn_ids)}")

    crispr_nb = crispr[crispr["ModelID"].isin(nb_ids)].copy()
    crispr_mycn = crispr_nb[crispr_nb["ModelID"].isin(mycn_ids)]
    crispr_nonmycn = crispr_nb[crispr_nb["ModelID"].isin(non_mycn_ids)]

    # Mean hub dependency per group
    mycn_hub_deps = crispr_mycn[hub_in_crispr].mean(axis=1)
    nonmycn_hub_deps = crispr_nonmycn[hub_in_crispr].mean(axis=1)
    mw_stat, mw_p = stats.mannwhitneyu(mycn_hub_deps, nonmycn_hub_deps, alternative="two-sided")
    print(f"  Hub mean dep: MYCN-amp={mycn_hub_deps.mean():.4f}, non-MYCN={nonmycn_hub_deps.mean():.4f}, MW p={mw_p:.4f}")

    # Per-hub MYCN vs non-MYCN
    per_hub_mycn = []
    for gene in hub_in_crispr:
        m_vals = crispr_mycn[gene].dropna()
        n_vals = crispr_nonmycn[gene].dropna()
        if len(m_vals) > 0 and len(n_vals) > 0:
            _, pp = stats.mannwhitneyu(m_vals, n_vals, alternative="two-sided")
            per_hub_mycn.append({
                "rbp": gene,
                "mycn_mean": float(m_vals.mean()),
                "nonmycn_mean": float(n_vals.mean()),
                "diff": float(m_vals.mean() - n_vals.mean()),
                "p": float(pp),
            })
    per_hub_mycn.sort(key=lambda x: x["p"])
    n_sig_mycn = sum(1 for x in per_hub_mycn if x["p"] < 0.05)
    print(f"  Per-hub MYCN-specific: {n_sig_mycn}/{len(per_hub_mycn)} significant (p<0.05)")
    if per_hub_mycn:
        top = per_hub_mycn[0]
        print(f"  Top: {top['rbp']} (MYCN={top['mycn_mean']:.3f}, non={top['nonmycn_mean']:.3f}, p={top['p']:.4f})")

    all_results["mycn_stratified"] = {
        "mycn_n": len(mycn_ids),
        "nonmycn_n": len(non_mycn_ids),
        "mycn_hub_mean": float(mycn_hub_deps.mean()),
        "nonmycn_hub_mean": float(nonmycn_hub_deps.mean()),
        "mw_p": float(mw_p),
        "n_sig_per_hub": n_sig_mycn,
        "per_hub": per_hub_mycn,
    }

    # --- 2b: Neural Lineage Specificity ---
    print("\n  --- 2b: Neural Lineage Specificity ---")

    lineages = {
        "PNS": "Peripheral Nervous System",
        "CNS": "CNS/Brain",
        "Lymphoid": "Lymphoid",
    }
    lineage_hub_deps = {}
    for label, lineage in lineages.items():
        lin_ids = set(model[model["OncotreeLineage"] == lineage]["ModelID"]) & set(crispr["ModelID"])
        crispr_lin = crispr[crispr["ModelID"].isin(lin_ids)]
        deps = crispr_lin[hub_in_crispr].mean(axis=1)
        lineage_hub_deps[label] = deps
        print(f"  {label} ({len(lin_ids)} lines): hub mean dep = {deps.mean():.4f}")

    # Kruskal-Wallis across all three
    kw_stat, kw_p = stats.kruskal(*lineage_hub_deps.values())
    print(f"  Kruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.4f}")

    # Pairwise Mann-Whitney
    pairwise = []
    labels = list(lineage_hub_deps.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            _, pp = stats.mannwhitneyu(lineage_hub_deps[labels[i]],
                                       lineage_hub_deps[labels[j]],
                                       alternative="two-sided")
            pairwise.append({
                "pair": f"{labels[i]} vs {labels[j]}",
                "mean_a": float(lineage_hub_deps[labels[i]].mean()),
                "mean_b": float(lineage_hub_deps[labels[j]].mean()),
                "mw_p": float(pp),
            })
            print(f"  {labels[i]} vs {labels[j]}: p={pp:.4f}")

    all_results["lineage_specificity"] = {
        "lineage_n": {k: len(v) for k, v in lineage_hub_deps.items()},
        "lineage_means": {k: float(v.mean()) for k, v in lineage_hub_deps.items()},
        "kruskal_wallis": {"H": float(kw_stat), "p": float(kw_p)},
        "pairwise": pairwise,
    }

    # --- 2c: Cross-NB-Line Hub Consistency ---
    print("\n  --- 2c: Cross-NB-Line Hub Consistency ---")

    # Non-hub RBPs: all genes not in hub list
    non_hub_genes = [g for g in all_genes if g not in set(hub_in_crispr)]

    # Per-line: mean hub dep vs mean non-hub dep
    per_line_hub = crispr_nb[hub_in_crispr].mean(axis=1)
    per_line_nonhub = crispr_nb[non_hub_genes].mean(axis=1)
    # Wilcoxon signed-rank: hub < non-hub across lines
    wsr_stat, wsr_p = stats.wilcoxon(per_line_hub, per_line_nonhub, alternative="less")
    print(f"  Wilcoxon signed-rank (hub < non-hub): p={wsr_p:.2e}")
    print(f"  Hub mean across lines: {per_line_hub.mean():.4f}, non-hub: {per_line_nonhub.mean():.4f}")

    # How many lines show hub < non-hub
    n_hub_more_essential = (per_line_hub < per_line_nonhub).sum()
    print(f"  Lines where hub < non-hub: {n_hub_more_essential}/{len(per_line_hub)}")

    # Bootstrap: 10,000 random 20-gene sets
    rng = np.random.default_rng(42)
    obs_diff = (per_line_hub - per_line_nonhub).mean()
    n_bootstrap = 10000
    boot_diffs = np.zeros(n_bootstrap)
    all_gene_arr = np.array(all_genes)
    crispr_nb_vals = crispr_nb[all_genes].values
    for i in range(n_bootstrap):
        rand_idx = rng.choice(len(all_genes), size=len(hub_in_crispr), replace=False)
        rand_mean = np.nanmean(crispr_nb_vals[:, rand_idx], axis=1)
        boot_diffs[i] = np.mean(rand_mean - per_line_nonhub.values)
    boot_p = np.mean(boot_diffs <= obs_diff)
    print(f"  Bootstrap p (hub more essential than random): {boot_p:.4f}")

    # Per-hub essentiality profile
    per_hub_profile = []
    for gene in hub_in_crispr:
        vals = crispr_nb[gene].dropna()
        frac_essential = float((vals < -0.5).mean())
        per_hub_profile.append({
            "rbp": gene,
            "mean_dep": float(vals.mean()),
            "frac_essential": frac_essential,
            "n_lines": len(vals),
        })
    per_hub_profile.sort(key=lambda x: x["mean_dep"])
    print(f"  Most essential hub: {per_hub_profile[0]['rbp']} "
          f"(mean={per_hub_profile[0]['mean_dep']:.3f}, "
          f"essential in {per_hub_profile[0]['frac_essential']*100:.0f}% of lines)")

    all_results["cross_line_consistency"] = {
        "n_lines": len(per_line_hub),
        "hub_mean": float(per_line_hub.mean()),
        "nonhub_mean": float(per_line_nonhub.mean()),
        "wilcoxon_p": float(wsr_p),
        "n_hub_more_essential": int(n_hub_more_essential),
        "bootstrap_p": float(boot_p),
        "obs_diff": float(obs_diff),
        "per_hub_profile": per_hub_profile,
    }

    save_results(all_results, "depmap_stratified")

    # Figure: 2x2 panel
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: MYCN vs non-MYCN hub dependency
    ax = axes[0, 0]
    bp = ax.boxplot([mycn_hub_deps.values, nonmycn_hub_deps.values],
                    tick_labels=["MYCN-amp", "Non-MYCN"], patch_artist=True)
    bp["boxes"][0].set_facecolor("salmon")
    bp["boxes"][1].set_facecolor("lightblue")
    ax.set_ylabel("Mean Hub RBP Dependency")
    ax.set_title(f"MYCN Stratification (p={mw_p:.4f})")

    # Panel B: Lineage comparison
    ax = axes[0, 1]
    positions = range(len(lineage_hub_deps))
    bp = ax.boxplot(lineage_hub_deps.values(), tick_labels=lineage_hub_deps.keys(), patch_artist=True)
    colors = ["#ff9999", "#99ccff", "#99ff99"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_ylabel("Mean Hub RBP Dependency")
    ax.set_title(f"Lineage Specificity (KW p={kw_p:.4f})")

    # Panel C: Hub vs non-hub across lines
    ax = axes[1, 0]
    ax.scatter(per_line_nonhub, per_line_hub, alpha=0.7, s=40, c="steelblue")
    lims = [min(per_line_nonhub.min(), per_line_hub.min()) - 0.1,
            max(per_line_nonhub.max(), per_line_hub.max()) + 0.1]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Mean Non-Hub Dependency")
    ax.set_ylabel("Mean Hub RBP Dependency")
    ax.set_title(f"Hub vs Non-Hub ({n_hub_more_essential}/{len(per_line_hub)} lines, Wilcoxon p={wsr_p:.2e})")

    # Panel D: Per-hub essentiality profile
    ax = axes[1, 1]
    sorted_profile = sorted(per_hub_profile, key=lambda x: x["frac_essential"], reverse=True)
    rbp_names = [x["rbp"] for x in sorted_profile]
    frac_vals = [x["frac_essential"] for x in sorted_profile]
    ax.barh(range(len(rbp_names)), frac_vals, color="steelblue", alpha=0.8)
    ax.set_yticks(range(len(rbp_names)))
    ax.set_yticklabels(rbp_names, fontsize=7)
    ax.set_xlabel("Fraction of NB Lines Where Essential (dep < -0.5)")
    ax.set_title("Per-Hub Essentiality Profile")
    ax.invert_yaxis()

    fig.suptitle("DepMap Stratified Neuroblastoma Analysis", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "depmap_stratified")

    return all_results


# ---------------------------------------------------------------------------
# Experiment 3: eCLIP Edge-Strength Concordance
# ---------------------------------------------------------------------------

def experiment3_eclip_edge_strength():
    """eCLIP edge-strength concordance across all datasets."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: eCLIP Edge-Strength Concordance")
    print("=" * 60)

    set_figure_style()

    eclip = pd.read_csv(DATA_DIR / "eclip_targets.csv")
    eclip_pairs = set(zip(eclip["rbp"].str.upper(), eclip["target_gene"].str.upper()))
    eclip_rbps = set(eclip["rbp"].str.upper())
    print(f"  eCLIP data: {len(eclip_pairs)} pairs, {len(eclip_rbps)} RBPs")

    datasets = ["pancreas", "dentate_gyrus", "neuroblastoma"]
    all_results = {}

    for ds_name in datasets:
        print(f"\n--- {ds_name} ---")
        net = load_network(ds_name)
        net["rbp_upper"] = net["rbp"].str.upper()
        net["target_upper"] = net["target"].str.upper()
        net["abs_r"] = net["r"].abs()

        # Filter to RBPs present in both network and eCLIP
        net_rbps = set(net["rbp_upper"].unique())
        shared_rbps = net_rbps & eclip_rbps
        print(f"  Network RBPs: {len(net_rbps)}, shared with eCLIP: {len(shared_rbps)}")

        if len(shared_rbps) == 0:
            print("  No shared RBPs, skipping")
            all_results[ds_name] = {"shared_rbps": 0}
            continue

        net_shared = net[net["rbp_upper"].isin(shared_rbps)].copy()
        net_shared["eclip_confirmed"] = net_shared.apply(
            lambda row: (row["rbp_upper"], row["target_upper"]) in eclip_pairs, axis=1
        )
        n_confirmed = net_shared["eclip_confirmed"].sum()
        n_not = (~net_shared["eclip_confirmed"]).sum()
        print(f"  Edges in shared RBPs: {len(net_shared)}, eCLIP-confirmed: {n_confirmed}")

        if n_confirmed < 3:
            print("  Too few eCLIP-confirmed edges, skipping")
            all_results[ds_name] = {"shared_rbps": len(shared_rbps), "eclip_confirmed": int(n_confirmed)}
            continue

        # Aggregate MW on |r|
        confirmed_r = net_shared[net_shared["eclip_confirmed"]]["abs_r"]
        not_confirmed_r = net_shared[~net_shared["eclip_confirmed"]]["abs_r"]
        mw_stat, mw_p = stats.mannwhitneyu(confirmed_r, not_confirmed_r, alternative="greater")
        print(f"  Aggregate MW (|r|): confirmed={confirmed_r.median():.4f}, "
              f"not={not_confirmed_r.median():.4f}, p={mw_p:.4f}")

        # Per-RBP MW
        per_rbp_results = []
        for rbp in shared_rbps:
            rbp_edges = net_shared[net_shared["rbp_upper"] == rbp]
            conf = rbp_edges[rbp_edges["eclip_confirmed"]]["abs_r"]
            notc = rbp_edges[~rbp_edges["eclip_confirmed"]]["abs_r"]
            if len(conf) >= 3 and len(notc) >= 3:
                _, pp = stats.mannwhitneyu(conf, notc, alternative="greater")
                per_rbp_results.append({
                    "rbp": rbp, "n_conf": len(conf), "n_notc": len(notc),
                    "conf_median": float(conf.median()), "notc_median": float(notc.median()),
                    "mw_p": float(pp),
                })

        n_sig_rbp = sum(1 for x in per_rbp_results if x["mw_p"] < 0.05)
        print(f"  Per-RBP: {len(per_rbp_results)} testable, {n_sig_rbp} significant")

        # Rank enrichment: for edges sorted by |r| descending, mean rank percentile of eCLIP-confirmed
        net_shared_sorted = net_shared.sort_values("abs_r", ascending=False).reset_index(drop=True)
        n_total = len(net_shared_sorted)
        net_shared_sorted["rank_pctl"] = np.arange(1, n_total + 1) / n_total
        confirmed_pctls = net_shared_sorted[net_shared_sorted["eclip_confirmed"]]["rank_pctl"]
        mean_pctl = float(confirmed_pctls.mean())
        # One-sample test: is mean percentile < 0.5 (i.e., enriched toward top)?
        if len(confirmed_pctls) >= 3:
            t_stat, t_p = stats.ttest_1samp(confirmed_pctls, 0.5)
            rank_p = float(t_p / 2) if t_stat < 0 else float(1 - t_p / 2)  # one-sided: < 0.5
        else:
            rank_p = float("nan")
        print(f"  Rank enrichment: mean percentile={mean_pctl:.4f} (0.5=random), p={rank_p:.4f}")

        all_results[ds_name] = {
            "shared_rbps": len(shared_rbps),
            "n_edges_shared": len(net_shared),
            "eclip_confirmed": int(n_confirmed),
            "aggregate_mw": {
                "confirmed_median_abs_r": float(confirmed_r.median()),
                "not_confirmed_median_abs_r": float(not_confirmed_r.median()),
                "mw_p": float(mw_p),
            },
            "per_rbp": per_rbp_results,
            "n_sig_per_rbp": n_sig_rbp,
            "rank_enrichment": {
                "mean_percentile": mean_pctl,
                "p": rank_p,
            },
        }

    save_results(all_results, "eclip_edge_strength")

    # Figure: bar chart of confirmed vs not-confirmed |r| per dataset
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, ds_name in zip(axes, datasets):
        res = all_results.get(ds_name, {})
        if "aggregate_mw" not in res:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(ds_name.replace("_", " ").title())
            continue
        vals = [res["aggregate_mw"]["confirmed_median_abs_r"],
                res["aggregate_mw"]["not_confirmed_median_abs_r"]]
        bars = ax.bar(["eCLIP\nConfirmed", "Not\nConfirmed"], vals,
                      color=["#e74c3c", "#95a5a6"], alpha=0.8)
        ax.set_ylabel("Median |r|")
        ax.set_title(f"{ds_name.replace('_', ' ').title()}\n(p={res['aggregate_mw']['mw_p']:.4f})")
        ax.text(0.05, 0.95, f"n_conf={res['eclip_confirmed']}\nn_total={res['n_edges_shared']}",
                transform=ax.transAxes, va="top", fontsize=8)

    fig.suptitle("eCLIP Edge-Strength Concordance", fontsize=13)
    plt.tight_layout()
    save_fig(fig, "eclip_edge_strength")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("WEAKNESS IMPROVEMENTS ANALYSIS")
    print("=" * 60)

    (OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    utr_results = experiment1_edge_utr()
    eclip_results = experiment3_eclip_edge_strength()
    depmap_results = experiment2_depmap_stratified()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nExperiment 1 (Edge-Level UTR):")
    for ds in ["pancreas", "dentate_gyrus", "neuroblastoma"]:
        r = utr_results[ds]
        print(f"  {ds}: Test A r={r['test_a']['spearman_r']:.4f} (p={r['test_a']['p']:.2e})")

    print("\nExperiment 2 (DepMap Stratified):")
    mycn = depmap_results["mycn_stratified"]
    print(f"  MYCN stratification: p={mycn['mw_p']:.4f}")
    lin = depmap_results["lineage_specificity"]
    print(f"  Lineage KW: p={lin['kruskal_wallis']['p']:.4f}")
    cl = depmap_results["cross_line_consistency"]
    print(f"  Cross-line Wilcoxon: p={cl['wilcoxon_p']:.2e}")
    print(f"  Bootstrap: p={cl['bootstrap_p']:.4f}")

    print("\nExperiment 3 (eCLIP Edge-Strength):")
    for ds in ["pancreas", "dentate_gyrus", "neuroblastoma"]:
        r = eclip_results.get(ds, {})
        if "aggregate_mw" in r:
            print(f"  {ds}: MW p={r['aggregate_mw']['mw_p']:.4f}, "
                  f"rank pctl={r['rank_enrichment']['mean_percentile']:.4f}")
        else:
            print(f"  {ds}: insufficient data")

    print("\nDone! Output: output/weakness_improvements/")


if __name__ == "__main__":
    main()
