#!/usr/bin/env python
"""Tier 2 validation: sequence-feature correlations and eCLIP validation.

T2-4: Correlate gamma with 3' UTR length and AU content
      (sequence-feature-based validation, replacing curated gene lists)
T2-5: Validate RBP-target network predictions against ENCODE eCLIP data
      (Fisher's exact test for overlap enrichment)
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

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "tier2_validation"
DATA_DIR = Path(__file__).parent.parent / "src" / "scptr" / "benchmark" / "data"


def save_fig(fig, name, subdir="figures"):
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
# T2-4: Sequence-feature validation
# =========================================================================
def sequence_feature_validation(adata, name, species):
    """Correlate per-gene gamma with 3' UTR length and AU content.

    Hypothesis:
    - Longer 3' UTRs → more regulatory elements → higher gamma (positive corr)
    - Higher AU content → ARE-mediated decay → higher gamma (positive corr)
    """
    print(f"\n{'='*60}")
    print(f"T2-4: SEQUENCE FEATURE VALIDATION ({name})")
    print(f"{'='*60}")

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load UTR features
    utr_file = DATA_DIR / f"{species}_utr_features.csv"
    if not utr_file.exists():
        print(f"  ERROR: {utr_file} not found. Run download_utr_features.py first.")
        return None
    utr_df = pd.read_csv(utr_file)
    print(f"  Loaded {len(utr_df)} {species} genes with UTR features")

    # Per-gene median gamma
    gamma = adata.layers["gamma"]
    gene_names = adata.var_names.tolist()
    median_gamma = np.median(gamma, axis=0)
    nonzero_frac = (gamma > 0).mean(axis=0)

    # Build gene-level DataFrame
    gamma_df = pd.DataFrame({
        "gene": gene_names,
        "median_gamma": median_gamma,
        "nonzero_frac": nonzero_frac,
    })

    # Filter to gamma-informative genes
    gamma_df = gamma_df[gamma_df["nonzero_frac"] >= 0.1].copy()
    print(f"  Gamma-informative genes: {len(gamma_df)}")

    # Case-insensitive merge
    gamma_df["gene_upper"] = gamma_df["gene"].str.upper()
    utr_df["gene_upper"] = utr_df["gene"].str.upper()

    merged = gamma_df.merge(utr_df[["gene_upper", "utr_length", "au_content"]],
                            on="gene_upper", how="inner")
    print(f"  Merged with UTR features: {len(merged)} genes")

    if len(merged) < 50:
        print("  Too few genes for analysis")
        return None

    # Filter extreme outliers
    merged = merged[merged["utr_length"] > 0].copy()
    merged["log_utr_length"] = np.log10(merged["utr_length"])
    merged["log_gamma"] = np.log1p(merged["median_gamma"])

    results = {}

    # 1. Gamma vs UTR length
    r_len, p_len = stats.spearmanr(merged["log_utr_length"], merged["median_gamma"])
    print(f"\n  Gamma vs log10(UTR length):")
    print(f"    Spearman r = {r_len:.4f}, p = {p_len:.2e}")
    print(f"    n = {len(merged)} genes")
    results["utr_length_spearman_r"] = float(r_len)
    results["utr_length_p"] = float(p_len)

    # 2. Gamma vs AU content
    r_au, p_au = stats.spearmanr(merged["au_content"], merged["median_gamma"])
    print(f"\n  Gamma vs AU content:")
    print(f"    Spearman r = {r_au:.4f}, p = {p_au:.2e}")
    results["au_content_spearman_r"] = float(r_au)
    results["au_content_p"] = float(p_au)

    # 3. Quartile analysis: genes in top vs bottom UTR length quartile
    q1 = merged["log_utr_length"].quantile(0.25)
    q4 = merged["log_utr_length"].quantile(0.75)
    short_utr = merged[merged["log_utr_length"] <= q1]
    long_utr = merged[merged["log_utr_length"] >= q4]

    median_gamma_short = short_utr["median_gamma"].median()
    median_gamma_long = long_utr["median_gamma"].median()
    u_stat, u_p = stats.mannwhitneyu(long_utr["median_gamma"],
                                      short_utr["median_gamma"],
                                      alternative="greater")
    print(f"\n  Quartile analysis (UTR length):")
    print(f"    Short UTR (Q1) median gamma: {median_gamma_short:.4f} (n={len(short_utr)})")
    print(f"    Long UTR (Q4) median gamma: {median_gamma_long:.4f} (n={len(long_utr)})")
    print(f"    Mann-Whitney (long > short): p = {u_p:.2e}")
    results["long_vs_short_utr_mw_p"] = float(u_p)
    results["median_gamma_short_utr"] = float(median_gamma_short)
    results["median_gamma_long_utr"] = float(median_gamma_long)

    # 4. AU content quartile
    au_q1 = merged["au_content"].quantile(0.25)
    au_q4 = merged["au_content"].quantile(0.75)
    low_au = merged[merged["au_content"] <= au_q1]
    high_au = merged[merged["au_content"] >= au_q4]

    median_gamma_low_au = low_au["median_gamma"].median()
    median_gamma_high_au = high_au["median_gamma"].median()
    au_u_stat, au_u_p = stats.mannwhitneyu(high_au["median_gamma"],
                                            low_au["median_gamma"],
                                            alternative="greater")
    print(f"\n  Quartile analysis (AU content):")
    print(f"    Low AU (Q1) median gamma: {median_gamma_low_au:.4f} (n={len(low_au)})")
    print(f"    High AU (Q4) median gamma: {median_gamma_high_au:.4f} (n={len(high_au)})")
    print(f"    Mann-Whitney (high AU > low AU): p = {au_u_p:.2e}")
    results["high_vs_low_au_mw_p"] = float(au_u_p)

    results["n_genes"] = len(merged)

    # Figure: 2x2 scatter + quartile boxplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Scatter: gamma vs UTR length
    axes[0, 0].scatter(merged["log_utr_length"], merged["log_gamma"],
                       s=2, alpha=0.3, color="steelblue")
    axes[0, 0].set_xlabel("log10(3' UTR length)")
    axes[0, 0].set_ylabel("log1p(median gamma)")
    axes[0, 0].set_title(f"Gamma vs 3' UTR Length ({name})\nr={r_len:.3f}, p={p_len:.1e}")

    # Scatter: gamma vs AU content
    axes[0, 1].scatter(merged["au_content"], merged["log_gamma"],
                       s=2, alpha=0.3, color="darkorange")
    axes[0, 1].set_xlabel("3' UTR AU content")
    axes[0, 1].set_ylabel("log1p(median gamma)")
    axes[0, 1].set_title(f"Gamma vs AU Content ({name})\nr={r_au:.3f}, p={p_au:.1e}")

    # Boxplot: UTR length quartiles
    quartile_data = []
    quartile_labels = []
    for qi, (lo, hi, label) in enumerate([
        (0, 0.25, "Q1\n(short)"), (0.25, 0.5, "Q2"), (0.5, 0.75, "Q3"),
        (0.75, 1.0, "Q4\n(long)")
    ]):
        qlo = merged["log_utr_length"].quantile(lo)
        qhi = merged["log_utr_length"].quantile(hi)
        mask = (merged["log_utr_length"] >= qlo) & (merged["log_utr_length"] <= qhi)
        quartile_data.append(merged.loc[mask, "median_gamma"].values)
        quartile_labels.append(label)
    bp = axes[1, 0].boxplot(quartile_data, labels=quartile_labels, patch_artist=True,
                             showfliers=False)
    colors = ["#2196F3", "#64B5F6", "#FFA726", "#E65100"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    axes[1, 0].set_ylabel("Median gamma")
    axes[1, 0].set_xlabel("3' UTR Length Quartile")
    axes[1, 0].set_title(f"Gamma by UTR Length Quartile\np={u_p:.1e}")

    # Boxplot: AU content quartiles
    au_data = []
    au_labels = []
    for qi, (lo, hi, label) in enumerate([
        (0, 0.25, "Q1\n(low AU)"), (0.25, 0.5, "Q2"), (0.5, 0.75, "Q3"),
        (0.75, 1.0, "Q4\n(high AU)")
    ]):
        qlo = merged["au_content"].quantile(lo)
        qhi = merged["au_content"].quantile(hi)
        mask = (merged["au_content"] >= qlo) & (merged["au_content"] <= qhi)
        au_data.append(merged.loc[mask, "median_gamma"].values)
        au_labels.append(label)
    bp2 = axes[1, 1].boxplot(au_data, labels=au_labels, patch_artist=True,
                              showfliers=False)
    colors2 = ["#4CAF50", "#81C784", "#FFB74D", "#FF5722"]
    for patch, color in zip(bp2["boxes"], colors2):
        patch.set_facecolor(color)
    axes[1, 1].set_ylabel("Median gamma")
    axes[1, 1].set_xlabel("3' UTR AU Content Quartile")
    axes[1, 1].set_title(f"Gamma by AU Content Quartile\np={au_u_p:.1e}")

    fig.suptitle(f"Sequence Feature Validation: {name}", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, f"seq_features_{name}")

    return results


# =========================================================================
# T2-5: eCLIP validation of RBP-target networks
# =========================================================================
def eclip_validation(adata, name):
    """Validate scPTR-predicted RBP-target edges against ENCODE eCLIP data.

    For each RBP with both scPTR predictions and eCLIP data:
    - Fisher's exact test: are predicted targets enriched for eCLIP-confirmed targets?
    - Report odds ratio and p-value
    """
    print(f"\n{'='*60}")
    print(f"T2-5: eCLIP VALIDATION ({name})")
    print(f"{'='*60}")

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load eCLIP targets
    eclip_file = DATA_DIR / "eclip_targets.csv"
    if not eclip_file.exists():
        print(f"  ERROR: {eclip_file} not found. Run download_eclip.py first.")
        return None
    eclip_df = pd.read_csv(eclip_file)
    print(f"  Loaded {len(eclip_df)} eCLIP RBP-target pairs")

    # Build eCLIP target sets per RBP (uppercase for matching)
    eclip_targets = {}
    for rbp, grp in eclip_df.groupby("rbp"):
        eclip_targets[rbp.upper()] = set(g.upper() for g in grp["target_gene"])

    # Get scPTR network edges
    gamma = adata.layers["gamma"]
    gene_names = adata.var_names.tolist()
    gene_upper = [g.upper() for g in gene_names]

    # Load RBP list
    rbp_path = Path(__file__).parent.parent / "src" / "scptr" / "tools" / "data" / "known_rbps.csv"
    rbps = pd.read_csv(rbp_path)["gene_symbol"].tolist()

    # Find RBPs in dataset
    adata_gene_map = {g.upper(): i for i, g in enumerate(gene_names)}
    rbp_in_data = {}
    for r in rbps:
        if r.upper() in adata_gene_map:
            rbp_in_data[r.upper()] = adata_gene_map[r.upper()]

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
    target_genes_upper = set(gene_upper[i] for i in target_indices)

    # Compute scPTR network edges via Spearman correlation
    print("  Computing scPTR network edges...")
    scptr_edges = {}  # rbp_upper -> set of target_gene_upper

    for rbp_upper, rbp_idx in rbp_in_data.items():
        rbp_expr = expr[:, rbp_idx]
        if np.std(rbp_expr) < 1e-6:
            continue

        targets = set()
        for ti in target_indices:
            target_gamma = gamma[:, ti]
            valid = target_gamma > 0
            if valid.sum() < 50:
                continue

            r, p = stats.spearmanr(rbp_expr[valid], target_gamma[valid])
            # Bonferroni correction
            if p < 0.05 / (len(rbp_in_data) * n_targets):
                targets.add(gene_upper[ti])

        if targets:
            scptr_edges[rbp_upper] = targets

    print(f"  scPTR edges: {sum(len(t) for t in scptr_edges.values())} total")
    print(f"  RBPs with edges: {len(scptr_edges)}")

    # All genes in dataset (uppercase) as universe
    all_genes_upper = set(gene_upper)

    # Fisher's exact test for each RBP with both scPTR and eCLIP data
    results = []

    for rbp_upper in sorted(set(scptr_edges.keys()) & set(eclip_targets.keys())):
        predicted = scptr_edges[rbp_upper]
        eclip = eclip_targets[rbp_upper]

        # Restrict eCLIP targets to genes in our dataset
        eclip_in_data = eclip & all_genes_upper
        if len(eclip_in_data) < 10:
            continue

        # 2x2 contingency table
        # predicted & eCLIP | predicted & ~eCLIP
        # ~predicted & eCLIP | ~predicted & ~eCLIP
        a = len(predicted & eclip_in_data)
        b = len(predicted - eclip_in_data)
        c = len(eclip_in_data - predicted)
        d = len(all_genes_upper - predicted - eclip_in_data)

        odds_ratio, p_val = stats.fisher_exact([[a, b], [c, d]], alternative="greater")

        # Also compute simple overlap statistics
        overlap_frac = a / max(len(predicted), 1)
        expected_frac = len(eclip_in_data) / max(len(all_genes_upper), 1)
        enrichment = overlap_frac / max(expected_frac, 1e-6)

        print(f"\n  {rbp_upper}:")
        print(f"    scPTR predicted targets: {len(predicted)}")
        print(f"    eCLIP confirmed targets: {len(eclip_in_data)}")
        print(f"    Overlap: {a}")
        print(f"    Enrichment fold: {enrichment:.2f}x")
        print(f"    Fisher's exact: OR={odds_ratio:.2f}, p={p_val:.4f}")

        results.append({
            "rbp": rbp_upper,
            "n_predicted": len(predicted),
            "n_eclip": len(eclip_in_data),
            "n_overlap": a,
            "odds_ratio": float(odds_ratio),
            "p_value": float(p_val),
            "enrichment_fold": float(enrichment),
        })

    if not results:
        print("  No RBPs with both scPTR and eCLIP data found")
        return None

    results_df = pd.DataFrame(results)
    results_df.to_csv(res_dir / f"eclip_validation_{name}.csv", index=False)

    # Summary
    n_sig = (results_df["p_value"] < 0.05).sum()
    print(f"\n  Summary: {n_sig}/{len(results_df)} RBPs have significant eCLIP overlap (p<0.05)")
    print(f"  Mean enrichment fold: {results_df['enrichment_fold'].mean():.2f}x")
    print(f"  Mean odds ratio: {results_df['odds_ratio'].mean():.2f}")

    # Figure: enrichment barplot
    if len(results_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Enrichment fold
        rbps = results_df["rbp"].values
        enrichments = results_df["enrichment_fold"].values
        pvals = results_df["p_value"].values
        colors = ["steelblue" if p < 0.05 else "lightgray" for p in pvals]

        bars = axes[0].bar(range(len(rbps)), enrichments, color=colors, edgecolor="black",
                           linewidth=0.5)
        axes[0].axhline(y=1, color="red", linestyle="--", alpha=0.5, label="Expected (random)")
        axes[0].set_xticks(range(len(rbps)))
        axes[0].set_xticklabels(rbps, rotation=45, ha="right", fontsize=9)
        axes[0].set_ylabel("Enrichment fold (observed/expected)")
        axes[0].set_title(f"eCLIP Validation: Target Enrichment ({name})")
        axes[0].legend()
        for i, (e, p) in enumerate(zip(enrichments, pvals)):
            sig = "*" if p < 0.05 else ""
            axes[0].text(i, e + 0.05, f"{e:.1f}x{sig}", ha="center", fontsize=8)

        # Overlap counts
        overlap_data = np.array([
            results_df["n_overlap"].values,
            results_df["n_predicted"].values - results_df["n_overlap"].values,
        ])
        axes[1].bar(range(len(rbps)), results_df["n_overlap"].values,
                    color="steelblue", label="eCLIP confirmed", edgecolor="black", linewidth=0.5)
        axes[1].bar(range(len(rbps)),
                    results_df["n_predicted"].values - results_df["n_overlap"].values,
                    bottom=results_df["n_overlap"].values,
                    color="lightgray", label="Not confirmed", edgecolor="black", linewidth=0.5)
        axes[1].set_xticks(range(len(rbps)))
        axes[1].set_xticklabels(rbps, rotation=45, ha="right", fontsize=9)
        axes[1].set_ylabel("Number of predicted targets")
        axes[1].set_title(f"Predicted Target Overlap with eCLIP ({name})")
        axes[1].legend()

        fig.tight_layout()
        save_fig(fig, f"eclip_validation_{name}")

    return results_df


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

    # Also load sci-fate
    sys.path.insert(0, str(Path(__file__).parent))
    from run_scifate import load_scifate_data, prepare_for_scptr
    adata_sf_raw = load_scifate_data()
    adata_sf = prepare_for_scptr(adata_sf_raw)
    adata_sf = run_pipeline(adata_sf, "scifate")

    datasets = {
        "pancreas": (adata_pan, "mouse"),
        "dentate_gyrus": (adata_dg, "mouse"),
        "scifate": (adata_sf, "human"),
    }

    # T2-4: Sequence feature validation
    print("\n" + "=" * 60)
    print("T2-4: SEQUENCE FEATURE VALIDATION")
    print("=" * 60)

    seq_results = {}
    for name, (adata, species) in datasets.items():
        res = sequence_feature_validation(adata, name, species)
        if res:
            seq_results[name] = res

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "sequence_features.json", "w") as f:
        json.dump(seq_results, f, indent=2)

    # T2-5: eCLIP validation (only for datasets with significant networks)
    print("\n" + "=" * 60)
    print("T2-5: eCLIP VALIDATION")
    print("=" * 60)

    for name, (adata, species) in datasets.items():
        eclip_validation(adata, name)

    print(f"\n{'='*60}")
    print("ALL TIER 2 VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
