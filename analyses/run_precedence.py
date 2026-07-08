#!/usr/bin/env python
"""Demonstrate that PT velocity precedes RNA velocity at cell fate transitions.

Central hypothesis: Post-transcriptional regulation (gamma changes) acts as an
early signal that precedes and potentially drives transcriptional changes during
cell fate transitions.

Strategy:
1. Order cells along pseudotime (diffusion pseudotime via scanpy)
2. Smooth gamma and expression along pseudotime
3. For transition-associated genes, detect when gamma change and expression
   change begin — gamma onset should precede expression onset
4. Cross-correlation analysis: gamma(t) should predict expression(t+delta)
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
from scipy import stats, signal, ndimage

sys.path.insert(0, str(Path(__file__).parent))
from _common import set_figure_style

import scptr

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "precedence"


def save_fig(fig, name, subdir="figures"):
    if fig is None:
        return
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def compute_pseudotime(adata, root_cluster):
    """Compute diffusion pseudotime from a root cluster."""
    # Use diffusion pseudotime via scanpy
    sc.tl.diffmap(adata)

    # Find root cell: centroid of root cluster in diffusion space
    root_mask = adata.obs["clusters"] == root_cluster
    root_cells = np.where(root_mask)[0]
    if len(root_cells) == 0:
        raise ValueError(f"No cells in cluster {root_cluster}")

    # Pick cell closest to cluster centroid in diffmap
    dm = adata.obsm["X_diffmap"]
    centroid = dm[root_cells].mean(axis=0)
    dists = np.linalg.norm(dm[root_cells] - centroid, axis=1)
    root_idx = root_cells[np.argmin(dists)]

    adata.uns["iroot"] = root_idx
    sc.tl.dpt(adata)

    pt = adata.obs["dpt_pseudotime"].values.copy()
    # Handle infinite values
    pt[~np.isfinite(pt)] = np.nanmax(pt[np.isfinite(pt)])
    return pt


def smooth_along_pseudotime(values, pseudotime, n_bins=100):
    """Bin and smooth values along pseudotime axis.

    Returns bin centers and smoothed values (per gene if 2D).
    """
    bins = np.linspace(0, np.max(pseudotime), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_idx = np.digitize(pseudotime, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    if values.ndim == 1:
        smoothed = np.zeros(n_bins)
        for i in range(n_bins):
            mask = bin_idx == i
            if mask.sum() > 0:
                smoothed[i] = np.mean(values[mask])
        # Gaussian smoothing
        smoothed = ndimage.gaussian_filter1d(smoothed, sigma=2)
        return bin_centers, smoothed

    # 2D: genes x bins
    n_genes = values.shape[1]
    smoothed = np.zeros((n_bins, n_genes))
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() > 0:
            smoothed[i] = np.mean(values[mask], axis=0)
    # Smooth each gene
    for g in range(n_genes):
        smoothed[:, g] = ndimage.gaussian_filter1d(smoothed[:, g], sigma=2)
    return bin_centers, smoothed


def detect_onset(trace, threshold_frac=0.1):
    """Detect onset of change: first index where signal exceeds
    threshold_frac * (max - baseline)."""
    baseline = np.mean(trace[:5])  # first 5 bins as baseline
    peak = np.max(np.abs(trace - baseline))
    threshold = baseline + threshold_frac * peak

    for i, val in enumerate(trace):
        if abs(val - baseline) > threshold_frac * peak:
            return i
    return len(trace) - 1


def cross_correlate_lag(gamma_trace, expr_trace, max_lag=20):
    """Compute cross-correlation to find temporal lag.

    Positive lag = gamma leads expression.
    Returns optimal lag and correlation at that lag.
    """
    # Normalize
    g = (gamma_trace - np.mean(gamma_trace))
    g_std = np.std(g)
    if g_std > 0:
        g = g / g_std
    e = (expr_trace - np.mean(expr_trace))
    e_std = np.std(e)
    if e_std > 0:
        e = e / e_std

    n = len(g)
    best_lag = 0
    best_corr = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            corr = np.corrcoef(g[:n-lag], e[lag:])[0, 1] if n - lag > 5 else 0
        else:
            corr = np.corrcoef(g[-lag:], e[:n+lag])[0, 1] if n + lag > 5 else 0

        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag

    return best_lag, best_corr


def run_precedence_analysis(adata, dataset_name, root_cluster, n_bins=100):
    """Run temporal precedence analysis on one dataset."""
    print(f"\n{'='*60}")
    print(f"PRECEDENCE ANALYSIS: {dataset_name}")
    print(f"{'='*60}")

    res_dir = OUTPUT_DIR / "results" / dataset_name
    res_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Compute pseudotime
    print("\n--- Computing pseudotime ---")
    pt = compute_pseudotime(adata, root_cluster)
    print(f"  Root cluster: {root_cluster}")
    print(f"  Pseudotime range: [{pt.min():.4f}, {pt.max():.4f}]")

    # Step 2: Get gamma and expression matrices
    gamma = adata.layers["gamma"]
    if hasattr(adata.X, 'toarray'):
        expr = adata.X.toarray()
    else:
        expr = np.asarray(adata.X)
    expr = np.log1p(expr)  # log-normalize for comparison

    # Step 3: Smooth both along pseudotime
    print("\n--- Smoothing along pseudotime ---")
    bin_centers, gamma_smooth = smooth_along_pseudotime(gamma, pt, n_bins)
    _, expr_smooth = smooth_along_pseudotime(expr, pt, n_bins)

    # Step 4: Identify transition genes (high variance along pseudotime)
    gamma_var = np.var(gamma_smooth, axis=0)
    expr_var = np.var(expr_smooth, axis=0)

    # Require both gamma and expression to vary along pseudotime
    gamma_var_thresh = np.percentile(gamma_var[gamma_var > 0], 75)
    expr_var_thresh = np.percentile(expr_var[expr_var > 0], 75)
    transition_mask = (gamma_var > gamma_var_thresh) & (expr_var > expr_var_thresh)
    transition_genes = adata.var_names[transition_mask]
    print(f"  Transition genes: {len(transition_genes)}")

    # Step 5: Onset detection
    print("\n--- Onset detection ---")
    onset_results = []
    for i, gene in enumerate(adata.var_names):
        if not transition_mask[i]:
            continue
        g_trace = gamma_smooth[:, i]
        e_trace = expr_smooth[:, i]

        g_onset = detect_onset(g_trace)
        e_onset = detect_onset(e_trace)
        lead_bins = e_onset - g_onset  # positive = gamma leads

        onset_results.append({
            "gene": gene,
            "gamma_onset_bin": g_onset,
            "expr_onset_bin": e_onset,
            "lead_bins": lead_bins,
        })

    onset_df = pd.DataFrame(onset_results)
    n_gamma_leads = (onset_df["lead_bins"] > 0).sum()
    n_expr_leads = (onset_df["lead_bins"] < 0).sum()
    n_simultaneous = (onset_df["lead_bins"] == 0).sum()
    print(f"  Gamma leads: {n_gamma_leads}/{len(onset_df)} genes")
    print(f"  Expression leads: {n_expr_leads}/{len(onset_df)} genes")
    print(f"  Simultaneous: {n_simultaneous}/{len(onset_df)} genes")
    print(f"  Mean lead (bins): {onset_df['lead_bins'].mean():.2f}")

    # Binomial test: is gamma-leading significantly more common than chance?
    n_nontied = n_gamma_leads + n_expr_leads
    if n_nontied > 0:
        binom_p = stats.binomtest(n_gamma_leads, n_nontied, 0.5).pvalue
        print(f"  Binomial test (gamma leads more): p = {binom_p:.4e}")
    else:
        binom_p = 1.0

    onset_df.to_csv(res_dir / "onset_detection.csv", index=False)

    # Step 6: Cross-correlation analysis
    print("\n--- Cross-correlation analysis ---")
    lag_results = []
    for i, gene in enumerate(adata.var_names):
        if not transition_mask[i]:
            continue
        g_trace = gamma_smooth[:, i]
        e_trace = expr_smooth[:, i]

        lag, corr = cross_correlate_lag(g_trace, e_trace, max_lag=15)
        lag_results.append({
            "gene": gene,
            "optimal_lag": lag,
            "cross_corr": corr,
        })

    lag_df = pd.DataFrame(lag_results)
    mean_lag = lag_df["optimal_lag"].mean()
    median_lag = lag_df["optimal_lag"].median()
    n_positive_lag = (lag_df["optimal_lag"] > 0).sum()
    print(f"  Mean optimal lag: {mean_lag:.2f} bins (positive = gamma leads)")
    print(f"  Median optimal lag: {median_lag:.1f} bins")
    print(f"  Genes with positive lag: {n_positive_lag}/{len(lag_df)}")

    lag_df.to_csv(res_dir / "cross_correlation.csv", index=False)

    # Step 7: Combine results
    results = {
        "n_transition_genes": len(transition_genes),
        "onset_gamma_leads": int(n_gamma_leads),
        "onset_expr_leads": int(n_expr_leads),
        "onset_simultaneous": int(n_simultaneous),
        "onset_mean_lead_bins": float(onset_df["lead_bins"].mean()),
        "onset_binomial_p": float(binom_p),
        "crosscorr_mean_lag": float(mean_lag),
        "crosscorr_median_lag": float(median_lag),
        "crosscorr_positive_lag_frac": float(n_positive_lag / len(lag_df)),
    }
    with open(res_dir / "precedence_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # =========================================================================
    # FIGURES
    # =========================================================================

    # Figure 1: Onset histogram
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(onset_df["lead_bins"], bins=30, color="steelblue",
                alpha=0.8, edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.5, label="Simultaneous")
    axes[0].axvline(onset_df["lead_bins"].mean(), color="darkred",
                   linestyle="-", lw=2,
                   label=f"Mean={onset_df['lead_bins'].mean():.1f}")
    axes[0].set_xlabel("Lead (bins): positive = gamma leads expression")
    axes[0].set_ylabel("Number of genes")
    axes[0].set_title(f"Onset detection ({n_gamma_leads}/{len(onset_df)} gamma-leading)")
    axes[0].legend()

    # Cross-correlation lag histogram
    axes[1].hist(lag_df["optimal_lag"], bins=30, color="darkorange",
                alpha=0.8, edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", alpha=0.5, label="No lag")
    axes[1].axvline(mean_lag, color="darkred", linestyle="-", lw=2,
                   label=f"Mean={mean_lag:.1f}")
    axes[1].set_xlabel("Optimal lag (bins): positive = gamma leads")
    axes[1].set_ylabel("Number of genes")
    axes[1].set_title(f"Cross-correlation lag ({n_positive_lag}/{len(lag_df)} positive)")
    axes[1].legend()

    fig.suptitle(f"PT Velocity Precedes RNA Velocity: {dataset_name}",
                fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, f"precedence_{dataset_name}")

    # Figure 2: Example gene traces
    # Pick top 6 genes with largest gamma-leading onset
    top_genes = onset_df.nlargest(6, "lead_bins")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for idx, (_, row) in enumerate(top_genes.iterrows()):
        if idx >= 6:
            break
        gene = row["gene"]
        gi = list(adata.var_names).index(gene)
        g_trace = gamma_smooth[:, gi]
        e_trace = expr_smooth[:, gi]

        # Normalize for comparison
        g_norm = (g_trace - g_trace.min()) / (g_trace.max() - g_trace.min() + 1e-10)
        e_norm = (e_trace - e_trace.min()) / (e_trace.max() - e_trace.min() + 1e-10)

        ax = axes[idx]
        ax.plot(bin_centers, g_norm, "b-", lw=2, label="Gamma (norm)")
        ax.plot(bin_centers, e_norm, "r-", lw=2, label="Expression (norm)")
        ax.axvline(bin_centers[int(row["gamma_onset_bin"])], color="blue",
                  linestyle=":", alpha=0.5)
        ax.axvline(bin_centers[int(row["expr_onset_bin"])], color="red",
                  linestyle=":", alpha=0.5)
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel("Normalized value")
        ax.set_title(f"{gene} (lead={int(row['lead_bins'])} bins)")
        ax.legend(fontsize=7)

    fig.suptitle(f"Top Gamma-Leading Genes: {dataset_name}", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, f"example_genes_{dataset_name}")

    return results


def main():
    set_figure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PANCREAS: Ductal → Beta cell lineage
    # =========================================================================
    print("=" * 60)
    print("LOADING AND PROCESSING PANCREAS")
    print("=" * 60)
    adata_pan = scptr.datasets.pancreas()
    scptr.pp.filter_genes(adata_pan)
    scptr.pp.normalize_layers(adata_pan)
    scptr.pp.neighbors(adata_pan, n_neighbors=30)
    scptr.pp.smooth_layers(adata_pan)
    scptr.tl.estimate_beta(adata_pan)
    scptr.tl.estimate_gamma(adata_pan)
    scptr.tl.variance_decomposition(adata_pan)
    scptr.tl.pt_states(adata_pan)
    scptr.tl.pt_velocity(adata_pan)
    print(f"  Pipeline complete: {adata_pan.shape}")

    # Root cluster for pseudotime: Ductal (progenitor)
    print(f"  Clusters: {adata_pan.obs['clusters'].unique().tolist()}")
    pan_results = run_precedence_analysis(
        adata_pan, "pancreas", root_cluster="Ductal"
    )

    # =========================================================================
    # DENTATE GYRUS: Radial glia → Granule neuron lineage
    # =========================================================================
    print("\n" + "=" * 60)
    print("LOADING AND PROCESSING DENTATE GYRUS")
    print("=" * 60)
    adata_dg = scptr.datasets.dentate_gyrus()
    scptr.pp.filter_genes(adata_dg)
    scptr.pp.normalize_layers(adata_dg)
    scptr.pp.neighbors(adata_dg, n_neighbors=30)
    scptr.pp.smooth_layers(adata_dg)
    scptr.tl.estimate_beta(adata_dg)
    scptr.tl.estimate_gamma(adata_dg)
    scptr.tl.variance_decomposition(adata_dg)
    scptr.tl.pt_states(adata_dg)
    scptr.tl.pt_velocity(adata_dg)
    print(f"  Pipeline complete: {adata_dg.shape}")

    print(f"  Clusters: {adata_dg.obs['clusters'].unique().tolist()}")
    dg_results = run_precedence_analysis(
        adata_dg, "dentate_gyrus", root_cluster="Radial Glia-like"
    )

    # =========================================================================
    # combined summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("COMBINED SUMMARY")
    print("=" * 60)

    for name, results in [("pancreas", pan_results), ("dentate_gyrus", dg_results)]:
        print(f"\n  {name}:")
        print(f"    Transition genes: {results['n_transition_genes']}")
        print(f"    Gamma leads: {results['onset_gamma_leads']}, "
              f"Expr leads: {results['onset_expr_leads']}")
        print(f"    Mean onset lead: {results['onset_mean_lead_bins']:.2f} bins")
        print(f"    Binomial p: {results['onset_binomial_p']:.4e}")
        print(f"    Cross-corr mean lag: {results['crosscorr_mean_lag']:.2f} bins")

    # Summary figure: comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    datasets = ["pancreas", "dentate_gyrus"]
    all_results = [pan_results, dg_results]

    # Left: onset detection
    leads = [r["onset_gamma_leads"] for r in all_results]
    follows = [r["onset_expr_leads"] for r in all_results]
    simult = [r["onset_simultaneous"] for r in all_results]
    x = np.arange(len(datasets))
    width = 0.25
    axes[0].bar(x - width, leads, width, label="Gamma leads", color="steelblue")
    axes[0].bar(x, simult, width, label="Simultaneous", color="gray")
    axes[0].bar(x + width, follows, width, label="Expression leads", color="salmon")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].set_ylabel("Number of genes")
    axes[0].set_title("Onset Detection: Which Changes First?")
    axes[0].legend()

    # Right: mean lag
    mean_lags = [r["crosscorr_mean_lag"] for r in all_results]
    colors = ["steelblue" if l > 0 else "salmon" for l in mean_lags]
    axes[1].bar(datasets, mean_lags, color=colors)
    axes[1].set_ylabel("Mean optimal lag (bins)")
    axes[1].set_title("Cross-Correlation: Positive = Gamma Leads")
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.3)

    fig.suptitle("Post-Transcriptional Changes Precede Transcriptional Changes",
                fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "combined_precedence")

    # Save combined results
    combined = {"pancreas": pan_results, "dentate_gyrus": dg_results}
    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "combined_precedence.json", "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\nAll results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
