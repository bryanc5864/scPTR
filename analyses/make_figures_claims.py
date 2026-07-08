#!/usr/bin/env python
"""Generate figures for specific scientific claims from existing computed data.

This script reads pre-computed JSON/CSV files from output/ and figures/results/
and produces publication-quality figures in figures/. It does NOT re-run the
scPTR pipeline.

Usage:
    python analyses/make_figures_claims.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "figures" / "results"
# The output/ directory lives in the main repo worktree (not in git sub-worktrees).
# Resolve by walking up from this file's location until we find an output/ dir.
_candidate = ROOT
while not (_candidate / "output").exists() and _candidate != _candidate.parent:
    _candidate = _candidate.parent
OUT_DIR = _candidate / "output"
# Fallback: use main repo path directly
if not OUT_DIR.exists():
    OUT_DIR = Path("/home/bcheng/scPTR/output")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 200,
        }
    )


def save_fig(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# A. PT velocity orthogonality
# ---------------------------------------------------------------------------

def fig_pt_velocity_orthogonality() -> None:
    """Bar chart: mean cosine similarity between PT velocity and RNA velocity."""
    pancreas_path = OUT_DIR / "gap_analysis" / "results" / "velocity_comparison" / "pancreas" / "velocity_comparison.json"
    dg_path = OUT_DIR / "gap_analysis" / "results" / "velocity_comparison" / "dentate_gyrus" / "velocity_comparison.json"

    with pancreas_path.open() as f:
        pancreas_data = json.load(f)
    with dg_path.open() as f:
        dg_data = json.load(f)

    datasets = ["Pancreas", "Dentate Gyrus"]
    cosines = [
        pancreas_data["mean_cosine_similarity"],
        dg_data["mean_cosine_similarity"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left panel: bar chart
    ax = axes[0]
    colors = ["#4C78A8", "#72B7B2"]
    bars = ax.bar(datasets, cosines, color=colors, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Mean cosine similarity", fontsize=13)
    ax.set_title("PT velocity vs RNA velocity cosine similarity", fontsize=13)
    ax.set_ylim(-0.25, 0.05)
    for bar, val in zip(bars, cosines):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val - 0.01,
            f"{val:.3f}",
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
    ax.text(
        0.5, -0.22,
        "Cosine ≈ 0 → orthogonal (independent information)",
        ha="center",
        fontsize=11,
        color="#555555",
        transform=ax.transData,
    )

    # Right panel: explanatory text box
    ax2 = axes[1]
    ax2.axis("off")
    explanation = (
        "Independent axes\n\n"
        "PT velocity captures\n"
        "different biology from\n"
        "RNA velocity.\n\n"
        "Cosine similarity ≈ 0\n"
        "means the two velocity\n"
        "vectors are orthogonal —\n"
        "neither is a mere\n"
        "rescaling of the other.\n\n"
        f"Pancreas: cos = {cosines[0]:.3f}\n"
        f"Dentate Gyrus: cos = {cosines[1]:.3f}\n\n"
        "RNA velocity ≈ dRNA/dt\n"
        "PT velocity ≈ dγ/dt"
    )
    ax2.text(
        0.5, 0.5,
        explanation,
        ha="center",
        va="center",
        fontsize=12,
        transform=ax2.transAxes,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#EEF4FF", edgecolor="#4C78A8", linewidth=1.5),
    )

    fig.suptitle("PT velocity ≈ orthogonal to RNA velocity", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_pt_velocity_orthogonality")


# ---------------------------------------------------------------------------
# B. Gamma leads expression (temporal precedence)
# ---------------------------------------------------------------------------

def fig_gamma_leads_expression() -> None:
    """Stacked bar showing gamma leads vs expression leads vs simultaneous."""
    path = OUT_DIR / "precedence" / "results" / "combined_precedence.json"
    with path.open() as f:
        data = json.load(f)

    datasets_info = {
        "Pancreas": data["pancreas"],
        "Dentate Gyrus": data["dentate_gyrus"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Left panel: stacked percentage bar
    ax = axes[0]
    labels = list(datasets_info.keys())
    x = np.arange(len(labels))
    width = 0.45

    gamma_pcts = []
    expr_pcts = []
    simul_pcts = []
    gamma_leads_abs = []
    n_totals = []

    for key in datasets_info:
        d = datasets_info[key]
        n = d["n_transition_genes"]
        g = d["onset_gamma_leads"]
        e = d["onset_expr_leads"]
        s = d["onset_simultaneous"]
        gamma_pcts.append(g / n * 100)
        expr_pcts.append(e / n * 100)
        simul_pcts.append(s / n * 100)
        gamma_leads_abs.append(g)
        n_totals.append(n)

    colors_gamma = "#4C78A8"
    colors_expr = "#E45756"
    colors_simul = "#AAAAAA"

    p1 = ax.bar(x, gamma_pcts, width, label="γ leads onset", color=colors_gamma)
    p2 = ax.bar(x, expr_pcts, width, bottom=gamma_pcts, label="Expression leads", color=colors_expr)
    bottoms_simul = [g + e for g, e in zip(gamma_pcts, expr_pcts)]
    p3 = ax.bar(x, simul_pcts, width, bottom=bottoms_simul, label="Simultaneous", color=colors_simul)

    # Annotate percentage that gamma leads
    for i, (pct, n) in enumerate(zip(gamma_pcts, n_totals)):
        ax.text(
            x[i],
            pct / 2,
            f"{pct:.0f}%\n(n={gamma_leads_abs[i]})",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\n(n={n})" for lbl, n in zip(labels, n_totals)], fontsize=12)
    ax.set_ylabel("% of transition genes", fontsize=13)
    ax.set_ylim(0, 105)
    ax.set_title("Onset timing: γ vs expression", fontsize=13)
    ax.legend(frameon=False, fontsize=11)

    # Right panel: p-value annotations
    ax2 = axes[1]
    ax2.axis("off")

    pval_pancreas = data["pancreas"]["onset_binomial_p"]
    pval_dg = data["dentate_gyrus"]["onset_binomial_p"]

    summary = (
        "Binomial test\n"
        "(H0: γ and expression lead equally often)\n\n"
        f"Pancreas:\n"
        f"  {data['pancreas']['onset_gamma_leads']}/{data['pancreas']['n_transition_genes']} genes: γ leads\n"
        f"  p = {pval_pancreas:.2e}\n\n"
        f"Dentate Gyrus:\n"
        f"  {data['dentate_gyrus']['onset_gamma_leads']}/{data['dentate_gyrus']['n_transition_genes']} genes: γ leads\n"
        f"  p = {pval_dg:.2e}\n\n"
        "Interpretation:\n"
        "Post-transcriptional regulation\n"
        "changes before mRNA expression\n"
        "at developmental transitions.\n\n"
        "γ changes first → RBPs prime\n"
        "fate before transcription."
    )
    ax2.text(
        0.5, 0.5,
        summary,
        ha="center",
        va="center",
        fontsize=12,
        transform=ax2.transAxes,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFF3E0", edgecolor="#F58518", linewidth=1.5),
    )

    fig.suptitle("γ (PT regulation) precedes expression onset at cell-fate transitions",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_gamma_leads_expression")


# ---------------------------------------------------------------------------
# C. Library-size correction
# ---------------------------------------------------------------------------

def fig_library_size_correction() -> None:
    """Before/after bar chart showing library-size correction effect."""
    path = OUT_DIR / "tier3" / "results" / "neuroblastoma_results.json"
    with path.open() as f:
        data = json.load(f)

    raw_destab = data["raw_destab_frac"]
    raw_stab = 1.0 - raw_destab
    corr_destab = data["corrected_destab_frac"]
    corr_stab = 1.0 - corr_destab
    n_raw = data["n_raw_edges"]
    n_corrected = data["n_network_edges"]
    removed = n_raw - n_corrected

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))

    categories = ["Destabilizing", "Stabilizing"]
    before_vals = [raw_destab * 100, raw_stab * 100]
    after_vals = [corr_destab * 100, corr_stab * 100]
    colors_dest = "#E45756"
    colors_stab = "#4C78A8"
    bar_colors = [colors_dest, colors_stab]

    # Left: Before correction
    ax1 = axes[0]
    bars1 = ax1.bar(categories, before_vals, color=bar_colors, width=0.5)
    ax1.set_title(f"Before correction\n(n={n_raw:,} edges)", fontsize=13)
    ax1.set_ylabel("% of edges", fontsize=13)
    ax1.set_ylim(0, 115)
    for bar, val in zip(bars1, before_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    # Right: After correction
    ax2 = axes[1]
    bars2 = ax2.bar(categories, after_vals, color=bar_colors, width=0.5)
    ax2.set_title(f"After correction\n(n={n_corrected:,} edges)", fontsize=13)
    ax2.set_ylabel("% of edges", fontsize=13)
    ax2.set_ylim(0, 115)
    for bar, val in zip(bars2, after_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    # Annotation
    ax2.text(
        0.5, 0.08,
        f"Removed {removed:,} library-size\nconfounded edges",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#555555",
        transform=ax2.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF0F0", edgecolor="#E45756", linewidth=1),
    )

    fig.suptitle("Library-size correction flips the neuroblastoma network bias",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_library_size_correction")


# ---------------------------------------------------------------------------
# D. DepMap essentials
# ---------------------------------------------------------------------------

def fig_depmap_essentials() -> None:
    """Bar chart: hub vs non-hub RBP DepMap dependency by dataset."""
    path = OUT_DIR / "tier3" / "results" / "depmap_validation.csv"
    df = pd.read_csv(path)

    datasets = df["dataset"].tolist()
    dataset_labels = [d.replace("_", " ").title() for d in datasets]

    fig, axes = plt.subplots(1, len(df), figsize=(13, 5.5), sharey=True)
    if len(df) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, df.iterrows()):
        label = row["dataset"].replace("_", " ").title()
        hub = row["hub_mean_dep"]
        nonhub = row["nonhub_mean_dep"]
        p = row["mannwhitney_p"]
        more_ess = row["hub_more_essential"]

        bar_colors = ["#E45756" if more_ess else "#AAAAAA", "#72B7B2"]
        bars = ax.bar(["Hub RBPs\n(top 20)", "Non-hub\nRBPs"], [hub, nonhub],
                      color=bar_colors, width=0.5)
        ax.set_title(label, fontsize=13)
        ax.set_ylabel("Mean DepMap score\n(more negative = more essential)", fontsize=11)

        # Annotate values
        for bar, val in zip(bars, [hub, nonhub]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val - 0.03,
                f"{val:.3f}",
                ha="center",
                va="top",
                fontsize=11,
                fontweight="bold",
                color="white",
            )

        # p-value annotation
        p_str = f"p = {p:.2e}" if p < 0.01 else f"p = {p:.3f}"
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "n.s.")
        ax.text(
            0.5, 0.97,
            f"{sig}\n{p_str}",
            ha="center",
            va="top",
            fontsize=11,
            transform=ax.transAxes,
            color="#333333",
        )

        # Reference line at 0
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")

    fig.suptitle("Hub RBPs are more essential in cancer (DepMap CRISPR)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_depmap_essentials")


# ---------------------------------------------------------------------------
# E. Neuroblastoma network
# ---------------------------------------------------------------------------

def fig_neuroblastoma_network() -> None:
    """Hub RBP bar chart + stabilizing/destabilizing breakdown."""
    results_path = OUT_DIR / "tier3" / "results" / "neuroblastoma_results.json"
    network_path = OUT_DIR / "tier3" / "results" / "neuroblastoma_network_corrected.csv"

    with results_path.open() as f:
        results = json.load(f)

    top_hubs = results["top_hubs"]
    corr_destab = results["corrected_destab_frac"]
    corr_stab = 1.0 - corr_destab
    n_corrected = results["n_network_edges"]

    # Read network to get per-hub direction breakdown
    network_df = pd.read_csv(network_path)
    hub_names = list(top_hubs.keys())

    # Count stabilizing vs destabilizing per hub
    hub_direction = {}
    for hub in hub_names:
        sub = network_df[network_df["rbp"] == hub]
        n_stab = (sub["direction"] == "stabilizing").sum()
        n_dest = (sub["direction"] == "destabilizing").sum()
        hub_direction[hub] = {"stabilizing": n_stab, "destabilizing": n_dest, "total": len(sub)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left panel: horizontal bar chart of top 10 hubs
    ax1 = axes[0]
    hubs_sorted = sorted(hub_names, key=lambda h: hub_direction[h]["total"])
    stab_counts = [hub_direction[h]["stabilizing"] for h in hubs_sorted]
    dest_counts = [hub_direction[h]["destabilizing"] for h in hubs_sorted]
    y = np.arange(len(hubs_sorted))

    ax1.barh(y, stab_counts, label="Stabilizing", color="#4C78A8", height=0.6)
    ax1.barh(y, dest_counts, left=stab_counts, label="Destabilizing", color="#E45756", height=0.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels(hubs_sorted, fontsize=11)
    ax1.set_xlabel("Edge count", fontsize=12)
    ax1.set_title("Top 10 hub RBPs\n(neuroblastoma, library-size corrected)", fontsize=12)
    ax1.legend(frameon=False, fontsize=11)

    # Annotate total
    for i, hub in enumerate(hubs_sorted):
        total = hub_direction[hub]["total"]
        ax1.text(
            total + 0.3, i,
            str(total),
            va="center",
            fontsize=10,
            color="#333333",
        )

    # Right panel: overall stabilizing vs destabilizing
    ax2 = axes[1]
    ax2.axis("off")

    # Draw a simple proportional bar
    bar_width = 0.6
    bar_height = 0.15
    stab_w = corr_stab * bar_width
    dest_w = corr_destab * bar_width
    bar_x = 0.2

    stab_patch = mpatches.FancyBboxPatch(
        (bar_x, 0.58), stab_w, bar_height,
        boxstyle="square,pad=0",
        facecolor="#4C78A8",
        transform=ax2.transAxes,
        zorder=3,
    )
    dest_patch = mpatches.FancyBboxPatch(
        (bar_x + stab_w, 0.58), dest_w, bar_height,
        boxstyle="square,pad=0",
        facecolor="#E45756",
        transform=ax2.transAxes,
        zorder=3,
    )
    ax2.add_patch(stab_patch)
    ax2.add_patch(dest_patch)

    ax2.text(bar_x + stab_w / 2, 0.58 + bar_height + 0.03,
             f"Stabilizing\n{corr_stab * 100:.1f}%",
             ha="center", va="bottom", fontsize=12, color="#4C78A8",
             transform=ax2.transAxes, fontweight="bold")
    ax2.text(bar_x + stab_w + dest_w / 2, 0.58 + bar_height + 0.03,
             f"Destabilizing\n{corr_destab * 100:.1f}%",
             ha="center", va="bottom", fontsize=12, color="#E45756",
             transform=ax2.transAxes, fontweight="bold")

    ax2.text(
        0.5, 0.45,
        f"Total corrected edges: n={n_corrected:,}",
        ha="center",
        va="top",
        fontsize=12,
        transform=ax2.transAxes,
        color="#333333",
    )
    ax2.text(
        0.5, 0.30,
        "Oncogenic regime:\nRBPs shift to stabilizing\nhigh-turnover targets\nin neuroblastoma",
        ha="center",
        va="top",
        fontsize=12,
        transform=ax2.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#EEF4FF", edgecolor="#4C78A8", linewidth=1.5),
    )

    fig.suptitle("Neuroblastoma RBP network: stabilizing-dominant after correction",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_neuroblastoma_network")


# ---------------------------------------------------------------------------
# F. Permutation control (silhouette gap as negative control)
# ---------------------------------------------------------------------------

def fig_permutation_control() -> None:
    """Silhouette comparison showing expression clustering fails to recover gamma structure."""
    pancreas_path = RES_DIR / "pancreas_subset_silhouettes.csv"
    dg_path = RES_DIR / "dentate_gyrus_subset_silhouettes.csv"

    pancreas_df = pd.read_csv(pancreas_path)
    dg_df = pd.read_csv(dg_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, df, title, dataset in [
        (axes[0], pancreas_df, "Pancreas", "pancreas"),
        (axes[1], dg_df, "Dentate Gyrus", "dentate_gyrus"),
    ]:
        df = df.sort_values("silhouette_gamma", ascending=True).reset_index(drop=True)
        y = np.arange(len(df))
        width = 0.35

        bars_gamma = ax.barh(y - width / 2, df["silhouette_gamma"], width,
                             color="#4C78A8", label="γ-space silhouette")
        bars_expr = ax.barh(y + width / 2, df["silhouette_expression"], width,
                            color="#E45756", label="Expression-space silhouette\n(negative control)")

        ax.set_yticks(y)
        ax.set_yticklabels(df["cluster"], fontsize=10)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Silhouette score", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(frameon=False, fontsize=10, loc="lower right")

        # Shade rows where gamma >> expression (gap > 0.25)
        for i, (_, row) in enumerate(df.iterrows()):
            if row["silhouette_gap"] > 0.25:
                ax.axhspan(i - 0.5, i + 0.5, alpha=0.08, color="#4C78A8", zorder=0)

    # Count cells where gamma > expression
    pan_gamma_better = (pancreas_df["silhouette_gap"] > 0).sum()
    pan_total = len(pancreas_df)
    dg_gamma_better = (dg_df["silhouette_gap"] > 0).sum()
    dg_total = len(dg_df)

    fig.suptitle(
        "Expression clustering does not recover γ-space structure (negative control)\n"
        f"{pan_gamma_better}/{pan_total} pancreas, {dg_gamma_better}/{dg_total} DG cell types: "
        "γ-space structure invisible to expression",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "fig_permutation_control")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating figures from existing computed data...")

    print("A. PT velocity orthogonality...")
    fig_pt_velocity_orthogonality()

    print("B. Gamma leads expression...")
    fig_gamma_leads_expression()

    print("C. Library-size correction...")
    fig_library_size_correction()

    print("D. DepMap essentials...")
    fig_depmap_essentials()

    print("E. Neuroblastoma network...")
    fig_neuroblastoma_network()

    print("F. Permutation control / silhouette negative control...")
    fig_permutation_control()

    print(f"\nDone. Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
