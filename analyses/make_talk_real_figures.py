#!/usr/bin/env python
"""Generate real-data figures for the scPTR talk.

The goal of this script is to avoid ambiguous or duplicated slide figures. It
recomputes the pancreas and dentate gyrus analyses from real spliced/unspliced
AnnData inputs, then writes a clean talk-focused figure folder:

    figures/talk_real_data/

The highest-priority outputs are:
  - matched expression-space vs gamma-space UMAPs
  - subset-level UMAP comparisons using the same gamma-substate labels
  - silhouette comparisons for every sufficiently large annotated cell type

No synthetic data are used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scptr
from anndata import AnnData
from scipy import stats
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "figures" / "talk_real_data"
RES_DIR = ROOT / "figures" / "talk_real_data_results"
SEED = 42


@dataclass
class DatasetResult:
    name: str
    adata: AnnData
    subset_stats: pd.DataFrame


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def save_table(df: pd.DataFrame, name: str) -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    path = RES_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"saved {path}")


def dense(x) -> np.ndarray:
    if issparse(x):
        return np.asarray(x.todense())
    return np.asarray(x)


def process_dataset(name: str, loader) -> AnnData:
    print(f"\n=== Processing {name} ===")
    adata = loader()
    print(f"input: {adata.n_obs} cells, {adata.n_vars} genes")

    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)

    # Expression-space graph and UMAP.
    scptr.pp.neighbors(adata, n_neighbors=30, random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)

    # scPTR estimator.
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)
    scptr.tl.variance_decomposition(adata)
    scptr.tl.pt_states(adata, random_state=SEED)
    scptr.tl.pt_velocity(adata)

    print(
        f"processed: {adata.n_obs} cells, {adata.n_vars} genes, "
        f"{adata.obs['pt_state'].nunique()} PT states"
    )
    return adata


def color_map(labels: pd.Series):
    cats = sorted(labels.astype(str).unique())
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(len(cats), 1))
    return cats, {cat: cmap(i) for i, cat in enumerate(cats)}


def plot_embedding(
    ax: plt.Axes,
    coords: np.ndarray,
    labels,
    title: str,
    palette: dict[str, tuple] | None = None,
    point_size: float = 5,
    legend: bool = False,
    legend_title: str | None = None,
) -> None:
    labels = pd.Series(labels).astype(str)
    cats = sorted(labels.unique())
    if palette is None:
        _, palette = color_map(labels)
    for cat in cats:
        mask = labels.values == cat
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=0.72,
            color=palette.get(cat),
            label=cat,
            linewidths=0,
            rasterized=True,
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    if legend:
        ax.legend(
            title=legend_title,
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            frameon=False,
            markerscale=2.2,
        )


def global_umap_figure(adata: AnnData, dataset_label: str, file_prefix: str) -> None:
    cell_labels = adata.obs["clusters"].astype(str)
    pt_labels = adata.obs["pt_state"].astype(str)
    _, cell_palette = color_map(cell_labels)
    _, pt_palette = color_map(pt_labels)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plot_embedding(
        axes[0, 0],
        adata.obsm["X_umap"],
        cell_labels,
        f"{dataset_label}: expression UMAP, cell types",
        cell_palette,
        legend=True,
        legend_title="Cell type",
    )
    plot_embedding(
        axes[0, 1],
        adata.obsm["X_gamma_umap"],
        cell_labels,
        f"{dataset_label}: gamma UMAP, same cell-type labels",
        cell_palette,
        legend=True,
        legend_title="Cell type",
    )
    plot_embedding(
        axes[1, 0],
        adata.obsm["X_umap"],
        pt_labels,
        f"{dataset_label}: expression UMAP, PT-state labels",
        pt_palette,
        legend=True,
        legend_title="PT state",
    )
    plot_embedding(
        axes[1, 1],
        adata.obsm["X_gamma_umap"],
        pt_labels,
        f"{dataset_label}: gamma UMAP, PT-state labels",
        pt_palette,
        legend=True,
        legend_title="PT state",
    )
    fig.suptitle(
        f"{dataset_label}: matched global UMAP comparisons\n"
        "Rows change labels; columns change feature space. UMAPs are recomputed in their own spaces.",
        y=1.02,
        fontsize=13,
    )
    fig.tight_layout()
    save_fig(fig, f"{file_prefix}_global_umap_expression_vs_gamma_matched")


def make_umap_from_features(features: np.ndarray, n_neighbors: int = 15) -> np.ndarray:
    tmp = AnnData(X=features.astype(np.float32))
    nn = min(n_neighbors, max(2, tmp.n_obs - 1))
    sc.pp.neighbors(tmp, n_neighbors=nn, use_rep="X", random_state=SEED)
    sc.tl.umap(tmp, random_state=SEED)
    return tmp.obsm["X_umap"]


def subset_analysis(adata: AnnData, dataset_name: str, min_cells: int = 50) -> pd.DataFrame:
    rows: list[dict] = []
    clusters = adata.obs["clusters"].astype(str)
    expr_global = dense(adata.X).astype(np.float32)
    gamma_global = np.log1p(dense(adata.layers["gamma"]).astype(np.float32))

    for cluster in sorted(clusters.unique()):
        mask = clusters.values == cluster
        n_cells = int(mask.sum())
        if n_cells < min_cells:
            continue

        expr_sub = expr_global[mask]
        gamma_sub = gamma_global[mask]
        n_pcs = min(15, n_cells - 1, expr_sub.shape[1] - 1, gamma_sub.shape[1] - 1)
        if n_pcs < 2:
            continue

        expr_pcs = PCA(n_components=n_pcs, random_state=SEED).fit_transform(expr_sub)
        gamma_pcs = PCA(n_components=n_pcs, random_state=SEED).fit_transform(gamma_sub)

        best = None
        for k in (2, 3):
            if n_cells < k * 10:
                continue
            labels = KMeans(n_clusters=k, random_state=SEED, n_init=25).fit_predict(gamma_pcs)
            min_size = int(np.bincount(labels).min())
            if min_size < max(10, int(0.05 * n_cells)):
                continue
            sil_gamma = float(silhouette_score(gamma_pcs, labels))
            sil_expr = float(silhouette_score(expr_pcs, labels))
            candidate = {
                "dataset": dataset_name,
                "cluster": cluster,
                "n_cells": n_cells,
                "n_substates": k,
                "silhouette_gamma": sil_gamma,
                "silhouette_expression": sil_expr,
                "silhouette_gap": sil_gamma - sil_expr,
                "min_substate_size": min_size,
                "labels": labels,
                "expr_pcs": expr_pcs,
                "gamma_pcs": gamma_pcs,
            }
            if best is None or candidate["silhouette_gamma"] > best["silhouette_gamma"]:
                best = candidate

        if best is None:
            continue
        rows.append({k: v for k, v in best.items() if k not in {"labels", "expr_pcs", "gamma_pcs"}})

    df = pd.DataFrame(rows).sort_values("silhouette_gap", ascending=False)
    return df.reset_index(drop=True)


def subset_details(adata: AnnData, cluster: str, n_substates: int | None = None) -> dict:
    clusters = adata.obs["clusters"].astype(str)
    mask = clusters.values == cluster
    expr_sub = dense(adata.X)[mask].astype(np.float32)
    gamma_sub = np.log1p(dense(adata.layers["gamma"])[mask].astype(np.float32))
    n_cells = int(mask.sum())
    n_pcs = min(15, n_cells - 1, expr_sub.shape[1] - 1, gamma_sub.shape[1] - 1)
    expr_pcs = PCA(n_components=n_pcs, random_state=SEED).fit_transform(expr_sub)
    gamma_pcs = PCA(n_components=n_pcs, random_state=SEED).fit_transform(gamma_sub)

    best = None
    ks = [n_substates] if n_substates is not None else [2, 3]
    for k in ks:
        labels = KMeans(n_clusters=k, random_state=SEED, n_init=25).fit_predict(gamma_pcs)
        sil_gamma = float(silhouette_score(gamma_pcs, labels))
        sil_expr = float(silhouette_score(expr_pcs, labels))
        candidate = (sil_gamma, sil_expr, labels)
        if best is None or sil_gamma > best[0]:
            best = candidate
    assert best is not None
    sil_gamma, sil_expr, labels = best
    return {
        "mask": mask,
        "labels": labels,
        "expr_pcs": expr_pcs,
        "gamma_pcs": gamma_pcs,
        "expr_umap": make_umap_from_features(expr_pcs),
        "gamma_umap": make_umap_from_features(gamma_pcs),
        "silhouette_gamma": sil_gamma,
        "silhouette_expression": sil_expr,
    }


def subset_umap_figure(
    adata: AnnData,
    dataset_label: str,
    cluster: str,
    file_name: str,
    n_substates: int | None = None,
) -> None:
    details = subset_details(adata, cluster, n_substates=n_substates)
    labels = pd.Series(details["labels"]).map(lambda x: f"gamma substate {x}")
    _, palette = color_map(labels)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    plot_embedding(
        axes[0],
        details["expr_umap"],
        labels,
        f"{cluster}: expression-space UMAP\nsame gamma-substate labels",
        palette,
        point_size=20,
        legend=False,
    )
    plot_embedding(
        axes[1],
        details["gamma_umap"],
        labels,
        f"{cluster}: gamma-space UMAP\nsame gamma-substate labels",
        palette,
        point_size=20,
        legend=True,
        legend_title="Substate",
    )
    vals = [details["silhouette_expression"], details["silhouette_gamma"]]
    axes[2].bar(["Expression\nspace", "Gamma\nspace"], vals, color=["#E45756", "#4C78A8"])
    axes[2].axhline(0, color="0.4", lw=0.8)
    axes[2].set_ylabel("Silhouette of gamma-substate labels")
    axes[2].set_title("Same labels, different feature spaces")
    for i, v in enumerate(vals):
        axes[2].text(i, v + (0.015 if v >= 0 else -0.035), f"{v:.3f}", ha="center")
    fig.suptitle(
        f"{dataset_label} {cluster}: expression-invisible PT substructure",
        y=1.02,
        fontsize=13,
    )
    fig.tight_layout()
    save_fig(fig, file_name)


def silhouette_overview(df: pd.DataFrame, dataset_label: str, file_name: str) -> None:
    plot_df = df.sort_values("silhouette_gap", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.35 * len(plot_df) + 1.5)))

    y = np.arange(len(plot_df))
    axes[0].barh(y - 0.18, plot_df["silhouette_expression"], height=0.36, color="#E45756", label="Expression")
    axes[0].barh(y + 0.18, plot_df["silhouette_gamma"], height=0.36, color="#4C78A8", label="Gamma")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot_df["cluster"])
    axes[0].axvline(0, color="0.4", lw=0.8)
    axes[0].set_xlabel("Silhouette score")
    axes[0].set_title("Same gamma-substate labels in both spaces")
    axes[0].legend(frameon=False)

    colors = np.where(plot_df["silhouette_gap"] > 0.05, "#4C78A8", "0.65")
    axes[1].barh(y, plot_df["silhouette_gap"], color=colors)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(plot_df["cluster"])
    axes[1].axvline(0, color="0.4", lw=0.8)
    axes[1].set_xlabel("Gamma silhouette minus expression silhouette")
    axes[1].set_title("Expression-invisibility score")
    fig.suptitle(f"{dataset_label}: subset PT-state separability", y=1.02, fontsize=13)
    fig.tight_layout()
    save_fig(fig, file_name)


def estimator_qc_figure(adata: AnnData) -> None:
    gene = "Malat1" if "Malat1" in adata.var_names else str(adata.var_names[0])
    gi = list(adata.var_names).index(gene)
    s = dense(adata.layers["Ms"])[:, gi]
    u = dense(adata.layers["Mu"])[:, gi]
    gamma = dense(adata.layers["gamma"])[:, gi]
    beta = float(adata.var.loc[gene, "beta"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sca = axes[0].scatter(s, u, c=np.log1p(gamma), s=4, alpha=0.55, cmap="magma", rasterized=True)
    xmax = np.percentile(s, 99.5)
    xs = np.linspace(0, xmax, 100)
    axes[0].plot(xs, beta * xs, "k--", lw=1.2, label=f"beta={beta:.2f}")
    axes[0].set_xlim(0, xmax)
    axes[0].set_ylim(0, np.percentile(u, 99.5))
    axes[0].set_xlabel("Smoothed spliced count")
    axes[0].set_ylabel("Smoothed unspliced count")
    axes[0].set_title(f"Phase portrait: {gene}")
    axes[0].legend(frameon=False)
    plt.colorbar(sca, ax=axes[0], label="log1p(gamma)")

    beta_vals = adata.var["beta"].values
    axes[1].hist(beta_vals[beta_vals > 0], bins=60, color="#4C78A8")
    axes[1].set_xlabel("Estimated beta")
    axes[1].set_ylabel("Genes")
    axes[1].set_title("Splicing-rate estimates")

    gamma_med = np.median(dense(adata.layers["gamma"]), axis=0)
    positive = gamma_med[gamma_med > 0]
    axes[2].hist(np.log10(positive + 1e-6), bins=60, color="#F58518")
    axes[2].set_xlabel("log10 median gamma")
    axes[2].set_ylabel("Genes")
    axes[2].set_title("Per-gene degradation-rate distribution")
    fig.suptitle("Pancreas estimator QC from real spliced/unspliced counts", y=1.02, fontsize=13)
    fig.tight_layout()
    save_fig(fig, "slide_08_pancreas_estimator_qc")


def halflife_figure(results: list[DatasetResult]) -> None:
    rows = []
    refs = [
        ("Herzog2017 mouse", scptr.datasets.herzog2017_halflives()),
        ("Schofield2018 human", scptr.datasets.schofield2018_halflives()),
    ]
    for res in results:
        for ref_name, ref in refs:
            corr = scptr.benchmark.correlate_with_halflives(res.adata, ref)
            rows.append({
                "dataset": res.name,
                "reference": ref_name,
                "spearman_r": corr["spearman_r"],
                "n_genes": corr["n_genes"],
            })
    df = pd.DataFrame(rows)
    save_table(df, "halflife_correlations")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    pivot = df.pivot(index="dataset", columns="reference", values="spearman_r")
    x = np.arange(len(pivot))
    width = 0.36
    refs_order = list(pivot.columns)
    colors = ["#4C78A8", "#F58518"]
    for i, ref in enumerate(refs_order):
        vals = pivot[ref].values
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=ref, color=colors[i])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val - 0.025, f"{val:.3f}", ha="center", va="top", fontsize=8)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Spearman correlation with half-life")
    ax.set_title("Real-data half-life validation: high gamma means shorter half-life")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, "slide_11_pancreas_dentate_halflife_validation")


def variance_figure(results: list[DatasetResult]) -> None:
    rows = []
    for res in results:
        rows.append({
            "dataset": res.name,
            "median_tf_score": float(np.median(res.adata.var["tf_score"].values)),
            "median_ptf_score": float(np.median(res.adata.var["ptf_score"].values)),
            "ptf_gt_half": int((res.adata.var["ptf_score"].values > 0.5).sum()),
            "n_genes": int(res.adata.n_vars),
        })
    df = pd.DataFrame(rows)
    save_table(df, "variance_decomposition_summary")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(df))
    axes[0].bar(x - 0.18, df["median_tf_score"], width=0.36, label="TF score", color="#E45756")
    axes[0].bar(x + 0.18, df["median_ptf_score"], width=0.36, label="PTF score", color="#4C78A8")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["dataset"])
    axes[0].set_ylabel("Median score")
    axes[0].set_title("Median variance decomposition")
    axes[0].legend(frameon=False)

    frac = df["ptf_gt_half"] / df["n_genes"]
    axes[1].bar(df["dataset"], frac, color="#4C78A8")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction of genes")
    axes[1].set_title("Genes with PTF score > 0.5")
    for i, val in enumerate(frac):
        axes[1].text(i, val + 0.02, f"{val:.0%}", ha="center")
    fig.suptitle("Most variation is post-transcriptional in these real datasets", y=1.02, fontsize=13)
    fig.tight_layout()
    save_fig(fig, "slide_13_variance_decomposition_pancreas_dentate")


def velocity_figures(results: list[DatasetResult]) -> None:
    for res in results:
        fig = scptr.pl.pt_velocity_embedding(res.adata, density=0.28, arrow_size=1.5, show=False)
        if fig is not None:
            fig.axes[0].set_title(f"{res.name}: post-transcriptional velocity on gamma UMAP")
            save_fig(fig, f"slide_17_{res.name}_pt_velocity")


def network_hub_figure(results: list[DatasetResult]) -> None:
    rows = []
    for res in results:
        known = scptr.tl.list_known_rbps(organism="mouse")
        rbps = [g for g in known if g in res.adata.var_names]
        if len(rbps) < 5:
            continue
        gamma = dense(res.adata.layers["gamma"])
        gamma_var = np.var(gamma, axis=0)
        top_idx = np.argsort(gamma_var)[::-1]
        targets = [res.adata.var_names[i] for i in top_idx if res.adata.var_names[i] not in rbps][:160]
        rbp_idx = [list(res.adata.var_names).index(g) for g in rbps]
        target_idx = [list(res.adata.var_names).index(g) for g in targets]
        expr = dense(res.adata.X)[:, rbp_idx]
        target_gamma = gamma[:, target_idx]
        edges = []
        for ri, rbp in enumerate(rbps):
            x = expr[:, ri]
            if np.std(x) < 1e-6:
                continue
            for ti, target in enumerate(targets):
                y = target_gamma[:, ti]
                if np.std(y) < 1e-6:
                    continue
                r, p = stats.spearmanr(x, y)
                if np.isfinite(r) and abs(r) > 0.2 and p < 0.01:
                    edges.append((rbp, target, r, p))
        edge_df = pd.DataFrame(edges, columns=["rbp", "target", "spearman_r", "p_value"])
        if edge_df.empty:
            continue
        counts = edge_df.groupby("rbp").size().sort_values(ascending=False).head(12)
        for rbp, count in counts.items():
            rows.append({"dataset": res.name, "rbp": rbp, "edges": int(count)})
        edge_df.to_csv(RES_DIR / f"{res.name}_network_edges.csv", index=False)

    hub_df = pd.DataFrame(rows)
    if hub_df.empty:
        return
    save_table(hub_df, "network_hub_counts")
    fig, axes = plt.subplots(1, len(results), figsize=(12, 5), squeeze=False)
    for ax, res in zip(axes[0], results):
        sub = hub_df[hub_df["dataset"] == res.name].sort_values("edges", ascending=True)
        ax.barh(sub["rbp"], sub["edges"], color="#4C78A8")
        ax.set_title(f"{res.name}: top RBP hubs")
        ax.set_xlabel("Significant target edges")
    fig.suptitle("RBP hub recovery from real gamma and expression matrices", y=1.02, fontsize=13)
    fig.tight_layout()
    save_fig(fig, "slide_18_pancreas_dentate_rbp_hubs")


def summary_figure(results: list[DatasetResult]) -> None:
    summary = []
    for res in results:
        summary.append({
            "dataset": res.name,
            "cells": res.adata.n_obs,
            "genes": res.adata.n_vars,
            "pt_states": res.adata.obs["pt_state"].nunique(),
            "eligible_subsets": len(res.subset_stats),
            "gamma_better_subsets": int((res.subset_stats["silhouette_gap"] > 0.05).sum()),
        })
    df = pd.DataFrame(summary)
    save_table(df, "talk_real_data_summary")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    axes[0].bar(df["dataset"], df["pt_states"], color="#4C78A8")
    axes[0].set_ylabel("PT states")
    axes[0].set_title("Gamma-space Leiden states")
    axes[1].bar(df["dataset"], df["genes"], color="#72B7B2")
    axes[1].set_ylabel("Filtered genes")
    axes[1].set_title("Real genes used")
    axes[2].bar(df["dataset"], df["gamma_better_subsets"], color="#F58518")
    axes[2].set_ylabel("Cell types")
    axes[2].set_title("Subsets where gamma separates better")
    fig.suptitle("Talk real-data figure set: pancreas and dentate gyrus", y=1.02, fontsize=13)
    fig.tight_layout()
    save_fig(fig, "slide_22_results_at_a_glance_pancreas_dentate")


def write_manifest(results: list[DatasetResult]) -> None:
    files = sorted(p.name for p in FIG_DIR.glob("*.png"))
    manifest = {
        "source": "Generated from real scPTR pancreas and dentate gyrus loaders.",
        "pdf_reference": "Local slide reference used during regeneration: ~/Downloads/scPTR talk.pdf",
        "output_dir": "figures/talk_real_data",
        "n_png": len(files),
        "files": files,
        "datasets": [
            {
                "name": res.name,
                "cells": int(res.adata.n_obs),
                "genes": int(res.adata.n_vars),
                "pt_states": int(res.adata.obs["pt_state"].nunique()),
            }
            for res in results
        ],
    }
    RES_DIR.mkdir(parents=True, exist_ok=True)
    (RES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    set_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    for stale in FIG_DIR.glob("*.png"):
        stale.unlink()
    for stale in RES_DIR.glob("*"):
        if stale.is_file():
            stale.unlink()

    pancreas = process_dataset("pancreas", scptr.datasets.pancreas)
    dentate = process_dataset("dentate_gyrus", scptr.datasets.dentate_gyrus)

    pancreas_stats = subset_analysis(pancreas, "pancreas")
    dentate_stats = subset_analysis(dentate, "dentate_gyrus")
    save_table(pancreas_stats, "pancreas_subset_silhouettes")
    save_table(dentate_stats, "dentate_gyrus_subset_silhouettes")

    results = [
        DatasetResult("pancreas", pancreas, pancreas_stats),
        DatasetResult("dentate_gyrus", dentate, dentate_stats),
    ]

    estimator_qc_figure(pancreas)
    halflife_figure(results)
    variance_figure(results)

    global_umap_figure(pancreas, "Pancreas", "slide_15_pancreas")
    global_umap_figure(dentate, "Dentate gyrus", "slide_15_dentate_gyrus")

    silhouette_overview(pancreas_stats, "Pancreas", "slide_16_pancreas_silhouette_overview")
    silhouette_overview(dentate_stats, "Dentate gyrus", "slide_16_dentate_gyrus_silhouette_overview")

    subset_umap_figure(pancreas, "Pancreas", "Epsilon", "slide_16_pancreas_epsilon_substate_umaps")
    subset_umap_figure(pancreas, "Pancreas", "Pre-endocrine", "slide_16_pancreas_pre_endocrine_substate_umaps")
    subset_umap_figure(dentate, "Dentate gyrus", "Endothelial", "slide_16_dentate_endothelial_substate_umaps")
    subset_umap_figure(dentate, "Dentate gyrus", "GABA", "slide_16_dentate_gaba_substate_umaps")
    subset_umap_figure(dentate, "Dentate gyrus", "Microglia", "slide_16_dentate_microglia_substate_umaps")
    subset_umap_figure(dentate, "Dentate gyrus", "Radial Glia-like", "slide_16_dentate_radial_glia_substate_umaps")

    velocity_figures(results)
    network_hub_figure(results)
    summary_figure(results)
    write_manifest(results)

    print(f"\nWrote {len(list(FIG_DIR.glob('*.png')))} figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
