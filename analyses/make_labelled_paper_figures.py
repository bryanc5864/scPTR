#!/usr/bin/env python
"""Create a labelled paper-figure export folder.

This script does two things:
1. Copies already regenerated figures into one folder with descriptive names.
2. Generates individual expression-vs-gamma UMAP panels that the stock scripts
   save only as combined figures.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import scptr


ROOT = Path(__file__).parent.parent
OUT = ROOT / "output" / "paper_figures_labelled"


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def copy_if_exists(src: str, dest_name: str) -> None:
    src_path = ROOT / src
    if not src_path.exists():
        print(f"missing {src_path}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{dest_name}.png"
    shutil.copy2(src_path, dest)
    print(f"copied {dest}")


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
    })


def categorical_umap(
    coords: np.ndarray,
    labels,
    title: str,
    xlabel: str = "UMAP 1",
    ylabel: str = "UMAP 2",
    legend_title: str | None = None,
    point_size: float = 6,
) -> plt.Figure:
    labels = pd.Series(labels).astype(str)
    cats = sorted(labels.unique())
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(len(cats), 1))
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for i, cat in enumerate(cats):
        mask = labels.values == cat
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=0.72,
            color=cmap(i),
            label=cat,
            rasterized=True,
            linewidths=0,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(
        title=legend_title,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        markerscale=2,
    )
    fig.tight_layout()
    return fig


def process_pancreas():
    adata = scptr.datasets.pancreas()
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    sc.tl.umap(adata, random_state=42)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)
    scptr.tl.variance_decomposition(adata)
    scptr.tl.pt_states(adata, random_state=42)
    scptr.tl.pt_velocity(adata)
    return adata


def make_split_umaps(adata) -> None:
    save(
        categorical_umap(
            adata.obsm["X_umap"],
            adata.obs["clusters"],
            "Pancreas expression-space UMAP",
            legend_title="Cell type",
            point_size=5,
        ),
        "figure_1b_pancreas_expression_space_umap_cell_types",
    )
    save(
        categorical_umap(
            adata.obsm["X_gamma_umap"],
            adata.obs["pt_state"],
            "Pancreas gamma-space UMAP",
            legend_title="PT state",
            point_size=5,
        ),
        "figure_1b_pancreas_gamma_space_umap_pt_states",
    )


def make_epsilon_umaps(adata) -> None:
    mask = adata.obs["clusters"].astype(str).values == "Epsilon"
    gamma_sub = adata.layers["gamma"][mask]
    n_pcs = min(10, gamma_sub.shape[0] - 1, gamma_sub.shape[1] - 1)
    pcs = PCA(n_components=n_pcs, random_state=42).fit_transform(gamma_sub)
    labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(pcs)

    expr_coords = adata.obsm["X_umap"][mask]
    gamma_coords = adata.obsm["X_gamma_umap"][mask]
    expr_sil = silhouette_score(expr_coords, labels)
    gamma_sil = silhouette_score(gamma_coords, labels)

    save(
        categorical_umap(
            expr_coords,
            labels,
            f"Epsilon cells: expression-space UMAP (sil={expr_sil:.3f})",
            legend_title="Gamma substate",
            point_size=22,
        ),
        "figure_3a_epsilon_expression_space_umap_gamma_substates",
    )
    save(
        categorical_umap(
            gamma_coords,
            labels,
            f"Epsilon cells: gamma-space UMAP (sil={gamma_sil:.3f})",
            legend_title="Gamma substate",
            point_size=22,
        ),
        "figure_3a_epsilon_gamma_space_umap_gamma_substates",
    )


def make_phase_portrait(adata) -> None:
    gene = "Malat1" if "Malat1" in adata.var_names else str(adata.var_names[0])
    fig = scptr.pl.phase_portrait(adata, gene, show=False)
    if fig is not None:
        fig.axes[0].set_title(f"Rate estimation phase portrait ({gene})")
        save(fig, "figure_1a_rate_estimation_phase_portrait")


def make_rbp_hub_bar() -> None:
    path = ROOT / "output" / "gap_analysis" / "results" / "network" / "pancreas" / "rbp_hub_counts.csv"
    if not path.exists():
        print(f"missing {path}")
        return
    hubs = pd.read_csv(path, header=None, names=["rbp", "targets"]).head(12)
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.barh(hubs["rbp"][::-1], hubs["targets"][::-1], color="#4C78A8")
    ax.set_xlabel("Significant target edges")
    ax.set_title("Pancreas RBP hub counts")
    fig.tight_layout()
    save(fig, "figure_1c_rbp_network_hub_bargraph")


def copy_generated_figures() -> None:
    copies = {
        "output/figures/aim1/halflife_scatter.png": "figure_2a_pancreas_halflife_scatter",
        "output/mirna_analysis/figures/mirna_analysis_pancreas.png": "figure_2b_pancreas_mirna_target_enrichment",
        "output/scifate_validation/figures/scifate_gamma_vs_ground_truth.png": "figure_2c_scifate_gamma_vs_ground_truth",
        "output/summary/figures/halflife_comparison.png": "figure_2d_cross_dataset_halflife_bargraph",
        "output/summary/figures/robustness_curves.png": "figure_2e_subsampling_robustness_curves",
        "output/figures/aim2/expression_vs_gamma_clustering.png": "figure_3a_pancreas_expression_vs_gamma_umap_combined",
        "output/gap_analysis/figures/invisible_states/invisible_states_pancreas.png": "figure_3b_pancreas_invisible_states_bargraph",
        "output/gap_analysis/figures/invisible_states/invisible_states_dentate_gyrus.png": "figure_3c_dentate_invisible_states_bargraph",
        "output/precedence/figures/precedence_pancreas.png": "figure_3d_pancreas_temporal_precedence",
        "output/precedence/figures/precedence_dentate_gyrus.png": "figure_3e_dentate_temporal_precedence",
        "output/precedence/figures/combined_precedence.png": "figure_3f_combined_temporal_precedence_bargraph",
        "output/figures/aim3/pt_velocity_embedding.png": "figure_4a_pancreas_pt_velocity_umap",
        "output/dentate_gyrus/figures/pt_velocity_embedding.png": "figure_4b_dentate_pt_velocity_umap",
        "output/gap_analysis/figures/velocity_comparison/velocity_comparison_pancreas.png": "figure_4c_pancreas_pt_vs_rna_velocity",
        "output/gap_analysis/figures/velocity_comparison/velocity_comparison_dentate_gyrus.png": "figure_4d_dentate_pt_vs_rna_velocity",
        "output/gap_analysis/figures/network/network_pancreas.png": "figure_5a_pancreas_rbp_network",
        "output/gap_analysis/figures/network/network_dentate_gyrus.png": "figure_5b_dentate_rbp_network",
        "output/deep_benchmarks/11_expression_vs_gamma/figures/pancreas_expr_vs_gamma.png": "supplement_expression_vs_gamma_pancreas",
        "output/deep_benchmarks/11_expression_vs_gamma/figures/dentate_gyrus_expr_vs_gamma.png": "supplement_expression_vs_gamma_dentate_gyrus",
        "output/scifate_validation/figures/scifate_per_timepoint.png": "supplement_scifate_per_timepoint",
        "output/scifate_validation/figures/scifate_top_bottom_boxplot.png": "supplement_scifate_top_bottom_boxplot",
        "output/summary/figures/cross_dataset_heatmap.png": "supplement_cross_dataset_gamma_heatmap",
        "output/summary/figures/enrichment_comparison.png": "supplement_are_nmd_enrichment_comparison",
        "output/dentate_gyrus/figures/halflife_scatter.png": "supplement_dentate_halflife_scatter",
        "output/dentate_gyrus/figures/pt_umap.png": "supplement_dentate_gamma_space_umap",
    }
    for src, dest in copies.items():
        copy_if_exists(src, dest)


def main() -> None:
    set_style()
    OUT.mkdir(parents=True, exist_ok=True)
    copy_generated_figures()
    adata = process_pancreas()
    make_split_umaps(adata)
    make_epsilon_umaps(adata)
    make_phase_portrait(adata)
    make_rbp_hub_bar()
    print(f"labelled paper figures written to {OUT}")


if __name__ == "__main__":
    main()
