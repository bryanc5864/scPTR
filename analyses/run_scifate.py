#!/usr/bin/env python
"""Validate scPTR gamma estimates against sci-fate metabolic labeling ground truth.

sci-fate (Cao et al. 2020, Nature Biotechnology) provides both total and newly
synthesized mRNA counts per cell via 4sU metabolic labeling. This allows us to
compute ground-truth degradation rates and compare them against scPTR's gamma
estimates from splicing kinetics alone.

Key idea:
- old RNA = total - new (pre-existing mRNA)
- degradation_rate ~ new / old (high ratio = fast turnover)
- We expect: genes with high scPTR gamma should have high new/old ratio

Data: A549 cells treated with dexamethasone (0-10h), GEO GSE131351.
"""

from __future__ import annotations

import gzip
import json
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from scipy.io import mmread
from scipy.sparse import csc_matrix

sys.path.insert(0, str(Path(__file__).parent))
from _common import set_figure_style

import scptr

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "scifate_validation"
CACHE_DIR = Path.home() / ".cache" / "scptr" / "scifate"


def save_fig(fig, name, subdir="figures"):
    """Save a matplotlib figure to output dir."""
    if fig is None:
        print(f"  [WARNING] {name}: plot returned None, skipping save")
        return
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def load_scifate_data():
    """Load sci-fate data from GEO-downloaded files.

    Returns AnnData with:
    - X: total gene counts (sparse)
    - layers['new']: newly synthesized counts (sparse)
    - obs: cell annotations (treatment_time, etc.)
    - var: gene annotations (gene_id, gene_short_name)
    """
    print("Loading sci-fate data from GEO files...")

    # Load cell annotations
    cell_ann = pd.read_csv(CACHE_DIR / "GSM3770930_A549_cell_annotate.txt.gz",
                           compression="gzip")
    print(f"  Cells: {len(cell_ann)}")

    # Load gene annotations
    gene_ann = pd.read_csv(CACHE_DIR / "GSM3770930_A549_gene_annotate.txt.gz",
                           compression="gzip")
    print(f"  Genes: {len(gene_ann)}")

    # Load total count matrix (MatrixMarket format, gzipped)
    print("  Loading total count matrix...")
    with gzip.open(CACHE_DIR / "GSM3770930_A549_gene_count.txt.gz", 'rb') as f:
        total_mat = mmread(f)  # genes x cells
    total_mat = csc_matrix(total_mat).T  # -> cells x genes

    # Load newly synthesized count matrix
    print("  Loading newly synthesized count matrix...")
    with gzip.open(CACHE_DIR / "GSM3770930_A549_gene_count_newly_synthesised.txt.gz", 'rb') as f:
        new_mat = mmread(f)  # genes x cells
    new_mat = csc_matrix(new_mat).T  # -> cells x genes

    print(f"  Total matrix: {total_mat.shape}")
    print(f"  New matrix: {new_mat.shape}")

    # Build AnnData
    import anndata as ad
    adata = ad.AnnData(
        X=total_mat,
        obs=cell_ann.set_index("sample"),
        var=gene_ann.set_index("gene_id"),
    )
    adata.layers["new"] = new_mat
    adata.var_names_make_unique()

    # Use gene short names
    adata.var["gene_id_full"] = adata.var_names.tolist()
    adata.var_names = adata.var["gene_short_name"].values
    adata.var_names_make_unique()

    print(f"  AnnData shape: {adata.shape}")
    print(f"  Treatment times: {adata.obs['treatment_time'].value_counts().to_dict()}")

    return adata


def compute_ground_truth_degradation(adata):
    """Compute per-gene ground-truth degradation rate from labeled/unlabeled RNA.

    Ground truth: degradation_rate_proxy = mean(new) / mean(old)
    where old = total - new.

    Genes with high turnover have high new/old ratio.
    """
    total = np.asarray(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)
    new = np.asarray(adata.layers["new"].toarray() if hasattr(adata.layers["new"], 'toarray') else adata.layers["new"])
    old = total - new

    # Per-gene: mean across cells
    mean_new = new.mean(axis=0)
    mean_old = old.mean(axis=0)
    mean_total = total.mean(axis=0)

    # Degradation rate proxy: new/old ratio (high = fast turnover)
    # Only for genes with sufficient expression
    min_expr = 0.5  # minimum mean total expression
    reliable = (mean_total >= min_expr) & (mean_old > 0.1)

    deg_rate = np.full(adata.n_vars, np.nan)
    deg_rate[reliable] = mean_new[reliable] / mean_old[reliable]

    # Also compute fraction-new (new/total), another degradation proxy
    frac_new = np.full(adata.n_vars, np.nan)
    frac_new[reliable] = mean_new[reliable] / mean_total[reliable]

    result = pd.DataFrame({
        "gene": adata.var_names,
        "mean_total": mean_total,
        "mean_new": mean_new,
        "mean_old": mean_old,
        "new_old_ratio": deg_rate,
        "frac_new": frac_new,
    })
    return result


def prepare_for_scptr(adata_scifate):
    """Prepare sci-fate data for scPTR pipeline.

    sci-fate doesn't have unspliced/spliced layers from velocity-style
    preprocessing. Instead, we use:
    - spliced = old RNA (pre-existing, ~steady-state pool)
    - unspliced = new RNA (recently transcribed, proxy for nascent)

    This mapping makes biological sense: newly synthesized RNA is analogous
    to the unspliced pool (recently produced), while old RNA represents the
    mature steady-state pool (analogous to spliced).
    """
    import anndata as ad

    total = adata_scifate.X.toarray() if hasattr(adata_scifate.X, 'toarray') else np.asarray(adata_scifate.X)
    new = adata_scifate.layers["new"].toarray() if hasattr(adata_scifate.layers["new"], 'toarray') else np.asarray(adata_scifate.layers["new"])
    old = total - new

    # Filter to protein-coding genes with sufficient expression
    mean_total = total.mean(axis=0)
    keep = mean_total >= 0.5  # min mean expression
    if "gene_type" in adata_scifate.var.columns:
        is_pc = adata_scifate.var["gene_type"] == "protein_coding"
        keep = keep & is_pc.values

    adata = ad.AnnData(
        X=total[:, keep].astype(np.float32),
        obs=adata_scifate.obs.copy(),
        var=adata_scifate.var.iloc[keep].copy(),
    )
    # Map: unspliced=new, spliced=old
    adata.layers["unspliced"] = new[:, keep].astype(np.float32)
    adata.layers["spliced"] = old[:, keep].astype(np.float32)

    print(f"  Prepared AnnData: {adata.shape}")
    print(f"  Protein-coding genes with mean expr >= 0.5: {keep.sum()}")
    return adata


def main():
    set_figure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # LOAD SCI-FATE DATA
    # =========================================================================
    print("=" * 60)
    print("LOADING SCI-FATE DATA")
    print("=" * 60)
    adata_raw = load_scifate_data()

    # =========================================================================
    # GROUND TRUTH DEGRADATION RATES
    # =========================================================================
    print("\n" + "=" * 60)
    print("COMPUTING GROUND TRUTH DEGRADATION RATES")
    print("=" * 60)

    gt = compute_ground_truth_degradation(adata_raw)
    n_reliable = gt["new_old_ratio"].notna().sum()
    print(f"  Reliable genes: {n_reliable} / {len(gt)}")
    print(f"  New/old ratio: median={gt['new_old_ratio'].median():.4f}, "
          f"mean={gt['new_old_ratio'].mean():.4f}")
    print(f"  Frac new: median={gt['frac_new'].median():.4f}")

    gt.to_csv(res_dir / "ground_truth_degradation.csv", index=False)

    # =========================================================================
    # PER-TIMEPOINT ANALYSIS
    # =========================================================================
    print("\n" + "=" * 60)
    print("PER-TIMEPOINT GROUND TRUTH")
    print("=" * 60)

    timepoints = sorted(adata_raw.obs["treatment_time"].unique())
    gt_by_time = {}
    for tp in timepoints:
        mask = adata_raw.obs["treatment_time"] == tp
        sub = adata_raw[mask].copy()
        gt_tp = compute_ground_truth_degradation(sub)
        gt_by_time[tp] = gt_tp
        n_rel = gt_tp["new_old_ratio"].notna().sum()
        med_ratio = gt_tp["new_old_ratio"].median()
        print(f"  {tp}: {mask.sum()} cells, {n_rel} reliable genes, "
              f"median new/old ratio = {med_ratio:.4f}")

    # Check consistency across timepoints
    print("\n--- Cross-timepoint consistency ---")
    tp_list = list(gt_by_time.keys())
    for i in range(len(tp_list)):
        for j in range(i + 1, len(tp_list)):
            a = gt_by_time[tp_list[i]].set_index("gene")
            b = gt_by_time[tp_list[j]].set_index("gene")
            shared = a.index.intersection(b.index)
            va = a.loc[shared, "new_old_ratio"].values
            vb = b.loc[shared, "new_old_ratio"].values
            valid = np.isfinite(va) & np.isfinite(vb)
            if valid.sum() > 10:
                r, p = stats.spearmanr(va[valid], vb[valid])
                print(f"  {tp_list[i]} vs {tp_list[j]}: Spearman r = {r:.4f} (n={valid.sum()})")

    # =========================================================================
    # RUN SCPTR PIPELINE
    # =========================================================================
    print("\n" + "=" * 60)
    print("RUNNING SCPTR PIPELINE ON SCI-FATE DATA")
    print("=" * 60)

    adata = prepare_for_scptr(adata_raw)

    # Preprocessing
    scptr.pp.filter_genes(adata)
    print(f"  After gene filtering: {adata.shape}")

    scptr.pp.normalize_layers(adata)
    print("  Normalized layers")

    scptr.pp.neighbors(adata, n_neighbors=30)
    print("  Built kNN graph")

    scptr.pp.smooth_layers(adata)
    print("  Smoothed layers")

    # Core analysis
    scptr.tl.estimate_beta(adata)
    beta = adata.var["beta"].values
    print(f"  Beta: median={np.median(beta):.4f}, max={np.max(beta):.4f}")

    scptr.tl.estimate_gamma(adata)
    gamma = adata.layers["gamma"]
    gamma_med = np.median(gamma, axis=0)
    print(f"  Gamma: shape={gamma.shape}, median of medians={np.median(gamma_med):.4f}")
    print(f"  Gamma max: {np.max(gamma):.4f}")

    # =========================================================================
    # CORRELATION: SCPTR GAMMA vs GROUND TRUTH
    # =========================================================================
    print("\n" + "=" * 60)
    print("SCPTR GAMMA vs GROUND TRUTH DEGRADATION RATES")
    print("=" * 60)
    print("  NOTE: Since gamma = beta * unspliced/spliced and we map")
    print("  new→unspliced, old→spliced, the gamma-vs-new/old correlation")
    print("  is partially tautological. The independent validation is the")
    print("  correlation with published half-lives (Schofield 2018).")

    # Build gene-level comparison
    gamma_series = pd.Series(gamma_med, index=adata.var_names)
    gt_indexed = gt.set_index("gene")

    shared = gamma_series.index.intersection(gt_indexed.index)
    print(f"  Shared genes: {len(shared)}")

    g = gamma_series[shared].values.astype(float)
    gt_ratio = gt_indexed.loc[shared, "new_old_ratio"].values.astype(float)
    gt_frac = gt_indexed.loc[shared, "frac_new"].values.astype(float)

    # Filter: need both values finite and positive
    valid_ratio = np.isfinite(g) & np.isfinite(gt_ratio) & (g > 0) & (gt_ratio > 0)
    valid_frac = np.isfinite(g) & np.isfinite(gt_frac) & (g > 0) & (gt_frac > 0)

    results = {}

    # Correlation with new/old ratio
    if valid_ratio.sum() > 10:
        g_r = g[valid_ratio]
        gt_r = gt_ratio[valid_ratio]
        sp_r, sp_p = stats.spearmanr(g_r, gt_r)
        pe_r, pe_p = stats.pearsonr(np.log1p(g_r), np.log1p(gt_r))
        print(f"\n  vs new/old ratio (n={valid_ratio.sum()}):")
        print(f"    Spearman r = {sp_r:.4f} (p = {sp_p:.2e})")
        print(f"    Pearson  r = {pe_r:.4f} (p = {pe_p:.2e}) [log-space]")
        results["new_old_ratio"] = {
            "spearman_r": float(sp_r), "spearman_p": float(sp_p),
            "pearson_r": float(pe_r), "pearson_p": float(pe_p),
            "n_genes": int(valid_ratio.sum()),
        }
    else:
        print("  Not enough shared genes for new/old ratio correlation.")
        results["new_old_ratio"] = {"n_genes": int(valid_ratio.sum())}

    # Correlation with fraction new
    if valid_frac.sum() > 10:
        g_f = g[valid_frac]
        gt_f = gt_frac[valid_frac]
        sp_r, sp_p = stats.spearmanr(g_f, gt_f)
        pe_r, pe_p = stats.pearsonr(np.log1p(g_f), np.log1p(gt_f))
        print(f"\n  vs fraction new (n={valid_frac.sum()}):")
        print(f"    Spearman r = {sp_r:.4f} (p = {sp_p:.2e})")
        print(f"    Pearson  r = {pe_r:.4f} (p = {pe_p:.2e}) [log-space]")
        results["frac_new"] = {
            "spearman_r": float(sp_r), "spearman_p": float(sp_p),
            "pearson_r": float(pe_r), "pearson_p": float(pe_p),
            "n_genes": int(valid_frac.sum()),
        }
    else:
        print("  Not enough shared genes for fraction new correlation.")
        results["frac_new"] = {"n_genes": int(valid_frac.sum())}

    # =========================================================================
    # INDEPENDENT VALIDATION: PUBLISHED HALF-LIVES (not tautological)
    # =========================================================================
    print("\n--- Independent validation: published half-life correlations ---")
    print("  (This is the key result — fully independent ground truth)")
    hl_human = scptr.datasets.schofield2018_halflives()
    corr_human = scptr.benchmark.correlate_with_halflives(adata, hl_human)
    print(f"  Human half-lives (Schofield 2018): Spearman r = {corr_human['spearman_r']:.4f} "
          f"(p={corr_human['spearman_p']:.2e}, n={corr_human['n_genes']})")
    results["halflife_human"] = {
        k: v for k, v in corr_human.items() if k != "matched_genes"
    }

    hl_mouse = scptr.datasets.herzog2017_halflives()
    corr_mouse = scptr.benchmark.correlate_with_halflives(adata, hl_mouse)
    print(f"  Mouse half-lives (Herzog 2017):    Spearman r = {corr_mouse['spearman_r']:.4f} "
          f"(p={corr_mouse['spearman_p']:.2e}, n={corr_mouse['n_genes']})")
    results["halflife_mouse"] = {
        k: v for k, v in corr_mouse.items() if k != "matched_genes"
    }

    with open(res_dir / "scifate_validation.json", "w") as f:
        json.dump(results, f, indent=2)

    # =========================================================================
    # SCATTER PLOTS
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: gamma vs new/old ratio
    if valid_ratio.sum() > 10:
        g_r = g[valid_ratio]
        gt_r = gt_ratio[valid_ratio]
        axes[0].scatter(gt_r, g_r, alpha=0.15, s=8, c="steelblue")
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Ground truth: new/old RNA ratio")
        axes[0].set_ylabel("scPTR median gamma")
        sp_r = results["new_old_ratio"]["spearman_r"]
        sp_p = results["new_old_ratio"]["spearman_p"]
        axes[0].set_title(f"vs New/Old ratio\n(Spearman r={sp_r:.3f}, p={sp_p:.1e})")

    # Panel 2: gamma vs fraction new
    if valid_frac.sum() > 10:
        g_f = g[valid_frac]
        gt_f = gt_frac[valid_frac]
        axes[1].scatter(gt_f, g_f, alpha=0.15, s=8, c="darkorange")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Ground truth: fraction new RNA")
        axes[1].set_ylabel("scPTR median gamma")
        sp_r = results["frac_new"]["spearman_r"]
        sp_p = results["frac_new"]["spearman_p"]
        axes[1].set_title(f"vs Fraction new\n(Spearman r={sp_r:.3f}, p={sp_p:.1e})")

    # Panel 3: Distribution comparison
    ax3 = axes[2]
    # Log-transform and z-score both, show rank correlation
    if valid_ratio.sum() > 10:
        g_log = np.log1p(g[valid_ratio])
        gt_log = np.log1p(gt_ratio[valid_ratio])
        # Rank both
        g_rank = stats.rankdata(g_log)
        gt_rank = stats.rankdata(gt_log)
        ax3.scatter(gt_rank / len(gt_rank), g_rank / len(g_rank),
                   alpha=0.1, s=5, c="purple")
        ax3.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)
        ax3.set_xlabel("Ground truth rank (fractional)")
        ax3.set_ylabel("scPTR gamma rank (fractional)")
        ax3.set_title("Rank-rank plot")

    fig.suptitle("sci-fate Validation: scPTR Gamma vs Ground Truth Degradation",
                fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "scifate_gamma_vs_ground_truth")

    # =========================================================================
    # PER-TIMEPOINT VALIDATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("PER-TIMEPOINT VALIDATION")
    print("=" * 60)

    tp_results = {}
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, tp in enumerate(timepoints):
        gt_tp = gt_by_time[tp].set_index("gene")
        shared_tp = gamma_series.index.intersection(gt_tp.index)
        g_tp = gamma_series[shared_tp].values.astype(float)
        gt_tp_ratio = gt_tp.loc[shared_tp, "new_old_ratio"].values.astype(float)
        valid = np.isfinite(g_tp) & np.isfinite(gt_tp_ratio) & (g_tp > 0) & (gt_tp_ratio > 0)

        if valid.sum() > 10:
            sp_r, sp_p = stats.spearmanr(g_tp[valid], gt_tp_ratio[valid])
            print(f"  {tp}: Spearman r = {sp_r:.4f} (n={valid.sum()})")
            tp_results[tp] = {"spearman_r": float(sp_r), "spearman_p": float(sp_p),
                             "n_genes": int(valid.sum())}

            if idx < len(axes):
                axes[idx].scatter(gt_tp_ratio[valid], g_tp[valid],
                                alpha=0.1, s=5, c="steelblue")
                axes[idx].set_xscale("log")
                axes[idx].set_yscale("log")
                axes[idx].set_xlabel("New/old ratio")
                axes[idx].set_ylabel("scPTR gamma")
                axes[idx].set_title(f"DEX {tp} (r={sp_r:.3f}, n={valid.sum()})")
        else:
            print(f"  {tp}: Not enough genes ({valid.sum()})")

    # Remove unused axes
    for idx in range(len(timepoints), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("sci-fate: Per-timepoint scPTR gamma vs ground truth",
                fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "scifate_per_timepoint")

    with open(res_dir / "scifate_per_timepoint.json", "w") as f:
        json.dump(tp_results, f, indent=2)

    # =========================================================================
    # TOP/BOTTOM GENE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 60)
    print("TOP/BOTTOM GENE ANALYSIS")
    print("=" * 60)

    if valid_ratio.sum() > 100:
        # Compare top/bottom gamma genes with ground truth ranking
        gene_df = pd.DataFrame({
            "gene": shared[valid_ratio],
            "gamma": g[valid_ratio],
            "new_old_ratio": gt_ratio[valid_ratio],
        })
        gene_df = gene_df.sort_values("gamma", ascending=False)

        # Top 10% gamma genes
        n10 = max(10, len(gene_df) // 10)
        top_gamma = gene_df.head(n10)
        bot_gamma = gene_df.tail(n10)

        top_gt_med = top_gamma["new_old_ratio"].median()
        bot_gt_med = bot_gamma["new_old_ratio"].median()

        print(f"\n  Top {n10} gamma genes: median new/old ratio = {top_gt_med:.4f}")
        print(f"  Bottom {n10} gamma genes: median new/old ratio = {bot_gt_med:.4f}")
        print(f"  Fold difference: {top_gt_med / bot_gt_med:.2f}x")

        # Mann-Whitney test
        u_stat, mw_p = stats.mannwhitneyu(
            top_gamma["new_old_ratio"].values,
            bot_gamma["new_old_ratio"].values,
            alternative="greater"
        )
        print(f"  Mann-Whitney p-value (top > bottom): {mw_p:.2e}")

        results["top_bottom_analysis"] = {
            "n_per_group": n10,
            "top_gamma_median_gt": float(top_gt_med),
            "bottom_gamma_median_gt": float(bot_gt_med),
            "fold_difference": float(top_gt_med / bot_gt_med),
            "mann_whitney_p": float(mw_p),
        }

        # Save updated results
        with open(res_dir / "scifate_validation.json", "w") as f:
            json.dump(results, f, indent=2)

        # Boxplot
        fig, ax = plt.subplots(figsize=(6, 5))
        positions = [1, 2]
        bp = ax.boxplot(
            [top_gamma["new_old_ratio"].values, bot_gamma["new_old_ratio"].values],
            positions=positions,
            widths=0.6,
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor("salmon")
        bp["boxes"][1].set_facecolor("lightblue")
        ax.set_xticks(positions)
        ax.set_xticklabels([f"Top {n10}\n(high gamma)", f"Bottom {n10}\n(low gamma)"])
        ax.set_ylabel("Ground truth: new/old RNA ratio")
        ax.set_title(f"High-gamma genes have higher turnover\n"
                    f"(fold={top_gt_med/bot_gt_med:.1f}x, p={mw_p:.1e})")
        fig.tight_layout()
        save_fig(fig, "scifate_top_bottom_boxplot")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Dataset: sci-fate A549 ({adata_raw.n_obs} cells, {adata_raw.n_vars} genes)")
    print(f"  scPTR pipeline: {adata.n_obs} cells, {adata.n_vars} genes")
    if "new_old_ratio" in results and "spearman_r" in results["new_old_ratio"]:
        print(f"  Gamma vs new/old ratio: Spearman r = {results['new_old_ratio']['spearman_r']:.4f}")
    if "frac_new" in results and "spearman_r" in results["frac_new"]:
        print(f"  Gamma vs frac new: Spearman r = {results['frac_new']['spearman_r']:.4f}")
    if "halflife_human" in results:
        print(f"  Human half-life (INDEPENDENT): Spearman r = {results['halflife_human']['spearman_r']:.4f}")
    if "halflife_mouse" in results and "spearman_r" in results["halflife_mouse"]:
        print(f"  Mouse half-life (INDEPENDENT): Spearman r = {results['halflife_mouse']['spearman_r']:.4f}")
    if "top_bottom_analysis" in results:
        tb = results["top_bottom_analysis"]
        print(f"  Top vs bottom gamma: {tb['fold_difference']:.1f}x fold diff (p={tb['mann_whitney_p']:.1e})")
    print(f"\nAll results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
