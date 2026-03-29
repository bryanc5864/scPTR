#!/usr/bin/env python
"""Honest analysis of the sci-fate tautology.

gamma = beta * Mu/Ms ≈ beta * new/old
ground_truth = new/old
Therefore gamma ≈ beta * ground_truth → high correlation is structural.

This script quantifies how much of r=0.99 is real vs tautological.
"""
from _common import *
import gzip
from scipy.io import mmread
from scipy.sparse import csc_matrix

OUT = output_dir("04_scifate_tautology")
CACHE_DIR = Path.home() / ".cache" / "scptr" / "scifate"


def main():
    set_figure_style()

    if not CACHE_DIR.exists():
        print("[SKIP] sci-fate data not cached. Run analyses/run_scifate.py first.")
        return

    print("Loading sci-fate data...")
    cell_ann = pd.read_csv(CACHE_DIR / "GSM3770930_A549_cell_annotate.txt.gz", compression="gzip")
    gene_ann = pd.read_csv(CACHE_DIR / "GSM3770930_A549_gene_annotate.txt.gz", compression="gzip")

    with gzip.open(CACHE_DIR / "GSM3770930_A549_gene_count.txt.gz", "rb") as f:
        total_mat = csc_matrix(mmread(f)).T
    with gzip.open(CACHE_DIR / "GSM3770930_A549_gene_count_newly_synthesised.txt.gz", "rb") as f:
        new_mat = csc_matrix(mmread(f)).T

    total = np.asarray(total_mat.todense())
    new = np.asarray(new_mat.todense())
    old = total - new

    mean_new, mean_old, mean_total = new.mean(0), old.mean(0), total.mean(0)
    reliable = (mean_total >= 0.5) & (mean_old > 0.1)

    gt_dict = {}
    for i, gn in enumerate(gene_ann["gene_short_name"].values):
        if isinstance(gn, str) and reliable[i] and gn not in gt_dict:
            gt_dict[gn] = mean_new[i] / mean_old[i]
    gt_s = pd.Series(gt_dict)

    # Run pipeline
    import anndata as ad
    keep = mean_total >= 0.5
    if "gene_type" in gene_ann.columns:
        keep = keep & (gene_ann["gene_type"] == "protein_coding").values

    adata = ad.AnnData(
        X=total[:, keep].astype(np.float32),
        obs=cell_ann.set_index("sample"),
        var=gene_ann.set_index("gene_id").iloc[keep].copy(),
    )
    adata.layers["unspliced"] = new[:, keep].astype(np.float32)
    adata.layers["spliced"] = old[:, keep].astype(np.float32)
    adata.var_names = adata.var["gene_short_name"].values
    adata.var_names_make_unique()

    scptr.pp.filter_genes(adata, min_unspliced_counts=1, min_unspliced_cells=1)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)

    gamma_s = pd.Series(np.median(adata.layers["gamma"], 0), index=adata.var_names)
    beta_s = pd.Series(adata.var["beta"].values, index=adata.var_names)

    shared = gamma_s.index.intersection(gt_s.dropna().index)
    g, t, b = gamma_s[shared].values, gt_s[shared].values, beta_s[shared].values
    v = np.isfinite(g) & np.isfinite(t) & (g > 0) & (t > 0) & np.isfinite(b)
    g, t, b = g[v].astype(float), t[v].astype(float), b[v].astype(float)

    r_gamma_gt, _ = stats.spearmanr(g, t)
    r_residual, _ = stats.spearmanr(g / (b + 1e-8), t)
    beta_cv = np.std(b) / np.mean(b)

    # Raw ratio baseline
    raw = np.median(new[:, keep], 0) / np.clip(np.median(old[:, keep], 0), 1e-8, None)
    raw_s = pd.Series(raw, index=adata.var_names[:len(raw)])
    sh2 = raw_s.index.intersection(gt_s.dropna().index)
    rv, tv2 = raw_s[sh2].values.astype(float), gt_s[sh2].values.astype(float)
    v2 = np.isfinite(rv) & np.isfinite(tv2) & (rv > 0) & (tv2 > 0)
    r_raw, _ = stats.spearmanr(rv[v2], tv2[v2]) if v2.sum() > 3 else (np.nan, np.nan)

    print(f"\n  gamma vs GT:        r = {r_gamma_gt:.4f} (n={len(g)})")
    print(f"  gamma/beta vs GT:   r = {r_residual:.4f} (after removing beta)")
    print(f"  raw new/old vs GT:  r = {r_raw:.4f} (no model)")
    print(f"  beta CV:            {beta_cv:.4f}")
    print(f"  Pipeline adds:      Δr = {r_gamma_gt - r_raw:.4f}")
    severity = "high" if r_residual > 0.98 else "moderate" if r_residual > 0.90 else "low"
    print(f"  Tautology severity: {severity}")

    results = {
        "r_gamma_gt": float(r_gamma_gt), "r_residual": float(r_residual),
        "r_raw": float(r_raw), "beta_cv": float(beta_cv),
        "pipeline_delta_r": float(r_gamma_gt - r_raw), "severity": severity,
        "n_genes": len(g),
    }
    save_json(results, "scifate_tautology", OUT)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(t, g, alpha=0.05, s=3, c="steelblue")
    axes[0].set_xlabel("Ground truth (new/old)"); axes[0].set_ylabel("scPTR gamma")
    axes[0].set_title(f"gamma vs GT (r={r_gamma_gt:.3f})"); axes[0].set_xscale("log"); axes[0].set_yscale("log")

    axes[1].scatter(t, g / (b + 1e-8), alpha=0.05, s=3, c="darkorange")
    axes[1].set_xlabel("Ground truth"); axes[1].set_ylabel("gamma / beta")
    axes[1].set_title(f"After removing beta (r={r_residual:.3f})"); axes[1].set_xscale("log"); axes[1].set_yscale("log")

    bars = axes[2].bar(["Raw\nnew/old", "scPTR\ngamma", "gamma/\nbeta"],
                       [abs(r_raw), abs(r_gamma_gt), abs(r_residual)],
                       color=["gray", "steelblue", "darkorange"], alpha=0.7)
    axes[2].set_ylabel("|Spearman r| with ground truth")
    axes[2].set_title("Tautology decomposition")
    axes[2].set_ylim(0.9, 1.01)

    fig.tight_layout()
    save_fig(fig, "scifate_tautology", OUT)


if __name__ == "__main__":
    main()
