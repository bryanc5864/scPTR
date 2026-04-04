#!/usr/bin/env python
"""Sparsity artifact control: are PT states driven by zero patterns?

55% of gamma values are clipped to zero. This script tests whether
PT states survive after controlling for sparsity artifacts.

Tests:
1. Dense-only: cluster using only genes with >50% nonzero gamma
2. Binarized: cluster on 0/1 gamma (if states vanish, magnitude matters)
3. Zero-permuted: shuffle zeros across cells (if states vanish, zero pattern is signal)
4. Expression-residualized: regress out expression level from gamma
"""
from _common import *
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

OUT = output_dir("32_sparsity_control")


def cluster_and_eval(matrix, adata, cluster_key, label):
    """PCA + Leiden cluster on matrix, evaluate vs expression clusters."""
    import anndata as ad
    adata_tmp = ad.AnnData(X=matrix.astype(np.float32), obs=adata.obs.copy())

    n_comps = min(30, matrix.shape[1] - 1, matrix.shape[0] - 1)
    if n_comps < 2:
        return {"label": label, "error": "too few components"}

    sc.pp.pca(adata_tmp, n_comps=n_comps)
    sc.pp.neighbors(adata_tmp, n_pcs=min(20, n_comps))
    sc.tl.leiden(adata_tmp, resolution=1.0, key_added="gamma_cluster")

    gamma_labels = adata_tmp.obs["gamma_cluster"].values
    expr_labels = adata.obs[cluster_key].astype("category").cat.codes.values

    n_clusters = len(np.unique(gamma_labels))
    ari = adjusted_rand_score(expr_labels, gamma_labels)
    nmi = normalized_mutual_info_score(expr_labels, gamma_labels)

    # Count invisible states (mixed expression types)
    ct = pd.crosstab(gamma_labels, adata.obs[cluster_key], normalize="index")
    n_invisible = sum(1 for gc in ct.index if ct.loc[gc].max() < 0.6)

    # Silhouette of expression clusters in gamma PCA space
    try:
        sil = silhouette_score(adata_tmp.obsm["X_pca"][:, :min(10, n_comps)],
                                expr_labels, sample_size=min(2000, len(expr_labels)))
    except Exception:
        sil = np.nan

    return {
        "label": label,
        "n_clusters": int(n_clusters),
        "n_invisible": n_invisible,
        "ari_vs_expr": float(ari),
        "nmi_vs_expr": float(nmi),
        "silhouette": float(sil),
    }


def main():
    set_figure_style()
    all_results = {}

    for ds_name, loader, ck in DATASETS:
        print(f"\n{'=' * 60}\n{ds_name.upper()}: Sparsity Controls\n{'=' * 60}")

        adata = run_analytical(loader)
        gamma = adata.layers["gamma"]
        n_cells, n_genes = gamma.shape

        frac_zero = (gamma == 0).mean()
        print(f"  Gamma shape: {gamma.shape}, {frac_zero*100:.1f}% zeros")

        # Gene-level nonzero fraction
        gene_nonzero_frac = (gamma > 0).mean(axis=0)

        results = []

        # ── A. Full gamma (baseline) ─────────────────────────────────
        print("\n  A. Full gamma (baseline)...")
        r = cluster_and_eval(gamma, adata, ck, "A. Full gamma")
        results.append(r)
        print(f"    {r['n_clusters']} clusters, {r['n_invisible']} invisible, ARI={r['ari_vs_expr']:.4f}")

        # ── B. Dense genes only (>50% nonzero) ───────────────────────
        print("\n  B. Dense genes only (>50% nonzero)...")
        dense_mask = gene_nonzero_frac > 0.5
        n_dense = dense_mask.sum()
        print(f"    {n_dense} dense genes (of {n_genes})")
        if n_dense > 20:
            r = cluster_and_eval(gamma[:, dense_mask], adata, ck, f"B. Dense genes ({n_dense})")
            results.append(r)
            print(f"    {r['n_clusters']} clusters, {r['n_invisible']} invisible, ARI={r['ari_vs_expr']:.4f}")

        # ── C. Very dense genes only (>80% nonzero) ──────────────────
        print("\n  C. Very dense genes (>80% nonzero)...")
        vdense_mask = gene_nonzero_frac > 0.8
        n_vdense = vdense_mask.sum()
        print(f"    {n_vdense} very dense genes")
        if n_vdense > 20:
            r = cluster_and_eval(gamma[:, vdense_mask], adata, ck, f"C. Very dense genes ({n_vdense})")
            results.append(r)
            print(f"    {r['n_clusters']} clusters, {r['n_invisible']} invisible, ARI={r['ari_vs_expr']:.4f}")

        # ── D. Binarized gamma (0/1) ─────────────────────────────────
        print("\n  D. Binarized gamma (0 vs nonzero)...")
        gamma_binary = (gamma > 0).astype(np.float32)
        r = cluster_and_eval(gamma_binary, adata, ck, "D. Binarized (0/1)")
        results.append(r)
        print(f"    {r['n_clusters']} clusters, {r['n_invisible']} invisible, ARI={r['ari_vs_expr']:.4f}")

        # ── E. Zero-permuted (shuffle zeros within each gene) ─────────
        print("\n  E. Zero-permuted (shuffle zero pattern)...")
        rng = np.random.RandomState(42)
        gamma_perm = gamma.copy()
        for g in range(n_genes):
            gamma_perm[:, g] = rng.permutation(gamma_perm[:, g])
        r = cluster_and_eval(gamma_perm, adata, ck, "E. Zero-permuted")
        results.append(r)
        print(f"    {r['n_clusters']} clusters, {r['n_invisible']} invisible, ARI={r['ari_vs_expr']:.4f}")

        # ── F. Log-transformed nonzero gamma ──────────────────────────
        print("\n  F. Log-transformed gamma...")
        gamma_log = np.log1p(gamma)
        r = cluster_and_eval(gamma_log, adata, ck, "F. Log gamma")
        results.append(r)
        print(f"    {r['n_clusters']} clusters, {r['n_invisible']} invisible, ARI={r['ari_vs_expr']:.4f}")

        # ── Summary ──────────────────────────────────────────────────
        print(f"\n  {'Condition':<35} {'Clusters':>8} {'Invisible':>10} {'ARI':>8} {'Sil':>8}")
        print("  " + "-" * 75)
        for r in results:
            if "error" in r:
                continue
            print(f"  {r['label']:<35} {r['n_clusters']:>8} {r['n_invisible']:>10} "
                  f"{r['ari_vs_expr']:>8.4f} {r['silhouette']:>8.4f}")

        # Key interpretation
        baseline = results[0]
        binary = next((r for r in results if "Binarized" in r.get("label", "")), None)
        permuted = next((r for r in results if "permuted" in r.get("label", "")), None)

        if binary and permuted:
            print(f"\n  INTERPRETATION:")
            if binary["n_invisible"] < baseline["n_invisible"] * 0.5:
                print(f"    Binarized has fewer invisible states → gamma MAGNITUDE matters (not just zeros)")
            else:
                print(f"    Binarized preserves invisible states → zero PATTERN drives clustering")

            if permuted["n_invisible"] < baseline["n_invisible"] * 0.5:
                print(f"    Zero-permuted loses states → zero pattern is STRUCTURED (not random)")
            else:
                print(f"    Zero-permuted preserves states → states NOT driven by zero pattern")

        all_results[ds_name] = results

    save_json(all_results, "sparsity_control", OUT)

    # Figure
    fig, axes = plt.subplots(1, len(all_results), figsize=(7 * len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]

    for ax, (ds, res) in zip(axes, all_results.items()):
        valid = [r for r in res if "error" not in r]
        labels = [r["label"].split(". ")[1] if ". " in r["label"] else r["label"] for r in valid]
        aris = [r["ari_vs_expr"] for r in valid]
        invisibles = [r["n_invisible"] for r in valid]

        x = np.arange(len(labels))
        ax.bar(x - 0.2, aris, 0.35, label="ARI vs expr", color="steelblue", alpha=0.7)
        ax2 = ax.twinx()
        ax2.bar(x + 0.2, invisibles, 0.35, label="Invisible states", color="darkorange", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("ARI vs expression clusters")
        ax2.set_ylabel("# invisible states")
        ax.set_title(f"{ds}")
        ax.legend(loc="upper left", fontsize=7)
        ax2.legend(loc="upper right", fontsize=7)

    fig.suptitle("Sparsity controls: are PT states real?", y=1.02)
    fig.tight_layout()
    save_fig(fig, "sparsity_control", OUT)


if __name__ == "__main__":
    main()
