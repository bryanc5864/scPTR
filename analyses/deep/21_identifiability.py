#!/usr/bin/env python
"""Empirical identifiability: negative controls for disentanglement.

Shows the z_T/z_PT disentanglement is real by comparing:
- Real data: z_T captures cell type, z_PT captures different structure
- Permuted data: both latents are random noise

If disentanglement disappears on permuted data, it's a real signal.
"""
from _common import *
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans

OUT = output_dir("21_identifiability")


def run_permutation_test(adata_base, cluster_key, n_perm=3):
    """Run DeepPTR on real vs permuted data."""
    from scipy.sparse import issparse

    # ── Real data ────────────────────────────────────────────────────
    print("  Real data...")
    adata_real = adata_base.copy()
    for key in ("spliced", "unspliced"):
        if key in adata_real.layers and issparse(adata_real.layers[key]):
            adata_real.layers[key] = np.asarray(adata_real.layers[key].todense())

    torch.set_num_threads(4)
    scptr.deep.fit_deepptr(adata_real, verbose=False, **DEEP_HP)

    labels = adata_real.obs[cluster_key].astype("category").cat.codes.values
    n_sample = min(2000, len(labels))

    sil_T_real = silhouette_score(adata_real.obsm["X_z_T"], labels, sample_size=n_sample)
    sil_PT_real = silhouette_score(adata_real.obsm["X_z_PT"], labels, sample_size=n_sample)

    # PT cluster vs expression cluster ARI
    km = KMeans(n_clusters=5, random_state=0, n_init=10)
    pt_labels = km.fit_predict(adata_real.obsm["X_z_PT"])
    ari_real = adjusted_rand_score(labels, pt_labels)

    # Count PT-specific genes
    gamma = adata_real.layers["gamma"]
    z_T = adata_real.obsm["X_z_T"]
    z_PT = adata_real.obsm["X_z_PT"]
    n_pt_genes = 0
    for g in range(adata_real.n_vars):
        gv = gamma[:, g]
        if gv.std() < 1e-8:
            continue
        r_T = max(abs(stats.spearmanr(gv, z_T[:, d]).statistic) for d in range(z_T.shape[1]))
        r_PT = max(abs(stats.spearmanr(gv, z_PT[:, d]).statistic) for d in range(z_PT.shape[1]))
        if r_PT > 0.3 and r_PT > r_T * 1.5:
            n_pt_genes += 1

    print(f"    sil_T={sil_T_real:.4f}, sil_PT={sil_PT_real:.4f}, ARI={ari_real:.4f}, PT_genes={n_pt_genes}")

    # ── Permuted data ────────────────────────────────────────────────
    perm_results = []
    for p in range(n_perm):
        print(f"  Permutation {p+1}/{n_perm}...")
        adata_perm = adata_base.copy()
        for key in ("spliced", "unspliced"):
            if key in adata_perm.layers and issparse(adata_perm.layers[key]):
                adata_perm.layers[key] = np.asarray(adata_perm.layers[key].todense())

        # Permute cells independently per gene (destroy cell-gene structure)
        rng = np.random.RandomState(p)
        s_perm = adata_perm.layers["spliced"].copy()
        u_perm = adata_perm.layers["unspliced"].copy()
        for g in range(s_perm.shape[1]):
            s_perm[:, g] = rng.permutation(s_perm[:, g])
            u_perm[:, g] = rng.permutation(u_perm[:, g])
        adata_perm.layers["spliced"] = s_perm
        adata_perm.layers["unspliced"] = u_perm

        torch.set_num_threads(4)
        try:
            scptr.deep.fit_deepptr(adata_perm, verbose=False, **{**DEEP_HP, "seed": p})
        except Exception as e:
            print(f"    Failed: {e}")
            continue

        sil_T_perm = silhouette_score(adata_perm.obsm["X_z_T"], labels, sample_size=n_sample)
        sil_PT_perm = silhouette_score(adata_perm.obsm["X_z_PT"], labels, sample_size=n_sample)

        pt_labels_perm = km.fit_predict(adata_perm.obsm["X_z_PT"])
        ari_perm = adjusted_rand_score(labels, pt_labels_perm)

        perm_results.append({
            "sil_T": float(sil_T_perm),
            "sil_PT": float(sil_PT_perm),
            "ari": float(ari_perm),
        })
        print(f"    sil_T={sil_T_perm:.4f}, sil_PT={sil_PT_perm:.4f}, ARI={ari_perm:.4f}")

    return {
        "real": {
            "sil_T": float(sil_T_real),
            "sil_PT": float(sil_PT_real),
            "ari": float(ari_real),
            "n_pt_genes": n_pt_genes,
        },
        "permuted": perm_results,
    }


def main():
    set_figure_style()
    all_results = {}

    for ds_name, loader, ck in DATASETS:
        print(f"\n{'=' * 60}\n{ds_name.upper()}: Identifiability test\n{'=' * 60}")

        adata_raw = loader()
        scptr.pp.filter_genes(adata_raw)
        scptr.pp.normalize_layers(adata_raw)
        scptr.pp.neighbors(adata_raw, n_neighbors=30)
        scptr.pp.smooth_layers(adata_raw)
        scptr.tl.estimate_beta(adata_raw)
        adata_base = select_top_genes(adata_raw, n_top=300)

        results = run_permutation_test(adata_base, ck, n_perm=3)
        all_results[ds_name] = results

        # Summary
        mean_perm_sil_T = np.mean([p["sil_T"] for p in results["permuted"]])
        mean_perm_ari = np.mean([p["ari"] for p in results["permuted"]])

        print(f"\n  SUMMARY:")
        print(f"    z_T silhouette: real={results['real']['sil_T']:.4f}, permuted={mean_perm_sil_T:.4f}")
        print(f"    PT-expr ARI:    real={results['real']['ari']:.4f}, permuted={mean_perm_ari:.4f}")
        print(f"    PT-specific genes: real={results['real']['n_pt_genes']}")

    save_json(all_results, "identifiability", OUT)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, (ds_name, res) in enumerate(all_results.items()):
        if ax_idx >= 2:
            break
        ax = axes[ax_idx]

        metrics = ["sil_T", "sil_PT", "ari"]
        labels = ["Silhouette z_T", "Silhouette z_PT", "ARI (PT vs expr)"]
        real_vals = [res["real"][m] for m in metrics]
        perm_vals = [np.mean([p[m] for p in res["permuted"]]) for m in metrics]
        perm_stds = [np.std([p[m] for p in res["permuted"]]) for m in metrics]

        x = np.arange(len(metrics))
        ax.bar(x - 0.2, real_vals, 0.35, label="Real", color="darkorange", alpha=0.8)
        ax.bar(x + 0.2, perm_vals, 0.35, yerr=perm_stds, label="Permuted",
               color="gray", alpha=0.6, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Score")
        ax.set_title(f"{ds_name}: Real vs permuted")
        ax.legend()
        ax.axhline(0, color="k", lw=0.5)

    fig.suptitle("Empirical identifiability: disentanglement is real", y=1.02)
    fig.tight_layout()
    save_fig(fig, "identifiability", OUT)


if __name__ == "__main__":
    main()
