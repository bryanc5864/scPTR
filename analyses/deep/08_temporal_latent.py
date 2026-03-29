#!/usr/bin/env python
"""Temporal analysis: do z_T and z_PT capture different aspects of development?

Correlates latent dimensions with pseudotime/differentiation markers to
test whether the disentanglement is biologically meaningful.
"""
from _common import *
import scanpy as sc

OUT = output_dir("08_temporal_latent")


def compute_pseudotime(adata):
    """Compute diffusion pseudotime if not present."""
    if "dpt_pseudotime" in adata.obs.columns:
        return adata.obs["dpt_pseudotime"].values

    # Use scanpy's DPT
    sc.tl.diffmap(adata)
    # Pick root as most common cell type's centroid
    adata.uns["iroot"] = 0
    sc.tl.dpt(adata)
    return adata.obs["dpt_pseudotime"].values


def run(name, loader, cluster_key):
    print(f"\n{'=' * 60}\n{name.upper()}: Temporal latent analysis\n{'=' * 60}")

    adata_dp, model, history = run_deep(loader)

    z_T = adata_dp.obsm["X_z_T"]
    z_PT = adata_dp.obsm["X_z_PT"]
    gamma = adata_dp.layers["gamma"]

    # Compute pseudotime
    try:
        ptime = compute_pseudotime(adata_dp)
        valid_pt = np.isfinite(ptime)
    except Exception as e:
        print(f"  Pseudotime failed: {e}")
        ptime = None
        valid_pt = None

    results = {}

    # 1. Correlation of each latent dimension with pseudotime
    if ptime is not None and valid_pt.sum() > 50:
        r_T_pt = [float(stats.spearmanr(z_T[valid_pt, d], ptime[valid_pt]).statistic)
                   for d in range(z_T.shape[1])]
        r_PT_pt = [float(stats.spearmanr(z_PT[valid_pt, d], ptime[valid_pt]).statistic)
                    for d in range(z_PT.shape[1])]

        max_r_T = max(abs(r) for r in r_T_pt)
        max_r_PT = max(abs(r) for r in r_PT_pt)

        print(f"  Max |r| z_T vs pseudotime:  {max_r_T:.4f}")
        print(f"  Max |r| z_PT vs pseudotime: {max_r_PT:.4f}")

        results["pseudotime"] = {
            "max_r_zT": max_r_T, "max_r_zPT": max_r_PT,
            "r_zT_dims": r_T_pt, "r_zPT_dims": r_PT_pt,
        }

    # 2. Cell-type purity in each latent space
    if cluster_key in adata_dp.obs.columns:
        from sklearn.metrics import silhouette_score
        labels = adata_dp.obs[cluster_key].astype("category").cat.codes.values
        n_sample = min(2000, len(labels))

        sil_T = silhouette_score(z_T, labels, sample_size=n_sample)
        sil_PT = silhouette_score(z_PT, labels, sample_size=n_sample)
        sil_gamma = silhouette_score(gamma, labels, sample_size=n_sample)

        # Also expression space
        if "X_pca" in adata_dp.obsm:
            sil_expr = silhouette_score(adata_dp.obsm["X_pca"][:, :8], labels, sample_size=n_sample)
        else:
            from sklearn.decomposition import PCA
            X_pca = PCA(n_components=8).fit_transform(
                np.log1p(np.asarray(adata_dp.layers["spliced"]))
            )
            sil_expr = silhouette_score(X_pca, labels, sample_size=n_sample)

        print(f"\n  Silhouette scores:")
        print(f"    Expression: {sil_expr:.4f}")
        print(f"    z_T:        {sil_T:.4f}")
        print(f"    z_PT:       {sil_PT:.4f}")
        print(f"    gamma:      {sil_gamma:.4f}")

        results["silhouette"] = {
            "expression": float(sil_expr), "z_T": float(sil_T),
            "z_PT": float(sil_PT), "gamma": float(sil_gamma),
        }

    # 3. Information content: variance explained by z_T vs z_PT
    from sklearn.decomposition import PCA
    var_T = PCA(n_components=min(8, z_T.shape[1])).fit(z_T).explained_variance_ratio_.sum()
    var_PT = PCA(n_components=min(8, z_PT.shape[1])).fit(z_PT).explained_variance_ratio_.sum()
    print(f"\n  Variance explained (top 8 PCs):")
    print(f"    z_T:  {var_T:.4f}")
    print(f"    z_PT: {var_PT:.4f}")
    results["variance_explained"] = {"z_T": float(var_T), "z_PT": float(var_PT)}

    # 4. KL contribution from training
    if history.train_kl:
        # Look at KL at end of training — how much info is encoded?
        final_kl = history.train_kl[-1]
        print(f"  Final KL: {final_kl:.4f}")
        results["final_kl"] = float(final_kl)

    save_json(results, f"{name}_temporal", OUT)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    if "silhouette" in results:
        sil = results["silhouette"]
        labels_plot = ["Expression", "z_T", "z_PT", "gamma"]
        vals = [sil["expression"], sil["z_T"], sil["z_PT"], sil["gamma"]]
        colors = ["gray", "steelblue", "darkorange", "seagreen"]
        axes[0].bar(labels_plot, vals, color=colors, alpha=0.7)
        axes[0].set_ylabel("Silhouette score")
        axes[0].set_title("Cell-type separation")
        axes[0].axhline(0, color="k", lw=0.5)

    if "pseudotime" in results:
        pt_r = results["pseudotime"]
        axes[1].bar(range(len(pt_r["r_zT_dims"])), [abs(r) for r in pt_r["r_zT_dims"]],
                    alpha=0.7, color="steelblue", label="z_T")
        axes[1].bar([x + 0.4 for x in range(len(pt_r["r_zPT_dims"]))],
                    [abs(r) for r in pt_r["r_zPT_dims"]],
                    alpha=0.7, width=0.4, color="darkorange", label="z_PT")
        axes[1].set_xlabel("Latent dimension")
        axes[1].set_ylabel("|r| with pseudotime")
        axes[1].set_title("Temporal correlation per dimension")
        axes[1].legend()

    # PCA of z_T colored by pseudotime
    if ptime is not None:
        z_2d = PCA(n_components=2).fit_transform(z_T)
        sc_plot = axes[2].scatter(z_2d[valid_pt, 0], z_2d[valid_pt, 1],
                                   c=ptime[valid_pt], cmap="viridis", alpha=0.3, s=3)
        plt.colorbar(sc_plot, ax=axes[2], label="Pseudotime")
        axes[2].set_title("z_T colored by pseudotime")
        axes[2].set_xlabel("PC1"); axes[2].set_ylabel("PC2")

    fig.suptitle(f"{name}: Temporal latent structure", y=1.02)
    fig.tight_layout()
    save_fig(fig, f"{name}_temporal", OUT)

    return results


def main():
    set_figure_style()
    all_results = {}
    for name, loader, ck in DATASETS:
        all_results[name] = run(name, loader, ck)
    save_json(all_results, "temporal_all", OUT)


if __name__ == "__main__":
    main()
