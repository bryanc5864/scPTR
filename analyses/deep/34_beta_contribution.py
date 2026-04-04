#!/usr/bin/env python
"""Beta contribution: does multiplying by beta improve gamma estimates?

Compares:
- gamma = beta * Mu/Ms (scPTR)
- gamma_naive = Mu/Ms (no beta)
- gamma_scvelo = regression slope (scVelo SS)

If beta doesn't help, scPTR is literally the same as Mu/Ms ratio.
"""
from _common import *

OUT = output_dir("34_beta_contribution")


def main():
    set_figure_style()
    _, hl_human = load_halflife_refs()
    hl_mouse, _ = load_halflife_refs()

    all_results = {}

    for ds_name, loader, ck in DATASETS:
        print(f"\n{'=' * 60}\n{ds_name.upper()}: Beta Contribution\n{'=' * 60}")

        adata = loader()
        scptr.pp.filter_genes(adata)
        scptr.pp.normalize_layers(adata)
        scptr.pp.neighbors(adata, n_neighbors=30)
        scptr.pp.smooth_layers(adata)

        from scipy.sparse import issparse
        Mu = adata.layers["Mu"]
        Ms = adata.layers["Ms"]
        if issparse(Mu):
            Mu = np.asarray(Mu.todense())
        if issparse(Ms):
            Ms = np.asarray(Ms.todense())
        Mu = np.asarray(Mu, dtype=float)
        Ms = np.asarray(Ms, dtype=float)

        # ── Naive gamma: Mu/Ms ───────────────────────────────────────
        gamma_naive = np.where(Ms > 0.01, Mu / Ms, 0)
        # Clip like scPTR
        for g in range(gamma_naive.shape[1]):
            col = gamma_naive[:, g]
            pos = col[col > 0]
            if len(pos) > 0:
                cap = np.percentile(pos, 99)
                gamma_naive[:, g] = np.clip(col, 0, cap)

        adata_naive = adata.copy()
        adata_naive.layers["gamma"] = gamma_naive.astype(np.float32)

        # ── scPTR gamma: beta * Mu/Ms ────────────────────────────────
        scptr.tl.estimate_beta(adata)
        scptr.tl.estimate_gamma(adata)

        # ── Compare half-life ─────────────────────────────────────────
        r_scptr_m, n_m = halflife_spearman(adata, hl_mouse)
        r_scptr_h, n_h = halflife_spearman(adata, hl_human)
        r_naive_m, n_nm = halflife_spearman(adata_naive, hl_mouse)
        r_naive_h, n_nh = halflife_spearman(adata_naive, hl_human)

        print(f"\n  {'Method':<25} {'Mouse r':>10} {'Human r':>10}")
        print("  " + "-" * 50)
        print(f"  {'Naive (Mu/Ms)':<25} {r_naive_m:>10.4f} {r_naive_h:>10.4f}")
        print(f"  {'scPTR (beta*Mu/Ms)':<25} {r_scptr_m:>10.4f} {r_scptr_h:>10.4f}")
        print(f"  {'Beta improvement':<25} {abs(r_scptr_m)-abs(r_naive_m):>10.4f} {abs(r_scptr_h)-abs(r_naive_h):>10.4f}")

        # ── Compare PT states ─────────────────────────────────────────
        from sklearn.metrics import silhouette_score
        from sklearn.decomposition import PCA

        labels = adata.obs[ck].astype("category").cat.codes.values

        pca_scptr = PCA(n_components=10).fit_transform(adata.layers["gamma"])
        pca_naive = PCA(n_components=10).fit_transform(gamma_naive)

        sil_scptr = silhouette_score(pca_scptr, labels, sample_size=min(2000, len(labels)))
        sil_naive = silhouette_score(pca_naive, labels, sample_size=min(2000, len(labels)))

        print(f"\n  Silhouette (cell-type in gamma PCA):")
        print(f"    Naive:  {sil_naive:.4f}")
        print(f"    scPTR:  {sil_scptr:.4f}")

        # ── Beta distribution ─────────────────────────────────────────
        beta = adata.var["beta"].values
        print(f"\n  Beta: median={np.median(beta):.4f}, CV={np.std(beta)/np.mean(beta):.4f}")
        print(f"  If CV≈0, beta is constant → no contribution")
        print(f"  Actual CV={np.std(beta)/np.mean(beta):.2f} → {'substantial' if np.std(beta)/np.mean(beta) > 0.5 else 'modest'} gene-specific effect")

        # ── Correlation between naive and scPTR gamma ─────────────────
        med_naive = np.median(gamma_naive, axis=0)
        med_scptr = np.median(adata.layers["gamma"], axis=0)
        valid = (med_naive > 0) & (med_scptr > 0)
        r_agree, _ = stats.spearmanr(med_naive[valid], med_scptr[valid])
        print(f"\n  Naive vs scPTR gamma agreement: r={r_agree:.4f}")

        all_results[ds_name] = {
            "naive_mouse": float(r_naive_m), "naive_human": float(r_naive_h),
            "scptr_mouse": float(r_scptr_m), "scptr_human": float(r_scptr_h),
            "beta_improvement_mouse": float(abs(r_scptr_m) - abs(r_naive_m)),
            "beta_improvement_human": float(abs(r_scptr_h) - abs(r_naive_h)),
            "sil_naive": float(sil_naive), "sil_scptr": float(sil_scptr),
            "beta_cv": float(np.std(beta) / np.mean(beta)),
            "naive_scptr_agreement": float(r_agree),
        }

    save_json(all_results, "beta_contribution", OUT)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, (ds, res) in enumerate(all_results.items()):
        x = i * 3
        axes[0].bar(x, abs(res["naive_human"]), 0.8, color="gray", alpha=0.7,
                     label="Naive (Mu/Ms)" if i == 0 else "")
        axes[0].bar(x + 1, abs(res["scptr_human"]), 0.8, color="darkorange", alpha=0.7,
                     label="scPTR (beta*Mu/Ms)" if i == 0 else "")
    axes[0].set_xticks([0.5, 3.5])
    axes[0].set_xticklabels(list(all_results.keys()))
    axes[0].set_ylabel("|r| with half-life (human)")
    axes[0].set_title("Beta contribution to half-life r")
    axes[0].legend()

    for i, (ds, res) in enumerate(all_results.items()):
        x = i * 3
        axes[1].bar(x, res["sil_naive"], 0.8, color="gray", alpha=0.7)
        axes[1].bar(x + 1, res["sil_scptr"], 0.8, color="darkorange", alpha=0.7)
    axes[1].set_xticks([0.5, 3.5])
    axes[1].set_xticklabels(list(all_results.keys()))
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Beta contribution to PT state quality")

    fig.suptitle("Does beta estimation improve gamma?", y=1.02)
    fig.tight_layout()
    save_fig(fig, "beta_contribution", OUT)


if __name__ == "__main__":
    main()
