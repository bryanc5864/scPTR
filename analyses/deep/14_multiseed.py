#!/usr/bin/env python
"""Multi-seed stability: how reproducible are DeepPTR results?"""
from _common import *
from sklearn.metrics import silhouette_score

OUT = output_dir("14_multiseed")
N_SEEDS = 5


def main():
    set_figure_style()

    print("Loading and preprocessing pancreas...")
    adata_raw = scptr.datasets.pancreas()
    scptr.pp.filter_genes(adata_raw)
    scptr.pp.normalize_layers(adata_raw)
    scptr.pp.neighbors(adata_raw, n_neighbors=30)
    scptr.pp.smooth_layers(adata_raw)
    scptr.tl.estimate_beta(adata_raw)
    adata_base = select_top_genes(adata_raw, n_top=300)

    _, hl_human = load_halflife_refs()

    records = []
    gamma_meds = []

    for seed in range(N_SEEDS):
        print(f"\n  Seed {seed}...")
        adata_v = adata_base.copy()
        from scipy.sparse import issparse
        for key in ("spliced", "unspliced"):
            if key in adata_v.layers and issparse(adata_v.layers[key]):
                adata_v.layers[key] = np.asarray(adata_v.layers[key].todense())

        hp = dict(DEEP_HP)
        hp["seed"] = seed
        torch.set_num_threads(4)
        model, history = scptr.deep.fit_deepptr(adata_v, verbose=False, **hp)

        gamma_med = np.median(adata_v.layers["gamma"], axis=0)
        gamma_meds.append(gamma_med)

        r, n = halflife_spearman(adata_v, hl_human)

        sil = np.nan
        if "clusters" in adata_v.obs.columns and "X_z_T" in adata_v.obsm:
            labels = adata_v.obs["clusters"].astype("category").cat.codes.values
            sil = silhouette_score(adata_v.obsm["X_z_T"], labels, sample_size=min(2000, len(labels)))

        records.append({
            "seed": seed, "halflife_r": float(r), "n_genes": n,
            "silhouette_zT": float(sil),
            "n_epochs": len(history.train_loss),
            "final_val_loss": history.val_loss[-1],
        })
        print(f"    HL r={r:.4f}, sil={sil:.4f}, epochs={len(history.train_loss)}")

    # Cross-seed gamma agreement
    cross_rs = []
    for i in range(N_SEEDS):
        for j in range(i + 1, N_SEEDS):
            r, _ = stats.spearmanr(gamma_meds[i], gamma_meds[j])
            cross_rs.append(float(r))

    # Gene ranking overlap
    top50_sets = [set(np.argsort(gm)[::-1][:50]) for gm in gamma_meds]
    overlaps = []
    for i in range(N_SEEDS):
        for j in range(i + 1, N_SEEDS):
            overlaps.append(len(top50_sets[i] & top50_sets[j]))

    # Summary
    hl_rs = [r["halflife_r"] for r in records]
    sils = [r["silhouette_zT"] for r in records]

    print(f"\n{'=' * 60}")
    print("MULTI-SEED SUMMARY (pancreas, N=5)")
    print("=" * 60)
    print(f"  Half-life r:     {np.mean(hl_rs):.4f} ± {np.std(hl_rs):.4f}")
    print(f"  Silhouette z_T:  {np.mean(sils):.4f} ± {np.std(sils):.4f}")
    print(f"  Cross-seed γ r:  {np.mean(cross_rs):.4f} ± {np.std(cross_rs):.4f}")
    print(f"  Top-50 overlap:  {np.mean(overlaps):.1f} ± {np.std(overlaps):.1f} / 50")

    results = {
        "per_seed": records,
        "halflife_mean": float(np.mean(hl_rs)),
        "halflife_std": float(np.std(hl_rs)),
        "silhouette_mean": float(np.mean(sils)),
        "silhouette_std": float(np.std(sils)),
        "cross_seed_gamma_r_mean": float(np.mean(cross_rs)),
        "cross_seed_gamma_r_std": float(np.std(cross_rs)),
        "top50_overlap_mean": float(np.mean(overlaps)),
        "top50_overlap_std": float(np.std(overlaps)),
    }
    save_json(results, "multiseed", OUT)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(range(N_SEEDS), [abs(r) for r in hl_rs], color="steelblue", alpha=0.7)
    axes[0].axhline(np.mean([abs(r) for r in hl_rs]), color="red", ls="--")
    axes[0].set_xlabel("Seed"); axes[0].set_ylabel("|r| with half-life")
    axes[0].set_title(f"Half-life r: {np.mean(hl_rs):.4f}±{np.std(hl_rs):.4f}")

    axes[1].bar(range(N_SEEDS), sils, color="darkorange", alpha=0.7)
    axes[1].axhline(np.mean(sils), color="red", ls="--")
    axes[1].set_xlabel("Seed"); axes[1].set_ylabel("Silhouette")
    axes[1].set_title(f"Silhouette: {np.mean(sils):.4f}±{np.std(sils):.4f}")

    axes[2].hist(cross_rs, bins=8, color="seagreen", alpha=0.7)
    axes[2].set_xlabel("Cross-seed gamma r"); axes[2].set_ylabel("Count")
    axes[2].set_title(f"Gamma agreement: {np.mean(cross_rs):.4f}±{np.std(cross_rs):.4f}")

    fig.suptitle("Multi-seed stability (N=5)", y=1.02)
    fig.tight_layout()
    save_fig(fig, "multiseed", OUT)


if __name__ == "__main__":
    main()
