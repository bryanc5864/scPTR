#!/usr/bin/env python
"""Gene-gene gamma correlation structure.

Do genes with correlated degradation rates share RBP regulators?
Identifies co-degradation modules and tests for shared eCLIP RBP targets.
"""
from _common import *
from sklearn.cluster import AgglomerativeClustering

OUT = output_dir("09_gamma_coexpression")


def run(name, loader, cluster_key):
    print(f"\n{'=' * 60}\n{name.upper()}: Gamma co-regulation\n{'=' * 60}")

    adata_an = run_analytical(loader)
    gamma = adata_an.layers["gamma"]

    # Per-gene median gamma
    gamma_med = np.median(gamma, axis=0)
    active = gamma_med > 0.01
    gamma_active = gamma[:, active]
    gene_names = adata_an.var_names[active].tolist()
    print(f"  Active genes (median gamma > 0.01): {len(gene_names)}")

    # Gene-gene correlation matrix (Spearman on per-cell gamma)
    # Subsample cells for speed
    rng = np.random.RandomState(42)
    n_sub = min(1000, gamma_active.shape[0])
    idx = rng.choice(gamma_active.shape[0], n_sub, replace=False)
    gamma_sub = gamma_active[idx]

    # Correlation matrix
    n_genes = gamma_sub.shape[1]
    if n_genes > 2000:
        # Too many — take top 500 by variance
        var = gamma_sub.var(axis=0)
        top = np.argsort(var)[::-1][:500]
        gamma_sub = gamma_sub[:, top]
        gene_names = [gene_names[i] for i in top]
        n_genes = 500

    print(f"  Computing {n_genes}x{n_genes} correlation matrix...")
    corr = np.corrcoef(gamma_sub.T)
    corr = np.nan_to_num(corr)

    # Cluster genes into co-degradation modules
    n_clusters = min(10, max(2, n_genes // 50))
    clust = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed",
                                     linkage="average")
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    labels = clust.fit_predict(dist)

    module_sizes = pd.Series(labels).value_counts().sort_index()
    print(f"  Found {n_clusters} co-degradation modules: {module_sizes.to_dict()}")

    # For each module: check eCLIP RBP enrichment
    eclip = pd.read_csv(DATA_DIR / "eclip_targets.csv")
    eclip_by_rbp = eclip.groupby("rbp")["target_gene"].apply(lambda x: set(x.str.upper())).to_dict()

    module_results = []
    for mod_id in range(n_clusters):
        mod_genes = [gene_names[i] for i in range(len(gene_names)) if labels[i] == mod_id]
        mod_upper = set(g.upper() for g in mod_genes)

        # Which RBPs target this module?
        rbp_counts = {}
        for rbp, targets in eclip_by_rbp.items():
            n_hit = len(mod_upper & targets)
            if n_hit > 0:
                rbp_counts[rbp] = n_hit

        top_rbp = sorted(rbp_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        # Mean within-module correlation
        mod_idx = [i for i in range(len(gene_names)) if labels[i] == mod_id]
        if len(mod_idx) > 1:
            sub_corr = corr[np.ix_(mod_idx, mod_idx)]
            mean_corr = (sub_corr.sum() - len(mod_idx)) / (len(mod_idx) * (len(mod_idx) - 1))
        else:
            mean_corr = 1.0

        module_results.append({
            "module": mod_id,
            "n_genes": len(mod_genes),
            "mean_within_corr": float(mean_corr),
            "top_rbps": [(r, c) for r, c in top_rbp],
            "example_genes": mod_genes[:10],
        })

        if top_rbp:
            rbp_str = ", ".join(f"{r}({c})" for r, c in top_rbp)
            print(f"  Module {mod_id}: {len(mod_genes)} genes, r={mean_corr:.3f}, RBPs: {rbp_str}")

    # Mean between-module correlation (should be lower)
    between_corrs = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            idx_i = [k for k in range(len(gene_names)) if labels[k] == i]
            idx_j = [k for k in range(len(gene_names)) if labels[k] == j]
            if idx_i and idx_j:
                between = corr[np.ix_(idx_i, idx_j)]
                between_corrs.append(between.mean())

    mean_within = np.mean([m["mean_within_corr"] for m in module_results])
    mean_between = np.mean(between_corrs) if between_corrs else 0

    print(f"\n  Mean within-module correlation:  {mean_within:.4f}")
    print(f"  Mean between-module correlation: {mean_between:.4f}")
    print(f"  Ratio: {mean_within / max(abs(mean_between), 1e-8):.2f}x")

    results = {
        "n_active_genes": len(gene_names),
        "n_modules": n_clusters,
        "mean_within_corr": float(mean_within),
        "mean_between_corr": float(mean_between),
        "modules": module_results,
    }
    save_json(results, f"{name}_gamma_coexpression", OUT)

    # Figure: heatmap of correlation matrix (clustered)
    order = np.argsort(labels)
    corr_ordered = corr[np.ix_(order, order)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(corr_ordered, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title(f"{name}: Gamma correlation ({n_genes} genes)")

    # Module size bar
    axes[1].bar(range(n_clusters),
                [m["mean_within_corr"] for m in module_results],
                color="steelblue", alpha=0.7)
    axes[1].axhline(mean_between, color="red", ls="--", label=f"Between={mean_between:.3f}")
    axes[1].set_xlabel("Module")
    axes[1].set_ylabel("Mean within-module correlation")
    axes[1].set_title("Co-degradation module structure")
    axes[1].legend()

    fig.tight_layout()
    save_fig(fig, f"{name}_gamma_corr", OUT)

    return results


def main():
    set_figure_style()
    for name, loader, ck in DATASETS:
        run(name, loader, ck)


if __name__ == "__main__":
    main()
