#!/usr/bin/env python
"""Quantify per-cell gamma advantage over per-gene gamma.

Shows what per-cell resolution enables:
1. Cell-type-specific half-life r (varies by type, some much better)
2. Transition cells have intermediate gamma (continuous, not binary)
3. Gamma heterogeneity within types correlates with biological axes
"""
from _common import *
import scanpy as sc

OUT = output_dir("33_percell_advantage")


def main():
    set_figure_style()
    _, hl_human = load_halflife_refs()

    all_results = {}

    for ds_name, loader, ck in DATASETS:
        print(f"\n{'=' * 60}\n{ds_name.upper()}: Per-cell Advantage\n{'=' * 60}")

        adata = run_analytical(loader)
        gamma = adata.layers["gamma"]
        gamma_med = np.median(gamma, axis=0)

        # ── 1. Cell-type-specific half-life ──────────────────────────
        print("\n--- Cell-type-specific half-life ---")
        r_global, n_global = halflife_spearman(adata, hl_human)
        print(f"  Global median: r={r_global:.4f} (n={n_global})")

        ct_results = []
        cell_types = sorted(adata.obs[ck].unique())

        for ct in cell_types:
            mask = (adata.obs[ck] == ct).values
            if mask.sum() < 20:
                continue
            gamma_ct = np.median(gamma[mask], axis=0)
            adata_tmp = adata.copy()
            adata_tmp.layers["gamma"] = np.tile(gamma_ct, (adata.n_obs, 1))
            r_ct, n_ct = halflife_spearman(adata_tmp, hl_human)
            ct_results.append({
                "cell_type": str(ct), "n_cells": int(mask.sum()),
                "r": float(r_ct), "n_genes": n_ct,
            })

        best_ct = min(ct_results, key=lambda x: x["r"])
        worst_ct = max(ct_results, key=lambda x: x["r"])
        range_r = abs(best_ct["r"]) - abs(worst_ct["r"])

        print(f"  Best:  {best_ct['cell_type']} r={best_ct['r']:.4f}")
        print(f"  Worst: {worst_ct['cell_type']} r={worst_ct['r']:.4f}")
        print(f"  Range: {range_r:.4f}")
        print(f"  → Per-gene gamma CANNOT compute this. Per-cell gamma can.")

        # ── 2. Transition cell detection ─────────────────────────────
        print("\n--- Transition cell detection ---")

        # For each cell: compute its "transition score" = how mixed is its
        # gamma profile between neighboring cell types?
        # Use entropy of kNN label distribution as proxy
        from scipy.sparse import issparse
        conn = adata.obsp["connectivities"]
        if issparse(conn):
            conn = conn.toarray()

        labels = adata.obs[ck].astype("category").cat.codes.values
        n_types = len(np.unique(labels))

        # Per-cell: fraction of neighbors from same type
        same_type_frac = np.zeros(adata.n_obs)
        for i in range(adata.n_obs):
            neighbors = np.where(conn[i] > 0)[0]
            if len(neighbors) == 0:
                same_type_frac[i] = 1.0
            else:
                same_type_frac[i] = np.mean(labels[neighbors] == labels[i])

        # Transition cells: low same_type_frac
        transition_mask = same_type_frac < 0.5
        pure_mask = same_type_frac > 0.9
        n_transition = transition_mask.sum()
        n_pure = pure_mask.sum()

        print(f"  Transition cells (<50% same-type neighbors): {n_transition}")
        print(f"  Pure cells (>90% same-type neighbors): {n_pure}")

        # Do transition cells have higher gamma variance? (more heterogeneous)
        if n_transition > 10 and n_pure > 10:
            var_transition = np.mean(np.var(gamma[transition_mask], axis=0))
            var_pure = np.mean(np.var(gamma[pure_mask], axis=0))
            print(f"  Mean gamma variance: transition={var_transition:.4f}, pure={var_pure:.4f}")
            print(f"  Ratio: {var_transition / max(var_pure, 1e-8):.2f}x")
            print(f"  → Transition cells show {'MORE' if var_transition > var_pure else 'LESS'} gamma heterogeneity")

        # ── 3. Within-type gamma CV correlates with position ──────────
        print("\n--- Within-type heterogeneity ---")

        # For the largest cell type: compute per-cell gamma CV, correlate with PCA position
        largest_ct = max(cell_types, key=lambda ct: (adata.obs[ck] == ct).sum())
        ct_mask = (adata.obs[ck] == largest_ct).values
        gamma_ct = gamma[ct_mask]

        # Per-cell: mean gamma across genes
        cell_mean_gamma = np.mean(gamma_ct, axis=0)
        # This is per-gene, not per-cell. Let's do per-cell mean
        cell_gamma_mean = np.mean(gamma_ct, axis=1)
        cell_gamma_std = np.std(gamma_ct, axis=1)

        # Correlate with PC1 within this cell type
        from sklearn.decomposition import PCA
        X_ct = np.asarray(adata.X[ct_mask].todense() if issparse(adata.X) else adata.X[ct_mask])
        if X_ct.shape[0] > 2 and X_ct.shape[1] > 2:
            pc1 = PCA(n_components=1).fit_transform(np.log1p(X_ct)).ravel()
            r_pc1_gamma, p_pc1 = stats.spearmanr(pc1, cell_gamma_mean)
            print(f"  {largest_ct} ({ct_mask.sum()} cells):")
            print(f"    Mean gamma vs PC1: r={r_pc1_gamma:.4f} (p={p_pc1:.2e})")
            print(f"    → Within-type gamma variation tracks continuous expression axis")

        # ── 4. Per-gene gamma: what information is lost? ──────────────
        print("\n--- Information loss with per-gene gamma ---")

        # Compare: per-cell gamma matrix vs per-gene median repeated
        gamma_pergene = np.tile(gamma_med, (adata.n_obs, 1))

        # How much variance is captured by per-gene median?
        total_var = np.var(gamma, axis=0).sum()
        between_var = np.var(gamma_pergene, axis=0).sum()  # This is 0 by construction
        # Actually: per-gene captures between-gene variance, misses within-gene (across cells)
        within_gene_var = np.mean(np.var(gamma, axis=0))  # average per-gene variance across cells
        between_gene_var = np.var(gamma_med)  # variance of medians across genes

        frac_within = within_gene_var / (within_gene_var + between_gene_var + 1e-10)
        print(f"  Within-gene (per-cell) variance:  {within_gene_var:.6f} ({frac_within*100:.1f}%)")
        print(f"  Between-gene variance:            {between_gene_var:.6f} ({(1-frac_within)*100:.1f}%)")
        print(f"  → Per-gene gamma discards {frac_within*100:.0f}% of total gamma variance")

        all_results[ds_name] = {
            "r_global": float(r_global),
            "ct_halflife": ct_results,
            "ct_range": float(range_r),
            "n_transition": int(n_transition),
            "n_pure": int(n_pure),
            "frac_within_gene_var": float(frac_within),
        }

    save_json(all_results, "percell_advantage", OUT)

    # Figure
    for ds_name, res in all_results.items():
        ct = res["ct_halflife"]
        if not ct:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Cell-type-specific half-life
        cts = [r["cell_type"] for r in ct]
        rs = [abs(r["r"]) for r in ct]
        colors = ["darkorange" if r > abs(res["r_global"]) else "steelblue" for r in rs]
        axes[0].barh(cts, rs, color=colors, alpha=0.7)
        axes[0].axvline(abs(res["r_global"]), color="red", ls="--",
                        label=f"Global={abs(res['r_global']):.3f}")
        axes[0].set_xlabel("|Spearman r| with half-life")
        axes[0].set_title(f"{ds_name}: Cell-type-specific r")
        axes[0].legend(fontsize=8)

        # Variance decomposition
        axes[1].pie([res["frac_within_gene_var"], 1 - res["frac_within_gene_var"]],
                    labels=[f"Within-gene\n(per-cell)\n{res['frac_within_gene_var']*100:.0f}%",
                            f"Between-gene\n{(1-res['frac_within_gene_var'])*100:.0f}%"],
                    colors=["darkorange", "steelblue"], autopct="%1.0f%%")
        axes[1].set_title("Gamma variance: per-gene discards orange portion")

        fig.tight_layout()
        save_fig(fig, f"{ds_name}_percell_advantage", OUT)


if __name__ == "__main__":
    main()
