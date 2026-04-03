#!/usr/bin/env python
"""CRITICAL: Do scVelo gamma-based clusters find the same "invisible states"?

If clustering scVelo's velocity_gamma gives the same invisible states
as scPTR, then scPTR's contribution is framing, not methodology.
"""
from _common import *
import scvelo as scv
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

OUT = output_dir("29_pt_states_comparison")


def cluster_gamma(gamma_matrix, adata, resolution=1.0, key_suffix=""):
    """PCA + Leiden clustering on a gamma matrix."""
    import anndata as ad

    adata_g = ad.AnnData(X=gamma_matrix, obs=adata.obs.copy())
    sc.pp.pca(adata_g, n_comps=min(30, gamma_matrix.shape[1] - 1))
    sc.pp.neighbors(adata_g, n_pcs=min(20, gamma_matrix.shape[1] - 1))
    sc.tl.leiden(adata_g, resolution=resolution, key_added=f"gamma_cluster{key_suffix}")
    return adata_g.obs[f"gamma_cluster{key_suffix}"].values


def check_invisible(gamma_clusters, expr_clusters):
    """Find clusters that are 'invisible' in expression (mixed expression types)."""
    ct = pd.crosstab(gamma_clusters, expr_clusters, normalize="index")
    # A gamma cluster is "invisible" if its dominant expression type < 60%
    invisible = []
    for gc in ct.index:
        max_frac = ct.loc[gc].max()
        if max_frac < 0.6:
            invisible.append(str(gc))
    return invisible


def main():
    set_figure_style()

    all_results = {}

    for ds_name, loader, ck in DATASETS:
        print(f"\n{'=' * 60}\n{ds_name.upper()}: PT States Comparison\n{'=' * 60}")

        adata_raw = loader()

        # ── scPTR gamma clustering ───────────────────────────────────
        adata_sp = run_analytical(loader)
        gamma_sp = adata_sp.layers["gamma"]
        clust_sp = cluster_gamma(gamma_sp, adata_sp, key_suffix="_scptr")
        expr_labels = adata_sp.obs[ck].values

        n_sp = len(np.unique(clust_sp))
        invis_sp = check_invisible(clust_sp, expr_labels)
        print(f"  scPTR: {n_sp} PT clusters, {len(invis_sp)} invisible")

        # ── scVelo SS gamma clustering ────────────────────────────────
        adata_sv = adata_raw.copy()
        scv.pp.filter_and_normalize(adata_sv, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(adata_sv, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(adata_sv, mode="steady_state")

        # Build per-cell gamma from scVelo: gamma_ig = velocity_gamma_g (broadcast)
        vg = adata_sv.var["velocity_gamma"].values.astype(float)
        # scVelo doesn't have per-cell gamma, so use Ms/Mu ratio approach
        Ms = np.asarray(adata_sv.layers["Ms"])
        Mu = np.asarray(adata_sv.layers["Mu"])
        gamma_sv = np.where(Ms > 0.01, Mu / Ms, 0) * vg[np.newaxis, :]

        # Match genes with scPTR
        shared = adata_sp.var_names.intersection(adata_sv.var_names)
        sp_idx = [list(adata_sp.var_names).index(g) for g in shared]
        sv_idx = [list(adata_sv.var_names).index(g) for g in shared]

        gamma_sv_shared = gamma_sv[:, sv_idx]

        # Need matching cells — use same raw data cells
        # scVelo may have filtered cells, so use scVelo's cell set
        clust_sv = cluster_gamma(gamma_sv_shared, adata_sv, key_suffix="_scvelo")
        expr_sv = adata_sv.obs[ck].values

        n_sv = len(np.unique(clust_sv))
        invis_sv = check_invisible(clust_sv, expr_sv)
        print(f"  scVelo SS: {n_sv} PT clusters, {len(invis_sv)} invisible")

        # ── Compare clusters ──────────────────────────────────────────
        # ARI between scPTR and scVelo gamma clusters (on shared cells)
        # Need to align cells
        shared_cells = adata_sp.obs_names.intersection(adata_sv.obs_names)
        if len(shared_cells) > 100:
            sp_mask = adata_sp.obs_names.isin(shared_cells)
            sv_mask = adata_sv.obs_names.isin(shared_cells)

            # Recluster on shared cells
            gamma_sp_shared = adata_sp.layers["gamma"][sp_mask][:, sp_idx]
            gamma_sv_for_compare = gamma_sv_shared[sv_mask]

            clust_sp_sh = cluster_gamma(gamma_sp_shared,
                                         adata_sp[sp_mask], key_suffix="_sp_sh")
            clust_sv_sh = cluster_gamma(gamma_sv_for_compare,
                                         adata_sv[sv_mask], key_suffix="_sv_sh")

            ari = adjusted_rand_score(clust_sp_sh, clust_sv_sh)
            nmi = normalized_mutual_info_score(clust_sp_sh, clust_sv_sh)

            # ARI with expression clusters
            expr_sp_sh = adata_sp.obs[ck].values[sp_mask]
            ari_sp_expr = adjusted_rand_score(clust_sp_sh, expr_sp_sh)
            ari_sv_expr = adjusted_rand_score(clust_sv_sh, adata_sv.obs[ck].values[sv_mask])

            print(f"\n  scPTR vs scVelo gamma clusters: ARI={ari:.4f}, NMI={nmi:.4f}")
            print(f"  scPTR gamma vs expression: ARI={ari_sp_expr:.4f}")
            print(f"  scVelo gamma vs expression: ARI={ari_sv_expr:.4f}")
        else:
            ari = nmi = ari_sp_expr = ari_sv_expr = np.nan

        # ── Which invisible states replicate? ─────────────────────────
        print(f"\n  Invisible states:")
        print(f"    scPTR:  {invis_sp if invis_sp else 'none'}")
        print(f"    scVelo: {invis_sv if invis_sv else 'none'}")

        print(f"\n  CONCLUSION: {'SAME structure' if ari > 0.5 else 'DIFFERENT structure' if ari < 0.2 else 'PARTIALLY overlapping'}")

        all_results[ds_name] = {
            "scptr_n_clusters": n_sp,
            "scvelo_n_clusters": n_sv,
            "scptr_invisible": invis_sp,
            "scvelo_invisible": invis_sv,
            "ari_scptr_vs_scvelo": float(ari) if np.isfinite(ari) else None,
            "nmi_scptr_vs_scvelo": float(nmi) if np.isfinite(nmi) else None,
            "ari_scptr_vs_expr": float(ari_sp_expr) if np.isfinite(ari_sp_expr) else None,
            "ari_scvelo_vs_expr": float(ari_sv_expr) if np.isfinite(ari_sv_expr) else None,
        }

    save_json(all_results, "pt_states_comparison", OUT)


if __name__ == "__main__":
    main()
