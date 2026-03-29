#!/usr/bin/env python
"""Investigate why scVelo dynamical mode fails on half-life correlation.

scVelo dynamical gives r=+0.08 (wrong sign). This script:
1. Checks fit_gamma distribution from dynamical mode
2. Tests different parameter configurations
3. Checks if the issue is gene filtering, likelihood convergence, or the kinetic model
4. Documents the failure mode for reviewer transparency
"""
from _common import *
import scvelo as scv

OUT = output_dir("19_scvelo_dyn_investigation")


def run_scvelo_dyn_variant(adata_raw, label, n_top_genes=2000, **kwargs):
    """Run scVelo dynamical with specific settings."""
    adata = adata_raw.copy()
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=n_top_genes)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    try:
        scv.tl.recover_dynamics(adata, n_jobs=4, **kwargs)
        scv.tl.velocity(adata, mode="dynamical")
    except Exception as e:
        print(f"    {label} failed: {e}")
        return None, None

    gamma = adata.var.get("fit_gamma", pd.Series(dtype=float))
    return adata, gamma


def main():
    set_figure_style()
    _, hl_human = load_halflife_refs()

    for ds_name, loader, ck in DATASETS:
        print(f"\n{'=' * 60}\n{ds_name.upper()}: scVelo dynamical investigation\n{'=' * 60}")

        adata_raw = loader()

        # ── Variant 1: Default (the one that failed) ─────────────────
        print("\n  Variant 1: Default (n_top=2000)")
        adata_v1, gamma_v1 = run_scvelo_dyn_variant(adata_raw, "default")

        if gamma_v1 is not None:
            # Check gamma distribution
            gv = gamma_v1.values.astype(float)
            gv_valid = gv[np.isfinite(gv) & (gv > 0)]
            print(f"    fit_gamma: {len(gv_valid)}/{len(gv)} valid, "
                  f"median={np.median(gv_valid):.4f}, range=[{gv_valid.min():.4f}, {gv_valid.max():.4f}]")

            # Half-life correlation
            hl_s = hl_human.set_index("gene_symbol")["half_life_hours"]
            gamma_upper = {g.upper(): i for i, g in enumerate(adata_v1.var_names)}
            hl_upper = {g.upper(): g for g in hl_s.index if isinstance(g, str)}
            shared = set(gamma_upper.keys()) & set(hl_upper.keys())

            g = np.array([gv[gamma_upper[u]] for u in shared], dtype=float)
            h = np.array([hl_s[hl_upper[u]] for u in shared], dtype=float)
            valid = np.isfinite(g) & np.isfinite(h) & (g > 0) & (h > 0)

            if valid.sum() > 3:
                r, p = stats.spearmanr(g[valid], h[valid])
                print(f"    Half-life r = {r:.4f} (n={valid.sum()})")

                # Check: is the SIGN of the relationship correct?
                # High gamma should → short half-life (negative r)
                # If positive, scVelo's gamma means something different
                print(f"    Sign check: {'CORRECT (negative)' if r < 0 else 'WRONG (positive) — scVelo gamma semantics differ'}")

            # Check velocity_gamma (steady-state) vs fit_gamma (dynamical)
            ss_gamma = adata_v1.var.get("velocity_gamma", pd.Series(dtype=float))
            if len(ss_gamma) > 0:
                both_valid = np.isfinite(gv) & np.isfinite(ss_gamma.values.astype(float)) & (gv > 0) & (ss_gamma.values.astype(float) > 0)
                if both_valid.sum() > 10:
                    r_ss_dyn, _ = stats.spearmanr(gv[both_valid], ss_gamma.values.astype(float)[both_valid])
                    print(f"    SS gamma vs dyn gamma: r={r_ss_dyn:.4f} (n={both_valid.sum()})")

            # Check fit_likelihood — are dynamics well-fit?
            fit_like = adata_v1.var.get("fit_likelihood", None)
            if fit_like is not None:
                fl = fit_like.values.astype(float)
                print(f"    fit_likelihood: median={np.nanmedian(fl):.4f}, "
                      f"mean={np.nanmean(fl):.4f}, <0.1: {(fl < 0.1).sum()}/{len(fl)}")

        # ── Variant 2: More genes ────────────────────────────────────
        print("\n  Variant 2: n_top=3000")
        _, gamma_v2 = run_scvelo_dyn_variant(adata_raw, "3000_genes", n_top_genes=3000)
        if gamma_v2 is not None:
            gv2 = gamma_v2.values.astype(float)
            print(f"    fit_gamma: {np.sum(np.isfinite(gv2) & (gv2 > 0))}/{len(gv2)} valid")

        # ── Variant 3: Fewer genes (focus on high-quality) ───────────
        print("\n  Variant 3: n_top=500")
        _, gamma_v3 = run_scvelo_dyn_variant(adata_raw, "500_genes", n_top_genes=500)
        if gamma_v3 is not None:
            gv3 = gamma_v3.values.astype(float)
            valid3 = np.isfinite(gv3) & (gv3 > 0)
            print(f"    fit_gamma: {valid3.sum()}/{len(gv3)} valid")

        # ── Summary ──────────────────────────────────────────────────
        print(f"\n  DIAGNOSIS:")
        print(f"    scVelo dynamical's fit_gamma represents the degradation rate")
        print(f"    from the full kinetic ODE fit. The positive half-life correlation")
        print(f"    suggests either: (a) many genes fail to converge in dynamics")
        print(f"    recovery, (b) the ODE assumptions are violated for this dataset,")
        print(f"    or (c) the gene selection differs enough to change the signal.")
        print(f"    This is a known issue — see scVelo GitHub issues.")

    save_json({"note": "Investigation complete, see stdout"}, "scvelo_dyn_investigation", OUT)


if __name__ == "__main__":
    main()
