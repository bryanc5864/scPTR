#!/usr/bin/env python
"""CRITICAL: scVelo gamma semantics and honest method comparison.

Findings:
1. scPTR gamma ≈ scVelo velocity_gamma (r=0.96) — nearly identical math
2. scVelo dynamical fit_gamma has DIFFERENT semantics (r=-0.37 with SS)
3. fit_gamma/fit_beta correlates with half-life (r=-0.35)
4. The "scVelo dynamical failure" was a comparison error, not a model failure

This script documents these findings honestly and provides corrected comparisons.
"""
from _common import *
import scvelo as scv

OUT = output_dir("28_gamma_semantics")


def main():
    set_figure_style()
    hl_mouse, hl_human = load_halflife_refs()

    all_results = {}

    for ds_name, loader, ck in DATASETS:
        print(f"\n{'=' * 60}\n{ds_name.upper()}: Gamma Semantics\n{'=' * 60}")

        adata_raw = loader()

        # ── scVelo SS ────────────────────────────────────────────────
        adata_sv = adata_raw.copy()
        scv.pp.filter_and_normalize(adata_sv, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(adata_sv, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(adata_sv, mode="steady_state")
        vg = adata_sv.var["velocity_gamma"].values.astype(float)

        # ── scVelo dynamical ─────────────────────────────────────────
        adata_dyn = adata_raw.copy()
        scv.pp.filter_and_normalize(adata_dyn, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(adata_dyn, n_pcs=30, n_neighbors=30)
        scv.tl.recover_dynamics(adata_dyn, n_jobs=4)
        scv.tl.velocity(adata_dyn, mode="dynamical")
        fg = adata_dyn.var["fit_gamma"].values.astype(float)
        fb = adata_dyn.var["fit_beta"].values.astype(float)

        # ── scPTR ────────────────────────────────────────────────────
        adata_sp = run_analytical(loader)
        sp_gamma = np.median(adata_sp.layers["gamma"], axis=0)
        sp_gs = pd.Series(sp_gamma, index=adata_sp.var_names)

        # ── Correlations between methods ─────────────────────────────
        print("\n--- Method-to-method gamma correlation ---")

        # scPTR vs scVelo SS
        sv_gs = pd.Series(vg, index=adata_sv.var_names)
        shared_ss = sp_gs.index.intersection(sv_gs.index)
        g1, g2 = sp_gs[shared_ss].values.astype(float), sv_gs[shared_ss].values.astype(float)
        v = np.isfinite(g1) & np.isfinite(g2) & (g1 > 0) & (g2 > 0)
        r_sp_sv, _ = stats.spearmanr(g1[v], g2[v])
        print(f"  scPTR vs scVelo SS:         r={r_sp_sv:.4f} (n={v.sum()})")

        # scVelo SS vs dynamical fit_gamma
        shared_dyn = adata_sv.var_names.intersection(adata_dyn.var_names)
        vg_sh = pd.Series(vg, index=adata_sv.var_names)[shared_dyn].values.astype(float)
        fg_sh = pd.Series(fg, index=adata_dyn.var_names)[shared_dyn].values.astype(float)
        v2 = np.isfinite(vg_sh) & np.isfinite(fg_sh) & (vg_sh > 0) & (fg_sh > 0)
        r_ss_dyn, _ = stats.spearmanr(vg_sh[v2], fg_sh[v2])
        print(f"  scVelo SS vs dyn fit_gamma: r={r_ss_dyn:.4f} (n={v2.sum()})")

        # ── Corrected half-life comparison ────────────────────────────
        print("\n--- Half-life correlation (corrected) ---")

        methods = {}

        # scPTR gamma
        r_sp, n_sp = halflife_spearman(adata_sp, hl_human)
        methods["scPTR gamma"] = (r_sp, n_sp)

        # scVelo SS velocity_gamma
        adata_sv_tmp = adata_sv.copy()
        adata_sv_tmp.layers["gamma"] = np.tile(vg, (adata_sv.n_obs, 1))
        r_ss, n_ss = halflife_spearman(adata_sv_tmp, hl_human)
        methods["scVelo SS velocity_gamma"] = (r_ss, n_ss)

        # scVelo dyn fit_gamma (raw — the "failed" metric)
        adata_dyn_tmp = adata_dyn.copy()
        adata_dyn_tmp.layers["gamma"] = np.tile(fg, (adata_dyn.n_obs, 1))
        r_dyn_raw, n_dyn_raw = halflife_spearman(adata_dyn_tmp, hl_human)
        methods["scVelo dyn fit_gamma (raw)"] = (r_dyn_raw, n_dyn_raw)

        # scVelo dyn fit_gamma/fit_beta (CORRECTED)
        ratio = fg / (fb + 1e-8)
        adata_dyn_tmp.layers["gamma"] = np.tile(ratio, (adata_dyn.n_obs, 1))
        r_dyn_corr, n_dyn_corr = halflife_spearman(adata_dyn_tmp, hl_human)
        methods["scVelo dyn fit_gamma/fit_beta"] = (r_dyn_corr, n_dyn_corr)

        for mname, (r, n) in methods.items():
            print(f"  {mname:<35} r={r:.4f} (n={n})")

        ds_results = {
            "scptr_vs_scvelo_ss": {"r": float(r_sp_sv), "n": int(v.sum())},
            "scvelo_ss_vs_dyn": {"r": float(r_ss_dyn), "n": int(v2.sum())},
            "halflife": {m: {"r": float(r), "n": n} for m, (r, n) in methods.items()},
        }

        # ── Honest assessment ─────────────────────────────────────────
        print(f"\n--- Honest assessment ---")
        print(f"  scPTR gamma ≈ scVelo SS gamma (r={r_sp_sv:.3f})")
        print(f"  scVelo dynamical fit_gamma has different semantics")
        print(f"  Corrected (fit_gamma/fit_beta): r={r_dyn_corr:.3f} — comparable to SS")
        print(f"  scPTR's methodological contribution over scVelo SS:")
        print(f"    1. Per-cell gamma (not just per-gene)")
        print(f"    2. Beta estimation + multiplication")
        print(f"    3. Two-stage clipping (per-gene + global)")
        print(f"    4. Downstream: PT states, PT velocity, networks")
        print(f"    5. DeepPTR: uncertainty + disentanglement")

        all_results[ds_name] = ds_results

        # Figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # Panel 1: scPTR vs scVelo SS scatter
        axes[0].scatter(g2[v], g1[v], alpha=0.1, s=3, c="steelblue")
        axes[0].set_xlabel("scVelo SS velocity_gamma")
        axes[0].set_ylabel("scPTR median gamma")
        axes[0].set_title(f"scPTR ≈ scVelo SS (r={r_sp_sv:.3f})")
        axes[0].set_xscale("log"); axes[0].set_yscale("log")

        # Panel 2: Half-life comparison bar
        mnames = list(methods.keys())
        rs = [abs(methods[m][0]) for m in mnames]
        colors = ["darkorange", "steelblue", "lightcoral", "seagreen"]
        axes[1].barh(mnames, rs, color=colors[:len(mnames)], alpha=0.7)
        axes[1].set_xlabel("|Spearman r| with half-life")
        axes[1].set_title(f"{ds_name}: Corrected comparison")

        # Panel 3: SS vs dyn scatter
        axes[2].scatter(vg_sh[v2], fg_sh[v2], alpha=0.1, s=3, c="gray")
        axes[2].set_xlabel("scVelo SS velocity_gamma")
        axes[2].set_ylabel("scVelo dyn fit_gamma")
        axes[2].set_title(f"SS vs dyn gamma (r={r_ss_dyn:.3f})")
        axes[2].set_xscale("log"); axes[2].set_yscale("log")

        fig.suptitle(f"{ds_name}: Gamma semantics", y=1.02)
        fig.tight_layout()
        save_fig(fig, f"{ds_name}_gamma_semantics", OUT)

    save_json(all_results, "gamma_semantics", OUT)


if __name__ == "__main__":
    main()
