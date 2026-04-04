#!/usr/bin/env python
"""Corrected method comparison: all methods on equal footing.

Includes corrected scVelo dynamical (fit_gamma/fit_beta) and
per-cell-type evaluation as a unique scPTR metric.
"""
from _common import *
import scvelo as scv

OUT = output_dir("35_corrected_comparison")


def main():
    set_figure_style()
    hl_mouse, hl_human = load_halflife_refs()

    all_results = {}

    for ds_name, loader, ck in DATASETS:
        print(f"\n{'=' * 60}\n{ds_name.upper()}: Corrected Comparison\n{'=' * 60}")

        adata_raw = loader()

        # ── scVelo SS ────────────────────────────────────────────────
        adata_sv = adata_raw.copy()
        scv.pp.filter_and_normalize(adata_sv, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(adata_sv, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(adata_sv, mode="steady_state")
        vg = adata_sv.var["velocity_gamma"].values.astype(float)

        # ── scVelo dynamical (CORRECTED) ─────────────────────────────
        adata_dyn = adata_raw.copy()
        scv.pp.filter_and_normalize(adata_dyn, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(adata_dyn, n_pcs=30, n_neighbors=30)
        scv.tl.recover_dynamics(adata_dyn, n_jobs=4)
        scv.tl.velocity(adata_dyn, mode="dynamical")
        fg = adata_dyn.var["fit_gamma"].values.astype(float)
        fb = adata_dyn.var["fit_beta"].values.astype(float)
        ratio = fg / (fb + 1e-8)  # CORRECTED: use ratio

        # ── scPTR ────────────────────────────────────────────────────
        adata_sp = run_analytical(loader)

        # ── Evaluate ─────────────────────────────────────────────────
        def eval_hl(gamma_vals, var_names, label):
            hl_s = hl_human.set_index("gene_symbol")["half_life_hours"]
            g_upper = {g.upper(): i for i, g in enumerate(var_names)}
            h_upper = {g.upper(): g for g in hl_s.index if isinstance(g, str)}
            shared = set(g_upper.keys()) & set(h_upper.keys())
            g = np.array([gamma_vals[g_upper[u]] for u in shared], dtype=float)
            h = np.array([hl_s[h_upper[u]] for u in shared], dtype=float)
            v = np.isfinite(g) & np.isfinite(h) & (g > 0) & (h > 0)
            if v.sum() < 3: return np.nan, 0
            r, _ = stats.spearmanr(g[v], h[v])
            return float(r), int(v.sum())

        results = {}

        # Global half-life
        r_ss, n_ss = eval_hl(vg, adata_sv.var_names, "scVelo SS")
        r_dyn_raw, n_dr = eval_hl(fg, adata_dyn.var_names, "scVelo dyn (raw)")
        r_dyn_corr, n_dc = eval_hl(ratio, adata_dyn.var_names, "scVelo dyn (corrected)")
        r_sp, n_sp = halflife_spearman(adata_sp, hl_human)

        print(f"\n  {'Method':<35} {'HL human r':>12} {'n':>6}")
        print("  " + "-" * 55)
        print(f"  {'scVelo SS':<35} {r_ss:>12.4f} {n_ss:>6}")
        print(f"  {'scVelo dyn (fit_gamma, RAW)':<35} {r_dyn_raw:>12.4f} {n_dr:>6}")
        print(f"  {'scVelo dyn (γ/β, CORRECTED)':<35} {r_dyn_corr:>12.4f} {n_dc:>6}")
        print(f"  {'scPTR analytical':<35} {r_sp:>12.4f} {n_sp:>6}")

        results["global_halflife"] = {
            "scvelo_ss": {"r": r_ss, "n": n_ss},
            "scvelo_dyn_raw": {"r": r_dyn_raw, "n": n_dr},
            "scvelo_dyn_corrected": {"r": r_dyn_corr, "n": n_dc},
            "scptr": {"r": r_sp, "n": n_sp},
        }

        # ── Per-cell-type half-life (UNIQUE TO scPTR) ─────────────────
        print(f"\n  Per-cell-type half-life (scPTR-unique capability):")
        if ck in adata_sp.obs.columns:
            ct_rs = []
            for ct in sorted(adata_sp.obs[ck].unique()):
                mask = (adata_sp.obs[ck] == ct).values
                if mask.sum() < 20: continue
                gamma_ct = np.median(adata_sp.layers["gamma"][mask], axis=0)
                adata_tmp = adata_sp.copy()
                adata_tmp.layers["gamma"] = np.tile(gamma_ct, (adata_sp.n_obs, 1))
                r_ct, _ = halflife_spearman(adata_tmp, hl_human)
                ct_rs.append({"cell_type": str(ct), "r": float(r_ct)})

            best = min(ct_rs, key=lambda x: x["r"])
            print(f"    Best cell type: {best['cell_type']} (r={best['r']:.4f})")
            print(f"    vs global: r={r_sp:.4f}")
            print(f"    → Cell-type resolution improves r by {abs(best['r'])-abs(r_sp):.4f}")
            results["best_celltype"] = best

        all_results[ds_name] = results

    save_json(all_results, "corrected_comparison", OUT)

    # Corrected summary table
    print(f"\n{'=' * 70}")
    print("CORRECTED METHOD COMPARISON (FINAL)")
    print("=" * 70)
    print(f"\n{'Method':<35} ", end="")
    for ds in all_results:
        print(f"{'|':>2} {ds:>15}", end="")
    print()
    print("-" * 70)

    for method in ["scvelo_ss", "scvelo_dyn_raw", "scvelo_dyn_corrected", "scptr"]:
        label = {"scvelo_ss": "scVelo SS", "scvelo_dyn_raw": "scVelo dyn (raw γ)",
                 "scvelo_dyn_corrected": "scVelo dyn (γ/β)", "scptr": "scPTR"}[method]
        print(f"  {label:<33} ", end="")
        for ds in all_results:
            r = all_results[ds]["global_halflife"][method]["r"]
            print(f"{'|':>2} {r:>15.4f}", end="")
        print()

    # Figure
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = ["scVelo SS", "scVelo dyn\n(raw γ)", "scVelo dyn\n(γ/β corrected)", "scPTR"]
    method_keys = ["scvelo_ss", "scvelo_dyn_raw", "scvelo_dyn_corrected", "scptr"]
    colors = ["#1f77b4", "#ff9999", "#2ca02c", "#ff7f0e"]

    x = np.arange(len(methods))
    width = 0.35
    for i, ds in enumerate(all_results):
        rs = [abs(all_results[ds]["global_halflife"][mk]["r"]) for mk in method_keys]
        offset = (i - 0.5) * width
        ax.bar(x + offset, rs, width, label=ds, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("|Spearman r| with half-life (human)")
    ax.set_title("Corrected Method Comparison")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "corrected_comparison", OUT)


if __name__ == "__main__":
    main()
