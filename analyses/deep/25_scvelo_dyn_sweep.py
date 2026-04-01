#!/usr/bin/env python
"""scVelo dynamical parameter sweep: is the failure robust?

Tests multiple configurations to show the positive half-life correlation
is not a misconfiguration artifact.
"""
from _common import *
import scvelo as scv

OUT = output_dir("25_scvelo_dyn_sweep")


def run_config(adata_raw, n_top, label):
    """Run scVelo dynamical with given n_top_genes."""
    adata = adata_raw.copy()
    try:
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=n_top)
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        scv.tl.recover_dynamics(adata, n_jobs=4)
        scv.tl.velocity(adata, mode="dynamical")

        gamma = adata.var.get("fit_gamma", pd.Series(dtype=float))
        fit_like = adata.var.get("fit_likelihood", pd.Series(dtype=float))
        return adata, gamma, fit_like
    except Exception as e:
        print(f"    {label} failed: {e}")
        return None, None, None


def eval_halflife(gamma_series, var_names, hl_df):
    """Evaluate half-life correlation."""
    if gamma_series is None:
        return np.nan, 0
    hl_s = hl_df.set_index("gene_symbol")["half_life_hours"]
    g_upper = {g.upper(): i for i, g in enumerate(var_names)}
    h_upper = {g.upper(): g for g in hl_s.index if isinstance(g, str)}
    shared = set(g_upper.keys()) & set(h_upper.keys())

    gv = gamma_series.values.astype(float)
    g = np.array([gv[g_upper[u]] for u in shared], dtype=float)
    h = np.array([hl_s[h_upper[u]] for u in shared], dtype=float)
    v = np.isfinite(g) & np.isfinite(h) & (g > 0) & (h > 0)
    if v.sum() < 3:
        return np.nan, 0
    r, _ = stats.spearmanr(g[v], h[v])
    return float(r), int(v.sum())


def main():
    set_figure_style()
    _, hl_human = load_halflife_refs()
    hl_mouse, _ = load_halflife_refs()

    configs = [500, 1000, 1500, 2000, 3000]
    all_results = {}

    for ds_name, loader, ck in DATASETS:
        print(f"\n{'=' * 60}\n{ds_name.upper()}: scVelo dynamical sweep\n{'=' * 60}")

        adata_raw = loader()
        ds_results = []

        for n_top in configs:
            label = f"n_top={n_top}"
            print(f"\n  {label}...")

            adata, gamma, fit_like = run_config(adata_raw, n_top, label)

            if gamma is not None:
                gv = gamma.values.astype(float)
                n_valid = np.sum(np.isfinite(gv) & (gv > 0))
                r_m, n_m = eval_halflife(gamma, adata.var_names, hl_mouse)
                r_h, n_h = eval_halflife(gamma, adata.var_names, hl_human)

                # Check fit quality
                fl = fit_like.values.astype(float) if fit_like is not None else np.array([])
                mean_like = float(np.nanmean(fl)) if len(fl) > 0 else np.nan
                low_like = int((fl < 0.1).sum()) if len(fl) > 0 else 0

                print(f"    valid gamma: {n_valid}/{len(gv)}")
                print(f"    HL mouse: r={r_m:.4f} (n={n_m})")
                print(f"    HL human: r={r_h:.4f} (n={n_h})")
                print(f"    mean fit_likelihood: {mean_like:.4f}, low_like (<0.1): {low_like}")

                ds_results.append({
                    "n_top_genes": n_top, "n_valid_gamma": int(n_valid),
                    "hl_mouse_r": r_m, "hl_mouse_n": n_m,
                    "hl_human_r": r_h, "hl_human_n": n_h,
                    "mean_fit_likelihood": mean_like, "n_low_likelihood": low_like,
                })
            else:
                ds_results.append({"n_top_genes": n_top, "error": True})

        # Also run steady-state for comparison
        print(f"\n  Steady-state (n_top=2000)...")
        adata_ss = adata_raw.copy()
        scv.pp.filter_and_normalize(adata_ss, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(adata_ss, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(adata_ss, mode="steady_state")
        ss_gamma = adata_ss.var.get("velocity_gamma", pd.Series(dtype=float))
        r_ss_m, _ = eval_halflife(ss_gamma, adata_ss.var_names, hl_mouse)
        r_ss_h, _ = eval_halflife(ss_gamma, adata_ss.var_names, hl_human)
        print(f"    SS: mouse={r_ss_m:.4f}, human={r_ss_h:.4f}")

        all_results[ds_name] = {
            "dynamical_sweep": ds_results,
            "steady_state": {"hl_mouse_r": r_ss_m, "hl_human_r": r_ss_h},
        }

        # Summary
        print(f"\n  SUMMARY: scVelo dynamical across configs")
        for r in ds_results:
            if "error" in r:
                print(f"    n_top={r['n_top_genes']}: FAILED")
            else:
                print(f"    n_top={r['n_top_genes']}: mouse={r['hl_mouse_r']:.4f}, human={r['hl_human_r']:.4f}")
        print(f"    Steady-state: mouse={r_ss_m:.4f}, human={r_ss_h:.4f}")

    save_json(all_results, "scvelo_dyn_sweep", OUT)

    # Figure
    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]

    for ax, (ds_name, res) in zip(axes, all_results.items()):
        sweep = [r for r in res["dynamical_sweep"] if "error" not in r]
        if not sweep:
            continue
        ntops = [r["n_top_genes"] for r in sweep]
        rs_h = [r["hl_human_r"] for r in sweep]

        ax.plot(ntops, rs_h, "o-", color="steelblue", label="Dynamical")
        ax.axhline(res["steady_state"]["hl_human_r"], color="red", ls="--",
                   label=f"SS={res['steady_state']['hl_human_r']:.3f}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("n_top_genes")
        ax.set_ylabel("Spearman r with half-life (human)")
        ax.set_title(f"{ds_name}")
        ax.legend()

    fig.suptitle("scVelo dynamical: failure across configurations", y=1.02)
    fig.tight_layout()
    save_fig(fig, "scvelo_dyn_sweep", OUT)


if __name__ == "__main__":
    main()
