#!/usr/bin/env python
"""Partial correlation deep dive: gamma signal after controlling for expression.

Addresses the r=-0.40 → r=-0.15 drop by:
1. Showing r=-0.15 is still highly significant
2. Comparing partial r across methods (scPTR vs scVelo)
3. Bootstrap CI on partial r
4. Decomposing: how much is expression, how much is unique PT regulation?
"""
from _common import *
import scvelo as scv

OUT = output_dir("22_partial_correlation")


def partial_halflife(gamma_med, expr_mean, hl_vals, gene_names, hl_df):
    """Compute raw and partial (controlling expression) half-life r."""
    hl_s = hl_df.set_index("gene_symbol")["half_life_hours"]

    gamma_upper = {g.upper(): i for i, g in enumerate(gene_names)}
    hl_upper = {g.upper(): g for g in hl_s.index if isinstance(g, str)}
    shared = set(gamma_upper.keys()) & set(hl_upper.keys())

    g = np.array([gamma_med[gamma_upper[u]] for u in shared], dtype=float)
    h = np.array([hl_s[hl_upper[u]] for u in shared], dtype=float)
    e = np.array([expr_mean[gamma_upper[u]] for u in shared], dtype=float)

    v = np.isfinite(g) & np.isfinite(h) & np.isfinite(e) & (g > 0) & (h > 0) & (e > 0)
    g, h, e = g[v], h[v], e[v]

    if len(g) < 10:
        return {"raw_r": np.nan, "partial_r": np.nan, "n": 0}

    # Raw
    r_raw, p_raw = stats.spearmanr(g, h)

    # Partial: residualize on log expression
    from numpy.polynomial.polynomial import polyfit, polyval
    log_e = np.log1p(e)
    log_g = np.log1p(g)
    log_h = np.log1p(h)

    coef_g = polyfit(log_e, log_g, 1)
    resid_g = log_g - polyval(log_e, coef_g)
    coef_h = polyfit(log_e, log_h, 1)
    resid_h = log_h - polyval(log_e, coef_h)

    r_partial, p_partial = stats.spearmanr(resid_g, resid_h)

    # Bootstrap CI on partial r
    rng = np.random.RandomState(42)
    boot_partial = []
    for _ in range(1000):
        idx = rng.choice(len(g), len(g), replace=True)
        coef_g_b = polyfit(log_e[idx], log_g[idx], 1)
        resid_g_b = log_g[idx] - polyval(log_e[idx], coef_g_b)
        coef_h_b = polyfit(log_e[idx], log_h[idx], 1)
        resid_h_b = log_h[idx] - polyval(log_e[idx], coef_h_b)
        r_b, _ = stats.spearmanr(resid_g_b, resid_h_b)
        boot_partial.append(r_b)

    ci_lo, ci_hi = np.percentile(boot_partial, [2.5, 97.5])

    return {
        "raw_r": float(r_raw), "raw_p": float(p_raw),
        "partial_r": float(r_partial), "partial_p": float(p_partial),
        "partial_ci_lo": float(ci_lo), "partial_ci_hi": float(ci_hi),
        "n": int(len(g)),
        "expression_confound": float(abs(r_raw) - abs(r_partial)),
    }


def main():
    set_figure_style()
    _, hl_human = load_halflife_refs()
    all_results = {}

    for ds_name, loader, ck in DATASETS:
        print(f"\n{'=' * 60}\n{ds_name.upper()}: Partial correlation\n{'=' * 60}")

        # scPTR
        adata_an = run_analytical(loader)
        from scipy.sparse import issparse
        X = adata_an.X
        if issparse(X):
            X = np.asarray(X.todense())
        expr_mean = np.asarray(X, dtype=float).mean(axis=0)
        gamma_med = np.median(adata_an.layers["gamma"], axis=0)

        r_scptr = partial_halflife(gamma_med, expr_mean, None, adata_an.var_names, hl_human)
        print(f"\n  scPTR:")
        print(f"    Raw r:     {r_scptr['raw_r']:.4f} (p={r_scptr['raw_p']:.2e})")
        print(f"    Partial r: {r_scptr['partial_r']:.4f} [{r_scptr['partial_ci_lo']:.4f}, {r_scptr['partial_ci_hi']:.4f}] (p={r_scptr['partial_p']:.2e})")
        print(f"    Confound:  Δr={r_scptr['expression_confound']:.4f}")

        # scVelo SS
        adata_sv = loader()
        scv.pp.filter_and_normalize(adata_sv, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(adata_sv, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(adata_sv, mode="steady_state")

        sv_gamma = adata_sv.var.get("velocity_gamma", pd.Series(dtype=float)).values.astype(float)
        X_sv = adata_sv.X
        if issparse(X_sv):
            X_sv = np.asarray(X_sv.todense())
        expr_sv = np.asarray(X_sv, dtype=float).mean(axis=0)

        r_scvelo = partial_halflife(sv_gamma, expr_sv, None, adata_sv.var_names, hl_human)
        print(f"\n  scVelo SS:")
        print(f"    Raw r:     {r_scvelo['raw_r']:.4f}")
        print(f"    Partial r: {r_scvelo['partial_r']:.4f} [{r_scvelo['partial_ci_lo']:.4f}, {r_scvelo['partial_ci_hi']:.4f}]")
        print(f"    Confound:  Δr={r_scvelo['expression_confound']:.4f}")

        # Key comparison
        print(f"\n  COMPARISON: scPTR retains {'MORE' if abs(r_scptr['partial_r']) > abs(r_scvelo['partial_r']) else 'LESS'} "
              f"signal after expression control")
        print(f"    scPTR partial:  {r_scptr['partial_r']:.4f}")
        print(f"    scVelo partial: {r_scvelo['partial_r']:.4f}")

        all_results[ds_name] = {"scptr": r_scptr, "scvelo_ss": r_scvelo}

    save_json(all_results, "partial_correlation", OUT)

    # Figure
    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]

    for ax, (ds_name, res) in zip(axes, all_results.items()):
        methods = ["scPTR", "scVelo SS"]
        raw_rs = [abs(res["scptr"]["raw_r"]), abs(res["scvelo_ss"]["raw_r"])]
        partial_rs = [abs(res["scptr"]["partial_r"]), abs(res["scvelo_ss"]["partial_r"])]

        x = np.arange(len(methods))
        ax.bar(x - 0.2, raw_rs, 0.35, label="Raw", color="steelblue", alpha=0.7)
        ax.bar(x + 0.2, partial_rs, 0.35, label="Partial (ctrl expr)", color="darkorange", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("|Spearman r| with half-life")
        ax.set_title(f"{ds_name}")
        ax.legend()

    fig.suptitle("Half-life r: raw vs expression-controlled", y=1.02)
    fig.tight_layout()
    save_fig(fig, "partial_correlation", OUT)


if __name__ == "__main__":
    main()
