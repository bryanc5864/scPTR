#!/usr/bin/env python
"""Expression level vs degradation rate: are they independent?

Tests the degree to which gamma is decorrelated from expression level.
High decorrelation validates that gamma captures post-transcriptional
regulation beyond what's explained by expression abundance.
"""
from _common import *

OUT = output_dir("11_expression_vs_gamma")


def run(name, loader, cluster_key):
    print(f"\n{'=' * 60}\n{name.upper()}: Expression vs gamma\n{'=' * 60}")

    adata = run_analytical(loader)

    from scipy.sparse import issparse
    X = adata.X
    if issparse(X):
        X = np.asarray(X.todense())
    X = np.asarray(X, dtype=np.float32)

    gamma = adata.layers["gamma"]
    gamma_med = np.median(gamma, axis=0)
    expr_mean = X.mean(axis=0)

    # Filter to expressed genes with gamma > 0
    valid = (gamma_med > 0) & (expr_mean > 0) & np.isfinite(gamma_med) & np.isfinite(expr_mean)
    g, e = gamma_med[valid], expr_mean[valid]

    r_expr_gamma, p = stats.spearmanr(np.log1p(e), np.log1p(g))
    print(f"  Expression vs gamma: r={r_expr_gamma:.4f} (p={p:.2e}, n={len(g)})")

    # Does controlling for expression improve half-life correlation?
    _, hl_human = load_halflife_refs()
    g_hl, h_hl, names = match_halflife(adata, hl_human)

    # Get expression for matched genes
    name_to_idx = {n: i for i, n in enumerate(adata.var_names)}
    expr_matched = np.array([expr_mean[name_to_idx[n]] for n in names])

    r_raw, _ = stats.spearmanr(g_hl, h_hl)

    # Partial correlation: gamma vs half-life, controlling for expression
    # Residualize gamma and halflife on expression
    from numpy.polynomial.polynomial import polyfit, polyval
    log_e = np.log1p(expr_matched)
    log_g = np.log1p(g_hl)
    log_h = np.log1p(h_hl)

    # Residualize
    coef_g = polyfit(log_e, log_g, 1)
    resid_g = log_g - polyval(log_e, coef_g)
    coef_h = polyfit(log_e, log_h, 1)
    resid_h = log_h - polyval(log_e, coef_h)

    r_partial, p_partial = stats.spearmanr(resid_g, resid_h)
    print(f"  Raw gamma-halflife: r={r_raw:.4f}")
    print(f"  Partial (ctrl expression): r={r_partial:.4f} (p={p_partial:.2e})")
    print(f"  Expression confound: Δr={abs(r_raw) - abs(r_partial):.4f}")

    # Variance decomposition: how much gamma variance is explained by expression?
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(np.log1p(e).reshape(-1, 1), np.log1p(g))
    r2_expr = lr.score(np.log1p(e).reshape(-1, 1), np.log1p(g))
    print(f"  R² of gamma ~ expression: {r2_expr:.4f} ({r2_expr*100:.1f}% explained)")
    print(f"  Residual gamma variance:  {1 - r2_expr:.4f} ({(1-r2_expr)*100:.1f}% independent)")

    results = {
        "r_expr_gamma": float(r_expr_gamma),
        "r_raw_halflife": float(r_raw),
        "r_partial_halflife": float(r_partial),
        "expression_confound": float(abs(r_raw) - abs(r_partial)),
        "r2_gamma_from_expression": float(r2_expr),
        "n_genes": int(len(g)),
    }
    save_json(results, f"{name}_expr_vs_gamma", OUT)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].scatter(np.log1p(e), np.log1p(g), alpha=0.05, s=3, c="steelblue")
    axes[0].set_xlabel("log(1 + mean expression)")
    axes[0].set_ylabel("log(1 + median gamma)")
    axes[0].set_title(f"Expression vs gamma (r={r_expr_gamma:.3f})")

    bars = axes[1].bar(["Raw", "Partial\n(ctrl expr)"],
                       [abs(r_raw), abs(r_partial)],
                       color=["steelblue", "darkorange"], alpha=0.7)
    axes[1].set_ylabel("|Spearman r| with half-life")
    axes[1].set_title(f"Expression confound (Δr={abs(r_raw)-abs(r_partial):.3f})")

    # Pie: variance decomposition
    axes[2].pie([r2_expr, 1 - r2_expr],
                labels=[f"Expression\n({r2_expr*100:.0f}%)",
                        f"Independent\n({(1-r2_expr)*100:.0f}%)"],
                colors=["lightcoral", "lightsteelblue"],
                autopct="%1.0f%%", startangle=90)
    axes[2].set_title("Gamma variance decomposition")

    fig.suptitle(f"{name}: Expression vs degradation", y=1.02)
    fig.tight_layout()
    save_fig(fig, f"{name}_expr_vs_gamma", OUT)

    return results


def main():
    set_figure_style()
    for name, loader, ck in DATASETS:
        run(name, loader, ck)


if __name__ == "__main__":
    main()
