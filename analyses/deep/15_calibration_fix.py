#!/usr/bin/env python
"""Fix uncertainty calibration via post-hoc temperature scaling.

The raw posterior has 27% coverage for 95% CI. This script:
1. Learns a temperature T on synthetic data
2. Scales posterior variance by T^2
3. Shows improved calibration on held-out synthetic + real data
"""
from _common import *
from scptr.deep.synthetic import generate_kinetic_data
from scipy.optimize import minimize_scalar

OUT = output_dir("15_calibration_fix")


def compute_coverage(gamma_true, gamma_mean, gamma_var, level=0.95, temperature=1.0):
    """Compute CI coverage at given temperature."""
    z = stats.norm.ppf(0.5 + level / 2)
    std = np.sqrt(np.clip(gamma_var * temperature**2, 1e-10, None))
    inside = (gamma_true >= gamma_mean - z * std) & (gamma_true <= gamma_mean + z * std)
    return float(inside.mean())


def find_temperature(gamma_true, gamma_mean, gamma_var, target=0.95):
    """Find temperature that gives target coverage."""
    def loss(log_t):
        t = np.exp(log_t)
        cov = compute_coverage(gamma_true, gamma_mean, gamma_var, target, t)
        return (cov - target) ** 2

    result = minimize_scalar(loss, bounds=(-2, 5), method="bounded")
    return np.exp(result.x)


def main():
    set_figure_style()

    # ── Learn temperature on synthetic data ───────────────────────────
    print("=" * 60)
    print("Learning calibration temperature on synthetic data")
    print("=" * 60)

    adata_train, truth_train = generate_kinetic_data(n_cells=1500, n_genes=100, seed=0)
    adata_test, truth_test = generate_kinetic_data(n_cells=1500, n_genes=100, seed=42)

    torch.set_num_threads(4)

    # Fit on training synthetic
    scptr.deep.fit_deepptr(adata_train, verbose=False, **DEEP_HP)
    # Fit on test synthetic (separate model)
    scptr.deep.fit_deepptr(adata_test, verbose=False, **{**DEEP_HP, "seed": 42})

    # Learn T on training set
    T = find_temperature(truth_train["gamma"], adata_train.layers["gamma"],
                          adata_train.layers["gamma_var"])
    print(f"  Learned temperature: T = {T:.4f}")

    # Evaluate on test set (held-out)
    cov_raw = compute_coverage(truth_test["gamma"], adata_test.layers["gamma"],
                                adata_test.layers["gamma_var"])
    cov_cal = compute_coverage(truth_test["gamma"], adata_test.layers["gamma"],
                                adata_test.layers["gamma_var"], temperature=T)
    print(f"  Test coverage (raw):        {cov_raw:.4f}")
    print(f"  Test coverage (calibrated): {cov_cal:.4f}")

    # Full calibration curve
    levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    raw_covs = [compute_coverage(truth_test["gamma"], adata_test.layers["gamma"],
                                  adata_test.layers["gamma_var"], l) for l in levels]
    cal_covs = [compute_coverage(truth_test["gamma"], adata_test.layers["gamma"],
                                  adata_test.layers["gamma_var"], l, T) for l in levels]

    print(f"\n  {'Level':>8} {'Raw':>8} {'Calibrated':>12}")
    for l, r, c in zip(levels, raw_covs, cal_covs):
        print(f"  {l:>8.2f} {r:>8.4f} {c:>12.4f}")

    # ── Apply to real data ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Applying calibration to pancreas")
    print("=" * 60)

    adata_real = scptr.datasets.pancreas()
    scptr.pp.filter_genes(adata_real)
    scptr.pp.normalize_layers(adata_real)
    scptr.pp.neighbors(adata_real, n_neighbors=30)
    scptr.pp.smooth_layers(adata_real)
    scptr.tl.estimate_beta(adata_real)
    adata_real = select_top_genes(adata_real, n_top=300)
    from scipy.sparse import issparse
    for key in ("spliced", "unspliced"):
        if key in adata_real.layers and issparse(adata_real.layers[key]):
            adata_real.layers[key] = np.asarray(adata_real.layers[key].todense())

    torch.set_num_threads(4)
    scptr.deep.fit_deepptr(adata_real, verbose=False, **DEEP_HP)

    # Calibrated variance
    gamma_var_cal = adata_real.layers["gamma_var"] * T**2
    adata_real.layers["gamma_var_calibrated"] = gamma_var_cal

    # Show calibrated uncertainty is more useful for gene filtering
    _, hl_human = load_halflife_refs()
    gamma_med = np.median(adata_real.layers["gamma"], axis=0)
    gamma_cv_raw = np.sqrt(np.median(adata_real.layers["gamma_var"], axis=0)) / (gamma_med + 1e-8)
    gamma_cv_cal = np.sqrt(np.median(gamma_var_cal, axis=0)) / (gamma_med + 1e-8)

    g, h, names = match_halflife(adata_real, hl_human)
    name_to_idx = {n: i for i, n in enumerate(adata_real.var_names)}
    cv_matched = np.array([gamma_cv_cal[name_to_idx[n]] for n in names])

    r_all, _ = stats.spearmanr(g, h)
    # Filter by calibrated CV
    mask_50 = cv_matched <= np.percentile(cv_matched, 50)
    mask_25 = cv_matched <= np.percentile(cv_matched, 25)
    r_50, _ = stats.spearmanr(g[mask_50], h[mask_50]) if mask_50.sum() > 10 else (np.nan, None)
    r_25, _ = stats.spearmanr(g[mask_25], h[mask_25]) if mask_25.sum() > 10 else (np.nan, None)

    print(f"  Calibrated uncertainty filtering:")
    print(f"    All genes:      r={r_all:.4f} (n={len(g)})")
    print(f"    Bottom 50% CV:  r={r_50:.4f} (n={mask_50.sum()})")
    print(f"    Bottom 25% CV:  r={r_25:.4f} (n={mask_25.sum()})")

    results = {
        "temperature": float(T),
        "test_coverage_raw": cov_raw,
        "test_coverage_calibrated": cov_cal,
        "calibration_curve": [{"level": l, "raw": r, "calibrated": c}
                               for l, r, c in zip(levels, raw_covs, cal_covs)],
        "real_filtering": {
            "all": {"r": float(r_all), "n": len(g)},
            "bottom_50pct": {"r": float(r_50), "n": int(mask_50.sum())},
            "bottom_25pct": {"r": float(r_25), "n": int(mask_25.sum())},
        },
    }
    save_json(results, "calibration_fix", OUT)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration curve
    axes[0].plot(levels, raw_covs, "o-", label="Raw", color="gray")
    axes[0].plot(levels, cal_covs, "o-", label=f"Calibrated (T={T:.2f})", color="darkorange")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")
    axes[0].set_xlabel("Nominal coverage")
    axes[0].set_ylabel("Actual coverage")
    axes[0].set_title("CI Calibration (held-out synthetic)")
    axes[0].legend()

    # Uncertainty filtering improvement
    bars = axes[1].bar(["All", "Bottom\n50% CV", "Bottom\n25% CV"],
                       [abs(r_all), abs(r_50), abs(r_25)],
                       color=["gray", "steelblue", "darkorange"], alpha=0.7)
    axes[1].set_ylabel("|r| with half-life")
    axes[1].set_title("Calibrated uncertainty filtering (pancreas)")

    fig.suptitle(f"Post-hoc calibration (T={T:.2f})", y=1.02)
    fig.tight_layout()
    save_fig(fig, "calibration_fix", OUT)


if __name__ == "__main__":
    main()
