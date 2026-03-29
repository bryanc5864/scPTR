#!/usr/bin/env python
"""CI coverage breakdown: diagnose why 95% CI has only 27% coverage."""
from _common import *
from scptr.deep.synthetic import generate_kinetic_data

OUT = output_dir("06_ci_coverage")


def main():
    set_figure_style()
    print("Generating synthetic data and fitting DeepPTR...")
    adata, truth = generate_kinetic_data(n_cells=1500, n_genes=100, seed=0)

    torch.set_num_threads(4)
    scptr.deep.fit_deepptr(adata, verbose=False, **DEEP_HP)

    gt = truth["gamma"]
    mu = adata.layers["gamma"]
    var = adata.layers["gamma_var"]

    z = 1.96
    std = np.sqrt(np.clip(var, 1e-10, None))
    inside = (gt >= mu - z * std) & (gt <= mu + z * std)

    overall = float(inside.mean())
    per_gene = inside.mean(axis=0)
    per_cell = inside.mean(axis=1)

    # Diagnosis
    ci_width = np.median(2 * z * std)
    true_range = np.median(np.ptp(gt, axis=0))
    rel_error = np.median(np.abs(mu - gt) / (gt + 1e-8))

    print(f"\n  Overall 95% CI coverage: {overall:.4f} (target: 0.95)")
    print(f"  Median CI width:         {ci_width:.4f}")
    print(f"  Median true range:       {true_range:.4f}")
    print(f"  CI/range ratio:          {ci_width / true_range:.4f}")
    print(f"  Median relative error:   {rel_error:.4f}")
    print(f"  Diagnosis: {'Overconfident' if overall < 0.5 else 'Moderate'} "
          f"(posterior {ci_width/true_range:.1%} of true range)")

    # What would coverage be at different CI levels?
    levels = [0.50, 0.80, 0.90, 0.95, 0.99]
    coverages = []
    for lev in levels:
        zl = stats.norm.ppf(0.5 + lev / 2)
        ins = (gt >= mu - zl * std) & (gt <= mu + zl * std)
        coverages.append(float(ins.mean()))
        print(f"  {lev*100:.0f}% CI → actual coverage: {ins.mean():.4f}")

    results = {
        "overall_coverage": overall,
        "ci_width": float(ci_width), "true_range": float(true_range),
        "rel_error": float(rel_error),
        "calibration": [{"nominal": l, "actual": c} for l, c in zip(levels, coverages)],
    }
    save_json(results, "ci_coverage", OUT)

    # Calibration plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(levels, coverages, "o-", color="darkorange", label="Actual")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    axes[0].set_xlabel("Nominal coverage")
    axes[0].set_ylabel("Actual coverage")
    axes[0].set_title("CI Calibration")
    axes[0].legend()

    axes[1].hist(per_gene, bins=20, color="steelblue", alpha=0.7)
    axes[1].axvline(0.95, color="red", ls="--", label="Target")
    axes[1].set_xlabel("Per-gene 95% CI coverage")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Coverage distribution (median={np.median(per_gene):.3f})")
    axes[1].legend()

    fig.tight_layout()
    save_fig(fig, "ci_calibration", OUT)


if __name__ == "__main__":
    main()
