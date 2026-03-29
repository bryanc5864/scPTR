#!/usr/bin/env python
"""Bootstrap confidence intervals on all key metrics."""
from _common import *

OUT = output_dir("02_bootstrap_ci")


def bootstrap_halflife(adata, hl_df, n_boot=1000, seed=42):
    g, h, _ = match_halflife(adata, hl_df)
    if len(g) < 10:
        return {"r": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "se": np.nan, "n": len(g)}

    r_point, _ = stats.spearmanr(g, h)
    rng = np.random.RandomState(seed)
    rs = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.choice(len(g), size=len(g), replace=True)
        rs[i], _ = stats.spearmanr(g[idx], h[idx])

    return {
        "r": float(r_point),
        "ci_lo": float(np.percentile(rs, 2.5)),
        "ci_hi": float(np.percentile(rs, 97.5)),
        "se": float(np.std(rs)),
        "n": len(g),
    }


def main():
    set_figure_style()
    hl_mouse, hl_human = load_halflife_refs()
    all_results = {}

    for name, loader, _ in DATASETS:
        print(f"\n{'=' * 60}\n{name.upper()}\n{'=' * 60}")
        adata_an = run_analytical(loader)
        results = {}
        for ref_name, hl_df in [("mouse", hl_mouse), ("human", hl_human)]:
            r = bootstrap_halflife(adata_an, hl_df)
            results[ref_name] = r
            print(f"  {ref_name}: r={r['r']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] (n={r['n']})")
        all_results[name] = results

    save_json(all_results, "bootstrap_ci", OUT)

    # Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    labels, rs, los, his = [], [], [], []
    for name in all_results:
        for ref in ("mouse", "human"):
            d = all_results[name][ref]
            labels.append(f"{name}\n{ref}")
            rs.append(d["r"])
            los.append(d["r"] - d["ci_lo"])
            his.append(d["ci_hi"] - d["r"])
    ax.barh(range(len(labels)), [-r for r in rs], xerr=[[lo for lo in los], [hi for hi in his]],
            color="steelblue", alpha=0.7, capsize=4)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("|Spearman r| with half-life (95% CI)")
    ax.set_title("Half-life correlation with bootstrap CIs")
    fig.tight_layout()
    save_fig(fig, "bootstrap_ci", OUT)


if __name__ == "__main__":
    main()
