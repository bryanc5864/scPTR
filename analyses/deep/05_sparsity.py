#!/usr/bin/env python
"""Sparsity analysis: how does unspliced detection rate affect gamma quality?"""
from _common import *

OUT = output_dir("05_sparsity")


def run(name, loader, _):
    print(f"\n{'=' * 60}\n{name.upper()}\n{'=' * 60}")
    adata = run_analytical(loader)
    _, hl_human = load_halflife_refs()

    from scipy.sparse import issparse
    u = adata.layers["unspliced"]
    if issparse(u):
        u = np.asarray(u.todense())
    frac_det = (np.asarray(u) > 0).mean(axis=0)

    g, h, names = match_halflife(adata, hl_human)
    # Get detection rate for matched genes
    det = np.array([frac_det[list(adata.var_names).index(n)] for n in names])

    quartiles = np.percentile(det, [25, 50, 75])
    bins = [
        ("Q1 (sparse)", det <= quartiles[0]),
        ("Q2", (det > quartiles[0]) & (det <= quartiles[1])),
        ("Q3", (det > quartiles[1]) & (det <= quartiles[2])),
        ("Q4 (dense)", det > quartiles[2]),
    ]

    records = []
    for label, mask in bins:
        if mask.sum() < 10:
            continue
        r, _ = stats.spearmanr(g[mask], h[mask])
        records.append({"quartile": label, "n": int(mask.sum()),
                        "r": float(r), "med_det": float(np.median(det[mask]))})
        print(f"  {label}: r={r:.4f} (n={mask.sum()}, det={np.median(det[mask]):.3f})")

    save_json({"dataset": name, "stratified": records}, f"{name}_sparsity", OUT)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([r["quartile"] for r in records], [abs(r["r"]) for r in records],
           color="steelblue", alpha=0.7)
    for i, r in enumerate(records):
        ax.text(i, abs(r["r"]) + 0.005, f"n={r['n']}", ha="center", fontsize=8)
    ax.set_ylabel("|Spearman r| with half-life")
    ax.set_title(f"{name}: Half-life r by unspliced detection rate")
    fig.tight_layout()
    save_fig(fig, f"{name}_sparsity", OUT)
    return records


def main():
    set_figure_style()
    for name, loader, ck in DATASETS:
        run(name, loader, ck)


if __name__ == "__main__":
    main()
