#!/usr/bin/env python
"""Per-cell-type half-life correlation.

Does splitting gamma by cell type improve correlation with published half-lives?
This tests whether cell-type-specific degradation captures biology better than
the global median.
"""
from _common import *

OUT = output_dir("10_celltype_halflife")


def run(name, loader, cluster_key):
    print(f"\n{'=' * 60}\n{name.upper()}: Per-cell-type half-life\n{'=' * 60}")

    adata = run_analytical(loader)
    _, hl_human = load_halflife_refs()

    if cluster_key not in adata.obs.columns:
        print("  [SKIP] No cluster key")
        return None

    # Global median
    r_global, n_global = halflife_spearman(adata, hl_human)
    print(f"  Global median gamma: r={r_global:.4f} (n={n_global})")

    # Per-cell-type
    cell_types = sorted(adata.obs[cluster_key].unique())
    records = []

    for ct in cell_types:
        mask = (adata.obs[cluster_key] == ct).values
        if mask.sum() < 20:
            continue

        # Create temp adata with cell-type-specific gamma
        gamma_ct = np.median(adata.layers["gamma"][mask], axis=0)
        adata_ct = adata.copy()
        adata_ct.layers["gamma"] = np.tile(gamma_ct, (adata.n_obs, 1))

        r_ct, n_ct = halflife_spearman(adata_ct, hl_human)
        records.append({
            "cell_type": str(ct), "n_cells": int(mask.sum()),
            "spearman_r": float(r_ct), "n_genes": n_ct,
        })
        diff = abs(r_ct) - abs(r_global)
        print(f"  {ct:25s}: r={r_ct:.4f} (n={n_ct}, {'↑' if diff > 0 else '↓'}{abs(diff):.4f})")

    # Best cell type
    if records:
        best = min(records, key=lambda x: x["spearman_r"])
        worst = max(records, key=lambda x: x["spearman_r"])
        print(f"\n  Best:  {best['cell_type']} (r={best['spearman_r']:.4f})")
        print(f"  Worst: {worst['cell_type']} (r={worst['spearman_r']:.4f})")
        print(f"  Range: {abs(best['spearman_r']) - abs(worst['spearman_r']):.4f}")

    results = {"global_r": r_global, "per_celltype": records}
    save_json(results, f"{name}_celltype_halflife", OUT)

    # Figure
    if records:
        fig, ax = plt.subplots(figsize=(8, 5))
        cts = [r["cell_type"] for r in records]
        rs_abs = [abs(r["spearman_r"]) for r in records]
        colors = ["darkorange" if abs(r["spearman_r"]) > abs(r_global) else "steelblue" for r in records]
        ax.barh(cts, rs_abs, color=colors, alpha=0.7)
        ax.axvline(abs(r_global), color="red", ls="--", label=f"Global={abs(r_global):.3f}")
        ax.set_xlabel("|Spearman r| with half-life")
        ax.set_title(f"{name}: Cell-type-specific half-life correlation")
        ax.legend()
        fig.tight_layout()
        save_fig(fig, f"{name}_celltype_halflife", OUT)

    return results


def main():
    set_figure_style()
    for name, loader, ck in DATASETS:
        run(name, loader, ck)


if __name__ == "__main__":
    main()
