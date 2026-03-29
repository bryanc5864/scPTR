#!/usr/bin/env python
"""Fair comparison: analytical vs DeepPTR on the SAME 300 genes.

The naive comparison is unfair because analytical uses ~5000-12000 genes
while DeepPTR uses 300. This script evaluates analytical gamma restricted
to the same gene set.
"""
from _common import *

OUT = output_dir("01_fair_comparison")


def run(name, loader, cluster_key):
    print(f"\n{'=' * 60}\n{name.upper()}\n{'=' * 60}")

    adata_an = run_analytical(loader)
    top_genes = select_top_genes(adata_an, n_top=300).var_names.tolist()

    # Analytical on same 300 genes
    an_300_idx = [list(adata_an.var_names).index(g) for g in top_genes if g in adata_an.var_names]
    adata_300 = adata_an[:, [adata_an.var_names[i] for i in an_300_idx]].copy()
    adata_300.layers["gamma"] = adata_an.layers["gamma"][:, an_300_idx]

    hl_mouse, hl_human = load_halflife_refs()

    # Load DeepPTR results from v1
    prev_file = PROJECT_ROOT / "output" / "deep_benchmark" / "results" / f"{name}_benchmark.json"
    prev = json.load(open(prev_file)) if prev_file.exists() else {}

    results = {}
    for ref_name, hl_df, hl_key in [
        ("mouse", hl_mouse, "mouse_herzog"),
        ("human", hl_human, "human_schofield"),
    ]:
        r_all, n_all = halflife_spearman(adata_an, hl_df)
        r_300, n_300 = halflife_spearman(adata_300, hl_df)
        dp = prev.get("halflife", {}).get(hl_key, {}).get("deepptr", {})
        r_dp, n_dp = dp.get("spearman_r", np.nan), dp.get("n_genes", 0)

        results[ref_name] = {
            "analytical_all": {"r": r_all, "n": n_all},
            "analytical_300": {"r": r_300, "n": n_300},
            "deepptr_300": {"r": r_dp, "n": n_dp},
        }
        print(f"  {ref_name}: all={r_all:.4f}(n={n_all})  300={r_300:.4f}(n={n_300})  deep={r_dp:.4f}(n={n_dp})")

    save_json(results, f"{name}_fair_comparison", OUT)
    return results


def main():
    set_figure_style()
    all_r = {}
    for name, loader, ck in DATASETS:
        all_r[name] = run(name, loader, ck)

    # Summary figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, ref in enumerate(["mouse", "human"]):
        labels, an_all, an_300, dp_300 = [], [], [], []
        for name in all_r:
            d = all_r[name].get(ref, {})
            labels.append(name)
            an_all.append(abs(d.get("analytical_all", {}).get("r", 0)))
            an_300.append(abs(d.get("analytical_300", {}).get("r", 0)))
            dp_300.append(abs(d.get("deepptr_300", {}).get("r", 0)))

        x = np.arange(len(labels))
        w = 0.25
        axes[ax_idx].bar(x - w, an_all, w, label="Analytical (all genes)", color="steelblue")
        axes[ax_idx].bar(x, an_300, w, label="Analytical (300 genes)", color="lightsteelblue")
        axes[ax_idx].bar(x + w, dp_300, w, label="DeepPTR (300 genes)", color="darkorange")
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels(labels)
        axes[ax_idx].set_ylabel("|Spearman r| with half-life")
        axes[ax_idx].set_title(f"{ref} reference")
        axes[ax_idx].legend(fontsize=8)

    fig.suptitle("Fair comparison: same gene set", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fair_comparison_summary", OUT)


if __name__ == "__main__":
    main()
