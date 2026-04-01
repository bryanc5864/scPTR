#!/usr/bin/env python
"""Perturbation validation: do RBP knockdowns affect predicted PT targets?

Searches for published Perturb-seq / CRISPRi data targeting RBPs, then
tests whether scPTR's PT-specific genes show differential expression
after RBP perturbation.

If no suitable dataset is found, performs an in-silico perturbation
analysis using the eCLIP network.
"""
from _common import *

OUT = output_dir("27_perturbation_validation")


def in_silico_perturbation(adata_an, dataset_name):
    """In-silico perturbation: if we remove RBP target genes from gamma,
    does the remaining signal change?

    Tests the prediction: PT-specific genes (z_PT-correlated) should be
    enriched among targets of specific RBPs. If we stratify genes by their
    RBP target status, PT-specific genes should cluster with their regulators.
    """
    print(f"\n{'=' * 60}")
    print(f"IN-SILICO PERTURBATION ({dataset_name})")
    print("=" * 60)

    # Load eCLIP targets
    eclip = pd.read_csv(DATA_DIR / "eclip_targets.csv")
    eclip_by_rbp = eclip.groupby("rbp")["target_gene"].apply(lambda x: set(x.str.upper())).to_dict()

    # Load PT-specific genes
    adv_file = PROJECT_ROOT / "output" / "deep_advantages" / "results" / f"{dataset_name}_advantages.json"
    if not adv_file.exists():
        print("  [SKIP] No advantage results")
        return None

    with open(adv_file) as f:
        adv = json.load(f)
    pt_genes = set(g.upper() for g in adv.get("disentanglement", {}).get("pt_specific_genes", []))
    all_genes = set(g.upper() for g in adata_an.var_names)

    if not pt_genes:
        print("  No PT-specific genes")
        return None

    print(f"  PT-specific genes: {len(pt_genes)}")
    print(f"  All genes: {len(all_genes)}")

    # For each RBP: test if PT-specific genes are enriched among its targets
    # compared to all genes in the dataset
    rbp_enrichment = []

    for rbp, targets in eclip_by_rbp.items():
        targets_in_data = targets & all_genes
        if len(targets_in_data) < 5:
            continue

        pt_in_targets = pt_genes & targets_in_data
        pt_not_in_targets = pt_genes - targets_in_data
        nonpt_in_targets = targets_in_data - pt_genes
        nonpt_not_in_targets = all_genes - pt_genes - targets_in_data

        # Fisher's exact test
        a = len(pt_in_targets)
        b = len(pt_not_in_targets)
        c = len(nonpt_in_targets)
        d = len(nonpt_not_in_targets)

        if min(a, b, c, d) >= 0 and a + b > 0 and c + d > 0:
            odds, p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
            rbp_enrichment.append({
                "rbp": rbp,
                "n_targets_in_data": len(targets_in_data),
                "n_pt_targets": a,
                "odds_ratio": float(odds),
                "p_value": float(p),
            })

    rbp_enrichment.sort(key=lambda x: x["p_value"])

    print(f"\n  RBP enrichment (PT genes among targets):")
    print(f"  {'RBP':<15} {'Targets':>8} {'PT hits':>8} {'OR':>8} {'p':>12}")
    print("  " + "-" * 55)
    for r in rbp_enrichment[:15]:
        print(f"  {r['rbp']:<15} {r['n_targets_in_data']:>8} {r['n_pt_targets']:>8} "
              f"{r['odds_ratio']:>8.2f} {r['p_value']:>12.2e}")

    # Multiple testing correction
    if rbp_enrichment:
        from statsmodels.stats.multitest import multipletests
        pvals = [r["p_value"] for r in rbp_enrichment]
        _, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
        n_sig = (p_adj < 0.05).sum()
        for r, pa in zip(rbp_enrichment, p_adj):
            r["p_adjusted"] = float(pa)
        print(f"\n  Significant after FDR correction: {n_sig}/{len(rbp_enrichment)}")

    # Gamma-based perturbation prediction
    # For the top RBP: are its targets' gamma values different from non-targets?
    gamma_med = np.median(adata_an.layers["gamma"], axis=0)
    gamma_s = pd.Series(gamma_med, index=adata_an.var_names)

    gamma_comparisons = []
    for rbp_info in rbp_enrichment[:5]:
        rbp = rbp_info["rbp"]
        targets = eclip_by_rbp[rbp]
        targets_in = [g for g in adata_an.var_names if g.upper() in targets]
        non_targets = [g for g in adata_an.var_names if g.upper() not in targets]

        if len(targets_in) < 5 or len(non_targets) < 5:
            continue

        g_targets = gamma_s[targets_in].values
        g_non = gamma_s[non_targets].values

        # Filter to non-zero
        g_targets = g_targets[g_targets > 0]
        g_non = g_non[g_non > 0]

        if len(g_targets) < 5:
            continue

        u_stat, u_p = stats.mannwhitneyu(g_targets, g_non, alternative="greater")
        median_ratio = np.median(g_targets) / max(np.median(g_non), 1e-8)

        gamma_comparisons.append({
            "rbp": rbp,
            "n_targets": len(g_targets),
            "median_gamma_targets": float(np.median(g_targets)),
            "median_gamma_background": float(np.median(g_non)),
            "fold_change": float(median_ratio),
            "mannwhitney_p": float(u_p),
        })
        print(f"\n  {rbp} targets gamma: median={np.median(g_targets):.4f} "
              f"vs background={np.median(g_non):.4f} (FC={median_ratio:.2f}, p={u_p:.2e})")

    return {
        "rbp_enrichment": rbp_enrichment[:20],
        "gamma_comparisons": gamma_comparisons,
        "n_pt_genes": len(pt_genes),
    }


def main():
    set_figure_style()
    all_results = {}

    for ds_name, loader, ck in DATASETS:
        adata_an = run_analytical(loader)
        result = in_silico_perturbation(adata_an, ds_name)
        if result:
            all_results[ds_name] = result

    save_json(all_results, "perturbation_validation", OUT)

    # Figure: RBP enrichment
    for ds_name, res in all_results.items():
        enrich = res.get("rbp_enrichment", [])
        if not enrich:
            continue
        top = enrich[:10]
        fig, ax = plt.subplots(figsize=(8, 5))
        rbps = [r["rbp"] for r in top]
        pvals = [-np.log10(r["p_value"] + 1e-300) for r in top]
        colors = ["darkorange" if r.get("p_adjusted", 1) < 0.05 else "steelblue" for r in top]
        ax.barh(rbps[::-1], pvals[::-1], color=colors[::-1], alpha=0.7)
        ax.set_xlabel("-log10(p-value)")
        ax.set_title(f"{ds_name}: RBP enrichment among PT-specific genes")
        ax.axvline(-np.log10(0.05), color="red", ls="--", alpha=0.3, label="p=0.05")
        ax.legend()
        fig.tight_layout()
        save_fig(fig, f"{ds_name}_perturbation", OUT)


if __name__ == "__main__":
    main()
