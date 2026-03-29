#!/usr/bin/env python
"""Validate PT-specific genes against eCLIP RBP binding data.

Tests whether genes identified as post-transcriptionally regulated
by DeepPTR's z_PT latent are confirmed RBP targets in ENCODE eCLIP.
"""
from _common import *

OUT = output_dir("03_eclip_validation")


def load_eclip():
    eclip = pd.read_csv(DATA_DIR / "eclip_targets.csv")
    return eclip


def load_pt_genes(dataset_name):
    adv_file = PROJECT_ROOT / "output" / "deep_advantages" / "results" / f"{dataset_name}_advantages.json"
    if not adv_file.exists():
        return []
    with open(adv_file) as f:
        adv = json.load(f)
    return adv.get("disentanglement", {}).get("pt_specific_genes", [])


def load_gene_list(filename):
    with open(DATA_DIR / filename) as f:
        return set(line.strip().upper() for line in f if line.strip())


def main():
    set_figure_style()
    eclip = load_eclip()
    eclip_targets = set(eclip["target_gene"].str.upper())
    eclip_by_rbp = eclip.groupby("rbp")["target_gene"].apply(lambda x: set(x.str.upper())).to_dict()

    are_genes = load_gene_list("are_genes.txt")
    nmd_genes = load_gene_list("nmd_genes.txt")

    all_results = {}

    for name, loader, ck in DATASETS:
        print(f"\n{'=' * 60}\n{name.upper()}\n{'=' * 60}")
        pt_genes = load_pt_genes(name)
        if not pt_genes:
            print("  No PT-specific genes found")
            continue

        pt_upper = set(g.upper() for g in pt_genes)

        # eCLIP overlap
        in_eclip = pt_upper & eclip_targets
        frac = len(in_eclip) / max(len(pt_upper), 1)
        print(f"  PT genes: {len(pt_genes)}, in eCLIP: {len(in_eclip)} ({frac*100:.0f}%)")
        if in_eclip:
            print(f"  Validated: {sorted(in_eclip)}")

        # Per-RBP
        rbp_hits = {}
        for rbp, targets in eclip_by_rbp.items():
            overlap = pt_upper & targets
            if overlap:
                rbp_hits[rbp] = sorted(overlap)

        print(f"\n  Top RBPs:")
        for rbp in sorted(rbp_hits, key=lambda x: len(rbp_hits[x]), reverse=True)[:10]:
            print(f"    {rbp}: {len(rbp_hits[rbp])} — {rbp_hits[rbp][:5]}")

        # Fisher's exact: PT genes vs random background for eCLIP enrichment
        # Background: use all genes from analytical pipeline
        adata_an = run_analytical(loader)
        all_upper = set(g.upper() for g in adata_an.var_names)
        bg_in_eclip = all_upper & eclip_targets

        # 2x2 table: [PT∩eCLIP, PT∩¬eCLIP; ¬PT∩eCLIP, ¬PT∩¬eCLIP]
        a = len(in_eclip)
        b = len(pt_upper) - a
        c = len(bg_in_eclip) - a
        d = len(all_upper) - len(pt_upper) - c
        if min(a, b, c, d) >= 0:
            odds, fisher_p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
            print(f"\n  Fisher's exact (PT enriched for eCLIP?): OR={odds:.2f}, p={fisher_p:.4f}")
        else:
            odds, fisher_p = np.nan, np.nan

        # ARE/NMD overlap
        are_overlap = pt_upper & are_genes
        nmd_overlap = pt_upper & nmd_genes
        print(f"  ARE overlap: {len(are_overlap)}, NMD overlap: {len(nmd_overlap)}")

        all_results[name] = {
            "n_pt_genes": len(pt_genes),
            "n_in_eclip": len(in_eclip),
            "frac_in_eclip": frac,
            "fisher_odds": float(odds) if np.isfinite(odds) else None,
            "fisher_p": float(fisher_p) if np.isfinite(fisher_p) else None,
            "validated_genes": sorted(in_eclip),
            "top_rbps": {k: v for k, v in sorted(rbp_hits.items(), key=lambda x: len(x[1]), reverse=True)[:10]},
            "are_overlap": sorted(are_overlap),
            "nmd_overlap": sorted(nmd_overlap),
        }

    save_json(all_results, "eclip_validation", OUT)

    # Summary figure: RBP target counts
    for name, res in all_results.items():
        rbps = res.get("top_rbps", {})
        if not rbps:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        rbp_names = list(rbps.keys())[:10]
        counts = [len(rbps[r]) for r in rbp_names]
        ax.barh(rbp_names, counts, color="darkorange", alpha=0.7)
        ax.set_xlabel("Number of PT-specific gene targets")
        ax.set_title(f"{name}: RBPs targeting PT-specific genes")
        fig.tight_layout()
        save_fig(fig, f"{name}_rbp_targets", OUT)


if __name__ == "__main__":
    main()
