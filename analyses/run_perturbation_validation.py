#!/usr/bin/env python
"""RBP perturbation validation: compare scPTR network predictions with
Replogle 2022 CRISPRi Perturb-seq data (via Harmonizome API).

For each RBP hub identified by scPTR (via Spearman correlation between
RBP expression and target gamma), we test whether its predicted targets
are enriched among genes differentially expressed upon RBP knockdown.

This validates the causal direction: if scPTR correctly identifies that RBP X
regulates gene Y's degradation, then knocking down RBP X should change Y's
expression level.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from _common import set_figure_style

import scptr

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "perturbation_validation"


def save_fig(fig, name, subdir="figures"):
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def load_perturb_seq_de(rbp: str) -> tuple[list[str], list[str]] | None:
    """Load differentially expressed genes from Replogle 2022 CRISPRi Perturb-seq
    via Harmonizome API.

    Returns (up_genes, down_genes): genes whose expression increases/decreases
    when the RBP is knocked down.
    """
    import requests

    rbp_ids = {
        "HNRNPA1": "3857_HNRNPA1_P1P2",
        "YBX1": "9921_YBX1_P1P2",
        "ELAVL1": "2583_ELAVL1_P1P2",
        "SRSF3": "8433_SRSF3_P1P2",
        "RBFOX2": "7148_RBFOX2_P1",
        "FUS": "3224_FUS_P1P2",
        "HNRNPC": "3861_HNRNPC_P1P2",
        "DDX5": "2134_DDX5_P1P2",
        "MBNL1": "4881_MBNL1_P1P2",
    }

    gene_set_id = rbp_ids.get(rbp)
    if gene_set_id is None:
        return None

    dataset_name = ("Replogle+et+al.,+Cell,+2022+K562+Genome-wide+"
                    "Perturb-seq+Gene+Perturbation+Signatures")
    url = (f"https://maayanlab.cloud/Harmonizome/api/1.0/gene_set/"
           f"{gene_set_id}/{dataset_name}")

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"    Harmonizome API error for {rbp}: {e}")
        return None

    associations = data.get("associations", [])
    if not associations:
        return None

    up_genes = []
    down_genes = []
    for assoc in associations:
        gene_name = assoc.get("gene", {}).get("symbol", "")
        value = assoc.get("standardizedValue", 0)
        if value > 0:
            up_genes.append(gene_name)
        else:
            down_genes.append(gene_name)

    return up_genes, down_genes


def infer_spearman_network(adata, rbp_list, n_top_targets=200):
    """Infer RBP-target network using vectorized Spearman partial correlation
    (library-size corrected) between RBP expression and target gene gamma.

    Vectorized approach: rank all columns once, residualize against library size
    ranks using matrix operations, then compute correlations via dot products.
    """

    gamma = np.array(scptr._utils.get_layer(adata, "gamma"))
    expression = np.array(scptr._utils.get_layer(adata, "Ms"))

    gene_names = [g.upper() for g in adata.var_names]
    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}

    n_cells, n_genes = gamma.shape

    # Library size ranks (once)
    lib_size = expression.sum(axis=1)
    lib_rank = stats.rankdata(lib_size)
    lib_rank_centered = lib_rank - lib_rank.mean()
    lib_ss = np.dot(lib_rank_centered, lib_rank_centered)

    # Rank all gamma columns (vectorized)
    gamma_ranks = np.zeros_like(gamma)
    gamma_valid = np.zeros(n_genes, dtype=bool)
    for j in range(n_genes):
        col = gamma[:, j]
        if np.std(col) < 1e-8:
            continue
        gamma_ranks[:, j] = stats.rankdata(col)
        gamma_valid[j] = True

    # Residualize gamma ranks against library size (vectorized)
    # slope_j = dot(lib_rank_centered, gamma_rank_j_centered) / dot(lib_rank_centered, lib_rank_centered)
    gamma_ranks_centered = gamma_ranks - gamma_ranks.mean(axis=0, keepdims=True)
    slopes_gamma = np.dot(lib_rank_centered, gamma_ranks_centered) / lib_ss
    gamma_resid = gamma_ranks - np.outer(lib_rank, slopes_gamma)
    gamma_resid_centered = gamma_resid - gamma_resid.mean(axis=0, keepdims=True)
    gamma_resid_std = np.sqrt((gamma_resid_centered ** 2).sum(axis=0))
    gamma_resid_std[gamma_resid_std < 1e-8] = 1.0  # avoid division by zero

    edges = []
    seen_rbps = set()

    for rbp in rbp_list:
        rbp_upper = rbp.upper()
        if rbp_upper in seen_rbps:
            continue
        if rbp_upper not in gene_name_to_idx:
            continue
        seen_rbps.add(rbp_upper)

        rbp_idx = gene_name_to_idx[rbp_upper]
        rbp_expr = expression[:, rbp_idx]

        if np.std(rbp_expr) < 1e-8:
            continue

        # Rank and residualize RBP expression
        rbp_rank = stats.rankdata(rbp_expr)
        rbp_rank_centered = rbp_rank - rbp_rank.mean()
        slope_rbp = np.dot(lib_rank_centered, rbp_rank_centered) / lib_ss
        rbp_resid = rbp_rank - slope_rbp * lib_rank
        rbp_resid_centered = rbp_resid - rbp_resid.mean()
        rbp_resid_std = np.sqrt(np.dot(rbp_resid_centered, rbp_resid_centered))

        if rbp_resid_std < 1e-8:
            continue

        # Vectorized correlation: r = dot(rbp_resid, gamma_resid) / (std_rbp * std_gamma)
        r_vals = np.dot(rbp_resid_centered, gamma_resid_centered) / (rbp_resid_std * gamma_resid_std)
        r_vals = np.clip(r_vals, -1.0, 1.0)

        # Compute p-values from t-distribution
        df = n_cells - 3  # partial correlation df
        t_vals = r_vals * np.sqrt(df / (1 - r_vals ** 2 + 1e-12))
        p_vals = 2 * stats.t.sf(np.abs(t_vals), df)

        # Filter to valid targets (not self, valid gamma)
        valid_mask = gamma_valid.copy()
        valid_mask[rbp_idx] = False
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            continue

        valid_r = r_vals[valid_indices]
        valid_p = p_vals[valid_indices]

        # Select top N targets by absolute correlation strength
        abs_r = np.abs(valid_r)
        top_k = min(n_top_targets, len(abs_r))
        top_indices = np.argsort(abs_r)[::-1][:top_k]

        for idx_in_valid in top_indices:
            gene_idx = valid_indices[idx_in_valid]
            edges.append({
                "regulator": rbp_upper,
                "target": gene_names[gene_idx],
                "weight": float(valid_r[idx_in_valid]),
                "p_value": float(valid_p[idx_in_valid]),
                "direction": "destabilizing" if valid_r[idx_in_valid] > 0 else "stabilizing",
            })

    result = pd.DataFrame(edges)
    if len(result) > 0:
        result = result.sort_values("weight", key=abs, ascending=False).reset_index(drop=True)

    return result


def validate_rbp_targets(adata, dataset_name, network_df):
    """For each hub RBP, test enrichment of its predicted targets among
    perturbation-responsive genes."""
    print(f"\n{'='*60}")
    print(f"PERTURBATION VALIDATION: {dataset_name}")
    print(f"{'='*60}")

    rbp_counts = network_df.groupby("regulator").size().sort_values(ascending=False)
    top_rbps = rbp_counts.head(15).index.tolist()
    print(f"  Top RBP hubs: {top_rbps[:10]}")

    results = []

    for rbp in top_rbps:
        rbp_upper = rbp.upper()

        rbp_edges = network_df[network_df["regulator"] == rbp_upper]
        predicted_targets = set(rbp_edges["target"].str.upper())
        predicted_destab = set(
            rbp_edges[rbp_edges["direction"] == "destabilizing"]["target"].str.upper()
        )
        predicted_stab = set(
            rbp_edges[rbp_edges["direction"] == "stabilizing"]["target"].str.upper()
        )

        n_targets = len(predicted_targets)
        if n_targets < 5:
            continue

        perturb_result = load_perturb_seq_de(rbp_upper)
        if perturb_result is not None:
            up_genes, down_genes = perturb_result
            up_set = set(g.upper() for g in up_genes)
            down_set = set(g.upper() for g in down_genes)

            all_genes = set(g.upper() for g in adata.var_names)

            # Destabilizing targets should be upregulated upon RBP knockdown
            if len(predicted_destab) > 0 and len(up_set) > 0:
                overlap_destab_up = len(predicted_destab & up_set)
                destab_not_up = len(predicted_destab - up_set)
                up_not_destab = len(up_set - predicted_destab)
                neither = len(all_genes - predicted_destab - up_set)

                table = [[overlap_destab_up, destab_not_up],
                         [up_not_destab, neither]]
                odds_ratio, fisher_p = stats.fisher_exact(table,
                                                          alternative="greater")

                print(f"\n  {rbp_upper} (Perturb-seq CRISPRi):")
                print(f"    Predicted destab targets: {len(predicted_destab)}")
                print(f"    Genes up upon KD: {len(up_set)}")
                print(f"    Overlap: {overlap_destab_up}")
                print(f"    Fisher OR={odds_ratio:.2f}, p={fisher_p:.3e}")

                results.append({
                    "rbp": rbp_upper,
                    "dataset": dataset_name,
                    "validation": "Perturb-seq_CRISPRi",
                    "n_predicted_targets": n_targets,
                    "n_predicted_destab": len(predicted_destab),
                    "n_predicted_stab": len(predicted_stab),
                    "n_perturbation_up": len(up_set),
                    "n_perturbation_down": len(down_set),
                    "overlap_destab_up": overlap_destab_up,
                    "fisher_or": float(odds_ratio),
                    "fisher_p": float(fisher_p),
                })

            # Stabilizing targets should be downregulated upon RBP knockdown
            if len(predicted_stab) > 0 and len(down_set) > 0:
                overlap_stab_down = len(predicted_stab & down_set)
                stab_not_down = len(predicted_stab - down_set)
                down_not_stab = len(down_set - predicted_stab)
                neither2 = len(all_genes - predicted_stab - down_set)

                table2 = [[overlap_stab_down, stab_not_down],
                           [down_not_stab, neither2]]
                or2, p2 = stats.fisher_exact(table2, alternative="greater")

                print(f"    Stabilizing->down: overlap={overlap_stab_down}, "
                      f"OR={or2:.2f}, p={p2:.3e}")
        else:
            print(f"\n  {rbp_upper}: No Perturb-seq data available")

    return results


def run_network_and_validate(adata, dataset_name):
    """Run scPTR pipeline, infer correlation-based network, validate."""
    import copy
    adata = copy.deepcopy(adata)
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)

    # RBPs to test (those with Perturb-seq data + known RBP hubs)
    rbp_list = [
        "HNRNPA1", "YBX1", "ELAVL1", "SRSF3", "RBFOX2",
        "FUS", "HNRNPC", "DDX5", "MBNL1",
        # Additional common RBP hubs
        "HNRNPD", "TRA2B", "ZFP36L1", "RBFOX1", "RBFOX3",
        "CELF2", "ELAVL3", "MATR3", "MBNL2", "PTBP1",
        # Mouse gene name variants
        "Hnrnpa1", "Ybx1", "Elavl1", "Srsf3", "Rbfox2",
        "Fus", "Hnrnpc", "Ddx5", "Mbnl1", "Hnrnpd",
        "Tra2b", "Zfp36l1", "Rbfox1", "Rbfox3", "Celf2",
        "Elavl3", "Matr3", "Mbnl2", "Ptbp1",
    ]

    print(f"  Inferring correlation network for {len(rbp_list)} candidate RBPs...")
    net_df = infer_spearman_network(adata, rbp_list, n_top_targets=200)

    if len(net_df) == 0:
        print(f"  No network edges for {dataset_name}")
        return []

    print(f"  Network: {len(net_df)} edges, "
          f"{net_df['regulator'].nunique()} regulators, "
          f"{net_df['target'].nunique()} targets")

    destab_frac = (net_df["direction"] == "destabilizing").mean()
    print(f"  Destabilizing fraction: {destab_frac:.1%}")

    return validate_rbp_targets(adata, dataset_name, net_df)


def main():
    set_figure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    adata_pan = scptr.datasets.pancreas()
    adata_dg = scptr.datasets.dentate_gyrus()

    all_results = []

    print("\n" + "#" * 60)
    print("# PANCREAS")
    print("#" * 60)
    results_pan = run_network_and_validate(adata_pan, "pancreas")
    all_results.extend(results_pan)

    print("\n" + "#" * 60)
    print("# DENTATE GYRUS")
    print("#" * 60)
    results_dg = run_network_and_validate(adata_dg, "dentate_gyrus")
    all_results.extend(results_dg)

    # Save results
    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(res_dir / "perturbation_validation.csv", index=False)

        # FDR correction across all tests
        from statsmodels.stats.multitest import multipletests
        _, fdr, _, _ = multipletests(results_df["fisher_p"], method="fdr_bh")
        results_df["fdr"] = fdr

        # Summary
        print(f"\n{'='*60}")
        print("PERTURBATION VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"  Total tests: {len(results_df)}")
        print(f"  Significant (p<0.05): {(results_df['fisher_p'] < 0.05).sum()}")
        print(f"  Significant (FDR<0.10): {(results_df['fdr'] < 0.10).sum()}")
        print(f"  Mean odds ratio: {results_df['fisher_or'].mean():.2f}")
        print(f"  Median odds ratio: {results_df['fisher_or'].median():.2f}")

        print(f"\n  Per-RBP results:")
        for _, row in results_df.sort_values("fisher_p").iterrows():
            sig = ("***" if row["fisher_p"] < 0.001 else
                   "**" if row["fisher_p"] < 0.01 else
                   "*" if row["fisher_p"] < 0.05 else "")
            print(f"    {row['rbp']:>10s} ({row['dataset']:>12s}): "
                  f"OR={row['fisher_or']:6.2f}  p={row['fisher_p']:.3e}  "
                  f"overlap={row['overlap_destab_up']:3d}/{row['n_predicted_destab']:3d} {sig}")

        # Figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        rbps = results_df["rbp"].values
        ors = results_df["fisher_or"].values
        ps = results_df["fisher_p"].values
        colors = ["red" if p < 0.05 else "gray" for p in ps]

        y_pos = np.arange(len(rbps))
        axes[0].barh(y_pos, np.log2(ors + 0.01), color=colors, edgecolor="black",
                     linewidth=0.5)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels([f"{r} ({d[:3]})" for r, d in
                                 zip(rbps, results_df["dataset"])], fontsize=8)
        axes[0].axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        axes[0].set_xlabel("log2(Odds Ratio)")
        axes[0].set_title("scPTR Target Enrichment in\nPerturb-seq DE Genes")

        axes[1].barh(y_pos, -np.log10(ps), color=colors, edgecolor="black",
                     linewidth=0.5)
        axes[1].axvline(x=-np.log10(0.05), color="blue", linestyle="--",
                        alpha=0.5, label="p=0.05")
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels([f"{r} ({d[:3]})" for r, d in
                                 zip(rbps, results_df["dataset"])], fontsize=8)
        axes[1].set_xlabel("-log10(p)")
        axes[1].set_title("Significance of Enrichment")
        axes[1].legend()

        fig.tight_layout()
        save_fig(fig, "perturbation_validation")

        results_df.to_csv(res_dir / "perturbation_validation.csv", index=False)

        summary = {
            "n_tests": len(results_df),
            "n_sig_005": int((results_df["fisher_p"] < 0.05).sum()),
            "n_sig_fdr_010": int((results_df["fdr"] < 0.10).sum()),
            "mean_or": float(results_df["fisher_or"].mean()),
            "median_or": float(results_df["fisher_or"].median()),
        }
        with open(res_dir / "perturbation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    else:
        print("  No perturbation validation results obtained.")

    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
