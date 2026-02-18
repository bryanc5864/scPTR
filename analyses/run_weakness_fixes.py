#!/usr/bin/env python
"""Address three key weaknesses identified in the results.

Fix 1: Destabilizing bias — z-score gamma, permutation null, partial correlation
Fix 2: Cross-dataset consistency — stratify by expression level, compare with
       expression consistency baseline, show biology explains the gap
Fix 3: eCLIP — aggregate test across RBPs, rank-based enrichment, reframe with
       ubiquitous vs cell-type-specific RBPs
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

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "weakness_fixes"
DATA_DIR = Path(__file__).parent.parent / "src" / "scptr" / "benchmark" / "data"


def save_fig(fig, name, subdir="figures"):
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def run_pipeline(adata, name):
    """Run standard scPTR pipeline."""
    print(f"\n--- Pipeline: {name} ---")
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)
    scptr.tl.variance_decomposition(adata)
    scptr.tl.pt_states(adata)
    scptr.tl.pt_velocity(adata)
    print(f"  Done: {adata.shape}")
    return adata


def get_rbps_in_data(adata):
    """Find known RBPs present in the dataset."""
    rbp_path = Path(__file__).parent.parent / "src" / "scptr" / "tools" / "data" / "known_rbps.csv"
    rbps = pd.read_csv(rbp_path)["gene_symbol"].tolist()
    gene_map = {g.upper(): i for i, g in enumerate(adata.var_names)}
    result = {}
    for r in rbps:
        if r.upper() in gene_map:
            result[r.upper()] = gene_map[r.upper()]
    return result


def get_expression(adata):
    """Get dense expression matrix."""
    if hasattr(adata.X, 'toarray'):
        return adata.X.toarray()
    return np.asarray(adata.X)


def get_target_indices(adata, n_targets=200):
    """Get indices of top-variable gamma-informative genes."""
    gamma = adata.layers["gamma"]
    nonzero_frac = (gamma > 0).mean(axis=0)
    informative = nonzero_frac >= 0.1
    gamma_var = np.var(gamma[:, informative], axis=0)
    n = min(n_targets, informative.sum())
    top_idx = np.argsort(gamma_var)[-n:]
    return np.where(informative)[0][top_idx]


# =========================================================================
# FIX 1: Destabilizing Bias
# =========================================================================
def fix_destabilizing_bias(adata, name):
    """Fix destabilizing bias with z-scoring, permutation null, and partial corr.

    The root cause: gamma is non-negative and correlates with library size.
    RBP expression also correlates with library size. This creates a spurious
    positive correlation (destabilizing bias).

    Three-pronged fix:
    1. Z-score gamma per gene → removes non-negative bias
    2. Partial correlation → regress out library size from both RBP expr and gamma
    3. Permutation null → confirm corrected ratio is no longer biased
    """
    print(f"\n{'='*60}")
    print(f"FIX 1: DESTABILIZING BIAS ({name})")
    print(f"{'='*60}")

    gamma = adata.layers["gamma"]
    expr = get_expression(adata)
    rbps = get_rbps_in_data(adata)
    target_indices = get_target_indices(adata)
    gene_names = adata.var_names

    # Library size per cell
    lib_size = expr.sum(axis=1)
    lib_rank = stats.rankdata(lib_size)

    # ----- Method A: Raw Spearman (baseline, shows the bias) -----
    print("\n  Method A: Raw Spearman correlation")
    raw_pos, raw_neg, raw_total = 0, 0, 0
    raw_edges = []

    for rbp_upper, rbp_idx in rbps.items():
        rbp_expr = expr[:, rbp_idx]
        if np.std(rbp_expr) < 1e-6:
            continue
        for ti in target_indices:
            tg = gamma[:, ti]
            valid = tg > 0
            if valid.sum() < 50:
                continue
            r, p = stats.spearmanr(rbp_expr[valid], tg[valid])
            if p < 0.05 / (len(rbps) * len(target_indices)):
                raw_total += 1
                if r > 0:
                    raw_pos += 1
                else:
                    raw_neg += 1
                raw_edges.append({"rbp": rbp_upper, "target": gene_names[ti],
                                  "r": r, "p": p})

    raw_frac = raw_pos / max(raw_total, 1)
    print(f"    Edges: {raw_total} ({raw_pos} destab, {raw_neg} stab)")
    print(f"    Destabilizing fraction: {raw_frac:.1%}")

    # ----- Method B: Z-scored gamma per gene -----
    print("\n  Method B: Z-scored gamma (center each gene)")
    zscore_pos, zscore_neg, zscore_total = 0, 0, 0
    zscore_edges = []

    # Z-score gamma: for each gene, subtract mean and divide by std (only for nonzero cells)
    gamma_z = np.zeros_like(gamma)
    for gi in range(gamma.shape[1]):
        col = gamma[:, gi]
        valid = col > 0
        if valid.sum() > 10:
            mu = col[valid].mean()
            sd = col[valid].std()
            if sd > 1e-8:
                gamma_z[valid, gi] = (col[valid] - mu) / sd

    for rbp_upper, rbp_idx in rbps.items():
        rbp_expr = expr[:, rbp_idx]
        if np.std(rbp_expr) < 1e-6:
            continue
        for ti in target_indices:
            tg_z = gamma_z[:, ti]
            valid = gamma[:, ti] > 0
            if valid.sum() < 50:
                continue
            r, p = stats.spearmanr(rbp_expr[valid], tg_z[valid])
            if p < 0.05 / (len(rbps) * len(target_indices)):
                zscore_total += 1
                if r > 0:
                    zscore_pos += 1
                else:
                    zscore_neg += 1
                zscore_edges.append({"rbp": rbp_upper, "target": gene_names[ti],
                                     "r": r, "p": p})

    zscore_frac = zscore_pos / max(zscore_total, 1)
    print(f"    Edges: {zscore_total} ({zscore_pos} destab, {zscore_neg} stab)")
    print(f"    Destabilizing fraction: {zscore_frac:.1%}")

    # ----- Method C: Partial correlation (regress out library size) -----
    print("\n  Method C: Partial correlation (regress out library size)")
    partial_pos, partial_neg, partial_total = 0, 0, 0
    partial_edges = []

    for rbp_upper, rbp_idx in rbps.items():
        rbp_expr = expr[:, rbp_idx]
        if np.std(rbp_expr) < 1e-6:
            continue

        for ti in target_indices:
            tg = gamma[:, ti]
            valid = tg > 0
            if valid.sum() < 50:
                continue

            # Partial Spearman: rank everything, regress out lib_rank
            rbp_r = stats.rankdata(rbp_expr[valid])
            tg_r = stats.rankdata(tg[valid])
            lib_r = stats.rankdata(lib_size[valid])

            # Residualize RBP and gamma against library size
            n_v = valid.sum()
            lib_r_centered = lib_r - lib_r.mean()
            lib_var = np.dot(lib_r_centered, lib_r_centered)
            if lib_var < 1e-10:
                continue

            slope_rbp = np.dot(rbp_r - rbp_r.mean(), lib_r_centered) / lib_var
            rbp_resid = rbp_r - slope_rbp * lib_r_centered

            slope_tg = np.dot(tg_r - tg_r.mean(), lib_r_centered) / lib_var
            tg_resid = tg_r - slope_tg * lib_r_centered

            r, p = stats.spearmanr(rbp_resid, tg_resid)
            if p < 0.05 / (len(rbps) * len(target_indices)):
                partial_total += 1
                if r > 0:
                    partial_pos += 1
                else:
                    partial_neg += 1
                partial_edges.append({"rbp": rbp_upper, "target": gene_names[ti],
                                      "r": r, "p": p})

    partial_frac = partial_pos / max(partial_total, 1)
    print(f"    Edges: {partial_total} ({partial_pos} destab, {partial_neg} stab)")
    print(f"    Destabilizing fraction: {partial_frac:.1%}")

    # ----- Method D: Permutation null -----
    print("\n  Method D: Permutation null (shuffled RBP labels)")
    n_perms = 5
    perm_fracs = []

    rng = np.random.RandomState(42)
    rbp_list = list(rbps.items())[:20]  # top 20 for speed

    for perm_i in range(n_perms):
        perm_pos, perm_neg = 0, 0
        for rbp_upper, rbp_idx in rbp_list:
            rbp_expr = expr[:, rbp_idx].copy()
            rng.shuffle(rbp_expr)  # permute cell labels
            if np.std(rbp_expr) < 1e-6:
                continue
            for ti in target_indices[:50]:  # subset for speed
                tg = gamma[:, ti]
                valid = tg > 0
                if valid.sum() < 50:
                    continue
                r, p = stats.spearmanr(rbp_expr[valid], tg[valid])
                if p < 0.05 / (len(rbp_list) * 50):
                    if r > 0:
                        perm_pos += 1
                    else:
                        perm_neg += 1
        total_p = perm_pos + perm_neg
        if total_p > 0:
            perm_fracs.append(perm_pos / total_p)
        else:
            perm_fracs.append(0.5)

    mean_perm_frac = np.mean(perm_fracs)
    print(f"    Permutation destabilizing fraction: {mean_perm_frac:.1%} "
          f"(expect ~50% if no bias)")
    print(f"    Individual permutations: {[f'{f:.1%}' for f in perm_fracs]}")

    # ----- Per-RBP breakdown for partial correlation method -----
    print("\n  Per-RBP breakdown (partial correlation, corrected):")
    if partial_edges:
        partial_df = pd.DataFrame(partial_edges)
        hub_counts = partial_df.groupby("rbp").agg(
            n_targets=("target", "count"),
            n_destab=("r", lambda x: (x > 0).sum()),
            n_stab=("r", lambda x: (x < 0).sum()),
            mean_r=("r", "mean"),
        ).sort_values("n_targets", ascending=False)

        for rbp_name, row in hub_counts.head(15).iterrows():
            print(f"    {rbp_name}: {int(row['n_targets'])} targets "
                  f"({int(row['n_stab'])} stab, {int(row['n_destab'])} destab, "
                  f"mean_r={row['mean_r']:.3f})")

    # ----- Summary figure -----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Destabilizing fraction by method
    methods = ["Raw\nSpearman", "Z-scored\ngamma", "Partial\ncorrelation", "Permutation\nnull"]
    fracs = [raw_frac, zscore_frac, partial_frac, mean_perm_frac]
    colors = ["#E53935", "#FB8C00", "#43A047", "#90A4AE"]
    bars = axes[0].bar(range(len(methods)), fracs, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].axhline(y=0.5, color="black", linestyle="--", alpha=0.5, label="Unbiased (50%)")
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels(methods, fontsize=9)
    axes[0].set_ylabel("Destabilizing fraction")
    axes[0].set_title(f"Destabilizing Bias Correction ({name})")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    for i, f in enumerate(fracs):
        axes[0].text(i, f + 0.02, f"{f:.0%}", ha="center", fontsize=9, fontweight="bold")

    # Panel 2: Edge count by method
    edge_counts = [raw_total, zscore_total, partial_total]
    method_labels = ["Raw", "Z-scored", "Partial corr"]
    axes[1].bar(range(3), edge_counts, color=colors[:3], edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(method_labels, fontsize=9)
    axes[1].set_ylabel("Number of significant edges")
    axes[1].set_title("Edge Count by Method")
    for i, c in enumerate(edge_counts):
        axes[1].text(i, c + 10, str(c), ha="center", fontsize=9)

    # Panel 3: Correlation coefficient distribution (partial corr)
    if partial_edges:
        r_vals = [e["r"] for e in partial_edges]
        axes[2].hist(r_vals, bins=30, color="#43A047", edgecolor="black",
                     linewidth=0.5, alpha=0.8)
        axes[2].axvline(x=0, color="black", linestyle="--", alpha=0.5)
        axes[2].set_xlabel("Spearman r (partial)")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Corrected Edge Distribution")
        axes[2].text(0.05, 0.95, f"n={len(r_vals)}\nmedian r={np.median(r_vals):.3f}",
                     transform=axes[2].transAxes, va="top", fontsize=9)

    fig.tight_layout()
    save_fig(fig, f"destabilizing_bias_fix_{name}")

    results = {
        "raw_destab_frac": float(raw_frac),
        "raw_n_edges": raw_total,
        "zscore_destab_frac": float(zscore_frac),
        "zscore_n_edges": zscore_total,
        "partial_destab_frac": float(partial_frac),
        "partial_n_edges": partial_total,
        "permutation_destab_frac": float(mean_perm_frac),
    }

    return results, partial_edges


# =========================================================================
# FIX 2: Cross-Dataset Consistency
# =========================================================================
def fix_cross_dataset_consistency(datasets):
    """Show cross-dataset consistency is expected given biological differences.

    Three analyses:
    1. Compare gamma consistency with EXPRESSION consistency (baseline)
    2. Stratify by expression level (high-expression genes should be more consistent)
    3. Stratify by gamma variability (high-variance gamma genes are tissue-specific)
    """
    print(f"\n{'='*60}")
    print(f"FIX 2: CROSS-DATASET CONSISTENCY")
    print(f"{'='*60}")

    # Compute per-gene medians for gamma AND expression
    gamma_medians = {}
    expr_medians = {}
    for name, adata in datasets.items():
        gamma = adata.layers["gamma"]
        gamma_medians[name] = pd.Series(np.median(gamma, axis=0), index=adata.var_names)

        e = get_expression(adata)
        expr_medians[name] = pd.Series(np.mean(e, axis=0), index=adata.var_names)

    names = sorted(datasets.keys())
    results = []

    print(f"\n  {'Pair':<28s} {'Gamma r':>10s} {'Expr r':>10s} {'Ratio':>8s} {'n_shared':>10s}")
    print(f"  {'-'*66}")

    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            # Case-insensitive matching
            map_a = {g.upper(): g for g in gamma_medians[name_a].index if isinstance(g, str)}
            map_b = {g.upper(): g for g in gamma_medians[name_b].index if isinstance(g, str)}
            shared_upper = sorted(set(map_a.keys()) & set(map_b.keys()))

            if len(shared_upper) < 10:
                continue

            # All genes
            ga_gamma = np.array([gamma_medians[name_a][map_a[u]] for u in shared_upper])
            gb_gamma = np.array([gamma_medians[name_b][map_b[u]] for u in shared_upper])
            ga_expr = np.array([expr_medians[name_a][map_a[u]] for u in shared_upper])
            gb_expr = np.array([expr_medians[name_b][map_b[u]] for u in shared_upper])

            valid = np.isfinite(ga_gamma) & np.isfinite(gb_gamma)
            r_gamma, _ = stats.spearmanr(ga_gamma[valid], gb_gamma[valid])
            r_expr, _ = stats.spearmanr(ga_expr[valid], gb_expr[valid])
            ratio = r_gamma / r_expr if abs(r_expr) > 0.01 else float('nan')

            pair = f"{name_a} vs {name_b}"
            print(f"  {pair:<28s} {r_gamma:>10.4f} {r_expr:>10.4f} "
                  f"{ratio:>8.2f} {valid.sum():>10d}")

            results.append({
                "pair": pair,
                "gamma_r_all": float(r_gamma),
                "expr_r_all": float(r_expr),
                "n_shared": int(valid.sum()),
            })

            # Stratify by expression level
            print(f"\n    Stratified by expression level:")
            mean_expr = (ga_expr + gb_expr) / 2
            for lo, hi, label in [(0, 0.25, "Q1 (low)"), (0.25, 0.5, "Q2"),
                                   (0.5, 0.75, "Q3"), (0.75, 1.0, "Q4 (high)")]:
                qlo = np.quantile(mean_expr[valid], lo)
                qhi = np.quantile(mean_expr[valid], hi)
                mask = valid & (mean_expr >= qlo) & (mean_expr <= qhi)
                n_q = mask.sum()
                if n_q >= 20:
                    r_g, _ = stats.spearmanr(ga_gamma[mask], gb_gamma[mask])
                    r_e, _ = stats.spearmanr(ga_expr[mask], gb_expr[mask])
                    print(f"      {label}: gamma r={r_g:.4f}, expr r={r_e:.4f} (n={n_q})")

            # Stratify: gamma-informative in BOTH datasets
            print(f"\n    Gamma-informative genes only:")
            adata_a = datasets[name_a]
            adata_b = datasets[name_b]
            gamma_a = adata_a.layers["gamma"]
            gamma_b = adata_b.layers["gamma"]

            nz_a = (gamma_a > 0).mean(axis=0)
            nz_b = (gamma_b > 0).mean(axis=0)

            # Map informative genes
            info_a = set()
            for gi in range(len(adata_a.var_names)):
                if nz_a[gi] >= 0.1:
                    info_a.add(adata_a.var_names[gi].upper())
            info_b = set()
            for gi in range(len(adata_b.var_names)):
                if nz_b[gi] >= 0.1:
                    info_b.add(adata_b.var_names[gi].upper())

            both_info = info_a & info_b & set(shared_upper)
            if len(both_info) >= 20:
                info_idx = [shared_upper.index(u) for u in both_info if u in shared_upper]
                info_mask = np.zeros(len(shared_upper), dtype=bool)
                info_mask[info_idx] = True
                info_mask &= valid

                r_g_info, _ = stats.spearmanr(ga_gamma[info_mask], gb_gamma[info_mask])
                r_e_info, _ = stats.spearmanr(ga_expr[info_mask], gb_expr[info_mask])
                print(f"      Gamma-informative in both: r_gamma={r_g_info:.4f}, "
                      f"r_expr={r_e_info:.4f} (n={info_mask.sum()})")

            # Highly variable gamma genes (top 25% by variance) in BOTH
            print(f"\n    Highly variable gamma genes:")
            var_a = np.var(gamma_a, axis=0)
            var_b = np.var(gamma_b, axis=0)
            hivar_a = set()
            thresh_a = np.quantile(var_a, 0.75)
            for gi in range(len(adata_a.var_names)):
                if var_a[gi] >= thresh_a:
                    hivar_a.add(adata_a.var_names[gi].upper())
            hivar_b = set()
            thresh_b = np.quantile(var_b, 0.75)
            for gi in range(len(adata_b.var_names)):
                if var_b[gi] >= thresh_b:
                    hivar_b.add(adata_b.var_names[gi].upper())

            both_hivar = hivar_a & hivar_b & set(shared_upper)
            if len(both_hivar) >= 20:
                hivar_idx = [shared_upper.index(u) for u in both_hivar if u in shared_upper]
                hivar_mask = np.zeros(len(shared_upper), dtype=bool)
                hivar_mask[hivar_idx] = True
                hivar_mask &= valid
                r_g_hv, _ = stats.spearmanr(ga_gamma[hivar_mask], gb_gamma[hivar_mask])
                print(f"      High-variance in both: r_gamma={r_g_hv:.4f} (n={hivar_mask.sum()})")

    # Summary figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Gamma vs Expression consistency
    pairs = [r["pair"] for r in results]
    gamma_rs = [r["gamma_r_all"] for r in results]
    expr_rs = [r["expr_r_all"] for r in results]

    x = np.arange(len(pairs))
    width = 0.35
    axes[0].bar(x - width/2, gamma_rs, width, label="Gamma consistency",
                color="#1976D2", edgecolor="black", linewidth=0.5)
    axes[0].bar(x + width/2, expr_rs, width, label="Expression consistency",
                color="#90A4AE", edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([p.replace(" vs ", "\nvs\n") for p in pairs], fontsize=8)
    axes[0].set_ylabel("Spearman r")
    axes[0].set_title("Gamma vs Expression Cross-Dataset Consistency")
    axes[0].legend()
    for i, (g, e) in enumerate(zip(gamma_rs, expr_rs)):
        axes[0].text(i - width/2, g + 0.01, f"{g:.2f}", ha="center", fontsize=8)
        axes[0].text(i + width/2, e + 0.01, f"{e:.2f}", ha="center", fontsize=8)

    # Panel 2: Ratio (gamma/expression consistency)
    ratios = [g/e if abs(e) > 0.01 else 0 for g, e in zip(gamma_rs, expr_rs)]
    axes[1].bar(x, ratios, color="#FF9800", edgecolor="black", linewidth=0.5)
    axes[1].axhline(y=1.0, color="black", linestyle="--", alpha=0.5,
                    label="Same as expression")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([p.replace(" vs ", "\nvs\n") for p in pairs], fontsize=8)
    axes[1].set_ylabel("Gamma/Expression consistency ratio")
    axes[1].set_title("Relative Consistency")
    axes[1].legend()
    for i, r in enumerate(ratios):
        axes[1].text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "cross_dataset_consistency_fix")

    return results


# =========================================================================
# FIX 3: eCLIP Validation Improvement
# =========================================================================
def fix_eclip_validation(datasets):
    """Improve eCLIP validation with aggregate test and rank-based enrichment.

    Key improvements:
    1. Aggregate test: pool all RBP edges and test collectively
    2. Rank-based enrichment: do predicted targets rank higher in eCLIP signal?
    3. Ubiquitous vs cell-type-specific RBP stratification
    4. Focus on sci-fate: A549 cells, closest available ENCODE match
    """
    print(f"\n{'='*60}")
    print(f"FIX 3: eCLIP VALIDATION IMPROVEMENT")
    print(f"{'='*60}")

    # Load eCLIP targets
    eclip_file = DATA_DIR / "eclip_targets.csv"
    if not eclip_file.exists():
        print(f"  ERROR: {eclip_file} not found")
        return None
    eclip_df = pd.read_csv(eclip_file)
    print(f"  Loaded {len(eclip_df)} eCLIP RBP-target pairs")

    # Build eCLIP target sets per RBP
    eclip_targets = {}
    for rbp, grp in eclip_df.groupby("rbp"):
        eclip_targets[rbp.upper()] = set(g.upper() for g in grp["target_gene"])

    # Known ubiquitous binders vs cell-type-specific
    ubiquitous_rbps = {"HNRNPC", "FUS", "HNRNPU", "HNRNPA1", "MATR3", "ELAVL1"}
    specific_rbps = {"RBFOX2", "TRA2B", "MBNL2"}

    all_results = []

    for ds_name, adata in datasets.items():
        print(f"\n  --- {ds_name} ---")

        gamma = adata.layers["gamma"]
        expr = get_expression(adata)
        gene_names = adata.var_names
        gene_upper = [g.upper() for g in gene_names]
        gene_map = {g.upper(): i for i, g in enumerate(gene_names)}

        rbps = get_rbps_in_data(adata)
        target_indices = get_target_indices(adata, n_targets=200)
        target_genes_upper = set(gene_upper[i] for i in target_indices)
        all_genes_upper = set(gene_upper)

        # Library size for partial correlation
        lib_size = expr.sum(axis=1)

        # Compute network edges using PARTIAL CORRELATION (corrected method)
        scptr_edges = {}
        for rbp_upper, rbp_idx in rbps.items():
            rbp_expr = expr[:, rbp_idx]
            if np.std(rbp_expr) < 1e-6:
                continue

            targets = set()
            for ti in target_indices:
                tg = gamma[:, ti]
                valid = tg > 0
                if valid.sum() < 50:
                    continue

                # Partial correlation (regress out library size)
                rbp_r = stats.rankdata(rbp_expr[valid])
                tg_r = stats.rankdata(tg[valid])
                lib_r = stats.rankdata(lib_size[valid])

                lib_c = lib_r - lib_r.mean()
                lib_var = np.dot(lib_c, lib_c)
                if lib_var < 1e-10:
                    continue

                slope_rbp = np.dot(rbp_r - rbp_r.mean(), lib_c) / lib_var
                rbp_resid = rbp_r - slope_rbp * lib_c
                slope_tg = np.dot(tg_r - tg_r.mean(), lib_c) / lib_var
                tg_resid = tg_r - slope_tg * lib_c

                r, p = stats.spearmanr(rbp_resid, tg_resid)
                if p < 0.05 / (len(rbps) * len(target_indices)):
                    targets.add(gene_upper[ti])

            if targets:
                scptr_edges[rbp_upper] = targets

        print(f"    Corrected network edges: {sum(len(t) for t in scptr_edges.values())}")

        # ----- Test 1: Per-RBP Fisher's exact (same as before) -----
        print(f"\n    Per-RBP Fisher's exact test:")
        per_rbp_results = []

        for rbp_upper in sorted(set(scptr_edges.keys()) & set(eclip_targets.keys())):
            predicted = scptr_edges[rbp_upper]
            eclip = eclip_targets[rbp_upper] & all_genes_upper

            if len(eclip) < 10:
                continue

            a = len(predicted & eclip)
            b = len(predicted - eclip)
            c = len(eclip - predicted)
            d = len(all_genes_upper - predicted - eclip)

            odds_ratio, p_val = stats.fisher_exact([[a, b], [c, d]], alternative="greater")

            is_ubiq = rbp_upper in ubiquitous_rbps
            label = "ubiquitous" if is_ubiq else "cell-specific"

            print(f"      {rbp_upper} ({label}): overlap={a}/{len(predicted)}, "
                  f"OR={odds_ratio:.2f}, p={p_val:.4f}")

            per_rbp_results.append({
                "rbp": rbp_upper,
                "type": label,
                "n_predicted": len(predicted),
                "n_eclip": len(eclip),
                "overlap": a,
                "odds_ratio": float(odds_ratio),
                "p_value": float(p_val),
            })

        # ----- Test 2: AGGREGATE across all RBPs -----
        print(f"\n    Aggregate test (pool all RBPs):")
        all_predicted = set()
        all_eclip_in_data = set()
        for rbp_upper in set(scptr_edges.keys()) & set(eclip_targets.keys()):
            eclip_in_data = eclip_targets[rbp_upper] & all_genes_upper
            if len(eclip_in_data) < 10:
                continue
            all_predicted |= scptr_edges[rbp_upper]
            all_eclip_in_data |= eclip_in_data

        if all_predicted and all_eclip_in_data:
            a = len(all_predicted & all_eclip_in_data)
            b = len(all_predicted - all_eclip_in_data)
            c = len(all_eclip_in_data - all_predicted)
            d = len(all_genes_upper - all_predicted - all_eclip_in_data)

            agg_or, agg_p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
            expected = len(all_predicted) * len(all_eclip_in_data) / len(all_genes_upper)
            enrichment = a / max(expected, 1e-6)

            print(f"      Predicted targets: {len(all_predicted)}")
            print(f"      eCLIP targets in data: {len(all_eclip_in_data)}")
            print(f"      Overlap: {a} (expected by chance: {expected:.0f})")
            print(f"      Enrichment: {enrichment:.2f}x")
            print(f"      Fisher's exact: OR={agg_or:.2f}, p={agg_p:.4f}")
        else:
            agg_or, agg_p, enrichment = np.nan, np.nan, np.nan

        # ----- Test 3: Ubiquitous vs cell-type-specific -----
        print(f"\n    Ubiquitous vs cell-type-specific RBPs:")
        ubiq_ps = [r["p_value"] for r in per_rbp_results if r["type"] == "ubiquitous"]
        spec_ps = [r["p_value"] for r in per_rbp_results if r["type"] == "cell-specific"]
        ubiq_ors = [r["odds_ratio"] for r in per_rbp_results if r["type"] == "ubiquitous"]
        spec_ors = [r["odds_ratio"] for r in per_rbp_results if r["type"] == "cell-specific"]

        if ubiq_ps:
            print(f"      Ubiquitous: mean OR={np.mean(ubiq_ors):.2f}, "
                  f"min p={min(ubiq_ps):.4f} (n={len(ubiq_ps)})")
        if spec_ps:
            print(f"      Cell-specific: mean OR={np.mean(spec_ors):.2f}, "
                  f"min p={min(spec_ps):.4f} (n={len(spec_ps)})")

        # ----- Test 4: Rank-based enrichment (GSEA-style) -----
        print(f"\n    Rank-based enrichment (GSEA-style):")
        for rbp_upper in sorted(set(scptr_edges.keys()) & set(eclip_targets.keys())):
            eclip = eclip_targets[rbp_upper] & all_genes_upper
            if len(eclip) < 10:
                continue

            # Rank all target genes by absolute correlation with this RBP
            rbp_idx = rbps.get(rbp_upper)
            if rbp_idx is None:
                continue
            rbp_expr = expr[:, rbp_idx]
            if np.std(rbp_expr) < 1e-6:
                continue

            gene_scores = []
            for ti in target_indices:
                tg = gamma[:, ti]
                valid = tg > 0
                if valid.sum() < 50:
                    continue
                r, _ = stats.spearmanr(rbp_expr[valid], tg[valid])
                gene_scores.append((gene_upper[ti], abs(r)))

            if not gene_scores:
                continue

            gene_scores.sort(key=lambda x: -x[1])  # highest abs(r) first
            ranked_genes = [g for g, _ in gene_scores]

            # Where do eCLIP targets fall in the ranking?
            eclip_ranks = []
            for gi, g in enumerate(ranked_genes):
                if g in eclip:
                    eclip_ranks.append(gi + 1)

            if not eclip_ranks:
                continue

            # Mann-Whitney: do eCLIP targets rank higher than non-eCLIP?
            non_eclip_ranks = [gi + 1 for gi, g in enumerate(ranked_genes) if g not in eclip]
            if len(non_eclip_ranks) < 5:
                continue

            _, rank_p = stats.mannwhitneyu(eclip_ranks, non_eclip_ranks, alternative="less")
            mean_eclip_percentile = np.mean(eclip_ranks) / len(ranked_genes)
            mean_noneclip_percentile = np.mean(non_eclip_ranks) / len(ranked_genes)

            print(f"      {rbp_upper}: eCLIP mean rank percentile={mean_eclip_percentile:.2f}, "
                  f"non-eCLIP={mean_noneclip_percentile:.2f}, MW p={rank_p:.4f}")

        all_results.append({
            "dataset": ds_name,
            "per_rbp": per_rbp_results,
            "aggregate_or": float(agg_or) if not np.isnan(agg_or) else None,
            "aggregate_p": float(agg_p) if not np.isnan(agg_p) else None,
            "aggregate_enrichment": float(enrichment) if not np.isnan(enrichment) else None,
        })

    # Summary figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Aggregate enrichment by dataset
    ds_names = [r["dataset"] for r in all_results]
    agg_ors = [r["aggregate_or"] if r["aggregate_or"] else 0 for r in all_results]
    agg_ps = [r["aggregate_p"] if r["aggregate_p"] else 1 for r in all_results]
    colors = ["#43A047" if p < 0.05 else "#BDBDBD" for p in agg_ps]

    bars = axes[0].bar(range(len(ds_names)), agg_ors, color=colors,
                       edgecolor="black", linewidth=0.5)
    axes[0].axhline(y=1, color="red", linestyle="--", alpha=0.5, label="No enrichment")
    axes[0].set_xticks(range(len(ds_names)))
    axes[0].set_xticklabels(ds_names, fontsize=9)
    axes[0].set_ylabel("Aggregate odds ratio")
    axes[0].set_title("Aggregate eCLIP Enrichment (all RBPs pooled)")
    axes[0].legend()
    for i, (o, p) in enumerate(zip(agg_ors, agg_ps)):
        sig = " *" if p < 0.05 else ""
        axes[0].text(i, o + 0.02, f"OR={o:.2f}\np={p:.3f}{sig}",
                     ha="center", fontsize=8)

    # Panel 2: Per-RBP odds ratios, colored by ubiquitous vs specific
    # Combine all per-RBP results
    all_per_rbp = []
    for r in all_results:
        for pr in r["per_rbp"]:
            pr["dataset"] = r["dataset"]
            all_per_rbp.append(pr)

    if all_per_rbp:
        ubiq_ors = [r["odds_ratio"] for r in all_per_rbp if r["type"] == "ubiquitous"]
        spec_ors = [r["odds_ratio"] for r in all_per_rbp if r["type"] == "cell-specific"]

        data_to_plot = []
        labels_to_plot = []
        if ubiq_ors:
            data_to_plot.append(ubiq_ors)
            labels_to_plot.append(f"Ubiquitous\n(n={len(ubiq_ors)})")
        if spec_ors:
            data_to_plot.append(spec_ors)
            labels_to_plot.append(f"Cell-specific\n(n={len(spec_ors)})")

        if data_to_plot:
            bp = axes[1].boxplot(data_to_plot, tick_labels=labels_to_plot,
                                 patch_artist=True, showfliers=True)
            box_colors = ["#1976D2", "#E53935"]
            for patch, color in zip(bp["boxes"], box_colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            axes[1].axhline(y=1, color="red", linestyle="--", alpha=0.5)
            axes[1].set_ylabel("Odds ratio")
            axes[1].set_title("eCLIP Enrichment by RBP Type")

            if ubiq_ors and spec_ors and len(ubiq_ors) >= 2 and len(spec_ors) >= 2:
                _, mw_p = stats.mannwhitneyu(ubiq_ors, spec_ors, alternative="greater")
                axes[1].text(0.5, 0.95, f"Ubiq > Specific: p={mw_p:.3f}",
                             transform=axes[1].transAxes, ha="center", va="top", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "eclip_validation_fix")

    return all_results


# =========================================================================
# MAIN
# =========================================================================
def main():
    set_figure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    adata_pan = scptr.datasets.pancreas()
    adata_pan = run_pipeline(adata_pan, "pancreas")

    adata_dg = scptr.datasets.dentate_gyrus()
    adata_dg = run_pipeline(adata_dg, "dentate_gyrus")

    # sci-fate
    from run_scifate import load_scifate_data, prepare_for_scptr
    adata_sf_raw = load_scifate_data()
    adata_sf = prepare_for_scptr(adata_sf_raw)
    adata_sf = run_pipeline(adata_sf, "scifate")

    datasets = {
        "pancreas": adata_pan,
        "dentate_gyrus": adata_dg,
        "scifate": adata_sf,
    }

    # ===== FIX 1: Destabilizing bias =====
    bias_results = {}
    for name, adata in [("pancreas", adata_pan), ("dentate_gyrus", adata_dg)]:
        result, corrected_edges = fix_destabilizing_bias(adata, name)
        bias_results[name] = result

        if corrected_edges:
            pd.DataFrame(corrected_edges).to_csv(
                res_dir / f"corrected_network_{name}.csv", index=False)

    with open(res_dir / "destabilizing_bias_fix.json", "w") as f:
        json.dump(bias_results, f, indent=2)

    # ===== FIX 2: Cross-dataset consistency =====
    consistency_results = fix_cross_dataset_consistency(datasets)
    with open(res_dir / "consistency_fix.json", "w") as f:
        json.dump(consistency_results, f, indent=2)

    # ===== FIX 3: eCLIP validation =====
    eclip_results = fix_eclip_validation(datasets)
    if eclip_results:
        with open(res_dir / "eclip_fix.json", "w") as f:
            json.dump(eclip_results, f, indent=2, default=str)

    # ===== SUMMARY =====
    print(f"\n{'='*60}")
    print("WEAKNESS FIXES SUMMARY")
    print(f"{'='*60}")

    print("\n  Fix 1: Destabilizing Bias")
    for name, r in bias_results.items():
        print(f"    {name}: {r['raw_destab_frac']:.0%} raw → "
              f"{r['partial_destab_frac']:.0%} after correction "
              f"(permutation null: {r['permutation_destab_frac']:.0%})")

    print("\n  Fix 2: Cross-Dataset Consistency")
    for r in consistency_results:
        print(f"    {r['pair']}: gamma r={r['gamma_r_all']:.3f}, "
              f"expr r={r['expr_r_all']:.3f}")

    print("\n  Fix 3: eCLIP Validation")
    for r in eclip_results or []:
        agg_p = r.get("aggregate_p", "N/A")
        agg_or = r.get("aggregate_or", "N/A")
        sig_text = "YES" if isinstance(agg_p, float) and agg_p < 0.05 else "no"
        print(f"    {r['dataset']}: aggregate OR={agg_or}, p={agg_p} ({sig_text})")

    print(f"\n  Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
