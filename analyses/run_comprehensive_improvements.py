#!/usr/bin/env python
"""Comprehensive improvements addressing 5 remaining weaknesses.

Experiment A: Network Target GO Enrichment (Weakness #1 — network validation)
Experiment B: Pathway-Level Cross-Dataset Consistency (Weakness #4 — low gene-level r)
Experiment C: Gamma vs Raw u/s on Downstream Tasks (Weakness #2 — marginal advantage)
Experiment D: NB Network Split-Half Robustness (Weakness #3 — single patient)
Experiment E: Corrected vs Uncorrected Network Quality (Weakness #5 — destabilizing bias)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import hypergeom

sys.path.insert(0, str(Path(__file__).parent))
from _common import set_figure_style

import scptr

# Force unbuffered stdout for progress visibility
sys.stdout.reconfigure(line_buffering=True)

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "comprehensive_improvements"
CACHE_DIR = Path(__file__).parent.parent / ".cache"
WEAKNESS_DIR = Path(__file__).parent.parent / "output" / "weakness_fixes" / "results"
TIER3_DIR = Path(__file__).parent.parent / "output" / "tier3" / "results"


def load_go_library():
    """Load GO BP gene sets from local cache (no network calls)."""
    cache_file = CACHE_DIR / "go_bp_2023.json"
    if cache_file.exists():
        with open(cache_file) as f:
            go_lib = json.load(f)
        return go_lib

    # Fallback: try to download and cache
    try:
        import gseapy as gp
        go_lib = gp.get_library("GO_Biological_Process_2023")
        with open(cache_file, "w") as f:
            json.dump(go_lib, f)
        return go_lib
    except Exception as e:
        print(f"  Failed to load GO library: {e}")
        return None


def hypergeometric_enrichment(gene_list, go_lib, background_size, alpha=0.05):
    """Run local hypergeometric GO enrichment (no API calls).

    Returns list of (term, p_value, overlap, term_size) for significant terms.
    """
    gene_set = set(g.upper() for g in gene_list)
    k = len(gene_set)  # drawn genes
    N = background_size  # population size

    results = []
    for term_name, term_genes in go_lib.items():
        term_upper = set(g.upper() for g in term_genes)
        K = len(term_upper)  # successes in population
        if K < 5 or K > N * 0.5:  # skip very small or very large terms
            continue
        overlap = gene_set & term_upper
        x = len(overlap)
        if x < 2:
            continue
        # P(X >= x) under hypergeometric
        p_val = hypergeom.sf(x - 1, N, K, k)
        results.append((term_name, p_val, x, K))

    # BH correction
    if not results:
        return []
    results.sort(key=lambda r: r[1])
    n_tests = len(results)
    corrected = []
    for i, (term, p, overlap, size) in enumerate(results):
        adj_p = p * n_tests / (i + 1)
        corrected.append((term, adj_p, overlap, size))

    # Enforce monotonicity
    min_p = 1.0
    for i in range(len(corrected) - 1, -1, -1):
        min_p = min(min_p, corrected[i][1])
        corrected[i] = (corrected[i][0], min_p, corrected[i][2], corrected[i][3])

    sig = [(t, p, o, s) for t, p, o, s in corrected if p < alpha]
    return sig


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


def get_expression(adata):
    if hasattr(adata.X, 'toarray'):
        return adata.X.toarray()
    return np.asarray(adata.X)


def get_rbps_in_data(adata):
    rbp_path = Path(__file__).parent.parent / "src" / "scptr" / "tools" / "data" / "known_rbps.csv"
    rbps = pd.read_csv(rbp_path)["gene_symbol"].tolist()
    gene_map = {g.upper(): i for i, g in enumerate(adata.var_names)}
    result = {}
    for r in rbps:
        if r.upper() in gene_map:
            result[r.upper()] = gene_map[r.upper()]
    return result


def get_target_indices(adata, n_targets=200):
    gamma = adata.layers["gamma"]
    nonzero_frac = (gamma > 0).mean(axis=0)
    informative = nonzero_frac >= 0.1
    gamma_var = np.var(gamma[:, informative], axis=0)
    n = min(n_targets, informative.sum())
    top_idx = np.argsort(gamma_var)[-n:]
    return np.where(informative)[0][top_idx]


# =========================================================================
# EXPERIMENT A: Network Target GO Enrichment
# =========================================================================
def experiment_a_go_enrichment():
    """Test whether predicted RBP targets share biological functions (GO enrichment).

    Uses local hypergeometric tests with cached GO BP gene sets — no API calls.
    """
    print(f"\n{'='*60}")
    print("EXPERIMENT A: NETWORK TARGET GO ENRICHMENT")
    print(f"{'='*60}")

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load GO gene sets from local cache
    print("  Loading GO Biological Process gene sets (local cache)...")
    go_lib = load_go_library()
    if go_lib is None:
        return None
    print(f"  Loaded {len(go_lib)} GO BP terms")

    # Load corrected networks
    networks = {}
    network_files = {
        "pancreas": WEAKNESS_DIR / "corrected_network_pancreas.csv",
        "dentate_gyrus": WEAKNESS_DIR / "corrected_network_dentate_gyrus.csv",
        "neuroblastoma": TIER3_DIR / "neuroblastoma_network_corrected.csv",
    }

    for name, path in network_files.items():
        if path.exists():
            df = pd.read_csv(path)
            print(f"  {name}: {len(df)} edges")
            networks[name] = df
        else:
            print(f"  {name}: file not found at {path}")

    # Estimate background gene count per organism
    # Use ~20,000 as a reasonable genome-wide background
    BACKGROUND_SIZE = 20000

    all_results = {}

    for ds_name, edges_df in networks.items():
        print(f"\n  --- {ds_name} ---")

        rbp_col = "rbp"
        target_col = "target"

        # Build RBP -> target sets
        rbp_targets = {}
        for rbp, grp in edges_df.groupby(rbp_col):
            targets = set(grp[target_col].tolist())
            rbp_targets[rbp] = targets

        # Background gene set (all unique targets in network)
        all_targets = set()
        for t in rbp_targets.values():
            all_targets |= t

        # Filter to RBPs with >= 10 targets
        eligible_rbps = {r: t for r, t in rbp_targets.items() if len(t) >= 10}
        print(f"    RBPs with >= 10 targets: {len(eligible_rbps)}")

        if not eligible_rbps:
            all_results[ds_name] = {"n_eligible_rbps": 0}
            continue

        # Run local hypergeometric enrichment for each eligible RBP
        rbp_enrichment_results = []
        n_with_sig = 0

        for rbp, targets in eligible_rbps.items():
            gene_list = list(targets)
            sig_terms = hypergeometric_enrichment(gene_list, go_lib, BACKGROUND_SIZE)
            n_sig = len(sig_terms)
            has_sig = n_sig > 0
            if has_sig:
                n_with_sig += 1

            top_terms = [t[0] for t in sig_terms[:5]]

            rbp_enrichment_results.append({
                "rbp": rbp,
                "n_targets": len(targets),
                "n_sig_terms": n_sig,
                "has_sig": has_sig,
                "top_terms": top_terms,
            })

        frac_with_sig = n_with_sig / max(len(eligible_rbps), 1)
        print(f"    RBPs with >= 1 significant GO term: {n_with_sig}/{len(eligible_rbps)} ({frac_with_sig:.1%})")

        # Known biology concordance
        known_biology = {
            "ELAVL1": ["mRNA stability", "mRNA stabilization", "RNA stability"],
            "RBFOX1": ["neuron", "neuronal", "synap", "axon"],
            "RBFOX2": ["neuron", "neuronal", "synap", "splicing"],
            "RBFOX3": ["neuron", "neuronal", "synap"],
            "SRSF3": ["splic", "mRNA processing", "RNA processing"],
            "HNRNPA1": ["splic", "mRNA processing", "RNA processing"],
            "YBX1": ["translation", "mRNA", "RNA"],
            "CELF2": ["splic", "neuron", "mRNA"],
        }

        concordance_hits = []
        for rbp_res in rbp_enrichment_results:
            rbp = rbp_res["rbp"]
            if rbp in known_biology and rbp_res["top_terms"]:
                expected_keywords = known_biology[rbp]
                all_terms_str = " ".join(rbp_res["top_terms"]).lower()
                matched = [kw for kw in expected_keywords if kw.lower() in all_terms_str]
                if matched:
                    concordance_hits.append({"rbp": rbp, "matched_keywords": matched})
                    print(f"    Known biology match: {rbp} -> {matched}")

        # Cross-RBP specificity (Jaccard between enriched GO term sets)
        enriched_term_sets = {}
        for rbp_res in rbp_enrichment_results:
            if rbp_res["top_terms"]:
                enriched_term_sets[rbp_res["rbp"]] = set(rbp_res["top_terms"])

        jaccard_values = []
        rbp_list = list(enriched_term_sets.keys())
        for i in range(len(rbp_list)):
            for j in range(i + 1, len(rbp_list)):
                s1 = enriched_term_sets[rbp_list[i]]
                s2 = enriched_term_sets[rbp_list[j]]
                union = s1 | s2
                if union:
                    jaccard_values.append(len(s1 & s2) / len(union))

        mean_jaccard = np.mean(jaccard_values) if jaccard_values else 0
        print(f"    Cross-RBP GO term Jaccard (specificity): {mean_jaccard:.3f} (lower = more specific)")

        # Bootstrap null: random gene sets from GENOME-WIDE background
        # (not from network targets, which are already enriched for biology)
        print(f"    Running bootstrap null (100 random genome-wide sets per RBP)...")
        n_bootstrap = 100
        rng = np.random.RandomState(42)
        # Build genome-wide gene list from GO library (covers ~20K genes)
        genome_genes = set()
        for genes in go_lib.values():
            genome_genes.update(g.upper() for g in genes)
        genome_genes_list = sorted(genome_genes)
        bootstrap_fracs = []

        test_rbps = list(eligible_rbps.items())[:min(10, len(eligible_rbps))]
        for rbp, targets in test_rbps:
            n_t = len(targets)
            null_sig_count = 0
            for _ in range(n_bootstrap):
                random_genes = rng.choice(genome_genes_list,
                                          size=min(n_t, len(genome_genes_list)),
                                          replace=False).tolist()
                sig_null = hypergeometric_enrichment(random_genes, go_lib, BACKGROUND_SIZE)
                if sig_null:
                    null_sig_count += 1
            bootstrap_fracs.append(null_sig_count / n_bootstrap)

        mean_null_frac = np.mean(bootstrap_fracs) if bootstrap_fracs else 0
        print(f"    Bootstrap null fraction with sig GO term: {mean_null_frac:.3f}")
        print(f"    Enrichment over null: {frac_with_sig / max(mean_null_frac, 0.01):.1f}x")

        all_results[ds_name] = {
            "n_eligible_rbps": len(eligible_rbps),
            "n_with_sig_go": n_with_sig,
            "frac_with_sig_go": float(frac_with_sig),
            "mean_cross_rbp_jaccard": float(mean_jaccard),
            "n_known_biology_matches": len(concordance_hits),
            "concordance_hits": concordance_hits,
            "bootstrap_null_frac": float(mean_null_frac),
            "per_rbp": rbp_enrichment_results,
        }

    # Save results
    with open(res_dir / "go_enrichment.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary figure
    ds_names = list(all_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Fraction with significant GO terms
    fracs = [all_results[d].get("frac_with_sig_go", 0) for d in ds_names]
    null_fracs = [all_results[d].get("bootstrap_null_frac", 0) for d in ds_names]
    x = np.arange(len(ds_names))
    width = 0.35
    axes[0].bar(x - width / 2, fracs, width, label="Real RBP targets",
                color="#1976D2", edgecolor="black", linewidth=0.5)
    axes[0].bar(x + width / 2, null_fracs, width, label="Random gene sets (null)",
                color="#BDBDBD", edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ds_names, fontsize=9)
    axes[0].set_ylabel("Fraction with >= 1 sig GO term")
    axes[0].set_title("GO Enrichment: Real vs Random Targets")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1.1)
    for i, (f, n) in enumerate(zip(fracs, null_fracs)):
        axes[0].text(i - width / 2, f + 0.02, f"{f:.0%}", ha="center", fontsize=8)
        axes[0].text(i + width / 2, n + 0.02, f"{n:.0%}", ha="center", fontsize=8)

    # Panel 2: Cross-RBP Jaccard (specificity)
    jaccards = [all_results[d].get("mean_cross_rbp_jaccard", 0) for d in ds_names]
    axes[1].bar(x, jaccards, color="#43A047", edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ds_names, fontsize=9)
    axes[1].set_ylabel("Mean Jaccard (lower = more specific)")
    axes[1].set_title("Cross-RBP GO Term Specificity")
    for i, j in enumerate(jaccards):
        axes[1].text(i, j + 0.005, f"{j:.3f}", ha="center", fontsize=9)

    fig.suptitle("Experiment A: Network Target GO Enrichment", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "experiment_a_go_enrichment")

    # Print summary
    print(f"\n  EXPERIMENT A SUMMARY:")
    for ds_name, res in all_results.items():
        print(f"    {ds_name}: {res.get('frac_with_sig_go', 0):.0%} RBPs with sig GO terms "
              f"(null: {res.get('bootstrap_null_frac', 0):.0%}, "
              f"concordance: {res.get('n_known_biology_matches', 0)} hits)")

    return all_results


# =========================================================================
# EXPERIMENT B: Pathway-Level Cross-Dataset Consistency
# =========================================================================
def experiment_b_pathway_consistency(datasets):
    """Show pathway-level gamma consistency is higher than gene-level."""
    print(f"\n{'='*60}")
    print("EXPERIMENT B: PATHWAY-LEVEL CROSS-DATASET CONSISTENCY")
    print(f"{'='*60}")

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load GO gene sets from local cache
    print("  Loading GO Biological Process gene sets (local cache)...")
    go_lib = load_go_library()
    if go_lib is None:
        return None
    print(f"  Loaded {len(go_lib)} GO BP terms")

    # Compute per-gene median gamma for each dataset
    gamma_medians = {}
    for name, adata in datasets.items():
        gamma = adata.layers["gamma"]
        gamma_med = np.median(gamma, axis=0)
        gamma_medians[name] = pd.Series(gamma_med, index=[g.upper() for g in adata.var_names])

    names = sorted(datasets.keys())
    results = []

    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            print(f"\n  --- {name_a} vs {name_b} ---")

            ga = gamma_medians[name_a]
            gb = gamma_medians[name_b]

            # Shared genes
            shared = sorted(set(ga.index) & set(gb.index))
            if len(shared) < 50:
                continue

            # Gene-level correlation (baseline)
            ga_shared = ga[shared].values
            gb_shared = gb[shared].values
            valid = np.isfinite(ga_shared) & np.isfinite(gb_shared)
            r_gene, p_gene = stats.spearmanr(ga_shared[valid], gb_shared[valid])
            print(f"    Gene-level Spearman r: {r_gene:.4f} (n={valid.sum()})")

            # Pathway-level: for each GO term with >= 10 shared genes,
            # compute mean gamma in each dataset
            pathway_gamma_a = []
            pathway_gamma_b = []
            pathway_names = []
            pathway_sizes = []

            for term_name, term_genes in go_lib.items():
                # Convert term genes to uppercase for matching
                term_genes_upper = set(g.upper() for g in term_genes)
                term_shared = term_genes_upper & set(shared)

                if len(term_shared) < 10:
                    continue

                genes_list = sorted(term_shared)
                idx = [shared.index(g) for g in genes_list]

                mean_a = np.mean(ga_shared[idx])
                mean_b = np.mean(gb_shared[idx])

                if np.isfinite(mean_a) and np.isfinite(mean_b):
                    pathway_gamma_a.append(mean_a)
                    pathway_gamma_b.append(mean_b)
                    pathway_names.append(term_name)
                    pathway_sizes.append(len(term_shared))

            if len(pathway_gamma_a) < 20:
                print(f"    Too few pathways with >= 10 shared genes: {len(pathway_gamma_a)}")
                continue

            r_pathway, p_pathway = stats.spearmanr(pathway_gamma_a, pathway_gamma_b)
            print(f"    Pathway-level Spearman r: {r_pathway:.4f} (n={len(pathway_gamma_a)} pathways)")
            print(f"    Improvement: {r_pathway:.3f} vs {r_gene:.3f} (gene-level)")

            results.append({
                "pair": f"{name_a} vs {name_b}",
                "gene_level_r": float(r_gene),
                "gene_level_p": float(p_gene),
                "n_shared_genes": int(valid.sum()),
                "pathway_level_r": float(r_pathway),
                "pathway_level_p": float(p_pathway),
                "n_pathways": len(pathway_gamma_a),
                "mean_pathway_size": float(np.mean(pathway_sizes)),
            })

    # Save results
    with open(res_dir / "pathway_consistency.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary figure
    if results:
        fig, ax = plt.subplots(figsize=(8, 5))
        pairs = [r["pair"] for r in results]
        gene_rs = [r["gene_level_r"] for r in results]
        pathway_rs = [r["pathway_level_r"] for r in results]

        x = np.arange(len(pairs))
        width = 0.35
        ax.bar(x - width / 2, gene_rs, width, label="Gene-level",
               color="#E53935", edgecolor="black", linewidth=0.5)
        ax.bar(x + width / 2, pathway_rs, width, label="Pathway-level",
               color="#1976D2", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace(" vs ", "\nvs\n") for p in pairs], fontsize=8)
        ax.set_ylabel("Spearman r")
        ax.set_title("Gamma Consistency: Gene vs Pathway Level")
        ax.legend()
        for i, (g, p) in enumerate(zip(gene_rs, pathway_rs)):
            ax.text(i - width / 2, g + 0.01, f"{g:.3f}", ha="center", fontsize=8)
            ax.text(i + width / 2, p + 0.01, f"{p:.3f}", ha="center", fontsize=8)

        fig.tight_layout()
        save_fig(fig, "experiment_b_pathway_consistency")

    print(f"\n  EXPERIMENT B SUMMARY:")
    for r in results:
        print(f"    {r['pair']}: gene r={r['gene_level_r']:.3f} -> pathway r={r['pathway_level_r']:.3f} "
              f"({r['n_pathways']} pathways)")

    return results


# =========================================================================
# EXPERIMENT C: Gamma vs Raw u/s on Downstream Tasks
# =========================================================================
def experiment_c_gamma_advantage(datasets):
    """Demonstrate gamma's downstream task advantage over raw u/s ratio."""
    print(f"\n{'='*60}")
    print("EXPERIMENT C: GAMMA vs RAW U/S ON DOWNSTREAM TASKS")
    print(f"{'='*60}")

    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for ds_name, adata in datasets.items():
        if ds_name == "scifate":
            continue  # Only pancreas and DG have expression clusters for comparison
        print(f"\n  --- {ds_name} ---")

        gamma = adata.layers["gamma"]
        Ms = adata.layers["Ms"]
        Mu = adata.layers["Mu"]

        # Construct smooth_ratio: same as gamma but WITHOUT beta multiplication
        reliable = Ms >= 0.01
        smooth_ratio = np.where(reliable, Mu / np.where(reliable, Ms, 1.0), 0.0)

        # Same per-gene 99th percentile clip as gamma
        for gi in range(smooth_ratio.shape[1]):
            col = smooth_ratio[:, gi]
            pos = col[col > 0]
            if len(pos) > 10:
                cap = np.percentile(pos, 99)
                smooth_ratio[:, gi] = np.clip(col, 0, cap)

        # Global cap at 10x 99th percentile of gene medians
        gene_medians = np.median(smooth_ratio, axis=0)
        pos_medians = gene_medians[gene_medians > 0]
        if len(pos_medians) > 0:
            global_cap = 10 * np.percentile(pos_medians, 99)
            smooth_ratio = np.clip(smooth_ratio, 0, global_cap)

        print(f"    Gamma shape: {gamma.shape}, max={gamma.max():.4f}")
        print(f"    Smooth ratio shape: {smooth_ratio.shape}, max={smooth_ratio.max():.4f}")

        # Get expression clusters
        clusters = adata.obs.get("clusters", adata.obs.get("cell_type"))
        if clusters is None:
            print(f"    No cluster labels found, skipping")
            continue
        clusters = clusters.astype(str)

        # ----- Task 1: PT State Discovery (Invisible States) -----
        print(f"\n    Task 1: Invisible State Discovery")

        invisible_results = {"gamma": [], "smooth_ratio": []}

        for method_name, layer_data in [("gamma", gamma), ("smooth_ratio", smooth_ratio)]:
            for cluster_name in sorted(clusters.unique()):
                mask = (clusters == cluster_name).values
                n_cells = mask.sum()
                if n_cells < 50:
                    continue

                sub = layer_data[mask]
                n_pcs = min(15, n_cells - 1, sub.shape[1] - 1)
                pca = PCA(n_components=n_pcs, random_state=42)
                pcs = pca.fit_transform(sub)

                best_k, best_sil, best_labels = 1, -1, np.zeros(n_cells, dtype=int)
                for k in [2, 3]:
                    if n_cells < k * 10:
                        continue
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = km.fit_predict(pcs)
                    if min(np.bincount(labels)) < 10:
                        continue
                    sil = silhouette_score(pcs, labels)
                    if sil > best_sil:
                        best_k, best_sil, best_labels = k, sil, labels

                # Expression silhouette for same labels
                expr_sub = get_expression(adata)[mask]
                n_expr_pcs = min(15, n_cells - 1, expr_sub.shape[1] - 1)
                pca_expr = PCA(n_components=n_expr_pcs, random_state=42)
                expr_pcs = pca_expr.fit_transform(expr_sub)

                if best_k > 1:
                    sil_method = best_sil
                    sil_expr = silhouette_score(expr_pcs, best_labels)
                else:
                    sil_method = 0
                    sil_expr = 0

                is_invisible = sil_method > 0.1 and sil_expr < 0.1

                invisible_results[method_name].append({
                    "cluster": cluster_name,
                    "n_cells": n_cells,
                    "sil_method": float(sil_method),
                    "sil_expr": float(sil_expr),
                    "invisibility": float(sil_method - sil_expr),
                    "is_invisible": is_invisible,
                })

        # Count invisible states for each method
        gamma_invisible = sum(1 for r in invisible_results["gamma"] if r["is_invisible"])
        ratio_invisible = sum(1 for r in invisible_results["smooth_ratio"] if r["is_invisible"])
        gamma_mean_invis = np.mean([r["invisibility"] for r in invisible_results["gamma"]])
        ratio_mean_invis = np.mean([r["invisibility"] for r in invisible_results["smooth_ratio"]])

        print(f"    Gamma: {gamma_invisible} invisible states, mean invisibility={gamma_mean_invis:.3f}")
        print(f"    Smooth ratio: {ratio_invisible} invisible states, mean invisibility={ratio_mean_invis:.3f}")

        # ----- Task 2: Cell-Type Variance Explained (eta-squared) -----
        print(f"\n    Task 2: Cell-Type Variance Explained (eta-squared)")

        cluster_labels = clusters.values
        unique_clusters = np.unique(cluster_labels)

        def compute_eta_squared(data, labels, unique_labels):
            """Compute eta-squared (fraction of variance explained by groups)."""
            n = data.shape[0]
            grand_mean = data.mean(axis=0)
            ss_total = np.sum((data - grand_mean) ** 2, axis=0)

            ss_between = np.zeros(data.shape[1])
            for cl in unique_labels:
                mask_cl = labels == cl
                n_cl = mask_cl.sum()
                if n_cl == 0:
                    continue
                group_mean = data[mask_cl].mean(axis=0)
                ss_between += n_cl * (group_mean - grand_mean) ** 2

            eta_sq = ss_between / np.clip(ss_total, 1e-10, None)
            return eta_sq

        eta_gamma = compute_eta_squared(gamma, cluster_labels, unique_clusters)
        eta_ratio = compute_eta_squared(smooth_ratio, cluster_labels, unique_clusters)

        # Filter to informative genes
        informative = (gamma > 0).mean(axis=0) >= 0.1
        eta_gamma_info = eta_gamma[informative]
        eta_ratio_info = eta_ratio[informative]

        gamma_wins = (eta_gamma_info > eta_ratio_info).sum()
        ratio_wins = (eta_ratio_info > eta_gamma_info).sum()
        total = len(eta_gamma_info)

        print(f"    Gamma eta-sq > smooth ratio: {gamma_wins}/{total} ({100*gamma_wins/total:.1f}%)")
        print(f"    Mean eta-sq — gamma: {eta_gamma_info.mean():.4f}, smooth ratio: {eta_ratio_info.mean():.4f}")

        # Wilcoxon test
        w_stat, w_p = stats.wilcoxon(eta_gamma_info, eta_ratio_info)
        print(f"    Wilcoxon signed-rank p: {w_p:.2e}")

        all_results[ds_name] = {
            "invisible_states": {
                "gamma_n_invisible": gamma_invisible,
                "smooth_ratio_n_invisible": ratio_invisible,
                "gamma_mean_invisibility": float(gamma_mean_invis),
                "smooth_ratio_mean_invisibility": float(ratio_mean_invis),
                "per_cluster": invisible_results,
            },
            "eta_squared": {
                "gamma_wins": int(gamma_wins),
                "ratio_wins": int(ratio_wins),
                "n_genes": int(total),
                "gamma_mean": float(eta_gamma_info.mean()),
                "ratio_mean": float(eta_ratio_info.mean()),
                "wilcoxon_p": float(w_p),
            },
        }

    # Save results
    with open(res_dir / "gamma_advantage.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Invisible state counts
    ds_labels = list(all_results.keys())
    gamma_invis = [all_results[d]["invisible_states"]["gamma_n_invisible"] for d in ds_labels]
    ratio_invis = [all_results[d]["invisible_states"]["smooth_ratio_n_invisible"] for d in ds_labels]
    x = np.arange(len(ds_labels))
    width = 0.35
    axes[0].bar(x - width / 2, gamma_invis, width, label="scPTR gamma",
                color="#1976D2", edgecolor="black", linewidth=0.5)
    axes[0].bar(x + width / 2, ratio_invis, width, label="Smooth u/s ratio (no beta)",
                color="#E53935", edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ds_labels, fontsize=9)
    axes[0].set_ylabel("Number of invisible states")
    axes[0].set_title("Invisible State Discovery")
    axes[0].legend(fontsize=8)
    for i, (g, r) in enumerate(zip(gamma_invis, ratio_invis)):
        axes[0].text(i - width / 2, g + 0.1, str(g), ha="center", fontsize=9)
        axes[0].text(i + width / 2, r + 0.1, str(r), ha="center", fontsize=9)

    # Panel 2: Eta-squared comparison
    gamma_means = [all_results[d]["eta_squared"]["gamma_mean"] for d in ds_labels]
    ratio_means = [all_results[d]["eta_squared"]["ratio_mean"] for d in ds_labels]
    axes[1].bar(x - width / 2, gamma_means, width, label="scPTR gamma",
                color="#1976D2", edgecolor="black", linewidth=0.5)
    axes[1].bar(x + width / 2, ratio_means, width, label="Smooth u/s ratio",
                color="#E53935", edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ds_labels, fontsize=9)
    axes[1].set_ylabel("Mean eta-squared")
    axes[1].set_title("Cell-Type Variance Explained")
    axes[1].legend(fontsize=8)
    for i, (g, r) in enumerate(zip(gamma_means, ratio_means)):
        axes[1].text(i - width / 2, g + 0.001, f"{g:.4f}", ha="center", fontsize=8)
        axes[1].text(i + width / 2, r + 0.001, f"{r:.4f}", ha="center", fontsize=8)

    fig.suptitle("Experiment C: Gamma vs Smooth Ratio Downstream Tasks", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "experiment_c_gamma_advantage")

    print(f"\n  EXPERIMENT C SUMMARY:")
    for ds_name, res in all_results.items():
        inv = res["invisible_states"]
        eta = res["eta_squared"]
        print(f"    {ds_name}: invisible states gamma={inv['gamma_n_invisible']} "
              f"vs ratio={inv['smooth_ratio_n_invisible']}; "
              f"eta-sq gamma={eta['gamma_mean']:.4f} vs ratio={eta['ratio_mean']:.4f} "
              f"(p={eta['wilcoxon_p']:.2e})")

    return all_results


# =========================================================================
# EXPERIMENT D: NB Network Split-Half Robustness
# =========================================================================
def experiment_d_nb_robustness():
    """Show NB network is internally robust via split-half cross-validation."""
    print(f"\n{'='*60}")
    print("EXPERIMENT D: NB NETWORK SPLIT-HALF ROBUSTNESS")
    print(f"{'='*60}")

    import scanpy as sc

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load NB data
    h5ad_path = CACHE_DIR / "neuroblastoma.h5ad"
    if not h5ad_path.exists():
        print(f"  NB data not found at {h5ad_path}")
        return None

    print("  Loading neuroblastoma dataset...")
    adata_full = sc.read_h5ad(str(h5ad_path))
    sc.pp.filter_genes(adata_full, min_cells=50)
    adata_full.layers["raw_spliced"] = adata_full.layers["spliced"].copy()
    adata_full.layers["raw_unspliced"] = adata_full.layers["unspliced"].copy()
    print(f"  Full dataset: {adata_full.shape}")

    def run_nb_pipeline(adata):
        """Run scPTR pipeline on NB data."""
        scptr.pp.filter_genes(adata)
        scptr.pp.normalize_layers(adata)
        scptr.pp.neighbors(adata, n_neighbors=30)
        scptr.pp.smooth_layers(adata)
        scptr.tl.estimate_beta(adata)
        scptr.tl.estimate_gamma(adata)
        return adata

    def infer_network(adata):
        """Run partial-correlation network inference (library-size corrected)."""
        gamma = adata.layers["gamma"]
        expr = get_expression(adata)
        rbps = get_rbps_in_data(adata)
        n_cells = adata.n_obs

        # Library size
        lib_size = expr.sum(axis=1)
        lib_rank = stats.rankdata(lib_size)
        lib_rank_centered = lib_rank - lib_rank.mean()
        lib_ss = np.dot(lib_rank_centered, lib_rank_centered)

        if lib_ss < 1e-10:
            return pd.DataFrame()

        # Target indices
        informative = (gamma > 0).mean(axis=0) >= 0.1
        if informative.sum() < 20:
            return pd.DataFrame()
        gamma_var = np.var(gamma[:, informative], axis=0)
        n_targets = min(200, informative.sum())
        top_var_idx = np.argsort(gamma_var)[-n_targets:]
        info_indices = np.where(informative)[0]
        target_indices = info_indices[top_var_idx]

        # Pre-compute residualized gamma ranks
        gamma_resid_map = {}
        for ti in target_indices:
            t_gamma = gamma[:, ti]
            if np.std(t_gamma) < 1e-8:
                continue
            t_rank = stats.rankdata(t_gamma)
            t_rank_c = t_rank - t_rank.mean()
            slope = np.dot(lib_rank_centered, t_rank_c) / lib_ss
            resid = t_rank - slope * lib_rank
            resid_c = resid - resid.mean()
            resid_std = np.sqrt(np.dot(resid_c, resid_c))
            if resid_std > 1e-8:
                gamma_resid_map[ti] = (resid_c, resid_std)

        edges = []
        for rbp_upper, rbp_idx in rbps.items():
            rbp_expr = expr[:, rbp_idx]
            if np.std(rbp_expr) < 1e-6:
                continue

            rbp_rank = stats.rankdata(rbp_expr)
            rbp_rank_c = rbp_rank - rbp_rank.mean()
            slope_rbp = np.dot(lib_rank_centered, rbp_rank_c) / lib_ss
            rbp_resid = rbp_rank - slope_rbp * lib_rank
            rbp_resid_c = rbp_resid - rbp_resid.mean()
            rbp_resid_std = np.sqrt(np.dot(rbp_resid_c, rbp_resid_c))
            if rbp_resid_std < 1e-8:
                continue

            for ti in target_indices:
                if ti not in gamma_resid_map:
                    continue
                g_resid_c, g_resid_std = gamma_resid_map[ti]
                r_corr = np.dot(rbp_resid_c, g_resid_c) / (rbp_resid_std * g_resid_std)
                r_corr = np.clip(r_corr, -1.0, 1.0)
                df = n_cells - 3
                t_val = r_corr * np.sqrt(df / (1 - r_corr ** 2 + 1e-12))
                p_corr = 2 * stats.t.sf(abs(t_val), df)

                if p_corr < 0.05 / (len(rbps) * n_targets):
                    edges.append({
                        "rbp": rbp_upper,
                        "target": adata.var_names[ti],
                        "r": float(r_corr),
                    })

        return pd.DataFrame(edges) if edges else pd.DataFrame(columns=["rbp", "target", "r"])

    def get_top_hubs(edges_df, n=20):
        if len(edges_df) == 0:
            return []
        hub_counts = edges_df.groupby("rbp").size().sort_values(ascending=False)
        return list(hub_counts.head(n).index)

    # Run full-data network first
    print("\n  Running full-data pipeline...")
    adata_full_processed = adata_full.copy()
    adata_full_processed = run_nb_pipeline(adata_full_processed)
    full_edges = infer_network(adata_full_processed)
    full_hubs = get_top_hubs(full_edges, n=20)
    full_hub_counts = full_edges.groupby("rbp").size() if len(full_edges) > 0 else pd.Series(dtype=int)
    print(f"  Full data: {len(full_edges)} edges, top hubs: {full_hubs[:5]}")

    # Split-half replicates
    n_replicates = 5
    rng = np.random.RandomState(42)
    n_cells = adata_full.n_obs

    replicate_results = []

    for rep_i in range(n_replicates):
        print(f"\n  Replicate {rep_i + 1}/{n_replicates}...")

        # Random split
        perm = rng.permutation(n_cells)
        half1_idx = perm[:n_cells // 2]
        half2_idx = perm[n_cells // 2:]

        half_hubs = []
        half_hub_counts_list = []

        for half_name, cell_idx in [("half1", half1_idx), ("half2", half2_idx)]:
            adata_half = adata_full[cell_idx].copy()
            # Restore raw layers
            adata_half.layers["spliced"] = adata_half.layers["raw_spliced"].copy()
            adata_half.layers["unspliced"] = adata_half.layers["raw_unspliced"].copy()

            try:
                adata_half = run_nb_pipeline(adata_half)
                edges_half = infer_network(adata_half)
                hubs = get_top_hubs(edges_half, n=20)
                hub_counts = edges_half.groupby("rbp").size() if len(edges_half) > 0 else pd.Series(dtype=int)
                print(f"    {half_name}: {len(edges_half)} edges, {len(hubs)} hubs")
            except Exception as e:
                print(f"    {half_name}: pipeline failed: {e}")
                hubs = []
                hub_counts = pd.Series(dtype=int)

            half_hubs.append(set(hubs))
            half_hub_counts_list.append(hub_counts)

        # Compare halves
        if half_hubs[0] and half_hubs[1]:
            union = half_hubs[0] | half_hubs[1]
            intersection = half_hubs[0] & half_hubs[1]
            jaccard = len(intersection) / len(union) if union else 0

            # Hub count correlation (all shared RBPs)
            shared_rbps = sorted(set(half_hub_counts_list[0].index) & set(half_hub_counts_list[1].index))
            if len(shared_rbps) >= 5:
                c1 = [half_hub_counts_list[0].get(r, 0) for r in shared_rbps]
                c2 = [half_hub_counts_list[1].get(r, 0) for r in shared_rbps]
                r_hub, p_hub = stats.spearmanr(c1, c2)
            else:
                r_hub, p_hub = np.nan, np.nan

            # Compare each half to full data hubs
            jaccard_h1_full = len(half_hubs[0] & set(full_hubs)) / len(half_hubs[0] | set(full_hubs)) if (half_hubs[0] | set(full_hubs)) else 0
            jaccard_h2_full = len(half_hubs[1] & set(full_hubs)) / len(half_hubs[1] | set(full_hubs)) if (half_hubs[1] | set(full_hubs)) else 0

            print(f"    Half-half Jaccard (top-20 hubs): {jaccard:.3f}")
            print(f"    Hub count Spearman r: {r_hub:.3f}")
            print(f"    Half1-vs-full Jaccard: {jaccard_h1_full:.3f}, Half2-vs-full: {jaccard_h2_full:.3f}")

            replicate_results.append({
                "replicate": rep_i + 1,
                "jaccard_half_half": float(jaccard),
                "hub_count_spearman_r": float(r_hub) if not np.isnan(r_hub) else None,
                "jaccard_half1_full": float(jaccard_h1_full),
                "jaccard_half2_full": float(jaccard_h2_full),
                "n_shared_rbps": len(shared_rbps),
                "overlap_hubs": sorted(intersection),
            })
        else:
            replicate_results.append({
                "replicate": rep_i + 1,
                "jaccard_half_half": 0,
                "hub_count_spearman_r": None,
                "jaccard_half1_full": 0,
                "jaccard_half2_full": 0,
            })

    # Summary statistics
    jaccards = [r["jaccard_half_half"] for r in replicate_results]
    hub_rs = [r["hub_count_spearman_r"] for r in replicate_results if r["hub_count_spearman_r"] is not None]

    mean_jaccard = np.mean(jaccards)
    std_jaccard = np.std(jaccards)
    mean_hub_r = np.mean(hub_rs) if hub_rs else np.nan

    print(f"\n  SUMMARY:")
    print(f"    Mean Jaccard (top-20 hubs): {mean_jaccard:.3f} +/- {std_jaccard:.3f}")
    print(f"    Mean hub count Spearman r: {mean_hub_r:.3f}")

    results = {
        "full_data_n_edges": len(full_edges),
        "full_data_top_hubs": full_hubs,
        "n_replicates": n_replicates,
        "mean_jaccard": float(mean_jaccard),
        "std_jaccard": float(std_jaccard),
        "mean_hub_count_r": float(mean_hub_r) if not np.isnan(mean_hub_r) else None,
        "replicates": replicate_results,
    }

    with open(res_dir / "nb_split_half.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Jaccard per replicate
    axes[0].bar(range(1, n_replicates + 1), jaccards, color="#1976D2",
                edgecolor="black", linewidth=0.5)
    axes[0].axhline(y=mean_jaccard, color="red", linestyle="--",
                    label=f"Mean={mean_jaccard:.3f}")
    axes[0].set_xlabel("Replicate")
    axes[0].set_ylabel("Jaccard similarity (top-20 hubs)")
    axes[0].set_title("Split-Half Hub Consistency")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    # Panel 2: Hub count correlation
    if hub_rs:
        axes[1].bar(range(1, len(hub_rs) + 1), hub_rs, color="#43A047",
                    edgecolor="black", linewidth=0.5)
        axes[1].axhline(y=mean_hub_r, color="red", linestyle="--",
                        label=f"Mean={mean_hub_r:.3f}")
        axes[1].set_xlabel("Replicate")
        axes[1].set_ylabel("Spearman r (hub target counts)")
        axes[1].set_title("Split-Half Hub Count Correlation")
        axes[1].legend()
        axes[1].set_ylim(-0.5, 1)

    fig.suptitle("Experiment D: NB Network Split-Half Robustness", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "experiment_d_nb_robustness")

    return results


# =========================================================================
# EXPERIMENT E: Corrected vs Uncorrected Network Quality
# =========================================================================
def experiment_e_correction_quality(go_results):
    """Compare GO enrichment quality between corrected and uncorrected networks."""
    print(f"\n{'='*60}")
    print("EXPERIMENT E: CORRECTED vs UNCORRECTED NETWORK QUALITY")
    print(f"{'='*60}")

    res_dir = OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Load uncorrected (raw) network for NB
    raw_nb_path = TIER3_DIR / "neuroblastoma_network_raw.csv"
    corr_nb_path = TIER3_DIR / "neuroblastoma_network_corrected.csv"

    # Also check for raw pancreas edges from gap_analysis
    raw_panc_path = Path(__file__).parent.parent / "output" / "gap_analysis" / "results" / "network" / "pancreas" / "network_edges.csv"

    networks_to_compare = {}

    if raw_nb_path.exists() and corr_nb_path.exists():
        raw_nb = pd.read_csv(raw_nb_path)
        corr_nb = pd.read_csv(corr_nb_path)
        networks_to_compare["neuroblastoma"] = {"raw": raw_nb, "corrected": corr_nb}
        print(f"  NB raw: {len(raw_nb)} edges, corrected: {len(corr_nb)} edges")

    if raw_panc_path.exists():
        raw_panc = pd.read_csv(raw_panc_path)
        corr_panc_path = WEAKNESS_DIR / "corrected_network_pancreas.csv"
        if corr_panc_path.exists():
            corr_panc = pd.read_csv(corr_panc_path)
            networks_to_compare["pancreas"] = {"raw": raw_panc, "corrected": corr_panc}
            print(f"  Pancreas raw: {len(raw_panc)} edges, corrected: {len(corr_panc)} edges")

    if not networks_to_compare:
        print("  No raw/corrected network pairs found")
        return None

    # Load GO library from local cache
    go_lib = load_go_library()
    if go_lib is None:
        return None

    BACKGROUND_SIZE = 20000

    all_results = {}

    for ds_name, net_pair in networks_to_compare.items():
        print(f"\n  --- {ds_name} ---")

        for method_name, edges_df in net_pair.items():
            print(f"\n    {method_name} network ({len(edges_df)} edges):")

            rbp_col = "rbp"
            target_col = "target"

            # Build RBP -> target sets
            rbp_targets = {}
            for rbp, grp in edges_df.groupby(rbp_col):
                rbp_key = rbp.upper() if isinstance(rbp, str) else str(rbp)
                rbp_targets[rbp_key] = set(str(t) for t in grp[target_col])

            eligible = {r: t for r, t in rbp_targets.items() if len(t) >= 10}
            print(f"      RBPs with >= 10 targets: {len(eligible)}")

            n_with_sig = 0
            for rbp, targets in eligible.items():
                gene_list = list(targets)
                sig_terms = hypergeometric_enrichment(gene_list, go_lib, BACKGROUND_SIZE)
                if sig_terms:
                    n_with_sig += 1

            frac = n_with_sig / max(len(eligible), 1)
            print(f"      Fraction with sig GO: {n_with_sig}/{len(eligible)} ({frac:.1%})")

            key = f"{ds_name}_{method_name}"
            all_results[key] = {
                "dataset": ds_name,
                "method": method_name,
                "n_edges": len(edges_df),
                "n_eligible_rbps": len(eligible),
                "n_with_sig_go": n_with_sig,
                "frac_with_sig_go": float(frac),
            }

    # Save results
    with open(res_dir / "correction_quality.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Also compare destabilizing fractions
    print("\n  Destabilizing fraction comparison:")
    for ds_name, net_pair in networks_to_compare.items():
        for method_name, edges_df in net_pair.items():
            # Find the correlation column
            r_col = None
            for c in ["r", "spearman_r"]:
                if c in edges_df.columns:
                    r_col = c
                    break
            if r_col:
                destab_frac = (edges_df[r_col] > 0).mean()
                print(f"    {ds_name} {method_name}: {destab_frac:.1%} destabilizing")

    # Summary figure
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = []
    raw_fracs = []
    corr_fracs = []

    for ds_name in networks_to_compare:
        raw_key = f"{ds_name}_raw"
        corr_key = f"{ds_name}_corrected"
        if raw_key in all_results and corr_key in all_results:
            labels.append(ds_name)
            raw_fracs.append(all_results[raw_key]["frac_with_sig_go"])
            corr_fracs.append(all_results[corr_key]["frac_with_sig_go"])

    if labels:
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, raw_fracs, width, label="Raw (uncorrected)",
               color="#E53935", edgecolor="black", linewidth=0.5)
        ax.bar(x + width / 2, corr_fracs, width, label="Library-size corrected",
               color="#1976D2", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Fraction of RBPs with sig GO enrichment")
        ax.set_title("GO Enrichment: Raw vs Corrected Networks")
        ax.legend()
        ax.set_ylim(0, 1.1)
        for i, (r, c) in enumerate(zip(raw_fracs, corr_fracs)):
            ax.text(i - width / 2, r + 0.02, f"{r:.0%}", ha="center", fontsize=9)
            ax.text(i + width / 2, c + 0.02, f"{c:.0%}", ha="center", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "experiment_e_correction_quality")

    print(f"\n  EXPERIMENT E SUMMARY:")
    for ds_name in networks_to_compare:
        raw_key = f"{ds_name}_raw"
        corr_key = f"{ds_name}_corrected"
        if raw_key in all_results and corr_key in all_results:
            print(f"    {ds_name}: raw GO={all_results[raw_key]['frac_with_sig_go']:.0%} "
                  f"-> corrected GO={all_results[corr_key]['frac_with_sig_go']:.0%}")

    return all_results


# =========================================================================
# MAIN
# =========================================================================
def main():
    set_figure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    # ===== Experiment A: GO Enrichment (CSV-only + API, fast) =====
    go_results = experiment_a_go_enrichment()

    # ===== Experiment E: Correction Quality (reuses GO, fast) =====
    correction_results = experiment_e_correction_quality(go_results)

    # ===== Load datasets for experiments B, C =====
    print(f"\n{'='*60}")
    print("LOADING DATASETS FOR EXPERIMENTS B, C")
    print(f"{'='*60}")

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

    # ===== Experiment B: Pathway Consistency =====
    pathway_results = experiment_b_pathway_consistency(datasets)

    # ===== Experiment C: Gamma Advantage =====
    gamma_adv_results = experiment_c_gamma_advantage(datasets)

    # ===== Experiment D: NB Split-Half Robustness (slowest) =====
    nb_results = experiment_d_nb_robustness()

    # ===== FINAL SUMMARY =====
    print(f"\n{'='*60}")
    print("COMPREHENSIVE IMPROVEMENTS COMPLETE")
    print(f"{'='*60}")

    print("\n  Experiment A (GO Enrichment):")
    if go_results:
        for ds, res in go_results.items():
            print(f"    {ds}: {res.get('frac_with_sig_go', 0):.0%} RBPs enriched "
                  f"(null: {res.get('bootstrap_null_frac', 0):.0%})")

    print("\n  Experiment B (Pathway Consistency):")
    if pathway_results:
        for r in pathway_results:
            print(f"    {r['pair']}: gene r={r['gene_level_r']:.3f} -> "
                  f"pathway r={r['pathway_level_r']:.3f}")

    print("\n  Experiment C (Gamma Advantage):")
    if gamma_adv_results:
        for ds, res in gamma_adv_results.items():
            inv = res["invisible_states"]
            eta = res["eta_squared"]
            print(f"    {ds}: invisible gamma={inv['gamma_n_invisible']} "
                  f"vs ratio={inv['smooth_ratio_n_invisible']}; "
                  f"eta-sq p={eta['wilcoxon_p']:.2e}")

    print("\n  Experiment D (NB Robustness):")
    if nb_results:
        print(f"    Mean Jaccard (top-20): {nb_results['mean_jaccard']:.3f} "
              f"+/- {nb_results['std_jaccard']:.3f}")

    print("\n  Experiment E (Correction Quality):")
    if correction_results:
        for key, res in correction_results.items():
            print(f"    {key}: {res['frac_with_sig_go']:.0%} sig GO")

    print(f"\n  All results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
