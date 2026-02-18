"""miRNA-target interaction analysis for post-transcriptional networks.

Integrates TargetScan predictions to identify miRNA-mediated regulation
of mRNA degradation rates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import stats

from .._constants import GAMMA
from .._utils import get_layer, require_layers, log_params


_CACHE_DIR = Path.home() / ".cache" / "scptr" / "targetscan"


def load_targetscan_predictions(
    species_id: int = 9606,
    min_context_score: float = -0.2,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load TargetScan conserved miRNA-target predictions.

    Parameters
    ----------
    species_id
        NCBI taxonomy ID. 9606 = human, 10090 = mouse.
    min_context_score
        Minimum (most negative = strongest) context++ score to include.
        Default -0.2 keeps moderately strong predictions.
    cache_dir
        Directory containing TargetScan files. If ``None``, looks in
        ``~/.cache/scptr/targetscan/`` and project ``.cache/targetscan/``.

    Returns
    -------
    DataFrame with columns ``['mirna_family', 'gene_symbol', 'context_score',
    'n_conserved_sites', 'representative_mirna']``.
    """
    # Search for the file in multiple locations
    search_dirs = []
    if cache_dir:
        search_dirs.append(Path(cache_dir))
    search_dirs.extend([
        _CACHE_DIR,
        Path.cwd() / ".cache" / "targetscan",
    ])

    summary_file = None
    for d in search_dirs:
        candidate = d / "Summary_Counts.default_predictions.txt"
        if candidate.exists():
            summary_file = candidate
            break

    if summary_file is None:
        raise FileNotFoundError(
            "TargetScan Summary_Counts.default_predictions.txt not found. "
            "Download from https://www.targetscan.org/vert_80/vert_80_data_download/"
            "Summary_Counts.default_predictions.txt.zip and extract to "
            f"one of: {[str(d) for d in search_dirs]}"
        )

    df = pd.read_csv(summary_file, sep="\t", low_memory=False)
    df = df[df["Species ID"] == species_id].copy()

    # Filter by context score (more negative = stronger)
    score_col = "Total context++ score"
    if score_col in df.columns:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        df = df[df[score_col] <= min_context_score].copy()

    result = pd.DataFrame({
        "mirna_family": df["miRNA family"],
        "gene_symbol": df["Gene Symbol"],
        "context_score": df[score_col] if score_col in df.columns else np.nan,
        "n_conserved_sites": df["Total num conserved sites"],
        "representative_mirna": df["Representative miRNA"],
    })

    return result.reset_index(drop=True)


def mirna_gamma_correlation(
    adata: AnnData,
    mirna_targets: pd.DataFrame,
    n_top_targets: int = 200,
    min_cells_expressing: int = 50,
) -> pd.DataFrame:
    """Test whether miRNA target genes have higher gamma (degradation).

    For each miRNA family, tests whether its predicted targets have
    systematically higher degradation rates than non-targets using
    Mann-Whitney U test.

    Parameters
    ----------
    adata
        Annotated data matrix with ``gamma`` layer.
    mirna_targets
        DataFrame from :func:`load_targetscan_predictions`.
    n_top_targets
        Number of top gamma-variable genes to use as background.
    min_cells_expressing
        Minimum cells with nonzero gamma for a gene to be included.

    Returns
    -------
    DataFrame with per-miRNA-family results.
    """
    require_layers(adata, GAMMA)
    gamma = get_layer(adata, GAMMA)

    # Per-gene median gamma
    med_gamma = np.median(gamma, axis=0)
    nonzero_frac = (gamma > 0).mean(axis=0)

    # Build gene lookup (case-insensitive)
    gene_map = {g.upper(): i for i, g in enumerate(adata.var_names)}

    # Filter to informative genes
    informative = nonzero_frac >= 0.1
    informative_genes = set(
        adata.var_names[i].upper() for i in range(len(adata.var_names)) if informative[i]
    )

    # All informative gamma values as background
    bg_gamma = med_gamma[informative]

    # Group targets by miRNA family
    targets_by_family = {}
    for _, row in mirna_targets.iterrows():
        family = row["mirna_family"]
        gene = str(row["gene_symbol"]).upper()
        if family not in targets_by_family:
            targets_by_family[family] = set()
        targets_by_family[family].add(gene)

    results = []
    for family, target_genes in sorted(targets_by_family.items()):
        # Map to dataset genes
        target_in_data = target_genes & informative_genes
        if len(target_in_data) < 5:
            continue

        target_gamma = [med_gamma[gene_map[g]] for g in target_in_data]
        nontarget_gamma = [
            med_gamma[gene_map[g]] for g in informative_genes - target_in_data
            if g in gene_map
        ]

        if len(nontarget_gamma) < 10:
            continue

        # Mann-Whitney: do targets have higher gamma?
        u_stat, p_val = stats.mannwhitneyu(
            target_gamma, nontarget_gamma, alternative="greater"
        )

        # Get representative miRNA name
        family_rows = mirna_targets[mirna_targets["mirna_family"] == family]
        rep_mirna = family_rows["representative_mirna"].iloc[0] if len(family_rows) > 0 else family

        results.append({
            "mirna_family": family,
            "representative_mirna": rep_mirna,
            "n_targets_in_data": len(target_in_data),
            "target_median_gamma": float(np.median(target_gamma)),
            "nontarget_median_gamma": float(np.median(nontarget_gamma)),
            "fold_enrichment": float(np.median(target_gamma) / (np.median(nontarget_gamma) + 1e-8)),
            "mannwhitney_p": float(p_val),
        })

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        # FDR correction
        from statsmodels.stats.multitest import multipletests
        _, result_df["fdr"], _, _ = multipletests(
            result_df["mannwhitney_p"], method="fdr_bh"
        )
        result_df = result_df.sort_values("mannwhitney_p")

    return result_df.reset_index(drop=True)
