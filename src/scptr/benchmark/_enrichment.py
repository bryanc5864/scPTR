"""Enrichment analysis for AU-rich element (ARE) and NMD target genes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import stats

from .._constants import GAMMA
from .._utils import get_layer, require_layers

_DATA_DIR = Path(__file__).parent / "data"


def _load_gene_list(filename: str) -> set[str]:
    """Load a gene list from a bundled text file (one gene per line)."""
    path = _DATA_DIR / filename
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def _enrichment_test(
    adata: AnnData,
    gene_set: set[str],
    label: str,
    min_gamma_fraction: float = 0.1,
) -> dict:
    """Mann-Whitney U test: do genes in gene_set have higher gamma?

    Only tests genes with sufficient non-zero gamma signal.
    """
    require_layers(adata, GAMMA)

    gamma = get_layer(adata, GAMMA)
    median_gamma = np.median(gamma, axis=0)

    # Filter to genes with reliable gamma estimates
    nonzero_frac = (gamma > 0).mean(axis=0)
    reliable = nonzero_frac >= min_gamma_fraction

    gene_names = adata.var_names.tolist()
    in_set = np.array([g in gene_set for g in gene_names])

    # Apply reliability filter
    in_set_reliable = in_set & reliable
    background_reliable = (~in_set) & reliable

    n_in = in_set_reliable.sum()
    n_out = background_reliable.sum()

    if n_in < 2 or n_out < 2:
        return {
            "label": label,
            "n_genes_in_set": int(n_in),
            "n_genes_in_set_unfiltered": int(in_set.sum()),
            "n_genes_background": int(n_out),
            "median_gamma_in_set": np.nan,
            "median_gamma_background": np.nan,
            "U_statistic": np.nan,
            "p_value": np.nan,
        }

    gamma_in = median_gamma[in_set_reliable]
    gamma_out = median_gamma[background_reliable]

    U, p = stats.mannwhitneyu(gamma_in, gamma_out, alternative="greater")

    return {
        "label": label,
        "n_genes_in_set": int(n_in),
        "n_genes_in_set_unfiltered": int(in_set.sum()),
        "n_genes_background": int(n_out),
        "median_gamma_in_set": float(np.median(gamma_in)),
        "median_gamma_background": float(np.median(gamma_out)),
        "U_statistic": float(U),
        "p_value": float(p),
    }


def are_enrichment(adata: AnnData) -> dict:
    """Test whether ARE genes have higher gamma than background.

    AU-rich elements (AREs) in 3' UTRs promote mRNA degradation.
    Genes with AREs should have higher degradation rates (gamma).

    Returns
    -------
    dict with test statistics including ``U_statistic`` and ``p_value``.
    """
    gene_set = _load_gene_list("are_genes.txt")
    return _enrichment_test(adata, gene_set, "ARE")


def nmd_enrichment(adata: AnnData) -> dict:
    """Test whether NMD target genes have higher gamma than background.

    Nonsense-mediated mRNA decay (NMD) targets should show higher
    degradation rates.

    Returns
    -------
    dict with test statistics including ``U_statistic`` and ``p_value``.
    """
    gene_set = _load_gene_list("nmd_genes.txt")
    return _enrichment_test(adata, gene_set, "NMD")
