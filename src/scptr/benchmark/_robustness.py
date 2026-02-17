"""Subsampling robustness analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import stats

from .._constants import GAMMA
from .._utils import get_layer, require_layers


def subsampling_robustness(
    adata: AnnData,
    fractions: list[float] | None = None,
    n_repeats: int = 3,
    random_state: int = 0,
) -> pd.DataFrame:
    """Evaluate robustness of gamma estimates by subsampling cells.

    For each fraction, subsample cells, rerun the pipeline, and correlate
    the resulting per-gene median gamma with the full-data estimate.

    Parameters
    ----------
    adata
        Fully analyzed AnnData (must have ``gamma`` layer, ``Mu``/``Ms``
        layers, and ``var['beta']``).
    fractions
        Cell fractions to test (default: [0.3, 0.5, 0.7, 0.9]).
    n_repeats
        Number of random repeats per fraction.
    random_state
        Base random seed.

    Returns
    -------
    DataFrame with columns: ``fraction``, ``repeat``, ``spearman_r``,
    ``pearson_r``, ``n_genes``.
    """
    require_layers(adata, GAMMA)

    if fractions is None:
        fractions = [0.3, 0.5, 0.7, 0.9]

    gamma_full = get_layer(adata, GAMMA)
    median_gamma_full = np.median(gamma_full, axis=0)

    rng = np.random.RandomState(random_state)
    records = []

    for frac in fractions:
        n_cells = max(int(adata.n_obs * frac), 10)
        for rep in range(n_repeats):
            idx = rng.choice(adata.n_obs, size=n_cells, replace=False)
            gamma_sub = gamma_full[idx, :]
            median_gamma_sub = np.median(gamma_sub, axis=0)

            # Remove genes with zero variance
            valid = (np.std(median_gamma_full) > 0) & (np.std(median_gamma_sub) > 0)
            if not valid:
                sp_r = pe_r = np.nan
            else:
                sp_r, _ = stats.spearmanr(median_gamma_full, median_gamma_sub)
                pe_r, _ = stats.pearsonr(median_gamma_full, median_gamma_sub)

            records.append({
                "fraction": frac,
                "repeat": rep,
                "spearman_r": float(sp_r),
                "pearson_r": float(pe_r),
                "n_genes": int(adata.n_vars),
                "n_cells_sampled": n_cells,
            })

    return pd.DataFrame(records)
