"""Cross-dataset consistency analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import stats

from .._constants import GAMMA
from .._utils import get_layer, require_layers


def cross_dataset_consistency(
    adatas_dict: dict[str, AnnData],
) -> pd.DataFrame:
    """Compare per-gene median gamma across multiple datasets.

    For each pair of datasets, computes Spearman and Pearson correlations
    of per-gene median gamma on their shared gene set.

    Parameters
    ----------
    adatas_dict
        Dictionary mapping dataset name to analyzed AnnData
        (each must have ``gamma`` layer).

    Returns
    -------
    DataFrame with columns: ``dataset_a``, ``dataset_b``, ``n_shared_genes``,
    ``spearman_r``, ``pearson_r``.
    """
    names = sorted(adatas_dict.keys())
    # Pre-compute median gamma for each dataset
    medians = {}
    for name in names:
        adata = adatas_dict[name]
        require_layers(adata, GAMMA)
        gamma = get_layer(adata, GAMMA)
        medians[name] = pd.Series(
            np.median(gamma, axis=0), index=adata.var_names
        )

    records = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            shared = medians[name_a].index.intersection(medians[name_b].index)
            n_shared = len(shared)

            if n_shared < 3:
                sp_r = pe_r = np.nan
            else:
                ga = medians[name_a][shared].values.astype(float)
                gb = medians[name_b][shared].values.astype(float)
                valid = np.isfinite(ga) & np.isfinite(gb)
                ga, gb = ga[valid], gb[valid]

                if len(ga) < 3:
                    sp_r = pe_r = np.nan
                else:
                    sp_r, _ = stats.spearmanr(ga, gb)
                    pe_r, _ = stats.pearsonr(ga, gb)

            records.append({
                "dataset_a": name_a,
                "dataset_b": name_b,
                "n_shared_genes": n_shared,
                "spearman_r": float(sp_r) if not np.isnan(sp_r) else np.nan,
                "pearson_r": float(pe_r) if not np.isnan(pe_r) else np.nan,
            })

    return pd.DataFrame(records)
