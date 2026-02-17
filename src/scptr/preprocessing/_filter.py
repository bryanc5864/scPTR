"""Gene and cell filtering for scPTR."""

from __future__ import annotations

import numpy as np
from anndata import AnnData
from scipy.sparse import issparse

from .._constants import (
    UNSPLICED,
    SPLICED,
    DEFAULT_MIN_UNSPLICED_COUNTS,
    DEFAULT_MIN_UNSPLICED_CELLS,
)
from .._utils import log_params


def filter_genes(
    adata: AnnData,
    min_unspliced_counts: int = DEFAULT_MIN_UNSPLICED_COUNTS,
    min_unspliced_cells: int = DEFAULT_MIN_UNSPLICED_CELLS,
    min_spliced_counts: int = 0,
) -> None:
    """Filter genes based on unspliced/spliced count thresholds.

    Modifies *adata* in place by subsetting to genes that pass thresholds.

    Parameters
    ----------
    adata
        Annotated data matrix with ``unspliced`` and ``spliced`` layers.
    min_unspliced_counts
        Minimum total unspliced counts across all cells.
    min_unspliced_cells
        Minimum number of cells with nonzero unspliced counts.
    min_spliced_counts
        Minimum total spliced counts across all cells.
    """
    u = adata.layers[UNSPLICED]
    s = adata.layers[SPLICED]

    if issparse(u):
        u_total = np.asarray(u.sum(axis=0)).ravel()
        u_ncells = np.asarray((u > 0).sum(axis=0)).ravel()
    else:
        u_total = np.asarray(u.sum(axis=0)).ravel()
        u_ncells = np.asarray((u > 0).sum(axis=0)).ravel()

    if issparse(s):
        s_total = np.asarray(s.sum(axis=0)).ravel()
    else:
        s_total = np.asarray(s.sum(axis=0)).ravel()

    keep = (
        (u_total >= min_unspliced_counts)
        & (u_ncells >= min_unspliced_cells)
        & (s_total >= min_spliced_counts)
    )

    n_before = adata.n_vars
    adata._inplace_subset_var(keep)
    n_after = adata.n_vars

    log_params(adata, "filter_genes", {
        "min_unspliced_counts": min_unspliced_counts,
        "min_unspliced_cells": min_unspliced_cells,
        "min_spliced_counts": min_spliced_counts,
        "n_genes_before": int(n_before),
        "n_genes_after": int(n_after),
    })


def filter_cells(
    adata: AnnData,
    min_unspliced_counts: int = 0,
    min_spliced_counts: int = 0,
) -> None:
    """Filter cells based on count thresholds.

    Modifies *adata* in place.

    Parameters
    ----------
    adata
        Annotated data matrix.
    min_unspliced_counts
        Minimum total unspliced counts per cell.
    min_spliced_counts
        Minimum total spliced counts per cell.
    """
    u = adata.layers[UNSPLICED]
    s = adata.layers[SPLICED]

    if issparse(u):
        u_total = np.asarray(u.sum(axis=1)).ravel()
    else:
        u_total = np.asarray(u.sum(axis=1)).ravel()

    if issparse(s):
        s_total = np.asarray(s.sum(axis=1)).ravel()
    else:
        s_total = np.asarray(s.sum(axis=1)).ravel()

    keep = (u_total >= min_unspliced_counts) & (s_total >= min_spliced_counts)

    n_before = adata.n_obs
    adata._inplace_subset_obs(keep)
    n_after = adata.n_obs

    log_params(adata, "filter_cells", {
        "min_unspliced_counts": min_unspliced_counts,
        "min_spliced_counts": min_spliced_counts,
        "n_cells_before": int(n_before),
        "n_cells_after": int(n_after),
    })
