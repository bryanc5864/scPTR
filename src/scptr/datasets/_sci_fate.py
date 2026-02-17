"""sci-fate metabolic labeling dataset."""

from __future__ import annotations

import anndata as ad
from anndata import AnnData

from ._registry import fetch


def sci_fate() -> AnnData:
    """Load the sci-fate metabolic labeling dataset.

    This dataset contains cells with metabolic labeling (Cao et al. 2020),
    useful for validating degradation rate estimates against direct
    measurements of RNA turnover.

    Returns
    -------
    AnnData with unspliced and spliced layers.
    """
    path = fetch("sci_fate.h5ad")
    return ad.read_h5ad(path)
