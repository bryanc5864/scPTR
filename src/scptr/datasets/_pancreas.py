"""Pancreas endocrinogenesis dataset."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
from anndata import AnnData


def pancreas() -> AnnData:
    """Load the pancreas endocrinogenesis dataset.

    This dataset contains ~3,700 cells from mouse pancreas
    endocrinogenesis (Bastidas-Ponce et al. 2019), commonly used
    for RNA velocity benchmarking.

    Returns
    -------
    AnnData with unspliced and spliced layers.
    """
    cache_path = Path.home() / ".cache" / "scptr" / "pancreas.h5ad"
    if cache_path.exists():
        return ad.read_h5ad(cache_path)

    from ._registry import fetch
    path = fetch("pancreas.h5ad")
    return ad.read_h5ad(path)
