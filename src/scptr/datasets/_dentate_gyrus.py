"""Dentate gyrus neurogenesis dataset."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
from anndata import AnnData


def dentate_gyrus() -> AnnData:
    """Load the dentate gyrus neurogenesis dataset.

    This dataset contains ~2,900 cells from mouse hippocampal
    dentate gyrus (Hochgerner et al. 2018), commonly used
    for RNA velocity benchmarking.

    Returns
    -------
    AnnData with unspliced and spliced layers.
    """
    cache_path = Path.home() / ".cache" / "scptr" / "dentate_gyrus.h5ad"
    if cache_path.exists():
        return ad.read_h5ad(cache_path)

    from ._registry import fetch
    path = fetch("dentate_gyrus.h5ad")
    return ad.read_h5ad(path)
