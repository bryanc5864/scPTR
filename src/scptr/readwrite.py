"""Read/write functions for scPTR."""

from __future__ import annotations

import logging
import warnings

import anndata as ad
from anndata import AnnData

from ._constants import UNSPLICED, SPLICED

logger = logging.getLogger("scptr")


def _validate_layers(adata: AnnData) -> None:
    """Warn if unspliced/spliced layers are missing."""
    missing = []
    if UNSPLICED not in adata.layers:
        missing.append(UNSPLICED)
    if SPLICED not in adata.layers:
        missing.append(SPLICED)
    if missing:
        warnings.warn(
            f"AnnData is missing expected layers: {missing}. "
            "scPTR requires 'unspliced' and 'spliced' layers for analysis.",
            UserWarning,
            stacklevel=3,
        )


def read_h5ad(path: str, **kwargs) -> AnnData:
    """Read an h5ad file and return an AnnData object.

    Thin wrapper around ``anndata.read_h5ad`` that validates
    the presence of unspliced/spliced layers.
    """
    adata = ad.read_h5ad(path, **kwargs)
    _validate_layers(adata)
    return adata


def read_loom(path: str, **kwargs) -> AnnData:
    """Read a loom file and return an AnnData object.

    Uses scanpy's read_loom which maps loom layers to AnnData layers.
    """
    import scanpy as sc

    adata = sc.read_loom(path, **kwargs)
    _validate_layers(adata)
    return adata
