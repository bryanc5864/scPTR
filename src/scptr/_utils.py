"""Utility functions for layer resolution, validation, and parameter logging."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from anndata import AnnData
from scipy.sparse import issparse

from . import _constants as C

logger = logging.getLogger("scptr")


def get_layer(adata: AnnData, layer: str) -> np.ndarray:
    """Return a dense numpy array for the given layer."""
    mat = adata.layers[layer]
    if issparse(mat):
        return np.asarray(mat.todense())
    return np.asarray(mat)


def require_layers(adata: AnnData, *layers: str) -> None:
    """Raise KeyError if any of the specified layers are missing."""
    missing = [l for l in layers if l not in adata.layers]
    if missing:
        raise KeyError(f"Missing required layers: {missing}")


def require_obs(adata: AnnData, *keys: str) -> None:
    """Raise KeyError if any of the specified obs columns are missing."""
    missing = [k for k in keys if k not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing required obs columns: {missing}")


def require_var(adata: AnnData, *keys: str) -> None:
    """Raise KeyError if any of the specified var columns are missing."""
    missing = [k for k in keys if k not in adata.var.columns]
    if missing:
        raise KeyError(f"Missing required var columns: {missing}")


def log_params(adata: AnnData, step: str, params: dict) -> None:
    """Record run parameters in adata.uns['scptr']."""
    if C.UNS_KEY not in adata.uns:
        adata.uns[C.UNS_KEY] = {}
    adata.uns[C.UNS_KEY][step] = params


def to_dense_float32(mat) -> np.ndarray:
    """Convert a matrix (sparse or dense) to dense float32."""
    if issparse(mat):
        mat = np.asarray(mat.todense())
    return np.asarray(mat, dtype=np.float32)
