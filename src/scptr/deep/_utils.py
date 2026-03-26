"""Initialization and utility helpers for DeepPTR."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from anndata import AnnData


def init_weights(module: nn.Module) -> None:
    """Xavier-uniform initialization for linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def beta_from_adata(adata: AnnData) -> torch.Tensor:
    """Extract analytical beta estimates from *adata* for warm-starting.

    Falls back to ones if ``adata.var['beta']`` is absent.

    Returns
    -------
    torch.Tensor
        Shape ``(n_genes,)``, dtype float32.
    """
    if "beta" in adata.var.columns:
        beta = adata.var["beta"].values.astype(np.float32)
        beta = np.clip(beta, 1e-4, None)
        return torch.from_numpy(np.log(beta))
    return torch.zeros(adata.n_vars, dtype=torch.float32)


def get_library_sizes(adata: AnnData) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-cell library sizes for unspliced and spliced layers.

    Uses raw integer counts stored in ``adata.layers``.

    Returns
    -------
    (l_u, l_s) : tuple of np.ndarray
        Each shape ``(n_obs,)``, float32.
    """
    from scipy.sparse import issparse

    for layer in ("spliced", "unspliced"):
        if layer not in adata.layers:
            raise KeyError(f"Missing required layer: {layer}")

    def _sum(mat: np.ndarray | "scipy.sparse.spmatrix") -> np.ndarray:
        if issparse(mat):
            return np.asarray(mat.sum(axis=1)).ravel().astype(np.float32)
        return np.asarray(mat.sum(axis=1)).ravel().astype(np.float32)

    l_s = _sum(adata.layers["spliced"])
    l_u = _sum(adata.layers["unspliced"])

    # Avoid zero library sizes
    l_s = np.clip(l_s, 1.0, None)
    l_u = np.clip(l_u, 1.0, None)

    return l_u, l_s
