"""Per-layer library-size normalization."""

from __future__ import annotations

import numpy as np
from anndata import AnnData
from scipy.sparse import issparse

from .._constants import UNSPLICED, SPLICED
from .._utils import log_params


def normalize_layers(
    adata: AnnData,
    target_sum: float | None = None,
    layers: tuple[str, ...] = (SPLICED, UNSPLICED),
) -> None:
    """Library-size normalize spliced and unspliced layers independently.

    Each cell's counts in each layer are divided by the cell's total
    counts in that layer and multiplied by *target_sum* (defaults to
    the median library size across cells for that layer).

    Modifies *adata* in place — layers are converted to dense float32.

    Parameters
    ----------
    adata
        Annotated data matrix.
    target_sum
        Target total counts per cell. If ``None``, use the median.
    layers
        Which layers to normalize.
    """
    for layer in layers:
        mat = adata.layers[layer]
        if issparse(mat):
            mat = np.asarray(mat.todense(), dtype=np.float32)
        else:
            mat = np.asarray(mat, dtype=np.float32)

        lib_sizes = mat.sum(axis=1, keepdims=True)
        lib_sizes = np.clip(lib_sizes, 1e-10, None)

        if target_sum is None:
            ts = np.median(lib_sizes)
        else:
            ts = target_sum

        mat = mat / lib_sizes * ts
        adata.layers[layer] = mat

    log_params(adata, "normalize_layers", {
        "target_sum": target_sum,
        "layers": list(layers),
    })
