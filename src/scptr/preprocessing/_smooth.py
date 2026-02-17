"""kNN Gaussian kernel smoothing of expression layers."""

from __future__ import annotations

import numpy as np
from anndata import AnnData
from scipy.sparse import issparse

from .._constants import (
    UNSPLICED,
    SPLICED,
    SMOOTHED_UNSPLICED,
    SMOOTHED_SPLICED,
    DEFAULT_BANDWIDTH,
    DEFAULT_GENE_BATCH_SIZE,
)
from .._utils import log_params, to_dense_float32
from .._numba_kernels import _smooth_kernel, _compute_adaptive_bandwidths


def smooth_layers(
    adata: AnnData,
    bandwidth: str | float = DEFAULT_BANDWIDTH,
    gene_batch_size: int = DEFAULT_GENE_BATCH_SIZE,
    layers: tuple[str, str] = (UNSPLICED, SPLICED),
    out_layers: tuple[str, str] = (SMOOTHED_UNSPLICED, SMOOTHED_SPLICED),
) -> None:
    """Smooth unspliced and spliced layers using kNN Gaussian kernel.

    Uses the precomputed kNN graph in ``adata.obsp['distances']``.
    Gene-batched for memory control on large datasets.

    Parameters
    ----------
    adata
        Annotated data matrix with precomputed neighbor graph.
    bandwidth
        ``'adaptive'`` (default) uses the median neighbor distance per
        cell. A float value sets a fixed bandwidth for all cells.
    gene_batch_size
        Number of genes to process per batch.
    layers
        Input layer names ``(unspliced, spliced)``.
    out_layers
        Output layer names ``(Mu, Ms)``.
    """
    if "distances" not in adata.obsp:
        raise ValueError(
            "Neighbor graph not found. Run scptr.pp.neighbors() first."
        )

    dist_mat = adata.obsp["distances"]
    if not issparse(dist_mat):
        from scipy.sparse import csr_matrix
        dist_mat = csr_matrix(dist_mat)
    dist_mat = dist_mat.tocsr()

    indices = dist_mat.indices.astype(np.int32)
    indptr = dist_mat.indptr.astype(np.int32)
    distances = dist_mat.data.astype(np.float32)

    # Compute bandwidths
    if bandwidth == "adaptive":
        bandwidths = _compute_adaptive_bandwidths(distances, indptr)
    else:
        bandwidths = np.full(adata.n_obs, float(bandwidth), dtype=np.float32)

    n_obs, n_vars = adata.n_obs, adata.n_vars

    for in_layer, out_layer in zip(layers, out_layers):
        data = to_dense_float32(adata.layers[in_layer])
        result = np.empty_like(data)

        # Process in gene batches
        for start in range(0, n_vars, gene_batch_size):
            end = min(start + gene_batch_size, n_vars)
            batch_in = np.ascontiguousarray(data[:, start:end])
            batch_out = np.empty_like(batch_in)

            _smooth_kernel(
                batch_in, indices, indptr, distances, bandwidths, batch_out
            )
            result[:, start:end] = batch_out

        adata.layers[out_layer] = result

    log_params(adata, "smooth_layers", {
        "bandwidth": bandwidth,
        "gene_batch_size": gene_batch_size,
    })
