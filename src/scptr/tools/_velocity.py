"""Post-transcriptional velocity computation."""

from __future__ import annotations

import numpy as np
from anndata import AnnData
from scipy.sparse import issparse

from .._constants import GAMMA, PT_VELOCITY
from .._utils import get_layer, require_layers, log_params
from .._numba_kernels import _velocity_kernel, _compute_adaptive_bandwidths


def pt_velocity(
    adata: AnnData,
    use_graph: str = "gamma",
) -> None:
    """Compute post-transcriptional velocity from gamma gradients.

    For each cell, the velocity is the weighted mean difference in
    gamma from its neighbors:
    ``v[i,g] = sum_j(w_ij * (gamma[j,g] - gamma[i,g]))``

    Parameters
    ----------
    adata
        Annotated data matrix with ``gamma`` layer and neighbor graph.
    use_graph
        Which neighbor graph to use: ``'gamma'`` for gamma-space graph
        (from ``pt_states``), ``'expression'`` for expression-space graph.
    """
    require_layers(adata, GAMMA)

    gamma = get_layer(adata, GAMMA).astype(np.float32)

    # Get the appropriate distance matrix
    if use_graph == "gamma" and "gamma_distances" in adata.obsp:
        dist_mat = adata.obsp["gamma_distances"]
    elif "distances" in adata.obsp:
        dist_mat = adata.obsp["distances"]
    else:
        raise ValueError(
            "No neighbor graph found. Run scptr.pp.neighbors() or "
            "scptr.tl.pt_states() first."
        )

    if not issparse(dist_mat):
        from scipy.sparse import csr_matrix
        dist_mat = csr_matrix(dist_mat)
    dist_mat = dist_mat.tocsr()

    indices = dist_mat.indices.astype(np.int32)
    indptr = dist_mat.indptr.astype(np.int32)
    distances = dist_mat.data.astype(np.float32)

    bandwidths = _compute_adaptive_bandwidths(distances, indptr)

    out = np.empty_like(gamma)
    _velocity_kernel(gamma, indices, indptr, distances, bandwidths, out)

    adata.layers[PT_VELOCITY] = out

    log_params(adata, "pt_velocity", {
        "use_graph": use_graph,
    })
