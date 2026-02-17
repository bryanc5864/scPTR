"""Neighbor graph computation (thin scanpy wrapper)."""

from __future__ import annotations

from anndata import AnnData

from .._constants import DEFAULT_N_NEIGHBORS
from .._utils import log_params


def neighbors(
    adata: AnnData,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    use_rep: str | None = None,
    **kwargs,
) -> None:
    """Compute a kNN graph using scanpy.

    Wrapper around :func:`scanpy.pp.neighbors` that logs parameters
    to ``adata.uns['scptr']``.

    Parameters
    ----------
    adata
        Annotated data matrix. If ``X_pca`` is not present, PCA is
        computed automatically by scanpy.
    n_neighbors
        Number of nearest neighbors.
    use_rep
        Representation to use. Passed to scanpy.
    **kwargs
        Additional keyword arguments passed to ``scanpy.pp.neighbors``.
    """
    import scanpy as sc

    if use_rep is None and "X_pca" not in adata.obsm:
        sc.tl.pca(adata)

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep, **kwargs)

    log_params(adata, "neighbors", {
        "n_neighbors": n_neighbors,
        "use_rep": use_rep,
    })
