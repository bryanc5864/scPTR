"""Post-transcriptional state discovery via Leiden clustering on gamma profiles."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from .._constants import GAMMA, PT_STATE, DEFAULT_LEIDEN_RESOLUTION
from .._utils import get_layer, require_layers, log_params


def pt_states(
    adata: AnnData,
    resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    n_pcs: int = 30,
    n_neighbors: int = 30,
    random_state: int = 0,
) -> None:
    """Discover post-transcriptional states by clustering gamma profiles.

    Steps:
    1. PCA on the gamma matrix.
    2. kNN graph in gamma-PCA space.
    3. Leiden clustering.
    4. UMAP embedding in gamma space.

    Results are stored in ``adata.obs['pt_state']`` and
    ``adata.obsm['X_gamma_pca']``, ``adata.obsm['X_gamma_umap']``.

    Parameters
    ----------
    adata
        Annotated data matrix with ``gamma`` layer.
    resolution
        Leiden clustering resolution.
    n_pcs
        Number of principal components for gamma PCA.
    n_neighbors
        Number of neighbors for the gamma-space kNN graph.
    random_state
        Random seed for reproducibility.
    """
    import scanpy as sc

    require_layers(adata, GAMMA)

    gamma = get_layer(adata, GAMMA)

    # Create a temporary AnnData for gamma-space analysis
    gamma_adata = AnnData(X=gamma.copy())
    gamma_adata.obs_names = adata.obs_names.copy()
    gamma_adata.var_names = adata.var_names.copy()

    # PCA on gamma matrix
    n_pcs_use = min(n_pcs, min(gamma.shape) - 1)
    sc.tl.pca(gamma_adata, n_comps=n_pcs_use, random_state=random_state)

    # kNN graph in gamma-PCA space
    sc.pp.neighbors(
        gamma_adata,
        n_neighbors=n_neighbors,
        use_rep="X_pca",
        random_state=random_state,
    )

    # Leiden clustering
    sc.tl.leiden(
        gamma_adata,
        resolution=resolution,
        random_state=random_state,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )

    # UMAP
    sc.tl.umap(gamma_adata, random_state=random_state)

    # Store results back in original adata
    adata.obs[PT_STATE] = gamma_adata.obs["leiden"].values
    adata.obs[PT_STATE] = adata.obs[PT_STATE].astype("category")
    adata.obsm["X_gamma_pca"] = gamma_adata.obsm["X_pca"]
    adata.obsm["X_gamma_umap"] = gamma_adata.obsm["X_umap"]

    # Store gamma-space neighbor graph
    adata.obsp["gamma_distances"] = gamma_adata.obsp["distances"]
    adata.obsp["gamma_connectivities"] = gamma_adata.obsp["connectivities"]

    log_params(adata, "pt_states", {
        "resolution": resolution,
        "n_pcs": n_pcs_use,
        "n_neighbors": n_neighbors,
        "n_states": int(adata.obs[PT_STATE].nunique()),
    })
