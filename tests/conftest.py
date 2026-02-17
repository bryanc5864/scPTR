"""Shared test fixtures for scPTR tests."""

import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix


@pytest.fixture
def synthetic_adata():
    """Create a synthetic AnnData with unspliced/spliced layers (500 cells, 200 genes)."""
    np.random.seed(42)
    n_obs, n_vars = 500, 200

    # Simulate spliced counts (Poisson-like)
    spliced = np.random.exponential(5, size=(n_obs, n_vars)).astype(np.float32)
    # Simulate unspliced as fraction of spliced with noise
    unspliced = (spliced * np.random.uniform(0.05, 0.5, size=(1, n_vars))
                 + np.random.exponential(0.5, size=(n_obs, n_vars))).astype(np.float32)

    adata = AnnData(
        X=csr_matrix(spliced),
        layers={
            "spliced": csr_matrix(spliced),
            "unspliced": csr_matrix(unspliced),
        },
    )
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]
    return adata


@pytest.fixture
def preprocessed_adata(synthetic_adata):
    """Synthetic AnnData that has been through the preprocessing pipeline."""
    import scptr

    scptr.pp.filter_genes(synthetic_adata, min_unspliced_counts=1, min_unspliced_cells=1)
    scptr.pp.normalize_layers(synthetic_adata)
    scptr.pp.neighbors(synthetic_adata, n_neighbors=30)
    scptr.pp.smooth_layers(synthetic_adata)
    return synthetic_adata


@pytest.fixture
def analyzed_adata(preprocessed_adata):
    """Preprocessed AnnData that has been through core analysis."""
    import scptr

    scptr.tl.estimate_beta(preprocessed_adata)
    scptr.tl.estimate_gamma(preprocessed_adata)
    scptr.tl.variance_decomposition(preprocessed_adata)
    scptr.tl.pt_states(preprocessed_adata)
    return preprocessed_adata


@pytest.fixture
def velocity_adata(preprocessed_adata):
    """Preprocessed AnnData with a synthetic velocity layer for dynamic gamma testing."""
    import scptr

    scptr.tl.estimate_beta(preprocessed_adata)

    n_obs, n_vars = preprocessed_adata.shape
    np.random.seed(99)
    preprocessed_adata.layers["velocity_S"] = np.random.randn(n_obs, n_vars).astype(
        np.float32
    )
    return preprocessed_adata
