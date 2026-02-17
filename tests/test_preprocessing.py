"""Tests for the preprocessing module."""

import numpy as np
import pytest
from scipy.sparse import issparse


def test_filter_genes(synthetic_adata):
    import scptr

    n_before = synthetic_adata.n_vars
    scptr.pp.filter_genes(
        synthetic_adata, min_unspliced_counts=10, min_unspliced_cells=10
    )
    assert synthetic_adata.n_vars <= n_before
    assert synthetic_adata.n_vars > 0
    assert "filter_genes" in synthetic_adata.uns["scptr"]


def test_filter_cells(synthetic_adata):
    import scptr

    n_before = synthetic_adata.n_obs
    scptr.pp.filter_cells(synthetic_adata, min_unspliced_counts=1)
    assert synthetic_adata.n_obs <= n_before
    assert synthetic_adata.n_obs > 0


def test_normalize_layers(synthetic_adata):
    import scptr

    scptr.pp.normalize_layers(synthetic_adata)
    # After normalization, layers should be dense float32
    assert not issparse(synthetic_adata.layers["spliced"])
    assert synthetic_adata.layers["spliced"].dtype == np.float32
    assert not issparse(synthetic_adata.layers["unspliced"])
    assert "normalize_layers" in synthetic_adata.uns["scptr"]


def test_neighbors(synthetic_adata):
    import scptr

    scptr.pp.normalize_layers(synthetic_adata)
    scptr.pp.neighbors(synthetic_adata, n_neighbors=10)
    assert "distances" in synthetic_adata.obsp
    assert "connectivities" in synthetic_adata.obsp
    assert "neighbors" in synthetic_adata.uns["scptr"]


def test_smooth_layers(synthetic_adata):
    import scptr

    scptr.pp.normalize_layers(synthetic_adata)
    scptr.pp.neighbors(synthetic_adata, n_neighbors=10)
    scptr.pp.smooth_layers(synthetic_adata)
    assert "Mu" in synthetic_adata.layers
    assert "Ms" in synthetic_adata.layers
    assert synthetic_adata.layers["Mu"].shape == synthetic_adata.shape
    assert synthetic_adata.layers["Ms"].shape == synthetic_adata.shape
    assert synthetic_adata.layers["Mu"].dtype == np.float32


def test_smooth_fixed_bandwidth(synthetic_adata):
    import scptr

    scptr.pp.normalize_layers(synthetic_adata)
    scptr.pp.neighbors(synthetic_adata, n_neighbors=10)
    scptr.pp.smooth_layers(synthetic_adata, bandwidth=1.0)
    assert "Mu" in synthetic_adata.layers
