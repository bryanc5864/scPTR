"""Tests for readwrite validation."""

import warnings

import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix


@pytest.fixture
def tmp_h5ad_with_layers(tmp_path):
    """Create a temporary h5ad file with unspliced/spliced layers."""
    adata = AnnData(
        X=csr_matrix(np.ones((10, 5), dtype=np.float32)),
        layers={
            "spliced": csr_matrix(np.ones((10, 5), dtype=np.float32)),
            "unspliced": csr_matrix(np.ones((10, 5), dtype=np.float32)),
        },
    )
    path = tmp_path / "with_layers.h5ad"
    adata.write_h5ad(path)
    return path


@pytest.fixture
def tmp_h5ad_without_layers(tmp_path):
    """Create a temporary h5ad file without unspliced/spliced layers."""
    adata = AnnData(X=csr_matrix(np.ones((10, 5), dtype=np.float32)))
    path = tmp_path / "without_layers.h5ad"
    adata.write_h5ad(path)
    return path


def test_read_h5ad_no_warning_with_layers(tmp_h5ad_with_layers):
    """read_h5ad does not warn when layers are present."""
    from scptr.readwrite import read_h5ad

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        adata = read_h5ad(str(tmp_h5ad_with_layers))

    assert "unspliced" in adata.layers
    assert "spliced" in adata.layers


def test_read_h5ad_warns_missing_layers(tmp_h5ad_without_layers):
    """read_h5ad warns when unspliced/spliced layers are missing."""
    from scptr.readwrite import read_h5ad

    with pytest.warns(UserWarning, match="missing expected layers"):
        read_h5ad(str(tmp_h5ad_without_layers))


def test_validate_layers_warns_partial(tmp_path):
    """_validate_layers warns when only one layer is missing."""
    from scptr.readwrite import _validate_layers

    adata = AnnData(
        X=csr_matrix(np.ones((10, 5), dtype=np.float32)),
        layers={"spliced": csr_matrix(np.ones((10, 5), dtype=np.float32))},
    )

    with pytest.warns(UserWarning, match="unspliced"):
        _validate_layers(adata)
