"""Tests for DeepPTR data loading utilities."""

import numpy as np
import pytest
import torch
from anndata import AnnData

from scptr.deep._data import setup_dataloaders
from scptr.deep._utils import get_library_sizes


@pytest.fixture
def simple_adata():
    """AnnData with spliced/unspliced counts."""
    rng = np.random.RandomState(42)
    n, g = 100, 20
    s = rng.poisson(5, size=(n, g)).astype(np.float32)
    u = rng.poisson(2, size=(n, g)).astype(np.float32)
    adata = AnnData(X=s)
    adata.layers["spliced"] = s
    adata.layers["unspliced"] = u
    adata.obs["cell_type"] = [f"type_{i % 3}" for i in range(n)]
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    return adata


class TestGetLibrarySizes:
    def test_shapes(self, simple_adata):
        l_u, l_s = get_library_sizes(simple_adata)
        assert l_u.shape == (simple_adata.n_obs,)
        assert l_s.shape == (simple_adata.n_obs,)

    def test_positive(self, simple_adata):
        l_u, l_s = get_library_sizes(simple_adata)
        assert (l_u >= 1.0).all()
        assert (l_s >= 1.0).all()

    def test_correct_sums(self, simple_adata):
        l_u, l_s = get_library_sizes(simple_adata)
        expected_s = simple_adata.layers["spliced"].sum(axis=1)
        expected_u = simple_adata.layers["unspliced"].sum(axis=1)
        np.testing.assert_allclose(l_s, np.clip(expected_s, 1.0, None), rtol=1e-5)
        np.testing.assert_allclose(l_u, np.clip(expected_u, 1.0, None), rtol=1e-5)

    def test_missing_layer_raises(self):
        adata = AnnData(X=np.zeros((5, 3)))
        with pytest.raises(KeyError, match="Missing required layer"):
            get_library_sizes(adata)


class TestSetupDataloaders:
    def test_returns_four(self, simple_adata):
        train_dl, val_dl, train_idx, val_idx = setup_dataloaders(
            simple_adata, batch_size=16, val_frac=0.2, seed=0
        )
        assert isinstance(train_dl, torch.utils.data.DataLoader)
        assert isinstance(val_dl, torch.utils.data.DataLoader)
        assert len(train_idx) + len(val_idx) == simple_adata.n_obs

    def test_no_overlap(self, simple_adata):
        _, _, train_idx, val_idx = setup_dataloaders(
            simple_adata, batch_size=16, val_frac=0.2, seed=0
        )
        assert len(set(train_idx) & set(val_idx)) == 0

    def test_batch_contents(self, simple_adata):
        train_dl, _, _, _ = setup_dataloaders(
            simple_adata, batch_size=16, val_frac=0.1, seed=0
        )
        s, u, l_s, l_u = next(iter(train_dl))
        assert s.ndim == 2
        assert u.ndim == 2
        assert l_s.ndim == 1
        assert l_u.ndim == 1
        assert s.shape[1] == simple_adata.n_vars

    def test_stratified_split(self, simple_adata):
        _, _, train_idx, val_idx = setup_dataloaders(
            simple_adata,
            batch_size=16,
            val_frac=0.2,
            stratify_key="cell_type",
            seed=0,
        )
        # All cell types should appear in both splits
        train_types = set(simple_adata.obs["cell_type"].values[train_idx])
        val_types = set(simple_adata.obs["cell_type"].values[val_idx])
        assert train_types == val_types

    def test_reproducible(self, simple_adata):
        _, _, idx1, _ = setup_dataloaders(simple_adata, seed=42)
        _, _, idx2, _ = setup_dataloaders(simple_adata, seed=42)
        np.testing.assert_array_equal(idx1, idx2)
