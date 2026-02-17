"""Tests for gamma estimation."""

import numpy as np
import pytest


def test_estimate_gamma(preprocessed_adata):
    import scptr

    scptr.tl.estimate_beta(preprocessed_adata)
    scptr.tl.estimate_gamma(preprocessed_adata)

    assert "gamma" in preprocessed_adata.layers
    gamma = preprocessed_adata.layers["gamma"]
    assert gamma.shape == preprocessed_adata.shape
    assert gamma.dtype == np.float32
    assert np.all(gamma >= 0)
    assert "estimate_gamma" in preprocessed_adata.uns["scptr"]


def test_gamma_clipping(preprocessed_adata):
    import scptr

    scptr.tl.estimate_beta(preprocessed_adata)
    scptr.tl.estimate_gamma(preprocessed_adata, clip_quantile=0.95)
    gamma_95 = preprocessed_adata.layers["gamma"].copy()

    scptr.tl.estimate_gamma(preprocessed_adata, clip_quantile=0.5)
    gamma_50 = preprocessed_adata.layers["gamma"].copy()

    # Stricter clipping should give lower or equal max values
    assert gamma_50.max() <= gamma_95.max() + 1e-6


def test_gamma_requires_beta(preprocessed_adata):
    import scptr

    with pytest.raises(KeyError):
        scptr.tl.estimate_gamma(preprocessed_adata)
