"""Tests for beta estimation."""

import numpy as np
import pytest


def test_estimate_beta(preprocessed_adata):
    import scptr

    scptr.tl.estimate_beta(preprocessed_adata)
    assert "beta" in preprocessed_adata.var.columns
    beta = preprocessed_adata.var["beta"].values
    assert beta.dtype == np.float32
    assert len(beta) == preprocessed_adata.n_vars
    assert np.all(beta >= 0)
    assert "estimate_beta" in preprocessed_adata.uns["scptr"]


def test_estimate_beta_quantile(preprocessed_adata):
    import scptr

    scptr.tl.estimate_beta(preprocessed_adata, quantile=0.90)
    beta_90 = preprocessed_adata.var["beta"].values.copy()

    scptr.tl.estimate_beta(preprocessed_adata, quantile=0.99)
    beta_99 = preprocessed_adata.var["beta"].values.copy()

    # Higher quantile should give higher or equal beta
    assert np.all(beta_99 >= beta_90 - 1e-6)
