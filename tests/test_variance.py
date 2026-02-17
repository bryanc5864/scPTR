"""Tests for variance decomposition."""

import numpy as np
import pytest


def test_variance_decomposition(preprocessed_adata):
    import scptr

    scptr.tl.estimate_beta(preprocessed_adata)
    scptr.tl.estimate_gamma(preprocessed_adata)
    scptr.tl.variance_decomposition(preprocessed_adata)

    assert "tf_score" in preprocessed_adata.var.columns
    assert "ptf_score" in preprocessed_adata.var.columns

    tf = preprocessed_adata.var["tf_score"].values
    ptf = preprocessed_adata.var["ptf_score"].values

    # TF + PTF should sum to ~1
    np.testing.assert_allclose(tf + ptf, 1.0, atol=1e-5)

    # Scores should be in [0, 1]
    assert np.all(tf >= 0) and np.all(tf <= 1)
    assert np.all(ptf >= 0) and np.all(ptf <= 1)
