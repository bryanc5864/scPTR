"""Tests for per-cell-type beta estimation (groupby parameter)."""

import numpy as np
import pytest


def test_groupby_produces_varm(preprocessed_adata):
    """groupby stores per-group betas in adata.varm['beta_groups']."""
    import scptr

    preprocessed_adata.obs["cell_type"] = np.random.choice(
        ["A", "B", "C"], size=preprocessed_adata.n_obs
    )
    scptr.tl.estimate_beta(preprocessed_adata, groupby="cell_type")

    assert "beta_groups" in preprocessed_adata.varm
    assert preprocessed_adata.varm["beta_groups"].shape == (
        preprocessed_adata.n_vars,
        3,
    )
    assert "beta" in preprocessed_adata.var.columns


def test_groupby_single_group_matches_global(preprocessed_adata):
    """When there is only one group, groupby result matches global."""
    import scptr

    # Global estimation
    scptr.tl.estimate_beta(preprocessed_adata)
    beta_global = preprocessed_adata.var["beta"].values.copy()

    # Single-group estimation
    preprocessed_adata.obs["one_group"] = "all"
    scptr.tl.estimate_beta(preprocessed_adata, groupby="one_group")
    beta_grouped = preprocessed_adata.var["beta"].values.copy()

    np.testing.assert_allclose(beta_global, beta_grouped, rtol=1e-5)


def test_groupby_missing_column_raises(preprocessed_adata):
    """groupby with nonexistent column raises KeyError."""
    import scptr

    with pytest.raises(KeyError, match="Missing required obs columns"):
        scptr.tl.estimate_beta(preprocessed_adata, groupby="nonexistent")


def test_groupby_consensus_is_clipped_median(preprocessed_adata):
    """Consensus beta is median across group betas, then globally clipped."""
    import scptr

    preprocessed_adata.obs["cell_type"] = np.random.choice(
        ["X", "Y"], size=preprocessed_adata.n_obs
    )
    scptr.tl.estimate_beta(preprocessed_adata, groupby="cell_type")

    beta_groups = preprocessed_adata.varm["beta_groups"]
    raw_median = np.nanmedian(beta_groups.values, axis=1)
    # Apply the same global clip as the implementation
    positive = raw_median[raw_median > 0]
    if len(positive) > 0:
        cap = np.percentile(positive, 99)
        expected = np.clip(raw_median, 0, cap).astype(np.float32)
    else:
        expected = raw_median.astype(np.float32)
    actual = preprocessed_adata.var["beta"].values

    np.testing.assert_allclose(actual, expected, rtol=1e-5)
