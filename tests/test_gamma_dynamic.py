"""Tests for dynamic-mode gamma estimation."""

import numpy as np
import pytest


def test_dynamic_mode_produces_gamma(preprocessed_adata):
    """Dynamic mode produces a gamma layer."""
    import scptr

    scptr.tl.estimate_beta(preprocessed_adata)

    # Create a synthetic velocity layer (ds/dt estimate)
    n_obs, n_vars = preprocessed_adata.shape
    preprocessed_adata.layers["velocity_S"] = np.random.randn(n_obs, n_vars).astype(
        np.float32
    )

    scptr.tl.estimate_gamma(
        preprocessed_adata, mode="dynamic", velocity_layer="velocity_S"
    )

    assert "gamma" in preprocessed_adata.layers
    assert preprocessed_adata.layers["gamma"].shape == (n_obs, n_vars)
    # Dynamic gamma should be non-negative (clipped)
    assert np.all(preprocessed_adata.layers["gamma"] >= 0)


def test_dynamic_mode_logs_params(preprocessed_adata):
    """Dynamic mode logs mode and velocity_layer in uns."""
    import scptr

    scptr.tl.estimate_beta(preprocessed_adata)
    n_obs, n_vars = preprocessed_adata.shape
    preprocessed_adata.layers["velocity_S"] = np.random.randn(n_obs, n_vars).astype(
        np.float32
    )

    scptr.tl.estimate_gamma(
        preprocessed_adata, mode="dynamic", velocity_layer="velocity_S"
    )

    params = preprocessed_adata.uns["scptr"]["estimate_gamma"]
    assert params["mode"] == "dynamic"
    assert params["velocity_layer"] == "velocity_S"


def test_dynamic_requires_velocity_layer(preprocessed_adata):
    """Dynamic mode without velocity_layer raises ValueError."""
    import scptr

    scptr.tl.estimate_beta(preprocessed_adata)

    with pytest.raises(ValueError, match="velocity_layer must be provided"):
        scptr.tl.estimate_gamma(preprocessed_adata, mode="dynamic")


def test_steady_state_mode_default(preprocessed_adata):
    """Default mode is steady_state and works as before."""
    import scptr

    scptr.tl.estimate_beta(preprocessed_adata)
    scptr.tl.estimate_gamma(preprocessed_adata)

    params = preprocessed_adata.uns["scptr"]["estimate_gamma"]
    assert params["mode"] == "steady_state"


def test_unknown_mode_raises(preprocessed_adata):
    """Unknown mode raises ValueError."""
    import scptr

    scptr.tl.estimate_beta(preprocessed_adata)

    with pytest.raises(ValueError, match="Unknown mode"):
        scptr.tl.estimate_gamma(preprocessed_adata, mode="invalid")
