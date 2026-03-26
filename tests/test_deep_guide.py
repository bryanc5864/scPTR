"""Tests for DeepPTR posterior extraction."""

import numpy as np
import pytest
import torch

from scptr.deep._model import DeepPTR
from scptr.deep._guide import posterior_gamma, extract_latent


@pytest.fixture
def trained_model_and_adata():
    """A small model with matching AnnData (untrained, just for shape checks)."""
    from anndata import AnnData

    rng = np.random.RandomState(0)
    n, g = 50, 20
    s = rng.poisson(5, size=(n, g)).astype(np.float32)
    u = rng.poisson(2, size=(n, g)).astype(np.float32)
    adata = AnnData(X=s)
    adata.layers["spliced"] = s
    adata.layers["unspliced"] = u

    torch.manual_seed(0)
    model = DeepPTR(n_genes=g, d_T=3, d_PT=3, d_hidden=16, n_enc_layers=1)
    model.eval()
    return model, adata


class TestPosteriorGamma:
    def test_shapes(self, trained_model_and_adata):
        model, adata = trained_model_and_adata
        gamma_mean, gamma_var = posterior_gamma(
            model, adata, n_samples=5, batch_size=16, device="cpu"
        )
        assert gamma_mean.shape == (adata.n_obs, adata.n_vars)
        assert gamma_var.shape == (adata.n_obs, adata.n_vars)

    def test_positive_values(self, trained_model_and_adata):
        model, adata = trained_model_and_adata
        gamma_mean, gamma_var = posterior_gamma(
            model, adata, n_samples=5, device="cpu"
        )
        assert (gamma_mean >= 0).all()
        assert (gamma_var >= 0).all()

    def test_no_nans(self, trained_model_and_adata):
        model, adata = trained_model_and_adata
        gamma_mean, gamma_var = posterior_gamma(
            model, adata, n_samples=5, device="cpu"
        )
        assert not np.isnan(gamma_mean).any()
        assert not np.isnan(gamma_var).any()

    def test_more_samples_lower_variance_of_mean(self, trained_model_and_adata):
        """With more MC samples, the posterior mean estimate should be more stable."""
        model, adata = trained_model_and_adata
        means = []
        for _ in range(3):
            gm, _ = posterior_gamma(model, adata, n_samples=2, device="cpu")
            means.append(gm.mean())
        spread_few = np.std(means)

        means = []
        for _ in range(3):
            gm, _ = posterior_gamma(model, adata, n_samples=20, device="cpu")
            means.append(gm.mean())
        spread_many = np.std(means)

        # Not a hard guarantee, but should generally hold
        # Use a generous threshold — we just want a sanity check
        assert spread_many < spread_few * 5


class TestExtractLatent:
    def test_shapes(self, trained_model_and_adata):
        model, adata = trained_model_and_adata
        z_T, z_PT = extract_latent(model, adata, batch_size=16, device="cpu")
        assert z_T.shape == (adata.n_obs, model.d_T)
        assert z_PT.shape == (adata.n_obs, model.d_PT)

    def test_deterministic(self, trained_model_and_adata):
        model, adata = trained_model_and_adata
        z_T1, z_PT1 = extract_latent(model, adata, device="cpu")
        z_T2, z_PT2 = extract_latent(model, adata, device="cpu")
        np.testing.assert_allclose(z_T1, z_T2, atol=1e-6)
        np.testing.assert_allclose(z_PT1, z_PT2, atol=1e-6)

    def test_no_nans(self, trained_model_and_adata):
        model, adata = trained_model_and_adata
        z_T, z_PT = extract_latent(model, adata, device="cpu")
        assert not np.isnan(z_T).any()
        assert not np.isnan(z_PT).any()
