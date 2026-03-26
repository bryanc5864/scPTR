"""Tests for DeepPTR model components."""

import numpy as np
import pytest
import torch

from scptr.deep._model import DeepPTR, Encoder, KineticDecoder


@pytest.fixture
def model_dims():
    return {"n_genes": 50, "d_T": 5, "d_PT": 5, "d_hidden": 32}


@pytest.fixture
def batch_data(model_dims):
    """Synthetic batch of data."""
    torch.manual_seed(0)
    n = 16
    G = model_dims["n_genes"]
    return {
        "s": torch.rand(n, G) * 10,
        "u": torch.rand(n, G) * 5,
        "l_s": torch.rand(n) * 1000 + 100,
        "l_u": torch.rand(n) * 500 + 50,
    }


class TestEncoder:
    def test_output_shapes(self, model_dims):
        enc = Encoder(
            n_genes=model_dims["n_genes"],
            d_hidden=model_dims["d_hidden"],
            d_T=model_dims["d_T"],
            d_PT=model_dims["d_PT"],
        )
        s = torch.rand(8, model_dims["n_genes"])
        u = torch.rand(8, model_dims["n_genes"])
        mu_T, logvar_T, mu_PT, logvar_PT = enc(s, u)
        assert mu_T.shape == (8, model_dims["d_T"])
        assert logvar_T.shape == (8, model_dims["d_T"])
        assert mu_PT.shape == (8, model_dims["d_PT"])
        assert logvar_PT.shape == (8, model_dims["d_PT"])

    def test_different_inputs_different_outputs(self, model_dims):
        enc = Encoder(
            n_genes=model_dims["n_genes"],
            d_hidden=model_dims["d_hidden"],
            d_T=model_dims["d_T"],
            d_PT=model_dims["d_PT"],
        )
        s1 = torch.rand(1, model_dims["n_genes"])
        u1 = torch.rand(1, model_dims["n_genes"])
        s2 = torch.rand(1, model_dims["n_genes"]) + 5
        u2 = torch.rand(1, model_dims["n_genes"]) + 5

        out1 = enc(s1, u1)
        out2 = enc(s2, u2)
        assert not torch.allclose(out1[0], out2[0])


class TestKineticDecoder:
    def test_output_shapes(self, model_dims):
        dec = KineticDecoder(
            n_genes=model_dims["n_genes"],
            d_T=model_dims["d_T"],
            d_PT=model_dims["d_PT"],
            d_hidden=model_dims["d_hidden"],
        )
        z_T = torch.randn(8, model_dims["d_T"])
        z_PT = torch.randn(8, model_dims["d_PT"])
        l_s = torch.ones(8) * 1000
        l_u = torch.ones(8) * 500
        out = dec(z_T, z_PT, l_s, l_u)

        G = model_dims["n_genes"]
        assert out["mu_s"].shape == (8, G)
        assert out["mu_u"].shape == (8, G)
        assert out["alpha"].shape == (8, G)
        assert out["gamma"].shape == (8, G)
        assert out["beta"].shape == (G,)
        assert out["theta_s"].shape == (G,)
        assert out["theta_u"].shape == (G,)

    def test_positive_outputs(self, model_dims):
        dec = KineticDecoder(
            n_genes=model_dims["n_genes"],
            d_T=model_dims["d_T"],
            d_PT=model_dims["d_PT"],
            d_hidden=model_dims["d_hidden"],
        )
        z_T = torch.randn(16, model_dims["d_T"])
        z_PT = torch.randn(16, model_dims["d_PT"])
        l_s = torch.ones(16) * 1000
        l_u = torch.ones(16) * 500
        out = dec(z_T, z_PT, l_s, l_u)

        for key in ("mu_s", "mu_u", "alpha", "gamma", "beta", "theta_s", "theta_u"):
            assert (out[key] >= 0).all(), f"{key} has negative values"

    def test_mu_scales_with_library_size(self, model_dims):
        dec = KineticDecoder(
            n_genes=model_dims["n_genes"],
            d_T=model_dims["d_T"],
            d_PT=model_dims["d_PT"],
            d_hidden=model_dims["d_hidden"],
        )
        z_T = torch.randn(1, model_dims["d_T"])
        z_PT = torch.randn(1, model_dims["d_PT"])

        out1 = dec(z_T, z_PT, torch.tensor([100.0]), torch.tensor([100.0]))
        out2 = dec(z_T, z_PT, torch.tensor([1000.0]), torch.tensor([1000.0]))

        ratio_s = out2["mu_s"].sum() / out1["mu_s"].sum()
        assert abs(ratio_s.item() - 10.0) < 1.0

    def test_beta_is_not_cell_specific(self, model_dims):
        """Beta should be the same regardless of input."""
        dec = KineticDecoder(
            n_genes=model_dims["n_genes"],
            d_T=model_dims["d_T"],
            d_PT=model_dims["d_PT"],
            d_hidden=model_dims["d_hidden"],
        )
        z_T1 = torch.randn(4, model_dims["d_T"])
        z_PT1 = torch.randn(4, model_dims["d_PT"])
        z_T2 = torch.randn(4, model_dims["d_T"])
        z_PT2 = torch.randn(4, model_dims["d_PT"])

        out1 = dec(z_T1, z_PT1, torch.ones(4), torch.ones(4))
        out2 = dec(z_T2, z_PT2, torch.ones(4), torch.ones(4))
        assert torch.allclose(out1["beta"], out2["beta"])


class TestDeepPTR:
    def test_forward_loss(self, model_dims, batch_data):
        model = DeepPTR(**model_dims)
        out = model(**batch_data)
        assert "loss" in out
        assert "recon_loss" in out
        assert "kl_loss" in out
        assert not torch.isnan(out["loss"])

    def test_backward(self, model_dims, batch_data):
        model = DeepPTR(**model_dims)
        out = model(**batch_data)
        out["loss"].backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"

    def test_kl_weight_zero(self, model_dims, batch_data):
        model = DeepPTR(**model_dims)
        out_0 = model(**batch_data, kl_weight=0.0)
        out_1 = model(**batch_data, kl_weight=1.0)
        # With kl_weight=0, loss should equal recon_loss
        assert torch.allclose(out_0["loss"], out_0["recon_loss"], atol=1e-5)
        # With kl_weight=1, loss > recon_loss (KL >= 0)
        assert out_1["loss"] >= out_1["recon_loss"] - 1e-5

    def test_get_latent(self, model_dims, batch_data):
        model = DeepPTR(**model_dims)
        mu_T, logvar_T, mu_PT, logvar_PT = model.get_latent(
            batch_data["s"], batch_data["u"]
        )
        assert mu_T.shape == (16, model_dims["d_T"])
        assert mu_PT.shape == (16, model_dims["d_PT"])

    def test_reparameterize_stochastic(self, model_dims):
        mu = torch.zeros(10, model_dims["d_T"])
        logvar = torch.zeros(10, model_dims["d_T"])
        z1 = DeepPTR.reparameterize(mu, logvar)
        z2 = DeepPTR.reparameterize(mu, logvar)
        # Two samples should differ (with overwhelming probability)
        assert not torch.allclose(z1, z2)
