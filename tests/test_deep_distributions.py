"""Tests for NB distribution utilities."""

import numpy as np
import pytest
import torch

from scptr.deep._distributions import log_nb_positive


class TestLogNBPositive:
    def test_output_shape(self):
        x = torch.tensor([[1.0, 2.0, 3.0]])
        mu = torch.tensor([[2.0, 2.0, 2.0]])
        theta = torch.tensor([5.0, 5.0, 5.0])
        ll = log_nb_positive(x, mu, theta)
        assert ll.shape == (1, 3)

    def test_non_positive(self):
        """Log-probabilities should be <= 0."""
        x = torch.randint(0, 20, (100, 50)).float()
        mu = torch.rand(100, 50) * 10 + 0.1
        theta = torch.rand(50) * 10 + 0.1
        ll = log_nb_positive(x, mu, theta)
        assert (ll <= 1e-5).all(), "Log probabilities should be non-positive"

    def test_peak_at_mean(self):
        """For integer means, likelihood should peak near the mean."""
        mu = torch.tensor([[10.0]])
        theta = torch.tensor([50.0])  # low dispersion
        xs = torch.arange(0, 30).float().unsqueeze(1)
        ll = log_nb_positive(xs, mu.expand(30, 1), theta)
        peak = ll.argmax().item()
        assert abs(peak - 10) <= 2, f"Peak at {peak}, expected near 10"

    def test_higher_theta_less_variance(self):
        """Higher theta (less dispersion) should give sharper distribution."""
        mu = torch.tensor([[5.0]])
        xs = torch.arange(0, 20).float().unsqueeze(1)

        theta_low = torch.tensor([1.0])
        theta_high = torch.tensor([100.0])

        ll_low = log_nb_positive(xs, mu.expand(20, 1), theta_low)
        ll_high = log_nb_positive(xs, mu.expand(20, 1), theta_high)

        # High theta should have higher peak probability
        assert ll_high.max() > ll_low.max()

    def test_gradient_flows(self):
        """Ensure gradients flow through all parameters."""
        x = torch.tensor([[3.0, 5.0]])
        mu = torch.tensor([[2.0, 4.0]], requires_grad=True)
        theta = torch.tensor([5.0, 5.0], requires_grad=True)
        ll = log_nb_positive(x, mu, theta).sum()
        ll.backward()
        assert mu.grad is not None
        assert theta.grad is not None
        assert not torch.isnan(mu.grad).any()
        assert not torch.isnan(theta.grad).any()

    def test_batch_consistency(self):
        """Batched computation should match individual computation."""
        torch.manual_seed(42)
        x = torch.randint(0, 10, (5, 3)).float()
        mu = torch.rand(5, 3) * 5 + 0.1
        theta = torch.rand(3) * 5 + 0.1

        ll_batch = log_nb_positive(x, mu, theta)
        for i in range(5):
            ll_single = log_nb_positive(
                x[i : i + 1], mu[i : i + 1], theta
            )
            assert torch.allclose(ll_batch[i], ll_single[0], atol=1e-5)
