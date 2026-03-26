"""Tests for DeepPTR synthetic data generation and metrics."""

import numpy as np
import pytest

from scptr.deep.synthetic import generate_kinetic_data, gamma_recovery, ci_coverage, latent_recovery


class TestGenerateKineticData:
    def test_shapes(self):
        adata, truth = generate_kinetic_data(n_cells=100, n_genes=30, seed=0)
        assert adata.n_obs == 100
        assert adata.n_vars == 30
        assert adata.layers["spliced"].shape == (100, 30)
        assert adata.layers["unspliced"].shape == (100, 30)

    def test_truth_keys(self):
        _, truth = generate_kinetic_data(n_cells=50, n_genes=20)
        for key in ("alpha", "gamma", "beta", "z_T", "z_PT"):
            assert key in truth

    def test_truth_shapes(self):
        adata, truth = generate_kinetic_data(n_cells=50, n_genes=20)
        assert truth["alpha"].shape == (50, 20)
        assert truth["gamma"].shape == (50, 20)
        assert truth["beta"].shape == (20,)
        assert truth["z_T"].shape[0] == 50
        assert truth["z_PT"].shape[0] == 50

    def test_non_negative_counts(self):
        adata, _ = generate_kinetic_data(n_cells=200, n_genes=50)
        assert (adata.layers["spliced"] >= 0).all()
        assert (adata.layers["unspliced"] >= 0).all()

    def test_cell_types(self):
        adata, _ = generate_kinetic_data(n_cells=100, n_cell_types=4)
        assert "cell_type" in adata.obs.columns
        assert adata.obs["cell_type"].nunique() <= 4

    def test_sparsity(self):
        adata_sparse, _ = generate_kinetic_data(n_cells=500, n_genes=100, sparsity=0.5)
        adata_dense, _ = generate_kinetic_data(n_cells=500, n_genes=100, sparsity=0.0)
        frac_zero_sparse = (adata_sparse.layers["spliced"] == 0).mean()
        frac_zero_dense = (adata_dense.layers["spliced"] == 0).mean()
        assert frac_zero_sparse > frac_zero_dense

    def test_reproducible(self):
        adata1, t1 = generate_kinetic_data(seed=42)
        adata2, t2 = generate_kinetic_data(seed=42)
        np.testing.assert_array_equal(
            adata1.layers["spliced"], adata2.layers["spliced"]
        )
        np.testing.assert_array_equal(t1["gamma"], t2["gamma"])


class TestGammaRecovery:
    def test_perfect_recovery(self):
        rng = np.random.RandomState(0)
        gamma = rng.rand(100, 20).astype(np.float32)
        r = gamma_recovery(gamma, gamma, per_gene=True)
        assert r > 0.99

    def test_random_is_low(self):
        rng = np.random.RandomState(0)
        g1 = rng.rand(100, 20).astype(np.float32)
        g2 = rng.rand(100, 20).astype(np.float32)
        r = gamma_recovery(g1, g2, per_gene=True)
        assert abs(r) < 0.3

    def test_global_mode(self):
        rng = np.random.RandomState(0)
        gamma = rng.rand(100, 20).astype(np.float32)
        r = gamma_recovery(gamma, gamma, per_gene=False)
        assert r > 0.99


class TestCICoverage:
    def test_perfect_coverage(self):
        rng = np.random.RandomState(0)
        gamma = rng.rand(100, 20).astype(np.float32)
        # Huge variance → everything covered
        cov = ci_coverage(gamma, gamma, np.ones_like(gamma) * 100.0)
        assert cov > 0.99

    def test_zero_variance_coverage(self):
        rng = np.random.RandomState(0)
        gamma_true = rng.rand(100, 20).astype(np.float32)
        gamma_pred = gamma_true + 1.0  # shifted
        # Tiny variance → nothing covered
        cov = ci_coverage(gamma_true, gamma_pred, np.ones_like(gamma_true) * 1e-10)
        assert cov < 0.05

    def test_returns_fraction(self):
        rng = np.random.RandomState(0)
        gamma = rng.rand(50, 10).astype(np.float32)
        cov = ci_coverage(gamma, gamma, np.ones_like(gamma) * 0.1)
        assert 0.0 <= cov <= 1.0


class TestLatentRecovery:
    def test_perfect_recovery(self):
        rng = np.random.RandomState(0)
        z = rng.randn(100, 5).astype(np.float32)
        r = latent_recovery(z, z)
        assert r > 0.99

    def test_random_is_lower(self):
        rng = np.random.RandomState(0)
        z1 = rng.randn(100, 5).astype(np.float32)
        z2 = rng.randn(100, 5).astype(np.float32)
        r = latent_recovery(z1, z2)
        assert r < 0.5
