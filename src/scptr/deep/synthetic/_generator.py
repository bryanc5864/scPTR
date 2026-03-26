"""Generate synthetic scRNA-seq data from known kinetic parameters + NB noise."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def generate_kinetic_data(
    n_cells: int = 3000,
    n_genes: int = 200,
    n_cell_types: int = 5,
    dispersion: float = 10.0,
    sparsity: float = 0.3,
    seed: int = 0,
) -> tuple[AnnData, dict[str, np.ndarray]]:
    """Generate (u, s) counts from a kinetic model with NB observation noise.

    The generative process:
      1. Sample latent factors z_T, z_PT per cell (different means per type).
      2. Derive alpha = softplus(W_alpha @ z_T), gamma = softplus(W_gamma @ z_PT).
      3. Set beta as gene-specific constants.
      4. Compute mu_u ∝ alpha/beta, mu_s ∝ alpha/gamma, scaled by library size.
      5. Draw counts from NB(mu, theta=dispersion).
      6. Apply zero-inflation (dropout) at rate ``sparsity``.

    Parameters
    ----------
    n_cells
        Number of cells.
    n_genes
        Number of genes.
    n_cell_types
        Number of simulated cell types.
    dispersion
        NB inverse dispersion (higher = less noise).
    sparsity
        Fraction of zeros injected (dropout).
    seed
        Random seed.

    Returns
    -------
    adata : AnnData
        With layers ``'spliced'``, ``'unspliced'``, and obs ``'cell_type'``.
    truth : dict
        Ground-truth arrays: ``alpha``, ``gamma``, ``beta``, ``z_T``, ``z_PT``.
    """
    rng = np.random.RandomState(seed)

    d_latent = 10  # latent dimension

    # Cell-type assignments
    cell_types = rng.choice(n_cell_types, size=n_cells)

    # Cell-type-specific latent means
    type_means_T = rng.randn(n_cell_types, d_latent).astype(np.float32)
    type_means_PT = rng.randn(n_cell_types, d_latent).astype(np.float32)

    z_T = type_means_T[cell_types] + 0.3 * rng.randn(n_cells, d_latent).astype(
        np.float32
    )
    z_PT = type_means_PT[cell_types] + 0.3 * rng.randn(n_cells, d_latent).astype(
        np.float32
    )

    # Decoder weights (fixed ground truth)
    W_alpha = rng.randn(d_latent, n_genes).astype(np.float32) * 0.5
    W_gamma = rng.randn(d_latent, n_genes).astype(np.float32) * 0.5

    # Kinetic parameters
    alpha = _softplus(z_T @ W_alpha)  # (n_cells, n_genes)
    gamma = _softplus(z_PT @ W_gamma)  # (n_cells, n_genes)
    beta = np.exp(rng.randn(n_genes).astype(np.float32) * 0.5 + 1.0)  # gene-specific

    # Expected counts (proportional)
    eps = 1e-8
    mu_u_raw = alpha / (beta[np.newaxis, :] + eps)
    mu_s_raw = alpha / (gamma + eps)

    # Library sizes
    l_u = rng.lognormal(mean=8.0, sigma=0.5, size=n_cells).astype(np.float32)
    l_s = rng.lognormal(mean=9.0, sigma=0.5, size=n_cells).astype(np.float32)

    # Normalize to proportions then scale by library size
    mu_u = (mu_u_raw / (mu_u_raw.sum(axis=1, keepdims=True) + eps)) * l_u[:, None]
    mu_s = (mu_s_raw / (mu_s_raw.sum(axis=1, keepdims=True) + eps)) * l_s[:, None]

    # NB sampling
    u_counts = _sample_nb(mu_u, dispersion, rng)
    s_counts = _sample_nb(mu_s, dispersion, rng)

    # Dropout
    if sparsity > 0:
        mask_u = rng.rand(n_cells, n_genes) > sparsity
        mask_s = rng.rand(n_cells, n_genes) > sparsity
        u_counts = u_counts * mask_u
        s_counts = s_counts * mask_s

    adata = AnnData(
        X=s_counts.astype(np.float32),
        layers={
            "spliced": s_counts.astype(np.float32),
            "unspliced": u_counts.astype(np.float32),
        },
    )
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs["cell_type"] = [f"type_{t}" for t in cell_types]
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")

    truth = {
        "alpha": alpha.astype(np.float32),
        "gamma": gamma.astype(np.float32),
        "beta": beta.astype(np.float32),
        "z_T": z_T,
        "z_PT": z_PT,
    }

    return adata, truth


def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus."""
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -20, 20))))


def _sample_nb(
    mu: np.ndarray, theta: float, rng: np.random.RandomState
) -> np.ndarray:
    """Sample from NB(mu, theta) using gamma-Poisson mixture."""
    mu = np.clip(mu, 1e-8, None)
    # Shape-rate parameterization: shape=theta, rate=theta/mu
    p = theta / (theta + mu)
    counts = rng.negative_binomial(theta, p)
    return counts.astype(np.float32)
