"""Posterior sampling and extraction for DeepPTR."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from anndata import AnnData

    from ._model import DeepPTR


def posterior_gamma(
    model: "DeepPTR",
    adata: "AnnData",
    n_samples: int = 50,
    batch_size: int = 512,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute posterior mean and variance of gamma via MC sampling.

    Parameters
    ----------
    model
        Trained :class:`DeepPTR` model.
    adata
        AnnData with ``layers['spliced']`` and ``layers['unspliced']``.
    n_samples
        Number of MC samples from the posterior.
    batch_size
        Inference batch size.
    device
        Torch device.

    Returns
    -------
    gamma_mean, gamma_var : np.ndarray
        Each shape ``(n_obs, n_genes)``, dtype float32.
    """
    from scipy.sparse import issparse

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    model.eval()

    def _dense(mat):
        if issparse(mat):
            return np.asarray(mat.todense())
        return np.asarray(mat)

    s_np = _dense(adata.layers["spliced"]).astype(np.float32)
    u_np = _dense(adata.layers["unspliced"]).astype(np.float32)

    s_t = torch.from_numpy(s_np)
    u_t = torch.from_numpy(u_np)
    ds = TensorDataset(s_t, u_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    n_obs = adata.n_obs
    n_genes = adata.n_vars

    # Accumulators (Welford online mean/var)
    mean_acc = np.zeros((n_obs, n_genes), dtype=np.float64)
    m2_acc = np.zeros((n_obs, n_genes), dtype=np.float64)

    with torch.no_grad():
        for k in range(n_samples):
            gamma_list: list[np.ndarray] = []
            for s_b, u_b in loader:
                s_b = s_b.to(device)
                u_b = u_b.to(device)

                mu_T, logvar_T, mu_PT, logvar_PT = model.encoder(s_b, u_b)
                z_PT = model.reparameterize(mu_PT, logvar_PT)

                gamma_b = F.softplus(model.decoder.f_gamma(z_PT))
                gamma_list.append(gamma_b.cpu().numpy())

            gamma_k = np.concatenate(gamma_list, axis=0)  # (n_obs, n_genes)

            # Welford update
            delta = gamma_k - mean_acc
            mean_acc += delta / (k + 1)
            delta2 = gamma_k - mean_acc
            m2_acc += delta * delta2

    gamma_mean = mean_acc.astype(np.float32)
    gamma_var = (m2_acc / max(n_samples - 1, 1)).astype(np.float32)

    return gamma_mean, gamma_var


def extract_latent(
    model: "DeepPTR",
    adata: "AnnData",
    batch_size: int = 512,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract posterior mean of z_T and z_PT for all cells.

    Returns
    -------
    z_T, z_PT : np.ndarray
        Shapes ``(n_obs, d_T)`` and ``(n_obs, d_PT)``.
    """
    from scipy.sparse import issparse

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    model.eval()

    def _dense(mat):
        if issparse(mat):
            return np.asarray(mat.todense())
        return np.asarray(mat)

    s_np = _dense(adata.layers["spliced"]).astype(np.float32)
    u_np = _dense(adata.layers["unspliced"]).astype(np.float32)

    s_t = torch.from_numpy(s_np)
    u_t = torch.from_numpy(u_np)
    ds = TensorDataset(s_t, u_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    z_T_list: list[np.ndarray] = []
    z_PT_list: list[np.ndarray] = []

    with torch.no_grad():
        for s_b, u_b in loader:
            s_b = s_b.to(device)
            u_b = u_b.to(device)
            mu_T, _, mu_PT, _ = model.encoder(s_b, u_b)
            z_T_list.append(mu_T.cpu().numpy())
            z_PT_list.append(mu_PT.cpu().numpy())

    return (
        np.concatenate(z_T_list, axis=0).astype(np.float32),
        np.concatenate(z_PT_list, axis=0).astype(np.float32),
    )
