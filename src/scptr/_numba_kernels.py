"""Numba-JIT compiled kernels for scPTR."""

from __future__ import annotations

import numpy as np
import numba as nb


@nb.njit(parallel=True, cache=True)
def _smooth_kernel(
    data: np.ndarray,
    indices_flat: np.ndarray,
    indptr: np.ndarray,
    distances_flat: np.ndarray,
    bandwidths: np.ndarray,
    out: np.ndarray,
) -> None:
    """Gaussian kernel smoothing over kNN graph.

    Parameters
    ----------
    data : (n_obs, n_genes) float32
        Gene expression matrix to smooth.
    indices_flat : (nnz,) int32
        CSR indices array of the kNN graph.
    indptr : (n_obs+1,) int32
        CSR indptr array of the kNN graph.
    distances_flat : (nnz,) float32
        CSR data array (distances) of the kNN graph.
    bandwidths : (n_obs,) float32
        Per-cell bandwidth for the Gaussian kernel.
    out : (n_obs, n_genes) float32
        Output array (pre-allocated).
    """
    n_obs = data.shape[0]
    n_genes = data.shape[1]

    for i in nb.prange(n_obs):
        start = indptr[i]
        end = indptr[i + 1]
        bw = bandwidths[i]
        bw2 = bw * bw
        if bw2 < 1e-12:
            bw2 = 1e-12

        # Compute weights
        n_neighbors = end - start
        weights = np.empty(n_neighbors + 1, dtype=np.float32)
        neighbor_idx = np.empty(n_neighbors + 1, dtype=np.int64)

        # Self-connection with weight 1
        weights[0] = 1.0
        neighbor_idx[0] = i
        total_weight = 1.0

        for k in range(n_neighbors):
            j = indices_flat[start + k]
            d = distances_flat[start + k]
            w = np.exp(-0.5 * d * d / bw2)
            weights[k + 1] = w
            neighbor_idx[k + 1] = j
            total_weight += w

        # Weighted average
        inv_total = 1.0 / total_weight
        for g in range(n_genes):
            val = 0.0
            for k in range(n_neighbors + 1):
                val += weights[k] * data[neighbor_idx[k], g]
            out[i, g] = val * inv_total


@nb.njit(parallel=True, cache=True)
def _compute_adaptive_bandwidths(
    distances_flat: np.ndarray,
    indptr: np.ndarray,
) -> np.ndarray:
    """Compute per-cell adaptive bandwidth as median neighbor distance.

    Parameters
    ----------
    distances_flat : (nnz,) float32
    indptr : (n_obs+1,) int32

    Returns
    -------
    bandwidths : (n_obs,) float32
    """
    n_obs = indptr.shape[0] - 1
    bandwidths = np.empty(n_obs, dtype=np.float32)

    for i in nb.prange(n_obs):
        start = indptr[i]
        end = indptr[i + 1]
        n_neighbors = end - start
        if n_neighbors == 0:
            bandwidths[i] = 1.0
            continue

        # Copy distances for this cell and sort to get median
        dists = np.empty(n_neighbors, dtype=np.float32)
        for k in range(n_neighbors):
            dists[k] = distances_flat[start + k]
        dists.sort()

        mid = n_neighbors // 2
        if n_neighbors % 2 == 0:
            bandwidths[i] = (dists[mid - 1] + dists[mid]) / 2.0
        else:
            bandwidths[i] = dists[mid]

        if bandwidths[i] < 1e-6:
            bandwidths[i] = 1.0

    return bandwidths


@nb.njit(parallel=True, cache=True)
def _compute_gamma_kernel(
    u_smooth: np.ndarray,
    s_smooth: np.ndarray,
    beta: np.ndarray,
    clip_vals: np.ndarray,
    out: np.ndarray,
) -> None:
    """Compute per-cell per-gene degradation rate gamma.

    gamma[i,g] = beta[g] * u_smooth[i,g] / max(s_smooth[i,g], 1e-6)
    Clipped at clip_vals[g] per gene.

    Parameters
    ----------
    u_smooth : (n_obs, n_genes) float32
    s_smooth : (n_obs, n_genes) float32
    beta : (n_genes,) float32
    clip_vals : (n_genes,) float32
        Per-gene clip values (e.g., 99th percentile).
    out : (n_obs, n_genes) float32
    """
    n_obs = u_smooth.shape[0]
    n_genes = u_smooth.shape[1]

    for i in nb.prange(n_obs):
        for g in range(n_genes):
            s_val = s_smooth[i, g]
            if s_val < 1e-6:
                s_val = 1e-6
            gamma_val = beta[g] * u_smooth[i, g] / s_val
            if gamma_val > clip_vals[g]:
                gamma_val = clip_vals[g]
            if gamma_val < 0.0:
                gamma_val = 0.0
            out[i, g] = gamma_val


@nb.njit(parallel=True, cache=True)
def _velocity_kernel(
    gamma: np.ndarray,
    indices_flat: np.ndarray,
    indptr: np.ndarray,
    distances_flat: np.ndarray,
    bandwidths: np.ndarray,
    out: np.ndarray,
) -> None:
    """Compute PT velocity as weighted mean gamma difference from neighbors.

    v[i,g] = sum_j(w_ij * (gamma[j,g] - gamma[i,g]))

    Parameters
    ----------
    gamma : (n_obs, n_genes) float32
    indices_flat, indptr, distances_flat : kNN graph CSR arrays
    bandwidths : (n_obs,) float32
    out : (n_obs, n_genes) float32
    """
    n_obs = gamma.shape[0]
    n_genes = gamma.shape[1]

    for i in nb.prange(n_obs):
        start = indptr[i]
        end = indptr[i + 1]
        bw = bandwidths[i]
        bw2 = bw * bw
        if bw2 < 1e-12:
            bw2 = 1e-12

        n_neighbors = end - start
        if n_neighbors == 0:
            for g in range(n_genes):
                out[i, g] = 0.0
            continue

        # Compute weights
        total_weight = 0.0
        for k in range(n_neighbors):
            d = distances_flat[start + k]
            w = np.exp(-0.5 * d * d / bw2)
            total_weight += w

        if total_weight < 1e-12:
            for g in range(n_genes):
                out[i, g] = 0.0
            continue

        inv_total = 1.0 / total_weight

        for g in range(n_genes):
            val = 0.0
            for k in range(n_neighbors):
                j = indices_flat[start + k]
                d = distances_flat[start + k]
                w = np.exp(-0.5 * d * d / bw2)
                val += w * (gamma[j, g] - gamma[i, g])
            out[i, g] = val * inv_total
