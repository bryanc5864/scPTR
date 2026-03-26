"""Evaluation metrics for synthetic recovery experiments."""

from __future__ import annotations

import numpy as np
from scipy import stats


def gamma_recovery(
    gamma_true: np.ndarray,
    gamma_pred: np.ndarray,
    per_gene: bool = True,
) -> float | np.ndarray:
    """Spearman correlation between true and predicted gamma.

    Parameters
    ----------
    gamma_true, gamma_pred
        Shape ``(n_cells, n_genes)``.
    per_gene
        If True, compute per-gene Spearman r and return the median.
        If False, flatten and compute a single global r.

    Returns
    -------
    float or np.ndarray
        Median per-gene r (if per_gene=True) or global r.
    """
    if per_gene:
        n_genes = gamma_true.shape[1]
        rs = np.empty(n_genes, dtype=np.float64)
        for g in range(n_genes):
            gt = gamma_true[:, g]
            gp = gamma_pred[:, g]
            if gt.std() < 1e-10 or gp.std() < 1e-10:
                rs[g] = 0.0
            else:
                rs[g] = stats.spearmanr(gt, gp).statistic
        return float(np.nanmedian(rs))
    else:
        return float(stats.spearmanr(gamma_true.ravel(), gamma_pred.ravel()).statistic)


def ci_coverage(
    gamma_true: np.ndarray,
    gamma_mean: np.ndarray,
    gamma_var: np.ndarray,
    level: float = 0.95,
) -> float:
    """Fraction of true gamma values within the posterior credible interval.

    Uses a Gaussian approximation: CI = mean +/- z * std.

    Parameters
    ----------
    gamma_true
        Ground-truth gamma, shape ``(n_cells, n_genes)``.
    gamma_mean
        Posterior mean, same shape.
    gamma_var
        Posterior variance, same shape.
    level
        Credible interval level (e.g. 0.95).

    Returns
    -------
    float
        Coverage fraction in [0, 1].
    """
    from scipy.stats import norm

    z = norm.ppf(0.5 + level / 2)
    std = np.sqrt(np.clip(gamma_var, 1e-10, None))
    lower = gamma_mean - z * std
    upper = gamma_mean + z * std
    inside = (gamma_true >= lower) & (gamma_true <= upper)
    return float(inside.mean())


def latent_recovery(
    z_true: np.ndarray,
    z_pred: np.ndarray,
) -> float:
    """Mean canonical correlation (CCA) between true and predicted latents.

    Parameters
    ----------
    z_true, z_pred
        Shape ``(n_cells, d_latent)``.

    Returns
    -------
    float
        Mean canonical correlation across components.
    """
    from sklearn.cross_decomposition import CCA

    d = min(z_true.shape[1], z_pred.shape[1])
    cca = CCA(n_components=d, max_iter=500)
    X_c, Y_c = cca.fit_transform(z_true, z_pred)

    correlations = np.array(
        [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(d)]
    )
    return float(np.mean(np.abs(correlations)))
