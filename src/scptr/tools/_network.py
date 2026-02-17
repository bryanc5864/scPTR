"""RBP-target regulatory network inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from .._constants import GAMMA, SMOOTHED_SPLICED
from .._utils import get_layer, require_layers, log_params


def infer_network(
    adata: AnnData,
    regulators: list[str] | None = None,
    targets: list[str] | None = None,
    method: str = "elasticnet",
    alpha: float = 0.5,
    n_top: int = 50,
    prior_network: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Infer post-transcriptional regulatory network.

    Regresses gamma of target genes on expression of regulator genes
    (putative RNA-binding proteins) using elastic net regression.

    Parameters
    ----------
    adata
        Annotated data matrix with ``gamma`` and ``Ms`` layers.
    regulators
        Gene names to use as regulators. If ``None``, all genes are used.
    targets
        Gene names to use as targets. If ``None``, all genes are used.
    method
        Regression method: ``'elasticnet'`` (default).
    alpha
        Elastic net mixing parameter (0=ridge, 1=lasso).
    n_top
        Number of top edges to return per target.
    prior_network
        Optional DataFrame with columns ``['regulator', 'target', 'weight']``
        containing prior knowledge (e.g. from motif analysis). When provided,
        regulator columns are scaled by prior weight before fitting, and
        coefficients are rescaled back, effectively biasing the model toward
        known interactions.

    Returns
    -------
    DataFrame with columns ``['regulator', 'target', 'weight']``.
    """
    from sklearn.linear_model import ElasticNet

    require_layers(adata, GAMMA, SMOOTHED_SPLICED)

    gamma = get_layer(adata, GAMMA)
    expression = get_layer(adata, SMOOTHED_SPLICED)

    gene_names = adata.var_names.tolist()

    if regulators is None:
        reg_idx = list(range(adata.n_vars))
    else:
        reg_idx = [gene_names.index(g) for g in regulators if g in gene_names]

    if targets is None:
        tgt_idx = list(range(adata.n_vars))
    else:
        tgt_idx = [gene_names.index(g) for g in targets if g in gene_names]

    X_reg = expression[:, reg_idx]
    reg_names = [gene_names[i] for i in reg_idx]

    # Build prior lookup for fast access
    prior_lookup = {}
    if prior_network is not None:
        for _, row in prior_network.iterrows():
            prior_lookup[(row["regulator"], row["target"])] = float(row["weight"])

    edges = []
    for ti in tgt_idx:
        y = gamma[:, ti]
        if np.std(y) < 1e-8:
            continue

        tgt_name = gene_names[ti]

        # Apply prior scaling if available
        if prior_network is not None:
            scale_factors = np.ones(len(reg_names), dtype=np.float64)
            for ri, rname in enumerate(reg_names):
                pw = prior_lookup.get((rname, tgt_name), 0.0)
                # Scale: higher prior weight → less regularization effect
                # Use 1 + |pw| so default (no prior) = 1 and priors boost
                scale_factors[ri] = 1.0 + abs(pw)

            X_scaled = X_reg * scale_factors[np.newaxis, :]
        else:
            X_scaled = X_reg
            scale_factors = None

        model = ElasticNet(alpha=0.01, l1_ratio=alpha, max_iter=1000)
        model.fit(X_scaled, y)

        coefs = model.coef_.copy()

        # Rescale coefficients back if prior scaling was applied
        if scale_factors is not None:
            coefs = coefs * scale_factors

        # Get top edges by absolute weight
        top_k = min(n_top, len(coefs))
        top_idx = np.argsort(np.abs(coefs))[::-1][:top_k]

        for idx in top_idx:
            if abs(coefs[idx]) > 1e-6:
                edges.append({
                    "regulator": reg_names[idx],
                    "target": tgt_name,
                    "weight": float(coefs[idx]),
                })

    result = pd.DataFrame(edges)
    if len(result) > 0:
        result = result.sort_values("weight", key=abs, ascending=False)
        result = result.reset_index(drop=True)

    adata.uns["pt_network"] = result

    log_params(adata, "infer_network", {
        "method": method,
        "alpha": alpha,
        "n_regulators": len(reg_idx),
        "n_targets": len(tgt_idx),
        "n_edges": len(result),
        "has_prior": prior_network is not None,
    })

    return result
