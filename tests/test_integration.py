"""Integration test: full pipeline on synthetic data."""

import numpy as np
import pytest


def test_full_pipeline(synthetic_adata):
    """Run the complete scPTR pipeline on synthetic data."""
    import scptr

    adata = synthetic_adata

    # Preprocessing
    scptr.pp.filter_genes(adata, min_unspliced_counts=1, min_unspliced_cells=1)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)

    assert "Mu" in adata.layers
    assert "Ms" in adata.layers

    # Core estimation
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)
    scptr.tl.variance_decomposition(adata)

    assert "beta" in adata.var.columns
    assert "gamma" in adata.layers
    assert "tf_score" in adata.var.columns
    assert "ptf_score" in adata.var.columns

    # PT states
    scptr.tl.pt_states(adata)
    assert "pt_state" in adata.obs.columns
    assert "X_gamma_pca" in adata.obsm
    assert "X_gamma_umap" in adata.obsm

    # Rank genes
    result = scptr.tl.rank_pt_genes(adata)
    assert len(result) > 0

    # PT velocity
    scptr.tl.pt_velocity(adata)
    assert "pt_velocity" in adata.layers
    assert adata.layers["pt_velocity"].shape == adata.shape

    # Check all uns parameters logged
    assert "scptr" in adata.uns
    params = adata.uns["scptr"]
    for step in [
        "filter_genes", "normalize_layers", "neighbors",
        "smooth_layers", "estimate_beta", "estimate_gamma",
        "variance_decomposition", "pt_states", "rank_pt_genes",
        "pt_velocity",
    ]:
        assert step in params, f"Missing params for {step}"


def test_network_inference(analyzed_adata):
    """Test network inference on analyzed data."""
    import scptr

    # Use a small subset of genes as regulators/targets for speed
    genes = analyzed_adata.var_names[:20].tolist()
    result = scptr.tl.infer_network(
        analyzed_adata,
        regulators=genes,
        targets=genes[:5],
    )
    assert "pt_network" in analyzed_adata.uns
    assert "infer_network" in analyzed_adata.uns["scptr"]
