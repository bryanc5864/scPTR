"""Tests for PT state discovery."""

import numpy as np
import pytest


def test_pt_states(analyzed_adata):
    assert "pt_state" in analyzed_adata.obs.columns
    assert "X_gamma_pca" in analyzed_adata.obsm
    assert "X_gamma_umap" in analyzed_adata.obsm
    assert analyzed_adata.obs["pt_state"].dtype.name == "category"
    n_states = analyzed_adata.obs["pt_state"].nunique()
    assert n_states >= 1


def test_rank_pt_genes(analyzed_adata):
    import scptr

    result = scptr.tl.rank_pt_genes(analyzed_adata, groupby="pt_state")
    assert len(result) > 0
    assert "names" in result.columns or "group" in result.columns
    assert "rank_pt_genes" in analyzed_adata.uns["scptr"]
