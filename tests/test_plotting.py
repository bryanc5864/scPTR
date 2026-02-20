"""Smoke tests for plotting functions."""

import matplotlib
matplotlib.use("Agg")

import pytest


def test_phase_portrait(analyzed_adata):
    import scptr

    gene = analyzed_adata.var_names[0]
    fig = scptr.pl.phase_portrait(analyzed_adata, gene, show=False)
    assert fig is not None


def test_gamma_heatmap(analyzed_adata):
    import scptr

    fig = scptr.pl.gamma_heatmap(analyzed_adata, show=False)
    assert fig is not None


def test_gamma_violin(analyzed_adata):
    import scptr

    gene = analyzed_adata.var_names[0]
    fig = scptr.pl.gamma_violin(analyzed_adata, gene, show=False)
    assert fig is not None


def test_pt_umap(analyzed_adata):
    import scptr

    fig = scptr.pl.pt_umap(analyzed_adata, show=False)
    assert fig is not None


def test_pt_comparison(analyzed_adata):
    import scptr

    fig = scptr.pl.pt_comparison(analyzed_adata, show=False)
    assert fig is not None


def test_tf_ptf_scatter(analyzed_adata):
    import scptr

    fig = scptr.pl.tf_ptf_scatter(analyzed_adata, show=False)
    assert fig is not None


def test_pt_velocity_embedding(analyzed_adata):
    import scptr

    scptr.tl.pt_velocity(analyzed_adata)
    fig = scptr.pl.pt_velocity_embedding(analyzed_adata, show=False)
    assert fig is not None


def test_pt_velocity_stream(analyzed_adata):
    import scptr

    scptr.tl.pt_velocity(analyzed_adata)
    fig = scptr.pl.pt_velocity_stream(analyzed_adata, grid_size=20, show=False)
    assert fig is not None
