"""Tests for the benchmark module."""

import numpy as np
import pandas as pd
import pytest


def test_correlate_with_halflives(analyzed_adata):
    """correlate_with_halflives returns correlation dict."""
    from scptr.benchmark import correlate_with_halflives

    # Create synthetic half-life data matching some gene names
    gene_names = analyzed_adata.var_names.tolist()
    hl_df = pd.DataFrame({
        "gene_symbol": gene_names[:50],
        "half_life_hours": np.random.exponential(5, size=50),
    })

    result = correlate_with_halflives(analyzed_adata, hl_df)

    assert "spearman_r" in result
    assert "pearson_r" in result
    assert "n_genes" in result
    assert result["n_genes"] == 50


def test_correlate_no_overlap(analyzed_adata):
    """correlate_with_halflives handles no gene overlap gracefully."""
    from scptr.benchmark import correlate_with_halflives

    hl_df = pd.DataFrame({
        "gene_symbol": ["FAKE_GENE_1", "FAKE_GENE_2"],
        "half_life_hours": [5.0, 10.0],
    })

    result = correlate_with_halflives(analyzed_adata, hl_df)
    assert result["n_genes"] == 0
    assert np.isnan(result["spearman_r"])


def test_are_enrichment(analyzed_adata):
    """are_enrichment runs without error."""
    from scptr.benchmark import are_enrichment

    result = are_enrichment(analyzed_adata)
    assert "label" in result
    assert result["label"] == "ARE"
    assert "p_value" in result


def test_nmd_enrichment(analyzed_adata):
    """nmd_enrichment runs without error."""
    from scptr.benchmark import nmd_enrichment

    result = nmd_enrichment(analyzed_adata)
    assert "label" in result
    assert result["label"] == "NMD"
    assert "p_value" in result


def test_subsampling_robustness(analyzed_adata):
    """subsampling_robustness returns DataFrame with expected columns."""
    from scptr.benchmark import subsampling_robustness

    result = subsampling_robustness(
        analyzed_adata, fractions=[0.5, 0.8], n_repeats=2
    )

    assert isinstance(result, pd.DataFrame)
    assert "fraction" in result.columns
    assert "spearman_r" in result.columns
    assert "pearson_r" in result.columns
    assert len(result) == 4  # 2 fractions * 2 repeats


def test_cross_dataset_consistency(analyzed_adata):
    """cross_dataset_consistency works with multiple copies."""
    from scptr.benchmark import cross_dataset_consistency

    result = cross_dataset_consistency({
        "dataset_A": analyzed_adata,
        "dataset_B": analyzed_adata,
    })

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1  # 1 pair
    assert "spearman_r" in result.columns
    # Same data should have perfect correlation
    assert result["spearman_r"].iloc[0] > 0.99


def test_enrichment_barplot(analyzed_adata):
    """enrichment_barplot produces a figure."""
    import matplotlib
    matplotlib.use("Agg")
    from scptr.plotting import enrichment_barplot

    results = [
        {"label": "ARE", "median_gamma_in_set": 0.5,
         "median_gamma_background": 0.3, "p_value": 0.01},
        {"label": "NMD", "median_gamma_in_set": 0.6,
         "median_gamma_background": 0.3, "p_value": 0.005},
    ]

    fig = enrichment_barplot(results)
    assert fig is not None


def test_halflife_scatter(analyzed_adata):
    """halflife_scatter produces a figure."""
    import matplotlib
    matplotlib.use("Agg")
    from scptr.plotting import halflife_scatter

    gene_names = analyzed_adata.var_names.tolist()
    hl_df = pd.DataFrame({
        "gene_symbol": gene_names[:20],
        "half_life_hours": np.random.exponential(5, size=20),
    })

    fig = halflife_scatter(analyzed_adata, hl_df)
    assert fig is not None
