"""Tests for the datasets module."""

import pytest
import pandas as pd


def test_herzog2017_halflives():
    """Bundled Herzog 2017 half-lives load correctly."""
    from scptr.datasets import herzog2017_halflives

    df = herzog2017_halflives()
    assert isinstance(df, pd.DataFrame)
    assert "gene_symbol" in df.columns
    assert "half_life_hours" in df.columns
    assert len(df) > 50
    assert df["half_life_hours"].min() > 0


def test_schofield2018_halflives():
    """Bundled Schofield 2018 half-lives load correctly."""
    from scptr.datasets import schofield2018_halflives

    df = schofield2018_halflives()
    assert isinstance(df, pd.DataFrame)
    assert "gene_symbol" in df.columns
    assert "half_life_hours" in df.columns
    assert len(df) > 50
    assert df["half_life_hours"].min() > 0


@pytest.mark.slow
def test_pancreas_download():
    """Pancreas dataset downloads and loads."""
    from scptr.datasets import pancreas

    adata = pancreas()
    assert adata.n_obs > 0
    assert adata.n_vars > 0


@pytest.mark.slow
def test_dentate_gyrus_download():
    """Dentate gyrus dataset downloads and loads."""
    from scptr.datasets import dentate_gyrus

    adata = dentate_gyrus()
    assert adata.n_obs > 0
    assert adata.n_vars > 0


