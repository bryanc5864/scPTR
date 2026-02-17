"""Bundled mRNA half-life reference datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"


def herzog2017_halflives() -> pd.DataFrame:
    """Load mRNA half-lives from Herzog et al. 2017 (SLAM-seq, mouse ESCs).

    Returns
    -------
    DataFrame with columns ``gene_symbol`` and ``half_life_hours``.
    """
    path = _DATA_DIR / "herzog2017_halflives.csv"
    return pd.read_csv(path)


def schofield2018_halflives() -> pd.DataFrame:
    """Load mRNA half-lives from Schofield et al. 2018 (TimeLapse-seq, K562).

    Returns
    -------
    DataFrame with columns ``gene_symbol`` and ``half_life_hours``.
    """
    path = _DATA_DIR / "schofield2018_halflives.csv"
    return pd.read_csv(path)
