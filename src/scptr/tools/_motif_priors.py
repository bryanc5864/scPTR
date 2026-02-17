"""Motif-guided priors for post-transcriptional network inference."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


_DATA_DIR = Path(__file__).parent / "data"


def load_motif_priors(source: str | Path) -> pd.DataFrame:
    """Load RBP-target prior weights from a user-provided CSV.

    The CSV must have at least columns ``regulator``, ``target``, and ``weight``.

    Parameters
    ----------
    source
        Path to a CSV file with columns ``regulator``, ``target``, ``weight``.

    Returns
    -------
    DataFrame with columns ``['regulator', 'target', 'weight']``.
    """
    df = pd.read_csv(source)
    required_cols = {"regulator", "target", "weight"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Prior network CSV is missing required columns: {missing}. "
            f"Expected columns: {sorted(required_cols)}"
        )
    return df[["regulator", "target", "weight"]].copy()


def list_known_rbps(organism: str | None = None) -> list[str]:
    """Return curated list of known RNA-binding proteins.

    Parameters
    ----------
    organism
        Filter by organism: ``'human'``, ``'mouse'``, or ``None`` (all).

    Returns
    -------
    Sorted list of gene symbols.
    """
    csv_path = _DATA_DIR / "known_rbps.csv"
    df = pd.read_csv(csv_path)
    if organism is not None:
        df = df[df["organism"] == organism]
    return sorted(df["gene_symbol"].tolist())
