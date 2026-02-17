"""Pooch-based download registry for scPTR datasets."""

from __future__ import annotations

from pathlib import Path

import pooch

_CACHE_DIR = Path.home() / ".cache" / "scptr"

# Base URL for scvelo's dataset hosting on GitHub
_SCVELO_BASE = "https://github.com/theislab/scvelo_notebooks/raw/master/"

REGISTRY = pooch.create(
    path=_CACHE_DIR,
    base_url="",
    registry={
        "pancreas.h5ad": None,
        "dentate_gyrus.h5ad": None,
    },
    urls={
        "pancreas.h5ad": _SCVELO_BASE + "data/Pancreas/endocrinogenesis_day15.h5ad",
        "dentate_gyrus.h5ad": _SCVELO_BASE + "data/DentateGyrus/10X43_1.h5ad",
    },
)


def fetch(name: str) -> str:
    """Fetch a dataset file, downloading if necessary.

    Returns the local file path.
    """
    return REGISTRY.fetch(name, progressbar=True)
