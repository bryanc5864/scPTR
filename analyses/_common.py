"""Shared configuration for analysis scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"


def setup_output_dirs(*subdirs: str) -> list[Path]:
    """Create output directories and return their paths."""
    dirs = []
    for sub in subdirs:
        d = OUTPUT_DIR / sub
        d.mkdir(parents=True, exist_ok=True)
        dirs.append(d)
    return dirs


def set_figure_style():
    """Set consistent figure style for all analyses."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (6, 5),
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig: plt.Figure, name: str, subdir: str = "figures") -> Path:
    """Save a figure to the output directory."""
    out_dir = OUTPUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    return path
