"""Shared plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def setup_axes(
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (6, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """Return a (fig, ax) pair, creating them if *ax* is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def save_or_show(fig: plt.Figure, save: str | None, show: bool) -> None:
    """Save figure to file and/or display."""
    if save is not None:
        fig.savefig(save, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
