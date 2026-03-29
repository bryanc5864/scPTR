"""Shared utilities for DeepPTR benchmark scripts.

All scripts in analyses/deep/ import from here for reproducibility.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import torch

torch.set_num_threads(4)

# Inline figure style (avoids name collision with parent _common.py)
def set_figure_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "figure.figsize": (6, 5), "axes.spines.top": False, "axes.spines.right": False,
    })

import scptr

# ── Paths ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "output" / "deep_benchmarks"
DATA_DIR = Path(scptr.benchmark.__file__).parent / "data"

DEEP_HP = dict(
    d_T=8, d_PT=8, d_hidden=48, n_enc_layers=2,
    batch_size=512, max_epochs=100, kl_warmup_epochs=20,
    patience=15, n_posterior_samples=15,
    device="cpu", seed=0,
)


def output_dir(script_name: str) -> Path:
    """Return output directory for a given script, e.g. '01_fair_comparison'."""
    d = OUTPUT_ROOT / script_name
    (d / "figures").mkdir(parents=True, exist_ok=True)
    (d / "results").mkdir(parents=True, exist_ok=True)
    return d


def save_fig(fig, name: str, out: Path, subdir: str = "figures"):
    if fig is None:
        return
    path = out / subdir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_json(data, name: str, out: Path):
    path = out / "results" / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")


# ── Data loading ───────────────────────────────────────────────────────────

def load_halflife_refs():
    return scptr.datasets.herzog2017_halflives(), scptr.datasets.schofield2018_halflives()


def select_top_genes(adata, n_top=300):
    """Select top genes by unspliced signal for DeepPTR."""
    from scipy.sparse import issparse

    u = adata.layers["unspliced"]
    if issparse(u):
        u = np.asarray(u.todense())
    u = np.asarray(u, dtype=np.float32)
    score = u.sum(axis=0) * (u > 0).mean(axis=0)
    top_idx = np.sort(np.argsort(score)[::-1][:n_top])
    adata_sub = adata[:, adata.var_names[top_idx]].copy()
    from scipy.sparse import issparse as _iss

    for key in ("spliced", "unspliced"):
        if key in adata_sub.layers and _iss(adata_sub.layers[key]):
            adata_sub.layers[key] = np.asarray(adata_sub.layers[key].todense())
    return adata_sub


def run_analytical(adata_loader):
    """Run full analytical scPTR pipeline, return adata."""
    adata = adata_loader()
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)
    return adata


def run_deep(adata_loader, n_top=300, verbose=True):
    """Run preprocessing + DeepPTR, return (adata_deep, model, history)."""
    adata = adata_loader()
    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    adata = select_top_genes(adata, n_top=n_top)
    torch.set_num_threads(4)
    model, history = scptr.deep.fit_deepptr(adata, verbose=verbose, **DEEP_HP)
    return adata, model, history


def run_both(adata_loader, n_top=300):
    """Return (adata_analytical, adata_deep, model, history)."""
    adata_an = run_analytical(adata_loader)
    adata_dp, model, history = run_deep(adata_loader, n_top=n_top)
    return adata_an, adata_dp, model, history


# ── Half-life matching ─────────────────────────────────────────────────────

def match_halflife(adata, hl_df, gene_col="gene_symbol", hl_col="half_life_hours"):
    """Match genes case-insensitively, return (gamma_vals, hl_vals, gene_names)."""
    gamma_med = np.median(adata.layers["gamma"], axis=0)
    hl_s = hl_df.set_index(gene_col)[hl_col]

    gamma_upper = {g.upper(): i for i, g in enumerate(adata.var_names)}
    hl_upper = {g.upper(): g for g in hl_s.index if isinstance(g, str)}
    shared = set(gamma_upper.keys()) & set(hl_upper.keys())

    idx = [gamma_upper[u] for u in shared]
    g = gamma_med[idx].astype(float)
    h = np.array([hl_s[hl_upper[u]] for u in shared], dtype=float)
    names = [adata.var_names[gamma_upper[u]] for u in shared]

    valid = np.isfinite(g) & np.isfinite(h) & (g > 0) & (h > 0)
    return g[valid], h[valid], [n for n, v in zip(names, valid) if v]


def halflife_spearman(adata, hl_df):
    """Quick Spearman r with half-life reference."""
    g, h, _ = match_halflife(adata, hl_df)
    if len(g) < 3:
        return np.nan, 0
    r, _ = stats.spearmanr(g, h)
    return float(r), len(g)


# ── Dataset registry ───────────────────────────────────────────────────────

DATASETS = [
    ("pancreas", scptr.datasets.pancreas, "clusters"),
    ("dentate_gyrus", scptr.datasets.dentate_gyrus, "clusters"),
]
