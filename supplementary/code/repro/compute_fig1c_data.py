"""Compute real data for fig1C: PT velocity field + RBP network hub info."""
import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import scanpy as sc
import json
from pathlib import Path

import scptr

OUT = Path("real_figure_data")
OUT.mkdir(exist_ok=True)

# ── Load and run pipeline ────────────────────────────────────────────────────
print("Loading pancreas dataset...")
adata = scptr.datasets.pancreas()
scptr.pp.filter_genes(adata)
scptr.pp.normalize_layers(adata)
scptr.pp.neighbors(adata)
scptr.pp.smooth_layers(adata)
scptr.tl.estimate_beta(adata)
scptr.tl.estimate_gamma(adata)

# ── PT states + velocity ─────────────────────────────────────────────────────
print("Computing PT states and velocity...")
scptr.tl.pt_states(adata)
scptr.tl.pt_velocity(adata)

# Get gamma UMAP coordinates and velocity projections
gamma_umap = adata.obsm["X_gamma_umap"]
velocity = adata.layers["pt_velocity"]

# Project velocity into gamma UMAP space using the gamma PCA loadings
gamma_pca = adata.obsm["X_gamma_pca"]

# Compute velocity in PCA space first, then project to 2D via neighbors
# Simpler approach: for each cell, compute the average UMAP displacement
# weighted by velocity magnitude toward neighbors
from scipy.sparse import issparse
dist_mat = adata.obsp["gamma_connectivities"]
if issparse(dist_mat):
    dist_mat = dist_mat.tocsr()

n_cells = adata.n_obs
vel_umap = np.zeros((n_cells, 2))

for i in range(n_cells):
    neighbors_i = dist_mat[i].nonzero()[1]
    if len(neighbors_i) == 0:
        continue
    # Direction in UMAP space to each neighbor
    dumap = gamma_umap[neighbors_i] - gamma_umap[i]
    # Velocity similarity: dot product of velocity with gamma difference
    dgamma = adata.layers["gamma"][neighbors_i] - adata.layers["gamma"][i]
    vel_i = velocity[i]
    # Cosine similarity between cell's velocity and direction to each neighbor
    cos_sim = np.array([
        np.dot(vel_i, dgamma[j]) / (np.linalg.norm(vel_i) * np.linalg.norm(dgamma[j]) + 1e-10)
        for j in range(len(neighbors_i))
    ])
    # Weight UMAP displacements by positive cosine similarity
    weights = np.clip(cos_sim, 0, None)
    if weights.sum() > 0:
        vel_umap[i] = (weights[:, None] * dumap).sum(axis=0) / weights.sum()

# Normalize arrow lengths for visualization
norms = np.linalg.norm(vel_umap, axis=1)
scale = np.percentile(norms[norms > 0], 95) if (norms > 0).any() else 1.0
vel_umap_norm = vel_umap / (scale + 1e-10)

# Cell type labels
cell_types = adata.obs["clusters"].values.astype(str)

np.savez(OUT / "pt_velocity_umap.npz",
         umap=gamma_umap,
         velocity=vel_umap_norm,
         cell_types=cell_types)
print("Saved pt_velocity_umap.npz")

# ── RBP network top hubs ─────────────────────────────────────────────────────
# Use documented results from RESULTS.md for the top pancreas RBP hubs
network_info = {
    "dataset": "pancreas",
    "total_edges": 2198,
    "n_rbps": 54,
    "top_hubs": [
        {"rbp": "Hnrnpa1", "destabilizing": 102, "stabilizing": 64},
        {"rbp": "Ybx1", "destabilizing": 138, "stabilizing": 20},
        {"rbp": "Srsf3", "destabilizing": 104, "stabilizing": 40},
        {"rbp": "Rbfox3", "destabilizing": 37, "stabilizing": 33},
        {"rbp": "Hnrnpd", "destabilizing": 45, "stabilizing": 3},
        {"rbp": "Elavl1", "destabilizing": 29, "stabilizing": 14},
    ]
}
with open(OUT / "rbp_network.json", "w") as f:
    json.dump(network_info, f, indent=2)
print("Saved rbp_network.json")
print("Done!")
