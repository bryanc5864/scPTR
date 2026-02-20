"""PT velocity embedding plot."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from scipy.sparse import issparse

from .._constants import PT_VELOCITY, GAMMA
from .._utils import get_layer, require_layers
from ._utils import setup_axes, save_or_show


def pt_velocity_embedding(
    adata: AnnData,
    basis: str = "X_gamma_umap",
    density: float = 1.0,
    arrow_size: float = 3.0,
    figsize: tuple[float, float] = (8, 6),
    save: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot PT velocity arrows on UMAP embedding.

    Projects high-dimensional velocity vectors onto 2D embedding
    using cosine similarity with displacement vectors to neighbors.

    Parameters
    ----------
    adata
        Annotated data matrix with ``pt_velocity`` layer and UMAP embedding.
    basis
        Key in ``adata.obsm`` for the 2D embedding.
    density
        Controls arrow density (fraction of cells to show).
    arrow_size
        Scaling factor for arrow size.
    figsize
        Figure size.
    save
        Path to save.
    show
        Whether to display.
    ax
        Pre-existing axes.
    """
    require_layers(adata, PT_VELOCITY)

    if basis not in adata.obsm:
        raise KeyError(f"Embedding {basis!r} not found.")

    velocity = get_layer(adata, PT_VELOCITY)
    gamma = get_layer(adata, GAMMA)
    coords = adata.obsm[basis]

    # Project velocity onto embedding using transition probability approach:
    # For each cell i, compare velocity direction with gamma displacement
    # to each neighbor. Neighbors whose gamma displacement aligns with the
    # velocity get higher weight for the embedding projection.
    n_obs = adata.n_obs
    v_emb = np.zeros((n_obs, 2), dtype=np.float64)

    # Use gamma-space graph if available
    if "gamma_connectivities" in adata.obsp:
        conn = adata.obsp["gamma_connectivities"]
    elif "connectivities" in adata.obsp:
        conn = adata.obsp["connectivities"]
    else:
        raise KeyError("No connectivities found.")

    from scipy.sparse import issparse
    if issparse(conn):
        conn = conn.tocsr()

    for i in range(n_obs):
        if issparse(conn):
            neighbors_i = conn[i].indices
            weights_i = conn[i].data
        else:
            neighbors_i = np.where(conn[i] > 0)[0]
            weights_i = conn[i, neighbors_i]

        if len(neighbors_i) == 0:
            continue

        # Velocity vector of cell i
        v_i = velocity[i]
        v_norm = np.linalg.norm(v_i)
        if v_norm < 1e-10:
            continue

        # Compare velocity direction with gamma displacement to neighbors
        total_w = 0.0
        for j_idx, j in enumerate(neighbors_i):
            # Gene-space displacement: where is neighbor j relative to cell i?
            dg = gamma[j] - gamma[i]
            dg_norm = np.linalg.norm(dg)
            if dg_norm < 1e-10:
                continue
            # Cosine similarity between velocity and gene displacement
            cos_sim = np.dot(v_i, dg) / (v_norm * dg_norm)
            # Only use neighbors in the velocity direction (positive cosine)
            w = weights_i[j_idx] * max(cos_sim, 0)
            # Accumulate embedding displacement
            de = coords[j] - coords[i]
            v_emb[i] += w * de
            total_w += w

        if total_w > 0:
            v_emb[i] /= total_w

    # Scale arrows: normalize then scale by a consistent factor
    norms = np.linalg.norm(v_emb, axis=1)
    cap = np.percentile(norms[norms > 0], 95) if (norms > 0).any() else 1.0
    # Clip extreme vectors
    scale_factor = np.minimum(norms / max(cap, 1e-10), 1.0)
    # Normalize direction, scale by capped magnitude
    safe_norms = np.clip(norms, 1e-10, None)
    v_emb = (v_emb / safe_norms[:, None]) * scale_factor[:, None]

    fig, ax = setup_axes(ax, figsize=figsize)

    # Subsample for density
    n_show = max(1, int(n_obs * min(density, 1.0)))
    idx = np.random.choice(n_obs, n_show, replace=False)

    # Color by velocity magnitude
    vel_mag = np.linalg.norm(velocity, axis=1)
    ax.scatter(
        coords[:, 0], coords[:, 1],
        s=5, alpha=0.4, c=vel_mag, cmap="coolwarm",
        rasterized=True, vmin=0, vmax=np.percentile(vel_mag, 95),
    )
    ax.quiver(
        coords[idx, 0], coords[idx, 1],
        v_emb[idx, 0], v_emb[idx, 1],
        scale=arrow_size, scale_units="inches",
        angles="xy", headwidth=4, headlength=5,
        alpha=0.6, color="black", linewidth=0.5,
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("PT Velocity")
    fig.tight_layout()

    save_or_show(fig, save, show)
    return fig if not show else None


def _project_velocity_to_2d(
    adata: AnnData,
    basis: str = "X_gamma_umap",
) -> np.ndarray:
    """Project high-dimensional velocity onto 2D embedding coordinates.

    Returns array of shape (n_obs, 2) with per-cell 2D velocity vectors.
    """
    velocity = get_layer(adata, PT_VELOCITY)
    gamma = get_layer(adata, GAMMA)
    coords = adata.obsm[basis]
    n_obs = adata.n_obs

    v_emb = np.zeros((n_obs, 2), dtype=np.float64)

    if "gamma_connectivities" in adata.obsp:
        conn = adata.obsp["gamma_connectivities"]
    elif "connectivities" in adata.obsp:
        conn = adata.obsp["connectivities"]
    else:
        raise KeyError("No connectivities found.")

    if issparse(conn):
        conn = conn.tocsr()

    for i in range(n_obs):
        if issparse(conn):
            neighbors_i = conn[i].indices
            weights_i = conn[i].data
        else:
            neighbors_i = np.where(conn[i] > 0)[0]
            weights_i = conn[i, neighbors_i]

        if len(neighbors_i) == 0:
            continue

        v_i = velocity[i]
        v_norm = np.linalg.norm(v_i)
        if v_norm < 1e-10:
            continue

        total_w = 0.0
        for j_idx, j in enumerate(neighbors_i):
            dg = gamma[j] - gamma[i]
            dg_norm = np.linalg.norm(dg)
            if dg_norm < 1e-10:
                continue
            cos_sim = np.dot(v_i, dg) / (v_norm * dg_norm)
            w = weights_i[j_idx] * max(cos_sim, 0)
            de = coords[j] - coords[i]
            v_emb[i] += w * de
            total_w += w

        if total_w > 0:
            v_emb[i] /= total_w

    return v_emb


def pt_velocity_stream(
    adata: AnnData,
    basis: str = "X_gamma_umap",
    grid_size: int = 50,
    smooth_sigma: float = 1.5,
    density: float = 1.0,
    color_key: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    save: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot PT velocity as streamlines on UMAP embedding.

    Parameters
    ----------
    adata
        Annotated data matrix with ``pt_velocity`` layer and UMAP embedding.
    basis
        Key in ``adata.obsm`` for the 2D embedding.
    grid_size
        Number of grid points per axis for the velocity field.
    smooth_sigma
        Gaussian smoothing sigma for the gridded velocity field.
    density
        Streamplot density parameter.
    color_key
        Column in ``adata.obs`` used to color the background scatter.
        If None, cells are colored by velocity magnitude.
    figsize
        Figure size.
    save
        Path to save.
    show
        Whether to display.
    ax
        Pre-existing axes.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.stats import binned_statistic_2d

    require_layers(adata, PT_VELOCITY)

    if basis not in adata.obsm:
        raise KeyError(f"Embedding {basis!r} not found.")

    coords = adata.obsm[basis]
    v_emb = _project_velocity_to_2d(adata, basis)

    # Build gridded velocity field
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05

    x_edges = np.linspace(x_min - pad_x, x_max + pad_x, grid_size + 1)
    y_edges = np.linspace(y_min - pad_y, y_max + pad_y, grid_size + 1)

    # Bin velocities into grid
    U, _, _, _ = binned_statistic_2d(
        coords[:, 0], coords[:, 1], v_emb[:, 0],
        statistic="mean", bins=[x_edges, y_edges],
    )
    V, _, _, _ = binned_statistic_2d(
        coords[:, 0], coords[:, 1], v_emb[:, 1],
        statistic="mean", bins=[x_edges, y_edges],
    )

    # Replace NaN with 0
    U = np.nan_to_num(U, nan=0.0)
    V = np.nan_to_num(V, nan=0.0)

    # Gaussian smooth
    U = gaussian_filter(U, sigma=smooth_sigma)
    V = gaussian_filter(V, sigma=smooth_sigma)

    # Grid centers
    gx = 0.5 * (x_edges[:-1] + x_edges[1:])
    gy = 0.5 * (y_edges[:-1] + y_edges[1:])
    GX, GY = np.meshgrid(gx, gy, indexing="ij")

    # Speed for coloring
    speed = np.sqrt(U**2 + V**2)

    fig, ax = setup_axes(ax, figsize=figsize)

    # Background scatter
    if color_key is not None and color_key in adata.obs.columns:
        cats = adata.obs[color_key]
        if hasattr(cats, "cat"):
            for ci, cat in enumerate(cats.cat.categories):
                mask = (cats == cat).values
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    s=3, alpha=0.2, label=cat,
                    c=[plt.cm.tab20(ci / 20)],
                    rasterized=True,
                )
            ax.legend(fontsize=6, markerscale=3, loc="best")
        else:
            ax.scatter(
                coords[:, 0], coords[:, 1],
                s=3, alpha=0.2, c="lightgray", rasterized=True,
            )
    else:
        vel_mag = np.linalg.norm(get_layer(adata, PT_VELOCITY), axis=1)
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=3, alpha=0.2, c=vel_mag, cmap="YlOrRd",
            vmin=0, vmax=np.percentile(vel_mag, 95),
            rasterized=True,
        )

    # Streamlines — transpose U, V so axes align with (x, y)
    ax.streamplot(
        gx, gy, U.T, V.T,
        color=speed.T, cmap="coolwarm",
        density=density, linewidth=0.8, arrowsize=1.2,
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("PT Velocity Streamlines")
    fig.tight_layout()

    save_or_show(fig, save, show)
    return fig if not show else None
