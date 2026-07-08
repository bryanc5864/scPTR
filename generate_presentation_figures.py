"""Generate high-quality presentation figures for scPTR."""
from __future__ import annotations

import json
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path

import sys
sys.path.insert(0, "/home/bcheng/scPTR/src")
import scptr

# ── Style ─────────────────────────────────────────────────────────────────────
BLUE   = "#2b5797"
RED    = "#c0392b"
GREEN  = "#2d7d4b"
GRAY   = "#888888"
LGRAY  = "#e8e8e8"
BG     = "white"

CT_COLORS = {
    "Ductal":         "#4e79a7",
    "Ngn3 low EP":    "#f28e2b",
    "Ngn3 high EP":   "#e15759",
    "Pre-endocrine":  "#76b7b2",
    "Beta":           "#59a14f",
    "Alpha":          "#edc948",
    "Delta":          "#b07aa1",
    "Epsilon":        "#ff9da7",
}

PT_CMAP = plt.colormaps["tab20"]

OUT = Path("/home/bcheng/scPTR/figures/presentation")
OUT.mkdir(parents=True, exist_ok=True)

def style_ax(ax, title="", xlabel="", ylabel="", title_size=13):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#444444", labelsize=10)
    if title:  ax.set_title(title, fontsize=title_size, fontweight="bold", color="#1a1a1a", pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=11, color="#444444")
    if ylabel: ax.set_ylabel(ylabel, fontsize=11, color="#444444")

def panel_label(ax, letter, x=-0.12, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=16, fontweight="bold", color="#1a1a1a", va="top")

def save(fig, name, dpi=200):
    path = OUT / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {path.name}")

# ── Load & run pipeline ───────────────────────────────────────────────────────
print("Loading pancreas dataset...")
adata = sc.read_h5ad("/home/bcheng/scPTR/data/Pancreas/endocrinogenesis_day15.h5ad")
print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")
print(f"  cell types: {sorted(adata.obs['clusters'].unique())}")

print("Preprocessing...")
scptr.pp.filter_genes(adata, min_unspliced_counts=10, min_unspliced_cells=5)
scptr.pp.normalize_layers(adata)
scptr.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
scptr.pp.smooth_layers(adata)

print("Estimating rates...")
scptr.tl.estimate_beta(adata, quantile=0.95)
scptr.tl.estimate_gamma(adata, clip_quantile=0.99, min_spliced=0.01)
scptr.tl.variance_decomposition(adata)

print("Discovering PT states...")
scptr.tl.pt_states(adata, resolution=0.5, n_pcs=20, n_neighbors=15, random_state=42)

print("Computing PT velocity...")
scptr.tl.pt_velocity(adata, use_graph="gamma")

print("Pipeline complete.")
print(f"  PT states: {adata.obs['pt_state'].nunique()}")

# ── Load reference data ───────────────────────────────────────────────────────
hl_mouse = pd.read_csv("/home/bcheng/scPTR/src/scptr/datasets/data/herzog2017_halflives.csv")
hl_human = pd.read_csv("/home/bcheng/scPTR/src/scptr/datasets/data/schofield2018_halflives.csv")

gamma_mat = adata.layers["gamma"]
median_gamma = pd.Series(np.median(gamma_mat, axis=0), index=adata.var_names)
beta = adata.var["beta"]
tf   = adata.var["tf_score"]
ptf  = adata.var["ptf_score"]
n_states = adata.obs["pt_state"].nunique()
state_colors = [PT_CMAP(i / max(n_states - 1, 1)) for i in range(n_states)]

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Method overview: kinetic model + halflife validation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] Method overview + halflife validation")

fig = plt.figure(figsize=(16, 6), facecolor=BG)
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

# ── Panel A: kinetic schematic ─────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.set_xlim(0, 10); ax0.set_ylim(0, 10); ax0.axis("off")
panel_label(ax0, "A", x=-0.04)

# Draw boxes
def box(ax, x, y, w, h, color, label, sub=""):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.15", linewidth=1.5,
        edgecolor=color, facecolor=color + "18")
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.25 if sub else 0), label,
            ha="center", va="center", fontsize=11, fontweight="bold", color=color)
    if sub:
        ax.text(x + w/2, y + h/2 - 0.35, sub,
                ha="center", va="center", fontsize=8.5, color="#555555")

def arrow(ax, x1, y1, x2, y2, label="", color="#555555"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.15, my, label, fontsize=9, color=color, va="center")

box(ax0, 0.5, 7.0, 3.5, 1.4, BLUE,   "Nascent RNA", "unspliced (u)")
box(ax0, 0.5, 4.0, 3.5, 1.4, GREEN,  "Mature mRNA", "spliced (s)")
box(ax0, 0.5, 1.0, 3.5, 1.4, RED,    "Degraded",    "")

arrow(ax0, 2.25, 7.0, 2.25, 5.4, "β (splicing)", color=BLUE)
arrow(ax0, 2.25, 4.0, 2.25, 2.4, "γ (degradation)", color=RED)

ax0.text(5.2, 5.5,  "Kinetic\nsteady state:",
         fontsize=10, color="#1a1a1a", va="center", fontweight="bold")
ax0.text(5.2, 4.2,
         r"$\frac{du}{dt} = \alpha - \beta u = 0$",
         fontsize=10, color=BLUE, va="center")
ax0.text(5.2, 3.2,
         r"$\frac{ds}{dt} = \beta u - \gamma s = 0$",
         fontsize=10, color=GREEN, va="center")
ax0.text(5.2, 2.0,
         r"$\Rightarrow\; \gamma_{ig} = \beta_g \cdot \frac{u_{ig}}{s_{ig}}$",
         fontsize=12, color=RED, va="center", fontweight="bold")

ax0.set_title("A  Kinetic Model", fontsize=13, fontweight="bold",
              color="#1a1a1a", pad=8, loc="left")

# ── Panel B: halflife scatter (log-log, mouse) ───────────────────────────────
ax1 = fig.add_subplot(gs[1])
panel_label(ax1, "B")

hl_m = hl_mouse.dropna(subset=["gene_symbol", "half_life_hours"])
shared = median_gamma.index.intersection(hl_m.set_index("gene_symbol").index)
mg_v = np.log10(median_gamma[shared].values + 1e-6)
hl_v = np.log10(hl_m.set_index("gene_symbol").loc[shared, "half_life_hours"].values + 1e-6)

mask = np.isfinite(mg_v) & np.isfinite(hl_v) & (median_gamma[shared].values > 0)
from scipy.stats import spearmanr
r, p = spearmanr(hl_v[mask], mg_v[mask])

h = ax1.hexbin(hl_v[mask], mg_v[mask], gridsize=40, cmap="Blues",
               mincnt=1, linewidths=0.2)
m, b_fit = np.polyfit(hl_v[mask], mg_v[mask], 1)
xl = np.array([hl_v[mask].min(), hl_v[mask].max()])
ax1.plot(xl, m*xl + b_fit, color=RED, lw=2, ls="--", zorder=5)
ax1.text(0.97, 0.95, f"ρ = {r:.2f}", transform=ax1.transAxes,
         ha="right", va="top", fontsize=11, fontweight="bold", color=RED)
ax1.text(0.97, 0.88, f"n = {mask.sum():,} genes", transform=ax1.transAxes,
         ha="right", va="top", fontsize=9, color=GRAY)

style_ax(ax1, title="B  Halflife Validation (Mouse)",
         xlabel="log₁₀ Published half-life (h)", ylabel="log₁₀ Median γ")
plt.colorbar(h, ax=ax1, label="Gene count", shrink=0.7)

# ── Panel C: halflife scatter (human) ────────────────────────────────────────
ax2 = fig.add_subplot(gs[2])
panel_label(ax2, "C")

hl_hu = hl_human.dropna()
hl_hcol = hl_hu.columns[1]  # half-life column
gene_col = hl_hu.columns[0]
hl_hu2 = hl_hu.set_index(gene_col)
shared_h = median_gamma.index.intersection(hl_hu2.index)
mg_h = np.log10(median_gamma[shared_h].values + 1e-6)
hl_h = np.log10(hl_hu2.loc[shared_h, hl_hcol].values.astype(float) + 1e-6)

mask_h = np.isfinite(mg_h) & np.isfinite(hl_h) & (median_gamma[shared_h].values > 0)
r_h, _ = spearmanr(hl_h[mask_h], mg_h[mask_h])

h2 = ax2.hexbin(hl_h[mask_h], mg_h[mask_h], gridsize=40, cmap="Purples",
                mincnt=1, linewidths=0.2)
m2, b2 = np.polyfit(hl_h[mask_h], mg_h[mask_h], 1)
xl2 = np.array([hl_h[mask_h].min(), hl_h[mask_h].max()])
ax2.plot(xl2, m2*xl2 + b2, color=RED, lw=2, ls="--", zorder=5)
ax2.text(0.97, 0.95, f"ρ = {r_h:.2f}", transform=ax2.transAxes,
         ha="right", va="top", fontsize=11, fontweight="bold", color=RED)
ax2.text(0.97, 0.88, f"n = {mask_h.sum():,} genes", transform=ax2.transAxes,
         ha="right", va="top", fontsize=9, color=GRAY)
style_ax(ax2, title="C  Halflife Validation (Human)",
         xlabel="log₁₀ Published half-life (h)", ylabel="log₁₀ Median γ")
plt.colorbar(h2, ax=ax2, label="Gene count", shrink=0.7)

fig.suptitle("scPTR: Kinetic Model and Halflife Validation", fontsize=14,
             fontweight="bold", color="#1a1a1a", y=1.01)
save(fig, "fig1_method_halflife.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Expression vs PT state organization (side-by-side UMAP)
# ═══════════════════════════════════════════════════════════════════════════════
print("[Fig 2] Expression vs PT UMAP")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)

def scatter_umap(ax, coords, labels, colors_dict, title, letter, label_points=True):
    panel_label(ax, letter)
    unique = sorted(set(labels), key=lambda x: (int(x) if str(x).isdigit() else 0, str(x)))
    for lbl in unique:
        mask = np.array(labels) == lbl
        color = colors_dict.get(str(lbl), "#aaaaaa")
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[color], s=8,
                   alpha=0.7, linewidths=0, label=str(lbl))
    if label_points:
        for lbl in unique:
            mask = np.array(labels) == lbl
            cx = coords[mask, 0].mean()
            cy = coords[mask, 1].mean()
            ax.text(cx, cy, str(lbl), fontsize=8.5, fontweight="bold",
                    color="white", ha="center", va="center",
                    path_effects=[pe.Stroke(linewidth=2.5, foreground="#333333"), pe.Normal()])
    ax.axis("off")
    style_ax(ax, title=title, title_size=12)

# Expression UMAP (has X_umap already)
expr_coords = adata.obsm["X_umap"]
expr_labels = adata.obs["clusters"].values
scatter_umap(axes[0], expr_coords, expr_labels, CT_COLORS,
             "A  Expression UMAP (cell type)", "A", label_points=True)
handles = [mpatches.Patch(color=CT_COLORS[ct], label=ct) for ct in CT_COLORS]
axes[0].legend(handles=handles, loc="lower left", fontsize=8,
               frameon=True, framealpha=0.9, edgecolor=LGRAY,
               bbox_to_anchor=(0, -0.02))

# γ-space UMAP
gamma_coords = adata.obsm["X_gamma_umap"]
pt_labels = adata.obs["pt_state"].astype(str).values
pt_color_map = {str(i): matplotlib.colors.to_hex(PT_CMAP(i / max(n_states - 1, 1)))
                for i in range(n_states)}
scatter_umap(axes[1], gamma_coords, pt_labels, pt_color_map,
             f"B  γ-space UMAP ({n_states} PT states)", "B", label_points=True)

fig.suptitle("scPTR Reveals Hidden Post-Transcriptional Organization",
             fontsize=14, fontweight="bold", color="#1a1a1a", y=1.01)
save(fig, "fig2_expression_vs_pt_umap.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Expression-invisible states (PT states mapped onto expression UMAP)
# ═══════════════════════════════════════════════════════════════════════════════
print("[Fig 3] Expression-invisible states")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=BG)

# Panel A: expression UMAP colored by cell type
panel_label(axes[0], "A")
for ct, color in CT_COLORS.items():
    mask = adata.obs["clusters"].values == ct
    axes[0].scatter(expr_coords[mask, 0], expr_coords[mask, 1],
                    c=[color], s=7, alpha=0.7, linewidths=0, label=ct)
axes[0].axis("off")
style_ax(axes[0], title="A  Cell type (expression)", title_size=12)
axes[0].legend(fontsize=7.5, loc="lower left", frameon=True,
               framealpha=0.9, edgecolor=LGRAY, markerscale=1.5)

# Panel B: expression UMAP colored by PT state
panel_label(axes[1], "B")
for i in range(n_states):
    mask = adata.obs["pt_state"].astype(str).values == str(i)
    c = matplotlib.colors.to_hex(PT_CMAP(i / max(n_states - 1, 1)))
    axes[1].scatter(expr_coords[mask, 0], expr_coords[mask, 1],
                    c=[c], s=7, alpha=0.7, linewidths=0, label=f"PT {i}")
axes[1].axis("off")
style_ax(axes[1], title="B  PT state on expression UMAP", title_size=12)

# Panel C: state-cell type heatmap (cross-tabulation)
ax = axes[2]
panel_label(ax, "C")
ct_order = ["Ductal", "Ngn3 low EP", "Ngn3 high EP", "Pre-endocrine", "Beta", "Alpha", "Delta", "Epsilon"]
cross = pd.crosstab(adata.obs["pt_state"].astype(int), adata.obs["clusters"])
cross = cross.reindex(columns=[c for c in ct_order if c in cross.columns])
cross_norm = cross.div(cross.sum(axis=1), axis=0)  # normalize per PT state

im = ax.imshow(cross_norm.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(cross_norm.columns)))
ax.set_xticklabels(cross_norm.columns, rotation=40, ha="right", fontsize=9)
ax.set_yticks(range(len(cross_norm.index)))
ax.set_yticklabels([f"PT {i}" for i in cross_norm.index], fontsize=9)
ax.set_xlabel("Cell type", fontsize=11)
ax.set_ylabel("PT state", fontsize=11)
plt.colorbar(im, ax=ax, label="Fraction of PT state", shrink=0.8)
ax.spines[:].set_visible(False)
style_ax(ax, title="C  PT State × Cell Type Composition", title_size=12)

fig.suptitle("Post-Transcriptional States Cut Across Expression Clusters",
             fontsize=14, fontweight="bold", color="#1a1a1a", y=1.02)
save(fig, "fig3_invisible_states.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Variance decomposition
# ═══════════════════════════════════════════════════════════════════════════════
print("[Fig 4] Variance decomposition")

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)

# Panel A: TF score distribution
ax = axes[0]
panel_label(ax, "A")
tf_pos = tf[tf > 0].values
ax.hist(tf_pos, bins=60, color=BLUE, alpha=0.75, linewidth=0, edgecolor="none")
ax.axvline(0.5, color=RED, lw=1.8, ls="--", label="TF = PTF")
ax.axvline(np.median(tf_pos), color=GREEN, lw=1.8, ls="-",
           label=f"Median = {np.median(tf_pos):.3f}")
n_ptf = (tf_pos < 0.5).sum()
ax.text(0.05, 0.92, f"{n_ptf:,} genes\nPTF-dominant", transform=ax.transAxes,
        fontsize=10, color=RED, va="top", fontweight="bold")
style_ax(ax, title="A  Transcriptional Fraction Distribution",
         xlabel="Transcriptional fraction (TF score)", ylabel="Gene count")
ax.legend(fontsize=9, frameon=False)

# Panel B: TF vs PTF scatter (per gene)
ax = axes[1]
panel_label(ax, "B")
tf_v  = tf.values
ptf_v = ptf.values
valid = np.isfinite(tf_v) & np.isfinite(ptf_v)
h = ax.hexbin(tf_v[valid], ptf_v[valid], gridsize=45, cmap="Blues",
              mincnt=1, linewidths=0.15, bins="log")
ax.plot([0, 1], [1, 0], color=RED, lw=1.5, ls="--", alpha=0.7, label="TF + PTF = 1")
# Top PTF genes
top_ptf_genes = ptf[valid].nlargest(5)
for g in top_ptf_genes.index:
    ax.annotate(g, xy=(tf[g], ptf[g]), xytext=(tf[g]+0.04, ptf[g]-0.06),
                fontsize=7.5, color="#333333", arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8))
plt.colorbar(h, ax=ax, label="log(gene count)", shrink=0.8)
style_ax(ax, title="B  Per-Gene Variance Decomposition",
         xlabel="TF score (transcriptional)", ylabel="PTF score (post-transcriptional)")
ax.legend(fontsize=8, frameon=False)

# Panel C: top 20 PTF genes bar chart
ax = axes[2]
panel_label(ax, "C")
top20 = ptf.sort_values(ascending=False).head(20)
colors_bar = [BLUE if v > 0.7 else "#7ba7d4" for v in top20.values]
ax.barh(range(len(top20)), top20.values, color=colors_bar, linewidth=0)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20.index, fontsize=9)
ax.invert_yaxis()
ax.axvline(0.5, color=RED, lw=1.2, ls="--", alpha=0.7)
style_ax(ax, title="C  Top PTF-Regulated Genes",
         xlabel="PTF score", ylabel="")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

fig.suptitle("Most Genes Are Post-Transcriptionally Regulated",
             fontsize=14, fontweight="bold", color="#1a1a1a", y=1.02)
save(fig, "fig4_variance_decomposition.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — PT velocity
# ═══════════════════════════════════════════════════════════════════════════════
print("[Fig 5] PT velocity")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=BG)

# Panel A: γ-space UMAP with velocity arrows, colored by PT state
ax = axes[0]
panel_label(ax, "A")
for i in range(n_states):
    mask = adata.obs["pt_state"].astype(str).values == str(i)
    c = matplotlib.colors.to_hex(PT_CMAP(i / max(n_states - 1, 1)))
    ax.scatter(gamma_coords[mask, 0], gamma_coords[mask, 1],
               c=[c], s=10, alpha=0.6, linewidths=0)
try:
    scptr.pl.pt_velocity_embedding(adata, ax=ax, arrow_size=2.5, show=False)
except Exception:
    pass
ax.axis("off")
style_ax(ax, title="A  PT Velocity (γ-space)", title_size=12)

# Panel B: expression UMAP with PT velocity projected
ax = axes[1]
panel_label(ax, "B")
for ct, color in CT_COLORS.items():
    mask = adata.obs["clusters"].values == ct
    ax.scatter(expr_coords[mask, 0], expr_coords[mask, 1],
               c=[color], s=10, alpha=0.6, linewidths=0, label=ct)
# Project velocity onto expression UMAP using gamma-space vectors
pv = adata.layers["pt_velocity"]
pv_mag = np.linalg.norm(pv, axis=1)
top_q = np.percentile(pv_mag[pv_mag > 0], 80)
show_mask = pv_mag > top_q
if show_mask.sum() > 50:
    # Project pt_velocity (cell × gene) onto expression UMAP via PCA on gamma layer
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    gamma_mat = adata.layers["gamma"]
    pca = PCA(n_components=2, random_state=0)
    pca.fit(gamma_mat)
    vel_2d = pv[show_mask] @ pca.components_.T
    vel_2d /= (np.linalg.norm(vel_2d, axis=1, keepdims=True) + 1e-9)
    vel_2d *= 0.5
    pts = gamma_coords[show_mask]
    # Map gamma coords to expression coords via nearest neighbors
    nn = NearestNeighbors(n_neighbors=1).fit(gamma_coords)
    _, idx = nn.kneighbors(pts)
    expr_pts = expr_coords[idx[:, 0]]
    ax.quiver(expr_pts[:, 0], expr_pts[:, 1],
              vel_2d[:, 0], vel_2d[:, 1],
              color="#333333", alpha=0.5, width=0.003,
              headwidth=4, headlength=5, scale=30)
ax.axis("off")
style_ax(ax, title="B  PT Velocity on Expression UMAP", title_size=12)
ax.legend(fontsize=7.5, loc="lower left", frameon=True,
          framealpha=0.9, edgecolor=LGRAY, markerscale=1.5)

fig.suptitle("Post-Transcriptional Velocity Captures Degradation Dynamics",
             fontsize=14, fontweight="bold", color="#1a1a1a", y=1.01)
save(fig, "fig5_pt_velocity.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Beta distribution + top genes
# ═══════════════════════════════════════════════════════════════════════════════
print("[Fig 6] Beta distribution + phase portraits")

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)

# Panel A: beta distribution
ax = axes[0]
panel_label(ax, "A")
beta_pos = beta[beta > 0].values
ax.hist(beta_pos, bins=60, color=GREEN, alpha=0.75, linewidth=0)
ax.axvline(np.median(beta_pos), color=RED, lw=2,
           label=f"Median β = {np.median(beta_pos):.3f}")
ax.text(0.97, 0.92, f"{len(beta_pos):,} genes\nestimated",
        transform=ax.transAxes, ha="right", va="top", fontsize=10, color=GRAY)
style_ax(ax, title="A  Splicing Rate (β) Distribution",
         xlabel="β (splicing rate)", ylabel="Gene count")
ax.legend(fontsize=9, frameon=False)

# Panel B: phase portrait of highest-β gene
ax = axes[1]
panel_label(ax, "B")
top_gene = beta.idxmax()
Ms = adata.layers["Ms"][:, adata.var_names.get_loc(top_gene)]
Mu = adata.layers["Mu"][:, adata.var_names.get_loc(top_gene)]
g_vals = adata.layers["gamma"][:, adata.var_names.get_loc(top_gene)]
sc_plot = ax.scatter(Ms, Mu, c=g_vals, cmap="viridis", s=8, alpha=0.6,
                     linewidths=0, vmin=0, vmax=np.percentile(g_vals, 97))
bv = beta[top_gene]
x_line = np.linspace(0, np.percentile(Ms, 98), 100)
ax.plot(x_line, bv * x_line, color=RED, lw=2, label=f"β = {bv:.3f}")
plt.colorbar(sc_plot, ax=ax, label="γ (degradation rate)", shrink=0.8)
style_ax(ax, title=f"B  Phase Portrait: {top_gene}",
         xlabel="Ms (smoothed spliced)", ylabel="Mu (smoothed unspliced)")
ax.legend(fontsize=9, frameon=False)

# Panel C: γ distribution per cell type
ax = axes[2]
panel_label(ax, "C")
gamma_flat = np.median(gamma_mat, axis=0)
ct_gamma = {}
for ct in sorted(CT_COLORS.keys()):
    mask = adata.obs["clusters"].values == ct
    ct_gamma[ct] = np.median(gamma_mat[mask], axis=1)

positions = range(len(CT_COLORS))
ct_list = sorted(CT_COLORS.keys())
vp = ax.violinplot([ct_gamma[ct] for ct in ct_list], positions=list(positions),
                    showmedians=True, showextrema=False)
for i, (body, ct) in enumerate(zip(vp["bodies"], ct_list)):
    body.set_facecolor(CT_COLORS[ct])
    body.set_alpha(0.7)
vp["cmedians"].set_color("#333333")
vp["cmedians"].set_linewidth(2)
ax.set_xticks(list(positions))
ax.set_xticklabels(ct_list, rotation=35, ha="right", fontsize=8.5)
style_ax(ax, title="C  Per-Cell γ Distribution by Cell Type",
         xlabel="Cell type", ylabel="Median γ per cell")
ax.spines["left"].set_color("#cccccc")

fig.suptitle("scPTR Splicing and Degradation Rate Estimation",
             fontsize=14, fontweight="bold", color="#1a1a1a", y=1.02)
save(fig, "fig6_rates.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Subsampling robustness
# ═══════════════════════════════════════════════════════════════════════════════
print("[Fig 7] Robustness")

robust_path = Path("/home/bcheng/scPTR/output/results/aim1/subsampling_robustness.csv")
if robust_path.exists():
    rob = pd.read_csv(robust_path)
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
    panel_label(ax, "A", x=-0.08)

    if "fraction" in rob.columns and "correlation" in rob.columns:
        ax.plot(rob["fraction"] * 100, rob["correlation"],
                color=BLUE, lw=2.5, marker="o", ms=7, zorder=5)
        ax.fill_between(rob["fraction"] * 100,
                        rob.get("ci_low", rob["correlation"]),
                        rob.get("ci_high", rob["correlation"]),
                        alpha=0.15, color=BLUE)
        ax.axhline(0.97, color=RED, lw=1.5, ls="--", label="r = 0.97")
        ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("Subsampling fraction (%)", fontsize=11)
    ax.set_ylabel("Gamma correlation (r)", fontsize=11)
    style_ax(ax, title="A  Subsampling Robustness", title_size=13)
    ax.legend(fontsize=10, frameon=False)
    fig.suptitle("scPTR Is Robust to Subsampling",
                 fontsize=14, fontweight="bold", color="#1a1a1a")
    save(fig, "fig7_robustness.png")
else:
    print("  (skipped — no robustness CSV found)")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Summary / key stats slide
# ═══════════════════════════════════════════════════════════════════════════════
print("[Fig 8] Summary stats")

fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
ax.axis("off")

stats = [
    ("ρ = −0.81",  "sci-fate halflife\ncorrelation",      BLUE),
    ("ρ = −0.40",  "10x developmental\nhalflives",        BLUE),
    (f"{n_states} PT states", "discovered\nin pancreas",   GREEN),
    ("54–78%",     "transition genes\nwith γ precedence", RED),
    ("p = 4.7×10⁻⁶⁵", "miRNA target\nenrichment",       "#9b59b6"),
    ("r > 0.97",   "at 20%\nsubsampling",                 "#16a085"),
]
xs = np.linspace(0.07, 0.93, len(stats))
for x, (val, lbl, color) in zip(xs, stats):
    ax.text(x, 0.68, val, ha="center", va="center", fontsize=19,
            fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(x, 0.40, lbl, ha="center", va="center", fontsize=10,
            color="#444444", transform=ax.transAxes, linespacing=1.5)
    ax.plot([x - 0.07, x + 0.07], [0.56, 0.56],
            color=color, lw=2.5, transform=ax.transAxes, alpha=0.3)

ax.text(0.5, 0.96, "scPTR: Key Results",
        ha="center", va="top", fontsize=16, fontweight="bold",
        color="#1a1a1a", transform=ax.transAxes)
ax.text(0.5, 0.10,
        "Single-Cell Post-Transcriptional Regulatory Decomposition  ·  Cheng & Jin 2026",
        ha="center", va="bottom", fontsize=10, color=GRAY, transform=ax.transAxes)

save(fig, "fig8_summary.png", dpi=150)

print(f"\nAll figures saved to {OUT}")
print("Files:")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}")
