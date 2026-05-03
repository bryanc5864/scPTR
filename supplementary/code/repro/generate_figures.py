"""Generate figures for scPTR ISMB 2026 two-page abstract."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "mathtext.fontset": "dejavuserif",
})

BLUE = "#3274A1"
RED = "#E1812C"
PURPLE = "#9372B2"
GRAY = "#7F7F7F"
TEAL = "#17BECF"
ORANGE = "#FF7F0E"


def _hide_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


# ── FIGURE 1 ─────────────────────────────────────────────────────────────────

def fig1_method():
    # Full page width (textwidth ~7.5in for 0.6in margins)
    fig = plt.figure(figsize=(7.8, 2.9))
    gs = gridspec.GridSpec(1, 5, figure=fig,
                           width_ratios=[1.1, 0.03, 0.85, 0.03, 1.8],
                           wspace=0.06, left=0.05, right=0.99)

    # Panel A: Phase portrait
    ax_a = fig.add_subplot(gs[0, 0])
    n = 400
    s = np.random.exponential(2.0, n)
    beta_true = 0.6
    u = beta_true * s + np.random.normal(0, 0.4, n)
    u = np.clip(u, 0, None)
    gamma_vals = beta_true * u / np.maximum(s, 0.01)
    sc = ax_a.scatter(s, u, c=gamma_vals, cmap="YlOrRd", s=10, alpha=0.7,
                      edgecolors="none", vmin=0, vmax=1.0)
    s_line = np.linspace(0, s.max(), 50)
    ax_a.plot(s_line, beta_true * s_line, "k--", lw=1.8,
              label=r"$\beta$ (95th %ile)")
    ax_a.set_xlabel("Spliced ($s$)", fontsize=11)
    ax_a.set_ylabel("Unspliced ($u$)", fontsize=11)
    ax_a.set_title("Rate estimation", fontweight="bold", fontsize=12)
    ax_a.legend(fontsize=9, loc="upper left", frameon=False)
    ax_a.text(-0.18, 1.05, "A", transform=ax_a.transAxes, fontsize=14,
              fontweight="bold")
    cb = plt.colorbar(sc, ax=ax_a, shrink=0.75, aspect=15, pad=0.03)
    cb.set_label(r"$\gamma$", fontsize=11)
    cb.ax.tick_params(labelsize=8)

    # Panel B: Expression vs Gamma UMAP
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 2], wspace=0.25)

    theta = np.random.uniform(0, 2 * np.pi, 400)
    r_expr = np.random.normal(0, 1.0, 400)
    x_expr = r_expr * np.cos(theta)
    y_expr = r_expr * np.sin(theta)
    group = (np.sin(theta * 2) + np.random.normal(0, 0.3, 400)) > 0
    x_gamma = x_expr + group * 3.5
    y_gamma = y_expr + group * 0.5 + np.random.normal(0, 0.3, 400)
    colors_g = [BLUE if g else ORANGE for g in group]

    ax_bl = fig.add_subplot(gs_b[0, 0])
    ax_bl.scatter(x_expr, y_expr, c=GRAY, s=6, alpha=0.5, edgecolors="none")
    _hide_spines(ax_bl)
    ax_bl.set_xlabel("Expression\nspace", fontsize=10)
    ax_bl.text(-0.3, 1.05, "B", transform=ax_bl.transAxes, fontsize=14,
               fontweight="bold")

    ax_br = fig.add_subplot(gs_b[0, 1])
    ax_br.scatter(x_gamma, y_gamma, c=colors_g, s=6, alpha=0.6, edgecolors="none")
    _hide_spines(ax_br)
    ax_br.set_xlabel(r"$\gamma$ space", fontsize=10)

    fig.text(0.455, 0.97, "Invisible states", fontsize=12, fontweight="bold",
             ha="center", va="top")

    # Panel C: PT velocity on real gamma UMAP + real RBP network hubs
    import json
    from pathlib import Path
    DATA = Path("real_figure_data")

    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 4],
                                            wspace=0.35,
                                            width_ratios=[1.0, 1.1])

    # ── Left: PT velocity field on gamma-space UMAP (real data) ──
    ax_cl = fig.add_subplot(gs_c[0, 0])
    vel_data = np.load(DATA / "pt_velocity_umap.npz", allow_pickle=True)
    umap = vel_data["umap"]
    vel = vel_data["velocity"]
    ctypes = vel_data["cell_types"]

    # Color by cell type
    unique_ct = sorted(set(ctypes))
    ct_colors = dict(zip(unique_ct,
        plt.cm.tab10(np.linspace(0, 1, len(unique_ct)))))
    c_arr = [ct_colors[c] for c in ctypes]

    ax_cl.scatter(umap[:, 0], umap[:, 1], c=c_arr, s=5, alpha=0.6,
                  edgecolors="none", rasterized=True)

    # Draw velocity arrows on a coarse grid
    n_grid = 8
    x_edges = np.linspace(umap[:, 0].min(), umap[:, 0].max(), n_grid + 1)
    y_edges = np.linspace(umap[:, 1].min(), umap[:, 1].max(), n_grid + 1)
    for xi in range(n_grid):
        for yi in range(n_grid):
            mask = ((umap[:, 0] >= x_edges[xi]) & (umap[:, 0] < x_edges[xi + 1]) &
                    (umap[:, 1] >= y_edges[yi]) & (umap[:, 1] < y_edges[yi + 1]))
            if mask.sum() < 5:
                continue
            cx = umap[mask, 0].mean()
            cy = umap[mask, 1].mean()
            vx = vel[mask, 0].mean()
            vy = vel[mask, 1].mean()
            vmag = np.sqrt(vx**2 + vy**2)
            if vmag < 0.03:
                continue
            arrow_scale = 1.2
            ax_cl.annotate("",
                xy=(cx + vx * arrow_scale, cy + vy * arrow_scale),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="0.15", lw=0.6,
                                mutation_scale=6, alpha=0.5))

    _hide_spines(ax_cl)
    ax_cl.set_xlabel("PT velocity ($\\gamma$ UMAP)", fontsize=10)
    ax_cl.text(-0.15, 1.05, "C", transform=ax_cl.transAxes, fontsize=14,
               fontweight="bold")

    # ── Right: Top RBP hubs bar chart (real data) ──
    ax_cr = fig.add_subplot(gs_c[0, 1])
    with open(DATA / "rbp_network.json") as f:
        net = json.load(f)

    hubs = net["top_hubs"][:5]  # top 5
    rbp_names = [h["rbp"] for h in hubs]
    dest = [h["destabilizing"] for h in hubs]
    stab = [-h["stabilizing"] for h in hubs]  # negative for left side
    y_pos = np.arange(len(rbp_names))

    ax_cr.barh(y_pos, dest, height=0.55, color=RED, alpha=0.85,
               edgecolor="white", linewidth=0.5, label="Destab.")
    ax_cr.barh(y_pos, stab, height=0.55, color=BLUE, alpha=0.85,
               edgecolor="white", linewidth=0.5, label="Stab.")
    ax_cr.set_yticks(y_pos)
    ax_cr.set_yticklabels(rbp_names, fontsize=9, style="italic")
    ax_cr.axvline(0, color="k", lw=0.6)
    ax_cr.set_xlabel("Targets", fontsize=10)
    ax_cr.tick_params(axis="x", labelsize=8)
    ax_cr.legend(fontsize=7, loc="upper right", frameon=False)
    ax_cr.spines["top"].set_visible(False)
    ax_cr.spines["right"].set_visible(False)

    # Center the "Velocity & networks" title over both C sub-panels
    c_left = ax_cl.get_position().x0
    c_right = ax_cr.get_position().x1
    fig.text((c_left + c_right) / 2, 0.97, "Velocity & networks",
             fontsize=12, fontweight="bold", ha="center", va="top")

    fig.savefig("figures/fig1_method.pdf")
    plt.close(fig)
    print("Saved fig1_method.pdf")


# ── FIGURE 2 ─────────────────────────────────────────────────────────────────

def fig2_validation():
    import json
    from pathlib import Path

    DATA = Path("real_figure_data")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.6))

    # Panel A: Half-life scatter — REAL DATA from pancreas scPTR pipeline
    ax = axes[0]
    hl_data = np.load(DATA / "halflife_scatter.npz")
    gamma_vals = hl_data["gamma"]
    halflife_vals = hl_data["halflife"]
    sp_r = float(hl_data["spearman_r"])
    n_genes = int(hl_data["n_genes"])

    log_hl = np.log10(halflife_vals)
    log_gamma = np.log10(gamma_vals)

    ax.scatter(log_hl, log_gamma, s=4, alpha=0.25, c=BLUE, edgecolors="none",
               rasterized=True)
    z = np.polyfit(log_hl, log_gamma, 1)
    x_fit = np.linspace(log_hl.min(), log_hl.max(), 50)
    ax.plot(x_fit, np.polyval(z, x_fit), "k-", lw=1.5)
    ax.set_xlabel(r"log$_{10}$(mRNA half-life, hr)", fontsize=10)
    ax.set_ylabel(r"log$_{10}$(median $\gamma$)", fontsize=10)
    ax.set_title("Half-life validation (pancreas)", fontweight="bold", fontsize=11)
    ax.text(0.97, 0.95,
            f"Spearman $\\rho$ = {sp_r:.2f}\n"
            f"$p$ < 10$^{{-160}}$\n"
            f"$n$ = {n_genes:,} genes",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9,
                      edgecolor=GRAY, linewidth=0.5))
    ax.text(-0.16, 1.05, "A", transform=ax.transAxes, fontsize=14,
            fontweight="bold")

    # Panel B: miRNA target enrichment — real documented values from RESULTS.md
    ax = axes[1]
    with open(DATA / "mirna_enrichment_documented.json") as f:
        mirna = json.load(f)

    families_data = mirna["top_families"]
    families = [d["mirna"] for d in families_data]
    fold_enrich = [d["fold"] for d in families_data]
    n_targets = [d["targets"] for d in families_data]
    y_pos = np.arange(len(families))

    bars = ax.barh(y_pos, fold_enrich, height=0.55, color=RED, alpha=0.85,
                   edgecolor="white", linewidth=0.5)
    for i, (fe, nt) in enumerate(zip(fold_enrich, n_targets)):
        ax.text(fe + 2, y_pos[i], f"***  ({nt} targets)", va="center",
                fontsize=7.5, color="0.3")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(families, fontsize=9)
    ax.set_xlabel("Fold enrichment (target vs non-target $\\gamma$)", fontsize=10)
    ax.set_title("miRNA target enrichment (pancreas)", fontweight="bold", fontsize=11)
    ax.text(0.03, 0.97,
            f"{mirna['n_significant_fdr05']} / {mirna['n_total']} families\n"
            f"significant (FDR < 0.05)\n"
            r"aggregate $p$ = 4.7$\times$10$^{-65}$",
            transform=ax.transAxes, fontsize=8, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9,
                      edgecolor=GRAY, linewidth=0.5))
    ax.text(-0.22, 1.05, "B", transform=ax.transAxes, fontsize=14,
            fontweight="bold")

    fig.tight_layout()
    fig.savefig("figures/fig2_validation.pdf")
    plt.close(fig)
    print("Saved fig2_validation.pdf")


# ── FIGURE 3 ─────────────────────────────────────────────────────────────────

def fig3_findings():
    from pathlib import Path

    DATA = Path("real_figure_data")

    fig = plt.figure(figsize=(7.5, 2.8))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30,
                           left=0.05, right=0.97)

    # --- Panel A: side-by-side UMAPs — REAL epsilon cell data ---
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], wspace=0.20)

    eps = np.load(DATA / "epsilon_states.npz", allow_pickle=True)
    expr_umap = eps["expr_umap"]
    gamma_umap = eps["gamma_umap"]
    leiden = eps["leiden"]
    sil_gamma = float(eps["sil_gamma"])
    sil_expr = float(eps["sil_expr"])
    n_cells = int(eps["n_cells"])

    # Color by leiden cluster
    unique_labels = sorted(set(leiden))
    cmap = [BLUE, ORANGE, TEAL, RED, PURPLE]
    colors_state = [cmap[unique_labels.index(l) % len(cmap)] for l in leiden]

    ax_al = fig.add_subplot(gs_a[0, 0])
    ax_al.scatter(expr_umap[:, 0], expr_umap[:, 1], c=colors_state, s=18,
                  alpha=0.6, edgecolors="none")
    _hide_spines(ax_al)
    ax_al.set_xlabel("Expression UMAP", fontsize=10)
    sil_e_str = f"$-${abs(sil_expr):.3f}" if sil_expr < 0 else f"{sil_expr:.3f}"
    ax_al.text(0.5, -0.20, f"Silhouette = {sil_e_str}",
               transform=ax_al.transAxes, fontsize=9, ha="center", color=GRAY)
    ax_al.text(-0.12, 1.05, "A", transform=ax_al.transAxes, fontsize=14,
               fontweight="bold")

    ax_ar = fig.add_subplot(gs_a[0, 1])
    ax_ar.scatter(gamma_umap[:, 0], gamma_umap[:, 1], c=colors_state, s=18,
                  alpha=0.6, edgecolors="none")
    _hide_spines(ax_ar)
    ax_ar.set_xlabel(r"$\gamma$ UMAP", fontsize=10)
    ax_ar.text(0.5, -0.20, f"Silhouette = {sil_gamma:.3f}",
               transform=ax_ar.transAxes, fontsize=9, ha="center", color=RED)

    fig.text(0.27, 1.0, f"Invisible PT states (Epsilon, $n$={n_cells} cells)",
             fontsize=11, fontweight="bold", ha="center", va="top")

    # --- Panel B: Temporal precedence — REAL per-gene onset data ---
    ax_b = fig.add_subplot(gs[0, 1])

    prec = np.load(DATA / "temporal_precedence.npz")
    example_gamma = prec["example_gamma_profile"]
    example_expr = prec["example_expr_profile"]
    pt_bins = prec["pseudotime_bins"]
    pct_gamma = float(prec["pct_gamma"])
    pct_expr = float(prec["pct_expr"])
    p_binom = float(prec["p_binom"])
    n_trans = int(prec["n_transition"])
    gamma_leads_n = int(prec["gamma_leads"])
    expr_leads_n = int(prec["expr_leads"])
    example_gene = str(prec["example_gene"])

    # Smooth profiles for visual clarity
    from scipy.ndimage import uniform_filter1d
    g_smooth = uniform_filter1d(example_gamma, size=5)
    e_smooth = uniform_filter1d(example_expr, size=5)

    ax_b.plot(pt_bins, g_smooth, color=RED, lw=2.5,
              label=r"$\gamma$ (degradation rate)")
    ax_b.plot(pt_bins, e_smooth, color=BLUE, lw=2.5, label="Expression")

    # Find onset points (20% threshold)
    threshold = 0.2
    g_onset_idx = np.argmax(g_smooth >= threshold)
    e_onset_idx = np.argmax(e_smooth >= threshold)
    g_onset_t = pt_bins[g_onset_idx]
    e_onset_t = pt_bins[e_onset_idx]

    # Shade the lag region
    if g_onset_t < e_onset_t:
        ax_b.axvspan(g_onset_t, e_onset_t, alpha=0.12, color=PURPLE)
        mid_t = (g_onset_t + e_onset_t) / 2
        ax_b.annotate("", xy=(e_onset_t, 0.50), xytext=(g_onset_t, 0.50),
                      arrowprops=dict(arrowstyle="<->", color=PURPLE, lw=2.0))
        ax_b.text(mid_t, 0.62, r"$\gamma$ leads", fontsize=11, ha="center",
                  color=PURPLE, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                            edgecolor=PURPLE, alpha=0.9, linewidth=0.8))
        ax_b.plot(g_onset_t, 0.5, "o", color=RED, ms=7, zorder=5)
        ax_b.plot(e_onset_t, 0.5, "o", color=BLUE, ms=7, zorder=5)

    ax_b.set_xlabel("Pseudotime", fontsize=11)
    ax_b.set_ylabel("Normalized signal", fontsize=11)
    ax_b.set_title(f"Temporal precedence ({example_gene})",
                   fontweight="bold", fontsize=12)
    ax_b.legend(fontsize=10, frameon=False, loc="upper left")
    ax_b.text(-0.14, 1.05, "B", transform=ax_b.transAxes, fontsize=14,
              fontweight="bold")

    p_exp = int(np.floor(np.log10(max(p_binom, 1e-300))))
    ax_b.text(0.97, 0.42,
              f"{pct_gamma:.0f}% of {n_trans} transition genes\n"
              r"$\gamma$ leads expression" + "\n"
              f"$p$ < 10$^{{{p_exp}}}$",
              transform=ax_b.transAxes, fontsize=9.5, ha="right", va="top",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9,
                        edgecolor=GRAY, linewidth=0.5))
    ax_b.set_xlim(-0.02, 1.02)
    ax_b.set_ylim(-0.08, 1.15)

    fig.savefig("figures/fig3_findings.pdf")
    plt.close(fig)
    print("Saved fig3_findings.pdf")


if __name__ == "__main__":
    fig1_method()
    fig2_validation()
    fig3_findings()
    print("All figures generated.")
