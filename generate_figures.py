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
    # Wider figure to fill page width
    fig = plt.figure(figsize=(7.5, 2.8))
    # Give panel C more room (increased from 1.0 to 1.4)
    gs = gridspec.GridSpec(1, 5, figure=fig,
                           width_ratios=[1.2, 0.03, 0.9, 0.03, 1.4],
                           wspace=0.06, left=0.06, right=0.98)

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
    ax_a.set_xlabel("Spliced ($s$)", fontsize=10)
    ax_a.set_ylabel("Unspliced ($u$)", fontsize=10)
    ax_a.set_title("Rate estimation", fontweight="bold", fontsize=11)
    ax_a.legend(fontsize=8, loc="upper left", frameon=False)
    ax_a.text(-0.18, 1.05, "A", transform=ax_a.transAxes, fontsize=14,
              fontweight="bold")
    cb = plt.colorbar(sc, ax=ax_a, shrink=0.75, aspect=15, pad=0.03)
    cb.set_label(r"$\gamma$", fontsize=10)
    cb.ax.tick_params(labelsize=7)

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
    ax_bl.set_xlabel("Expression\nspace", fontsize=8)
    ax_bl.text(-0.3, 1.05, "B", transform=ax_bl.transAxes, fontsize=14,
               fontweight="bold")

    ax_br = fig.add_subplot(gs_b[0, 1])
    ax_br.scatter(x_gamma, y_gamma, c=colors_g, s=6, alpha=0.6, edgecolors="none")
    _hide_spines(ax_br)
    ax_br.set_xlabel(r"$\gamma$ space", fontsize=8)

    fig.text(0.455, 0.97, "Invisible states", fontsize=11, fontweight="bold",
             ha="center", va="top")

    # Panel C: Velocity + Network — give network more room
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 4],
                                            wspace=0.15,
                                            width_ratios=[0.8, 1.2])

    # Streamlines
    ax_cl = fig.add_subplot(gs_c[0, 0])
    t = np.linspace(0, 1, 200)
    x_traj = t + np.random.normal(0, 0.07, 200)
    y_traj = np.sin(t * 2.5) * 0.8 + np.random.normal(0, 0.07, 200)
    ax_cl.scatter(x_traj, y_traj, c=t, cmap="viridis", s=8, alpha=0.6,
                  edgecolors="none")
    for i in range(0, 180, 25):
        dx = x_traj[i + 15] - x_traj[i]
        dy = y_traj[i + 15] - y_traj[i]
        ax_cl.annotate("", xy=(x_traj[i] + dx * 0.6, y_traj[i] + dy * 0.6),
                        xytext=(x_traj[i], y_traj[i]),
                        arrowprops=dict(arrowstyle="->", color="k", lw=1.0))
    _hide_spines(ax_cl)
    ax_cl.set_xlabel("PT velocity", fontsize=8)
    ax_cl.text(-0.2, 1.05, "C", transform=ax_cl.transAxes, fontsize=14,
               fontweight="bold")

    # Network — bigger, more space for labels
    ax_cr = fig.add_subplot(gs_c[0, 1])
    n_nodes = 6
    labels = ["RBP1", "RBP2", "T1", "T2", "T3", "T4"]
    # Spread nodes further apart
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False) - np.pi / 2
    radius = 1.0
    nx_pos = radius * np.cos(angles)
    ny_pos = radius * np.sin(angles)
    edges = [(0, 2, RED), (0, 3, RED), (1, 4, BLUE), (1, 5, BLUE),
             (0, 4, RED), (1, 2, BLUE)]
    for i, j, c in edges:
        # Shorten arrows so they don't overlap nodes
        dx = nx_pos[j] - nx_pos[i]
        dy = ny_pos[j] - ny_pos[i]
        dist = np.sqrt(dx**2 + dy**2)
        shrink = 0.15 / dist
        ax_cr.annotate("",
                        xy=(nx_pos[j] - dx * shrink, ny_pos[j] - dy * shrink),
                        xytext=(nx_pos[i] + dx * shrink, ny_pos[i] + dy * shrink),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2,
                                        alpha=0.7, mutation_scale=12))
    for i, lab in enumerate(labels):
        fc = TEAL if "RBP" in lab else "#DDDDDD"
        ax_cr.plot(nx_pos[i], ny_pos[i], "o", ms=12, color=fc,
                   markeredgecolor="k", markeredgewidth=0.7)
        # Place label outside the node, further out for clarity
        label_r = 1.55
        lx = label_r * np.cos(angles[i])
        ly = label_r * np.sin(angles[i])
        ax_cr.text(lx, ly, lab, fontsize=8, ha="center", va="center",
                   fontweight="bold")
    ax_cr.set_xlim(-2.1, 2.1)
    ax_cr.set_ylim(-2.1, 2.1)
    # xlim/ylim set above after label placement
    ax_cr.set_aspect("equal")
    _hide_spines(ax_cr)
    ax_cr.set_xlabel("RBP network", fontsize=8)

    fig.text(0.80, 0.97, "Velocity & networks", fontsize=11, fontweight="bold",
             ha="center", va="top")

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
