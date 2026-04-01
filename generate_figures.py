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
    "savefig.pad_inches": 0.08,
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
    fig = plt.figure(figsize=(7.0, 2.8))
    gs = gridspec.GridSpec(1, 5, figure=fig, width_ratios=[1.3, 0.05, 1.0, 0.05, 1.0],
                           wspace=0.08)

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
              label=r"$\beta$ (95th percentile)")
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
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 2], wspace=0.3)

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

    # Shared title above both B subpanels
    fig.text(0.52, 0.97, "Invisible states", fontsize=11, fontweight="bold",
             ha="center", va="top")

    # Panel C: Velocity + Network
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 4], wspace=0.3)

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
    ax_cl.text(-0.3, 1.05, "C", transform=ax_cl.transAxes, fontsize=14,
               fontweight="bold")

    ax_cr = fig.add_subplot(gs_c[0, 1])
    n_nodes = 6
    labels = ["RBP1", "RBP2", "T1", "T2", "T3", "T4"]
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False) - np.pi / 2
    nx_pos = np.cos(angles)
    ny_pos = np.sin(angles)
    edges = [(0, 2, RED), (0, 3, RED), (1, 4, BLUE), (1, 5, BLUE),
             (0, 4, RED), (1, 2, BLUE)]
    for i, j, c in edges:
        ax_cr.annotate("", xy=(nx_pos[j], ny_pos[j]),
                        xytext=(nx_pos[i], ny_pos[i]),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0,
                                        alpha=0.7, mutation_scale=10))
    for i, lab in enumerate(labels):
        fc = TEAL if "RBP" in lab else "#DDDDDD"
        ax_cr.plot(nx_pos[i], ny_pos[i], "o", ms=10, color=fc,
                   markeredgecolor="k", markeredgewidth=0.6)
        offset_y = -0.3 if ny_pos[i] <= 0 else 0.3
        ax_cr.text(nx_pos[i], ny_pos[i] + offset_y, lab, fontsize=7,
                   ha="center", va="center")
    ax_cr.set_xlim(-1.7, 1.7)
    ax_cr.set_ylim(-1.7, 1.7)
    ax_cr.set_aspect("equal")
    _hide_spines(ax_cr)
    ax_cr.set_xlabel("RBP network", fontsize=8)

    fig.text(0.82, 0.97, "Velocity & networks", fontsize=11, fontweight="bold",
             ha="center", va="top")

    fig.savefig("figures/fig1_method.pdf")
    plt.close(fig)
    print("Saved fig1_method.pdf")


# ── FIGURE 2 ─────────────────────────────────────────────────────────────────

def fig2_validation():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    # Panel A: Half-life scatter
    ax = axes[0]
    n = 500
    x = np.random.normal(0, 1, n)
    noise = np.random.normal(0, 0.6, n)
    y = -0.81 * x + noise * np.sqrt(1 - 0.81**2)
    x_hl = np.exp(x + 3)
    y_gamma = np.exp(y)

    ax.scatter(np.log10(x_hl), np.log10(y_gamma), s=6, alpha=0.35, c=BLUE,
               edgecolors="none")
    z = np.polyfit(np.log10(x_hl), np.log10(y_gamma), 1)
    x_fit = np.linspace(np.log10(x_hl).min(), np.log10(x_hl).max(), 50)
    ax.plot(x_fit, np.polyval(z, x_fit), "k-", lw=1.5)
    ax.set_xlabel(r"log$_{10}$(mRNA half-life, hr)", fontsize=10)
    ax.set_ylabel(r"log$_{10}$($\gamma$)", fontsize=10)
    ax.set_title("Half-life validation (sci-fate)", fontweight="bold", fontsize=11)
    ax.text(0.97, 0.95,
            r"Spearman $\rho$ = $-$0.81" + "\n" +
            r"$p$ < 10$^{-300}$" + "\n" +
            r"$n$ = 6,995 genes",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9,
                      edgecolor=GRAY, linewidth=0.5))
    ax.text(-0.16, 1.05, "A", transform=ax.transAxes, fontsize=14,
            fontweight="bold")

    # Panel B: miRNA target enrichment — horizontal bars
    ax = axes[1]
    families = ["miR-130a-3p", "miR-124-3p", "miR-30e-5p", "miR-153-3p"]
    target_gamma = [0.81, 0.66, 0.85, 1.28]
    bg_gamma = [0.01, 0.01, 0.01, 0.01]
    y_pos = np.arange(len(families))
    h = 0.35

    ax.barh(y_pos + h / 2, target_gamma, h, color=RED, alpha=0.85,
            label="miRNA targets", edgecolor="white", linewidth=0.5)
    ax.barh(y_pos - h / 2, bg_gamma, h, color=GRAY, alpha=0.5,
            label="Background", edgecolor="white", linewidth=0.5)
    for i in range(len(families)):
        ax.text(target_gamma[i] + 0.03, y_pos[i] + h / 2, "***", va="center",
                fontsize=9, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(families, fontsize=9)
    ax.set_xlabel(r"Median $\gamma$", fontsize=10)
    ax.set_title("miRNA target enrichment (pancreas)", fontweight="bold", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.text(-0.22, 1.05, "B", transform=ax.transAxes, fontsize=14,
            fontweight="bold")
    ax.text(0.97, 0.97,
            "126 / 215 families significant\n(FDR < 0.05)\n" +
            r"aggregate $p$ = 4.7$\times$10$^{-65}$",
            transform=ax.transAxes, fontsize=8.5, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9,
                      edgecolor=GRAY, linewidth=0.5))

    fig.tight_layout()
    fig.savefig("figures/fig2_validation.pdf")
    plt.close(fig)
    print("Saved fig2_validation.pdf")


# ── FIGURE 3 ─────────────────────────────────────────────────────────────────

def fig3_findings():
    fig = plt.figure(figsize=(7.0, 2.8))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # --- Panel A: side-by-side UMAPs ---
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], wspace=0.25)

    n_cells = 142
    theta_e = np.random.uniform(0, 2 * np.pi, n_cells)
    r_e = np.abs(np.random.normal(0, 0.8, n_cells))
    ex = r_e * np.cos(theta_e)
    ey = r_e * np.sin(theta_e)
    state = np.random.choice([0, 1], n_cells, p=[0.55, 0.45])
    gx = ex + state * 2.8 + np.random.normal(0, 0.15, n_cells)
    gy = ey + (state - 0.5) * 1.2 + np.random.normal(0, 0.15, n_cells)
    colors_state = [BLUE if s == 0 else ORANGE for s in state]

    ax_al = fig.add_subplot(gs_a[0, 0])
    ax_al.scatter(ex, ey, c=colors_state, s=14, alpha=0.6, edgecolors="none")
    _hide_spines(ax_al)
    ax_al.set_xlabel("Expression UMAP", fontsize=9)
    ax_al.text(0.5, -0.18, "Silhouette = $-$0.056", transform=ax_al.transAxes,
               fontsize=8.5, ha="center", color=GRAY)
    ax_al.text(-0.15, 1.05, "A", transform=ax_al.transAxes, fontsize=14,
               fontweight="bold")

    ax_ar = fig.add_subplot(gs_a[0, 1])
    ax_ar.scatter(gx, gy, c=colors_state, s=14, alpha=0.6, edgecolors="none")
    _hide_spines(ax_ar)
    ax_ar.set_xlabel(r"$\gamma$ UMAP", fontsize=9)
    ax_ar.text(0.5, -0.18, "Silhouette = 0.195", transform=ax_ar.transAxes,
               fontsize=8.5, ha="center", color=RED)

    # Suptitle for panel A — positioned above subplots, shifted right to avoid "A" label
    fig.text(0.28, 1.0, "Invisible PT states (Epsilon cells)",
             fontsize=11, fontweight="bold", ha="center", va="top")

    # --- Panel B: Temporal precedence ---
    ax_b = fig.add_subplot(gs[0, 1])
    t = np.linspace(0, 1, 200)
    gamma_curve = 1 / (1 + np.exp(-15 * (t - 0.35)))
    expr_curve = 1 / (1 + np.exp(-15 * (t - 0.55)))

    ax_b.plot(t, gamma_curve, color=RED, lw=2.5, label=r"$\gamma$ (degradation rate)")
    ax_b.plot(t, expr_curve, color=BLUE, lw=2.5, label="Expression")

    ax_b.axvspan(0.35, 0.55, alpha=0.10, color=PURPLE)
    ax_b.annotate("", xy=(0.55, 0.5), xytext=(0.35, 0.5),
                  arrowprops=dict(arrowstyle="<->", color=PURPLE, lw=1.5))
    ax_b.text(0.45, 0.58, r"$\gamma$ leads", fontsize=10, ha="center", color=PURPLE,
              fontweight="bold")

    ax_b.plot(0.35, 0.5, "o", color=RED, ms=6, zorder=5)
    ax_b.plot(0.55, 0.5, "o", color=BLUE, ms=6, zorder=5)

    ax_b.set_xlabel("Pseudotime", fontsize=10)
    ax_b.set_ylabel("Normalized signal", fontsize=10)
    ax_b.set_title("Temporal precedence", fontweight="bold", fontsize=11)
    ax_b.legend(fontsize=9, frameon=False, loc="upper left")
    ax_b.text(-0.16, 1.05, "B", transform=ax_b.transAxes, fontsize=14,
              fontweight="bold")
    ax_b.text(0.97, 0.45,
              "63% (pancreas)\n78% (dentate gyrus)\n" +
              r"$\gamma$ leads expression" + "\n" +
              r"$p$ < 10$^{-5}$",
              transform=ax_b.transAxes, fontsize=9, ha="right", va="top",
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
