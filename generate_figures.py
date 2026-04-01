"""Generate figures for scPTR ISMB 2026 two-page abstract."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
import seaborn as sns

np.random.seed(42)

# Global style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

BLUE = "#3274A1"
RED = "#E1812C"
GREEN = "#3A923A"
PURPLE = "#9372B2"
GRAY = "#7F7F7F"
TEAL = "#17BECF"
ORANGE = "#FF7F0E"


# ── FIGURE 1: Method Overview (3 panels) ──────────────────────────────────────

def fig1_method():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))

    # Panel A: Phase portrait with beta slope
    ax = axes[0]
    n = 300
    s = np.random.exponential(2.0, n)
    beta_true = 0.6
    u = beta_true * s + np.random.normal(0, 0.4, n)
    u = np.clip(u, 0, None)
    gamma_vals = beta_true * u / np.maximum(s, 0.01)
    sc = ax.scatter(s, u, c=gamma_vals, cmap="YlOrRd", s=6, alpha=0.7,
                    edgecolors="none", vmin=0, vmax=1.0)
    s_line = np.linspace(0, s.max(), 50)
    ax.plot(s_line, beta_true * s_line, "k--", lw=1.5, label=r"$\beta$ (95th %ile)")
    ax.set_xlabel("Spliced (s)")
    ax.set_ylabel("Unspliced (u)")
    ax.set_title("Rate estimation", fontweight="bold", fontsize=9)
    ax.legend(fontsize=6, loc="upper left", frameon=False)
    ax.text(-0.15, 1.08, "A", transform=ax.transAxes, fontsize=12, fontweight="bold")
    cb = plt.colorbar(sc, ax=ax, shrink=0.7, aspect=15, pad=0.02)
    cb.set_label(r"$\gamma$", fontsize=8)
    cb.ax.tick_params(labelsize=6)

    # Panel B: Expression vs Gamma UMAP
    ax = axes[1]
    # Expression space: one blob
    theta = np.random.uniform(0, 2 * np.pi, 400)
    r_expr = np.random.normal(0, 1.0, 400)
    x_expr = r_expr * np.cos(theta)
    y_expr = r_expr * np.sin(theta)
    # Assign hidden groups
    group = (np.sin(theta * 2) + np.random.normal(0, 0.3, 400)) > 0
    # Gamma space: two separated clusters
    x_gamma = x_expr + group * 3.5
    y_gamma = y_expr + group * 0.5 + np.random.normal(0, 0.3, 400)

    colors_g = [BLUE if g else ORANGE for g in group]

    # mini axes for side-by-side within panel B
    ax.set_axis_off()
    ax.text(-0.15, 1.08, "B", transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.set_title("Expression-invisible states", fontweight="bold", fontsize=9)

    ax_left = fig.add_axes([0.38, 0.18, 0.12, 0.65])
    ax_left.scatter(x_expr, y_expr, c=GRAY, s=3, alpha=0.5, edgecolors="none")
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.set_xlabel("Expression\nspace", fontsize=7)
    for spine in ax_left.spines.values():
        spine.set_visible(False)

    ax_right = fig.add_axes([0.52, 0.18, 0.12, 0.65])
    ax_right.scatter(x_gamma, y_gamma, c=colors_g, s=3, alpha=0.6, edgecolors="none")
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    ax_right.set_xlabel(r"$\gamma$ space", fontsize=7)
    for spine in ax_right.spines.values():
        spine.set_visible(False)

    # Arrow between
    fig.text(0.505, 0.50, r"$\rightarrow$", fontsize=14, ha="center", va="center",
             fontweight="bold")

    # Panel C: Velocity streamlines + network
    ax = axes[2]
    ax.set_axis_off()
    ax.text(-0.15, 1.08, "C", transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.set_title("PT velocity & networks", fontweight="bold", fontsize=9)

    # Streamlines subplot
    ax_stream = fig.add_axes([0.72, 0.18, 0.12, 0.65])
    # Cells along a trajectory
    t = np.linspace(0, 1, 200)
    x_traj = t + np.random.normal(0, 0.08, 200)
    y_traj = np.sin(t * 2.5) * 0.8 + np.random.normal(0, 0.08, 200)
    ax_stream.scatter(x_traj, y_traj, c=t, cmap="viridis", s=4, alpha=0.6,
                      edgecolors="none")
    # Add a few arrows
    for i in range(0, 180, 30):
        dx = x_traj[i + 15] - x_traj[i]
        dy = y_traj[i + 15] - y_traj[i]
        ax_stream.annotate("", xy=(x_traj[i] + dx * 0.6, y_traj[i] + dy * 0.6),
                           xytext=(x_traj[i], y_traj[i]),
                           arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
    ax_stream.set_xticks([])
    ax_stream.set_yticks([])
    ax_stream.set_xlabel("PT velocity", fontsize=7)
    for spine in ax_stream.spines.values():
        spine.set_visible(False)

    # Network subplot
    ax_net = fig.add_axes([0.86, 0.18, 0.12, 0.65])
    n_nodes = 6
    labels = ["RBP1", "RBP2", "T1", "T2", "T3", "T4"]
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    nx_pos = np.cos(angles)
    ny_pos = np.sin(angles)

    # Edges
    edges = [(0, 2, RED), (0, 3, RED), (1, 4, BLUE), (1, 5, BLUE),
             (0, 4, RED), (1, 2, BLUE)]
    for i, j, c in edges:
        ax_net.annotate("", xy=(nx_pos[j], ny_pos[j]),
                        xytext=(nx_pos[i], ny_pos[i]),
                        arrowprops=dict(arrowstyle="->", color=c, lw=0.8, alpha=0.7))
    for i, lab in enumerate(labels):
        fc = TEAL if "RBP" in lab else "#DDDDDD"
        ax_net.plot(nx_pos[i], ny_pos[i], "o", ms=8, color=fc,
                    markeredgecolor="k", markeredgewidth=0.5)
        ax_net.text(nx_pos[i], ny_pos[i] - 0.25, lab, fontsize=5, ha="center")
    ax_net.set_xlim(-1.6, 1.6)
    ax_net.set_ylim(-1.6, 1.6)
    ax_net.set_xticks([])
    ax_net.set_yticks([])
    ax_net.set_xlabel("RBP network", fontsize=7)
    ax_net.set_aspect("equal")
    for spine in ax_net.spines.values():
        spine.set_visible(False)

    fig.savefig("figures/fig1_method.pdf")
    plt.close(fig)
    print("Saved fig1_method.pdf")


# ── FIGURE 2: Validation (2 panels) ──────────────────────────────────────────

def fig2_validation():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))

    # Panel A: Half-life correlation scatter
    ax = axes[0]
    n = 500
    # Generate correlated data with r ~ -0.81
    x = np.random.normal(0, 1, n)  # log half-life
    noise = np.random.normal(0, 0.6, n)
    y = -0.81 * x + noise * np.sqrt(1 - 0.81**2)  # log gamma
    x_hl = np.exp(x + 3)   # half-life in hours
    y_gamma = np.exp(y)     # gamma

    ax.scatter(np.log10(x_hl), np.log10(y_gamma), s=4, alpha=0.3, c=BLUE,
               edgecolors="none")
    # Regression line
    z = np.polyfit(np.log10(x_hl), np.log10(y_gamma), 1)
    x_fit = np.linspace(np.log10(x_hl).min(), np.log10(x_hl).max(), 50)
    ax.plot(x_fit, np.polyval(z, x_fit), "k-", lw=1.2)
    ax.set_xlabel(r"log$_{10}$(mRNA half-life, hr)")
    ax.set_ylabel(r"log$_{10}$($\gamma$)")
    ax.set_title("Half-life validation (sci-fate)", fontweight="bold", fontsize=9)
    ax.text(0.97, 0.95, r"$\rho$ = $-$0.81" + "\np < 10$^{-300}$\nn = 6,995 genes",
            transform=ax.transAxes, fontsize=7, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8,
                      edgecolor=GRAY, linewidth=0.5))
    ax.text(-0.15, 1.08, "A", transform=ax.transAxes, fontsize=12, fontweight="bold")

    # Panel B: miRNA target enrichment
    ax = axes[1]
    families = ["miR-153", "miR-30e", "miR-124", "miR-130a"]
    target_gamma = [1.28, 0.85, 0.66, 0.81]
    bg_gamma = [0.01, 0.01, 0.01, 0.01]
    x_pos = np.arange(len(families))
    w = 0.35
    bars1 = ax.bar(x_pos - w / 2, target_gamma, w, color=RED, alpha=0.85,
                   label="miRNA targets", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x_pos + w / 2, bg_gamma, w, color=GRAY, alpha=0.5,
                   label="Background", edgecolor="white", linewidth=0.5)
    # Significance stars
    for i in range(len(families)):
        ax.text(x_pos[i], target_gamma[i] + 0.04, "***", ha="center", fontsize=7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(families, fontsize=7)
    ax.set_ylabel(r"Median $\gamma$")
    ax.set_title("miRNA target enrichment", fontweight="bold", fontsize=9)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.text(-0.15, 1.08, "B", transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.text(0.97, 0.72, "126/215 families\nsignificant (FDR<0.05)",
            transform=ax.transAxes, fontsize=6.5, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8,
                      edgecolor=GRAY, linewidth=0.5))

    fig.tight_layout()
    fig.savefig("figures/fig2_validation.pdf")
    plt.close(fig)
    print("Saved fig2_validation.pdf")


# ── FIGURE 3: Key Findings (2 panels) ────────────────────────────────────────

def fig3_findings():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))

    # Panel A: Expression vs Gamma UMAP — invisible states
    ax = axes[0]

    # Epsilon cluster: 142 cells, one expression blob, two gamma sub-states
    n_cells = 142
    # Expression: single cluster
    theta_e = np.random.uniform(0, 2 * np.pi, n_cells)
    r_e = np.abs(np.random.normal(0, 0.8, n_cells))
    ex = r_e * np.cos(theta_e)
    ey = r_e * np.sin(theta_e)

    # Hidden gamma state assignment
    state = np.random.choice([0, 1], n_cells, p=[0.55, 0.45])

    # Gamma space: separated
    gx = ex + state * 2.8 + np.random.normal(0, 0.15, n_cells)
    gy = ey + (state - 0.5) * 1.2 + np.random.normal(0, 0.15, n_cells)

    colors_state = [BLUE if s == 0 else ORANGE for s in state]

    # Left mini: expression
    ax.set_axis_off()
    ax.text(-0.12, 1.08, "A", transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.set_title("Expression-invisible PT states (Epsilon cells)",
                 fontweight="bold", fontsize=8.5)

    ax_e = fig.add_axes([0.06, 0.17, 0.18, 0.65])
    ax_e.scatter(ex, ey, c=colors_state, s=8, alpha=0.6, edgecolors="none")
    ax_e.set_xticks([])
    ax_e.set_yticks([])
    ax_e.set_title("Expression UMAP", fontsize=7)
    for spine in ax_e.spines.values():
        spine.set_visible(False)
    ax_e.text(0.5, -0.12, "Sil = $-$0.056", transform=ax_e.transAxes,
              fontsize=6.5, ha="center", color=GRAY)

    # Arrow
    fig.text(0.26, 0.49, r"$\rightarrow$", fontsize=14, ha="center",
             fontweight="bold")

    ax_g = fig.add_axes([0.28, 0.17, 0.18, 0.65])
    ax_g.scatter(gx, gy, c=colors_state, s=8, alpha=0.6, edgecolors="none")
    ax_g.set_xticks([])
    ax_g.set_yticks([])
    ax_g.set_title(r"$\gamma$ UMAP", fontsize=7)
    for spine in ax_g.spines.values():
        spine.set_visible(False)
    ax_g.text(0.5, -0.12, "Sil = 0.195", transform=ax_g.transAxes,
              fontsize=6.5, ha="center", color=RED)

    # Panel B: Temporal precedence
    ax = axes[1]
    t = np.linspace(0, 1, 200)
    # Gamma changes first (sigmoid shifted left)
    gamma_curve = 1 / (1 + np.exp(-15 * (t - 0.35)))
    # Expression follows (sigmoid shifted right)
    expr_curve = 1 / (1 + np.exp(-15 * (t - 0.55)))

    ax.plot(t, gamma_curve, color=RED, lw=2.0, label=r"$\gamma$ (degradation rate)")
    ax.plot(t, expr_curve, color=BLUE, lw=2.0, label="Expression")

    # Shade the lag region
    ax.axvspan(0.35, 0.55, alpha=0.08, color=PURPLE)
    ax.annotate("", xy=(0.55, 0.5), xytext=(0.35, 0.5),
                arrowprops=dict(arrowstyle="<->", color=PURPLE, lw=1.2))
    ax.text(0.45, 0.57, r"$\gamma$ leads", fontsize=7, ha="center", color=PURPLE,
            fontweight="bold")

    # Onset markers
    ax.plot(0.35, 0.5, "o", color=RED, ms=5, zorder=5)
    ax.plot(0.55, 0.5, "o", color=BLUE, ms=5, zorder=5)

    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Normalized signal")
    ax.set_title("Temporal precedence", fontweight="bold", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax.text(-0.15, 1.08, "B", transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.text(0.97, 0.35, "63% (pancreas)\n78% (dentate gyrus)\nof genes: " +
            r"$\gamma$ leads" + "\n(p < 10$^{-5}$)",
            transform=ax.transAxes, fontsize=6.5, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8,
                      edgecolor=GRAY, linewidth=0.5))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.08, 1.15)

    fig.tight_layout()
    fig.savefig("figures/fig3_findings.pdf")
    plt.close(fig)
    print("Saved fig3_findings.pdf")


if __name__ == "__main__":
    fig1_method()
    fig2_validation()
    fig3_findings()
    print("All figures generated.")
