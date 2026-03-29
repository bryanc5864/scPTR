#!/usr/bin/env python
"""Half-life ceiling analysis: is r=-0.40 near the theoretical maximum?

Compares published half-life datasets against EACH OTHER to establish
inter-study agreement. If Herzog and Schofield agree at r=0.5-0.6,
then r=0.40 is 67-80% of the ceiling — near-optimal.
"""
from _common import *

OUT = output_dir("18_halflife_ceiling")


def main():
    set_figure_style()

    hl_mouse, hl_human = load_halflife_refs()

    # ── Inter-study agreement: mouse vs human half-lives ─────────────
    print("=" * 60)
    print("INTER-STUDY HALF-LIFE AGREEMENT")
    print("=" * 60)

    hl_m = hl_mouse.set_index("gene_symbol")["half_life_hours"]
    hl_h = hl_human.set_index("gene_symbol")["half_life_hours"]

    # Case-insensitive match
    m_upper = {g.upper(): g for g in hl_m.index if isinstance(g, str)}
    h_upper = {g.upper(): g for g in hl_h.index if isinstance(g, str)}
    shared = set(m_upper.keys()) & set(h_upper.keys())

    m_vals = np.array([hl_m[m_upper[u]] for u in shared], dtype=float)
    h_vals = np.array([hl_h[h_upper[u]] for u in shared], dtype=float)
    valid = np.isfinite(m_vals) & np.isfinite(h_vals) & (m_vals > 0) & (h_vals > 0)
    m_vals, h_vals = m_vals[valid], h_vals[valid]

    r_inter, p_inter = stats.spearmanr(m_vals, h_vals)
    r_inter_log, p_inter_log = stats.pearsonr(np.log(m_vals), np.log(h_vals))

    print(f"\n  Herzog (mouse) vs Schofield (human):")
    print(f"    Shared genes: {len(m_vals)}")
    print(f"    Spearman r = {r_inter:.4f} (p={p_inter:.2e})")
    print(f"    Pearson r (log) = {r_inter_log:.4f}")

    # ── Ceiling fractions ─────────────────────────────────────────────
    print(f"\n  Ceiling analysis:")
    print(f"    Inter-study agreement (ceiling): |r| = {abs(r_inter):.4f}")

    methods = {
        "scPTR analytical (pancreas, human)": -0.4021,
        "scPTR analytical (DG, human)": -0.3812,
        "scVelo SS (pancreas, human)": -0.3730,
        "scVelo SS (DG, human)": -0.3675,
        "velVI (pancreas, human)": -0.2783,
        "velVI (DG, human)": -0.3522,
        "DeepPTR (pancreas, human)": -0.2767,
        "DeepPTR (DG, human)": -0.3577,
    }

    print(f"\n    {'Method':<45} {'|r|':>6} {'% ceiling':>10}")
    print("    " + "-" * 65)
    for name, r in sorted(methods.items(), key=lambda x: abs(x[1]), reverse=True):
        pct = abs(r) / abs(r_inter) * 100
        print(f"    {name:<45} {abs(r):.4f} {pct:>9.1f}%")

    # ── Bootstrap CI on inter-study agreement ─────────────────────────
    rng = np.random.RandomState(42)
    boot_rs = []
    for _ in range(1000):
        idx = rng.choice(len(m_vals), len(m_vals), replace=True)
        boot_rs.append(stats.spearmanr(m_vals[idx], h_vals[idx]).statistic)
    ci_lo, ci_hi = np.percentile(boot_rs, [2.5, 97.5])

    print(f"\n  Inter-study 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    # ── Split-half reliability of gamma itself ─────────────────────────
    print(f"\n--- Split-half reliability of scPTR gamma ---")
    for ds_name, loader, ck in DATASETS:
        adata = run_analytical(loader)
        gamma = adata.layers["gamma"]

        rng = np.random.RandomState(0)
        n = adata.n_obs
        perm = rng.permutation(n)
        half1 = perm[:n // 2]
        half2 = perm[n // 2:]

        med1 = np.median(gamma[half1], axis=0)
        med2 = np.median(gamma[half2], axis=0)

        valid = (med1 > 0) & (med2 > 0) & np.isfinite(med1) & np.isfinite(med2)
        r_split, _ = stats.spearmanr(med1[valid], med2[valid])

        # Spearman-Brown correction for full reliability
        r_full = 2 * r_split / (1 + r_split)

        print(f"  {ds_name}: split-half r={r_split:.4f}, Spearman-Brown corrected={r_full:.4f}")

    results = {
        "inter_study_r": float(r_inter),
        "inter_study_r_log": float(r_inter_log),
        "inter_study_n": int(len(m_vals)),
        "inter_study_ci": [float(ci_lo), float(ci_hi)],
        "method_ceiling_pct": {k: abs(v) / abs(r_inter) * 100 for k, v in methods.items()},
    }
    save_json(results, "halflife_ceiling", OUT)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: inter-study scatter
    axes[0].scatter(np.log10(m_vals), np.log10(h_vals), alpha=0.1, s=5, c="gray")
    axes[0].set_xlabel("log10(Herzog mouse half-life)")
    axes[0].set_ylabel("log10(Schofield human half-life)")
    axes[0].set_title(f"Inter-study agreement (r={r_inter:.3f}, n={len(m_vals)})")

    # Panel 2: ceiling fraction bar chart
    names_short = [n.split("(")[0].strip() for n in methods]
    pcts = [abs(v) / abs(r_inter) * 100 for v in methods.values()]
    colors = ["darkorange" if "scPTR" in n else "steelblue" if "scVelo" in n
              else "seagreen" if "velVI" in n else "gray" for n in methods]
    axes[1].barh(range(len(methods)), pcts, color=colors, alpha=0.7)
    axes[1].set_yticks(range(len(methods)))
    axes[1].set_yticklabels(list(methods.keys()), fontsize=7)
    axes[1].set_xlabel("% of inter-study ceiling")
    axes[1].set_title("Method performance relative to ceiling")
    axes[1].axvline(100, color="red", ls="--", alpha=0.3, label="Ceiling")
    axes[1].legend()

    fig.tight_layout()
    save_fig(fig, "halflife_ceiling", OUT)


if __name__ == "__main__":
    main()
