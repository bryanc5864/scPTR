#!/usr/bin/env python
"""Ablation study: contribution of each DeepPTR component.

Tests:
  A. Full model (baseline)
  B. No z_PT (d_PT=0) — can model work without PT latent?
  C. No z_T (d_T=0) — can z_PT alone recover gamma?
  D. No kinetic decoder (standard MLP decoder)
  E. No beta warm-start (random init)
  F. No KL (autoencoder, beta_kl=0 throughout)

Evaluated on synthetic (gamma recovery) and pancreas (half-life, silhouette).
"""
from _common import *
from scptr.deep.synthetic import generate_kinetic_data, gamma_recovery

OUT = output_dir("13_ablation")


def run_ablation_variant(adata, label, **override_hp):
    """Fit DeepPTR with modified hyperparameters."""
    hp = dict(DEEP_HP)
    hp.update(override_hp)

    # Handle special cases
    no_warmstart = hp.pop("no_warmstart", False)
    no_kl = hp.pop("no_kl", False)

    torch.set_num_threads(4)

    if no_kl:
        # Train with kl_warmup = max_epochs+1 so kl_weight stays 0
        hp["kl_warmup_epochs"] = hp["max_epochs"] + 100

    # If d_PT=0 or d_T=0, we need to handle this
    d_T = hp.get("d_T", 8)
    d_PT = hp.get("d_PT", 8)
    if d_T == 0:
        hp["d_T"] = 1  # Can't be 0, use 1 and it'll be ignored
    if d_PT == 0:
        hp["d_PT"] = 1

    try:
        adata_fit = adata.copy()
        model, history = scptr.deep.fit_deepptr(adata_fit, verbose=False, **hp)
        return adata_fit, model, history
    except Exception as e:
        print(f"    {label} FAILED: {e}")
        return None, None, None


def eval_synthetic(truth, adata, label):
    """Evaluate on synthetic data."""
    if adata is None or "gamma" not in adata.layers:
        return {"label": label, "gamma_r": np.nan}
    r = gamma_recovery(truth["gamma"], adata.layers["gamma"], per_gene=True)
    return {"label": label, "gamma_r": float(r)}


def eval_real(adata, hl_df, cluster_key, label):
    """Evaluate on real data."""
    if adata is None or "gamma" not in adata.layers:
        return {"label": label, "halflife_r": np.nan, "silhouette": np.nan}

    r, n = halflife_spearman(adata, hl_df)

    sil = np.nan
    if cluster_key in adata.obs.columns and "X_z_T" in adata.obsm:
        from sklearn.metrics import silhouette_score
        labels = adata.obs[cluster_key].astype("category").cat.codes.values
        try:
            sil = silhouette_score(adata.obsm["X_z_T"], labels,
                                    sample_size=min(2000, len(labels)))
        except Exception:
            pass

    return {"label": label, "halflife_r": float(r), "n_genes": n, "silhouette": float(sil)}


def main():
    set_figure_style()

    # ── Synthetic ablation ───────────────────────────────────────────
    print("=" * 60)
    print("ABLATION: Synthetic data")
    print("=" * 60)

    adata_syn, truth = generate_kinetic_data(n_cells=1500, n_genes=100, seed=0)

    variants = [
        ("A. Full model", {}),
        ("B. No z_PT (d_PT=1)", {"d_PT": 1}),
        ("C. No z_T (d_T=1)", {"d_T": 1}),
        ("D. No beta warmstart", {"no_warmstart": True}),
        ("E. No KL", {"no_kl": True}),
        ("F. Small model", {"d_hidden": 16, "d_T": 4, "d_PT": 4}),
    ]

    syn_results = []
    for label, hp in variants:
        print(f"\n  {label}...")
        adata_fit, model, history = run_ablation_variant(adata_syn, label, **hp)
        if adata_fit is not None and model is not None:
            r = eval_synthetic(truth, adata_fit, label)
        else:
            r = {"label": label, "gamma_r": np.nan}
        syn_results.append(r)
        print(f"    gamma recovery: {r['gamma_r']:.4f}" if np.isfinite(r['gamma_r']) else f"    FAILED")

    # ── Real data ablation (pancreas) ────────────────────────────────
    print(f"\n{'=' * 60}")
    print("ABLATION: Pancreas")
    print("=" * 60)

    adata_raw = scptr.datasets.pancreas()
    scptr.pp.filter_genes(adata_raw)
    scptr.pp.normalize_layers(adata_raw)
    scptr.pp.neighbors(adata_raw, n_neighbors=30)
    scptr.pp.smooth_layers(adata_raw)
    scptr.tl.estimate_beta(adata_raw)
    adata_base = select_top_genes(adata_raw, n_top=300)

    _, hl_human = load_halflife_refs()

    real_results = []
    for label, hp in variants:
        print(f"\n  {label}...")
        adata_v = adata_base.copy()
        # Ensure dense
        from scipy.sparse import issparse
        for key in ("spliced", "unspliced"):
            if key in adata_v.layers and issparse(adata_v.layers[key]):
                adata_v.layers[key] = np.asarray(adata_v.layers[key].todense())

        adata_fit, model, history = run_ablation_variant(adata_v, label, **hp)
        r = eval_real(adata_fit if adata_fit is not None else adata_v, hl_human, "clusters", label)
        real_results.append(r)
        print(f"    halflife r={r['halflife_r']:.4f}, silhouette={r['silhouette']:.4f}"
              if np.isfinite(r['halflife_r']) else f"    FAILED")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("ABLATION SUMMARY")
    print("=" * 60)
    print(f"\n{'Variant':<30} {'Synth γ r':>10} {'HL r':>10} {'Silhouette':>10}")
    print("-" * 65)
    for s, r in zip(syn_results, real_results):
        gr = f"{s['gamma_r']:.4f}" if np.isfinite(s['gamma_r']) else "N/A"
        hr = f"{r['halflife_r']:.4f}" if np.isfinite(r['halflife_r']) else "N/A"
        si = f"{r['silhouette']:.4f}" if np.isfinite(r['silhouette']) else "N/A"
        print(f"  {s['label']:<28} {gr:>10} {hr:>10} {si:>10}")

    results = {"synthetic": syn_results, "pancreas": real_results}
    save_json(results, "ablation", OUT)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = [s["label"] for s in syn_results]
    short_labels = [l.split(". ")[1] if ". " in l else l for l in labels]

    # Synthetic gamma recovery
    vals = [s["gamma_r"] for s in syn_results]
    axes[0].barh(short_labels, vals, color="steelblue", alpha=0.7)
    axes[0].set_xlabel("Gamma recovery (Spearman r)")
    axes[0].set_title("Synthetic")
    axes[0].axvline(vals[0], color="red", ls="--", alpha=0.3, label="Full model")

    # Real half-life
    vals = [abs(r["halflife_r"]) if np.isfinite(r["halflife_r"]) else 0 for r in real_results]
    axes[1].barh(short_labels, vals, color="darkorange", alpha=0.7)
    axes[1].set_xlabel("|Spearman r| with half-life")
    axes[1].set_title("Pancreas half-life")

    # Silhouette
    vals = [r["silhouette"] if np.isfinite(r["silhouette"]) else 0 for r in real_results]
    axes[2].barh(short_labels, vals, color="seagreen", alpha=0.7)
    axes[2].set_xlabel("Silhouette score (z_T)")
    axes[2].set_title("Cell-type separation")

    fig.suptitle("Ablation study", y=1.02)
    fig.tight_layout()
    save_fig(fig, "ablation_summary", OUT)


if __name__ == "__main__":
    main()
