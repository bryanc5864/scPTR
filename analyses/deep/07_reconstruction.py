#!/usr/bin/env python
"""Reconstruction quality: how well does DeepPTR fit the observed data?

Evaluates NB log-likelihood on held-out cells, compares reconstruction
error per gene and per cell type.
"""
from _common import *
from scptr.deep._distributions import log_nb_positive

OUT = output_dir("07_reconstruction")


def evaluate_reconstruction(adata_dp, model, history, name, cluster_key="clusters"):
    """Evaluate model fit quality."""
    print(f"\n{'=' * 60}\n{name.upper()}: Reconstruction quality\n{'=' * 60}")

    from scipy.sparse import issparse

    s_np = np.asarray(adata_dp.layers["spliced"]).astype(np.float32)
    u_np = np.asarray(adata_dp.layers["unspliced"]).astype(np.float32)

    s_t = torch.from_numpy(s_np)
    u_t = torch.from_numpy(u_np)
    l_s = torch.from_numpy(s_np.sum(1).clip(1).astype(np.float32))
    l_u = torch.from_numpy(u_np.sum(1).clip(1).astype(np.float32))

    model.eval()
    with torch.no_grad():
        out = model(s_t, u_t, l_s, l_u, kl_weight=1.0)

    # Per-gene reconstruction error
    ll_s = log_nb_positive(s_t, out["mu_s"], out["theta_s"]).numpy()
    ll_u = log_nb_positive(u_t, out["mu_u"], out["theta_u"]).numpy()

    ll_per_gene_s = ll_s.mean(axis=0)
    ll_per_gene_u = ll_u.mean(axis=0)
    ll_per_gene = ll_per_gene_s + ll_per_gene_u

    # Per-cell
    ll_per_cell = ll_s.sum(axis=1) + ll_u.sum(axis=1)

    # Poisson baseline (mu = mean count)
    from scipy.stats import poisson
    s_mean = s_np.mean(0, keepdims=True).clip(1e-8)
    u_mean = u_np.mean(0, keepdims=True).clip(1e-8)
    ll_baseline_s = np.mean(poisson.logpmf(s_np.clip(0, 100).astype(int), s_mean), axis=0)
    ll_baseline_u = np.mean(poisson.logpmf(u_np.clip(0, 100).astype(int), u_mean), axis=0)
    ll_baseline = ll_baseline_s + ll_baseline_u

    improvement = ll_per_gene - ll_baseline
    n_improved = (improvement > 0).sum()

    print(f"  Median per-gene NB log-lik: {np.median(ll_per_gene):.4f}")
    print(f"  Median Poisson baseline:    {np.median(ll_baseline):.4f}")
    print(f"  Genes improved over baseline: {n_improved}/{len(ll_per_gene)}")
    print(f"  Training loss: {history.train_loss[-1]:.2f}, Val loss: {history.val_loss[-1]:.2f}")

    # Per-cell-type reconstruction
    ct_results = []
    if cluster_key in adata_dp.obs.columns:
        print(f"\n  Per-cell-type reconstruction:")
        for ct in sorted(adata_dp.obs[cluster_key].unique()):
            mask = (adata_dp.obs[cluster_key] == ct).values
            if mask.sum() < 5:
                continue
            mean_ll = ll_per_cell[mask].mean()
            ct_results.append({"cell_type": str(ct), "n_cells": int(mask.sum()),
                              "mean_ll": float(mean_ll)})
            print(f"    {ct}: mean LL = {mean_ll:.2f} (n={mask.sum()})")

    results = {
        "median_ll_per_gene": float(np.median(ll_per_gene)),
        "median_ll_baseline": float(np.median(ll_baseline)),
        "n_improved": int(n_improved),
        "n_genes": int(len(ll_per_gene)),
        "final_train_loss": float(history.train_loss[-1]),
        "final_val_loss": float(history.val_loss[-1]),
        "per_celltype": ct_results,
    }

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].scatter(ll_baseline, ll_per_gene, alpha=0.3, s=8, c="steelblue")
    lim = [min(ll_baseline.min(), ll_per_gene.min()), max(ll_baseline.max(), ll_per_gene.max())]
    axes[0].plot(lim, lim, "k--", alpha=0.3)
    axes[0].set_xlabel("Poisson baseline LL")
    axes[0].set_ylabel("DeepPTR NB LL")
    axes[0].set_title(f"{n_improved}/{len(ll_per_gene)} genes improved")

    axes[1].plot(history.train_loss, label="train")
    axes[1].plot(history.val_loss, label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training curve")
    axes[1].legend()

    if ct_results:
        cts = [r["cell_type"] for r in ct_results]
        lls = [r["mean_ll"] for r in ct_results]
        axes[2].barh(cts, lls, color="steelblue", alpha=0.7)
        axes[2].set_xlabel("Mean log-likelihood")
        axes[2].set_title("Per-cell-type fit")

    fig.suptitle(f"{name}: Reconstruction quality", y=1.02)
    fig.tight_layout()
    save_fig(fig, f"{name}_reconstruction", OUT)

    return results


def main():
    set_figure_style()
    all_results = {}
    for name, loader, ck in DATASETS:
        adata_dp, model, history = run_deep(loader)
        all_results[name] = evaluate_reconstruction(adata_dp, model, history, name, ck)
    save_json(all_results, "reconstruction", OUT)


if __name__ == "__main__":
    main()
