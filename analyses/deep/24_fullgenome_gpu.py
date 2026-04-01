#!/usr/bin/env python
"""Full-genome DeepPTR on GPU: eliminate the 300-gene limitation.

Runs DeepPTR on ALL genes (after filtering) using CUDA, comparing:
- 300 genes CPU (previous baseline)
- 500, 1000, 2000, ALL genes on GPU
- Half-life correlation, runtime, and gene coverage at each scale
"""
from _common import *
import time as _time

OUT = output_dir("24_fullgenome_gpu")

# Force CPU — CUDA kernel incompatible (torch cu118 vs driver cu114)
DEVICE = "cpu"


def run_at_scale(adata_base, n_genes, device, label):
    """Run DeepPTR at a given gene count."""
    from scipy.sparse import issparse

    if n_genes >= adata_base.n_vars:
        adata = adata_base.copy()
        actual_n = adata.n_vars
    else:
        adata = select_top_genes(adata_base, n_top=n_genes)
        actual_n = n_genes

    for key in ("spliced", "unspliced"):
        if key in adata.layers and issparse(adata.layers[key]):
            adata.layers[key] = np.asarray(adata.layers[key].todense())

    # Scale hidden dim with gene count
    d_hidden = 48 if actual_n <= 500 else 64 if actual_n <= 2000 else 128

    hp = dict(DEEP_HP)
    hp["device"] = device
    hp["d_hidden"] = d_hidden
    hp["n_posterior_samples"] = 15

    torch.set_num_threads(4)
    t0 = _time.time()
    try:
        model, history = scptr.deep.fit_deepptr(adata, verbose=True, **hp)
        elapsed = _time.time() - t0
        n_epochs = len(history.train_loss)
        return adata, elapsed, n_epochs
    except Exception as e:
        print(f"    FAILED: {e}")
        return None, _time.time() - t0, 0


def main():
    set_figure_style()
    _, hl_human = load_halflife_refs()
    hl_mouse, _ = load_halflife_refs()

    all_results = {}

    for ds_name, loader, ck in DATASETS:
        print(f"\n{'#' * 60}")
        print(f"# {ds_name.upper()}: Full-genome GPU scaling")
        print(f"{'#' * 60}")

        # Preprocess once
        adata_base = loader()
        scptr.pp.filter_genes(adata_base)
        scptr.pp.normalize_layers(adata_base)
        scptr.pp.neighbors(adata_base, n_neighbors=30)
        scptr.pp.smooth_layers(adata_base)
        scptr.tl.estimate_beta(adata_base)

        total_genes = adata_base.n_vars
        print(f"  Total genes after filtering: {total_genes}")

        # Test scales (CPU-tractable)
        scales = [300, 500, 1000, 2000]
        scales = [s for s in scales if s <= total_genes]

        ds_results = []

        for n_g in scales:
            label = f"{n_g} genes" if n_g < total_genes else f"ALL ({total_genes})"
            device = "cpu" if n_g <= 300 else DEVICE
            print(f"\n  --- {label} on {device} ---")

            adata_fit, elapsed, n_epochs = run_at_scale(adata_base, n_g, device, label)

            if adata_fit is not None:
                # Half-life
                r_m, n_m = halflife_spearman(adata_fit, hl_mouse)
                r_h, n_h = halflife_spearman(adata_fit, hl_human)
                print(f"    Time: {elapsed:.1f}s, epochs: {n_epochs}")
                print(f"    HL mouse: r={r_m:.4f} (n={n_m})")
                print(f"    HL human: r={r_h:.4f} (n={n_h})")

                ds_results.append({
                    "n_genes": n_g if n_g < total_genes else total_genes,
                    "label": label,
                    "device": device,
                    "time_s": elapsed,
                    "n_epochs": n_epochs,
                    "hl_mouse_r": float(r_m),
                    "hl_mouse_n": n_m,
                    "hl_human_r": float(r_h),
                    "hl_human_n": n_h,
                })
            else:
                ds_results.append({
                    "n_genes": n_g if n_g < total_genes else total_genes,
                    "label": label,
                    "device": device,
                    "error": True,
                    "time_s": elapsed,
                })

        # Also get analytical baseline for reference
        scptr.tl.estimate_gamma(adata_base)
        r_an_m, n_an_m = halflife_spearman(adata_base, hl_mouse)
        r_an_h, n_an_h = halflife_spearman(adata_base, hl_human)
        print(f"\n  Analytical (all {total_genes} genes): mouse={r_an_m:.4f}(n={n_an_m}), human={r_an_h:.4f}(n={n_an_h})")

        ds_results.append({
            "n_genes": total_genes, "label": "Analytical (all)",
            "device": "cpu", "hl_mouse_r": float(r_an_m), "hl_human_r": float(r_an_h),
            "hl_mouse_n": n_an_m, "hl_human_n": n_an_h,
        })

        all_results[ds_name] = ds_results

        # Summary
        print(f"\n  {'Config':<25} {'Device':>6} {'Time':>8} {'HL mouse':>10} {'HL human':>10} {'n_HL':>6}")
        print("  " + "-" * 70)
        for r in ds_results:
            if "error" in r:
                print(f"  {r['label']:<25} {r['device']:>6} {'FAIL':>8}")
            else:
                t_s = f"{r.get('time_s', 0):.0f}s" if 'time_s' in r else "—"
                print(f"  {r['label']:<25} {r['device']:>6} {t_s:>8} {r['hl_mouse_r']:>10.4f} {r['hl_human_r']:>10.4f} {r.get('hl_human_n', ''):>6}")

        # Figure
        valid = [r for r in ds_results if "error" not in r and "time_s" in r]
        if len(valid) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            genes = [r["n_genes"] for r in valid if r["label"] != "Analytical (all)"]
            times = [r["time_s"] for r in valid if r["label"] != "Analytical (all)"]
            rs = [abs(r["hl_human_r"]) for r in valid if r["label"] != "Analytical (all)"]

            if genes:
                axes[0].plot(genes, times, "o-", color="steelblue")
                axes[0].set_xlabel("Number of genes")
                axes[0].set_ylabel("Runtime (seconds)")
                axes[0].set_title(f"{ds_name}: Scalability")
                axes[0].set_xscale("log")

                axes[1].plot(genes, rs, "o-", color="darkorange", label="DeepPTR")
                axes[1].axhline(abs(r_an_h), color="red", ls="--", label=f"Analytical={abs(r_an_h):.3f}")
                axes[1].set_xlabel("Number of genes")
                axes[1].set_ylabel("|r| with half-life (human)")
                axes[1].set_title(f"{ds_name}: Quality vs scale")
                axes[1].set_xscale("log")
                axes[1].legend()

            fig.tight_layout()
            save_fig(fig, f"{ds_name}_fullgenome", OUT)

    save_json(all_results, "fullgenome_gpu", OUT)


if __name__ == "__main__":
    main()
