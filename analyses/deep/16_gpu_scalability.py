#!/usr/bin/env python
"""GPU scalability: full-genome DeepPTR with CUDA.

Demonstrates that DeepPTR scales to full gene sets when GPU is available,
comparing runtime and quality vs the 300-gene CPU subset.
"""
from _common import *

OUT = output_dir("16_gpu_scalability")


def main():
    set_figure_style()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("  [WARN] No GPU available. Running reduced comparison.")

    # Load and preprocess
    adata_raw = scptr.datasets.pancreas()
    scptr.pp.filter_genes(adata_raw)
    scptr.pp.normalize_layers(adata_raw)
    scptr.pp.neighbors(adata_raw, n_neighbors=30)
    scptr.pp.smooth_layers(adata_raw)
    scptr.tl.estimate_beta(adata_raw)

    _, hl_human = load_halflife_refs()
    results = {}

    # ── CPU 300 genes (baseline) ──────────────────────────────────────
    print(f"\n{'=' * 60}\nCPU: 300 genes\n{'=' * 60}")
    adata_300 = select_top_genes(adata_raw, n_top=300)
    from scipy.sparse import issparse
    for key in ("spliced", "unspliced"):
        if key in adata_300.layers and issparse(adata_300.layers[key]):
            adata_300.layers[key] = np.asarray(adata_300.layers[key].todense())

    torch.set_num_threads(4)
    t0 = _time.time() if 'time' not in dir() else __import__('time').time()
    import time as _time
    t0 = _time.time()
    scptr.deep.fit_deepptr(adata_300, device="cpu", verbose=True, **DEEP_HP)
    t_cpu_300 = _time.time() - t0

    r_300, n_300 = halflife_spearman(adata_300, hl_human)
    print(f"  Time: {t_cpu_300:.1f}s, HL r={r_300:.4f} (n={n_300})")
    results["cpu_300"] = {"time": t_cpu_300, "r": r_300, "n_genes": 300, "n_hl": n_300}

    # ── GPU scaling experiments ───────────────────────────────────────
    gene_counts = [500, 1000, 2000]
    if device == "cpu":
        gene_counts = [500]  # Reduced for CPU-only

    for n_genes in gene_counts:
        if n_genes > adata_raw.n_vars:
            continue
        label = f"{device}_{n_genes}"
        print(f"\n{'=' * 60}\n{device.upper()}: {n_genes} genes\n{'=' * 60}")

        adata_n = select_top_genes(adata_raw, n_top=n_genes)
        for key in ("spliced", "unspliced"):
            if key in adata_n.layers and issparse(adata_n.layers[key]):
                adata_n.layers[key] = np.asarray(adata_n.layers[key].todense())

        hp = dict(DEEP_HP)
        hp["device"] = device
        if n_genes > 1000:
            hp["d_hidden"] = 64  # Scale up for more genes

        torch.set_num_threads(4)
        t0 = _time.time()
        try:
            scptr.deep.fit_deepptr(adata_n, verbose=True, **hp)
            elapsed = _time.time() - t0
            r_n, n_n = halflife_spearman(adata_n, hl_human)
            print(f"  Time: {elapsed:.1f}s, HL r={r_n:.4f} (n={n_n})")
            results[label] = {"time": elapsed, "r": r_n, "n_genes": n_genes, "n_hl": n_n}
        except Exception as e:
            print(f"  FAILED: {e}")
            results[label] = {"error": str(e), "n_genes": n_genes}

    # ── Full genome attempt ───────────────────────────────────────────
    if device == "cuda":
        n_full = adata_raw.n_vars
        print(f"\n{'=' * 60}\nGPU: Full genome ({n_full} genes)\n{'=' * 60}")

        adata_full = adata_raw.copy()
        for key in ("spliced", "unspliced"):
            if key in adata_full.layers and issparse(adata_full.layers[key]):
                adata_full.layers[key] = np.asarray(adata_full.layers[key].todense())

        hp = dict(DEEP_HP)
        hp["device"] = "cuda"
        hp["d_hidden"] = 128
        hp["batch_size"] = 256

        t0 = _time.time()
        try:
            scptr.deep.fit_deepptr(adata_full, verbose=True, **hp)
            elapsed = _time.time() - t0
            r_full, n_full_hl = halflife_spearman(adata_full, hl_human)
            print(f"  Time: {elapsed:.1f}s, HL r={r_full:.4f} (n={n_full_hl})")
            results[f"gpu_full_{n_full}"] = {
                "time": elapsed, "r": r_full, "n_genes": n_full, "n_hl": n_full_hl
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            results[f"gpu_full_{n_full}"] = {"error": str(e), "n_genes": n_full}

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SCALABILITY SUMMARY")
    print("=" * 60)
    print(f"  {'Config':<25} {'Genes':>8} {'Time':>10} {'HL r':>10} {'HL n':>8}")
    for label, d in results.items():
        if "error" in d:
            print(f"  {label:<25} {d['n_genes']:>8} {'FAIL':>10}")
        else:
            print(f"  {label:<25} {d['n_genes']:>8} {d['time']:>9.1f}s {d['r']:>10.4f} {d['n_hl']:>8}")

    save_json(results, "gpu_scalability", OUT)

    # Figure
    configs = [k for k in results if "error" not in results[k]]
    if len(configs) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        genes = [results[k]["n_genes"] for k in configs]
        times = [results[k]["time"] for k in configs]
        rs = [abs(results[k]["r"]) for k in configs]

        axes[0].plot(genes, times, "o-", color="steelblue")
        axes[0].set_xlabel("Number of genes")
        axes[0].set_ylabel("Runtime (seconds)")
        axes[0].set_title("Scalability")

        axes[1].plot(genes, rs, "o-", color="darkorange")
        axes[1].set_xlabel("Number of genes")
        axes[1].set_ylabel("|r| with half-life")
        axes[1].set_title("Quality vs gene count")

        fig.tight_layout()
        save_fig(fig, "gpu_scalability", OUT)


if __name__ == "__main__":
    main()
