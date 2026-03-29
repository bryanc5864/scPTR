#!/usr/bin/env python
"""Head-to-head comparison: scPTR vs scVelo vs velVI.

Compares degradation rate / kinetic parameter estimates across methods
using the same evaluation framework:
  1. Half-life correlation (mouse + human references)
  2. Cell-type discrimination (silhouette, ANOVA F-stat)
  3. Latent quality (pseudotime correlation, cell-type silhouette)
  4. Runtime

Datasets: pancreas, dentate gyrus.
"""
from _common import *
import scanpy as sc
import scvelo as scv
import time as _time
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

OUT = output_dir("12_method_comparison")


# ── scVelo steady-state ────────────────────────────────────────────────────

def run_scvelo_steady(adata_raw, name):
    """Run scVelo steady-state mode, extract gamma-like parameter."""
    print(f"\n  scVelo steady-state ({name})...")
    adata = adata_raw.copy()
    t0 = _time.time()

    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(adata, mode="steady_state")
    scv.tl.velocity_graph(adata)

    elapsed = _time.time() - t0

    # scVelo stores velocity_gamma in adata.var
    # gamma_ss = fit parameter from u = gamma * s + offset
    gamma = adata.var.get("velocity_gamma", pd.Series(dtype=float))

    # For cell-type discrimination, use velocity PCA
    if "velocity" in adata.layers:
        vel = np.asarray(adata.layers["velocity"])
        vel = np.nan_to_num(vel, 0)
        vel_pca = PCA(n_components=min(10, vel.shape[1])).fit_transform(vel)
    else:
        vel_pca = None

    print(f"    Done in {elapsed:.1f}s, {len(gamma)} genes, {adata.n_vars} after filter")
    return adata, gamma, vel_pca, elapsed


# ── scVelo dynamical ──────────────────────────────────────────────────────

def run_scvelo_dynamical(adata_raw, name):
    """Run scVelo dynamical mode, extract gamma and latent time."""
    print(f"\n  scVelo dynamical ({name})...")
    adata = adata_raw.copy()
    t0 = _time.time()

    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.recover_dynamics(adata, n_jobs=4)
    scv.tl.velocity(adata, mode="dynamical")
    scv.tl.velocity_graph(adata)
    scv.tl.latent_time(adata)

    elapsed = _time.time() - t0

    gamma = adata.var.get("fit_gamma", pd.Series(dtype=float))
    latent_time = adata.obs.get("latent_time", pd.Series(dtype=float))

    if "velocity" in adata.layers:
        vel = np.asarray(adata.layers["velocity"])
        vel = np.nan_to_num(vel, 0)
        vel_pca = PCA(n_components=min(10, vel.shape[1])).fit_transform(vel)
    else:
        vel_pca = None

    print(f"    Done in {elapsed:.1f}s, {len(gamma)} genes")
    return adata, gamma, latent_time, vel_pca, elapsed


# ── velVI ─────────────────────────────────────────────────────────────────

def run_velovi(adata_raw, name):
    """Run velVI (VELOVI from scvi-tools)."""
    print(f"\n  velVI ({name})...")
    from scvi.external import VELOVI

    adata = adata_raw.copy()
    t0 = _time.time()

    # Preprocessing (velVI needs spliced/unspliced in layers + scVelo moments)
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=1000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

    # Setup and train (CPU — CUDA kernel incompatible on this system)
    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    model = VELOVI(adata, n_latent=10, n_hidden=128, n_layers=1)
    model.train(max_epochs=50, early_stopping=True, early_stopping_patience=10,
                accelerator="cpu")

    elapsed = _time.time() - t0

    # Extract latent representation
    latent = model.get_latent_representation()
    adata.obsm["X_velovi"] = latent

    # Extract per-gene parameters
    # velVI stores gamma-like parameters internally
    # Get velocity outputs
    try:
        outputs = model.get_velocity()
        if "velocity" not in adata.layers:
            adata.layers["velocity"] = outputs
    except Exception:
        pass

    # velVI gamma from the model
    gamma_vals = None
    try:
        gamma_vals = model.get_rates()
        if isinstance(gamma_vals, dict) and "gamma" in gamma_vals:
            gamma_vals = gamma_vals["gamma"]
    except Exception:
        pass

    vel_pca = None
    if "velocity" in adata.layers:
        vel = np.asarray(adata.layers["velocity"])
        vel = np.nan_to_num(vel, 0)
        vel_pca = PCA(n_components=min(10, vel.shape[1])).fit_transform(vel)

    print(f"    Done in {elapsed:.1f}s")
    return adata, gamma_vals, latent, vel_pca, elapsed


# ── scPTR analytical ──────────────────────────────────────────────────────

def run_scptr_analytical(adata_raw, name):
    """Run scPTR analytical pipeline."""
    print(f"\n  scPTR analytical ({name})...")
    adata = adata_raw.copy()
    t0 = _time.time()

    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    scptr.tl.estimate_gamma(adata)

    elapsed = _time.time() - t0

    gamma_med = np.median(adata.layers["gamma"], axis=0)
    gamma_s = pd.Series(gamma_med, index=adata.var_names)

    # Gamma PCA for cell-type discrimination
    gamma_pca = PCA(n_components=min(10, adata.layers["gamma"].shape[1])).fit_transform(
        adata.layers["gamma"]
    )

    print(f"    Done in {elapsed:.1f}s, {adata.n_vars} genes")
    return adata, gamma_s, gamma_pca, elapsed


# ── scPTR DeepPTR ─────────────────────────────────────────────────────────

def run_scptr_deep(adata_raw, name):
    """Run DeepPTR."""
    print(f"\n  DeepPTR ({name})...")
    adata = adata_raw.copy()
    t0 = _time.time()

    scptr.pp.filter_genes(adata)
    scptr.pp.normalize_layers(adata)
    scptr.pp.neighbors(adata, n_neighbors=30)
    scptr.pp.smooth_layers(adata)
    scptr.tl.estimate_beta(adata)
    adata = select_top_genes(adata, n_top=300)

    torch.set_num_threads(4)
    model, history = scptr.deep.fit_deepptr(adata, verbose=False, **DEEP_HP)

    elapsed = _time.time() - t0

    gamma_med = np.median(adata.layers["gamma"], axis=0)
    gamma_s = pd.Series(gamma_med, index=adata.var_names)

    print(f"    Done in {elapsed:.1f}s, {adata.n_vars} genes")
    return adata, gamma_s, adata.obsm.get("X_z_T"), elapsed


# ── Evaluation ────────────────────────────────────────────────────────────

def eval_halflife(gamma_series_or_adata, hl_df, method_name):
    """Evaluate half-life correlation for any method's gamma estimates."""
    if isinstance(gamma_series_or_adata, pd.Series):
        gamma_s = gamma_series_or_adata
    else:
        # It's an adata with gamma layer
        adata = gamma_series_or_adata
        gamma_med = np.median(adata.layers["gamma"], axis=0)
        gamma_s = pd.Series(gamma_med, index=adata.var_names)

    hl_s = hl_df.set_index("gene_symbol")["half_life_hours"]

    # Case-insensitive match
    gamma_upper = {g.upper(): g for g in gamma_s.index}
    hl_upper = {g.upper(): g for g in hl_s.index if isinstance(g, str)}
    shared = set(gamma_upper.keys()) & set(hl_upper.keys())

    g = np.array([gamma_s[gamma_upper[u]] for u in shared], dtype=float)
    h = np.array([hl_s[hl_upper[u]] for u in shared], dtype=float)

    valid = np.isfinite(g) & np.isfinite(h) & (g > 0) & (h > 0)
    if valid.sum() < 3:
        return np.nan, 0
    r, _ = stats.spearmanr(g[valid], h[valid])
    return float(r), int(valid.sum())


def eval_celltype(latent_or_pca, labels, method_name):
    """Evaluate cell-type discrimination in latent/kinetic space."""
    if latent_or_pca is None:
        return np.nan
    labels_arr = np.asarray(labels)
    if isinstance(labels_arr[0], str):
        from sklearn.preprocessing import LabelEncoder
        labels_arr = LabelEncoder().fit_transform(labels_arr)

    n_sample = min(2000, len(labels_arr))
    try:
        return float(silhouette_score(latent_or_pca, labels_arr, sample_size=n_sample))
    except Exception:
        return np.nan


def eval_anova_fstat(gamma_matrix, labels, gene_names):
    """Median F-statistic across genes for cell-type gamma differences."""
    from scipy.stats import f_oneway
    cts = sorted(set(labels))
    f_stats = []
    for g in range(gamma_matrix.shape[1]):
        groups = [gamma_matrix[labels == ct, g] for ct in cts if (labels == ct).sum() >= 5]
        if len(groups) < 2:
            continue
        try:
            f, p = f_oneway(*groups)
            if np.isfinite(f):
                f_stats.append(f)
        except Exception:
            pass
    return float(np.median(f_stats)) if f_stats else np.nan


# ── Main comparison ───────────────────────────────────────────────────────

def run_comparison(name, loader, cluster_key):
    print(f"\n{'#' * 60}")
    print(f"# {name.upper()}: METHOD COMPARISON")
    print(f"{'#' * 60}")

    adata_raw = loader()
    hl_mouse, hl_human = load_halflife_refs()
    labels = adata_raw.obs.get(cluster_key, pd.Series(dtype=str)).values

    results = {}

    # ── Run each method ───────────────────────────────────────────────
    # 1. scVelo steady-state
    try:
        sv_ss_adata, sv_ss_gamma, sv_ss_pca, sv_ss_time = run_scvelo_steady(adata_raw, name)
        results["scvelo_ss"] = {"time": sv_ss_time, "n_genes": len(sv_ss_gamma)}
    except Exception as e:
        print(f"  scVelo SS failed: {e}")
        sv_ss_gamma, sv_ss_pca = pd.Series(dtype=float), None
        results["scvelo_ss"] = {"error": str(e)}

    # 2. scVelo dynamical
    try:
        sv_dyn_adata, sv_dyn_gamma, sv_dyn_ltime, sv_dyn_pca, sv_dyn_time = run_scvelo_dynamical(adata_raw, name)
        results["scvelo_dyn"] = {"time": sv_dyn_time, "n_genes": len(sv_dyn_gamma)}
    except Exception as e:
        print(f"  scVelo dyn failed: {e}")
        sv_dyn_gamma, sv_dyn_ltime, sv_dyn_pca = pd.Series(dtype=float), None, None
        results["scvelo_dyn"] = {"error": str(e)}

    # 3. velVI
    try:
        vi_adata, vi_gamma, vi_latent, vi_pca, vi_time = run_velovi(adata_raw, name)
        results["velovi"] = {"time": vi_time}
    except Exception as e:
        print(f"  velVI failed: {e}")
        vi_gamma, vi_latent, vi_pca = None, None, None
        results["velovi"] = {"error": str(e)}

    # 4. scPTR analytical
    sp_adata, sp_gamma, sp_pca, sp_time = run_scptr_analytical(adata_raw, name)
    results["scptr_analytical"] = {"time": sp_time, "n_genes": len(sp_gamma)}

    # 5. DeepPTR
    dp_adata, dp_gamma, dp_latent, dp_time = run_scptr_deep(adata_raw, name)
    results["deepptr"] = {"time": dp_time, "n_genes": len(dp_gamma)}

    # ── Evaluate: half-life correlations ─────────────────────────────
    print(f"\n--- Half-life correlations ---")
    methods_gamma = {
        "scVelo SS": sv_ss_gamma,
        "scVelo dyn": sv_dyn_gamma,
        "scPTR analytical": sp_gamma,
        "DeepPTR": dp_gamma,
    }

    for ref_name, hl_df in [("mouse", hl_mouse), ("human", hl_human)]:
        print(f"\n  {ref_name}:")
        for mname, gamma in methods_gamma.items():
            if gamma is not None and len(gamma) > 0:
                r, n = eval_halflife(gamma, hl_df, mname)
                results.setdefault(mname.lower().replace(" ", "_"), {})
                results[mname.lower().replace(" ", "_")][f"halflife_{ref_name}"] = {"r": r, "n": n}
                print(f"    {mname:25s}: r={r:.4f} (n={n})")
            else:
                print(f"    {mname:25s}: N/A")

        # velVI (if gamma available as pd.Series or array)
        if vi_gamma is not None:
            if isinstance(vi_gamma, np.ndarray) and vi_gamma.ndim == 1:
                # Try to match with velVI adata gene names
                try:
                    vi_gs = pd.Series(vi_gamma, index=vi_adata.var_names)
                    r, n = eval_halflife(vi_gs, hl_df, "velVI")
                except Exception:
                    r, n = np.nan, 0
            elif isinstance(vi_gamma, pd.Series):
                r, n = eval_halflife(vi_gamma, hl_df, "velVI")
            else:
                r, n = np.nan, 0
            results.setdefault("velovi", {})
            results["velovi"][f"halflife_{ref_name}"] = {"r": r, "n": n}
            print(f"    {'velVI':25s}: r={r:.4f} (n={n})")

    # ── Evaluate: cell-type discrimination ────────────────────────────
    print(f"\n--- Cell-type discrimination (silhouette) ---")
    if cluster_key in adata_raw.obs.columns:
        spaces = {
            "scVelo SS velocity": sv_ss_pca,
            "scVelo dyn velocity": sv_dyn_pca,
            "velVI latent": vi_latent,
            "velVI velocity": vi_pca,
            "scPTR gamma": sp_pca,
            "DeepPTR z_T": dp_latent,
        }
        for sname, space in spaces.items():
            if space is not None:
                # Need matching labels
                if "scVelo" in sname:
                    if "SS" in sname:
                        lbl = sv_ss_adata.obs.get(cluster_key, pd.Series(dtype=str)).values
                    else:
                        lbl = sv_dyn_adata.obs.get(cluster_key, pd.Series(dtype=str)).values
                elif "velVI" in sname:
                    lbl = vi_adata.obs.get(cluster_key, pd.Series(dtype=str)).values
                elif "scPTR" in sname:
                    lbl = sp_adata.obs.get(cluster_key, pd.Series(dtype=str)).values
                else:
                    lbl = dp_adata.obs.get(cluster_key, pd.Series(dtype=str)).values

                sil = eval_celltype(space, lbl, sname)
                key = sname.lower().replace(" ", "_")
                results[key] = results.get(key, {})
                results[key]["silhouette"] = sil
                print(f"    {sname:25s}: {sil:.4f}")

    # ── Evaluate: ANOVA F-stat on scPTR gamma ─────────────────────────
    print(f"\n--- Gamma cell-type F-statistic ---")
    if cluster_key in sp_adata.obs.columns:
        f_an = eval_anova_fstat(sp_adata.layers["gamma"],
                                sp_adata.obs[cluster_key].values,
                                sp_adata.var_names)
        f_dp = eval_anova_fstat(dp_adata.layers["gamma"],
                                dp_adata.obs[cluster_key].values,
                                dp_adata.var_names)
        print(f"    scPTR analytical: median F = {f_an:.2f}")
        print(f"    DeepPTR:          median F = {f_dp:.2f}")
        results["gamma_fstat"] = {"analytical": f_an, "deepptr": f_dp}

    # ── Runtime comparison ────────────────────────────────────────────
    print(f"\n--- Runtime ---")
    runtimes = {
        "scVelo SS": results.get("scvelo_ss", {}).get("time", np.nan),
        "scVelo dyn": results.get("scvelo_dyn", {}).get("time", np.nan),
        "velVI": results.get("velovi", {}).get("time", np.nan),
        "scPTR analytical": results.get("scptr_analytical", {}).get("time", np.nan),
        "DeepPTR": results.get("deepptr", {}).get("time", np.nan),
    }
    for mname, t in runtimes.items():
        print(f"    {mname:25s}: {t:.1f}s" if np.isfinite(t) else f"    {mname:25s}: N/A")

    save_json(results, f"{name}_comparison", OUT)

    # ── Summary figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Half-life correlation
    methods = ["scVelo SS", "scVelo dyn", "scPTR analytical", "DeepPTR"]
    method_keys = ["scvelo_ss", "scvelo_dyn", "scptr_analytical", "deepptr"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    for ref_idx, ref_name in enumerate(["mouse", "human"]):
        x_offset = ref_idx * 0.4
        for i, (mname, mkey) in enumerate(zip(methods, method_keys)):
            r = results.get(mkey, {}).get(f"halflife_{ref_name}", {}).get("r", np.nan)
            if np.isfinite(r):
                axes[0].bar(i + x_offset, abs(r), 0.35, color=colors[i],
                           alpha=0.7 if ref_idx == 0 else 0.4)

    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("|Spearman r| with half-life")
    axes[0].set_title(f"{name}: Half-life correlation")

    # Panel 2: Runtime
    valid_runtimes = {k: v for k, v in runtimes.items() if np.isfinite(v)}
    if valid_runtimes:
        axes[1].barh(list(valid_runtimes.keys()), list(valid_runtimes.values()),
                     color=colors[:len(valid_runtimes)], alpha=0.7)
        axes[1].set_xlabel("Runtime (seconds)")
        axes[1].set_title("Runtime comparison")

    # Panel 3: Silhouette scores
    sil_data = {}
    for sname in ["scVelo SS velocity", "scVelo dyn velocity", "scPTR gamma", "DeepPTR z_T"]:
        key = sname.lower().replace(" ", "_")
        s = results.get(key, {}).get("silhouette", np.nan)
        if np.isfinite(s):
            sil_data[sname] = s

    if sil_data:
        axes[2].bar(list(sil_data.keys()), list(sil_data.values()), color="steelblue", alpha=0.7)
        axes[2].set_ylabel("Silhouette score")
        axes[2].set_title("Cell-type discrimination")
        plt.setp(axes[2].get_xticklabels(), rotation=30, ha="right", fontsize=8)

    fig.suptitle(f"{name}: Method comparison", y=1.02)
    fig.tight_layout()
    save_fig(fig, f"{name}_method_comparison", OUT)

    return results


def main():
    set_figure_style()
    all_results = {}
    for name, loader, ck in DATASETS:
        all_results[name] = run_comparison(name, loader, ck)

    # Print summary table
    print(f"\n{'=' * 80}")
    print("METHOD COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<25} {'HL mouse':>10} {'HL human':>10} {'Runtime':>10}")
    print("-" * 60)

    for ds_name in all_results:
        print(f"\n  {ds_name.upper()}")
        r = all_results[ds_name]
        for mkey, mname in [("scvelo_ss", "scVelo SS"), ("scvelo_dyn", "scVelo dyn"),
                             ("velovi", "velVI"),
                             ("scptr_analytical", "scPTR analytical"), ("deepptr", "DeepPTR")]:
            d = r.get(mkey, {})
            r_m = d.get("halflife_mouse", {}).get("r", np.nan)
            r_h = d.get("halflife_human", {}).get("r", np.nan)
            t = d.get("time", np.nan)
            r_m_s = f"{r_m:.4f}" if np.isfinite(r_m) else "N/A"
            r_h_s = f"{r_h:.4f}" if np.isfinite(r_h) else "N/A"
            t_s = f"{t:.0f}s" if np.isfinite(t) else "N/A"
            print(f"  {mname:<25} {r_m_s:>10} {r_h_s:>10} {t_s:>10}")

    save_json(all_results, "comparison_all", OUT)


if __name__ == "__main__":
    main()
