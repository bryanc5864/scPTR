# scPTR

**Single-Cell Post-Transcriptional Regulatory Decomposition**

scPTR estimates per-cell, per-gene mRNA degradation rates from scRNA-seq spliced/unspliced counts and uses them as a primary analytical axis — complementary to RNA velocity.

## What scPTR does

- **Degradation rate estimation**: Per-cell, per-gene gamma from kinetic steady-state relationships with kNN Gaussian-kernel smoothing
- **Expression-invisible states**: Discovers cell subpopulations with distinct post-transcriptional programs undetectable by standard expression analysis
- **Post-transcriptional velocity**: Neighbor-averaged gamma gradient that captures degradation dynamics orthogonal to RNA velocity
- **RBP-target networks**: Library-size-corrected inference of RNA-binding protein regulatory networks with elastic net
- **DeepPTR**: Structured VAE with a kinetic decoder that disentangles transcriptional and post-transcriptional latent spaces

## Installation

```bash
pip install .
```

Optional dependencies:

```bash
pip install ".[deep]"      # PyTorch for DeepPTR
pip install ".[datasets]"  # Pooch for dataset downloads
pip install ".[dev]"       # pytest for testing
```

## Quick start

```python
import scptr

# Load data with spliced/unspliced layers
adata = scptr.read_h5ad("your_data.h5ad")

# Preprocessing
scptr.pp.filter_genes(adata)
scptr.pp.normalize_layers(adata)
scptr.pp.neighbors(adata)
scptr.pp.smooth_layers(adata)

# Estimate rates
scptr.tl.estimate_beta(adata)
scptr.tl.estimate_gamma(adata)

# Downstream analysis
scptr.tl.variance_decomposition(adata)
scptr.tl.pt_states(adata)
scptr.tl.pt_velocity(adata)
scptr.tl.infer_network(adata)
```

## Pipeline overview

```
Raw scRNA-seq (spliced + unspliced)
  -> Gene/cell filtering
  -> Library-size normalization (per layer)
  -> kNN graph + Gaussian smoothing
  -> Beta estimation (quantile regression on u/s phase portraits)
  -> Gamma estimation (gamma = beta * u / s, per cell per gene)
  -> Variance decomposition (transcriptional vs post-transcriptional)
  -> PT states (PCA + Leiden clustering in gamma-space)
  -> PT velocity (neighbor-averaged gamma gradient)
  -> RBP-target network inference (elastic net, library-size corrected)
```

## Validation

scPTR gamma estimates have been validated against:

| Validation | Result |
|------------|--------|
| Published mRNA half-lives | ρ = −0.81 (sci-fate), −0.33 to −0.40 (10x developmental) |
| Method comparison | Outperforms scVelo steady-state (−0.37) and velVI (−0.28) |
| miRNA target enrichment | 59% of 215 families enriched (p = 4.7×10⁻⁶⁵) |
| 3′ UTR sequence features | UTR length ρ = 0.34 (p < 10⁻²⁰⁰), AU content ρ = 0.30 |
| DepMap CRISPR essentiality | Hub RBPs more essential (p = 6.4×10⁻⁵) |
| Subsampling robustness | r > 0.97 at 20% subsampling |

## Key findings

- **Expression-invisible states**: 3/8 pancreatic and 6/11 hippocampal cell types harbor post-transcriptional subpopulations undetectable by expression analysis (confirmed by zero-permutation control, ARI ≈ 0), enriched for ER stress/autophagy and synaptic plasticity pathways
- **Temporal precedence**: degradation-rate changes precede expression changes for 54% of transition genes in pancreas (p < 10⁻⁵⁷) and 78% in dentate gyrus (p = 9.9×10⁻¹³)
- **RBP networks**: library-size-corrected inference identifies essential hub regulators (HNRNPA1, YBX1, ELAVL1/HuR); neuroblastoma shows 66% stabilizing edges vs. destabilizing bias in developmental tissues

## Datasets

Built-in dataset loaders (downloaded via Pooch):

```python
adata = scptr.datasets.pancreas()        # Mouse endocrinogenesis (3,696 cells)
adata = scptr.datasets.dentate_gyrus()   # Mouse hippocampal neurogenesis (2,930 cells)
adata = scptr.datasets.sci_fate()        # Human A549 dexamethasone response (7,404 cells)
```

## Requirements

- Python >= 3.9
- anndata >= 0.8, scanpy >= 1.9, numpy >= 1.21, scipy >= 1.7, numba >= 0.55
- Optional: torch >= 2.0 (DeepPTR), pooch >= 1.6 (datasets)

## Reproducing results

All analyses can be reproduced from scratch using the scripts in `analyses/`:

```bash
# full pipeline on pancreas (aims 1–4)
python analyses/run_all.py

# dentate gyrus
python analyses/run_dentate_gyrus.py

# sci-fate metabolic labeling validation
python analyses/run_scifate.py

# expression-invisible states + temporal precedence + RBP networks
python analyses/run_gaps.py

# deepPTR benchmarks
python analyses/run_deep_benchmark_v2.py

# cross-dataset summary
python analyses/run_summary.py
```

Outputs (figures and JSON results) go to `output/`. The 53-test suite covers all core estimators:

```bash
pip install ".[dev]"
pytest tests/ -v
```

## Figures

All committed figures are in `figures/` with descriptive `fig_` names:

| File | What it shows |
|------|---------------|
| `fig_gamma_estimation.png` | Phase portrait, beta distribution, median gamma distribution |
| `fig_halflife_validation.png` | Half-life Spearman correlations (pancreas + DG) |
| `fig_variance_decomposition.png` | TF/PTF variance decomposition |
| `fig_pancreas_global_umap.png` | Expression vs γ-space UMAP comparison (pancreas) |
| `fig_dentate_global_umap.png` | Expression vs γ-space UMAP comparison (DG) |
| `fig_invisible_states_pancreas.png` | Silhouette overview: 3/8 pancreas cell types invisible to expression |
| `fig_invisible_states_dentate.png` | Silhouette overview: 6/11 DG cell types invisible to expression |
| `fig_invisible_epsilon.png` | Epsilon substate UMAPs (sil γ=0.276, expr=−0.042) |
| `fig_invisible_pre_endocrine.png` | Pre-endocrine substate UMAPs |
| `fig_invisible_dg_*.png` | DG per-cell-type invisible-state UMAPs |
| `fig_permutation_control.png` | Expression clustering fails to recover γ-space structure |
| `fig_pt_velocity_pancreas.png` | PT velocity field on γ-space UMAP (pancreas) |
| `fig_pt_velocity_dentate.png` | PT velocity field on γ-space UMAP (DG) |
| `fig_pt_velocity_orthogonality.png` | PT velocity ≈ orthogonal to RNA velocity (cosine ≈ 0) |
| `fig_gamma_leads_expression.png` | γ leads expression at transitions: 63% pancreas, 78% DG |
| `fig_rbp_hubs.png` | Top RBP hubs from network inference |
| `fig_library_size_correction.png` | Library-size correction: 98.9% → 33.7% destabilizing |
| `fig_depmap_essentials.png` | Hub RBPs more essential in cancer (p = 6.4×10⁻⁵) |
| `fig_neuroblastoma_network.png` | Neuroblastoma: 66% stabilizing, HNRNPA2B1/PABPC1 top hubs |
| `fig_results_summary.png` | Summary: PT states, genes used, γ-better subsets |

Numeric tables: `figures/results/`  
Generators: `analyses/make_figures.py` (pipeline figures), `analyses/make_figures_claims.py` (claim figures from existing output)  
Documentation: `docs/FIGURES.md`

## Repository structure

```
src/scptr/          core package
  preprocessing/    filter, normalize, smooth
  tools/            beta/gamma estimation, PT states, velocity, network inference
  plotting/         UMAP, phase portraits, network plots
  benchmark/        half-life validation, miRNA enrichment
  datasets/         built-in loaders (pancreas, dentate gyrus, sci-fate)
  deep/             DeepPTR VAE
analyses/           analysis scripts reproducing every result
  run_all.py        end-to-end pancreas pipeline
  run_gaps.py       invisible states, velocity, networks
  make_figures.py   regenerate all pipeline figures
  make_figures_claims.py  claim-backing figures from existing output/
  deep/             individual deep-benchmark scripts (01–35)
tests/              pytest suite (53 tests)
figures/            committed figures (fig_*.png)
  results/          numeric tables behind figures
output/             analysis results (JSON + CSV, gitignored)
```

## Citation

If you use scPTR, please cite:

> scPTR: Decomposing Post-Transcriptional Regulation at Single-Cell Resolution (2026)
