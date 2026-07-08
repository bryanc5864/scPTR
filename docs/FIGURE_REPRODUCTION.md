# Figure Reproduction Notes

This document records how the regenerated scPTR figure bundle was produced and
what each labelled output file corresponds to. The generated PNG files are
committed under:

```text
figures/reproduced_paper_labelled/
```

The figure-generation entry point that consolidates the labelled export is:

```text
analyses/make_labelled_paper_figures.py
```

The notes below are intentionally explicit so that the figure bundle can be
rebuilt from a fresh checkout.

## Environment

The figures were regenerated on Windows with Python 3.11.9. The local package
was installed editable from the repository root:

```powershell
python -m pip install -e .
python -m pip install scvelo
```

The environment used for this regeneration included:

```text
scptr==0.1.0
scanpy==1.11.5
anndata==0.12.10
numpy==1.26.4
scipy==1.17.1
pandas==2.3.3
matplotlib==3.10.8
seaborn==0.13.2
scikit-learn==1.8.0
scvelo==0.3.4
torch==2.6.0+cu124
leidenalg==0.11.0
umap-learn==0.5.11
statsmodels==0.14.6
pooch==1.9.0
pillow==12.2.0
```

For Windows consoles, use UTF-8 output when running scripts that print Greek
letters or other scientific symbols:

```powershell
$env:PYTHONIOENCODING='utf-8'
```

## Data Inputs

### Pancreas and Dentate Gyrus

The repository loaders fetch the two scVelo tutorial datasets through Pooch:

```python
scptr.datasets.pancreas()
scptr.datasets.dentate_gyrus()
```

They are cached under:

```text
~/.cache/scptr/
```

The source URLs are defined in `src/scptr/datasets/_registry.py` and point to
the Theis lab scVelo notebook data:

```text
https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad
https://github.com/theislab/scvelo_notebooks/raw/master/data/DentateGyrus/10X43_1.h5ad
```

### TargetScan miRNA Predictions

`analyses/run_mirna_analysis.py` expects the TargetScan summary file here:

```text
.cache/targetscan/Summary_Counts.default_predictions.txt
```

The file was downloaded from TargetScan 8.0:

```powershell
$dir = ".cache\targetscan"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$url = "https://www.targetscan.org/vert_80/vert_80_data_download/Summary_Counts.default_predictions.txt.zip"
Invoke-WebRequest -Uri $url -OutFile "$dir\Summary_Counts.default_predictions.txt.zip"
Expand-Archive -Path "$dir\Summary_Counts.default_predictions.txt.zip" -DestinationPath $dir -Force
```

### sci-fate GEO Files

`analyses/run_scifate.py` reads four GEO supplemental files from:

```text
~/.cache/scptr/scifate/
```

They were downloaded from:

```text
https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3770nnn/GSM3770930/suppl/GSM3770930_A549_cell_annotate.txt.gz
https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3770nnn/GSM3770930/suppl/GSM3770930_A549_gene_annotate.txt.gz
https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3770nnn/GSM3770930/suppl/GSM3770930_A549_gene_count.txt.gz
https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3770nnn/GSM3770930/suppl/GSM3770930_A549_gene_count_newly_synthesised.txt.gz
```

PowerShell download command:

```powershell
$dir = Join-Path $HOME ".cache\scptr\scifate"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$base = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3770nnn/GSM3770930/suppl"
$files = @(
  "GSM3770930_A549_cell_annotate.txt.gz",
  "GSM3770930_A549_gene_annotate.txt.gz",
  "GSM3770930_A549_gene_count.txt.gz",
  "GSM3770930_A549_gene_count_newly_synthesised.txt.gz"
)
foreach ($f in $files) {
  Invoke-WebRequest -Uri "$base/$f" -OutFile (Join-Path $dir $f)
}
```

## Reproduction Command Sequence

Run these from the repository root after installation and data setup.

### 1. Pancreas core figures

```powershell
$env:PYTHONIOENCODING='utf-8'
python analyses\run_all.py
```

This regenerates:

```text
output/figures/aim1/halflife_scatter.png
output/figures/aim1/enrichment_barplot.png
output/figures/aim1/subsampling_robustness.png
output/figures/aim2/pt_umap.png
output/figures/aim2/tf_ptf_scatter.png
output/figures/aim2/gamma_heatmap.png
output/figures/aim3/pt_velocity_embedding.png
```

Observed run summary:

```text
Pancreas input: 3,696 cells, 27,998 genes
After filtering: 3,696 cells, 11,906 genes
PT states: 11
Human half-life Spearman: -0.4018, n=4,308
Mouse half-life Spearman: -0.3505, n=4,611
```

### 2. Expression-vs-gamma benchmark panels

```powershell
$env:PYTHONIOENCODING='utf-8'
python analyses\deep\11_expression_vs_gamma.py
```

Outputs:

```text
output/deep_benchmarks/11_expression_vs_gamma/figures/pancreas_expr_vs_gamma.png
output/deep_benchmarks/11_expression_vs_gamma/figures/dentate_gyrus_expr_vs_gamma.png
```

Observed run summary:

```text
Pancreas expression vs gamma Spearman: -0.7522
Pancreas gamma residual variance independent of expression: 85.8%
Dentate expression vs gamma Spearman: -0.6604
Dentate gamma residual variance independent of expression: 88.9%
```

### 3. Expression-space vs gamma-space UMAP comparison

```powershell
$env:PYTHONIOENCODING='utf-8'
python analyses\aim2_hidden_states\run_pt_states_comparison.py
```

Output:

```text
output/figures/aim2/expression_vs_gamma_clustering.png
```

Observed adjusted Rand index between expression clustering and gamma clustering:

```text
ARI = 0.418
```

### 4. Dentate gyrus core figures

```powershell
$env:PYTHONIOENCODING='utf-8'
python analyses\run_dentate_gyrus.py
```

Outputs:

```text
output/dentate_gyrus/figures/halflife_scatter.png
output/dentate_gyrus/figures/enrichment_barplot.png
output/dentate_gyrus/figures/pt_umap.png
output/dentate_gyrus/figures/tf_ptf_scatter.png
output/dentate_gyrus/figures/gamma_heatmap.png
output/dentate_gyrus/figures/pt_velocity_embedding.png
```

### 5. Gap analysis: invisible states, RNA velocity comparison, networks

```powershell
$env:PYTHONIOENCODING='utf-8'
python analyses\run_gaps.py
```

Outputs:

```text
output/gap_analysis/figures/invisible_states/invisible_states_pancreas.png
output/gap_analysis/figures/invisible_states/invisible_states_dentate_gyrus.png
output/gap_analysis/figures/velocity_comparison/velocity_comparison_pancreas.png
output/gap_analysis/figures/velocity_comparison/velocity_comparison_dentate_gyrus.png
output/gap_analysis/figures/network/network_pancreas.png
output/gap_analysis/figures/network/network_dentate_gyrus.png
```

Observed run summary:

```text
Pancreas invisible or partially invisible states:
  Epsilon, Pre-endocrine, Delta, Ngn3 high EP
Dentate invisible or partially invisible states:
  Granule immature, Granule mature, Microglia, Radial Glia-like, Mossy, OL, OPC
Pancreas PT/RNA velocity mean cosine similarity: -0.0561
Dentate PT/RNA velocity mean cosine similarity: -0.1615
Pancreas network significant edges: 1,112
Dentate network significant edges: 3,054
```

### 6. Temporal precedence figures

```powershell
$env:PYTHONIOENCODING='utf-8'
python analyses\run_precedence.py
```

Outputs:

```text
output/precedence/figures/precedence_pancreas.png
output/precedence/figures/example_genes_pancreas.png
output/precedence/figures/precedence_dentate_gyrus.png
output/precedence/figures/example_genes_dentate_gyrus.png
output/precedence/figures/combined_precedence.png
```

Observed run summary:

```text
Pancreas transition genes: 188
Pancreas gamma leads expression: 119/188 genes
Pancreas binomial p-value: 5.2949e-06
Dentate transition genes: 146
Dentate gamma leads expression: 114/146 genes
Dentate binomial p-value: 9.9323e-13
```

### 7. miRNA target enrichment

```powershell
$env:PYTHONIOENCODING='utf-8'
python analyses\run_mirna_analysis.py
```

Outputs:

```text
output/mirna_analysis/figures/mirna_analysis_pancreas.png
output/mirna_analysis/figures/mirna_analysis_dentate_gyrus.png
```

Observed pancreas summary:

```text
miRNA families tested: 215
Significant at FDR < 0.05: 126
Significant at FDR < 0.10: 141
Aggregate p-value: 4.68e-65
```

### 8. sci-fate validation

```powershell
$env:PYTHONIOENCODING='utf-8'
python analyses\run_scifate.py
```

Outputs:

```text
output/scifate_validation/figures/scifate_gamma_vs_ground_truth.png
output/scifate_validation/figures/scifate_per_timepoint.png
output/scifate_validation/figures/scifate_top_bottom_boxplot.png
```

Observed run summary:

```text
sci-fate input: 7,404 cells, 43,167 genes
scPTR pipeline: 7,404 cells, 7,970 genes
Gamma vs new/old ratio Spearman: 0.9928
Gamma vs fraction-new Spearman: 0.9928
Human half-life Spearman: -0.8123, n=6,995
Mouse half-life Spearman: -0.6732, n=6,816
Top-vs-bottom gamma turnover fold difference: 32.4x
```

### 9. Cross-dataset summary figures

```powershell
$env:PYTHONIOENCODING='utf-8'
python analyses\run_summary.py
```

Outputs:

```text
output/summary/figures/halflife_comparison.png
output/summary/figures/robustness_curves.png
output/summary/figures/cross_dataset_heatmap.png
output/summary/figures/enrichment_comparison.png
```

Observed summary:

```text
pancreas:       3,696 cells, 11,906 genes, 11 PT states
dentate_gyrus:  2,930 cells,  5,325 genes, 13 PT states
scifate:        7,404 cells,  7,970 genes, 13 PT states
```

### 10. Labelled figure export

After all upstream scripts have run, create the single labelled export folder:

```powershell
$env:PYTHONIOENCODING='utf-8'
python analyses\make_labelled_paper_figures.py
```

This copies and generates labelled figures under:

```text
output/paper_figures_labelled/
```

The committed copy of that folder is:

```text
figures/reproduced_paper_labelled/
```

## Compatibility Notes

Two small script-level compatibility changes were needed for this Windows/Python
environment and are included in the committed scripts:

1. `analyses/aim2_hidden_states/run_pt_states_comparison.py`
   now calls `scptr.pl.pt_comparison(adata, show=False)` so the plotting helper
   returns a Matplotlib figure object that can be saved.

2. `analyses/run_gaps.py`
   calls `scv.pp.filter_and_normalize(adata_scv, min_shared_counts=20)`.
   The installed `scvelo==0.3.4` forwards unknown keyword arguments to
   `normalize_per_cell`, so the older `n_top_genes=2000` argument caused an API
   error in this environment.

The consolidation script `analyses/make_labelled_paper_figures.py` is new. It
does not change the estimator; it only copies generated figures into a labelled
folder and creates split expression/gamma UMAP panels that the stock scripts
saved as combined multi-panel images.

## Figure Manifest

The regenerated labelled bundle contains 31 PNG files.

| File | Description | Source |
|---|---|---|
| `figure_1a_rate_estimation_phase_portrait.png` | Pancreas rate-estimation phase portrait panel. | `make_labelled_paper_figures.py` |
| `figure_1b_pancreas_expression_space_umap_cell_types.png` | Pancreas expression-space UMAP coloured by annotated cell type. | `make_labelled_paper_figures.py` |
| `figure_1b_pancreas_gamma_space_umap_pt_states.png` | Pancreas gamma-space UMAP coloured by PT state. | `make_labelled_paper_figures.py` |
| `figure_1c_rbp_network_hub_bargraph.png` | Pancreas RBP hub-count bar graph from network inference. | `make_labelled_paper_figures.py`, `run_gaps.py` |
| `figure_2a_pancreas_halflife_scatter.png` | Pancreas median gamma vs published half-life scatter. | `run_all.py` |
| `figure_2b_pancreas_mirna_target_enrichment.png` | Pancreas miRNA target enrichment panels. | `run_mirna_analysis.py` |
| `figure_2c_scifate_gamma_vs_ground_truth.png` | sci-fate gamma vs metabolic-labeling ground-truth panels. | `run_scifate.py` |
| `figure_2d_cross_dataset_halflife_bargraph.png` | Cross-dataset half-life validation bar chart. | `run_summary.py` |
| `figure_2e_subsampling_robustness_curves.png` | Subsampling robustness curves across datasets. | `run_summary.py` |
| `figure_3a_epsilon_expression_space_umap_gamma_substates.png` | Epsilon-cell expression UMAP labelled by gamma substate. | `make_labelled_paper_figures.py` |
| `figure_3a_epsilon_gamma_space_umap_gamma_substates.png` | Epsilon-cell gamma UMAP labelled by gamma substate. | `make_labelled_paper_figures.py` |
| `figure_3a_pancreas_expression_vs_gamma_umap_combined.png` | Combined expression/gamma UMAP comparison for pancreas. | `run_pt_states_comparison.py` |
| `figure_3b_pancreas_invisible_states_bargraph.png` | Pancreas invisible-state silhouette comparison. | `run_gaps.py` |
| `figure_3c_dentate_invisible_states_bargraph.png` | Dentate invisible-state silhouette comparison. | `run_gaps.py` |
| `figure_3d_pancreas_temporal_precedence.png` | Pancreas temporal precedence histograms. | `run_precedence.py` |
| `figure_3e_dentate_temporal_precedence.png` | Dentate temporal precedence histograms. | `run_precedence.py` |
| `figure_3f_combined_temporal_precedence_bargraph.png` | Combined precedence bar chart. | `run_precedence.py` |
| `figure_4a_pancreas_pt_velocity_umap.png` | Pancreas PT velocity UMAP. | `run_all.py` |
| `figure_4b_dentate_pt_velocity_umap.png` | Dentate PT velocity UMAP. | `run_dentate_gyrus.py` |
| `figure_4c_pancreas_pt_vs_rna_velocity.png` | Pancreas PT velocity vs scVelo RNA velocity comparison. | `run_gaps.py` |
| `figure_4d_dentate_pt_vs_rna_velocity.png` | Dentate PT velocity vs scVelo RNA velocity comparison. | `run_gaps.py` |
| `figure_5a_pancreas_rbp_network.png` | Pancreas RBP-target network summary. | `run_gaps.py` |
| `figure_5b_dentate_rbp_network.png` | Dentate RBP-target network summary. | `run_gaps.py` |
| `supplement_are_nmd_enrichment_comparison.png` | ARE/NMD enrichment comparison. | `run_summary.py` |
| `supplement_cross_dataset_gamma_heatmap.png` | Cross-dataset gamma consistency heatmap. | `run_summary.py` |
| `supplement_dentate_gamma_space_umap.png` | Dentate gamma-space UMAP. | `run_dentate_gyrus.py` |
| `supplement_dentate_halflife_scatter.png` | Dentate half-life scatter. | `run_dentate_gyrus.py` |
| `supplement_expression_vs_gamma_dentate_gyrus.png` | Dentate expression-vs-gamma benchmark. | `deep/11_expression_vs_gamma.py` |
| `supplement_expression_vs_gamma_pancreas.png` | Pancreas expression-vs-gamma benchmark. | `deep/11_expression_vs_gamma.py` |
| `supplement_scifate_per_timepoint.png` | sci-fate per-timepoint validation. | `run_scifate.py` |
| `supplement_scifate_top_bottom_boxplot.png` | sci-fate top/bottom gamma turnover boxplot. | `run_scifate.py` |

## Verification

After the labelled export was produced, every PNG was opened with Pillow and
checked for non-flat RGB extrema. All 31 files were nonblank.

Verification snippet:

```python
from pathlib import Path
from PIL import Image

folder = Path("output/paper_figures_labelled")
files = sorted(folder.glob("*.png"))
assert len(files) == 31
for path in files:
    image = Image.open(path).convert("RGB")
    assert any(lo != hi for lo, hi in image.getextrema()), path
```

## Notes On Reproducibility

The scripts use fixed random seeds in the main Scanpy/scikit-learn calls where
the original code exposes them. Some upstream routines, especially approximate
neighbors and UMAP layout, can still vary slightly across platform, dependency
version, and BLAS/threading configuration. The committed PNGs are the exact
outputs generated in the environment recorded above.

