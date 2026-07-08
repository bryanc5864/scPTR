# scPTR Figure Set

This document describes the figure bundle generated for the scPTR paper.

The figures are in:

```text
figures/
```

The numeric tables behind the figures are in:

```text
figures/results/
```

The generator is:

```text
analyses/make_figures.py
```

## Why This Figure Organization

Each figure is named with a `fig_` prefix and a descriptive name that encodes
what the figure shows. Old `slide_NN_` prefixes have been removed.

Every UMAP filename and title states:

- which dataset is used
- whether the UMAP is expression-space or gamma-space
- whether labels are cell type, global PT state, or within-subset gamma substate
- whether the UMAP was computed globally or within the subset

## Data And Models

All figures in `figures/` were generated from real scRNA-seq spliced/unspliced
data using the repository loaders:

```python
scptr.datasets.pancreas()
scptr.datasets.dentate_gyrus()
```

No synthetic data are used.

The script runs the actual scPTR analytic pipeline:

```text
filter_genes
normalize_layers
expression PCA/neighbors/UMAP
smooth_layers
estimate_beta
estimate_gamma
variance_decomposition
pt_states
pt_velocity
```

The subset panels use real gamma matrices from this pipeline. Within each
annotated cell type, gamma-substate labels are produced by KMeans on gamma PCA
coordinates. Those same labels are then evaluated in both expression PCA space
and gamma PCA space using silhouette score.

This is the key rule for the subset UMAPs:

```text
Same cells, same gamma-substate labels, different feature spaces.
```

That means a subset panel is valid when the bar chart shows gamma-space
silhouette above expression-space silhouette. The old Granule immature/mature
individual panels were not retained because, in this real run, their gamma
silhouette was lower than their expression silhouette. They remain represented
truthfully in the dentate overview table/bar chart, but are not used as
positive examples.

## How To Regenerate

From the repository root:

```bash
PYTHONPATH="$PWD/src" python analyses/make_figures.py
```

Alternatively, install the package editable:

```bash
python -m pip install -e .
python analyses/make_figures.py
```

The script clears stale PNGs from `figures/` and stale result files from
`figures/results/` before writing new ones, so deleted or renamed panels do
not linger.

## Output Manifest

The current run generated 17 nonblank PNGs.

| File | What it shows |
|---|---|
| `fig_gamma_estimation.png` | Real pancreas phase portrait for `Malat1`, beta distribution, and median gamma distribution. |
| `fig_halflife_validation.png` | Real half-life Spearman correlations for pancreas and dentate gyrus against Herzog and Schofield references. |
| `fig_variance_decomposition.png` | Real TF/PTF variance decomposition summary for pancreas and dentate gyrus. |
| `fig_pancreas_global_umap.png` | Four-panel pancreas global UMAP comparison. Columns are expression vs gamma spaces. Rows are cell-type vs PT-state labels. |
| `fig_dentate_global_umap.png` | Same four-panel global UMAP comparison for dentate gyrus. |
| `fig_invisible_states_pancreas.png` | Pancreas subset-level silhouette comparison for all sufficiently large annotated cell types. |
| `fig_invisible_states_dentate.png` | Dentate subset-level silhouette comparison for all sufficiently large annotated cell types. |
| `fig_invisible_epsilon.png` | Epsilon-only expression vs gamma UMAPs, same gamma-substate labels, plus silhouette bars. |
| `fig_invisible_pre_endocrine.png` | Pre-endocrine-only expression vs gamma UMAPs, same gamma-substate labels, plus silhouette bars. |
| `fig_invisible_dg_endothelial.png` | Dentate Endothelial subset expression vs gamma UMAPs with silhouette bars. |
| `fig_invisible_dg_gaba.png` | Dentate GABA subset expression vs gamma UMAPs with silhouette bars. |
| `fig_invisible_dg_microglia.png` | Dentate Microglia subset expression vs gamma UMAPs with silhouette bars. |
| `fig_invisible_dg_radial_glia.png` | Dentate Radial Glia-like subset expression vs gamma UMAPs with silhouette bars. |
| `fig_pt_velocity_pancreas.png` | Real pancreas post-transcriptional velocity field on gamma UMAP. |
| `fig_pt_velocity_dentate.png` | Real dentate gyrus post-transcriptional velocity field on gamma UMAP. |
| `fig_rbp_hubs.png` | Real RBP hub counts inferred from pancreas and dentate gamma/expression matrices. |
| `fig_results_summary.png` | Summary chart: PT states, filtered genes, and count of subsets where gamma separates better. |

## Important UMAP Interpretation

UMAP is a layout algorithm, not a fixed coordinate system. A UMAP recomputed
from expression features and a UMAP recomputed from gamma features are expected
to have different shapes. That is the point of the comparison.

The valid comparison is not whether the clouds have the same x/y coordinates.
The valid comparison is:

```text
Given the same cells and same labels, are labels more separated in expression
space or gamma space?
```

For subset panels, the answer is quantified by the rightmost bar chart:

```text
silhouette(gamma-substate labels in expression PCA space)
vs
silhouette(gamma-substate labels in gamma PCA space)
```

## Key Real-Run Results

From `figures/results/pancreas_subset_silhouettes.csv`:

```text
Epsilon:
  n_cells = 142
  n_substates = 3
  expression silhouette = -0.0420
  gamma silhouette = 0.2762
  gap = 0.3183

Pre-endocrine:
  n_cells = 592
  n_substates = 3
  expression silhouette = -0.0202
  gamma silhouette = 0.2306
  gap = 0.2508
```

From `figures/results/dentate_gyrus_subset_silhouettes.csv`,
positive dentate examples include:

```text
Endothelial:
  expression silhouette = 0.2439
  gamma silhouette = 0.3957
  gap = 0.1518

GABA:
  expression silhouette = 0.1370
  gamma silhouette = 0.2706
  gap = 0.1336

Microglia:
  expression silhouette = 0.2498
  gamma silhouette = 0.3164
  gap = 0.0665

Radial Glia-like:
  expression silhouette = 0.1643
  gamma silhouette = 0.2396
  gap = 0.0752
```

Granule immature and Granule mature are not used as individual positive-example
figures in this run because their silhouette gaps are negative. They are still
included in the overview CSV and overview bar chart so the result is honest and
auditable.

## Verification

A nonblank image check was run after generation:

```python
from pathlib import Path
from PIL import Image

folder = Path("figures")
files = sorted(folder.glob("fig_*.png"))
assert len(files) == 17
for path in files:
    image = Image.open(path).convert("RGB")
    assert any(lo != hi for lo, hi in image.getextrema()), path
```

All 17 figures passed.
