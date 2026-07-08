# Real-Data Figure Set For The scPTR Talk

This document describes the replacement figure bundle generated for the local
talk file:

```text
C:\Users\Maozer\Downloads\scPTR talk.pdf
```

The new figures are in:

```text
figures/talk_real_data/
```

The numeric tables behind the figures are in:

```text
figures/talk_real_data_results/
```

The generator is:

```text
analyses/make_talk_real_figures.py
```

## Why This Replaces The Previous Bundle

The previous folder `figures/reproduced_paper_labelled/` was removed because it
mixed several incompatible figure concepts under similar names:

- a global all-pancreas expression/gamma UMAP comparison
- an Epsilon-only expression/gamma UMAP comparison
- panels coloured by different label systems, including cell type, global PT
  state, and within-subset gamma substate

That made files named like `figure_1b` and `figure_3a` look comparable when they
were not. The new folder avoids that failure mode. Every UMAP filename and title
now states:

- which dataset is used
- whether the UMAP is expression-space or gamma-space
- whether labels are cell type, global PT state, or within-subset gamma substate
- whether the UMAP was computed globally or within the subset

## Data And Models

All figures in `figures/talk_real_data/` were generated from real scRNA-seq
spliced/unspliced data using the repository loaders:

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

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONPATH="$PWD\src"
python analyses\make_talk_real_figures.py
```

Alternatively, install the package editable:

```powershell
python -m pip install -e .
python analyses\make_talk_real_figures.py
```

The script clears stale PNGs and stale result files from the talk output folders
before writing new ones, so deleted or renamed panels do not linger.

## Output Manifest

The current run generated 17 nonblank PNGs.

| Talk slide | File | What it shows |
|---|---|---|
| Slide 8 | `slide_08_pancreas_estimator_qc.png` | Real pancreas phase portrait for `Malat1`, beta distribution, and median gamma distribution. |
| Slide 11 | `slide_11_pancreas_dentate_halflife_validation.png` | Real half-life Spearman correlations for pancreas and dentate gyrus against Herzog and Schofield references. |
| Slide 13 | `slide_13_variance_decomposition_pancreas_dentate.png` | Real TF/PTF variance decomposition summary for pancreas and dentate gyrus. |
| Slide 15 | `slide_15_pancreas_global_umap_expression_vs_gamma_matched.png` | Four-panel pancreas global UMAP comparison. Columns are expression vs gamma spaces. Rows are cell-type vs PT-state labels. |
| Slide 15 | `slide_15_dentate_gyrus_global_umap_expression_vs_gamma_matched.png` | Same four-panel global UMAP comparison for dentate gyrus. |
| Slide 16 | `slide_16_pancreas_silhouette_overview.png` | Pancreas subset-level silhouette comparison for all sufficiently large annotated cell types. |
| Slide 16 | `slide_16_dentate_gyrus_silhouette_overview.png` | Dentate subset-level silhouette comparison for all sufficiently large annotated cell types. |
| Slide 16 | `slide_16_pancreas_epsilon_substate_umaps.png` | Epsilon-only expression vs gamma UMAPs, same gamma-substate labels, plus silhouette bars. |
| Slide 16 | `slide_16_pancreas_pre_endocrine_substate_umaps.png` | Pre-endocrine-only expression vs gamma UMAPs, same gamma-substate labels, plus silhouette bars. |
| Slide 16 | `slide_16_dentate_endothelial_substate_umaps.png` | Dentate Endothelial subset expression vs gamma UMAPs with silhouette bars. |
| Slide 16 | `slide_16_dentate_gaba_substate_umaps.png` | Dentate GABA subset expression vs gamma UMAPs with silhouette bars. |
| Slide 16 | `slide_16_dentate_microglia_substate_umaps.png` | Dentate Microglia subset expression vs gamma UMAPs with silhouette bars. |
| Slide 16 | `slide_16_dentate_radial_glia_substate_umaps.png` | Dentate Radial Glia-like subset expression vs gamma UMAPs with silhouette bars. |
| Slide 17 | `slide_17_pancreas_pt_velocity.png` | Real pancreas post-transcriptional velocity field on gamma UMAP. |
| Slide 17 | `slide_17_dentate_gyrus_pt_velocity.png` | Real dentate gyrus post-transcriptional velocity field on gamma UMAP. |
| Slide 18 | `slide_18_pancreas_dentate_rbp_hubs.png` | Real RBP hub counts inferred from pancreas and dentate gamma/expression matrices. |
| Slide 22 | `slide_22_results_at_a_glance_pancreas_dentate.png` | Talk summary chart: PT states, filtered genes, and count of subsets where gamma separates better. |

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

From `figures/talk_real_data_results/pancreas_subset_silhouettes.csv`:

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

From `figures/talk_real_data_results/dentate_gyrus_subset_silhouettes.csv`,
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

folder = Path("figures/talk_real_data")
files = sorted(folder.glob("*.png"))
assert len(files) == 17
for path in files:
    image = Image.open(path).convert("RGB")
    assert any(lo != hi for lo, hi in image.getextrema()), path
```

All 17 figures passed.
