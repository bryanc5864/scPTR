# scPTR Results

Comprehensive results from the scPTR (Single-Cell Post-Transcriptional Regulatory Decomposition) framework across three datasets.

## Datasets

| Dataset | Species | Platform | Cells | Genes | PT States | Description |
|---------|---------|----------|-------|-------|-----------|-------------|
| Pancreas | Mouse | 10X Chromium | 3,696 | 11,906 | 11 | Endocrinogenesis (day 15.5) |
| Dentate Gyrus | Mouse | 10X Chromium | 2,930 | 5,325 | 12 | Hippocampal neurogenesis |
| sci-fate A549 | Human | sci (combinatorial indexing) | 7,404 | 7,970 | 14 | Dexamethasone response (0-10h) |

---

## Aim 1: Benchmarking and Validation

### Half-life Correlation

scPTR gamma estimates correlate negatively with published mRNA half-lives (high gamma = fast degradation = short half-life). All correlations are highly significant (p < 1e-31).

| Dataset | Reference | Spearman r | p-value | Pearson r | n genes |
|---------|-----------|-----------|---------|-----------|---------|
| Pancreas | Mouse (Herzog 2017) | **-0.351** | 2.16e-133 | -0.232 | 4,611 |
| Pancreas | Human (Schofield 2018) | **-0.402** | 7.29e-167 | -0.283 | 4,308 |
| Dentate Gyrus | Mouse (Herzog 2017) | **-0.327** | 1.20e-31 | -0.223 | 1,217 |
| Dentate Gyrus | Human (Schofield 2018) | **-0.389** | 4.70e-42 | -0.287 | 1,126 |
| sci-fate | Mouse (Herzog 2017) | **-0.673** | < 1e-300 | -0.544 | 6,816 |
| sci-fate | Human (Schofield 2018) | **-0.812** | < 1e-300 | -0.646 | 6,995 |

The sci-fate dataset yields the strongest correlations because: (1) it is a human cell line (closer to steady-state assumption), (2) same species as the human half-life reference, and (3) larger cell count with clean labeling data.

### sci-fate Metabolic Labeling Ground Truth

sci-fate (Cao et al. 2020) provides both total and newly synthesized mRNA counts per cell via 4sU metabolic labeling, enabling direct validation of degradation rate estimates.

| Metric | Value |
|--------|-------|
| Genes with ground truth | 7,928 |
| Top 10% gamma genes: median new/old ratio | 1.871 |
| Bottom 10% gamma genes: median new/old ratio | 0.058 |
| **Fold difference** | **32.4x** |
| Mann-Whitney p-value | 1.80e-260 |

Per-timepoint validation (DEX treatment time course):

| Timepoint | Spearman r | n genes |
|-----------|-----------|---------|
| 0h | 0.989 | 7,764 |
| 2h | 0.982 | 7,814 |
| 4h | 0.989 | 7,701 |
| 6h | 0.991 | 7,887 |
| 8h | 0.988 | 7,848 |
| 10h | 0.988 | 7,748 |

> **Note**: The gamma-vs-new/old correlation (r ~ 0.99) is partially tautological because gamma = beta * unspliced/spliced and we map new RNA to the unspliced layer. The independent validations (published half-life correlations above) are the key results.

### ARE/NMD Enrichment

AU-rich element (ARE) genes and nonsense-mediated decay (NMD) targets should have higher degradation rates (gamma). Mann-Whitney U test (one-sided: gene set > background):

| Dataset | Test | Genes in set | p-value | Significant? |
|---------|------|-------------|---------|-------------|
| Pancreas | ARE | 13 | 0.975 | No |
| Pancreas | NMD | 28 | 0.254 | No |
| Dentate Gyrus | ARE | 5 | 0.939 | No |
| Dentate Gyrus | NMD | 10 | **0.003** | Yes |
| sci-fate | ARE | 31 | **7.84e-10** | Yes |
| sci-fate | NMD | 37 | 0.327 | No |

ARE enrichment is significant in sci-fate (human cell line where ARE-containing cytokine/signaling genes are expressed) but not in developmental datasets. NMD enrichment is significant in dentate gyrus.

### Subsampling Robustness

Gamma estimates are highly robust to cell subsampling. Spearman correlation with full-data estimates:

| Fraction | Pancreas | Dentate Gyrus | sci-fate |
|----------|----------|---------------|---------|
| 20% | 0.988 | 0.977 | 1.000 |
| 40% | 0.992 | 0.985 | 1.000 |
| 60% | 0.996 | 0.990 | 1.000 |
| 80% | 0.997 | 0.994 | 1.000 |
| 90% | 0.998 | 0.996 | 1.000 |

### Cross-Dataset Consistency

Per-gene median gamma correlation between dataset pairs (case-insensitive gene matching):

| Dataset A | Dataset B | Shared genes | Spearman r |
|-----------|-----------|-------------|-----------|
| Pancreas | Dentate Gyrus | 4,915 | 0.193 |
| Pancreas | sci-fate | 6,444 | 0.277 |
| Dentate Gyrus | sci-fate | 3,382 | 0.085 |

Modest cross-dataset consistency is expected: degradation rates are cell-type-specific, and these datasets contain entirely different cell populations.

---

## Aim 2: Expression-Invisible Post-Transcriptional States

Sub-clustering within expression-defined clusters using gamma profiles reveals cell populations that are invisible to standard expression analysis. "Invisible" = higher silhouette score in gamma space than expression space.

### Pancreas

| Cluster | Cells | Subclusters | Sil (gamma) | Sil (expr) | Invisibility | Status |
|---------|-------|-------------|-------------|------------|-------------|--------|
| Epsilon | 142 | 3 | 0.193 | -0.056 | 0.249 | **INVISIBLE** |
| Pre-endocrine | 592 | 2 | 0.144 | 0.065 | 0.080 | **INVISIBLE** |
| Delta | 70 | 2 | 0.277 | 0.136 | 0.141 | PARTIALLY |
| Ngn3 high EP | 642 | 2 | 0.260 | 0.206 | 0.054 | PARTIALLY |
| Alpha | 481 | 2 | 0.153 | 0.185 | -0.032 | VISIBLE |
| Beta | 591 | 2 | 0.158 | 0.210 | -0.052 | VISIBLE |
| Ductal | 916 | 2 | 0.191 | 0.310 | -0.118 | VISIBLE |
| Ngn3 low EP | 262 | 2 | 0.192 | 0.282 | -0.089 | VISIBLE |

### Dentate Gyrus

| Cluster | Cells | Subclusters | Sil (gamma) | Sil (expr) | Invisibility | Status |
|---------|-------|-------------|-------------|------------|-------------|--------|
| Microglia | 81 | 2 | 0.344 | 0.036 | 0.307 | **INVISIBLE** |
| Radial Glia-like | 51 | 2 | 0.310 | 0.028 | 0.281 | **INVISIBLE** |
| OL | 50 | 2 | 0.367 | 0.141 | 0.225 | **INVISIBLE** |
| OPC | 53 | 2 | 0.393 | 0.176 | 0.217 | **INVISIBLE** |
| Granule immature | 785 | 2 | 0.217 | 0.005 | 0.213 | **INVISIBLE** |
| Granule mature | 1,070 | 3 | 0.142 | -0.041 | 0.183 | **INVISIBLE** |
| Mossy | 75 | 3 | 0.278 | 0.215 | 0.062 | PARTIALLY |
| Astrocytes | 120 | 2 | 0.272 | 0.227 | 0.045 | VISIBLE |
| GABA | 61 | 3 | 0.243 | 0.205 | 0.038 | VISIBLE |

---

## Aim 3: PT Velocity and Temporal Precedence

### PT Velocity vs RNA Velocity

PT velocity and RNA velocity (scvelo) capture complementary biological signals:

| Dataset | Shared genes | Magnitude Spearman r | Mean cosine similarity |
|---------|-------------|---------------------|----------------------|
| Pancreas | 2,000 | -0.175 | -0.056 |
| Dentate Gyrus | 2,000 | 0.138 | -0.170 |

Low correlation and near-zero cosine similarity confirm that post-transcriptional and transcriptional velocity capture independent information.

### Temporal Precedence: Gamma Changes Precede Expression Changes

Along developmental pseudotime (diffusion pseudotime), gamma changes occur before expression changes for a significant majority of transition genes:

| Dataset | Transition genes | Gamma leads | Expr leads | Simultaneous | Binomial p |
|---------|-----------------|-------------|-----------|--------------|-----------|
| Pancreas | 188 | **119 (63%)** | 58 (31%) | 11 (6%) | **5.29e-06** |
| Dentate Gyrus | 146 | **114 (78%)** | 30 (21%) | 2 (1%) | **9.93e-13** |

Cross-correlation analysis confirms the temporal ordering:

| Dataset | Mean lag (bins) | Median lag (bins) | Positive lag fraction |
|---------|----------------|------------------|---------------------|
| Pancreas | 2.27 | 4.0 | 55.3% |
| Dentate Gyrus | 2.09 | 3.0 | 55.5% |

Positive lag = gamma change precedes expression change. This supports the hypothesis that post-transcriptional regulation acts as an early signal during cell fate transitions.

---

## Aim 4: RBP-Target Regulatory Networks

Spearman correlation between RBP expression and target gene gamma identifies putative post-transcriptional regulatory networks. Edges filtered at FDR < 0.05.

### Pancreas

- **1,112 significant edges** (54 RBPs, 200 target genes)
- Top RBP hubs:

| RBP | Total targets | Stabilizing | Destabilizing |
|-----|-------------|-------------|--------------|
| Hnrnpa1 | 166 | 64 | 102 |
| Ybx1 | 158 | 20 | 138 |
| Srsf3 | 144 | 40 | 104 |
| Rbfox3 | 70 | 33 | 37 |
| Hnrnpd | 48 | 3 | 45 |
| Tra2b | 47 | 10 | 37 |
| Elavl1 (HuR) | 43 | 14 | 29 |
| Fus | 32 | 4 | 28 |
| Zfp36l1 (TTP) | 30 | 4 | 26 |

### Dentate Gyrus

- **3,050 significant edges** (30 RBPs, 200 target genes)
- Top RBP hubs:

| RBP | Total targets | Stabilizing | Destabilizing |
|-----|-------------|-------------|--------------|
| Ybx1 | 550 | 109 | 441 |
| Rbfox1 | 230 | 50 | 180 |
| Celf2 | 214 | 44 | 170 |
| Hnrnpa1 | 209 | 21 | 188 |
| Rbfox3 | 197 | 39 | 158 |
| Elavl3 | 194 | 35 | 159 |
| Rbfox2 | 143 | 17 | 126 |
| Matr3 | 134 | 17 | 117 |
| Elavl1 (HuR) | 129 | 5 | 124 |
| Mbnl2 | 118 | 17 | 101 |

Biologically meaningful: Elavl1/HuR is a well-characterized mRNA stability factor. Zfp36l1 is a TTP-family destabilizing factor. Rbfox proteins are neuron-specific splicing/stability regulators prominent in dentate gyrus.

---

## Pipeline Statistics

| Parameter | Pancreas | Dentate Gyrus | sci-fate |
|-----------|----------|---------------|---------|
| Beta median | 1.093 | 0.882 | 0.614 |
| Gamma median of medians | 0.000 | 0.000 | 0.174 |
| Gamma max | 1,340 | 1,133 | 291 |
| TF score median | 0.033 | 0.030 | 0.658 |
| Genes with TF > 0.5 | 3,487 | 1,332 | 4,662 |

---

## Analysis Scripts

| Script | Description |
|--------|-------------|
| `analyses/run_all.py` | Full pipeline on pancreas dataset |
| `analyses/run_dentate_gyrus.py` | Full pipeline on dentate gyrus dataset |
| `analyses/run_scifate.py` | sci-fate metabolic labeling validation |
| `analyses/run_gaps.py` | Invisible states, velocity comparison, network inference |
| `analyses/run_summary.py` | Cross-dataset validation summary |
| `analyses/run_precedence.py` | Temporal precedence analysis |

All figures saved to `output/` subdirectories. 53 tests passing.

---

## References

- Cao, J. et al. (2020). Sci-fate characterizes the dynamics of gene expression in single cells. *Nature Biotechnology*, 38, 980-988.
- Herzog, V.A. et al. (2017). Thiol-linked alkylation of RNA to assess expression dynamics. *Nature Methods*, 14, 1198-1204.
- Schofield, J.A. et al. (2018). TimeLapse-seq: adding a temporal dimension to RNA sequencing through nucleoside recoding. *Nature Methods*, 15, 221-225.
- Bergen, V. et al. (2020). Generalizing RNA velocity to transient cell states through dynamical modeling. *Nature Biotechnology*, 38, 1408-1414.
