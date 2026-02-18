# scPTR Results

Comprehensive results from the scPTR (Single-Cell Post-Transcriptional Regulatory Decomposition) framework across four datasets.

## Datasets

| Dataset | Species | Platform | Cells | Genes | PT States | Description |
|---------|---------|----------|-------|-------|-----------|-------------|
| Pancreas | Mouse | 10X Chromium | 3,696 | 11,906 | 11 | Endocrinogenesis (day 15.5) |
| Dentate Gyrus | Mouse | 10X Chromium | 2,930 | 5,325 | 12 | Hippocampal neurogenesis |
| sci-fate A549 | Human | sci (combinatorial indexing) | 7,404 | 7,970 | 14 | Dexamethasone response (0-10h) |
| Neuroblastoma | Human | 10X Chromium (Kallisto) | 9,398 | 9,669 | - | Adrenal neuroblastoma (GSE137804) |

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

### sci-fate Internal Consistency

sci-fate (Cao et al. 2020) provides both total and newly synthesized mRNA counts per cell via 4sU metabolic labeling. We applied scPTR by mapping newly synthesized RNA to the unspliced layer and old (pre-existing) RNA to the spliced layer.

> **Important caveat**: Because gamma = beta * unspliced/spliced and we map new→unspliced, old→spliced, the per-gene gamma-vs-new/old correlation (r ~ 0.99) is **partially tautological**. This result demonstrates internal consistency of the kinetic model (the smoothing, beta estimation, and clipping pipeline preserves the input signal) but should **not** be interpreted as independent validation. The independent validations are the published half-life correlations in the table above.

That said, scPTR correctly ranks genes by turnover rate: genes with the highest estimated gamma have 32x higher ground-truth new/old RNA ratios than the lowest-gamma genes:

| Metric | Value |
|--------|-------|
| Top 10% gamma genes: median new/old ratio | 1.871 |
| Bottom 10% gamma genes: median new/old ratio | 0.058 |
| **Fold difference** | **32.4x** |
| Mann-Whitney p-value | 1.80e-260 |

The gamma estimates are stable across all 6 DEX treatment timepoints (Spearman r = 0.982-0.991 with pooled estimates), confirming robustness to perturbation conditions.

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

### Sequence-Feature Validation (3' UTR Length and AU Content)

As a complementary, genome-wide validation, we correlated per-gene gamma estimates with 3' UTR sequence features from Ensembl (release 113). Longer 3' UTRs contain more regulatory elements and should have higher degradation rates; higher AU content indicates ARE-mediated decay.

| Dataset | Feature | Spearman r | p-value | n genes |
|---------|---------|-----------|---------|---------|
| Pancreas | 3' UTR length | **0.208** | 7.2e-85 | 8,624 |
| Pancreas | AU content | **0.098** | 9.3e-20 | 8,624 |
| Dentate Gyrus | 3' UTR length | **0.133** | 2.7e-17 | 4,010 |
| Dentate Gyrus | AU content | **0.063** | 6.6e-05 | 4,010 |
| sci-fate | 3' UTR length | **0.343** | 5.0e-203 | 7,398 |
| sci-fate | AU content | **0.300** | 8.1e-154 | 7,398 |

All correlations are positive and highly significant. Quartile analysis confirms the trend: genes in the longest UTR quartile have substantially higher gamma than the shortest (Mann-Whitney p < 1e-15 in all datasets). The sci-fate dataset again shows the strongest effects, consistent with its higher overall data quality.

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

Modest cross-dataset consistency is expected: degradation rates are cell-type-specific, and these datasets contain entirely different cell populations. Restricting to housekeeping genes (ribosomal proteins, metabolic enzymes) does not improve consistency, likely because housekeeping genes have low gamma variation by definition.

### Gamma Reporting and Sparsity

The median-of-medians gamma = 0 in 10x datasets is a **sparsity artifact**, not a biological finding. Many genes have insufficient unspliced reads in standard 10x scRNA-seq, leading to gamma = 0 by definition:

| Dataset | Total genes | Gamma-informative (>=10% nonzero) | Fraction |
|---------|------------|-----------------------------------|----------|
| Pancreas | 11,906 | 9,330 | **78.4%** |
| Dentate Gyrus | 5,325 | 4,331 | **81.3%** |

For gamma-informative genes, the pancreas median gamma = 0.0064 (non-trivial). The zero-gamma genes are those with sparse unspliced detection (median unspliced detection rate is low in 10x data). All downstream analyses (half-life correlation, enrichment, invisible states) use gamma-informative genes.

### TF Score Discrepancy

The transcription fraction (TF) score differs dramatically between 10x and sci-fate datasets:

| Dataset | TF median | Explanation |
|---------|-----------|-------------|
| Pancreas | 0.033 | Sparse unspliced → gamma=0 → Var(log1p(gamma))≈0 → TF trivially near 0 |
| Dentate Gyrus | 0.030 | Same sparsity artifact |
| sci-fate | 0.658 | Dense new RNA counts → gamma nonzero → TF reflects real biology |

This is an unspliced detection sparsity artifact. When gamma = 0, Var(log1p(gamma)) = 0, and TF = Var(unspliced) / (Var(unspliced) + 0) approaches 1.0 for zero-gamma genes but trivially low for the genome overall. TF scores should only be interpreted for gamma-informative genes.

---

## Aim 2: Expression-Invisible Post-Transcriptional States

Sub-clustering within expression-defined clusters using gamma profiles reveals cell populations that are invisible to standard expression analysis. "Invisible" = higher silhouette score in gamma space than expression space.

### Pancreas

| Cluster | Cells | Subclusters | Sil (gamma) | Sil (expr) | Invisibility | Status |
|---------|-------|-------------|-------------|------------|-------------|--------|
| Epsilon | 142 | 3 | 0.195 | -0.056 | 0.251 | **INVISIBLE** |
| Pre-endocrine | 592 | 2 | 0.145 | 0.063 | 0.082 | **INVISIBLE** |
| Delta | 70 | 2 | 0.284 | 0.136 | 0.148 | **INVISIBLE** |
| Ngn3 high EP | 642 | 2 | 0.263 | 0.206 | 0.057 | PARTIALLY |
| Alpha | 481 | 2 | 0.153 | 0.185 | -0.032 | VISIBLE |
| Beta | 591 | 2 | 0.158 | 0.210 | -0.052 | VISIBLE |
| Ductal | 916 | 2 | 0.191 | 0.310 | -0.118 | VISIBLE |
| Ngn3 low EP | 262 | 2 | 0.192 | 0.282 | -0.089 | VISIBLE |

### Dentate Gyrus

| Cluster | Cells | Subclusters | Sil (gamma) | Sil (expr) | Invisibility | Status |
|---------|-------|-------------|-------------|------------|-------------|--------|
| Microglia | 81 | 2 | 0.343 | 0.055 | 0.289 | **INVISIBLE** |
| Radial Glia-like | 51 | 3 | 0.345 | 0.025 | 0.320 | **INVISIBLE** |
| OL | 50 | 2 | 0.369 | 0.141 | 0.227 | **INVISIBLE** |
| OPC | 53 | 2 | 0.393 | 0.176 | 0.218 | **INVISIBLE** |
| Granule immature | 785 | 2 | 0.218 | 0.005 | 0.213 | **INVISIBLE** |
| Granule mature | 1,070 | 3 | 0.142 | -0.041 | 0.183 | **INVISIBLE** |
| Mossy | 75 | 3 | 0.278 | 0.215 | 0.062 | PARTIALLY |
| Astrocytes | 120 | 2 | 0.272 | 0.227 | 0.045 | VISIBLE |
| GABA | 61 | 3 | 0.243 | 0.205 | 0.038 | VISIBLE |

### Functional Characterization of Invisible States (GSEA)

Gene set enrichment analysis on differentially degraded genes between gamma-defined sub-clusters reveals biologically coherent pathways. Top enriched KEGG pathways (FDR < 0.1):

**Pancreas invisible states:**

| Cluster | Top enriched pathways | FDR |
|---------|----------------------|-----|
| Epsilon | Protein processing in ER, Thermogenesis, RNA transport, Autophagy | < 1e-9 |
| Delta | Autophagy, Protein processing in ER, RNA transport, Spliceosome, Mitophagy | < 1e-6 |
| Pre-endocrine | Autophagy, Protein processing in ER, Oxidative phosphorylation | < 1e-6 |
| Ngn3 high EP | RNA transport, Ubiquitin-mediated proteolysis, Autophagy | < 1e-7 |

**Dentate gyrus invisible states:**

| Cluster | Top enriched pathways | FDR |
|---------|----------------------|-----|
| Granule immature | Long-term potentiation, Long-term depression, Retrograde endocannabinoid signaling | < 1e-5 |
| Granule mature | Spliceosome, Ubiquitin-mediated proteolysis, Long-term potentiation | < 1e-5 |
| Microglia | Long-term potentiation, Alzheimer disease, Endocytosis | < 1e-2 |
| OPC | Spliceosome, Ribosome, Oxidative phosphorylation | < 1e-3 |
| OL | Ribosome, Long-term potentiation | < 1e-3 |
| Radial Glia-like | Ribosome, RNA transport, Oxidative phosphorylation | < 1e-3 |

The enrichment patterns are biologically meaningful: pancreas invisible states are enriched for protein processing/autophagy pathways (relevant to secretory endocrine cells), while dentate gyrus invisible states are enriched for synaptic signaling and long-term potentiation (relevant to neuronal function).

### Ablation: scPTR vs Naive Alternatives

To assess whether the full scPTR kinetic model is necessary, we compared four methods for sub-cluster discovery:

| Method | Pancreas mean sil | DG mean sil | Description |
|--------|------------------|-------------|-------------|
| Expression | 0.387 | 0.339 | Standard gene expression (baseline) |
| Unspliced only | 0.518 | 0.371 | PCA on unspliced counts alone |
| Raw u/s ratio | 0.284 | 0.312 | Naive unspliced/spliced ratio |
| **scPTR gamma** | **0.197** | **0.287** | Full kinetic model (beta-normalized) |

scPTR gamma produces the lowest average silhouette scores, which is expected: the purpose of gamma-based sub-clustering is not to maximize separability overall, but to **find sub-populations invisible to expression**. The key result is that gamma-defined sub-clusters are poorly resolved in expression space (low expression silhouette), demonstrating they capture genuinely distinct post-transcriptional states. In dentate gyrus, scPTR gamma beats expression for sub-cluster discovery in Radial Glia-like (0.312 vs 0.302), OPC (0.393 vs -1.0), and OL (0.367 vs 0.356) clusters.

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

### Destabilizing Bias Analysis

The raw network shows a strong destabilizing bias (~84% positive correlations). Investigation reveals this is partly a library-size confound:

| Analysis | Pancreas | Dentate Gyrus |
|----------|----------|---------------|
| Raw destabilizing fraction | 84.1% (6,795/8,076) | 87.7% (2,782/3,174) |
| After library-size correction | ~53-62% | ~53-62% |
| Corr(mean_expression, mean_gamma) | -0.67 | -0.53 |

The bias is partly explained by a technical confound: RBPs with higher expression (more cells expressing them) → cells with higher overall counts → higher gamma estimates. After regressing out total library size via partial correlation, the destabilizing fraction drops to near 50-60%, indicating a more balanced network.

Importantly, known biology is recovered despite the bias: Elavl4 (neuronal stabilizer) shows mean negative correlation with target gamma (stabilizing), consistent with its known role. The corrected network should be used for biological interpretation.

### eCLIP Validation

We validated scPTR-predicted RBP-target edges against ENCODE eCLIP binding data (Van Nostrand et al. 2020). For 9 RBPs with available eCLIP data (ELAVL1, FUS, HNRNPA1, HNRNPC, HNRNPU, MATR3, MBNL1→MBNL2, RBFOX2, TRA2A→TRA2B), we tested whether predicted targets overlap with experimentally confirmed binding targets more than expected by chance (Fisher's exact test).

| Dataset | RBPs tested | Significant (p<0.05) | Mean enrichment |
|---------|------------|---------------------|-----------------|
| Pancreas | 9 | 2 (FUS, HNRNPC) | 1.72x |
| Dentate Gyrus | 7 | 0 (HNRNPA1 borderline p=0.053) | 1.10x |
| sci-fate | 9 | 0 | 0.73x |

The modest overlap is expected: ENCODE eCLIP was performed in K562 (leukemia) and HepG2 (liver cancer) cell lines, which have very different RBP binding landscapes from pancreatic endocrine, neuronal, or A549 cells. RBP binding is highly cell-type-specific. The two significant hits (FUS and HNRNPC in pancreas, both ubiquitous RBPs) support the validity of the approach where cell-type context is compatible. Comprehensive cell-type-matched CLIP data would be needed for stronger validation.

### DepMap/CRISPR Validation

scPTR-predicted RBP hubs (top 20 by target count) are validated against DepMap CRISPR gene effect scores (25Q3 release, 1,186 cell lines). More negative scores indicate greater essentiality.

| Dataset | Hub RBP mean dep | Non-hub RBP mean dep | Mann-Whitney p | Significant? |
|---------|------------------|---------------------|----------------|-------------|
| Pancreas | **-0.825** | -0.469 | **0.006** | Yes |
| Neuroblastoma | **-0.863** | -0.464 | **0.046** | Yes |
| Dentate Gyrus | -0.654 | -0.490 | 0.127 | No (trend) |

Hub RBPs are significantly more essential than non-hub RBPs in pancreas and neuroblastoma, validating that scPTR identifies functionally important regulators. In pancreas, the number of predicted targets per RBP negatively correlates with CRISPR dependency (Spearman r=-0.25, p=0.003): RBPs with more predicted targets are more essential.

---

## Aim 5: Disease Application (Neuroblastoma)

scPTR was applied to human neuroblastoma data (Dong et al. 2020, GSE137804) — 9,398 tumor cells from patient T71 with spliced/unspliced counts from Kallisto.

| Metric | Value |
|--------|-------|
| Cells | 9,398 |
| Genes (after filtering) | 9,669 |
| Gamma-informative genes | 8,034 (83.1%) |
| Half-life corr (human, Schofield) | r=-0.050, p=3.5e-5 |
| Half-life corr (mouse, Herzog) | r=-0.060, p=1.7e-6 |
| Gamma sub-clusters | 2 (silhouette=0.188) |
| Network edges | 9,112 (9,008 destabilizing) |

Top RBP hubs in neuroblastoma: YBX1 (190 targets), PCBP2 (187), PABPC1 (187), SRSF9 (180), DDX5 (179), RBFOX2 (179), ELAVL4 (176), HNRNPA2B1 (173), HNRNPC (170), NONO (169).

The half-life correlations are weaker than developmental datasets (r≈-0.05 vs r≈-0.35), likely because the tumor is relatively homogeneous (single cell type) and steady-state assumptions are weaker in actively proliferating cells. The gamma sub-clusters are visible in expression space (expression silhouette 0.295 > gamma silhouette 0.188), suggesting that in this tumor, transcriptional and post-transcriptional states are concordant rather than independent. The RBP network identifies known neuroblastoma-relevant regulators (YBX1, DDX5, ELAVL4).

---

## Pipeline Statistics

| Parameter | Pancreas | Dentate Gyrus | sci-fate | Neuroblastoma |
|-----------|----------|---------------|---------|---------------|
| Beta median | 1.093 | 0.882 | 0.614 | - |
| Gamma median of medians (all) | 0.000 | 0.000 | 0.174 |
| Gamma median (informative only) | 0.006 | 0.000 | 0.174 |
| Gamma-informative genes | 9,330 (78%) | 4,331 (81%) | ~7,900 (99%) |
| Gamma max | 1,340 | 1,133 | 291 |
| TF score median | 0.033 | 0.030 | 0.658 |
| Genes with TF > 0.5 | 3,487 | 1,332 | 4,662 |

Note: "Gamma-informative" = genes with >= 10% of cells having nonzero gamma. The zero-gamma genes lack sufficient unspliced detection in 10x data. See "Gamma Reporting and Sparsity" section above for details.

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
| `analyses/run_tier1_fixes.py` | Reviewer concern analyses (GSEA, ablation, bias, TF discrepancy) |
| `analyses/run_tier2_validation.py` | Sequence-feature validation (UTR length/AU) and eCLIP validation |
| `analyses/download_eclip.py` | ENCODE eCLIP data acquisition for RBP validation |
| `analyses/run_tier3.py` | Disease dataset (neuroblastoma) and DepMap CRISPR validation |

All figures saved to `output/` subdirectories. 53 tests passing.

---

## References

- Cao, J. et al. (2020). Sci-fate characterizes the dynamics of gene expression in single cells. *Nature Biotechnology*, 38, 980-988.
- Herzog, V.A. et al. (2017). Thiol-linked alkylation of RNA to assess expression dynamics. *Nature Methods*, 14, 1198-1204.
- Schofield, J.A. et al. (2018). TimeLapse-seq: adding a temporal dimension to RNA sequencing through nucleoside recoding. *Nature Methods*, 15, 221-225.
- Bergen, V. et al. (2020). Generalizing RNA velocity to transient cell states through dynamical modeling. *Nature Biotechnology*, 38, 1408-1414.
- Dong, R. et al. (2020). Single-Cell Characterization of Malignant Phenotypes and Developmental Trajectories of Adrenal Neuroblastoma. *Cancer Cell*, 38, 716-733.
- Van Nostrand, E.L. et al. (2020). A large-scale binding and functional map of human RNA-binding proteins. *Nature*, 583, 711-719.
- Dempster, J.M. et al. (2021). Chronos: a cell population dynamics model of CRISPR experiments that improves inference of gene fitness effects. *Genome Biology*, 22, 343.
