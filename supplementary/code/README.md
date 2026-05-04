# scPTR — supplementary code (NeurIPS 2026 anonymous submission)

This directory contains the complete, executable code accompanying the
paper *scPTR: Decomposing Post-Transcriptional Regulation at Single-Cell
Resolution*. All author and institutional information has been removed
for double-blind review. The code is self-contained: every figure,
table, and ablation reported in the paper can be regenerated from this
tree with the commands below.

## 1. Layout

```
code/
├── README.md              <- this file
├── LICENSE                <- MIT (anonymous)
├── requirements.txt       <- pinned dependency versions
├── pyproject.toml         <- pip-installable package definition
│
├── src/scptr/             <- the library
│   ├── preprocessing/         filtering, normalisation, kNN smoothing
│   ├── tools/                 β / γ estimation, PT states, PT velocity, networks
│   ├── deep/                  DeepPTR generative model (PyTorch)
│   ├── benchmark/             half-life / miRNA / RBP benchmarks
│   ├── plotting/              figure-style helpers
│   ├── datasets/              Pooch-backed dataset loaders
│   └── readwrite.py           AnnData I/O helpers
│
├── tests/                 <- pytest suite (24 modules)
│
├── scripts/               <- analysis scripts grouped by paper claim
│   ├── aim1_benchmarking/     §4.2: half-life validation, miRNA enrichment, robustness
│   ├── aim2_hidden_states/    §4.3: PT-state discovery + invisibility scores
│   ├── aim3_pt_velocity/      §4.3: PT velocity + temporal precedence
│   ├── aim4_cancer/           §4.4: RBP network on neuroblastoma
│   ├── deep/                  §4.5: DeepPTR ablations (numbered 01-35)
│   ├── _common.py             shared utilities (paths, plot styles)
│   └── download_eclip.py      fetcher for the eCLIP RBP catalogue
│
├── notebooks/             <- per-figure reproduction notebooks
│   ├── fig1_method.ipynb         §3 / Fig 1
│   ├── fig2_validation.ipynb     §4.2 / Fig 2
│   ├── fig3_findings.ipynb       §4.3 / Fig 3
│   └── compute_*.py / generate_figures.py  (notebook backends)
│
└── data/
    └── README.md          <- dataset access (Pooch + manual links)
```

## 2. Install

Tested with Python 3.10 and 3.11 on Windows 11 and Linux. A clean conda
or venv environment is recommended.

```bash
python -m venv .venv && source .venv/bin/activate   # or: conda create -n scptr python=3.11
pip install -r requirements.txt
pip install -e .                                    # makes "import scptr" work
```

Optional: `pip install ".[deep]"` to add a CUDA-enabled PyTorch for the
DeepPTR scripts; everything in §4.1–§4.4 works on CPU only.

## 3. Reproduce the paper

The three main-text figures are reproduced end-to-end by the notebooks
in `notebooks/`. Random seed `42` is fixed throughout.

| Paper element | Command | Approx. runtime (CPU) |
|---|---|---|
| Figure 1 (method overview) | `jupyter execute notebooks/fig1_method.ipynb` | ~2 min |
| Figure 2 (half-life validation, miRNA enrichment) | `jupyter execute notebooks/fig2_validation.ipynb` | ~8 min |
| Figure 3 (Epsilon PT states, temporal precedence) | `jupyter execute notebooks/fig3_findings.ipynb` | ~12 min |
| Table 2 (half-life Spearman vs scVelo / velVI / UniTVelo) | `python scripts/aim1_benchmarking/run_halflife_validation.py` | ~3 min |
| Table 3 (runtime scaling) | `python scripts/aim1_benchmarking/run_robustness.py` | ~6 min |
| Table 4 (DeepPTR ablations) | `bash scripts/deep/run_all.sh` | ~2 h, GPU recommended |
| §4.3 hidden states (full hippocampus + pancreas) | `python scripts/aim2_hidden_states/run_pt_states_pancreas.py` | ~5 min |
| §4.3 temporal precedence (pancreas + dentate gyrus) | `python scripts/aim3_pt_velocity/run_velocity_pancreas.py` | ~4 min |
| §4.4 RBP network (pancreas + neuroblastoma) | `python scripts/aim4_cancer/run_network_inference.py` | ~10 min |

The notebook backends (`notebooks/compute_real_figure_data.py`,
`compute_fig1c_data.py`, `generate_figures.py`) write intermediate
arrays into `real_figure_data/` and rendered PDFs into `figures/`; both
folders are created on first run.

## 4. Datasets

All datasets used in the paper are public and fetched on first use via
[Pooch](https://www.fatiando.org/pooch/) with pinned hashes. See
`data/README.md` for direct links, sizes, licences, and instructions
for offline / manual download.

| Dataset | Cells | Size | Used in |
|---|---|---|---|
| Pancreatic endocrinogenesis (Bastidas-Ponce 2019) | 3{,}696 | 32 MB | §4.2-§4.4, all main figures |
| Dentate gyrus neurogenesis (Hochgerner 2018) | 2{,}930 | 25 MB | §4.3 (PT velocity) |
| A549 dexamethasone response (Cao 2020 / sci-fate) | 7{,}404 | 65 MB | §4.4 (cross-platform) |
| Neuroblastoma (Dong 2020) | 19{,}723 | 180 MB | §4.4 (oncogenic RBPs) |
| Schofield 2018 mRNA half-lives (SLAM-seq, mouse 3T3) | 4{,}308 genes | 0.4 MB | §4.2 (reference) |
| Herzog 2017 mRNA half-lives (SLAM-seq) | 11{,}979 genes | 0.6 MB | §4.2 (reference) |
| TargetScan 8.0 miRNA target predictions (mouse) | 215 families | 30 MB | §4.2 (miRNA enrichment) |

Total dataset footprint: ~330 MB. All raw data is third-party,
redistributed under the original licences listed in `data/README.md`.

## 5. Tests

A pytest suite covers the estimator, smoothing, network inference,
DeepPTR likelihood, and plotting:

```bash
pytest tests/ -q                # full suite (~3 min)
pytest tests/ -q -m "not slow"  # fast subset (~30 s)
```

## 6. Reproducibility checklist

- All experiments use random seed **42**, set via `numpy`, `random`,
  `torch`, and `scanpy` global seeds inside each entry script.
- Datasets are pinned by Pooch hash; the registry lives at
  `src/scptr/datasets/_registry.py`.
- DeepPTR training is deterministic on CPU and on a single GPU when
  `torch.use_deterministic_algorithms(True)` is set (default in our
  scripts; impacts speed by <10%).
- Hyperparameters and architectures are listed in Section A of the
  paper and in the docstring of every entry script.

## 7. Licence

MIT (anonymous; the full attribution text will be released alongside
the camera-ready paper if accepted). Third-party datasets retain their
original licences.

## 8. Anonymity statement

This release was generated by `supplementary/build.sh`, which copies
the live source tree, overlays anonymous `README.md` / `LICENSE`
files, and aborts if any author-identifying string (e.g. real name,
GitHub handle, institutional email) is detected. Reviewers are
encouraged to report any anonymity violations they find.
