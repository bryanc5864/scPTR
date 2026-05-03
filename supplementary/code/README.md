# scPTR — anonymized supplementary code

This directory contains the code accompanying the NeurIPS 2026 submission
"scPTR: Decomposing Post-Transcriptional Regulation at Single-Cell Resolution".
All author and institutional information has been removed for double-blind
review.

## Install

```bash
pip install .                 # core
pip install ".[deep]"         # PyTorch / DeepPTR
pip install ".[datasets]"     # Pooch dataset loaders
pip install ".[dev]"          # pytest
```

## Reproducing paper figures

Notebooks in `repro/` regenerate every figure in the main paper end-to-end.
Random seed `42` is fixed throughout. Datasets are fetched on first use via
Pooch; the manifest in `src/scptr/datasets/` lists hashes.

| Figure | Notebook | Approx. runtime |
|---|---|---|
| Fig. 1 (method overview) | `repro/fig1_method.ipynb` | 2 min |
| Fig. 2 (validation)      | `repro/fig2_validation.ipynb` | 8 min |
| Fig. 3 (findings)        | `repro/fig3_findings.ipynb` | 12 min |

## Layout

```
src/scptr/      core library (preprocessing, kinetics, plotting, deep)
tests/          pytest suite (24 modules)
analyses/       full benchmark scripts grouped by aim (aim1=halflife,
                aim2=PT states, aim3=PT velocity, aim4=cancer; deep/
                holds all DeepPTR ablations)
repro/          notebooks regenerating each main-paper figure end-to-end
                  fig1_method.ipynb       (~2 min)
                  fig2_validation.ipynb   (~8 min)
                  fig3_findings.ipynb     (~12 min)
                + the original computation scripts they call:
                  compute_fig1c_data.py, compute_real_figure_data.py,
                  generate_figures.py
```

## Datasets

Datasets are fetched on first use via Pooch from public repositories;
hashes are pinned in `src/scptr/datasets/`. Half-life references:
Schofield 2018 (mouse 3T3) and Herzog 2017 (SLAM-seq); the sci-fate
metabolic-labelling validation uses Cao et al. 2020.

## License

MIT (anonymous; full license text on acceptance).
