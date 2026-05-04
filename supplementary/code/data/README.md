# Datasets

All datasets are public, third-party, and fetched on first use via
[Pooch](https://www.fatiando.org/pooch/) with pinned SHA-256 hashes.
The registry lives at `src/scptr/datasets/_registry.py`. No data is
shipped inside this supplementary archive.

## Automatic download (default)

Running any of the analysis scripts or reproduction notebooks triggers
the download. By default Pooch caches under
`~/.cache/scptr/` (Linux/macOS) or `%LOCALAPPDATA%\scptr\` (Windows).

```python
import scptr
adata = scptr.datasets.pancreas()         # downloads on first call
hl    = scptr.datasets.schofield2018_halflives()
```

## Manual download / offline use

If your reproduction environment has no internet access, download the
files below to the cache directory above (or set `SCPTR_DATA_DIR`
to a local path before importing `scptr`).

### Single-cell expression datasets

| Dataset | Source | URL | Size | Licence |
|---|---|---|---|---|
| Pancreatic endocrinogenesis | Bastidas-Ponce et al. 2019, *Development* | https://figshare.com/ndownloader/files/24464755 | 32 MB | CC-BY-4.0 |
| Dentate gyrus neurogenesis | Hochgerner et al. 2018, *Nat Neurosci* | https://github.com/theislab/scvelo_notebooks/raw/master/data/DentateGyrus/10X43_1.loom | 25 MB | CC-BY-4.0 |
| A549 dexamethasone (sci-fate) | Cao et al. 2020, *Nat Biotechnol* | GEO: GSE141834 | 65 MB | Original GEO terms |
| Neuroblastoma | Dong et al. 2020, *Cancer Cell* | GEO: GSE137804 | 180 MB | Original GEO terms |

### Reference half-life catalogues

| Dataset | Source | Size | Licence |
|---|---|---|---|
| Schofield 2018 | TimeLapse-seq, mouse 3T3 fibroblasts | 0.4 MB | CC-BY-4.0 |
| Herzog 2017 | SLAM-seq, mouse ES cells | 0.6 MB | CC-BY-4.0 |

Both catalogues are bundled into the Pooch registry with stable mirror
URLs.

### miRNA target predictions

TargetScan 8.0 predictions (mouse-mm10) must be downloaded manually
from the TargetScan website due to their distribution policy:

1. Visit https://www.targetscan.org/vert_80/vert_80_data_download/
2. Download `Summary_Counts.default_predictions.txt.zip` (~30 MB).
3. Extract `Summary_Counts.default_predictions.txt` to either
   `~/.cache/scptr/targetscan/` or the project-local
   `.cache/targetscan/`.

If the file is not present, `scripts/aim1_benchmarking/run_enrichment_analysis.py`
falls back to the documented summary statistics from the paper.

### eCLIP RBP catalogue (optional, for §4.4)

A subset of ENCODE eCLIP peaks is fetched on demand by
`scripts/download_eclip.py`. The full catalogue is large (~12 GB);
the script downloads only the RBPs that appear as inferred hubs in
the paper.

## Reproducing per-dataset claims

| Claim | Dataset(s) | Script |
|---|---|---|
| Spearman ρ = −0.81 on sci-fate | sci-fate | `scripts/aim1_benchmarking/run_halflife_validation.py` |
| Spearman ρ = −0.40 on Schofield (pancreas) | Pancreas + Schofield | `scripts/aim1_benchmarking/run_halflife_validation.py` |
| 126 / 215 miRNA families significant | Pancreas + TargetScan 8.0 | `scripts/aim1_benchmarking/run_enrichment_analysis.py` |
| Epsilon silhouette γ-space 0.215 vs expr −0.011 | Pancreas | `scripts/aim2_hidden_states/run_pt_states_pancreas.py` |
| 54% of 1{,}472 transition genes have γ leading | Pancreas | `scripts/aim3_pt_velocity/run_velocity_pancreas.py` |
| 78% / 146 dentate-gyrus genes | Dentate gyrus | `scripts/aim3_pt_velocity/run_velocity_dentate_gyrus.py` |
| Neuroblastoma 66% stabilising edges | Neuroblastoma | `scripts/aim4_cancer/run_network_inference.py` |
| A549 ρ = −0.78 with sci-fate matched half-lives | A549 + sci-fate | `notebooks/fig2_validation.ipynb` |

## Verifying a clean fetch

After running any script, hash-verify the cached files:

```bash
python -c "import scptr.datasets as d; d.verify_cache()"
```

The function recomputes SHA-256 for every cached file and prints
`OK` / `MISMATCH` per file.
