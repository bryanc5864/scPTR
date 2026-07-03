# scPTR Web Interface

Interactive web application for single-cell post-transcriptional regulatory decomposition.

## Run locally

```bash
pip install streamlit
streamlit run web/app.py
```

Opens at `http://localhost:8501`

## Deploy (Streamlit Community Cloud)

1. Push repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo, set main file to `web/app.py`
4. Deploy

## Workflow

1. **Load Data** — upload .h5ad or use built-in pancreas / dentate gyrus datasets
2. **Preprocess** — filter genes, normalize, build kNN graph, smooth layers
3. **Estimate Rates** — β via quantile regression, γ = β · u / s per cell
4. **Discover PT States** — Leiden clustering in γ-space; PT velocity; RBP networks
5. **Results** — UMAP visualization, gene rankings, download outputs
