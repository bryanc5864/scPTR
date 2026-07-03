"""scPTR web application — single-cell post-transcriptional regulatory decomposition."""

from __future__ import annotations

import io
import tempfile
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import streamlit as st

warnings.filterwarnings("ignore")

# ── page config (must be first) ───────────────────────────────────────────────
st.set_page_config(
    page_title="scPTR",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── reset & base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #f4f4f4;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 14px;
    color: #1a1a1a;
}

/* hide default chrome */
#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #d4d4d4;
    padding: 0;
}
[data-testid="stSidebar"] > div:first-child { padding: 0; }

/* ── main content padding ── */
.block-container {
    padding: 2rem 2.5rem 2rem 2.5rem !important;
    max-width: 1100px;
}

/* ── cards / panels ── */
.scptr-card {
    background: #ffffff;
    border: 1px solid #d4d4d4;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
}
.scptr-card-header {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #888888;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #ebebeb;
}

/* ── metric grid ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1px;
    background: #d4d4d4;
    border: 1px solid #d4d4d4;
    margin-bottom: 1.25rem;
}
.metric-cell {
    background: #ffffff;
    padding: 1rem 1.25rem;
}
.metric-label {
    font-size: 11px;
    color: #888888;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.25rem;
}
.metric-value {
    font-size: 22px;
    font-weight: 600;
    color: #1a1a1a;
    font-variant-numeric: tabular-nums;
    font-family: "SF Mono", "Fira Code", Consolas, monospace;
}
.metric-sub {
    font-size: 11px;
    color: #aaaaaa;
    margin-top: 0.15rem;
}

/* ── step nav in sidebar ── */
.step-nav {
    padding: 1.5rem 0 0 0;
}
.step-nav-title {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #1a1a1a;
    padding: 0 1.5rem 1rem 1.5rem;
    border-bottom: 1px solid #ebebeb;
    margin-bottom: 0.5rem;
}
.step-nav-sub {
    font-size: 11px;
    color: #aaaaaa;
    font-weight: 400;
    letter-spacing: 0;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 1.5rem;
    cursor: default;
    border-left: 3px solid transparent;
}
.step-item.active {
    background: #f0f4f9;
    border-left: 3px solid #2b5797;
}
.step-item.done {
    opacity: 0.7;
}
.step-num {
    width: 22px;
    height: 22px;
    border: 1.5px solid #c0c0c0;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 600;
    color: #888;
    flex-shrink: 0;
}
.step-item.active .step-num {
    background: #2b5797;
    border-color: #2b5797;
    color: #fff;
}
.step-item.done .step-num {
    background: #e8f0e8;
    border-color: #4a9a5a;
    color: #4a9a5a;
}
.step-label { font-size: 13px; color: #333; }
.step-item.active .step-label { font-weight: 600; color: #1a1a1a; }

/* ── sidebar info block ── */
.sidebar-info {
    margin: 1rem 1.5rem 0 1.5rem;
    padding: 0.75rem 1rem;
    background: #f7f7f7;
    border: 1px solid #e0e0e0;
    font-size: 12px;
    color: #555;
    line-height: 1.6;
}
.sidebar-info strong { color: #1a1a1a; }

/* ── page title ── */
.page-title {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #1a1a1a;
    margin-bottom: 0.25rem;
}
.page-desc {
    font-size: 13px;
    color: #666;
    margin-bottom: 1.5rem;
    line-height: 1.5;
}

/* ── result table ── */
.result-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.result-table th {
    background: #f7f7f7;
    border-bottom: 2px solid #d4d4d4;
    padding: 0.5rem 0.75rem;
    text-align: left;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #666;
    font-weight: 600;
}
.result-table td {
    padding: 0.45rem 0.75rem;
    border-bottom: 1px solid #ebebeb;
    color: #1a1a1a;
    font-family: "SF Mono", "Fira Code", Consolas, monospace;
    font-size: 12px;
}
.result-table tr:last-child td { border-bottom: none; }
.result-table tr:hover td { background: #fafafa; }

/* ── tag ── */
.tag {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    background: #e8eef5;
    color: #2b5797;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.tag.green { background: #e8f0e8; color: #2d7d4b; }
.tag.grey { background: #f0f0f0; color: #666; }

/* ── buttons ── */
.stButton > button {
    background: #2b5797 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.5rem 1.25rem !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
}
.stButton > button:hover { background: #1e3f6e !important; }
.stButton > button:active { background: #163060 !important; }

/* secondary button */
.stDownloadButton > button {
    background: #ffffff !important;
    color: #2b5797 !important;
    border: 1.5px solid #2b5797 !important;
    border-radius: 0 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

/* ── form elements ── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label,
[data-testid="stFileUploader"] label {
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #555 !important;
}
[data-testid="stFileUploader"] {
    border: 1.5px dashed #c0c0c0 !important;
    background: #fafafa !important;
    padding: 0.5rem !important;
}

/* ── tabs ── */
[data-testid="stTabs"] button {
    border-radius: 0 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ── divider ── */
hr { border: none; border-top: 1px solid #e0e0e0; margin: 1.25rem 0; }

/* ── figure ── */
.stImage img {
    border: 1px solid #d4d4d4;
    display: block;
}
</style>
""", unsafe_allow_html=True)

# ── session state ─────────────────────────────────────────────────────────────
for key, default in {
    "step": 1,
    "adata": None,
    "dataset_name": None,
    "preprocessed": False,
    "estimated": False,
    "states_done": False,
    "network_done": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── helpers ───────────────────────────────────────────────────────────────────
def go_to(step: int):
    st.session_state.step = step

def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def metric_card(metrics: list[tuple[str, str, str]]) -> str:
    """metrics = [(label, value, sub), ...]"""
    cells = "".join(
        f'<div class="metric-cell">'
        f'<div class="metric-label">{lbl}</div>'
        f'<div class="metric-value">{val}</div>'
        f'<div class="metric-sub">{sub}</div>'
        f'</div>'
        for lbl, val, sub in metrics
    )
    return f'<div class="metric-grid">{cells}</div>'

def card(header: str, body: str) -> str:
    return (
        f'<div class="scptr-card">'
        f'<div class="scptr-card-header">{header}</div>'
        f'{body}'
        f'</div>'
    )

# ── sidebar ───────────────────────────────────────────────────────────────────
STEPS = [
    "Load Data",
    "Preprocess",
    "Estimate Rates",
    "Discover PT States",
    "Results",
]

with st.sidebar:
    st.markdown(
        '<div class="step-nav">'
        '<div class="step-nav-title">scPTR'
        '<div class="step-nav-sub">Post-transcriptional decomposition</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    for i, label in enumerate(STEPS, 1):
        cls = "active" if i == st.session_state.step else ("done" if i < st.session_state.step else "")
        st.markdown(
            f'<div class="step-item {cls}">'
            f'<div class="step-num">{i}</div>'
            f'<div class="step-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.adata is not None:
        adata = st.session_state.adata
        st.markdown(
            f'<div class="sidebar-info">'
            f'<strong>{st.session_state.dataset_name}</strong><br>'
            f'{adata.n_obs:,} cells &nbsp;·&nbsp; {adata.n_vars:,} genes<br>'
            f'{"✓ preprocessed" if st.session_state.preprocessed else "not preprocessed"}<br>'
            f'{"✓ rates estimated" if st.session_state.estimated else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── main ──────────────────────────────────────────────────────────────────────
step = st.session_state.step

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
if step == 1:
    st.markdown('<div class="page-title">Load Data</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">Upload an AnnData file (.h5ad) with <code>spliced</code> and '
        '<code>unspliced</code> count layers, or start with one of the built-in example datasets.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("**Example datasets**")
        example = st.selectbox(
            "Dataset",
            ["— select —", "Pancreas (mouse, 3,696 cells)", "Dentate Gyrus (mouse, 2,930 cells)"],
            label_visibility="collapsed",
        )
        if st.button("Load example", disabled=example == "— select —"):
            import scptr
            with st.spinner("Loading…"):
                if "Pancreas" in example:
                    adata = scptr.datasets.pancreas()
                    st.session_state.dataset_name = "Pancreas"
                else:
                    adata = scptr.datasets.dentate_gyrus()
                    st.session_state.dataset_name = "Dentate Gyrus"
            st.session_state.adata = adata
            st.session_state.preprocessed = False
            st.session_state.estimated = False
            st.session_state.states_done = False
            st.session_state.network_done = False
            go_to(2)
            st.rerun()

    with col2:
        st.markdown("**Upload your own**")
        uploaded = st.file_uploader("H5AD file", type=["h5ad"], label_visibility="collapsed")
        if uploaded and st.button("Load file"):
            with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
                f.write(uploaded.read())
                tmp_path = f.name
            with st.spinner("Loading…"):
                adata = sc.read_h5ad(tmp_path)
            missing = [l for l in ["spliced", "unspliced"] if l not in adata.layers]
            if missing:
                st.error(f"Missing layers: {', '.join(missing)}. File must have 'spliced' and 'unspliced' layers.")
            else:
                st.session_state.adata = adata
                st.session_state.dataset_name = uploaded.name.replace(".h5ad", "")
                st.session_state.preprocessed = False
                st.session_state.estimated = False
                st.session_state.states_done = False
                st.session_state.network_done = False
                go_to(2)
                st.rerun()

    st.markdown("---")
    st.markdown(
        card("About scPTR",
             '<p style="font-size:13px;color:#444;line-height:1.7;margin:0">'
             'scPTR estimates per-cell, per-gene mRNA degradation rates (γ) from standard scRNA-seq '
             'spliced/unspliced counts using the kinetic steady-state relation '
             '<code>γ = β · u / s</code>, where β is the gene-specific splicing rate. '
             'It uses γ as a primary analytical axis to discover expression-invisible '
             'post-transcriptional cell states, compute post-transcriptional velocity, '
             'and infer RNA-binding protein regulatory networks.'
             '</p>'
             '<p style="font-size:12px;color:#aaa;margin:0.75rem 0 0 0">'
             'Input: AnnData with <code>spliced</code> and <code>unspliced</code> count layers.'
             '</p>'),
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════
elif step == 2:
    import scptr

    st.markdown('<div class="page-title">Preprocess</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">Filter low-abundance genes, normalize counts, build a cell '
        'neighborhood graph, and apply Gaussian kernel smoothing to the spliced/unspliced layers.</div>',
        unsafe_allow_html=True,
    )

    adata = st.session_state.adata
    st.markdown(
        metric_card([
            ("Cells", f"{adata.n_obs:,}", ""),
            ("Genes", f"{adata.n_vars:,}", "before filtering"),
        ]),
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        min_spliced = st.number_input("Min spliced counts per gene", value=10, min_value=1, step=1)
        min_cells = st.number_input("Min cells expressing gene", value=5, min_value=1, step=1)
        n_neighbors = st.number_input("Neighbors (kNN graph)", value=30, min_value=5, max_value=100, step=5)
    with col2:
        n_pcs = st.number_input("PCA components", value=30, min_value=10, max_value=100, step=5)
        bandwidth = st.selectbox("Smoothing bandwidth", ["adaptive (median distance)", "fixed"])
        fixed_bw = None
        if bandwidth == "fixed":
            fixed_bw = st.number_input("Bandwidth value", value=1.0, min_value=0.1, step=0.1)

    if st.button("Run preprocessing"):
        adata_work = st.session_state.adata.copy()
        with st.spinner("Filtering genes…"):
            scptr.pp.filter_genes(adata_work, min_spliced=min_spliced, min_cells=min_cells)
        with st.spinner("Normalizing layers…"):
            scptr.pp.normalize_layers(adata_work)
        with st.spinner("Building neighborhood graph…"):
            scptr.pp.neighbors(adata_work, n_neighbors=n_neighbors, n_pcs=n_pcs)
        with st.spinner("Smoothing spliced/unspliced…"):
            if fixed_bw:
                scptr.pp.smooth_layers(adata_work, bandwidth=fixed_bw)
            else:
                scptr.pp.smooth_layers(adata_work)

        st.session_state.adata = adata_work
        st.session_state.preprocessed = True
        st.session_state.estimated = False
        st.session_state.states_done = False

        st.markdown(
            metric_card([
                ("Cells", f"{adata_work.n_obs:,}", ""),
                ("Genes retained", f"{adata_work.n_vars:,}", "after filtering"),
                ("Neighbors", f"{n_neighbors}", "kNN"),
            ]),
            unsafe_allow_html=True,
        )
        st.success("Preprocessing complete.")

    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("← Back"):
            go_to(1); st.rerun()
    with c2:
        if st.button("Next →", disabled=not st.session_state.preprocessed):
            go_to(3); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ESTIMATE RATES
# ══════════════════════════════════════════════════════════════════════════════
elif step == 3:
    import scptr

    st.markdown('<div class="page-title">Estimate Rates</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">Estimate gene-specific splicing rates (β) via quantile regression '
        'on unspliced/spliced phase portraits, then compute per-cell degradation rates '
        'γ<sub>ig</sub> = β<sub>g</sub> · u<sub>ig</sub> / s<sub>ig</sub>.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        quantile = st.slider("Phase portrait quantile (β)", 0.80, 0.99, 0.95, 0.01,
                             help="Upper quantile of u/s ratios used to estimate splicing rate.")
        min_r2 = st.slider("Min R² for beta fit", 0.0, 0.9, 0.0, 0.05)
    with col2:
        min_spliced_gamma = st.number_input("Min smoothed spliced (Ms) for γ", value=0.01,
                                            min_value=0.001, step=0.005, format="%.3f")
        clip_pct = st.slider("Gamma clip percentile", 90, 100, 99,
                             help="Per-gene clipping at this percentile to remove outliers.")

    if st.button("Estimate β and γ"):
        adata_work = st.session_state.adata
        with st.spinner("Estimating splicing rates (β)…"):
            scptr.tl.estimate_beta(adata_work, quantile=quantile)
        with st.spinner("Estimating degradation rates (γ)…"):
            scptr.tl.estimate_gamma(adata_work, min_spliced=min_spliced_gamma,
                                    clip_percentile=clip_pct)
        with st.spinner("Variance decomposition…"):
            scptr.tl.variance_decomposition(adata_work)

        st.session_state.adata = adata_work
        st.session_state.estimated = True
        st.session_state.states_done = False

        beta = adata_work.var["beta"]
        gamma = adata_work.layers["gamma"]
        gamma_pos = gamma[gamma > 0]

        st.markdown(
            metric_card([
                ("Median β", f"{np.median(beta):.3f}", "splicing rate"),
                ("Median γ", f"{np.median(gamma_pos):.4f}", "positive cells"),
                ("Max γ", f"{np.max(gamma):.2f}", ""),
                ("Informative genes", f"{(np.median(gamma, axis=0) > 0).sum():,}", "median γ > 0"),
            ]),
            unsafe_allow_html=True,
        )

        # Phase portrait for top gene
        top_gene = adata_work.var["beta"].idxmax()
        fig, ax = plt.subplots(figsize=(4, 3.2))
        Ms = adata_work.layers["Ms"][:, adata_work.var_names.get_loc(top_gene)]
        Mu = adata_work.layers["Mu"][:, adata_work.var_names.get_loc(top_gene)]
        ax.scatter(Ms, Mu, s=4, alpha=0.4, color="#2b5797", linewidths=0)
        beta_val = adata_work.var.loc[top_gene, "beta"]
        x = np.linspace(0, Ms.max(), 100)
        ax.plot(x, beta_val * x, color="#c0392b", lw=1.5, label=f"β = {beta_val:.3f}")
        ax.set_xlabel("Ms (smoothed spliced)", fontsize=10)
        ax.set_ylabel("Mu (smoothed unspliced)", fontsize=10)
        ax.set_title(f"Phase portrait: {top_gene}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=9, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.image(fig_to_png(fig), width=440)
        plt.close(fig)

        st.success("Rate estimation complete.")

    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("← Back"):
            go_to(2); st.rerun()
    with c2:
        if st.button("Next →", disabled=not st.session_state.estimated):
            go_to(4); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — DISCOVER PT STATES
# ══════════════════════════════════════════════════════════════════════════════
elif step == 4:
    import scptr

    st.markdown('<div class="page-title">Discover PT States</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">Cluster cells in γ-space (PCA → kNN → Leiden) to reveal '
        'post-transcriptional states invisible to expression analysis. Optionally compute '
        'PT velocity and infer RBP–target networks.</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["PT STATES", "PT VELOCITY", "RBP NETWORKS"])

    with tab1:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            n_pcs_gamma = st.number_input("PCA dims (γ-space)", value=20, min_value=5, max_value=50)
            n_neighbors_gamma = st.number_input("Neighbors (γ-space)", value=15, min_value=5, max_value=50)
        with col2:
            resolution = st.slider("Leiden resolution", 0.1, 2.0, 0.5, 0.1)
            random_state = st.number_input("Random seed", value=42)

        if st.button("Find PT states"):
            adata_work = st.session_state.adata
            with st.spinner("Clustering in γ-space…"):
                scptr.tl.pt_states(
                    adata_work,
                    n_pcs=n_pcs_gamma,
                    n_neighbors=n_neighbors_gamma,
                    resolution=resolution,
                    random_state=random_state,
                )
            st.session_state.adata = adata_work
            st.session_state.states_done = True

            n_states = adata_work.obs["pt_state"].nunique()
            st.markdown(
                metric_card([("PT States", str(n_states), "Leiden clusters in γ-space")]),
                unsafe_allow_html=True,
            )

            # UMAP plot
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
            sc.pl.umap(adata_work, color="pt_state", ax=axes[0], show=False,
                       title="PT states (γ-space)", frameon=False, legend_loc="right margin",
                       size=18)
            if "leiden" in adata_work.obs.columns or "clusters" in adata_work.obs.columns:
                expr_col = "leiden" if "leiden" in adata_work.obs.columns else "clusters"
                sc.pl.umap(adata_work, color=expr_col, ax=axes[1], show=False,
                           title="Expression clusters", frameon=False, size=18)
            else:
                axes[1].set_visible(False)
            plt.tight_layout()
            st.image(fig_to_png(fig), use_container_width=True)
            plt.close(fig)
            st.success(f"Found {n_states} post-transcriptional states.")

    with tab2:
        if not st.session_state.states_done:
            st.info("Run PT state discovery first.")
        else:
            if st.button("Compute PT velocity"):
                adata_work = st.session_state.adata
                with st.spinner("Computing PT velocity…"):
                    scptr.tl.pt_velocity(adata_work)
                st.session_state.adata = adata_work
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                scptr.pl.pt_velocity_embedding(adata_work, ax=ax, show=False)
                ax.set_title("Post-transcriptional velocity", fontsize=11, fontweight="bold")
                plt.tight_layout()
                st.image(fig_to_png(fig), width=520)
                plt.close(fig)
                st.success("PT velocity computed.")

    with tab3:
        if not st.session_state.states_done:
            st.info("Run PT state discovery first.")
        else:
            col1, col2 = st.columns(2, gap="large")
            with col1:
                n_rbps = st.number_input("Max RBPs to test", value=50, min_value=10, max_value=200)
                fdr_threshold = st.number_input("FDR threshold", value=0.05, min_value=0.001,
                                                max_value=0.2, step=0.01, format="%.3f")
            with col2:
                correct_library = st.checkbox("Library-size correction", value=True)

            if st.button("Infer RBP networks"):
                adata_work = st.session_state.adata
                with st.spinner("Running elastic net regression… (this may take a few minutes)"):
                    scptr.tl.infer_network(
                        adata_work,
                        n_rbps=n_rbps,
                        fdr=fdr_threshold,
                        correct_library_size=correct_library,
                    )
                st.session_state.adata = adata_work
                st.session_state.network_done = True

                edges = adata_work.uns.get("network_edges", pd.DataFrame())
                if len(edges):
                    n_sig = len(edges)
                    hubs = (
                        edges.groupby("rbp")
                        .size()
                        .sort_values(ascending=False)
                        .head(10)
                        .reset_index()
                    )
                    hubs.columns = ["RBP", "Target count"]
                    st.markdown(
                        metric_card([("Significant edges", f"{n_sig:,}", f"FDR < {fdr_threshold}")]),
                        unsafe_allow_html=True,
                    )
                    rows = "".join(
                        f"<tr><td>{row.RBP}</td><td>{row[1]:,}</td></tr>"
                        for row in hubs.itertuples()
                    )
                    st.markdown(
                        f'<table class="result-table"><thead><tr><th>RBP hub</th>'
                        f'<th>Targets</th></tr></thead><tbody>{rows}</tbody></table>',
                        unsafe_allow_html=True,
                    )
                st.success("Network inference complete.")

    st.markdown("---")
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("← Back"):
            go_to(3); st.rerun()
    with c2:
        if st.button("View results →", disabled=not st.session_state.states_done):
            go_to(5); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif step == 5:
    import scptr

    st.markdown('<div class="page-title">Results</div>', unsafe_allow_html=True)
    adata = st.session_state.adata

    # ── summary metrics ───────────────────────────────────────────────────────
    gamma = adata.layers.get("gamma")
    gamma_pos = gamma[gamma > 0] if gamma is not None else np.array([])
    n_states = adata.obs["pt_state"].nunique() if "pt_state" in adata.obs.columns else 0
    beta = adata.var.get("beta")

    st.markdown(
        metric_card([
            ("Cells", f"{adata.n_obs:,}", ""),
            ("Genes", f"{adata.n_vars:,}", ""),
            ("PT States", str(n_states), "γ-space clusters"),
            ("Median β", f"{np.median(beta):.3f}" if beta is not None else "—", "splicing rate"),
            ("Median γ", f"{np.median(gamma_pos):.4f}" if len(gamma_pos) else "—", "positive"),
        ]),
        unsafe_allow_html=True,
    )

    rtab1, rtab2, rtab3 = st.tabs(["VISUALIZATION", "GENE RANKINGS", "DOWNLOAD"])

    with rtab1:
        viz_col = st.selectbox(
            "Color by",
            (["pt_state"] if "pt_state" in adata.obs.columns else []) + [
                c for c in adata.obs.columns if c not in ["pt_state"]
            ][:20],
            label_visibility="collapsed",
        )

        use_gamma_umap = st.checkbox("Use γ-space UMAP", value=True,
                                     help="UMAP computed from γ-space PCA")

        if st.button("Plot UMAP"):
            basis = "X_gamma_umap" if (use_gamma_umap and "X_gamma_umap" in adata.obsm) else "X_umap"
            if basis not in adata.obsm:
                if "X_umap" not in adata.obsm:
                    with st.spinner("Computing UMAP…"):
                        sc.tl.umap(adata)
                basis = "X_umap"

            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            sc.pl.embedding(adata, basis=basis, color=viz_col, ax=ax, show=False,
                            frameon=False, size=18,
                            title=f"{viz_col} — {'γ-space' if 'gamma' in basis else 'expression'} UMAP")
            plt.tight_layout()
            st.image(fig_to_png(fig), width=520)
            plt.close(fig)

        if "gamma" in adata.layers and n_states > 0:
            if st.button("Gamma heatmap"):
                fig = scptr.pl.gamma_heatmap(adata, groupby="pt_state", show=False)
                if fig:
                    st.image(fig_to_png(fig), use_container_width=True)
                    plt.close(fig)

    with rtab2:
        if st.button("Rank PT genes"):
            with st.spinner("Ranking…"):
                scptr.tl.rank_pt_genes(adata)
            if "rank_pt_genes" in adata.uns:
                names = adata.uns["rank_pt_genes"].get("names", {})
                if names:
                    groups = list(names.keys())[:5]
                    rows = ""
                    for g in groups:
                        top = list(names[g])[:10]
                        rows += f"<tr><td><strong>PT state {g}</strong></td><td>{'&nbsp; '.join(top)}</td></tr>"
                    st.markdown(
                        f'<table class="result-table"><thead><tr>'
                        f'<th>State</th><th>Top differentially degraded genes</th>'
                        f'</tr></thead><tbody>{rows}</tbody></table>',
                        unsafe_allow_html=True,
                    )

        if beta is not None:
            st.markdown("**Top genes by splicing rate (β)**")
            top_beta = adata.var["beta"].sort_values(ascending=False).head(15)
            rows = "".join(
                f"<tr><td>{gene}</td><td>{val:.4f}</td></tr>"
                for gene, val in top_beta.items()
            )
            st.markdown(
                f'<table class="result-table" style="max-width:360px">'
                f'<thead><tr><th>Gene</th><th>β</th></tr></thead>'
                f'<tbody>{rows}</tbody></table>',
                unsafe_allow_html=True,
            )

    with rtab3:
        st.markdown("**Download results**")
        col1, col2 = st.columns(2)

        with col1:
            # h5ad download
            with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
                tmp = f.name
            adata.write_h5ad(tmp)
            with open(tmp, "rb") as f:
                st.download_button(
                    "Download AnnData (.h5ad)",
                    f.read(),
                    file_name=f"scptr_{st.session_state.dataset_name}.h5ad",
                    mime="application/octet-stream",
                )

        with col2:
            # gamma CSV
            if gamma is not None:
                gamma_df = pd.DataFrame(
                    gamma,
                    index=adata.obs_names,
                    columns=adata.var_names,
                )
                csv = gamma_df.to_csv()
                st.download_button(
                    "Download γ matrix (.csv)",
                    csv,
                    file_name=f"gamma_{st.session_state.dataset_name}.csv",
                    mime="text/csv",
                )

        if "pt_state" in adata.obs.columns:
            obs_csv = adata.obs[["pt_state"]].copy()
            if "tf_score" in adata.obs.columns:
                obs_csv["tf_score"] = adata.obs["tf_score"]
            if "ptf_score" in adata.obs.columns:
                obs_csv["ptf_score"] = adata.obs["ptf_score"]
            st.download_button(
                "Download cell metadata (.csv)",
                obs_csv.to_csv(),
                file_name=f"metadata_{st.session_state.dataset_name}.csv",
                mime="text/csv",
            )

    st.markdown("---")
    if st.button("← Start over"):
        for key in ["adata", "preprocessed", "estimated", "states_done", "network_done", "dataset_name"]:
            st.session_state[key] = None if key in ["adata", "dataset_name"] else False
        go_to(1)
        st.rerun()
