"""scPTR — Single-Cell Post-Transcriptional Regulatory Decomposition."""

from __future__ import annotations

import io
import os
import tempfile
import traceback
import warnings

import matplotlib
matplotlib.use("Agg")
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import streamlit as st

warnings.filterwarnings("ignore")

import scptr  # noqa: E402 — imported after matplotlib backend set

st.set_page_config(
    page_title="scPTR",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* animations */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideIn {
    from { opacity: 0; transform: translateX(-8px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes countUp {
    from { opacity: 0; transform: scale(0.92); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes stepPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(43,87,151,0.3); }
    50%       { box-shadow: 0 0 0 4px rgba(43,87,151,0.1); }
}

/* base */
html, body, [data-testid="stAppViewContainer"] {
    background: #f2f3f5;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 14px;
    color: #1a1a1a;
}
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

.block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1080px;
}

/* sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #d0d0d0;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

/* sidebar brand */
.sb-brand {
    padding: 1.25rem 1.5rem 1rem;
    border-bottom: 1px solid #e8e8e8;
}
.sb-name {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #2b5797;
}
.sb-tag {
    font-size: 11px;
    color: #888;
    line-height: 1.5;
    margin-top: 0.1rem;
}

/* sidebar step list */
.sb-steps { padding: 0.5rem 0; }
.sb-step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.55rem 1.5rem;
    border-left: 3px solid transparent;
}
.sb-step.active {
    background: #eef2f9;
    border-left-color: #2b5797;
    animation: slideIn 0.25s ease both;
}
.sb-step.done { opacity: 0.6; }
.sb-step-num {
    width: 22px; height: 22px;
    border: 1.5px solid #ccc;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 700; color: #888;
    flex-shrink: 0;
}
.sb-step.active .sb-step-num {
    background: #2b5797; border-color: #2b5797; color: #fff;
    animation: stepPulse 2.5s ease infinite;
}
.sb-step.done .sb-step-num {
    background: #e5f3ec; border-color: #2d7d4b; color: #2d7d4b;
}
.sb-step-label { font-size: 12px; color: #555; }
.sb-step.active .sb-step-label { font-weight: 700; color: #1a1a1a; }

/* data info */
.sb-info {
    margin: 0.75rem 1rem;
    padding: 0.7rem 1rem;
    background: #f7f8fa;
    border: 1px solid #e0e0e0;
    font-size: 12px; color: #444; line-height: 1.8;
    animation: fadeUp 0.3s ease;
}
.sb-info b { color: #1a1a1a; font-weight: 700; }
.sb-ok { color: #2d7d4b; font-size: 11px; }

/* page title */
.pt { font-size: 22px; font-weight: 700; letter-spacing: -0.02em;
      color: #1a1a1a; margin-bottom: 0.2rem;
      animation: fadeUp 0.3s ease both; }
.pd { font-size: 13px; color: #555; line-height: 1.65;
      margin-bottom: 1.5rem; animation: fadeUp 0.4s ease both; }

/* section label */
.sl {
    font-size: 11px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #888;
    margin: 1.5rem 0 0.6rem; padding-bottom: 0.35rem;
    border-bottom: 1px solid #ebebeb;
}

/* cards */
.card {
    background: #fff; border: 1px solid #ddd;
    padding: 1.4rem 1.75rem; margin-bottom: 1.25rem;
    animation: fadeUp 0.4s ease both;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.card:hover { border-color: #b0bac8; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
.card-title {
    font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #777;
    margin-bottom: 0.6rem; padding-bottom: 0.4rem;
    border-bottom: 1px solid #ebebeb;
}

/* metric grid */
.mg {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1px; background: #d0d0d0;
    border: 1px solid #d0d0d0;
    margin-bottom: 1.5rem;
    animation: fadeUp 0.4s ease both;
}
.mc { background: #fff; padding: 1rem 1.25rem;
      animation: fadeUp 0.35s ease both; }
.mc:nth-child(1) { animation-delay: 0.03s; }
.mc:nth-child(2) { animation-delay: 0.07s; }
.mc:nth-child(3) { animation-delay: 0.10s; }
.mc:nth-child(4) { animation-delay: 0.13s; }
.mc:nth-child(5) { animation-delay: 0.16s; }
.mc:nth-child(6) { animation-delay: 0.19s; }
.mc-label {
    font-size: 10px; color: #888;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 0.25rem;
}
.mc-val {
    font-size: 23px; font-weight: 600; color: #1a1a1a;
    font-family: "SF Mono","Fira Code",Consolas,monospace;
    font-variant-numeric: tabular-nums;
    animation: countUp 0.4s ease both;
}
.mc-sub { font-size: 11px; color: #aaa; margin-top: 0.1rem; }
.mc.blue .mc-val { color: #2b5797; }
.mc.green .mc-val { color: #2d7d4b; }

/* tables */
.tbl { width: 100%; border-collapse: collapse; font-size: 13px;
       animation: fadeUp 0.4s ease both; }
.tbl th {
    background: #f7f8fa; border-bottom: 2px solid #d0d0d0;
    padding: 0.5rem 0.75rem; text-align: left;
    font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #666;
}
.tbl td {
    padding: 0.42rem 0.75rem; border-bottom: 1px solid #ebebeb;
    font-family: "SF Mono","Fira Code",Consolas,monospace; font-size: 12px;
}
.tbl tr:last-child td { border-bottom: none; }
.tbl tr:hover td { background: #fafbfc; }
.tbl td.txt { font-family: inherit; font-size: 13px; }

/* status boxes */
.ibox { background: #eef2f9; border-left: 3px solid #2b5797;
        padding: 0.7rem 1rem; margin: 0.75rem 0;
        font-size: 13px; color: #2a3a5a; line-height: 1.6;
        animation: fadeUp 0.3s ease both;
        transition: background 0.15s; }
.ibox:hover { background: #e5ecf6; }
.sbox { background: #eaf5ef; border-left: 3px solid #2d7d4b;
        padding: 0.7rem 1rem; margin: 0.75rem 0;
        font-size: 13px; color: #1a3d28; line-height: 1.6;
        animation: fadeUp 0.3s ease both;
        transition: background 0.15s; }
.sbox:hover { background: #dff0e8; }
.ebox { background: #fdf0f0; border-left: 3px solid #c0392b;
        padding: 0.7rem 1rem; margin: 0.75rem 0;
        font-size: 13px; color: #5a1a1a; line-height: 1.6;
        animation: fadeUp 0.3s ease both;
        transition: background 0.15s; }
.ebox:hover { background: #f9e5e5; }
.wbox { background: #fef9f0; border-left: 3px solid #d4860a;
        padding: 0.7rem 1rem; margin: 0.75rem 0;
        font-size: 13px; color: #4a3010; line-height: 1.6;
        animation: fadeUp 0.3s ease both;
        transition: background 0.15s; }
.wbox:hover { background: #fdf1e0; }

/* radio as horizontal nav */
[data-testid="stSidebar"] [data-testid="stRadio"] { margin: 0.5rem 0.75rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    display: flex !important; gap: 0 !important; flex-direction: row !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    flex: 1 !important; text-align: center !important;
    padding: 0.4rem 0 !important; margin: 0 !important;
    background: #f4f4f4 !important; border: 1px solid #d0d0d0 !important;
    border-right: none !important;
    font-size: 10px !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.07em !important;
    color: #666 !important; cursor: pointer !important;
    transition: background 0.15s !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:last-of-type {
    border-right: 1px solid #d0d0d0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #2b5797 !important; color: #fff !important;
    border-color: #2b5797 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* buttons */
.stButton > button {
    background: #2b5797 !important; color: #fff !important;
    border: none !important; border-radius: 0 !important;
    padding: 0.48rem 1.3rem !important;
    font-size: 11px !important; font-weight: 700 !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #1e3f6e !important; }
.stButton > button:active { background: #163060 !important; }
.stButton > button:disabled { background: #b0bac8 !important; }
.stDownloadButton > button {
    background: #fff !important; color: #2b5797 !important;
    border: 1.5px solid #2b5797 !important; border-radius: 0 !important;
    font-size: 11px !important; font-weight: 700 !important;
    letter-spacing: 0.04em !important; text-transform: uppercase !important;
}
.stDownloadButton > button:hover { background: #eef2f9 !important; }

/* form labels */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label,
[data-testid="stFileUploader"] label,
[data-testid="stCheckbox"] label,
[data-testid="stRadio"] label,
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label {
    font-size: 11px !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.07em !important;
    color: #555 !important;
}
[data-testid="stFileUploader"] {
    border: 1.5px dashed #bbb !important;
    background: #fafafa !important;
}
/* sidebar jump buttons — secondary style (override primary) */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important; color: #2b5797 !important;
    border: 1px solid #c8d4e8 !important;
    font-size: 10px !important; padding: 0.3rem 0.75rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #eef2f9 !important; border-color: #2b5797 !important;
}

/* tabs */
[data-testid="stTabs"] { overflow-x: auto; }
[data-testid="stTabs"] > div > div:first-child {
    flex-wrap: nowrap; white-space: nowrap;
}
[data-testid="stTabs"] button {
    border-radius: 0 !important; font-size: 11px !important;
    font-weight: 700 !important; text-transform: uppercase !important;
    letter-spacing: 0.07em !important; white-space: nowrap !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2b5797 !important;
    border-bottom-color: #2b5797 !important;
}
/* dividers */
hr { border: none; border-top: 1px solid #e0e0e0; margin: 1.25rem 0; }
/* images */
.stImage img { border: 1px solid #d0d0d0; display: block; }
/* expander */
[data-testid="stExpander"] { border: 1px solid #ddd !important; border-radius: 0 !important; }

/* hero (home page) */
.hero {
    background: #fff; border: 1px solid #ddd;
    padding: 2.25rem 2rem 1.75rem;
    margin-bottom: 1.5rem;
    animation: fadeUp 0.4s ease both;
}
.hero-title {
    font-size: 34px; font-weight: 800; letter-spacing: -0.04em;
    color: #1a1a1a; line-height: 1.1; margin-bottom: 0.5rem;
}
.hero-title .accent { color: #2b5797; }
.hero-sub {
    font-size: 14px; color: #555; line-height: 1.65;
    max-width: 600px; margin-bottom: 1.25rem;
}
.hero-eq {
    display: inline-block;
    background: #f7f8fa; border: 1px solid #ddd;
    padding: 0.65rem 1.25rem;
    font-family: "SF Mono","Fira Code",Consolas,monospace;
    font-size: 15px; color: #2b5797; letter-spacing: 0.03em;
}

/* pipeline steps (home) */
.pipe { display: flex; margin: 1rem 0 1.5rem; }
.pipe-step {
    flex: 1; background: #fff; border: 1px solid #ddd; border-right: none;
    padding: 0.75rem 0.9rem;
    animation: fadeUp 0.4s ease both;
    transition: background 0.15s, border-color 0.15s;
}
.pipe-step:hover { background: #f7f8fa; border-color: #b8c4d4; }
.pipe-step:nth-child(1) { animation-delay: 0.05s; }
.pipe-step:nth-child(2) { animation-delay: 0.10s; }
.pipe-step:nth-child(3) { animation-delay: 0.15s; }
.pipe-step:nth-child(4) { animation-delay: 0.20s; }
.pipe-step:nth-child(5) { animation-delay: 0.25s; }
.pipe-step:last-child { border-right: 1px solid #ddd; }
.pipe-step.done { border-top: 2px solid #2d7d4b; }
.pipe-step.done .pipe-n { color: #2d7d4b; }
.pipe-step.active-step { border-top: 2px solid #2b5797; }
.pipe-step.active-step .pipe-n { color: #2b5797; }
.pipe-n { font-size: 10px; font-weight: 700; color: #999;
          text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.2rem; }
.pipe-name { font-size: 12px; font-weight: 600; color: #1a1a1a; margin-bottom: 0.15rem; }
.pipe-desc { font-size: 11px; color: #777; line-height: 1.4; }

/* docs */
.doc-sec {
    background: #fff; border: 1px solid #ddd;
    padding: 1.5rem 1.75rem; margin-bottom: 1.25rem;
    animation: fadeUp 0.4s ease both;
}
.doc-h2 {
    font-size: 15px; font-weight: 700; color: #1a1a1a;
    margin-bottom: 0.75rem; padding-bottom: 0.5rem;
    border-bottom: 2px solid #2b5797;
}
.doc-h3 {
    font-size: 12px; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; color: #555;
    margin: 1.2rem 0 0.5rem;
}
.doc-p { font-size: 13px; color: #333; line-height: 1.75; margin-bottom: 0.6rem; }
.doc-code {
    background: #f7f8fa; border: 1px solid #e0e0e0;
    padding: 0.75rem 1rem;
    font-family: "SF Mono","Fira Code",Consolas,monospace;
    font-size: 12px; color: #1a1a1a;
    white-space: pre; overflow-x: auto;
    margin: 0.5rem 0 1rem;
}
.param-name { font-family: monospace; font-size: 12px; color: #2b5797; font-weight: 600; }
.param-type { font-size: 11px; color: #888; font-style: italic; }
.param-desc { font-size: 12px; color: #444; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ── session state ─────────────────────────────────────────────────────────────
for k, v in {
    "page": "home",
    "step": 1,
    "adata": None,
    "dataset_name": None,
    "preprocessed": False,
    "estimated": False,
    "states_done": False,
    "velocity_done": False,
    "network_done": False,
    "_tmps": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── helpers ───────────────────────────────────────────────────────────────────
def go(step: int) -> None:
    st.session_state.step = step

def nav(page: str) -> None:
    st.session_state.page = page

def fig_png(fig: plt.Figure, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.read()

def make_tmp(suffix: str = ".h5ad") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    st.session_state._tmps.append(path)
    return path

def clean_tmps() -> None:
    for p in st.session_state._tmps:
        try:
            os.unlink(p)
        except Exception:
            pass
    st.session_state._tmps = []

def mg(*metrics) -> str:
    """metric_grid(*[(label, value, sub, cls?), ...])"""
    cells = []
    for m in metrics:
        lbl, val, sub = m[0], m[1], m[2]
        cls = m[3] if len(m) > 3 else ""
        cells.append(
            f'<div class="mc {cls}">'
            f'<div class="mc-label">{lbl}</div>'
            f'<div class="mc-val">{val}</div>'
            f'<div class="mc-sub">{sub}</div>'
            f'</div>'
        )
    return f'<div class="mg">{"".join(cells)}</div>'

def reset_downstream(from_step: int) -> None:
    if from_step <= 1:
        st.session_state.preprocessed = False
        st.session_state.estimated = False
        st.session_state.states_done = False
        st.session_state.velocity_done = False
        st.session_state.network_done = False
    elif from_step <= 2:
        st.session_state.estimated = False
        st.session_state.states_done = False
        st.session_state.velocity_done = False
        st.session_state.network_done = False
    elif from_step <= 3:
        st.session_state.states_done = False
        st.session_state.velocity_done = False
        st.session_state.network_done = False


# ── sidebar ───────────────────────────────────────────────────────────────────
STEPS = ["Load Data", "Preprocess", "Estimate Rates", "Discover PT States", "Results"]

with st.sidebar:
    st.markdown(
        '<div class="sb-brand">'
        '<div class="sb-name">scPTR</div>'
        '<div class="sb-tag">Single-Cell Post-Transcriptional<br>Regulatory Decomposition</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    _page_map = {"Home": "home", "Analysis": "analysis", "Docs": "docs"}
    _page_rev = {v: k for k, v in _page_map.items()}
    _nav_choice = st.radio(
        "nav",
        list(_page_map.keys()),
        index=list(_page_map.keys()).index(_page_rev.get(st.session_state.page, "Home")),
        horizontal=True,
        label_visibility="collapsed",
        key="sidebar_nav",
    )
    if _page_map[_nav_choice] != st.session_state.page:
        st.session_state.page = _page_map[_nav_choice]
        st.rerun()

    # Show data summary on home/docs pages too
    if st.session_state.page != "analysis" and st.session_state.adata is not None:
        _a = st.session_state.adata
        _done_count = sum([
            st.session_state.preprocessed,
            st.session_state.estimated,
            st.session_state.states_done,
        ])
        st.markdown(
            f'<div class="sb-info">'
            f'<b>{st.session_state.dataset_name}</b><br>'
            f'{_a.n_obs:,} cells · {_a.n_vars:,} genes<br>'
            f'<span class="sb-ok">{_done_count}/3 steps complete</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("Resume Analysis →", use_container_width=True):
            nav("analysis"); st.rerun()

    if st.session_state.page == "analysis":
        st.markdown('<hr style="margin:0.75rem 0">', unsafe_allow_html=True)
        st.markdown('<div class="sb-steps">', unsafe_allow_html=True)

        step = st.session_state.step
        done = {
            1: st.session_state.adata is not None,
            2: st.session_state.preprocessed,
            3: st.session_state.estimated,
            4: st.session_state.states_done,
        }
        for i, label in enumerate(STEPS, 1):
            if i == step:
                cls = "active"
            elif done.get(i, False):
                cls = "done"
            else:
                cls = ""
            num = "✓" if (done.get(i, False) and i != step) else str(i)

            # Clickable for completed steps and step 5 (always accessible after states)
            can_jump = done.get(i, False) and i != step
            if can_jump:
                # Show step as a compact link button
                col_btn, = st.columns([1])
                if st.button(f"↩ {label}", key=f"jump_{i}", use_container_width=True):
                    go(i); st.rerun()
            else:
                st.markdown(
                    f'<div class="sb-step {cls}">'
                    f'<div class="sb-step-num">{num}</div>'
                    f'<div class="sb-step-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.adata is not None:
            adata = st.session_state.adata
            lines = [
                f"<b>{st.session_state.dataset_name}</b>",
                f"{adata.n_obs:,} cells · {adata.n_vars:,} genes",
            ]
            if st.session_state.preprocessed:
                lines.append('<span class="sb-ok">✓ preprocessed</span>')
            if st.session_state.estimated:
                lines.append('<span class="sb-ok">✓ rates estimated</span>')
            if st.session_state.states_done:
                n_s = adata.obs["pt_state"].nunique() if "pt_state" in adata.obs else "?"
                lines.append(f'<span class="sb-ok">✓ {n_s} PT states</span>')
            if st.session_state.network_done:
                lines.append('<span class="sb-ok">✓ network inferred</span>')
            st.markdown(
                f'<div class="sb-info">{"<br>".join(lines)}</div>',
                unsafe_allow_html=True,
            )


# ── page routing ──────────────────────────────────────────────────────────────
page = st.session_state.page


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "home":
    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">Single-cell <span class="accent">post-transcriptional</span><br>regulatory decomposition</div>'
        '<div class="hero-sub">'
        'scPTR estimates per-cell, per-gene mRNA degradation rates from standard scRNA-seq '
        'spliced/unspliced counts. It uses the degradation rate γ as a primary analytical '
        'axis — complementary to RNA velocity — to discover expression-invisible cell states, '
        'compute post-transcriptional velocity, and infer RNA-binding protein networks.'
        '</div>'
        '<div class="hero-eq">γ<sub>ig</sub> = β<sub>g</sub> · u<sub>ig</sub> / s<sub>ig</sub></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    home_c1, home_c2 = st.columns([1, 2])
    with home_c1:
        if st.button("Start Analysis →"):
            nav("analysis"); st.rerun()
    with home_c2:
        if st.session_state.adata is not None:
            _prog_items = [
                ("Load Data", True),
                ("Preprocess", st.session_state.preprocessed),
                ("Estimate Rates", st.session_state.estimated),
                ("PT States", st.session_state.states_done),
                ("Results", st.session_state.states_done),
            ]
            _pct = sum(1 for _, v in _prog_items if v) / len(_prog_items) * 100
            bars = "".join(
                f'<div style="flex:1;height:4px;background:{"#2d7d4b" if v else "#e0e0e0"};margin-right:2px"></div>'
                for _, v in _prog_items
            )
            st.markdown(
                f'<div style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.3rem">'
                f'Analysis progress · {_pct:.0f}%</div>'
                f'<div style="display:flex;align-items:center;gap:0">{bars}</div>'
                f'<div style="font-size:12px;color:#555;margin-top:0.4rem">'
                f'{st.session_state.dataset_name} — '
                f'{st.session_state.adata.n_obs:,} cells · {st.session_state.adata.n_vars:,} genes</div>',
                unsafe_allow_html=True,
            )
            if st.button("Resume →"):
                nav("analysis"); st.rerun()

    # Pipeline — dynamic done/active state
    _s = st.session_state
    _pipe_states = [
        _s.adata is not None,
        _s.preprocessed,
        _s.estimated,
        _s.states_done,
        _s.states_done,
    ]
    _next_step = next((i for i, v in enumerate(_pipe_states) if not v), len(_pipe_states))
    def _pipe_cls(i):
        if _pipe_states[i]: return "done"
        if i == _next_step: return "active-step"
        return ""
    def _pipe_icon(i):
        return "✓ " if _pipe_states[i] else ""
    st.markdown('<div class="sl">Analysis pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pipe">'
        f'<div class="pipe-step {_pipe_cls(0)}"><div class="pipe-n">{_pipe_icon(0)}01</div><div class="pipe-name">Load Data</div>'
        '<div class="pipe-desc">Upload .h5ad or use built-in pancreas / dentate gyrus datasets</div></div>'
        f'<div class="pipe-step {_pipe_cls(1)}"><div class="pipe-n">{_pipe_icon(1)}02</div><div class="pipe-name">Preprocess</div>'
        '<div class="pipe-desc">Filter genes, normalize counts, kNN graph, Gaussian smoothing</div></div>'
        f'<div class="pipe-step {_pipe_cls(2)}"><div class="pipe-n">{_pipe_icon(2)}03</div><div class="pipe-name">Estimate Rates</div>'
        '<div class="pipe-desc">β via quantile regression on u/s portraits; γ = β · u / s per cell</div></div>'
        f'<div class="pipe-step {_pipe_cls(3)}"><div class="pipe-n">{_pipe_icon(3)}04</div><div class="pipe-name">PT States</div>'
        '<div class="pipe-desc">Leiden clustering in γ-space; PT velocity; RBP–target networks</div></div>'
        f'<div class="pipe-step {_pipe_cls(4)}"><div class="pipe-n">{_pipe_icon(4)}05</div><div class="pipe-name">Results</div>'
        '<div class="pipe-desc">UMAP visualization, gene rankings, download outputs</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="sl">Validation</div>', unsafe_allow_html=True)
        _g = 'background:#eaf5ef'
        st.markdown(
            '<table class="tbl">'
            '<thead><tr><th>Validation</th><th>Result</th></tr></thead>'
            '<tbody>'
            f'<tr style="{_g}"><td class="txt">sci-fate metabolic labeling</td><td><b>ρ = −0.81</b></td></tr>'
            f'<tr><td class="txt">10x developmental half-lives</td><td>ρ = −0.33 to −0.40</td></tr>'
            f'<tr><td class="txt">vs. scVelo steady-state</td><td>−0.37 (scPTR: −0.40)</td></tr>'
            f'<tr><td class="txt">vs. velVI</td><td>−0.28 (scPTR: −0.40)</td></tr>'
            f'<tr style="{_g}"><td class="txt">miRNA target enrichment</td><td><b>59% of 215 families</b>, p = 4.7×10⁻⁶⁵</td></tr>'
            f'<tr style="{_g}"><td class="txt">DepMap CRISPR essentiality</td><td>hub RBPs, <b>p = 6.4×10⁻⁵</b></td></tr>'
            f'<tr><td class="txt">Subsampling robustness</td><td>r &gt; 0.97 at 20%</td></tr>'
            '</tbody></table>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown('<div class="sl">Key findings</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card" style="margin-bottom:0.75rem">'
            '<div class="card-title">Expression-invisible states</div>'
            '<div style="font-size:13px;color:#333;line-height:1.65">'
            '3/8 pancreatic and 6/11 hippocampal cell types harbor post-transcriptional '
            'subpopulations undetectable by expression analysis. Confirmed by zero-permutation '
            'control (ARI ≈ 0).'
            '</div></div>'
            '<div class="card" style="margin-bottom:0.75rem">'
            '<div class="card-title">Temporal precedence</div>'
            '<div style="font-size:13px;color:#333;line-height:1.65">'
            'Degradation-rate changes precede expression changes for 54% of pancreas '
            'transition genes (p &lt; 10⁻⁵⁷) and 78% in dentate gyrus (p = 9.9×10⁻¹³).'
            '</div></div>'
            '<div class="card">'
            '<div class="card-title">RBP networks</div>'
            '<div style="font-size:13px;color:#333;line-height:1.65">'
            'Library-size-corrected inference identifies essential hub regulators '
            '(HNRNPA1, YBX1, ELAVL1/HuR). Neuroblastoma shows 66% stabilizing edges.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sl">Input requirements</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ibox">'
        'scPTR requires an AnnData file (<code>.h5ad</code>) with two count matrix layers: '
        '<code>spliced</code> and <code>unspliced</code>. These can be generated with '
        '<strong>STARsolo</strong>, <strong>Alevin-fry</strong>, <strong>kallisto|bustools</strong>, '
        'or <strong>velocyto</strong>. Raw (un-normalized) counts are expected.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="wbox" style="font-size:12px">'
        '<b>Session data lives in browser memory.</b> Keep this tab open while running analysis. '
        'Refreshing the page will reset all results. Download your AnnData file in step 5 to save progress.'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS — STEP 1: LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "analysis" and st.session_state.step == 1:
    st.markdown('<div class="pt">Load Data <span style="font-size:13px;font-weight:400;color:#aaa;letter-spacing:0">— step 1 of 5</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pd">Upload an AnnData file (.h5ad) containing <code>spliced</code> and '
        '<code>unspliced</code> count layers, or start with a built-in example dataset.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sl">Quick start</div>', unsafe_allow_html=True)
    qs_c1, qs_c2, qs_c3 = st.columns(3, gap="large")
    with qs_c1:
        st.markdown(
            '<div class="card" style="animation-delay:0.05s">'
            '<div class="card-title">Pancreas</div>'
            '<div style="font-size:13px;color:#333;line-height:1.6;margin-bottom:0.75rem">'
            'Mouse endocrinogenesis · 3,696 cells<br>'
            '<span style="color:#888;font-size:12px">Bastidas-Ponce et al. 2019</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("Load Pancreas", key="qs_pancreas", use_container_width=True):
            with st.spinner("Downloading pancreas dataset…"):
                try:
                    _adata = scptr.datasets.pancreas()
                    st.session_state.adata = _adata
                    st.session_state.dataset_name = "Pancreas"
                    reset_downstream(1)
                    go(2); st.rerun()
                except Exception as e:
                    st.markdown(f'<div class="ebox">Error: {e}</div>', unsafe_allow_html=True)
    with qs_c2:
        st.markdown(
            '<div class="card" style="animation-delay:0.10s">'
            '<div class="card-title">Dentate Gyrus</div>'
            '<div style="font-size:13px;color:#333;line-height:1.6;margin-bottom:0.75rem">'
            'Mouse hippocampal neurogenesis · 2,930 cells<br>'
            '<span style="color:#888;font-size:12px">Hochgerner et al. 2018</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("Load Dentate Gyrus", key="qs_dg", use_container_width=True):
            with st.spinner("Downloading dentate gyrus dataset…"):
                try:
                    _adata = scptr.datasets.dentate_gyrus()
                    st.session_state.adata = _adata
                    st.session_state.dataset_name = "Dentate Gyrus"
                    reset_downstream(1)
                    go(2); st.rerun()
                except Exception as e:
                    st.markdown(f'<div class="ebox">Error: {e}</div>', unsafe_allow_html=True)
    with qs_c3:
        st.markdown(
            '<div class="card" style="animation-delay:0.15s">'
            '<div class="card-title">Upload</div>'
            '<div style="font-size:13px;color:#333;line-height:1.6;margin-bottom:0.75rem">'
            'Your own .h5ad file<br>'
            '<span style="color:#888;font-size:12px">Requires spliced + unspliced layers</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("Upload File →", key="qs_upload", use_container_width=True):
            st.session_state["_show_upload"] = True
            st.rerun()

    st.markdown('<hr>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="sl">Example datasets</div>', unsafe_allow_html=True)
        example = st.selectbox(
            "Dataset",
            ["— select —",
             "Pancreas (mouse endocrinogenesis, 3,696 cells)",
             "Dentate Gyrus (mouse hippocampal neurogenesis, 2,930 cells)"],
            label_visibility="collapsed",
        )
        st.markdown(
            '<div class="ibox" style="font-size:12px">'
            'Both datasets include spliced/unspliced counts from scVelo. '
            'Requires <code>pooch</code> — datasets are downloaded on first use (~30 MB each).'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("Load Example Dataset", disabled=(example == "— select —")):
            with st.spinner("Downloading and loading dataset…"):
                try:
                    if "Pancreas" in example:
                        adata = scptr.datasets.pancreas()
                        name = "Pancreas"
                    else:
                        adata = scptr.datasets.dentate_gyrus()
                        name = "Dentate Gyrus"
                    st.session_state.adata = adata
                    st.session_state.dataset_name = name
                    reset_downstream(1)
                    go(2)
                    st.rerun()
                except Exception as e:
                    st.markdown(
                        f'<div class="ebox"><b>Error loading dataset:</b> {e}<br>'
                        f'Ensure <code>pip install "scptr[datasets]"</code> is installed.</div>',
                        unsafe_allow_html=True,
                    )

    with col2:
        if st.session_state.pop("_show_upload", False):
            st.markdown('<div class="ibox">Use the file uploader below to load your .h5ad file.</div>', unsafe_allow_html=True)
        st.markdown('<div class="sl">Upload your own</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "H5AD file",
            type=["h5ad"],
            label_visibility="collapsed",
            help="AnnData file with 'spliced' and 'unspliced' layers",
        )
        if uploaded is not None:
            st.markdown(
                f'<div style="font-size:12px;color:#555;margin-bottom:0.5rem">'
                f'File: <code>{uploaded.name}</code> ({uploaded.size / 1e6:.1f} MB)</div>',
                unsafe_allow_html=True,
            )
            if st.button("Load File"):
                tmp = make_tmp(".h5ad")
                try:
                    with open(tmp, "wb") as f:
                        f.write(uploaded.read())
                    with st.spinner("Reading file…"):
                        adata = sc.read_h5ad(tmp)

                    missing = [l for l in ["spliced", "unspliced"] if l not in adata.layers]
                    if missing:
                        st.markdown(
                            f'<div class="ebox"><b>Missing layers:</b> {", ".join(missing)}<br>'
                            f'The file must contain <code>spliced</code> and <code>unspliced</code> '
                            f'count matrix layers. Use velocyto, STARsolo, or Alevin-fry to generate them.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.session_state.adata = adata
                        st.session_state.dataset_name = uploaded.name.removesuffix(".h5ad")
                        reset_downstream(1)
                        go(2)
                        st.rerun()
                except Exception as e:
                    st.markdown(
                        f'<div class="ebox"><b>Could not read file:</b> {e}</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card">'
        '<div class="card-title">What scPTR needs</div>'
        '<table class="tbl">'
        '<thead><tr><th>Slot</th><th>Content</th><th>Required</th></tr></thead>'
        '<tbody>'
        '<tr><td>adata.layers["spliced"]</td><td class="txt">Spliced (mature) mRNA counts per cell × gene</td><td>✓</td></tr>'
        '<tr><td>adata.layers["unspliced"]</td><td class="txt">Unspliced (nascent) mRNA counts per cell × gene</td><td>✓</td></tr>'
        '<tr><td>adata.obs columns</td><td class="txt">Cell metadata (cell type, cluster, etc.) for visualization</td><td>optional</td></tr>'
        '<tr><td>adata.obsm["X_umap"]</td><td class="txt">UMAP embedding (scPTR will compute one if absent)</td><td>optional</td></tr>'
        '</tbody></table>'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS — STEP 2: PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "analysis" and st.session_state.step == 2:
    adata = st.session_state.adata
    st.markdown('<div class="pt">Preprocess <span style="font-size:13px;font-weight:400;color:#aaa;letter-spacing:0">— step 2 of 5</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pd">Filter low-quality genes, normalize counts to library size, '
        'build a cell neighborhood graph (kNN in PCA space), and apply Gaussian kernel '
        'smoothing to the spliced and unspliced layers.</div>',
        unsafe_allow_html=True,
    )

    # Data quality summary
    _s_layer = adata.layers["spliced"]
    _u_layer = adata.layers["unspliced"]
    _s_sparsity = 1.0 - float(np.count_nonzero(_s_layer)) / _s_layer.size
    _u_sparsity = 1.0 - float(np.count_nonzero(_u_layer)) / _u_layer.size
    _has_obs = list(adata.obs.columns)[:5]
    st.markdown(
        mg(
            ("Cells", f"{adata.n_obs:,}", ""),
            ("Genes", f"{adata.n_vars:,}", "before filtering"),
            ("Spliced sparsity", f"{_s_sparsity:.1%}", "zeros in matrix"),
            ("Unspliced sparsity", f"{_u_sparsity:.1%}", "zeros in matrix"),
        ),
        unsafe_allow_html=True,
    )
    _layer_ok = all(l in adata.layers for l in ["spliced", "unspliced"])
    _extra_layers = [l for l in adata.layers if l not in ["spliced", "unspliced"]]
    _layer_info = (
        '<span style="color:#2d7d4b;font-weight:700">✓ spliced</span> &nbsp; '
        '<span style="color:#2d7d4b;font-weight:700">✓ unspliced</span>'
    )
    if _extra_layers:
        _layer_info += f' &nbsp;· also: {", ".join(_extra_layers[:4])}'
    if _has_obs:
        _layer_info += f'<br><span style="color:#888;font-size:11px">obs columns: {", ".join(_has_obs)}</span>'
    st.markdown(
        f'<div class="ibox" style="font-size:12px">'
        f'Layers detected: {_layer_info}'
        f'</div>',
        unsafe_allow_html=True,
    )

    _rn = st.session_state.get("pp_reset_n", 0)  # reset counter — changes keys to force re-init
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown('<div class="sl">Gene filtering</div>', unsafe_allow_html=True)
        min_unspliced = st.number_input(
            "Min total unspliced counts per gene",
            value=10, min_value=1, step=1,
            key=f"pp_mu_{_rn}",
            help="Genes with fewer total unspliced counts across all cells are removed.",
        )
        min_cells = st.number_input(
            "Min cells with nonzero unspliced",
            value=5, min_value=1, step=1,
            key=f"pp_mc_{_rn}",
            help="Genes expressed in fewer cells are removed.",
        )
    with col2:
        st.markdown('<div class="sl">Cell filtering</div>', unsafe_allow_html=True)
        min_cell_unspliced = st.number_input(
            "Min unspliced counts per cell",
            value=0, min_value=0, step=100,
            key=f"pp_mcu_{_rn}",
            help="Cells with fewer total unspliced counts are removed. 0 = no filter.",
        )
        min_cell_spliced = st.number_input(
            "Min spliced counts per cell",
            value=0, min_value=0, step=100,
            key=f"pp_mcs_{_rn}",
            help="Cells with fewer total spliced counts are removed. 0 = no filter.",
        )
    with col3:
        st.markdown('<div class="sl">Neighborhood graph</div>', unsafe_allow_html=True)
        n_neighbors = st.number_input(
            "Neighbors (kNN graph)",
            value=30, min_value=5, max_value=100, step=5,
            key=f"pp_nn_{_rn}",
            help="Number of nearest neighbors for the cell graph. Larger = smoother.",
        )
        n_pcs = st.number_input(
            "PCA components",
            value=30, min_value=10, max_value=100, step=5,
            key=f"pp_np_{_rn}",
            help="PCA dimensions used for neighbor search.",
        )

    with st.expander("Advanced: Smoothing options"):
        bw_mode = st.radio(
            "Smoothing bandwidth",
            ["Adaptive (median distance)", "Fixed value"],
            horizontal=True,
        )
        fixed_bw = None
        if bw_mode == "Fixed value":
            fixed_bw = st.number_input("Bandwidth", value=1.0, min_value=0.05, step=0.05)
        st.markdown(
            '<div class="ibox" style="font-size:12px">'
            'Gaussian kernel smoothing reduces noise in spliced/unspliced counts by averaging '
            'over cell neighbors. Adaptive bandwidth uses the median distance to the k-th neighbor.'
            '</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.preprocessed:
        st.markdown(
            '<div class="sbox">Preprocessing already complete. '
            'Click below to re-run with new parameters.</div>',
            unsafe_allow_html=True,
        )

    run_col, reset_col = st.columns([2, 1])
    if "pp_reset_n" not in st.session_state:
        st.session_state.pp_reset_n = 0
    with reset_col:
        if st.button("↻ Reset to defaults", help="Restore all preprocessing parameters to defaults"):
            st.session_state.pp_reset_n += 1
            st.rerun()
    with run_col:
        _run_pp = st.button("Run Preprocessing", use_container_width=True)
    if _run_pp:
        adata_work = st.session_state.adata.copy()
        try:
            prog = st.progress(0, text="Filtering genes…")
            scptr.pp.filter_genes(
                adata_work,
                min_unspliced_counts=min_unspliced,
                min_unspliced_cells=min_cells,
            )
            if min_cell_unspliced > 0 or min_cell_spliced > 0:
                prog.progress(15, text="Filtering cells…")
                scptr.pp.filter_cells(
                    adata_work,
                    min_unspliced_counts=min_cell_unspliced,
                    min_spliced_counts=min_cell_spliced,
                )
            prog.progress(25, text="Normalizing layers…")
            scptr.pp.normalize_layers(adata_work)
            prog.progress(50, text="Building neighborhood graph…")
            scptr.pp.neighbors(adata_work, n_neighbors=n_neighbors, n_pcs=n_pcs)
            prog.progress(75, text="Smoothing spliced/unspliced…")
            if fixed_bw is not None:
                scptr.pp.smooth_layers(adata_work, bandwidth=fixed_bw)
            else:
                scptr.pp.smooth_layers(adata_work)
            prog.progress(100, text="Done.")

            st.session_state.adata = adata_work
            st.session_state.preprocessed = True
            reset_downstream(2)

            _cells_removed = adata.n_obs - adata_work.n_obs
            _genes_removed = adata.n_vars - adata_work.n_vars
            st.markdown(
                mg(
                    ("Cells retained", f"{adata_work.n_obs:,}", f"−{_cells_removed:,} filtered" if _cells_removed else "all kept", "green"),
                    ("Genes retained", f"{adata_work.n_vars:,}", f"−{_genes_removed:,} filtered"),
                    ("Neighbors", str(n_neighbors), "kNN graph"),
                    ("PCA dims", str(n_pcs), "for graph"),
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="sbox">'
                'Preprocessing complete — genes filtered, layers normalized, kNN graph built, counts smoothed. '
                '<b>Next:</b> estimate β (splicing rate) and γ (degradation rate). '
                'Variance decomposition runs automatically with rate estimation.'
                '</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.markdown(
                f'<div class="ebox"><b>Preprocessing failed:</b> {e}<br>'
                f'<span style="font-size:12px;opacity:0.8">Common causes: '
                f'too few genes after filtering (lower thresholds in step 2) · '
                f'missing spliced/unspliced layers · '
                f'dataset too small for PCA (n_pcs may exceed n_cells or n_genes)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

    st.markdown('<hr>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("← Back"):
            go(1); st.rerun()
    with c2:
        if st.button("Next: Estimate Rates →", disabled=not st.session_state.preprocessed):
            go(3); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS — STEP 3: ESTIMATE RATES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "analysis" and st.session_state.step == 3:
    adata = st.session_state.adata
    st.markdown('<div class="pt">Estimate Rates <span style="font-size:13px;font-weight:400;color:#aaa;letter-spacing:0">— step 3 of 5</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pd">Estimate gene-specific splicing rates (β) from unspliced/spliced phase '
        'portraits using quantile regression. Then compute per-cell, per-gene degradation rates '
        'γ<sub>ig</sub> = β<sub>g</sub> · u<sub>ig</sub> / s<sub>ig</sub> using the '
        'kinetic steady-state model.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="sl">Beta estimation (β)</div>', unsafe_allow_html=True)
        quantile = st.slider(
            "Phase portrait quantile",
            0.80, 0.99, 0.95, 0.01,
            help="Upper quantile of the u/s ratio distribution used to estimate β. "
                 "Typical: 0.95 for 10x Chromium; 0.90–0.92 for Smart-seq (denser coverage). "
                 "Lower values = more conservative boundary.",
        )
        min_r2 = st.slider(
            "Min R² for beta fit",
            0.0, 0.90, 0.0, 0.05,
            help="Genes with a phase portrait R² below this threshold are excluded.",
        )
    with col2:
        st.markdown('<div class="sl">Gamma estimation (γ)</div>', unsafe_allow_html=True)
        min_spliced_thr = st.number_input(
            "Min smoothed spliced (Ms) for γ",
            value=0.01, min_value=0.001, step=0.005, format="%.3f",
            help="Cells with Ms < threshold for a given gene receive γ = 0 (prevents division by near-zero).",
        )
        clip_q = st.slider(
            "Gamma clip quantile",
            0.90, 1.00, 0.99, 0.01,
            help="Per-gene clipping at this quantile to remove extreme outliers before analysis.",
        )

    if st.session_state.estimated:
        st.markdown(
            '<div class="sbox">Rate estimation already complete. Re-run to update with new parameters.</div>',
            unsafe_allow_html=True,
        )

    if st.button("Estimate β and γ"):
        adata_work = st.session_state.adata
        try:
            prog = st.progress(0, text="Estimating splicing rates (β)…")
            scptr.tl.estimate_beta(adata_work, quantile=quantile)
            prog.progress(40, text="Estimating degradation rates (γ)…")
            scptr.tl.estimate_gamma(
                adata_work,
                clip_quantile=clip_q,
                min_spliced=min_spliced_thr,
            )
            prog.progress(80, text="Variance decomposition…")
            scptr.tl.variance_decomposition(adata_work)
            prog.progress(100, text="Done.")

            st.session_state.adata = adata_work
            st.session_state.estimated = True
            reset_downstream(3)

            beta = adata_work.var["beta"]
            gamma = adata_work.layers["gamma"]
            gamma_pos = gamma[gamma > 0]
            inf_genes = int((np.median(gamma, axis=0) > 0).sum())

            st.markdown(
                mg(
                    ("Median β", f"{np.median(beta):.3f}", "splicing rate"),
                    ("Median γ", f"{np.median(gamma_pos):.4f}", "positive cells", "blue"),
                    ("Max γ", f"{np.max(gamma):.2f}", "clipped"),
                    ("Informative genes", f"{inf_genes:,}", "median γ > 0", "green"),
                ),
                unsafe_allow_html=True,
            )

            # Phase portrait for highest-β gene
            top_gene = beta.idxmax()
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

            # Phase portrait
            ax = axes[0]
            Ms = adata_work.layers["Ms"][:, adata_work.var_names.get_loc(top_gene)]
            Mu = adata_work.layers["Mu"][:, adata_work.var_names.get_loc(top_gene)]
            ax.scatter(Ms, Mu, s=5, alpha=0.35, color="#2b5797", linewidths=0)
            bv = adata_work.var.loc[top_gene, "beta"]
            x = np.linspace(0, np.percentile(Ms, 99), 100)
            ax.plot(x, bv * x, color="#c0392b", lw=1.5, label=f"β = {bv:.3f}")
            ax.set_xlabel("Ms (smoothed spliced)", fontsize=10)
            ax.set_ylabel("Mu (smoothed unspliced)", fontsize=10)
            ax.set_title(f"Phase portrait: {top_gene}", fontsize=10, fontweight="bold")
            ax.legend(fontsize=9, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)

            # Beta distribution
            ax2 = axes[1]
            ax2.hist(beta, bins=50, color="#2b5797", alpha=0.7, linewidth=0)
            ax2.axvline(np.median(beta), color="#c0392b", lw=1.5,
                        label=f"median = {np.median(beta):.3f}")
            ax2.set_xlabel("β (splicing rate)", fontsize=10)
            ax2.set_ylabel("Gene count", fontsize=10)
            ax2.set_title("Distribution of β across genes", fontsize=10, fontweight="bold")
            ax2.legend(fontsize=9, frameon=False)
            ax2.spines[["top", "right"]].set_visible(False)

            plt.tight_layout()
            st.image(fig_png(fig), use_container_width=True)
            plt.close(fig)

            # β quality summary
            _beta_pos_pct = (beta > 0).mean() * 100
            _gamma_pos_pct = (np.median(adata_work.layers["gamma"], axis=0) > 0).mean() * 100
            _quality = "good" if _beta_pos_pct > 60 else "low"
            _quality_color = "#2d7d4b" if _quality == "good" else "#d4860a"
            st.markdown(
                f'<div class="ibox" style="font-size:12px">'
                f'<b>Data quality:</b> '
                f'{_beta_pos_pct:.0f}% of genes have estimable β · '
                f'{_gamma_pos_pct:.0f}% have estimable γ · '
                f'<span style="color:{_quality_color};font-weight:700">{_quality.upper()}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="sbox">'
                'β and γ estimated, variance decomposition complete. '
                '<b>Next:</b> cluster cells in γ-space to discover PT states. '
                'Or explore the phase portrait below for individual genes.'
                '</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.markdown(
                f'<div class="ebox"><b>Rate estimation failed:</b> {e}</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Traceback"):
                st.code(traceback.format_exc())

    # Phase portrait explorer (available once rates are estimated)
    if st.session_state.estimated:
        st.markdown('<div class="sl">Phase portrait explorer</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ibox" style="font-size:12px">'
            'Inspect the unspliced vs spliced phase portrait for any gene, colored by γ. '
            'The diagonal line shows the estimated β slope. Cells above = accelerating; below = decelerating.'
            '</div>',
            unsafe_allow_html=True,
        )
        adata_est = st.session_state.adata
        pp_col1, pp_col2 = st.columns([3, 1])
        with pp_col1:
            pp_gene = st.selectbox(
                "Gene",
                sorted(adata_est.var_names.tolist()),
                key="pp_gene_select",
            )
        with pp_col2:
            pp_cmap = st.selectbox("Color", ["viridis", "plasma", "RdBu_r", "coolwarm"], key="pp_cmap")
        if st.button("Plot Phase Portrait", key="pp_btn"):
            try:
                fig = scptr.pl.phase_portrait(
                    adata_est, genes=pp_gene, color_by="gamma",
                    cmap=pp_cmap, show=False,
                )
                if fig is not None:
                    st.image(fig_png(fig), width=450)
                    plt.close(fig)
            except Exception as e:
                st.markdown(f'<div class="ebox">Phase portrait failed: {e}</div>', unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("← Back"):
            go(2); st.rerun()
    with c2:
        if st.button("Next: Discover PT States →", disabled=not st.session_state.estimated):
            go(4); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS — STEP 4: DISCOVER PT STATES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "analysis" and st.session_state.step == 4:
    adata = st.session_state.adata
    st.markdown('<div class="pt">Discover PT States <span style="font-size:13px;font-weight:400;color:#aaa;letter-spacing:0">— step 4 of 5</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pd">Cluster cells in γ-space (PCA → kNN → Leiden) to reveal '
        'post-transcriptional states invisible to expression analysis. '
        'Optionally compute PT velocity and infer RBP–target regulatory networks.</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["PT STATES", "PT VELOCITY", "RBP NETWORKS"])

    # ── Tab 1: PT States ──────────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<div class="sl">Dimensionality reduction</div>', unsafe_allow_html=True)
            n_pcs_g = st.number_input(
                "PCA dims (γ-space)",
                value=20, min_value=5, max_value=50,
                help="PCA components computed from the γ matrix before clustering.",
            )
            n_nbrs_g = st.number_input(
                "Neighbors (γ-space kNN)",
                value=15, min_value=5, max_value=50,
                help="Number of nearest neighbors in γ-PCA space.",
            )
        with col2:
            st.markdown('<div class="sl">Leiden clustering</div>', unsafe_allow_html=True)
            resolution = st.slider(
                "Resolution",
                0.1, 2.0, 0.5, 0.05,
                help="Higher resolution → more, smaller clusters. "
                     "Typical range: 0.3–0.8. Start at 0.5 and increase if clusters are too coarse.",
            )
            rand_seed = st.number_input("Random seed", value=42, step=1)

        st.markdown(
            '<div class="ibox">'
            'PT state discovery performs PCA on the γ matrix, builds a kNN graph in γ-PCA space, '
            'then runs Leiden community detection. The resulting clusters represent groups of cells '
            'with similar post-transcriptional programs — which may differ from their expression-based identity.'
            '</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.states_done:
            n_s = adata.obs["pt_state"].nunique() if "pt_state" in adata.obs else "?"
            st.markdown(
                f'<div class="sbox">PT states already computed ({n_s} states). '
                f'Re-run to update with new parameters.</div>',
                unsafe_allow_html=True,
            )

        if st.button("Find PT States"):
            try:
                with st.spinner("Clustering cells in γ-space…"):
                    scptr.tl.pt_states(
                        adata,
                        n_pcs=n_pcs_g,
                        n_neighbors=n_nbrs_g,
                        resolution=resolution,
                        random_state=rand_seed,
                    )
                st.session_state.adata = adata
                st.session_state.states_done = True
                st.session_state.velocity_done = False
                st.session_state.network_done = False

                n_states = adata.obs["pt_state"].nunique()
                state_sizes = adata.obs["pt_state"].value_counts().sort_index()

                st.markdown(
                    mg(
                        ("PT States", str(n_states), "Leiden clusters in γ-space", "blue"),
                        ("Largest state", f"{state_sizes.max():,}", "cells"),
                        ("Smallest state", f"{state_sizes.min():,}", "cells"),
                    ),
                    unsafe_allow_html=True,
                )

                # Side-by-side UMAP: γ-space vs expression
                fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

                # γ-space UMAP
                coords_g = adata.obsm["X_gamma_umap"]
                labels = adata.obs["pt_state"].values
                unique_labels = sorted(set(labels), key=lambda x: int(x) if str(x).isdigit() else x)
                cmap = plt.colormaps["tab20"].resampled(len(unique_labels))
                label_map = {l: i for i, l in enumerate(unique_labels)}
                c = [label_map[l] for l in labels]
                sc_plot = axes[0].scatter(
                    coords_g[:, 0], coords_g[:, 1],
                    c=c, cmap=cmap, s=8, alpha=0.6, linewidths=0,
                )
                axes[0].set_title("PT states (γ-space UMAP)", fontsize=10, fontweight="bold")
                axes[0].axis("off")
                # legend
                for l in unique_labels:
                    axes[0].scatter([], [], color=cmap(label_map[l]), label=str(l), s=20)
                axes[0].legend(fontsize=8, frameon=False, ncol=2, loc="upper right")

                # Expression UMAP (if available)
                if "X_umap" in adata.obsm:
                    coords_e = adata.obsm["X_umap"]
                    axes[1].scatter(
                        coords_e[:, 0], coords_e[:, 1],
                        c=c, cmap=cmap, s=8, alpha=0.6, linewidths=0,
                    )
                    axes[1].set_title("PT states (expression UMAP)", fontsize=10, fontweight="bold")
                    axes[1].axis("off")
                else:
                    axes[1].text(0.5, 0.5, "No expression UMAP\n(add X_umap to adata.obsm)",
                                 ha="center", va="center", transform=axes[1].transAxes,
                                 fontsize=10, color="#888")
                    axes[1].axis("off")

                plt.tight_layout()
                st.image(fig_png(fig), use_container_width=True)
                plt.close(fig)

                # State size table
                rows = "".join(
                    f"<tr><td>State {s}</td><td>{n:,}</td>"
                    f'<td>{n / len(adata.obs) * 100:.1f}%</td></tr>'
                    for s, n in state_sizes.items()
                )
                st.markdown(
                    f'<table class="tbl" style="max-width:340px">'
                    f'<thead><tr><th>PT State</th><th>Cells</th><th>%</th></tr></thead>'
                    f'<tbody>{rows}</tbody></table>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="sbox">PT state discovery complete. '
                    'Proceed to Results or explore PT Velocity and RBP Networks.</div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.markdown(
                    f'<div class="ebox"><b>PT state discovery failed:</b> {e}</div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())

    # ── Tab 2: PT Velocity ────────────────────────────────────────────────────
    with tab2:
        if not st.session_state.states_done:
            st.markdown(
                '<div class="wbox">Run PT state discovery (Tab 1) before computing velocity.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="ibox">'
                'Post-transcriptional velocity computes, for each cell, the weighted mean '
                'difference in γ from its neighbors: <code>v[i,g] = Σ w_ij · (γ[j,g] − γ[i,g])</code>. '
                'This captures the direction and magnitude of post-transcriptional change, '
                'orthogonal to RNA velocity.'
                '</div>',
                unsafe_allow_html=True,
            )

            vel_c1, vel_c2 = st.columns(2, gap="large")
            with vel_c1:
                graph_choice = st.radio(
                    "Neighbor graph",
                    ["γ-space graph (from PT states)", "Expression-space graph"],
                    horizontal=False,
                    help="Which kNN graph to use when averaging neighbor γ values.",
                )
                use_graph = "gamma" if "γ-space" in graph_choice else "expression"
            with vel_c2:
                viz_style = st.radio(
                    "Visualization style",
                    ["Arrows (embedding)", "Streamlines"],
                    horizontal=False,
                    help="Arrows show per-cell velocity; streamlines show flow patterns.",
                )
                arrow_size = st.slider("Arrow size", 1.0, 8.0, 3.0, 0.5, key="vel_arrow")

            if st.session_state.velocity_done:
                st.markdown(
                    '<div class="sbox">PT velocity already computed. Re-run to update.</div>',
                    unsafe_allow_html=True,
                )

            if st.button("Compute PT Velocity"):
                try:
                    with st.spinner("Computing PT velocity…"):
                        scptr.tl.pt_velocity(adata, use_graph=use_graph)
                    st.session_state.adata = adata
                    st.session_state.velocity_done = True

                    fig, ax = plt.subplots(figsize=(6, 5))
                    try:
                        if "Stream" in viz_style:
                            scptr.pl.pt_velocity_stream(adata, ax=ax, show=False)
                        else:
                            scptr.pl.pt_velocity_embedding(adata, ax=ax, arrow_size=arrow_size, show=False)
                    except Exception:
                        coords = adata.obsm.get("X_gamma_umap", adata.obsm.get("X_umap"))
                        if coords is not None:
                            ax.scatter(coords[:, 0], coords[:, 1], s=6, alpha=0.4,
                                       color="#2b5797", linewidths=0)
                    ax.set_title(f"Post-transcriptional velocity ({'streamlines' if 'Stream' in viz_style else 'arrows'})", fontsize=10)
                    ax.axis("off")
                    plt.tight_layout()
                    st.image(fig_png(fig), width=520)
                    plt.close(fig)

                    st.markdown(
                        '<div class="sbox">PT velocity computed and stored in '
                        '<code>adata.layers["pt_velocity"]</code>.</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.markdown(
                        f'<div class="ebox"><b>PT velocity failed:</b> {e}</div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc())

    # ── Tab 3: RBP Networks ───────────────────────────────────────────────────
    with tab3:
        if not st.session_state.states_done:
            st.markdown(
                '<div class="wbox">Run PT state discovery (Tab 1) before inferring networks.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="ibox">'
                'RBP–target network inference regresses γ of each target gene on the '
                'smoothed expression of regulator genes using elastic net regression. '
                'Non-zero coefficients indicate putative regulatory relationships. '
                'Runtime scales with number of genes — start with defaults for speed.'
                '</div>',
                unsafe_allow_html=True,
            )

            # Core settings
            use_known_rbps = st.checkbox(
                "Restrict regulators to known RBPs (recommended — faster, more interpretable)",
                value=True,
                help="Filter regulators to curated RNA-binding proteins only.",
            )
            rbp_organism = None
            if use_known_rbps:
                rbp_organism = st.selectbox(
                    "RBP organism",
                    ["human", "mouse", None],
                    format_func=lambda x: x if x else "all species",
                    key="rbp_org",
                )

            with st.expander("Advanced options"):
                net_c1, net_c2 = st.columns(2, gap="large")
                with net_c1:
                    n_top = st.number_input(
                        "Top edges per target gene",
                        value=50, min_value=5, max_value=500,
                        help="Maximum number of regulator–target edges to retain per gene.",
                    )
                    alpha = st.slider(
                        "Elastic net mixing (α)",
                        0.0, 1.0, 0.5, 0.05,
                        help="0 = ridge (all regulators retained), 1 = lasso (sparse selection). 0.5 = elastic net.",
                    )
                with net_c2:
                    custom_regs = st.text_area(
                        "Additional regulators (optional)",
                        value="",
                        height=80,
                        help="Comma or newline-separated gene names. Appended to known RBPs if that option is checked.",
                    )
                    custom_tgts = st.text_area(
                        "Restrict to these targets (optional)",
                        value="",
                        height=80,
                        help="Leave empty to use all genes as targets.",
                    )

            if st.session_state.network_done:
                net = adata.uns.get("pt_network", pd.DataFrame())
                st.markdown(
                    f'<div class="sbox">Network already inferred ({len(net):,} edges). '
                    f'Re-run to update.</div>',
                    unsafe_allow_html=True,
                )

            if st.button("Infer RBP–Target Network"):
                regs = None
                tgts = None
                # Build regulator list
                if use_known_rbps:
                    try:
                        known = scptr.tl.list_known_rbps(organism=rbp_organism if use_known_rbps else None)
                        # Keep only those present in adata
                        known = [g for g in known if g in adata.var_names]
                        regs = known if known else None
                    except Exception:
                        regs = None
                if custom_regs.strip():
                    extra = [g.strip() for g in custom_regs.replace(",", "\n").split("\n") if g.strip()]
                    regs = list(set((regs or []) + extra)) if regs else extra
                if custom_tgts.strip():
                    tgts = [g.strip() for g in custom_tgts.replace(",", "\n").split("\n") if g.strip()]
                if regs:
                    st.markdown(
                        f'<div class="ibox" style="font-size:12px">Using {len(regs):,} regulators.</div>',
                        unsafe_allow_html=True,
                    )

                try:
                    with st.spinner("Running elastic net regression… (may take several minutes for large datasets)"):
                        scptr.tl.infer_network(
                            adata,
                            regulators=regs,
                            targets=tgts,
                            method="elasticnet",
                            alpha=alpha,
                            n_top=n_top,
                        )
                    st.session_state.adata = adata
                    st.session_state.network_done = True

                    net = adata.uns.get("pt_network", pd.DataFrame())
                    if len(net) == 0:
                        st.markdown(
                            '<div class="wbox">No significant edges found. '
                            'Try reducing α or increasing n_top.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        n_destab = int((net["weight"] > 0).sum())
                        n_stab = int((net["weight"] < 0).sum())
                        st.markdown(
                            mg(
                                ("Total edges", f"{len(net):,}", "regulator–target pairs"),
                                ("Destabilizing", f"{n_destab:,}", "weight > 0", "blue"),
                                ("Stabilizing", f"{n_stab:,}", "weight < 0", "green"),
                                ("Hub regulators", f"{net['regulator'].nunique():,}", "unique"),
                            ),
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            '<div class="ibox" style="font-size:12px">'
                            '<b>Edge weights:</b> positive = regulator destabilizes target (promotes degradation); '
                            'negative = regulator stabilizes target (protects from degradation).'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                        # Hub table with direction
                        hubs_dest = net[net["weight"] > 0].groupby("regulator").size()
                        hubs_stab = net[net["weight"] < 0].groupby("regulator").size()
                        hubs_all = net.groupby("regulator").size().sort_values(ascending=False).head(15)
                        rows = "".join(
                            f"<tr><td>{rbp}</td><td>{cnt:,}</td>"
                            f"<td>{hubs_dest.get(rbp, 0):,}</td>"
                            f"<td>{hubs_stab.get(rbp, 0):,}</td></tr>"
                            for rbp, cnt in hubs_all.items()
                        )
                        st.markdown(
                            f'<table class="tbl" style="max-width:500px">'
                            f'<thead><tr><th>Regulator</th><th>Total</th><th>Destab.</th><th>Stab.</th></tr></thead>'
                            f'<tbody>{rows}</tbody></table>',
                            unsafe_allow_html=True,
                        )

                        # Network plot
                        try:
                            fig, ax = plt.subplots(figsize=(7, 6))
                            scptr.pl.network_graph(adata, n_edges=60, ax=ax, show=False)
                            ax.set_title("Top regulatory edges", fontsize=10, fontweight="bold")
                            plt.tight_layout()
                            st.image(fig_png(fig), use_container_width=True)
                            plt.close(fig)
                        except Exception as _ne:
                            st.markdown(
                                f'<div class="wbox">Network plot could not be rendered: {_ne}</div>',
                                unsafe_allow_html=True,
                            )

                        st.markdown(
                            '<div class="sbox">Network inference complete. '
                            'Results stored in <code>adata.uns["pt_network"]</code>.</div>',
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    st.markdown(
                        f'<div class="ebox"><b>Network inference failed:</b> {e}</div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc())

    st.markdown('<hr>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("← Back"):
            go(3); st.rerun()
    with c2:
        if st.button("View Results →", disabled=not st.session_state.states_done):
            go(5); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS — STEP 5: RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "analysis" and st.session_state.step == 5:
    adata = st.session_state.adata
    st.markdown('<div class="pt">Results <span style="font-size:13px;font-weight:400;color:#aaa;letter-spacing:0">— step 5 of 5</span></div>', unsafe_allow_html=True)

    # Summary metrics
    gamma = adata.layers.get("gamma")
    gamma_pos = gamma[gamma > 0] if gamma is not None else np.array([])
    n_states = adata.obs["pt_state"].nunique() if "pt_state" in adata.obs else 0
    beta = adata.var.get("beta")
    net = adata.uns.get("pt_network", pd.DataFrame())

    st.markdown(
        mg(
            ("Cells", f"{adata.n_obs:,}", ""),
            ("Genes", f"{adata.n_vars:,}", ""),
            ("PT States", str(n_states), "γ-space Leiden", "blue"),
            ("Median β", f"{np.median(beta):.3f}" if beta is not None else "—", "splicing rate"),
            ("Median γ", f"{np.median(gamma_pos):.4f}" if len(gamma_pos) else "—", "positive cells"),
            ("Network edges", f"{len(net):,}" if len(net) else "—", "RBP–target", "green"),
        ),
        unsafe_allow_html=True,
    )

    # Analysis completion summary
    _done_items = []
    if beta is not None: _done_items.append("β estimated")
    if gamma is not None: _done_items.append("γ estimated")
    if "tf_score" in adata.var.columns: _done_items.append("variance decomposed")
    if n_states > 0: _done_items.append(f"{n_states} PT states")
    if st.session_state.velocity_done: _done_items.append("PT velocity")
    if st.session_state.network_done: _done_items.append(f"{len(net):,} network edges")
    if _done_items:
        st.markdown(
            '<div class="ibox" style="font-size:12px">'
            f'<b>Analyses complete:</b> {" · ".join(_done_items)}'
            '</div>',
            unsafe_allow_html=True,
        )

    rtab1, rtab2, rtab3, rtab4, rtab5 = st.tabs(["VISUALIZATION", "VARIANCE DECOMP", "GENE RANKINGS", "GAMMA EXPLORER", "DOWNLOAD"])

    # ── Visualization ─────────────────────────────────────────────────────────
    with rtab1:
        st.markdown(
            '<div class="ibox">Choose a column to color by and click <b>Plot UMAP</b> to visualize cells. '
            'Toggle <b>γ-space UMAP</b> to compare expression vs. post-transcriptional organization.</div>',
            unsafe_allow_html=True,
        )
        col_a, col_b = st.columns([2, 1])
        with col_a:
            color_opts = []
            if "pt_state" in adata.obs.columns:
                color_opts.append("pt_state")
            color_opts += [c for c in adata.obs.columns if c != "pt_state"][:30]
            viz_col = st.selectbox("Color by", color_opts, label_visibility="visible")
        with col_b:
            use_g_umap = st.checkbox(
                "γ-space UMAP",
                value=True,
                help="Use UMAP computed from γ-space PCA (vs. expression UMAP)",
            )

        if st.button("Plot UMAP"):
            basis = "X_gamma_umap" if (use_g_umap and "X_gamma_umap" in adata.obsm) else "X_umap"
            if basis not in adata.obsm and "X_umap" not in adata.obsm:
                with st.spinner("Computing expression UMAP…"):
                    sc.tl.umap(adata)
                basis = "X_umap"
            elif basis not in adata.obsm:
                basis = "X_umap"

            fig, ax = plt.subplots(figsize=(6, 5))
            sc.pl.embedding(
                adata, basis=basis, color=viz_col, ax=ax, show=False,
                frameon=False, size=14,
                title=f"{viz_col} · {'γ-space' if 'gamma' in basis else 'expression'} UMAP",
            )
            plt.tight_layout()
            st.image(fig_png(fig), use_container_width=True)
            plt.close(fig)

        # PT state UMAP using native scptr function
        if n_states > 0 and "X_gamma_umap" in adata.obsm:
            st.markdown('<hr>', unsafe_allow_html=True)
            if st.button("PT State UMAP (γ-space)"):
                try:
                    fig = scptr.pl.pt_umap(adata, show=False)
                    if fig is not None:
                        st.image(fig_png(fig), use_container_width=True)
                        plt.close(fig)
                except Exception as e:
                    st.markdown(f'<div class="ebox">PT UMAP failed: {e}</div>', unsafe_allow_html=True)

        if gamma is not None and n_states > 0:
            st.markdown('<hr>', unsafe_allow_html=True)
            if st.button("Plot Gamma Heatmap"):
                with st.spinner("Building heatmap…"):
                    try:
                        fig = scptr.pl.gamma_heatmap(adata, groupby="pt_state", show=False)
                        if fig is not None:
                            st.image(fig_png(fig), use_container_width=True)
                            plt.close(fig)
                    except Exception as e:
                        st.markdown(f'<div class="ebox">Heatmap failed: {e}</div>', unsafe_allow_html=True)

        if "X_gamma_umap" in adata.obsm and "X_umap" in adata.obsm:
            st.markdown('<hr>', unsafe_allow_html=True)
            if st.button("γ-space vs Expression Comparison"):
                with st.spinner("Building side-by-side comparison…"):
                    try:
                        fig = scptr.pl.pt_comparison(adata, figsize=(12, 5), show=False)
                        if fig:
                            st.image(fig_png(fig), use_container_width=True)
                            plt.close(fig)
                    except Exception as e:
                        # Fallback: manual side-by-side
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        for ax, basis, title in [
                            (axes[0], "X_umap", "Expression UMAP"),
                            (axes[1], "X_gamma_umap", "γ-space UMAP"),
                        ]:
                            coords = adata.obsm[basis]
                            if "pt_state" in adata.obs.columns:
                                labels = adata.obs["pt_state"].values
                                unique = sorted(set(labels), key=lambda x: int(x) if str(x).isdigit() else x)
                                cmap = plt.colormaps["tab20"].resampled(len(unique))
                                lmap = {l: i for i, l in enumerate(unique)}
                                c = [lmap[l] for l in labels]
                                ax.scatter(coords[:, 0], coords[:, 1], c=c, cmap=cmap,
                                           s=8, alpha=0.6, linewidths=0)
                            else:
                                ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.5,
                                           color="#2b5797", linewidths=0)
                            ax.set_title(title, fontsize=10, fontweight="bold")
                            ax.axis("off")
                        plt.tight_layout()
                        st.image(fig_png(fig), use_container_width=True)
                        plt.close(fig)

        if st.session_state.velocity_done:
            st.markdown('<hr>', unsafe_allow_html=True)
            if st.button("Plot PT Velocity"):
                try:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    scptr.pl.pt_velocity_embedding(adata, ax=ax, show=False)
                    ax.set_title("Post-transcriptional velocity", fontsize=10)
                    plt.tight_layout()
                    st.image(fig_png(fig), use_container_width=True)
                    plt.close(fig)
                except Exception as e:
                    st.markdown(f'<div class="ebox">Velocity plot failed: {e}</div>', unsafe_allow_html=True)

    # ── Variance decomposition ────────────────────────────────────────────────
    with rtab2:
        has_vd = "tf_score" in adata.var.columns and "ptf_score" in adata.var.columns
        if not has_vd:
            st.markdown(
                '<div class="wbox">Variance decomposition not computed. '
                'Return to step 3 and run Estimate β and γ (variance decomposition runs automatically).</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="ibox">'
                'Each gene\'s variance is decomposed into a <b>transcriptional fraction</b> (TF, driven by changes in '
                'unspliced/nascent RNA) and a <b>post-transcriptional fraction</b> (PTF, driven by changes in γ). '
                'Genes with high PTF score are primarily regulated post-transcriptionally.'
                '</div>',
                unsafe_allow_html=True,
            )

            tf = adata.var["tf_score"]
            ptf = adata.var["ptf_score"]

            # Aggregate stats
            ptf_dom = (ptf > 0.5).sum()
            tf_dom = (tf > 0.5).sum()
            st.markdown(
                mg(
                    ("PTF-dominant genes", f"{ptf_dom:,}", "PTF score > 0.5", "blue"),
                    ("TF-dominant genes", f"{tf_dom:,}", "TF score > 0.5"),
                    ("Median PTF score", f"{ptf.median():.3f}", "across genes"),
                ),
                unsafe_allow_html=True,
            )

            n_label = st.slider("Label top N genes", 5, 30, 10, 5, key="vd_label")
            if st.button("Plot Variance Decomposition"):
                try:
                    fig = scptr.pl.tf_ptf_scatter(adata, label_top=n_label, show=False)
                    if fig:
                        st.image(fig_png(fig), use_container_width=True)
                        plt.close(fig)
                except Exception as e:
                    # Fallback: manual plot
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
                    axes[0].hist(ptf, bins=50, color="#2b5797", alpha=0.75, linewidth=0)
                    axes[0].axvline(0.5, color="#c0392b", lw=1.5, ls="--", label="PTF=0.5")
                    axes[0].set_xlabel("PTF score", fontsize=10)
                    axes[0].set_ylabel("Gene count", fontsize=10)
                    axes[0].set_title("Distribution of PTF scores", fontsize=10, fontweight="bold")
                    axes[0].legend(fontsize=9, frameon=False)
                    axes[0].spines[["top","right"]].set_visible(False)

                    rank = np.argsort(ptf.values)
                    axes[1].scatter(range(len(ptf)), ptf.values[rank], s=4, alpha=0.5,
                                    color="#2b5797", linewidths=0)
                    axes[1].axhline(0.5, color="#c0392b", lw=1.2, ls="--")
                    axes[1].set_xlabel("Gene rank", fontsize=10)
                    axes[1].set_ylabel("PTF score", fontsize=10)
                    axes[1].set_title("Ranked PTF scores", fontsize=10, fontweight="bold")
                    axes[1].spines[["top","right"]].set_visible(False)
                    plt.tight_layout()
                    st.image(fig_png(fig), use_container_width=True)
                    plt.close(fig)

            # Top PTF genes table
            st.markdown('<div class="sl">Top post-transcriptionally regulated genes</div>', unsafe_allow_html=True)
            top_ptf = ptf.sort_values(ascending=False).head(30)
            ptf_df = pd.DataFrame({
                "Gene": top_ptf.index,
                "PTF score": top_ptf.values.round(4),
                "TF score": tf[top_ptf.index].values.round(4),
            })
            st.dataframe(ptf_df, use_container_width=True, hide_index=True, height=420)

    # ── Gene rankings ─────────────────────────────────────────────────────────
    with rtab3:
        if n_states == 0:
            st.markdown(
                '<div class="wbox">No PT states found. Run <b>Discover PT States → PT STATES</b> tab first, '
                'then return here to rank differentially degraded genes.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="ibox">Identifies genes with significantly different γ (degradation rate) '
                'between PT states — analogous to differential expression but in γ-space.</div>',
                unsafe_allow_html=True,
            )
            rank_method = st.selectbox(
                "Statistical test",
                ["t-test", "wilcoxon"],
                help="t-test is faster; Wilcoxon rank-sum is more robust to outliers.",
            )
            if st.button("Rank PT-Differential Genes"):
                with st.spinner("Running differential γ test…"):
                    try:
                        scptr.tl.rank_pt_genes(adata, method=rank_method)
                        st.session_state.adata = adata
                    except Exception as e:
                        st.markdown(f'<div class="ebox">Ranking failed: {e}</div>', unsafe_allow_html=True)

            if "rank_pt_genes" in adata.uns:
                names = adata.uns["rank_pt_genes"].get("names", {})
                if names:
                    groups = list(names.keys())
                    show_groups = st.multiselect(
                        "Show states",
                        groups,
                        default=groups[:min(4, len(groups))],
                    )
                    if show_groups:
                        top_n_genes = st.slider("Genes per state", 5, 50, 15, 5, key="rank_n")
                        all_rows = []
                        for g in show_groups:
                            gene_list = list(names[g])[:top_n_genes]
                            for rank, gene in enumerate(gene_list, 1):
                                all_rows.append({"State": f"PT {g}", "Rank": rank, "Gene": gene})
                        if all_rows:
                            rank_df = pd.DataFrame(all_rows)
                            st.dataframe(rank_df, use_container_width=True, hide_index=True, height=380)
                            st.download_button(
                                "Download ranked genes (.csv)",
                                rank_df.to_csv(index=False).encode(),
                                file_name=f"ranked_genes_{st.session_state.dataset_name}.csv",
                                mime="text/csv",
                            )

        if beta is not None:
            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown('<div class="sl">Genes by splicing rate (β)</div>', unsafe_allow_html=True)
            top_n = st.slider("Show top N genes", 10, 100, 30, 10, key="top_beta_n")
            top_beta = beta.sort_values(ascending=False).head(top_n)
            beta_df = pd.DataFrame({"Gene": top_beta.index, "β (splicing rate)": top_beta.values.round(5)})
            st.dataframe(beta_df, use_container_width=True, hide_index=True, height=350)

    # ── Gamma explorer ────────────────────────────────────────────────────────
    with rtab4:
        if gamma is None:
            st.markdown('<div class="wbox">No γ data available.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="ibox">Explore the distribution of degradation rates for individual genes '
                'or compare γ across PT states.</div>',
                unsafe_allow_html=True,
            )
            ge_col1, ge_col2 = st.columns([2, 1])
            with ge_col1:
                gene_input = st.selectbox(
                    "Select gene",
                    sorted(adata.var_names.tolist()),
                    key="gamma_gene",
                )
            with ge_col2:
                groupby_col = "pt_state" if "pt_state" in adata.obs.columns else None
                if groupby_col is None:
                    st.markdown('<div style="font-size:12px;color:#888;margin-top:1.5rem">No PT states computed yet</div>', unsafe_allow_html=True)

            if gene_input and st.button("Plot γ for selected gene"):
                g_idx = adata.var_names.get_loc(gene_input)
                g_vals = gamma[:, g_idx]
                g_pos = g_vals[g_vals > 0]

                # Try scptr.pl.gamma_violin first, fall back to manual
                if groupby_col:
                    try:
                        fig = scptr.pl.gamma_violin(adata, genes=gene_input, groupby=groupby_col, show=False)
                        if fig:
                            st.image(fig_png(fig), use_container_width=True)
                            plt.close(fig)
                        raise StopIteration  # skip fallback
                    except StopIteration:
                        pass
                    except Exception:
                        pass  # fall through to manual

                # Manual histogram + violin fallback
                fig, axes = plt.subplots(1, 2 if groupby_col else 1, figsize=(9 if groupby_col else 5, 3.8))
                ax0 = axes[0] if groupby_col else axes

                ax0.hist(g_pos, bins=40, color="#2b5797", alpha=0.75, linewidth=0)
                if len(g_pos):
                    ax0.axvline(np.median(g_pos), color="#c0392b", lw=1.5,
                                label=f"median = {np.median(g_pos):.4f}")
                    ax0.legend(fontsize=9, frameon=False)
                ax0.set_xlabel(f"γ ({gene_input})", fontsize=10)
                ax0.set_ylabel("Cell count", fontsize=10)
                ax0.set_title(f"γ distribution: {gene_input}", fontsize=10, fontweight="bold")
                ax0.spines[["top", "right"]].set_visible(False)

                if groupby_col:
                    states = sorted(adata.obs[groupby_col].unique(),
                                    key=lambda x: int(x) if str(x).isdigit() else x)
                    data_by_state = [g_vals[adata.obs[groupby_col] == s] for s in states]
                    vp = axes[1].violinplot(data_by_state, positions=range(len(states)),
                                            showmedians=True, showextrema=False)
                    for body in vp["bodies"]:
                        body.set_facecolor("#2b5797"); body.set_alpha(0.6)
                    vp["cmedians"].set_color("#c0392b")
                    axes[1].set_xticks(range(len(states)))
                    axes[1].set_xticklabels([str(s) for s in states], fontsize=9)
                    axes[1].set_xlabel("PT State", fontsize=10)
                    axes[1].set_ylabel(f"γ ({gene_input})", fontsize=10)
                    axes[1].set_title("γ by PT state", fontsize=10, fontweight="bold")
                    axes[1].spines[["top", "right"]].set_visible(False)

                plt.tight_layout()
                st.image(fig_png(fig), use_container_width=True)
                plt.close(fig)

                st.markdown(
                    mg(
                        ("Median γ", f"{np.median(g_pos):.4f}" if len(g_pos) else "—", "positive cells"),
                        ("Max γ", f"{np.max(g_vals):.4f}", ""),
                        ("Cells γ > 0", f"{int((g_vals > 0).sum()):,}", f"of {adata.n_obs:,}"),
                    ),
                    unsafe_allow_html=True,
                )

    # ── Download ──────────────────────────────────────────────────────────────
    with rtab5:
        st.markdown('<div class="sl">Export results</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Full AnnData**")
            st.markdown(
                '<div style="font-size:12px;color:#555;margin-bottom:0.5rem">'
                'All layers (γ, β, Ms, Mu) plus PT state assignments and embeddings.</div>',
                unsafe_allow_html=True,
            )
            if st.button("Prepare AnnData (.h5ad)"):
                tmp = make_tmp(".h5ad")
                with st.spinner("Writing…"):
                    adata.write_h5ad(tmp)
                with open(tmp, "rb") as f:
                    st.download_button(
                        "Download .h5ad",
                        f.read(),
                        file_name=f"scptr_{st.session_state.dataset_name}.h5ad",
                        mime="application/octet-stream",
                    )

        with col2:
            if gamma is not None:
                st.markdown("**Gamma matrix**")
                st.markdown(
                    '<div style="font-size:12px;color:#555;margin-bottom:0.5rem">'
                    'Cells × genes CSV of degradation rates γ.</div>',
                    unsafe_allow_html=True,
                )
                if st.button("Prepare γ matrix (.csv)"):
                    with st.spinner("Building CSV…"):
                        gamma_df = pd.DataFrame(
                            gamma,
                            index=adata.obs_names,
                            columns=adata.var_names,
                        )
                        csv_bytes = gamma_df.to_csv().encode()
                    st.download_button(
                        "Download gamma.csv",
                        csv_bytes,
                        file_name=f"gamma_{st.session_state.dataset_name}.csv",
                        mime="text/csv",
                    )

        with col3:
            st.markdown("**Cell metadata**")
            st.markdown(
                '<div style="font-size:12px;color:#555;margin-bottom:0.5rem">'
                'PT state assignments per cell (CSV).</div>',
                unsafe_allow_html=True,
            )
            obs_cols = [c for c in ["pt_state"] if c in adata.obs.columns]
            if obs_cols:
                obs_csv = adata.obs[obs_cols].copy().to_csv().encode()
                st.download_button(
                    "Download metadata.csv",
                    obs_csv,
                    file_name=f"metadata_{st.session_state.dataset_name}.csv",
                    mime="text/csv",
                )
            else:
                st.markdown(
                    '<div class="wbox" style="font-size:12px">Run <b>Discover PT States</b> first to populate cell metadata.</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<hr>', unsafe_allow_html=True)
        col_d1, col_d2, col_d3 = st.columns(3)

        with col_d1:
            if beta is not None:
                st.markdown("**Gene splicing rates (β)**")
                st.markdown(
                    '<div style="font-size:12px;color:#555;margin-bottom:0.5rem">'
                    'Per-gene β from phase portrait quantile regression.</div>',
                    unsafe_allow_html=True,
                )
                beta_df = pd.DataFrame({"gene": beta.index, "beta": beta.values})
                st.download_button(
                    "Download beta.csv",
                    beta_df.to_csv(index=False).encode(),
                    file_name=f"beta_{st.session_state.dataset_name}.csv",
                    mime="text/csv",
                )

        with col_d2:
            if "tf_score" in adata.var.columns:
                st.markdown("**Variance decomposition**")
                st.markdown(
                    '<div style="font-size:12px;color:#555;margin-bottom:0.5rem">'
                    'TF and PTF score per gene (0–1).</div>',
                    unsafe_allow_html=True,
                )
                vd_df = pd.DataFrame({
                    "gene": adata.var_names,
                    "tf_score": adata.var["tf_score"].values,
                    "ptf_score": adata.var["ptf_score"].values,
                })
                st.download_button(
                    "Download variance_decomp.csv",
                    vd_df.to_csv(index=False).encode(),
                    file_name=f"variance_decomp_{st.session_state.dataset_name}.csv",
                    mime="text/csv",
                )

        with col_d3:
            st.markdown("**Analysis parameters**")
            st.markdown(
                '<div style="font-size:12px;color:#555;margin-bottom:0.5rem">'
                'Reproducibility: parameters logged by scPTR.</div>',
                unsafe_allow_html=True,
            )
            params_log = adata.uns.get("scptr", {})
            if params_log:
                st.download_button(
                    "Download parameters.json",
                    json.dumps(params_log, indent=2, default=str).encode(),
                    file_name=f"parameters_{st.session_state.dataset_name}.json",
                    mime="application/json",
                )
            else:
                st.markdown(
                    '<div class="wbox" style="font-size:12px">No logged parameters yet — run the full pipeline to populate.</div>',
                    unsafe_allow_html=True,
                )

        if len(net) > 0:
            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown("**RBP–target network**")
            st.markdown(
                '<div style="font-size:12px;color:#555;margin-bottom:0.5rem">'
                'Columns: regulator, target, weight. Positive weight = destabilizing; negative = stabilizing.</div>',
                unsafe_allow_html=True,
            )
            net_csv = net.to_csv(index=False).encode()
            st.download_button(
                "Download network.csv",
                net_csv,
                file_name=f"network_{st.session_state.dataset_name}.csv",
                mime="text/csv",
            )

    st.markdown('<hr>', unsafe_allow_html=True)
    if st.button("← Start Over"):
        clean_tmps()
        for k, v in {
            "step": 1,
            "adata": None,
            "dataset_name": None,
            "preprocessed": False,
            "estimated": False,
            "states_done": False,
            "velocity_done": False,
            "network_done": False,
        }.items():
            st.session_state[k] = v
        nav("analysis"); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENTATION PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "docs":
    st.markdown('<div class="pt">Documentation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pd">Method details, parameter reference, and interpretation guide.</div>',
        unsafe_allow_html=True,
    )

    dtab1, dtab2, dtab3, dtab4, dtab5, dtab6 = st.tabs(["METHOD", "PARAMETERS", "INTERPRETATION", "QUICK START", "TROUBLESHOOTING", "CITATION"])

    # ── Method ────────────────────────────────────────────────────────────────
    with dtab1:
        st.markdown(
            '<div class="doc-sec">'
            '<div class="doc-h2">The kinetic model</div>'
            '<div class="doc-p">'
            'scPTR is based on the kinetic model of RNA metabolism where each gene g has a '
            'transcription rate α, splicing rate β, and degradation rate γ. At steady state, '
            'the rates satisfy:'
            '</div>'
            '<div class="doc-code">'
            'du/dt = α  −  β · u  =  0   →   u* = α/β\n'
            'ds/dt = β · u  −  γ · s  =  0   →   s* = α/γ\n\n'
            'Therefore:  γ_ig = β_g · u_ig / s_ig'
            '</div>'
            '<div class="doc-p">'
            'This means γ is directly estimable from observed spliced and unspliced counts, '
            'given an estimate of β. Critically, γ can vary <em>per cell</em> because of '
            'post-transcriptional regulatory programs (miRNA-mediated repression, RBP binding, '
            'codon usage, etc.).'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="doc-sec">'
            '<div class="doc-h2">Beta estimation</div>'
            '<div class="doc-p">'
            'The splicing rate β<sub>g</sub> is gene-specific and estimated from the phase portrait '
            '(u vs. s plot) using quantile regression on the upper boundary of the u/s ratio. '
            'This approach captures the slope of the kinetic upper boundary, which reflects the '
            'maximum u/s ratio observed across cells — corresponding to minimal degradation.'
            '</div>'
            '<div class="doc-p">'
            'Concretely, β̂<sub>g</sub> = quantile(u<sub>ig</sub> / s<sub>ig</sub>, q=0.95) '
            'clipped at the 99th percentile of positive values.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="doc-sec">'
            '<div class="doc-h2">Post-transcriptional state discovery</div>'
            '<div class="doc-p">'
            'To identify cell populations with distinct degradation programs, scPTR:'
            '</div>'
            '<ol style="font-size:13px;color:#333;line-height:2;margin:0 0 0.5rem 1.25rem">'
            '<li>Computes PCA on the γ matrix (cells × genes)</li>'
            '<li>Builds a kNN graph in γ-PCA space</li>'
            '<li>Runs Leiden community detection</li>'
            '<li>Embeds in 2D using UMAP for visualization</li>'
            '</ol>'
            '<div class="doc-p">'
            'These clusters can differ from expression-based clusters — a cell may be '
            'transcriptionally similar to its neighbors but have a distinct degradation program '
            '("expression-invisible" state).'
            '</div>'
            '<div class="doc-p">'
            '<b>Zero-permutation control:</b> scPTR validates that PT states are not artifactual '
            'by permuting γ values randomly across cells and recomputing clusters. The adjusted '
            'Rand index (ARI) between real and permuted PT states should be near 0.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="doc-sec">'
            '<div class="doc-h2">RBP–target network inference</div>'
            '<div class="doc-p">'
            'For each target gene, scPTR regresses its γ values across cells on the '
            'smoothed expression of regulator genes (putative RBPs) using elastic net regression:'
            '</div>'
            '<div class="doc-code">γ_target ~ Σ_r  coef_r · expr_r  +  ε</div>'
            '<div class="doc-p">'
            'Non-zero coefficients indicate regulatory relationships. Positive coefficients imply '
            'the regulator promotes degradation (destabilizing); negative implies protection '
            '(stabilizing).'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Parameters ────────────────────────────────────────────────────────────
    with dtab2:
        def param_row(name, type_, default, desc):
            return (
                f'<tr>'
                f'<td><span class="param-name">{name}</span></td>'
                f'<td><span class="param-type">{type_}</span></td>'
                f'<td>{default}</td>'
                f'<td class="txt param-desc">{desc}</td>'
                f'</tr>'
            )

        tables = [
            ("pp.filter_genes", [
                ("min_unspliced_counts", "int", "10", "Minimum total unspliced count across all cells."),
                ("min_unspliced_cells", "int", "5", "Minimum cells with nonzero unspliced expression."),
                ("min_spliced_counts", "int", "0", "Minimum total spliced count (rarely changed)."),
            ]),
            ("pp.neighbors", [
                ("n_neighbors", "int", "30", "Number of nearest neighbors for the cell graph."),
                ("n_pcs", "int", "30", "PCA components used to build the neighbor graph."),
            ]),
            ("pp.smooth_layers", [
                ("bandwidth", "float | None", "None", "Gaussian kernel bandwidth. None = adaptive (median kNN distance)."),
            ]),
            ("tl.estimate_beta", [
                ("quantile", "float", "0.95", "Upper quantile of u/s ratio used as the kinetic boundary."),
            ]),
            ("tl.estimate_gamma", [
                ("clip_quantile", "float", "0.99", "Per-gene clipping quantile to remove extreme γ values."),
                ("min_spliced", "float", "0.01", "Minimum smoothed spliced (Ms) for reliable γ; below this γ = 0."),
                ("mode", "str", "'steady_state'", "'steady_state' or 'dynamic' (uses ds/dt from RNA velocity)."),
            ]),
            ("tl.pt_states", [
                ("resolution", "float", "0.5", "Leiden resolution. Higher → more, smaller clusters."),
                ("n_pcs", "int", "30", "PCA components computed from the γ matrix."),
                ("n_neighbors", "int", "30", "kNN neighbors in γ-PCA space."),
                ("random_state", "int", "0", "Random seed for PCA, UMAP, and Leiden."),
            ]),
            ("tl.pt_velocity", [
                ("use_graph", "str", "'gamma'", "'gamma' = γ-space kNN graph; 'expression' = expression kNN graph."),
            ]),
            ("tl.infer_network", [
                ("regulators", "list | None", "None", "Regulator gene names. None = all genes used as regulators."),
                ("targets", "list | None", "None", "Target gene names. None = all genes used as targets."),
                ("method", "str", "'elasticnet'", "Regression method. Currently only 'elasticnet'."),
                ("alpha", "float", "0.5", "Elastic net mixing: 0 = ridge, 1 = lasso, 0.5 = elastic net."),
                ("n_top", "int", "50", "Maximum edges to retain per target gene."),
            ]),
        ]

        for fn_name, params in tables:
            rows = "".join(param_row(*p) for p in params)
            st.markdown(
                f'<div class="doc-sec">'
                f'<div class="doc-h2"><code>{fn_name}</code></div>'
                f'<table class="tbl">'
                f'<thead><tr><th>Parameter</th><th>Type</th><th>Default</th><th>Description</th></tr></thead>'
                f'<tbody>{rows}</tbody>'
                f'</table>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Interpretation ────────────────────────────────────────────────────────
    with dtab3:
        st.markdown(
            '<div class="doc-sec">'
            '<div class="doc-h2">Interpreting γ values</div>'
            '<div class="doc-p">'
            'γ (gamma) represents the <b>mRNA degradation rate</b> — higher γ means faster '
            'turnover. Key relationships:'
            '</div>'
            '<table class="tbl">'
            '<thead><tr><th>γ value</th><th>Meaning</th></tr></thead>'
            '<tbody>'
            '<tr><td>γ = 0</td><td class="txt">Insufficient spliced counts; rate not estimable</td></tr>'
            '<tr><td>γ → small (e.g., 0.001)</td><td class="txt">Slow degradation; stable mRNA</td></tr>'
            '<tr><td>γ → large (e.g., 0.5+)</td><td class="txt">Rapid degradation; unstable mRNA</td></tr>'
            '<tr><td>γ varies across cells</td><td class="txt">Post-transcriptional regulation; likely RBP or miRNA activity</td></tr>'
            '</tbody></table>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="doc-sec">'
            '<div class="doc-h2">Interpreting PT states</div>'
            '<div class="doc-p">'
            'A PT state is a cluster of cells with similar γ profiles. Biologically, this means:'
            '</div>'
            '<ul style="font-size:13px;color:#333;line-height:2;margin:0 0 0.5rem 1.25rem">'
            '<li><b>Same PT state, same expression cluster</b>: transcriptional and post-transcriptional programs are aligned</li>'
            '<li><b>Different PT states within one expression cluster</b>: expression-invisible state — cells look identical transcriptionally but differ in RNA stability</li>'
            '<li><b>Same PT state, different expression clusters</b>: convergent post-transcriptional programs across distinct cell types</li>'
            '</ul>'
            '<div class="doc-p">'
            'Use the <b>silhouette score</b> to assess separation: higher in γ-space than '
            'expression-space indicates a genuine post-transcriptional signature.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="doc-sec">'
            '<div class="doc-h2">Interpreting PT velocity</div>'
            '<div class="doc-p">'
            'PT velocity captures the <b>direction and magnitude of change in γ</b> across the '
            'γ-space neighborhood graph. Unlike RNA velocity (which reflects nascent → mature RNA '
            'flow), PT velocity is orthogonal — it reflects where in γ-space a cell is headed.'
            '</div>'
            '<ul style="font-size:13px;color:#333;line-height:2;margin:0 0 0.5rem 1.25rem">'
            '<li><b>Arrow direction</b>: predicted future γ profile of the cell</li>'
            '<li><b>Arrow length</b>: magnitude of degradation-rate change (longer = faster transition)</li>'
            '<li><b>Coherent arrows</b> in a region suggest a shared post-transcriptional program being gained or lost</li>'
            '<li><b>Divergent arrows</b> suggest a decision point where cells adopt different degradation programs</li>'
            '</ul>'
            '<div class="doc-p">'
            'PT velocity precedes expression changes for 54–78% of transition genes '
            '(pancreas p &lt; 10⁻⁵⁷, dentate gyrus p = 9.9×10⁻¹³), making it a leading '
            'indicator of cell-fate decisions.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="doc-sec">'
            '<div class="doc-h2">Interpreting network edges</div>'
            '<div class="doc-p">'
            'RNA-binding proteins (RBPs) are post-transcriptional regulators that bind mRNA to control '
            'stability, splicing, and translation. scPTR infers RBP→target edges where RBP expression '
            'predicts γ of the target, using library-size-corrected elastic net regression.'
            '</div>'
            '<div class="doc-p">'
            'Each edge has a <b>weight</b> (regression coefficient):'
            '</div>'
            '<ul style="font-size:13px;color:#333;line-height:2;margin:0 0 0.5rem 1.25rem">'
            '<li><b>Positive weight</b>: RBP expression correlates with higher γ → '
            'the RBP <em>destabilizes</em> the target mRNA</li>'
            '<li><b>Negative weight</b>: RBP expression correlates with lower γ → '
            'the RBP <em>stabilizes</em> the target mRNA</li>'
            '<li><b>|weight| magnitude</b>: filter by |weight| &gt; 0.05 to retain high-confidence edges</li>'
            '<li><b>Hub regulators</b> (many edges) are likely master regulators '
            '— validated hubs include HNRNPA1, YBX1, ELAVL1/HuR, RBFOX1</li>'
            '</ul>'
            '<div class="doc-p">'
            'To validate edges, cross-reference with CLIP-seq databases (e.g., ENCODE eCLIP, '
            'ATtRACT) or miRNA target databases (TargetScan) for experimental support.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Quick Start ───────────────────────────────────────────────────────────
    with dtab4:
        st.markdown(
            '<div class="doc-sec">'
            '<div class="doc-h2">Python quick start</div>'
            '<div class="doc-code">'
            'import scptr\n\n'
            '# Load data (must have spliced + unspliced layers)\n'
            'adata = scptr.read_h5ad("your_data.h5ad")\n\n'
            '# Preprocessing\n'
            'scptr.pp.filter_genes(adata, min_unspliced_counts=10, min_unspliced_cells=5)\n'
            'scptr.pp.normalize_layers(adata)\n'
            'scptr.pp.neighbors(adata, n_neighbors=30, n_pcs=30)\n'
            'scptr.pp.smooth_layers(adata)  # Gaussian kernel smoothing\n\n'
            '# Rate estimation\n'
            'scptr.tl.estimate_beta(adata, quantile=0.95)\n'
            'scptr.tl.estimate_gamma(adata, clip_quantile=0.99, min_spliced=0.01)\n'
            'scptr.tl.variance_decomposition(adata)\n\n'
            '# PT state discovery\n'
            'scptr.tl.pt_states(adata, resolution=0.5, n_pcs=20, n_neighbors=15)\n\n'
            '# Optional: PT velocity\n'
            'scptr.tl.pt_velocity(adata, use_graph="gamma")\n\n'
            '# Optional: RBP–target network\n'
            'net = scptr.tl.infer_network(adata, alpha=0.5, n_top=50)\n'
            '# Results in adata.uns["pt_network"]\n\n'
            '# Visualization\n'
            'scptr.pl.pt_umap(adata)  # γ-space UMAP\n'
            'scptr.pl.gamma_heatmap(adata, groupby="pt_state")\n'
            'scptr.pl.network_graph(adata, n_edges=50)'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="doc-sec">'
            '<div class="doc-h2">Installation</div>'
            '<div class="doc-code">'
            'pip install .                    # core package\n'
            'pip install ".[datasets]"        # built-in pancreas / dentate gyrus\n'
            'pip install ".[deep]"            # DeepPTR (PyTorch)\n'
            'pip install ".[dev]"             # pytest for testing'
            '</div>'
            '<div class="doc-h3">Requirements</div>'
            '<div class="doc-p">'
            'Python ≥ 3.9, anndata ≥ 0.8, scanpy ≥ 1.9, numpy ≥ 1.21, scipy ≥ 1.7, '
            'scikit-learn ≥ 1.0, numba ≥ 0.55, matplotlib ≥ 3.5'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="doc-sec">'
            '<div class="doc-h2">Data preparation</div>'
            '<div class="doc-p">'
            'scPTR requires <b>raw, un-normalized</b> spliced and unspliced counts '
            'in an AnnData <code>.h5ad</code> file. Tools for generating these:'
            '</div>'
            '<table class="tbl">'
            '<thead><tr><th>Tool</th><th>Notes</th></tr></thead>'
            '<tbody>'
            '<tr><td>STARsolo</td><td class="txt">Produces spliced/unspliced with <code>--soloFeatures SJ Gene Velocyto</code></td></tr>'
            '<tr><td>Alevin-fry</td><td class="txt">Use <code>splici</code> index; outputs spliced + unspliced per cell</td></tr>'
            '<tr><td>velocyto</td><td class="txt">Run on BAM files post-alignment; outputs loom → convert to h5ad</td></tr>'
            '<tr><td>kallisto|bustools</td><td class="txt">Use kb-python with <code>--workflow lamanno</code></td></tr>'
            '</tbody></table>'
            '<div class="doc-h3">Input data requirements</div>'
            '<div class="doc-p">'
            'scPTR requires <b>raw (un-normalized) counts</b> in both layers. '
            'Do <em>not</em> pre-normalize with Seurat, Scanpy, or other pipelines — '
            'scPTR performs its own library-size normalization internally. '
            'STARsolo and Alevin-fry output is already raw; velocyto loom files are also raw.'
            '</div>'
            '<div class="doc-h3">Verify your data has required layers</div>'
            '<div class="doc-code">'
            'import anndata as ad\n'
            'adata = ad.read_h5ad("your_data.h5ad")\n'
            'print(list(adata.layers.keys()))  # must include "spliced" and "unspliced"\n'
            'print(adata.shape)                # (n_cells, n_genes)\n'
            '# Check counts are raw (integers)\n'
            'import numpy as np\n'
            'print(np.allclose(adata.layers["spliced"] % 1, 0))  # True = raw counts'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Troubleshooting ───────────────────────────────────────────────────────
    with dtab5:
        issues = [
            ("Missing spliced/unspliced layers",
             "Error: 'spliced' not in adata.layers",
             "Your file lacks the required layers. Re-run alignment with STARsolo "
             "(<code>--soloFeatures Velocyto</code>), velocyto, or Alevin-fry with a splici index."),
            ("All γ values are zero",
             "Gamma layer exists but all values are 0",
             "This happens when Ms (smoothed spliced) is below the <code>min_spliced</code> threshold everywhere. "
             "Try reducing Min smoothed spliced (Ms) to 0.001 in step 3, or check that normalization ran correctly."),
            ("Very few genes after filtering",
             "n_vars drops to &lt; 100 after preprocessing",
             "Lower <em>Min total unspliced counts</em> or <em>Min cells with nonzero unspliced</em> in step 2. "
             "Some datasets have sparse unspliced counts — try 5 and 3 respectively."),
            ("PT states = 1 (all cells in one cluster)",
             "Only 1 Leiden cluster found",
             "Increase the Leiden resolution (try 1.0–2.0). Also check that the γ matrix has variance — "
             "if most values are 0, the clustering will be uninformative."),
            ("Network inference is slow",
             "Elastic net taking &gt; 10 minutes",
             "Check 'Restrict regulators to known RBPs' to reduce the feature matrix from all genes "
             "to ~200 curated RBPs. Also reduce <em>Top edges per target</em> to 20–30."),
            ("Dataset download fails",
             "pooch.HTTPError or FileNotFoundError for example datasets",
             "Ensure <code>pip install \"scptr[datasets]\"</code> is installed and you have an internet connection. "
             "The pancreas dataset is ~30 MB and dentate gyrus ~25 MB."),
            ("Memory error on large dataset",
             "MemoryError or kernel crash",
             "The γ matrix (cells × genes) can be large. Try filtering more aggressively in step 2 "
             "(higher min counts). For &gt;20,000 cells, consider subsetting to highly variable genes first."),
            ("Phase portrait shows no clear line",
             "β estimates are noisy / R² is very low",
             "This is normal for some genes. Raise the <em>Phase portrait quantile</em> to 0.97–0.99 "
             "to better capture the kinetic upper boundary. Genes with very low counts will always have noisy portraits."),
        ]
        for title, symptom, fix in issues:
            st.markdown(
                f'<div class="doc-sec">'
                f'<div class="doc-h2">{title}</div>'
                f'<div class="doc-h3">Symptom</div>'
                f'<div class="doc-code">{symptom}</div>'
                f'<div class="doc-h3">Fix</div>'
                f'<div class="doc-p">{fix}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Citation ──────────────────────────────────────────────────────────────
    with dtab6:
        st.markdown(
            '<div class="doc-sec">'
            '<div class="doc-h2">Citing scPTR</div>'
            '<div class="doc-p">'
            'If you use scPTR in your research, please cite:'
            '</div>'
            '<div class="doc-code">'
            'Bryan Cheng*, Austin Jin*\n'
            'scPTR: Decomposing Post-Transcriptional Regulation at Single-Cell Resolution\n'
            '(2026)'
            '</div>'
            '<div class="doc-h3">BibTeX</div>'
            '<div class="doc-code">'
            '@article{cheng2026scptr,\n'
            '  title   = {scPTR: Decomposing Post-Transcriptional Regulation\n'
            '             at Single-Cell Resolution},\n'
            '  author  = {Cheng, Bryan and Jin, Austin},\n'
            '  year    = {2026},\n'
            '}'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="doc-sec">'
            '<div class="doc-h2">Dependencies to cite</div>'
            '<div class="doc-p">'
            'scPTR builds on these foundational tools — please also cite them as appropriate:'
            '</div>'
            '<table class="tbl">'
            '<thead><tr><th>Tool</th><th>Use in scPTR</th><th>Reference</th></tr></thead>'
            '<tbody>'
            '<tr><td>scanpy</td><td class="txt">PCA, neighbors, UMAP, Leiden clustering</td>'
            '<td class="txt">Wolf et al., Genome Biology 2018</td></tr>'
            '<tr><td>anndata</td><td class="txt">Core data structure</td>'
            '<td class="txt">Virshup et al., Nat Methods 2024</td></tr>'
            '<tr><td>scikit-learn</td><td class="txt">Elastic net regression (network inference)</td>'
            '<td class="txt">Pedregosa et al., JMLR 2011</td></tr>'
            '<tr><td>scVelo</td><td class="txt">Steady-state kinetic model inspiration</td>'
            '<td class="txt">Bergen et al., Nat Biotechnol 2020</td></tr>'
            '</tbody></table>'
            '</div>',
            unsafe_allow_html=True,
        )
