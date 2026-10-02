from pathlib import Path
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="BloomDetect AI", page_icon="🌊", layout="wide", initial_sidebar_state="collapsed")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "latest_bloom_risk_predictions.csv"
INFOGRAPHIC_PATH = BASE_DIR / "bloomdetect_bloom_process.png"

# -----------------------------
# DATA
# -----------------------------
@st.cache_data(show_spinner="Loading satellite observations...")
def load_data():
    """Load the existing prediction CSV. If the deployed local copy is empty,
    try the public GitHub raw file as a safety fallback."""
    local_error = None

    def clean(d):
        if d is None or d.empty:
            raise pd.errors.EmptyDataError("The prediction CSV is empty.")
        d.columns = [str(c).strip().lower().replace(" ", "_") for c in d.columns]
        required = {"latitude", "longitude", "date", "chla", "risk_label"}
        missing = required - set(d.columns)
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        for c in ["latitude", "longitude", "chla"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        for c in ["risk_probability", "model_score", "previous_chla", "historical_baseline", "recent_mean", "recent_max", "chla_anomaly", "chla_change"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["latitude", "longitude", "date", "chla"]).copy()
        if d.empty:
            raise ValueError("The CSV contains no valid observation rows after cleaning.")
        d["risk_flag"] = d["risk_label"].astype(str).str.strip().str.lower().eq("potential bloom risk")
        d["plot_lon"] = ((d["longitude"] + 180) % 360) - 180
        return d

    try:
        if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
            raise pd.errors.EmptyDataError("Local prediction CSV is missing or empty.")
        return clean(pd.read_csv(DATA_PATH))
    except Exception as exc:
        local_error = exc

    # Public GitHub fallback. This does not change the dataset or model.
    raw_url = "https://raw.githubusercontent.com/TenetiSrujana/BloomDetect-AI/main/latest_bloom_risk_predictions.csv"
    try:
        return clean(pd.read_csv(raw_url))
    except Exception as remote_error:
        raise RuntimeError(
            "The prediction CSV could not be loaded locally or from the GitHub fallback. "
            f"Local error: {local_error}. GitHub error: {remote_error}"
        )

try:
    df = load_data()
except Exception as e:
    st.error("BloomDetect could not load the prediction dataset.")
    st.markdown(
        "**Most likely cause:** `latest_bloom_risk_predictions.csv` in GitHub is empty, "
        "truncated, or not the real prediction CSV. The app also tried the public GitHub raw copy."
    )
    st.code(str(e))
    st.stop()

latest_date = df["date"].max()
latest = df[df["date"] == latest_date].copy()
risk = latest[latest["risk_flag"]].copy()

# Project screening thresholds used when the proxy target was created.
ANOMALY_THRESHOLD = 0.059076173328496344
CHANGE_THRESHOLD = 0.02007450088858604

# -----------------------------
# NAVIGATION
# -----------------------------
st.markdown('<div class="brand-bar"><div class="brand-left"><div class="logo">≈</div><div><div class="brand-name">BloomDetect AI</div><div class="brand-sub">Satellite-based potential bloom-risk screening</div></div></div></div>', unsafe_allow_html=True)
nav_cols = st.columns(8, gap="small")
for col, page in zip(nav_cols, PAGES):
    with col:
        if st.button(NAV[page], key=f"nav_{page}", use_container_width=True, type="primary" if st.session_state.page == page else "secondary"):
            st.session_state.page = page
            st.rerun()

# -----------------------------
# HELPERS
# -----------------------------
def metric(label, value, note):
    st.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>', unsafe_allow_html=True)
