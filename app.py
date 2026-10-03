
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# BLOOMDETECT AI
# Stage-1 Coastal & Ocean Intelligence Platform
#
# Current dataset supports:
#   1. Ocean & Bloom-Risk Screening
#   2. Spatial Hotspot Intelligence
#   3. Location Intelligence
#   4. Chlorophyll-a / Ocean-Colour Signal Explorer
#   5. Early-Warning Screening Summary
#   6. Regional Screening
#   7. Investigation Queue
#   8. Data & Research / Downloads
#
# It does NOT claim confirmed HAB detection, species/toxin detection,
# fisheries prediction, pollution detection, or real-time monitoring.
# ============================================================

st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "latest_bloom_risk_predictions.csv"
INFOGRAPHIC_PATH = BASE_DIR / "bloomdetect_bloom_process.png"

# Project proxy thresholds used to create the stored screening label.
ANOMALY_THRESHOLD = 0.059076173328496344
CHANGE_THRESHOLD = 0.02007450088858604

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "latest_bloom_risk_predictions.csv was not found beside app.py."
        )

    d = pd.read_csv(DATA_PATH)

    required = {"latitude", "longitude", "date", "chla", "risk_label"}
    missing = required.difference(d.columns)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    d["date"] = pd.to_datetime(d["date"], errors="coerce")

    for col in ["latitude", "longitude", "chla"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    optional_numeric = [
        "risk_probability",
        "model_score",
        "previous_chla",
        "historical_baseline",
        "recent_mean",
        "recent_max",
        "chla_anomaly",
        "chla_change",
    ]

    for col in optional_numeric:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(
        subset=["latitude", "longitude", "date", "chla"]
    ).copy()

    d["plot_lon"] = ((d["longitude"] + 180) % 360) - 180
    d["risk_flag"] = (
        d["risk_label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("potential bloom risk")
    )

    return d
