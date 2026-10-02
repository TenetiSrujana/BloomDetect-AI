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
@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("latest_bloom_risk_predictions.csv must be in the same folder as app.py")
    d = pd.read_csv(DATA_PATH)
    required = {"latitude", "longitude", "date", "chla", "risk_label"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ["latitude", "longitude", "chla"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    numeric = ["risk_probability", "model_score", "previous_chla", "historical_baseline", "recent_mean", "recent_max", "chla_anomaly", "chla_change"]
    for c in numeric:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["latitude", "longitude", "date", "chla"]).copy()
    d["risk_flag"] = d["risk_label"].astype(str).str.strip().str.lower().eq("potential bloom risk")
    d["plot_lon"] = ((d["longitude"] + 180) % 360) - 180
    return d

try:
    df = load_data()
except Exception as e:
    st.error("BloomDetect could not load the dataset.")
    st.code(str(e))
    st.stop()

latest_date = df["date"].max()
latest = df[df["date"] == latest_date].copy()
risk = latest[latest["risk_flag"]].copy()

# Project screening thresholds used when the proxy target was created.
ANOMALY_THRESHOLD = 0.059076173328496344
CHANGE_THRESHOLD = 0.02007450088858604

PAGES = ["home", "map", "checker", "hotspots", "insights", "method", "data"]
NAV = {
    "home": "Home",
    "map": "Risk Map",
    "checker": "Risk Checker",
    "hotspots": "Hotspots",
    "insights": "Insights",
    "method": "How It Works",
    "data": "Data",
}
if st.session_state.get("page") not in PAGES:
    st.session_state.page = "home"

# -----------------------------
# CSS
# -----------------------------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root{--ink:#103f49;--muted:#5f8088;--aqua:#0ca7b2;--deep:#07566a;--line:#9fcfd4;--pale:#f1fbfb;--shadow:0 14px 38px rgba(9,80,94,.09)}
html,body,[data-testid="stAppViewContainer"]{background:#f1fbfb!important;color:var(--ink)!important;font-family:'DM Sans',sans-serif!important}
.stApp{background:linear-gradient(180deg,#fbffff 0%,#effafa 50%,#fbffff 100%)!important;overflow-x:hidden}
[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer,[data-testid="stSidebar"]{display:none!important}
.block-container{max-width:1260px!important;padding:24px 34px 72px!important}
/* Subtle ocean motion, deliberately lightweight so the app stays responsive. */
.stApp:before{content:"";position:fixed;left:-10%;right:-10%;bottom:-145px;height:260px;z-index:0;pointer-events:none;background:radial-gradient(ellipse at 15% 55%,rgba(28,193,201,.13) 0 17%,transparent 18%),radial-gradient(ellipse at 55% 45%,rgba(69,221,218,.10) 0 20%,transparent 21%),radial-gradient(ellipse at 90% 60%,rgba(15,171,191,.10) 0 18%,transparent 19%);animation:waterMove 12s ease-in-out infinite alternate}
.stApp:after{content:"◦   ·     ◦      ·     ◦      ·";position:fixed;left:5%;bottom:-90px;z-index:0;pointer-events:none;color:rgba(8,153,169,.10);font-size:24px;letter-spacing:55px;animation:bubbleRise 20s linear infinite}
@keyframes waterMove{from{transform:translateX(-2%)}to{transform:translateX(2%)}}
@keyframes bubbleRise{from{transform:translateY(80px);opacity:.02}50%{opacity:.13}to{transform:translateY(-100vh);opacity:.01}}
/* Brand */
.brand-bar{position:relative;z-index:10;display:flex;align-items:center;padding:15px 20px;border:1.5px solid #c5e4e6;background:rgba(255,255,255,.86);border-radius:22px;box-shadow:var(--shadow);backdrop-filter:blur(18px)}
.brand-left{display:flex;align-items:center;gap:13px}.logo{width:48px;height:48px;border-radius:16px;background:linear-gradient(145deg,#28cdd0,#087b8e);display:grid;place-items:center;color:white;font-size:20px;font-weight:800;box-shadow:0 10px 24px rgba(9,144,157,.18)}
.brand-name{font:800 1.22rem Manrope,sans-serif;color:#103f49;letter-spacing:-.03em}.brand-sub{font-size:.73rem;color:#78959b;margin-top:2px}
/* Buttons */
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{height:46px!important;border-radius:14px!important;background:#ffffff!important;border:2px solid #164e5b!important;color:#123f49!important;font-weight:800!important;font-size:.91rem!important;box-shadow:0 6px 16px rgba(15,70,82,.09)!important;transition:all .18s ease!important}
.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{background:#e2f8f8!important;border-color:#087d8c!important;transform:translateY(-1px)!important;box-shadow:0 10px 22px rgba(15,91,101,.13)!important}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#07576a,#0b8b99)!important;border-color:#063d4b!important;color:#fff!important;box-shadow:0 9px 22px rgba(8,82,99,.22),0 0 0 3px rgba(16,174,183,.10)!important}
.stButton>button[kind="primary"]:hover{background:linear-gradient(135deg,#064b5c,#087c8b)!important;color:#fff!important}
.stButton>button:focus,.stButton>button:active{outline:none!important;box-shadow:0 0 0 3px rgba(16,174,183,.16),0 9px 22px rgba(15,91,101,.12)!important}
.stButton>button p,.stButton>button span,.stDownloadButton>button p,.stDownloadButton>button span{color:inherit!important}
.nav-wrap{position:relative;z-index:12;margin:14px 0 8px}.nav-spacer{height:2px}
/* Type */
.kicker{font:800 .70rem Manrope,sans-serif;letter-spacing:.19em;text-transform:uppercase;color:#0797a5;margin-bottom:12px}
.section{position:relative;z-index:2;padding:36px 0 18px}.section h2{font:800 clamp(2.25rem,4vw,4.0rem)/1.02 Manrope,sans-serif;letter-spacing:-.06em;color:#103f49;margin:0 0 14px}.section p{color:#5f8088;line-height:1.68;margin:0;max-width:1120px;font-size:1.04rem}
