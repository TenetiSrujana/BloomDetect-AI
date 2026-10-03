import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# BLOOMDETECT AI - CLEAN STAGE 1 DASHBOARD
# ============================================================

st.set_page_config(
    page_title="BloomDetect AI | Coastal & Ocean Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "latest_bloom_risk_predictions.csv"

# -----------------------------
# Theme
# -----------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root{
 --navy:#073B4C; --navy2:#0B5267; --aqua:#0EA5B7; --mint:#E6F8F6;
 --ink:#102A33; --muted:#5D747C; --line:#B9DDE0; --white:#FFFFFF;
 --green:#2F9E72; --green-soft:#E8F7EF; --red:#D64545; --red-soft:#FFF0F0;
 --shadow:0 9px 28px rgba(7,59,76,.07);
}
html,body,[class*="css"]{font-family:"DM Sans",sans-serif;color:var(--ink);}
.stApp{
 background:
 radial-gradient(circle at 8% 4%,rgba(94,215,215,.17),transparent 24%),
 radial-gradient(circle at 94% 10%,rgba(14,165,183,.08),transparent 26%),
 linear-gradient(180deg,#F9FEFD 0%,#EEF9F8 100%);
}
.block-container{max-width:1180px;padding-top:1.1rem!important;padding-bottom:3.8rem!important;}
#MainMenu,footer,header{visibility:hidden;}
h1,h2,h3{font-family:"Space Grotesk",sans-serif!important;color:var(--navy)!important;letter-spacing:-.02em;}
h1{font-size:2.35rem!important;} h2{font-size:1.55rem!important;} h3{font-size:1.05rem!important;}
p,label{color:var(--ink);}

.brand-bar{background:rgba(255,255,255,.95);border:1px solid var(--line);border-radius:22px;padding:16px 20px;box-shadow:var(--shadow);display:flex;align-items:center;justify-content:space-between;gap:18px;margin-bottom:16px;}
.brand-left{display:flex;align-items:center;gap:13px;}
.logo-mark{width:46px;height:46px;border-radius:14px;display:grid;place-items:center;background:linear-gradient(145deg,var(--navy),var(--aqua));color:#fff;font-size:23px;box-shadow:0 8px 18px rgba(7,59,76,.16);}
.brand-name{font-family:"Space Grotesk",sans-serif;font-size:1.18rem;font-weight:700;color:var(--navy);}
.brand-sub,.brand-date{font-size:.76rem;color:var(--muted);}
.brand-date{text-align:right;}

.nav-wrap{margin:4px 0 22px;}
div[data-testid="stHorizontalBlock"]{gap:11px!important;}
.stButton>button{width:100%;min-height:44px;border-radius:12px;border:1.6px solid var(--navy);background:rgba(255,255,255,.97);color:var(--navy);font-size:.82rem;font-weight:600;padding:9px 8px;box-shadow:0 4px 12px rgba(7,59,76,.05);transition:all .18s ease;}
.stButton>button:hover{border-color:var(--aqua);background:var(--mint);transform:translateY(-1px);}
.stButton>button:focus{box-shadow:0 0 0 3px rgba(14,165,183,.14);border-color:var(--aqua);}
.nav-active .stButton>button{background:linear-gradient(135deg,var(--navy),var(--aqua));color:#fff;border-color:var(--navy);}

.hero{position:relative;overflow:hidden;min-height:330px;border-radius:26px;border:1px solid #8FCED3;box-shadow:var(--shadow);background:linear-gradient(100deg,rgba(4,45,59,.91),rgba(7,81,99,.58)),url("https://images.unsplash.com/photo-1530053969600-caed2596d242?auto=format&fit=crop&w=1800&q=82") center/cover no-repeat;color:#fff;padding:42px;margin:8px 0 22px;}
.hero:after{content:"";position:absolute;left:-10%;right:-10%;bottom:-58px;height:125px;background:rgba(72,205,210,.20);border-radius:50%;animation:wave 5s ease-in-out infinite;}
@keyframes wave{0%,100%{transform:translateX(-2%) scaleY(1)}50%{transform:translateX(3%) scaleY(1.25)}}
.hero-content{position:relative;z-index:2;max-width:730px;}
.kicker{display:inline-flex;padding:7px 11px;border:1px solid rgba(255,255,255,.35);background:rgba(255,255,255,.12);border-radius:999px;font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;}
.hero h1{color:#fff!important;margin:18px 0 12px!important;font-size:2.55rem!important;}
.hero p{color:rgba(255,255,255,.93);font-size:1rem;line-height:1.65;margin:0;}

.section-kicker{color:var(--aqua);font-size:.71rem;text-transform:uppercase;letter-spacing:.13em;font-weight:800;margin-bottom:5px;}
.section-title{font-family:"Space Grotesk",sans-serif;color:var(--navy);font-size:1.52rem;font-weight:700;margin-bottom:6px;}
.section-copy{color:var(--muted);font-size:.89rem;line-height:1.55;margin-bottom:15px;}

.card,.metric-card,.download-card,.legend-card{background:rgba(255,255,255,.95);border:1px solid var(--line);border-radius:17px;box-shadow:0 7px 22px rgba(7,59,76,.05);}
.card{padding:19px;height:100%;}
.card-title{font-family:"Space Grotesk",sans-serif;font-weight:700;color:var(--navy);font-size:1rem;margin-bottom:6px;}
.card-text{color:var(--muted);font-size:.83rem;line-height:1.55;}
.metric-card{padding:16px 18px;min-height:118px;}
.metric-label{font-size:.71rem;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.06em;}
.metric-value{font-family:"Space Grotesk",sans-serif;font-size:1.55rem;font-weight:700;color:var(--navy);margin-top:7px;line-height:1.15;}
.metric-note{font-size:.71rem;color:var(--muted);margin-top:4px;}

.status{border-radius:16px;padding:15px 17px;border:1.5px solid;}
.status-green{background:var(--green-soft);border-color:#83CDAF;color:#1D6F4E;}
.status-red{background:var(--red-soft);border-color:#E7A1A1;color:#9C3030;}
.status-title{font-family:"Space Grotesk",sans-serif;font-size:1rem;font-weight:700;}
.status-copy{font-size:.79rem;margin-top:4px;line-height:1.5;}
.notice{border-left:4px solid var(--aqua);background:#EAF8F7;border-radius:0 13px 13px 0;padding:13px 15px;font-size:.8rem;line-height:1.55;}

.step-card{background:#fff;border:1.5px solid var(--navy);border-radius:16px;padding:16px;min-height:145px;box-shadow:0 6px 18px rgba(7,59,76,.05);}
.step-num{width:30px;height:30px;display:grid;place-items:center;border-radius:50%;background:var(--navy);color:#fff;font-weight:700;font-size:.76rem;margin-bottom:9px;}
.step-card h3{margin:0 0 6px!important;font-size:.96rem!important;}
.step-card p{margin:0;color:var(--muted);font-size:.78rem;line-height:1.48;}

.legend-card{padding:15px 17px;}
.legend-row{display:flex;gap:18px;align-items:center;flex-wrap:wrap;font-size:.78rem;color:var(--muted);}
