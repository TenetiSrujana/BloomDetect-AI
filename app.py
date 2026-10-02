from pathlib import Path
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "latest_bloom_risk_predictions.csv"
INFOGRAPHIC_PATH = BASE_DIR / "bloomdetect_bloom_process.png"


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data(show_spinner=False)
def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "latest_bloom_risk_predictions.csv must be in the same folder as app.py"
        )

    data = pd.read_csv(DATA_PATH)

    required = {
        "latitude",
        "longitude",
        "date",
        "chla",
        "risk_label"
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    for column in [
        "latitude",
        "longitude",
        "chla"
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    optional_numeric = [
        "risk_probability",
        "model_score",
        "previous_chla",
        "historical_baseline",
        "recent_mean",
        "recent_max",
        "chla_anomaly",
        "chla_change"
    ]

    for column in optional_numeric:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    data = data.dropna(
        subset=[
            "latitude",
            "longitude",
            "date",
            "chla"
        ]
    ).copy()

    data["risk_flag"] = (
        data["risk_label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("potential bloom risk")
    )

    # Keep longitude inside standard geographic range
    data["plot_lon"] = (
        (data["longitude"] + 180) % 360
    ) - 180

    return data


# =========================================================
# LOAD DATA SAFELY
# =========================================================

try:

    df = load_data()

except Exception as error:

    st.error("BloomDetect could not load the dataset.")

    st.code(str(error))

    st.stop()


# =========================================================
# LATEST DATA
# =========================================================

latest_date = df["date"].max()

latest = df[
    df["date"] == latest_date
].copy()

risk = latest[
    latest["risk_flag"]
].copy()


# =========================================================
# PROJECT THRESHOLDS
# =========================================================

ANOMALY_THRESHOLD = 0.059076173328496344

CHANGE_THRESHOLD = 0.02007450088858604


# =========================================================
# NAVIGATION
# =========================================================

PAGES = [
    "home",
    "map",
    "checker",
    "hotspots",
    "insights",
    "method",
    "data"
]

NAV = {
    "home": "Home",
    "map": "Risk Map",
    "checker": "Risk Checker",
    "hotspots": "Hotspots",
    "insights": "Insights",
    "method": "How It Works",
    "data": "Data"
}

if st.session_state.get("page") not in PAGES:

    st.session_state.page = "home"


# =========================================================
# GLOBAL CSS
# =========================================================

st.markdown(
r"""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap'
);


/* =====================================================
   GLOBAL
   ===================================================== */

:root{

    --ink:#103f49;
    --deep:#07566a;
    --aqua:#0ca7b2;
    --aqua2:#13b7bd;
    --muted:#628189;

    --line:#9fcfd4;
    --soft:#effafa;

    --blue:#237fa4;
    --green:#19a878;
    --red:#ed4f61;

    --shadow:
        0 14px 38px rgba(9,80,94,.09);

}


html,
body,
[data-testid="stAppViewContainer"]{

    background:#effafa !important;

    color:var(--ink) !important;

    font-family:
        'DM Sans',
        sans-serif !important;

}


.stApp{

    min-height:100vh;

    background:

        linear-gradient(
            180deg,
            #fbffff 0%,
            #effafa 48%,
            #f8ffff 100%
        ) !important;

    overflow-x:hidden;

}


/* =====================================================
   HIDE STREAMLIT DEFAULT ELEMENTS
   ===================================================== */

[data-testid="stHeader"],
[data-testid="stToolbar"],
#MainMenu,
footer,
[data-testid="stSidebar"]{

    display:none !important;

}


.block-container{

    max-width:1280px !important;

    padding:
        22px 34px 70px !important;

}


/* =====================================================
   ANIMATED WATER BACKGROUND
   ===================================================== */

.stApp::before{

    content:"";

    position:fixed;

    left:-10%;
    right:-10%;
    bottom:-130px;

    height:300px;

    z-index:0;

    pointer-events:none;

    background:

        radial-gradient(
            ellipse at 10% 60%,
            rgba(24,193,201,.16) 0 15%,
            transparent 16%
        ),

        radial-gradient(
            ellipse at 45% 40%,
            rgba(69,221,218,.12) 0 18%,
            transparent 19%
        ),

        radial-gradient(
            ellipse at 80% 65%,
            rgba(15,171,191,.12) 0 16%,
            transparent 17%
        );

    animation:
        waterMove
        12s
        ease-in-out
        infinite
        alternate;

}


.stApp::after{

    content:
        "○     ·       ○        ·       ○       ·       ○";

    position:fixed;

    left:5%;

    bottom:-80px;

    z-index:0;

    pointer-events:none;

    color:rgba(8,153,169,.11);

    font-size:24px;

    letter-spacing:38px;

    animation:
        bubbles
        22s
        linear
        infinite;

}


@keyframes waterMove{

    from{
        transform:translateX(-3%);
    }

    to{
        transform:translateX(3%);
    }

}


@keyframes bubbles{

    from{

        transform:
            translateY(100px);

        opacity:.03;

    }

    45%{

        opacity:.12;

    }

    to{

        transform:
            translateY(-100vh);

        opacity:.01;

    }

}


/* =====================================================
   BRAND
   ===================================================== */

.brand-bar{

    position:relative;

    z-index:20;

    display:flex;

    align-items:center;

    justify-content:space-between;

    padding:15px 20px;

    border:
        1.5px solid
        #b9dfe2;

    background:
        rgba(255,255,255,.88);

    border-radius:22px;

    box-shadow:var(--shadow);

    backdrop-filter:
        blur(18px);

}


.brand-left{

    display:flex;

    align-items:center;

    gap:13px;

}


.logo{

    width:48px;

    height:48px;

    border-radius:16px;

    background:
        linear-gradient(
            145deg,
            #28cdd0,
            #087b8e
        );

    display:grid;

    place-items:center;

    color:white;

    font-size:23px;

    font-weight:800;

    box-shadow:
        0 10px 24px
        rgba(9,144,157,.18);

}


.brand-name{

    font:
        800 1.22rem
        Manrope,
        sans-serif;

    color:#103f49;

    letter-spacing:-.03em;

}


.brand-sub{

    font-size:.73rem;

    color:#78959b;

    margin-top:2px;

}


/* =====================================================
   BUTTONS
   ===================================================== */

.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button{

    min-height:46px !important;

    border-radius:14px !important;

    background:#ffffff !important;

    border:
        2px solid
        #164e5b !important;

    color:#123f49 !important;

    font-weight:800 !important;

    font-size:.90rem !important;

    box-shadow:
        0 6px 16px
        rgba(15,70,82,.09) !important;

    transition:
        all .18s ease !important;

}


.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover{

    background:#e2f8f8 !important;

    border-color:#087d8c !important;

    transform:
        translateY(-2px) !important;

    box-shadow:
        0 10px 22px
        rgba(15,91,101,.13) !important;

}


.stButton > button[kind="primary"]{

    background:
        linear-gradient(
            135deg,
            #07576a,
            #0b8b99
        ) !important;

    border-color:#063d4b !important;

    color:white !important;

    box-shadow:
        0 9px 22px
        rgba(8,82,99,.22),
        0 0 0 3px
        rgba(16,174,183,.10) !important;

}


.stButton > button[kind="primary"]:hover{

    background:
        linear-gradient(
            135deg,
            #064b5c,
            #087c8b
        ) !important;

    color:white !important;

}


.stButton > button:focus,
.stButton > button:active{

    outline:none !important;

    box-shadow:
        0 0 0 3px
        rgba(16,174,183,.16),
        0 9px 22px
        rgba(15,91,101,.12) !important;

}


.stButton > button p,
.stButton > button span,
.stDownloadButton > button p,
.stDownloadButton > button span{

    color:inherit !important;

}


/* Navigation */

.nav-wrap{

    position:relative;

    z-index:20;

    margin:
        14px 0 8px;

}


/* =====================================================
   TYPOGRAPHY
   ===================================================== */

.kicker{

    font:
        800 .70rem
        Manrope,
        sans-serif;

    letter-spacing:.19em;

    text-transform:uppercase;

    color:#0797a5;

    margin-bottom:12px;

}


.section{

    position:relative;

    z-index:2;

    padding:
        34px 0 18px;

}


.section h2{

    font:
        800 clamp(
            2.15rem,
            4vw,
            3.65rem
        )/1.02
        Manrope,
        sans-serif;

    letter-spacing:-.06em;

    color:#103f49;

    margin:
        0 0 14px;

}


.section p{

    color:#5f8088;

    line-height:1.62;

    margin:0;

    max-width:1120px;

    font-size:1rem;

}


/* =====================================================
   CARDS
   ===================================================== */

.card{

    position:relative;

    z-index:2;

    background:
        rgba(255,255,255,.88);

    border:
        1.5px solid
        #a9d6da;

    border-radius:23px;

    box-shadow:var(--shadow);

    padding:22px;

    backdrop-filter:
        blur(12px);

}


.card h3{

    font:
        800 1.32rem
        Manrope,
        sans-serif;

    color:#123f49;

    margin:
        0 0 9px;

}


.card p{

    color:#66848b;

    line-height:1.62;

    margin:0;

    font-size:.95rem;

}


.mini-label{

    font:
        800 .67rem
        Manrope,
        sans-serif;

    letter-spacing:.15em;

    text-transform:uppercase;

    color:#6d8c93;

    margin-bottom:8px;

}


/* =====================================================
   METRIC CARDS
   ===================================================== */

.metric{

    position:relative;

    z-index:2;

    min-height:118px;

    height:100%;

    box-sizing:border-box;

    padding:18px;

    background:
        linear-gradient(
            145deg,
            #ffffff,
            #eaf8f8
        );

    border:
        1.5px solid
        #a6d5da;

    border-radius:19px;

    box-shadow:
        0 11px 28px
        rgba(15,91,101,.08);

    display:flex;

    flex-direction:column;

    justify-content:center;

}


.metric .label{

    font:
        800 .66rem
        Manrope,
        sans-serif;

    letter-spacing:.12em;

    text-transform:uppercase;

    color:#6d8c93;

}


.metric .value{

    font:
        800 1.42rem
        Manrope,
        sans-serif;

    color:#123f49;

    margin-top:7px;

    white-space:nowrap;

}


.metric .note{

    font-size:.75rem;

    color:#78959b;

    margin-top:5px;

}


/* =====================================================
   HERO
   ===================================================== */

.hero{

    position:relative;

    z-index:2;

    overflow:hidden;

    min-height:455px;

    border-radius:31px;

    padding:
        68px 60px;

    background:

        linear-gradient(
            135deg,
            #063d51,
            #076d7e 55%,
            #13a9ad
        );

    box-shadow:
        0 28px 72px
        rgba(6,86,100,.16);

}


.hero:before{

    content:"";

    position:absolute;

    inset:-25%;

    background:

        repeating-radial-gradient(
            ellipse at 20% 115%,
            transparent 0 55px,
            rgba(181,255,251,.12)
            57px 59px,
            transparent 61px 105px
        );

    transform:rotate(-7deg);

    animation:
        waveLines
        14s
        linear
        infinite;

}


.hero:after{

    content:"";

    position:absolute;

    left:-5%;

    right:-5%;

    bottom:-130px;

    height:270px;

    background:
        rgba(142,244,237,.14);

    border-radius:50%;

    animation:
        heroWave
        8s
        ease-in-out
        infinite
        alternate;

}


.hero-content{

    position:relative;

    z-index:2;

    max-width:880px;

}


.hero .kicker{

    color:#a6fffa;

}


.hero h1{

    font:
        800 clamp(
            3.1rem,
            6vw,
            5.8rem
        )/.91
        Manrope,
        sans-serif;

    letter-spacing:-.075em;

    color:#e4ffff;

    margin:
        0 0 22px;

}


.hero p{

    font-size:1.08rem;

    line-height:1.78;

    color:#e0fbfb;

    max-width:790px;

}


.hero-badges{

    display:flex;

    gap:9px;

    flex-wrap:wrap;

    margin-top:23px;

}


.badge{

    padding:
        9px 13px;

    border-radius:999px;

    background:
        rgba(255,255,255,.14);

    border:
        1px solid
        rgba(255,255,255,.25);

    color:#efffff;

    font-size:.80rem;

    font-weight:700;

}


@keyframes waveLines{

    from{
        transform:
            translateX(-4%)
            rotate(-7deg);
    }

    to{
        transform:
            translateX(4%)
            rotate(-7deg);
    }

}


@keyframes heroWave{

    from{
        transform:
            translateX(-2%)
            rotate(-1deg);
    }

    to{
        transform:
            translateX(2%)
            rotate(1deg);
    }

}


/* =====================================================
   HOME EDUCATION
   ===================================================== */

.bloom-grid{

    position:relative;

    z-index:2;

    display:grid;

    grid-template-columns:
        1fr 1.15fr;

    gap:22px;

    align-items:stretch;

}


.bloom-panel,
.bloom-visual{

    background:
        rgba(255,255,255,.82);

    border:
        1.5px solid
        #b9dfe2;

    border-radius:23px;

    box-shadow:var(--shadow);

    padding:26px;

}


.bloom-panel{

    min-height:360px;

}


.bloom-panel h3{

    font:
        800 1.4rem/1.2
        Manrope;

    color:#123f49;

    margin:
        0 0 13px;

}


.bloom-panel p{

    font-size:.97rem;

    line-height:1.75;

    color:#66848b;

    margin:0;

}


.bloom-panel .important{

    margin-top:20px;

    padding:
        14px 16px;

    background:#e5f9f9;

    border-left:
        4px solid
        #13aab2;

    border-radius:
        0 14px 14px 0;

    color:#3f6971;

    line-height:1.62;

}


.bloom-visual{

    padding:9px;

    min-height:360px;

}


.bloom-visual img{

    display:block;

    width:100%;

    height:100%;

    min-height:340px;

    object-fit:cover;

    border-radius:17px;

}


/* =====================================================
   FEATURE CARDS
   ===================================================== */

.feature-grid{

    display:grid;

    grid-template-columns:
        repeat(4,1fr);

    gap:16px;

}


.feature{

    position:relative;

    z-index:2;

    padding:23px;

    background:
        rgba(255,255,255,.82);

    border:
        1.5px solid
        #b5dde0;

    border-radius:20px;

    box-shadow:var(--shadow);

    min-height:175px;

    transition:
        .2s;

}


.feature:hover{

    transform:
        translateY(-4px);

    border-color:
        #5fb7bf;

}


.feature .icon{

    font-size:1.5rem;

    margin-bottom:12px;

}


.feature h3{

    font:
        800 1.1rem
        Manrope;

    color:#123f49;

    margin:
        0 0 7px;

}


.feature p{

    font-size:.92rem;

    line-height:1.62;

    color:#66848b;

    margin:0;

}


/* =====================================================
   MAP
   ===================================================== */

.map-card{

    position:relative;

    z-index:2;

    background:#e7f8f8;

    border-radius:24px;

    padding:5px;

    border:
        2px solid
        #7fbfc7;

    box-shadow:
        0 16px 42px
        rgba(15,91,101,.09);

    overflow:hidden;

}


.map-study-label{

    height:47px;

    display:flex;

    align-items:center;

    gap:11px;

    padding:
        0 14px;

    color:#174b56;

}


.map-study-label span{

    font:
        800 .65rem
        Manrope;

    letter-spacing:.15em;

    color:#0a9ba8;

}


.map-study-label b{

    font-size:.94rem;

}


.map-legend{

    display:flex;

    align-items:center;

    gap:11px;

    flex-wrap:wrap;

    color:#4f747b;

    font-size:.83rem;

    padding:
        13px 14px;

}


.legend-blue{

    width:16px;

    height:10px;

    border-radius:4px;

    background:#277fa4;

}


.legend-green{

    width:16px;

    height:10px;

    border-radius:4px;

    background:#19a878;

}


.risk-dot{

    width:12px;

    height:12px;

    border-radius:50%;

    background:#ef4f5e;

    border:
        2px solid
        white;

}


/* =====================================================
   CHECKER
   ===================================================== */

.checker-grid{

    display:grid;

    grid-template-columns:
        .9fr 1.1fr;

    gap:25px;

    align-items:start;

}


.checker-form{

    padding-top:2px;

}


.coord-label{

    font:
        800 .80rem
        Manrope;

    color:#285761;

    margin:
        0 0 6px !important;

}


.entered-line{

    margin-top:15px;

    padding:
        12px 14px;

    border-radius:14px;

    background:#e7f8f8;

    border:
        1px solid
        #c9e9eb;

    color:#4f747b;

    font-size:.88rem;

    line-height:1.5;

}


.nearest-line{

    color:#66848b;

    font-size:.88rem;

    line-height:1.5;

}


.result-grid{

    display:grid;

    grid-template-columns:
        1fr 1fr;

    gap:11px;

    margin-top:15px;

}


.result-grid div{

    padding:
        13px 14px;

    background:
        linear-gradient(
            145deg,
            #f5fdfd,
            #e9f8f8
        );

    border:
        1px solid
        #c7e5e7;

    border-radius:14px;

    min-height:59px;

}


.result-grid span{

    display:block;

    font-size:.72rem;

    color:#78959b;

    margin-bottom:5px;

}


.result-grid b{

    font-size:.94rem;

    color:#194b55;

}


.status-box{

    margin-top:15px;

    padding:
        15px 16px;

    border-radius:15px;

    border:2px solid;

    line-height:1.55;

}


.status-box.yes{

    background:#fff0f2;

    border-color:#f06a78;

    color:#a62d3c;

}


.status-box.no{

    background:#eafaf4;

    border-color:#49b995;

    color:#176f58;

}


.status-box strong{

    font:
        800 .92rem
        Manrope;

}


.signal-grid{

    display:grid;

    grid-template-columns:
        repeat(4,1fr);

    gap:10px;

    margin-top:15px;

}


.signal{

    padding:
        12px 13px;

    border-radius:14px;

    background:#f7fcfc;

    border:
        1px solid
        #c9e5e7;

}


.signal .slabel{

    font-size:.70rem;

    color:#78959b;

}


.signal .sval{

    font:
        800 .94rem
        Manrope;

    color:#164a55;

    margin-top:4px;

}


.note{

    position:relative;

    z-index:2;

    margin-top:17px;

    padding:
        13px 16px;

    background:#e6f8f8;

    border-left:
        4px solid
        #11a9b2;

    border-radius:
        0 14px 14px 0;

    color:#52757c;

    font-size:.88rem;

    line-height:1.6;

}


/* =====================================================
   HOTSPOT CARDS
   ===================================================== */

.hotspot-grid{

    display:grid;

    grid-template-columns:
        repeat(3,1fr);

    gap:15px;

}


.hotspot-card{

    background:
        rgba(255,255,255,.88);

    border:
        2px solid
        #a9d6da;

    border-radius:20px;

    padding:20px;

    box-shadow:var(--shadow);

}


.hotspot-card:hover{

    border-color:
        #5bb5be;

}


.hotspot-number{

    width:38px;

    height:38px;

    display:grid;

    place-items:center;

    border-radius:12px;

    background:#e9f8f8;

    border:
        2px solid
        #164e5b;

    color:#07566a;

    font-weight:800;

    margin-bottom:12px;

}


.hotspot-card h3{

    font:
        800 1.05rem
        Manrope;

    color:#123f49;

    margin:
        0 0 6px;

}


.hotspot-card p{

    color:#66848b;

    font-size:.88rem;

    line-height:1.5;

    margin:0;

}


/* =====================================================
   METHOD STEPS
   ===================================================== */

.method-grid{

    display:grid;

    grid-template-columns:
        repeat(2,1fr);

    gap:14px;

}


.step{

    position:relative;

    z-index:2;

    min-height:130px;

    display:grid;

    grid-template-columns:
        52px 1fr;

    gap:15px;

    align-items:start;

    padding:
        19px;

    border:
        2px solid
        #174e5b;

    border-radius:18px;

    background:
        linear-gradient(
            100deg,
            rgba(255,255,255,.97),
            rgba(232,249,249,.90)
        );

    box-shadow:
        0 9px 23px
        rgba(15,91,101,.07);

}


.step-num{

    width:43px;

    height:43px;

    border-radius:12px;

    background:#0d5264;

    color:#fff;

    display:grid;

    place-items:center;

    font:
        800 .85rem
        Manrope;

    border:
        2px solid
        #082f3b;

}


.step h3{

    font:
        800 1.12rem
        Manrope;

    color:#123f49;

    margin:
        2px 0 6px;

}


.step p{

    color:#66848b;

    line-height:1.5;

    margin:0;

    font-size:.92rem;

}


/* =====================================================
   DOWNLOADS
   ===================================================== */

.download-panel{

    min-height:128px;

    box-sizing:border-box;

    padding:21px;

    border-radius:19px;

    background:
        linear-gradient(
            135deg,
            #087b8b,
            #13adb3
        );

    border:
        2px solid
        #07576a;

    box-shadow:
        0 13px 28px
        rgba(8,91,102,.16);

    color:white;

}


.download-panel h3{

    font:
        800 1.2rem
        Manrope;

    color:white;

    margin:
        0 0 7px;

}


.download-panel p{

    font-size:.88rem;

    color:#e5ffff;

    line-height:1.45;

    margin:0;

}


/* =====================================================
   CHARTS
   ===================================================== */

.chart-shell{

    background:
        rgba(255,255,255,.90);

    border:
        1.5px solid
        #a9d6da;

    border-radius:20px;

    padding:
        16px 17px 5px;

    box-shadow:var(--shadow);

}


.chart-shell h3{

    font:
        800 1.16rem
        Manrope;

    color:#123f49;

    margin:
        0 0 5px;

}


.chart-shell p{

    font-size:.87rem;

    color:#6b878d;

    margin:
        0 0 2px;

}


/* =====================================================
   DATA TABLE
   ===================================================== */

.table-wrap{

    position:relative;

    z-index:2;

    border-radius:18px;

    overflow:hidden;

    border:
        1px solid
        #c5e3e5;

    box-shadow:var(--shadow);

    background:#fff;

}


.table-wrap table{

    width:100%;

    border-collapse:collapse;

    font-size:.88rem;

}


.table-wrap th{

    background:#0e5663;

    color:#fff;

    text-align:left;

    padding:
        12px 14px;

}


.table-wrap td{

    padding:
        11px 14px;

    border-top:
        1px solid
        #e3eeee;

    color:#315b63;

    background:#fff;

}


.table-wrap tr:nth-child(even) td{

    background:#f7fcfc;

}


/* =====================================================
   FOOTER
   ===================================================== */

.footer{

    position:relative;

    z-index:2;

    border-top:
        1px solid
        #d5ebed;

    margin-top:44px;

    padding-top:18px;

    color:#76959b;

    font-size:.72rem;

}


/* =====================================================
   RESPONSIVE
   ===================================================== */

@media(max-width:950px){

    .block-container{

        padding:
            18px 18px 55px !important;

    }

    .feature-grid{

        grid-template-columns:
            1fr 1fr;

    }

    .bloom-grid,
    .checker-grid{

        grid-template-columns:
            1fr;

    }

    .signal-grid{

        grid-template-columns:
            1fr 1fr;

    }

    .method-grid{

        grid-template-columns:
            1fr;

    }

    .hotspot-grid{

        grid-template-columns:
            1fr 1fr;

    }

    .hero{

        padding:
            48px 35px;

    }

}


@media(max-width:650px){

    .feature-grid,
    .hotspot-grid{

        grid-template-columns:
            1fr;

    }

    .signal-grid{

        grid-template-columns:
            1fr;

    }

    .hero h1{

        font-size:3rem;

    }

    .block-container{

        padding:
            15px 12px 45px !important;

    }

}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
"""
<div class="brand-bar">

    <div class="brand-left">

        <div class="logo">≈</div>

        <div>

            <div class="brand-name">
                BloomDetect AI
            </div>

            <div class="brand-sub">
                Satellite-based potential bloom-risk screening
            </div>

        </div>

    </div>

</div>
""",
unsafe_allow_html=True
)


# =========================================================
# NAVIGATION
# =========================================================

st.markdown(
'<div class="nav-wrap"></div>',
unsafe_allow_html=True
)

nav_cols = st.columns(
    len(PAGES),
    gap="small"
)

for col, page in zip(
    nav_cols,
    PAGES
):

    with col:

        if st.button(
            NAV[page],
            key=f"nav_{page}",
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.page == page
                else "secondary"
            )
        ):

            st.session_state.page = page

            st.rerun()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def metric(
    label,
    value,
    note
):

    st.markdown(
        f"""
        <div class="metric">

            <div class="label">
                {label}
            </div>

            <div class="value">
                {value}
            </div>

            <div class="note">
                {note}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def section(
    kicker,
    title,
    copy
):

    st.markdown(
        f"""
        <div class="section">

            <div class="kicker">
                {kicker}
            </div>

            <h2>
                {title}
            </h2>

            <p>
                {copy}
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


def safe(
    value,
    digits=3
):

    if pd.isna(value):

        return "Unavailable"

    number = float(value)

    if abs(number) < 0.0005:

        return "0.0"

    return f"{number:.{digits}f}"


def probability_text(value):

    if pd.isna(value):

        return "Unavailable"

    number = float(value)

    if number <= 1:

        number *= 100

    return f"{number:.1f}%"


def nearest_row(
    latitude,
    longitude
):

    dlat = (
        latest["latitude"].to_numpy()
        - latitude
    ) ** 2

    dlon = (
        np.abs(
            latest["longitude"].to_numpy()
            - longitude
        )
        .clip(0, 180)
    ) ** 2

    index = int(
        np.argmin(
            dlat + dlon
        )
    )

    return latest.iloc[index]


def study_area(data):

    return data[
        data["latitude"].between(
            -40,
            30
        )
        &
        data["plot_lon"].between(
            20,
            120
        )
    ].copy()


def region_name(
    latitude,
    longitude
):

    if (
        5 <= latitude <= 30
        and
        45 <= longitude <= 75
    ):

        return "Arabian Sea"

    if (
        0 <= latitude <= 25
        and
        75 < longitude <= 100
    ):

        return "Bay of Bengal"

    if (
        -30 <= latitude < 5
        and
        40 <= longitude <= 100
    ):

        return "Southern Indian Ocean"

    if (
        5 <= latitude <= 30
        and
        75 < longitude <= 120
    ):

        return "Northern Indian Ocean"

    return "Other study area"


def image_data_uri(path):

    if not path.exists():

        return ""

    mime = (
        "image/png"
        if path.suffix.lower() == ".png"
        else "image/jpeg"
    )

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode()

    return (
        f"data:{mime};base64,{encoded}"
    )


# =========================================================
# MAP FUNCTION
# =========================================================

def map_figure(
    show_risk=True
):

    data = study_area(
        latest
    )

    if data.empty:

        data = latest.copy()

    # -----------------------------------------------------
    # BLUE BACKGROUND FIELD
    # -----------------------------------------------------

    normal = data[
        ~data["risk_flag"]
    ].copy()

    normal["lat_bin"] = (
        np.floor(
            normal["latitude"]
        ) + .5
    ).round(2)

    normal["lon_bin"] = (
        np.floor(
            normal["plot_lon"]
        ) + .5
    ).round(2)

    blue = (
        normal
        .groupby(
            [
                "lat_bin",
                "lon_bin"
            ],
            as_index=False
        )
        .agg(
            chla=("chla", "mean")
        )
    )

    # -----------------------------------------------------
    # BLUE FIELD
    # -----------------------------------------------------

    fig = px.scatter_geo(
        blue,
        lat="lat_bin",
        lon="lon_bin",
        color="chla",
        custom_data=[
            "lat_bin",
            "lon_bin",
            "chla"
        ],
        projection="equirectangular",
        color_continuous_scale=[
            [0, "#165d8b"],
            [.50, "#218db0"],
            [1, "#55c8c7"]
        ],
        range_color=(
            0,
            max(
                float(
                    data["chla"].quantile(
                        .995
                    )
                ),
                .01
            )
        )
    )

    fig.update_traces(
        marker=dict(
            size=7.5,
            opacity=.78,
            line=dict(
                width=0
            )
        ),
        hovertemplate=
        "Latitude %{customdata[0]:.1f}°"
        "<br>Longitude %{customdata[1]:.1f}°"
        "<br>Mean Chl-a %{customdata[2]:.4f}"
        "<extra></extra>"
    )

    # -----------------------------------------------------
    # RISK CELLS
    # -----------------------------------------------------

    risk_data = data[
        data["risk_flag"]
    ].copy()

    if (
        show_risk
        and
        not risk_data.empty
    ):

        # Green screening area
        green = (
            risk_data
            .groupby(
                [
                    "latitude",
                    "plot_lon"
                ],
                as_index=False
            )
            .agg(
                chla=("chla", "mean")
            )
        )

        green_fig = px.scatter_geo(
            green,
            lat="latitude",
            lon="plot_lon"
        )

        fig.add_trace(
            green_fig.data[0]
        )

        fig.data[-1].marker = dict(
            size=10,
            color="#19a878",
            opacity=.68,
            line=dict(
                width=1,
                color="#ffffff"
            )
        )

        fig.data[-1].name = (
            "Potential-risk area"
        )

        fig.data[-1].hovertemplate = (
            "Potential-risk screening area"
            "<br>Latitude %{lat:.2f}°"
            "<br>Longitude %{lon:.2f}°"
            "<extra></extra>"
        )

        # Red points on top for clear flags
        red_fig = px.scatter_geo(
            risk_data,
            lat="latitude",
            lon="plot_lon",
            custom_data=[
                "latitude",
                "longitude",
                "chla"
            ]
        )

        fig.add_trace(
            red_fig.data[0]
        )

        fig.data[-1].marker = dict(
            size=7.5,
            color="#ef4f5e",
            opacity=.95,
            line=dict(
                width=1.4,
                color="#ffffff"
            )
        )

        fig.data[-1].name = (
            "Potential bloom-risk flag"
        )

        fig.data[-1].hovertemplate = (
            "Potential bloom-risk flag"
            "<br>Latitude %{customdata[0]:.2f}°"
            "<br>Longitude %{customdata[1]:.2f}°"
            "<br>Chl-a %{customdata[2]:.4f}"
            "<extra></extra>"
        )

    # -----------------------------------------------------
    # MAP STYLE
    # -----------------------------------------------------

    fig.update_geos(

        showland=True,

        landcolor="#d9e9e7",

        showocean=True,

        oceancolor="#d9f5f6",

        showcoastlines=True,

        coastlinecolor="#398995",

        coastlinewidth=1.1,

        showcountries=True,

        countrycolor="#8eafb4",

        bgcolor="#d9f5f6",

        lataxis_range=[
            -40,
            30
        ],

        lonaxis_range=[
            20,
            120
        ],

        projection_scale=1.05,

        center=dict(
            lat=-5,
            lon=70
        )

    )

    fig.update_layout(

        height=610,

        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0
        ),

        paper_bgcolor="#d9f5f6",

        plot_bgcolor="#d9f5f6",

        font=dict(
            color="#174b56"
        ),

        showlegend=False,

        coloraxis_colorbar=dict(

            title=dict(
                text="Chl-a",
                font=dict(
                    color="#174b56"
                )
            ),

            tickfont=dict(
                color="#174b56"
            ),

            bgcolor=
                "rgba(255,255,255,.90)",

            outlinewidth=0,

            thickness=14,

            len=.55

        )

    )

    return fig


# =========================================================
# HOME
# =========================================================

if st.session_state.page == "home":

    st.markdown(
    """
    <div class="hero">

        <div class="hero-content">

            <div class="kicker">
                EOS-06 · OCM-3 · BLOOMDETECT AI
            </div>

            <h1>
                See the ocean<br>
                more clearly.
            </h1>

            <p>
                BloomDetect AI turns satellite-derived
                chlorophyll-a observations into an
                interactive screening system for finding
                ocean locations whose recent patterns
                may deserve a closer look.
            </p>

            <div class="hero-badges">

                <span class="badge">
                    🌊 Spatial screening
                </span>

                <span class="badge">
                    🛰️ Satellite observations
                </span>

                <span class="badge">
                    🔥 Hotspot detection
                </span>

                <span class="badge">
                    📍 Location investigation
                </span>

            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # WHAT IS A BLOOM
    # -----------------------------------------------------

    section(
        "01 · Understand the signal",
        "What is an algal bloom?",
        "An algal bloom occurs when algae or phytoplankton become unusually concentrated in part of a water body. Some blooms are harmless, while some can have ecological or health effects. Chlorophyll-a can show where conditions deserve closer investigation, but it does not by itself prove a harmful bloom or identify toxins."
    )

    uri = image_data_uri(
        INFOGRAPHIC_PATH
    )

    if uri:

        st.markdown(
        f"""
        <div class="bloom-grid">

            <div class="bloom-panel">

                <div class="mini-label">
                    WHY THIS MATTERS
                </div>

                <h3>
                    Small organisms can create
                    a large environmental signal.
                </h3>

                <p>
                    Phytoplankton use light and nutrients
                    to grow. When many cells accumulate,
                    ocean colour and chlorophyll-a can
                    change. BloomDetect uses that observable
                    signal to screen locations, then points
                    people toward places worth investigating
                    with additional evidence.
                </p>

                <div class="important">

                    <b>Important:</b>

                    This is an early-warning support tool,
                    not a species, toxin, or laboratory
                    confirmation system.

                </div>

            </div>

            <div class="bloom-visual">

                <img
                    src="{uri}"
                    alt="Algal bloom process"
                >

            </div>

        </div>
        """,
        unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    section(
        "02 · Explore",
        "A different job on every page.",
        "The dashboard is deliberately task-based: map the field, find concentrations, inspect a coordinate, understand the latest pattern and download the evidence."
    )

    feature_columns = st.columns(
        4,
        gap="medium"
    )

    features = [

        (
            "🌍",
            "Risk Map",
            "See the latest chlorophyll-a field across the Indian Ocean study area and the potential-risk flags."
        ),

        (
            "🔥",
            "Hotspots",
            "Find spatial concentrations of nearby potential-risk cells."
        ),

        (
            "📍",
            "Risk Checker",
            "Enter a coordinate and inspect the nearest processed satellite cell."
        ),

        (
            "📊",
            "Insights",
            "Understand the latest field using compact statistics and charts."
        )

    ]

    for col, feature in zip(
        feature_columns,
        features
    ):

        icon, title, text = feature

        with col:

            st.markdown(
            f"""
            <div class="feature">

                <div class="icon">
                    {icon}
                </div>

                <h3>
                    {title}
                </h3>

                <p>
                    {text}
                </p>

            </div>
            """,
            unsafe_allow_html=True
            )

    # -----------------------------------------------------
    # SCIENTIFIC BOUNDARY
    # -----------------------------------------------------

    section(
        "03 · Scientific boundary",
        "What the flag means.",
        "The model screens a proxy pattern derived from chlorophyll-a behaviour. A potential-risk flag is a reason for additional investigation, not proof of a harmful algal bloom, harmful species or toxin."
    )


# =========================================================
# RISK MAP
# =========================================================

elif st.session_state.page == "map":

    section(
        "01 · Spatial screening",
        "See where the signal changes.",
        "Blue shows the latest chlorophyll-a field across the project study area. Green highlights locations included in the potential-risk screening, while red points mark individual potential-risk flags."
    )

    left, right = st.columns(
        [1.4, .7],
        gap="medium"
    )

    with left:

        show_risk = st.toggle(
            "Show potential-risk overlay",
            value=True,
            key="map_risk_toggle"
        )

    with right:

        metric(
            "Visible cells",
            f"{len(study_area(latest)):,}",
            "latest observation"
        )

    st.markdown(
    """
    <div class="map-card">

        <div class="map-study-label">

            <span>
                STUDY AREA
            </span>

            <b>
                Indian Ocean · Arabian Sea · Bay of Bengal
            </b>

        </div>
    """,
    unsafe_allow_html=True
    )

    st.plotly_chart(
        map_figure(
            show_risk
        ),
        use_container_width=True,
        config={
            "scrollZoom": False,
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "lasso2d",
                "select2d"
            ]
        }
    )

    st.markdown(
    """
        <div class="map-legend">

            <span>
                Normal field
            </span>

            <span class="legend-blue"></span>

            <span>
                Blue = latest chlorophyll-a field
            </span>

            <span class="legend-green"></span>

            <span>
                Green = potential-risk screening area
            </span>

            <span class="risk-dot"></span>

            <span>
                Red = potential bloom-risk flag
            </span>

        </div>

    </div>
    """,
    unsafe_allow_html=True
    )

    st.markdown(
    """
    <div class="note">

        <b>How to use:</b>

        Find a red concentration, open
        <b>Hotspots</b> to inspect the cluster,
        then use <b>Risk Checker</b> to investigate
        individual coordinates.

    </div>
    """,
    unsafe_allow_html=True
    )


# =========================================================
# RISK CHECKER
# =========================================================

elif st.session_state.page == "checker":

    section(
        "02 · Coordinate screening",
        "Check a location.",
        "Enter the coordinate you want to investigate. BloomDetect keeps your entered coordinate separate from the nearest processed satellite grid cell."
    )

    # -----------------------------------------------------
    # DEFAULTS
    # -----------------------------------------------------

    if "input_lat" not in st.session_state:

        st.session_state.input_lat = 18.00

    if "input_lon" not in st.session_state:

        st.session_state.input_lon = 78.00

    if "checked_lat" not in st.session_state:

        st.session_state.checked_lat = 18.00

    if "checked_lon" not in st.session_state:

        st.session_state.checked_lon = 78.00

    # -----------------------------------------------------
    # FORM
    # -----------------------------------------------------

    with st.form(
        "checker_form",
        clear_on_submit=False
    ):

        c1, c2 = st.columns(
            2,
            gap="large"
        )

        with c1:

            st.markdown(
                '<div class="coord-label">Latitude</div>',
                unsafe_allow_html=True
            )

            latitude = st.number_input(
                "lat",
                min_value=-90.0,
                max_value=90.0,
                step=.25,
                format="%.2f",
                key="input_lat",
                label_visibility="collapsed"
            )

        with c2:

            st.markdown(
                '<div class="coord-label">Longitude</div>',
                unsafe_allow_html=True
            )

            longitude = st.number_input(
                "lon",
                min_value=-180.0,
                max_value=180.0,
                step=.25,
                format="%.2f",
                key="input_lon",
                label_visibility="collapsed"
            )

        submitted = st.form_submit_button(
            "🔎 Check this coordinate",
            use_container_width=True,
            type="primary"
        )

    if submitted:

        st.session_state.checked_lat = float(
            latitude
        )

        st.session_state.checked_lon = float(
            longitude
        )

    latitude = float(
        st.session_state.checked_lat
    )

    longitude = float(
        st.session_state.checked_lon
    )

    row = nearest_row(
        latitude,
        longitude
    )

    risk_yes = bool(
        row["risk_flag"]
    )

    probability = probability_text(
        row.get(
            "risk_probability",
            np.nan
        )
    )

    anomaly = row.get(
        "chla_anomaly",
        np.nan
    )

    change = row.get(
        "chla_change",
        np.nan
    )

    # -----------------------------------------------------
    # INPUT + RESULT
    # -----------------------------------------------------

    st.markdown(
        '<div class="checker-grid">',
        unsafe_allow_html=True
    )

    # LEFT
    st.markdown(
    f"""
    <div class="checker-form">

        <div class="mini-label">
            YOUR INPUT
        </div>

        <div class="card">

            <h3 style="font-size:1.55rem">
                {latitude:.2f}° · {longitude:.2f}°
            </h3>

            <p>
                This is the exact coordinate you entered.
                It is kept separate from the satellite
                grid location used for screening.
            </p>

        </div>

    </div>
    """,
    unsafe_allow_html=True
    )

    # RIGHT
    status_class = (
        "yes"
        if risk_yes
        else
        "no"
    )

    status_text = (
        "POTENTIAL BLOOM-RISK FLAG"
        if risk_yes
        else
        "NO POTENTIAL-RISK FLAG"
    )

    status_icon = (
        "🔴"
        if risk_yes
        else
        "🟢"
    )

    status_description = (
        "This processed cell is included in the current potential-risk screening shortlist."
        if risk_yes
        else
        "This processed cell is not included in the current potential-risk screening shortlist."
    )

    st.markdown(
    f"""
    <div>

        <div class="card">

            <div class="mini-label">
                NEAREST PROCESSED CELL
            </div>

            <h3 style="font-size:1.75rem">

                {row.latitude:.2f}°
                ·
                {row.longitude:.2f}°

            </h3>

            <div class="nearest-line">

                Your input:
                <b>
                    {latitude:.2f}°
                    ·
                    {longitude:.2f}°
                </b>

            </div>

            <div class="nearest-line">

                Nearest processed cell:
                <b>
                    {row.latitude:.2f}°
                    ·
                    {row.longitude:.2f}°
                </b>

            </div>

            <div class="result-grid">

                <div>

                    <span>
                        Date
                    </span>

                    <b>
                        {row.date.strftime("%d %B %Y")}
                    </b>

                </div>

                <div>

                    <span>
                        Chlorophyll-a
                    </span>

                    <b>
                        {safe(row.chla)}
                    </b>

                </div>

                <div>

                    <span>
                        Model probability
                    </span>

                    <b>
                        {probability}
                    </b>

                </div>

                <div>

                    <span>
                        Region
                    </span>

                    <b>
                        {region_name(
                            row.latitude,
                            row.longitude
                        )}
                    </b>

                </div>

            </div>

            <div class="status-box {status_class}">

                <strong>
                    {status_icon}
                    {status_text}
                </strong>

                <br>

                <span>
                    {status_description}
                </span>

            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # SIGNALS
    # -----------------------------------------------------

    st.markdown(
    """
    <div
        class="mini-label"
        style="margin-top:22px"
    >
        SIGNALS BEHIND THE SCREENING
    </div>
    """,
    unsafe_allow_html=True
    )

    signals = [

        (
            "Current Chl-a",
            safe(row.chla)
        ),

        (
            "Historical baseline",
            safe(
                row.get(
                    "historical_baseline",
                    np.nan
                )
            )
        ),

        (
            "Anomaly",
            safe(anomaly)
        ),

        (
            "Recent change",
            safe(change)
        )

    ]

    signal_columns = st.columns(
        4,
        gap="small"
    )

    for col, item in zip(
        signal_columns,
        signals
    ):

        label, value = item

        with col:

            st.markdown(
            f"""
            <div class="signal">

                <div class="slabel">
                    {label}
                </div>

                <div class="sval">
                    {value}
                </div>

            </div>
            """,
            unsafe_allow_html=True
            )

    # -----------------------------------------------------
    # EXPLANATION
    # -----------------------------------------------------

    if risk_yes:

        explanation = (
            "The latest cell crosses the two proxy screening "
            "conditions used to create the potential-risk label: "
            "chlorophyll anomaly and recent change are both at or "
            "above their project thresholds."
        )

    else:

        checks = []

        if pd.notna(anomaly):

            checks.append(
                "anomaly is above"
                if anomaly >= ANOMALY_THRESHOLD
                else
                "anomaly is below"
            )

        if pd.notna(change):

            checks.append(
                "recent change is above"
                if change >= CHANGE_THRESHOLD
                else
                "recent change is below"
            )

        if checks:

            explanation = (
                "This cell is not flagged by the current "
                "proxy rule. "
                + " and ".join(checks)
                + " the project threshold."
            )

        else:

            explanation = (
                "Supporting historical signals "
                "are unavailable."
            )

    st.markdown(
    f"""
    <div class="note">

        <b>
            Why this result?
        </b>

        {explanation}

        This is a screening explanation,
        not proof of a harmful bloom.

    </div>
    """,
    unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # DOWNLOAD LOCATION REPORT
    # -----------------------------------------------------

    report = pd.DataFrame(
        [
            {
                "input_latitude":
                    latitude,

                "input_longitude":
                    longitude,

                "nearest_latitude":
                    row.latitude,

                "nearest_longitude":
                    row.longitude,

                "date":
                    row.date.strftime(
                        "%Y-%m-%d"
                    ),

                "chla":
                    row.chla,

                "historical_baseline":
                    row.get(
                        "historical_baseline",
                        np.nan
                    ),

                "chla_anomaly":
                    anomaly,

                "chla_change":
                    change,

                "risk_label":
                    row.risk_label,

                "risk_probability":
                    row.get(
                        "risk_probability",
                        np.nan
                    )

            }
        ]
    )

    st.download_button(
        "⬇ Download this location report",
        report.to_csv(
            index=False
        ).encode(),
        "bloomdetect_location_report.csv",
        "text/csv",
        use_container_width=True
    )


# =========================================================
# HOTSPOTS
# =========================================================

elif st.session_state.page == "hotspots":

    section(
        "03 · Spatial concentration",
        "Find the areas worth investigating.",
        "Nearby flagged cells are grouped into 2° × 2° zones. This turns individual screening flags into a smaller investigation list without treating the zones as confirmed HAB severity."
    )

    study = study_area(
        latest
    )

    risk_study = study[
        study["risk_flag"]
    ].copy()

    if risk_study.empty:

        st.markdown(
        """
        <div class="card">

            <h3>
                No potential-risk cells
                in the study area.
            </h3>

            <p>
                The latest processed field has no
                screening flags in the current
                study view.
            </p>

        </div>
        """,
        unsafe_allow_html=True
        )

    else:

        risk_study["lat_zone"] = (
            np.floor(
                risk_study["latitude"] / 2
            ) * 2
        ).round(2)

        risk_study["lon_zone"] = (
            np.floor(
                risk_study["plot_lon"] / 2
            ) * 2
        ).round(2)

        zones = (

            risk_study
            .groupby(
                [
                    "lat_zone",
                    "lon_zone"
                ],
                as_index=False
            )
            .agg(
                flagged_cells=(
                    "risk_flag",
                    "size"
                ),

                mean_chla=(
                    "chla",
                    "mean"
                ),

                max_chla=(
                    "chla",
                    "max"
                )
            )

            .sort_values(
                [
                    "flagged_cells",
                    "mean_chla"
                ],
                ascending=False
            )

            .head(15)

        )

        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        metric_columns = st.columns(
            3,
            gap="medium"
        )

        with metric_columns[0]:

            metric(
                "Flagged cells",
                f"{len(risk_study):,}",
                "current study area"
            )

        with metric_columns[1]:

            metric(
                "Hotspot zones",
                f"{len(zones):,}",
                "2° × 2° grouping"
            )

        with metric_columns[2]:

            metric(
                "Largest cluster",
                f"{int(zones.iloc[0].flagged_cells):,}",
                "flagged cells"
            )

        # -------------------------------------------------
        # HOTSPOT MAP
        # -------------------------------------------------

        st.markdown(
        """
        <div class="section"
             style="padding-top:30px">

            <div class="kicker">
                01 · PRIORITY AREAS
            </div>

            <h2>
                Where the flags concentrate.
            </h2>

            <p>
                Larger markers mean more flagged cells
                in that 2° × 2° zone. They represent
                investigation concentrations, not
                confirmed severity rankings.
            </p>

        </div>
        """,
        unsafe_allow_html=True
        )

        hotspot_fig = px.scatter_geo(
            zones,
            lat="lat_zone",
            lon="lon_zone",
            size="flagged_cells",
            color="flagged_cells",
            color_continuous_scale=[
                [0, "#bceee0"],
                [.45, "#39bd91"],
                [1, "#e94e60"]
            ],
            projection="equirectangular",
            hover_data={
                "lat_zone": ":.2f",
                "lon_zone": ":.2f",
                "flagged_cells": True,
                "mean_chla": ":.4f",
                "max_chla": ":.4f"
            }
        )

        hotspot_fig.update_geos(

            showland=True,

            landcolor="#dceae8",

            showocean=True,

            oceancolor="#dff7f8",

            showcoastlines=True,

            coastlinecolor="#4f9aa4",

            showcountries=True,

            countrycolor="#9abdc2",

            bgcolor="#dff7f8",

            lataxis_range=[
                -40,
                30
            ],

            lonaxis_range=[
                20,
                120
            ],

            center=dict(
                lat=-5,
                lon=70
            ),

            projection_scale=1.08

        )

        hotspot_fig.update_layout(

            height=520,

            margin=dict(
                l=0,
                r=0,
                t=0,
                b=0
            ),

            paper_bgcolor="#dff7f8",

            plot_bgcolor="#dff7f8",

            font=dict(
                color="#174b56"
            ),

            coloraxis_colorbar=dict(
                title="Flagged cells",
                tickfont=dict(
                    color="#174b56"
                ),
                bgcolor=
                    "rgba(255,255,255,.88)",
                thickness=13,
                len=.52
            )

        )

        st.plotly_chart(
            hotspot_fig,
            use_container_width=True,
            config={
                "scrollZoom": False,
                "displaylogo": False
            }
        )

        # -------------------------------------------------
        # HOTSPOT CARDS
        # -------------------------------------------------

        st.markdown(
        """
        <div class="mini-label"
             style="margin-top:20px">
            TOP INVESTIGATION AREAS
        </div>
        """,
        unsafe_allow_html=True
        )

        card_columns = st.columns(
            3,
            gap="medium"
        )

        top_zones = zones.head(6)

        for index, (
            col,
            (_, zone)
        ) in enumerate(
            zip(
                card_columns * 2,
                top_zones.iterrows()
            ),
            start=1
        ):

            with col:

                st.markdown(
                f"""
                <div class="hotspot-card">

                    <div class="hotspot-number">
                        {index}
                    </div>

                    <h3>
                        {zone.lat_zone:.2f}°
                        ·
                        {zone.lon_zone:.2f}°
                    </h3>

                    <p>
                        <b>
                            {int(zone.flagged_cells)}
                        </b>
                        flagged cells
                    </p>

                    <p>
                        Mean Chl-a:
                        <b>
                            {zone.mean_chla:.4f}
                        </b>
                    </p>

                    <p>
                        Max Chl-a:
                        <b>
                            {zone.max_chla:.4f}
                        </b>
                    </p>

                </div>
                """,
                unsafe_allow_html=True
                )

        # -------------------------------------------------
        # TABLE
        # -------------------------------------------------

        show = zones.copy()

        show["Zone"] = show.apply(
            lambda row:
                f"{row.lat_zone:.2f}° , "
                f"{row.lon_zone:.2f}°",
            axis=1
        )

        show["Mean Chl-a"] = (
            show["mean_chla"]
            .round(4)
        )

        show["Max Chl-a"] = (
            show["max_chla"]
            .round(4)
        )

        show = show[
            [
                "Zone",
                "flagged_cells",
                "Mean Chl-a",
                "Max Chl-a"
            ]
        ].rename(
            columns={
                "flagged_cells":
                    "Flagged cells"
            }
        )

        st.markdown(
        """
        <div class="mini-label"
             style="margin-top:25px">
            INVESTIGATION LIST
        </div>
        """,
        unsafe_allow_html=True
        )

        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
        """
        <div class="note">

            <b>
                Recommended workflow:
            </b>

            Use a hotspot to identify a spatial
            concentration, inspect representative
            coordinates in Risk Checker, and combine
            the satellite signal with field or
            environmental evidence.

        </div>
        """,
        unsafe_allow_html=True
        )


# =========================================================
# INSIGHTS
# =========================================================

elif st.session_state.page == "insights":

    section(
        "04 · Latest-field insights",
        "What stands out right now?",
        "A compact operational view showing how many cells are flagged, where the flags concentrate, and which coordinates may deserve further investigation."
    )

    normal_count = int(
        (~latest["risk_flag"]).sum()
    )

    risk_count = int(
        latest["risk_flag"].sum()
    )

    risk_share = (
        100 * risk_count / len(latest)
        if len(latest)
        else 0
    )

    # -----------------------------------------------------
    # TOP METRICS
    # -----------------------------------------------------

    metrics = [

        (
            "Observation cells",
            f"{len(latest):,}",
            "latest field"
        ),

        (
            "Potential-risk cells",
            f"{risk_count:,}",
            "model screening"
        ),

        (
            "Normal cells",
            f"{normal_count:,}",
            "latest field"
        ),

        (
            "Risk share",
            f"{risk_share:.2f}%",
            "latest field"
        )

    ]

    metric_columns = st.columns(
        4,
        gap="medium"
    )

    for col, item in zip(
        metric_columns,
        metrics
    ):

        with col:

            metric(*item)

    # -----------------------------------------------------
    # REGIONAL WATCHLIST
    # -----------------------------------------------------

    regional = []

    region_conditions = [

        (
            "Arabian Sea",
            (
                latest.latitude.between(
                    5,
                    30
                )
                &
                latest.plot_lon.between(
                    45,
                    75
                )
            )
        ),

        (
            "Bay of Bengal",
            (
                latest.latitude.between(
                    0,
                    25
                )
                &
                latest.plot_lon.between(
                    75.01,
                    100
                )
            )
        ),

        (
            "Southern Indian Ocean",
            (
                latest.latitude.between(
                    -30,
                    5
                )
                &
                latest.plot_lon.between(
                    40,
                    100
                )
            )
        )

    ]

    for name, condition in region_conditions:

        subset = latest[
            condition
        ]

        regional.append(
            (
                name,
                len(subset),
                int(
                    subset.risk_flag.sum()
                ),
                (
                    100 *
                    subset.risk_flag.mean()
                    if len(subset)
                    else 0
                )
            )
        )

    regional_df = pd.DataFrame(
        regional,
        columns=[
            "Region",
            "Cells",
            "Potential-risk cells",
            "Risk share"
        ]
    )

    st.markdown(
    """
    <div class="mini-label"
         style="margin-top:25px">
        REGIONAL WATCHLIST
    </div>
    """,
    unsafe_allow_html=True
    )

    st.dataframe(
        regional_df.style.format(
            {
                "Risk share":
                    "{:.2f}%"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # CHARTS
    # -----------------------------------------------------

    chart_left, chart_right = st.columns(
        2,
        gap="large"
    )

    # -----------------------------------------------------
    # CHART 1
    # -----------------------------------------------------

    with chart_left:

        st.markdown(
        """
        <div class="chart-shell">

            <div class="mini-label">
                SCREENING COMPOSITION
            </div>

            <h3>
                Normal vs potential bloom-risk
            </h3>

            <p>
                Count of the two latest-field
                screening outcomes.
            </p>

        </div>
        """,
        unsafe_allow_html=True
        )

        chart_data = pd.DataFrame(
            {
                "Classification": [
                    "Normal",
                    "Potential bloom risk"
                ],

                "Cells": [
                    normal_count,
                    risk_count
                ]
            }
        )

        fig = px.bar(
            chart_data,
            x="Classification",
            y="Cells",
            text="Cells",
            color="Classification",
            color_discrete_map={
                "Normal":
                    "#3e9eac",

                "Potential bloom risk":
                    "#ef5362"
            }
        )

        fig.update_traces(
            textposition="outside",
            cliponaxis=False
        )

        fig.update_layout(

            height=315,

            margin=dict(
                l=60,
                r=20,
                t=30,
                b=65
            ),

            paper_bgcolor=
                "rgba(0,0,0,0)",

            plot_bgcolor="#ffffff",

            showlegend=False,

            font=dict(
                color="#174b56"
            ),

            xaxis=dict(

                title="Screening outcome",

                title_font=dict(
                    size=13,
                    color="#174b56"
                ),

                tickfont=dict(
                    size=12,
                    color="#174b56"
                )

            ),

            yaxis=dict(

                title="Number of cells",

                title_font=dict(
                    size=13,
                    color="#174b56"
                ),

                tickfont=dict(
                    size=12,
                    color="#174b56"
                ),

                gridcolor="#d6e7e8"

            )

        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False
            }
        )

    # -----------------------------------------------------
    # CHART 2
    # -----------------------------------------------------

    with chart_right:

        st.markdown(
        """
        <div class="chart-shell">

            <div class="mini-label">
                CHLOROPHYLL-A DISTRIBUTION
            </div>

            <h3>
                Where values cluster
            </h3>

            <p>
                Distribution of the current
                satellite-derived field.
            </p>

        </div>
        """,
        unsafe_allow_html=True
        )

        hist = px.histogram(
            latest,
            x="chla",
            nbins=30,
            color_discrete_sequence=[
                "#159eaa"
            ]
        )

        hist.update_layout(

            height=315,

            margin=dict(
                l=60,
                r=20,
                t=30,
                b=65
            ),

            paper_bgcolor=
                "rgba(0,0,0,0)",

            plot_bgcolor="#ffffff",

            font=dict(
                color="#174b56"
            ),

            xaxis=dict(

                title="Chlorophyll-a",

                title_font=dict(
                    size=13,
                    color="#174b56"
                ),

                tickfont=dict(
                    size=12,
                    color="#174b56"
                ),

                gridcolor="#eef4f4"

            ),

            yaxis=dict(

                title="Number of cells",

                title_font=dict(
                    size=13,
                    color="#174b56"
                ),

                tickfont=dict(
                    size=12,
                    color="#174b56"
                ),

                gridcolor="#d6e7e8"

            )

        )

        st.plotly_chart(
            hist,
            use_container_width=True,
            config={
                "displaylogo": False
            }
        )

    # -----------------------------------------------------
    # QUEUE
    # -----------------------------------------------------

    st.markdown(
    """
    <div class="section"
         style="padding-top:25px">

        <div class="kicker">
            INVESTIGATION QUEUE
        </div>

        <h2>
            Currently flagged coordinates.
        </h2>

        <p>
            This is a follow-up list, not a ranking
            of confirmed HAB severity.
        </p>

    </div>
    """,
    unsafe_allow_html=True
    )

    queue = (
        risk
        .sort_values(
            ["chla"],
            ascending=False
        )
        .head(25)
        .copy()
    )

    queue_columns = [
        column
        for column in [
            "latitude",
            "longitude",
            "chla",
            "chla_anomaly",
            "chla_change",
            "risk_probability",
            "risk_label"
        ]
        if column in queue.columns
    ]

    queue = queue[
        queue_columns
    ]

    for column in [
        "chla",
        "chla_anomaly",
        "chla_change",
        "risk_probability"
    ]:

        if column in queue.columns:

            queue[column] = (
                queue[column]
                .round(4)
            )

    st.dataframe(
        queue,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# HOW IT WORKS
# =========================================================

elif st.session_state.page == "method":

    section(
        "05 · Project pipeline",
        "From satellite observation to screening support.",
        "The pipeline is deliberately visible: each step explains what happens before a potential-risk flag reaches the map."
    )

    steps = [

        (
            "01",
            "Observe",
            "EOS-06 OCM-3 analysed chlorophyll-a observations provide the environmental signal."
        ),

        (
            "02",
            "Clean",
            "Invalid observations are removed and spatial-temporal records are organized consistently."
        ),

        (
            "03",
            "Build context",
            "Previous observation, historical baseline, recent mean, recent maximum, anomaly and change features provide temporal context."
        ),

        (
            "04",
            "Screen",
            "A Decision Tree model screens the prepared features for the project proxy potential-risk label."
        ),

        (
            "05",
            "Map",
            "The latest screening output is returned to geographic coordinates so spatial concentrations can be inspected."
        ),

        (
            "06",
            "Investigate",
            "Flagged cells and hotspot concentrations become a shortlist for additional observations and validation."
        )

    ]

    st.markdown(
        '<div class="method-grid">',
        unsafe_allow_html=True
    )

    for number, title, description in steps:

        st.markdown(
        f"""
        <div class="step">

            <div class="step-num">
                {number}
            </div>

            <div>

                <h3>
                    {title}
                </h3>

                <p>
                    {description}
                </p>

            </div>

        </div>
        """,
        unsafe_allow_html=True
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # MODEL DETAILS
    # -----------------------------------------------------

    st.markdown(
    """
    <div class="section"
         style="padding-top:30px">

        <div class="kicker">
            MODEL CONTEXT
        </div>

        <h2>
            What the model actually uses.
        </h2>

        <p>
            The final Decision Tree uses five prepared
            features derived from satellite chlorophyll-a
            observations and their historical context.
        </p>

    </div>
    """,
    unsafe_allow_html=True
    )

    model_columns = st.columns(
        5,
        gap="small"
    )

    model_features = [

        "Current Chl-a",

        "Previous Chl-a",

        "Historical baseline",

        "Recent mean",

        "Recent maximum"

    ]

    for col, feature in zip(
        model_columns,
        model_features
    ):

        with col:

            st.markdown(
            f"""
            <div class="signal">

                <div class="slabel">
                    MODEL FEATURE
                </div>

                <div class="sval">
                    {feature}
                </div>

            </div>
            """,
            unsafe_allow_html=True
            )

    st.markdown(
    """
    <div class="note">

        <b>
            Scientific boundary:
        </b>

        The current satellite dataset does not contain
        confirmed harmful-bloom species or toxin labels.
        The system therefore reports
        <b>potential bloom risk</b>, not confirmed HAB
        detection. Satellite screening should be combined
        with field observations and other environmental
        evidence.

    </div>
    """,
    unsafe_allow_html=True
    )


# =========================================================
# DATA
# =========================================================

elif st.session_state.page == "data":

    section(
        "06 · Dataset & outputs",
        "Dataset, fields and useful downloads.",
        "This page contains the source product, grid structure, processed fields and downloadable outputs. Other pages focus on analysis rather than repeating the documentation."
    )

    # -----------------------------------------------------
    # DATA METRICS
    # -----------------------------------------------------

    data_metrics = [

        (
            "Product",
            "E06OCM_L4_AC",
            "EOS-06 / OCM-3"
        ),

        (
            "Grid",
            "0.25°",
            "latitude × longitude"
        ),

        (
            "Latest cells",
            f"{len(latest):,}",
            "processed observation"
        ),

        (
            "Latest date",
            latest_date.strftime(
                "%d %b %Y"
            ),
            "processed dataset"
        )

    ]

    data_columns = st.columns(
        4,
        gap="medium"
    )

    for col, item in zip(
        data_columns,
        data_metrics
    ):

        with col:

            metric(*item)

    # -----------------------------------------------------
    # SOURCE
    # -----------------------------------------------------

    st.markdown(
    """
    <div class="card"
         style="margin-top:18px">

        <div class="mini-label">
            SOURCE & INTERPRETATION
        </div>

        <h3>
            What the dashboard is built from.
        </h3>

        <p>
            <b>
                Satellite product:
            </b>

            EOS-06 OCM-3 Level-4 Analysed
            Chlorophyll Product
            (E06OCM_L4_AC).
        </p>

        <p>
            <b>
                Primary variable:
            </b>

            chlorophyll-a
            (<code>chla</code>).
        </p>

        <p>
            <b>
                Spatial structure:
            </b>

            0.25° latitude × 0.25° longitude
            global-ocean grid.
        </p>

        <p>
            <b>
                Interpretation:
            </b>

            the product provides an environmental
            observation signal. It does not contain
            confirmed harmful-bloom species or toxin
            labels.
        </p>

    </div>
    """,
    unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # DATA DICTIONARY
    # -----------------------------------------------------

    section(
        "Data dictionary",
        "Fields available in the processed file.",
        "The public dashboard uses these fields to inspect observations and screening context."
    )

    fields = [

        (
            "latitude",
            "Grid latitude",
            "degrees"
        ),

        (
            "longitude",
            "Grid longitude",
            "degrees"
        ),

        (
            "date",
            "Observation date",
            "date"
        ),

        (
            "chla",
            "Satellite-derived chlorophyll-a",
            "product value"
        ),

        (
            "risk_label",
            "Model screening result",
            "Normal / Potential Bloom Risk"
        ),

        (
            "risk_probability",
            "Model probability when available",
            "stored model probability"
        ),

        (
            "previous_chla",
            "Previous observation",
            "derived feature"
        ),

        (
            "historical_baseline",
            "Historical baseline",
            "derived feature"
        ),

        (
            "recent_mean",
            "Recent mean",
            "derived feature"
        ),

        (
            "recent_max",
            "Recent maximum",
            "derived feature"
        ),

        (
            "chla_anomaly",
            "Chlorophyll anomaly",
            "derived feature"
        ),

        (
            "chla_change",
            "Change from previous observation",
            "derived feature"
        )

    ]

    table_rows = ""

    for field, meaning, field_type in fields:

        table_rows += f"""
        <tr>

            <td>
                {field}
            </td>

            <td>
                {meaning}
            </td>

            <td>
                {field_type}
            </td>

        </tr>
        """

    st.markdown(
    f"""
    <div class="table-wrap">

        <table>

            <thead>

                <tr>

                    <th>
                        Field
                    </th>

                    <th>
                        Meaning
                    </th>

                    <th>
                        Type
                    </th>

                </tr>

            </thead>

            <tbody>

                {table_rows}

            </tbody>

        </table>

    </div>
    """,
    unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # DOWNLOADS
    # -----------------------------------------------------

    section(
        "Useful outputs",
        "Take the screening result with you.",
        "Downloads are kept here so the dashboard remains easy to audit and reuse."
    )

    download_left, download_right = st.columns(
        2,
        gap="large"
    )

    with download_left:

        st.markdown(
        """
        <div class="download-panel">

            <h3>
                Latest observation table
            </h3>

            <p>
                Complete latest processed prediction
                table used by the dashboard.
            </p>

        </div>
        """,
        unsafe_allow_html=True
        )

        st.download_button(
            "⬇ Download latest observations",
            latest.to_csv(
                index=False
            ).encode(),
            "bloomdetect_latest_observations.csv",
            "text/csv",
            use_container_width=True
        )

    with download_right:

        st.markdown(
        """
        <div class="download-panel">

            <h3>
                Potential-risk shortlist
            </h3>

            <p>
                Only the latest locations currently
                flagged for potential bloom risk.
            </p>

        </div>
        """,
        unsafe_allow_html=True
        )

        st.download_button(
            "⬇ Download potential-risk locations",
            risk.to_csv(
                index=False
            ).encode(),
            "bloomdetect_potential_risk.csv",
            "text/csv",
            use_container_width=True
        )

    st.markdown(
    """
    <div class="note">

        <b>
            Data boundary:
        </b>

        This is not a live real-time monitoring feed.
        Satellite screening should be combined with
        field observations and other environmental
        evidence before decisions about a harmful event
        are made.

    </div>
    """,
    unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
f"""
<div class="footer">

    BloomDetect AI · EOS-06 / OCM-3 ·
    Potential bloom-risk screening ·
    Current data through
    {latest_date.strftime("%d %B %Y")}

</div>
""",
unsafe_allow_html=True
)
