import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import textwrap
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "latest_bloom_risk_predictions.csv"

# Optional local image
INFOGRAPHIC_PATH = BASE_DIR / "bloomdetect_bloom_process.png"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        st.error(
            "latest_bloom_risk_predictions.csv was not found. "
            "Make sure it is in the same GitHub folder as app.py."
        )
        st.stop()

    df = pd.read_csv(DATA_PATH)

    # Standardize column names
    df.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in df.columns
    ]

    # Convert numeric columns where available
    numeric_columns = [
        "latitude",
        "longitude",
        "chla",
        "risk_probability",
        "probability"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Find risk column
    if "risk_label" not in df.columns:

        if "prediction" in df.columns:
            df["risk_label"] = df["prediction"]

        elif "risk" in df.columns:
            df["risk_label"] = df["risk"]

        else:
            df["risk_label"] = 0

    return df


df = load_data()


# ============================================================
# CLEAN RISK LABEL
# ============================================================

def is_risk(value):

    text = str(value).strip().lower()

    return (
        text in [
            "1",
            "1.0",
            "true",
            "risk",
            "potential bloom risk",
            "potential_bloom_risk",
            "flagged"
        ]
    )


df["is_risk"] = df["risk_label"].apply(is_risk)


# ============================================================
# PROBABILITY
# ============================================================

if "risk_probability" in df.columns:

    probability_column = "risk_probability"

elif "probability" in df.columns:

    probability_column = "probability"

else:

    probability_column = None


def probability_text(value):

    if pd.isna(value):
        return "N/A"

    value = float(value)

    if value <= 1:
        value = value * 100

    return f"{value:.1f}%"


# ============================================================
# BASIC DATA
# ============================================================

latest_date = "2026-03-30"

if "date" in df.columns:

    try:

        dates = pd.to_datetime(df["date"], errors="coerce")

        if dates.notna().any():
            latest_date = dates.max().strftime("%Y-%m-%d")

    except Exception:
        pass


total_cells = len(df)

risk_cells = int(df["is_risk"].sum())

normal_cells = total_cells - risk_cells

risk_percentage = (
    (risk_cells / total_cells) * 100
    if total_cells > 0
    else 0
)


# ============================================================
# REGIONAL CLASSIFICATION
# ============================================================

def classify_region(lat, lon):

    if pd.isna(lat) or pd.isna(lon):
        return "Other"

    if 5 <= lat <= 30 and 40 <= lon <= 80:
        return "Arabian Sea"

    if 5 <= lat <= 30 and 80 < lon <= 100:
        return "Bay of Bengal"

    if -10 <= lat < 5 and 40 <= lon <= 100:
        return "Equatorial Indian Ocean"

    if -40 <= lat < -10 and 20 <= lon <= 120:
        return "Southern Indian Ocean"

    return "Other"


df["region"] = df.apply(
    lambda row: classify_region(
        row.get("latitude"),
        row.get("longitude")
    ),
    axis=1
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"


def go_to(page):

    st.session_state.page = page


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    textwrap.dedent(
        """
        <style>

        /* ==================================================
           GLOBAL
        ================================================== */

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        header {
            visibility: hidden;
        }

        .stApp {
            background:
                radial-gradient(
                    circle at 10% 10%,
                    rgba(83, 196, 211, 0.14),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 90% 80%,
                    rgba(17, 113, 148, 0.10),
                    transparent 30%
                ),
                #eefafb;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        /* ==================================================
           WATER ANIMATION
        ================================================== */

        .water-animation {
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            height: 85px;
            overflow: hidden;
            pointer-events: none;
            z-index: 0;
            opacity: 0.25;
        }

        .water-wave {
            position: absolute;
            left: -10%;
            width: 120%;
            height: 55px;
            border-radius: 50%;
            border-top: 4px solid #1597aa;
            animation: waveMove 9s linear infinite;
        }

        .water-wave:nth-child(2) {
            top: 20px;
            opacity: 0.6;
            animation-duration: 12s;
            animation-direction: reverse;
        }

        .water-wave:nth-child(3) {
            top: 38px;
            opacity: 0.35;
            animation-duration: 15s;
        }

        @keyframes waveMove {

            0% {
                transform: translateX(-3%);
            }

            50% {
                transform: translateX(3%);
            }

            100% {
                transform: translateX(-3%);
            }

        }

        /* ==================================================
           BRAND
        ================================================== */

        .brand-card {
            background: rgba(255,255,255,0.82);
            border: 2px solid #b7dfe5;
            border-radius: 24px;
            padding: 20px 24px;
            box-shadow: 0 12px 35px rgba(12, 75, 91, 0.08);
            margin-bottom: 14px;
        }

        .brand-left {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .logo {
            width: 52px;
            height: 52px;
            border-radius: 16px;
            background: linear-gradient(
                145deg,
                #0b5366,
                #16a1ad
            );
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 25px;
            font-weight: 800;
            box-shadow: 0 8px 18px rgba(12, 99, 117, 0.25);
        }

        .brand-name {
            font-size: 1.55rem;
            font-weight: 800;
            color: #073d4d;
            letter-spacing: -0.5px;
        }

        .brand-sub {
            font-size: 0.82rem;
            color: #52727b;
            margin-top: 2px;
        }

        /* ==================================================
           NAVIGATION
        ================================================== */

        div[data-testid="stHorizontalBlock"] {
            gap: 10px;
        }

        .nav-caption {
            color: #456a73;
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 7px;
        }

        /* All normal buttons */

        .stButton > button {
            width: 100%;
            min-height: 42px;
            border-radius: 13px;
            border: 2px solid #07536a;
            background: rgba(255,255,255,0.88);
            color: #073d4d;
            font-weight: 700;
            font-size: 0.82rem;
            transition: all 0.2s ease;
            box-shadow: 0 4px 10px rgba(7, 83, 106, 0.05);
        }

        .stButton > button:hover {
            background: #e4f7fa;
            border-color: #043d50;
            transform: translateY(-2px);
            box-shadow: 0 7px 16px rgba(7, 83, 106, 0.12);
        }

        /* ==================================================
           HERO
        ================================================== */

        .hero {
            position: relative;
            overflow: hidden;
            border-radius: 28px;
            min-height: 440px;
            margin-top: 14px;
            margin-bottom: 22px;

            background:
                linear-gradient(
                    115deg,
                    rgba(3, 63, 80, 0.96),
                    rgba(8, 124, 142, 0.84)
                ),
                url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1800&q=80");

            background-size: cover;
            background-position: center;
            box-shadow:
                0 22px 55px rgba(3, 73, 88, 0.22);
        }

        .hero::before {
            content: "";
            position: absolute;
            width: 650px;
            height: 650px;
            border: 2px solid rgba(255,255,255,0.13);
            border-radius: 50%;
            right: -180px;
            top: -260px;
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 500px;
            height: 500px;
            border: 2px solid rgba(255,255,255,0.10);
            border-radius: 50%;
            right: -40px;
            bottom: -330px;
        }

        .hero-content {
            position: relative;
            z-index: 2;
            padding: 58px 52px;
            max-width: 720px;
        }

        .kicker {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 30px;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.30);
            color: #eaffff;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.7px;
            margin-bottom: 18px;
        }

        .hero h1 {
            color: white;
            font-size: 3.25rem;
            line-height: 1.05;
            margin: 0 0 20px 0;
            letter-spacing: -1.8px;
        }

        .hero p {
            color: #e1f8fa;
            font-size: 1.04rem;
            line-height: 1.65;
            max-width: 650px;
        }

        .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 9px;
            margin-top: 24px;
        }

        .badge {
            padding: 8px 13px;
            border-radius: 20px;
            background: rgba(255,255,255,0.13);
            border: 1px solid rgba(255,255,255,0.25);
            color: white;
            font-size: 0.78rem;
            font-weight: 700;
        }

        /* ==================================================
           SECTION HEADERS
        ================================================== */

        .section-title {
            color: #073d4d;
            font-size: 1.65rem;
            font-weight: 800;
            margin: 22px 0 5px 0;
        }

        .section-subtitle {
            color: #55737a;
            font-size: 0.9rem;
            margin-bottom: 15px;
        }

        /* ==================================================
           CARDS
        ================================================== */

        .glass-card {
            background: rgba(255,255,255,0.78);
            border: 2px solid #b9dfe5;
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 10px 25px rgba(12, 75, 91, 0.07);
            height: 100%;
        }

        .glass-card h3 {
            color: #073d4d;
            margin: 0 0 8px 0;
            font-size: 1.05rem;
        }

        .glass-card p {
            color: #58727a;
            font-size: 0.88rem;
            line-height: 1.55;
        }

        .feature-icon {
            font-size: 1.75rem;
            margin-bottom: 8px;
        }

        /* ==================================================
           METRIC CARDS
        ================================================== */

        .metric-card {
            background: rgba(255,255,255,0.84);
            border: 2px solid #b7dfe5;
            border-radius: 18px;
            padding: 18px;
            text-align: center;
            min-height: 112px;
            box-shadow: 0 8px 20px rgba(8, 73, 88, 0.06);
        }

        .metric-value {
            color: #07566b;
            font-size: 1.65rem;
            font-weight: 850;
        }

        .metric-label {
            color: #607980;
            font-size: 0.76rem;
            font-weight: 700;
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* ==================================================
           STATUS BOX
        ================================================== */

        .status-risk {
            background: rgba(255, 235, 235, 0.92);
            border: 2px solid #d94b4b;
            border-radius: 18px;
            padding: 19px;
            color: #8c1f1f;
            text-align: center;
        }

        .status-normal {
            background: rgba(231, 250, 240, 0.92);
            border: 2px solid #23945d;
            border-radius: 18px;
            padding: 19px;
            color: #17653f;
            text-align: center;
        }

        .status-title {
            font-size: 1.12rem;
            font-weight: 850;
            margin-bottom: 5px;
        }

        .status-small {
            font-size: 0.82rem;
        }

        /* ==================================================
           INPUT CARDS
        ================================================== */

        .input-card {
            background: rgba(244, 251, 252, 0.90);
            border: 2px solid #8fcbd4;
            border-radius: 18px;
            padding: 17px;
            height: 100%;
        }

        .input-title {
            color: #42666f;
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.7px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .input-value {
            color: #073d4d;
            font-size: 1.12rem;
            font-weight: 800;
        }

        /* ==================================================
           METHOD STEPS
        ================================================== */

        .step-card {
            background: rgba(255,255,255,0.82);
            border: 2px solid #0b5269;
            border-radius: 18px;
            padding: 18px;
            min-height: 165px;
            box-shadow: 0 7px 17px rgba(8, 73, 88, 0.07);
        }

        .step-number {
            width: 34px;
            height: 34px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #0a5268;
            color: white;
            font-weight: 850;
            margin-bottom: 11px;
        }

        .step-card h4 {
            color: #073d4d;
            margin: 0 0 7px 0;
        }

        .step-card p {
            color: #5b7379;
            font-size: 0.82rem;
            line-height: 1.5;
        }

        /* ==================================================
           DOWNLOAD
        ================================================== */

        .download-card {
            background:
                linear-gradient(
                    135deg,
                    rgba(5, 81, 103, 0.97),
                    rgba(15, 133, 148, 0.95)
                );
            border: 2px solid #073d4d;
            border-radius: 20px;
            padding: 20px;
            color: white;
            min-height: 145px;
        }

        .download-card h3 {
            margin: 0 0 7px 0;
            color: white;
        }

        .download-card p {
            color: #d9f5f7;
            font-size: 0.83rem;
        }

        /* ==================================================
           STREAMLIT DOWNLOAD BUTTON
        ================================================== */

        .stDownloadButton > button {
            width: 100%;
            border-radius: 12px;
            border: 2px solid #073d4d;
            background: white;
            color: #073d4d;
            font-weight: 800;
        }

        .stDownloadButton > button:hover {
            background: #e5f8fa;
            border-color: #052f3c;
        }

        /* ==================================================
           CHART CARDS
        ================================================== */

        .chart-title {
            color: #073d4d;
            font-size: 0.98rem;
            font-weight: 800;
            margin-bottom: 4px;
        }

        .chart-note {
            color: #607980;
            font-size: 0.76rem;
            margin-bottom: 6px;
        }

        /* ==================================================
           FOOTER
        ================================================== */

        .footer {
            margin-top: 35px;
            padding: 20px;
            border-top: 1px solid #b9dfe5;
            text-align: center;
            color: #668087;
            font-size: 0.75rem;
        }

        /* ==================================================
           RESPONSIVE
        ================================================== */

        @media (max-width: 800px) {

            .hero-content {
                padding: 38px 25px;
            }

            .hero h1 {
                font-size: 2.35rem;
            }

            .brand-name {
                font-size: 1.25rem;
            }

        }

        </style>
        """
    ),
    unsafe_allow_html=True
)


# ============================================================
# WATER ANIMATION
# ============================================================

st.markdown(
    textwrap.dedent(
        """
        <div class="water-animation">
            <div class="water-wave"></div>
            <div class="water-wave"></div>
            <div class="water-wave"></div>
        </div>
        """
    ),
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    textwrap.dedent(
        """
        <div class="brand-card">

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
        """
    ),
    unsafe_allow_html=True
)


# ============================================================
# NAVIGATION
# ============================================================

pages = [
    "Home",
    "Risk Map",
    "Risk Checker",
    "Hotspots",
    "Insights",
    "How It Works",
    "Data"
]

nav_columns = st.columns(len(pages))

for i, page in enumerate(pages):

    with nav_columns[i]:

        if st.button(
            page,
            key=f"nav_{page}",
            type="primary" if st.session_state.page == page else "secondary"
        ):
            go_to(page)


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "Home":

    st.markdown(
        textwrap.dedent(
            f"""
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
                        chlorophyll-a observations into an interactive
                        screening system for identifying locations whose
                        recent patterns may deserve closer attention.
                    </p>

                    <div class="hero-badges">

                        <span class="badge">
                            🌊 Spatial screening
                        </span>

                        <span class="badge">
                            🛰️ EOS-06 OCM-3
                        </span>

                        <span class="badge">
                            🌱 Potential bloom-risk
                        </span>

                        <span class="badge">
                            🤖 AI-assisted
                        </span>

                    </div>

                </div>

            </div>
            """
        ),
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # QUICK METRICS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Today’s screening snapshot</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Latest processed satellite observations.</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{total_cells:,}</div>
                <div class="metric-label">Processed cells</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{risk_cells:,}</div>
                <div class="metric-label">Potential risk cells</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{risk_percentage:.2f}%</div>
                <div class="metric-label">Risk share</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{latest_date}</div>
                <div class="metric-label">Latest observation</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Explore BloomDetect AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Move through the system using the sections below.</div>',
        unsafe_allow_html=True
    )

    f1, f2, f3, f4 = st.columns(4)

    features = [
        (
            f1,
            "🗺️",
            "Risk Map",
            "View the focused Indian Ocean study region and potential bloom-risk cells.",
            "Risk Map"
        ),
        (
            f2,
            "📍",
            "Risk Checker",
            "Enter a latitude and longitude and inspect the nearest processed observation.",
            "Risk Checker"
        ),
        (
            f3,
            "🔥",
            "Hotspots",
            "See where potential risk cells are concentrated in the latest screening.",
            "Hotspots"
        ),
        (
            f4,
            "📊",
            "Insights",
            "Explore compact summaries and useful patterns from the screening.",
            "Insights"
        )
    ]

    for column, icon, title, description, target in features:

        with column:

            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="glass-card">

                        <div class="feature-icon">
                            {icon}
                        </div>

                        <h3>{title}</h3>

                        <p>
                            {description}
                        </p>

                    </div>
                    """
                ),
                unsafe_allow_html=True
            )

            if st.button(
                f"Open {title}",
                key=f"home_{target}"
            ):
                go_to(target)
                st.rerun()

    # --------------------------------------------------------
    # SCIENTIFIC NOTE
    # --------------------------------------------------------

    st.markdown(
        textwrap.dedent(
            """
            <div class="glass-card" style="margin-top:20px;">

                <h3>Important interpretation</h3>

                <p>
                    BloomDetect AI identifies <b>potential bloom-risk patterns</b>
                    from satellite-derived observations. A high chlorophyll-a
                    value by itself does not confirm a harmful algal bloom,
                    species, or toxin presence. Satellite screening is intended
                    as early-warning support and should be complemented by
                    field validation.
                </p>

            </div>
            """
        ),
        unsafe_allow_html=True
    )


# ============================================================
# MAP FUNCTION
# ============================================================

def create_risk_map(data):

    map_data = data.copy()

    # Focused Indian Ocean study area
    map_data = map_data[
        (map_data["latitude"] >= -40) &
        (map_data["latitude"] <= 30) &
        (map_data["longitude"] >= 20) &
        (map_data["longitude"] <= 120)
    ].copy()

    normal_data = map_data[~map_data["is_risk"]].copy()
    risk_data = map_data[map_data["is_risk"]].copy()

    # --------------------------------------------------------
    # Normal background grid
    # Aggregate to 1-degree cells for performance
    # --------------------------------------------------------

    if len(normal_data) > 0:

        normal_data["lat_bin"] = np.floor(
            normal_data["latitude"]
        ).astype(int)

        normal_data["lon_bin"] = np.floor(
            normal_data["longitude"]
        ).astype(int)

        normal_grid = (
            normal_data
            .groupby(["lat_bin", "lon_bin"], as_index=False)
            .agg(
                latitude=("latitude", "mean"),
                longitude=("longitude", "mean"),
                cells=("latitude", "size")
            )
        )

    else:

        normal_grid = pd.DataFrame(
            columns=["latitude", "longitude", "cells"]
        )

    # --------------------------------------------------------
    # Risk zones
    # Aggregate risk cells into 2-degree zones
    # --------------------------------------------------------

    if len(risk_data) > 0:

        risk_data["lat_zone"] = np.floor(
            risk_data["latitude"] / 2
        ) * 2

        risk_data["lon_zone"] = np.floor(
            risk_data["longitude"] / 2
        ) * 2

        risk_zones = (
            risk_data
            .groupby(
                ["lat_zone", "lon_zone"],
                as_index=False
            )
            .agg(
                latitude=("latitude", "mean"),
                longitude=("longitude", "mean"),
                risk_cells=("latitude", "size")
            )
        )

    else:

        risk_zones = pd.DataFrame(
            columns=[
                "latitude",
                "longitude",
                "risk_cells"
            ]
        )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig = go.Figure()

    # Normal ocean field
    if len(normal_grid) > 0:

        fig.add_trace(
            go.Scattergeo(
                lat=normal_grid["latitude"],
                lon=normal_grid["longitude"],
                mode="markers",
                name="Normal processed region",
                marker=dict(
                    size=5,
                    color="#4b9bb5",
                    opacity=0.55
                ),
                hovertemplate=(
                    "<b>Processed region</b><br>"
                    "Latitude: %{lat:.2f}<br>"
                    "Longitude: %{lon:.2f}"
                    "<extra></extra>"
                )
            )
        )

    # Green potential risk zones
    if len(risk_zones) > 0:

        zone_sizes = np.clip(
            np.sqrt(risk_zones["risk_cells"]) * 3,
            10,
            32
        )

        fig.add_trace(
            go.Scattergeo(
                lat=risk_zones["latitude"],
                lon=risk_zones["longitude"],
                mode="markers",
                name="Potential risk zone",
                marker=dict(
                    size=zone_sizes,
                    color="#36a86a",
                    opacity=0.48,
                    line=dict(
                        color="#197446",
                        width=1
                    )
                ),
                customdata=risk_zones["risk_cells"],
                hovertemplate=(
                    "<b>Potential bloom-risk zone</b><br>"
                    "Latitude: %{lat:.2f}<br>"
                    "Longitude: %{lon:.2f}<br>"
                    "Flagged cells: %{customdata:,}"
                    "<extra></extra>"
                )
            )
        )

    # Red individual risk cells
    if len(risk_data) > 0:

        hover_text = []

        for _, row in risk_data.iterrows():

            chla_value = row.get("chla", np.nan)

            if pd.isna(chla_value):
                chla_text = "N/A"
            else:
                chla_text = f"{float(chla_value):.4f}"

            if probability_column:
                prob = probability_text(
                    row.get(probability_column)
                )
            else:
                prob = "N/A"

            hover_text.append(
                f"<b>Potential bloom-risk cell</b><br>"
                f"Latitude: {row['latitude']:.2f}<br>"
                f"Longitude: {row['longitude']:.2f}<br>"
                f"Chlorophyll-a: {chla_text}<br>"
                f"Model probability: {prob}"
            )

        fig.add_trace(
            go.Scattergeo(
                lat=risk_data["latitude"],
                lon=risk_data["longitude"],
                mode="markers",
                name="Flagged cell",
                marker=dict(
                    size=7,
                    color="#d93636",
                    opacity=0.90,
                    line=dict(
                        color="#8d1818",
                        width=0.7
                    )
                ),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>"
            )
        )

    # Study area boundary
    boundary_lat = [-40, -40, 30, 30, -40]
    boundary_lon = [20, 120, 120, 20, 20]

    fig.add_trace(
        go.Scattergeo(
            lat=boundary_lat,
            lon=boundary_lon,
            mode="lines",
            name="Study area",
            line=dict(
                color="#073d4d",
                width=1.5,
                dash="dot"
            ),
            hoverinfo="skip"
        )
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.update_geos(
        projection_type="mercator",
        center=dict(
            lat=-5,
            lon=70
        ),
        lataxis=dict(
            range=[-40, 30]
        ),
        lonaxis=dict(
            range=[20, 120]
        ),
        showland=True,
        landcolor="#dfe9e8",
        showocean=True,
        oceancolor="#d9f3f7",
        showcountries=True,
        countrycolor="#9ab6bc",
        coastlinecolor="#6e9098",
        showlakes=True,
        lakecolor="#d9f3f7",
        resolution=50
    )

    fig.update_layout(
        height=620,
        margin=dict(
            l=0,
            r=0,
            t=10,
            b=0
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.86)",
            bordercolor="#9ccbd2",
            borderwidth=1
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig


# ============================================================
# RISK MAP PAGE
# ============================================================

if st.session_state.page == "Risk Map":

    st.markdown(
        '<div class="section-title">Potential Bloom-Risk Map</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        textwrap.dedent(
            f"""
            <div class="section-subtitle">
                Latest processed screening for {latest_date}.
                The map focuses on the Indian Ocean study region.
            </div>
            """
        ),
        unsafe_allow_html=True
    )

    fig = create_risk_map(df)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "displaylogo": False
        }
    )

    st.markdown(
        textwrap.dedent(
            """
            <div class="glass-card">

                <h3>Map legend</h3>

                <p>
                    <b style="color:#4b9bb5;">● Blue</b>
                    = normal processed region
                    &nbsp;&nbsp;
                    <b style="color:#36a86a;">● Green</b>
                    = concentrated potential-risk zone
                    &nbsp;&nbsp;
                    <b style="color:#d93636;">● Red</b>
                    = individual flagged cell
                </p>

                <p>
                    Red cells represent model-screened locations whose
                    recent satellite-derived patterns meet the project's
                    potential bloom-risk screening criteria.
                </p>

            </div>
            """
        ),
        unsafe_allow_html=True
    )


# ============================================================
# RISK CHECKER PAGE
# ============================================================

if st.session_state.page == "Risk Checker":

    st.markdown(
        '<div class="section-title">Location / Risk Checker</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Enter a location and inspect the nearest processed satellite cell.</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:

        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=17.3850,
            step=0.01,
            format="%.4f"
        )

    with c2:

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=78.4867,
            step=0.01,
            format="%.4f"
        )

    check_button = st.button(
        "🔎 Check this location",
        key="check_location",
        type="primary"
    )

    if check_button:

        valid = df.dropna(
            subset=["latitude", "longitude"]
        ).copy()

        # Distance approximation
        valid["distance"] = (
            (valid["latitude"] - latitude) ** 2
            +
            (valid["longitude"] - longitude) ** 2
        )

        nearest = valid.loc[
            valid["distance"].idxmin()
        ]

        nearest_lat = float(nearest["latitude"])
        nearest_lon = float(nearest["longitude"])

        nearest_risk = bool(
            nearest["is_risk"]
        )

        nearest_chla = nearest.get(
            "chla",
            np.nan
        )

        if probability_column:

            nearest_probability = probability_text(
                nearest.get(probability_column)
            )

        else:

            nearest_probability = "N/A"

        # ----------------------------------------------------
        # Input and nearest observation
        # ----------------------------------------------------

        left, right = st.columns(2)

        with left:

            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="input-card">

                        <div class="input-title">
                            Your input
                        </div>

                        <div class="input-value">
                            Latitude: {latitude:.4f}<br>
                            Longitude: {longitude:.4f}
                        </div>

                    </div>
                    """
                ),
                unsafe_allow_html=True
            )

        with right:

            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="input-card">

                        <div class="input-title">
                            Nearest processed cell
                        </div>

                        <div class="input-value">
                            Latitude: {nearest_lat:.4f}<br>
                            Longitude: {nearest_lon:.4f}
                        </div>

                    </div>
                    """
                ),
                unsafe_allow_html=True
            )

        st.write("")

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if nearest_risk:

            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="status-risk">

                        <div class="status-title">
                            🔴 POTENTIAL BLOOM-RISK FLAGGED
                        </div>

                        <div class="status-small">
                            The nearest processed cell was flagged by
                            the BloomDetect AI screening model.
                        </div>

                        <br>

                        <b>Chlorophyll-a:</b>
                        {nearest_chla:.4f}
                        &nbsp;&nbsp; | &nbsp;&nbsp;
                        <b>Model probability:</b>
                        {nearest_probability}

                    </div>
                    """
                ),
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="status-normal">

                        <div class="status-title">
                            🟢 NOT FLAGGED
                        </div>

                        <div class="status-small">
                            The nearest processed cell was not flagged
                            by the BloomDetect AI screening model.
                        </div>

                        <br>

                        <b>Chlorophyll-a:</b>
                        {nearest_chla:.4f}
                        &nbsp;&nbsp; | &nbsp;&nbsp;
                        <b>Model probability:</b>
                        {nearest_probability}

                    </div>
                    """
                ),
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # Location information
        # ----------------------------------------------------

        region = classify_region(
            nearest_lat,
            nearest_lon
        )

        st.write("")

        a, b, c = st.columns(3)

        with a:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                        {nearest_lat:.2f}°
                    </div>
                    <div class="metric-label">
                        Processed latitude
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with b:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                        {nearest_lon:.2f}°
                    </div>
                    <div class="metric-label">
                        Processed longitude
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                        {region}
                    </div>
                    <div class="metric-label">
                        Region
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # Study area warning
        # ----------------------------------------------------

        if not (
            -40 <= latitude <= 30
            and
            20 <= longitude <= 120
        ):

            st.warning(
                "The entered coordinates are outside the project's "
                "focused Indian Ocean study area. The checker therefore "
                "returns the nearest available processed observation."
            )


# ============================================================
# HOTSPOTS PAGE
# ============================================================

if st.session_state.page == "Hotspots":

    st.markdown(
        '<div class="section-title">Potential Bloom-Risk Hotspots</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Locations where flagged cells are concentrated in the latest screening.</div>',
        unsafe_allow_html=True
    )

    risk_data = df[df["is_risk"]].copy()

    if len(risk_data) == 0:

        st.info(
            "No potential bloom-risk cells were flagged in the current dataset."
        )

    else:

        # ----------------------------------------------------
        # Hotspot aggregation
        # ----------------------------------------------------

        risk_data["lat_zone"] = np.floor(
            risk_data["latitude"] / 2
        ) * 2

        risk_data["lon_zone"] = np.floor(
            risk_data["longitude"] / 2
        ) * 2

        hotspots = (
            risk_data
            .groupby(
                ["lat_zone", "lon_zone"],
                as_index=False
            )
            .agg(
                flagged_cells=("latitude", "size"),
                avg_chla=("chla", "mean")
            )
            .sort_values(
                "flagged_cells",
                ascending=False
            )
            .head(10)
        )

        hotspots["region"] = hotspots.apply(
            lambda row: classify_region(
                row["lat_zone"],
                row["lon_zone"]
            ),
            axis=1
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        h1, h2, h3 = st.columns(3)

        with h1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                        {len(risk_data):,}
                    </div>
                    <div class="metric-label">
                        Flagged cells
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with h2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                        {len(hotspots)}
                    </div>
                    <div class="metric-label">
                        Concentration zones
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with h3:

            top_region = (
                hotspots.iloc[0]["region"]
                if len(hotspots) > 0
                else "N/A"
            )

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                        {top_region}
                    </div>
                    <div class="metric-label">
                        Largest hotspot zone
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        # ----------------------------------------------------
        # Hotspot chart
        # ----------------------------------------------------

        fig_hotspot = px.bar(
            hotspots.sort_values(
                "flagged_cells",
                ascending=True
            ),
            x="flagged_cells",
            y="region",
            orientation="h",
            labels={
                "flagged_cells": "Flagged cells",
                "region": "Region"
            }
        )

        fig_hotspot.update_traces(
            marker_color="#1597aa"
        )

        fig_hotspot.update_layout(
            height=390,
            margin=dict(
                l=10,
                r=10,
                t=15,
                b=10
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="Number of flagged cells",
                showgrid=True
            ),
            yaxis=dict(
                title="Region"
            )
        )

        st.plotly_chart(
            fig_hotspot,
            use_container_width=True
        )

        # ----------------------------------------------------
        # Table
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Top concentration zones</div>',
            unsafe_allow_html=True
        )

        display_hotspots = hotspots.copy()

        display_hotspots["lat_zone"] = (
            display_hotspots["lat_zone"]
            .round(2)
        )

        display_hotspots["lon_zone"] = (
            display_hotspots["lon_zone"]
            .round(2)
        )

        display_hotspots["avg_chla"] = (
            display_hotspots["avg_chla"]
            .round(4)
        )

        display_hotspots.columns = [
            "Latitude zone",
            "Longitude zone",
            "Flagged cells",
            "Average Chl-a",
            "Region"
        ]

        st.dataframe(
            display_hotspots,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# INSIGHTS PAGE
# ============================================================

if st.session_state.page == "Insights":

    st.markdown(
        '<div class="section-title">Screening Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">A compact view of the latest screening results.</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Standout metrics
    # --------------------------------------------------------

    i1, i2, i3, i4 = st.columns(4)

    with i1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {risk_percentage:.2f}%
                </div>
                <div class="metric-label">
                    Cells flagged
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with i2:

        risk_ratio = (
            risk_cells / normal_cells
            if normal_cells > 0
            else 0
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    1 : {1/risk_ratio:.0f}
                </div>
                <div class="metric-label">
                    Risk to normal ratio
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with i3:

        avg_chla = (
            df["chla"].mean()
            if "chla" in df.columns
            else np.nan
        )

        avg_chla_text = (
            f"{avg_chla:.3f}"
            if not pd.isna(avg_chla)
            else "N/A"
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {avg_chla_text}
                </div>
                <div class="metric-label">
                    Mean Chl-a
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with i4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {latest_date}
                </div>
                <div class="metric-label">
                    Screening date
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # Chart 1
    # --------------------------------------------------------

    st.write("")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            '<div class="chart-title">Normal vs potential risk</div>',
            unsafe_allow_html=True
        )

        chart_df = pd.DataFrame(
            {
                "Category": [
                    "Normal",
                    "Potential bloom-risk"
                ],
                "Cells": [
                    normal_cells,
                    risk_cells
                ]
            }
        )

        fig1 = px.pie(
            chart_df,
            names="Category",
            values="Cells",
            hole=0.55
        )

        fig1.update_traces(
            marker=dict(
                colors=[
                    "#4b9bb5",
                    "#d93636"
                ]
            ),
            textinfo="percent+label"
        )

        fig1.update_layout(
            height=340,
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Chart 2
    # --------------------------------------------------------

    with c2:

        st.markdown(
            '<div class="chart-title">Flagged cells by region</div>',
            unsafe_allow_html=True
        )

        region_counts = (
            df[df["is_risk"]]
            .groupby("region")
            .size()
            .reset_index(name="Flagged cells")
            .sort_values(
                "Flagged cells",
                ascending=False
            )
        )

        if len(region_counts) > 0:

            fig2 = px.bar(
                region_counts,
                x="region",
                y="Flagged cells",
                labels={
                    "region": "Region",
                    "Flagged cells": "Flagged cells"
                }
            )

            fig2.update_traces(
                marker_color="#36a86a"
            )

            fig2.update_layout(
                height=340,
                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=45
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Region",
                yaxis_title="Flagged cells"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

        else:

            st.info(
                "No regional potential-risk observations are available."
            )

    # --------------------------------------------------------
    # Chl-a distribution
    # --------------------------------------------------------

    if "chla" in df.columns:

        st.markdown(
            '<div class="section-title">Chlorophyll-a distribution</div>',
            unsafe_allow_html=True
        )

        plot_df = df[
            df["chla"].notna()
            &
            (df["chla"] >= 0)
        ].copy()

        if len(plot_df) > 0:

            upper = plot_df["chla"].quantile(0.99)

            plot_df = plot_df[
                plot_df["chla"] <= upper
            ]

            plot_df["Screening"] = np.where(
                plot_df["is_risk"],
                "Potential bloom-risk",
                "Normal"
            )

            fig3 = px.histogram(
                plot_df,
                x="chla",
                color="Screening",
                nbins=45,
                labels={
                    "chla": "Chlorophyll-a",
                    "count": "Number of cells"
                }
            )

            fig3.update_layout(
                height=380,
                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=45
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Chlorophyll-a",
                yaxis_title="Number of cells"
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    st.markdown(
        textwrap.dedent(
            """
            <div class="glass-card">

                <h3>What stands out right now?</h3>

                <p>
                    The screening highlights a relatively small subset of
                    processed ocean cells as potential bloom-risk locations.
                    These locations are screening signals rather than
                    confirmed harmful algal blooms.
                </p>

                <p>
                    Chlorophyll-a patterns should therefore be interpreted
                    together with temporal behaviour, spatial context and,
                    where available, environmental information and field
                    observations.
                </p>

            </div>
            """
        ),
        unsafe_allow_html=True
    )


# ============================================================
# HOW IT WORKS PAGE
# ============================================================

if st.session_state.page == "How It Works":

    st.markdown(
        '<div class="section-title">How BloomDetect AI Works</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">From satellite observations to potential bloom-risk screening.</div>',
        unsafe_allow_html=True
    )

    steps = [
        (
            "1",
            "Satellite data",
            "EOS-06 OCM-3 ocean-colour observations provide the satellite-derived chlorophyll-a data used by the system."
        ),
        (
            "2",
            "Historical context",
            "Previous observations are used to build a historical baseline and recent temporal features for each processed location."
        ),
        (
            "3",
            "Feature preparation",
            "Current chlorophyll-a, previous chlorophyll-a, historical baseline, recent mean and recent maximum are prepared for model screening."
        ),
        (
            "4",
            "AI screening",
            "A Decision Tree model classifies each processed cell according to the project's potential bloom-risk screening target."
        ),
        (
            "5",
            "Spatial mapping",
            "The predictions are placed back onto geographic coordinates to create a spatial potential bloom-risk map."
        ),
        (
            "6",
            "Early-warning support",
            "The resulting signals can help identify locations that may deserve closer monitoring or validation."
        )
    ]

    row1 = st.columns(3)

    for i in range(3):

        number, title, description = steps[i]

        with row1[i]:

            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="step-card">

                        <div class="step-number">
                            {number}
                        </div>

                        <h4>
                            {title}
                        </h4>

                        <p>
                            {description}
                        </p>

                    </div>
                    """
                ),
                unsafe_allow_html=True
            )

    st.write("")

    row2 = st.columns(3)

    for i in range(3, 6):

        number, title, description = steps[i]

        with row2[i - 3]:

            st.markdown(
                textwrap.dedent(
                    f"""
                    <div class="step-card">

                        <div class="step-number">
                            {number}
                        </div>

                        <h4>
                            {title}
                        </h4>

                        <p>
                            {description}
                        </p>

                    </div>
                    """
                ),
                unsafe_allow_html=True
            )

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Current model</div>',
        unsafe_allow_html=True
    )

    model1, model2, model3 = st.columns(3)

    with model1:

        st.markdown(
            """
            <div class="glass-card">

                <h3>Model</h3>

                <p>
                    Decision Tree Classifier
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with model2:

        st.markdown(
            """
            <div class="glass-card">

                <h3>Input features</h3>

                <p>
                    Chl-a, previous Chl-a, historical baseline,
                    recent mean and recent maximum.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with model3:

        st.markdown(
            """
            <div class="glass-card">

                <h3>Purpose</h3>

                <p>
                    Potential bloom-risk screening and spatial
                    early-warning support.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # Scientific limitation
    # --------------------------------------------------------

    st.markdown(
        textwrap.dedent(
            """
            <div class="glass-card" style="margin-top:20px;">

                <h3>Scientific limitation</h3>

                <p>
                    The current dataset does not contain confirmed
                    harmful-algal-bloom species or toxin labels.
                    Therefore, the model output should be described as
                    <b>potential bloom-risk screening</b>, not confirmed
                    HAB detection.
                </p>

                <p>
                    Satellite observations complement field measurements
                    and cannot independently confirm species or toxin
                    presence.
                </p>

            </div>
            """
        ),
        unsafe_allow_html=True
    )


# ============================================================
# DATA PAGE
# ============================================================

if st.session_state.page == "Data":

    st.markdown(
        '<div class="section-title">Dataset & Downloads</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Information about the processed data used by the dashboard.</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Dataset metrics
    # --------------------------------------------------------

    d1, d2, d3, d4 = st.columns(4)

    with d1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {total_cells:,}
                </div>
                <div class="metric-label">
                    Latest observations
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with d2:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">
                    0.25°
                </div>
                <div class="metric-label">
                    Approx. grid resolution
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with d3:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">
                    OCM-3
                </div>
                <div class="metric-label">
                    Satellite sensor
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with d4:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">
                    DT
                </div>
                <div class="metric-label">
                    Final ML model
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Data source</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        textwrap.dedent(
            """
            <div class="glass-card">

                <h3>EOS-06 OCM-3 Level-4 Analysed Chlorophyll Product</h3>

                <p>
                    Product ID:
                    <b>E06OCM_L4_AC</b>
                </p>

                <p>
                    Variable used:
                    <b>Chlorophyll-a (chla)</b>
                </p>

                <p>
                    Format:
                    <b>NetCDF</b>
                </p>

                <p>
                    The dashboard displays the processed latest-date
                    predictions rather than the full raw satellite archive.
                </p>

            </div>
            """
        ),
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Downloads
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Downloads</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Useful project outputs available directly from the dashboard.</div>',
        unsafe_allow_html=True
    )

    # CSV download
    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    # Summary text
    summary_text = f"""
BloomDetect AI
==============================

Project:
AI-Based Harmful Algal Bloom Detection and Early Warning System

Latest screening date:
{latest_date}

Processed cells:
{total_cells:,}

Potential bloom-risk cells:
{risk_cells:,}

Risk share:
{risk_percentage:.2f}%

Model:
Decision Tree Classifier

Satellite product:
EOS-06 OCM-3 E06OCM_L4_AC

Interpretation:
The output represents potential bloom-risk screening.
It does not independently confirm a harmful algal bloom,
species, or toxin presence.

Satellite screening should be complemented by field validation.
"""

    download1, download2 = st.columns(2)

    with download1:

        st.markdown(
            textwrap.dedent(
                """
                <div class="download-card">

                    <h3>📄 Screening data</h3>

                    <p>
                        Download the latest processed prediction table.
                    </p>

                </div>
                """
            ),
            unsafe_allow_html=True
        )

        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="latest_bloom_risk_predictions.csv",
            mime="text/csv",
            key="download_csv"
        )

    with download2:

        st.markdown(
            textwrap.dedent(
                """
                <div class="download-card">

                    <h3>📝 Project summary</h3>

                    <p>
                        Download a compact summary of the current screening.
                    </p>

                </div>
                """
            ),
            unsafe_allow_html=True
        )

        st.download_button(
            "Download Summary",
            data=summary_text,
            file_name="BloomDetect_AI_Screening_Summary.txt",
            mime="text/plain",
            key="download_summary"
        )

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Data preview</div>',
        unsafe_allow_html=True
    )

    preview_columns = [
        col for col in [
            "latitude",
            "longitude",
            "chla",
            "risk_label"
        ]
        if col in df.columns
    ]

    preview = df[preview_columns].head(15).copy()

    st.dataframe(
        preview,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    textwrap.dedent(
        """
        <div class="footer">

            <b>BloomDetect AI</b>
            · EOS-06 OCM-3 satellite-based potential bloom-risk screening

            <br><br>

            This system provides early-warning support.
            Satellite screening does not independently confirm HAB species
            or toxin presence and should be complemented by field validation.

        </div>
        """
    ),
    unsafe_allow_html=True
)
