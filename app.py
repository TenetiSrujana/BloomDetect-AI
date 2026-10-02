import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# =========================================================
# BLOOMDETECT AI
# =========================================================

st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="≈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "latest_bloom_risk_predictions.csv"
INFOGRAPHIC_PATH = BASE_DIR / "bloomdetect_bloom_process.png"

HERO_IMAGE_URL = (
    "https://images.unsplash.com/photo-1500375592092-40eb2168fd21"
    "?auto=format&fit=crop&w=1800&q=85"
)

# =========================================================
# GLOBAL STYLE
# =========================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root {
    --ink: #103f4c;
    --muted: #607f89;
    --aqua: #0f91a0;
    --aqua2: #24b9c1;
    --deep: #07596a;
    --line: #a9dce2;
    --bg: #f2fbfc;
    --card: #ffffff;
    --green: #238b63;
    --red: #d94c4c;
}

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(36,185,193,.12),
            transparent 25%
        ),
        radial-gradient(
            circle at 90% 30%,
            rgba(15,145,160,.10),
            transparent 26%
        ),
        linear-gradient(
            180deg,
            #f8ffff 0%,
            #eef9fa 100%
        );
    color: var(--ink);
}

.block-container {
    max-width: 1180px;
    padding-top: 2.2rem !important;
    padding-bottom: 4rem !important;
}

header[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    visibility: hidden;
    height: 0;
}

h1, h2, h3, h4 {
    font-family: "Manrope", sans-serif !important;
    color: var(--ink) !important;
    letter-spacing: -0.035em;
}

p, li, label, .stMarkdown {
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.65;
}

/* =========================================================
   BRAND
   ========================================================= */

.brand {
    background: rgba(255,255,255,.94);
    border: 2px solid #b8e3e7;
    border-radius: 24px;
    padding: 18px 22px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 16px 35px rgba(5,82,99,.08);
    margin-bottom: 14px;
}

.brand-logo {
    width: 58px;
    height: 58px;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 28px;
    font-weight: 800;
    background: linear-gradient(135deg, #1cc4c5, #087f96);
    box-shadow: 0 9px 20px rgba(7,89,106,.22);
}

.brand-title {
    font-family: "Manrope", sans-serif;
    font-size: 1.42rem;
    font-weight: 800;
    color: var(--ink);
    line-height: 1.1;
}

.brand-subtitle {
    margin-top: 4px;
    color: #73939b;
    font-size: .84rem;
    font-weight: 600;
}

/* =========================================================
   NAVIGATION
   ========================================================= */

.stButton > button {
    min-height: 45px;
    border-radius: 14px;
    border: 2px solid #15586a !important;
    background: rgba(255,255,255,.96) !important;
    color: #154653 !important;
    font-weight: 700 !important;
    font-size: .92rem !important;
    box-shadow: 0 5px 12px rgba(8,77,91,.07);
    transition: all .18s ease;
}

.stButton > button:hover {
    border-color: #0c8c9a !important;
    color: #087d8b !important;
    transform: translateY(-2px);
    box-shadow: 0 9px 18px rgba(8,77,91,.13);
}

.active-nav > button {
    background: linear-gradient(
        135deg,
        #08677a,
        #12a5ae
    ) !important;
    color: white !important;
    border-color: #064d5d !important;
}

/* =========================================================
   HOME HERO
   ========================================================= */

.hero {
    position: relative;
    overflow: hidden;
    min-height: 500px;
    border-radius: 30px;
    padding: 50px 48px 38px;
    color: white;

    background:
        linear-gradient(
            110deg,
            rgba(2,64,78,.96),
            rgba(5,116,127,.89) 58%,
            rgba(19,169,170,.80)
        ),
        url("https://images.unsplash.com/photo-1500375592092-40eb2168fd21?auto=format&fit=crop&w=1800&q=85")
        center/cover no-repeat;

    box-shadow: 0 24px 45px rgba(4,82,96,.19);
}

.hero::before,
.hero::after {
    content: "";
    position: absolute;
    width: 150%;
    height: 90px;
    left: -25%;
    border-top: 3px solid rgba(170,239,242,.22);
    border-radius: 50%;
    animation: oceanWave 9s ease-in-out infinite;
}

.hero::before {
    top: 70px;
    transform: rotate(5deg);
}

.hero::after {
    top: 230px;
    animation-delay: -3s;
    transform: rotate(-4deg);
}

@keyframes oceanWave {
    0%, 100% {
        transform: translateX(-2%) rotate(4deg);
    }

    50% {
        transform: translateX(3%) rotate(-2deg);
    }
}

.hero-content {
    position: relative;
    z-index: 3;
    max-width: 660px;
}

.hero-kicker {
    color: #bdf8f6;
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .18em;
    text-transform: uppercase;
    margin-bottom: 20px;
}

.hero h1 {
    color: white !important;
    font-size: clamp(2.4rem, 5vw, 4.45rem) !important;
    line-height: .98 !important;
    max-width: 650px;
    margin: 0 0 20px 0 !important;
    text-shadow: 0 3px 20px rgba(0,0,0,.22);
}

.hero-copy {
    color: rgba(255,255,255,.94) !important;
    font-size: 1.05rem !important;
    max-width: 690px;
    line-height: 1.7 !important;
}

.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 23px;
}

.hero-pill {
    display: inline-block;
    padding: 8px 12px;
    border-radius: 999px;
    color: white;
    background: rgba(255,255,255,.14);
    border: 1px solid rgba(255,255,255,.27);
    backdrop-filter: blur(8px);
    font-size: .82rem;
    font-weight: 700;
}

.hero-side {
    position: absolute;
    z-index: 4;
    right: 36px;
    top: 42px;
    width: 330px;
    border-radius: 24px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,.48);
    background: rgba(255,255,255,.13);
    backdrop-filter: blur(9px);
    box-shadow: 0 15px 35px rgba(0,0,0,.18);
}

.hero-side img {
    display: block;
    width: 100%;
    height: 205px;
    object-fit: cover;
}

.hero-side-text {
    padding: 16px 18px 18px;
}

.hero-side-title {
    color: white;
    font-weight: 800;
    font-size: 1.05rem;
}

.hero-side-copy {
    color: rgba(255,255,255,.82);
    font-size: .82rem;
    margin-top: 5px;
    line-height: 1.55;
}

.hero-stats {
    position: relative;
    z-index: 5;
    margin-top: 44px;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    max-width: 800px;
}

.hero-stat {
    background: rgba(255,255,255,.14);
    border: 1px solid rgba(255,255,255,.25);
    border-radius: 17px;
    padding: 15px 16px;
    backdrop-filter: blur(8px);
}

.hero-stat-value {
    color: white;
    font-size: 1.55rem;
    font-weight: 800;
}

.hero-stat-label {
    color: rgba(255,255,255,.80);
    font-size: .73rem;
    margin-top: 2px;
}

/* =========================================================
   HOME SECTIONS
   ========================================================= */

.section-kicker {
    color: #078b9b;
    font-size: .76rem;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
    margin-top: 42px;
    margin-bottom: 9px;
}

.section-title {
    font-family: "Manrope", sans-serif;
    font-size: clamp(2rem, 4vw, 3.25rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -.045em;
    color: var(--ink);
    margin-bottom: 12px;
}

.section-copy {
    color: var(--muted);
    max-width: 920px;
    font-size: 1rem;
    line-height: 1.7;
    margin-bottom: 22px;
}

.info-card {
    background: rgba(255,255,255,.95);
    border: 1.5px solid #b9e1e5;
    border-radius: 19px;
    padding: 23px;
    min-height: 220px;
    box-shadow: 0 12px 26px rgba(8,82,97,.07);
    transition: transform .18s ease, box-shadow .18s ease;
}

.info-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 18px 30px rgba(8,82,97,.11);
}

.info-icon {
    font-size: 1.55rem;
    margin-bottom: 13px;
}

.info-title {
    color: var(--ink);
    font-family: "Manrope", sans-serif;
    font-size: 1.04rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.info-copy {
    color: var(--muted);
    font-size: .89rem;
    line-height: 1.65;
}

/* =========================================================
   WORKFLOW
   ========================================================= */

.workflow {
    background: linear-gradient(
        135deg,
        rgba(227,249,250,.98),
        rgba(255,255,255,.98)
    );
    border: 1.5px solid #b6e0e5;
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 12px 28px rgba(7,83,99,.06);
}

.workflow-title {
    font-family: "Manrope", sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    color: var(--ink);
    margin-bottom: 5px;
}

.workflow-copy {
    color: var(--muted);
    font-size: .9rem;
}

.flow-step {
    background: white;
    border: 1.5px solid #a9dce2;
    border-radius: 15px;
    padding: 17px;
    min-height: 125px;
}

.flow-num {
    color: #078b9b;
    font-size: .7rem;
    font-weight: 800;
    letter-spacing: .12em;
}

.flow-name {
    color: var(--ink);
    font-weight: 800;
    margin-top: 7px;
    font-size: .95rem;
}

.flow-text {
    color: var(--muted);
    font-size: .79rem;
    line-height: 1.5;
    margin-top: 4px;
}

/* =========================================================
   CAVEAT
   ========================================================= */

.caveat {
    background: #e9f9fa;
    border-left: 5px solid #11a1ad;
    border-radius: 12px;
    padding: 14px 17px;
    color: #416d77;
    font-size: .84rem;
    line-height: 1.6;
    margin-top: 22px;
}

/* =========================================================
   STANDARD CARDS
   ========================================================= */

.page-card {
    background: rgba(255,255,255,.95);
    border: 1.5px solid #b7dfe4;
    border-radius: 20px;
    padding: 23px;
    box-shadow: 0 10px 25px rgba(8,82,97,.06);
}

.metric-card {
    background: white;
    border: 1.5px solid #b7dfe4;
    border-radius: 17px;
    padding: 18px;
    min-height: 112px;
}

.metric-value {
    color: var(--ink);
    font-family: "Manrope", sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
}

.metric-label {
    color: var(--muted);
    font-size: .78rem;
    margin-top: 3px;
}

/* =========================================================
   RISK CHECKER
   ========================================================= */

.check-card {
    background: white;
    border: 1.5px solid #b7dfe4;
    border-radius: 18px;
    padding: 19px;
    min-height: 140px;
}

.check-label {
    color: #6d8b94;
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .1em;
    text-transform: uppercase;
}

.check-value {
    color: var(--ink);
    font-size: 1.25rem;
    font-weight: 800;
    margin-top: 8px;
}

.flagged {
    background: #fff1f1;
    border: 2px solid #e06a6a;
    color: #a83232;
    border-radius: 18px;
    padding: 18px;
}

.clear {
    background: #edf9f4;
    border: 2px solid #59ad8a;
    color: #216d50;
    border-radius: 18px;
    padding: 18px;
}

.flag-title {
    font-weight: 800;
    font-size: 1.12rem;
}

.flag-copy {
    margin-top: 4px;
    font-size: .86rem;
}

/* =========================================================
   DOWNLOADS
   ========================================================= */

.download-card {
    background: linear-gradient(
        135deg,
        #e9fafb,
        #ffffff
    );
    border: 2px solid #8fd1d8;
    border-radius: 18px;
    padding: 18px;
    min-height: 150px;
}

/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    margin-top: 55px;
    padding-top: 20px;
    border-top: 1px solid #c9e5e8;
    color: #78959d;
    font-size: .76rem;
    line-height: 1.6;
}

/* =========================================================
   RESPONSIVE
   ========================================================= */

@media (max-width: 900px) {

    .hero {
        padding: 36px 27px 30px;
    }

    .hero-side {
        position: relative;
        right: auto;
        top: auto;
        width: 100%;
        margin-top: 28px;
    }

    .hero-stats {
        grid-template-columns: 1fr;
        margin-top: 25px;
    }
}

@media (max-width: 640px) {

    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    .brand {
        padding: 14px;
    }

    .brand-title {
        font-size: 1.15rem;
    }

    .hero h1 {
        font-size: 2.35rem !important;
    }

    .section-title {
        font-size: 2rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data(show_spinner=False)
def load_data():

    if not DATA_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(DATA_PATH)

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ):
        return pd.DataFrame()

    if df.empty:
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

    for col in [
        "latitude",
        "longitude",
        "chla",
        "risk_probability",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    if "risk_label" in df.columns:
        df["risk_label"] = df["risk_label"].astype(str)

    return df


df = load_data()

if df.empty:

    st.error(
        "The screening CSV could not be read. "
        "Make sure latest_bloom_risk_predictions.csv "
        "is in the same GitHub folder as app.py."
    )

    st.stop()

# =========================================================
# COMMON DATA
# =========================================================

if "risk_label" not in df.columns:

    if "prediction" in df.columns:

        df["risk_label"] = np.where(
            df["prediction"].astype(str).str.contains("1"),
            "Potential Bloom Risk",
            "Normal",
        )

    else:

        df["risk_label"] = "Normal"


latest_date = (
    df["date"].max()
    if "date" in df.columns
    else pd.NaT
)

if pd.notna(latest_date):

    latest = df[
        df["date"] == latest_date
    ].copy()

else:

    latest = df.copy()


risk_mask = latest[
    "risk_label"
].str.lower().str.contains(
    "risk|flag|bloom",
    na=False
)

risk_df = latest[risk_mask].copy()

normal_df = latest[~risk_mask].copy()

risk_count = len(risk_df)

latest_count = len(latest)

# =========================================================
# NAVIGATION
# =========================================================

if "page" not in st.session_state:

    st.session_state.page = "Home"


pages = [
    "Home",
    "Risk Map",
    "Risk Checker",
    "Hotspots",
    "Insights",
    "How It Works",
    "Data",
]


st.markdown(
    """
<div class="brand">
    <div class="brand-logo">≈</div>

    <div>
        <div class="brand-title">
            BloomDetect AI
        </div>

        <div class="brand-subtitle">
            Satellite-based potential bloom-risk screening
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


nav_cols = st.columns(
    7,
    gap="small"
)


for i, page_name in enumerate(pages):

    with nav_cols[i]:

        if st.button(
            page_name,
            key=f"nav_{page_name}",
            use_container_width=True,
        ):

            st.session_state.page = page_name

            st.rerun()


# =========================================================
# HOME
# =========================================================

if st.session_state.page == "Home":

    hero_side = f"""
<div class="hero-side">

    <img
        src="{HERO_IMAGE_URL}"
        alt="Ocean surface"
    >

    <div class="hero-side-text">

        <div class="hero-side-title">
            From a huge ocean field to a focused shortlist.
        </div>

        <div class="hero-side-copy">
            Use the screening output to identify locations
            that deserve a closer look, then investigate them
            using the map, checker and hotspot views.
        </div>

    </div>

</div>
"""

    st.markdown(
        f"""
<div class="hero">

    <div class="hero-content">

        <div class="hero-kicker">
            EOS-06 · OCM-3 · BLOOMDETECT AI
        </div>

        <h1>
            See change<br>
            beneath the surface.
        </h1>

        <div class="hero-copy">
            BloomDetect AI turns satellite-derived chlorophyll-a
            observations into a practical screening workflow for
            finding ocean locations whose recent patterns may
            deserve closer investigation.
        </div>

        <div class="hero-pills">

            <span class="hero-pill">
                🛰 Satellite screening
            </span>

            <span class="hero-pill">
                🌊 Spatial context
            </span>

            <span class="hero-pill">
                🔥 Potential-risk hotspots
            </span>

            <span class="hero-pill">
                📍 Coordinate investigation
            </span>

        </div>

    </div>

    {hero_side}

    <div class="hero-stats">

        <div class="hero-stat">

            <div class="hero-stat-value">
                {latest_count:,}
            </div>

            <div class="hero-stat-label">
                latest processed grid cells
            </div>

        </div>


        <div class="hero-stat">

            <div class="hero-stat-value">
                {risk_count:,}
            </div>

            <div class="hero-stat-label">
                current potential-risk flags
            </div>

        </div>


        <div class="hero-stat">

            <div class="hero-stat-value">
                {
                    latest_date.strftime("%d %b %Y")
                    if pd.notna(latest_date)
                    else "—"
                }
            </div>

            <div class="hero-stat-label">
                latest observation date
            </div>

        </div>

    </div>

</div>
""",
        unsafe_allow_html=True,
    )

    # =====================================================
    # WHY THIS MATTERS
    # =====================================================

    st.markdown(
        """
<div class="section-kicker">
    01 · WHY THIS MATTERS
</div>

<div class="section-title">
    Algal blooms are an environmental signal,
    not just a colour on a map.
</div>

<div class="section-copy">
    Algae and phytoplankton are normal parts of marine ecosystems.
    When their concentration changes sharply or becomes unusually high,
    an area can deserve closer ecological attention. Some blooms are
    harmless, while some can be associated with environmental or health
    impacts. The practical challenge is deciding where to look first
    across a very large ocean area.
</div>
""",
        unsafe_allow_html=True,
    )

    info_cols = st.columns(
        3,
        gap="large"
    )

    info_cards = [

        (
            "🌱",
            "What are algae?",
            """
            Microscopic algae and phytoplankton live throughout
            the ocean. They form the base of many marine food webs
            and contain pigments such as chlorophyll-a that can be
            observed from satellites.
            """
        ),

        (
            "🐟",
            "Why does a change matter?",
            """
            A strong or unusual change in chlorophyll-a can indicate
            that something in the ocean environment deserves closer
            investigation. It is a screening signal, not automatic
            proof of a harmful bloom.
            """
        ),

        (
            "🔬",
            "Who can use the output?",
            """
            Researchers, students, coastal and environmental teams
            can use the map and flagged coordinates to narrow a large
            search area before deeper analysis, field observation
            or validation.
            """
        ),
    ]

    for col, (icon, title, copy) in zip(
        info_cols,
        info_cards
    ):

        with col:

            st.markdown(
                f"""
<div class="info-card">

    <div class="info-icon">
        {icon}
    </div>

    <div class="info-title">
        {title}
    </div>

    <div class="info-copy">
        {copy}
    </div>

</div>
""",
                unsafe_allow_html=True,
            )

    # =====================================================
    # WHAT BLOOMDETECT ADDS
    # =====================================================

    st.markdown(
        """
<div class="section-kicker">
    02 · WHAT BLOOMDETECT ADDS
</div>

<div class="section-title">
    It helps answer one practical question:
    where should we look closer?
</div>

<div class="section-copy">
    Instead of manually scanning thousands of ocean cells,
    the system combines the current satellite observation
    with historical and recent temporal context to create
    a shortlist of locations for investigation.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="workflow">

    <div class="workflow-title">
        Turn a large satellite field into an investigation workflow.
    </div>

    <div class="workflow-copy">
        The dashboard does not replace field sampling or scientific
        confirmation. It reduces the first-search problem:
        which locations deserve attention first?
    </div>

</div>
""",
        unsafe_allow_html=True,
    )

    flow_cols = st.columns(
        4,
        gap="medium"
    )

    steps = [

        (
            "01",
            "MAP",
            "See the full study area and understand the spatial pattern."
        ),

        (
            "02",
            "HOTSPOTS",
            "Find concentrated groups of currently flagged cells."
        ),

        (
            "03",
            "CHECK",
            "Inspect a coordinate and its nearest processed observation."
        ),

        (
            "04",
            "ACT",
            "Download the screening output for further analysis or validation."
        ),
    ]

    for col, (num, name, text_value) in zip(
        flow_cols,
        steps
    ):

        with col:

            st.markdown(
                f"""
<div class="flow-step">

    <div class="flow-num">
        {num}
    </div>

    <div class="flow-name">
        {name}
    </div>

    <div class="flow-text">
        {text_value}
    </div>

</div>
""",
                unsafe_allow_html=True,
            )

    # =====================================================
    # HOME ACTION BUTTONS
    # =====================================================

    action_cols = st.columns(
        4,
        gap="medium"
    )

    actions = [

        (
            "🗺 Open Risk Map",
            "Risk Map"
        ),

        (
            "🔥 Find Hotspots",
            "Hotspots"
        ),

        (
            "📍 Check a Location",
            "Risk Checker"
        ),

        (
            "📊 View Insights",
            "Insights"
        ),
    ]

    for col, (label, target) in zip(
        action_cols,
        actions
    ):

        with col:

            if st.button(
                label,
                key=f"home_action_{target}",
                use_container_width=True,
            ):

                st.session_state.page = target

                st.rerun()

    # =====================================================
    # USE RESULT CORRECTLY
    # =====================================================

    st.markdown(
        """
<div class="section-kicker">
    03 · USE THE RESULT CORRECTLY
</div>

<div class="section-title">
    A potential-risk flag is a starting point
    for investigation.
</div>

<div class="section-copy">
    BloomDetect uses satellite-derived chlorophyll-a behaviour
    as a screening signal. High chlorophyll-a alone does not prove
    a harmful algal bloom, and satellite observations cannot identify
    every species or toxin. The output is therefore designed as
    early-warning support that can be combined with field observations,
    environmental information and scientific validation.
</div>

<div class="caveat">
    <b>Scientific boundary:</b>
    BloomDetect reports potential bloom-risk signals,
    not confirmed HAB events. The current dataset is satellite-derived
    and does not contain confirmed harmful-bloom ground-truth labels.
</div>
""",
        unsafe_allow_html=True,
    )

    # =====================================================
    # USER-GENERATED IMAGE
    # =====================================================

    if INFOGRAPHIC_PATH.exists():

        st.markdown(
            """
<div class="section-kicker">
    04 · HOW A BLOOM CAN DEVELOP
</div>

<div class="section-title">
    From ocean conditions to satellite observation.
</div>

<div class="section-copy">
    This visual gives a quick practical view of the process behind
    the signal: environmental conditions influence algal growth,
    water movement changes where material accumulates, and satellite
    observations provide the chlorophyll-a information used by the
    screening workflow.
</div>
""",
            unsafe_allow_html=True,
        )

        st.image(
            str(INFOGRAPHIC_PATH),
            use_container_width=True
        )


# =========================================================
# RISK MAP
# =========================================================

elif st.session_state.page == "Risk Map":

    st.markdown(
        '<div class="section-kicker">SPATIAL VIEW</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Potential bloom-risk screening map</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="section-copy">
    Blue shows the processed ocean field. Green shows areas
    containing flagged cells. Red marks individual potential-risk
    cells from the latest observation date.
</div>
""",
        unsafe_allow_html=True,
    )

    map_df = latest.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    if len(map_df) > 30000:

        base_df = map_df.sample(
            30000,
            random_state=42
        )

    else:

        base_df = map_df

    risk_zone = risk_df.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    if not risk_zone.empty:

        risk_zone["lat_bin"] = (
            np.floor(risk_zone["latitude"] / 2) * 2
        )

        risk_zone["lon_bin"] = (
            np.floor(risk_zone["longitude"] / 2) * 2
        )

        zones = (
            risk_zone
            .groupby(
                ["lat_bin", "lon_bin"]
            )
            .size()
            .reset_index(
                name="flagged_cells"
            )
        )

        zones["lat"] = zones["lat_bin"] + 1
        zones["lon"] = zones["lon_bin"] + 1

        zones["marker_size"] = np.clip(
            12
            + np.sqrt(
                zones["flagged_cells"]
            ) * 5,
            12,
            42
        )

    else:

        zones = pd.DataFrame(
            columns=[
                "lat",
                "lon",
                "flagged_cells",
                "marker_size"
            ]
        )

    fig = go.Figure()

    # Blue normal field
    fig.add_trace(
        go.Scattergeo(

            lat=base_df["latitude"],
            lon=base_df["longitude"],

            mode="markers",

            name="Normal / processed field",

            marker=dict(
                size=4.5,
                color="#2c8db1",
                opacity=0.34,
            ),

            hovertemplate=(
                "<b>Processed cell</b><br>"
                "Latitude: %{lat:.2f}<br>"
                "Longitude: %{lon:.2f}"
                "<extra></extra>"
            ),
        )
    )

    # Green risk zones
    if not zones.empty:

        fig.add_trace(
            go.Scattergeo(

                lat=zones["lat"],
                lon=zones["lon"],

                mode="markers",

                name="Potential-risk zone",

                marker=dict(
                    size=zones["marker_size"],
                    color="#38a66f",
                    opacity=0.42,
                    line=dict(
                        color="#1f7e54",
                        width=1
                    ),
                ),

                customdata=zones[
                    ["flagged_cells"]
                ].values,

                hovertemplate=(
                    "<b>Potential-risk zone</b><br>"
                    "Flagged cells: %{customdata[0]}<br>"
                    "Approx. latitude: %{lat:.1f}<br>"
                    "Approx. longitude: %{lon:.1f}"
                    "<extra></extra>"
                ),
            )
        )

    # Red individual risk cells
    if not risk_df.empty:

        risk_plot = risk_df.dropna(
            subset=["latitude", "longitude"]
        ).copy()

        if len(risk_plot) > 5000:

            risk_plot = risk_plot.sample(
                5000,
                random_state=42
            )

        if "chla" in risk_plot.columns:

            hover_chla = (
                risk_plot["chla"]
                .round(4)
                .astype(str)
            )

        else:

            hover_chla = [
                "—"
            ] * len(risk_plot)

        fig.add_trace(
            go.Scattergeo(

                lat=risk_plot["latitude"],
                lon=risk_plot["longitude"],

                mode="markers",

                name="Flagged cell",

                marker=dict(
                    size=7,
                    color="#d94c4c",
                    opacity=0.88,
                    line=dict(
                        color="#8e2929",
                        width=0.6
                    ),
                ),

                hovertemplate=(
                    "<b>Potential bloom-risk flag</b><br>"
                    "Latitude: %{lat:.2f}<br>"
                    "Longitude: %{lon:.2f}<br>"
                    "Chlorophyll-a: %{text}"
                    "<extra></extra>"
                ),

                text=hover_chla,
            )
        )

    fig.update_geos(

        projection_type="mercator",

        center=dict(
            lat=-5,
            lon=70
        ),

        lataxis_range=[
            -40,
            30
        ],

        lonaxis_range=[
            20,
            120
        ],

        showland=True,
        landcolor="#e7f1ef",

        showocean=True,
        oceancolor="#dff4f6",

        showcountries=True,
        countrycolor="#8db4bb",

        coastlinecolor="#6f9ca4",
        coastlinewidth=1,

        showlakes=True,
        lakecolor="#dff4f6",

        bgcolor="rgba(0,0,0,0)",
    )

    fig.update_layout(

        height=650,

        margin=dict(
            l=0,
            r=0,
            t=15,
            b=0
        ),

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        """
<div class="caveat">
    <b>Map interpretation:</b>
    blue represents the processed screening field,
    green represents areas where flagged cells cluster,
    and red represents individual potential-risk cells.
    These are screening outputs, not confirmed harmful algal
    bloom boundaries.
</div>
""",
        unsafe_allow_html=True,
    )


# =========================================================
# RISK CHECKER
# =========================================================

elif st.session_state.page == "Risk Checker":

    st.markdown(
        '<div class="section-kicker">COORDINATE INVESTIGATION</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Check a location</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="section-copy">
    Enter a latitude and longitude. BloomDetect finds the nearest
    processed observation from the latest available date and shows
    its screening status.
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(
        2,
        gap="large"
    )

    with c1:

        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=10.0,
            step=0.25,
            format="%.2f",
        )

    with c2:

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=70.0,
            step=0.25,
            format="%.2f",
        )

    if st.button(
        "Check this location",
        type="primary",
        use_container_width=True,
    ):

        valid = latest.dropna(
            subset=[
                "latitude",
                "longitude"
            ]
        ).copy()

        if valid.empty:

            st.warning(
                "No processed observations are available."
            )

        else:

            valid["_distance"] = (
                (valid["latitude"] - latitude) ** 2
                +
                (valid["longitude"] - longitude) ** 2
            )

            nearest = valid.loc[
                valid["_distance"].idxmin()
            ]

            nearest_distance = float(
                np.sqrt(
                    nearest["_distance"]
                )
            )

            label = str(
                nearest.get(
                    "risk_label",
                    ""
                )
            ).lower()

            is_risk = (
                "risk" in label
                or "flag" in label
                or label in [
                    "1",
                    "true"
                ]
            )

            st.markdown(
                """
<div class="section-kicker">
    RESULT
</div>
""",
                unsafe_allow_html=True,
            )

            result_cols = st.columns(
                2,
                gap="large"
            )

            with result_cols[0]:

                st.markdown(
                    f"""
<div class="check-card">

    <div class="check-label">
        YOUR INPUT
    </div>

    <div class="check-value">
        {latitude:.2f}°, {longitude:.2f}°
    </div>

</div>
""",
                    unsafe_allow_html=True,
                )

            with result_cols[1]:

                st.markdown(
                    f"""
<div class="check-card">

    <div class="check-label">
        NEAREST PROCESSED CELL
    </div>

    <div class="check-value">
        {float(nearest['latitude']):.2f}°,
        {float(nearest['longitude']):.2f}°
    </div>

</div>
""",
                    unsafe_allow_html=True,
                )

            st.write("")

            if is_risk:

                st.markdown(
                    f"""
<div class="flagged">

    <div class="flag-title">
        🔴 Potential bloom-risk signal detected
    </div>

    <div class="flag-copy">
        The nearest processed cell is flagged for potential
        bloom risk on
        <b>
            {
                latest_date.strftime("%d %b %Y")
                if pd.notna(latest_date)
                else "the latest date"
            }
        </b>.

        This is a screening signal and requires further
        investigation or validation.
    </div>

</div>
""",
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    f"""
<div class="clear">

    <div class="flag-title">
        🟢 Not flagged in the current screening
    </div>

    <div class="flag-copy">
        The nearest processed cell is not currently flagged
        for potential bloom risk on
        <b>
            {
                latest_date.strftime("%d %b %Y")
                if pd.notna(latest_date)
                else "the latest date"
            }
        </b>.

        A non-flag does not mean the location can never
        experience a bloom.
    </div>

</div>
""",
                    unsafe_allow_html=True,
                )

            value_cols = st.columns(
                3,
                gap="medium"
            )

            with value_cols[0]:

                chla_value = nearest.get(
                    "chla",
                    np.nan
                )

                st.markdown(
                    f"""
<div class="metric-card">

    <div class="metric-value">
        {float(chla_value):.4f}
    </div>

    <div class="metric-label">
        Chlorophyll-a
    </div>

</div>
""",
                    unsafe_allow_html=True,
                )

            with value_cols[1]:

                if (
                    "risk_probability" in nearest.index
                    and pd.notna(
                        nearest["risk_probability"]
                    )
                ):

                    p = float(
                        nearest["risk_probability"]
                    )

                    if p <= 1:
                        p *= 100

                    p_text = f"{p:.1f}%"

                else:

                    p_text = "—"

                st.markdown(
                    f"""
<div class="metric-card">

    <div class="metric-value">
        {p_text}
    </div>

    <div class="metric-label">
        Model screening probability
    </div>

</div>
""",
                    unsafe_allow_html=True,
                )

            with value_cols[2]:

                st.markdown(
                    f"""
<div class="metric-card">

    <div class="metric-value">
        {nearest_distance:.3f}°
    </div>

    <div class="metric-label">
        Distance from input to nearest cell
    </div>

</div>
""",
                    unsafe_allow_html=True,
                )


# =========================================================
# HOTSPOTS
# =========================================================

elif st.session_state.page == "Hotspots":

    st.markdown(
        '<div class="section-kicker">FOCUSED SEARCH</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Where are the current flags concentrated?</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="section-copy">
    Hotspots group nearby flagged cells so a large list of
    coordinates becomes easier to investigate.
</div>
""",
        unsafe_allow_html=True,
    )

    h = risk_df.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    ).copy()

    if h.empty:

        st.info(
            "No potential-risk cells are present in the latest observation."
        )

    else:

        h["lat_bin"] = (
            np.floor(
                h["latitude"] / 2
            ) * 2
        )

        h["lon_bin"] = (
            np.floor(
                h["longitude"] / 2
            ) * 2
        )

        hotspots = (
            h.groupby(
                [
                    "lat_bin",
                    "lon_bin"
                ]
            )
            .agg(
                flagged_cells=(
                    "latitude",
                    "size"
                ),
                mean_chla=(
                    "chla",
                    "mean"
                ),
            )
            .reset_index()
        )

        hotspots["center_lat"] = (
            hotspots["lat_bin"] + 1
        )

        hotspots["center_lon"] = (
            hotspots["lon_bin"] + 1
        )

        hotspots = hotspots.sort_values(
            "flagged_cells",
            ascending=False
        ).reset_index(
            drop=True
        )

        hotspots.insert(
            0,
            "Rank",
            np.arange(
                1,
                len(hotspots) + 1
            )
        )

        m1, m2, m3 = st.columns(
            3,
            gap="medium"
        )

        with m1:

            st.markdown(
                f"""
<div class="metric-card">

    <div class="metric-value">
        {len(hotspots):,}
    </div>

    <div class="metric-label">
        spatial hotspot zones
    </div>

</div>
""",
                unsafe_allow_html=True,
            )

        with m2:

            st.markdown(
                f"""
<div class="metric-card">

    <div class="metric-value">
        {risk_count:,}
    </div>

    <div class="metric-label">
        flagged cells
    </div>

</div>
""",
                unsafe_allow_html=True,
            )

        with m3:

            largest = int(
                hotspots.iloc[0][
                    "flagged_cells"
                ]
            )

            st.markdown(
                f"""
<div class="metric-card">

    <div class="metric-value">
        {largest:,}
    </div>

    <div class="metric-label">
        flags in the largest zone
    </div>

</div>
""",
                unsafe_allow_html=True,
            )

        st.write("")

        top = hotspots.head(20).copy()

        top["mean_chla"] = (
            top["mean_chla"]
            .round(4)
        )

        st.dataframe(
            top[
                [
                    "Rank",
                    "center_lat",
                    "center_lon",
                    "flagged_cells",
                    "mean_chla",
                ]
            ].rename(
                columns={
                    "center_lat":
                        "Approx. latitude",

                    "center_lon":
                        "Approx. longitude",

                    "flagged_cells":
                        "Flagged cells",

                    "mean_chla":
                        "Mean chlorophyll-a",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# INSIGHTS
# =========================================================

elif st.session_state.page == "Insights":

    st.markdown(
        '<div class="section-kicker">SCREENING SUMMARY</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">What stands out right now?</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="section-copy">
    A compact view of the latest screening output.
    These summaries describe the current model output and
    should not be interpreted as confirmed HAB measurements.
</div>
""",
        unsafe_allow_html=True,
    )

    r1, r2, r3, r4 = st.columns(
        4,
        gap="medium"
    )

    with r1:

        st.markdown(
            f"""
<div class="metric-card">

    <div class="metric-value">
        {latest_count:,}
    </div>

    <div class="metric-label">
        latest grid cells
    </div>

</div>
""",
            unsafe_allow_html=True,
        )

    with r2:

        st.markdown(
            f"""
<div class="metric-card">

    <div class="metric-value">
        {risk_count:,}
    </div>

    <div class="metric-label">
        potential-risk flags
    </div>

</div>
""",
            unsafe_allow_html=True,
        )

    with r3:

        share = (
            risk_count
            / latest_count
            * 100
            if latest_count
            else 0
        )

        st.markdown(
            f"""
<div class="metric-card">

    <div class="metric-value">
        {share:.2f}%
    </div>

    <div class="metric-label">
        flagged share of latest field
    </div>

</div>
""",
            unsafe_allow_html=True,
        )

    with r4:

        if "chla" in latest.columns:

            median_chla = latest[
                "chla"
            ].median()

        else:

            median_chla = np.nan

        st.markdown(
            f"""
<div class="metric-card">

    <div class="metric-value">
        {median_chla:.4f}
    </div>

    <div class="metric-label">
        median chlorophyll-a
    </div>

</div>
""",
            unsafe_allow_html=True,
        )

    st.write("")

    chart1, chart2 = st.columns(
        2,
        gap="large"
    )

    with chart1:

        status_counts = pd.DataFrame(
            {
                "Status": [
                    "Normal / not flagged",
                    "Potential bloom risk"
                ],

                "Cells": [
                    len(normal_df),
                    len(risk_df)
                ],
            }
        )

        fig1 = px.bar(
            status_counts,
            x="Status",
            y="Cells",
            text="Cells",
        )

        fig1.update_traces(
            marker_color=[
                "#2c8db1",
                "#d94c4c"
            ],
            textposition="outside",
        )

        fig1.update_layout(
            title="Latest screening status",
            height=390,
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=30
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Screening class",
            yaxis_title="Number of cells",
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with chart2:

        if not risk_df.empty:

            rz = risk_df.copy()

            rz["lat_band"] = pd.cut(
                rz["latitude"],
                bins=[
                    -40,
                    -20,
                    0,
                    20,
                    30
                ],
                labels=[
                    "-40 to -20",
                    "-20 to 0",
                    "0 to 20",
                    "20 to 30"
                ],
                include_lowest=True,
            )

            region_counts = (
                rz.groupby(
                    "lat_band",
                    observed=False
                )
                .size()
                .reset_index(
                    name="Flagged cells"
                )
            )

            fig2 = px.bar(
                region_counts,
                x="lat_band",
                y="Flagged cells",
                text="Flagged cells",
            )

            fig2.update_traces(
                marker_color="#38a66f",
                textposition="outside"
            )

            fig2.update_layout(
                title="Flagged cells by latitude band",
                height=390,
                margin=dict(
                    l=20,
                    r=20,
                    t=55,
                    b=30
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Latitude band",
                yaxis_title="Flagged cells",
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

        else:

            st.info(
                "No flagged cells are available for the hotspot summary."
            )

    if "chla" in latest.columns:

        chla = latest[
            "chla"
        ].dropna()

        if not chla.empty:

            q99 = chla.quantile(.99)

            clipped = chla.clip(
                upper=q99
            )

            fig3 = px.histogram(
                x=clipped,
                nbins=45,
                labels={
                    "x": "Chlorophyll-a",
                    "count": "Grid cells"
                },
            )

            fig3.update_traces(
                marker_color="#0f91a0"
            )

            fig3.update_layout(
                title=(
                    "Chlorophyll-a distribution, "
                    "clipped at the 99th percentile"
                ),
                height=360,
                margin=dict(
                    l=20,
                    r=20,
                    t=55,
                    b=30
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Chlorophyll-a",
                yaxis_title="Grid cells",
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )


# =========================================================
# HOW IT WORKS
# =========================================================

elif st.session_state.page == "How It Works":

    st.markdown(
        '<div class="section-kicker">PROJECT WORKFLOW</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">How BloomDetect AI works</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="section-copy">
    The project converts satellite-derived ocean observations
    into a structured screening output that can support further
    investigation.
</div>
""",
        unsafe_allow_html=True,
    )

    how_steps = [

        (
            "01",
            "Satellite data",
            "EOS-06 OCM-3 analysed chlorophyll-a observations "
            "are used as the main satellite-derived signal."
        ),

        (
            "02",
            "Historical context",
            "Previous observations at the same grid location "
            "provide temporal context and a historical baseline."
        ),

        (
            "03",
            "Feature engineering",
            "Current chlorophyll-a, previous value, historical "
            "baseline and recent temporal statistics are prepared "
            "for the model."
        ),

        (
            "04",
            "AI screening",
            "A trained Decision Tree classifier produces the "
            "current potential bloom-risk screening output."
        ),

        (
            "05",
            "Spatial interpretation",
            "The latest predictions are placed back on the "
            "geographic grid to reveal individual flags and "
            "concentrated zones."
        ),

        (
            "06",
            "Investigation support",
            "Users can inspect the map, check coordinates, review "
            "hotspots and download the output."
        ),
    ]

    for row_start in range(
        0,
        len(how_steps),
        3
    ):

        cols = st.columns(
            3,
            gap="large"
        )

        for col, item in zip(
            cols,
            how_steps[
                row_start:row_start + 3
            ]
        ):

            num, title, text_value = item

            with col:

                st.markdown(
                    f"""
<div class="info-card"
     style="min-height:220px;">

    <div class="flow-num">
        {num}
    </div>

    <div class="info-title"
         style="margin-top:12px;">

        {title}

    </div>

    <div class="info-copy">
        {text_value}
    </div>

</div>
""",
                    unsafe_allow_html=True,
                )

    st.markdown(
        """
<div class="caveat">
    <b>Important:</b>
    The present system is an early-warning / screening support
    system. It does not establish that a flagged location contains
    a harmful algal bloom, a particular species, or a toxin.
</div>
""",
        unsafe_allow_html=True,
    )


# =========================================================
# DATA
# =========================================================

elif st.session_state.page == "Data":

    st.markdown(
        '<div class="section-kicker">DATA & OUTPUT</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Project data at a glance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="section-copy">
    The dashboard currently displays the latest processed
    screening output used by the application.
</div>
""",
        unsafe_allow_html=True,
    )

    d1, d2, d3, d4 = st.columns(
        4,
        gap="medium"
    )

    metrics = [

        (
            f"{len(df):,}",
            "rows in latest-output CSV"
        ),

        (
            f"{latest_count:,}",
            "latest-date cells"
        ),

        (
            f"{risk_count:,}",
            "latest potential-risk flags"
        ),

        (
            "Decision Tree",
            "current screening model"
        ),
    ]

    for col, (value, label) in zip(
        [d1, d2, d3, d4],
        metrics
    ):

        with col:

            st.markdown(
                f"""
<div class="metric-card">

    <div class="metric-value">
        {value}
    </div>

    <div class="metric-label">
        {label}
    </div>

</div>
""",
                unsafe_allow_html=True,
            )

    st.write("")

    st.markdown(
        """
<div class="page-card">

    <h3 style="margin-top:0;">
        Dataset source
    </h3>

    <p>
        <b>EOS-06 / OCM-3</b> satellite-derived analysed
        chlorophyll-a product.
        The current project uses chlorophyll-a as an observable
        ocean-colour signal for potential bloom-risk screening.
    </p>

    <p>
        <b>Latest observation:</b>
        the date shown on the Home and Map pages.
        <br>

        <b>Spatial output:</b>
        latitude / longitude grid cells.
        <br>

        <b>Primary output:</b>
        Normal or Potential Bloom Risk.
    </p>

</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-kicker">DOWNLOADS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="section-title"
     style="font-size:2rem;">

    Take the screening output with you.

</div>
""",
        unsafe_allow_html=True,
    )

    summary_text = f"""
BloomDetect AI - Screening Summary

Latest observation date:
{
    latest_date.strftime("%Y-%m-%d")
    if pd.notna(latest_date)
    else "N/A"
}

Latest processed cells:
{latest_count}

Potential bloom-risk flags:
{risk_count}

Model:
Decision Tree

Satellite product:
EOS-06 OCM-3 analysed chlorophyll-a

Scientific note:
The output represents potential bloom-risk screening,
not confirmed harmful algal bloom ground truth.

High chlorophyll-a alone does not prove a harmful algal bloom.

Further environmental analysis and field/scientific
validation are required.
"""

    dl1, dl2 = st.columns(
        2,
        gap="large"
    )

    with dl1:

        st.markdown(
            """
<div class="download-card">

    <div class="info-title">
        📄 Latest screening CSV
    </div>

    <div class="info-copy">
        All processed grid-cell predictions in the latest
        output file.
    </div>

</div>
""",
            unsafe_allow_html=True,
        )

        st.download_button(
            "Download screening CSV",
            data=df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name=(
                "latest_bloom_risk_predictions.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    with dl2:

        st.markdown(
            """
<div class="download-card">

    <div class="info-title">
        📝 Project screening summary
    </div>

    <div class="info-copy">
        A small text summary of the current dataset,
        model and scientific boundary.
    </div>

</div>
""",
            unsafe_allow_html=True,
        )

        st.download_button(
            "Download project summary",
            data=summary_text.encode(
                "utf-8"
            ),
            file_name=(
                "BloomDetect_AI_Screening_Summary.txt"
            ),
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown(
        '<div class="section-kicker">LATEST RECORDS</div>',
        unsafe_allow_html=True
    )

    preview_cols = [
        c
        for c in [
            "date",
            "latitude",
            "longitude",
            "chla",
            "risk_label",
            "risk_probability",
        ]
        if c in latest.columns
    ]

    preview = latest[
        preview_cols
    ].head(100).copy()

    if "date" in preview.columns:

        preview["date"] = (
            preview["date"]
            .dt.strftime("%Y-%m-%d")
        )

    st.dataframe(
        preview,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">

    <b>BloomDetect AI</b>
    · AI-based potential harmful algal bloom screening
    and early-warning support.

    <br>

    Satellite screening is intended to narrow where further
    investigation may be useful. It does not replace field
    observations, laboratory testing or scientific confirmation.

</div>
""",
    unsafe_allow_html=True,
)
