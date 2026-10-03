from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# BLOOMDETECT AI
# STAGE 1
# Coastal & Ocean Intelligence Platform
# ============================================================

st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "latest_bloom_risk_predictions.csv"
PROCESS_IMAGE = BASE_DIR / "bloomdetect_bloom_process.png"
PHYTO_IMAGE = BASE_DIR / "phytoplankton_signal.png"
SATELLITE_IMAGE = BASE_DIR / "satellite_signal_illustration.png"


# ============================================================
# PROJECT SETTINGS
# ============================================================

ANOMALY_THRESHOLD = 0.059076173328496344
CHANGE_THRESHOLD = 0.02007450088858604


PAGES = [
    ("home", "Home"),
    ("map", "Risk Map"),
    ("hotspots", "Hotspots"),
    ("checker", "Location"),
    ("insights", "Insights"),
    ("explorer", "Satellite Explorer"),
    ("environment", "Environment"),
    ("fisheries", "Fisheries"),
    ("alerts", "Early Warning"),
    ("data", "Data & Research"),
]


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "latest_bloom_risk_predictions.csv was not found "
            "in the same folder as app.py."
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = {
        "latitude",
        "longitude",
        "date",
        "chla",
        "risk_label",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            "Missing required CSV columns: "
            + ", ".join(sorted(missing))
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    for column in [
        "latitude",
        "longitude",
        "chla",
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    numeric_columns = [
        "risk_probability",
        "model_score",
        "previous_chla",
        "historical_baseline",
        "recent_mean",
        "recent_max",
        "chla_anomaly",
        "chla_change",
    ]

    for column in numeric_columns:

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
            "chla",
        ]
    ).copy()

    data["risk_flag"] = (
        data["risk_label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("potential bloom risk")
    )

    data["plot_lon"] = (
        (data["longitude"] + 180) % 360
    ) - 180

    return data


try:

    df = load_data()

except Exception as error:

    st.error(
        "BloomDetect could not load the dataset."
    )

    st.code(str(error))

    st.stop()


latest_date = df["date"].max()

latest = df[
    df["date"] == latest_date
].copy()

risk = latest[
    latest["risk_flag"]
].copy()

normal = latest[
    ~latest["risk_flag"]
].copy()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def fmt_num(value, digits=3):

    if pd.isna(value):
        return "Unavailable"

    return f"{float(value):.{digits}f}"


def fmt_probability(value):

    if pd.isna(value):
        return "Unavailable"

    value = float(value)

    if value <= 1:
        value *= 100

    return f"{value:.1f}%"


def region_name(latitude, longitude):

    if (
        5 <= latitude <= 30
        and 45 <= longitude <= 75
    ):
        return "Arabian Sea"

    if (
        0 <= latitude <= 25
        and 75 < longitude <= 100
    ):
        return "Bay of Bengal"

    if (
        -30 <= latitude < 5
        and 40 <= longitude <= 100
    ):
        return "Southern Indian Ocean"

    if (
        -40 <= latitude <= 30
        and 20 <= longitude <= 120
    ):
        return "Indian Ocean region"

    return "Outside main study view"


def study_area(data):

    return data[
        data["latitude"].between(-40, 30)
        &
        data["plot_lon"].between(20, 120)
    ].copy()


def nearest_row(latitude, longitude):

    if latest.empty:
        return None

    latitude_difference = (
        latest["latitude"].to_numpy()
        - float(latitude)
    )

    longitude_difference = np.abs(
        latest["longitude"].to_numpy()
        - float(longitude)
    )

    longitude_difference = np.minimum(
        longitude_difference,
        360 - longitude_difference
    )

    distance = (
        latitude_difference ** 2
        +
        longitude_difference ** 2
    )

    index = int(
        np.argmin(distance)
    )

    return latest.iloc[index]


def go_page(page):

    st.session_state.page = page

    st.rerun()


def metric_card(
    label,
    value,
    note=""
):

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                {label}
            </div>

            <div class="metric-value">
                {value}
            </div>

            <div class="metric-note">
                {note}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


def section(
    kicker,
    title,
    description
):

    st.markdown(
        f"""
        <div class="section">

            <div class="kicker">
                {kicker}
            </div>

            <h1>
                {title}
            </h1>

            <p>
                {description}
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


def information_card(
    title,
    text,
    icon=""
):

    st.markdown(
        f"""
        <div class="card">

            <div class="card-icon">
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
        unsafe_allow_html=True,
    )


# ============================================================
# MAP FUNCTION
# ============================================================

def create_risk_map(
    show_risk=True
):

    data = study_area(latest)

    if data.empty:
        data = latest.copy()

    # --------------------------------------------------------
    # BLUE NORMAL / BASE SATELLITE FIELD
    # --------------------------------------------------------

    base = data.copy()

    base["lat_bin"] = (
        np.floor(base["latitude"])
        + 0.5
    )

    base["lon_bin"] = (
        np.floor(base["plot_lon"])
        + 0.5
    )

    blue = (
        base
        .groupby(
            [
                "lat_bin",
                "lon_bin"
            ],
            as_index=False
        )
        .agg(
            chla=("chla", "mean"),
            cells=("chla", "size")
        )
    )

    fig = px.scatter_geo(
        blue,

        lat="lat_bin",
        lon="lon_bin",

        color="chla",

        size="cells",

        size_max=13,

        color_continuous_scale=[
            [0.00, "#084b7a"],
            [0.30, "#087ca0"],
            [0.60, "#13a5b1"],
            [1.00, "#66d8d1"],
        ],

        custom_data=[
            "lat_bin",
            "lon_bin",
            "chla",
            "cells",
        ],

        projection="mercator",
    )

    fig.update_traces(

        marker=dict(
            opacity=0.78,
            line=dict(width=0),
        ),

        hovertemplate=(
            "Normal satellite field<br>"
            "Latitude: %{customdata[0]:.1f}°<br>"
            "Longitude: %{customdata[1]:.1f}°<br>"
            "Mean Chl-a: %{customdata[2]:.4f}<br>"
            "Cells: %{customdata[3]}<extra></extra>"
        ),

        name="Normal field",
    )

    # --------------------------------------------------------
    # GREEN RISK CONCENTRATIONS
    # --------------------------------------------------------

    if show_risk:

        risk_region = study_area(
            risk
        )

        if not risk_region.empty:

            green_data = risk_region.copy()

            green_data["lat_zone"] = (
                np.floor(
                    green_data["latitude"] / 2
                )
                * 2
                + 1
            )

            green_data["lon_zone"] = (
                np.floor(
                    green_data["plot_lon"] / 2
                )
                * 2
                + 1
            )

            green = (
                green_data
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
                    )
                )
            )

            fig.add_trace(

                go.Scattergeo(

                    lat=green["lat_zone"],

                    lon=green["lon_zone"],

                    mode="markers",

                    marker=dict(

                        size=np.clip(
                            green["flagged_cells"]
                            * 0.8
                            + 12,
                            14,
                            32,
                        ),

                        color=(
                            "rgba(45,190,105,0.36)"
                        ),

                        line=dict(
                            color=(
                                "rgba(27,135,78,0.75)"
                            ),
                            width=1,
                        ),
                    ),

                    name="Risk concentration",

                    hovertemplate=(
                        "Potential-risk concentration<br>"
                        "Latitude: %{lat:.1f}°<br>"
                        "Longitude: %{lon:.1f}°<extra></extra>"
                    ),
                )
            )

            # ------------------------------------------------
            # RED INDIVIDUAL RISK CELLS
            # ------------------------------------------------

            fig.add_trace(

                go.Scattergeo(

                    lat=risk_region[
                        "latitude"
                    ],

                    lon=risk_region[
                        "plot_lon"
                    ],

                    mode="markers",

                    marker=dict(
                        size=7,
                        color="#ed4f5f",
                        opacity=0.97,
                        line=dict(
                            color="white",
                            width=1,
                        ),
                    ),

                    name="Potential bloom risk",

                    customdata=risk_region[
                        [
                            "latitude",
                            "longitude",
                            "chla",
                        ]
                    ],

                    hovertemplate=(
                        "Potential bloom-risk flag<br>"
                        "Latitude: %{customdata[0]:.2f}°<br>"
                        "Longitude: %{customdata[1]:.2f}°<br>"
                        "Chl-a: %{customdata[2]:.4f}<extra></extra>"
                    ),
                )
            )

    # --------------------------------------------------------
    # MAP STYLE
    # --------------------------------------------------------

    fig.update_geos(

        showland=True,
        landcolor="#dcebea",

        showocean=True,
        oceancolor="#dff7f8",

        showcoastlines=True,
        coastlinecolor="#438995",
        coastlinewidth=1.0,

        showcountries=True,
        countrycolor="#9bbec2",

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

        projection_scale=1.08,
    )

    fig.update_layout(

        height=575,

        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0,
        ),

        paper_bgcolor="#dff7f8",

        plot_bgcolor="#dff7f8",

        font=dict(
            color="#174b56"
        ),

        legend=dict(

            orientation="h",

            yanchor="bottom",
            y=0.01,

            xanchor="left",
            x=0.02,

            bgcolor=(
                "rgba(255,255,255,.88)"
            ),

            bordercolor="#9fcfd4",
            borderwidth=1,
        ),

        coloraxis_colorbar=dict(
            title="Chl-a",
            thickness=13,
            len=.52,
        ),
    )

    return fig


# ============================================================
# CHART STYLE
# ============================================================

def chart_layout(
    figure,
    height=330
):

    figure.update_layout(

        height=height,

        margin=dict(
            l=58,
            r=20,
            t=32,
            b=62,
        ),

        paper_bgcolor=(
            "rgba(0,0,0,0)"
        ),

        plot_bgcolor="#ffffff",

        font=dict(
            color="#174b56"
        ),

        xaxis=dict(

            title_font=dict(
                size=13,
                color="#174b56",
            ),

            tickfont=dict(
                size=12,
                color="#174b56",
            ),

            gridcolor="#d6e7e8",

            automargin=True,
        ),

        yaxis=dict(

            title_font=dict(
                size=13,
                color="#174b56",
            ),

            tickfont=dict(
                size=12,
                color="#174b56",
            ),

            gridcolor="#d6e7e8",

            automargin=True,
        ),
    )

    return figure


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap'
);

:root{

    --ink:#103f49;

    --muted:#62828a;

    --deep:#07566a;

    --aqua:#0ca7b2;

    --line:#a9d6da;

    --pale:#effafa;

    --shadow:
        0 13px 34px
        rgba(9,80,94,.08);
}


html,
body,
[data-testid="stAppViewContainer"]{

    background:#f2fbfb!important;

    color:var(--ink)!important;

    font-family:
        'DM Sans',
        sans-serif!important;
}


.stApp{

    background:

        radial-gradient(
            circle at 12% 18%,
            rgba(16,174,183,.06),
            transparent 23%
        ),

        radial-gradient(
            circle at 88% 62%,
            rgba(28,191,199,.05),
            transparent 25%
        ),

        linear-gradient(
            180deg,
            #fbffff 0%,
            #effafa 50%,
            #fbffff 100%
        )!important;

    overflow-x:hidden;
}


[data-testid="stHeader"],
[data-testid="stToolbar"],
#MainMenu,
footer,
[data-testid="stSidebar"]{

    display:none!important;
}


.block-container{

    max-width:1280px!important;

    padding:
        20px
        32px
        70px!important;
}


/* ----------------------------------------------------------
   WATER ANIMATION
---------------------------------------------------------- */

.stApp:before{

    content:"";

    position:fixed;

    left:-10%;
    right:-10%;

    bottom:-160px;

    height:270px;

    z-index:0;

    pointer-events:none;

    background:

        radial-gradient(
            ellipse at 15% 55%,
            rgba(28,193,201,.13)
            0 17%,
            transparent 18%
        ),

        radial-gradient(
            ellipse at 55% 45%,
            rgba(69,221,218,.10)
            0 20%,
            transparent 21%
        ),

        radial-gradient(
            ellipse at 90% 60%,
            rgba(15,171,191,.10)
            0 18%,
            transparent 19%
        );

    animation:
        waterMove
        12s
        ease-in-out
        infinite
        alternate;
}


@keyframes waterMove{

    from{
        transform:translateX(-2%);
    }

    to{
        transform:translateX(2%);
    }
}


/* ----------------------------------------------------------
   BRAND
---------------------------------------------------------- */

.brand{

    position:relative;

    z-index:5;

    display:flex;

    align-items:center;

    justify-content:space-between;

    gap:20px;

    padding:
        15px
        19px;

    border:
        1.5px
        solid
        #c3e3e5;

    background:
        rgba(255,255,255,.90);

    border-radius:22px;

    box-shadow:var(--shadow);

    backdrop-filter:
        blur(15px);
}


.brand-left{

    display:flex;

    align-items:center;

    gap:13px;
}


.logo{

    width:47px;

    height:47px;

    border-radius:15px;

    display:grid;

    place-items:center;

    color:white;

    font-size:22px;

    font-weight:800;

    background:
        linear-gradient(
            145deg,
            #29cdd0,
            #087b8e
        );

    box-shadow:
        0 9px 22px
        rgba(9,144,157,.18);
}


.brand-name{

    font:
        800
        1.2rem
        Manrope,
        sans-serif;

    color:#103f49;

    letter-spacing:-.03em;
}


.brand-sub{

    color:#78959b;

    font-size:.73rem;

    margin-top:2px;
}


/* ----------------------------------------------------------
   BUTTONS
---------------------------------------------------------- */

.stButton>button,
.stDownloadButton>button,
.stFormSubmitButton>button{

    min-height:45px!important;

    border-radius:13px!important;

    background:#ffffff!important;

    border:
        2px
        solid
        #164e5b!important;

    color:#123f49!important;

    font-weight:800!important;

    font-size:.88rem!important;

    box-shadow:
        0 5px 15px
        rgba(15,70,82,.08)!important;

    transition:
        all
        .18s
        ease!important;
}


.stButton>button:hover,
.stDownloadButton>button:hover,
.stFormSubmitButton>button:hover{

    background:#e1f8f8!important;

    border-color:#087d8c!important;

    transform:
        translateY(-1px)!important;
}


.stButton>button[kind="primary"]{

    background:
        linear-gradient(
            135deg,
            #07576a,
            #0b8b99
        )!important;

    border-color:#063d4b!important;

    color:#fff!important;
}


.stButton>button[kind="primary"] p,
.stButton>button[kind="primary"] span{

    color:#fff!important;
}


.nav-row{

    position:relative;

    z-index:6;

    margin:
        12px
        0
        5px;
}


/* ----------------------------------------------------------
   SECTION
---------------------------------------------------------- */

.section{

    position:relative;

    z-index:2;

    padding:
        35px
        0
        17px;
}


.kicker{

    color:#0797a5;

    font:
        800
        .68rem
        Manrope,
        sans-serif;

    letter-spacing:.18em;

    text-transform:uppercase;

    margin-bottom:10px;
}


.section h1{

    font:
        800
        clamp(
            2.15rem,
            4vw,
            3.8rem
        )
        /
        1.03
        Manrope,
        sans-serif;

    letter-spacing:-.06em;

    color:#103f49;

    margin:
        0
        0
        12px;
}


.section p{

    color:#5f8088;

    line-height:1.65;

    margin:0;

    max-width:1120px;

    font-size:.98rem;
}


/* ----------------------------------------------------------
   HERO
---------------------------------------------------------- */

.hero{

    position:relative;

    z-index:2;

    overflow:hidden;

    min-height:470px;

    border-radius:30px;

    padding:
        55px;

    background:

        linear-gradient(
            90deg,
            rgba(3,48,64,.94),
            rgba(4,105,121,.82)
        ),

        url(
            "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1800&q=85"
        );

    background-size:cover;

    background-position:center;

    box-shadow:
        0 25px 65px
        rgba(6,86,100,.16);
}


.hero:after{

    content:"";

    position:absolute;

    left:-5%;
    right:-5%;

    bottom:-135px;

    height:260px;

    background:
        rgba(142,244,237,.12);

    border-radius:50%;

    animation:
        heroWave
        8s
        ease-in-out
        infinite
        alternate;
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


.hero-content{

    position:relative;

    z-index:2;

    max-width:850px;
}


.hero .kicker{

    color:#a6fffa;
}


.hero h1{

    color:#e8ffff;

    font:
        800
        clamp(
            3rem,
            6vw,
            5.7rem
        )
        /
        .92
        Manrope,
        sans-serif;

    letter-spacing:-.075em;

    margin:
        0
        0
        20px;
}


.hero p{

    color:#e1fbfb;

    line-height:1.75;

    font-size:1.06rem;

    max-width:790px;
}


.badges{

    display:flex;

    gap:8px;

    flex-wrap:wrap;

    margin-top:22px;
}


.badge{

    padding:
        8px
        12px;

    border-radius:999px;

    background:
        rgba(255,255,255,.13);

    border:
        1px
        solid
        rgba(255,255,255,.27);

    color:#efffff;

    font-size:.78rem;

    font-weight:700;
}


/* ----------------------------------------------------------
   CARDS
---------------------------------------------------------- */

.card{

    position:relative;

    z-index:2;

    height:100%;

    box-sizing:border-box;

    background:
        rgba(255,255,255,.90);

    border:
        1.5px
        solid
        #abd7da;

    border-radius:20px;

    box-shadow:var(--shadow);

    padding:21px;
}


.card-icon{

    font-size:1.35rem;

    margin-bottom:7px;
}


.card h3{

    font:
        800
        1.16rem
        Manrope,
        sans-serif;

    color:#123f49;

    margin:
        0
        0
        7px;
}


.card p{

    color:#66848b;

    line-height:1.58;

    margin:0;

    font-size:.90rem;
}


/* ----------------------------------------------------------
   METRICS
---------------------------------------------------------- */

.metric-card{

    position:relative;

    z-index:2;

    min-height:116px;

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
        1.5px
        solid
        #a6d5da;

    border-radius:18px;

    box-shadow:
        0 10px 27px
        rgba(15,91,101,.07);
}


.metric-label{

    color:#6d8c93;

    font:
        800
        .63rem
        Manrope,
        sans-serif;

    letter-spacing:.12em;

    text-transform:uppercase;
}


.metric-value{

    color:#123f49;

    font:
        800
        1.45rem
        Manrope,
        sans-serif;

    margin-top:7px;

    white-space:nowrap;
}


.metric-note{

    color:#78959b;

    font-size:.72rem;

    margin-top:4px;
}


/* ----------------------------------------------------------
   FLOW
---------------------------------------------------------- */

.flow{

    display:grid;

    grid-template-columns:
        repeat(4,1fr);

    gap:13px;
}


.flow-step{

    padding:18px;

    border:
        2px
        solid
        #174e5b;

    border-radius:17px;

    background:
        linear-gradient(
            135deg,
            #ffffff,
            #eaf8f8
        );

    min-height:125px;
}


.flow-step b{

    color:#0b98a5;

    font:
        800
        .72rem
        Manrope;

    letter-spacing:.13em;
}


.flow-step h3{

    color:#123f49;

    font:
        800
        1rem
        Manrope;

    margin:
        8px
        0
        5px;
}


.flow-step p{

    color:#66848b;

    font-size:.82rem;

    line-height:1.45;

    margin:0;
}


/* ----------------------------------------------------------
   MAP
---------------------------------------------------------- */

.map-card{

    position:relative;

    z-index:2;

    background:#eefafa;

    border:
        2px
        solid
        #8fcbd1;

    border-radius:22px;

    padding:5px;

    overflow:hidden;

    box-shadow:
        0 15px 40px
        rgba(15,91,101,.09);
}


.map-head{

    padding:
        10px
        12px;

    display:flex;

    align-items:center;

    justify-content:space-between;

    gap:15px;
}


.map-head small{

    color:#0797a5;

    font-weight:800;

    letter-spacing:.13em;
}


.map-head b{

    color:#174b56;

    font-size:.92rem;
}


/* ----------------------------------------------------------
   LOCATION
---------------------------------------------------------- */

.input-panel{

    position:relative;

    z-index:2;

    padding:22px;

    background:
        rgba(255,255,255,.90);

    border:
        1.5px
        solid
        #abd7da;

    border-radius:20px;

    box-shadow:var(--shadow);
}


.input-title{

    color:#6d8c93;

    font:
        800
        .68rem
        Manrope;

    letter-spacing:.14em;

    text-transform:uppercase;

    margin-bottom:8px;
}


.input-value{

    color:#123f49;

    font:
        800
        1.75rem
        Manrope;

    margin-bottom:7px;
}


.input-copy{

    color:#66848b;

    line-height:1.58;

    font-size:.88rem;
}


.result-box{

    position:relative;

    z-index:2;

    padding:22px;

    background:
        rgba(255,255,255,.92);

    border:
        1.5px
        solid
        #abd7da;

    border-radius:20px;

    box-shadow:var(--shadow);
}


.result-title{

    color:#6d8c93;

    font:
        800
        .68rem
        Manrope;

    letter-spacing:.14em;

    text-transform:uppercase;
}


.result-coordinate{

    color:#123f49;

    font:
        800
        1.65rem
        Manrope;

    margin:
        6px
        0
        12px;
}


.result-grid{

    display:grid;

    grid-template-columns:
        repeat(2,1fr);

    gap:9px;
}


.result-item{

    padding:12px;

    border-radius:13px;

    background:#f4fbfb;

    border:
        1px
        solid
        #c8e5e7;
}


.result-item span{

    display:block;

    color:#78959b;

    font-size:.68rem;

    margin-bottom:4px;
}


.result-item b{

    color:#174b56;

    font-size:.88rem;
}


.status{

    margin-top:13px;

    padding:
        13px
        15px;

    border-radius:14px;

    border:2px solid;

    font-size:.84rem;

    line-height:1.5;
}


.status.yes{

    background:#fff0f2;

    border-color:#ef6473;

    color:#a42e3d;
}


.status.no{

    background:#eafaf4;

    border-color:#49b995;

    color:#176f58;
}


.signal-grid{

    display:grid;

    grid-template-columns:
        repeat(4,1fr);

    gap:10px;

    margin-top:14px;
}


.signal{

    padding:12px;

    border-radius:13px;

    background:#f7fcfc;

    border:
        1px
        solid
        #c9e5e7;
}


.signal span{

    color:#78959b;

    font-size:.68rem;
}


.signal b{

    display:block;

    color:#164a55;

    font:
        800
        .93rem
        Manrope;

    margin-top:4px;
}


/* ----------------------------------------------------------
   CHARTS
---------------------------------------------------------- */

.chart-card{

    position:relative;

    z-index:2;

    padding:
        17px
        18px
        5px;

    border-radius:19px;

    background:
        rgba(255,255,255,.90);

    border:
        1.5px
        solid
        #abd7da;

    box-shadow:var(--shadow);
}


.chart-card h3{

    color:#123f49;

    font:
        800
        1.14rem
        Manrope;

    margin:
        2px
        0
        5px;
}


.chart-card p{

    color:#66848b;

    font-size:.82rem;

    margin:0;
}


/* ----------------------------------------------------------
   METHOD / STEPS
---------------------------------------------------------- */

.step{

    position:relative;

    z-index:2;

    display:grid;

    grid-template-columns:
        50px
        1fr;

    gap:14px;

    align-items:center;

    padding:
        15px
        17px;

    margin:
        9px
        0;

    border:
        2px
        solid
        #174e5b;

    border-radius:17px;

    background:
        linear-gradient(
            100deg,
            rgba(255,255,255,.96),
            rgba(232,249,249,.90)
        );

    box-shadow:
        0 8px 21px
        rgba(15,91,101,.06);
}


.step-number{

    width:42px;

    height:42px;

    border-radius:12px;

    display:grid;

    place-items:center;

    background:#0d5264;

    color:#fff;

    border:
        2px
        solid
        #082f3b;

    font:
        800
        .84rem
        Manrope;
}


.step h3{

    color:#123f49;

    font:
        800
        1.06rem
        Manrope;

    margin:
        0
        0
        4px;
}


.step p{

    color:#66848b;

    line-height:1.48;

    font-size:.88rem;

    margin:0;
}


/* ----------------------------------------------------------
   DOWNLOADS
---------------------------------------------------------- */

.download-card{

    min-height:135px;

    box-sizing:border-box;

    padding:20px;

    border-radius:19px;

    background:
        linear-gradient(
            135deg,
            #087b8b,
            #13adb3
        );

    border:
        2px
        solid
        #07576a;

    box-shadow:
        0 13px 28px
        rgba(8,91,102,.16);

    color:white;
}


.download-card h3{

    color:white;

    font:
        800
        1.13rem
        Manrope;

    margin:
        0
        0
        6px;
}


.download-card p{

    color:#e5ffff;

    font-size:.82rem;

    line-height:1.45;

    margin:0;
}


/* ----------------------------------------------------------
   NOTE
---------------------------------------------------------- */

.note{

    position:relative;

    z-index:2;

    margin-top:16px;

    padding:
        13px
        16px;

    background:#e6f8f8;

    border-left:
        4px
        solid
        #11a9b2;

    border-radius:
        0
        14px
        14px
        0;

    color:#52757c;

    font-size:.86rem;

    line-height:1.58;
}


/* ----------------------------------------------------------
   FOOTER
---------------------------------------------------------- */

.footer{

    position:relative;

    z-index:2;

    border-top:
        1px
        solid
        #d5ebed;

    margin-top:40px;

    padding-top:17px;

    color:#76959b;

    font-size:.71rem;
}


/* ----------------------------------------------------------
   RESPONSIVE
---------------------------------------------------------- */

@media(max-width:950px){

    .flow{

        grid-template-columns:
            1fr
            1fr;
    }

    .checker-grid{

        grid-template-columns:
            1fr;
    }

}


@media(max-width:700px){

    .block-container{

        padding:
            15px
            12px
            50px!important;
    }

    .flow{

        grid-template-columns:
            1fr;
    }

    .signal-grid{

        grid-template-columns:
            1fr
            1fr;
    }

    .result-grid{

        grid-template-columns:
            1fr;
    }

    .hero{

        padding:
            42px
            28px;
    }

    .hero h1{

        font-size:3rem;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# INITIAL PAGE
# ============================================================

if "page" not in st.session_state:

    st.session_state.page = "home"


# ============================================================
# BRAND
# ============================================================

st.markdown(
    f"""
    <div class="brand">

        <div class="brand-left">

            <div class="logo">
                ≈
            </div>

            <div>

                <div class="brand-name">
                    BloomDetect AI
                </div>

                <div class="brand-sub">
                    Coastal & Ocean Intelligence Platform · EOS-06 OCM-3
                </div>

            </div>

        </div>

        <div class="brand-sub">
            Data through {latest_date.strftime("%d %b %Y")}
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NAVIGATION
# ============================================================

for navigation_row in [
    PAGES[:5],
    PAGES[5:],
]:

    st.markdown(
        '<div class="nav-row">',
        unsafe_allow_html=True,
    )

    columns = st.columns(
        5,
        gap="small"
    )

    for column, (
        page_key,
        page_name
    ) in zip(
        columns,
        navigation_row
    ):

        with column:

            if st.button(
                page_name,

                key=f"nav_{page_key}",

                use_container_width=True,

                type=(
                    "primary"
                    if st.session_state.page
                    == page_key
                    else "secondary"
                ),
            ):

                go_page(page_key)

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


page = st.session_state.page


# ============================================================
# HOME
# ============================================================

if page == "home":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-content">

                <div class="kicker">
                    EOS-06 · OCM-3 · STAGE 1
                </div>

                <h1>
                    See the ocean differently.
                </h1>

                <p>
                    BloomDetect AI turns satellite-derived chlorophyll-a
                    observations into a practical coastal and ocean
                    screening workspace.
                    Explore the latest field, locate potential bloom-risk
                    signals, inspect coordinates and organize areas that
                    deserve closer observation.
                </p>

                <div class="badges">

                    <span class="badge">
                        🌊 Ocean intelligence
                    </span>

                    <span class="badge">
                        🛰️ Satellite screening
                    </span>

                    <span class="badge">
                        📍 Location intelligence
                    </span>

                    <span class="badge">
                        🚨 Early-warning support
                    </span>

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="height:18px"></div>',
        unsafe_allow_html=True,
    )

    image_col, information_col = st.columns(
        [1.15, .85],
        gap="large",
    )

    with image_col:

        if PROCESS_IMAGE.exists():

            st.image(
                str(PROCESS_IMAGE),
                use_container_width=True,
            )

        elif PHYTO_IMAGE.exists():

            st.image(
                str(PHYTO_IMAGE),
                use_container_width=True,
            )

        else:

            st.markdown(
                """
                <div class="card"
                     style="
                     min-height:320px;
                     display:grid;
                     place-items:center;
                     text-align:center;
                     ">

                    <div>

                        <div style="
                            font-size:4rem;
                        ">
                            🌊
                        </div>

                        <h3>
                            Ocean observation workspace
                        </h3>

                        <p>
                            EOS-06 OCM-3 chlorophyll-a screening.
                        </p>

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    with information_col:

        information_card(
            "What is BloomDetect?",
            "A satellite-based screening platform using chlorophyll-a and temporal context to identify locations that may deserve additional observation.",
            "🌱",
        )

        st.markdown(
            '<div style="height:12px"></div>',
            unsafe_allow_html=True,
        )

        information_card(
            "What can Stage 1 support?",
            "Bloom-risk screening, hotspot discovery, coordinate inspection, satellite-signal exploration, environmental indicators, fisheries productivity context and early-warning workflows.",
            "🛰️",
        )

        st.markdown(
            '<div style="height:12px"></div>',
            unsafe_allow_html=True,
        )

        information_card(
            "Scientific boundary",
            "A potential-risk flag is not proof of a harmful algal bloom. Chlorophyll-a alone cannot identify harmful species or toxins. Field validation and additional environmental evidence remain important.",
            "🔬",
        )

    st.markdown(
        '<div style="height:24px"></div>',
        unsafe_allow_html=True,
    )

    section(
        "01 · Platform workflow",
        "One dataset. Multiple useful views.",
        "Stage 1 extracts practical value from the processed EOS-06 OCM-3 chlorophyll-a dataset without inventing unavailable variables.",
    )

    workflow = [
        (
            "01",
            "OBSERVE",
            "Inspect the latest satellite field."
        ),
        (
            "02",
            "LOCATE",
            "Find potential-risk cells and concentrations."
        ),
        (
            "03",
            "EXPLORE",
            "Inspect chlorophyll and environmental indicators."
        ),
        (
            "04",
            "ACT",
            "Create a shortlist for further observation."
        ),
    ]

    st.markdown(
        '<div class="flow">',
        unsafe_allow_html=True,
    )

    for number, title, text in workflow:

        st.markdown(
            f"""
            <div class="flow-step">

                <b>
                    {number}
                </b>

                <h3>
                    {title}
                </h3>

                <p>
                    {text}
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="height:20px"></div>',
        unsafe_allow_html=True,
    )

    action_columns = st.columns(
        5,
        gap="small"
    )

    actions = [
        ("🌍 Risk Map", "map"),
        ("🔥 Hotspots", "hotspots"),
        ("📍 Location", "checker"),
        ("🛰️ Explorer", "explorer"),
        ("🚨 Early Warning", "alerts"),
    ]

    for column, (
        label,
        target
    ) in zip(
        action_columns,
        actions
    ):

        with column:

            if st.button(
                label,
                key=f"home_{target}",
                use_container_width=True,
            ):

                go_page(target)

    st.markdown(
        '<div style="height:22px"></div>',
        unsafe_allow_html=True,
    )

    home_metrics = [

        (
            "Latest cells",
            f"{len(latest):,}",
            "processed grid cells",
        ),

        (
            "Potential-risk cells",
            f"{len(risk):,}",
            "latest screening",
        ),

        (
            "Risk share",
            f"{100 * len(risk) / len(latest):.2f}%",
            "latest field",
        ),

        (
            "Grid",
            "0.25°",
            "source product",
        ),

    ]

    metric_columns = st.columns(
        4,
        gap="medium"
    )

    for column, values in zip(
        metric_columns,
        home_metrics
    ):

        with column:

            metric_card(*values)

    st.markdown(
        """
        <div class="note">

            <b>Important:</b>
            BloomDetect is an early-warning support and screening platform.
            It is not a real-time monitoring system and does not independently
            confirm a harmful algal bloom.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RISK MAP
# ============================================================

elif page == "map":

    section(
        "02 · Ocean & bloom risk",
        "Potential bloom-risk map",
        "Blue shows the latest processed chlorophyll-a field. Green marks concentrations of flagged cells and red marks individual potential-risk screening cells.",
    )

    map_left, map_right = st.columns(
        [1.45, .55],
        gap="medium",
    )

    with map_left:

        show_risk = st.toggle(
            "Show risk concentrations and flagged cells",
            value=True,
            key="show_risk_map",
        )

    with map_right:

        metric_card(
            "Study area",
            "20°–120°E",
            "40°S–30°N",
        )

    st.markdown(
        '<div class="map-card">',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="map-head">

            <div>

                <small>
                    STUDY REGION
                </small>

                <br>

                <b>
                    Indian Ocean · Arabian Sea · Bay of Bengal
                </b>

            </div>

            <div style="text-align:right">

                <small>
                    LATEST OBSERVATION
                </small>

                <br>

                <b>
                    Current processed field
                </b>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        create_risk_map(show_risk),
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": False,
            "modeBarButtonsToRemove": [
                "lasso2d",
                "select2d",
            ],
        },
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="note">

            <b>Map legend:</b>

            🔵 Blue = latest satellite field

            &nbsp;&nbsp;

            🟢 Green = potential-risk concentration

            &nbsp;&nbsp;

            🔴 Red = individual potential-risk screening cell

            <br>

            A red cell is a screening signal, not confirmed HAB evidence.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HOTSPOTS
# ============================================================

elif page == "hotspots":

    section(
        "03 · Hotspot intelligence",
        "Where do potential-risk cells concentrate?",
        "Nearby flagged cells are grouped into 2° × 2° investigation zones. These are spatial concentrations, not rankings of confirmed bloom severity.",
    )

    study = study_area(latest)

    risk_study = study[
        study["risk_flag"]
    ].copy()

    if risk_study.empty:

        st.markdown(
            """
            <div class="card">

                <h3>
                    No potential-risk cells in the study area.
                </h3>

                <p>
                    The latest processed field contains no screening
                    flags in the current study view.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        risk_study["lat_zone"] = (
            np.floor(
                risk_study["latitude"] / 2
            )
            * 2
            + 1
        )

        risk_study["lon_zone"] = (
            np.floor(
                risk_study["plot_lon"] / 2
            )
            * 2
            + 1
        )

        zones = (

            risk_study
            .groupby(
                [
                    "lat_zone",
                    "lon_zone",
                ],
                as_index=False,
            )

            .agg(

                flagged_cells=(
                    "risk_flag",
                    "size",
                ),

                mean_chla=(
                    "chla",
                    "mean",
                ),

                max_chla=(
                    "chla",
                    "max",
                ),
            )

            .sort_values(
                [
                    "flagged_cells",
                    "mean_chla",
                ],
                ascending=False,
            )

            .reset_index(drop=True)
        )

        hotspot_metrics = [

            (
                "Flagged cells",
                f"{len(risk_study):,}",
                "study area",
            ),

            (
                "Hotspot zones",
                f"{len(zones):,}",
                "2° × 2° grouping",
            ),

            (
                "Largest concentration",
                f"{int(zones.iloc[0]['flagged_cells']):,}",
                "flagged cells",
            ),

            (
                "Mean Chl-a",
                fmt_num(
                    risk_study["chla"].mean(),
                    4,
                ),
                "flagged cells",
            ),

        ]

        columns = st.columns(
            4,
            gap="medium"
        )

        for column, values in zip(
            columns,
            hotspot_metrics
        ):

            with column:

                metric_card(*values)

        hotspot_fig = px.scatter_geo(

            zones,

            lat="lat_zone",

            lon="lon_zone",

            size="flagged_cells",

            color="flagged_cells",

            size_max=40,

            color_continuous_scale=[

                [0, "#a9ead0"],

                [.5, "#48bf86"],

                [1, "#d63f51"],

            ],

            custom_data=[
                "lat_zone",
                "lon_zone",
                "flagged_cells",
                "mean_chla",
            ],

            projection="mercator",
        )

        hotspot_fig.update_traces(

            hovertemplate=(
                "Investigation concentration<br>"
                "Latitude: %{customdata[0]:.1f}°<br>"
                "Longitude: %{customdata[1]:.1f}°<br>"
                "Flagged cells: %{customdata[2]}<br>"
                "Mean Chl-a: %{customdata[3]:.4f}"
                "<extra></extra>"
            )
        )

        hotspot_fig.update_geos(

            showland=True,
            landcolor="#dcebea",

            showocean=True,
            oceancolor="#dff7f8",

            showcoastlines=True,
            coastlinecolor="#438995",

            showcountries=True,
            countrycolor="#9bbec2",

            bgcolor="#dff7f8",

            lataxis_range=[
                -40,
                30,
            ],

            lonaxis_range=[
                20,
                120,
            ],

            center=dict(
                lat=-5,
                lon=70,
            ),

            projection_scale=1.08,
        )

        hotspot_fig.update_layout(

            height=520,

            margin=dict(
                l=0,
                r=0,
                t=0,
                b=0,
            ),

            paper_bgcolor="#dff7f8",

            plot_bgcolor="#dff7f8",

            font=dict(
                color="#174b56"
            ),

            coloraxis_colorbar=dict(
                title="Flagged cells",
                thickness=13,
                len=.5,
            ),
        )

        st.markdown(
            '<div class="chart-card">',
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            hotspot_fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        table = zones.copy()

        table["Zone"] = table.apply(
            lambda row:
            f"{row.lat_zone:.1f}°, "
            f"{row.lon_zone:.1f}°",
            axis=1,
        )

        table["Mean Chl-a"] = (
            table["mean_chla"]
            .round(4)
        )

        table["Max Chl-a"] = (
            table["max_chla"]
            .round(4)
        )

        table = table[
            [
                "Zone",
                "flagged_cells",
                "Mean Chl-a",
                "Max Chl-a",
            ]
        ].rename(
            columns={
                "flagged_cells":
                    "Potential-risk cells"
            }
        )

        st.markdown(
            """
            <div class="section"
                 style="padding-top:25px">

                <div class="kicker">
                    Investigation list
                </div>

                <h1 style="font-size:2rem">
                    Concentrated areas
                </h1>

                <p>
                    Use these zones to decide where to inspect
                    individual coordinates next.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(
            table.head(15),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇ Download hotspot investigation list",

            zones.to_csv(
                index=False
            ).encode("utf-8"),

            "bloomdetect_hotspot_zones.csv",

            "text/csv",

            use_container_width=True,
        )


# ============================================================
# LOCATION
# ============================================================

elif page == "checker":

    section(
        "04 · Location intelligence",
        "Check a coordinate",
        "Enter any coordinate. BloomDetect preserves your exact input and separately reports the nearest processed satellite grid cell.",
    )

    if "input_lat" not in st.session_state:
        st.session_state.input_lat = 18.00

    if "input_lon" not in st.session_state:
        st.session_state.input_lon = 78.00

    if "checked_lat" not in st.session_state:
        st.session_state.checked_lat = 18.00

    if "checked_lon" not in st.session_state:
        st.session_state.checked_lon = 78.00

    with st.form(
        "coordinate_form",
        clear_on_submit=False,
    ):

        c1, c2, c3 = st.columns(
            [1, 1, .75],
            gap="medium",
        )

        with c1:

            latitude = st.number_input(
                "Latitude",

                min_value=-90.0,

                max_value=90.0,

                step=.25,

                format="%.2f",

                key="input_lat",
            )

        with c2:

            longitude = st.number_input(
                "Longitude",

                min_value=-180.0,

                max_value=180.0,

                step=.25,

                format="%.2f",

                key="input_lon",
            )

        with c3:

            st.markdown(
                "<div style='height:29px'></div>",
                unsafe_allow_html=True,
            )

            submitted = st.form_submit_button(
                "🔎 Check location",

                use_container_width=True,

                type="primary",
            )

    if submitted:

        st.session_state.checked_lat = (
            float(latitude)
        )

        st.session_state.checked_lon = (
            float(longitude)
        )

    checked_lat = float(
        st.session_state.checked_lat
    )

    checked_lon = float(
        st.session_state.checked_lon
    )

    row = nearest_row(
        checked_lat,
        checked_lon,
    )

    if row is None:

        st.error(
            "No processed observations are available."
        )

        st.stop()

    is_risk = bool(
        row["risk_flag"]
    )

    status_class = (
        "yes"
        if is_risk
        else "no"
    )

    status_title = (

        "🔴 POTENTIAL BLOOM-RISK FLAG"

        if is_risk

        else

        "🟢 NO POTENTIAL-RISK FLAG"
    )

    status_text = (

        "The nearest processed cell is included in the current potential-risk screening shortlist."

        if is_risk

        else

        "The nearest processed cell is not included in the current potential-risk screening shortlist."
    )

    left, right = st.columns(
        [.85, 1.15],
        gap="large",
    )

    with left:

        st.markdown(
            f"""
            <div class="input-panel">

                <div class="input-title">
                    YOUR INPUT
                </div>

                <div class="input-value">
                    {checked_lat:.2f}°
                    ·
                    {checked_lon:.2f}°
                </div>

                <div class="input-copy">
                    This is the coordinate you entered.
                    It is not silently replaced by the
                    satellite grid location.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            f"""
            <div class="result-box">

                <div class="result-title">
                    NEAREST PROCESSED OBSERVATION
                </div>

                <div class="result-coordinate">
                    {row.latitude:.2f}°
                    ·
                    {row.longitude:.2f}°
                </div>

                <div class="result-grid">

                    <div class="result-item">

                        <span>
                            Observation date
                        </span>

                        <b>
                            {row.date.strftime("%d %B %Y")}
                        </b>

                    </div>

                    <div class="result-item">

                        <span>
                            Chlorophyll-a
                        </span>

                        <b>
                            {fmt_num(row.chla,4)}
                        </b>

                    </div>

                    <div class="result-item">

                        <span>
                            Model probability
                        </span>

                        <b>
                            {fmt_probability(
                                row.get(
                                    "risk_probability",
                                    np.nan
                                )
                            )}
                        </b>

                    </div>

                    <div class="result-item">

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

                <div class="status {status_class}">

                    <b>
                        {status_title}
                    </b>

                    <br>

                    {status_text}

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="kicker" style="margin-top:22px">'
        'Screening signals'
        '</div>',
        unsafe_allow_html=True,
    )

    signals = [

        (
            "Current Chl-a",
            fmt_num(row.chla),
        ),

        (
            "Historical baseline",
            fmt_num(
                row.get(
                    "historical_baseline",
                    np.nan
                )
            ),
        ),

        (
            "Chl-a anomaly",
            fmt_num(
                row.get(
                    "chla_anomaly",
                    np.nan
                )
            ),
        ),

        (
            "Recent change",
            fmt_num(
                row.get(
                    "chla_change",
                    np.nan
                )
            ),
        ),

    ]

    signal_columns = st.columns(
        4,
        gap="small",
    )

    for column, (
        label,
        value
    ) in zip(
        signal_columns,
        signals
    ):

        with column:

            st.markdown(
                f"""
                <div class="signal">

                    <span>
                        {label}
                    </span>

                    <b>
                        {value}
                    </b>

                </div>
                """,
                unsafe_allow_html=True,
            )

    if is_risk:

        reason = (
            "The nearest cell meets the project's "
            "proxy screening conditions for anomaly "
            "and recent chlorophyll-a change."
        )

    else:

        reason = (
            "The nearest cell does not meet both "
            "proxy conditions used to create the "
            "current potential-risk label."
        )

    st.markdown(
        f"""
        <div class="note">

            <b>
                Screening explanation:
            </b>

            {reason}

            This does not confirm a harmful algal bloom,
            species or toxin.

        </div>
        """,
        unsafe_allow_html=True,
    )

    location_report = pd.DataFrame(
        [
            {

                "input_latitude":
                    checked_lat,

                "input_longitude":
                    checked_lon,

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
                        np.nan,
                    ),

                "chla_anomaly":
                    row.get(
                        "chla_anomaly",
                        np.nan,
                    ),

                "chla_change":
                    row.get(
                        "chla_change",
                        np.nan,
                    ),

                "risk_label":
                    row.risk_label,

                "risk_probability":
                    row.get(
                        "risk_probability",
                        np.nan,
                    ),
            }
        ]
    )

    st.download_button(

        "⬇ Download this location report",

        location_report
        .to_csv(
            index=False
        )
        .encode("utf-8"),

        "bloomdetect_location_report.csv",

        "text/csv",

        use_container_width=True,
    )


# ============================================================
# INSIGHTS
# ============================================================

elif page == "insights":

    section(
        "05 · Ocean insights",
        "What stands out in the latest field?",
        "A compact analytical view of the current processed observation: screening composition, chlorophyll distribution, regional context and flagged coordinates.",
    )

    total = len(latest)

    risk_count = len(risk)

    normal_count = len(normal)

    risk_share = (
        100 * risk_count / total
        if total
        else 0
    )

    insight_metrics = [

        (
            "Observation cells",
            f"{total:,}",
            "latest field",
        ),

        (
            "Potential-risk cells",
            f"{risk_count:,}",
            "screening output",
        ),

        (
            "Normal cells",
            f"{normal_count:,}",
            "latest field",
        ),

        (
            "Risk share",
            f"{risk_share:.2f}%",
            "latest field",
        ),

    ]

    columns = st.columns(
        4,
        gap="medium"
    )

    for column, values in zip(
        columns,
        insight_metrics
    ):

        with column:

            metric_card(*values)

    st.markdown(
        '<div style="height:18px"></div>',
        unsafe_allow_html=True,
    )

    chart_left, chart_right = st.columns(
        2,
        gap="large",
    )

    with chart_left:

        st.markdown(
            """
            <div class="chart-card">

                <h3>
                    Normal vs potential bloom-risk
                </h3>

                <p>
                    Current screening composition.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        composition = pd.DataFrame(
            {
                "Classification": [
                    "Normal",
                    "Potential bloom risk",
                ],

                "Cells": [
                    normal_count,
                    risk_count,
                ],
            }
        )

        figure = px.bar(
            composition,

            x="Classification",

            y="Cells",

            text="Cells",

            color="Classification",

            color_discrete_map={
                "Normal":
                    "#3e9daf",

                "Potential bloom risk":
                    "#ef5362",
            },
        )

        figure.update_traces(
            textposition="outside",
            cliponaxis=False,
        )

        figure.update_layout(
            showlegend=False,

            xaxis_title=
                "Screening outcome",

            yaxis_title=
                "Number of cells",
        )

        chart_layout(
            figure
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
            config={
                "displaylogo": False
            },
        )

    with chart_right:

        st.markdown(
            """
            <div class="chart-card">

                <h3>
                    Chlorophyll-a distribution
                </h3>

                <p>
                    Distribution of the latest satellite-derived field.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        figure = px.histogram(
            latest,

            x="chla",

            nbins=30,

            color_discrete_sequence=[
                "#159eaa"
            ],
        )

        figure.update_layout(
            xaxis_title=
                "Chlorophyll-a",

            yaxis_title=
                "Number of cells",
        )

        chart_layout(
            figure
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
            config={
                "displaylogo": False
            },
        )

    # --------------------------------------------------------
    # REGIONAL VIEW
    # --------------------------------------------------------

    regional = []

    regional_definitions = [

        (
            "Arabian Sea",

            latest["latitude"].between(
                5,
                30
            )
            &
            latest["plot_lon"].between(
                45,
                75
            ),
        ),

        (
            "Bay of Bengal",

            latest["latitude"].between(
                0,
                25
            )
            &
            latest["plot_lon"].between(
                75.01,
                100
            ),
        ),

        (
            "Southern Indian Ocean",

            latest["latitude"].between(
                -30,
                5
            )
            &
            latest["plot_lon"].between(
                40,
                100
            ),
        ),

    ]

    for name, condition in regional_definitions:

        subset = latest[
            condition
        ]

        regional.append(

            {

                "Region":
                    name,

                "Cells":
                    len(subset),

                "Potential-risk cells":
                    int(
                        subset["risk_flag"]
                        .sum()
                    ),

                "Risk share":
                    (
                        100
                        * subset[
                            "risk_flag"
                        ].mean()
                        if len(subset)
                        else 0
                    ),

                "Mean Chl-a":
                    (
                        subset["chla"].mean()
                        if len(subset)
                        else np.nan
                    ),
            }
        )

    regional_df = pd.DataFrame(
        regional
    )

    regional_df[
        "Risk share"
    ] = regional_df[
        "Risk share"
    ].round(2)

    regional_df[
        "Mean Chl-a"
    ] = regional_df[
        "Mean Chl-a"
    ].round(4)

    st.markdown(
        '<div class="kicker" style="margin-top:18px">'
        'Regional view'
        '</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        regional_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="note">

            <b>
                Interpretation:
            </b>

            These are current-field descriptive indicators.
            They should not be interpreted as direct measures
            of harmfulness, fisheries catch or confirmed HAB severity.

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="kicker" style="margin-top:25px">'
        'Investigation queue'
        '</div>',
        unsafe_allow_html=True,
    )

    queue = (
        risk
        .sort_values(
            "chla",
            ascending=False
        )
        .head(25)
        .copy()
    )

    keep_columns = [

        column

        for column in [

            "latitude",
            "longitude",
            "chla",
            "chla_anomaly",
            "chla_change",
            "risk_probability",
            "risk_label",

        ]

        if column in queue.columns
    ]

    queue = queue[
        keep_columns
    ]

    for column in [
        "chla",
        "chla_anomaly",
        "chla_change",
        "risk_probability",
    ]:

        if column in queue.columns:

            queue[column] = (
                queue[column]
                .round(4)
            )

    st.dataframe(
        queue,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SATELLITE EXPLORER
# ============================================================

elif page == "explorer":

    section(
        "06 · Satellite explorer",
        "Explore the satellite signal",
        "Use the current EOS-06 OCM-3 chlorophyll-a field as an ocean-colour exploration layer. This page describes what the dataset actually contains rather than inventing additional satellite variables.",
    )

    explorer_metrics = [

        (
            "Product",
            "E06OCM_L4_AC",
            "EOS-06 / OCM-3",
        ),

        (
            "Variable",
            "Chl-a",
            "chlorophyll-a",
        ),

        (
            "Grid",
            "0.25°",
            "latitude × longitude",
        ),

        (
            "Latest date",
            latest_date.strftime(
                "%d %b %Y"
            ),
            "processed",
        ),

    ]

    columns = st.columns(
        4,
        gap="medium"
    )

    for column, values in zip(
        columns,
        explorer_metrics
    ):

        with column:

            metric_card(*values)

    image_column, info_column = st.columns(
        [1.1, .9],
        gap="large",
    )

    with image_column:

        if SATELLITE_IMAGE.exists():

            st.image(
                str(SATELLITE_IMAGE),
                use_container_width=True,
            )

        elif PHYTO_IMAGE.exists():

            st.image(
                str(PHYTO_IMAGE),
                use_container_width=True,
            )

    with info_column:

        information_card(
            "What is chlorophyll-a?",
            "Chlorophyll-a is a satellite-observable indicator related to phytoplankton biomass and ocean productivity. It is useful for screening spatial changes, but it is not by itself a harmfulness or toxin measurement.",
            "🌱",
        )

        st.markdown(
            '<div style="height:12px"></div>',
            unsafe_allow_html=True,
        )

        information_card(
            "What can this page support?",
            "Spatial exploration, comparison of current values, identification of elevated screening signals and selection of coordinates for further investigation.",
            "🛰️",
        )

        st.markdown(
            '<div style="height:12px"></div>',
            unsafe_allow_html=True,
        )

        information_card(
            "What is not available?",
            "The current CSV does not provide direct measurements for toxins, harmful species identity, sea-surface temperature, salinity, currents, nutrients, turbidity or dissolved oxygen.",
            "🔬",
        )

    st.markdown(
        '<div style="height:20px"></div>',
        unsafe_allow_html=True,
    )

    explorer_study = study_area(
        latest
    )

    if not explorer_study.empty:

        top_values = (
            explorer_study
            .nlargest(
                20,
                "chla"
            )
            [
                [
                    "latitude",
                    "longitude",
                    "chla",
                    "risk_label",
                ]
            ]
            .copy()
        )

        top_values[
            "chla"
        ] = top_values[
            "chla"
        ].round(4)

        st.markdown(
            '<div class="kicker">'
            'Highest current Chl-a observations'
            '</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            top_values,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        """
        <div class="note">

            <b>
                Important:
            </b>

            A high chlorophyll-a value is an observation
            signal. It does not prove a harmful algal bloom.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ENVIRONMENT
# ============================================================

elif page == "environment":

    section(
        "07 · Coastal environment",
        "Environmental condition indicators",
        "The current dataset supports chlorophyll-based environmental screening. It does not support a complete water-quality diagnosis because several physical and chemical variables are not present.",
    )

    mean_chla = latest[
        "chla"
    ].mean()

    median_chla = latest[
        "chla"
    ].median()

    max_chla = latest[
        "chla"
    ].max()

    percentile_95 = latest[
        "chla"
    ].quantile(.95)

    environment_metrics = [

        (
            "Mean Chl-a",
            fmt_num(
                mean_chla,
                4
            ),
            "latest field",
        ),

        (
            "Median Chl-a",
            fmt_num(
                median_chla,
                4
            ),
            "latest field",
        ),

        (
            "95th percentile",
            fmt_num(
                percentile_95,
                4
            ),
            "latest field",
        ),

        (
            "Maximum Chl-a",
            fmt_num(
                max_chla,
                4
            ),
            "latest field",
        ),

    ]

    columns = st.columns(
        4,
        gap="medium"
    )

    for column, values in zip(
        columns,
        environment_metrics
    ):

        with column:

            metric_card(*values)

    chart_left, chart_right = st.columns(
        2,
        gap="large"
    )

    with chart_left:

        summary = pd.DataFrame(
            {

                "Indicator": [
                    "Mean",
                    "Median",
                    "95th percentile",
                    "Maximum",
                ],

                "Chl-a": [
                    mean_chla,
                    median_chla,
                    percentile_95,
                    max_chla,
                ],

            }
        )

        figure = px.bar(
            summary,

            x="Indicator",

            y="Chl-a",

            text="Chl-a",
        )

        figure.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside",
        )

        figure.update_layout(
            xaxis_title=
                "Summary statistic",

            yaxis_title=
                "Chlorophyll-a",
        )

        chart_layout(
            figure
        )

        st.markdown(
            """
            <div class="chart-card">

                <h3>
                    Latest-field Chl-a summary
                </h3>

                <p>
                    Descriptive environmental signal.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
            config={
                "displaylogo": False
            },
        )

    with chart_right:

        regional_environment = []

        for name, condition in regional_definitions:

            subset = latest[
                condition
            ]

            regional_environment.append(

                {

                    "Region":
                        name,

                    "Mean Chl-a":
                        (
                            subset["chla"].mean()
                            if len(subset)
                            else np.nan
                        ),

                }
            )

        environment_df = pd.DataFrame(
            regional_environment
        )

        figure = px.bar(
            environment_df,

            x="Region",

            y="Mean Chl-a",

            text="Mean Chl-a",
        )

        figure.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside",
        )

        figure.update_layout(
            xaxis_title="Region",

            yaxis_title=
                "Mean chlorophyll-a",
        )

        chart_layout(
            figure
        )

        st.markdown(
            """
            <div class="chart-card">

                <h3>
                    Regional Chl-a context
                </h3>

                <p>
                    Current processed field comparison.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
            config={
                "displaylogo": False
            },
        )

    st.markdown(
        """
        <div class="note">

            <b>
                Data limitation:
            </b>

            This is a chlorophyll-based environmental indicator,
            not a complete water-quality assessment.

            Temperature, turbidity, salinity, nutrients and
            other observations can be added later.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FISHERIES
# ============================================================

elif page == "fisheries":

    section(
        "08 · Fisheries intelligence",
        "Chlorophyll-based productivity context",
        "Chlorophyll-a can provide environmental context for marine productivity. With the current dataset, this is an indicator layer only, not a fish-abundance or catch prediction system.",
    )

    low_cut = latest[
        "chla"
    ].quantile(.33)

    high_cut = latest[
        "chla"
    ].quantile(.67)

    fisheries_data = latest.copy()

    def productivity_band(
        value
    ):

        if value >= high_cut:
            return "Higher Chl-a zone"

        if value <= low_cut:
            return "Lower Chl-a zone"

        return "Intermediate Chl-a zone"

    fisheries_data[
        "productivity_context"
    ] = fisheries_data[
        "chla"
    ].apply(
        productivity_band
    )

    order = [

        "Lower Chl-a zone",

        "Intermediate Chl-a zone",

        "Higher Chl-a zone",

    ]

    distribution = (

        fisheries_data[
            "productivity_context"
        ]

        .value_counts()

        .reindex(
            order,
            fill_value=0
        )

        .reset_index()
    )

    distribution.columns = [
        "Zone",
        "Cells",
    ]

    metrics = [

        (
            "Lower Chl-a zone",
            f"{distribution.iloc[0]['Cells']:,}",
            "relative field",
        ),

        (
            "Intermediate zone",
            f"{distribution.iloc[1]['Cells']:,}",
            "relative field",
        ),

        (
            "Higher Chl-a zone",
            f"{distribution.iloc[2]['Cells']:,}",
            "relative field",
        ),

    ]

    columns = st.columns(
        3,
        gap="medium"
    )

    for column, values in zip(
        columns,
        metrics
    ):

        with column:

            metric_card(*values)

    figure = px.bar(
        distribution,

        x="Zone",

        y="Cells",

        text="Cells",
    )

    figure.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    figure.update_layout(
        xaxis_title=
            "Relative chlorophyll-a zone",

        yaxis_title=
            "Number of cells",
    )

    chart_layout(
        figure
    )

    st.markdown(
        """
        <div class="chart-card">

            <h3>
                Productivity-context distribution
            </h3>

            <p>
                Zones are defined relative to the current
                field distribution.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "displaylogo": False
        },
    )

    st.markdown(
        """
        <div class="note">

            <b>
                Scientific boundary:
            </b>

            This indicator does not estimate fish abundance,
            catch, species composition, fishing grounds or
            stock health.

            It uses chlorophyll-a only as environmental
            productivity context.

        </div>
        """,
        unsafe_allow_html=True,
    )

    fisheries_table = fisheries_data[
        [
            "latitude",
            "longitude",
            "chla",
            "productivity_context",
            "risk_label",
        ]
    ].copy()

    fisheries_table[
        "chla"
    ] = fisheries_table[
        "chla"
    ].round(4)

    st.download_button(

        "⬇ Download productivity-context table",

        fisheries_table
        .to_csv(
            index=False
        )
        .encode("utf-8"),

        "bloomdetect_fisheries_productivity_context.csv",

        "text/csv",

        use_container_width=True,
    )


# ============================================================
# EARLY WARNING
# ============================================================

elif page == "alerts":

    section(
        "09 · Early-warning dashboard",
        "Turn screening into an action queue",
        "This page converts the latest potential-risk output into a practical investigation workflow: count, concentration, priority cells and downloadable evidence.",
    )

    total = len(latest)

    flagged = len(risk)

    share = (
        100 * flagged / total
        if total
        else 0
    )

    alert_metrics = [

        (
            "Current field",
            latest_date.strftime(
                "%d %b"
            ),
            "latest processed",
        ),

        (
            "Flagged cells",
            f"{flagged:,}",
            "potential risk",
        ),

        (
            "Risk share",
            f"{share:.2f}%",
            "of latest field",
        ),

        (
            "Action",
            "Investigate",
            "screening output",
        ),

    ]

    columns = st.columns(
        4,
        gap="medium"
    )

    for column, values in zip(
        columns,
        alert_metrics
    ):

        with column:

            metric_card(*values)

    if flagged:

        queue = risk.copy()

        # A simple organizational signal.
        # This is NOT a severity ranking.
        score_parts = []

        if "chla" in queue.columns:
            score_parts.append(
                queue["chla"].rank(
                    pct=True
                )
            )

        if "chla_anomaly" in queue.columns:
            score_parts.append(
                queue["chla_anomaly"].rank(
                    pct=True
                )
            )

        if "chla_change" in queue.columns:
            score_parts.append(
                queue["chla_change"].rank(
                    pct=True
                )
            )

        if score_parts:

            queue[
                "screening_priority_signal"
            ] = sum(
                score_parts
            )

            queue = queue.sort_values(
                "screening_priority_signal",
                ascending=False,
            )

        display_columns = [

            column

            for column in [

                "latitude",
                "longitude",
                "chla",
                "chla_anomaly",
                "chla_change",
                "risk_probability",
                "risk_label",

            ]

            if column in queue.columns
        ]

        queue = queue[
            display_columns
        ].head(50).copy()

        for column in [

            "chla",
            "chla_anomaly",
            "chla_change",
            "risk_probability",

        ]:

            if column in queue.columns:

                queue[column] = (
                    queue[column]
                    .round(4)
                )

        st.markdown(
            """
            <div class="section"
                 style="padding-top:28px">

                <div class="kicker">
                    Investigation queue
                </div>

                <h1 style="font-size:2.2rem">
                    Cells to inspect
                </h1>

                <p>
                    This queue organizes available screening
                    signals for investigation. It is not a
                    ranking of confirmed HAB severity.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(
            queue,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(

            "⬇ Download early-warning investigation queue",

            queue
            .to_csv(
                index=False
            )
            .encode("utf-8"),

            "bloomdetect_early_warning_queue.csv",

            "text/csv",

            use_container_width=True,
        )

    else:

        st.markdown(
            """
            <div class="card">

                <h3>
                    No current potential-risk flags
                </h3>

                <p>
                    The latest processed field contains no
                    potential-risk screening cells.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="note">

            <b>
                Workflow:
            </b>

            Map → Hotspot → Coordinate check →
            Field/environmental validation.

            The dashboard supports early attention;
            it does not issue a confirmed environmental event alert.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DATA & RESEARCH
# ============================================================

elif page == "data":

    section(
        "10 · Data & research",
        "Dataset, fields and downloads",
        "The research page keeps source information and reusable outputs in one place so the analytical pages stay clean.",
    )

    data_metrics = [

        (
            "Product",
            "E06OCM_L4_AC",
            "EOS-06 / OCM-3",
        ),

        (
            "Grid",
            "0.25°",
            "latitude × longitude",
        ),

        (
            "Latest cells",
            f"{len(latest):,}",
            "processed",
        ),

        (
            "Latest date",
            latest_date.strftime(
                "%d %b %Y"
            ),
            "processed",
        ),

    ]

    columns = st.columns(
        4,
        gap="medium"
    )

    for column, values in zip(
        columns,
        data_metrics
    ):

        with column:

            metric_card(*values)

    st.markdown(
        """
        <div class="card"
             style="margin-top:18px">

            <h3>
                Source product
            </h3>

            <p>

                <b>
                    Satellite:
                </b>
                EOS-06 / Oceansat-3 OCM-3

                <br>

                <b>
                    Product:
                </b>
                E06OCM_L4_AC Level-4 Analysed Chlorophyll

                <br>

                <b>
                    Primary variable:
                </b>
                chla

                <br>

                <b>
                    Spatial grid:
                </b>
                0.25° latitude × 0.25° longitude

                <br>

                <b>
                    Interpretation:
                </b>
                satellite-derived environmental signal
                used for screening and spatial analysis.

            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section"
             style="padding-top:28px">

            <div class="kicker">
                Data dictionary
            </div>

            <h1 style="font-size:2.15rem">
                Fields used by BloomDetect
            </h1>

            <p>
                Fields already available in the processed CSV.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    field_rows = [

        (
            "latitude",
            "Grid latitude",
            "degrees",
        ),

        (
            "longitude",
            "Grid longitude",
            "degrees",
        ),

        (
            "date",
            "Observation date",
            "date",
        ),

        (
            "chla",
            "Satellite-derived chlorophyll-a",
            "numeric",
        ),

        (
            "risk_label",
            "Potential-risk screening output",
            "category",
        ),

        (
            "risk_probability",
            "Stored model probability",
            "numeric",
        ),

        (
            "previous_chla",
            "Previous observation",
            "derived",
        ),

        (
            "historical_baseline",
            "Historical baseline",
            "derived",
        ),

        (
            "recent_mean",
            "Recent mean",
            "derived",
        ),

        (
            "recent_max",
            "Recent maximum",
            "derived",
        ),

        (
            "chla_anomaly",
            "Chlorophyll anomaly",
            "derived",
        ),

        (
            "chla_change",
            "Change from previous observation",
            "derived",
        ),

    ]

    field_df = pd.DataFrame(

        field_rows,

        columns=[
            "Field",
            "Meaning",
            "Type",
        ],
    )

    st.dataframe(
        field_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="section"
             style="padding-top:28px">

            <div class="kicker">
                Downloads
            </div>

            <h1 style="font-size:2.15rem">
                Research outputs
            </h1>

            <p>
                Reusable files generated directly from
                the current dashboard state.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    download_left, download_right = st.columns(
        2,
        gap="large",
    )

    with download_left:

        st.markdown(
            """
            <div class="download-card">

                <h3>
                    Latest processed observations
                </h3>

                <p>
                    Complete latest-date table used by
                    the dashboard.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.download_button(

            "⬇ Download latest observations",

            latest
            .to_csv(
                index=False
            )
            .encode("utf-8"),

            "bloomdetect_latest_observations.csv",

            "text/csv",

            use_container_width=True,
        )

    with download_right:

        st.markdown(
            """
            <div class="download-card">

                <h3>
                    Potential-risk shortlist
                </h3>

                <p>
                    Current locations carrying a
                    potential bloom-risk screening flag.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.download_button(

            "⬇ Download potential-risk locations",

            risk
            .to_csv(
                index=False
            )
            .encode("utf-8"),

            "bloomdetect_potential_risk.csv",

            "text/csv",

            use_container_width=True,
        )

    st.markdown(
        '<div style="height:12px"></div>',
        unsafe_allow_html=True,
    )

    project_summary = pd.DataFrame(
        [

            {

                "project":
                    "BloomDetect AI",

                "product":
                    "E06OCM_L4_AC",

                "latest_date":
                    latest_date.strftime(
                        "%Y-%m-%d"
                    ),

                "latest_cells":
                    len(latest),

                "potential_risk_cells":
                    len(risk),

                "risk_share_percent":
                    round(
                        risk_share,
                        4
                    ),

                "study_region":
                    "Indian Ocean / Arabian Sea / Bay of Bengal",

                "scientific_boundary":
                    (
                        "Potential bloom-risk screening only; "
                        "not confirmed HAB/species/toxin detection"
                    ),

            }

        ]
    )

    st.markdown(
        """
        <div class="download-card">

            <h3>
                Project screening summary
            </h3>

            <p>
                Compact audit-friendly summary for
                reports, presentations and documentation.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.download_button(

        "⬇ Download project screening summary",

        project_summary
        .to_csv(
            index=False
        )
        .encode("utf-8"),

        "bloomdetect_project_summary.csv",

        "text/csv",

        use_container_width=True,
    )

    st.markdown(
        """
        <div class="note">

            <b>
                Research boundary:
            </b>

            The current processed file does not contain
            confirmed harmful-bloom species or toxin labels.

            Satellite screening should be combined with
            field observations and other environmental evidence.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="footer">

        BloomDetect AI · EOS-06 / OCM-3 ·
        Potential bloom-risk screening ·
        Latest processed data:
        {latest_date.strftime("%d %B %Y")}

    </div>
    """,
    unsafe_allow_html=True,
)
