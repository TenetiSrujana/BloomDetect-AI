from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# BLOOMDETECT AI
# Home | Risk Map | Location | Insights | Data
# ============================================================

st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "latest_bloom_risk_predictions.csv"
IMAGE_PATH = BASE_DIR / "bloomdetect_bloom_process.png"

HISTORY_FILES = [
    BASE_DIR / "bloomdetect_history_web.csv.gz",
    BASE_DIR / "bloomdetect_history.csv.gz",
    BASE_DIR / "bloomdetect_history.csv",
]

LAT_MIN, LAT_MAX = -40.0, 30.0
LON_MIN, LON_MAX = 20.0, 120.0


# ============================================================
# DATA
# ============================================================

@st.cache_data(show_spinner="Loading BloomDetect data...")
def load_predictions():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "latest_bloom_risk_predictions.csv is missing beside app.py."
        )

    d = pd.read_csv(DATA_PATH)

    required = {"latitude", "longitude", "date", "chla", "risk_label"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError("Prediction CSV is missing: " + ", ".join(sorted(missing)))

    d["date"] = pd.to_datetime(d["date"], errors="coerce")

    for col in ["latitude", "longitude", "chla"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    for col in [
        "risk_probability",
        "model_score",
        "previous_chla",
        "historical_baseline",
        "recent_mean",
        "recent_max",
        "chla_anomaly",
        "chla_change",
    ]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["latitude", "longitude", "date", "chla"]).copy()

    d["risk_flag"] = (
        d["risk_label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("potential bloom risk")
    )

    return d


@st.cache_data(show_spinner="Loading date history...")
def load_history():
    path = next((p for p in HISTORY_FILES if p.exists()), None)

    if path is None:
        return pd.DataFrame(), ""

    h = pd.read_csv(path)

    required = {"date", "lat_bin", "lon_bin", "chla", "risk"}
    if not required.issubset(h.columns):
        return pd.DataFrame(), ""

    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    for col in ["lat_bin", "lon_bin", "chla", "risk"]:
        h[col] = pd.to_numeric(h[col], errors="coerce")

    h = h.replace([np.inf, -np.inf], np.nan)
    h = h.dropna(subset=["date", "lat_bin", "lon_bin", "chla"]).copy()
    h["risk"] = h["risk"].fillna(0).astype("int8")

    h = h[
        h["lat_bin"].between(LAT_MIN, LAT_MAX)
        & h["lon_bin"].between(LON_MIN, LON_MAX)
    ].copy()

    return h, path.name


try:
    df = load_predictions()
    history, history_name = load_history()
except Exception as exc:
    st.error("BloomDetect AI could not load the project data.")
    st.code(str(exc))
    st.stop()

latest_date = df["date"].max()
latest = df[df["date"].eq(latest_date)].copy()
latest_risk = latest[latest["risk_flag"]].copy()


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root{
    --ink:#103f49;
    --muted:#64838a;
    --aqua:#08a7b3;
    --line:#a8d9dd;
    --bg:#f2fbfb;
    --green:#20a477;
    --red:#ef4f5e;
    --blue:#197da5;
}

html, body, [data-testid="stAppViewContainer"]{
    background:var(--bg)!important;
    color:var(--ink)!important;
    font-family:'DM Sans',sans-serif!important;
}

.stApp{
    background:
        radial-gradient(circle at 8% 8%,rgba(45,210,212,.08),transparent 24%),
        radial-gradient(circle at 92% 75%,rgba(24,160,180,.06),transparent 28%),
        linear-gradient(180deg,#fbffff 0%,#effafa 55%,#fbffff 100%)!important;
}

[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer,
[data-testid="stSidebar"]{
    display:none!important;
}

.block-container{
    max-width:1220px!important;
    padding:25px 28px 65px!important;
    margin:0 auto!important;
}

.brand{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:20px;
    padding:15px 20px;
    border:1.5px solid #c4e4e6;
    background:#fff;
    border-radius:22px;
    box-shadow:0 12px 30px rgba(9,80,94,.08);
}

.brand-left{
    display:flex;
    align-items:center;
    gap:13px;
}

.logo{
    width:48px;
    height:48px;
    border-radius:15px;
    background:linear-gradient(145deg,#2bcdd0,#087b8e);
    display:grid;
    place-items:center;
    color:white;
    font-size:20px;
    font-weight:800;
}

.brand-name{
    font:800 1.2rem Manrope,sans-serif;
    color:var(--ink);
}

.brand-sub{
    color:#78959b;
    font-size:.72rem;
    margin-top:2px;
}

.latest{
    color:#78959b;
    font-size:.67rem;
    text-align:right;
    line-height:1.4;
}

.latest b{color:#174b56;font-size:.77rem}

.nav-space{height:10px}

.stButton>button, .stDownloadButton>button, .stFormSubmitButton>button{
    min-height:44px!important;
    border-radius:13px!important;
    background:#fff!important;
    border:2px solid #164e5b!important;
    color:#123f49!important;
    font-weight:800!important;
    font-size:.84rem!important;
    box-shadow:0 5px 15px rgba(15,70,82,.08)!important;
}

.stButton>button:hover, .stDownloadButton>button:hover,
.stFormSubmitButton>button:hover{
    background:#e2f8f8!important;
    border-color:#087d8c!important;
}

.stButton>button[kind="primary"]{
    background:linear-gradient(135deg,#d9f7f7,#bceeee)!important;
    border-color:#078c9b!important;
}

.kicker{
    font:800 .66rem Manrope,sans-serif;
    letter-spacing:.17em;
    text-transform:uppercase;
    color:#0797a5;
    margin-bottom:9px;
}

h1, h2, h3{
    font-family:Manrope,sans-serif!important;
    color:var(--ink)!important;
}

.page-title{
    margin:25px 0 15px;
}

.page-title h2{
    font:800 clamp(2rem,4vw,3.25rem)/1.05 Manrope,sans-serif!important;
    letter-spacing:-.055em;
    margin:0 0 10px!important;
}

.page-title p{
    color:#5f8088;
    line-height:1.62;
    max-width:1050px;
    font-size:.95rem;
    margin:0;
}

.hero{
    margin-top:4px;
    min-height:390px;
    padding:55px;
    border-radius:30px;
    background:linear-gradient(135deg,#063d51,#076d7e 55%,#13a9ad);
    color:white;
    position:relative;
    overflow:hidden;
    box-shadow:0 24px 65px rgba(6,86,100,.15);
}

.hero:before{
    content:"";
    position:absolute;
    inset:-25%;
    background:repeating-radial-gradient(
        ellipse at 20% 115%,
        transparent 0 55px,
        rgba(181,255,251,.12) 57px 59px,
        transparent 61px 105px
    );
    transform:rotate(-7deg);
}

.hero-content{
    position:relative;
    z-index:1;
    max-width:800px;
}

.hero .kicker{color:#a6fffa}

.hero h1{
    color:#e4ffff!important;
    font:800 clamp(3rem,6vw,5.7rem)/.92 Manrope,sans-serif!important;
    letter-spacing:-.075em;
    margin:0 0 18px!important;
}

.hero p{
    color:#e0fbfb;
    font-size:1.03rem;
    line-height:1.75;
    margin:0;
    max-width:760px;
}

.badges{
    display:flex;
    flex-wrap:wrap;
    gap:9px;
    margin-top:22px;
}

.badge{
    padding:8px 12px;
    border-radius:999px;
    background:rgba(255,255,255,.14);
    border:1px solid rgba(255,255,255,.25);
    color:#efffff;
    font-size:.75rem;
    font-weight:700;
}

.metric{
    min-height:110px;
    padding:17px;
    background:linear-gradient(145deg,#fff,#eaf8f8);
    border:1.5px solid #a6d5da;
    border-radius:18px;
    box-shadow:0 10px 25px rgba(15,91,101,.07);
}

.metric-label{
    font:800 .62rem Manrope,sans-serif;
    letter-spacing:.11em;
    text-transform:uppercase;
    color:#6d8c93;
}

.metric-value{
    font:800 1.4rem Manrope,sans-serif;
    color:#123f49;
    margin-top:6px;
}

.metric-note{
    color:#78959b;
    font-size:.71rem;
    margin-top:3px;
}

.card{
    background:#fff;
    border:1.5px solid #a9d6da;
    border-radius:21px;
    box-shadow:0 12px 32px rgba(9,80,94,.07);
    padding:22px;
    height:100%;
    box-sizing:border-box;
}

.card h3{
    font-size:1.2rem!important;
    margin:0 0 8px!important;
}

.card p{
    color:#66848b;
    line-height:1.62;
    font-size:.9rem;
    margin:0;
}

.note{
    margin-top:15px;
    padding:13px 15px;
    background:#e6f8f8;
    border-left:4px solid #11a9b2;
    border-radius:0 13px 13px 0;
    color:#52757c;
    font-size:.83rem;
    line-height:1.55;
}

.tool-card{
    min-height:145px;
    padding:18px;
    background:#fff;
    border:1.5px solid #b5dde0;
    border-radius:19px;
    box-shadow:0 10px 25px rgba(9,80,94,.07);
    box-sizing:border-box;
}

.tool-icon{font-size:1.25rem;margin-bottom:8px}
.tool-card h3{font-size:1rem!important;margin:0 0 6px!important}
.tool-card p{color:#66848b;font-size:.81rem;line-height:1.5;margin:0}

.section-gap{height:28px}

.timeline{
    background:#fff;
    border:1.5px solid #a9d6da;
    border-radius:17px;
    padding:14px 18px;
    margin:12px 0 16px;
    box-shadow:0 9px 23px rgba(9,80,94,.06);
}

.timeline-row{
    display:flex;
    align-items:end;
    justify-content:space-between;
    gap:15px;
}

.timeline-title{
    font:800 .86rem Manrope;
    color:#174b56;
}

.timeline-help{
    font-size:.71rem;
    color:#6c8b92;
    margin-top:3px;
}

.timeline-date{
    font:800 .98rem Manrope;
    color:#103f49;
}

.map-shell{
    background:#dff7f8;
    border:2px solid #8fcbd1;
    border-radius:22px;
    padding:4px;
    overflow:hidden;
}

.map-heading{
    display:flex;
    justify-content:space-between;
    gap:10px;
    padding:10px 12px;
    color:#174b56;
    font-size:.86rem;
}

.map-heading span{
    color:#6b8b92;
    font-size:.68rem;
}

.legend-row{
    display:flex;
    flex-wrap:wrap;
    gap:8px 12px;
    align-items:center;
    padding:10px 13px;
    color:#4f747b;
    font-size:.75rem;
}

.dot{
    width:11px;
    height:11px;
    border-radius:50%;
    display:inline-block;
}
.dot-blue{background:#197da5}
.dot-green{background:#20a477}
.dot-red{background:#ef4f5e}

.result-box{
    border:1.5px solid #a9d6da;
    border-radius:21px;
    background:#fff;
    padding:22px;
    box-shadow:0 12px 32px rgba(9,80,94,.07);
}

.result-box h3{font-size:1.45rem!important;margin:0 0 7px!important}
.result-box p{color:#66848b;font-size:.88rem;line-height:1.55}

.status-risk{
    background:#fff0f2;
    border:2px solid #f06a78;
    color:#9e2d3c;
    border-radius:15px;
    padding:14px;
    margin-top:12px;
}

.status-normal{
    background:#eafaf4;
    border:2px solid #49b995;
    color:#176f58;
    border-radius:15px;
    padding:14px;
    margin-top:12px;
}

.status-title{
    font:800 .88rem Manrope;
    margin-bottom:4px;
}

.signal{
    padding:12px;
    background:#f7fcfc;
    border:1px solid #c9e5e7;
    border-radius:13px;
}

.signal-label{font-size:.66rem;color:#78959b}
.signal-value{font:800 .88rem Manrope;color:#164a55;margin-top:4px}

.chart-box{
    background:#fff;
    border:1.5px solid #b6dfe2;
    border-radius:19px;
    padding:14px;
    box-shadow:0 10px 27px rgba(9,80,94,.07);
}

.download-card{
    min-height:135px;
    padding:19px;
    border-radius:19px;
    background:linear-gradient(135deg,#087b8b,#13adb3);
    border:2px solid #07576a;
    color:white;
    box-sizing:border-box;
}

.download-card h3{color:white!important;font-size:1.05rem!important;margin:0 0 6px!important}
.download-card p{color:#e5ffff;font-size:.81rem;line-height:1.45;margin:0}

.footer{
    border-top:1px solid #d5ebed;
    margin-top:38px;
    padding-top:14px;
    color:#76959b;
    font-size:.68rem;
}

@media(max-width:900px){
    .hero{padding:42px 32px}
}
@media(max-width:620px){
    .block-container{padding:14px 12px 45px!important}
    .hero{padding:36px 23px}
    .hero h1{font-size:3rem!important}
    .latest{display:none}
    .timeline-row{align-items:flex-start;flex-direction:column}
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def metric(label, value, note):
    st.markdown(
        f"""
        <div class="metric">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def heading(kicker, title, description):
    st.markdown(
        f"""
        <div class="page-title">
            <div class="kicker">{kicker}</div>
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_num(value, digits=4):
    try:
        if pd.isna(value):
            return "Unavailable"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "Unavailable"


def probability_text(value):
    try:
        if pd.isna(value):
            return "Unavailable"
        x = float(value)
        if x <= 1:
            x *= 100
        return f"{x:.1f}%"
    except Exception:
        return "Unavailable"


def region_name(lat, lon):
    if 5 <= lat <= 30 and 45 <= lon <= 75:
        return "Arabian Sea"
    if 0 <= lat <= 25 and 75 < lon <= 100:
        return "Bay of Bengal"
    if -30 <= lat < 5 and 40 <= lon <= 100:
        return "Southern Indian Ocean"
    if 5 <= lat <= 30 and 75 < lon <= 120:
        return "Northern Indian Ocean"
    return "Other study area"


def nearest_latest(lat, lon):
    a = latest.copy()
    lon_scale = max(np.cos(np.deg2rad(float(lat))), 0.25)
    dist = (
        (a["latitude"].to_numpy() - float(lat)) ** 2
        + (
            (a["longitude"].to_numpy() - float(lon)) * lon_scale
        ) ** 2
    )
    return a.iloc[int(np.argmin(dist))]


def history_dates():
    if history.empty:
        return []
    return sorted(pd.to_datetime(history["date"].dropna().unique()))


def history_for_date(date_value):
    if history.empty:
        return pd.DataFrame()
    h = history[history["date"].eq(pd.Timestamp(date_value))].copy()
    return h[
        h["lat_bin"].between(LAT_MIN, LAT_MAX)
        & h["lon_bin"].between(LON_MIN, LON_MAX)
    ].copy()


def make_map(date_value):
    selected = pd.Timestamp(date_value)

    if selected == latest_date:
        field = latest.copy()
        field["lat_bin"] = np.floor(field["latitude"]) + 0.5
        field["lon_bin"] = np.floor(field["longitude"]) + 0.5

        blue = (
            field.groupby(["lat_bin", "lon_bin"], as_index=False)
            .agg(mean_chla=("chla", "mean"), cells=("chla", "size"))
        )

        red = field[field["risk_flag"]].copy()
        processed_count = len(field)
        risk_count = int(field["risk_flag"].sum())

    else:
        h = history_for_date(selected)

        if h.empty:
            return make_map(latest_date)

        blue = h.rename(
            columns={
                "lat_bin": "lat",
                "lon_bin": "lon",
                "chla": "mean_chla",
            }
        ).copy()

        red = blue[blue["risk"].gt(0)].copy()
        red = red.rename(
            columns={
                "lat": "latitude",
                "lon": "longitude",
                "mean_chla": "chla",
            }
        )

        processed_count = len(blue)
        risk_count = int(blue["risk"].gt(0).sum())

    fig = go.Figure()

    if not blue.empty:
        lat_col = "lat_bin" if "lat_bin" in blue.columns else "lat"
        lon_col = "lon_bin" if "lon_bin" in blue.columns else "lon"

        fig.add_trace(
            go.Scattergeo(
                lat=blue[lat_col],
                lon=blue[lon_col],
                mode="markers",
                name="Processed ocean field",
                marker=dict(size=7, color="#197da5", opacity=.58),
                text=[f"Mean Chl-a: {x:.4f}" for x in blue["mean_chla"]],
                hovertemplate="<b>Processed ocean field</b><br>%{text}<extra></extra>",
            )
        )

    if not red.empty:
        red = red.copy()
        red["zone_lat"] = np.floor(red["latitude"] / 2) * 2 + 1
        red["zone_lon"] = np.floor(red["longitude"] / 2) * 2 + 1

        zones = (
            red.groupby(["zone_lat", "zone_lon"], as_index=False)
            .size()
            .rename(columns={"size": "flagged_cells"})
        )

        fig.add_trace(
            go.Scattergeo(
                lat=zones["zone_lat"],
                lon=zones["zone_lon"],
                mode="markers",
                name="Flag concentration zone",
                marker=dict(
                    size=np.clip(zones["flagged_cells"] * 1.3 + 9, 10, 34),
                    color="#20a477",
                    opacity=.42,
                    line=dict(width=1.2, color="#fff"),
                ),
                text=zones["flagged_cells"],
                hovertemplate="<b>Flag concentration zone</b><br>Flagged cells: %{text}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scattergeo(
                lat=red["latitude"],
                lon=red["longitude"],
                mode="markers",
                name="Potential bloom-risk cell",
                marker=dict(
                    size=6,
                    color="#ef4f5e",
                    opacity=.92,
                    line=dict(width=.7, color="#fff"),
                ),
                text=[f"Chl-a: {x:.4f}" for x in red["chla"]],
                hovertemplate="<b>Potential bloom-risk screening</b><br>%{text}<extra></extra>",
            )
        )

    fig.update_geos(
        showland=True,
        landcolor="#dce9e7",
        showocean=True,
        oceancolor="#dff7f8",
        showcoastlines=True,
        coastlinecolor="#4d9099",
        coastlinewidth=1,
        showcountries=True,
        countrycolor="#9ab9be",
        bgcolor="#dff7f8",
        lataxis_range=[LAT_MIN, LAT_MAX],
        lonaxis_range=[LON_MIN, LON_MAX],
        center=dict(lat=-5, lon=70),
        projection_type="equirectangular",
    )

    fig.update_layout(
        height=610,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#dff7f8",
        plot_bgcolor="#dff7f8",
        font=dict(color="#174b56"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=.01,
            xanchor="left",
            x=.01,
            bgcolor="rgba(255,255,255,.92)",
            bordercolor="#b9dfe2",
            borderwidth=1,
            font=dict(size=10, color="#174b56"),
        ),
    )

    return fig, processed_count, risk_count, selected


def hotspot_table():
    r = latest[latest["risk_flag"]].copy()

    if r.empty:
        return pd.DataFrame()

    r["lat_zone"] = np.floor(r["latitude"] / 2) * 2
    r["lon_zone"] = np.floor(r["longitude"] / 2) * 2

    z = (
        r.groupby(["lat_zone", "lon_zone"], as_index=False)
        .agg(
            flagged_cells=("risk_flag", "size"),
            mean_chla=("chla", "mean"),
            max_chla=("chla", "max"),
        )
        .sort_values(["flagged_cells", "mean_chla"], ascending=False)
        .head(12)
    )

    z["Zone"] = z.apply(
        lambda x: f"{x.lat_zone:.0f}°–{x.lat_zone + 2:.0f}°, "
                  f"{x.lon_zone:.0f}°–{x.lon_zone + 2:.0f}°",
        axis=1,
    )
    z["Flagged cells"] = z["flagged_cells"].astype(int)
    z["Mean Chl-a"] = z["mean_chla"].round(4)
    z["Max Chl-a"] = z["max_chla"].round(4)

    return z[["Zone", "Flagged cells", "Mean Chl-a", "Max Chl-a"]]


# ============================================================
# HEADER + NAVIGATION
# ============================================================

st.markdown(
    f"""
    <div class="brand">
        <div class="brand-left">
            <div class="logo">≈</div>
            <div>
                <div class="brand-name">BloomDetect AI</div>
                <div class="brand-sub">Coastal &amp; Ocean Intelligence · EOS-06 OCM-3</div>
            </div>
        </div>
        <div class="latest">
            Latest processed field<br>
            <b>{latest_date.strftime("%d %b %Y")}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    ("home", "⌂ Home"),
    ("map", "🗺 Risk Map"),
    ("location", "📍 Location"),
    ("insights", "📊 Insights"),
    ("data", "⇩ Data"),
]

if st.session_state.get("page") not in [p[0] for p in PAGES]:
    st.session_state.page = "home"

st.markdown('<div class="nav-space"></div>', unsafe_allow_html=True)

nav_cols = st.columns(5, gap="small")
for col, (page, label) in zip(nav_cols, PAGES):
    with col:
        if st.button(
            label,
            key=f"nav_{page}",
            width="stretch",
            type="primary" if st.session_state.page == page else "secondary",
        ):
            st.session_state.page = page
            st.rerun()


# ============================================================
# HOME
# ============================================================

if st.session_state.page == "home":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-content">
                <div class="kicker">EOS-06 · OCM-3 · SATELLITE INTELLIGENCE</div>
                <h1>Read the ocean signal.</h1>
                <p>
                    BloomDetect AI turns satellite-derived chlorophyll-a
                    observations and the project screening output into a
                    practical view of where unusual patterns may deserve
                    closer investigation.
                </p>
                <div class="badges">
                    <span class="badge">🌊 Ocean colour</span>
                    <span class="badge">🛰 EOS-06 OCM-3</span>
                    <span class="badge">🌱 Potential bloom-risk screening</span>
                    <span class="badge">📍 Spatial intelligence</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    risk_share = 100 * len(latest_risk) / len(latest) if len(latest) else 0

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    cols = st.columns(4, gap="medium")
    home_metrics = [
        ("Processed cells", f"{len(latest):,}", "latest field"),
        ("Potential-risk cells", f"{len(latest_risk):,}", "screening output"),
        ("Risk share", f"{risk_share:.2f}%", "of processed cells"),
        ("Maximum Chl-a", safe_num(latest["chla"].max()), "latest field"),
    ]
    for col, item in zip(cols, home_metrics):
        with col:
            metric(*item)

    heading(
        "PROJECT SNAPSHOT",
        "One clear view of the signal.",
        "The home page introduces the project and points to the four analysis tools. Detailed investigation stays inside the relevant page instead of being repeated everywhere.",
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            """
            <div class="card">
                <div class="kicker">WHAT BLOOMDETECT ADDS</div>
                <h3>From satellite observation to investigation support.</h3>
                <p>
                    The platform combines the latest satellite-derived
                    chlorophyll-a field with the project screening result,
                    then exposes that signal through a temporal map,
                    coordinate lookup and compact spatial insights.
                </p>
                <div class="note">
                    <b>Scientific note:</b> a potential bloom-risk flag is
                    not confirmation of a harmful algal bloom. Species,
                    toxin presence and ecological impact require additional
                    evidence and field validation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        if IMAGE_PATH.exists():
            st.image(
                str(IMAGE_PATH),
                width="stretch",
                caption="How satellite ocean-colour observations can support bloom-risk investigation",
            )
        else:
            st.info("Project illustration is not present beside app.py.")

    heading(
        "EXPLORE THE PLATFORM",
        "Four focused tools.",
        "Each page has one clear purpose, so the same information is not repeated across the dashboard.",
    )

    tools = [
        ("🗺️", "Risk Map", "Move through available dates and compare the spatial screening field.", "map", "Open Risk Map"),
        ("📍", "Location", "Check a coordinate against the nearest processed satellite cell.", "location", "Open Location"),
        ("📊", "Insights", "Review hotspot concentration, regional patterns and current Chl-a signals.", "insights", "Open Insights"),
        ("⇩", "Data", "Download the latest observations and potential-risk shortlist.", "data", "Open Data"),
    ]

    tool_cols = st.columns(4, gap="medium")
    for col, (icon, title, desc, target, button_text) in zip(tool_cols, tools):
        with col:
            st.markdown(
                f"""
                <div class="tool-card">
                    <div class="tool-icon">{icon}</div>
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(button_text, key=f"home_{target}", width="stretch"):
                st.session_state.page = target
                st.rerun()


# ============================================================
# RISK MAP
# ============================================================

elif st.session_state.page == "map":

    heading(
        "01 · SPATIAL INTELLIGENCE",
        "See how the field changes.",
        "Use the date slider to move through the available processed observations. The map keeps the same Indian Ocean study window so spatial changes are easy to compare.",
    )

    dates = history_dates()

    if len(dates) > 1:
        if (
            "map_date" not in st.session_state
            or pd.Timestamp(st.session_state.map_date) not in dates
        ):
            st.session_state.map_date = dates[-1]

        selected = st.slider(
            "Observation date",
            min_value=dates[0].date(),
            max_value=dates[-1].date(),
            value=pd.Timestamp(st.session_state.map_date).date(),
            format="DD MMM YYYY",
            key="map_date_slider",
        )
        selected_date = pd.Timestamp(selected)
        st.session_state.map_date = selected_date

        st.markdown(
            f"""
            <div class="timeline">
                <div class="timeline-row">
                    <div>
                        <div class="timeline-title">MAP TIMELINE</div>
                        <div class="timeline-help">Drag the slider to compare available satellite fields.</div>
                    </div>
                    <div class="timeline-date">{selected_date.strftime("%d %b %Y")}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        selected_date = latest_date
        st.info(
            "The compact history file is not available in this deployment. "
            "The latest processed field is shown."
        )

    fig, processed_count, risk_count, actual_date = make_map(selected_date)
    share = 100 * risk_count / processed_count if processed_count else 0

    cols = st.columns(4, gap="medium")
    items = [
        ("Selected date", actual_date.strftime("%d %b %Y"), "observation"),
        ("Map cells", f"{processed_count:,}", "selected observation"),
        ("Potential-risk", f"{risk_count:,}", "selected field"),
        ("Risk share", f"{share:.2f}%", "selected field"),
    ]
    for col, item in zip(cols, items):
        with col:
            metric(*item)

    st.markdown(
        """
        <div class="map-shell">
            <div class="map-heading">
                <b>Indian Ocean · Arabian Sea · Bay of Bengal</b>
                <span>Blue field · green concentration · red screening flags</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displaylogo": False,
            "scrollZoom": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )

    st.markdown(
        """
            <div class="legend-row">
                <span class="dot dot-blue"></span><span>Blue = processed ocean field</span>
                <span class="dot dot-green"></span><span>Green = flagged-cell concentration</span>
                <span class="dot dot-red"></span><span>Red = individual potential-risk cell</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Reading the map: green areas indicate spatial concentration of "
        "screening flags. They are not a separate severity score. Red cells "
        "are potential-risk screening results, not confirmed harmful algal blooms."
    )


# ============================================================
# LOCATION
# ============================================================

elif st.session_state.page == "location":

    heading(
        "02 · LOCATION INTELLIGENCE",
        "Check one coordinate across the available evidence.",
        "Your input is shown separately from the nearest processed 0.25° satellite cell, so there is no confusion about which location was actually evaluated.",
    )

    if "checked_lat" not in st.session_state:
        st.session_state.checked_lat = 18.0
    if "checked_lon" not in st.session_state:
        st.session_state.checked_lon = 78.0

    left, right = st.columns([.82, 1.18], gap="large")

    with left:
        st.markdown(
            """
            <div class="card">
                <div class="kicker">YOUR INPUT</div>
                <h3>Coordinates</h3>
                <p>Enter decimal degrees. Example: 17.38, 78.49.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("location_form", clear_on_submit=False):
            lat = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=float(st.session_state.checked_lat),
                step=.25,
                format="%.4f",
            )
            lon = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=float(st.session_state.checked_lon),
                step=.25,
                format="%.4f",
            )
            submitted = st.form_submit_button(
                "🔎 Check location",
                width="stretch",
                type="primary",
            )

        if submitted:
            st.session_state.checked_lat = float(lat)
            st.session_state.checked_lon = float(lon)
            st.rerun()

    lat = float(st.session_state.checked_lat)
    lon = float(st.session_state.checked_lon)
    row = nearest_latest(lat, lon)
    flagged = bool(row["risk_flag"])

    with right:
        st.markdown(
            f"""
            <div class="result-box">
                <div class="kicker">NEAREST PROCESSED CELL</div>
                <h3>{float(row.latitude):.4f}° · {float(row.longitude):.4f}°</h3>
                <p>
                    This is the satellite grid cell used for the lookup.<br>
                    <b>Your input:</b> {lat:.4f}° · {lon:.4f}°
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        a, b, c, d = st.columns(4, gap="small")
        with a:
            st.metric("Observation", row.date.strftime("%d %b %Y"))
        with b:
            st.metric("Chl-a", safe_num(row.chla))
        with c:
            st.metric(
                "Risk probability",
                probability_text(row.get("risk_probability", np.nan)),
            )
        with d:
            st.metric(
                "Region",
                region_name(float(row.latitude), float(row.longitude)),
            )

        if flagged:
            st.markdown(
                """
                <div class="status-risk">
                    <div class="status-title">🔴 POTENTIAL BLOOM-RISK FLAG</div>
                    This processed cell is included in the current screening shortlist.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="status-normal">
                    <div class="status-title">🟢 NOT FLAGGED</div>
                    This processed cell is not included in the current potential-risk shortlist.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("**Supporting signals**")

    signal_cols = st.columns(4, gap="small")
    signals = [
        ("Current Chl-a", safe_num(row.chla)),
        ("Historical baseline", safe_num(row.get("historical_baseline", np.nan))),
        ("Anomaly", safe_num(row.get("chla_anomaly", np.nan))),
        ("Recent change", safe_num(row.get("chla_change", np.nan))),
    ]

    for col, (label, value) in zip(signal_cols, signals):
        with col:
            st.markdown(
                f"""
                <div class="signal">
                    <div class="signal-label">{label}</div>
                    <div class="signal-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    report = pd.DataFrame([{
        "input_latitude": lat,
        "input_longitude": lon,
        "nearest_processed_latitude": row.latitude,
        "nearest_processed_longitude": row.longitude,
        "date": row.date.strftime("%Y-%m-%d"),
        "chla": row.chla,
        "historical_baseline": row.get("historical_baseline", np.nan),
        "chla_anomaly": row.get("chla_anomaly", np.nan),
        "chla_change": row.get("chla_change", np.nan),
        "risk_label": row.risk_label,
        "risk_probability": row.get("risk_probability", np.nan),
    }])

    st.download_button(
        "⬇ Download location report",
        report.to_csv(index=False).encode(),
        "bloomdetect_location_report.csv",
        "text/csv",
        width="stretch",
    )


# ============================================================
# INSIGHTS
# ============================================================

elif st.session_state.page == "insights":

    heading(
        "03 · OCEAN INTELLIGENCE",
        "What stands out in the latest field?",
        "Hotspot concentration and data insights are combined here because they answer the same question: where should the current satellite signal receive closer investigation?",
    )

    total_flags = int(latest["risk_flag"].sum())
    share = 100 * total_flags / len(latest) if len(latest) else 0

    cols = st.columns(4, gap="medium")
    insight_metrics = [
        ("Observation cells", f"{len(latest):,}", "latest field"),
        ("Potential-risk", f"{total_flags:,}", "screening output"),
        ("Risk share", f"{share:.2f}%", "latest field"),
        ("Mean Chl-a", safe_num(latest["chla"].mean()), "latest field"),
    ]
    for col, item in zip(cols, insight_metrics):
        with col:
            metric(*item)

    heading(
        "HOTSPOT INTELLIGENCE",
        "Where are the flags concentrating?",
        "Nearby screening flags are grouped into 2° × 2° investigation zones. This is a spatial concentration view, not a severity ranking.",
    )

    zones = hotspot_table()

    if not zones.empty:
        hz = latest[latest["risk_flag"]].copy()
        hz["lat_zone"] = np.floor(hz["latitude"] / 2) * 2 + 1
        hz["lon_zone"] = np.floor(hz["longitude"] / 2) * 2 + 1

        plot_zones = (
            hz.groupby(["lat_zone", "lon_zone"], as_index=False)
            .size()
            .rename(columns={"size": "flagged_cells"})
            .sort_values("flagged_cells", ascending=True)
            .tail(12)
        )

        fig = px.bar(
            plot_zones,
            x="flagged_cells",
            y=[
                f"{lat:.0f}°–{lat+2:.0f}°, {lon:.0f}°–{lon+2:.0f}°"
                for lat, lon in zip(plot_zones["lat_zone"], plot_zones["lon_zone"])
            ],
            orientation="h",
            text="flagged_cells",
        )
        fig.update_traces(
            marker_color="#20a477",
            textposition="outside",
            cliponaxis=False,
        )
        fig.update_layout(
            height=420,
            margin=dict(l=165, r=55, t=25, b=65),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#fff",
            showlegend=False,
            font=dict(color="#174b56", size=12),
            xaxis=dict(
                title="Potential-risk cells in zone",
                title_font=dict(size=14, color="#174b56"),
                tickfont=dict(size=11, color="#174b56"),
                gridcolor="#d6e7e8",
            ),
            yaxis=dict(
                title="Investigation zone",
                title_font=dict(size=14, color="#174b56"),
                tickfont=dict(size=11, color="#174b56"),
            ),
        )

        st.markdown('<div class="chart-box"><b>Top investigation zones</b></div>', unsafe_allow_html=True)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

        st.dataframe(zones, width="stretch", hide_index=True)

    else:
        st.success("No potential-risk cells are present in the latest study field.")

    heading(
        "CURRENT FIELD",
        "What does the latest observation look like?",
        "These charts describe the latest processed field without repeating the spatial map.",
    )

    c1, c2 = st.columns(2, gap="large")

    with c1:
        clean = latest["chla"].replace([np.inf, -np.inf], np.nan).dropna()
        fig1 = px.histogram(
            pd.DataFrame({"Chlorophyll-a": clean}),
            x="Chlorophyll-a",
            nbins=32,
        )
        fig1.update_traces(marker_color="#197da5")
        fig1.update_layout(
            height=380,
            margin=dict(l=70, r=25, t=25, b=75),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#fff",
            showlegend=False,
            font=dict(color="#174b56", size=12),
            xaxis=dict(
                title="Satellite-derived chlorophyll-a",
                title_font=dict(size=14, color="#174b56"),
                tickfont=dict(size=11, color="#174b56"),
                gridcolor="#d6e7e8",
            ),
            yaxis=dict(
                title="Number of processed cells",
                title_font=dict(size=14, color="#174b56"),
                tickfont=dict(size=11, color="#174b56"),
                gridcolor="#d6e7e8",
            ),
        )
        st.markdown('<div class="chart-box"><b>Chlorophyll-a distribution</b></div>', unsafe_allow_html=True)
        st.plotly_chart(fig1, width="stretch", config={"displaylogo": False})

    with c2:
        comp = pd.DataFrame({
            "Screening group": ["Normal", "Potential bloom risk"],
            "Mean chlorophyll-a": [
                latest.loc[~latest.risk_flag, "chla"].mean(),
                latest.loc[latest.risk_flag, "chla"].mean(),
            ],
        })
        fig2 = px.bar(
            comp,
            x="Screening group",
            y="Mean chlorophyll-a",
            text="Mean chlorophyll-a",
        )
        fig2.update_traces(
            marker_color=["#197da5", "#ef4f5e"],
            texttemplate="%{text:.4f}",
            textposition="outside",
            cliponaxis=False,
        )
        fig2.update_layout(
            height=380,
            margin=dict(l=70, r=25, t=25, b=75),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#fff",
            showlegend=False,
            font=dict(color="#174b56", size=12),
            xaxis=dict(
                title="Screening group",
                title_font=dict(size=14, color="#174b56"),
                tickfont=dict(size=11, color="#174b56"),
            ),
            yaxis=dict(
                title="Mean chlorophyll-a",
                title_font=dict(size=14, color="#174b56"),
                tickfont=dict(size=11, color="#174b56"),
                gridcolor="#d6e7e8",
            ),
        )
        st.markdown('<div class="chart-box"><b>Mean chlorophyll-a by screening group</b></div>', unsafe_allow_html=True)
        st.plotly_chart(fig2, width="stretch", config={"displaylogo": False})

    regional_defs = [
        ("Arabian Sea", latest.latitude.between(5, 30) & latest.longitude.between(45, 75)),
        ("Bay of Bengal", latest.latitude.between(0, 25) & latest.longitude.between(75.01, 100)),
        ("Southern Indian Ocean", latest.latitude.between(-30, 5) & latest.longitude.between(40, 100)),
        ("Northern Indian Ocean", latest.latitude.between(5, 30) & latest.longitude.between(75.01, 120)),
    ]

    rows = []
    for name, mask in regional_defs:
        sub = latest[mask]
        if len(sub):
            rows.append({
                "Region": name,
                "Cells": len(sub),
                "Potential-risk cells": int(sub["risk_flag"].sum()),
                "Risk share": f"{100 * sub['risk_flag'].mean():.2f}%",
                "Mean Chl-a": round(sub["chla"].mean(), 4),
            })

    if rows:
        heading(
            "REGIONAL SIGNAL",
            "Broad-area comparison.",
            "Descriptive regional summaries of the latest processed field.",
        )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.info(
        "Interpretation: these charts describe satellite-derived chlorophyll-a "
        "and the project screening output. They do not independently establish "
        "harmfulness, species identity or toxin presence."
    )

    if not zones.empty:
        st.download_button(
            "⬇ Download hotspot report",
            zones.to_csv(index=False).encode(),
            "bloomdetect_hotspot_report.csv",
            "text/csv",
            width="stretch",
        )


# ============================================================
# DATA
# ============================================================

elif st.session_state.page == "data":

    heading(
        "04 · DATA & OUTPUTS",
        "The evidence behind the dashboard.",
        "Keep the source product, current processed field and downloadable outputs in one place. No duplicate analysis panels here.",
    )

    cols = st.columns(4, gap="medium")
    data_metrics = [
        ("Product", "E06OCM_L4_AC", "EOS-06 / OCM-3"),
        ("Grid", "0.25°", "latitude × longitude"),
        ("Latest cells", f"{len(latest):,}", "processed observation"),
        ("Latest date", latest_date.strftime("%d %b %Y"), "processed dataset"),
    ]
    for col, item in zip(cols, data_metrics):
        with col:
            metric(*item)

    st.markdown(
        """
        <div class="card">
            <div class="kicker">SOURCE PRODUCT</div>
            <h3>EOS-06 OCM-3 analysed chlorophyll-a</h3>
            <p>
                The dashboard uses the EOS-06 / Oceansat-3 OCM-3 Level-4
                analysed chlorophyll product, E06OCM_L4_AC. The project output
                is a potential bloom-risk screening layer built from
                satellite-derived chlorophyll-a and temporal context.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    heading(
        "DOWNLOADS",
        "Take the actual project outputs.",
        "These downloads are generated directly from the data used by the dashboard.",
    )

    d1, d2 = st.columns(2, gap="large")

    with d1:
        st.markdown(
            """
            <div class="download-card">
                <h3>Latest observation table</h3>
                <p>All processed cells from the latest available field, including the stored screening result and supporting signals.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇ Download latest observations",
            latest.to_csv(index=False).encode(),
            "bloomdetect_latest_observations.csv",
            "text/csv",
            width="stretch",
        )

    with d2:
        st.markdown(
            """
            <div class="download-card">
                <h3>Potential-risk shortlist</h3>
                <p>Only cells currently screened as potential bloom risk in the latest processed field.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇ Download potential-risk locations",
            latest_risk.to_csv(index=False).encode(),
            "bloomdetect_potential_risk.csv",
            "text/csv",
            width="stretch",
        )

    heading(
        "SCIENTIFIC USE",
        "Use the screening result correctly.",
        "The dashboard is an early-warning support tool, not a confirmation system.",
    )

    st.info(
        "Potential bloom risk is a project screening output. It is not "
        "confirmation of a harmful algal bloom, species identity or toxin "
        "presence. Satellite observations should be combined with field "
        "observations and additional environmental evidence."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="footer">
        BloomDetect AI · EOS-06 / OCM-3 · Potential bloom-risk screening ·
        Latest processed field: {latest_date.strftime("%d %B %Y")}
    </div>
    """,
    unsafe_allow_html=True,
)
