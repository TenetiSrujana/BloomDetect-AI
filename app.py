from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# BLOOMDETECT AI
# Final Streamlit application
# Navigation: Home | Risk Map | Location | Insights | Data
# ============================================================

st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).resolve().parent
PREDICTION_FILE = BASE / "latest_bloom_risk_predictions.csv"
IMAGE_FILE = BASE / "bloomdetect_bloom_process.png"
HISTORY_FILES = [
    BASE / "bloomdetect_history_web.csv.gz",
    BASE / "bloomdetect_history.csv.gz",
    BASE / "bloomdetect_history.csv",
]

LAT_MIN, LAT_MAX = -40.0, 30.0
LON_MIN, LON_MAX = 20.0, 120.0
ANOMALY_THRESHOLD = 0.059076173328496344
CHANGE_THRESHOLD = 0.02007450088858604

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner="Loading BloomDetect field…")
def load_predictions():
    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            "latest_bloom_risk_predictions.csv was not found beside app.py."
        )

    df = pd.read_csv(PREDICTION_FILE)
    required = {"latitude", "longitude", "date", "chla", "risk_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Prediction file is missing: " + ", ".join(sorted(missing)))

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric = [
        "latitude", "longitude", "chla", "risk_probability", "model_score",
        "previous_chla", "historical_baseline", "recent_mean", "recent_max",
        "chla_anomaly", "chla_change",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["latitude", "longitude", "date", "chla"]).copy()
    df["risk_flag"] = (
        df["risk_label"].astype(str).str.strip().str.lower()
        == "potential bloom risk"
    )
    return df


@st.cache_data(show_spinner="Loading observation timeline…")
def load_history():
    source = next((p for p in HISTORY_FILES if p.exists()), None)
    if source is None:
        return pd.DataFrame(), None

    h = pd.read_csv(source)
    required = {"date", "lat_bin", "lon_bin", "chla", "risk"}
    if not required.issubset(h.columns):
        return pd.DataFrame(), None

    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    for col in ["lat_bin", "lon_bin", "chla", "risk", "cells", "max_chla"]:
        if col in h.columns:
            h[col] = pd.to_numeric(h[col], errors="coerce")
    h = h.replace([np.inf, -np.inf], np.nan)
    h = h.dropna(subset=["date", "lat_bin", "lon_bin", "chla"]).copy()
    h["risk"] = h["risk"].fillna(0).astype(int)
    h = h[
        h["lat_bin"].between(LAT_MIN, LAT_MAX)
        & h["lon_bin"].between(LON_MIN, LON_MAX)
    ].copy()
    return h, source.name


try:
    df = load_predictions()
except Exception as exc:
    st.error("BloomDetect AI could not load the project data.")
    st.code(str(exc))
    st.stop()

history, history_name = load_history()
history_available = not history.empty
latest_date = df["date"].max()
latest = df[df["date"].dt.normalize() == latest_date.normalize()].copy()
risk_latest = latest[latest["risk_flag"]].copy()

# ============================================================
# VISUAL SYSTEM
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root{
  --ink:#0b3e49;
  --ink2:#165a65;
  --teal:#079eaa;
  --teal2:#0b7280;
  --mint:#dff7f5;
  --mist:#f4fbfb;
  --line:#c4e3e5;
  --muted:#6b858b;
  --red:#e84e5d;
  --green:#22a878;
  --blue:#2387aa;
  --gold:#d8952e;
}

html,body,[data-testid="stAppViewContainer"]{
  background:#f5fbfb!important;
  color:var(--ink)!important;
  font-family:'DM Sans',sans-serif!important;
}
.stApp{
  background:
    radial-gradient(circle at 15% 0%,rgba(7,158,170,.07),transparent 28%),
    radial-gradient(circle at 90% 12%,rgba(34,168,120,.06),transparent 24%),
    #f5fbfb!important;
}
[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer,[data-testid="stSidebar"]{display:none!important;}
.block-container{max-width:1360px!important;padding:22px 34px 52px!important;margin:auto!important;}

/* Header */
.topbar{
  display:flex;align-items:center;justify-content:space-between;gap:20px;
  padding:16px 18px;background:rgba(255,255,255,.94);border:1px solid var(--line);
  border-radius:20px;box-shadow:0 12px 35px rgba(9,76,86,.07);
}
.brand{display:flex;align-items:center;gap:12px;min-width:0}.brandmark{
  width:48px;height:48px;border-radius:15px;display:grid;place-items:center;
  color:white;font:800 20px Manrope;background:linear-gradient(145deg,#12b9bd,#086778);
  box-shadow:0 8px 18px rgba(8,103,120,.22);
}.brandtitle{font:800 1.15rem Manrope;letter-spacing:-.04em;color:var(--ink)}
.brandsub{font-size:.68rem;color:var(--muted);margin-top:2px}.topmeta{text-align:right;font-size:.66rem;color:var(--muted)}
.topmeta strong{display:block;color:var(--ink2);font:700 .76rem Manrope;margin-top:2px}

/* Native Streamlit controls */
.stButton>button,.stDownloadButton>button{
  border:1px solid #8dbec4!important;border-radius:13px!important;background:white!important;
  color:var(--ink)!important;font-weight:700!important;min-height:43px!important;
  box-shadow:0 4px 13px rgba(10,76,87,.05)!important;transition:.15s ease!important;
}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--teal)!important;background:#e9fbfa!important;transform:translateY(-1px)}
.stButton>button[kind="primary"]{background:var(--ink)!important;color:white!important;border-color:var(--ink)!important}
.stButton>button[kind="primary"]:hover{background:var(--teal2)!important}
[data-baseweb="select"]>div,.stNumberInput input{
  border:1px solid #9fcbd0!important;border-radius:11px!important;background:#fff!important;
}
/* Location number fields: real editable text, clearly visible on the light UI. */
[data-testid="stNumberInput"] input,
.stNumberInput input{
  color:#0b3e49!important;
  -webkit-text-fill-color:#0b3e49!important;
  caret-color:#0b3e49!important;
  opacity:1!important;
  font-weight:600!important;
}
[data-testid="stNumberInput"] input::placeholder,
.stNumberInput input::placeholder{
  color:#8aa1a6!important;
  -webkit-text-fill-color:#8aa1a6!important;
  opacity:1!important;
}
[data-testid="stNumberInput"] button{
  color:#0b3e49!important;
  background:#f2f8f8!important;
  opacity:1!important;
}
[data-testid="stNumberInput"] button:hover{
  background:#dff7f5!important;
  color:#0b3e49!important;
}
/* Location inputs: keep labels visible and consistent with the BloomDetect theme. */
[data-testid="stNumberInput"] label,
.stNumberInput label{
  color:#0b3e49!important;font-family:'DM Sans',sans-serif!important;
  font-size:.78rem!important;font-weight:700!important;
  margin-bottom:4px!important;
}
[data-testid="stNumberInput"] label p,
.stNumberInput label p{color:#0b3e49!important;}
/* Streamlit alerts should use the same readable dark text as the rest of the app. */
[data-testid="stAlert"]{
  border-radius:14px!important;border:1px solid #c4e3e5!important;
  background:#eaf8f7!important;color:#0b3e49!important;
}
[data-testid="stAlert"] *{color:#0b3e49!important;}
[data-testid="stSlider"]{padding-top:4px!important}

/* Page typography */
.kicker{font:800 .63rem Manrope;letter-spacing:.17em;text-transform:uppercase;color:var(--teal);margin-bottom:8px}
.title{font:800 clamp(2.25rem,4.6vw,4.25rem)/1 Manrope;color:var(--ink);letter-spacing:-.065em;margin:0 0 12px}
.subtitle{font-size:.92rem;line-height:1.65;color:#5d7a81;max-width:940px;margin-bottom:22px}
.page-head{padding:28px 2px 8px}

/* Hero */
.hero{position:relative;overflow:hidden;min-height:420px;border-radius:30px;margin-top:20px;
  background:linear-gradient(135deg,#063c4e 0%,#086a78 48%,#0ba9aa 100%);
  box-shadow:0 25px 60px rgba(5,75,87,.18);padding:48px 52px;display:flex;align-items:center}
.hero:before{content:"";position:absolute;width:520px;height:520px;right:-180px;top:-210px;border:70px solid rgba(220,255,254,.09);border-radius:50%;box-shadow:0 0 0 55px rgba(220,255,254,.045),0 0 0 120px rgba(220,255,254,.025)}
.hero:after{content:"";position:absolute;left:-140px;bottom:-180px;width:380px;height:380px;border:1px solid rgba(255,255,255,.11);border-radius:50%}
.hero-content{position:relative;z-index:2;max-width:790px}.hero .kicker{color:#a6fffb}.hero h1{font:800 clamp(3.2rem,6.4vw,6rem)/.92 Manrope;color:#efffff;letter-spacing:-.08em;margin:0 0 20px}.hero p{color:#dcf7f8;font-size:1rem;line-height:1.7;max-width:740px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:22px}.chip{padding:8px 12px;border:1px solid rgba(255,255,255,.25);background:rgba(255,255,255,.11);border-radius:999px;color:#efffff;font-size:.7rem;font-weight:700}

/* Cards / metrics */
.metric{min-height:112px;padding:18px;border:1px solid var(--line);border-radius:18px;background:white;box-shadow:0 9px 25px rgba(10,78,89,.055)}
.metric-label{font:800 .58rem Manrope;letter-spacing:.13em;text-transform:uppercase;color:#759096}.metric-value{font:800 1.55rem Manrope;color:var(--ink);margin-top:5px;letter-spacing:-.035em}.metric-note{font-size:.67rem;color:var(--muted);margin-top:4px}
.card{padding:22px;border:1px solid var(--line);border-radius:20px;background:white;box-shadow:0 9px 25px rgba(10,78,89,.055)}
.card-title{font:800 1.1rem Manrope;color:var(--ink);margin-bottom:7px}.card-copy{font-size:.82rem;line-height:1.6;color:#668188}
.callout{padding:13px 15px;margin-top:14px;border-left:4px solid var(--teal);border-radius:0 12px 12px 0;background:#e8f8f8;color:#56767c;font-size:.75rem;line-height:1.55}.callout strong{color:var(--ink2)}
.status{padding:16px;border-radius:16px;border:1px solid;margin-top:16px}.status-risk{background:#fff1f3;border-color:#f19aa3;color:#8e2f3b}.status-ok{background:#eafaf3;border-color:#70c8aa;color:#16684f}.status-title{font:800 .9rem Manrope;margin-bottom:4px}

/* Map */
.map-shell{border:1px solid #a9d4d8;border-radius:22px;overflow:hidden;background:#e5f7f8;box-shadow:0 14px 35px rgba(9,80,92,.08)}
.map-header{display:flex;justify-content:space-between;gap:16px;align-items:center;padding:13px 16px;background:white;border-bottom:1px solid var(--line)}
.map-header strong{font:800 .82rem Manrope;color:var(--ink)}.map-header span{font-size:.66rem;color:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:15px;padding:11px 15px;background:white;border-top:1px solid var(--line);font-size:.68rem;color:#5d7a81}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}

/* Tables / charts */
.section-label{font:800 .61rem Manrope;letter-spacing:.16em;text-transform:uppercase;color:#0b3e49;margin:25px 0 9px}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden}
[data-testid="stDataFrame"] *{color:#0b3e49!important}
.chart-card{padding:4px 0 0}

.footer{margin-top:38px;padding-top:13px;border-top:1px solid #d6e9ea;color:#789197;font-size:.64rem}
.small-muted{font-size:.69rem;color:var(--muted)}

@media(max-width:900px){.block-container{padding:15px 18px 40px!important}.hero{padding:36px 27px;min-height:400px}.hero h1{font-size:3.3rem}.topmeta{display:none}}
@media(max-width:620px){.block-container{padding:10px 11px 32px!important}.hero{padding:31px 22px}.hero h1{font-size:2.8rem}.map-header{display:block}.map-header span{display:block;margin-top:5px}}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================

def fmt_date(value):
    return pd.to_datetime(value).strftime("%d %b %Y")


def fmt_num(value, digits=4):
    try:
        if pd.isna(value):
            return "Unavailable"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "Unavailable"


def metric(label, value, note):
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def page_head(kicker, title, copy):
    st.markdown(
        f'<div class="page-head"><div class="kicker">{kicker}</div><h1 class="title">{title}</h1><div class="subtitle">{copy}</div></div>',
        unsafe_allow_html=True,
    )


def card(title, copy, extra=""):
    st.markdown(
        f'<div class="card"><div class="card-title">{title}</div><div class="card-copy">{copy}</div>{extra}</div>',
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        f'<div class="footer">BloomDetect AI · EOS-06 / Oceansat-3 OCM-3 · Potential bloom-risk screening · Latest processed field: {fmt_date(latest_date)}</div>',
        unsafe_allow_html=True,
    )


def geo_style():
    return dict(
        projection_type="mercator",
        showland=True,
        landcolor="#dce8e7",
        showocean=True,
        oceancolor="#e4f7f8",
        showcountries=True,
        countrycolor="#9abdc2",
        coastlinecolor="#6fa6ae",
        showlakes=True,
        lakecolor="#e4f7f8",
        bgcolor="#e4f7f8",
        lonaxis=dict(range=[LON_MIN, LON_MAX], dtick=10, showgrid=True, gridcolor="#c7e3e5"),
        lataxis=dict(range=[LAT_MIN, LAT_MAX], dtick=10, showgrid=True, gridcolor="#c7e3e5"),
    )


def base_map(height=620):
    fig = go.Figure()
    fig.update_geos(**geo_style())
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#e4f7f8",
        plot_bgcolor="#e4f7f8",
        font=dict(family="DM Sans", color="#0b3e49"),
        legend=dict(
            orientation="h", y=0.01, x=0.02,
            bgcolor="rgba(255,255,255,.92)", bordercolor="#b9dfe2", borderwidth=1,
        ),
        hoverlabel=dict(bgcolor="#0b3e49", font_color="white"),
    )
    return fig


def latest_map(frame, view):
    fig = base_map()
    field = frame.copy()
    if len(field) > 16000:
        field = field.sample(16000, random_state=42)

    if view == "Screening":
        fig.add_trace(go.Scattergeo(
            lon=field.longitude, lat=field.latitude, mode="markers",
            name="Processed field",
            marker=dict(size=3.8, color="#2387aa", opacity=.30),
            hovertemplate="Lat %{lat:.3f}°<br>Lon %{lon:.3f}°<br>Processed ocean cell<extra></extra>",
        ))
        flagged = frame[frame.risk_flag].copy()
        if not flagged.empty:
            fig.add_trace(go.Scattergeo(
                lon=flagged.longitude, lat=flagged.latitude, mode="markers",
                name="Potential-risk cell",
                marker=dict(size=7.5, color="#e84e5d", opacity=.92, line=dict(width=.6, color="white")),
                customdata=np.c_[
                    flagged.chla,
                    flagged["risk_probability"] if "risk_probability" in flagged else np.nan,
                ],
                hovertemplate=(
                    "Lat %{lat:.3f}°<br>Lon %{lon:.3f}°<br>"
                    "Chl-a %{customdata[0]:.4f}<br>"
                    "Screening probability %{customdata[1]:.1%}<extra></extra>"
                ),
            ))
    else:
        column = {
            "Chlorophyll-a": "chla",
            "Recent change": "chla_change",
            "Historical anomaly": "chla_anomaly",
        }[view]
        if column not in frame.columns:
            return None
        plot = frame.dropna(subset=[column]).copy()
        if plot.empty:
            return None
        if len(plot) > 16000:
            plot = plot.sample(16000, random_state=42)
        scale = "YlGnBu" if view == "Chlorophyll-a" else "RdBu_r"
        title = "Chl-a" if view == "Chlorophyll-a" else "Signal"
        fig.add_trace(go.Scattergeo(
            lon=plot.longitude, lat=plot.latitude, mode="markers", name=view,
            marker=dict(
                size=4.2, color=plot[column], colorscale=scale, opacity=.78,
                colorbar=dict(title=title, thickness=13),
            ),
            customdata=np.c_[plot.chla, plot[column]],
            hovertemplate="Lat %{lat:.3f}°<br>Lon %{lon:.3f}°<br>Chl-a %{customdata[0]:.4f}<br>Signal %{customdata[1]:.4f}<extra></extra>",
        ))
        flagged = frame[frame.risk_flag].copy()
        if not flagged.empty:
            fig.add_trace(go.Scattergeo(
                lon=flagged.longitude, lat=flagged.latitude, mode="markers",
                name="Potential-risk cells",
                marker=dict(size=6.5, color="#e84e5d", opacity=.9),
                hovertemplate="Potential bloom-risk screening cell<extra></extra>",
            ))
    return fig


def history_map(selected):
    fig = base_map()
    field = selected.copy()
    if len(field) > 18000:
        field = field.sample(18000, random_state=42)

    fig.add_trace(go.Scattergeo(
        lon=field.lon_bin, lat=field.lat_bin, mode="markers",
        name="Processed field",
        marker=dict(size=5.4, color="#2387aa", opacity=.38),
        customdata=field.chla,
        hovertemplate="Lat %{lat:.1f}°<br>Lon %{lon:.1f}°<br>Chl-a %{customdata:.4f}<extra></extra>",
    ))
    high = selected[selected.chla >= selected.chla.quantile(.90)].copy()
    if not high.empty:
        if len(high) > 4500:
            high = high.sample(4500, random_state=42)
        fig.add_trace(go.Scattergeo(
            lon=high.lon_bin, lat=high.lat_bin, mode="markers",
            name="Higher concentration",
            marker=dict(size=7, color="#22a878", opacity=.52),
            hovertemplate="Higher Chl-a concentration<extra></extra>",
        ))
    flagged = selected[selected.risk == 1]
    if not flagged.empty:
        fig.add_trace(go.Scattergeo(
            lon=flagged.lon_bin, lat=flagged.lat_bin, mode="markers",
            name="Potential-risk cell",
            marker=dict(size=9, color="#e84e5d", opacity=.94, line=dict(width=.5, color="white")),
            hovertemplate="Potential bloom-risk screening cell<extra></extra>",
        ))
    return fig


def nearest_ocean_cell(lat, lon):
    valid = latest[
        latest.chla.notna()
        & latest.latitude.between(LAT_MIN, LAT_MAX)
        & latest.longitude.between(LON_MIN, LON_MAX)
        & (latest.chla > 0)
    ].copy()
    if valid.empty:
        return None, None
    lat_value = float(lat)
    lon_value = float(lon)
    distance = ((valid.latitude - lat_value) / 70.0) ** 2 + ((valid.longitude - lon_value) / 100.0) ** 2
    idx = distance.idxmin()
    row = valid.loc[idx]
    separation = float(np.hypot(float(row.latitude) - lat_value, float(row.longitude) - lon_value))
    return row, separation


def go_to(page):
    st.session_state.page = page
    st.rerun()


# ============================================================
# HEADER / NAVIGATION
# ============================================================

st.markdown(
    f'<div class="topbar"><div class="brand"><div class="brandmark">≈</div><div><div class="brandtitle">BloomDetect AI</div><div class="brandsub">Coastal & Ocean Intelligence · EOS-06 OCM-3</div></div></div><div class="topmeta">Latest processed field<strong>{fmt_date(latest_date)}</strong></div></div>',
    unsafe_allow_html=True,
)

if st.session_state.get("page") not in {"home", "map", "location", "insights", "data"}:
    st.session_state.page = "home"

# Small clean space between header and navigation buttons
st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

pages = [
    ("home", "⌂ Home"),
    ("map", "◉ Risk Map"),
    ("location", "⌖ Location"),
    ("insights", "▥ Insights"),
    ("data", "↓ Data"),
]

nav = st.columns(5)

for col, (key, label) in zip(nav, pages):
    with col:
        if st.button(
            label,
            key="nav_" + key,
            width="stretch",
            type="primary" if st.session_state.page == key else "secondary",
        ):
            go_to(key)
            
# ============================================================
# HOME
# ============================================================

if st.session_state.page == "home":
    st.markdown(
        '''<section class="hero"><div class="hero-content"><div class="kicker">EOS-06 · OCM-3 · Satellite intelligence</div><h1>Read the ocean signal.</h1><p>BloomDetect AI screens satellite-derived chlorophyll-a and temporal behaviour to surface locations whose patterns may deserve closer investigation. It is an early-warning support system, not a claim of confirmed harmful algal bloom detection.</p><div class="chips"><span class="chip">🌊 Ocean colour</span><span class="chip">🛰 EOS-06 OCM-3</span><span class="chip">📍 Spatial screening</span><span class="chip">⚠ Potential bloom-risk signal</span></div></div></section>''',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric("Processed cells", f"{len(latest):,}", "latest processed field")

    with c2:
        metric("Potential-risk cells", f"{len(risk_latest):,}", "current screening")

    with c3:
        metric(
            "Risk share",
            f"{(len(risk_latest)/len(latest)*100 if len(latest) else 0):.2f}%",
            "of processed cells",
        )

    with c4:
        metric("Maximum Chl-a", fmt_num(latest.chla.max()), "latest field")

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.05, .95], gap="large")

    with left:
        card(
            "From satellite observation to investigation support",
            "BloomDetect combines current chlorophyll-a with temporal context such as the previous observation, historical baseline and recent behaviour. The resulting flag is a potential bloom-risk screening signal.",
            '<div class="callout"><strong>Scientific boundary:</strong> high chlorophyll-a alone does not prove a harmful algal bloom. Species, toxin presence and ecological impact require additional evidence and field validation.</div>',
        )

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="card"><div class="card-title">A focused workflow</div><div class="card-copy">Use <b>Risk Map</b> for the spatial field, <b>Location</b> for one coordinate, <b>Insights</b> for patterns, and <b>Data</b> for source files and project evidence. Each view has one job, so the same information does not keep haunting you across five pages.</div></div>',
            unsafe_allow_html=True,
        )

    with right:
        if IMAGE_FILE.exists():
            st.image(
                str(IMAGE_FILE),
                width="stretch",
                caption="Satellite ocean-colour observations supporting bloom-risk investigation",
            )
        else:
            card(
                "Project visual",
                "The project illustration is not present beside app.py, so no substitute image is being invented.",
            )

    if history_available:
        first_hist = history.date.min()
        last_hist = history.date.max()

        st.markdown(
            f'<div class="callout"><strong>Timeline:</strong> the available compact historical screening layer covers {fmt_date(first_hist)} to {fmt_date(last_hist)}. Earlier dates are historical screening context, not additional ML model predictions.</div>',
            unsafe_allow_html=True,
        )

    footer()

# ============================================================
# RISK MAP
# ============================================================

elif st.session_state.page == "map":

    page_head(
        "01 · SPATIAL INTELLIGENCE",
        "See the field, then change the lens.",
        "Explore the study window across the available observation timeline. Blue shows the processed ocean field, green highlights higher concentration zones, and red marks individual potential-risk screening cells.",
    )

    if history_available:

        # --------------------------------------------------------
        # OBSERVATION TIMELINE
        # --------------------------------------------------------

        dates = sorted(
            history.date.dt.normalize().dropna().unique()
        )

        date_values = [
            pd.Timestamp(d).date()
            for d in dates
        ]

        selected_date = st.select_slider(
            "Observation date",
            options=date_values,
            value=date_values[-1],
            format_func=lambda x: pd.Timestamp(x).strftime(
                "%d %b %Y"
            ),
            key="timeline_date",
        )

        selected_ts = pd.Timestamp(selected_date)

        selected = history[
            history.date.dt.normalize()
            == selected_ts.normalize()
        ].copy()

        is_latest = (
            selected_ts.normalize()
            == latest_date.normalize()
        )

        # ========================================================
        # SAME MAP FORMAT FOR ALL DATES
        # ========================================================

        if is_latest:

            # ----------------------------------------------------
            # ACTUAL LATEST FIELD
            # ----------------------------------------------------

            processed_count = len(latest)

            risk_count = len(risk_latest)

            risk_share = (
                risk_count / processed_count * 100
                if processed_count
                else 0
            )

            # ----------------------------------------------------
            # Convert latest full-resolution field into the same
            # compact spatial representation used by history.
            # ----------------------------------------------------

            latest_compact = latest[
                [
                    "latitude",
                    "longitude",
                    "chla",
                ]
            ].copy()

            latest_compact = latest_compact.dropna(
                subset=[
                    "latitude",
                    "longitude",
                    "chla",
                ]
            )

            latest_compact = latest_compact[
                latest_compact["chla"] > 0
            ].copy()

            # Same spatial binning used for the timeline display.
            latest_compact["lat_bin"] = (
                latest_compact["latitude"].round()
            )

            latest_compact["lon_bin"] = (
                latest_compact["longitude"].round()
            )

            latest_compact = (
                latest_compact
                .groupby(
                    [
                        "lat_bin",
                        "lon_bin",
                    ],
                    as_index=False,
                )
                .agg(
                    chla=("chla", "mean"),
                )
            )

            # ----------------------------------------------------
            # Mark compact cells containing at least one actual
            # latest potential-risk cell.
            # ----------------------------------------------------

            latest_risk = latest[
                latest["risk_flag"].astype(bool)
            ][
                [
                    "latitude",
                    "longitude",
                ]
            ].copy()

            if not latest_risk.empty:

                latest_risk["lat_bin"] = (
                    latest_risk["latitude"].round()
                )

                latest_risk["lon_bin"] = (
                    latest_risk["longitude"].round()
                )

                risk_bins = (
                    latest_risk[
                        [
                            "lat_bin",
                            "lon_bin",
                        ]
                    ]
                    .drop_duplicates()
                    .assign(risk=1)
                )

                latest_compact = latest_compact.merge(
                    risk_bins,
                    on=[
                        "lat_bin",
                        "lon_bin",
                    ],
                    how="left",
                )

                latest_compact["risk"] = (
                    latest_compact["risk"]
                    .fillna(0)
                    .astype(int)
                )

            else:

                latest_compact["risk"] = 0

            # ----------------------------------------------------
            # SAME MAP RENDERER AS HISTORICAL DATES
            # ----------------------------------------------------

            map_fig = history_map(
                latest_compact
            )

            timeline_note = (
                "Latest field uses the stored final screening "
                "output and is displayed on the same compact "
                "spatial grid used across the historical timeline."
            )

            # ----------------------------------------------------
            # CURRENT RISK CELLS FOR SPATIAL ANALYSIS
            # ----------------------------------------------------

            current_risk_for_analysis = latest[
                latest["risk_flag"].astype(bool)
            ].copy()

            current_risk_for_analysis["lat_bin"] = (
                current_risk_for_analysis["latitude"].round()
            )

            current_risk_for_analysis["lon_bin"] = (
                current_risk_for_analysis["longitude"].round()
            )

        else:

            # ----------------------------------------------------
            # HISTORICAL FIELD
            # ----------------------------------------------------

            processed_count = len(selected)

            risk_count = int(
                selected["risk"].sum()
            )

            risk_share = (
                risk_count / processed_count * 100
                if processed_count
                else 0
            )

            # Same renderer as latest date.
            map_fig = history_map(
                selected
            )

            timeline_note = (
                "Earlier dates use compact historical Chl-a "
                "screening, not another ML prediction."
            )

            # Historical risk cells already use the compact grid.
            current_risk_for_analysis = selected[
                selected["risk"].astype(bool)
            ].copy()

        # ========================================================
        # METRIC CARDS
        # ========================================================

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric(
                "Selected date",
                fmt_date(selected_ts),
                "observation",
            )

        with c2:
            metric(
                "Processed cells",
                f"{processed_count:,}",
                "selected field",
            )

        with c3:
            metric(
                "Potential-risk",
                f"{risk_count:,}",
                "screening cells",
            )

        with c4:
            metric(
                "Risk share",
                f"{risk_share:.2f}%",
                "selected field",
            )

        # ========================================================
        # MAP CONTAINER
        # ========================================================

        st.markdown(
            '<div class="map-shell">'
            '<div class="map-header">'
            '<strong>'
            'Indian Ocean · Arabian Sea · Bay of Bengal'
            '</strong>'
            '<span>'
            'Blue processed field · green concentration · '
            'red potential-risk screening'
            '</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            map_fig,
            width="stretch",
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "responsive": True,
            },
        )

        st.markdown(
            '<div class="legend">'
            '<span>'
            '<i style="background:#2387aa"></i>'
            'Processed ocean field'
            '</span>'
            '<span>'
            '<i style="background:#22a878"></i>'
            'Higher concentration'
            '</span>'
            '<span>'
            '<i style="background:#e84e5d"></i>'
            'Potential-risk cell'
            '</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )

        # ========================================================
        # SPATIAL PATTERN
        # ========================================================

        if not current_risk_for_analysis.empty:

            cells = set()

            for lat, lon in zip(
                current_risk_for_analysis["lat_bin"],
                current_risk_for_analysis["lon_bin"],
            ):

                if pd.notna(lat) and pd.notna(lon):

                    cells.add(
                        (
                            int(round(float(lat))),
                            int(round(float(lon))),
                        )
                    )

            clusters = 0
            largest_cluster = 0
            remaining = set(cells)

            while remaining:

                start = remaining.pop()

                stack = [start]

                cluster_size = 1

                while stack:

                    r, c = stack.pop()

                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):

                            if dr == 0 and dc == 0:
                                continue

                            neighbour = (
                                r + dr,
                                c + dc,
                            )

                            if neighbour in remaining:

                                remaining.remove(
                                    neighbour
                                )

                                stack.append(
                                    neighbour
                                )

                                cluster_size += 1

                clusters += 1

                largest_cluster = max(
                    largest_cluster,
                    cluster_size,
                )

            concentration = (
                largest_cluster
                / len(cells)
                * 100
                if cells
                else 0
            )

            pattern_text = (
                f"The selected field contains "
                f"<b>{clusters}</b> spatial risk "
                f"cluster"
                f"{'s' if clusters != 1 else ''}. "
                f"The largest cluster contains "
                f"<b>{largest_cluster}</b> grid cell"
                f"{'s' if largest_cluster != 1 else ''} "
                f"({concentration:.1f}% of clustered "
                f"risk cells). "
                f"This describes spatial concentration only; "
                f"it is not a forecast of bloom movement."
            )

        else:

            pattern_text = (
                "No potential-risk cells are present in the "
                "selected field, so no spatial risk cluster "
                "is reported."
            )

        st.markdown(
            f'<div class="callout">'
            f'<strong>Spatial pattern:</strong> '
            f'{pattern_text}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ========================================================
        # RISK PERSISTENCE
        # ========================================================

        current_cells = set()

        for lat, lon in zip(
            current_risk_for_analysis["lat_bin"],
            current_risk_for_analysis["lon_bin"],
        ):

            if pd.notna(lat) and pd.notna(lon):

                current_cells.add(
                    (
                        int(round(float(lat))),
                        int(round(float(lon))),
                    )
                )

        # Find the previous observation date.
        previous_dates = [
            d
            for d in date_values
            if pd.Timestamp(d) < selected_ts
        ]

        if previous_dates and current_cells:

            previous_date = pd.Timestamp(
                previous_dates[-1]
            )

            previous = history[
                history.date.dt.normalize()
                == previous_date.normalize()
            ].copy()

            previous_risk = previous[
                previous["risk"].astype(bool)
            ].copy()

            previous_cells = set()

            for lat, lon in zip(
                previous_risk["lat_bin"],
                previous_risk["lon_bin"],
            ):

                if pd.notna(lat) and pd.notna(lon):

                    previous_cells.add(
                        (
                            int(round(float(lat))),
                            int(round(float(lon))),
                        )
                    )

            persistent_cells = (
                current_cells
                & previous_cells
            )

            persistence_rate = (
                len(persistent_cells)
                / len(current_cells)
                * 100
                if current_cells
                else 0
            )

            persistence_text = (
                f"<b>{persistence_rate:.1f}%</b> of the "
                f"current potential-risk grid cells were "
                f"also screened in the previous observation "
                f"({previous_date.strftime('%d %b %Y')}). "
                f"This indicates spatial persistence in the "
                f"screening signal, not confirmed bloom persistence."
            )

        elif not previous_dates:

            persistence_text = (
                "No previous observation is available for "
                "this date, so persistence cannot be calculated."
            )

        else:

            persistence_text = (
                "No potential-risk cells are present in the "
                "selected field, so persistence is not reported."
            )

        st.markdown(
            f'<div class="callout">'
            f'<strong>Risk persistence:</strong> '
            f'{persistence_text}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ========================================================
        # HOW TO READ
        # ========================================================

        st.markdown(
            f'<div class="callout">'
            f'<strong>How to read this:</strong> '
            f'{timeline_note} '
            f'Red cells are screening results, not confirmed '
            f'harmful algal blooms.'
            f'</div>',
            unsafe_allow_html=True,
        )

    else:

        st.info(
            "The compact history file is not present in this "
            "deployment. The latest processed field is still "
            "available below. No fake timeline is being created."
        )

        view = st.selectbox(
            "Map detail",
            [
                "Screening",
                "Chlorophyll-a",
                "Recent change",
                "Historical anomaly",
            ],
            key="map_view",
        )

        fig = latest_map(
            latest,
            view,
        )

        if fig is not None:

            st.markdown(
                '<div class="map-shell">'
                '<div class="map-header">'
                '<strong>'
                'Indian Ocean · Arabian Sea · Bay of Bengal'
                '</strong>'
                '<span>'
                'Study window: 40°S–30°N · 20°E–120°E'
                '</span>'
                '</div>',
                unsafe_allow_html=True,
            )

            st.plotly_chart(
                fig,
                width="stretch",
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "responsive": True,
                },
            )

            st.markdown(
                '<div class="legend">'
                '<span>'
                '<i style="background:#2387aa"></i>'
                'Processed ocean field'
                '</span>'
                '<span>'
                '<i style="background:#e84e5d"></i>'
                'Potential-risk cell'
                '</span>'
                '</div></div>',
                unsafe_allow_html=True,
            )

    footer()


# ============================================================
# LOCATION
# ============================================================

elif st.session_state.page == "location":

    page_head(
        "02 · LOCATION INTELLIGENCE",
        "Interrogate one coordinate.",
        "Enter a real coordinate inside the study window. BloomDetect keeps your input separate from the nearest valid processed ocean cell and reports only evidence available in the dataset.",
    )

    # ========================================================
    # INPUT + MAIN RESULT
    # ========================================================

    left, right = st.columns([0.82, 1.18], gap="large")

    # --------------------------------------------------------
    # LEFT COLUMN : INPUT
    # --------------------------------------------------------

    with left:

        card(
            "Your input",
            "Use decimal degrees. The fields start at the study-window boundary and can be edited before checking the coordinate."
        )

        lat = st.number_input(
            "Latitude",
            min_value=float(LAT_MIN),
            max_value=float(LAT_MAX),
            value=float(LAT_MIN),
            step=0.01,
            format="%.4f",
            key="lookup_lat",
        )

        lon = st.number_input(
            "Longitude",
            min_value=float(LON_MIN),
            max_value=float(LON_MAX),
            value=float(LON_MIN),
            step=0.01,
            format="%.4f",
            key="lookup_lon",
        )

        check = st.button(
            "Check coordinate",
            width="stretch",
            type="primary"
        )

        st.markdown(
            f"""
            <div class="small-muted">
                Study window: {LAT_MIN:.0f}° to {LAT_MAX:.0f}° latitude
                · {LON_MIN:.0f}° to {LON_MAX:.0f}° longitude
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # FIND COORDINATE
    # --------------------------------------------------------

    row = None
    separation = None
    lookup_error = None

    if check:

        if not (
            LAT_MIN <= float(lat) <= LAT_MAX
            and LON_MIN <= float(lon) <= LON_MAX
        ):
            lookup_error = (
                f"This coordinate is outside the study window: "
                f"{LAT_MIN}° to {LAT_MAX}° latitude and "
                f"{LON_MIN}° to {LON_MAX}° longitude."
            )

        else:
            row, separation = nearest_ocean_cell(
                float(lat),
                float(lon)
            )

            if row is None:
                lookup_error = (
                    "No valid processed ocean cell is available "
                    "for this lookup."
                )

    # --------------------------------------------------------
    # RIGHT COLUMN : RESULT
    # --------------------------------------------------------

    with right:

        if not check:

            card(
                "Nearest processed ocean cell",
                "Enter a coordinate and run the lookup. Results will appear here using only processed dataset observations."
            )

        elif lookup_error:

            st.error(lookup_error)

        else:

            flagged = bool(row.risk_flag)

            card(
                "Nearest processed ocean cell",
                f"""
                <b>{row.latitude:.4f}° · {row.longitude:.4f}°</b><br>
                Your input: {float(lat):.4f}° · {float(lon):.4f}°<br>
                Approximate angular separation: {separation:.2f}°
                """
            )

            # ------------------------------------------------
            # OBSERVATION DETAILS
            # ------------------------------------------------

            a, b = st.columns(2)

            with a:
                metric(
                    "Observation date",
                    fmt_date(row.date),
                    "nearest valid cell"
                )

            with b:
                metric(
                    "Chlorophyll-a",
                    fmt_num(row.chla),
                    "processed observation"
                )

            c, d = st.columns(2)

            with c:

                prob = row.get(
                    "risk_probability",
                    np.nan
                )

                metric(
                    "Screening probability",
                    (
                        f"{float(prob):.1%}"
                        if pd.notna(prob)
                        else "Unavailable"
                    ),
                    "stored model output"
                )

            with d:

                metric(
                    "Screening status",
                    "Potential risk" if flagged else "Not flagged",
                    "latest field"
                )

            # ------------------------------------------------
            # SCREENING STATUS
            # ------------------------------------------------

            if flagged:

                st.markdown(
                    """
                    <div class="status status-risk">
                        <div class="status-title">
                            🔴 Potential bloom-risk screening
                        </div>
                        This processed cell is included in the latest
                        potential-risk screening output.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="status status-ok">
                        <div class="status-title">
                            🟢 Not flagged
                        </div>
                        This valid processed ocean cell is not included
                        in the latest potential-risk screening output.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ========================================================
    # SUPPORTING SIGNALS
    # Placed below the two-column top section so the left side
    # is no longer a giant empty space.
    # ========================================================

    if check and row is not None and lookup_error is None:

        st.markdown(
            '<div class="section-label">Supporting signals</div>',
            unsafe_allow_html=True
        )

        values = [
            (
                "Current Chl-a",
                row.get("chla", np.nan)
            ),
            (
                "Historical baseline",
                row.get("historical_baseline", np.nan)
            ),
            (
                "Anomaly",
                row.get("chla_anomaly", np.nan)
            ),
            (
                "Recent change",
                row.get("chla_change", np.nan)
            ),
        ]

        signal_cols = st.columns(4, gap="medium")

        for col, (label, value) in zip(signal_cols, values):

            with col:

                metric(
                    label,
                    fmt_num(value),
                    "available signal"
                )

    # ========================================================
    # LOCATION HISTORY
    # Full-width so the graph is not squeezed into the
    # right-hand column.
    # ========================================================

    if (
        check
        and row is not None
        and lookup_error is None
        and history_available
    ):

        hlat = float(row.latitude)
        hlon = float(row.longitude)

        # IMPORTANT:
        # Parentheses prevent pandas '&' precedence problems.
        same = history[
            (
                (history.lat_bin - hlat).abs() <= 1.0
            )
            &
            (
                (history.lon_bin - hlon).abs() <= 1.0
            )
        ].copy()

        if not same.empty:

            trend = (
                same
                .groupby("date", as_index=False)
                .chla
                .mean()
                .sort_values("date")
            )

            if len(trend) >= 2:

                st.markdown(
                    '<div class="section-label">Location history</div>',
                    unsafe_allow_html=True
                )

                # ------------------------------------------------
                # FULL-WIDTH HISTORY CHART
                # ------------------------------------------------

                fig = px.line(
                    trend,
                    x="date",
                    y="chla",
                    markers=True,
                )

                fig.update_traces(
                    line=dict(
                        color="#78BDF2",
                        width=3
                    ),
                    marker=dict(
                        size=7,
                        color="#78BDF2"
                    ),
                    hovertemplate=(
                        "<b>%{x|%d %b %Y}</b>"
                        "<br>Mean Chl-a: %{y:.4f}"
                        "<extra></extra>"
                    ),
                )

                fig.update_layout(
                    height=360,

                    margin=dict(
                        l=65,
                        r=25,
                        t=20,
                        b=65
                    ),

                    paper_bgcolor="white",
                    plot_bgcolor="white",

                    font=dict(
                        family="DM Sans",
                        color="#0B3E49"
                    ),

                    # --------------------------------------------
                    # X AXIS
                    # --------------------------------------------

                    xaxis=dict(
                        title=dict(
                            text="Observation date",
                            font=dict(
                                color="#111111",
                                size=14
                            )
                        ),

                        tickfont=dict(
                            color="#111111",
                            size=12
                        ),

                        showline=True,
                        linecolor="#111111",
                        linewidth=1.5,

                        showgrid=True,
                        gridcolor="#E5E7EB",

                        zeroline=False,
                    ),

                    # --------------------------------------------
                    # Y AXIS
                    # --------------------------------------------

                    yaxis=dict(
                        title=dict(
                            text="Mean Chl-a",
                            font=dict(
                                color="#111111",
                                size=14
                            )
                        ),

                        tickfont=dict(
                            color="#111111",
                            size=12
                        ),

                        showline=True,
                        linecolor="#111111",
                        linewidth=1.5,

                        showgrid=True,
                        gridcolor="#E5E7EB",

                        zeroline=False,
                    ),

                    hoverlabel=dict(
                        bgcolor="white",
                        font=dict(
                            color="#0B3E49"
                        )
                    ),
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                    config={
                        "displaylogo": False,
                        "responsive": True
                    }
                )

    # ========================================================
    # SCIENTIFIC BOUNDARY
    # ========================================================

    if check and row is not None and lookup_error is None:

        st.markdown(
            """
            <div class="callout">
                <strong>Scientific boundary:</strong>
                this lookup is a satellite screening aid. It cannot
                independently establish harmfulness, species identity,
                toxin presence or ecological impact.
            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # FOOTER
    # ========================================================

    footer()


# ============================================================
# INSIGHTS
# ============================================================

elif st.session_state.page == "insights":
    page_head(
        "03 · OCEAN INTELLIGENCE",
        "Find the patterns that matter.",
        "The latest processed field is summarized through spatial concentration, Chl-a distribution, change and current screening priorities. Each chart answers a different question and uses only stored project data.",
    )

    change_series = pd.to_numeric(latest.get("chla_change", pd.Series(index=latest.index, dtype=float)), errors="coerce")
    anomaly_series = pd.to_numeric(latest.get("chla_anomaly", pd.Series(index=latest.index, dtype=float)), errors="coerce")
    positive_change = int((change_series > 0).sum())
    anomaly_count = int((anomaly_series >= ANOMALY_THRESHOLD).sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric("Potential-risk cells", f"{len(risk_latest):,}", "latest screening")
    with c2: metric("Positive Chl-a change", f"{positive_change:,}", "change > 0")
    with c3: metric("Anomalous cells", f"{anomaly_count:,}", f"anomaly ≥ {ANOMALY_THRESHOLD:.4f}")
    with c4: metric("Normal screening field", f"{max(len(latest)-len(risk_latest),0):,}", "not currently flagged")

    # Spatial concentration
    st.markdown('<div class="section-label">Spatial concentration</div>', unsafe_allow_html=True)
    if not risk_latest.empty:
        zone = risk_latest.copy()
        zone["lat_zone"] = np.floor(zone.latitude / 5) * 5
        zone["lon_zone"] = np.floor(zone.longitude / 5) * 5
        zone = zone.groupby(["lat_zone", "lon_zone"], as_index=False).size().rename(columns={"size": "cells"})
        zone["zone"] = zone.apply(lambda r: f"{r.lat_zone:.0f}°–{r.lat_zone+5:.0f}° · {r.lon_zone:.0f}°–{r.lon_zone+5:.0f}°", axis=1)
        zone = zone.nlargest(10, "cells").sort_values("cells")
        fig = px.bar(zone, x="cells", y="zone", orientation="h", text="cells")
        fig.update_traces(marker_color="#e84e5d", textposition="outside")
        fig.update_layout(
            height=430,
            margin=dict(l=150, r=45, t=20, b=60),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            font=dict(family="DM Sans", color="#0b3e49"),
            showlegend=False,
            xaxis=dict(
                title=dict(text="Potential-risk screening cells", font=dict(color="#0b3e49", size=13)),
                tickfont=dict(color="#0b3e49", size=11),
                gridcolor="#d7e9ea",
                zerolinecolor="#9fcbd0",
            ),
            yaxis=dict(
                title=dict(text="", font=dict(color="#0b3e49")),
                tickfont=dict(color="#0b3e49", size=11),
            ),
        )
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    else:
        st.info("No potential-risk cells are present in the latest field, so there is no concentration chart to fabricate.")

    # Two genuinely different distributions
    a, b = st.columns(2, gap="large")
    with a:
        card("Chl-a distribution", "Distribution of valid positive chlorophyll-a observations in the latest field.")
        valid = latest.loc[latest.chla > 0, "chla"].dropna()
        if not valid.empty:
            fig = px.histogram(valid, nbins=45)
            fig.update_traces(marker_color="#2387aa")
            fig.update_layout(
                height=360,
                margin=dict(l=55, r=15, t=18, b=58),
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(family="DM Sans", color="#0b3e49"),
                showlegend=False,
                xaxis=dict(
                    title=dict(text="Chlorophyll-a", font=dict(color="#0b3e49", size=13)),
                    tickfont=dict(color="#0b3e49", size=11),
                    gridcolor="#d7e9ea",
                    zerolinecolor="#9fcbd0",
                ),
                yaxis=dict(
                    title=dict(text="Cells", font=dict(color="#0b3e49", size=13)),
                    tickfont=dict(color="#0b3e49", size=11),
                    gridcolor="#d7e9ea",
                ),
            )
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    with b:
        card("Recent Chl-a change", "The distribution of change relative to the previous observation, where that field exists.")
        change = change_series.dropna()
        if not change.empty:
            fig = px.histogram(change, nbins=45)
            fig.update_traces(marker_color="#22a878")
            fig.add_vline(x=CHANGE_THRESHOLD, line_dash="dash", line_color="#d8952e")
            fig.update_layout(
                height=360,
                margin=dict(l=55, r=15, t=18, b=58),
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(family="DM Sans", color="#0b3e49"),
                showlegend=False,
                xaxis=dict(
                    title=dict(text="Chl-a change", font=dict(color="#0b3e49", size=13)),
                    tickfont=dict(color="#0b3e49", size=11),
                    gridcolor="#d7e9ea",
                    zerolinecolor="#9fcbd0",
                ),
                yaxis=dict(
                    title=dict(text="Cells", font=dict(color="#0b3e49", size=13)),
                    tickfont=dict(color="#0b3e49", size=11),
                    gridcolor="#d7e9ea",
                ),
            )
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        else:
            st.info("Recent-change values are not present in the latest field.")

    # Screening priority table
    st.markdown('<div class="section-label">Current screening priorities</div>', unsafe_allow_html=True)
    if not risk_latest.empty:
        sort_col = "risk_probability" if "risk_probability" in risk_latest.columns else "chla"
        priority = risk_latest.sort_values(sort_col, ascending=False).head(12).copy()
        priority["Location"] = priority.apply(lambda r: f"{r.latitude:.3f}°, {r.longitude:.3f}°", axis=1)
        table = pd.DataFrame({
            "Location": priority.Location,
            "Chl-a": priority.chla.round(4),
            "Risk probability": priority["risk_probability"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "Unavailable") if "risk_probability" in priority else "Unavailable",
            "Anomaly": priority["chla_anomaly"].round(4) if "chla_anomaly" in priority else np.nan,
            "Recent change": priority["chla_change"].round(4) if "chla_change" in priority else np.nan,
        })
        st.dataframe(table, width="stretch", hide_index=True)
    else:
        st.info("No current screening priorities are available in the latest field.")

    # Regional summary is useful and is not a duplicate of the risk table.
    st.markdown('<div class="section-label">Regional view</div>', unsafe_allow_html=True)
    regional = latest.copy()
    def region(lat, lon):
        if lon >= 75 and lat >= 0:
            return "Bay of Bengal"
        if lon < 75 and lat >= -5:
            return "Arabian Sea"
        if lon >= 55 and lat < 0:
            return "Southern Indian Ocean"
        return "Northern Indian Ocean"
    regional["Region"] = [region(a, b) for a, b in zip(regional.latitude, regional.longitude)]
    summary = regional.groupby("Region").agg(
        Processed_cells=("Region", "size"),
        Potential_risk=("risk_flag", "sum"),
        Mean_Chl_a=("chla", "mean"),
        Maximum_Chl_a=("chla", "max"),
    ).reset_index()
    summary["Risk_share"] = summary.Potential_risk / summary.Processed_cells * 100
    summary = summary.sort_values("Potential_risk", ascending=False)
    st.dataframe(summary.rename(columns={
        "Processed_cells": "Processed cells",
        "Potential_risk": "Potential-risk cells",
        "Mean_Chl_a": "Mean Chl-a",
        "Maximum_Chl_a": "Maximum Chl-a",
        "Risk_share": "Risk share (%)",
    }).round(4), width="stretch", hide_index=True)

    st.markdown('<div class="callout"><strong>Scientific caution:</strong> these are descriptive satellite-derived signals and proxy screening results. They do not identify algal species or toxins and do not confirm a harmful algal bloom.</div>', unsafe_allow_html=True)
    footer()

# ============================================================
# DATA
# ============================================================

elif st.session_state.page == "data":
    page_head(
        "04 · DATA & RESEARCH",
        "The evidence behind the dashboard.",
        "Source, coverage, processing scope and downloadable outputs live here. The other pages stay focused on analysis rather than repeating dataset documentation.",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric("Product", "E06OCM_L4_AC", "EOS-06 / OCM-3")
    with c2: metric("Grid", "0.25°", "latest processed field")
    with c3: metric("Coverage", f"{fmt_date(df.date.min())} → {fmt_date(latest_date)}", "available prediction records")
    with c4: metric("Latest field", f"{len(latest):,}", fmt_date(latest_date))

    st.markdown('<div class="section-label">Source product</div>', unsafe_allow_html=True)
    card(
        "EOS-06 / Oceansat-3 OCM-3 Level-4 analysed chlorophyll-a",
        "The dashboard uses the E06OCM_L4_AC satellite-derived ocean-colour product. The current application layer combines chlorophyll-a with temporal signals already stored in the processed project data.",
        '<div class="callout"><strong>Study window:</strong> approximately 20°E–120°E longitude and 40°S–30°N latitude for the dashboard spatial view.</div>',
    )

    st.markdown('<div class="section-label">Project outputs</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2, gap="large")
    with d1:
        card("Latest observations", "All valid observations from the latest processed field.")
        st.download_button(
            "Download latest observations",
            data=latest.to_csv(index=False).encode("utf-8"),
            file_name="bloomdetect_latest_observations.csv",
            mime="text/csv",
            width="stretch",
        )
    with d2:
        card("Potential-risk shortlist", "Only cells currently included in the latest potential-risk screening output.")
        st.download_button(
            "Download potential-risk locations",
            data=risk_latest.to_csv(index=False).encode("utf-8"),
            file_name="bloomdetect_potential_risk_locations.csv",
            mime="text/csv",
            width="stretch",
        )

    st.markdown('<div class="section-label">Model evidence</div>', unsafe_allow_html=True)
    model = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        "Final Decision Tree": [0.9993, 0.8331, 0.9988, 0.9085, 0.9997],
    })
    st.dataframe(model, width="stretch", hide_index=True)
    st.markdown('<div class="callout"><strong>Interpretation:</strong> these metrics evaluate the project’s proxy screening target derived from chlorophyll-a temporal behaviour. They are not field-validated HAB detection accuracy. Satellite observations cannot independently establish species identity or toxin presence.</div>', unsafe_allow_html=True)

    if history_available:
        st.markdown('<div class="section-label">Historical support layer</div>', unsafe_allow_html=True)
        card("Compact observation history", f"The Risk Map timeline is backed by <b>{history_name}</b>. It is an aggregated historical screening layer, not a second ML model. Available dates: {fmt_date(history.date.min())} to {fmt_date(history.date.max())}.")

    footer()
