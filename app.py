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
# Environmental / SCAT-3 data
ENVIRONMENT_FILE = BASE / "data" / "bloomdetect_environmental_signals.csv"
OCEAN_ATMOSPHERE_FILE = BASE / "data" / "bloomdetect_ocean_atmosphere.csv"
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


@st.cache_data(show_spinner="Loading ocean-atmosphere observations…")
def load_ocean_atmosphere():
    """
    Load the merged OCM-3 + SCAT-3 dataset.

    The source contains the satellite chlorophyll-a field together
    with analyzed wind and atmosphere-ocean forcing variables.
    """

    if not OCEAN_ATMOSPHERE_FILE.exists():
        return pd.DataFrame()

    required = [
        "date",
        "lat",
        "lon",
        "chla",
        "risk",
        "U",
        "V",
        "TAUX",
        "TAUY",
        "DIVG",
        "CURL",
        "QLH",
        "QSH",
        "NS",
        "wind_speed",
        "wind_direction",
        "wind_stress",
    ]

    # Read only the required columns.
    header = pd.read_csv(
        OCEAN_ATMOSPHERE_FILE,
        nrows=0
    )

    available = set(header.columns)

    usecols = [
        col
        for col in required
        if col in available
    ]

    if not {"date", "lat", "lon", "chla"}.issubset(usecols):
        return pd.DataFrame()

    env = pd.read_csv(
        OCEAN_ATMOSPHERE_FILE,
        usecols=usecols
    )

    env["date"] = pd.to_datetime(
        env["date"],
        errors="coerce"
    )

    numeric_cols = [
        col
        for col in usecols
        if col not in {"date"}
    ]

    for col in numeric_cols:
        env[col] = pd.to_numeric(
            env[col],
            errors="coerce"
        )

    env = env.replace(
        [np.inf, -np.inf],
        np.nan
    )

    env = env.dropna(
        subset=[
            "date",
            "lat",
            "lon",
            "chla",
        ]
    ).copy()

    env = env[
        env["lat"].between(
            LAT_MIN,
            LAT_MAX
        )
        &
        env["lon"].between(
            LON_MIN,
            LON_MAX
        )
    ].copy()

    # Recalculate wind speed if the stored field is unavailable.
    if "wind_speed" not in env.columns:
        if {"U", "V"}.issubset(env.columns):
            env["wind_speed"] = np.sqrt(
                env["U"] ** 2 +
                env["V"] ** 2
            )

    # Recalculate wind direction if unavailable.
    if "wind_direction" not in env.columns:
        if {"U", "V"}.issubset(env.columns):
            env["wind_direction"] = (
                np.degrees(
                    np.arctan2(
                        env["V"],
                        env["U"]
                    )
                )
                + 360
            ) % 360

    # Recalculate wind stress magnitude if unavailable.
    if "wind_stress" not in env.columns:
        if {"TAUX", "TAUY"}.issubset(env.columns):
            env["wind_stress"] = np.sqrt(
                env["TAUX"] ** 2 +
                env["TAUY"] ** 2
            )

    return env


ocean_atmosphere = load_ocean_atmosphere()

latest_date = df["date"].max()

latest = df[
    df["date"].dt.normalize()
    == latest_date.normalize()
].copy()

risk_latest = latest[
    latest["risk_flag"]
].copy()


# Environmental dataset has a slightly different temporal endpoint.
# Use the most recent environmental observation available at or before
# the OCM-3 latest field.
if not ocean_atmosphere.empty:

    environmental_latest_date = ocean_atmosphere[
        ocean_atmosphere["date"] <= latest_date
    ]["date"].max()

    if pd.notna(environmental_latest_date):

        environmental_latest = ocean_atmosphere[
            ocean_atmosphere["date"].dt.normalize()
            == environmental_latest_date.normalize()
        ].copy()

    else:

        environmental_latest_date = pd.NaT
        environmental_latest = pd.DataFrame()

else:

    environmental_latest_date = pd.NaT
    environmental_latest = pd.DataFrame()

CHLA_ELEVATION_THRESHOLD = 0.10576990386471145
WIND_FORCING_THRESHOLD = 8.7317875
STRESS_FORCING_THRESHOLD = 0.11441081

def environmental_state(row):
    chla = pd.to_numeric(row.get("chla", np.nan), errors="coerce")
    wind = pd.to_numeric(row.get("wind_speed", np.nan), errors="coerce")
    stress = pd.to_numeric(row.get("wind_stress", np.nan), errors="coerce")
    biological = pd.notna(chla) and chla >= CHLA_ELEVATION_THRESHOLD
    forcing = ((pd.notna(wind) and wind >= WIND_FORCING_THRESHOLD) or (pd.notna(stress) and stress >= STRESS_FORCING_THRESHOLD))
    if biological and forcing: return "Combined elevation + forcing"
    if biological: return "Biological elevation"
    if forcing: return "Environmental forcing"
    return "Baseline"

def environmental_features(frame):
    if frame.empty: return frame.copy()
    out = frame.copy()
    for col in ["chla", "wind_speed", "wind_stress"]:
        if col in out.columns: out[col] = pd.to_numeric(out[col], errors="coerce")
        else: out[col] = np.nan
    out["biological_elevation"] = out["chla"] >= CHLA_ELEVATION_THRESHOLD
    out["environmental_forcing"] = (out["wind_speed"] >= WIND_FORCING_THRESHOLD) | (out["wind_stress"] >= STRESS_FORCING_THRESHOLD)
    out["combined_signal"] = out["biological_elevation"] & out["environmental_forcing"]
    out["signal_state"] = np.select([out["combined_signal"],out["biological_elevation"],out["environmental_forcing"]],["Combined elevation + forcing","Biological elevation","Environmental forcing"],default="Baseline")
    return out

@st.cache_data(show_spinner=False)
def location_exposure_summary(frame):
    """Summarize forcing exposure and screening rate by location."""
    cols=[c for c in ["lat","lon","risk","wind_speed","wind_stress"] if c in frame.columns]
    if len(cols)<5:
        return pd.DataFrame()
    z=frame[cols].dropna().copy()
    if z.empty:
        return pd.DataFrame()
    z["forcing"]=(z["wind_speed"]>=WIND_FORCING_THRESHOLD)|(z["wind_stress"]>=STRESS_FORCING_THRESHOLD)
    summary=z.groupby(["lat","lon"],as_index=False).agg(exposure=("forcing","mean"),screening_rate=("risk","mean"),observations=("risk","size"))
    summary["exposure_band"]=pd.cut(summary["exposure"],[ -0.01,0.25,0.50,0.75,1.01],labels=["Low","Moderate","High","Very high"])
    return summary

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
@keyframes floatSoft{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
@keyframes pulseSoft{0%,100%{box-shadow:0 0 0 0 rgba(7,158,170,.12)}50%{box-shadow:0 0 0 12px rgba(7,158,170,0)}}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.hero{animation:fadeUp .55s ease both}.hero:before{animation:pulseSoft 6s ease-in-out infinite}.metric,.card{animation:fadeUp .45s ease both}.brandmark{animation:floatSoft 4s ease-in-out infinite}
.feature-pill{display:inline-flex;align-items:center;gap:7px;padding:8px 11px;border-radius:999px;background:#effafa;border:1px solid #cbe8e8;color:#0b5964;font-size:.69rem;font-weight:700;margin:3px 4px 3px 0}.feature-dot{width:7px;height:7px;border-radius:50%;background:#079eaa;display:inline-block}
.home-feature{padding:20px;border:1px solid var(--line);border-radius:20px;background:white;box-shadow:0 9px 25px rgba(10,78,89,.055);margin-bottom:12px}.home-feature h3{font:800 1rem Manrope;color:var(--ink);margin:0 0 6px}.home-feature p{font-size:.76rem;line-height:1.55;color:#668188;margin:0}.home-mini{padding:14px;border-radius:16px;background:#f1fbfb;border:1px solid #d2ecec;margin-top:10px}.home-mini strong{color:var(--ink2);font-size:.78rem}.home-mini span{display:block;color:#6b858b;font-size:.7rem;line-height:1.45;margin-top:3px}

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
        f'<div class="footer">BloomDetect AI · EOS-06 OCM-3 + SCAT-3 · Ocean intelligence & screening · Latest processed field: {fmt_date(latest_date)}</div>',
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

    distance = (
        ((valid.latitude - lat_value) / 70.0) ** 2
        + ((valid.longitude - lon_value) / 100.0) ** 2
    )

    idx = distance.idxmin()
    row = valid.loc[idx]

    separation = float(
        np.hypot(
            float(row.latitude) - lat_value,
            float(row.longitude) - lon_value
        )
    )

    return row, separation


def nearest_environmental_cell(latitude, longitude, target_date=None):
    """Find the nearest available SCAT-3 observation for a coordinate/date."""

    if ocean_atmosphere.empty:
        return None, None

    env = ocean_atmosphere

    if target_date is not None:
        target = pd.Timestamp(target_date).normalize()

        same_date = env[
            env["date"].dt.normalize() == target
        ]

        if not same_date.empty:
            env = same_date
        else:
            before = env[env["date"] <= target]
            if not before.empty:
                fallback_date = before["date"].max()
                env = before[
                    before["date"].dt.normalize()
                    == fallback_date.normalize()
                ]

    if env.empty:
        return None, None

    lat_scale = max(LAT_MAX - LAT_MIN, 1)
    lon_scale = max(LON_MAX - LON_MIN, 1)

    distance = (
        ((env["lat"] - float(latitude)) / lat_scale) ** 2
        + ((env["lon"] - float(longitude)) / lon_scale) ** 2
    )

    idx = distance.idxmin()
    result = env.loc[idx].copy()
    separation = float(np.sqrt(distance.loc[idx]))

    return result, separation

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
    ("home", "⌂ Overview"),
    ("map", "◉ Ocean Map"),
    ("location", "⌖ Location Check"),
    ("insights", "▥ Ocean Insights"),
    ("data", "↓ Research Data"),
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
        '<section class="hero"><div class="hero-content"><div class="kicker">EOS-06 · OCM-3 + SCAT-3 · Multi-sensor ocean intelligence</div><h1>One ocean.<br>Many signals.</h1><p>BloomDetect AI brings together ocean colour, biological change and surface environmental conditions to help people understand where the ocean deserves a closer look. It is an investigation-support system, not a claim of confirmed harmful algal bloom detection.</p><div class="chips"><span class="chip">🌊 Chl-a & biology</span><span class="chip">💨 Wind & forcing</span><span class="chip">🗺 Spatial patterns</span><span class="chip">📍 Local conditions</span><span class="chip">🔬 Research support</span></div></div></section>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1: metric("Ocean cells",f"{len(latest):,}","latest OCM-3 field")
    with c2: metric("Potential-risk",f"{len(risk_latest):,}","screening signal")
    with c3: metric("Wind context",(fmt_num(environmental_latest.wind_speed.mean(),1)+" m/s") if not environmental_latest.empty and "wind_speed" in environmental_latest.columns else "Unavailable","SCAT-3 field")
    with c4: metric("Risk share",f"{(len(risk_latest)/len(latest)*100 if len(latest) else 0):.2f}%","latest screening")

    st.markdown('<div class="section-label">What BloomDetect can understand</div>',unsafe_allow_html=True)
    h1,h2=st.columns(2,gap="medium")
    with h1:
        st.markdown('<div class="home-feature"><h3>🌱 Biological activity</h3><p>Tracks chlorophyll-a, historical baseline, recent change and anomaly patterns to surface places whose biological signal deserves investigation.</p><div><span class="feature-pill"><i class="feature-dot"></i>Chl-a</span><span class="feature-pill"><i class="feature-dot"></i>Anomaly</span><span class="feature-pill"><i class="feature-dot"></i>Recent change</span></div><div class="home-mini"><strong>Human-readable answer</strong><span>“This location has an elevated or changing ocean-colour signal.”</span></div></div>',unsafe_allow_html=True)
    with h2:
        st.markdown('<div class="home-feature"><h3>💨 Environmental conditions</h3><p>Uses SCAT-3 analyzed wind and surface-flux observations to add environmental context around the biological signal.</p><div><span class="feature-pill"><i class="feature-dot"></i>Wind speed</span><span class="feature-pill"><i class="feature-dot"></i>Wind stress</span><span class="feature-pill"><i class="feature-dot"></i>Heat flux</span></div><div class="home-mini"><strong>Human-readable answer</strong><span>“This signal is occurring alongside stronger or weaker surface forcing.”</span></div></div>',unsafe_allow_html=True)
    h3,h4=st.columns(2,gap="medium")
    with h3:
        st.markdown('<div class="home-feature"><h3>🧭 Real-world monitoring support</h3><p>Useful for researchers, coastal observers and marine-resource teams who need a fast way to inspect broad ocean conditions before deciding where closer investigation may be worthwhile.</p><div><span class="feature-pill"><i class="feature-dot"></i>Coastal watch</span><span class="feature-pill"><i class="feature-dot"></i>Marine research</span><span class="feature-pill"><i class="feature-dot"></i>Environmental screening</span></div></div>',unsafe_allow_html=True)
    with h4:
        st.markdown('<div class="home-feature"><h3>🔬 Beyond bloom screening</h3><p>The same observations support ocean-condition analysis, environmental forcing studies, spatial pattern analysis and ocean–atmosphere research. They do not directly predict fish catch, toxins or species identity.</p><div><span class="feature-pill"><i class="feature-dot"></i>Ocean conditions</span><span class="feature-pill"><i class="feature-dot"></i>Forcing patterns</span><span class="feature-pill"><i class="feature-dot"></i>Research evidence</span></div></div>',unsafe_allow_html=True)

    st.markdown('<div class="section-label">Choose what you want to understand</div>',unsafe_allow_html=True)
    b1,b2,b3,b4,b5=st.columns(5)
    nav_targets=[(b1,"🗺️","Ocean Map","See patterns across the study region.","map"),(b2,"📍","Location Check","Understand one coordinate.","location"),(b3,"📊","Ocean Insights","Compare biological and environmental signals.","insights"),(b4,"🧪","Research Data","Inspect sources and download evidence.","data"),(b5,"🌊","Overview","Return to the project overview.","home")]
    for col,icon,title,desc,target in nav_targets:
        with col:
            st.markdown(f'<div class="home-mini"><strong>{icon} {title}</strong><span>{desc}</span></div>',unsafe_allow_html=True)
            if st.button("Open",key="home_"+target,width="stretch",type="primary" if target!="home" else "secondary"):
                go_to(target)

    if not environmental_latest.empty:
        ef=environmental_features(environmental_latest)
        combined_count=int(ef["combined_signal"].sum())
        st.markdown(f'<div class="callout"><strong>Today’s research lens:</strong> the integrated dataset separates biological elevation, environmental forcing and their overlap. The latest matched field contains <b>{combined_count:,}</b> cells in the combined category. This is an observational screening signal, not proof that environmental forcing caused the biological change.</div>',unsafe_allow_html=True)

    st.markdown('<div class="callout"><strong>What we do not claim:</strong> BloomDetect does not identify algal species, detect toxins, confirm a harmful algal bloom, predict fish catch or establish cause-and-effect from wind. Those questions need additional observations and field validation.</div>',unsafe_allow_html=True)
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
        # ENVIRONMENTAL LENS
        # ========================================================

        if is_latest and not environmental_latest.empty:
            st.markdown('<div class="section-label">Environmental lens</div>',unsafe_allow_html=True)
            st.markdown('<div class="callout"><strong>New:</strong> switch from the biological map to the surface environment. This helps answer not only “where is the signal?” but also “what conditions are present there?”</div>',unsafe_allow_html=True)
            lens=st.selectbox("Map lens",["No environmental overlay","Wind speed","Wind stress","Combined elevation + forcing"],key="environmental_map_lens")
            if lens != "No environmental overlay":
                env_plot=environmental_latest.copy()
                if lens=="Wind speed": field_col,label,scale="wind_speed","Wind speed (m/s)","Turbo"
                elif lens=="Wind stress": field_col,label,scale="wind_stress","Wind stress (Pa)","Viridis"
                else:
                    env_plot=environmental_features(env_plot); env_plot["combined_value"]=env_plot["combined_signal"].astype(int); field_col,label,scale="combined_value","Combined signal","YlGnBu"
                if field_col in env_plot.columns:
                    env_plot=env_plot.dropna(subset=["lat","lon",field_col]).copy()
                    if len(env_plot)>14000: env_plot=env_plot.sample(14000,random_state=42)
                    fig_env=base_map(height=520)
                    fig_env.add_trace(go.Scattergeo(lon=env_plot["lon"],lat=env_plot["lat"],mode="markers",name=label,marker=dict(size=4.2,color=env_plot[field_col],colorscale=scale,opacity=.72,colorbar=dict(title=label,thickness=13)),customdata=np.c_[env_plot["chla"],env_plot[field_col]],hovertemplate="Lat %{lat:.2f}°<br>Lon %{lon:.2f}°<br>Chl-a %{customdata[0]:.4f}<br>"+label+" %{customdata[1]:.3f}<extra></extra>"))
                    st.plotly_chart(fig_env,width="stretch",config={"displaylogo":False,"scrollZoom":True,"responsive":True})

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


        if not environmental_latest.empty:
            st.markdown(
                '<div class="section-label">Ocean–atmosphere context</div>',
                unsafe_allow_html=True,
            )

            env_valid = environmental_latest.dropna(
                subset=["wind_speed"]
            ).copy() if "wind_speed" in environmental_latest.columns else pd.DataFrame()

            if not env_valid.empty:
                ec1, ec2, ec3, ec4 = st.columns(4)

                with ec1:
                    metric(
                        "Environmental date",
                        fmt_date(environmental_latest_date),
                        "latest available SCAT-3"
                    )

                with ec2:
                    metric(
                        "Mean wind speed",
                        f"{env_valid.wind_speed.mean():.2f} m/s",
                        "study window"
                    )

                with ec3:
                    stress_mean = (
                        env_valid["wind_stress"].dropna().mean()
                        if "wind_stress" in env_valid.columns
                        else np.nan
                    )
                    metric(
                        "Mean wind stress",
                        f"{stress_mean:.3f} Pa" if pd.notna(stress_mean) else "Unavailable",
                        "study window"
                    )

                with ec4:
                    high_wind = env_valid["wind_speed"] >= 8.73
                    metric(
                        "Higher-wind cells",
                        f"{int(high_wind.sum()):,}",
                        "wind ≥ 75th percentile"
                    )

                env_map = env_valid[["lat", "lon", "wind_speed"]].copy()

                if len(env_map) > 10000:
                    env_map = env_map.sample(10000, random_state=42)

                fig_env = px.scatter_geo(
                    env_map,
                    lat="lat",
                    lon="lon",
                    color="wind_speed",
                    color_continuous_scale=[
                        "#D8F3F0", "#58B4AE", "#0B7285", "#C0394B"
                    ],
                    hover_data={
                        "lat": ":.2f",
                        "lon": ":.2f",
                        "wind_speed": ":.2f",
                    },
                    labels={"wind_speed": "Wind speed (m/s)"},
                )

                fig_env.update_geos(
                    showcountries=True,
                    countrycolor="#B8C9CC",
                    showcoastlines=True,
                    coastlinecolor="#6F858A",
                    showland=True,
                    landcolor="#F2F5F5",
                    projection_type="equirectangular",
                    lataxis_range=[LAT_MIN, LAT_MAX],
                    lonaxis_range=[LON_MIN, LON_MAX],
                )

                fig_env.update_layout(
                    height=500,
                    margin=dict(l=10, r=10, t=20, b=20),
                    paper_bgcolor="white",
                    font=dict(family="DM Sans", color="#0b3e49"),
                    coloraxis_colorbar=dict(title="Wind speed<br>(m/s)"),
                )

                st.plotly_chart(
                    fig_env,
                    width="stretch",
                    config={"displaylogo": False, "scrollZoom": True, "responsive": True},
                )

                st.markdown(
                    f"""
                    <div class="callout">
                        <strong>Environmental context:</strong>
                        the map shows SCAT-3 analyzed wind speed for
                        {fmt_date(environmental_latest_date)} alongside
                        the selected OCM-3 screening field. Higher wind
                        conditions may be associated with different
                        ocean-surface and mixing conditions, but this
                        relationship is not itself evidence of harmfulness.
                    </div>
                    """,
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

        # ====================================================
        # OCEAN-ATMOSPHERE SUPPORT
        # ====================================================

        environmental_row, environmental_separation = (
            nearest_environmental_cell(
                float(row.latitude),
                float(row.longitude),
                row.date,
            )
        )

        if environmental_row is not None:

            st.markdown(
                '<div class="section-label">'
                'Ocean–atmosphere conditions'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="callout">'
                '<strong>SCAT-3 environmental support:</strong> '
                'wind and surface-flux observations are shown '
                'alongside the OCM-3 chlorophyll-a screening signal. '
                'These variables describe environmental conditions '
                'associated with the observation; they do not prove '
                'that wind caused a bloom.'
                '</div>',
                unsafe_allow_html=True
            )

            e1, e2, e3, e4 = st.columns(4)

            with e1:
                metric(
                    "Wind speed",
                    (
                        f"{environmental_row.wind_speed:.2f} m/s"
                        if pd.notna(
                            environmental_row.get(
                                "wind_speed",
                                np.nan
                            )
                        )
                        else "Unavailable"
                    ),
                    "SCAT-3 analyzed wind"
                )

            with e2:
                metric(
                    "Wind stress",
                    (
                        f"{environmental_row.wind_stress:.3f} Pa"
                        if pd.notna(
                            environmental_row.get(
                                "wind_stress",
                                np.nan
                            )
                        )
                        else "Unavailable"
                    ),
                    "surface forcing"
                )

            with e3:
                metric(
                    "Latent heat flux",
                    (
                        f"{environmental_row.QLH:.1f} W/m²"
                        if pd.notna(
                            environmental_row.get(
                                "QLH",
                                np.nan
                            )
                        )
                        else "Unavailable"
                    ),
                    "air–sea heat exchange"
                )

            with e4:
                metric(
                    "Sensible heat flux",
                    (
                        f"{environmental_row.QSH:.1f} W/m²"
                        if pd.notna(
                            environmental_row.get(
                                "QSH",
                                np.nan
                            )
                        )
                        else "Unavailable"
                    ),
                    "air–sea heat exchange"
                )

            # Environmental interpretation
            wind_value = environmental_row.get(
                "wind_speed",
                np.nan
            )

            stress_value = environmental_row.get(
                "wind_stress",
                np.nan
            )

            chla_value = row.get(
                "chla",
                np.nan
            )

            anomaly_value = row.get(
                "chla_anomaly",
                np.nan
            )

            change_value = row.get(
                "chla_change",
                np.nan
            )

            environmental_notes = []

            if pd.notna(wind_value):
                if wind_value >= 8.73:
                    environmental_notes.append(
                        "elevated wind forcing"
                    )
                else:
                    environmental_notes.append(
                        "moderate wind forcing"
                    )

            if pd.notna(stress_value):
                if stress_value >= 0.114:
                    environmental_notes.append(
                        "higher wind stress"
                    )
                else:
                    environmental_notes.append(
                        "lower wind stress"
                    )

            if (
                pd.notna(anomaly_value)
                and anomaly_value >= ANOMALY_THRESHOLD
            ):
                environmental_notes.append(
                    "elevated Chl-a relative to the local baseline"
                )

            if (
                pd.notna(change_value)
                and change_value > CHANGE_THRESHOLD
            ):
                environmental_notes.append(
                    "recently increasing Chl-a"
                )

            if environmental_notes:

                interpretation = (
                    ", ".join(
                        environmental_notes
                    )
                    + "."
                )

            else:

                interpretation = (
                    "No strong environmental or temporal "
                    "signal is identified from the available "
                    "fields."
                )

            st.markdown(
                f"""
                <div class="callout">
                    <strong>Environmental interpretation:</strong>
                    {interpretation}
                    This is contextual evidence for screening,
                    not a causal diagnosis of harmful algal bloom
                    formation.
                </div>
                """,
                unsafe_allow_html=True
            )

            state = environmental_state({
                "chla": chla_value,
                "wind_speed": wind_value,
                "wind_stress": stress_value,
            })
            direction_value = environmental_row.get("wind_direction", np.nan)
            st.markdown('<div class="section-label">Plain-language reading</div>', unsafe_allow_html=True)
            p1,p2,p3=st.columns(3)
            with p1: metric("Environment state",state,"biological + forcing")
            with p2: metric("Wind direction",f"{float(direction_value):.0f}°" if pd.notna(direction_value) else "Unavailable","analyzed direction")
            with p3: metric("Chl-a signal", "Elevated" if pd.notna(chla_value) and chla_value >= CHLA_ELEVATION_THRESHOLD else "Lower","relative screening lens")

            if (
                pd.notna(environmental_row.date)
                and environmental_row.date.normalize()
                != pd.Timestamp(row.date).normalize()
            ):

                st.markdown(
                    f"""
                    <div class="small-muted">
                        Environmental observation used:
                        {fmt_date(environmental_row.date)}.
                        The closest available SCAT-3 observation
                        was used because an exact-date environmental
                        record was not available for this cell.
                    </div>
                    """,
                    unsafe_allow_html=True
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
        "See what the ocean is doing.",
        "BloomDetect turns the same observations into simple views of biological activity, environmental forcing and the places where those signals overlap.",
    )

    change_series=pd.to_numeric(latest.get("chla_change",pd.Series(index=latest.index,dtype=float)),errors="coerce")
    anomaly_series=pd.to_numeric(latest.get("chla_anomaly",pd.Series(index=latest.index,dtype=float)),errors="coerce")
    chla_series=pd.to_numeric(latest.get("chla",pd.Series(index=latest.index,dtype=float)),errors="coerce")
    positive_change=int((change_series>0).sum())
    anomaly_count=int((anomaly_series>=ANOMALY_THRESHOLD).sum())

    i1,i2,i3,i4=st.columns(4)
    with i1: metric("Potential-risk",f"{len(risk_latest):,}","latest screening")
    with i2: metric("Positive change",f"{positive_change:,}","Chl-a change > 0")
    with i3: metric("Anomalous cells",f"{anomaly_count:,}",f"anomaly ≥ {ANOMALY_THRESHOLD:.3f}")
    with i4: metric("Median Chl-a",fmt_num(chla_series.median()),"latest field")

    st.markdown('<div class="section-label">A simple way to read the ocean</div>',unsafe_allow_html=True)
    state_counts=pd.Series(dtype=int)
    if not environmental_latest.empty:
        state_counts=environmental_features(environmental_latest)["signal_state"].value_counts()
    q1,q2,q3,q4=st.columns(4,gap="medium")
    cards=[(q1,"🌱 Biological elevation","Higher Chl-a relative to the project screening threshold.","Biological elevation"),(q2,"💨 Environmental forcing","Stronger surface wind or wind stress in the screening lens.","Environmental forcing"),(q3,"✨ Combined signal","Biological elevation alongside stronger environmental forcing.","Combined elevation + forcing"),(q4,"🌊 Baseline","Neither signal is elevated in the current lens.","Baseline")]
    for col,title,copy,state in cards:
        count=int(state_counts.get(state,0))
        with col:
            st.markdown(f'<div class="home-feature"><h3>{title}</h3><p>{copy}</p><div class="metric-value" style="margin-top:10px">{count:,}</div><div class="metric-note">matched environmental cells</div></div>',unsafe_allow_html=True)

    st.markdown('<div class="section-label">Environment × biology</div>',unsafe_allow_html=True)
    if not environmental_latest.empty:
        ef=environmental_features(environmental_latest)
        matrix=pd.crosstab(ef["biological_elevation"].map({True:"Elevated biology",False:"Lower biology"}),ef["environmental_forcing"].map({True:"Higher forcing",False:"Lower forcing"}))
        matrix=matrix.reindex(index=["Lower biology","Elevated biology"],columns=["Lower forcing","Higher forcing"],fill_value=0)
        st.dataframe(matrix,width="stretch")
        st.markdown('<div class="callout"><strong>Research lens:</strong> the overlap category is useful for prioritising investigation. It is not a causal diagnosis.</div>',unsafe_allow_html=True)
    else:
        st.info("The SCAT-3 integrated layer is not available for this deployment.")

    st.markdown('<div class="section-label">Ocean–atmosphere relationships</div>',unsafe_allow_html=True)
    if not ocean_atmosphere.empty and {"chla","wind_speed","wind_stress"}.issubset(ocean_atmosphere.columns):
        env_sample=ocean_atmosphere[["chla","wind_speed","wind_stress"]].dropna().copy()
        if len(env_sample)>6000: env_sample=env_sample.sample(6000,random_state=42)
        r1,r2=st.columns(2,gap="large")
        for col,xcol,title,xlabel in [(r1,"wind_speed","Chl-a and wind speed","Wind speed (m/s)"),(r2,"wind_stress","Chl-a and wind stress","Wind stress (Pa)")]:
            with col:
                fig=px.scatter(env_sample,x=xcol,y="chla",opacity=.30,labels={xcol:xlabel,"chla":"Chlorophyll-a"},title=title)
                fig.update_traces(marker=dict(size=5,color="#0b8f9b"))
                fig.update_layout(height=390,paper_bgcolor="white",plot_bgcolor="white",font=dict(family="DM Sans",color="#0b3e49"),margin=dict(l=65,r=20,t=55,b=65),xaxis=dict(tickfont=dict(color="#0b3e49",size=11),title_font=dict(color="#0b3e49",size=13)),yaxis=dict(tickfont=dict(color="#0b3e49",size=11),title_font=dict(color="#0b3e49",size=13)))
                st.plotly_chart(fig,width="stretch",config={"displaylogo":False,"responsive":True})
        wind_corr=env_sample[["chla","wind_speed"]].corr().iloc[0,1]
        stress_corr=env_sample[["chla","wind_stress"]].corr().iloc[0,1]
        st.markdown(f'<div class="callout"><strong>Observed association:</strong> Chl-a and wind speed correlation = <b>{wind_corr:.3f}</b>; Chl-a and wind stress correlation = <b>{stress_corr:.3f}</b>. These statistics describe association in the processed observations, not cause and effect.</div>',unsafe_allow_html=True)

    st.markdown('<div class="section-label">Environmental exposure gradient</div>',unsafe_allow_html=True)
    if not ocean_atmosphere.empty and {"lat","lon","risk","wind_speed","wind_stress"}.issubset(ocean_atmosphere.columns):
        exposure=location_exposure_summary(ocean_atmosphere)
        if not exposure.empty:
            grad=(exposure.groupby("exposure_band",observed=False).agg(locations=("screening_rate","size"),mean_screening=("screening_rate","mean")).reset_index())
            grad["mean_screening_pct"]=grad["mean_screening"]*100
            fig=px.bar(grad,x="exposure_band",y="mean_screening_pct",text="mean_screening_pct",labels={"exposure_band":"Environmental exposure","mean_screening_pct":"Mean screening rate (%)"},title="Screening rate across environmental exposure")
            fig.update_traces(marker_color="#0b8f9b",texttemplate="%{text:.2f}%",textposition="outside")
            fig.update_layout(height=350,paper_bgcolor="white",plot_bgcolor="white",font=dict(family="DM Sans",color="#0b3e49"),margin=dict(l=55,r=25,t=55,b=55),xaxis=dict(tickfont=dict(color="#0b3e49",size=11),title_font=dict(color="#0b3e49",size=13)),yaxis=dict(tickfont=dict(color="#0b3e49",size=11),title_font=dict(color="#0b3e49",size=13)))
            st.plotly_chart(fig,width="stretch",config={"displaylogo":False,"responsive":True})
            st.markdown('<div class="callout"><strong>How to read it:</strong> locations are grouped by the share of their observations experiencing the project’s environmental-forcing screen. A higher screening rate is an association, not evidence that forcing caused the biological signal.</div>',unsafe_allow_html=True)

    st.markdown('<div class="section-label">Where this can help</div>',unsafe_allow_html=True)
    u1,u2,u3=st.columns(3,gap="medium")
    for col,title,copy in [(u1,"🐟 Marine-resource support","Identify unusual ocean conditions that may deserve ecological or fisheries investigation. This is not a fish-catch predictor."),(u2,"🏖 Coastal environmental watch","Inspect biological and environmental signals together before prioritising a field visit or closer observation."),(u3,"🧪 Research & education","Explore how ocean-colour observations and atmospheric forcing behave together across space and time.")]:
        with col: card(title,copy)

    st.markdown('<div class="section-label">Current biological signal</div>',unsafe_allow_html=True)
    valid_signal=pd.DataFrame({"change":change_series,"anomaly":anomaly_series}).dropna()
    if not valid_signal.empty:
        emerging=int(((valid_signal.anomaly>=ANOMALY_THRESHOLD)&(valid_signal.change>CHANGE_THRESHOLD)).sum())
        persistent=int(((valid_signal.anomaly>=ANOMALY_THRESHOLD)&(valid_signal.change<=CHANGE_THRESHOLD)).sum())
        rapid=int(((valid_signal.anomaly<ANOMALY_THRESHOLD)&(valid_signal.change>CHANGE_THRESHOLD)).sum())
        baseline=int(((valid_signal.anomaly<ANOMALY_THRESHOLD)&(valid_signal.change<=CHANGE_THRESHOLD)).sum())
        s1,s2,s3,s4=st.columns(4)
        with s1: metric("Emerging",f"{emerging:,}","anomaly + rising")
        with s2: metric("Persistent",f"{persistent:,}","elevated, not rising")
        with s3: metric("Rapid change",f"{rapid:,}","rising without anomaly")
        with s4: metric("Baseline",f"{baseline:,}","neither elevated")

    st.markdown('<div class="callout"><strong>Scientific boundary:</strong> Chl-a is a biological proxy. The combined environmental categories are screening aids and observational associations. They do not confirm HAB species, toxins, ecological impact or causation.</div>',unsafe_allow_html=True)
    footer()

# ============================================================
# DATA
# ============================================================

elif st.session_state.page == "data":
    page_head(
        "04 · RESEARCH DATA",
        "Know what is behind every signal.",
        "Two EOS-06 observation streams are brought together here: OCM-3 for ocean-colour biology and SCAT-3 for surface environmental conditions. Download the evidence, not a mystery box.",
    )

    c1,c2,c3,c4=st.columns(4)
    with c1: metric("OCM-3","Chl-a","biological signal")
    with c2: metric("SCAT-3","Wind","environmental context")
    with c3: metric("Integrated rows",f"{len(ocean_atmosphere):,}" if not ocean_atmosphere.empty else "Unavailable","matched observations")
    with c4: metric("Latest field",fmt_date(latest_date),"OCM-3")

    st.markdown('<div class="section-label">What each dataset contributes</div>',unsafe_allow_html=True)
    d1,d2=st.columns(2,gap="medium")
    with d1: card("🛰 OCM-3 · Biological view","Satellite-derived chlorophyll-a plus temporal behaviour: previous observation, baseline, anomaly and recent change. This powers the existing potential-risk screening layer.",'<div class="callout"><strong>Use it for:</strong> spatial screening, biological-pattern analysis and locating unusual ocean-colour signals.</div>')
    with d2: card("💨 SCAT-3 · Environmental view","Analyzed zonal/meridional wind, wind speed, direction, wind stress, fluxes, divergence, curl and sample-count information where available.",'<div class="callout"><strong>Use it for:</strong> environmental context, forcing analysis and ocean–atmosphere relationship studies.</div>')

    if not ocean_atmosphere.empty:
        env_cells=len(ocean_atmosphere); env_dates=ocean_atmosphere.date.nunique(); env_first=ocean_atmosphere.date.min(); env_last=ocean_atmosphere.date.max()
        st.markdown('<div class="section-label">Integrated project layer</div>',unsafe_allow_html=True)
        e1,e2,e3,e4=st.columns(4)
        with e1: metric("Matched observations",f"{env_cells:,}","OCM-3 + SCAT-3")
        with e2: metric("Observation dates",f"{env_dates:,}","integrated layer")
        with e3: metric("Coverage start",fmt_date(env_first),"integrated")
        with e4: metric("Coverage end",fmt_date(env_last),"integrated")
        ef=environmental_features(environmental_latest) if not environmental_latest.empty else pd.DataFrame()
        if not ef.empty:
            st.markdown('<div class="section-label">New derived research features</div>',unsafe_allow_html=True)
            f1,f2,f3=st.columns(3,gap="medium")
            biological=int(ef.biological_elevation.sum()); forcing=int(ef.environmental_forcing.sum()); combined=int(ef.combined_signal.sum())
            with f1: metric("Biological elevation",f"{biological:,}",f"Chl-a ≥ {CHLA_ELEVATION_THRESHOLD:.3f}")
            with f2: metric("Environmental forcing",f"{forcing:,}","wind / stress threshold")
            with f3: metric("Combined signal",f"{combined:,}","elevation + forcing")
            st.markdown('<div class="callout"><strong>Feature meaning:</strong> these are transparent screening categories derived from the integrated observations. They are deliberately interpretable so a non-technical user can understand why a location was placed in a category.</div>',unsafe_allow_html=True)
        if not environmental_latest.empty:
            st.download_button("Download latest integrated field",data=environmental_latest.to_csv(index=False).encode("utf-8"),file_name="bloomdetect_latest_ocean_atmosphere.csv",mime="text/csv",width="stretch")
        with st.expander("Variables in the integrated dataset"):
            st.write(", ".join([c for c in ["date","lat","lon","chla","U","V","wind_speed","wind_direction","TAUX","TAUY","wind_stress","DIVG","CURL","QLH","QSH","NS"] if c in ocean_atmosphere.columns]))
    else:
        st.warning("The integrated OCM-3 + SCAT-3 dataset could not be loaded.")

    st.markdown('<div class="section-label">Existing project evidence</div>',unsafe_allow_html=True)
    d1,d2=st.columns(2,gap="medium")
    with d1: st.download_button("Download latest OCM-3 observations",data=latest.to_csv(index=False).encode("utf-8"),file_name="bloomdetect_latest_observations.csv",mime="text/csv",width="stretch")
    with d2: st.download_button("Download potential-risk shortlist",data=risk_latest.to_csv(index=False).encode("utf-8"),file_name="bloomdetect_potential_risk_locations.csv",mime="text/csv",width="stretch")

    st.markdown('<div class="section-label">Model evidence</div>',unsafe_allow_html=True)
    model=pd.DataFrame({"Metric":["Accuracy","Precision","Recall","F1","ROC-AUC"],"Final Decision Tree":[0.9993,0.8331,0.9988,0.9085,0.9997]})
    st.dataframe(model,width="stretch",hide_index=True)
    st.markdown('<div class="callout"><strong>Important:</strong> these metrics evaluate the project proxy screening target derived from chlorophyll-a temporal behaviour. They are not field-validated HAB detection accuracy.</div>',unsafe_allow_html=True)
    st.markdown('<div class="callout"><strong>What is not included:</strong> species identification, toxin detection, fish-catch prediction, confirmed bloom diagnosis or causal claims about wind and biology. Those require additional evidence and validation.</div>',unsafe_allow_html=True)
    footer()
