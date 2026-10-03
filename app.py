from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# BLOOMDETECT AI
# 5-page Coastal & Ocean Intelligence dashboard
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
HISTORY_PATH = BASE_DIR / "bloomdetect_history_web.csv.gz"
BLOOM_IMAGE = BASE_DIR / "bloomdetect_bloom_process.png"

LAT_MIN, LAT_MAX = -40.0, 30.0
LON_MIN, LON_MAX = 20.0, 120.0

# These are the proxy-screening thresholds used in the project label.
ANOMALY_THRESHOLD = 0.059076173328496344
CHANGE_THRESHOLD = 0.02007450088858604

# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

@st.cache_data(show_spinner="Loading satellite data...")
def load_prediction_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "latest_bloom_risk_predictions.csv was not found beside app.py."
        )

    d = pd.read_csv(DATA_PATH)
    required = {"latitude", "longitude", "date", "chla", "risk_label"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ["latitude", "longitude", "chla"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    numeric_cols = [
        "risk_probability", "model_score", "previous_chla",
        "historical_baseline", "recent_mean", "recent_max",
        "chla_anomaly", "chla_change",
    ]
    for c in numeric_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["latitude", "longitude", "date", "chla"]).copy()
    d["risk_flag"] = (
        d["risk_label"].astype(str).str.strip().str.lower()
        == "potential bloom risk"
    )
    d["plot_lon"] = ((d["longitude"] + 180) % 360) - 180
    return d


@st.cache_data(show_spinner="Loading map history...")
def load_history():
    if not HISTORY_PATH.exists():
        return pd.DataFrame()

    h = pd.read_csv(HISTORY_PATH)
    required = {"date", "lat_bin", "lon_bin", "chla", "risk"}
    if not required.issubset(h.columns):
        return pd.DataFrame()

    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    for c in ["lat_bin", "lon_bin", "chla", "risk"]:
        if c in h.columns:
            h[c] = pd.to_numeric(h[c], errors="coerce")
    h = h.dropna(subset=["date", "lat_bin", "lon_bin", "chla"]).copy()

    # Derive historical context without changing the stored lightweight file.
    h = h.sort_values(["lat_bin", "lon_bin", "date"]).reset_index(drop=True)
    g = h.groupby(["lat_bin", "lon_bin"])["chla"]
    h["previous_chla"] = g.shift(1)
    h["historical_baseline"] = g.transform(lambda x: x.shift(1).expanding().mean())
    h["chla_change"] = h["chla"] - h["previous_chla"]
    h["chla_anomaly"] = h["chla"] - h["historical_baseline"]
    return h


try:
    df = load_prediction_data()
except Exception as exc:
    st.error("BloomDetect AI could not load the prediction dataset.")
    st.code(str(exc))
    st.stop()

latest_date = df["date"].max()
latest = df[df["date"].eq(latest_date)].copy()
risk = latest[latest["risk_flag"]].copy()

history = load_history()

# ------------------------------------------------------------
# STATE / NAVIGATION
# ------------------------------------------------------------

PAGES = ["home", "map", "location", "insights", "data"]
NAV = {
    "home": "⌂ Home",
    "map": "🗺 Risk Map",
    "location": "📍 Location",
    "insights": "📊 Insights",
    "data": "⇩ Data",
}

if st.session_state.get("page") not in PAGES:
    st.session_state.page = "home"

# ------------------------------------------------------------
# STYLE
# ------------------------------------------------------------

st.markdown(
    r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root{
    --ink:#103f49; --deep:#07566a; --aqua:#0aa8b5; --muted:#63828a;
    --line:#a7d8dc; --pale:#f1fbfb; --green:#22a879; --red:#ef4f5e;
    --blue:#197da5; --shadow:0 14px 38px rgba(9,80,94,.09);
}
html,body,[data-testid="stAppViewContainer"]{
    background:#f1fbfb!important; color:var(--ink)!important;
    font-family:'DM Sans',sans-serif!important;
}
.stApp{
    background:
      radial-gradient(circle at 8% 10%,rgba(55,211,210,.08),transparent 25%),
      radial-gradient(circle at 92% 72%,rgba(19,156,175,.07),transparent 28%),
      linear-gradient(180deg,#fbffff 0%,#effafa 52%,#fbffff 100%)!important;
    overflow-x:hidden;
}
[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer,[data-testid="stSidebar"]{
    display:none!important;
}
.block-container{max-width:1260px!important;padding:24px 34px 70px!important;}
.stApp:before{
    content:"";position:fixed;left:-10%;right:-10%;bottom:-170px;height:300px;z-index:0;
    pointer-events:none;
    background:
      radial-gradient(ellipse at 18% 55%,rgba(24,193,201,.12) 0 17%,transparent 18%),
      radial-gradient(ellipse at 56% 42%,rgba(69,221,218,.09) 0 20%,transparent 21%),
      radial-gradient(ellipse at 88% 60%,rgba(15,171,191,.10) 0 18%,transparent 19%);
    animation:waterMove 13s ease-in-out infinite alternate;
}
@keyframes waterMove{from{transform:translateX(-2%)}to{transform:translateX(2%)}}
.brand-bar{
    position:relative;z-index:10;display:flex;align-items:center;justify-content:space-between;
    gap:20px;padding:15px 20px;border:1.5px solid #c5e4e6;background:rgba(255,255,255,.92);
    border-radius:22px;box-shadow:var(--shadow);backdrop-filter:blur(18px);
}
.brand-left{display:flex;align-items:center;gap:13px}.logo{width:48px;height:48px;border-radius:16px;
    background:linear-gradient(145deg,#29cdd0,#087b8e);display:grid;place-items:center;color:#fff;font-size:20px;font-weight:800;
    box-shadow:0 10px 24px rgba(9,144,157,.18)}
.brand-name{font:800 1.22rem Manrope,sans-serif;color:#103f49;letter-spacing:-.03em}
.brand-sub{font-size:.73rem;color:#78959b;margin-top:2px}.latest-label{text-align:right;color:#78959b;font-size:.68rem;line-height:1.35}
.latest-label b{color:#174b56;font-size:.78rem}
.nav-wrap{position:relative;z-index:12;margin:13px 0 6px}
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{
    min-height:45px!important;border-radius:14px!important;background:#fff!important;border:2px solid #164e5b!important;
    color:#123f49!important;font-weight:800!important;font-size:.87rem!important;box-shadow:0 6px 16px rgba(15,70,82,.09)!important;transition:.18s!important;
}
.stButton>button:hover,.stDownloadButton>button:hover,.stFormSubmitButton>button:hover{
    background:#e2f8f8!important;border-color:#087d8c!important;transform:translateY(-1px)!important;
}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#d8f7f7,#b9eeee)!important;border-color:#078c9b!important;color:#0a5966!important}
.section{position:relative;z-index:2;padding:26px 0 14px}.kicker{font:800 .68rem Manrope,sans-serif;letter-spacing:.18em;text-transform:uppercase;color:#0797a5;margin-bottom:11px}
.section h2{font:800 clamp(2.15rem,4vw,3.65rem)/1.03 Manrope,sans-serif;letter-spacing:-.06em;color:#103f49;margin:0 0 13px}
.section p{color:#5f8088;line-height:1.62;margin:0;max-width:1080px;font-size:.98rem}
.hero{position:relative;z-index:2;overflow:hidden;min-height:360px;border-radius:30px;padding:48px;
    background:linear-gradient(135deg,#063d51,#076d7e 55%,#13a9ad);box-shadow:0 28px 72px rgba(6,86,100,.16)}
.hero:before{content:"";position:absolute;inset:-25%;background:repeating-radial-gradient(ellipse at 20% 115%,transparent 0 55px,rgba(181,255,251,.12) 57px 59px,transparent 61px 105px);transform:rotate(-7deg);animation:waveLines 14s linear infinite}
.hero:after{content:"";position:absolute;left:-5%;right:-5%;bottom:-135px;height:275px;background:rgba(142,244,237,.14);border-radius:50%;animation:heroWave 8s ease-in-out infinite alternate}
.hero-content{position:relative;z-index:2;max-width:820px}.hero .kicker{color:#a6fffa}
.hero h1{font:800 clamp(3rem,6vw,5.8rem)/.91 Manrope,sans-serif;letter-spacing:-.075em;color:#e4ffff;margin:0 0 20px}
.hero p{font-size:1.05rem;line-height:1.78;color:#e0fbfb;max-width:760px}.hero-badges{display:flex;gap:9px;flex-wrap:wrap;margin-top:23px}
.badge{padding:9px 13px;border-radius:999px;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.25);color:#efffff;font-size:.78rem;font-weight:700}
@keyframes waveLines{from{transform:translateX(-4%) rotate(-7deg)}to{transform:translateX(4%) rotate(-7deg)}}
@keyframes heroWave{from{transform:translateX(-2%) rotate(-1deg)}to{transform:translateX(2%) rotate(1deg)}}
.card{position:relative;z-index:2;background:rgba(255,255,255,.88);border:1.5px solid #a9d6da;border-radius:22px;box-shadow:var(--shadow);padding:22px;backdrop-filter:blur(12px)}
.card h3{font:800 1.28rem Manrope,sans-serif;color:#123f49;margin:0 0 8px}.card p{color:#66848b;line-height:1.62;margin:0;font-size:.93rem}
.metric{position:relative;z-index:2;min-height:112px;height:100%;box-sizing:border-box;padding:17px;background:linear-gradient(145deg,#fff,#eaf8f8);border:1.5px solid #a6d5da;border-radius:18px;box-shadow:0 11px 28px rgba(15,91,101,.08);display:flex;flex-direction:column;justify-content:center}
.metric .label{font:800 .64rem Manrope,sans-serif;letter-spacing:.12em;text-transform:uppercase;color:#6d8c93}.metric .value{font:800 1.45rem Manrope,sans-serif;color:#123f49;margin-top:6px;white-space:nowrap}.metric .note{font-size:.73rem;color:#78959b;margin-top:4px}
.story-grid{display:grid;grid-template-columns:1.05fr .95fr;gap:20px;align-items:stretch;margin-top:20px}.story-image{border:1.5px solid #a9d6da;border-radius:22px;overflow:hidden;background:#dff7f8;box-shadow:var(--shadow);padding:8px}.story-image img{display:block;width:100%;height:320px;object-fit:cover;border-radius:16px}
.tool-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin-top:18px}.tool-card{position:relative;z-index:2;min-height:145px;padding:18px;background:rgba(255,255,255,.86);border:1.5px solid #b5dde0;border-radius:19px;box-shadow:var(--shadow);transition:.2s}.tool-card:hover{transform:translateY(-3px)}
.tool-card .icon{font-size:1.35rem;margin-bottom:10px}.tool-card h3{font:800 1.05rem Manrope;color:#123f49;margin:0 0 7px}.tool-card p{font-size:.86rem;line-height:1.55;color:#66848b;margin:0}
.map-shell{position:relative;z-index:2;background:#dff7f8;border-radius:23px;padding:4px;border:2px solid #8fcbd1;box-shadow:0 16px 42px rgba(15,91,101,.09);overflow:hidden}.map-title{min-height:44px;display:flex;align-items:center;justify-content:space-between;gap:15px;padding:0 12px;color:#174b56}.map-title b{font-size:.92rem}.map-title span{font-size:.70rem;color:#6b8b92}
.map-legend{display:flex;align-items:center;gap:9px;flex-wrap:wrap;color:#4f747b;font-size:.79rem;padding:12px 14px}.legend{width:15px;height:10px;border-radius:4px;display:inline-block}.blue{background:#197da5}.green{background:#22a879}.red{background:#ef4f5e;border-radius:50%;width:11px;height:11px}
.timeline-card{margin:16px 0;padding:16px 18px;background:rgba(255,255,255,.92);border:1.5px solid #a9d6da;border-radius:18px;box-shadow:var(--shadow)}.timeline-top{display:flex;justify-content:space-between;gap:15px;align-items:end}.timeline-title{font:800 .9rem Manrope;color:#174b56}.timeline-date{font:800 1.05rem Manrope;color:#103f49}.timeline-help{font-size:.76rem;color:#6c8b92;margin-top:4px}
.status{margin-top:16px;padding:15px 16px;border-radius:15px;border:2px solid}.status.risk{background:#fff0f2;border-color:#f06a78;color:#9e2d3c}.status.normal{background:#eafaf4;border-color:#49b995;color:#176f58}.status-title{font:800 .92rem Manrope}
.result-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:15px}.result-item{padding:12px 13px;background:#f5fcfc;border:1px solid #c7e5e7;border-radius:13px}.result-item span{display:block;font-size:.70rem;color:#78959b;margin-bottom:4px}.result-item b{color:#194b55;font-size:.91rem}
.signal-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}.signal{padding:12px 13px;border-radius:14px;background:#f7fcfc;border:1px solid #c9e5e7}.signal .label{font-size:.69rem;color:#78959b}.signal .value{font:800 .92rem Manrope;color:#164a55;margin-top:4px}
.note{position:relative;z-index:2;margin-top:17px;padding:13px 16px;background:#e6f8f8;border-left:4px solid #11a9b2;border-radius:0 14px 14px 0;color:#52757c;font-size:.86rem;line-height:1.58}
.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}.chart-card{position:relative;z-index:2;background:#fff;border:1.5px solid #b6dfe2;border-radius:20px;box-shadow:var(--shadow);padding:14px}.chart-title{font:800 .95rem Manrope;color:#123f49;padding:4px 4px 10px}
.hotspot-table{position:relative;z-index:2;border-radius:18px;overflow:hidden;border:1px solid #c5e3e5;box-shadow:var(--shadow);background:#fff}.hotspot-table table{width:100%;border-collapse:collapse;font-size:.87rem}.hotspot-table th{background:#0e5663;color:#fff;text-align:left;padding:11px 13px}.hotspot-table td{padding:10px 13px;border-top:1px solid #e3eeee;color:#315b63;background:#fff}.hotspot-table tr:nth-child(even) td{background:#f7fcfc}
.download-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}.download-card{min-height:145px;padding:20px;border-radius:19px;background:linear-gradient(135deg,#087b8b,#13adb3);border:2px solid #07576a;box-shadow:0 13px 28px rgba(8,91,102,.16);color:#fff}.download-card h3{font:800 1.12rem Manrope;color:#fff;margin:0 0 6px}.download-card p{font-size:.84rem;color:#e5ffff;line-height:1.45;margin:0}
.mini-label{font:800 .64rem Manrope,sans-serif;letter-spacing:.14em;text-transform:uppercase;color:#6d8c93;margin-bottom:7px}.footer{position:relative;z-index:2;border-top:1px solid #d5ebed;margin-top:42px;padding-top:16px;color:#76959b;font-size:.70rem}
.stNumberInput input{border:2px solid #164e5b!important;border-radius:12px!important;background:#fff!important}.stSlider label{color:#174b56!important;font-weight:800!important}
@media(max-width:950px){.block-container{padding:18px 18px 55px!important}.tool-grid,.story-grid,.chart-grid,.download-grid{grid-template-columns:1fr 1fr}.signal-grid{grid-template-columns:1fr 1fr}.hero{padding:45px 32px}}
@media(max-width:620px){.tool-grid,.story-grid,.chart-grid,.download-grid,.signal-grid{grid-template-columns:1fr}.hero h1{font-size:3rem}.hero{padding:40px 24px}.block-container{padding:14px 12px 45px!important}.latest-label{display:none}}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def metric(label, value, note):
    st.markdown(
        f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def section(kicker, title, copy):
    st.markdown(
        f'<div class="section"><div class="kicker">{kicker}</div><h2>{title}</h2><p>{copy}</p></div>',
        unsafe_allow_html=True,
    )


def safe_num(value, digits=4):
    if pd.isna(value):
        return "Unavailable"
    return f"{float(value):.{digits}f}"


def probability_text(value):
    if pd.isna(value):
        return "Unavailable"
    x = float(value)
    if x <= 1:
        x *= 100
    return f"{x:.1f}%"


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


def nearest_row(lat, lon):
    a = latest
    # Longitude is adjusted by latitude so that distance is less distorted.
    lon_scale = max(np.cos(np.deg2rad(float(lat))), 0.25)
    dist = (a["latitude"].to_numpy() - lat) ** 2 + (
        (a["longitude"].to_numpy() - lon) * lon_scale
    ) ** 2
    return a.iloc[int(np.argmin(dist))]


def study_data_for_date(date_value):
    d = df[df["date"].eq(pd.Timestamp(date_value))].copy()
    return d[
        d["latitude"].between(LAT_MIN, LAT_MAX)
        & d["plot_lon"].between(LON_MIN, LON_MAX)
    ].copy()


def make_map_for_date(selected_date, detail="Current screening"):
    """Render the lightweight 1-degree historical screening grid."""
    selected_date = pd.Timestamp(selected_date)
    h = history[history["date"].eq(selected_date)].copy()
    h = h[h["lat_bin"].between(LAT_MIN, LAT_MAX) & h["lon_bin"].between(LON_MIN, LON_MAX)]

    if h.empty:
        return go.Figure(), 0, 0

    fig = go.Figure()

    if detail == "Current screening":
        # Keep the full field visible, then overlay concentration zones and risk cells.
        fig.add_trace(go.Scattergeo(
            lat=h["lat_bin"], lon=h["lon_bin"], mode="markers",
            name="Processed field",
            marker=dict(size=6, color="#197da5", opacity=.58),
            text=[f"Chl-a: {x:.4f}" for x in h["chla"]],
            hovertemplate="<b>Processed grid cell</b><br>%{text}<extra></extra>",
        ))

        risk_h = h[h["risk"].eq(1)].copy()
        if not risk_h.empty:
            risk_h["zone_lat"] = np.floor(risk_h["lat_bin"] / 2) * 2 + 1
            risk_h["zone_lon"] = np.floor(risk_h["lon_bin"] / 2) * 2 + 1
            zones = (risk_h.groupby(["zone_lat", "zone_lon"], as_index=False)
                     .size().rename(columns={"size": "flagged_cells"}))
            fig.add_trace(go.Scattergeo(
                lat=zones["zone_lat"], lon=zones["zone_lon"], mode="markers",
                name="Concentration zone",
                marker=dict(size=np.clip(zones["flagged_cells"] * 1.7 + 9, 11, 34),
                            color="#22a879", opacity=.52,
                            line=dict(width=1.2, color="#ffffff")),
                text=zones["flagged_cells"],
                hovertemplate="<b>Concentration zone</b><br>Potential-risk cells: %{text}<extra></extra>",
            ))
            fig.add_trace(go.Scattergeo(
                lat=risk_h["lat_bin"], lon=risk_h["lon_bin"], mode="markers",
                name="Potential-risk cell",
                marker=dict(size=7.5, color="#ef4f5e", opacity=.95,
                            line=dict(width=.8, color="#ffffff")),
                text=[f"Chl-a: {x:.4f}" for x in risk_h["chla"]],
                hovertemplate="<b>Potential bloom-risk screening flag</b><br>%{text}<extra></extra>",
            ))
    else:
        value_col = {
            "Chlorophyll-a": "chla",
            "Change from previous observation": "chla_change",
            "Historical anomaly": "chla_anomaly",
        }[detail]
        plot = h.dropna(subset=[value_col]).copy()
        if not plot.empty:
            if detail == "Chlorophyll-a":
                vals = plot[value_col].astype(float)
                cmin, cmax = float(vals.min()), float(vals.max())
                colorscale = [[0.0, "#e8f7f7"], [0.35, "#45b8b4"], [0.7, "#087b8b"], [1.0, "#063d51"]]
                title = "Chl-a"
            else:
                vals = plot[value_col].astype(float)
                vmax = max(float(vals.abs().quantile(.98)), 1e-6)
                cmin, cmax = -vmax, vmax
                colorscale = [[0.0, "#197da5"], [0.5, "#f4fbfb"], [1.0, "#ef4f5e"]]
                title = "Change" if detail.startswith("Change") else "Anomaly"
            fig.add_trace(go.Scattergeo(
                lat=plot["lat_bin"], lon=plot["lon_bin"], mode="markers",
                name=detail,
                marker=dict(size=7, color=plot[value_col], colorscale=colorscale,
                            cmin=cmin, cmax=cmax,
                            colorbar=dict(title=title, thickness=13, len=.60)),
                text=[f"{title}: {x:.4f}" for x in plot[value_col]],
                hovertemplate="<b>%{text}</b><extra></extra>",
            ))

    fig.update_geos(
        showland=True, landcolor="#dce9e7", showocean=True, oceancolor="#dff7f8",
        showcoastlines=True, coastlinecolor="#4d9099", coastlinewidth=1,
        showcountries=True, countrycolor="#9ab9be", bgcolor="#dff7f8",
        lataxis_range=[LAT_MIN, LAT_MAX], lonaxis_range=[LON_MIN, LON_MAX],
        center=dict(lat=-5, lon=70), projection_type="equirectangular",
    )
    fig.update_layout(
        height=540, margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor="#dff7f8", plot_bgcolor="#dff7f8",
        font=dict(color="#174b56"), showlegend=(detail == "Current screening"),
        legend=dict(orientation="h", yanchor="bottom", y=.01, xanchor="left", x=.01,
                    bgcolor="rgba(255,255,255,.92)", bordercolor="#b9dfe2", borderwidth=1,
                    font=dict(size=10, color="#174b56")),
    )
    return fig, len(h), int(h["risk"].sum())

def hotspot_table(data):
    r = data[data["risk_flag"]].copy()
    if r.empty:
        return pd.DataFrame()
    r["lat_zone"] = np.floor(r["latitude"] / 2) * 2
    r["lon_zone"] = np.floor(r["plot_lon"] / 2) * 2
    z = (r.groupby(["lat_zone","lon_zone"],as_index=False)
           .agg(flagged_cells=("risk_flag","size"),mean_chla=("chla","mean"),max_chla=("chla","max"))
           .sort_values(["flagged_cells","mean_chla"],ascending=False).head(12))
    z["Zone"] = z.apply(lambda x:f"{x.lat_zone:.0f}°–{x.lat_zone+2:.0f}°, {x.lon_zone:.0f}°–{x.lon_zone+2:.0f}°",axis=1)
    z["Flagged cells"] = z["flagged_cells"].astype(int)
    z["Mean Chl-a"] = z["mean_chla"].round(4)
    z["Max Chl-a"] = z["max_chla"].round(4)
    return z[["Zone","Flagged cells","Mean Chl-a","Max Chl-a"]]


def navigate(page):
    st.session_state.page = page
    st.rerun()

# ------------------------------------------------------------
# HEADER / NAV
# ------------------------------------------------------------

st.markdown(
    f'<div class="brand-bar"><div class="brand-left"><div class="logo">≈</div><div><div class="brand-name">BloomDetect AI</div><div class="brand-sub">Coastal & Ocean Intelligence · EOS-06 OCM-3</div></div></div><div class="latest-label">Latest processed field<br><b>{latest_date.strftime("%d %b %Y")}</b></div></div>',
    unsafe_allow_html=True,
)

nav_cols = st.columns(len(PAGES), gap="small")
for col, page in zip(nav_cols, PAGES):
    with col:
        if st.button(NAV[page], key=f"nav_{page}", width="stretch", type="primary" if st.session_state.page == page else "secondary"):
            navigate(page)

# ============================================================
# HOME
# ============================================================

if st.session_state.page == "home":
    st.markdown(
        '<div class="hero"><div class="hero-content"><div class="kicker">EOS-06 · OCM-3 · SATELLITE INTELLIGENCE</div><h1>Read the ocean signal.</h1><p>BloomDetect AI turns satellite-derived chlorophyll-a observations and the project screening output into a practical view of where unusual patterns may deserve closer investigation.</p><div class="hero-badges"><span class="badge">🌊 Ocean colour</span><span class="badge">🛰 EOS-06 OCM-3</span><span class="badge">🌱 Potential bloom-risk screening</span><span class="badge">📍 Spatial intelligence</span></div></div></div>',
        unsafe_allow_html=True,
    )

    risk_share = 100 * len(risk) / len(latest) if len(latest) else 0
    cols = st.columns(4, gap="medium")
    for col,(label,value,note) in zip(cols,[
        ("Processed cells",f"{len(latest):,}","latest field"),
        ("Potential-risk cells",f"{len(risk):,}","screening output"),
        ("Risk share",f"{risk_share:.2f}%","of processed cells"),
        ("Maximum Chl-a",safe_num(latest["chla"].max()),"latest field"),
    ]):
        with col: metric(label,value,note)

    st.markdown('<div class="section"><div class="kicker">PROJECT SNAPSHOT</div><h2>One clear view of the signal.</h2><p>The home page stays intentionally light. The analysis pages contain the actual investigation tools, so the same information is not repeated five times.</p></div>',unsafe_allow_html=True)

    left,right = st.columns([1.05,.95],gap="large")
    with left:
        st.markdown('<div class="card"><div class="mini-label">WHAT BLOOMDETECT ADDS</div><h3>From satellite observation to investigation support.</h3><p>It combines the latest satellite-derived chlorophyll-a field with the project screening result, then exposes that result through a temporal map, coordinate lookup, hotspot analysis and compact insights.</p><div class="note"><b>Scientific note:</b> a potential bloom-risk flag is not confirmation of a harmful algal bloom. Species, toxin presence and ecological impact require additional evidence and field validation.</div></div>',unsafe_allow_html=True)
    with right:
        if BLOOM_IMAGE.exists():
            st.markdown('<div class="story-image">',unsafe_allow_html=True)
            st.image(str(BLOOM_IMAGE),width="stretch",caption="How satellite ocean-colour observations can support bloom-risk investigation")
            st.markdown('</div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><h3>EOS-06 ocean-colour observation</h3><p>Add bloomdetect_bloom_process.png beside app.py to display the project illustration here.</p></div>',unsafe_allow_html=True)

    st.markdown('<div class="section"><div class="kicker">EXPLORE THE PLATFORM</div><h2>Four focused tools.</h2><p>Each page has one job. No duplicate dashboards wearing different hats.</p></div>',unsafe_allow_html=True)
    tools=[
        ("🗺️","Risk Map","Move through available dates and see how the spatial screening field changes.","map"),
        ("📍","Location","Check a coordinate against the nearest processed satellite grid cell.","location"),
        ("🔥","Insights","See spatial concentrations, regional patterns and the latest Chl-a distribution in one analysis page.","insights"),
        ("⇩","Data","Download the latest observation table, screening shortlist and hotspot output.","data"),
    ]
    tcols=st.columns(4,gap="medium")
    for col,(icon,title,desc,target) in zip(tcols,tools):
        with col:
            st.markdown(f'<div class="tool-card"><div class="icon">{icon}</div><h3>{title}</h3><p>{desc}</p></div>',unsafe_allow_html=True)
            if st.button(f"Open {title}",key=f"home_{target}",width="stretch"): navigate(target)

# ============================================================
# RISK MAP
# ============================================================

elif st.session_state.page == "map":
    section("01 · SPATIAL INTELLIGENCE", "See how the field changes.", "Move through the available satellite observations and inspect the same Indian Ocean study window over time.")

    if not history.empty:
        available_dates = sorted(pd.to_datetime(history["date"].dropna().unique()))
        selected_default = pd.Timestamp(available_dates[-1])
        if "map_selected_date" in st.session_state:
            try:
                candidate = pd.Timestamp(st.session_state.map_selected_date)
                if candidate in pd.to_datetime(available_dates):
                    selected_default = candidate
            except Exception:
                pass
        selected_date = st.select_slider(
            "Observation date",
            options=[pd.Timestamp(x) for x in available_dates],
            value=selected_default,
            format_func=lambda x: pd.Timestamp(x).strftime("%d %b %Y"),
            key="map_date_slider",
        )
        st.session_state.map_selected_date = selected_date
    else:
        selected_date = latest_date
        st.error("Historical map data is missing. Upload bloomdetect_history_web.csv.gz beside app.py.")

    selected_detail = st.selectbox(
        "Map detail",
        ["Current screening", "Chlorophyll-a", "Change from previous observation", "Historical anomaly"],
        key="map_detail",
        help="Choose which satellite-derived signal to emphasize on the selected date."
    )

    fig, processed_count, risk_count = make_map_for_date(selected_date, selected_detail)
    share = 100 * risk_count / processed_count if processed_count else 0

    m1,m2,m3,m4=st.columns(4,gap="medium")
    with m1: metric("Selected date",selected_date.strftime("%d %b %Y"),"observation")
    with m2: metric("History grid cells",f"{processed_count:,}","1° screening grid")
    with m3: metric("Potential-risk cells",f"{risk_count:,}","screening output")
    with m4: metric("Risk share",f"{share:.2f}%","selected field")

    st.markdown(f'<div class="map-shell"><div class="map-title"><div><b>Indian Ocean · Arabian Sea · Bay of Bengal</b><br><span>{selected_detail} · historical 1° screening grid</span></div><span>{selected_date.strftime("%d %b %Y")}</span></div>',unsafe_allow_html=True)
    st.plotly_chart(fig,width="stretch",config={"displaylogo":False,"scrollZoom":False,"modeBarButtonsToRemove":["lasso2d","select2d"]})
    st.markdown('<div class="map-legend"><span class="legend blue"></span><span>Processed field</span><span class="legend green"></span><span>Concentration zone</span><span class="legend red"></span><span>Potential-risk cell</span></div></div>',unsafe_allow_html=True)
    st.markdown('<div class="note"><b>Reading the map:</b> the screening layer indicates potential bloom-risk patterns. It is not confirmation of a harmful algal bloom, species identity or toxin presence.</div>',unsafe_allow_html=True)

# ============================================================
# LOCATION
# ============================================================

elif st.session_state.page == "location":
    section("02 · LOCATION INTELLIGENCE","Check one coordinate across the available evidence.","Your input is always shown separately from the nearest processed 0.25° satellite cell, so there is no confusion about which location was actually evaluated.")

    if "checked_lat" not in st.session_state: st.session_state.checked_lat=18.0
    if "checked_lon" not in st.session_state: st.session_state.checked_lon=78.0

    left,right=st.columns([.82,1.18],gap="large")
    with left:
        st.markdown('<div class="card"><div class="mini-label">YOUR INPUT</div><h3>Coordinates</h3><p>Enter decimal degrees. Example: 17.38, 78.49.</p></div>',unsafe_allow_html=True)
        with st.form("location_form",clear_on_submit=False):
            lat=st.number_input("Latitude",min_value=-90.0,max_value=90.0,value=float(st.session_state.checked_lat),step=.25,format="%.4f")
            lon=st.number_input("Longitude",min_value=-180.0,max_value=180.0,value=float(st.session_state.checked_lon),step=.25,format="%.4f")
            submitted=st.form_submit_button("🔎 Check location",width="stretch",type="primary")
        if submitted:
            st.session_state.checked_lat=float(lat); st.session_state.checked_lon=float(lon)

    lat=float(st.session_state.checked_lat); lon=float(st.session_state.checked_lon)
    row=nearest_row(lat,lon); flagged=bool(row["risk_flag"])

    with right:
        st.markdown(f'<div class="card"><div class="mini-label">NEAREST PROCESSED CELL</div><h3 style="font-size:1.65rem">{float(row.latitude):.4f}° · {float(row.longitude):.4f}°</h3><p>This is the satellite grid cell used for the lookup. <b>Your input:</b> {lat:.4f}° · {lon:.4f}°</p><div class="result-grid"><div class="result-item"><span>Observation date</span><b>{row.date.strftime("%d %b %Y")}</b></div><div class="result-item"><span>Chlorophyll-a</span><b>{safe_num(row.chla)}</b></div><div class="result-item"><span>Risk probability</span><b>{probability_text(row.get("risk_probability",np.nan))}</b></div><div class="result-item"><span>Region</span><b>{region_name(row.latitude,row.longitude)}</b></div></div></div>',unsafe_allow_html=True)
        if flagged:
            st.markdown('<div class="status risk"><div class="status-title">🔴 POTENTIAL BLOOM-RISK FLAG</div><div>This processed cell is included in the current screening shortlist.</div></div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="status normal"><div class="status-title">🟢 NOT FLAGGED</div><div>This processed cell is not included in the current potential-risk shortlist.</div></div>',unsafe_allow_html=True)

    st.markdown('<div class="mini-label" style="margin-top:22px">SUPPORTING SIGNALS</div>',unsafe_allow_html=True)
    signals=[("Current Chl-a",safe_num(row.chla)),("Historical baseline",safe_num(row.get("historical_baseline",np.nan))),("Anomaly",safe_num(row.get("chla_anomaly",np.nan))), ("Recent change",safe_num(row.get("chla_change",np.nan)))]
    sc=st.columns(4,gap="small")
    for col,(label,value) in zip(sc,signals):
        with col: st.markdown(f'<div class="signal"><div class="label">{label}</div><div class="value">{value}</div></div>',unsafe_allow_html=True)

    report=pd.DataFrame([{
        "input_latitude":lat,"input_longitude":lon,"nearest_processed_latitude":row.latitude,"nearest_processed_longitude":row.longitude,
        "date":row.date.strftime("%Y-%m-%d"),"chla":row.chla,"historical_baseline":row.get("historical_baseline",np.nan),
        "chla_anomaly":row.get("chla_anomaly",np.nan),"chla_change":row.get("chla_change",np.nan),"risk_label":row.risk_label,
        "risk_probability":row.get("risk_probability",np.nan),
    }])
    st.download_button("⬇ Download location report",report.to_csv(index=False).encode(),"bloomdetect_location_report.csv","text/csv",width="stretch")

# ============================================================
# INSIGHTS = HOTSPOTS + INSIGHTS TOGETHER
# ============================================================

elif st.session_state.page == "insights":
    section("03 · OCEAN INTELLIGENCE","What stands out in the latest field?","Hotspot concentration and data insights are combined here because they answer the same question: where should the current satellite signal receive closer investigation?")

    total_flags=int(latest["risk_flag"].sum()); normal_count=len(latest)-total_flags; share=100*total_flags/len(latest) if len(latest) else 0
    top=st.columns(4,gap="medium")
    for col,(label,value,note) in zip(top,[
        ("Observation cells",f"{len(latest):,}","latest field"),("Potential-risk",f"{total_flags:,}","screening output"),
        ("Risk share",f"{share:.2f}%","latest field"),("Mean Chl-a",safe_num(latest["chla"].mean()),"latest field")]):
        with col: metric(label,value,note)

    # ---------------- HOTSPOTS ----------------
    zones=hotspot_table(latest)
    st.markdown('<div class="section" style="padding-top:30px"><div class="kicker">HOTSPOT INTELLIGENCE</div><h2>Where are the flags concentrating?</h2><p>Nearby screening flags are grouped into 2° × 2° investigation zones. This is a spatial concentration view, not a severity ranking.</p></div>',unsafe_allow_html=True)

    if zones.empty:
        st.markdown('<div class="card"><h3>No potential-risk cells in the latest study area.</h3><p>The latest processed field contains no screening flags inside the selected study window.</p></div>',unsafe_allow_html=True)
    else:
        hz=latest[latest["risk_flag"]].copy(); hz["lat_zone"]=np.floor(hz["latitude"]/2)*2+1; hz["lon_zone"]=np.floor(hz["plot_lon"]/2)*2+1
        plot_zones=(hz.groupby(["lat_zone","lon_zone"],as_index=False).size().rename(columns={"size":"flagged_cells"}).sort_values("flagged_cells",ascending=False).head(12))
        hfig=px.bar(plot_zones,x="flagged_cells",y=plot_zones.apply(lambda x:f"{x.lat_zone:.0f}°–{x.lat_zone+2:.0f}°, {x.lon_zone:.0f}°–{x.lon_zone+2:.0f}°",axis=1),orientation="h",text="flagged_cells")
        hfig.update_traces(marker_color="#22a879",textposition="outside",cliponaxis=False)
        hfig.update_layout(height=430,margin=dict(l=155,r=45,t=20,b=65),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fff",showlegend=False,
            xaxis=dict(title="Potential-risk cells in zone",title_font=dict(color="#174b56",size=13),tickfont=dict(color="#174b56",size=11),gridcolor="#d6e7e8"),
            yaxis=dict(title="Investigation zone",title_font=dict(color="#174b56",size=13),tickfont=dict(color="#174b56",size=10)),font=dict(color="#174b56"))
        st.markdown('<div class="chart-card"><div class="chart-title">Top investigation zones</div>',unsafe_allow_html=True)
        st.plotly_chart(hfig,width="stretch",config={"displaylogo":False})
        st.markdown('</div>',unsafe_allow_html=True)

        st.markdown('<div class="mini-label" style="margin-top:18px">ZONE DETAILS</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="hotspot-table">{zones.to_html(index=False,border=0)}</div>',unsafe_allow_html=True)

    # ---------------- CURRENT FIELD ----------------
    st.markdown('<div class="section" style="padding-top:32px"><div class="kicker">CURRENT FIELD</div><h2>What does the latest observation look like?</h2><p>These charts use the selected field and do not repeat the spatial map.</p></div>',unsafe_allow_html=True)

    c1,c2=st.columns(2,gap="large")
    with c1:
        clean=latest["chla"].replace([np.inf,-np.inf],np.nan).dropna()
        f1=px.histogram(pd.DataFrame({"Chlorophyll-a":clean}),x="Chlorophyll-a",nbins=32)
        f1.update_traces(marker_color="#197da5")
        f1.update_layout(height=390,margin=dict(l=60,r=25,t=20,b=70),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fff",showlegend=False,font=dict(color="#174b56"),
            xaxis=dict(title="Satellite-derived chlorophyll-a",title_font=dict(color="#174b56",size=13),tickfont=dict(color="#174b56",size=11),gridcolor="#d6e7e8"),
            yaxis=dict(title="Number of processed cells",title_font=dict(color="#174b56",size=13),tickfont=dict(color="#174b56",size=11),gridcolor="#d6e7e8"))
        st.markdown('<div class="chart-card"><div class="chart-title">Chlorophyll-a distribution</div>',unsafe_allow_html=True); st.plotly_chart(f1,width="stretch",config={"displaylogo":False}); st.markdown('</div>',unsafe_allow_html=True)

    with c2:
        comp=pd.DataFrame({"Screening group":["Normal","Potential bloom risk"],"Mean chlorophyll-a":[latest.loc[~latest.risk_flag,"chla"].mean(),latest.loc[latest.risk_flag,"chla"].mean()]})
        f2=px.bar(comp,x="Screening group",y="Mean chlorophyll-a",text="Mean chlorophyll-a")
        f2.update_traces(marker_color=["#197da5","#ef4f5e"],texttemplate="%{text:.4f}",textposition="outside",cliponaxis=False)
        f2.update_layout(height=390,margin=dict(l=60,r=25,t=20,b=70),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fff",showlegend=False,font=dict(color="#174b56"),
            xaxis=dict(title="Screening group",title_font=dict(color="#174b56",size=13),tickfont=dict(color="#174b56",size=11)),
            yaxis=dict(title="Mean chlorophyll-a",title_font=dict(color="#174b56",size=13),tickfont=dict(color="#174b56",size=11),gridcolor="#d6e7e8"))
        st.markdown('<div class="chart-card"><div class="chart-title">Mean chlorophyll-a by screening group</div>',unsafe_allow_html=True); st.plotly_chart(f2,width="stretch",config={"displaylogo":False}); st.markdown('</div>',unsafe_allow_html=True)

    # ---------------- REGIONAL SIGNAL ----------------
    regional_defs=[
        ("Arabian Sea",latest.latitude.between(5,30)&latest.plot_lon.between(45,75)),
        ("Bay of Bengal",latest.latitude.between(0,25)&latest.plot_lon.between(75.01,100)),
        ("Southern Indian Ocean",latest.latitude.between(-30,5)&latest.plot_lon.between(40,100)),
        ("Northern Indian Ocean",latest.latitude.between(5,30)&latest.plot_lon.between(75.01,120)),
    ]
    rows=[]
    for name,cond in regional_defs:
        sub=latest[cond]
        if len(sub): rows.append({"Region":name,"Cells":len(sub),"Potential-risk cells":int(sub.risk_flag.sum()),"Risk share":100*sub.risk_flag.mean(),"Mean Chl-a":sub.chla.mean()})
    regional=pd.DataFrame(rows)
    if not regional.empty:
        regional["Risk share"]=regional["Risk share"].map(lambda x:f"{x:.2f}%")
        regional["Mean Chl-a"]=regional["Mean Chl-a"].round(4)
        st.markdown('<div class="section" style="padding-top:28px"><div class="kicker">REGIONAL SIGNAL</div><h2>Broad-area comparison.</h2><p>These are descriptive regional summaries of the latest processed field.</p></div>',unsafe_allow_html=True)
        st.dataframe(regional,hide_index=True,width="stretch")

    st.markdown('<div class="note"><b>Interpretation:</b> the charts describe satellite-derived chlorophyll-a and the project screening output. They do not independently establish harmfulness, species identity or toxin presence.</div>',unsafe_allow_html=True)
    if not zones.empty:
        st.download_button("⬇ Download hotspot report",zones.to_csv(index=False).encode(),"bloomdetect_hotspot_report.csv","text/csv",width="stretch")

# ============================================================
# DATA
# ============================================================

elif st.session_state.page == "data":
    section("04 · DATA & OUTPUTS","The evidence behind the dashboard.","Keep the source product, current processed field and downloadable outputs in one place. No duplicate analysis panels here.")

    cols=st.columns(4,gap="medium")
    for col,(label,value,note) in zip(cols,[
        ("Product","E06OCM_L4_AC","EOS-06 / OCM-3"),("Grid","0.25°","latitude × longitude"),
        ("Latest cells",f"{len(latest):,}","processed observation"),("Latest date",latest_date.strftime("%d %b %Y"),"processed dataset")]):
        with col: metric(label,value,note)

    st.markdown('<div class="card" style="margin-top:18px"><div class="mini-label">SOURCE PRODUCT</div><h3>EOS-06 OCM-3 analysed chlorophyll-a</h3><p>The dashboard uses the EOS-06 / Oceansat-3 OCM-3 Level-4 analysed chlorophyll product, E06OCM_L4_AC. The current project output is a potential bloom-risk screening layer built from the satellite-derived chlorophyll-a signal and temporal context.</p></div>',unsafe_allow_html=True)

    st.markdown('<div class="section" style="padding-top:30px"><div class="kicker">DOWNLOADS</div><h2>Take the actual project outputs.</h2><p>These downloads are directly generated from the data used by the dashboard.</p></div>',unsafe_allow_html=True)
    d1,d2=st.columns(2,gap="large")
    with d1:
        st.markdown('<div class="download-card"><h3>Latest observation table</h3><p>All processed cells from the latest available field, including the stored screening result and supporting signals.</p></div>',unsafe_allow_html=True)
        st.download_button("⬇ Download latest observations",latest.to_csv(index=False).encode(),"bloomdetect_latest_observations.csv","text/csv",width="stretch")
    with d2:
        st.markdown('<div class="download-card"><h3>Potential-risk shortlist</h3><p>Only cells currently screened as potential bloom risk in the latest processed field.</p></div>',unsafe_allow_html=True)
        st.download_button("⬇ Download potential-risk locations",risk.to_csv(index=False).encode(),"bloomdetect_potential_risk.csv","text/csv",width="stretch")

    st.markdown('<div class="section" style="padding-top:30px"><div class="kicker">SCIENTIFIC USE</div><h2>Use the screening result correctly.</h2></div><div class="card"><p><b>Potential bloom risk</b> is a project screening output. It is not confirmation of a harmful algal bloom, species identity or toxin presence. Satellite observations should be combined with field observations and additional environmental evidence.</p></div>',unsafe_allow_html=True)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.markdown(f'<div class="footer">BloomDetect AI · EOS-06 / OCM-3 · Potential bloom-risk screening · Latest processed field: {latest_date.strftime("%d %B %Y")}</div>',unsafe_allow_html=True)
