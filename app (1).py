
import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide"
)

# --------------------------------------------------
# Load prediction data
# --------------------------------------------------

DATA_PATH = "/content/latest_bloom_risk_predictions.csv"

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

latest_date = df["date"].max()

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🌊 BloomDetect AI")
st.subheader("AI-Based Harmful Algal Bloom Detection and Early Warning System")

st.info(
    "This system identifies locations with potential bloom risk "
    "using EOS-06 OCM-3 chlorophyll-a observations and machine learning."
)

# --------------------------------------------------
# Summary statistics
# --------------------------------------------------

total_locations = len(df)

risk_locations = (
    df["risk_label"] == "Potential Bloom Risk"
).sum()

normal_locations = (
    df["risk_label"] == "Normal"
).sum()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Observation Locations",
        f"{total_locations:,}"
    )

with col2:
    st.metric(
        "Potential Bloom Risk",
        f"{risk_locations:,}"
    )

with col3:
    st.metric(
        "Latest Date",
        latest_date.strftime("%d %b %Y")
    )

# --------------------------------------------------
# Risk Map
# --------------------------------------------------

st.markdown("## 🗺️ Potential Bloom Risk Map")

risk_data = df[
    df["risk_label"] == "Potential Bloom Risk"
].copy()

fig = px.scatter_geo(
    risk_data,
    lat="latitude",
    lon="longitude",
    hover_data=[
        "date",
        "chla",
        "risk_probability"
    ],
    projection="mercator",
    title=f"Potential Bloom Risk Areas - {latest_date.date()}"
)

fig.update_geos(
    showcountries=True,
    showcoastlines=True,
    showland=True,
    coastlinecolor="black",
    lataxis_range=[-40, 30],
    lonaxis_range=[20, 120]
)

fig.update_traces(
    marker=dict(
        size=8,
        color="red"
    )
)

fig.update_layout(
    height=650
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# Risk information
# --------------------------------------------------

st.markdown("## 📊 Risk Summary")

st.write(
    f"**{risk_locations:,} locations** were flagged as "
    f"potential bloom-risk areas on "
    f"**{latest_date.strftime('%d %B %Y')}**."
)

st.caption(
    "Note: Potential bloom risk is a model-derived indicator "
    "and does not represent confirmed harmful algal bloom occurrence."
)
