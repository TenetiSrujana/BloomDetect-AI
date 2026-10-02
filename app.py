import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from collections import deque

# ============================================================
# BLOOMDETECT AI
# Final-year major project dashboard
# EOS-06 / OCM-3 potential bloom-risk screening
# ============================================================

st.set_page_config(
    page_title="BloomDetect AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root {
    --navy:#07324a;
    --blue:#0b6280;
    --aqua:#16a9b5;
    --aqua-soft:#e8f8fa;
    --ink:#173747;
    --muted:#637d88;
    --line:#c9e1e6;
    --bg:#f8fcfd;
    --white:#ffffff;
    --red:#d9414a;
    --red-soft:#fff1f2;
    --green:#138a63;
    --green-soft:#edf9f4;
    --yellow:#a86c00;
    --yellow-soft:#fff8e8;
}

html, body, [class*="css"] {
    font-family:"DM Sans",sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 5%, rgba(22,169,181,.08), transparent 22%),
        radial-gradient(circle at 92% 12%, rgba(11,98,128,.07), transparent 24%),
        linear-gradient(180deg,#ffffff 0%,#f8fcfd 58%,#f4fafb 100%);
    color:var(--ink);
}

.block-container {
    max-width:1450px;
    padding-top:1.2rem;
    padding-bottom:4rem;
}

section[data-testid="stSidebar"] {
    background:#f7fcfd;
    border-right:1px solid var(--line);
}

section[data-testid="stSidebar"] > div {
    padding-top:1.3rem;
}

h1,h2,h3,h4 {
    font-family:"Manrope",sans-serif;
    color:var(--navy);
}

p,li,label,.stMarkdown {
    color:var(--ink);
}

.brandbar {
    border:2px solid var(--navy);
    border-radius:20px;
    background:rgba(255,255,255,.96);
    padding:14px 16px 12px;
    margin-bottom:16px;
    box-shadow:0 8px 28px rgba(7,50,74,.07);
}

.brandrow {
    display:flex;
    align-items:center;
    gap:12px;
}
