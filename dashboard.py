"""
Pfizer Prescriber Clustering — Executive Dashboard

Run with:
    streamlit run dashboard.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score, v_measure_score,
                              adjusted_rand_score, normalized_mutual_info_score)
from scipy.cluster.hierarchy import linkage, fcluster

# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────
SEED = 42

# Pfizer brand-inspired palette
PFIZER_BLUE       = "#0072CE"
PFIZER_DARK_BLUE  = "#003B71"
PFIZER_LIGHT_BLUE = "#00B5E2"
PFIZER_ORANGE     = "#F47B20"
PFIZER_GREEN      = "#00A651"
PFIZER_PURPLE     = "#7C3F98"
PFIZER_GRAY       = "#535554"

CLUSTER_PALETTE = [
    PFIZER_BLUE, PFIZER_ORANGE, PFIZER_GREEN,
    PFIZER_PURPLE, "#FDB913", PFIZER_LIGHT_BLUE,
    "#A6228C", PFIZER_GRAY,
]

ATSEG_COLORS = {
    "SEG_A": PFIZER_BLUE,
    "SEG_B": PFIZER_ORANGE,
    "SEG_C": PFIZER_GREEN,
    "No ATSEG": "#CFD8DC",
}

# ─────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pfizer Prescriber Clustering",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────
# Custom CSS — executive look
# ─────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Premium typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    /* Global — Dark Mode */
    .stApp, [data-testid="stAppViewContainer"], .main,
    [data-testid="stAppViewContainer"] > .main {
        background:
            radial-gradient(ellipse 1200px 600px at 0% 0%, rgba(0, 114, 206, 0.10) 0%, transparent 50%),
            radial-gradient(ellipse 800px 400px at 100% 0%, rgba(124, 63, 152, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse 600px 400px at 50% 100%, rgba(0, 181, 226, 0.06) 0%, transparent 50%),
            #0B1120 !important;
        color: #CBD5E1;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }

    /* ── Premium Animations ── */
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes borderGlow {
        0%   { opacity: 0.4; }
        50%  { opacity: 1; }
        100% { opacity: 0.4; }
    }
    @keyframes pulseDot {
        0%, 100% { box-shadow: 0 0 0 0 currentColor; }
        50%      { box-shadow: 0 0 0 4px transparent; }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 0.3; }
        25%      { transform: translateY(-15px) rotate(3deg); opacity: 0.6; }
        50%      { transform: translateY(-25px) rotate(-2deg); opacity: 0.4; }
        75%      { transform: translateY(-10px) rotate(1deg); opacity: 0.5; }
    }
    @keyframes breathe {
        0%, 100% { box-shadow: 0 0 15px rgba(0,114,206,0.05); }
        50%      { box-shadow: 0 0 25px rgba(0,114,206,0.15); }
    }

    /* ── Floating particles (background decoration) ── */
    .main::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image:
            radial-gradient(2px 2px at 10% 20%, rgba(0,181,226,0.25) 0%, transparent 100%),
            radial-gradient(2px 2px at 30% 60%, rgba(0,114,206,0.20) 0%, transparent 100%),
            radial-gradient(1.5px 1.5px at 55% 15%, rgba(124,63,152,0.18) 0%, transparent 100%),
            radial-gradient(2px 2px at 70% 75%, rgba(0,181,226,0.15) 0%, transparent 100%),
            radial-gradient(1.5px 1.5px at 85% 35%, rgba(244,123,32,0.12) 0%, transparent 100%),
            radial-gradient(2px 2px at 20% 85%, rgba(0,166,81,0.15) 0%, transparent 100%),
            radial-gradient(1px 1px at 45% 45%, rgba(255,255,255,0.08) 0%, transparent 100%),
            radial-gradient(1px 1px at 90% 90%, rgba(255,255,255,0.06) 0%, transparent 100%);
        pointer-events: none;
        z-index: 0;
        animation: float 20s ease-in-out infinite;
    }
    .main > .block-container { position: relative; z-index: 1; }

    /* ── Staggered entrance for KPI cards ── */
    .kpi-card:nth-child(1) { animation-delay: 0.05s; }
    .kpi-card:nth-child(2) { animation-delay: 0.10s; }
    .kpi-card:nth-child(3) { animation-delay: 0.15s; }
    .kpi-card:nth-child(4) { animation-delay: 0.20s; }
    .kpi-card:nth-child(5) { animation-delay: 0.25s; }

    /* Hero header — premium dark */
    .hero {
        position: relative;
        background:
            radial-gradient(ellipse at top right, rgba(0,181,226,0.22) 0%, transparent 55%),
            radial-gradient(ellipse at bottom left, rgba(124,63,152,0.15) 0%, transparent 50%),
            linear-gradient(135deg, #0a1628 0%, #0d2847 35%, #0f3d6e 100%);
        background-size: 200% 200%;
        animation: slideUp 0.5s ease, gradientShift 12s ease infinite;
        color: white;
        padding: 36px 44px 30px 44px;
        border-radius: 20px;
        margin-bottom: 28px;
        box-shadow:
            0 12px 48px rgba(0, 0, 0, 0.55),
            0 0 80px rgba(0, 114, 206, 0.18),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
        overflow: hidden;
        border: 1px solid rgba(0, 114, 206, 0.25);
    }
    /* Decorative shapes */
    .hero::before {
        content: "";
        position: absolute;
        top: -50%; right: -10%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(0,181,226,0.30) 0%, transparent 65%);
        pointer-events: none;
    }
    .hero::after {
        content: "";
        position: absolute;
        bottom: -60%; left: 30%;
        width: 360px; height: 360px;
        background: radial-gradient(circle, rgba(244,123,32,0.18) 0%, transparent 65%);
        pointer-events: none;
    }
    .hero-content {
        position: relative;
        z-index: 1;
    }
    .hero-eyebrow {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 2.4px;
        opacity: 0.85;
        font-weight: 600;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-eyebrow::before {
        content: "";
        display: inline-block;
        width: 24px;
        height: 2px;
        background: #00B5E2;
        border-radius: 2px;
    }
    .hero h1 {
        margin: 0;
        font-size: 34px;
        font-weight: 800;
        letter-spacing: -0.6px;
        line-height: 1.1;
        background: linear-gradient(135deg, #ffffff 0%, #93C5FD 40%, #00B5E2 70%, #C084FC 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 6s ease infinite;
    }
    .hero p {
        margin: 8px 0 0 0;
        opacity: 0.86;
        font-size: 14px;
        font-weight: 400;
        max-width: 660px;
        line-height: 1.5;
    }
    .hero-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,181,226,0.4), rgba(124,63,152,0.3), transparent);
        background-size: 200% 100%;
        animation: shimmer 4s linear infinite;
        margin: 18px 0 14px 0;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255,255,255,0.12);
        backdrop-filter: blur(6px);
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 12px;
        margin-right: 8px;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.18);
    }
    .hero-pill b {
        font-weight: 700;
        color: #FFE7D2;
    }

    /* KPI cards — glassmorphic dark with animated border */
    .kpi-card {
        position: relative;
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 20px 22px 18px 22px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow:
            0 4px 24px rgba(0, 0, 0, 0.30),
            inset 0 1px 0 rgba(255,255,255,0.05);
        height: 100%;
        transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
        overflow: hidden;
        animation: slideUp 0.5s ease both, breathe 4s ease-in-out infinite;
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #0072CE, #00B5E2, #0072CE);
        background-size: 200% 100%;
        animation: shimmer 3s linear infinite;
    }
    .kpi-card::after {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 60px;
        background: linear-gradient(180deg, rgba(0,114,206,0.06), transparent);
        pointer-events: none;
    }
    .kpi-card:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow:
            0 12px 40px rgba(0, 0, 0, 0.45),
            0 0 30px rgba(0, 114, 206, 0.15),
            inset 0 1px 0 rgba(255,255,255,0.08);
        border-color: rgba(0, 114, 206, 0.30);
    }
    .kpi-card.good::before    { background: linear-gradient(90deg, #00A651, #4ADE80, #00A651); background-size: 200% 100%; animation: shimmer 3s linear infinite; }
    .kpi-card.good::after     { background: linear-gradient(180deg, rgba(0,166,81,0.06), transparent); }
    .kpi-card.good:hover      { box-shadow: 0 12px 40px rgba(0,0,0,0.45), 0 0 25px rgba(0,166,81,0.15); border-color: rgba(0,166,81,0.30); }
    .kpi-card.warn::before    { background: linear-gradient(90deg, #F47B20, #FDB913, #F47B20); background-size: 200% 100%; animation: shimmer 3s linear infinite; }
    .kpi-card.warn::after     { background: linear-gradient(180deg, rgba(244,123,32,0.06), transparent); }
    .kpi-card.warn:hover      { box-shadow: 0 12px 40px rgba(0,0,0,0.45), 0 0 25px rgba(244,123,32,0.15); border-color: rgba(244,123,32,0.30); }
    .kpi-card.accent::before  { background: linear-gradient(90deg, #7C3F98, #A6228C, #7C3F98); background-size: 200% 100%; animation: shimmer 3s linear infinite; }
    .kpi-card.accent::after   { background: linear-gradient(180deg, rgba(124,63,152,0.06), transparent); }
    .kpi-card.accent:hover    { box-shadow: 0 12px 40px rgba(0,0,0,0.45), 0 0 25px rgba(124,63,152,0.15); border-color: rgba(124,63,152,0.30); }
    .kpi-card.neutral::before { background: linear-gradient(90deg, #475569, #94A3B8); }

    .kpi-icon {
        position: absolute;
        top: 16px; right: 16px;
        width: 36px; height: 36px;
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 16px;
        background: rgba(0, 114, 206, 0.15); color: #60A5FA;
    }
    .kpi-card.good   .kpi-icon { background: rgba(0, 166, 81, 0.15); color: #4ADE80; }
    .kpi-card.warn   .kpi-icon { background: rgba(244, 123, 32, 0.15); color: #FB923C; }
    .kpi-card.accent .kpi-icon { background: rgba(124, 63, 152, 0.15); color: #C084FC; }

    .kpi-label {
        color: #64748B;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #F1F5F9;
        font-size: 30px;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.5px;
        font-feature-settings: "tnum";
    }
    .kpi-delta {
        font-size: 11px;
        color: #94A3B8;
        margin-top: 6px;
        font-weight: 500;
        line-height: 1.3;
    }

    /* Section header — dark */
    .section-header {
        margin: 32px 0 14px 0;
        animation: slideUp 0.4s ease;
    }
    .section-header-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 6px;
    }
    .section-icon {
        width: 38px; height: 38px;
        border-radius: 10px;
        background: linear-gradient(135deg, #0072CE, #00B5E2);
        display: flex; align-items: center; justify-content: center;
        font-size: 16px;
        color: white;
        box-shadow: 0 4px 16px rgba(0, 114, 206, 0.40), 0 0 20px rgba(0,114,206,0.15);
        flex-shrink: 0;
    }
    .section-header h2 {
        margin: 0;
        color: #E2E8F0;
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.4px;
    }
    .section-header p {
        margin: 0 0 10px 48px;
        color: #64748B;
        font-size: 13px;
        font-weight: 400;
        line-height: 1.5;
    }
    .section-rule {
        height: 2px;
        background: linear-gradient(90deg, #0072CE 0%, #00B5E2 18%, rgba(255,255,255,0.06) 18%, rgba(255,255,255,0.06) 100%);
        background-size: 200% 100%;
        animation: shimmer 6s linear infinite;
        border-radius: 2px;
        margin: 0 0 6px 0;
    }

    /* ── Sidebar — dark glass with animated border ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1526 0%, #0a1120 100%);
        border-right: 1px solid rgba(0,114,206,0.12);
        animation: breathe 6s ease-in-out infinite;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* ── Sidebar brand header — glowing ── */
    .sb-brand {
        position: relative;
        background:
            radial-gradient(ellipse at top right, rgba(0,181,226,0.25) 0%, transparent 60%),
            linear-gradient(135deg, #0a1628 0%, #0d2847 50%, #0f3d6e 100%);
        color: white;
        padding: 18px 18px 16px 18px;
        border-radius: 16px;
        margin: 0 0 18px 0;
        box-shadow:
            0 8px 30px rgba(0, 0, 0, 0.40),
            0 0 30px rgba(0, 114, 206, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.06);
        overflow: hidden;
        border: 1px solid rgba(0, 114, 206, 0.25);
    }
    .sb-brand::after {
        content: "";
        position: absolute;
        bottom: -40%; left: -20%;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(124,63,152,0.20) 0%, transparent 65%);
        pointer-events: none;
    }
    .sb-brand-row {
        display: flex;
        align-items: center;
        gap: 12px;
        position: relative;
        z-index: 1;
    }
    .sb-brand-logo {
        width: 40px; height: 40px;
        background: rgba(0,114,206,0.20);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(0,181,226,0.30);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 20px;
        flex-shrink: 0;
    }
    .sb-brand-text {
        line-height: 1.15;
    }
    .sb-brand-title {
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 0.6px;
        margin: 0;
    }
    .sb-brand-sub {
        font-size: 10px;
        opacity: 0.82;
        margin: 2px 0 0 0;
        font-weight: 500;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }

    /* ── Section label (above each widget) ── */
    .sb-label {
        display: flex;
        align-items: center;
        gap: 9px;
        margin: 18px 0 8px 0;
    }
    .sb-label-bar {
        width: 3px;
        height: 18px;
        border-radius: 999px;
        flex-shrink: 0;
    }
    .sb-label-bar.blue   { background: linear-gradient(180deg, #0072CE, #00B5E2); }
    .sb-label-bar.orange { background: linear-gradient(180deg, #F47B20, #FDB913); }
    .sb-label-bar.purple { background: linear-gradient(180deg, #7C3F98, #A6228C); }
    .sb-label-bar.green  { background: linear-gradient(180deg, #00A651, #4ADE80); }
    .sb-label-text {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
    }
    .sb-label-icon {
        font-size: 13px;
        margin-right: 2px;
    }

    /* ── Sidebar selectboxes — dark glass ── */
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        display: none;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
        border: 1.5px solid rgba(255,255,255,0.10) !important;
        min-height: 42px !important;
        transition: all .15s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.20);
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div:hover {
        border-color: rgba(0,114,206,0.35) !important;
        box-shadow: 0 2px 12px rgba(0, 114, 206, 0.15);
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div:focus-within {
        border-color: #0072CE !important;
        box-shadow: 0 0 0 3px rgba(0, 114, 206, 0.20) !important;
    }
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
        font-weight: 600;
        color: #E2E8F0;
        font-size: 13px;
    }

    /* ── Sidebar radio — segmented dark ── */
    section[data-testid="stSidebar"] .stRadio > div {
        flex-direction: row !important;
        background: rgba(255,255,255,0.04);
        padding: 4px;
        border-radius: 10px;
        gap: 2px !important;
        border: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stRadio > div > label {
        flex: 1 1 0;
        text-align: center;
        background: transparent !important;
        border: none !important;
        padding: 8px 4px !important;
        margin: 0 !important;
        border-radius: 8px !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        color: #64748B;
        transition: all .15s ease;
        cursor: pointer;
        min-width: 0;
    }
    section[data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        color: #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] {
        width: 100%;
    }
    section[data-testid="stSidebar"] .stRadio input[type="radio"] {
        display: none;
    }
    section[data-testid="stSidebar"] .stRadio > div > label > div:first-child {
        display: none;
    }
    section[data-testid="stSidebar"] .stRadio > div > label[data-baseweb="radio"] > div:first-child {
        display: none !important;
    }
    section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background: rgba(0,114,206,0.20) !important;
        color: #60A5FA !important;
        box-shadow: 0 2px 10px rgba(0, 114, 206, 0.20) !important;
    }

    /* ── Sidebar slider — premium ── */
    section[data-testid="stSidebar"] .stSlider {
        padding: 4px 4px 0 4px;
    }
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div > div > div {
        background: linear-gradient(90deg, #0072CE, #00B5E2) !important;
        height: 6px !important;
    }
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div > div {
        background: rgba(255,255,255,0.12) !important;
        height: 6px !important;
        border-radius: 999px !important;
    }
    section[data-testid="stSidebar"] .stSlider [role="slider"] {
        background: #E2E8F0 !important;
        border: 3px solid #0072CE !important;
        box-shadow: 0 3px 8px rgba(0, 114, 206, 0.30) !important;
        width: 18px !important;
        height: 18px !important;
    }
    /* Slider value tick text */
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] {
        color: #94A3B8 !important;
        font-size: 10px !important;
    }

    /* ── Live current-value chip below each widget ── */
    .sb-current {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(0, 114, 206, 0.08);
        border: 1px solid rgba(0, 114, 206, 0.15);
        padding: 8px 12px;
        border-radius: 10px;
        margin-top: 8px;
        margin-bottom: 4px;
    }
    .sb-current-label {
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        color: #64748B;
        font-weight: 700;
    }
    .sb-current-value {
        font-size: 12px;
        color: #60A5FA;
        font-weight: 700;
        text-align: right;
        max-width: 60%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    /* k=value display next to slider */
    .sb-k-display {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 11px;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 2px;
    }
    .sb-k-value {
        background: linear-gradient(135deg, #0072CE, #00B5E2);
        color: white;
        font-weight: 800;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 12px;
        letter-spacing: 0.3px;
        box-shadow: 0 2px 6px rgba(0, 114, 206, 0.25);
    }

    /* Section divider */
    .sb-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        margin: 16px 0 4px 0;
    }

    /* Active selection summary chip */
    .sb-chip {
        display: inline-block;
        background: rgba(0, 114, 206, 0.12);
        color: #60A5FA;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 999px;
        margin-top: 6px;
        border: 1px solid rgba(0, 114, 206, 0.20);
    }

    /* Footer block in sidebar */
    .sb-footer {
        margin-top: 18px;
        padding: 12px 14px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        text-align: center;
    }
    .sb-footer-row {
        display: flex;
        justify-content: space-between;
        font-size: 10px;
        color: #475569;
        margin-top: 4px;
    }
    .sb-footer-key {
        color: #60A5FA;
        font-weight: 700;
        font-size: 11px;
    }

    /* Tabs — dark glass */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255,255,255,0.04);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.20);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #64748B;
        font-weight: 600;
        font-size: 13px;
        padding: 8px 16px;
        transition: all .15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.06);
        color: #CBD5E1;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0d2847, #0072CE) !important;
        color: white !important;
        box-shadow: 0 4px 14px rgba(0, 114, 206, 0.35);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background: transparent !important;
    }

    /* Badges — refined */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 11px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.3px;
        border: 1px solid transparent;
    }
    .badge-blue   { background: rgba(0,114,206,0.15); color: #60A5FA; border-color: rgba(0,114,206,0.25); }
    .badge-green  { background: rgba(0,166,81,0.15); color: #4ADE80; border-color: rgba(0,166,81,0.25); }
    .badge-orange { background: rgba(244,123,32,0.15); color: #FB923C; border-color: rgba(244,123,32,0.25); }
    .badge-purple { background: rgba(124,63,152,0.15); color: #C084FC; border-color: rgba(124,63,152,0.25); }

    /* Streamlit alerts — soften */
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid transparent !important;
    }

    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
    }

    /* Plotly chart container */
    .stPlotlyChart {
        background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
        backdrop-filter: blur(10px);
        border-radius: 18px;
        padding: 12px 10px 6px 10px;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow:
            0 4px 16px rgba(0, 0, 0, 0.25),
            0 8px 32px rgba(0, 0, 0, 0.15);
        transition: box-shadow .25s ease, border-color .25s ease, transform .25s ease;
    }
    .stPlotlyChart:hover {
        transform: translateY(-2px);
        box-shadow:
            0 8px 24px rgba(0, 0, 0, 0.35),
            0 16px 48px rgba(0, 0, 0, 0.20),
            0 0 20px rgba(0, 114, 206, 0.08);
        border-color: rgba(0, 114, 206, 0.20);
    }

    /* ── Context strip ── */
    .context-strip {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 12px 18px;
        margin: -10px 0 22px 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.20);
        backdrop-filter: blur(10px);
    }
    .context-label {
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #475569;
        margin-right: 4px;
    }
    .context-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 12px;
        color: #CBD5E1;
        font-weight: 500;
    }
    .context-pill .pill-key {
        color: #64748B;
        font-weight: 500;
    }
    .context-pill .pill-val {
        color: #E2E8F0;
        font-weight: 700;
    }
    .context-pill.accent {
        background: linear-gradient(135deg, rgba(0,59,113,0.6), rgba(0,114,206,0.4));
        color: white;
        border-color: rgba(0,114,206,0.30);
        box-shadow: 0 0 12px rgba(0, 114, 206, 0.15);
    }
    .context-pill.accent .pill-key { color: rgba(255,255,255,0.78); }
    .context-pill.accent .pill-val { color: #93C5FD; }

    /* ── Status badges in KPI cards ── */
    .kpi-status {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        padding: 2px 8px;
        border-radius: 999px;
        margin-top: 8px;
    }
    .kpi-status.ok      { background: rgba(0,166,81,0.15); color: #4ADE80; }
    .kpi-status.fair    { background: rgba(244,123,32,0.15); color: #FB923C; }
    .kpi-status.poor    { background: rgba(239,68,68,0.15); color: #F87171; }
    .kpi-status.info    { background: rgba(0,114,206,0.15); color: #60A5FA; }
    .kpi-status::before {
        content: "";
        width: 6px; height: 6px;
        border-radius: 50%;
        background: currentColor;
        animation: pulseDot 2s ease-in-out infinite;
    }

    /* ── Subsection title ── */
    .subsection {
        color: #475569;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin: 22px 0 8px 0;
        padding-left: 10px;
        border-left: 3px solid #0072CE;
    }

    /* ── Streamlit native widget polish ── */
    .stRadio > div { gap: 8px; }
    .stSelectbox label, .stSlider label {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    /* Main area selectbox refinement */
    .main .stSelectbox > div > div {
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.10);
    }
    .main .stSelectbox > div > div:focus-within {
        border-color: #0072CE;
        box-shadow: 0 0 0 3px rgba(0, 114, 206, 0.20);
    }
    /* Slider in main area */
    .main .stSlider [data-baseweb="slider"] > div > div > div > div {
        background: linear-gradient(90deg, #0072CE, #00B5E2) !important;
    }
    .main .stSlider [role="slider"] {
        background: #E2E8F0 !important;
        border: 3px solid #0072CE !important;
        box-shadow: 0 2px 8px rgba(0, 114, 206, 0.30) !important;
    }
    /* Radio buttons in main area */
    .main .stRadio label {
        background: rgba(255,255,255,0.04);
        padding: 8px 14px;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.08);
        transition: all .12s ease;
    }
    .main .stRadio label:hover {
        background: rgba(255,255,255,0.06);
        border-color: rgba(0,114,206,0.25);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        color: #E2E8F0 !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0072CE, #00B5E2) !important;
    }

    /* Footer — dark premium */
    .footer {
        text-align: center;
        color: #64748B;
        font-size: 11px;
        margin-top: 40px;
        padding: 22px;
        border-top: 1px solid rgba(255,255,255,0.06);
        background: rgba(255,255,255,0.02);
        border-radius: 14px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.05);
    }
    .footer-brand {
        color: #E2E8F0;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 0.4px;
    }
    .footer-divider {
        display: inline-block;
        width: 4px; height: 4px;
        background: rgba(0,114,206,0.5);
        border-radius: 50%;
        margin: 0 10px;
        vertical-align: middle;
    }

    /* Hide only the Streamlit "Made with…" footer and main menu;
       KEEP the top header/toolbar so the sidebar collapse/expand button
       and the rerun controls remain accessible. */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.5rem;
    }
    /* Make sure the sidebar collapse button is always visible */
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        background: rgba(15,21,38,0.9);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.30);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────
# Plotly theme
# ─────────────────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_FONT = dict(family="Inter, -apple-system, Segoe UI, sans-serif",
                    size=12, color="#CBD5E1")


def _style_fig(fig, height=500, title=None, legend_top=False, subtitle=None):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=PLOTLY_FONT,
        height=height,
        margin=dict(l=12, r=12, t=58 if title else 24, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,17,32,0.6)",
        hoverlabel=dict(
            bgcolor="rgba(15,25,50,0.92)",
            bordercolor="rgba(0,114,206,0.3)",
            font=dict(family="Inter, sans-serif", size=12, color="#E2E8F0"),
        ),
        title=None,
    )
    if title:
        title_text = (
            f"<span style='font-size:14px;color:#E2E8F0;font-weight:700'>{title}</span>"
            + (f"<br><span style='font-size:11px;color:#64748B;font-weight:400'>{subtitle}</span>"
               if subtitle else "")
        )
        fig.update_layout(title=dict(text=title_text, x=0.015, y=0.96,
                                      xanchor="left", yanchor="top",
                                      pad=dict(t=8)))

    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.08)",
        tickfont=dict(color="#94A3B8", size=11),
        title=dict(font=dict(color="#94A3B8", size=12)),
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.08)",
        tickfont=dict(color="#94A3B8", size=11),
        title=dict(font=dict(color="#94A3B8", size=12)),
    )
    if legend_top:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                       xanchor="right", x=1,
                                       font=dict(size=11, color="#94A3B8"),
                                       bgcolor="rgba(0,0,0,0)"))
    return fig


# ─────────────────────────────────────────────────────────────────────────
# Status helpers — interpret metric quality
# ─────────────────────────────────────────────────────────────────────────
def silhouette_status(v):
    if v >= 0.5: return "ok", "Excellent"
    if v >= 0.25: return "fair", "Acceptable"
    return "poor", "Weak"

def davies_bouldin_status(v):
    if v <= 1.0: return "ok", "Tight"
    if v <= 1.6: return "fair", "Moderate"
    return "poor", "Loose"

def vmeasure_status(v):
    if v >= 0.20: return "ok", "Strong"
    if v >= 0.08: return "fair", "Moderate"
    return "poor", "Weak"

def ari_status(v):
    if v >= 0.20: return "ok", "Strong"
    if v >= 0.08: return "fair", "Moderate"
    return "poor", "Weak"


# ─────────────────────────────────────────────────────────────────────────
# Cached pipeline
# ─────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(f"{dataset_name}.csv")
    df["ATSEG_first"] = df["ATSEG_first"].replace("0", np.nan)
    return df


@st.cache_data
def preprocess(dataset_name: str):
    df = load_data(dataset_name)
    drop_cols = ["NUEVO_ID", "ATSEG_first"] + [c for c in df.columns if c.startswith("WEEK_ID")]
    cat_dummies = [c for c in df.columns
                   if c.startswith(("SPEC_", "STATE_", "STS_")) or c.startswith("(")]
    features = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in drop_cols and c not in cat_dummies]
    df_feat = df[features].copy()

    skew = df_feat.skew()
    for col in skew[skew.abs() > 2].index:
        cmin = df_feat[col].min()
        df_feat[col] = np.log1p(df_feat[col] - cmin) if cmin < 0 else np.log1p(df_feat[col])

    X = RobustScaler().fit_transform(df_feat)
    pca3 = PCA(n_components=3, random_state=SEED).fit(X)
    pca2 = PCA(n_components=2, random_state=SEED).fit(X)
    loadings = pd.DataFrame(pca2.components_.T, index=features, columns=["PC1", "PC2"])
    loadings["abs_total"] = loadings["PC1"].abs() + loadings["PC2"].abs()

    return {
        "df": df,
        "features": features,
        "X": X,
        "atseg": df["ATSEG_first"],
        "X_pca3": pca3.transform(X),
        "pca3": pca3,
        "loadings": loadings,
        "variance_explained": pca3.explained_variance_ratio_,
    }


@st.cache_data
def compute_tsne(dataset_name: str):
    ctx = preprocess(dataset_name)
    pca30 = PCA(n_components=min(30, ctx["X"].shape[1]),
                random_state=SEED).fit_transform(ctx["X"])
    return TSNE(n_components=3, random_state=SEED, perplexity=40,
                max_iter=1000, init="random").fit_transform(pca30)


@st.cache_data
def run_clustering(dataset_name: str, algo: str, k: int):
    ctx = preprocess(dataset_name)
    X = ctx["X"]

    if algo == "K-Means":
        labels = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit_predict(X)
    elif algo == "Hierarchical (Ward)":
        pca30 = PCA(n_components=min(30, X.shape[1]),
                    random_state=SEED).fit_transform(X)
        Z = linkage(pca30, method="ward", metric="euclidean")
        labels = fcluster(Z, t=k, criterion="maxclust") - 1
    elif algo == "GMM":
        labels = GaussianMixture(n_components=k, random_state=SEED,
                                  covariance_type="full",
                                  n_init=5).fit_predict(X)
    else:
        raise ValueError(algo)

    metrics = {
        "Silhouette": silhouette_score(X, labels, sample_size=5000, random_state=SEED),
        "Davies-Bouldin": davies_bouldin_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels),
    }
    has = ctx["atseg"].notna()
    if has.sum() > 0:
        atseg_codes = ctx["atseg"][has].map({"SEG_A": 0, "SEG_B": 1, "SEG_C": 2}).values
        labs_sub = labels[has]
        metrics["V-Measure"] = v_measure_score(atseg_codes, labs_sub)
        metrics["ARI"] = adjusted_rand_score(atseg_codes, labs_sub)
        metrics["NMI"] = normalized_mutual_info_score(atseg_codes, labs_sub)
    return labels, metrics


def get_projection(ctx, dataset_name, kind, custom_feats=None):
    if kind == "PCA":
        return ctx["X_pca3"], ["PC1", "PC2", "PC3"]
    if kind == "t-SNE":
        return compute_tsne(dataset_name), ["t-SNE 1", "t-SNE 2", "t-SNE 3"]
    if kind == "Top Features":
        top3 = ctx["loadings"].nlargest(3, "abs_total").index.tolist()
        idx = [ctx["features"].index(f) for f in top3]
        return ctx["X"][:, idx], top3
    if kind == "Custom":
        feats = custom_feats or ctx["features"][:3]
        # Validate — only keep features actually in the dataset
        feats = [f for f in feats if f in ctx["features"]]
        while len(feats) < 3:
            for cand in ctx["features"]:
                if cand not in feats:
                    feats.append(cand)
                    break
        feats = feats[:3]
        idx = [ctx["features"].index(f) for f in feats]
        return ctx["X"][:, idx], feats
    raise ValueError(kind)


def kpi_card(col, label, value, helper="", style="", icon="",
             status=None, status_label=""):
    """Render a KPI card. Status: 'ok' | 'fair' | 'poor' | 'info'."""
    cls = f"kpi-card {style}".strip()
    icon_html = f'<div class="kpi-icon">{icon}</div>' if icon else ""
    status_html = (f'<div class="kpi-status {status}">{status_label}</div>'
                   if status else "")
    col.markdown(
        f"""
        <div class="{cls}">
            {icon_html}
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta">{helper}</div>
            {status_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def context_strip(items, accent_index=None):
    """Render a horizontal context strip showing active configuration.

    items: list of (key, value) tuples.
    accent_index: index of pill to render in accent style.
    """
    pills = ['<span class="context-label">VIEWING</span>']
    for i, (key, val) in enumerate(items):
        cls = "context-pill accent" if i == accent_index else "context-pill"
        pills.append(
            f'<span class="{cls}">'
            f'<span class="pill-key">{key}</span>'
            f'<span class="pill-val">{val}</span></span>'
        )
    st.markdown(f'<div class="context-strip">{"".join(pills)}</div>',
                 unsafe_allow_html=True)


def section(title, subtitle="", icon="📊"):
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-header-row">
                <div class="section-icon">{icon}</div>
                <h2>{title}</h2>
            </div>
            {f'<p>{subtitle}</p>' if subtitle else ''}
            <div class="section-rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────
def sb_label(icon: str, title: str, color: str = "blue"):
    """Sidebar section label — colored bar + icon + title."""
    st.sidebar.markdown(
        f"""
        <div class="sb-label">
            <div class="sb-label-bar {color}"></div>
            <span class="sb-label-icon">{icon}</span>
            <span class="sb-label-text">{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sb_current(label: str, value: str):
    """Live current-value chip shown below a widget."""
    st.sidebar.markdown(
        f"""
        <div class="sb-current">
            <span class="sb-current-label">{label}</span>
            <span class="sb-current-value">{value}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sb_divider():
    st.sidebar.markdown('<div class="sb-divider"></div>',
                         unsafe_allow_html=True)


# ── Brand header ──
st.sidebar.markdown(
    """
    <div class="sb-brand">
        <div class="sb-brand-row">
            <div class="sb-brand-logo">💊</div>
            <div class="sb-brand-text">
                <div class="sb-brand-title">PFIZER</div>
                <div class="sb-brand-sub">Clustering Intelligence</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── View Mode (segmented control) ──
sb_label("👁", "View Mode", color="purple")
mode = st.sidebar.radio(
    "Mode", ["Single dataset", "Compare both datasets"],
    label_visibility="collapsed",
    horizontal=True,
)

sb_divider()

# ── Algorithm ──
sb_label("🧠", "Algorithm", color="blue")
algo = st.sidebar.selectbox(
    "Algorithm", ["K-Means", "Hierarchical (Ward)", "GMM"],
    label_visibility="collapsed",
    format_func=lambda x: {
        "K-Means": "🎯  K-Means",
        "Hierarchical (Ward)": "🌲  Hierarchical (Ward)",
        "GMM": "🔮  Gaussian Mixture (GMM)",
    }[x],
)

# ── k slider with live value badge ──
st.sidebar.markdown(
    f"""
    <div class="sb-k-display">
        <span>NUMBER OF CLUSTERS</span>
        <span class="sb-k-value" id="k-val">k = {st.session_state.get('k_slider', 3)}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
k = st.sidebar.slider(
    "k", 3, 8, 3, key="k_slider",
    label_visibility="collapsed",
    help="Business floor is k=3 (matches ATSEG)",
)

sb_divider()

# ── Dataset & Projection (only in single-dataset mode) ──
if mode == "Single dataset":
    sb_label("📂", "Dataset", color="orange")
    dataset = st.sidebar.selectbox(
        "Dataset", ["doctors_means", "doctors_sums"],
        label_visibility="collapsed",
        format_func=lambda x: ("📊  Doctor Means (averages)"
                                if x == "doctors_means"
                                else "📈  Doctor Sums (totals)"),
    )

    sb_divider()

    sb_label("🌐", "3D Projection", color="green")
    projection = st.sidebar.selectbox(
        "Projection",
        ["PCA", "t-SNE", "Top Features", "Custom"],
        label_visibility="collapsed",
        format_func=lambda x: {
            "PCA": "📐  PCA (3 components)",
            "t-SNE": "🌌  t-SNE (3D)",
            "Top Features": "⭐  Top PCA Loadings",
            "Custom": "🛠  Custom Features",
        }[x],
    )

    custom_feats = None
    if projection == "Custom":
        _ctx_for_picker = preprocess(dataset)
        feat_list = _ctx_for_picker["features"]

        defaults = ["ORAL_TRX", "N_CLMOTHERS", "TOTAL_TRX"]
        defaults = [f for f in defaults if f in feat_list]
        while len(defaults) < 3:
            for cand in feat_list:
                if cand not in defaults:
                    defaults.append(cand)
                    break

        st.sidebar.markdown(
            "<div style='font-size:10px;color:#94A3B8;margin:10px 0 6px 4px;"
            "font-weight:700;letter-spacing:.6px;text-transform:uppercase;'>"
            "Pick 3 axes</div>",
            unsafe_allow_html=True,
        )

        col_axis_x, col_axis_y, col_axis_z = st.sidebar.columns(3)
        with col_axis_x:
            st.markdown("<div style='font-size:10px;font-weight:700;color:#0072CE;"
                         "text-align:center;margin-bottom:2px;'>X</div>",
                         unsafe_allow_html=True)
        with col_axis_y:
            st.markdown("<div style='font-size:10px;font-weight:700;color:#F47B20;"
                         "text-align:center;margin-bottom:2px;'>Y</div>",
                         unsafe_allow_html=True)
        with col_axis_z:
            st.markdown("<div style='font-size:10px;font-weight:700;color:#00A651;"
                         "text-align:center;margin-bottom:2px;'>Z</div>",
                         unsafe_allow_html=True)

        feat_x = st.sidebar.selectbox(
            "X axis", feat_list,
            index=feat_list.index(defaults[0]),
            key="custom_x", label_visibility="collapsed",
        )
        feat_y = st.sidebar.selectbox(
            "Y axis", feat_list,
            index=feat_list.index(defaults[1]),
            key="custom_y", label_visibility="collapsed",
        )
        feat_z = st.sidebar.selectbox(
            "Z axis", feat_list,
            index=feat_list.index(defaults[2]),
            key="custom_z", label_visibility="collapsed",
        )
        custom_feats = [feat_x, feat_y, feat_z]

        if len(set(custom_feats)) < 3:
            st.sidebar.warning("⚠️ Pick 3 different features for best results.")

# ── Live "Now showing" summary ──
sb_divider()
st.sidebar.markdown(
    "<div style='font-size:10px;color:#94A3B8;margin:14px 0 4px 4px;"
    "font-weight:700;letter-spacing:.7px;text-transform:uppercase;'>"
    "Active Configuration</div>",
    unsafe_allow_html=True,
)
sb_current("Mode", mode.replace("dataset", "").strip().capitalize())
sb_current("Algorithm", algo)
sb_current("k", str(k))
if mode == "Single dataset":
    ds_short = "Means" if dataset == "doctors_means" else "Sums"
    sb_current("Dataset", f"Doctor {ds_short}")
    sb_current("Projection", projection)

# ── Footer info ──
st.sidebar.markdown(
    f"""
    <div class="sb-footer">
        <div style="font-size:10px;color:#94A3B8;letter-spacing:.7px;
                    font-weight:700;text-transform:uppercase;">Reproducibility</div>
        <div class="sb-footer-row">
            <span>Random seed</span>
            <span class="sb-footer-key">{SEED}</span>
        </div>
        <div class="sb-footer-row">
            <span>k floor</span>
            <span class="sb-footer-key">3 (ATSEG)</span>
        </div>
        <div class="sb-footer-row">
            <span>Data</span>
            <span class="sb-footer-key">Anonymized</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-content">
            <div class="hero-eyebrow">PFIZER · COMMERCIAL ANALYTICS</div>
            <h1>Prescriber Clustering Intelligence</h1>
            <p>Doctor segmentation analytics for the VELSIPITY franchise.
            Discover distinct prescriber archetypes, validate against existing ATSEG segmentation,
            and surface actionable insights across multiple clustering algorithms.</p>
            <div class="hero-divider"></div>
            <span class="hero-pill">🧠 Algorithm <b>{algo}</b></span>
            <span class="hero-pill">🎯 Clusters <b>k={k}</b></span>
            <span class="hero-pill">👁 Mode <b>{mode}</b></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODE 1: Single dataset
# ============================================================
if mode == "Single dataset":
    with st.spinner("Running clustering pipeline..."):
        ctx = preprocess(dataset)
        labels, metrics = run_clustering(dataset, algo, k)

    # ── Context strip ──
    dataset_label = ("Doctor Means" if dataset == "doctors_means"
                      else "Doctor Sums")
    context_strip([
        ("Dataset", dataset_label),
        ("Algorithm", algo),
        ("k", str(k)),
        ("Projection", projection),
    ], accent_index=0)

    # ── Top-level KPI strip with status indicators ──
    sil_status, sil_label = silhouette_status(metrics["Silhouette"])
    db_status, db_label = davies_bouldin_status(metrics["Davies-Bouldin"])
    vm_status, vm_label = vmeasure_status(metrics.get("V-Measure", 0))
    ari_status_v, ari_label = ari_status(metrics.get("ARI", 0))

    cols = st.columns(5)
    kpi_card(cols[0], "Doctors Analyzed", f"{len(ctx['df']):,}",
             helper=f"{len(ctx['features'])} features",
             style="neutral", icon="👥",
             status="info", status_label="Loaded")
    kpi_card(cols[1], "Silhouette",
             f"{metrics['Silhouette']:.4f}",
             helper="Cluster separation quality",
             style="good", icon="🎯",
             status=sil_status, status_label=sil_label)
    kpi_card(cols[2], "Davies-Bouldin",
             f"{metrics['Davies-Bouldin']:.4f}",
             helper="Cluster compactness",
             style="warn", icon="📐",
             status=db_status, status_label=db_label)
    kpi_card(cols[3], "V-Measure vs ATSEG",
             f"{metrics.get('V-Measure', 0):.4f}",
             helper="Agreement with ATSEG",
             style="accent", icon="🤝",
             status=vm_status, status_label=vm_label)
    kpi_card(cols[4], "ARI vs ATSEG",
             f"{metrics.get('ARI', 0):.4f}",
             helper="Adjusted Rand Index",
             style="accent", icon="📊",
             status=ari_status_v, status_label=ari_label)

    # ── Tabs ──
    tabs = st.tabs([
        "🔍  Data Exploration",
        "🌐  Cluster Map",
        "📊  Cluster Profile",
        "⭐  Feature Drivers",
        "🔥  Cluster Heatmap",
        "📑  ATSEG Validation",
        "📋  Data Explorer",
    ])

    # ── Data Exploration ──
    with tabs[0]:
        section("ATSEG Segmentation Overview",
                "Distribution of doctors across the existing ATSEG segmentation",
                icon="🔍")

        atseg_full = ctx["atseg"].fillna("No ATSEG")
        vc = atseg_full.value_counts()
        total_doctors = len(atseg_full)
        with_atseg = atseg_full.ne("No ATSEG").sum()
        no_atseg = atseg_full.eq("No ATSEG").sum()
        coverage = with_atseg / total_doctors * 100
        n_segments = (vc.index != "No ATSEG").sum()

        # KPI strip
        cols = st.columns(4)
        cov_status = ("ok" if coverage >= 50 else "fair" if coverage >= 25 else "poor")
        cov_label = ("High" if coverage >= 50 else "Partial" if coverage >= 25 else "Low")
        kpi_card(cols[0], "Total Doctors", f"{total_doctors:,}",
                 helper="In dataset",
                 style="neutral", icon="👥",
                 status="info", status_label="Loaded")
        kpi_card(cols[1], "With ATSEG", f"{with_atseg:,}",
                 helper=f"{coverage:.1f}% coverage",
                 style="good", icon="✓",
                 status=cov_status, status_label=f"{cov_label} coverage")
        kpi_card(cols[2], "Without ATSEG", f"{no_atseg:,}",
                 helper=f"{100-coverage:.1f}% unsegmented",
                 style="warn", icon="⚠",
                 status="fair" if no_atseg > with_atseg else "info",
                 status_label="Unlabeled")
        kpi_card(cols[3], "ATSEG Segments", f"{n_segments}",
                 helper="Distinct categories",
                 style="accent", icon="🏷",
                 status="info", status_label="Reference")

        st.markdown("&nbsp;")

        # Bar + donut side by side
        df_atseg = vc.reset_index()
        df_atseg.columns = ["Segment", "Doctors"]
        df_atseg["Pct"] = df_atseg["Doctors"] / total_doctors * 100
        df_atseg = df_atseg.sort_values("Segment")

        col_a, col_b = st.columns([3, 2])
        with col_a:
            fig = px.bar(
                df_atseg, x="Segment", y="Doctors",
                color="Segment",
                color_discrete_map=ATSEG_COLORS,
                text=df_atseg.apply(
                    lambda r: f"{int(r['Doctors']):,}<br>({r['Pct']:.1f}%)",
                    axis=1),
            )
            fig.update_traces(textposition="outside",
                              marker_line_width=0, opacity=0.92)
            _style_fig(fig, height=460, title="Doctors per ATSEG Segment")
            fig.update_layout(showlegend=False,
                              xaxis_title="",
                              yaxis_title="Number of Doctors")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            colors_pie = [ATSEG_COLORS.get(s, "#94A3B8")
                          for s in df_atseg["Segment"]]
            fig = go.Figure(data=[go.Pie(
                labels=df_atseg["Segment"],
                values=df_atseg["Doctors"],
                hole=0.58,
                marker=dict(colors=colors_pie,
                            line=dict(color="rgba(11,17,32,0.8)", width=2)),
                textinfo="percent",
                textfont=dict(size=14, color="white"),
                hovertemplate="<b>%{label}</b><br>%{value:,} doctors"
                              "<br>%{percent}<extra></extra>",
            )])
            fig.update_layout(
                annotations=[dict(
                    text=f"<b>{total_doctors:,}</b>"
                         f"<br><span style='font-size:11px;"
                         f"color:#64748B'>doctors</span>",
                    x=0.5, y=0.5, font_size=22,
                    font=dict(color="#E2E8F0"), showarrow=False,
                )],
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.12,
                            xanchor="center", x=0.5,
                            font=dict(size=11)),
            )
            _style_fig(fig, height=460, title="ATSEG Share")
            st.plotly_chart(fig, use_container_width=True)

        # ── Feature distribution by ATSEG ──
        section("Feature Distribution by ATSEG",
                "Compare how key features distribute across each ATSEG segment",
                icon="📈")

        top_feats = ctx["loadings"].nlargest(8, "abs_total").index.tolist()
        feat_to_show = st.selectbox(
            "Feature to inspect", top_feats, key="explore_feat",
        )

        df_dist = ctx["df"][[feat_to_show, "ATSEG_first"]].copy()
        df_dist["ATSEG_first"] = df_dist["ATSEG_first"].fillna("No ATSEG")

        col_h, col_b2 = st.columns(2)
        with col_h:
            fig = px.histogram(
                df_dist, x=feat_to_show, color="ATSEG_first",
                category_orders={"ATSEG_first":
                                 ["SEG_A", "SEG_B", "SEG_C", "No ATSEG"]},
                color_discrete_map=ATSEG_COLORS,
                barmode="overlay",
                opacity=0.55,
                nbins=50,
            )
            _style_fig(fig, height=420,
                       title=f"Histogram — {feat_to_show}")
            fig.update_layout(legend_title_text="ATSEG",
                              yaxis_title="Doctor Count",
                              xaxis_title=feat_to_show)
            st.plotly_chart(fig, use_container_width=True)

        with col_b2:
            fig = px.box(
                df_dist, x="ATSEG_first", y=feat_to_show,
                color="ATSEG_first",
                category_orders={"ATSEG_first":
                                 ["SEG_A", "SEG_B", "SEG_C", "No ATSEG"]},
                color_discrete_map=ATSEG_COLORS,
                points=False,
            )
            _style_fig(fig, height=420,
                       title=f"Box Plot — {feat_to_show} by ATSEG")
            fig.update_layout(showlegend=False,
                              xaxis_title="",
                              yaxis_title=feat_to_show)
            st.plotly_chart(fig, use_container_width=True)

        # ── Feature averages by ATSEG ──
        section("Average Feature Values by ATSEG",
                "Mean (raw, unscaled) of top features for each segment",
                icon="🧮")

        top10 = ctx["loadings"].nlargest(10, "abs_total").index.tolist()
        df_avg = ctx["df"][top10 + ["ATSEG_first"]].copy()
        df_avg["ATSEG_first"] = df_avg["ATSEG_first"].fillna("No ATSEG")
        avg_table = df_avg.groupby("ATSEG_first")[top10].mean().round(3)
        # Reorder
        order = [s for s in ["SEG_A", "SEG_B", "SEG_C", "No ATSEG"]
                 if s in avg_table.index]
        avg_table = avg_table.loc[order]

        fig = px.imshow(
            avg_table, text_auto=".2f", aspect="auto",
            color_continuous_scale="Blues",
        )
        fig.update_xaxes(tickangle=35)
        _style_fig(fig, height=380,
                   title="Mean Feature Values by ATSEG Segment")
        st.plotly_chart(fig, use_container_width=True)

    # ── Cluster Map (3D) ──
    with tabs[1]:
        section("Interactive 3D Cluster Map",
                "Compare cluster assignments against the existing ATSEG segmentation",
                icon="🌐")

        proj_data, ax_labels = get_projection(ctx, dataset, projection,
                                               custom_feats=custom_feats)
        df_plot = pd.DataFrame(proj_data, columns=ax_labels)
        df_plot["Cluster"] = [f"C{l}" for l in labels]
        df_plot["ATSEG"] = ctx["atseg"].fillna("No ATSEG").values

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**By Cluster** &nbsp; <span class='badge badge-blue'>{algo}</span>",
                        unsafe_allow_html=True)
            fig = px.scatter_3d(
                df_plot, x=ax_labels[0], y=ax_labels[1], z=ax_labels[2],
                color="Cluster", color_discrete_sequence=CLUSTER_PALETTE,
                opacity=0.55,
            )
            fig.update_traces(marker=dict(size=2.5, line=dict(width=0)))
            _style_fig(fig, height=560)
            fig.update_layout(scene=dict(
                xaxis=dict(backgroundcolor="rgba(11,17,32,0.4)", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(backgroundcolor="rgba(11,17,32,0.4)", gridcolor="rgba(255,255,255,0.06)"),
                zaxis=dict(backgroundcolor="rgba(11,17,32,0.4)", gridcolor="rgba(255,255,255,0.06)"),
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("**By ATSEG** &nbsp; <span class='badge badge-orange'>Reference</span>",
                        unsafe_allow_html=True)
            atseg_order = ["SEG_A", "SEG_B", "SEG_C", "No ATSEG"]
            fig = px.scatter_3d(
                df_plot, x=ax_labels[0], y=ax_labels[1], z=ax_labels[2],
                color="ATSEG",
                category_orders={"ATSEG": atseg_order},
                color_discrete_map=ATSEG_COLORS,
                opacity=0.55,
            )
            fig.update_traces(marker=dict(size=2.5, line=dict(width=0)))
            _style_fig(fig, height=560)
            fig.update_layout(scene=dict(
                xaxis=dict(backgroundcolor="rgba(11,17,32,0.4)", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(backgroundcolor="rgba(11,17,32,0.4)", gridcolor="rgba(255,255,255,0.06)"),
                zaxis=dict(backgroundcolor="rgba(11,17,32,0.4)", gridcolor="rgba(255,255,255,0.06)"),
            ))
            st.plotly_chart(fig, use_container_width=True)

        if projection == "PCA":
            ve = ctx["variance_explained"]
            st.info(
                f"**PCA variance captured** — PC1: {ve[0]*100:.1f}%  ·  "
                f"PC2: {ve[1]*100:.1f}%  ·  PC3: {ve[2]*100:.1f}%  ·  "
                f"**Total: {ve.sum()*100:.1f}%**",
                icon="📐",
            )

    # ── Cluster Profile ──
    with tabs[2]:
        section("Cluster Size Distribution",
                "How prescribers are distributed across the discovered segments",
                icon="📊")

        unique, counts = np.unique(labels, return_counts=True)
        total = counts.sum()
        df_sz = pd.DataFrame({
            "Cluster": [f"C{u}" for u in unique],
            "Doctors": counts,
            "Pct": counts / total * 100,
        })

        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig = px.bar(
                df_sz, x="Cluster", y="Doctors",
                color="Cluster", color_discrete_sequence=CLUSTER_PALETTE,
                text=df_sz.apply(lambda r: f"{int(r['Doctors']):,}<br>({r['Pct']:.1f}%)",
                                  axis=1),
            )
            fig.update_traces(textposition="outside",
                              marker_line_width=0, opacity=0.92)
            _style_fig(fig, height=480, title="Doctors per Cluster")
            fig.update_layout(showlegend=False, yaxis_title="Number of Doctors",
                              xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig = go.Figure(data=[go.Pie(
                labels=df_sz["Cluster"], values=df_sz["Doctors"],
                hole=0.55,
                marker=dict(colors=CLUSTER_PALETTE[:len(df_sz)],
                            line=dict(color="rgba(11,17,32,0.8)", width=2)),
                textinfo="percent", textfont=dict(size=14, color="white"),
            )])
            fig.update_layout(
                annotations=[dict(text=f"{total:,}<br><span style='font-size:12px;color:#64748B'>doctors</span>",
                                   x=0.5, y=0.5, font_size=22,
                                   font=dict(color="#E2E8F0"), showarrow=False)],
                showlegend=False,
            )
            _style_fig(fig, height=480, title="Share of Population")
            st.plotly_chart(fig, use_container_width=True)

        # Balance flag
        max_pct = df_sz["Pct"].max()
        min_pct = df_sz["Pct"].min()
        if max_pct > 70:
            st.warning(f"⚠️ **Imbalanced clusters detected** — largest cluster holds "
                       f"{max_pct:.1f}% of doctors. Consider trying a different algorithm.")
        else:
            st.success(f"✅ **Balanced distribution** — largest cluster: {max_pct:.1f}%, "
                       f"smallest: {min_pct:.1f}%")

    # ── Feature Drivers ──
    with tabs[3]:
        section("Top Features Driving the Segmentation",
                "Variables with highest contribution to PC1 + PC2",
                icon="⭐")

        n_top = st.slider("Number of features to display", 5, 30, 15)
        top = ctx["loadings"].nlargest(n_top, "abs_total").reset_index()
        top.columns = ["Feature", "PC1", "PC2", "abs_total"]

        fig = px.bar(
            top, x="abs_total", y="Feature", orientation="h",
            text=top["abs_total"].round(3),
            color="abs_total",
            color_continuous_scale=[[0, "#BFDBFE"], [1, "#003B71"]],
        )
        fig.update_traces(textposition="outside",
                          marker_line_width=0,
                          texttemplate="%{text:.3f}")
        fig.update_layout(yaxis={"categoryorder": "total ascending"},
                          coloraxis_showscale=False,
                          xaxis_title="|PC1| + |PC2| contribution",
                          yaxis_title="")
        _style_fig(fig, height=max(450, 32 * n_top),
                   title=f"Top {n_top} Features — {dataset}")
        st.plotly_chart(fig, use_container_width=True)

    # ── Heatmap ──
    with tabs[4]:
        section("Cluster Profiles by Key Features",
                "Mean scaled values per cluster — green = high, red = low",
                icon="🔥")

        target = ["ORAL_TRX", "N_CLMOTHERS", "TOTAL_TRX", "IL23_TRX",
                  "BRAND1_TRX", "BRAND2_TRX", "TOTAL_NRX",
                  "ENGAGEMENT_SCORE", "RTE", "SAMPLES", "DETAILS"]
        feats = [f for f in target if f in ctx["features"]]
        idx = [ctx["features"].index(f) for f in feats]
        unique_clusters = sorted(set(labels))
        means = []
        for c in unique_clusters:
            mask = labels == c
            means.append(ctx["X"][mask][:, idx].mean(axis=0))
        df_h = pd.DataFrame(means, columns=feats,
                             index=[f"C{c}" for c in unique_clusters])

        fig = px.imshow(
            df_h, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
        )
        fig.update_xaxes(tickangle=35)
        _style_fig(fig, height=400 + 35 * len(unique_clusters),
                   title=f"Cluster Profiles — {algo} k={k}")
        st.plotly_chart(fig, use_container_width=True)

    # ── ATSEG Validation ──
    with tabs[5]:
        section("ATSEG Validation",
                "How well do the discovered clusters align with current segmentation?",
                icon="🤝")

        has = ctx["atseg"].notna()
        if has.sum() == 0:
            st.warning("No ATSEG labels available for this dataset.")
        else:
            df_cm = pd.DataFrame({
                "ATSEG": ctx["atseg"][has].values,
                "Cluster": [f"C{l}" for l in labels[has]],
            })
            ct = pd.crosstab(df_cm["ATSEG"], df_cm["Cluster"])
            ct_norm = ct.div(ct.sum(axis=1), axis=0)
            purity = ct.values.max(axis=0).sum() / ct.values.sum()

            kcols = st.columns(4)
            ari_v = metrics.get("ARI", 0)
            vm_v = metrics.get("V-Measure", 0)
            kpi_card(kcols[0], "Cluster Purity", f"{purity*100:.1f}%",
                     helper="Largest ATSEG class per cluster",
                     style="good", icon="✓",
                     status="info", status_label="Reference")
            kpi_card(kcols[1], "ARI", f"{ari_v:.4f}",
                     helper="Adjusted Rand Index", icon="📊",
                     status=ari_status(ari_v)[0],
                     status_label=ari_status(ari_v)[1])
            kpi_card(kcols[2], "NMI", f"{metrics.get('NMI', 0):.4f}",
                     helper="Normalized Mutual Information", icon="ℹ",
                     status="info", status_label="Info")
            kpi_card(kcols[3], "V-Measure", f"{vm_v:.4f}",
                     helper="Homogeneity × Completeness",
                     style="accent", icon="🤝",
                     status=vmeasure_status(vm_v)[0],
                     status_label=vmeasure_status(vm_v)[1])

            st.markdown("&nbsp;")
            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.imshow(
                    ct_norm, text_auto=".1%", aspect="auto",
                    color_continuous_scale="Blues",
                )
                _style_fig(fig, height=420,
                           title="Row-normalized (ATSEG → Cluster %)")
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                fig = px.imshow(
                    ct, text_auto=True, aspect="auto",
                    color_continuous_scale="Blues",
                )
                _style_fig(fig, height=420, title="Raw Doctor Counts")
                st.plotly_chart(fig, use_container_width=True)

    # ── Data Explorer ──
    with tabs[6]:
        section("Dataset Overview",
                f"Source: <code>{dataset}.csv</code> — anonymized prescriber data",
                icon="📋")

        c1, c2, c3 = st.columns(3)
        kpi_card(c1, "Rows (Doctors)", f"{len(ctx['df']):,}",
                 style="neutral", icon="👥")
        kpi_card(c2, "Total Columns", f"{ctx['df'].shape[1]}",
                 style="neutral", icon="📋")
        kpi_card(c3, "Features Used", f"{len(ctx['features'])}",
                 style="good", icon="✓")

        st.markdown("&nbsp;")
        st.markdown("**Sample (first 50 rows)**")
        st.dataframe(ctx["df"].head(50), use_container_width=True, height=350)

        with st.expander("📚  View full feature list"):
            chunk = 4
            feat_table = pd.DataFrame({
                f"Set {i+1}": (ctx["features"][i*chunk:(i+1)*chunk]
                              + [""] * chunk)[:chunk]
                for i in range((len(ctx["features"]) + chunk - 1) // chunk)
            })
            st.dataframe(feat_table.T, use_container_width=True)


# ============================================================
# MODE 2: Compare both datasets
# ============================================================
else:
    context_strip([
        ("Mode", "Compare both datasets"),
        ("Algorithm", algo),
        ("k", str(k)),
    ], accent_index=0)

    progress = st.progress(0, text="Computing both datasets...")
    summary_rows = []
    cluster_results = {}

    for i, dn in enumerate(["doctors_means", "doctors_sums"]):
        progress.progress((i + 0.5) / 2, text=f"Processing {dn}...")
        ctx = preprocess(dn)
        labels, metrics = run_clustering(dn, algo, k)
        cluster_results[dn] = {"ctx": ctx, "labels": labels, "metrics": metrics}
        summary_rows.append({"Dataset": dn, "Algorithm": algo, "k": k, **metrics})
    progress.empty()

    df_sum = pd.DataFrame(summary_rows).round(4)

    # ── Summary KPIs (winner of each metric) ──
    section("Executive Summary",
            "Best dataset for each clustering quality metric",
            icon="🏆")

    best_sil = df_sum.loc[df_sum["Silhouette"].idxmax()]
    best_db  = df_sum.loc[df_sum["Davies-Bouldin"].idxmin()]
    best_vm  = df_sum.loc[df_sum["V-Measure"].idxmax()] if "V-Measure" in df_sum.columns else None

    kc = st.columns(3)
    kpi_card(kc[0], "Best by Silhouette",
             best_sil["Dataset"].replace("doctors_", "").upper(),
             helper=f"Score: {best_sil['Silhouette']:.4f}",
             style="good", icon="🎯")
    kpi_card(kc[1], "Best by Davies-Bouldin",
             best_db["Dataset"].replace("doctors_", "").upper(),
             helper=f"Score: {best_db['Davies-Bouldin']:.4f}",
             style="warn", icon="📐")
    if best_vm is not None:
        kpi_card(kc[2], "Best by V-Measure (vs ATSEG)",
                 best_vm["Dataset"].replace("doctors_", "").upper(),
                 helper=f"Score: {best_vm['V-Measure']:.4f}",
                 style="accent", icon="🤝")

    # ── Detailed metrics ──
    section("Side-by-Side Metrics",
            "Quantitative comparison across both datasets",
            icon="📋")
    st.dataframe(df_sum, use_container_width=True, height=120)

    # ── Bar charts ──
    section("Metrics Comparison",
            "Visual comparison of clustering quality indicators",
            icon="📊")
    metric_cols = ["Silhouette", "Davies-Bouldin", "V-Measure"]
    fig = make_subplots(rows=1, cols=3,
                         subplot_titles=[
                             "Silhouette  ↑ better",
                             "Davies-Bouldin  ↓ better",
                             "V-Measure vs ATSEG  ↑ better",
                         ])
    ds_colors = [PFIZER_BLUE, PFIZER_ORANGE]
    ds_labels = {"doctors_means": "MEANS", "doctors_sums": "SUMS"}
    for ci, metric in enumerate(metric_cols, start=1):
        for di, ds in enumerate(["doctors_means", "doctors_sums"]):
            val = df_sum[df_sum["Dataset"] == ds][metric].values[0]
            fig.add_trace(go.Bar(
                x=[ds_labels[ds]], y=[val],
                text=[f"{val:.4f}"], textposition="outside",
                marker_color=ds_colors[di],
                marker_line_width=0,
                name=ds_labels[ds], showlegend=(ci == 1),
                width=0.5,
            ), row=1, col=ci)

    _style_fig(fig, height=460)
    fig.update_layout(barmode="group",
                       legend=dict(orientation="h", yanchor="bottom", y=1.08,
                                    xanchor="center", x=0.5))
    for sa in fig["layout"]["annotations"]:
        sa["font"] = dict(size=13, color="#E2E8F0")
    st.plotly_chart(fig, use_container_width=True)

    # ── 3D scatter side-by-side ──
    section("3D Cluster Maps — Both Datasets",
            "Side-by-side view of cluster geometry",
            icon="🌐")
    proj_kind = st.radio("Projection", ["PCA", "t-SNE"], horizontal=True)

    col_a, col_b = st.columns(2)
    for ax, dn in zip([col_a, col_b], ["doctors_means", "doctors_sums"]):
        with ax:
            r = cluster_results[dn]
            ctx_d = r["ctx"]
            labels_d = r["labels"]
            proj_data, ax_labels = get_projection(ctx_d, dn, proj_kind)
            df_plot = pd.DataFrame(proj_data, columns=ax_labels)
            df_plot["Cluster"] = [f"C{l}" for l in labels_d]
            badge = ("badge-blue" if dn == "doctors_means" else "badge-orange")
            st.markdown(
                f"<span class='badge {badge}'>"
                f"{ds_labels[dn]}</span> &nbsp; {algo} k={k}",
                unsafe_allow_html=True)
            fig = px.scatter_3d(
                df_plot, x=ax_labels[0], y=ax_labels[1], z=ax_labels[2],
                color="Cluster", color_discrete_sequence=CLUSTER_PALETTE,
                opacity=0.55,
            )
            fig.update_traces(marker=dict(size=2.5, line=dict(width=0)))
            _style_fig(fig, height=520)
            fig.update_layout(scene=dict(
                xaxis=dict(backgroundcolor="rgba(11,17,32,0.4)", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(backgroundcolor="rgba(11,17,32,0.4)", gridcolor="rgba(255,255,255,0.06)"),
                zaxis=dict(backgroundcolor="rgba(11,17,32,0.4)", gridcolor="rgba(255,255,255,0.06)"),
            ))
            st.plotly_chart(fig, use_container_width=True)

    # ── Cluster sizes side by side ──
    section("Cluster Size Distribution — Both Datasets",
            "Population balance across discovered segments",
            icon="📊")
    col_a, col_b = st.columns(2)
    for ax, dn in zip([col_a, col_b], ["doctors_means", "doctors_sums"]):
        with ax:
            labels_d = cluster_results[dn]["labels"]
            unique, counts = np.unique(labels_d, return_counts=True)
            df_sz = pd.DataFrame({"Cluster": [f"C{u}" for u in unique],
                                   "Doctors": counts})
            fig = px.bar(
                df_sz, x="Cluster", y="Doctors",
                text="Doctors", color="Cluster",
                color_discrete_sequence=CLUSTER_PALETTE,
            )
            fig.update_traces(textposition="outside",
                              marker_line_width=0, opacity=0.92)
            _style_fig(fig, height=380, title=ds_labels[dn])
            fig.update_layout(showlegend=False,
                               xaxis_title="", yaxis_title="Doctors")
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        <span class="footer-brand">PFIZER · COMMERCIAL ANALYTICS</span>
        <span class="footer-divider"></span>
        Prescriber Clustering Intelligence
        <span class="footer-divider"></span>
        Built with Streamlit
        <span class="footer-divider"></span>
        Random seed locked at 42 for reproducibility
    </div>
    """,
    unsafe_allow_html=True,
)
