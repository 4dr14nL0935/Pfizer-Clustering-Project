"""
Pfizer HCP Segmentation — Executive Dashboard
=============================================
Two XGBoost models predicting ATSEG segments (SEG_A / SEG_B / SEG_C):

  1) ARGMAX BASELINE         — multi:softprob with balanced sample_weight
  2) BUSINESS-CALIBRATED ORDINAL — two-stage XGBoost ordinal with
                                   custom decision thresholds, optimised
                                   for minimising the catastrophic
                                   SEG_C → SEG_A error.

Run with:
    streamlit run dashboard.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64

import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────
SEED = 42
VALID_LABELS = ("SEG_A", "SEG_B", "SEG_C")

# Brand palette
PFIZER_BLUE       = "#0070BF"
PFIZER_DARK_BLUE  = "#003B71"
PFIZER_LIGHT_BLUE = "#00AFF0"
PFIZER_ORANGE     = "#F47B20"
PFIZER_GREEN      = "#00A651"
PFIZER_PURPLE     = "#7C3F98"
PFIZER_GRAY       = "#535554"

SEG_COLORS = {
    "SEG_A": "#0070BF",
    "SEG_B": "#F47B20",
    "SEG_C": "#7C3F98",
    "Unlabeled": "#CFD8DC",
    "Unlab.": "#CFD8DC",
    "": "#CFD8DC",
}

# Hyperparameters — Ordinal model (capstone final)
ORDINAL_PARAMS = dict(
    n_estimators=800, max_depth=5, learning_rate=0.04,
    subsample=0.85, colsample_bytree=0.7, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
    eval_metric="logloss", random_state=SEED, n_jobs=-1,
)
THR_A, THR_C = 0.70, 0.30
SEG_C_SAMPLE_WEIGHT = 2.0

# Hyperparameters — Argmax baseline
ARGMAX_PARAMS = dict(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    objective="multi:softprob", num_class=3,
    random_state=SEED, n_jobs=-1, eval_metric="mlogloss",
)

# Key features for explorer (suffix _sum — match doctors_aggregated.csv columns).
# Each row is (column, friendly_name, description, is_actionable).
# ACTIONABLE = Pfizer can directly influence (engagement levers).
# SIGNAL     = doctor's prescribing behaviour (outcomes / indicators).
KEY_FEATURES = [
    ("DETAILS_sum",   "Sales detail visits",        "Pfizer rep visits to this HCP",            True),
    ("SAMPLES_sum",   "Drug samples",               "Samples delivered to HCP",                  True),
    ("UC_TRX_sum",    "UC prescriptions (TRx)",     "Total Rx for ulcerative colitis",           False),
    ("UC_NRX_sum",    "UC new prescriptions (NRx)", "Patients newly starting UC therapy",        False),
    ("ORAL_TRX_sum",  "Oral therapy Rx",            "Oral UC treatments — Velsipity's category", False),
    ("IL23_TRX_sum",  "IL-23 inhibitor Rx",         "Competitor biologics for UC",               False),
    ("TOTAL_TRX_sum", "Total Rx volume",            "Overall prescribing activity",              False),
]

N_SHAP_TOP = 8                  # Top SHAP features shown per doctor
N_CONVERSION_CANDIDATES = 300   # Top predicted-B doctors by P_C for conversion tab

# Hard-coded dataset
DATASET_NAME = "doctors_aggregated"

# Data preparation pipeline (shown in the hover tooltip)
DATA_PREP_STEPS = [
    ("Aggregate",     "Collapsed weekly rows into 1 row per HCP (sum + mean)."),
    ("Engineer",      "Added TOTAL_TRX, ENGAGEMENT_SCORE, ratios, R4 sums."),
    ("Encode",        "One-hot SPEC_*, STATE_*, age buckets."),
    ("Label & clean", "Joined ATSEG_first; imputed nulls. Fully numeric."),
]

# ─────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────
logo_pfizer = Image.open("logo2.png")
st.set_page_config(
    page_title="Pfizer HCP Segmentation",
    page_icon=logo_pfizer,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────
# Custom CSS — executive look
# ─────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    /* Increase base font for Streamlit text, but NOT Plotly internals */
    .stMarkdown, .stText, [data-testid="stMetric"],
    p, span, li, td, th, label, .stSelectbox, .stRadio {
        font-size: 16px !important;
    }
    .stPlotlyChart * { font-size: unset !important; }

    /* ── BASE THEME (PFIZER PROFESSIONAL LIGHT) ── */
    .stApp {
        background: linear-gradient(160deg, #E8EDF4 0%, #DCE4F0 40%, #CEDAF0 100%) !important;
        background-attachment: fixed !important;
        color: #1E293B;
    }
    .main { background: transparent; }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 4rem;
        max-width: 1440px;
    }

    @keyframes fadeInSlide {
        from { opacity: 0; transform: translateY(15px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── COMPACT & EVEN STREAMLIT TABS ── */
    [data-baseweb="tab-list"] {
        gap: 4px !important;
        width: 100%;
    }
    button[data-baseweb="tab"] {
        flex: 1 !important;
        padding-left: 4px !important;
        padding-right: 4px !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    button[data-baseweb="tab"] p {
        font-size: 14px !important;
        text-align: center !important;
        width: 100% !important;
    }

    /* ── HERO SECTION ── */
    .hero {
        position: relative;
        background: #FFFFFF;
        border: 1px solid #D6DEE8;
        box-shadow: 0 2px 8px rgba(0,40,100,0.07), 0 8px 32px rgba(0,112,191,0.06);
        padding: 36px 44px 32px 44px;
        border-radius: 16px;
        margin-bottom: 28px;
        overflow: hidden;
        animation: fadeInSlide 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .hero::before {
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 6px;
        background: linear-gradient(90deg, #003B71, #0070BF, #00D4FF);
        box-shadow: 0 2px 8px rgba(0,112,191,0.3);
    }
    .hero-content { position: relative; z-index: 1; }
    .hero-eyebrow {
        font-size: 16px !important; text-transform: uppercase; letter-spacing: 2.5px;
        color: #0070BF; font-weight: 700; margin-bottom: 12px;
        display: flex; align-items: center; gap: 10px;
    }
    .hero-title {
        margin: 0 0 10px 0; 
        font-family: 'Outfit', -apple-system, sans-serif !important;
        font-size: 44px !important; 
        font-weight: 700 !important;
        letter-spacing: -0.5px !important; 
        line-height: 1.2 !important;
        color: #0F172A !important;
        display: block;
    }
    .hero p {
        margin: 18px 0 0 0; color: #475569; font-size: 20px !important;
        font-weight: 400; max-width: 800px; line-height: 1.6;
    }
    .hero-divider {
        height: 1px; background: linear-gradient(90deg, rgba(0,112,191,0.2), transparent);
        margin: 24px 0 20px 0;
        border: none;
    }
    .hero-pill {
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        padding: 10px 18px; border-radius: 999px;
        font-size: 16px !important; margin-right: 14px; margin-top: 10px; color: #334155;
        font-weight: 500; border: 1px solid rgba(0, 112, 191, 0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .hero-pill:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0, 112, 191, 0.1); }
    .hero-pill b { font-weight: 700; color: #0070BF; font-size: 16px !important; }

    /* ── KPI CARDS (PROFESSIONAL) ── */
    .kpi-card {
        position: relative;
        background: #FFFFFF;
        border: 1px solid #D6DEE8;
        box-shadow: 0 2px 8px rgba(0,40,100,0.07);
        padding: 22px 24px 18px 24px;
        border-radius: 12px;
        height: 280px !important;
        margin-bottom: 24px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.25s ease;
        overflow: hidden;
        animation: fadeInSlide 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .kpi-card::before {
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 5px;
        background: linear-gradient(90deg, #0058A3, #0088E0, #00D4FF);
        box-shadow: 0 2px 6px rgba(0,112,191,0.25);
    }
    .kpi-card.good::before   { background: linear-gradient(90deg, #047857, #10B981, #34D399); box-shadow: 0 2px 6px rgba(16,185,129,0.3); }
    .kpi-card.warn::before   { background: linear-gradient(90deg, #B45309, #F59E0B, #FBBF24); box-shadow: 0 2px 6px rgba(245,158,11,0.3); }
    .kpi-card.accent::before { background: linear-gradient(90deg, #6D28D9, #8B5CF6, #A78BFA); box-shadow: 0 2px 6px rgba(139,92,246,0.3); }
    .kpi-card.danger::before { background: linear-gradient(90deg, #9F1239, #E11D48, #FB7185); box-shadow: 0 2px 6px rgba(225,29,72,0.3); }
    .kpi-card:nth-child(1) { animation-delay: 0.05s; }
    .kpi-card:nth-child(2) { animation-delay: 0.1s; }
    .kpi-card:nth-child(3) { animation-delay: 0.15s; }
    .kpi-card:nth-child(4) { animation-delay: 0.2s; }
    .kpi-card:nth-child(5) { animation-delay: 0.25s; }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,112,191,0.1);
        border-color: rgba(0,112,191,0.25);
    }
    .kpi-card.compact {
        height: 170px !important;
        padding: 16px 20px 14px 20px;
    }
    .kpi-card.compact .kpi-value { font-size: 32px; }
    .kpi-card.compact .kpi-icon { width: 40px; height: 40px; font-size: 18px; top: 16px; right: 16px; }
    .kpi-card.compact .kpi-label { font-size: 12px; margin-bottom: 6px; }
    .kpi-card.compact .kpi-delta { font-size: 13px; margin-top: 4px; }
    .kpi-card.compact .kpi-status { margin-top: 8px; padding: 4px 12px; font-size: 11px; }
    .kpi-icon {
        position: absolute; top: 22px; right: 22px;
        width: 50px; height: 50px; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 24px;
        background: #F0F7FF; color: #0070BF;
        border: 1px solid #DBEAFE;
    }
    .kpi-card.good   .kpi-icon { background: #ECFDF5; color: #059669; border-color: #A7F3D0; }
    .kpi-card.warn   .kpi-icon { background: #FFFBEB; color: #D97706; border-color: #FDE68A; }
    .kpi-card.accent .kpi-icon { background: #F5F3FF; color: #7C3AED; border-color: #DDD6FE; }
    .kpi-card.danger .kpi-icon { background: #FFF1F2; color: #BE123C; border-color: #FECDD3; }
    .kpi-label {
        color: #64748B; font-size: 14px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 12px;
        padding-right: 64px;
        line-height: 1.4;
    }
    .kpi-value {
        color: #0F172A; font-size: 44px; font-weight: 700;
        line-height: 1.1; letter-spacing: -0.8px;
        font-feature-settings: "tnum";
    }
    .kpi-delta {
        font-size: 15px; color: #94A3B8; margin-top: 10px; font-weight: 500;
        line-height: 1.4;
    }
    .kpi-status {
        display: inline-flex; align-items: center; gap: 6px;
        font-size: 13px; font-weight: 700; letter-spacing: 0.5px;
        text-transform: uppercase; padding: 6px 14px;
        border-radius: 999px; margin-top: 16px;
    }
    .kpi-status.ok   { color: #059669; background: #ECFDF5; border: 1px solid #A7F3D0; }
    .kpi-status.fair { color: #D97706; background: #FFFBEB; border: 1px solid #FDE68A; }
    .kpi-status.poor { color: #BE123C; background: #FFF1F2; border: 1px solid #FECDD3; }
    .kpi-status.info { color: #0070BF; background: #F0F7FF; border: 1px solid #DBEAFE; }
    .kpi-status::before {
        content: ""; width: 6px; height: 6px;
        border-radius: 50%; background: currentColor;
    }

    /* ── SECTION HEADER ── (no-emoji, professional look) */
    .section-header {
        position: relative;
        margin: 40px 0 20px 0;
        padding: 0 0 0 18px;
        animation: fadeInSlide 0.5s ease both;
    }
    /* Brand vertical accent bar replaces the old icon square */
    .section-header::before {
        content: "";
        position: absolute;
        left: 0; top: 4px; bottom: 4px;
        width: 4px;
        border-radius: 4px;
        background: linear-gradient(180deg, #003B71 0%, #0070BF 60%, #00B5E2 100%);
        box-shadow: 0 2px 8px rgba(0, 114, 206, 0.25);
    }
    .section-header-row {
        display: flex; align-items: baseline; gap: 14px; margin-bottom: 4px;
        flex-wrap: wrap;
    }
    .section-eyebrow {
        font-size: 10.5px;
        font-weight: 800;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        color: #0070BF;
        margin-right: 4px;
    }
    .section-header h2 {
        margin: 0; color: #0F172A; font-size: 22px;
        font-weight: 800; letter-spacing: -0.3px;
        line-height: 1.2;
    }
    .section-header p {
        margin: 6px 0 0 0; color: #64748B;
        font-size: 13.5px; font-weight: 400;
        line-height: 1.55;
        max-width: 820px;
    }
    /* Decorative underline (replaces the legacy .section-rule div if used) */
    .section-rule {
        height: 1px;
        background: linear-gradient(90deg,
            rgba(0, 114, 206, 0.35) 0%,
            rgba(226, 232, 240, 0.6) 30%,
            transparent 100%);
        margin-top: 12px;
    }

    /* ── SIDEBAR ── */
    section[data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #D6DEE8;
        box-shadow: 2px 0 16px rgba(0,40,100,0.06);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem; padding-left: 1.5rem; padding-right: 1.5rem;
    }
    .sb-brand {
        position: relative;
        background: linear-gradient(135deg, #003B71, #0070BF);
        color: white; padding: 20px;
        border-radius: 12px; margin: 0 0 24px 0;
        box-shadow: 0 4px 16px rgba(0,59,113,0.25);
        overflow: hidden;
    }
    .sb-brand::after {
        content: ""; position: absolute; right: -20px; top: -20px;
        width: 100px; height: 100px; border-radius: 50%;
        background: radial-gradient(circle, rgba(0,175,240,0.3) 0%, transparent 70%);
    }
    .sb-brand-row {
        display: flex; align-items: center; gap: 14px;
        position: relative; z-index: 1;
    }
    .sb-brand-logo {
        width: 44px; height: 44px;
        background: rgba(255,255,255,0.15);
        border-radius: 10px; border: 1px solid rgba(255,255,255,0.2);
        display: flex; align-items: center; justify-content: center;
        font-size: 22px; flex-shrink: 0;
        color: #FFFFFF;
    }
    .sb-brand-text { line-height: 1.2; }
    .sb-brand-title {
        font-size: 16px; font-weight: 700; letter-spacing: 0.5px; margin: 0;
        color: #FFFFFF;
    }
    .sb-brand-sub {
        font-size: 11px; color: rgba(255,255,255,0.7); margin: 3px 0 0 0;
        font-weight: 600; letter-spacing: 1px; text-transform: uppercase;
    }
    /* Sidebar widgets */
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label { display: none; }
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: #F8FAFC !important;
        border-radius: 10px !important;
        border: 1px solid #E2E8F0 !important;
        min-height: 44px !important;
        color: #0F172A !important;
        transition: all 0.2s ease;
    }
    section[data-testid="stSidebar"] .stSelectbox > div > div:hover {
        border-color: #0070BF !important;
        box-shadow: 0 0 0 3px rgba(0,112,191,0.08) !important;
    }

    /* PLOTLY CHARTS */
    .stPlotlyChart {
        background: #FFFFFF;
        border-radius: 12px; padding: 16px;
        border: 1px solid #D6DEE8;
        box-shadow: 0 2px 8px rgba(0,40,100,0.07);
        transition: all 0.25s ease;
        overflow: visible !important;
    }
    .stPlotlyChart > div, .stPlotlyChart iframe,
    .stPlotlyChart .js-plotly-plot, .stPlotlyChart .plot-container {
        overflow: visible !important;
    }
    .stPlotlyChart:hover {
        box-shadow: 0 8px 24px rgba(0,112,191,0.08);
    }
    /* DOCTOR CARD */
    .doctor-card {
        background: #FFFFFF;
        border-radius: 14px; padding: 28px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 20px;
        animation: fadeInSlide 0.5s ease both;
    }
    .doctor-id {
        font-family: 'Outfit', monospace; font-weight: 700;
        font-size: 32px; color: #0F172A; letter-spacing: -0.5px;
        background: linear-gradient(90deg, #0070BF, #00AFF0);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    /* ── SEGMENT PILLS ── */
    .seg-pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 6px 16px; border-radius: 999px;
        font-size: 12px; font-weight: 600; letter-spacing: 0.5px;
        border: 1px solid transparent;
    }
    .seg-pill.SEG_A { background: rgba(0,112,191,0.1); color: #0070BF; border-color: rgba(0,112,191,0.2); }
    .seg-pill.SEG_B { background: rgba(244,123,32,0.1); color: #F47B20; border-color: rgba(244,123,32,0.2); }
    .seg-pill.SEG_C { background: rgba(190,18,60,0.1); color: #BE123C; border-color: rgba(190,18,60,0.2); }
    .seg-pill.Unlabeled { background: rgba(100,116,139,0.1); color: #64748B; border-color: rgba(100,116,139,0.2); }

    /* ── VERDICT CARDS ── */
    .verdict-card {
        background: #FFFFFF;
        border-radius: 12px; padding: 22px 24px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: all 0.25s ease; height: 100%;
    }
    .verdict-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,112,191,0.08);
    }
    .verdict-card-header {
        display: flex; align-items: center; justify-content: space-between;
        gap: 8px; margin-bottom: 14px;
        padding-bottom: 14px; border-bottom: 1px dashed rgba(0,0,0,0.08);
    }
    .verdict-model-name {
        font-size: 12px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.8px; color: #0070BF;
    }
    .verdict-prediction {
        font-size: 34px; font-weight: 700; color: #0F172A; margin: 6px 0;
    }
    .verdict-prediction.SEG_A { color: #0070BF; }
    .verdict-prediction.SEG_B { color: #F47B20; }
    .verdict-prediction.SEG_C { color: #BE123C; }
    .verdict-confidence { font-size: 13px; color: #64748B; font-weight: 400; }
    .verdict-confidence b { color: #0F172A; font-weight: 600; }
    .verdict-correct {
        background: rgba(16,185,129,0.1); color: #059669;
        padding: 5px 12px; border-radius: 999px;
        font-size: 10px; font-weight: 700; letter-spacing: 0.5px;
        border: 1px solid rgba(16,185,129,0.2);
    }
    .verdict-wrong {
        background: rgba(225,29,72,0.1); color: #BE123C;
        padding: 5px 12px; border-radius: 999px;
        font-size: 10px; font-weight: 700; letter-spacing: 0.5px;
        border: 1px solid rgba(225,29,72,0.2);
    }
    .verdict-na {
        background: rgba(100,116,139,0.1); color: #475569;
        padding: 5px 12px; border-radius: 999px;
        font-size: 10px; font-weight: 700; letter-spacing: 0.5px;
        border: 1px solid rgba(100,116,139,0.2);
    }

    /* ── PROBABILITY BAR ── */
    .prob-bar-row { margin: 14px 0; }
    .prob-bar-header {
        display: flex; justify-content: space-between;
        align-items: center; margin-bottom: 8px;
    }
    .prob-bar-label {
        font-size: 12px; color: #475569; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.8px;
    }
    .prob-bar-track {
        display: flex; height: 32px; border-radius: 8px; overflow: hidden;
        background: rgba(0,0,0,0.04); border: 1px solid rgba(0,0,0,0.06);
    }
    .prob-seg-fill {
        display: flex; align-items: center; justify-content: center;
        color: white; font-size: 11px; font-weight: 700;
        transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .prob-seg-fill.A { background: #0070BF; }
    .prob-seg-fill.B { background: #F47B20; }
    .prob-seg-fill.C { background: #7C3F98; }

    /* ── CONTEXT STRIP ── */
    .context-strip {
        display: flex; align-items: center; flex-wrap: wrap; gap: 10px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px; padding: 14px 22px;
        margin: -10px 0 24px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .context-label {
        font-size: 10px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.8px; color: #64748B; margin-right: 6px;
    }
    .context-pill {
        display: inline-flex; align-items: center; gap: 8px;
        background: #F8FAFC; border: 1px solid #E2E8F0;
        padding: 6px 14px; border-radius: 999px;
        font-size: 14px; color: #334155; font-weight: 500;
    }
    .context-pill .pill-key { color: #64748B; font-weight: 400; }
    .context-pill .pill-val { color: #0F172A; font-weight: 600; }
    .context-pill.accent {
        background: rgba(0,112,191,0.08); border-color: rgba(0,112,191,0.15);
    }
    .context-pill.accent .pill-val { color: #0070BF; }

    /* ── SIDEBAR LABELS ── (no-emoji, professional look) */
    .sb-label {
        display: flex; align-items: center; gap: 10px;
        margin: 26px 0 10px 0;
        padding-bottom: 6px;
        border-bottom: 1px dashed #E2E8F0;
    }
    /* Thicker accent bar with a soft glow — replaces the missing icon */
    .sb-label-bar {
        width: 4px; height: 20px; border-radius: 999px;
        flex-shrink: 0;
    }
    .sb-label-bar.blue   {
        background: linear-gradient(180deg, #003B71, #0070BF);
        box-shadow: 0 2px 6px rgba(0, 112, 191, 0.30);
    }
    .sb-label-bar.orange {
        background: linear-gradient(180deg, #C2410C, #F47B20);
        box-shadow: 0 2px 6px rgba(244, 123, 32, 0.30);
    }
    .sb-label-bar.purple {
        background: linear-gradient(180deg, #5B21B6, #7C3AED);
        box-shadow: 0 2px 6px rgba(124, 58, 237, 0.30);
    }
    .sb-label-bar.green  {
        background: linear-gradient(180deg, #047857, #059669);
        box-shadow: 0 2px 6px rgba(5, 150, 105, 0.30);
    }
    .sb-label-text {
        font-size: 11px; font-weight: 800; text-transform: uppercase;
        letter-spacing: 1.2px; color: #0F172A;
        flex: 1;
    }
    /* Icon slot — only shown when sb_label() is called with a non-empty icon */
    .sb-label-icon { font-size: 14px; color: #64748B; }

    .sb-current {
        display: flex; align-items: center; justify-content: space-between;
        background: #F8FAFC; border: 1px solid #E2E8F0;
        padding: 10px 14px; border-radius: 8px;
        margin-top: 8px; margin-bottom: 6px;
    }
    .sb-current-label {
        font-size: 9px; text-transform: uppercase; letter-spacing: 0.8px;
        color: #64748B; font-weight: 700;
    }
    .sb-current-value {
        font-size: 12px; color: #0F172A; font-weight: 600;
        text-align: right; max-width: 60%;
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .sb-divider {
        height: 1px; background: #E2E8F0; margin: 20px 0 8px 0;
    }
    .sb-footer {
        margin-top: 24px; padding: 14px;
        background: #F8FAFC; border: 1px solid #E2E8F0;
        border-radius: 10px;
    }
    .sb-footer-row {
        display: flex; justify-content: space-between;
        font-size: 10px; color: #64748B; margin-top: 6px;
    }
    .sb-footer-key { color: #0F172A; font-weight: 600; font-size: 11px; }

    /* Sidebar radio toggle */
    section[data-testid="stSidebar"] .stRadio > div {
        flex-direction: row !important;
        background: #F1F5F9; padding: 4px;
        border-radius: 8px; gap: 4px !important;
        border: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] .stRadio > div > label {
        flex: 1 1 0; text-align: center;
        background: transparent !important;
        border: none !important; padding: 8px 4px !important;
        margin: 0 !important; border-radius: 6px !important;
        font-size: 12px !important; font-weight: 600 !important;
        color: #64748B; transition: all .2s ease; cursor: pointer;
    }
    section[data-testid="stSidebar"] .stRadio input[type="radio"] { display: none; }
    section[data-testid="stSidebar"] .stRadio > div > label > div:first-child { display: none; }
    section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background: #FFFFFF !important; color: #0070BF !important;
        border: 1px solid #DBEAFE !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    }

    /* ── SECTION RULE ── */
    .section-rule {
        height: 1px; background: rgba(0,0,0,0.06); margin-top: 12px;
    }

    /* ── TABS ── (no-emoji, professional pills) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04),
                    inset 0 1px 0 rgba(255, 255, 255, 0.6);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 9px;
        color: #475569;
        font-weight: 600;
        font-size: 13.5px;
        letter-spacing: 0.2px;
        padding: 11px 22px;
        transition: all .18s ease;
        position: relative;
        border: 1px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #EFF6FF;
        color: #003B71;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F0F7FF 100%) !important;
        color: #003B71 !important;
        font-weight: 800 !important;
        border: 1px solid #BFDBFE !important;
        box-shadow: 0 2px 6px rgba(0, 114, 206, 0.12),
                    0 1px 2px rgba(15, 23, 42, 0.06);
    }
    /* Subtle bottom accent on the active tab */
    .stTabs [aria-selected="true"]::after {
        content: "";
        position: absolute;
        left: 50%; bottom: -2px;
        transform: translateX(-50%);
        width: 28px; height: 3px;
        background: linear-gradient(90deg, #003B71, #0070BF);
        border-radius: 2px;
    }
    .stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }

    /* ── DATAFRAME ── */
    [data-testid="stDataFrame"] {
        border-radius: 10px; overflow: hidden;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        background: #FFFFFF;
    }

    /* ── EMPTY STATE ── */
    .empty-state {
        text-align: center;
        background: #FFFFFF; border: 2px dashed #CBD5E1;
        border-radius: 12px; padding: 60px 30px; margin: 24px 0;
    }
    .empty-state-icon { font-size: 54px; opacity: 0.6; margin-bottom: 16px; }
    .empty-state-title {
        color: #0F172A; font-size: 20px; font-weight: 600; margin-bottom: 8px;
    }
    .empty-state-text {
        color: #475569; font-size: 14px;
        max-width: 480px; margin: 0 auto; line-height: 1.6;
    }

    /* ── INFO PANEL ── */
    .info-panel {
        display: flex; align-items: flex-start; gap: 14px;
        background: #F0F7FF;
        border: 1px solid #DBEAFE;
        border-left: 4px solid #0070BF;
        border-radius: 10px; padding: 18px 22px; margin: 16px 0;
    }
    .info-panel-icon { font-size: 20px; flex-shrink: 0; }
    .info-panel-content {
        font-size: 14px; color: #1E293B; line-height: 1.6;
    }
    .info-panel-content b { color: #0070BF; font-weight: 600; }

    /* ── FOOTER ── */
    .footer {
        text-align: center; color: #64748B; font-size: 13px;
        margin-top: 40px; padding: 24px;
        border-top: 1px solid #E2E8F0;
        background: #FFFFFF; border-radius: 10px;
    }
    .footer-brand {
        color: #0F172A; font-weight: 600; font-size: 12px;
        letter-spacing: 0.6px; text-transform: uppercase;
    }
    .footer-divider {
        display: inline-block; width: 4px; height: 4px;
        background: #CBD5E1; border-radius: 50%;
        margin: 0 12px; vertical-align: middle;
    }

    /* ── ALERTS / PROGRESS ── */
    .stAlert { border-radius: 12px !important; }
    .stProgress > div > div > div > div { background: #0070BF !important; }

    /* ────────────────────────────────────────────────────────────
       Dashboard animations — minimal & safe.

       Streamlit re-renders every tab's content on every widget
       interaction, even tabs that aren't visible.  Animating tab
       panels or every Plotly chart re-fires those animations on
       each rerun and causes layout shifts that scroll the page
       (the user reported this when toggling the Probability Map
       color radio).  So we keep ONLY:

         · KPI card fade (top of each tab — short distance, OK)
         · Tab pill scale-pulse (fixed-size element, can't shift layout)
         · Info-panel fade
         · Hover micro-interactions

       Charts, dataframes, tab panels, and expanders get NO entrance
       animation — they render in place, no jump.
    ──────────────────────────────────────────────────────────── */

    @keyframes opacityIn { from { opacity: 0; } to { opacity: 1; } }

    /* KPI cards fade in — staggered cascade.  Safe because cards
       have fixed dimensions and the animation is pure opacity. */
    section[data-testid="stMain"] .kpi-card {
        animation: opacityIn 0.35s ease-out both;
    }
    section[data-testid="stMain"] [data-testid="stHorizontalBlock"]
        > [data-testid="column"]:nth-child(1) .kpi-card { animation-delay: 0ms; }
    section[data-testid="stMain"] [data-testid="stHorizontalBlock"]
        > [data-testid="column"]:nth-child(2) .kpi-card { animation-delay: 50ms; }
    section[data-testid="stMain"] [data-testid="stHorizontalBlock"]
        > [data-testid="column"]:nth-child(3) .kpi-card { animation-delay: 100ms; }
    section[data-testid="stMain"] [data-testid="stHorizontalBlock"]
        > [data-testid="column"]:nth-child(4) .kpi-card { animation-delay: 150ms; }
    section[data-testid="stMain"] [data-testid="stHorizontalBlock"]
        > [data-testid="column"]:nth-child(5) .kpi-card { animation-delay: 200ms; }

    /* Tab pill micro-pulse — pure scale on fixed-size element */
    .stTabs [aria-selected="true"] {
        animation: tabSelect 0.28s cubic-bezier(.2, .8, .25, 1);
    }
    @keyframes tabSelect {
        0%   { transform: scale(0.96); }
        50%  { transform: scale(1.03); }
        100% { transform: scale(1); }
    }

    /* Info panels fade in */
    .info-panel { animation: opacityIn 0.35s ease both; }

    /* Smooth chart container hover (the lift was previously instant) */
    .stPlotlyChart {
        transition: transform 0.25s ease, box-shadow 0.25s ease !important;
    }

    /* Buttons get a subtle press effect */
    section[data-testid="stMain"] .stButton button {
        transition: transform 0.12s ease, box-shadow 0.18s ease,
                    background 0.18s ease !important;
    }
    section[data-testid="stMain"] .stButton button:active {
        transform: translateY(1px) scale(0.985);
    }

    /* Inputs get a smooth focus glow */
    section[data-testid="stMain"] input,
    section[data-testid="stMain"] textarea,
    section[data-testid="stMain"] [data-baseweb="select"] > div {
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }

    /* Sliders get a smooth thumb transition */
    section[data-testid="stMain"] [data-baseweb="slider"] [role="slider"] {
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }

    /* Reduce-motion respect: disable entrance animations for users
       who have prefers-reduced-motion set */
    @media (prefers-reduced-motion: reduce) {
        section[data-testid="stMain"] .kpi-card,
        .info-panel,
        .stTabs [aria-selected="true"] {
            animation: none !important;
        }
    }

    /* ── Dataset card with hover-tooltip "+" icon ── */
    .dataset-card {
        position: relative;
        background:
            linear-gradient(135deg, rgba(0,114,206,0.04) 0%, rgba(0,181,226,0.04) 100%),
            white;
        border: 1px solid #DBEAFE;
        border-left: 4px solid #0072CE;
        border-radius: 14px;
        padding: 18px 22px;
        margin: -10px 0 22px 0;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        display: flex;
        align-items: center;
        gap: 18px;
        flex-wrap: wrap;
    }
    .ds-icon {
        width: 48px; height: 48px;
        background: linear-gradient(135deg, #003B71, #0072CE);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px; color: white;
        box-shadow: 0 4px 10px rgba(0, 114, 206, 0.25);
        flex-shrink: 0;
    }
    .ds-text { flex: 1; min-width: 240px; }
    .ds-label {
        font-size: 10px; text-transform: uppercase;
        letter-spacing: 0.7px; color: #64748B;
        font-weight: 700; margin-bottom: 2px;
    }
    .ds-name {
        font-family: 'Inter', monospace;
        font-size: 18px; font-weight: 800; color: #0B1B33;
        letter-spacing: -0.3px;
        display: inline-flex; align-items: center; gap: 8px;
    }
    .ds-meta {
        font-size: 12px; color: #475569;
        margin-top: 4px; font-weight: 500;
    }

    /* The "+" info icon with hover tooltip */
    .ds-info {
        position: relative;
        display: inline-flex;
        align-items: center; justify-content: center;
        width: 22px; height: 22px;
        border-radius: 50%;
        background: white;
        border: 1.5px solid #0072CE;
        color: #0072CE;
        font-size: 14px; font-weight: 700;
        cursor: help;
        transition: all .15s ease;
        line-height: 1;
        margin-left: 4px;
    }
    .ds-info:hover {
        background: #0072CE; color: white;
        transform: scale(1.1);
        box-shadow: 0 4px 10px rgba(0, 114, 206, 0.30);
    }
    .ds-tooltip {
        visibility: hidden; opacity: 0;
        position: absolute;
        top: calc(100% + 14px);
        left: 50%; transform: translateX(-50%);
        z-index: 1000;
        width: 380px;
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow:
            0 10px 30px rgba(15, 23, 42, 0.15),
            0 2px 6px rgba(15, 23, 42, 0.08);
        text-align: left;
        font-size: 12px; line-height: 1.5; color: #0B1B33;
        transition: opacity .18s ease, transform .18s ease;
        pointer-events: none;
    }
    .ds-tooltip::before {
        content: ""; position: absolute;
        top: -7px; left: 50%; transform: translateX(-50%);
        width: 14px; height: 14px;
        background: white;
        border-left: 1px solid #E2E8F0;
        border-top: 1px solid #E2E8F0;
        rotate: 45deg;
    }
    .ds-info:hover .ds-tooltip {
        visibility: visible; opacity: 1;
        transform: translateX(-50%) translateY(0);
    }
    .ds-tooltip-title {
        font-size: 11px; text-transform: uppercase;
        letter-spacing: 0.7px; color: #003B71;
        font-weight: 800; margin-bottom: 10px;
        padding-bottom: 8px; border-bottom: 1px solid #E2E8F0;
    }
    .ds-step {
        display: flex; gap: 10px; margin-bottom: 10px;
    }
    .ds-step:last-child { margin-bottom: 0; }
    .ds-step-num {
        flex-shrink: 0;
        width: 22px; height: 22px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0072CE, #00B5E2);
        color: white;
        display: flex; align-items: center; justify-content: center;
        font-size: 11px; font-weight: 800;
    }
    .ds-step-text { flex: 1; font-size: 12px; line-height: 1.45; }
    .ds-step-name {
        font-weight: 700; color: #003B71;
        display: block; margin-bottom: 2px;
        font-size: 12px;
    }



    /* Streamlit native expander tweak (used both in main and sidebar) */
    [data-testid="stExpander"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        background: white !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        background: linear-gradient(135deg, #F8FAFC, #EEF2F7) !important;
        font-weight: 700 !important;
        color: #003B71 !important;
        font-size: 12px !important;
        padding: 10px 14px !important;
    }

    /* ── Sidebar dataset card (compact) ── */
    /* Allow tooltip to escape the sidebar */
    section[data-testid="stSidebar"] {
        overflow: visible !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        overflow: visible !important;
    }
    section[data-testid="stSidebar"] .block-container {
        overflow: visible !important;
    }

    /* Clickable dataset-name button (styled to look like a card) */
    section[data-testid="stSidebar"] .stButton button {
        background:
            linear-gradient(135deg, rgba(0,114,206,0.06) 0%, rgba(0,181,226,0.06) 100%),
            white !important;
        border: 1px solid #DBEAFE !important;
        border-left: 4px solid #0072CE !important;
        border-radius: 12px !important;
        color: #0B1B33 !important;
        font-family: 'Inter', monospace !important;
        font-size: 13px !important;
        font-weight: 800 !important;
        text-align: left !important;
        padding: 14px 16px !important;
        height: auto !important;
        line-height: 1.3 !important;
        letter-spacing: -0.2px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
        transition: all .15s ease !important;
        white-space: normal !important;
        cursor: pointer;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background:
            linear-gradient(135deg, rgba(0,114,206,0.14) 0%, rgba(0,181,226,0.14) 100%),
            white !important;
        border-color: #0072CE !important;
        border-left-color: #003B71 !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(0, 114, 206, 0.18) !important;
        color: #003B71 !important;
    }
    section[data-testid="stSidebar"] .stButton button:active,
    section[data-testid="stSidebar"] .stButton button:focus {
        background:
            linear-gradient(135deg, rgba(0,114,206,0.20) 0%, rgba(0,181,226,0.20) 100%),
            white !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(0, 114, 206, 0.18) !important;
    }
    section[data-testid="stSidebar"] .stButton button p {
        font-family: 'Inter', monospace !important;
        font-size: 13px !important;
        font-weight: 800 !important;
        margin: 0 !important;
    }

    /* Meta + "+" row that sits below the dataset button */
    .sb-ds-meta-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 4px 0 8px;
        margin-bottom: 8px;
    }
    .sb-ds-meta-text {
        font-size: 10px;
        color: #64748B;
        font-weight: 600;
    }

    /* The "+" info icon — tooltip pops to the RIGHT (sidebar is on the left) */
    .sb-ds-info {
        position: relative;
        display: inline-flex;
        align-items: center; justify-content: center;
        width: 22px; height: 22px;
        border-radius: 50%;
        background: white;
        border: 1.5px solid #0072CE;
        color: #0072CE;
        font-size: 14px; font-weight: 700;
        cursor: help;
        transition: all .15s ease;
        line-height: 1;
        flex-shrink: 0;
    }
    .sb-ds-info:hover {
        background: #0072CE; color: white;
        transform: scale(1.1);
        box-shadow: 0 4px 10px rgba(0, 114, 206, 0.30);
    }
    .sb-ds-tooltip {
        visibility: hidden; opacity: 0;
        position: absolute;
        top: 50%;
        left: calc(100% + 16px);
        transform: translateY(-50%);
        z-index: 10000;
        width: 340px;
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow:
            0 12px 30px rgba(15, 23, 42, 0.18),
            0 2px 6px rgba(15, 23, 42, 0.08);
        text-align: left;
        font-size: 12px; line-height: 1.45; color: #0B1B33;
        transition: opacity .18s ease, transform .18s ease;
        pointer-events: none;
    }
    /* Arrow pointing left */
    .sb-ds-tooltip::before {
        content: ""; position: absolute;
        top: 50%; left: -7px; transform: translateY(-50%);
        width: 14px; height: 14px;
        background: white;
        border-left: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        rotate: 45deg;
    }
    .sb-ds-info:hover .sb-ds-tooltip {
        visibility: visible; opacity: 1;
        transform: translateY(-50%);
    }

    /* ────────────────────────────────────────────────────────────
       Effective 90% zoom on the MAIN content area only.
       Reproduces the "Ctrl + −" feel without forcing the user to
       zoom out manually.  Sidebar stays at native size.
    ──────────────────────────────────────────────────────────── */
    section[data-testid="stMain"] .block-container {
        zoom: 0.9 !important;
        max-width: 1600px;
    }
    [data-testid="stDialog"], [role="dialog"] {
        zoom: 0.9 !important;
    }

    /* ────────────────────────────────────────────────────────────
       Animations — used across the app
    ──────────────────────────────────────────────────────────── */
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    @keyframes pulse {
        0%, 100% { opacity: .35; transform: scale(1); }
        50%      { opacity: 1;   transform: scale(1.15); }
    }
    @keyframes shimmer {
        0%   { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    @keyframes progress {
        0%   { width: 0%; }
        50%  { width: 70%; }
        100% { width: 100%; }
    }

    /* Style Streamlit's built-in spinner with brand colors + bigger ring */
    [data-testid="stSpinner"] {
        text-align: center;
    }
    [data-testid="stSpinner"] > div {
        border-color: #0072CE transparent transparent transparent !important;
        border-width: 4px !important;
        animation-duration: 0.9s !important;
    }
    [data-testid="stSpinner"] + div,
    [data-testid="stSpinner"] p {
        color: #003B71 !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }

    /* ── Custom branded loader (overlay) ──────────────────────── */
    .brand-loader {
        text-align: center;
        padding: 36px 28px;
        background: white;
        border-radius: 18px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 12px 32px rgba(15,23,42,0.10);
        max-width: 460px;
        margin: 60px auto;
        animation: slideUp 0.4s ease;
    }
    .brand-loader-ring {
        width: 68px; height: 68px;
        border: 5px solid #DBEAFE;
        border-top-color: #0072CE;
        border-right-color: #00B5E2;
        border-radius: 50%;
        margin: 0 auto 18px auto;
        animation: spin 0.9s linear infinite;
    }
    .brand-loader-title {
        font-size: 16px;
        font-weight: 800;
        color: #003B71;
        letter-spacing: 0.4px;
        margin-bottom: 4px;
    }
    .brand-loader-sub {
        font-size: 12px;
        color: #64748B;
        font-weight: 500;
    }
    .brand-loader-bar {
        margin: 22px auto 0 auto;
        max-width: 240px;
        height: 6px;
        background: #EEF2F7;
        border-radius: 999px;
        overflow: hidden;
    }
    .brand-loader-bar > span {
        display: block;
        height: 100%;
        background: linear-gradient(90deg, #0072CE, #00B5E2);
        border-radius: 999px;
        animation: progress 1.8s ease-in-out infinite;
    }
    .brand-loader-dots {
        display: inline-flex; gap: 6px;
        margin-top: 12px;
    }
    .brand-loader-dots > span {
        width: 6px; height: 6px;
        background: #0072CE;
        border-radius: 50%;
        animation: pulse 1.2s ease-in-out infinite;
    }
    .brand-loader-dots > span:nth-child(2) { animation-delay: .15s; }
    .brand-loader-dots > span:nth-child(3) { animation-delay: .30s; }
    .brand-loader-dots > span:nth-child(4) { animation-delay: .45s; }

    /* Shimmer skeleton for plotly chart placeholders */
    .skeleton {
        background: linear-gradient(90deg,
            rgba(241,245,249,0.6) 0%,
            rgba(226,232,240,0.9) 50%,
            rgba(241,245,249,0.6) 100%);
        background-size: 1000px 100%;
        animation: shimmer 1.5s linear infinite;
        border-radius: 14px;
    }
    .skeleton-chart {
        height: 320px;
        margin-bottom: 14px;
    }

    /* Fade-in everything that's a top-level section header */
    .section-header {
        animation: slideUp 0.35s ease both;
    }

    /* Hide streamlit chrome */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] {
        background: transparent !important; height: 3rem;
    }
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        background: #FFFFFF; border: 1px solid #E2E8F0;
        border-radius: 8px; color: #0F172A;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    </style>
    """, unsafe_allow_html=True)


def brand_loader(title: str = "Loading", sub: str = ""):
    """Return a context manager that shows a branded loading card while a
    block of code runs (replaces st.spinner for the heavier waits).
    """
    placeholder = st.empty()
    html = (
        f'<div class="brand-loader">'
        f'<div class="brand-loader-ring"></div>'
        f'<div class="brand-loader-title">{title}</div>'
        f'<div class="brand-loader-sub">{sub}</div>'
        f'<div class="brand-loader-bar"><span></span></div>'
        f'<div class="brand-loader-dots">'
        f'<span></span><span></span><span></span><span></span>'
        f'</div></div>'
    )
    placeholder.markdown(html, unsafe_allow_html=True)
    return placeholder

# ─────────────────────────────────────────────────────────────────────────
# Plotly theme helper
# ─────────────────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_FONT = dict(family="Outfit, -apple-system, sans-serif",
                    size=14, color="#475569")

def _style_fig(fig, height=400, title=None, legend_top=False, subtitle=None):
    fig.update_layout(
        template=PLOTLY_TEMPLATE, font=PLOTLY_FONT, height=height,
        margin=dict(l=24, r=24, t=72 if title else 32, b=32),
        paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)",
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#CBD5E1",
                         font=dict(family="Outfit, sans-serif",
                                    size=13, color="#0F172A")),
        title=None,
    )
    if title:
        title_text = (
            f"<span style='font-size:16px;color:#0F172A;font-weight:700'>{title}</span>"
            + (f"<br><span style='font-size:13px;color:#64748B;font-weight:400'>{subtitle}</span>"
               if subtitle else "")
        )
        fig.update_layout(title=dict(text=title_text, x=0.02, y=0.96,
                                      xanchor="left", yanchor="top",
                                      pad=dict(t=8)))
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.04)", linecolor="rgba(0,0,0,0.08)",
                     zerolinecolor="rgba(0,0,0,0.08)",
                     tickfont=dict(color="#64748B", size=13))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.04)", linecolor="rgba(0,0,0,0.08)",
                     zerolinecolor="rgba(0,0,0,0.08)",
                     tickfont=dict(color="#64748B", size=13))
    if legend_top:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom",
                                       y=1.05, xanchor="right", x=1,
                                       font=dict(size=13, color="#475569"),
                                       bgcolor="rgba(255,255,255,0)"))
    return fig


# ─────────────────────────────────────────────────────────────────────────
# Status helpers
# ─────────────────────────────────────────────────────────────────────────
def accuracy_status(v):
    if v >= 0.65: return "ok", "Strong"
    if v >= 0.50: return "fair", "Acceptable"
    return "poor", "Weak"

def recall_c_status(v):
    if v >= 0.55: return "ok", "Strong"
    if v >= 0.40: return "fair", "Acceptable"
    return "poor", "Weak"

def c_loss_status(v):
    if v <= 0.18: return "ok", "Low"
    if v <= 0.25: return "fair", "Moderate"
    return "poor", "High"


# ─────────────────────────────────────────────────────────────────────────
# Modeling pipeline
# ─────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(f"{dataset_name}.csv")
    df["ATSEG_first"] = df["ATSEG_first"].replace("0", np.nan)
    return df


@st.cache_data(show_spinner=False)
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered ratio + diversity + engagement features (from 3d.2.py)."""
    out = df.copy()
    eps = 1e-6
    # Pairwise ratios
    for new, num, den in [
        ("ratio_UC_over_ORAL",       "UC_TRX_sum",     "ORAL_TRX_sum"),
        ("ratio_UC_over_IL23",       "UC_TRX_sum",     "IL23_TRX_sum"),
        ("ratio_ORAL_over_IL23",     "ORAL_TRX_sum",   "IL23_TRX_sum"),
        ("ratio_UC_NRX_TRX",         "UC_NRX_sum",     "UC_TRX_sum"),
        ("ratio_DETAILS_per_TRX",    "DETAILS_sum",    "UC_TRX_sum"),
        ("ratio_BRAND1_over_UC",     "BRAND1_TRX_sum", "UC_TRX_sum"),
        ("ratio_BRAND1_over_ORAL",   "BRAND1_TRX_sum", "ORAL_TRX_sum"),
        ("ratio_NRX_over_TRX_ORAL",  "ORAL_NRX_sum",   "ORAL_TRX_sum"),
        ("ratio_NBRX_over_NRX_B1",   "BRAND1_NBRX_sum", "BRAND1_NRX_sum"),
        ("ratio_SAMPLES_per_DETAIL", "SAMPLES_sum",    "DETAILS_sum"),
    ]:
        if num in out.columns and den in out.columns:
            out[new] = out[num] / (out[den] + eps)

    # Brand diversity / concentration
    bc = [c for c in out.columns
          if c.startswith("BRAND") and "_TRX_sum" in c]
    if bc:
        out["brand_diversity"] = (out[bc] > 0).sum(axis=1)
    bn = [c for c in out.columns
          if c.startswith("BRAND") and "_NRX_sum" in c]
    if bn:
        t = out[bn].sum(axis=1)
        out["brand_concentration"] = out[bn].max(axis=1) / (t + eps)

    # Total engagement
    eng_cols = [c for c in out.columns if any(
        e in c for e in ["DETAILS_sum", "SAMPLES_sum", "RTE_sum",
                          "SPK_sum", "COPAY_sum", "DIRECTMAIL_sum"])]
    if eng_cols:
        out["total_engagement"] = out[eng_cols].sum(axis=1)

    # Claim diversity / total
    clm = [c for c in out.columns if c.startswith("N_CLM") and "_sum" in c]
    if clm:
        out["claims_diversity"] = (out[clm] > 0).sum(axis=1)
        out["total_claims"]     = out[clm].sum(axis=1)

    # log1p of large-magnitude columns (helps tree splits stay numerically stable)
    for col in ["UC_TRX_sum", "ORAL_TRX_sum", "IL23_TRX_sum",
                 "TOTAL_TRX_sum", "UC_NRX_sum", "DETAILS_sum"]:
        if col in out.columns:
            out[f"log_{col}"] = np.log1p(np.maximum(out[col], 0))
    return out


# Backwards-compat alias so any leftover callers still work
add_ratios = add_features


def fit_ordinal(X_train, y_train):
    y_geq_b = (y_train != "SEG_A").astype(int)
    y_geq_c = (y_train == "SEG_C").astype(int)
    m1 = xgb.XGBClassifier(**ORDINAL_PARAMS); m1.fit(X_train, y_geq_b)
    sw = np.where(y_geq_c == 1, SEG_C_SAMPLE_WEIGHT, 1.0)
    m2 = xgb.XGBClassifier(**ORDINAL_PARAMS); m2.fit(X_train, y_geq_c, sample_weight=sw)
    return m1, m2


def predict_ordinal(m1, m2, X):
    """Decision cascade — thresholds + dominance rule.

      P(A) ≥ 0.70  → SEG_A
      P(C) ≥ 0.30  → SEG_C
      P(C) > P(B)  → SEG_C   ← dominance rule (v3 capstone change)
      else         → SEG_B
    """
    p_b = m1.predict_proba(X)[:, 1]
    p_c = np.minimum(m2.predict_proba(X)[:, 1], p_b)
    P_A, P_B, P_C = 1 - p_b, p_b - p_c, p_c
    s = P_A + P_B + P_C + 1e-9
    P_A, P_B, P_C = P_A / s, P_B / s, P_C / s
    pred = np.where(P_A >= THR_A, "SEG_A",
           np.where(P_C >= THR_C, "SEG_C",
           np.where(P_C > P_B, "SEG_C", "SEG_B")))
    return P_A, P_B, P_C, pred


def fit_argmax(X_train, y_train):
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    sw = compute_sample_weight("balanced", y=y_enc)
    m = xgb.XGBClassifier(**ARGMAX_PARAMS)
    m.fit(X_train, y_enc, sample_weight=sw)
    return m, le


def predict_argmax(m, le, X):
    proba = m.predict_proba(X)
    classes = list(le.classes_)
    a, b, c = classes.index("SEG_A"), classes.index("SEG_B"), classes.index("SEG_C")
    P_A, P_B, P_C = proba[:, a], proba[:, b], proba[:, c]
    pred_idx = proba.argmax(axis=1)
    pred = le.inverse_transform(pred_idx).astype(object)
    return P_A, P_B, P_C, pred


def metrics_for(y_true, pred):
    acc = float((pred == y_true).mean())
    bal_acc = float(balanced_accuracy_score(y_true, pred))
    mask_C = y_true == "SEG_C"
    recall_C = float((pred[mask_C] == "SEG_C").mean()) if mask_C.sum() > 0 else 0.0
    c_lost_pct = float(((y_true == "SEG_C") & (pred == "SEG_A")).sum()
                        / max(mask_C.sum(), 1))
    cm = confusion_matrix(y_true, pred, labels=list(VALID_LABELS))
    return {
        "accuracy": acc, "balanced_accuracy": bal_acc,
        "recall_C": recall_C, "c_lost_pct": c_lost_pct,
        "confusion_matrix": cm,
    }


# ─────────────────────────────────────────────────────────────────────────
# SHAP helpers (3d.2.py)
# ─────────────────────────────────────────────────────────────────────────
def get_shap_values(model, X):
    """SHAP-like contributions from XGBoost (drop the bias term)."""
    dm = xgb.DMatrix(X)
    contribs = model.get_booster().predict(dm, pred_contribs=True)
    return contribs[:, :-1]


def top_shap_per_doctor(shap_vals, top_n=N_SHAP_TOP):
    """Return the top-N feature indices and values per doctor."""
    out = []
    for i in range(shap_vals.shape[0]):
        v = shap_vals[i]
        idx = np.argsort(np.abs(v))[::-1][:top_n]
        out.append([(int(j), float(v[j])) for j in idx])
    return out


@st.cache_resource(show_spinner=False)
def train_pipeline(dataset_name: str, n_folds: int = 5):
    """OOF CV for both models on labeled HCPs + score unlabeled.

    Adds (vs the previous version):
      - Per-doctor 95% confidence intervals on (P_A, P_B, P_C) from CV folds
      - SHAP top-N feature contributions per labeled doctor (Ordinal P>=C model)
    """
    df = load_data(dataset_name)
    df = add_features(df)

    # Identify labeled
    is_labeled = df["ATSEG_first"].isin(VALID_LABELS).values
    y_all = df["ATSEG_first"].fillna("").astype(str).values
    ids_all = df["NUEVO_ID"].astype(str).values

    # Feature columns: drop ID, label, and any week columns + non-numeric
    drop_cols = {"NUEVO_ID", "ATSEG_first"}
    drop_cols |= {c for c in df.columns if c.startswith("WEEK_ID")}
    feat_cols = [c for c in df.columns
                 if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    X_all = df[feat_cols].fillna(0).values
    X_lab = X_all[is_labeled]
    y_lab = y_all[is_labeled]
    X_unlab = X_all[~is_labeled]
    n_labeled = int(is_labeled.sum())
    n_unlabeled = int((~is_labeled).sum())

    # OOF on labeled — also store per-fold predictions on the full labeled set
    # to derive 95% CI bounds.
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    n = len(y_lab)
    oof = {
        "ord": {"P_A": np.zeros(n), "P_B": np.zeros(n), "P_C": np.zeros(n),
                 "pred": np.empty(n, dtype=object)},
        "amx": {"P_A": np.zeros(n), "P_B": np.zeros(n), "P_C": np.zeros(n),
                 "pred": np.empty(n, dtype=object)},
    }
    cf_probs = np.zeros((n, n_folds, 3))   # ordinal CV probs on full labeled

    for fold_i, (tr, va) in enumerate(skf.split(X_lab, y_lab)):
        m1, m2 = fit_ordinal(X_lab[tr], y_lab[tr])
        pa, pb, pc, pr = predict_ordinal(m1, m2, X_lab[va])
        oof["ord"]["P_A"][va] = pa; oof["ord"]["P_B"][va] = pb
        oof["ord"]["P_C"][va] = pc; oof["ord"]["pred"][va] = pr

        # Score full labeled set with this fold's models for CI calculation
        pa_all, pb_all, pc_all, _ = predict_ordinal(m1, m2, X_lab)
        cf_probs[:, fold_i, :] = np.column_stack([pa_all, pb_all, pc_all])

        m, le = fit_argmax(X_lab[tr], y_lab[tr])
        pa, pb, pc, pr = predict_argmax(m, le, X_lab[va])
        oof["amx"]["P_A"][va] = pa; oof["amx"]["P_B"][va] = pb
        oof["amx"]["P_C"][va] = pc; oof["amx"]["pred"][va] = pr

    # 95% CI bounds (mean ± 1.96·std across folds)
    cf_mean = cf_probs.mean(axis=1)
    cf_std  = cf_probs.std(axis=1)
    ci_lo_lab = np.clip(cf_mean - 1.96 * cf_std, 0, 1)
    ci_hi_lab = np.clip(cf_mean + 1.96 * cf_std, 0, 1)

    # Train final on all labeled, score unlabeled
    if n_unlabeled > 0:
        m1, m2 = fit_ordinal(X_lab, y_lab)
        ord_unlab = predict_ordinal(m1, m2, X_unlab)
        m_amx, le_amx = fit_argmax(X_lab, y_lab)
        amx_unlab = predict_argmax(m_amx, le_amx, X_unlab)
        # Final-model SHAP on labeled HCPs (using P>=C model which carries the
        # SEG_C signal — the most actionable for the business)
        shap_lab = get_shap_values(m2, X_lab)
    else:
        ord_unlab = amx_unlab = (np.array([]),) * 4
        m1, m2 = fit_ordinal(X_lab, y_lab)
        shap_lab = get_shap_values(m2, X_lab)

    top_shap_lab = top_shap_per_doctor(shap_lab, top_n=N_SHAP_TOP)

    # Reassemble full vectors
    full = {}
    for mod, oof_d, unlab_d in [("ord", oof["ord"], ord_unlab),
                                ("amx", oof["amx"], amx_unlab)]:
        PA = np.zeros(len(df)); PB = np.zeros(len(df)); PC = np.zeros(len(df))
        pred = np.empty(len(df), dtype=object)
        PA[is_labeled] = oof_d["P_A"]; PB[is_labeled] = oof_d["P_B"]
        PC[is_labeled] = oof_d["P_C"]; pred[is_labeled] = oof_d["pred"]
        if n_unlabeled > 0:
            PA[~is_labeled] = unlab_d[0]; PB[~is_labeled] = unlab_d[1]
            PC[~is_labeled] = unlab_d[2]; pred[~is_labeled] = unlab_d[3]
        full[mod] = {"P_A": PA, "P_B": PB, "P_C": PC, "pred": pred}

    # CI bounds aligned to full df length (zeros for unlabeled)
    ci_full_lo = np.zeros((len(df), 3))
    ci_full_hi = np.zeros((len(df), 3))
    ci_full_lo[is_labeled] = ci_lo_lab
    ci_full_hi[is_labeled] = ci_hi_lab

    # SHAP aligned to full df length (None for unlabeled)
    shap_full = [None] * len(df)
    lab_indices = np.where(is_labeled)[0]
    for k, idx in enumerate(lab_indices):
        shap_full[idx] = top_shap_lab[k]

    # Metrics on labeled only
    metrics = {}
    for mod in ["ord", "amx"]:
        metrics[mod] = metrics_for(y_lab, oof[mod]["pred"])

    # Available key features (drop the actionable flag for the legacy 3-tuple
    # consumers; keep the full 4-tuple for the conversion strategy).
    avail_keys      = [(k, n_, d_) for k, n_, d_, _ in KEY_FEATURES if k in df.columns]
    avail_keys_full = [(k, n_, d_, a) for k, n_, d_, a in KEY_FEATURES if k in df.columns]

    # Raw (un-engineered) feature values for the conversion strategy /
    # explorer profile — read straight from df by their original column name.
    raw_fv = {k: df[k].values.copy() for k, _, _, _ in KEY_FEATURES if k in df.columns}

    return {
        "df": df,
        "ids": ids_all,
        "is_labeled": is_labeled,
        "y_true": y_all,
        "n_labeled": n_labeled,
        "n_unlabeled": n_unlabeled,
        "feat_cols": feat_cols,
        "full": full,
        "metrics": metrics,
        "available_features": avail_keys,
        # New (3d.2.py)
        "available_features_full": avail_keys_full,  # 4-tuple incl. actionable
        "raw_fv": raw_fv,                # raw key-feature values per HCP
        "ci_lo": ci_full_lo,             # (N, 3) lower CI bounds for P_A/B/C
        "ci_hi": ci_full_hi,             # (N, 3) upper CI bounds for P_A/B/C
        "shap_top": shap_full,           # list of [(feat_idx, shap_val), ...] per HCP
        # Trained models (needed for counterfactual simulation)
        "final_m1": m1,                  # P(>=B) model
        "final_m2": m2,                  # P(>=C) model
    }


# ─────────────────────────────────────────────────────────────────────────
# Segment medians for the individual radar (centroids_v2.py style)
# ─────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def segment_medians(dataset_name: str, top_n: int = 6):
    """Median value of the top-N actionable+signal key features per segment.

    Returns:
      feats   — list of (column, friendly name) actually used (top_n entries)
      medians — dict {segment: [median per feature, …]}
    """
    R = train_pipeline(dataset_name)
    df = R["df"]
    avail = R["available_features_full"]  # 4-tuple
    # Use the first top_n features that exist in the dataset
    feats_full = [(k, n_) for k, n_, _, _ in avail][:top_n]
    medians = {}
    for seg in VALID_LABELS:
        mask = df["ATSEG_first"] == seg
        medians[seg] = [
            float(df.loc[mask, k].median()) if mask.sum() > 0 else 0.0
            for k, _ in feats_full
        ]
    return feats_full, medians


# ─────────────────────────────────────────────────────────────────────────
# Conversion Strategy (B → C) — adapted from 3d.2.py
# ─────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
# ─────────────────────────────────────────────────────────────────────────
# Counterfactual simulation (from counterfactual_simulation.py)
# ─────────────────────────────────────────────────────────────────────────
def _simulate(df_raw, feat_cols, _m1, _m2, seg_b_idx,
               details_delta=None, details_target=None,
               samples_delta=None, samples_target=None):
    """Modify raw DETAILS/SAMPLES for SEG_B doctors → re-engineer → re-predict.

    The features must be re-derived from raw because ratios and logs
    (e.g. ratio_DETAILS_per_TRX, log_DETAILS_sum) depend on the same
    columns being perturbed.
    """
    df_sim = df_raw.copy()
    sb_rows = df_sim.index[seg_b_idx]

    if details_delta is not None:
        df_sim.loc[sb_rows, "DETAILS_sum"] = (
            df_sim.loc[sb_rows, "DETAILS_sum"] + details_delta)
    if details_target is not None:
        df_sim.loc[sb_rows, "DETAILS_sum"] = (
            df_sim.loc[sb_rows, "DETAILS_sum"].clip(lower=details_target))
    if samples_delta is not None:
        df_sim.loc[sb_rows, "SAMPLES_sum"] = (
            df_sim.loc[sb_rows, "SAMPLES_sum"] + samples_delta)
    if samples_target is not None:
        df_sim.loc[sb_rows, "SAMPLES_sum"] = (
            df_sim.loc[sb_rows, "SAMPLES_sum"].clip(lower=samples_target))

    df_sim = add_features(df_sim)
    X_sim = df_sim.reindex(columns=feat_cols, fill_value=0).fillna(0).values
    PA_s, PB_s, PC_s, pred_s = predict_ordinal(_m1, _m2, X_sim[seg_b_idx])
    return PA_s, PB_s, PC_s, pred_s


@st.cache_data(show_spinner=False)
def run_counterfactual_scenarios(dataset_name: str):
    """Counterfactual scenarios for predicted-SEG_B doctors.

    Returns a dict with:
      - summary table (scenarios × outcomes)
      - flippers vs stayers indices for the "+1 visit" scenario
      - per-doctor P(C) before and after +1 visit (for distribution shift)
      - diminishing-returns sweep (0..10 added visits → cumulative B→C count)
    """
    R = train_pipeline(dataset_name)
    m1, m2 = R["final_m1"], R["final_m2"]
    feat_cols = R["feat_cols"]
    df_raw = load_data(dataset_name)

    # Identify predicted-SEG_B doctors using the model output
    pred_all = R["full"]["ord"]["pred"]
    PC_all   = R["full"]["ord"]["P_C"]
    seg_b_idx = np.where(pred_all == "SEG_B")[0]
    n_seg_b = int(len(seg_b_idx))

    # SEG_C medians on labeled doctors only
    y_all = R["y_true"]; is_lab = R["is_labeled"]
    c_mask = is_lab & (y_all == "SEG_C")
    details_c_med = (float(np.median(df_raw["DETAILS_sum"].values[c_mask]))
                      if c_mask.sum() > 0 else 0.0)
    samples_c_med = (float(np.median(df_raw["SAMPLES_sum"].values[c_mask]))
                      if c_mask.sum() > 0 else 0.0)

    # ── Scenario list ──
    scenarios = [
        ("DETAILS", "+1 visit",
         dict(details_delta=1)),
        ("DETAILS", "+2 visits",
         dict(details_delta=2)),
        ("DETAILS", "+3 visits",
         dict(details_delta=3)),
        ("DETAILS", "+5 visits",
         dict(details_delta=5)),
        ("DETAILS", f"→ SEG_C median (≈{details_c_med:.0f})",
         dict(details_target=details_c_med)),
        ("SAMPLES", "+1 sample",
         dict(samples_delta=1)),
        ("SAMPLES", "+2 samples",
         dict(samples_delta=2)),
        ("COMBINED", "+1 visit + 1 sample",
         dict(details_delta=1, samples_delta=1)),
        ("COMBINED", "+2 visits + 2 samples",
         dict(details_delta=2, samples_delta=2)),
        ("COMBINED", "Both → SEG_C medians",
         dict(details_target=details_c_med, samples_target=samples_c_med)),
    ]

    rows = []
    for group, label, spec in scenarios:
        _, _, PC_s, pred_s = _simulate(df_raw, feat_cols, m1, m2,
                                        seg_b_idx, **spec)
        b_to_c = int((pred_s == "SEG_C").sum())
        b_to_a = int((pred_s == "SEG_A").sum())
        b_stay = int((pred_s == "SEG_B").sum())
        rows.append({
            "Group": group, "Scenario": label,
            "B→C": b_to_c, "B stays": b_stay, "B→A": b_to_a,
            "Total B": n_seg_b,
            "% to C": round(b_to_c / max(n_seg_b, 1) * 100, 1),
            "Mean P(C) after": round(float(PC_s.mean()), 4),
        })
    summary_df = pd.DataFrame(rows)

    # ── Detailed +1 visit run for the distribution-shift chart ──
    _, _, PC_plus1, pred_plus1 = _simulate(df_raw, feat_cols, m1, m2,
                                            seg_b_idx, details_delta=1)
    flippers_mask = pred_plus1 == "SEG_C"
    flippers_idx = seg_b_idx[flippers_mask]
    stayers_mask = pred_plus1 == "SEG_B"
    stayers_idx = seg_b_idx[stayers_mask]
    pc_base_seg_b = PC_all[seg_b_idx]

    # ── Diminishing returns sweep: 0..10 additional visits ──
    dim_returns = []
    for delta in range(0, 11):
        if delta == 0:
            n_flip = 0  # baseline
        else:
            _, _, _, pred_d = _simulate(df_raw, feat_cols, m1, m2,
                                          seg_b_idx, details_delta=delta)
            n_flip = int((pred_d == "SEG_C").sum())
        dim_returns.append({"Added visits": delta, "B→C cumulative": n_flip})
    dim_df = pd.DataFrame(dim_returns)
    # Compute incremental gains
    dim_df["Incremental"] = (dim_df["B→C cumulative"].diff().fillna(0)
                              .astype(int))

    return {
        "n_seg_b": n_seg_b,
        "details_c_med": details_c_med,
        "samples_c_med": samples_c_med,
        "summary": summary_df,
        "dim_returns": dim_df,
        "pc_base_seg_b": pc_base_seg_b,
        "pc_after_plus1": PC_plus1,
        "flippers_idx": flippers_idx,
        "stayers_idx": stayers_idx,
        "seg_b_idx": seg_b_idx,
        "details_baseline": df_raw["DETAILS_sum"].values,
        "samples_baseline": df_raw["SAMPLES_sum"].values,
    }


def simulate_single_hcp(dataset_name: str, hcp_idx: int,
                         details_delta: int = 0,
                         samples_delta: int = 0):
    """Per-HCP counterfactual — used by the Doctor Explorer."""
    R = train_pipeline(dataset_name)
    m1, m2 = R["final_m1"], R["final_m2"]
    feat_cols = R["feat_cols"]
    df_raw = load_data(dataset_name)
    df_sim = df_raw.copy()
    if details_delta:
        df_sim.loc[df_sim.index[hcp_idx], "DETAILS_sum"] += details_delta
    if samples_delta:
        df_sim.loc[df_sim.index[hcp_idx], "SAMPLES_sum"] += samples_delta
    df_sim = add_features(df_sim)
    X_sim = df_sim.reindex(columns=feat_cols, fill_value=0).fillna(0).values
    PA, PB, PC, pred = predict_ordinal(m1, m2, X_sim[[hcp_idx]])
    return float(PA[0]), float(PB[0]), float(PC[0]), str(pred[0])


def compute_conversion_strategy(dataset_name: str,
                                 n_candidates: int = N_CONVERSION_CANDIDATES):
    """Find predicted-SEG_B doctors whose P_C is highest and quantify the
    feature-level gaps that hold them back from being predicted as SEG_C.

    Returns:
      candidates_df  — DataFrame of N candidates (id, probs, true label, gap_*)
      aggregate_df   — DataFrame: per key feature, % of candidates below SEG_C
                       median + actionable flag, sorted by % below desc
      c_medians      — dict {feature_key: SEG_C median value}
    """
    R = train_pipeline(dataset_name)
    is_labeled = R["is_labeled"]
    y_all      = R["y_true"]
    raw_fv     = R["raw_fv"]
    feat_meta  = {k: (n_, d_, a) for (k, n_, d_, a) in R["available_features_full"]}
    PA = R["full"]["ord"]["P_A"]
    PB = R["full"]["ord"]["P_B"]
    PC = R["full"]["ord"]["P_C"]
    pred_ord = R["full"]["ord"]["pred"]
    ids_all  = R["ids"]

    # SEG_C medians (labeled doctors only)
    c_mask = is_labeled & (y_all == "SEG_C")
    c_medians = {k: float(np.median(raw_fv[k][c_mask]))
                 for k in raw_fv if c_mask.sum() > 0}

    # Candidates: predicted-B doctors, sorted by P_C descending
    pb_idx = np.where(pred_ord == "SEG_B")[0]
    pb_sorted = pb_idx[np.argsort(-PC[pb_idx])]
    cand_idx = pb_sorted[:n_candidates]

    # Build per-candidate row (probabilities + per-feature gap)
    rows = []
    for i in cand_idx:
        row = {
            "HCP_ID":   ids_all[i],
            "True":     y_all[i] if is_labeled[i] else "Unlabeled",
            "P(A)":     round(float(PA[i]), 3),
            "P(B)":     round(float(PB[i]), 3),
            "P(C)":     round(float(PC[i]), 3),
        }
        worst_gap = 0.0
        worst_feat = None
        for k in c_medians:
            v = float(raw_fv[k][i])
            gap = v - c_medians[k]
            row[f"{feat_meta[k][0]}"] = round(v, 2)
            # Track most-negative actionable gap → "top action"
            if feat_meta[k][2] and gap < worst_gap:
                worst_gap = gap
                worst_feat = feat_meta[k][0]
        row["Top action"] = (f"↑ {worst_feat}"
                              if worst_feat else "—")
        row["Action gap"] = round(worst_gap, 2) if worst_feat else 0.0
        rows.append(row)
    candidates_df = pd.DataFrame(rows)

    # Aggregate: per feature, % of candidates below SEG_C median
    agg_rows = []
    for k in c_medians:
        n_below = sum(1 for i in cand_idx
                       if float(raw_fv[k][i]) < c_medians[k])
        agg_rows.append({
            "Feature":     feat_meta[k][0],
            "Code":        k,
            "Actionable":  feat_meta[k][2],
            "% below SEG_C median": round(
                n_below / max(len(cand_idx), 1) * 100, 1),
            "SEG_C median": round(c_medians[k], 2),
        })
    aggregate_df = (pd.DataFrame(agg_rows)
                     .sort_values("% below SEG_C median", ascending=False)
                     .reset_index(drop=True))

    return candidates_df, aggregate_df, c_medians


# ─────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────
def kpi_card(col, label, value, helper="", style="", icon="",
             status=None, status_label=""):
    cls = f"kpi-card {style}".strip()
    icon_html = f'<div class="kpi-icon">{icon}</div>' if icon else ""
    status_html = f'<div class="kpi-status {status}">{status_label}</div>' if status else ""
    
    # Eliminamos las tabulaciones internas para obligar a Streamlit a leerlo como HTML puro
    html_content = f'<div class="{cls}">{icon_html}<div class="kpi-content-top"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-delta">{helper}</div></div><div class="kpi-content-bottom">{status_html}</div></div>'
    
    col.markdown(html_content, unsafe_allow_html=True)


def section(title, subtitle="", icon="", eyebrow=""):
    """Render a section header with the new emoji-free professional layout.

    A vertical brand-blue accent bar runs down the left edge; the title
    sits beside it in dark slate.  An optional `eyebrow` (small uppercase
    line above the title) lets callers add a discreet context tag.
    The `icon=` argument is accepted for backwards-compat and ignored.
    """
    eyebrow_html = (
        f'<span class="section-eyebrow">{eyebrow}</span>' if eyebrow else ""
    )
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    html = (
        f'<div class="section-header">'
        f'<div class="section-header-row">'
        f'{eyebrow_html}'
        f'<h2>{title}</h2>'
        f'</div>'
        f'{sub}'
        f'<div class="section-rule"></div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def context_strip(items, accent_index=None):
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


def sb_label(icon, title, color="blue"):
    """Sidebar group label. `icon` is kept for backwards-compat but is only
    rendered when non-empty so an empty string doesn't leave a visual gap.
    The colored vertical bar on the left replaces the missing icon slot.

    The HTML is built as a single-line string because Streamlit's markdown
    pipeline otherwise wraps blank-line segments in <p> tags and escapes
    the surrounding tag characters as plain text.
    """
    icon_html = (
        f'<span class="sb-label-icon">{icon}</span>'
        if icon and icon.strip() else ""
    )
    html = (
        f'<div class="sb-label">'
        f'<div class="sb-label-bar {color}"></div>'
        f'{icon_html}'
        f'<span class="sb-label-text">{title}</span>'
        f'</div>'
    )
    st.sidebar.markdown(html, unsafe_allow_html=True)


def sb_current(label, value):
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


MODEL_LABELS = {
    "ord": "Business-Calibrated Ordinal",
    "amx": "Argmax Baseline",
}
MODEL_LABELS_SHORT = {"ord": "Ordinal", "amx": "Argmax"}


# ─────────────────────────────────────────────────────────────────────────
# Cover / landing page — shown before the dashboard
# ─────────────────────────────────────────────────────────────────────────
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Init session state for the cover gate
if "entered_dashboard" not in st.session_state:
    st.session_state.entered_dashboard = False


def _render_cover():
    """Splash / landing page rendered before the dashboard.

    Shows: brand, project description, predicted-segment overview
    (live model output), and an arrow CTA to enter the dashboard.
    """

    # Hide the sidebar entirely while the cover is showing — it would just
    # add visual noise to a landing page.
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }
        /* While on the cover, narrow the main container so EVERY block —
           including Streamlit-rendered widgets like st.columns and plotly
           charts — shares the same left/right margins as our HTML cards.
           Zoom is handled globally at the stAppViewContainer level, so we
           don't redeclare it here (would compound to 0.81). */
        section[data-testid="stMain"] .block-container {
            max-width: 1100px !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }
        .cover-wrap {
            width: 100%;
            margin: 12px auto 0 auto;
            animation: slideUp 0.45s ease;
        }
        .cover-hero {
            position: relative;
            background:
                radial-gradient(ellipse at top right, rgba(0,181,226,0.30) 0%, transparent 60%),
                linear-gradient(135deg, #001F3F 0%, #003B71 35%, #0072CE 100%);
            color: white;
            padding: 48px 56px 44px 56px;
            border-radius: 22px;
            margin-bottom: 28px;
            box-shadow:
                0 16px 40px rgba(0, 59, 113, 0.28),
                0 4px 10px rgba(0, 59, 113, 0.14),
                inset 0 1px 0 rgba(255,255,255,0.08);
            overflow: hidden;
        }
        .cover-hero::before {
            content: "";
            position: absolute;
            top: -40%; right: -10%;
            width: 520px; height: 520px;
            background: radial-gradient(circle, rgba(0,181,226,0.32) 0%, transparent 65%);
            pointer-events: none;
        }
        .cover-hero::after {
            content: "";
            position: absolute;
            bottom: -55%; left: 20%;
            width: 460px; height: 460px;
            background: radial-gradient(circle, rgba(244,123,32,0.22) 0%, transparent 65%);
            pointer-events: none;
        }
        .cover-content { position: relative; z-index: 1; }
        .cover-eyebrow {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 3px;
            opacity: 0.9;
            font-weight: 700;
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .cover-eyebrow::before {
            content: "";
            display: inline-block;
            width: 36px; height: 2px;
            background: #00B5E2;
            border-radius: 2px;
        }
        .cover-title {
            font-size: 44px;
            font-weight: 800;
            letter-spacing: -1px;
            line-height: 1.05;
            margin: 0 0 10px 0;
        }
        .cover-subtitle {
            font-size: 16px;
            opacity: 0.88;
            font-weight: 400;
            margin: 0 0 6px 0;
            max-width: 760px;
            line-height: 1.55;
        }
        .cover-divider {
            height: 1px;
            background: rgba(255,255,255,0.18);
            margin: 24px 0 18px 0;
        }
        .cover-meta {
            display: flex; flex-wrap: wrap; gap: 10px;
        }
        .cover-pill {
            display: inline-flex; align-items: center; gap: 6px;
            background: rgba(255,255,255,0.12);
            backdrop-filter: blur(6px);
            padding: 8px 16px; border-radius: 999px;
            font-size: 13px;
            font-weight: 500;
            border: 1px solid rgba(255,255,255,0.18);
        }
        .cover-pill b { color: #FFE7D2; font-weight: 700; }

        /* Section title above the segment cards */
        .cover-sec-title {
            font-size: 14px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #003B71;
            margin: 22px 0 12px 4px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .cover-sec-title::before {
            content: "";
            width: 4px; height: 18px; border-radius: 3px;
            background: linear-gradient(180deg, #0072CE, #00B5E2);
        }

        /* Segment KPI cards */
        .seg-stat {
            background: white;
            border-radius: 16px;
            padding: 22px 24px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 12px rgba(15,23,42,0.04);
            position: relative;
            overflow: hidden;
            transition: transform .18s ease, box-shadow .18s ease;
        }
        .seg-stat:hover {
            transform: translateY(-3px);
            box-shadow: 0 14px 30px rgba(15,23,42,0.10);
        }
        .seg-stat::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; height: 5px;
        }
        .seg-stat.SEG_A::before { background: linear-gradient(90deg, #2E6CB0, #0072CE); }
        .seg-stat.SEG_B::before { background: linear-gradient(90deg, #F47B20, #FDB913); }
        .seg-stat.SEG_C::before { background: linear-gradient(90deg, #C44E52, #DC2626); }
        .seg-stat.Unlabeled::before { background: linear-gradient(90deg, #94A3B8, #CBD5E1); }
        .seg-stat-name {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.7px;
            color: #64748B;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .seg-stat-value {
            font-size: 38px;
            font-weight: 800;
            color: #0B1B33;
            line-height: 1.05;
            letter-spacing: -1px;
            font-feature-settings: "tnum";
        }
        .seg-stat-pct {
            font-size: 14px;
            color: #475569;
            font-weight: 600;
            margin-top: 6px;
        }
        .seg-stat-desc {
            font-size: 12px;
            color: #64748B;
            margin-top: 10px;
            line-height: 1.5;
        }

        /* Description card */
        .desc-card {
            background: white;
            border-radius: 16px;
            border: 1px solid #E2E8F0;
            border-left: 5px solid #0072CE;
            padding: 24px 28px;
            box-shadow: 0 4px 14px rgba(15,23,42,0.04);
            margin-bottom: 8px;
        }
        .desc-card h3 {
            margin: 0 0 10px 0;
            color: #003B71;
            font-size: 18px;
            font-weight: 700;
        }
        .desc-card p {
            color: #334155;
            font-size: 14.5px;
            line-height: 1.65;
            margin: 0 0 10px 0;
        }
        .desc-card .pillar {
            display: inline-block;
            background: #EFF6FF;
            border: 1px solid #DBEAFE;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 11px;
            color: #1E40AF;
            font-weight: 700;
            margin-right: 6px;
            margin-top: 4px;
            letter-spacing: 0.3px;
        }

        /* Enter button — fully styled by targeting the only button in main */
        section[data-testid="stMain"] .stButton button {
            background: linear-gradient(135deg, #003B71 0%, #0072CE 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 999px !important;
            padding: 18px 42px !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            letter-spacing: 0.5px !important;
            box-shadow: 0 10px 24px rgba(0, 114, 206, 0.30) !important;
            transition: all .18s ease !important;
            font-family: 'Inter', sans-serif !important;
            text-align: center !important;
            cursor: pointer;
            white-space: nowrap;
        }
        section[data-testid="stMain"] .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 32px rgba(0, 114, 206, 0.45) !important;
            background: linear-gradient(135deg, #002A52 0%, #005CA8 100%) !important;
        }
        section[data-testid="stMain"] .stButton button p {
            margin: 0 !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Load data & train (cached) to surface segment counts ──
    _loader = brand_loader(
        "Training XGBoost models",
        "5-fold CV · scoring 20,931 HCPs · computing SHAP · ~60s first run, instant after",
    )
    R = train_pipeline(DATASET_NAME)
    _loader.empty()
    pred_all = R["full"]["ord"]["pred"]
    is_lab = R["is_labeled"]
    n_total = len(R["ids"])

    seg_a_n = int((pred_all == "SEG_A").sum())
    seg_b_n = int((pred_all == "SEG_B").sum())
    seg_c_n = int((pred_all == "SEG_C").sum())
    lab_n   = int(is_lab.sum())
    unlab_n = int(n_total - lab_n)

    seg_descs = {
        "SEG_A": "Already strong prescribers — protect & maintain",
        "SEG_B": "Growth potential — primary targets for engagement",
        "SEG_C": "High-value prescribers — priority targets",
    }

    # Inject extra cover-specific styles for the new sections
    st.markdown(
        """
        <style>
        .cover-section {
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.4px;
            color: #003B71;
            margin: 32px 0 12px 4px;
            display: flex; align-items: center; gap: 10px;
        }
        .cover-section::before {
            content: "";
            width: 4px; height: 18px; border-radius: 3px;
            background: linear-gradient(180deg, #0072CE, #00B5E2);
        }
        .cover-section .count-chip {
            font-size: 11px; letter-spacing: .4px;
            background: #EFF6FF; color: #1E40AF;
            padding: 2px 10px; border-radius: 999px;
            border: 1px solid #DBEAFE; font-weight: 700;
        }

        /* Two-column model card */
        .model-card {
            background: white;
            border-radius: 16px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 14px rgba(15,23,42,0.05);
            overflow: hidden;
            margin-bottom: 14px;
        }
        .model-card-header {
            background: linear-gradient(135deg, #003B71, #0072CE);
            color: white;
            padding: 18px 24px;
            display: flex; align-items: center; gap: 14px;
        }
        .model-card-icon {
            width: 48px; height: 48px;
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.30);
            border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 18px;
            font-weight: 800;
            color: #FFFFFF;
            font-family: 'Inter', sans-serif;
            letter-spacing: 0.4px;
            flex-shrink: 0;
            position: relative;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.18);
        }
        /* Subtle accent corner inside the monogram tile */
        .model-card-icon::after {
            content: "";
            position: absolute;
            top: 6px; right: 6px;
            width: 6px; height: 6px;
            background: #00B5E2;
            border-radius: 50%;
        }
        .model-card-title {
            font-size: 18px; font-weight: 800;
            line-height: 1.15;
            margin: 0;
        }
        .model-card-sub {
            font-size: 12px; opacity: 0.86;
            font-weight: 500;
            letter-spacing: 0.3px;
            margin-top: 2px;
        }
        .model-card-body {
            padding: 22px 26px 24px 26px;
        }
        .model-card-body p {
            color: #334155;
            font-size: 14px;
            line-height: 1.7;
            margin: 0 0 14px 0;
        }
        .model-card-body p:last-child { margin-bottom: 0; }
        .model-card-body b { color: #003B71; }

        /* Pipeline steps */
        .pipeline {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }
        @media(max-width: 1100px){ .pipeline { grid-template-columns: repeat(2, 1fr); } }
        .pipe-step {
            position: relative;
            background: white;
            border-radius: 12px;
            border: 1px solid #E2E8F0;
            padding: 16px 18px;
            box-shadow: 0 2px 8px rgba(15,23,42,0.04);
            transition: transform .15s ease, box-shadow .15s ease;
        }
        .pipe-step:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 22px rgba(15,23,42,0.08);
        }
        .pipe-step::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, #0072CE, #00B5E2);
            border-radius: 12px 12px 0 0;
        }
        .pipe-num {
            width: 24px; height: 24px;
            background: linear-gradient(135deg, #003B71, #0072CE);
            color: white;
            border-radius: 50%;
            display: inline-flex; align-items: center; justify-content: center;
            font-size: 12px; font-weight: 800;
            margin-bottom: 8px;
        }
        .pipe-title {
            font-size: 13px;
            font-weight: 700;
            color: #003B71;
            margin: 0 0 4px 0;
        }
        .pipe-desc {
            font-size: 11.5px;
            color: #64748B;
            line-height: 1.5;
        }

        /* Tabs preview */
        .tabs-preview {
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 8px;
            position: relative;        /* anchor for tooltips */
            overflow: visible;
        }
        @media(max-width: 1100px){ .tabs-preview { grid-template-columns: repeat(3, 1fr); } }
        @media(max-width: 720px) { .tabs-preview { grid-template-columns: repeat(2, 1fr); } }

        .tab-card {
            position: relative;        /* anchor the popup to the card */
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 14px 14px 12px 14px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(15,23,42,0.04);
            transition: all .15s ease;
            cursor: help;
        }
        .tab-card:hover {
            transform: translateY(-2px);
            border-color: #BFDBFE;
            background: linear-gradient(180deg, #F8FAFC, white);
            box-shadow: 0 8px 20px rgba(15,23,42,0.10);
            z-index: 10;               /* sit above neighbours so popup wins */
        }
        .tab-card-icon {
            font-size: 24px;
            margin-bottom: 6px;
        }
        .tab-card-name {
            font-size: 11.5px;
            font-weight: 700;
            color: #003B71;
            letter-spacing: 0.3px;
        }

        /* ── Hover popup for each tab card ───────────────────────── */
        .tab-card-popup {
            position: absolute;
            bottom: calc(100% + 12px);
            left: 50%;
            transform: translateX(-50%) translateY(6px);
            min-width: 240px;
            max-width: 280px;
            background: white;
            color: #0B1B33;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 12px 14px;
            box-shadow:
                0 12px 28px rgba(15,23,42,0.18),
                0 3px 8px rgba(15,23,42,0.08);
            font-size: 11.5px;
            line-height: 1.55;
            text-align: left;
            opacity: 0;
            visibility: hidden;
            pointer-events: none;
            transition: opacity .18s ease, transform .18s ease,
                          visibility .18s ease;
            z-index: 100;
        }
        .tab-card-popup::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%) rotate(45deg);
            width: 12px; height: 12px;
            background: white;
            border-right: 1px solid #E2E8F0;
            border-bottom: 1px solid #E2E8F0;
            margin-top: -6px;
        }
        .tab-card-popup-title {
            display: block;
            font-size: 11.5px;
            font-weight: 800;
            color: #003B71;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
            padding-bottom: 6px;
            border-bottom: 1px solid #E2E8F0;
        }
        .tab-card-popup-text {
            color: #475569;
        }
        .tab-card:hover .tab-card-popup {
            opacity: 1;
            visibility: visible;
            transform: translateX(-50%) translateY(0);
        }
        /* Flip the first card's popup to the right so it doesn't get clipped
           by the container's left edge, and the last card's popup to the
           left so it doesn't get clipped by the right edge. */
        .tabs-preview .tab-card:first-child .tab-card-popup {
            left: 0;
            transform: translateY(6px);
        }
        .tabs-preview .tab-card:first-child:hover .tab-card-popup {
            transform: translateY(0);
        }
        .tabs-preview .tab-card:first-child .tab-card-popup::after {
            left: 20%;
        }
        .tabs-preview .tab-card:last-child .tab-card-popup {
            left: auto;
            right: 0;
            transform: translateY(6px);
        }
        .tabs-preview .tab-card:last-child:hover .tab-card-popup {
            transform: translateY(0);
        }
        .tabs-preview .tab-card:last-child .tab-card-popup::after {
            left: 80%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ─── HERO ─── (logo + title on the same row)
    try:
        _logo_b64 = get_image_base64("logo3.png")
        _logo_html = (
            f'<div style="display:inline-flex;align-items:center;'
            f'justify-content:center;width:88px;height:88px;'
            f'background:rgba(255,255,255,0.95);'
            f'border:1px solid rgba(255,255,255,0.35);'
            f'border-radius:20px;padding:10px;'
            f'box-shadow:0 6px 18px rgba(0,0,0,0.18);'
            f'flex-shrink:0;">'
            f'<img src="data:image/png;base64,{_logo_b64}" '
            f'style="width:100%;height:auto;display:block;" />'
            f'</div>'
        )
    except Exception:
        _logo_html = ""

    st.markdown(
        f"""
        <div class="cover-wrap">
          <div class="cover-hero" style="padding:40px 56px;">
            <div class="cover-content" style="display:flex;align-items:center;
                                                gap:28px;flex-wrap:wrap;">
              {_logo_html}
              <div class="cover-title" style="font-size:48px;margin:0;">
                Prescriber Probability Engine
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════════
    # 1. MODEL DESCRIPTION (moved here from the sidebar tooltip)
    # ═══════════════════════════════════════════════════════════════
    st.markdown(
        """
        <div class="cover-wrap">
          <div class="cover-section">About the Model</div>
          <div class="model-card">
            <div class="model-card-header">
              <div class="model-card-icon">Pfz</div>
              <div>
                <div class="model-card-title">Ordinal XGBoost Framework</div>
                <div class="model-card-sub">Business-calibrated · v3.2 · SHAP-explainable</div>
              </div>
            </div>
            <div class="model-card-body">
              <p>
                The final model is an <b>ordinal XGBoost framework</b> designed
                to segment healthcare providers into three categories based on
                their likelihood of prescribing Velsipity: <b>SEG_A</b>,
                <b>SEG_B</b>, and <b>SEG_C</b>. Instead of using a traditional
                multiclass classifier, the system models the problem as an
                <b>ordered classification task</b> through two sequential binary
                XGBoost models, allowing it to better capture the natural
                progression between physician segments. The model incorporates
                extensive feature engineering — prescription activity, sales
                representative interactions, drug sample distribution,
                competitor prescriptions, ratio-based metrics, and logarithmic
                transformations — to improve predictive performance and
                robustness.
              </p>
              <p>
                To align the predictions with Pfizer's commercial objectives,
                the model applies <b>custom business-calibrated thresholds</b>
                and a dominance rule that prioritises the identification of
                high-value prescribers while minimising costly false negatives.
                The final system also integrates <b>SHAP explainability</b> to
                provide transparent feature-level insights for each prediction
                and includes a <b>conversion analysis module</b> capable of
                identifying SEG_B physicians with strong potential to
                transition into SEG_C. Overall, the model functions not only
                as a predictive tool, but also as a strategic decision-support
                system for physician targeting and resource optimisation.
              </p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════════
    # 2. HCPs Classified per Segment — KPI cards + donut side by side
    # ═══════════════════════════════════════════════════════════════
    st.markdown(
        f'<div class="cover-wrap">'
        f'<div class="cover-section">HCPs Classified per Segment '
        f'<span class="count-chip">{n_total:,} total</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    def _seg_html(seg, count, total, desc):
        pct = count / max(total, 1) * 100
        return (
            f'<div class="seg-stat {seg}">'
            f'<div class="seg-stat-name">{seg.replace("_", " ")}</div>'
            f'<div class="seg-stat-value">{count:,}</div>'
            f'<div class="seg-stat-pct">{pct:.1f}% of {total:,} HCPs</div>'
            f'<div class="seg-stat-desc">{desc}</div>'
            f'</div>'
        )

    left, right = st.columns([3, 2])
    with left:
        # Vertical stack of segment cards
        st.markdown(_seg_html("SEG_A", seg_a_n, n_total, seg_descs["SEG_A"]),
                     unsafe_allow_html=True)
        st.markdown(_seg_html("SEG_B", seg_b_n, n_total, seg_descs["SEG_B"]),
                     unsafe_allow_html=True)
        st.markdown(_seg_html("SEG_C", seg_c_n, n_total, seg_descs["SEG_C"]),
                     unsafe_allow_html=True)
        if unlab_n > 0:
            st.markdown(
                f'<div style="font-size:12px;color:#64748B;'
                f'padding:14px 4px 4px 4px;line-height:1.55;">'
                f'<b style="color:#003B71">{lab_n:,}</b> HCPs have '
                f'ground-truth ATSEG labels · '
                f'<b style="color:#003B71">{unlab_n:,}</b> were scored '
                f'exclusively by the model.'
                f'</div>',
                unsafe_allow_html=True,
            )

    with right:
        df_seg = pd.DataFrame({
            "Segment": ["SEG_A", "SEG_B", "SEG_C"],
            "HCPs":    [seg_a_n, seg_b_n, seg_c_n],
        })
        fig = go.Figure(data=[go.Pie(
            labels=df_seg["Segment"], values=df_seg["HCPs"],
            hole=0.64,
            marker=dict(
                colors=[SEG_COLORS["SEG_A"], SEG_COLORS["SEG_B"],
                          SEG_COLORS["SEG_C"]],
                line=dict(color="white", width=3),
            ),
            textinfo="label+percent",
            textfont=dict(size=13, color="white",
                            family="Inter, sans-serif"),
            hovertemplate=("<b>%{label}</b><br>%{value:,} HCPs<br>%{percent}"
                            "<extra></extra>"),
        )])
        fig.update_layout(
            annotations=[dict(
                text=f"<b>{n_total:,}</b><br><span style='font-size:12px;"
                     f"color:#64748B'>HCPs scored</span>",
                x=0.5, y=0.5, font_size=24,
                font=dict(color="#003B71", family="Inter, sans-serif"),
                showarrow=False,
            )],
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=10),
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════
    # 3. How it works — 4-step pipeline
    # ═══════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="cover-wrap">'
        '<div class="cover-section">How the engine works</div>'
        '<div class="pipeline">'
        '<div class="pipe-step">'
        '<div class="pipe-num">1</div>'
        '<div class="pipe-title">Engineer features</div>'
        '<div class="pipe-desc">'
        '10 ratio features, brand diversity, engagement totals, '
        'log-transforms — all derived from raw doctor data.'
        '</div></div>'

        '<div class="pipe-step">'
        '<div class="pipe-num">2</div>'
        '<div class="pipe-title">Two-stage XGBoost</div>'
        '<div class="pipe-desc">'
        'Model A learns P(≥B). Model B learns P(≥C) with SEG_C '
        'samples weighted 2× to prevent under-prediction.'
        '</div></div>'

        '<div class="pipe-step">'
        '<div class="pipe-num">3</div>'
        '<div class="pipe-title">Calibrated decision</div>'
        '<div class="pipe-desc">'
        'P(A) ≥ 0.70 → A · P(C) ≥ 0.30 → C · P(C) &gt; P(B) → C · '
        'otherwise B. Tuned to minimise C→A misclassifications.'
        '</div></div>'

        '<div class="pipe-step">'
        '<div class="pipe-num">4</div>'
        '<div class="pipe-title">Explain & act</div>'
        '<div class="pipe-desc">'
        'SHAP highlights drivers per HCP. Counterfactual simulation '
        'shows how many B doctors flip to C under engagement deltas.'
        '</div></div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ═══════════════════════════════════════════════════════════════
    # 4. Inside the dashboard — 7 tab teaser (with hover popups)
    # ═══════════════════════════════════════════════════════════════
    tabs_meta = [
        ("🔍", "Data Overview",
         "ATSEG distribution by segment, histograms and box plots of the "
         "key prescribing features broken down by SEG_A / SEG_B / SEG_C."),
        ("📈", "Performance & CIs",
         "Confusion matrices (counts + row-normalised), per-segment "
         "precision & recall, and the 95% confidence-interval section: "
         "mean CI widths per segment, distribution histogram, box plot "
         "by predicted class, and the most uncertain HCPs to review."),
        ("🌐", "Probability Map",
         "Interactive 3D scatter of P(A) · P(B) · P(C) for every HCP, "
         "coloured by predicted or true segment. Includes a live HCP "
         "search box that highlights one point with a hover-style tooltip."),
        ("🔬", "Doctor Explorer",
         "Per-HCP profile: ATSEG label vs model prediction, probability "
         "breakdown, per-HCP counterfactual sliders, 95% CIs, SHAP top "
         "features, prescribing profile table, and the individual radar "
         "with multiple scaling options."),
        ("🎯", "Conversion Strategy",
         "Predicted-SEG_B doctors closest to SEG_C, ranked by P(C), with "
         "the engineered-feature gaps that hold each one back and the "
         "single top actionable lever per candidate."),
        ("🔮", "Counterfactual",
         "Simulates 10 engagement deltas (+1, +2, +3, +5 visits, +1/+2 "
         "samples, combined, → SEG_C median) on the SEG_B universe. "
         "Stacked-bar scenario impact, diminishing-returns sweep, "
         "P(C) distribution shift, flipper vs stayer profile, and a "
         "downloadable list of easy-win HCPs."),
        ("📋", "Predictions Table",
         "Full sortable table of every HCP — true ATSEG, predicted "
         "segment, P(A)/P(B)/P(C) with progress-bar rendering, CI lo/hi "
         "bounds per class, max CI width column, an uncertainty filter, "
         "and a one-click CSV export."),
    ]
    tabs_html = "".join(
        f'<div class="tab-card">'
        f'<div class="tab-card-icon">{icon}</div>'
        f'<div class="tab-card-name">{name}</div>'
        f'<div class="tab-card-popup">'
        f'<span class="tab-card-popup-title">{icon}  {name}</span>'
        f'<div class="tab-card-popup-text">{desc}</div>'
        f'</div>'
        f'</div>'
        for icon, name, desc in tabs_meta
    )
    st.markdown(
        f'<div class="cover-wrap">'
        f'<div class="cover-section">Inside the dashboard</div>'
        f'<div class="tabs-preview">{tabs_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ─── Enter button (the requested arrow) ───
    # Use an on_click CALLBACK to flip session_state BEFORE the script
    # reruns. With this pattern Streamlit performs a single clean rerun
    # (cover is skipped because state is already True on next execution).
    # The previous `set state + st.rerun()` inside the click branch
    # produced a double-rerun, briefly leaving the cover content in the
    # DOM without its CSS while the dashboard re-rendered.
    def _enter_dashboard_cb():
        st.session_state.entered_dashboard = True
        # Flag picked up by the dashboard render to scroll the page to
        # the very top — otherwise the browser keeps the user's scroll
        # position from the cover (often near the Enter button at the
        # bottom of the cover page).
        st.session_state.scroll_top_after_enter = True

    st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
    bcol_l, bcol_m, bcol_r = st.columns([1, 1, 1])
    with bcol_m:
        st.button("Enter Dashboard  →", key="enter_dashboard_btn",
                   on_click=_enter_dashboard_cb,
                   use_container_width=True)

    # Footer credit
    st.markdown(
        '<div style="text-align:center;color:#94A3B8;font-size:11px;'
        'margin-top:30px;letter-spacing:0.5px;">'
        'PFIZER · COMMERCIAL ANALYTICS · Built with Streamlit & XGBoost · '
        'Random seed locked at 42'
        '</div>',
        unsafe_allow_html=True,
    )


# Render cover and stop the rest of the script from drawing.
# The on_click callback used by the "Enter Dashboard" button already
# prevents the double-rerun that previously left stale cover content
# visible — no need to wrap the cover in an st.empty() container, which
# was breaking widget rendering inside the cover.
if not st.session_state.entered_dashboard:
    _render_cover()
    st.stop()


# ─────────────────────────────────────────────────────────────────────────
# Scroll to top after entering the dashboard.
# Without this, the browser keeps the user's scroll position from the
# cover (usually at the very bottom, near the Enter Dashboard button)
# and the dashboard appears to "open partway down".
#
# IMPORTANT: This fires ONLY on the one-shot transition from cover to
# dashboard.  We do NOT use a MutationObserver — a previous version did,
# and it interfered with Streamlit's normal scroll handling whenever the
# user toggled a widget inside any tab (e.g. the Probability Map color
# radio), making the page jump around.  Three staggered timeouts are
# enough to catch async-rendered content from the cover→dashboard
# transition.
# ─────────────────────────────────────────────────────────────────────────
if st.session_state.get("scroll_top_after_enter"):
    st.session_state.scroll_top_after_enter = False
    components.html(
        """
        <script>
            (function() {
                function jumpTop() {
                    try {
                        var w = window.parent;
                        var d = w.document;
                        w.scrollTo(0, 0);
                        if (d.documentElement) d.documentElement.scrollTop = 0;
                        if (d.body) d.body.scrollTop = 0;
                        var main = d.querySelector(
                            'section[data-testid="stMain"]'
                        );
                        if (main) main.scrollTop = 0;
                        var app = d.querySelector(
                            '[data-testid="stAppViewContainer"]'
                        );
                        if (app) app.scrollTop = 0;
                    } catch (e) {}
                }
                // Three staggered attempts — covers the typical async
                // hydration of charts after the initial paint.  No
                // MutationObserver: subsequent re-renders (radio toggle,
                // tab switch, slider change) must NOT trigger scroll.
                jumpTop();
                setTimeout(jumpTop, 100);
                setTimeout(jumpTop, 500);
            })();
        </script>
        """,
        height=0,
        width=0,
    )


# ─────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# 2. Convierte tu archivo local (asegúrate de que la ruta sea correcta)
img_base64 = get_image_base64("logo3.png") 

# 3. Lo inyectas en el HTML usando f-strings
st.sidebar.markdown(
    f"""
    <div class="sb-brand">
        <div class="sb-brand-row">
            <div class="sb-brand-logo">
                <img src="data:image/png;base64,{img_base64}" width="40" style="vertical-align: middle;">
            </div>
            <div class="sb-brand-text">
                <div class="sb-brand-title">PFIZER</div>
                <div class="sb-brand-sub">HCP Segmentation</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Dataset (hard-coded)
dataset = DATASET_NAME

_df_for_card = load_data(DATASET_NAME)
_n_rows = len(_df_for_card)
_n_cols = _df_for_card.shape[1]

# Build the tooltip HTML — single-line to avoid Streamlit markdown interference
_steps_html = "".join(
    (f'<div class="ds-step"><div class="ds-step-num">{i+1}</div>'
     f'<div class="ds-step-text"><span class="ds-step-name">{name}</span>'
     f'{desc}</div></div>')
    for i, (name, desc) in enumerate(DATA_PREP_STEPS)
)


@st.dialog("📊 Dataset Preview", width="large")
def show_dataset_preview():
    df_full = load_data(DATASET_NAME)

    st.markdown(
        f"""
        <div style="margin:-12px 0 14px 0;">
            <div style="font-family:'Inter',monospace;font-size:18px;
                        font-weight:800;color:#0B1B33;letter-spacing:-0.3px;">
                📂  doctors_aggregated.csv
            </div>
            <div style="font-size:12px;color:#64748B;margin-top:4px;">
                Pfizer VELSIPITY · doctor-level aggregated dataset
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick stats
    cols = st.columns(4)
    cols[0].metric("Total HCPs", f"{len(df_full):,}")
    cols[1].metric("Columns", f"{df_full.shape[1]}")
    labeled_cnt = (df_full["ATSEG_first"] != "0").sum()
    cols[2].metric("Labeled", f"{labeled_cnt:,}",
                    f"{labeled_cnt/len(df_full)*100:.1f}%")
    seg_c_cnt = (df_full["ATSEG_first"] == "SEG_C").sum()
    cols[3].metric("SEG_C HCPs", f"{seg_c_cnt:,}")

    st.markdown(
        "<div style='margin:14px 0 6px 0;font-size:11px;text-transform:uppercase;"
        "letter-spacing:.7px;color:#003B71;font-weight:800;'>"
        "First 30 rows</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(df_full.head(30), use_container_width=True, height=380)

    st.markdown(
        "<div style='margin:18px 0 6px 0;font-size:11px;text-transform:uppercase;"
        "letter-spacing:.7px;color:#003B71;font-weight:800;'>"
        "ATSEG distribution</div>",
        unsafe_allow_html=True,
    )
    vc = (df_full["ATSEG_first"]
          .replace("0", "Unlabeled")
          .value_counts()
          .reindex(["SEG_A", "SEG_B", "SEG_C", "Unlabeled"])
          .fillna(0).astype(int))
    breakdown = pd.DataFrame({
        "Segment": vc.index,
        "Doctors": vc.values,
        "Pct (%)": (vc.values / len(df_full) * 100).round(1),
    })
    st.dataframe(breakdown, use_container_width=True, hide_index=True,
                  column_config={
                      "Doctors": st.column_config.NumberColumn(
                          "Doctors", format="%d"),
                      "Pct (%)": st.column_config.ProgressColumn(
                          "Pct (%)", format="%.1f", min_value=0, max_value=100),
                  })


sb_label("", "Dataset", color="orange")

# Clickable styled button → opens the preview popup
if st.sidebar.button(
    "doctors_aggregated.csv",
    key="ds_preview_btn",
    use_container_width=True,
    help="Click to preview the dataset in a popup",
):
    show_dataset_preview()

# Meta row with "+" hover tooltip below the button
# Built as a single-line string so Streamlit's markdown processor doesn't
# wrap pieces of the HTML in <p> tags and leave stray closing tags behind.
_tooltip_html = (
    f'<div class="sb-ds-tooltip">'
    f'<div class="ds-tooltip-title">Data preparation pipeline</div>'
    f'{_steps_html}'
    f'</div>'
)
_meta_html = (
    f'<div class="sb-ds-meta-row">'
    f'<span class="sb-ds-meta-text">{_n_rows:,} HCPs · {_n_cols} cols</span>'
    f'<div class="sb-ds-info">+{_tooltip_html}</div>'
    f'</div>'
)
st.sidebar.markdown(_meta_html, unsafe_allow_html=True)

# Single-model dashboard — the Business-Calibrated Ordinal (3d.2.py) is the
# only model in production. Picker removed.
model_pick = "ord"
mode = "Single model"

sb_divider()

# Active model summary card (replaces the old picker)
sb_label("" ,"Model", color="blue")
st.sidebar.markdown(
    '<div class="sb-current">'
    '<span class="sb-current-label">Active</span>'
    '<span class="sb-current-value"> Ordinal (v3.2)</span>'
    '</div>',
    unsafe_allow_html=True,
)
# Inject technical-spec tooltip styles (only once, scoped to this block)
st.sidebar.markdown(
    """
    <style>
    .tech-section {
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #003B71;
        font-weight: 800;
        margin: 8px 0 4px 0;
        padding-bottom: 3px;
        border-bottom: 1px solid #E2E8F0;
    }
    .tech-section:first-child { margin-top: 0; }
    .tech-row {
        display: flex; justify-content: space-between;
        align-items: center;
        padding: 2px 0;
        font-size: 10.5px;
        color: #334155;
        line-height: 1.4;
        gap: 8px;
        min-width: 0;
    }
    .tech-row > span {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .tech-row code {
        background: #F1F5F9;
        color: #0B1B33;
        font-family: 'Inter', monospace;
        font-size: 10px;
        font-weight: 700;
        padding: 1px 6px;
        border-radius: 4px;
        border: 1px solid #E2E8F0;
        line-height: 1.4;
        flex-shrink: 0;
    }
    /* 4-column grid — wider popup means every param has its own slot */
    .tech-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        column-gap: 14px;
        row-gap: 4px;
    }
    .tech-grid.two-col {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .tech-cascade {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 8px 10px;
        font-family: 'Inter', monospace;
        font-size: 10.5px;
        color: #0B1B33;
        line-height: 1.7;
        margin-top: 4px;
        display: grid;
        grid-template-columns: 1fr 1fr;
        column-gap: 14px;
        row-gap: 4px;
    }
    .tech-cascade b { color: #003B71; }
    .tech-cascade .arrow { color: #94A3B8; margin: 0 4px; }
    .sb-ds-tooltip.tech-tooltip .ds-tooltip-title {
        font-size: 12px !important;
        margin-bottom: 8px !important;
        padding-bottom: 6px !important;
    }
    /* Wide tooltip override — make sure it doesn't get clipped */
    .sb-ds-tooltip.tech-tooltip {
        max-width: 90vw;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Technical-specs tooltip — trimmed to the essentials only.
# Removed: subsample, colsample_bytree, min_child_weight, reg_alpha/λ,
# tree_method, class_weight  (regularisation / engine-level knobs that
# don't change the business interpretation of the model).
_model_tooltip_html = (
    '<div class="sb-ds-tooltip tech-tooltip" '
    'style="width:360px;padding:12px 14px;">'
    '<div class="ds-tooltip-title">Model Technical Specs</div>'

    # Architecture (2x2)
    '<div class="tech-section">Architecture</div>'
    '<div class="tech-grid two-col">'
    '<div class="tech-row"><span>Type</span><code>Ordinal XGB</code></div>'
    '<div class="tech-row"><span>Stages</span><code>2 binary</code></div>'
    '<div class="tech-row"><span>Stage 1</span><code>P(≥B)</code></div>'
    '<div class="tech-row"><span>Stage 2</span><code>P(≥C)</code></div>'
    '</div>'

    # Hyperparameters — only the ones that drive the business behaviour
    '<div class="tech-section">Hyperparameters</div>'
    '<div class="tech-grid two-col">'
    '<div class="tech-row"><span>n_est.</span><code>800</code></div>'
    '<div class="tech-row"><span>depth</span><code>5</code></div>'
    '<div class="tech-row"><span>lr</span><code>0.04</code></div>'
    '<div class="tech-row"><span>SEG_C wt</span><code>×2.0</code></div>'
    '</div>'

    # Decision cascade
    '<div class="tech-section">Decision Cascade</div>'
    '<div class="tech-cascade">'
    '<div>if P(A) ≥ <b>0.70</b><span class="arrow">→</span> A</div>'
    '<div>elif P(C) ≥ <b>0.30</b><span class="arrow">→</span> C</div>'
    '<div>elif P(C) &gt; P(B)<span class="arrow">→</span> C</div>'
    '<div>else<span class="arrow">→</span> B</div>'
    '</div>'

    # Training & evaluation
    '<div class="tech-section">Training & evaluation</div>'
    '<div class="tech-grid two-col">'
    '<div class="tech-row"><span>CV</span><code>5-fold</code></div>'
    '<div class="tech-row"><span>CIs</span><code>μ±1.96σ</code></div>'
    '<div class="tech-row"><span>XAI</span><code>SHAP P≥C</code></div>'
    '<div class="tech-row"><span>seed</span><code>42</code></div>'
    '</div>'

    '</div>'
)
_model_meta_html = (
    '<div class="sb-ds-meta-row">'
    '<span class="sb-ds-meta-text">Business Calibrated · SHAP</span>'
    '<div class="sb-ds-info">+' + _model_tooltip_html + '</div>'
    '</div>'
)
st.sidebar.markdown(_model_meta_html, unsafe_allow_html=True)

sb_divider()

# ─────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-content">
            <div class="hero-eyebrow">PFIZER · VELSIPITY HCP SEGMENTATION</div>
            <div class="hero-title">Prescriber Probability Engine</div>
            <p>XGBoost-driven segmentation predicting SEG_A / SEG_B / SEG_C
            for every HCP using the business-calibrated Ordinal model
            (capstone v3.2). Decision cascade is tuned to minimise the
            catastrophic SEG_C → SEG_A misclassification.</p>
            <div class="hero-divider"></div>
            <span class="hero-pill">Dataset <b>doctors_aggregated.csv</b></span>
            <span class="hero-pill">Model <b>Ordinal v3.2</b></span>
            <span class="hero-pill">P(A)≥<b>{THR_A}</b> · P(C)≥<b>{THR_C}</b> · dominance rule</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────
# Train pipeline (cached).
# The branded loader only shows on the FIRST dashboard load.  Showing it
# on every rerun caused a layout shift (loader inserted, then removed)
# that scrolled the page whenever a widget like the Probability Map's
# color radio was clicked.
# ─────────────────────────────────────────────────────────────────────────
if not st.session_state.get("dashboard_first_render_done"):
    _pl_loader = brand_loader(
        "Loading dashboard",
        "Reading cached model · scoring HCPs · preparing visualisations",
    )
    R = train_pipeline(dataset)
    _pl_loader.empty()
    st.session_state.dashboard_first_render_done = True
else:
    R = train_pipeline(dataset)   # cached → instant, no loader needed

# (Context strip removed — the hero already shows the active model + dataset)


# ============================================================
# Single-model dashboard (Ordinal v3.2 from 3d.2.py)
# ============================================================
if True:
    m = R["metrics"][model_pick]

    # KPI status helpers (used inside Data Overview)
    acc_s, acc_l = accuracy_status(m["accuracy"])
    rec_s, rec_l = recall_c_status(m["recall_C"])
    closs_s, closs_l = c_loss_status(m["c_lost_pct"])


    tabs = st.tabs([
        "Data Overview",
        "Model Performance",
        "Probability Map",
        "Doctor Explorer",
        "Conversion Strategy",
        "Counterfactual",
        "Predictions Table",
    ])

    # ── Data Overview ──
    with tabs[0]:
        section("ATSEG Segmentation Overview",
                "Model performance metrics and HCP distribution across segments",
                icon="")



        # ── Row 1: Model performance KPIs ──
        cols = st.columns(5)
        kpi_card(cols[0], "HCPs Analyzed", f"{len(R['ids']):,}",
                 helper=f"{R['n_labeled']:,} labeled · {R['n_unlabeled']:,} unlabeled",
                 style="neutral", icon="",
                 status="info", status_label="Loaded")
        kpi_card(cols[1], "Accuracy", f"{m['accuracy']*100:.1f}%",
                 helper="Overall correctness",
                 style="good", icon="",
                 status=acc_s, status_label=acc_l)
        kpi_card(cols[2], "Balanced Accuracy", f"{m['balanced_accuracy']*100:.1f}%",
                 helper="Class-balanced score",
                 style="accent", icon="",
                 status=acc_s, status_label=acc_l)
        kpi_card(cols[3], "Recall (SEG_C)", f"{m['recall_C']*100:.1f}%",
                 helper="C captured by the model",
                 style="warn", icon="",
                 status=rec_s, status_label=rec_l)
        kpi_card(cols[4], "SEG_C → SEG_A Loss", f"{m['c_lost_pct']*100:.1f}%",
                 helper="Catastrophic mis-classifications",
                 style="danger", icon="",
                 status=closs_s, status_label=closs_l)

        # ── Row 2: Data distribution KPIs ──
        with_atseg = R["n_labeled"]
        no_atseg = R["n_unlabeled"]
        coverage = with_atseg / (with_atseg + no_atseg) * 100
        cnt_a = (R["y_true"] == "SEG_A").sum()
        cnt_b = (R["y_true"] == "SEG_B").sum()
        cnt_c = (R["y_true"] == "SEG_C").sum()

        cols = st.columns(3)
        kpi_card(cols[0], "With ATSEG", f"{with_atseg:,}",
                 helper=f"{coverage:.1f}% coverage",
                 style="good compact", icon="",
                 status="ok", status_label="Labeled")
        kpi_card(cols[1], "SEG_C HCPs",
                 f"{cnt_c:,}",
                 helper="High-value targets",
                 style="danger compact", icon="",
                 status="info", status_label="Priority")
        kpi_card(cols[2], "Unlabeled", f"{no_atseg:,}",
                 helper=f"{100-coverage:.1f}% to score",
                 style="warn compact", icon="",
                 status="info", status_label="Scored")

        st.markdown("&nbsp;")

        col_a, col_b = st.columns([3, 2])
        vc = pd.Series(R["y_true"]).replace({"": "Unlabeled", "nan": "Unlabeled"})
        vc = vc.value_counts().reindex(["SEG_A", "SEG_B", "SEG_C", "Unlabeled"]).fillna(0)
        df_atseg = vc.reset_index()
        df_atseg.columns = ["Segment", "Doctors"]
        df_atseg["Segment"] = df_atseg["Segment"].replace("Unlabeled", "Unlab.")
        df_atseg["Pct"] = df_atseg["Doctors"] / df_atseg["Doctors"].sum() * 100

        with col_a:
            fig = px.bar(
                df_atseg, x="Segment", y="Doctors",
                color="Segment", color_discrete_map=SEG_COLORS,
                text=df_atseg.apply(
                    lambda r: f"{int(r['Doctors']):,}<br>({r['Pct']:.1f}%)",
                    axis=1),
            )
            fig.update_traces(textposition="outside", marker_line_width=0, opacity=0.92)
            _style_fig(fig, height=370, title="HCPs per ATSEG Segment")
            fig.update_layout(showlegend=False, xaxis_title="",
                              yaxis_title="Number of HCPs")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            seg_order = list(df_atseg["Segment"])
            # Dark text on light slices, white on dark slices
            txt_colors = ["#FFFFFF" if s != "Unlab." else "#334155" for s in seg_order]
            fig = go.Figure(data=[go.Pie(
                labels=df_atseg["Segment"], values=df_atseg["Doctors"],
                hole=0.5,
                marker=dict(colors=[SEG_COLORS.get(s, "#94A3B8") for s in seg_order],
                             line=dict(color="white", width=2.5)),
                textinfo="label+percent", textposition="inside",
                insidetextorientation="horizontal",
                textfont=dict(size=13, family="Outfit, sans-serif"),
                hovertemplate="<b>%{label}</b><br>%{value:,} HCPs<br>%{percent}<extra></extra>",
            )])
            # Set per-slice text colors
            fig.update_traces(textfont_color=txt_colors)
            fig.update_layout(
                annotations=[dict(text=f"<b>{int(df_atseg['Doctors'].sum()):,}</b>"
                                       f"<br><span style='font-size:11px;color:#64748B'>HCPs</span>",
                                    x=0.5, y=0.5, font_size=20,
                                    font=dict(color="#003B71"), showarrow=False)],
                showlegend=False,
                margin=dict(l=8, r=8, t=64, b=16),
            )
            _style_fig(fig, height=380, title="ATSEG Share")
            st.plotly_chart(fig, use_container_width=True)

        section("Key Feature Distribution by ATSEG",
                "Compare how key prescribing features differ across segments",
                icon="")

        if R["available_features"]:
            feat_options = [k for k, _, _ in R["available_features"]]
            feat_pick = st.selectbox("Feature to inspect", feat_options,
                                      key="explore_feat")

            # --- VISTA 1: ORIGINAL (CON UNLABELED) ---
            df_dist = R["df"][[feat_pick, "ATSEG_first"]].copy()
            df_dist["ATSEG_first"] = df_dist["ATSEG_first"].fillna("Unlabeled").replace("Unlabeled", "Unlab.")

            col_h, col_b2 = st.columns(2)
            with col_h:
                fig = px.histogram(
                    df_dist, x=feat_pick, color="ATSEG_first",
                    category_orders={"ATSEG_first":
                                     ["SEG_A", "SEG_B", "SEG_C", "Unlab."]},
                    color_discrete_map=SEG_COLORS, barmode="overlay",
                    opacity=0.6, nbins=50,
                )
                _style_fig(fig, height=400, title=f"Histogram — {feat_pick}")
                fig.update_layout(legend_title_text="",
                                   yaxis_title="HCP Count",
                                   margin=dict(l=70, r=24, t=72, b=100),
                                   legend=dict(orientation="h", yanchor="top",
                                               y=-0.25, xanchor="center", x=0.5,
                                               font=dict(size=12)),
                                   xaxis=dict(tickfont=dict(size=13)),
                                   yaxis=dict(tickfont=dict(size=13),
                                              title=dict(font=dict(size=14))))
                st.plotly_chart(fig, use_container_width=True)

            with col_b2:
                fig = px.box(
                    df_dist, x="ATSEG_first", y=feat_pick,
                    color="ATSEG_first",
                    category_orders={"ATSEG_first":
                                     ["SEG_A", "SEG_B", "SEG_C", "Unlab."]},
                    color_discrete_map=SEG_COLORS, points=False,
                )
                _style_fig(fig, height=360,
                           title=f"Box Plot — {feat_pick} by ATSEG")
                fig.update_layout(showlegend=False, xaxis_title="",
                                  margin=dict(l=70, r=40, t=72, b=50),
                                  xaxis=dict(tickfont=dict(size=14)),
                                  yaxis=dict(tickfont=dict(size=13),
                                             title=dict(font=dict(size=14))))
                st.plotly_chart(fig, use_container_width=True)


    # ── Model Performance (confusion matrix + bars) ──
    with tabs[1]:
        section(f"{MODEL_LABELS[model_pick]} — Performance Detail",
                "Confusion matrix + per-segment metrics computed on OOF predictions",
                icon="")

        cm = m["confusion_matrix"]
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        cm_df = pd.DataFrame(
            cm,
            index=[f"True {l}" for l in VALID_LABELS],
            columns=[f"Pred {l}" for l in VALID_LABELS],
        )

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.imshow(cm_df, text_auto=True, aspect="auto",
                             color_continuous_scale="Blues")
            fig.update_traces(textfont=dict(size=15))
            _style_fig(fig, height=380, title="Confusion Matrix (counts)")
            fig.update_layout(margin=dict(l=80, r=24, t=72, b=60),
                              xaxis=dict(tickfont=dict(size=13)),
                              yaxis=dict(tickfont=dict(size=13)))
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            cm_pct = pd.DataFrame(
                cm_norm,
                index=[f"True {l}" for l in VALID_LABELS],
                columns=[f"Pred {l}" for l in VALID_LABELS],
            )
            fig = px.imshow(cm_pct, text_auto=".1%", aspect="auto",
                             color_continuous_scale="Blues",
                             zmin=0, zmax=1)
            fig.update_traces(textfont=dict(size=15))
            _style_fig(fig, height=380, title="Confusion Matrix (row-normalized)")
            fig.update_layout(margin=dict(l=80, r=24, t=72, b=60),
                              xaxis=dict(tickfont=dict(size=13)),
                              yaxis=dict(tickfont=dict(size=13)))
            st.plotly_chart(fig, use_container_width=True)

        # Per-segment performance bar
        recall_a = float(cm[0,0] / max(cm[0].sum(), 1))
        recall_b = float(cm[1,1] / max(cm[1].sum(), 1))
        recall_c = float(cm[2,2] / max(cm[2].sum(), 1))
        prec_a = float(cm[0,0] / max(cm[:,0].sum(), 1))
        prec_b = float(cm[1,1] / max(cm[:,1].sum(), 1))
        prec_c = float(cm[2,2] / max(cm[:,2].sum(), 1))

        df_perf = pd.DataFrame({
            "Segment": ["SEG_A", "SEG_B", "SEG_C"] * 2,
            "Metric": ["Recall"]*3 + ["Precision"]*3,
            "Value": [recall_a, recall_b, recall_c, prec_a, prec_b, prec_c],
        })

        fig = px.bar(
            df_perf, x="Segment", y="Value", color="Metric",
            barmode="group", text=df_perf["Value"].apply(lambda v: f"{v:.1%}"),
            color_discrete_sequence=[PFIZER_BLUE, PFIZER_ORANGE],
        )
        fig.update_traces(textposition="outside")
        _style_fig(fig, height=360, title="Precision & Recall per Segment")
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title="", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        # ── Population-level Confidence Intervals (5-fold CV) ──
        section("Prediction confidence intervals (95% CI)",
                "Width = upper bound − lower bound across 5 CV folds for each "
                "labeled HCP. Narrow bands ⇒ stable predictions; wide bands "
                "⇒ noisy predictions worth manual review.",
                icon="")

        # CI bounds are only meaningful for labeled HCPs (computed from CV folds)
        lab_mask = R["is_labeled"]
        if lab_mask.sum() > 0:
            ci_lo_lab = R["ci_lo"][lab_mask]
            ci_hi_lab = R["ci_hi"][lab_mask]
            widths = ci_hi_lab - ci_lo_lab           # shape (n_labeled, 3)
            mean_widths = widths.mean(axis=0)        # per segment
            max_widths  = widths.max(axis=0)
            pred_lab    = R["full"]["ord"]["pred"][lab_mask]

            # KPI strip
            ks = st.columns(4)
            kpi_card(ks[0], "Mean CI width — P(A)",
                     f"±{mean_widths[0]/2*100:.1f}pp",
                     helper="Avg uncertainty on SEG_A probability",
                     style="accent", icon="",
                     status="info", status_label="Stability")
            kpi_card(ks[1], "Mean CI width — P(B)",
                     f"±{mean_widths[1]/2*100:.1f}pp",
                     helper="Avg uncertainty on SEG_B probability",
                     style="warn", icon="",
                     status="info", status_label="Stability")
            kpi_card(ks[2], "Mean CI width — P(C)",
                     f"±{mean_widths[2]/2*100:.1f}pp",
                     helper="Avg uncertainty on SEG_C probability",
                     style="danger", icon="",
                     status="info", status_label="Stability")
            # share of HCPs with stable predictions (max CI width below 15pp)
            stable_mask = widths.max(axis=1) < 0.15
            stable_pct  = float(stable_mask.mean() * 100)
            kpi_card(ks[3], "Stable predictions", f"{stable_pct:.1f}%",
                     helper="Max CI width < 15pp across A/B/C",
                     style="good", icon="",
                     status=("ok" if stable_pct > 75
                              else "fair" if stable_pct > 50 else "poor"),
                     status_label="Stable")

            # ── Histogram of CI widths per segment ──
            df_w = pd.DataFrame({
                "Width": np.concatenate([widths[:, 0], widths[:, 1], widths[:, 2]]),
                "Segment": (["P(A)"] * len(widths)
                             + ["P(B)"] * len(widths)
                             + ["P(C)"] * len(widths)),
            })
            fig_w = px.histogram(
                df_w, x="Width", color="Segment",
                color_discrete_map={"P(A)": SEG_COLORS["SEG_A"],
                                       "P(B)": SEG_COLORS["SEG_B"],
                                       "P(C)": SEG_COLORS["SEG_C"]},
                barmode="overlay", opacity=0.55, nbins=40,
            )
            _style_fig(fig_w, height=380,
                        title="Distribution of 95% CI widths")
            fig_w.update_layout(
                xaxis=dict(title="CI width (upper − lower)",
                            tickformat=".0%"),
                yaxis_title="Labeled HCP count",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                              xanchor="right", x=1),
            )
            st.plotly_chart(fig_w, use_container_width=True)

            # ── Box plot: CI width per predicted segment ──
            col_box, col_top = st.columns([3, 2])
            with col_box:
                # Compute the CI width corresponding to each HCP's predicted class
                idx_class = {"SEG_A": 0, "SEG_B": 1, "SEG_C": 2}
                pred_class_idx = np.array([idx_class.get(p, 0) for p in pred_lab])
                width_pred = widths[np.arange(len(pred_lab)), pred_class_idx]
                df_box = pd.DataFrame({
                    "Predicted segment": pred_lab,
                    "CI width": width_pred,
                })
                fig_box = px.box(
                    df_box, x="Predicted segment", y="CI width",
                    color="Predicted segment",
                    color_discrete_map=SEG_COLORS,
                    category_orders={"Predicted segment":
                                      ["SEG_A", "SEG_B", "SEG_C"]},
                    points=False,
                )
                _style_fig(fig_box, height=380,
                            title="CI width on the predicted segment")
                fig_box.update_layout(
                    yaxis=dict(tickformat=".0%"),
                    showlegend=False, xaxis_title="",
                )
                st.plotly_chart(fig_box, use_container_width=True)

            with col_top:
                # Most uncertain HCPs — top 12 by max CI width
                max_w_per_hcp = widths.max(axis=1)
                ids_lab   = R["ids"][lab_mask]
                y_true_lab = R["y_true"][lab_mask]
                order = np.argsort(-max_w_per_hcp)[:12]
                df_unc = pd.DataFrame({
                    "HCP_ID":   ids_lab[order],
                    "True":     y_true_lab[order],
                    "Predicted": pred_lab[order],
                    "Max CI width": max_w_per_hcp[order],
                }).round(3)
                st.markdown(
                    "<div style='font-size:13px;color:#0F172A;font-weight:700;"
                    "padding:6px 0 2px 2px;'>Most uncertain labeled HCPs</div>"
                    "<div style='font-size:11px;color:#64748B;"
                    "padding:0 0 8px 2px;'>"
                    "Widest CI across A / B / C — candidates for human review."
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    df_unc, use_container_width=True, hide_index=True,
                    height=360,
                    column_config={
                        "Max CI width": st.column_config.ProgressColumn(
                            "Max CI width", format="%.2f",
                            min_value=0, max_value=1),
                    },
                )

            st.markdown(
                '<div class="info-panel">'
                '<div class="info-panel-icon">💡</div>'
                '<div class="info-panel-content">'
                '<b>CIs are computed only for labeled HCPs.</b> Each labeled '
                'doctor is scored by all 5 cross-validation folds; the CI is '
                '<code>mean ± 1.96 × std</code> across folds (clipped to '
                '[0, 1]). Unlabeled doctors are scored by the final model on '
                'all labeled data — they get a point prediction without CI. '
                'Search any labeled HCP in the Doctor Explorer to see its '
                'individual CI bars.'
                '</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No labeled HCPs available — CIs require ground-truth "
                     "labels for the cross-validation procedure.")

# ── Probability Map (3D) ──
    with tabs[2]:
        section("Probability Map",
                f"Each HCP plotted by ({MODEL_LABELS_SHORT[model_pick]}) "
                "P(A) · P(B) · P(C). Color = predicted segment",
                icon="")

        # ── Fragment isolation ──────────────────────────────────────
        # Wrapping the entire probability-map UI in @st.fragment makes
        # the radio / search-box interactions re-run ONLY this block —
        # NOT the whole script.  This fixes the bug where clicking the
        # "Color points by" radio caused the active-tab state to reset
        # and the page jumped to the bottom of Data Overview.
        @st.fragment
        def _probability_map_fragment():
            # Build full data frame (all HCPs)
            df_full_map = pd.DataFrame({
                "HCP_ID": R["ids"],
                "P(A)": R["full"][model_pick]["P_A"],
                "P(B)": R["full"][model_pick]["P_B"],
                "P(C)": R["full"][model_pick]["P_C"],
                "Predicted": R["full"][model_pick]["pred"],
                "True ATSEG": np.where(R["is_labeled"], R["y_true"], "Unlabeled"),
            })

            # Controls row: color toggle + search input
            col_clr, col_search = st.columns([2, 3])
            with col_clr:
                # 🔥 CORRECCIÓN: Añadimos la opción filtrada en el componente de Radio
                color_by = st.radio("Color points by",
                                     ["Predicted Segment", "True ATSEG", "True Segment (Labeled Only)"],
                                     horizontal=True, key="map_color_by")
            with col_search:
                highlight_q = st.text_input(
                    "🔍  Highlight HCP",
                    placeholder="Enter HCP ID (NUEVO_ID) to highlight in the map",
                    key="map_highlight_id",
                    label_visibility="collapsed",
                )
        
            # 🔥 CORRECCIÓN: Si eligen filtrar, removemos los Unlabeled antes de graficar todo el mapa
            if color_by == "True Segment (Labeled Only)":
                df_full_map = df_full_map[df_full_map["True ATSEG"] != "Unlabeled"].reset_index(drop=True)
                color_col = "True ATSEG"
            else:
                color_col = "Predicted" if color_by == "Predicted Segment" else "True ATSEG"

            # Resolve highlight
            highlight_row = None
            if highlight_q.strip():
                # Buscamos el ID en el dataframe actual disponible
                matches = df_full_map[df_full_map["HCP_ID"] == highlight_q.strip()]
                if len(matches) == 0:
                    st.warning(f"No HCP found with ID `{highlight_q.strip()}` en esta vista.")
                else:
                    # Extraemos la fila directamente como un diccionario
                    highlight_row = matches.iloc[0].to_dict()

            # Subsample background points for performance, always keep highlight
            N_MAX = 5000
            if len(df_full_map) > N_MAX:
                df_plot = df_full_map.sample(N_MAX, random_state=SEED)
                if highlight_row is not None and (
                    highlight_row["HCP_ID"] not in df_plot["HCP_ID"].values
                ):
                    df_plot = pd.concat([df_plot, df_full_map[
                        df_full_map["HCP_ID"] == highlight_row["HCP_ID"]
                    ]], ignore_index=True)
            else:
                df_plot = df_full_map

            if highlight_row is not None:
                # Background: every point in gray so the highlighted HCP stands out
                df_bg = df_plot[df_plot["HCP_ID"] != highlight_row["HCP_ID"]]
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                    x=df_bg["P(A)"], y=df_bg["P(B)"], z=df_bg["P(C)"],
                    mode="markers",
                    marker=dict(size=3.0, color="#78909C", opacity=0.75,
                                 line=dict(width=0)),
                    name="Other HCPs",
                    hovertemplate=("HCP %{customdata[0]}<br>"
                                    "Pred %{customdata[1]} · True %{customdata[2]}<br>"
                                    "P(A) %{x:.2f} · P(B) %{y:.2f} · P(C) %{z:.2f}"
                                    "<extra></extra>"),
                    customdata=df_bg[["HCP_ID", "Predicted", "True ATSEG"]].values,
                    showlegend=True,
                ))
                # Highlighted HCP keeps its segment color and looks like a normal
                # point — the "hover" tooltip is drawn as a scene annotation.
                hi_seg = (highlight_row[color_col]
                           if highlight_row[color_col] in SEG_COLORS
                           else highlight_row["Predicted"])
                seg_color = SEG_COLORS.get(hi_seg, "#0072CE")
                fig.add_trace(go.Scatter3d(
                    x=[highlight_row["P(A)"]],
                    y=[highlight_row["P(B)"]],
                    z=[highlight_row["P(C)"]],
                    mode="markers",
                    marker=dict(
                        size=7, color=seg_color, opacity=1.0,
                        line=dict(color="white", width=2),
                    ),
                    name=f"HCP {highlight_row['HCP_ID']}",
                    hovertemplate=(
                        f"<b>HCP {highlight_row['HCP_ID']}</b><br>"
                        f"Predicted: {highlight_row['Predicted']}<br>"
                        f"True ATSEG: {highlight_row['True ATSEG']}<br>"
                        f"P(A): {highlight_row['P(A)']:.3f}<br>"
                        f"P(B): {highlight_row['P(B)']:.3f}<br>"
                        f"P(C): {highlight_row['P(C)']:.3f}"
                        "<extra></extra>"
                    ),
                ))

                # Compact tooltip — formatted to avoid Plotly's bounding box calculation bugs with spaces
                tooltip_text = (
                    f"<b>HCP {highlight_row['HCP_ID']}</b><br>"
                    f"Pred: <b>{highlight_row['Predicted']}</b> | True: <b>{highlight_row['True ATSEG']}</b><br>"
                    f"P(A): <b>{highlight_row['P(A)']:.2f}</b> | P(B): <b>{highlight_row['P(B)']:.2f}</b> | P(C): <b>{highlight_row['P(C)']:.2f}</b>"
                )

                # Build the scene config — keep axis titles in the xaxis/yaxis/zaxis
                # dicts so they're not lost when the styling block runs later.
                fig.update_layout(scene=dict(
                    xaxis=dict(title=dict(text="P(A)",
                                            font=dict(color="#475569", size=12))),
                    yaxis=dict(title=dict(text="P(B)",
                                            font=dict(color="#475569", size=12))),
                    zaxis=dict(title=dict(text="P(C)",
                                            font=dict(color="#475569", size=12))),
                    annotations=[dict(
                        x=highlight_row["P(A)"],
                        y=highlight_row["P(B)"],
                        z=highlight_row["P(C)"],
                        text=tooltip_text,
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1.5,
                        arrowcolor=seg_color,
                        ax=90, ay=-90,
                        xanchor="left",
                        yanchor="middle",
                        align="left",
                        bgcolor="white",
                        bordercolor=seg_color,
                        borderwidth=2,
                        borderpad=14,
                        width=260,
                        height=85,
                        opacity=1.0,
                        font=dict(size=12, color="#0B1B33",
                                    family="Inter, sans-serif"),
                    )],
                ))
            else:
                # Normal coloured map by segment
                fig = px.scatter_3d(
                    df_plot, x="P(A)", y="P(B)", z="P(C)",
                    color=color_col,
                    color_discrete_map=SEG_COLORS,
                    opacity=0.6,
                    hover_data={"HCP_ID": True, "Predicted": True,
                                  "True ATSEG": True,
                                  "P(A)": ":.2f", "P(B)": ":.2f", "P(C)": ":.2f"},
                )
                fig.update_traces(marker=dict(size=2.6, line=dict(width=0)))

            _style_fig(fig, height=780)
        
            # Add non-breaking spaces to trace names to force Plotly to calculate
            # a wider SVG clip-path, preventing the last letter from being cut off.
            fig.for_each_trace(lambda t: t.update(name=t.name + "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;") if t.name else None)

            # Style the 3D scene without wiping the axis titles set above
            axis_title_font = dict(color="#475569", size=13,
                                    family="Inter, sans-serif")
            axis_tick_font  = dict(color="#475569", size=10,
                                    family="Inter, sans-serif")
            fig.update_layout(
                # Explicit empty title — prevents the spurious "undefined" string
                title=dict(text=""),
                legend_title_text=color_col,
                # Generous right margin so the legend doesn't get clipped
                margin=dict(l=10, r=30, t=20, b=10),
                # Move the legend INSIDE the chart (top-left of the scene)
                # to prevent it from being clipped by Plotly's SVG boundaries
                legend=dict(
                    x=0.02, y=0.98,
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.85)",
                    borderwidth=0,
                    font=dict(size=12, color="#0B1B33",
                                family="Inter, sans-serif"),
                    itemsizing="constant",
                ),
                scene=dict(
                    xaxis=dict(
                        title=dict(text="P(A)", font=axis_title_font),
                        tickfont=axis_tick_font,
                        backgroundcolor="rgba(244,247,250,0.6)",
                        gridcolor="rgba(0,0,0,0.08)",
                        showbackground=True, zeroline=False,
                    ),
                    yaxis=dict(
                        title=dict(text="P(B)", font=axis_title_font),
                        tickfont=axis_tick_font,
                        backgroundcolor="rgba(244,247,250,0.6)",
                        gridcolor="rgba(0,0,0,0.08)",
                        showbackground=True, zeroline=False,
                    ),
                    zaxis=dict(
                        title=dict(text="P(C)", font=axis_title_font),
                        tickfont=axis_tick_font,
                        backgroundcolor="rgba(244,247,250,0.6)",
                        gridcolor="rgba(0,0,0,0.08)",
                        showbackground=True, zeroline=False,
                    ),
                    aspectmode="cube",
                    # Slightly pulled-back camera so the simplex isn't clipped
                    camera=dict(eye=dict(x=1.6, y=1.6, z=1.4)),
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

            # If a highlight is active, show a small detail card below
            if highlight_row is not None:
                seg_class = (highlight_row["True ATSEG"]
                              if highlight_row["True ATSEG"] in VALID_LABELS
                              else "Unlabeled")
                st.markdown(
                    f"""
                    <div class="info-panel">
                        <div class="info-panel-icon"></div>
                        <div class="info-panel-content">
                            <b>HCP {highlight_row['HCP_ID']}</b> highlighted in gold.
                            Predicted <b>{highlight_row['Predicted']}</b>
                            (P(A)={highlight_row['P(A)']:.2f},
                            P(B)={highlight_row['P(B)']:.2f},
                            P(C)={highlight_row['P(C)']:.2f}).
                            Ground truth ATSEG:
                            <span class="seg-pill {seg_class}" style="margin-left:6px;">
                                {highlight_row['True ATSEG']}
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        _probability_map_fragment()

    # ── Doctor Explorer ──
    with tabs[3]:
        section("Doctor Explorer",
                "Search any HCP to inspect probabilities and prescribing profile",
                icon="")

        col_s, col_help = st.columns([3, 2])
        with col_s:
            search_q = st.text_input(
                "Search by HCP ID (NUEVO_ID)",
                placeholder="Enter HCP ID (e.g. 100012345)",
                label_visibility="collapsed",
            )
        with col_help:
            st.markdown(
                "<div style='font-size:11px;color:#64748B;padding:10px 6px;"
                "text-align:right;'>"
                f"<b style='color:#0F172A'>{len(R['ids']):,}</b> HCPs available "
                "· search by exact NUEVO_ID</div>",
                unsafe_allow_html=True,
            )

        if search_q.strip():
            matches = np.where(R["ids"] == search_q.strip())[0]
            if len(matches) == 0:
                st.markdown(
                    f"""
                    <div class="empty-state">
                        <div class="empty-state-icon"></div>
                        <div class="empty-state-title">No HCP found</div>
                        <div class="empty-state-text">
                            No record with NUEVO_ID
                            <b style="color:#0F172A;">{search_q.strip()}</b>.
                            Double-check the ID and try again.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                idx = matches[0]
                hcp_id = R["ids"][idx]
                true_seg = R["y_true"][idx] if R["is_labeled"][idx] else "Unlabeled"
                true_class = true_seg if true_seg in VALID_LABELS else "Unlabeled"

                pa_o = float(R["full"]["ord"]["P_A"][idx])
                pb_o = float(R["full"]["ord"]["P_B"][idx])
                pc_o = float(R["full"]["ord"]["P_C"][idx])
                pred_o = R["full"]["ord"]["pred"][idx]
                max_o = max(pa_o, pb_o, pc_o)
                is_labeled_hcp = R["is_labeled"][idx]

                # ── Profile header card ──
                profile_sub = ("Labeled — ground truth available"
                                if is_labeled_hcp
                                else "No ATSEG label — model prediction only")
                st.markdown(
                    f"""
                    <div class="doctor-card">
                        <div style="display:flex;justify-content:space-between;
                                    align-items:center;flex-wrap:wrap;gap:14px;">
                            <div>
                                <div style="font-size:11px;color:#64748B;letter-spacing:.4px;
                                            text-transform:uppercase;font-weight:700;">HCP Identifier</div>
                                <div class="doctor-id">{hcp_id}</div>
                                <div style="font-size:12px;color:#64748B;margin-top:4px;">
                                    {profile_sub}
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ── Segment cards ──
                # If labeled  → 2 cards side by side: Current ATSEG | Predicted
                # If unlabeled → 1 card centered:                     Predicted
                section("Segment", icon="")

                pred_class = pred_o if pred_o in VALID_LABELS else ""

                def segment_card(title, value, value_class, footer_html=""):
                    return (
                        f'<div class="verdict-card">'
                        f'<div class="verdict-card-header">'
                        f'<span class="verdict-model-name">{title}</span>'
                        f'</div>'
                        f'<div style="font-size:10px;color:#64748B;'
                        f'text-transform:uppercase;font-weight:700;'
                        f'letter-spacing:.5px;">Segment</div>'
                        f'<div class="verdict-prediction {value_class}">{value}</div>'
                        f'{footer_html}'
                        f'</div>'
                    )

                if is_labeled_hcp:
                    # Match badge for the predicted card
                    if pred_o == true_class:
                        match_badge = (
                            '<div class="verdict-confidence">'
                            '<span class="verdict-correct">✓ Matches ATSEG</span>'
                            '</div>'
                        )
                    else:
                        match_badge = (
                            '<div class="verdict-confidence">'
                            '<span class="verdict-wrong">✗ Differs from ATSEG</span>'
                            '</div>'
                        )
                    pred_footer = (
                        f'<div class="verdict-confidence">Probability '
                        f'<b style="color:#0F172A">{max_o*100:.1f}%</b></div>'
                        f'{match_badge}'
                    )
                    cur_card = segment_card("📋 Current ATSEG",
                                              true_seg, true_class,
                                              footer_html=(
                                                  '<div class="verdict-confidence">'
                                                  'Ground truth label'
                                                  '</div>'))
                    pred_card = segment_card("🎯 Predicted Segment",
                                              pred_o, pred_class,
                                              footer_html=pred_footer)
                    col_l, col_r = st.columns(2)
                    with col_l:
                        st.markdown(cur_card, unsafe_allow_html=True)
                    with col_r:
                        st.markdown(pred_card, unsafe_allow_html=True)
                else:
                    pred_footer = (
                        f'<div class="verdict-confidence">Probability '
                        f'<b style="color:#0F172A">{max_o*100:.1f}%</b></div>'
                        f'<div class="verdict-confidence">'
                        f'<span class="verdict-na">No ATSEG label</span>'
                        f'</div>'
                    )
                    pred_card = segment_card("🎯 Predicted Segment",
                                              pred_o, pred_class,
                                              footer_html=pred_footer)
                    # Center the single card with surrounding empty columns
                    cl, cc, cr = st.columns([1, 2, 1])
                    with cc:
                        st.markdown(pred_card, unsafe_allow_html=True)

                # ── Stacked probability bar (Ordinal only) ──
                def stacked_bar(label, pa, pb, pc):
                    def w(v): return max(v * 100, 0)
                    label_a = "SEG_A" if pa > 0.15 else ""
                    label_b = "SEG_B" if pb > 0.15 else ""
                    label_c = "SEG_C" if pc > 0.15 else ""
                    return (
                        f'<div class="prob-bar-row">'
                        f'<div class="prob-bar-header">'
                        f'<span class="prob-bar-label">{label}</span>'
                        f'<span style="font-size:11px;color:#64748B;">'
                        f'A {pa:.0%} · B {pb:.0%} · C {pc:.0%}</span>'
                        f'</div>'
                        f'<div class="prob-bar-track">'
                        f'<div class="prob-seg-fill A" style="width:{w(pa)}%;">'
                        f'{label_a}</div>'
                        f'<div class="prob-seg-fill B" style="width:{w(pb)}%;">'
                        f'{label_b}</div>'
                        f'<div class="prob-seg-fill C" style="width:{w(pc)}%;">'
                        f'{label_c}</div>'
                        f'</div>'
                        f'</div>'
                    )

                breakdown_html = (
                    '<div class="doctor-card" style="padding:20px 24px;">'
                    '<div style="font-size:13px;color:#0F172A;font-weight:700;'
                    'margin-bottom:10px;">Probability breakdown</div>'
                    + stacked_bar("🎯  Ordinal", pa_o, pb_o, pc_o)
                    + '</div>'
                )
                st.markdown(breakdown_html, unsafe_allow_html=True)

                # ── Per-HCP counterfactual ── (only meaningful for SEG_B doctors)
                if pred_o == "SEG_B":
                    section("Counterfactual — what if we engage more?",
                            "Pull the sliders to simulate extra rep visits "
                            "or samples; the Ordinal model re-scores this "
                            "HCP in real time.",
                            icon="")
                    c_left, c_right = st.columns([2, 3])
                    with c_left:
                        cf_visits = st.slider(
                            "Add visits (DETAILS_sum)", 0, 10, 1,
                            key=f"cf_visits_{hcp_id}")
                        cf_samples = st.slider(
                            "Add samples (SAMPLES_sum)", 0, 10, 0,
                            key=f"cf_samples_{hcp_id}")
                    pa_cf, pb_cf, pc_cf, pred_cf = simulate_single_hcp(
                        dataset, idx,
                        details_delta=cf_visits, samples_delta=cf_samples)
                    with c_right:
                        # Show before/after stacked bars
                        st.markdown(
                            '<div class="doctor-card" style="padding:18px 20px;">'
                            '<div style="font-size:13px;color:#0F172A;'
                            'font-weight:700;margin-bottom:10px;">'
                            'Probability comparison'
                            '</div>'
                            + stacked_bar("📍  Today",      pa_o,  pb_o,  pc_o)
                            + stacked_bar("🔮  After Δ",   pa_cf, pb_cf, pc_cf)
                            + '</div>',
                            unsafe_allow_html=True,
                        )

                    # Verdict
                    flip_html = ""
                    if pred_cf == "SEG_C" and pred_o != "SEG_C":
                        flip_html = (
                            '<div class="info-panel" style="background:#F0FDF4;'
                            'border-color:#86EFAC;border-left-color:#16A34A;">'
                            '<div class="info-panel-icon"></div>'
                            f'<div class="info-panel-content">'
                            f'<b>Conversion candidate.</b> With +{cf_visits} '
                            f'visit(s) and +{cf_samples} sample(s) this HCP '
                            f'flips <b>SEG_B → SEG_C</b>: '
                            f'P(C) moves from <b>{pc_o*100:.1f}%</b> to '
                            f'<b>{pc_cf*100:.1f}%</b>.'
                            f'</div></div>')
                    elif pred_cf == "SEG_A":
                        flip_html = (
                            '<div class="info-panel" style="background:#FFF7ED;'
                            'border-color:#FED7AA;border-left-color:#F47B20;">'
                            '<div class="info-panel-icon">⚠</div>'
                            f'<div class="info-panel-content">'
                            f'Under this intervention the model would '
                            f'reclassify this HCP as <b>SEG_A</b> '
                            f'(P(A) = <b>{pa_cf*100:.1f}%</b>) — likely an '
                            f'edge case worth reviewing.'
                            f'</div></div>')
                    else:
                        flip_html = (
                            '<div class="info-panel">'
                            '<div class="info-panel-icon">💡</div>'
                            f'<div class="info-panel-content">'
                            f'Still <b>SEG_B</b> after intervention. '
                            f'P(C) shifts from <b>{pc_o*100:.1f}%</b> '
                            f'to <b>{pc_cf*100:.1f}%</b> '
                            f'(Δ {(pc_cf - pc_o)*100:+.1f}pp). '
                            f'Try a larger Δ or layer in samples.'
                            f'</div></div>')
                    st.markdown(flip_html, unsafe_allow_html=True)

                # ── Confidence intervals (Ordinal, 5-fold CV) ──
                if R["is_labeled"][idx]:
                    section("Prediction confidence — 95% CI",
                            "Range of probabilities across 5 cross-validation "
                            "folds. Wider bars = less stable prediction.",
                            icon="")
                    ci_lo = R["ci_lo"][idx]
                    ci_hi = R["ci_hi"][idx]
                    point = [pa_o, pb_o, pc_o]
                    seg_names = ["SEG_A", "SEG_B", "SEG_C"]
                    seg_colors_list = [SEG_COLORS["SEG_A"],
                                         SEG_COLORS["SEG_B"],
                                         SEG_COLORS["SEG_C"]]
                    fig_ci = go.Figure()
                    for i_, (s, c, lo, hi, mn) in enumerate(zip(
                            seg_names, seg_colors_list, ci_lo, ci_hi, point)):
                        fig_ci.add_trace(go.Bar(
                            x=[hi - lo], y=[s], base=[lo], orientation="h",
                            marker=dict(color=c, opacity=0.35,
                                          line=dict(width=0)),
                            text=[f"&nbsp;&nbsp;{lo*100:.0f}%–{hi*100:.0f}%"],
                            textposition="outside",
                            cliponaxis=False,
                            hovertemplate=(f"<b>{s}</b><br>"
                                            f"Range: {lo*100:.1f}% – "
                                            f"{hi*100:.1f}%<br>"
                                            f"Point estimate: "
                                            f"{mn*100:.1f}%<extra></extra>"),
                            showlegend=False,
                        ))
                        # Point estimate marker on top
                        fig_ci.add_trace(go.Scatter(
                            x=[mn], y=[s], mode="markers",
                            marker=dict(size=12, color=c, symbol="line-ns",
                                          line=dict(color=c, width=3)),
                            hovertemplate=(f"Point estimate: "
                                            f"{mn*100:.1f}%<extra></extra>"),
                            showlegend=False,
                        ))
                    _style_fig(fig_ci, height=260)
                    fig_ci.update_layout(
                        title=dict(text=""),
                        margin=dict(r=60),
                        xaxis=dict(range=[0, 1.1], tickformat=".0%",
                                    title=""),
                        yaxis=dict(title="", autorange="reversed"),
                        bargap=0.45,
                    )
                    st.plotly_chart(fig_ci, use_container_width=True)

                # ── SHAP top features (labeled HCPs only) ──
                if R["is_labeled"][idx] and R["shap_top"][idx]:
                    section("Top features driving this prediction",
                            "SHAP contributions to the SEG_C decision. "
                            "Red = pushes toward SEG_C, blue = pushes away.",
                            icon="")
                    shap_rows = []
                    feat_cols = R["feat_cols"]
                    for j, val in R["shap_top"][idx]:
                        if j < len(feat_cols):
                            shap_rows.append({
                                "Feature": feat_cols[j],
                                "Contribution": float(val),
                            })
                    df_shap = pd.DataFrame(shap_rows)
                    df_shap["Direction"] = np.where(
                        df_shap["Contribution"] >= 0, "↑ Toward SEG_C",
                        "↓ Away from SEG_C")
                    df_shap["abs"] = df_shap["Contribution"].abs()
                    df_shap = df_shap.sort_values("abs")  # smallest first → biggest at top after reverse
                    fig_sh = px.bar(
                        df_shap, x="Contribution", y="Feature", orientation="h",
                        color="Direction",
                        color_discrete_map={"↑ Toward SEG_C": SEG_COLORS["SEG_C"],
                                              "↓ Away from SEG_C": SEG_COLORS["SEG_A"]},
                        text=df_shap["Contribution"].apply(
                            lambda v: f"{'+' if v>=0 else ''}{v:.3f}"),
                    )
                    fig_sh.update_traces(textposition="outside",
                                          marker_line_width=0, opacity=0.85,
                                          cliponaxis=False)

                    _style_fig(fig_sh, height=380)
                    
                    # Expand the x-axis so the outer text labels don't get clipped
                    val_max = df_shap["Contribution"].abs().max()
                    
                    fig_sh.update_layout(
                        title=dict(text=""),
                        yaxis_title="", xaxis_title="SHAP value",
                        xaxis=dict(range=[-val_max * 1.3, val_max * 1.3]),
                        margin=dict(l=220, r=60, t=60),
                        showlegend=False,
                        annotations=[
                            dict(
                                x=0.0, y=1.05, xref="paper", yref="paper",
                                text="<span style='color:#0070BF'>■</span> ↓ Away from SEG_C",
                                showarrow=False, xanchor="left", yanchor="bottom",
                                font=dict(size=13, color="#475569", family="Inter, sans-serif")
                            ),
                            dict(
                                x=1.0, y=1.05, xref="paper", yref="paper",
                                text="<span style='color:#7C3F98'>■</span> ↑ Toward SEG_C",
                                showarrow=False, xanchor="right", yanchor="bottom",
                                font=dict(size=13, color="#475569", family="Inter, sans-serif")
                            )
                        ]
                    )
                    st.plotly_chart(fig_sh, use_container_width=True)

                # ── Insight panel: model vs ATSEG mismatch (labeled HCPs) ──
                if is_labeled_hcp and pred_o != true_class:
                    st.markdown(
                        f"""
                        <div class="info-panel">
                            <div class="info-panel-icon">💡</div>
                            <div class="info-panel-content">
                                <b>Model differs from ATSEG.</b> The Ordinal model
                                predicts <b>{pred_o}</b> while the current
                                ATSEG assignment is <b>{true_class}</b>.
                                The Ordinal model is calibrated to favour SEG_C
                                when there is meaningful risk
                                (P(C) ≥ {THR_C}) — review this HCP if the
                                SEG_B / SEG_C distinction is operationally
                                important.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ── Prescribing Profile ──
                section("Prescribing Profile",
                        "How this HCP compares against the median for each segment",
                        icon="")
                if R["available_features"]:
                    feat_rows = []
                    for k, name, _desc in R["available_features"]:
                        val = float(R["df"][k].values[idx])
                        row = {"Feature": name, "Code": k,
                                "This HCP": val}
                        for seg in VALID_LABELS:
                            mask = R["df"]["ATSEG_first"] == seg
                            row[f"{seg} median"] = (float(R["df"].loc[mask, k].median())
                                                     if mask.sum() > 0 else 0.0)
                        feat_rows.append(row)
                    df_feat = pd.DataFrame(feat_rows).round(2)

                    # Use a column config to render the "This HCP" value with a colored progress bar
                    val_max = max(df_feat[["This HCP", "SEG_A median",
                                              "SEG_B median", "SEG_C median"]
                                            ].max(axis=1).max(), 1)
                    st.dataframe(
                        df_feat, use_container_width=True, hide_index=True,
                        column_config={
                            "This HCP": st.column_config.ProgressColumn(
                                "This HCP", format="%.2f",
                                min_value=0, max_value=float(val_max),
                            ),
                            "SEG_A median": st.column_config.NumberColumn(
                                "SEG_A median", format="%.2f",
                            ),
                            "SEG_B median": st.column_config.NumberColumn(
                                "SEG_B median", format="%.2f",
                            ),
                            "SEG_C median": st.column_config.NumberColumn(
                                "SEG_C median", format="%.2f",
                            ),
                        },
                    )

                # ── Individual Profile Radar ──
                section("Profile Radar — HCP vs segment centroids",
                        "Where does this HCP sit relative to the median "
                        "profile of each ATSEG segment? Try different scaling "
                        "approaches to surface different patterns.",
                        icon="")

                feats_full, seg_meds = segment_medians(dataset, top_n=6)
                feat_keys  = [k for k, _ in feats_full]
                feat_names = [n_ for _, n_ in feats_full]

                # This HCP's raw values for each radar feature
                hcp_vals = [float(R["df"][k].values[idx]) for k in feat_keys]

                # ── Scale picker ──
                col_scale, col_help = st.columns([2, 3])
                with col_scale:
                    scale_mode = st.selectbox(
                        "Radar scale",
                        ["Linear (per-axis max)",
                          "Logarithmic (log1p)",
                          "Percentile rank (vs population)",
                          "Z-score (clipped)"],
                        key="radar_scale",
                    )
                scale_hints = {
                    "Linear (per-axis max)":
                        "Each axis ÷ max of the 4 traces. Good for relative "
                        "comparison but small differences may flatten when "
                        "one value dominates.",
                    "Logarithmic (log1p)":
                        "log1p of values, then per-axis max. Best for the "
                        "long-tail prescription metrics — compresses large "
                        "values, expands small ones so quiet features remain visible.",
                    "Percentile rank (vs population)":
                        "Each value becomes its percentile in the labeled HCP "
                        "population (0–100). Most interpretable: 'this HCP "
                        "is in the 80th percentile for DETAILS'.",
                    "Z-score (clipped)":
                        "(value − mean)/std over the labeled population, "
                        "clipped to ±2σ then mapped to 0–1. Highlights "
                        "atypical values regardless of feature magnitude.",
                }
                with col_help:
                    st.markdown(
                        f'<div style="font-size:12px;color:#475569;'
                        f'padding:32px 6px 0 0;line-height:1.5;'
                        f'text-align:right;">'
                        f'{scale_hints[scale_mode]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # ── Build normalised vectors based on the chosen scale ──
                df_full = R["df"]
                lab_mask = R["is_labeled"]
                pop_arrays = {k: df_full.loc[lab_mask, k].values
                                for k in feat_keys}

                norm_hcp, norm_a, norm_b, norm_c = [], [], [], []

                if scale_mode == "Linear (per-axis max)":
                    for fi in range(len(feat_keys)):
                        quartet = [hcp_vals[fi], seg_meds["SEG_A"][fi],
                                    seg_meds["SEG_B"][fi], seg_meds["SEG_C"][fi]]
                        mx = max(quartet) if max(quartet) > 0 else 1.0
                        norm_hcp.append(hcp_vals[fi] / mx)
                        norm_a.append(seg_meds["SEG_A"][fi] / mx)
                        norm_b.append(seg_meds["SEG_B"][fi] / mx)
                        norm_c.append(seg_meds["SEG_C"][fi] / mx)

                elif scale_mode == "Logarithmic (log1p)":
                    for fi in range(len(feat_keys)):
                        quartet = [hcp_vals[fi], seg_meds["SEG_A"][fi],
                                    seg_meds["SEG_B"][fi], seg_meds["SEG_C"][fi]]
                        log_q = [np.log1p(max(v, 0)) for v in quartet]
                        mx = max(log_q) if max(log_q) > 0 else 1.0
                        norm_hcp.append(log_q[0] / mx)
                        norm_a.append(log_q[1] / mx)
                        norm_b.append(log_q[2] / mx)
                        norm_c.append(log_q[3] / mx)

                elif scale_mode == "Percentile rank (vs population)":
                    for fi, k in enumerate(feat_keys):
                        pop = pop_arrays[k]
                        n_pop = len(pop)
                        def _pct(v, _pop=pop, _n=n_pop):
                            return float((_pop <= v).sum()) / _n if _n else 0.0
                        norm_hcp.append(_pct(hcp_vals[fi]))
                        norm_a.append(_pct(seg_meds["SEG_A"][fi]))
                        norm_b.append(_pct(seg_meds["SEG_B"][fi]))
                        norm_c.append(_pct(seg_meds["SEG_C"][fi]))

                else:  # Z-score (clipped)
                    for fi, k in enumerate(feat_keys):
                        pop = pop_arrays[k]
                        if len(pop) == 0:
                            norm_hcp.append(0.5); norm_a.append(0.5)
                            norm_b.append(0.5);   norm_c.append(0.5)
                            continue
                        mu = float(pop.mean()); sd = float(pop.std()) or 1.0
                        def _z01(v, _mu=mu, _sd=sd):
                            z = max(min((v - _mu) / _sd, 2.0), -2.0)
                            return (z + 2.0) / 4.0
                        norm_hcp.append(_z01(hcp_vals[fi]))
                        norm_a.append(_z01(seg_meds["SEG_A"][fi]))
                        norm_b.append(_z01(seg_meds["SEG_B"][fi]))
                        norm_c.append(_z01(seg_meds["SEG_C"][fi]))

                # Close each polygon by repeating the first value
                theta = feat_names + [feat_names[0]]
                def _close(xs): return xs + [xs[0]]

                radar_fig = go.Figure()

                # Segment centroids (low opacity fills)
                for seg, vals, color in [
                    ("SEG_A", norm_a, SEG_COLORS["SEG_A"]),
                    ("SEG_B", norm_b, SEG_COLORS["SEG_B"]),
                    ("SEG_C", norm_c, SEG_COLORS["SEG_C"]),
                ]:
                    radar_fig.add_trace(go.Scatterpolar(
                        r=_close(vals), theta=theta,
                        fill="toself", name=f"{seg} median&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
                        line=dict(color=color, width=1.5),
                        fillcolor=color, opacity=0.20,
                        hovertemplate=("<b>%{theta}</b><br>"
                                        f"{seg} median: " "%{r:.2f}"
                                        "<extra></extra>"),
                    ))

                # The HCP itself — solid line, more prominent
                radar_fig.add_trace(go.Scatterpolar(
                    r=_close(norm_hcp), theta=theta,
                    fill="toself", name=f"HCP {hcp_id}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;",
                    line=dict(color="#0B1B33", width=3, dash="solid"),
                    fillcolor="rgba(11,27,51,0.10)",
                    marker=dict(size=8, color="#0B1B33",
                                  line=dict(color="white", width=1.5)),
                    mode="lines+markers",
                    hovertemplate=("<b>%{theta}</b><br>"
                                    "This HCP: %{r:.2f}<extra></extra>"),
                ))

                radar_fig.update_layout(
                    polar=dict(
                        bgcolor="rgba(244,247,250,0.4)",
                        radialaxis=dict(
                            visible=True, range=[0, 1],
                            tickfont=dict(size=10, color="#64748B"),
                            gridcolor="rgba(0,0,0,0.10)",
                            linecolor="rgba(0,0,0,0.15)",
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=11, color="#0B1B33",
                                            family="Inter, sans-serif"),
                            gridcolor="rgba(0,0,0,0.06)",
                            linecolor="rgba(0,0,0,0.10)",
                        ),
                    ),
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="top", y=0.9,
                                xanchor="left", x=1.05,
                                font=dict(size=11, color="#0B1B33")),
                )
                _style_fig(radar_fig, height=520)
                
                # Fix undefined title issue after _style_fig sets title=None
                # Adjust margins symmetrically so the chart shrinks a bit and stays centered
                radar_fig.update_layout(
                    title=dict(text=""), 
                    margin=dict(l=100, r=160, t=50, b=50)
                )
                
                st.plotly_chart(radar_fig, use_container_width=True)

                # Quick textual summary: which segment is this HCP closest to?
                # Use Euclidean distance between the HCP vector and each
                # segment-centroid vector in normalised space.
                dist = {}
                for seg, vals in [("SEG_A", norm_a), ("SEG_B", norm_b),
                                    ("SEG_C", norm_c)]:
                    dist[seg] = float(np.sqrt(sum(
                        (h - v) ** 2 for h, v in zip(norm_hcp, vals))))
                closest = min(dist, key=dist.get)
                closest_color = SEG_COLORS[closest]
                st.markdown(
                    f'<div class="info-panel">'
                    f'<div class="info-panel-icon"></div>'
                    f'<div class="info-panel-content">'
                    f'Using the <b>{scale_mode}</b> scale, this HCP is closest to '
                    f'<b style="color:{closest_color}">{closest}</b> '
                    f'(distance {dist[closest]:.2f}). '
                    f'All distances: SEG_A {dist["SEG_A"]:.2f} · '
                    f'SEG_B {dist["SEG_B"]:.2f} · '
                    f'SEG_C {dist["SEG_C"]:.2f}. '
                    f'<i>Switch the scale above to see if this changes.</i>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-state-icon"></div>
                    <div class="empty-state-title">Search any HCP to begin</div>
                    <div class="empty-state-text">
                        Enter a <b>NUEVO_ID</b> in the search box above to see:
                        the ground-truth ATSEG (if available),
                        side-by-side predictions from both models,
                        probability breakdown bars, and the HCP's
                        prescribing profile vs segment medians.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Conversion Strategy (B → C) ──
    with tabs[4]:
        section("Conversion Strategy — B to C movement plan",
                "Predicted SEG_B doctors closest to SEG_C, plus the engagement "
                "gaps holding them back",
                icon="")

        # Loader only on first computation — subsequent reruns hit cache
        if not st.session_state.get("cs_first_render_done"):
            _cs_loader = brand_loader(
                "Computing conversion strategy",
                "Identifying SEG_B doctors closest to SEG_C · "
                "scoring engagement gaps",
            )
            cand_df, agg_df, c_meds = compute_conversion_strategy(dataset)
            _cs_loader.empty()
            st.session_state.cs_first_render_done = True
        else:
            cand_df, agg_df, c_meds = compute_conversion_strategy(dataset)

        # KPI strip
        n_cand = len(cand_df)
        n_actionable_below = sum(1 for _, r in agg_df.iterrows()
                                  if r["Actionable"]
                                  and r["% below SEG_C median"] >= 50)
        avg_pc = (cand_df["P(C)"].mean() if n_cand else 0)
        top_action_count = (cand_df["Top action"].ne("—").sum()
                             if n_cand else 0)

        cols = st.columns(4)
        kpi_card(cols[0], "Candidates Identified", f"{n_cand:,}",
                 helper="Predicted SEG_B with highest P(C)",
                 style="accent", icon="",
                 status="info", status_label="Pipeline")
        kpi_card(cols[1], "Avg P(C) at Candidates",
                 f"{avg_pc*100:.1f}%",
                 helper="Mean SEG_C probability",
                 style="warn", icon="",
                 status="info", status_label="Confidence")
        kpi_card(cols[2], "With Actionable Gap",
                 f"{top_action_count:,}",
                 helper="Have at least 1 actionable lever to pull",
                 style="good", icon="",
                 status="ok", status_label="Targetable")
        kpi_card(cols[3], "Critical Levers",
                 f"{n_actionable_below}",
                 helper="Actionable features ≥50% below SEG_C median",
                 style="danger", icon="",
                 status="fair", status_label="Focus")

        # Insight panel
        st.markdown(
            '<div class="info-panel">'
            '<div class="info-panel-icon">💡</div>'
            '<div class="info-panel-content">'
            '<b>How to read this tab:</b> these are HCPs the Ordinal model '
            'predicts as <b>SEG_B</b> but with the highest <b>P(C)</b>. '
            'They are statistically the closest to becoming SEG_C. '
            'For each, the <i>top action</i> is the most-negative gap on an '
            '<b>actionable</b> feature (rep visits, samples, etc.) — '
            'levers Pfizer can directly influence.'
            '</div></div>',
            unsafe_allow_html=True,
        )

        # Aggregate gap chart
        section("Aggregate gaps — what's holding them back",
                "% of B→C candidates whose feature value sits below the "
                "SEG_C median. Higher = bigger collective gap to close.",
                icon="")

        agg_plot = agg_df.copy()
        agg_plot["Type"] = np.where(agg_plot["Actionable"],
                                       "Actionable", "Signal")

        fig = px.bar(
            agg_plot, x="% below SEG_C median", y="Feature", orientation="h",
            color="Type",
            color_discrete_map={"Actionable": PFIZER_GREEN, "Signal": PFIZER_GRAY},
            text=agg_plot["% below SEG_C median"].apply(lambda v: f"{v:.0f}%"),
            hover_data=["Code", "SEG_C median"],
        )
        fig.update_traces(textposition="outside", marker_line_width=0,
                          opacity=0.92)
        _style_fig(fig, height=420,
                    title="Feature gaps across all conversion candidates")
        fig.update_layout(yaxis={"categoryorder": "total ascending"},
                          xaxis_title="% of candidates below SEG_C median",
                          yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        # Candidate table
        section("Conversion candidates",
                f"Top {n_cand:,} predicted-SEG_B doctors ranked by P(C)",
                icon="")

        if n_cand > 0:
            # Format the dataframe with progress bars on probability cols
            display_cols = ["HCP_ID", "True", "P(A)", "P(B)", "P(C)",
                              "Top action", "Action gap"]
            extra_cols = [c for c in cand_df.columns if c not in display_cols]
            ordered_cols = display_cols + extra_cols
            st.dataframe(
                cand_df[ordered_cols], use_container_width=True,
                hide_index=True, height=520,
                column_config={
                    "P(A)": st.column_config.ProgressColumn(
                        "P(A)", format="%.2f", min_value=0, max_value=1),
                    "P(B)": st.column_config.ProgressColumn(
                        "P(B)", format="%.2f", min_value=0, max_value=1),
                    "P(C)": st.column_config.ProgressColumn(
                        "P(C)", format="%.2f", min_value=0, max_value=1),
                },
            )
            csv_cand = cand_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇  Download conversion plan (CSV)",
                                data=csv_cand,
                                file_name="hcp_conversion_strategy.csv",
                                mime="text/csv")
        else:
            st.info("No SEG_B candidates available.")

    # ── Counterfactual simulation ──
    with tabs[5]:
        section("Counterfactual — What if Pfizer increases engagement?",
                "Simulate engagement deltas on predicted-SEG_B doctors. "
                "Modifies raw DETAILS / SAMPLES values, re-derives ALL "
                "engineered features, and re-scores the Ordinal model to "
                "count how many doctors flip from SEG_B to SEG_C.",
                icon="")

        # Loader only on first computation — subsequent reruns hit cache
        if not st.session_state.get("cf_first_render_done"):
            _cf_loader = brand_loader(
                "Running counterfactual scenarios",
                "Simulating 10 engagement deltas · re-engineering features · "
                "re-scoring the SEG_B universe",
            )
            CF = run_counterfactual_scenarios(dataset)
            _cf_loader.empty()
            st.session_state.cf_first_render_done = True
        else:
            CF = run_counterfactual_scenarios(dataset)

        n_seg_b = CF["n_seg_b"]
        sm = CF["summary"]
        dim = CF["dim_returns"]

        # Headline KPIs from the +1 visit scenario (most common ask)
        plus1_row = sm[sm["Scenario"] == "+1 visit"].iloc[0]
        plus1_flips = int(plus1_row["B→C"])
        plus1_pct   = float(plus1_row["% to C"])

        kcols = st.columns(4)
        kpi_card(kcols[0], "Predicted SEG_B HCPs", f"{n_seg_b:,}",
                 helper="Universe to influence",
                 style="warn", icon="",
                 status="info", status_label="Universe")
        kpi_card(kcols[1], "+1 visit converts",
                 f"{plus1_flips:,}",
                 helper=f"{plus1_pct:.1f}% of SEG_B flip to SEG_C",
                 style="good", icon="",
                 status=("ok" if plus1_pct >= 5
                          else "fair" if plus1_pct >= 2 else "poor"),
                 status_label=("Strong" if plus1_pct >= 5
                                else "Moderate" if plus1_pct >= 2
                                else "Weak"))
        # Diminishing-returns headline: +2 incremental over +1
        d2 = int(dim[dim["Added visits"] == 2]["B→C cumulative"].iloc[0])
        delta_2_vs_1 = d2 - plus1_flips
        kpi_card(kcols[2], "+2nd visit incremental",
                 f"{delta_2_vs_1:,}",
                 helper="Extra converts from the 2nd added visit",
                 style="accent", icon="",
                 status="info", status_label="Marginal")
        # SEG_C median DETAILS scenario
        med_row = sm[sm["Group"] == "DETAILS"].iloc[-1]
        med_flips = int(med_row["B→C"])
        kpi_card(kcols[3], "→ SEG_C median visits",
                 f"{med_flips:,}",
                 helper=f"Clip DETAILS to ≥{CF['details_c_med']:.0f}",
                 style="danger", icon="",
                 status="info", status_label="Ceiling")

        st.markdown(
            '<div class="info-panel">'
            '<div class="info-panel-icon">💡</div>'
            '<div class="info-panel-content">'
            '<b>How to read this tab:</b> we modify the raw '
            '<code>DETAILS_sum</code> / <code>SAMPLES_sum</code> for the '
            'predicted SEG_B universe, re-derive every engineered feature '
            '(ratios, logs, total_engagement, etc.) so the perturbation '
            'propagates correctly, then re-score with the Ordinal model. '
            'The "% to C" figure is the share of SEG_B doctors who flip to '
            'SEG_C under each scenario — the business signal for ROI.'
            '</div></div>',
            unsafe_allow_html=True,
        )

        # ── Scenario impact bar chart ──
        section("Scenario impact",
                "How many SEG_B doctors move to each outcome per scenario",
                icon="")

        # Long-format for stacked bars
        long = sm.melt(id_vars=["Group", "Scenario", "Total B", "% to C",
                                 "Mean P(C) after"],
                        value_vars=["B→C", "B stays", "B→A"],
                        var_name="Outcome", value_name="Count")
        outcome_colors = {
            "B→C":     SEG_COLORS["SEG_C"],
            "B stays": SEG_COLORS["SEG_B"],
            "B→A":     SEG_COLORS["SEG_A"],
        }
        fig = px.bar(
            long, x="Count", y="Scenario", color="Outcome",
            orientation="h", barmode="stack",
            color_discrete_map=outcome_colors,
            hover_data=["Group", "% to C", "Mean P(C) after"],
            category_orders={"Outcome": ["B→C", "B stays", "B→A"]},
        )
        fig.update_traces(marker_line_width=0, opacity=0.92)
        _style_fig(fig, height=440, title="SEG_B outcome distribution per scenario")
        fig.update_layout(
            yaxis={"categoryorder": "array",
                    "categoryarray": sm["Scenario"].tolist()[::-1]},
            xaxis_title="Number of SEG_B doctors",
            yaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                          xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Diminishing returns line chart ──
        section("Diminishing returns — DETAILS sweep",
                "Cumulative B→C flips as we add 0..10 visits to each "
                "predicted-SEG_B doctor. The slope shows when extra visits "
                "stop adding meaningful conversions.",
                icon="")

        col_a, col_b = st.columns([3, 2])
        with col_a:
            fig_dim = go.Figure()
            fig_dim.add_trace(go.Scatter(
                x=dim["Added visits"], y=dim["B→C cumulative"],
                mode="lines+markers", name="Cumulative B→C",
                line=dict(color=PFIZER_BLUE, width=3),
                marker=dict(size=8, color=PFIZER_BLUE,
                              line=dict(color="white", width=1.5)),
                hovertemplate=("Added visits: %{x}<br>"
                                "Cumulative B→C: %{y:,}<extra></extra>"),
                fill="tozeroy",
                fillcolor="rgba(0,114,206,0.10)",
            ))
            _style_fig(fig_dim, height=380,
                        title="Cumulative converts vs added visits")
            fig_dim.update_layout(
                xaxis=dict(title="Additional visits per SEG_B HCP",
                            dtick=1),
                yaxis_title="Cumulative B→C count",
            )
            st.plotly_chart(fig_dim, use_container_width=True)

        with col_b:
            fig_inc = px.bar(
                dim[dim["Added visits"] > 0],
                x="Added visits", y="Incremental",
                text="Incremental",
                color="Incremental", color_continuous_scale="Blues",
            )
            fig_inc.update_traces(textposition="outside",
                                    marker_line_width=0)
            _style_fig(fig_inc, height=380,
                        title="Incremental converts per added visit")
            fig_inc.update_layout(
                coloraxis_showscale=False,
                xaxis=dict(title="Visit #", dtick=1),
                yaxis_title="New B→C from this visit",
            )
            st.plotly_chart(fig_inc, use_container_width=True)

        # ── P(C) distribution shift histogram ──
        section("P(C) distribution shift — SEG_B before vs after +1 visit",
                "How the SEG_C probability moves for the SEG_B universe "
                "when each doctor receives one extra visit.",
                icon="")

        pc_base = CF["pc_base_seg_b"]
        pc_after = CF["pc_after_plus1"]

        bins = np.linspace(0, 1, 21)
        df_hist = pd.DataFrame({
            "P(C)": np.concatenate([pc_base, pc_after]),
            "When": (["Baseline"] * len(pc_base)
                      + ["After +1 visit"] * len(pc_after)),
        })
        fig_pc = px.histogram(
            df_hist, x="P(C)", color="When",
            barmode="overlay", opacity=0.55, nbins=20,
            color_discrete_map={
                "Baseline": "#94A3B8",
                "After +1 visit": SEG_COLORS["SEG_C"],
            },
        )
        # Vertical line at P(C)=0.30 threshold
        fig_pc.add_shape(type="line", x0=THR_C, x1=THR_C,
                          y0=0, y1=1, yref="paper",
                          line=dict(color="#0B1B33", width=2, dash="dash"))
        fig_pc.add_annotation(
            x=THR_C, y=1.02, yref="paper",
            text=f"<b>P(C) ≥ {THR_C}</b> → SEG_C",
            showarrow=False, font=dict(size=11, color="#0B1B33"),
        )
        _style_fig(fig_pc, height=420)
        fig_pc.update_layout(
            xaxis_tickformat=".0%",
            yaxis_title="SEG_B doctor count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                          xanchor="right", x=1),
        )
        st.plotly_chart(fig_pc, use_container_width=True)

        # ── Flippers vs Stayers profile (+1 visit) ──
        section("Flippers vs Stayers (+1 visit)",
                "Who actually converts with one extra visit? Compare the "
                "two cohorts on raw engagement and prescribing metrics.",
                icon="")

        flippers = CF["flippers_idx"]
        stayers  = CF["stayers_idx"]
        details_baseline = CF["details_baseline"]
        samples_baseline = CF["samples_baseline"]
        df_full_raw = load_data(dataset)

        def cohort_stats(name, idx):
            return {
                "Cohort": name,
                "n": int(len(idx)),
                "Mean P(C) before": float(CF["pc_base_seg_b"][
                    np.where(np.isin(CF["seg_b_idx"], idx))[0]
                ].mean()) if len(idx) > 0 else 0,
                "Mean DETAILS": float(details_baseline[idx].mean()) if len(idx) > 0 else 0,
                "Mean SAMPLES": float(samples_baseline[idx].mean()) if len(idx) > 0 else 0,
                "Mean UC_TRX_sum": float(df_full_raw["UC_TRX_sum"].values[idx].mean()) if len(idx) > 0 else 0,
                "Mean ORAL_TRX_sum": float(df_full_raw["ORAL_TRX_sum"].values[idx].mean()) if len(idx) > 0 else 0,
                "Mean IL23_TRX_sum": float(df_full_raw["IL23_TRX_sum"].values[idx].mean()) if len(idx) > 0 else 0,
            }
        df_cohort = pd.DataFrame([
            cohort_stats("Flippers (B→C)", flippers),
            cohort_stats("Stayers (B stays)", stayers),
        ])
        st.dataframe(df_cohort.round(3), use_container_width=True,
                      hide_index=True)

        # Side-by-side bar of mean features per cohort
        feat_compare = ["Mean DETAILS", "Mean SAMPLES",
                         "Mean UC_TRX_sum", "Mean ORAL_TRX_sum",
                         "Mean IL23_TRX_sum"]
        plot_rows = []
        for _, r in df_cohort.iterrows():
            for f in feat_compare:
                plot_rows.append({"Cohort": r["Cohort"],
                                    "Feature": f.replace("Mean ", ""),
                                    "Value": float(r[f])})
        plot_df = pd.DataFrame(plot_rows)
        fig_cohort = px.bar(
            plot_df, x="Feature", y="Value", color="Cohort",
            barmode="group", text=plot_df["Value"].round(2),
            color_discrete_sequence=[SEG_COLORS["SEG_C"], "#94A3B8"],
        )
        fig_cohort.update_traces(textposition="outside",
                                  marker_line_width=0)
        _style_fig(fig_cohort, height=420,
                    title="Mean raw feature values — Flippers vs Stayers")
        fig_cohort.update_layout(
            yaxis_title="",
            xaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                          xanchor="right", x=1),
        )
        st.plotly_chart(fig_cohort, use_container_width=True)

        # ── Top flippers table (CSV export) ──
        section("Top flippers — easy wins",
                "Predicted-SEG_B doctors who flip to SEG_C with just +1 visit, "
                "ranked by P(C) after intervention.",
                icon="")

        if len(flippers) > 0:
            pc_after_flippers = CF["pc_after_plus1"][
                np.isin(CF["seg_b_idx"], flippers)]
            df_flip = pd.DataFrame({
                "HCP_ID": R["ids"][flippers],
                "True ATSEG": np.where(R["is_labeled"][flippers],
                                         R["y_true"][flippers], "Unlabeled"),
                "P(C) before": CF["pc_base_seg_b"][
                    np.isin(CF["seg_b_idx"], flippers)],
                "P(C) after +1": pc_after_flippers,
                "DETAILS before": details_baseline[flippers],
                "SAMPLES before": samples_baseline[flippers],
                "UC_TRX_sum": df_full_raw["UC_TRX_sum"].values[flippers],
            }).round(3).sort_values("P(C) after +1", ascending=False)

            st.dataframe(
                df_flip, use_container_width=True, hide_index=True, height=480,
                column_config={
                    "P(C) before": st.column_config.ProgressColumn(
                        "P(C) before", format="%.2f",
                        min_value=0, max_value=1),
                    "P(C) after +1": st.column_config.ProgressColumn(
                        "P(C) after +1", format="%.2f",
                        min_value=0, max_value=1),
                },
            )
            csv_flip = df_flip.to_csv(index=False).encode("utf-8")
            st.download_button("⬇  Download flippers (CSV)",
                                data=csv_flip,
                                file_name="counterfactual_flippers.csv",
                                mime="text/csv")
        else:
            st.info("No SEG_B doctors flip to SEG_C with +1 visit.")

        # ── Raw scenario table ──
        with st.expander("All scenarios — raw counts"):
            st.dataframe(sm, use_container_width=True, hide_index=True)

    # ── Predictions Table ──
    with tabs[6]:
        section("Full Predictions Table",
                "All HCPs with the Ordinal model probabilities and predictions — filter & export",
                icon="")

        # CI widths per HCP (0 for unlabeled — CIs are CV-derived)
        ci_w   = R["ci_hi"] - R["ci_lo"]                   # (N, 3)
        ci_max = ci_w.max(axis=1)
        df_out = pd.DataFrame({
            "HCP_ID": R["ids"],
            "True ATSEG": np.where(R["is_labeled"], R["y_true"], "Unlabeled"),
            "Predicted": R["full"]["ord"]["pred"],
            "P(A)": R["full"]["ord"]["P_A"],
            "P(B)": R["full"]["ord"]["P_B"],
            "P(C)": R["full"]["ord"]["P_C"],
            # 95% CI low/high bounds for the predicted-segment probability
            "P(A) lo": R["ci_lo"][:, 0],
            "P(A) hi": R["ci_hi"][:, 0],
            "P(B) lo": R["ci_lo"][:, 1],
            "P(B) hi": R["ci_hi"][:, 1],
            "P(C) lo": R["ci_lo"][:, 2],
            "P(C) hi": R["ci_hi"][:, 2],
            "Max CI width": ci_max,
        }).round(3)

        # Filters
        col_seg, col_pred, col_unc = st.columns([1, 1, 1])
        with col_seg:
            seg_filter = st.multiselect(
                "True ATSEG", ["SEG_A", "SEG_B", "SEG_C", "Unlabeled"],
                placeholder="All segments")
        with col_pred:
            pred_filter = st.multiselect(
                "Predicted segment", ["SEG_A", "SEG_B", "SEG_C"],
                placeholder="All predictions")
        with col_unc:
            uncertainty = st.selectbox(
                "Uncertainty filter",
                ["All", "Stable (max CI < 15pp)",
                  "Borderline (max CI 15–30pp)", "Uncertain (max CI > 30pp)"],
            )

        if seg_filter:
            df_out = df_out[df_out["True ATSEG"].isin(seg_filter)]
        if pred_filter:
            df_out = df_out[df_out["Predicted"].isin(pred_filter)]
        if uncertainty == "Stable (max CI < 15pp)":
            df_out = df_out[df_out["Max CI width"] < 0.15]
        elif uncertainty == "Borderline (max CI 15–30pp)":
            df_out = df_out[(df_out["Max CI width"] >= 0.15)
                              & (df_out["Max CI width"] < 0.30)]
        elif uncertainty == "Uncertain (max CI > 30pp)":
            df_out = df_out[df_out["Max CI width"] >= 0.30]

        # Summary KPIs
        cols = st.columns(4)
        kpi_card(cols[0], "Filtered HCPs", f"{len(df_out):,}",
                 helper=f"of {len(R['ids']):,} total",
                 style="neutral", icon="",
                 status="info", status_label="Selection")

        labeled_count = df_out["True ATSEG"].isin(VALID_LABELS).sum()
        hits = ((df_out["Predicted"] == df_out["True ATSEG"])
                 & df_out["True ATSEG"].isin(VALID_LABELS)).sum()
        misses = labeled_count - hits

        kpi_card(cols[1], "Hits (vs ATSEG)", f"{hits:,}",
                 helper=f"{(hits/max(labeled_count,1))*100:.1f}% of labeled",
                 style="good", icon="",
                 status="ok" if labeled_count > 0 else "info",
                 status_label="Match")
        kpi_card(cols[2], "Misses (vs ATSEG)", f"{misses:,}",
                 helper=f"{(misses/max(labeled_count,1))*100:.1f}% of labeled",
                 style="warn", icon="",
                 status="info", status_label="Differ")

        n_unlab = (df_out["True ATSEG"] == "Unlabeled").sum()
        kpi_card(cols[3], "Unlabeled scored", f"{n_unlab:,}",
                 helper="HCPs with no ATSEG, scored by the model",
                 style="accent", icon="",
                 status="info", status_label="Predicted")

        st.markdown("&nbsp;")
        st.dataframe(
            df_out, use_container_width=True, hide_index=True, height=560,
            column_config={
                "P(A)": st.column_config.ProgressColumn(
                    "P(A)", format="%.2f", min_value=0, max_value=1),
                "P(B)": st.column_config.ProgressColumn(
                    "P(B)", format="%.2f", min_value=0, max_value=1),
                "P(C)": st.column_config.ProgressColumn(
                    "P(C)", format="%.2f", min_value=0, max_value=1),
                "P(A) lo": st.column_config.NumberColumn(
                    "A lo", format="%.2f"),
                "P(A) hi": st.column_config.NumberColumn(
                    "A hi", format="%.2f"),
                "P(B) lo": st.column_config.NumberColumn(
                    "B lo", format="%.2f"),
                "P(B) hi": st.column_config.NumberColumn(
                    "B hi", format="%.2f"),
                "P(C) lo": st.column_config.NumberColumn(
                    "C lo", format="%.2f"),
                "P(C) hi": st.column_config.NumberColumn(
                    "C hi", format="%.2f"),
                "Max CI width": st.column_config.ProgressColumn(
                    "Max CI",
                    help="Widest 95% CI across A/B/C "
                         "(higher = more uncertain)",
                    format="%.2f", min_value=0, max_value=1),
            },
            column_order=("HCP_ID", "True ATSEG", "Predicted",
                            "P(A)", "P(B)", "P(C)", "Max CI width",
                            "P(A) lo", "P(A) hi",
                            "P(B) lo", "P(B) hi",
                            "P(C) lo", "P(C) hi"),
        )

        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇  Download filtered CSV",
                            data=csv, file_name="hcp_predictions.csv",
                            mime="text/csv")


# ─────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        <span class="footer-brand">PFIZER · COMMERCIAL ANALYTICS</span>
        <span class="footer-divider"></span>
        HCP Probability Engine
        <span class="footer-divider"></span>
        Built with Streamlit · XGBoost
        <span class="footer-divider"></span>
        Random seed locked at 42 for reproducibility
    </div>
    """,
    unsafe_allow_html=True,
)
