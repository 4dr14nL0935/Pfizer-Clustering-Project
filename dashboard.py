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
import plotly.express as px
import plotly.graph_objects as go

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
st.set_page_config(
    page_title="Pfizer HCP Segmentation",
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

    /* ── SECTION HEADER ── */
    .section-header { margin: 36px 0 18px 0; animation: fadeInSlide 0.5s ease both; }
    .section-header-row {
        display: flex; align-items: center; gap: 14px; margin-bottom: 8px;
    }
    .section-icon {
        width: 38px; height: 38px; border-radius: 10px;
        background: linear-gradient(135deg, #003B71, #0070BF);
        display: flex; align-items: center; justify-content: center;
        font-size: 18px; color: white;
        box-shadow: 0 4px 12px rgba(0,112,191,0.2); flex-shrink: 0;
    }
    .section-header h2 {
        margin: 0; color: #0F172A; font-size: 24px;
        font-weight: 700; letter-spacing: -0.3px;
    }
    .section-header p {
        margin: 0 0 0 52px; color: #64748B;
        font-size: 15px; font-weight: 400;
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

    /* ── SIDEBAR LABELS ── */
    .sb-label {
        display: flex; align-items: center; gap: 10px; margin: 24px 0 10px 0;
    }
    .sb-label-bar { width: 3px; height: 18px; border-radius: 999px; flex-shrink: 0; }
    .sb-label-bar.blue   { background: #0070BF; }
    .sb-label-bar.orange { background: #F47B20; }
    .sb-label-bar.purple { background: #7C3AED; }
    .sb-label-bar.green  { background: #059669; }
    .sb-label-text {
        font-size: 11px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1px; color: #0F172A;
    }
    .sb-label-icon { font-size: 14px; margin-right: 2px; color: #64748B; }

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

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: #F1F5F9; padding: 5px;
        border-radius: 10px; border: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border-radius: 8px;
        color: #64748B; font-weight: 500; font-size: 14px;
        padding: 10px 18px; transition: all .2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { background: #FFFFFF; color: #0F172A; }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #0070BF !important;
        border: 1px solid #DBEAFE !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
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
    status_html = (f'<div class="kpi-status {status}">{status_label}</div>'
                    if status else "")
    col.markdown(
        f"""
        <div class="{cls}">
            {icon_html}
            <div class="kpi-content-top">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-delta">{helper}</div>
            </div>
            <div class="kpi-content-bottom">
                {status_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title, subtitle="", icon="📊"):
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    html = (
        f'<div class="section-header">'
        f'<div class="section-header-row">'
        f'<div class="section-icon">{icon}</div>'
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
# Sidebar
# ─────────────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div class="sb-brand">
        <div class="sb-brand-row">
            <div class="sb-brand-logo">💊</div>
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


sb_label("📂", "Dataset", color="orange")

# Clickable styled button → opens the preview popup
if st.sidebar.button(
    "📂  doctors_aggregated.csv",
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
sb_label("🧠", "Model", color="blue")
st.sidebar.markdown(
    '<div class="sb-current">'
    '<span class="sb-current-label">Active</span>'
    '<span class="sb-current-value">🎯 Ordinal (v3.2)</span>'
    '</div>',
    unsafe_allow_html=True,
)

sb_divider()

# Active configuration summary
st.sidebar.markdown(
    "<div style='font-size:10px;color:#64748B;margin:14px 0 4px 4px;"
    "font-weight:700;letter-spacing:.7px;text-transform:uppercase;'>"
    "Active Configuration</div>",
    unsafe_allow_html=True,
)
sb_current("Dataset", "doctors_aggregated.csv")
sb_current("Model",   "Ordinal v3.2")

st.sidebar.markdown(
    f"""
    <div class="sb-footer">
        <div style="font-size:10px;color:#64748B;letter-spacing:.7px;
                    font-weight:700;text-transform:uppercase;">Reproducibility</div>
        <div class="sb-footer-row"><span>Random seed</span><span class="sb-footer-key">{SEED}</span></div>
        <div class="sb-footer-row"><span>CV folds</span><span class="sb-footer-key">5</span></div>
        <div class="sb-footer-row"><span>P(A) cutoff</span><span class="sb-footer-key">{THR_A}</span></div>
        <div class="sb-footer-row"><span>P(C) cutoff</span><span class="sb-footer-key">{THR_C}</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
            <span class="hero-pill">📂 Dataset <b>doctors_aggregated.csv</b></span>
            <span class="hero-pill">🧠 Model <b>Ordinal v3.2</b></span>
            <span class="hero-pill">🎯 P(A)≥<b>{THR_A}</b> · P(C)≥<b>{THR_C}</b> · dominance rule</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────
# Train pipeline (cached)
# ─────────────────────────────────────────────────────────────────────────
with st.spinner("Training XGBoost models (cached after first run)..."):
    R = train_pipeline(dataset)

# ─────────────────────────────────────────────────────────────────────────
# Context strip
# ─────────────────────────────────────────────────────────────────────────
context_items = [
    ("HCPs (total)", f"{len(R['ids']):,}"),
    ("HCPs (labeled)", f"{R['n_labeled']:,}"),
    ("HCPs (unlabeled)", f"{R['n_unlabeled']:,}"),
    ("Model", "Ordinal v3.2"),
]
context_strip(context_items, accent_index=0)


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
        "🔍  Data Overview",
        "📈  Model Performance",
        "🌐  Probability Map",
        "🔬  Doctor Explorer",
        "🎯  Conversion Strategy",
        "📋  Predictions Table",
    ])

    # ── Data Overview ──
    with tabs[0]:
        section("ATSEG Segmentation Overview",
                "Model performance metrics and HCP distribution across segments",
                icon="🔍")



        # ── Row 1: Model performance KPIs ──
        cols = st.columns(5)
        kpi_card(cols[0], "HCPs Analyzed", f"{len(R['ids']):,}",
                 helper=f"{R['n_labeled']:,} labeled · {R['n_unlabeled']:,} unlabeled",
                 style="neutral", icon="👥",
                 status="info", status_label="Loaded")
        kpi_card(cols[1], "Accuracy", f"{m['accuracy']*100:.1f}%",
                 helper="Overall correctness",
                 style="good", icon="🎯",
                 status=acc_s, status_label=acc_l)
        kpi_card(cols[2], "Balanced Accuracy", f"{m['balanced_accuracy']*100:.1f}%",
                 helper="Class-balanced score",
                 style="accent", icon="⚖",
                 status=acc_s, status_label=acc_l)
        kpi_card(cols[3], "Recall (SEG_C)", f"{m['recall_C']*100:.1f}%",
                 helper="C captured by the model",
                 style="warn", icon="🔬",
                 status=rec_s, status_label=rec_l)
        kpi_card(cols[4], "SEG_C → SEG_A Loss", f"{m['c_lost_pct']*100:.1f}%",
                 helper="Catastrophic mis-classifications",
                 style="danger", icon="⚠",
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
                 style="good compact", icon="✓",
                 status="ok", status_label="Labeled")
        kpi_card(cols[1], "SEG_C HCPs",
                 f"{cnt_c:,}",
                 helper="High-value targets",
                 style="danger compact", icon="🎯",
                 status="info", status_label="Priority")
        kpi_card(cols[2], "Unlabeled", f"{no_atseg:,}",
                 helper=f"{100-coverage:.1f}% to score",
                 style="warn compact", icon="❓",
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
                icon="📈")

        if R["available_features"]:
            feat_options = [k for k, _, _ in R["available_features"]]
            feat_pick = st.selectbox("Feature to inspect", feat_options,
                                      key="explore_feat")

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
                icon="📈")

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

    # ── Probability Map (3D) ──
    with tabs[2]:
        section("Probability Map",
                f"Each HCP plotted by ({MODEL_LABELS_SHORT[model_pick]}) "
                "P(A) · P(B) · P(C). Color = predicted segment",
                icon="🌐")

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
            color_by = st.radio("Color points by",
                                 ["Predicted Segment", "True ATSEG"],
                                 horizontal=True, key="map_color_by")
        with col_search:
            highlight_q = st.text_input(
                "🔍  Highlight HCP",
                placeholder="Enter HCP ID (NUEVO_ID) to highlight in the map",
                key="map_highlight_id",
                label_visibility="collapsed",
            )
        color_col = "Predicted" if color_by == "Predicted Segment" else "True ATSEG"

        # Resolve highlight
        highlight_row = None
        if highlight_q.strip():
            matches = np.where(R["ids"] == highlight_q.strip())[0]
            if len(matches) == 0:
                st.warning(f"No HCP found with ID `{highlight_q.strip()}`.")
            else:
                hi_idx = matches[0]
                highlight_row = {
                    "HCP_ID": R["ids"][hi_idx],
                    "P(A)": float(R["full"][model_pick]["P_A"][hi_idx]),
                    "P(B)": float(R["full"][model_pick]["P_B"][hi_idx]),
                    "P(C)": float(R["full"][model_pick]["P_C"][hi_idx]),
                    "Predicted": str(R["full"][model_pick]["pred"][hi_idx]),
                    "True ATSEG": (R["y_true"][hi_idx]
                                    if R["is_labeled"][hi_idx] else "Unlabeled"),
                }

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
                    <div class="info-panel-icon">⭐</div>
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

    # ── Doctor Explorer ──
    with tabs[3]:
        section("Doctor Explorer",
                "Search any HCP to inspect probabilities and prescribing profile",
                icon="🔬")

        col_s, col_help = st.columns([3, 2])
        with col_s:
            search_q = st.text_input(
                "Search by HCP ID (NUEVO_ID)",
                placeholder="🔍  Enter HCP ID (e.g. 100012345)",
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
                        <div class="empty-state-icon">🔎</div>
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
                section("Segment", icon="🧠")

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

                # ── Confidence intervals (Ordinal, 5-fold CV) ──
                if R["is_labeled"][idx]:
                    section("Prediction confidence — 95% CI",
                            "Range of probabilities across 5 cross-validation "
                            "folds. Wider bars = less stable prediction.",
                            icon="📐")
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
                            icon="🧠")
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
                        icon="🧬")
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
                        icon="🕸")

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
                    f'<div class="info-panel-icon">🎯</div>'
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
                    <div class="empty-state-icon">🔍</div>
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
                icon="🎯")

        with st.spinner("Computing conversion candidates..."):
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
                 style="accent", icon="🎯",
                 status="info", status_label="Pipeline")
        kpi_card(cols[1], "Avg P(C) at Candidates",
                 f"{avg_pc*100:.1f}%",
                 helper="Mean SEG_C probability",
                 style="warn", icon="📐",
                 status="info", status_label="Confidence")
        kpi_card(cols[2], "With Actionable Gap",
                 f"{top_action_count:,}",
                 helper="Have at least 1 actionable lever to pull",
                 style="good", icon="✓",
                 status="ok", status_label="Targetable")
        kpi_card(cols[3], "Critical Levers",
                 f"{n_actionable_below}",
                 helper="Actionable features ≥50% below SEG_C median",
                 style="danger", icon="⚠",
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
                icon="📊")

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
                icon="📋")

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

    # ── Predictions Table ──
    with tabs[5]:
        section("Full Predictions Table",
                "All HCPs with both models' probabilities and predictions — filter & export",
                icon="📋")

        df_out = pd.DataFrame({
            "HCP_ID": R["ids"],
            "True ATSEG": np.where(R["is_labeled"], R["y_true"], "Unlabeled"),
            "Predicted": R["full"]["ord"]["pred"],
            "P(A)": R["full"]["ord"]["P_A"],
            "P(B)": R["full"]["ord"]["P_B"],
            "P(C)": R["full"]["ord"]["P_C"],
        }).round(3)

        # Filters
        col_seg, col_pred = st.columns([1, 1])
        with col_seg:
            seg_filter = st.multiselect(
                "True ATSEG", ["SEG_A", "SEG_B", "SEG_C", "Unlabeled"],
                placeholder="All segments")
        with col_pred:
            pred_filter = st.multiselect(
                "Predicted segment", ["SEG_A", "SEG_B", "SEG_C"],
                placeholder="All predictions")

        if seg_filter:
            df_out = df_out[df_out["True ATSEG"].isin(seg_filter)]
        if pred_filter:
            df_out = df_out[df_out["Predicted"].isin(pred_filter)]

        # Summary KPIs
        cols = st.columns(4)
        kpi_card(cols[0], "Filtered HCPs", f"{len(df_out):,}",
                 helper=f"of {len(R['ids']):,} total",
                 style="neutral", icon="👥",
                 status="info", status_label="Selection")

        labeled_count = df_out["True ATSEG"].isin(VALID_LABELS).sum()
        hits = ((df_out["Predicted"] == df_out["True ATSEG"])
                 & df_out["True ATSEG"].isin(VALID_LABELS)).sum()
        misses = labeled_count - hits

        kpi_card(cols[1], "Hits (vs ATSEG)", f"{hits:,}",
                 helper=f"{(hits/max(labeled_count,1))*100:.1f}% of labeled",
                 style="good", icon="🎯",
                 status="ok" if labeled_count > 0 else "info",
                 status_label="Match")
        kpi_card(cols[2], "Misses (vs ATSEG)", f"{misses:,}",
                 helper=f"{(misses/max(labeled_count,1))*100:.1f}% of labeled",
                 style="warn", icon="↔",
                 status="info", status_label="Differ")

        n_unlab = (df_out["True ATSEG"] == "Unlabeled").sum()
        kpi_card(cols[3], "Unlabeled scored", f"{n_unlab:,}",
                 helper="HCPs with no ATSEG, scored by the model",
                 style="accent", icon="❓",
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
            },
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
