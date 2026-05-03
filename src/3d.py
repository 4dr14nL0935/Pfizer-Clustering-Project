"""
build_probability_visualization_v3.py
=====================================
Generates a multi-tab HTML dashboard for HCP segmentation:
  Tab 1 — Overview & Methodology (model explanation, segment defs, error costs)
  Tab 2 — Probability Map         (ternary + 3D scatter)
  Tab 3 — Doctor Explorer         (per-HCP feature comparison and explanation)

Recreates the capstone's final ordinal model:
  Stage 1: P(>=B) = P(not A)         — XGBoost binary
  Stage 2: P(>=C) = P(is C)          — XGBoost binary, sample_weight=2.0 for C
  Ordinal coherence: P(>=C) = min(P(>=C), P(>=B))
  P(A) = 1 - P(>=B);  P(B) = P(>=B) - P(>=C);  P(C) = P(>=C)
  Decision: P(A) >= 0.70 -> A;  elif P(C) >= 0.30 -> C;  else B

USAGE:
    python build_probability_visualization_v3.py --input doctors_aggregated.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

try:
    import xgboost as xgb
except ImportError:
    sys.exit("xgboost not installed.  Run: pip install xgboost")


# ---------- Hyperparameters (from final model) ----------
XGB_PARAMS = dict(
    n_estimators=800, max_depth=5, learning_rate=0.04,
    subsample=0.85, colsample_bytree=0.7, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
    eval_metric="logloss", random_state=42, n_jobs=-1,
)

THR_A, THR_C = 0.70, 0.30
SEG_C_SAMPLE_WEIGHT = 2.0
VALID_LABELS = ("SEG_A", "SEG_B", "SEG_C")

# Features used for the per-doctor explanation
KEY_FEATURES = [
    ("UC_TRX_sum",    "UC prescriptions (TRx)",          "Total Rx for ulcerative colitis"),
    ("UC_NRX_sum",    "UC new prescriptions (NRx)",      "Patients newly starting UC therapy"),
    ("ORAL_TRX_sum",  "Oral therapy Rx",                 "Oral UC treatments — Velsipity's category"),
    ("IL23_TRX_sum",  "IL-23 inhibitor Rx",              "Competitor biologics for UC"),
    ("DETAILS_sum",   "Sales detail visits",             "Pfizer rep visits to this HCP"),
    ("TOTAL_TRX_sum", "Total prescription volume",       "Overall prescribing activity"),
]


def add_ratios(df):
    eps = 1e-6
    out = df.copy()
    pairs = [
        ("ratio_UC_over_ORAL",    "UC_TRX_sum",   "ORAL_TRX_sum"),
        ("ratio_UC_over_IL23",    "UC_TRX_sum",   "IL23_TRX_sum"),
        ("ratio_ORAL_over_IL23",  "ORAL_TRX_sum", "IL23_TRX_sum"),
        ("ratio_UC_NRX_TRX",      "UC_NRX_sum",   "UC_TRX_sum"),
        ("ratio_DETAILS_per_TRX", "DETAILS_sum",  "UC_TRX_sum"),
    ]
    n = 0
    for new, num, den in pairs:
        if num in out.columns and den in out.columns:
            out[new] = out[num] / (out[den] + eps); n += 1
    bcols = [c for c in out.columns if c.startswith("BRAND") and "_TRX_sum" in c]
    if bcols:
        out["brand_diversity"] = (out[bcols] > 0).sum(axis=1); n += 1
    print(f"  Derived features added: {n}")
    return out


def fit_ordinal(X_train, y_train):
    y_geq_b = (y_train != "SEG_A").astype(int)
    y_geq_c = (y_train == "SEG_C").astype(int)
    m1 = xgb.XGBClassifier(**XGB_PARAMS); m1.fit(X_train, y_geq_b)
    sw = np.where(y_geq_c == 1, SEG_C_SAMPLE_WEIGHT, 1.0)
    m2 = xgb.XGBClassifier(**XGB_PARAMS); m2.fit(X_train, y_geq_c, sample_weight=sw)
    return m1, m2


def predict_ordinal(m1, m2, X):
    p_b = m1.predict_proba(X)[:, 1]
    p_c = np.minimum(m2.predict_proba(X)[:, 1], p_b)
    P_A, P_B, P_C = 1 - p_b, p_b - p_c, p_c
    s = P_A + P_B + P_C
    return P_A/s, P_B/s, P_C/s


def decide(P_A, P_C):
    return np.where(P_A >= THR_A, "SEG_A",
           np.where(P_C >= THR_C, "SEG_C", "SEG_B"))


def oof_predictions(X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    P_A = np.zeros(len(y)); P_B = np.zeros(len(y)); P_C = np.zeros(len(y))
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"    fold {fold}/{n_splits}...")
        m1, m2 = fit_ordinal(X[tr], y[tr])
        P_A[va], P_B[va], P_C[va] = predict_ordinal(m1, m2, X[va])
    return P_A, P_B, P_C


def compute_segment_stats(df_labeled, label_col, key_feature_cols):
    """Per-segment quartiles for key features. Used by JS to explain predictions."""
    stats = {}
    for feat in key_feature_cols:
        if feat not in df_labeled.columns:
            continue
        per_seg = {}
        for seg in VALID_LABELS:
            vals = df_labeled.loc[df_labeled[label_col] == seg, feat].values
            per_seg[seg] = {
                "p25":    float(np.quantile(vals, 0.25)),
                "median": float(np.median(vals)),
                "p75":    float(np.quantile(vals, 0.75)),
            }
        stats[feat] = per_seg
    return stats


# ============================== HTML TEMPLATE ==============================
HTML_TEMPLATE = r'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>HCP Segmentation Dashboard — Velsipity</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{
  --bg:#f5f7fb;--card:#fff;--border:#e3e8f0;--text:#1a2330;--muted:#6b7c93;
  --pfizer:#0093D0;--pfd:#00255D;
  --a:#2e6cb0;--b:#f0a830;--c:#c83a3a;--u:#9aa6b4;
  --ok:#2a8b4a;--warn:#cc7a00;--bad:#b22a2a;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
  background:var(--bg);color:var(--text);padding:18px;line-height:1.5}
header{background:linear-gradient(135deg,var(--pfd),var(--pfizer));color:#fff;
  padding:18px 24px;border-radius:10px;margin-bottom:14px;box-shadow:0 2px 8px rgba(0,37,93,.15)}
header h1{font-size:22px;margin-bottom:4px}
header p{font-size:13px;opacity:.92}
.tabs{display:flex;gap:4px;margin-bottom:14px;border-bottom:2px solid var(--border)}
.tab{padding:10px 18px;background:none;border:none;font-size:14px;font-weight:500;
  color:var(--muted);cursor:pointer;border-bottom:3px solid transparent;margin-bottom:-2px;
  font-family:inherit}
.tab:hover{color:var(--pfd)}
.tab.active{color:var(--pfd);border-bottom-color:var(--pfizer);font-weight:600}
.tab-content{display:none}
.tab-content.active{display:block}

/* Cards & layout */
.panel{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:18px;margin-bottom:14px}
.panel h2{font-size:16px;color:var(--pfd);margin-bottom:8px}
.panel h3{font-size:13px;color:var(--pfd);margin:14px 0 6px;text-transform:uppercase;letter-spacing:.5px}
.panel p{font-size:13px;color:var(--text);margin-bottom:8px}
.panel ul{font-size:13px;margin-left:20px;margin-bottom:8px}
.panel ul li{margin-bottom:4px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
@media(max-width:1100px){.grid2,.grid3{grid-template-columns:1fr}}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.metric{background:#f8fafc;border:1px solid var(--border);border-radius:8px;padding:10px 12px}
.metric .label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.metric .value{font-size:20px;font-weight:700;color:var(--pfd);margin-top:2px}
.metric .delta{font-size:10px;color:var(--ok);margin-top:1px}

/* Segment cards in overview */
.seg-card{padding:14px;border-radius:8px;border-left:5px solid;background:#f8fafc}
.seg-A{border-left-color:var(--a)}
.seg-B{border-left-color:var(--b)}
.seg-C{border-left-color:var(--c)}
.seg-card .name{font-weight:700;font-size:14px;margin-bottom:2px}
.seg-card .pct{font-size:11px;color:var(--muted);margin-bottom:6px}
.seg-card .desc{font-size:12px}
.seg-card .action{font-size:11px;color:var(--muted);margin-top:4px;font-style:italic}

/* Cost asymmetry table */
.cost-row{display:grid;grid-template-columns:30px 1fr auto;gap:10px;padding:10px;
  border-radius:6px;margin-bottom:6px;font-size:13px;align-items:center}
.cost-bad{background:#fdebec}.cost-warn{background:#fdf6e8}.cost-ok{background:#eaf6ed}
.cost-icon{font-size:18px;text-align:center}
.cost-tag{font-weight:700;font-size:11px;padding:2px 8px;border-radius:4px;color:#fff}
.cost-tag.bad{background:var(--bad)}.cost-tag.warn{background:var(--warn)}
.cost-tag.ok{background:var(--ok)}

/* Plot containers */
.plot{width:100%;height:460px}

/* Toggles */
.controls{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;align-items:center}
.ctrl-label{font-size:12px;color:var(--muted);margin-right:4px}
.ctrl-btn{background:#fff;border:1px solid var(--border);border-radius:6px;
  padding:6px 12px;font-size:12px;cursor:pointer;color:var(--text);font-family:inherit}
.ctrl-btn:hover{border-color:var(--pfizer)}
.ctrl-btn.active{background:var(--pfizer);color:#fff;border-color:var(--pfizer)}

/* Legend */
.legend-row{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-bottom:8px}
.legend-item{display:inline-flex;align-items:center;gap:6px}
.swatch{width:12px;height:12px;border-radius:3px;display:inline-block}

/* Info boxes */
.info{background:#fffbe8;border:1px solid #f0d780;border-radius:8px;
  padding:10px 14px;font-size:12px;color:#6b5500;margin-bottom:14px}
.info.tip{background:#eaf3fb;border-color:#b8d8ed;color:#0a4775}

/* Doctor explorer */
.explorer{display:grid;grid-template-columns:1.1fr 1fr;gap:14px}
@media(max-width:1100px){.explorer{grid-template-columns:1fr}}
.search-box{width:100%;padding:10px 12px;border:1px solid var(--border);border-radius:6px;
  font-size:14px;font-family:inherit;margin-bottom:10px}
.search-box:focus{outline:none;border-color:var(--pfizer)}
.profile{min-height:300px}
.profile-empty{text-align:center;color:var(--muted);padding:60px 20px;font-size:13px;font-style:italic}
.profile-header{padding-bottom:10px;border-bottom:1px solid var(--border);margin-bottom:12px}
.profile-id{font-size:18px;font-weight:700;color:var(--pfd)}
.profile-tags{display:flex;gap:8px;margin-top:6px;flex-wrap:wrap}
.tag{padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}
.tag-true{background:#eef2f7;color:var(--text)}
.tag-pred{background:var(--pfizer);color:#fff}
.prob-bar{display:flex;height:24px;border-radius:4px;overflow:hidden;margin-top:10px;background:#e8edf3}
.prob-bar > div{display:flex;align-items:center;justify-content:center;color:#fff;
  font-size:11px;font-weight:700;transition:width 0.3s}

/* Feature comparison rows */
.feat-row{display:grid;grid-template-columns:160px 60px 1fr;gap:10px;align-items:center;
  padding:8px 0;border-bottom:1px solid #f0f3f8;font-size:12px}
.feat-name{font-weight:600;color:var(--text)}
.feat-name .desc{font-weight:400;color:var(--muted);font-size:10px;display:block;margin-top:1px}
.feat-value{font-weight:700;text-align:right;color:var(--pfd);font-size:14px;font-variant-numeric:tabular-nums}
.feat-bar-wrap{position:relative;height:18px;background:#f0f3f8;border-radius:3px;overflow:visible}
.feat-bar-segments{position:absolute;top:0;left:0;right:0;height:100%;display:flex}
.feat-bar-segment{height:100%;opacity:0.55}
.feat-bar-marker{position:absolute;top:-2px;width:3px;height:22px;background:#000;
  border-radius:1px;box-shadow:0 0 0 2px #fff}
.feat-legend{display:flex;gap:10px;font-size:10px;color:var(--muted);margin:4px 0 8px}

.explanation{background:#f0f7fc;border-left:3px solid var(--pfizer);padding:12px 14px;
  border-radius:4px;font-size:13px;line-height:1.55;margin-top:14px}

footer{text-align:center;font-size:11px;color:var(--muted);margin-top:18px;padding:14px}
</style></head><body>

<header>
  <h1>HCP Segmentation Dashboard — Velsipity</h1>
  <p>Ordinal classification model with business-calibrated decision rule · Capstone Project, Tec de Monterrey × Pfizer</p>
</header>

<div class="tabs">
  <button class="tab active" data-tab="overview">1 · Overview & Methodology</button>
  <button class="tab" data-tab="map">2 · Probability Map</button>
  <button class="tab" data-tab="explorer">3 · Doctor Explorer</button>
</div>

<!-- ================== TAB 1: OVERVIEW ================== -->
<div class="tab-content active" id="tab-overview">

  <div class="panel">
    <h2>The Business Problem</h2>
    <p>Pfizer launched <strong>Velsipity</strong>, an oral therapy for ulcerative colitis (FDA-approved Oct 2023). The commercial team needs to know — for each of <strong id="kpi_total">—</strong> healthcare providers (HCPs) — how likely they are to prescribe, so sales force time goes where it has the most impact.</p>
    <p>The original segmentation came from interviewing <strong>205 HCPs</strong> and was extrapolated to <strong id="kpi_labeled">—</strong> doctors. Another <strong id="kpi_unlabeled">—</strong> remained unlabeled (<em>untyped</em>). The capstone goal: build a model that scales the segmentation to the entire population.</p>
  </div>

  <div class="panel">
    <h2>The Three Segments (Ordinal: A &lt; B &lt; C)</h2>
    <div class="grid3" style="margin-top:8px">
      <div class="seg-card seg-A">
        <div class="name" style="color:var(--a)">SEG_A</div>
        <div class="pct"><span id="kpi_a_pct">—</span> · Will NOT prescribe</div>
        <div class="desc">Doctors with low UC prescribing activity and weak alignment with oral-therapy prescribing patterns.</div>
        <div class="action">Action: don't waste sales effort here</div>
      </div>
      <div class="seg-card seg-B">
        <div class="name" style="color:var(--b)">SEG_B</div>
        <div class="pct"><span id="kpi_b_pct">—</span> · Neutral / undecided</div>
        <div class="desc">Ambiguous middle ground. Could go either way depending on engagement and education.</div>
        <div class="action">Action: educate and convince</div>
      </div>
      <div class="seg-card seg-C">
        <div class="name" style="color:var(--c)">SEG_C</div>
        <div class="pct"><span id="kpi_c_pct">—</span> · Will prescribe</div>
        <div class="desc">Active UC prescribers, often already using oral therapies — Velsipity's natural target.</div>
        <div class="action">Action: top priority — never miss</div>
      </div>
    </div>
    <div class="info tip" style="margin-top:14px;margin-bottom:0">
      <strong>Why ordinal matters:</strong> the segments are not three unrelated classes. Confusing SEG_A with SEG_B is much smaller error than confusing SEG_A with SEG_C, because the underlying scale is "propensity to prescribe." The model treats it that way.
    </div>
  </div>

  <div class="panel">
    <h2>Cost of Errors — Why Plain Accuracy Is the Wrong Metric</h2>
    <p>Some classification errors are far more expensive to the business than others. The model is calibrated to minimize the worst kind first.</p>
    <div class="cost-row cost-bad">
      <div class="cost-icon">✕</div>
      <div><strong>Predicting SEG_A when true is SEG_C</strong> — A real prescriber was abandoned. Direct revenue loss; the doctor will keep prescribing competitor brands, unreached.</div>
      <div class="cost-tag bad">CATASTROPHIC</div>
    </div>
    <div class="cost-row cost-warn">
      <div class="cost-icon">!</div>
      <div><strong>Predicting SEG_C when true is SEG_A</strong> — Sales force visits a non-prescriber. Time wasted, but recoverable; the rep can de-prioritize after a few visits.</div>
      <div class="cost-tag warn">COSTLY</div>
    </div>
    <div class="cost-row cost-ok">
      <div class="cost-icon">~</div>
      <div><strong>Any error involving SEG_B</strong> — SEG_B is itself a residual / ambiguous bucket; errors here are inherent to the segmentation, not the model's fault.</div>
      <div class="cost-tag ok">ACCEPTABLE</div>
    </div>
    <div class="info" style="margin-top:14px;margin-bottom:0">
      <strong>The model is tuned to minimize the catastrophic error.</strong> It's willing to "over-call" SEG_C (more A→C false positives) so that real SEG_C doctors are rarely missed (fewer C→A misses).
    </div>
  </div>

  <div class="panel">
    <h2>The Model — Two-Stage Ordinal Classifier</h2>
    <p>The core model respects the ordinal structure A &lt; B &lt; C using two binary classifiers chained together:</p>
    <ul>
      <li><strong>Stage 1:</strong> P(≥B) — answers "is this doctor NOT a clear A?" (XGBoost)</li>
      <li><strong>Stage 2:</strong> P(≥C) — answers "is this doctor a strong C?" (XGBoost, with extra weight on C training examples)</li>
    </ul>
    <p>From these, the three probabilities are derived so that they always sum to 1 and are coherent with the ordering:</p>
    <ul>
      <li>P(A) = 1 − P(≥B)</li>
      <li>P(B) = P(≥B) − P(≥C)</li>
      <li>P(C) = P(≥C)</li>
    </ul>

    <h3>The Decision Rule (business-calibrated)</h3>
    <p>Once probabilities are computed, the segment is assigned by:</p>
    <pre style="background:#f8fafc;padding:10px;border-radius:6px;font-size:12px;font-family:Menlo,monospace;border:1px solid var(--border)">
if  P(A) >= 0.70   →   SEG_A    (only when very confident in "won't prescribe")
elif P(C) >= 0.30  →   SEG_C    (low bar — losing a real C is catastrophic)
else               →   SEG_B</pre>
    <p>The asymmetric thresholds (0.70 vs 0.30) directly encode the cost asymmetry above.</p>
  </div>

  <div class="panel">
    <h2>Why This Problem Is Hard</h2>
    <p>The internal Pfizer report acknowledges:</p>
    <ul>
      <li><strong>98.6% feature-space overlap</strong> between SEG_A and SEG_C</li>
      <li><strong>35.8% of cases are "inherently difficult"</strong></li>
      <li><strong>SEG_B is a "residual ambiguous category"</strong></li>
    </ul>
    <p>After testing 10 different strategies (temporal features, label-noise cleaning, hierarchical models, stacking, ordinal calibration), all approaches converged near the same ceiling: <strong>~58–60% balanced accuracy</strong>. Beyond that, model improvements run into the noise floor of the labels themselves — the labels were extrapolated from 205 interviewed doctors, and 17.3% of them are inconsistent with observed prescribing behavior.</p>
    <p>The model's value is therefore not just the prediction — it's the <strong>probability</strong>, which lets the business reason about confidence, identify ambiguous cases for manual review, and prioritize within segments.</p>
  </div>

  <div class="panel">
    <h2>Performance on Your Data</h2>
    <div class="metrics" style="margin-top:6px">
      <div class="metric"><div class="label">HCPs evaluated (CV)</div><div class="value" id="m_n_lab">—</div></div>
      <div class="metric"><div class="label">HCPs newly classified</div><div class="value" id="m_n_unlab">—</div></div>
      <div class="metric"><div class="label">Accuracy</div><div class="value" id="m_acc">—</div></div>
      <div class="metric"><div class="label">Balanced accuracy</div><div class="value" id="m_ba">—</div></div>
      <div class="metric"><div class="label">SEG_C recall</div><div class="value" id="m_rc">—</div><div class="delta">vs ~49% baseline</div></div>
      <div class="metric"><div class="label">SEG_C → A losses</div><div class="value" id="m_lost">—</div><div class="delta">vs ~24% baseline</div></div>
    </div>
    <p style="margin-top:14px;font-size:12px;color:var(--muted)">Metrics on labeled HCPs from 5-fold cross-validation (out-of-fold predictions — each doctor predicted by a model that did not see them during training).</p>
  </div>

</div>

<!-- ================== TAB 2: PROBABILITY MAP ================== -->
<div class="tab-content" id="tab-map">

  <div class="info">
    <strong>How to read this view.</strong> Each dot is one doctor. Position = the model's probability distribution over the three segments. Doctors near a vertex are confidently classified; doctors near the center are ambiguous (the "inherently difficult" cases). The two views show the same information — the ternary is the canonical representation, the 3D scatter is the same data with three explicit axes.
  </div>

  <div class="panel">
    <div class="controls">
      <span class="ctrl-label">Color by:</span>
      <button class="ctrl-btn active" data-color="true">Original label</button>
      <button class="ctrl-btn" data-color="pred">Model prediction</button>
      <button class="ctrl-btn" data-color="confidence">Confidence (max P)</button>
    </div>
    <div class="controls">
      <span class="ctrl-label">Show:</span>
      <button class="ctrl-btn active" data-show="all">All HCPs</button>
      <button class="ctrl-btn" data-show="labeled">With ground truth (labeled)</button>
      <button class="ctrl-btn" data-show="unlabeled">No ground truth (untyped)</button>
    </div>
    <div class="legend-row">
      <span class="legend-item"><span class="swatch" style="background:var(--a)"></span>SEG_A</span>
      <span class="legend-item"><span class="swatch" style="background:var(--b)"></span>SEG_B</span>
      <span class="legend-item"><span class="swatch" style="background:var(--c)"></span>SEG_C</span>
      <span class="legend-item"><span class="swatch" style="background:var(--u)"></span>No ground truth (only visible in "Original label" mode)</span>
    </div>
    <div class="info tip">
      <strong>About "No ground truth" doctors:</strong> these <span id="info_unlab">—</span> HCPs never had an original label — they're the <em>untyped</em> population. The model assigned them a predicted segment, which is exactly the project deliverable. In <em>"Original label"</em> mode they appear gray (no truth to color); in <em>"Model prediction"</em> mode they appear with their predicted color, just like any other doctor.
    </div>
  </div>

  <div class="grid2">
    <div class="panel">
      <h2>Probability Simplex (Ternary)</h2>
      <p style="font-size:12px;color:var(--muted);margin-bottom:8px">The mathematically canonical view. The three vertices are pure SEG_A, SEG_B, SEG_C. The center is maximum ambiguity.</p>
      <div id="plot_ternary" class="plot"></div>
    </div>
    <div class="panel">
      <h2>3D Scatter — P(A), P(B), P(C)</h2>
      <p style="font-size:12px;color:var(--muted);margin-bottom:8px">Same data with three explicit axes. All points lie on the plane P(A)+P(B)+P(C)=1.</p>
      <div id="plot_3d" class="plot"></div>
    </div>
  </div>
</div>

<!-- ================== TAB 3: DOCTOR EXPLORER ================== -->
<div class="tab-content" id="tab-explorer">

  <div class="info">
    <strong>How to use this view.</strong> Click any dot in the scatter (or search by HCP ID) to see why the model assigned its probabilities. The right panel compares the doctor's key metrics against the typical ranges of each segment, and produces a plain-language explanation.
  </div>

  <div class="explorer">
    <div class="panel">
      <h2>Find a Doctor</h2>
      <input type="text" class="search-box" id="search" placeholder="Search by HCP ID (e.g. 1234) — or click a dot in the scatter below">
      <div id="search_result" style="font-size:12px;color:var(--muted);margin-bottom:10px"></div>
      <h3 style="margin-top:6px">UC Rx vs Oral Rx (top discriminating features)</h3>
      <p style="font-size:11px;color:var(--muted);margin-bottom:6px">Axes are log(1+x). UC_TRX has the strongest signal (AUC 0.73). Click any point.</p>
      <div id="plot_scatter" class="plot" style="height:430px"></div>
      <div class="legend-row" style="margin-top:6px">
        <span class="legend-item"><span class="swatch" style="background:var(--a)"></span>Predicted SEG_A</span>
        <span class="legend-item"><span class="swatch" style="background:var(--b)"></span>Predicted SEG_B</span>
        <span class="legend-item"><span class="swatch" style="background:var(--c)"></span>Predicted SEG_C</span>
      </div>
    </div>

    <div class="panel">
      <h2>Doctor Profile</h2>
      <div id="profile" class="profile">
        <div class="profile-empty">Click a dot in the scatter, or search by HCP ID, to see a detailed profile and explanation of why the model predicted what it did.</div>
      </div>
    </div>
  </div>
</div>

<footer>
  Capstone Project · Tec de Monterrey × Pfizer Global Commercial Analytics · Predictions: out-of-fold for ground-truth doctors, forward-scored for untyped doctors
</footer>

<script>
const DATA = __DATA_PLACEHOLDER__;
const COLORS = {SEG_A:'#2e6cb0', SEG_B:'#f0a830', SEG_C:'#c83a3a', UNLABELED:'#9aa6b4'};

/* ---------- Tab switcher ---------- */
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('tab-' + t.dataset.tab).classList.add('active');
    // re-render plots in the activated tab to fix sizing
    if (t.dataset.tab === 'map') { plotTernary(); plot3D(); }
    if (t.dataset.tab === 'explorer') { plotScatter(); }
  });
});

/* ---------- Fill overview metrics ---------- */
const M = DATA.metrics;
document.getElementById('kpi_total').textContent = (M.n_labeled + M.n_unlabeled).toLocaleString();
document.getElementById('kpi_labeled').textContent = M.n_labeled.toLocaleString();
document.getElementById('kpi_unlabeled').textContent = M.n_unlabeled.toLocaleString();
document.getElementById('kpi_a_pct').textContent = (M.dist_A*100).toFixed(1) + '%';
document.getElementById('kpi_b_pct').textContent = (M.dist_B*100).toFixed(1) + '%';
document.getElementById('kpi_c_pct').textContent = (M.dist_C*100).toFixed(1) + '%';
document.getElementById('m_n_lab').textContent = M.n_labeled.toLocaleString();
document.getElementById('m_n_unlab').textContent = M.n_unlabeled.toLocaleString();
document.getElementById('m_acc').textContent = (M.accuracy*100).toFixed(1) + '%';
document.getElementById('m_ba').textContent = (M.balanced_accuracy*100).toFixed(1) + '%';
document.getElementById('m_rc').textContent = (M.recall_C*100).toFixed(1) + '%';
document.getElementById('m_lost').textContent = (M.c_lost_pct*100).toFixed(1) + '%';
document.getElementById('info_unlab').textContent = M.n_unlabeled.toLocaleString();

/* ---------- Probability map state ---------- */
const VIZ = DATA.viz;       // subsampled doctors used in the plots
const ALL = DATA.all;       // full doctor lookup map  {id: {pa,pb,pc,t,p,is_l, feats:[...]}}
const FEATS = DATA.features; // [{key,name,desc}, ...] in order
const STATS = DATA.segment_stats;  // {feat_key: {SEG_A:{p25,median,p75}, ...}}

let colorMode='true', showMode='all';
function getMask(){const lb=VIZ.is_labeled;
  if(showMode==='all') return lb.map(()=>true);
  if(showMode==='labeled') return lb;
  return lb.map(x=>!x)}
function getColors(mode){
  if(mode==='true') return VIZ.true.map((s,i)=>VIZ.is_labeled[i]?(COLORS[s]||COLORS.UNLABELED):COLORS.UNLABELED);
  if(mode==='pred') return VIZ.pred.map(s=>COLORS[s]);
  return VIZ.P_A.map((_,i)=>Math.max(VIZ.P_A[i],VIZ.P_B[i],VIZ.P_C[i]))}
function filt(arr,mask){return arr.filter((_,i)=>mask[i])}
const HOVER = VIZ.hcp_id.map((_,i)=>{
  const t = VIZ.is_labeled[i] ? VIZ.true[i] : '(untyped)';
  return `HCP ${VIZ.hcp_id[i]}<br>Original: <b>${t}</b> · Predicted: <b>${VIZ.pred[i]}</b><br>` +
    `P(A)=${(VIZ.P_A[i]*100).toFixed(1)}% · P(B)=${(VIZ.P_B[i]*100).toFixed(1)}% · P(C)=${(VIZ.P_C[i]*100).toFixed(1)}%`});

function plotTernary(){
  const m=getMask(), c=getColors(colorMode), isC=colorMode==='confidence';
  Plotly.react('plot_ternary',[{type:'scatterternary',mode:'markers',
    a:filt(VIZ.P_A,m), b:filt(VIZ.P_B,m), c:filt(VIZ.P_C,m),
    text:filt(HOVER,m), hoverinfo:'text',
    marker:{size:5, color:filt(c,m),
      colorscale: isC?'Viridis':undefined, cmin:isC?0.33:undefined, cmax:isC?1:undefined,
      showscale:isC, opacity:0.7, line:{width:0.3,color:'rgba(0,0,0,0.15)'}}}],
    {ternary:{sum:1,
      aaxis:{title:'P(SEG_A)',min:0}, baxis:{title:'P(SEG_B)',min:0}, caxis:{title:'P(SEG_C)',min:0},
      bgcolor:'#fafbfd'},
     margin:{l:50,r:30,t:20,b:30}, showlegend:false},
    {responsive:true, displaylogo:false});
}
function plot3D(){
  const m=getMask(), c=getColors(colorMode), isC=colorMode==='confidence';
  Plotly.react('plot_3d',[{type:'scatter3d',mode:'markers',
    x:filt(VIZ.P_A,m), y:filt(VIZ.P_B,m), z:filt(VIZ.P_C,m),
    text:filt(HOVER,m), hoverinfo:'text',
    marker:{size:3, color:filt(c,m),
      colorscale:isC?'Viridis':undefined, cmin:isC?0.33:undefined, cmax:isC?1:undefined, opacity:0.8}}],
    {scene:{xaxis:{title:'P(A)',range:[0,1]}, yaxis:{title:'P(B)',range:[0,1]}, zaxis:{title:'P(C)',range:[0,1]},
            camera:{eye:{x:1.4,y:1.4,z:1}}},
     margin:{l:0,r:0,t:0,b:0}},
    {responsive:true, displaylogo:false});
}
document.querySelectorAll('.ctrl-btn[data-color]').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('.ctrl-btn[data-color]').forEach(x=>x.classList.remove('active'));
  b.classList.add('active'); colorMode=b.dataset.color; plotTernary(); plot3D();
}));
document.querySelectorAll('.ctrl-btn[data-show]').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('.ctrl-btn[data-show]').forEach(x=>x.classList.remove('active'));
  b.classList.add('active'); showMode=b.dataset.show; plotTernary(); plot3D();
}));

/* ---------- Doctor explorer ---------- */
function plotScatter(){
  const colors = VIZ.pred.map(s => COLORS[s]);
  Plotly.react('plot_scatter',[{type:'scattergl', mode:'markers',
    x: VIZ.log_uc_trx, y: VIZ.log_oral_trx,
    text: HOVER, hoverinfo:'text', customdata: VIZ.hcp_id,
    marker:{size:6, color:colors, opacity:0.6, line:{width:0.3,color:'rgba(0,0,0,0.2)'}}}],
    {xaxis:{title:'log(1 + UC_TRX_sum)', gridcolor:'#eef2f7'},
     yaxis:{title:'log(1 + ORAL_TRX_sum)', gridcolor:'#eef2f7'},
     margin:{l:50,r:20,t:10,b:40}, plot_bgcolor:'#fafbfd', hovermode:'closest'},
    {responsive:true, displaylogo:false});
  document.getElementById('plot_scatter').on('plotly_click', ev=>{
    showProfile(ev.points[0].customdata);
  });
}

/* Compare a value v to segment quartiles. Returns {seg, position}
   where position is one of "low", "typical", "high" relative to that seg. */
function classifyFeatureValue(v, statsForFeat){
  // For each segment, check whether v is below/within/above its [P25, P75] range
  const results = {};
  for (const seg of ['SEG_A','SEG_B','SEG_C']) {
    const s = statsForFeat[seg];
    if (v < s.p25) results[seg] = 'below';
    else if (v > s.p75) results[seg] = 'above';
    else results[seg] = 'within';
  }
  return results;
}

/* For each feature, find the segment whose median is closest to v */
function closestSegment(v, statsForFeat){
  let best=null, bestD=Infinity;
  for (const seg of ['SEG_A','SEG_B','SEG_C']) {
    const d = Math.abs(v - statsForFeat[seg].median);
    if (d < bestD) { bestD = d; best = seg; }
  }
  return best;
}

function fmtNum(v){
  if (v === null || v === undefined) return '—';
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

function buildExplanation(doctor){
  const parts = [];
  const closestVotes = {SEG_A:0, SEG_B:0, SEG_C:0};
  const featDescriptions = [];

  FEATS.forEach((f, i) => {
    const v = doctor.f[i];
    const stats = STATS[f.key];
    if (!stats) return;
    const cls = classifyFeatureValue(v, stats);
    const closest = closestSegment(v, stats);
    closestVotes[closest]++;

    // Build a sentence about this feature
    let segLabel;
    if (cls.SEG_A === 'within' && cls.SEG_C === 'within') {
      segLabel = 'overlaps with both SEG_A and SEG_C ranges (ambiguous zone)';
    } else if (cls.SEG_C === 'within' || cls.SEG_C === 'above') {
      segLabel = `is in the typical range of SEG_C (median: ${fmtNum(stats.SEG_C.median)})`;
    } else if (cls.SEG_A === 'within' || cls.SEG_A === 'below') {
      segLabel = `is in the typical range of SEG_A (median: ${fmtNum(stats.SEG_A.median)})`;
    } else {
      segLabel = `falls in the SEG_B range (median: ${fmtNum(stats.SEG_B.median)})`;
    }
    featDescriptions.push(`<strong>${f.name}</strong> = ${fmtNum(v)} — ${segLabel}.`);
  });

  // Determine dominant predicted segment
  const probs = {SEG_A: doctor.pa, SEG_B: doctor.pb, SEG_C: doctor.pc};
  const topSeg = Object.keys(probs).reduce((a,b)=>probs[a]>probs[b]?a:b);

  // Lead sentence
  let lead;
  const maxP = Math.max(doctor.pa, doctor.pb, doctor.pc);
  if (maxP >= 0.70) {
    lead = `The model is <strong>highly confident</strong> this HCP belongs to <strong style="color:${COLORS[topSeg]}">${topSeg}</strong> (P = ${(maxP*100).toFixed(0)}%).`;
  } else if (maxP >= 0.50) {
    lead = `The model leans toward <strong style="color:${COLORS[topSeg]}">${topSeg}</strong> (P = ${(maxP*100).toFixed(0)}%), but with meaningful uncertainty across other segments.`;
  } else {
    lead = `This HCP is in the <strong>ambiguous zone</strong> — no single segment has more than 50% probability. The model's best guess is <strong style="color:${COLORS[topSeg]}">${topSeg}</strong> (P = ${(maxP*100).toFixed(0)}%), but a manual review is recommended.`;
  }

  // Decision rule reasoning
  let ruleNote;
  if (doctor.pa >= 0.70) {
    ruleNote = `The decision rule classified this doctor as <strong>SEG_A</strong> because P(A) = ${(doctor.pa*100).toFixed(0)}% ≥ 70% threshold.`;
  } else if (doctor.pc >= 0.30) {
    ruleNote = `The decision rule classified this doctor as <strong>SEG_C</strong> because P(C) = ${(doctor.pc*100).toFixed(0)}% ≥ 30% threshold (low bar by design — losing a real prescriber is the most costly error).`;
  } else {
    ruleNote = `The decision rule defaulted to <strong>SEG_B</strong> because neither P(A) ≥ 70% nor P(C) ≥ 30% — the doctor doesn't have enough signal in either direction.`;
  }

  return `<p>${lead}</p>
<p style="margin-top:8px">${ruleNote}</p>
<p style="margin-top:10px"><strong>Feature-by-feature:</strong></p>
<ul style="margin-top:4px;margin-left:20px">${featDescriptions.map(d=>`<li>${d}</li>`).join('')}</ul>`;
}

function buildFeatureRow(f, value, stats){
  // Compute axis range for visualization: 0 to max(P75 of all segs) * 1.3
  const maxRef = Math.max(stats.SEG_A.p75, stats.SEG_B.p75, stats.SEG_C.p75) * 1.3 + 0.01;
  // Three colored bands showing each segment's IQR
  const bandPct = seg => {
    const s = stats[seg];
    const left  = Math.max(0, Math.min(100, s.p25 / maxRef * 100));
    const right = Math.max(0, Math.min(100, s.p75 / maxRef * 100));
    return {left, width: Math.max(0, right - left)};
  };
  const bA = bandPct('SEG_A'), bB = bandPct('SEG_B'), bC = bandPct('SEG_C');
  // Doctor's value as marker
  const markerPct = Math.max(0, Math.min(100, value / maxRef * 100));

  return `
    <div class="feat-row">
      <div class="feat-name">${f.name}<span class="desc">${f.desc}</span></div>
      <div class="feat-value">${fmtNum(value)}</div>
      <div>
        <div class="feat-bar-wrap">
          <div class="feat-bar-segments">
            <div class="feat-bar-segment" style="margin-left:${bA.left}%;width:${bA.width}%;background:${COLORS.SEG_A};position:absolute;top:0"></div>
            <div class="feat-bar-segment" style="margin-left:${bB.left}%;width:${bB.width}%;background:${COLORS.SEG_B};position:absolute;top:0"></div>
            <div class="feat-bar-segment" style="margin-left:${bC.left}%;width:${bC.width}%;background:${COLORS.SEG_C};position:absolute;top:0"></div>
          </div>
          <div class="feat-bar-marker" style="left:calc(${markerPct}% - 1.5px)" title="This doctor: ${fmtNum(value)}"></div>
        </div>
      </div>
    </div>`;
}

function showProfile(hcpId){
  const id = String(hcpId);
  const d = ALL[id];
  if (!d) {
    document.getElementById('profile').innerHTML =
      '<div class="profile-empty">HCP not found.</div>';
    return;
  }
  const tagTrue = d.is_l ? `<span class="tag tag-true">Original: ${d.t}</span>` : '<span class="tag tag-true">No ground truth (untyped)</span>';
  const tagPred = `<span class="tag tag-pred">Predicted: ${d.p}</span>`;
  const pa = (d.pa*100).toFixed(0), pb = (d.pb*100).toFixed(0), pc = (d.pc*100).toFixed(0);

  const featRows = FEATS.map((f, i) => {
    const stats = STATS[f.key];
    if (!stats) return '';
    return buildFeatureRow(f, d.f[i], stats);
  }).join('');

  document.getElementById('profile').innerHTML = `
    <div class="profile-header">
      <div class="profile-id">HCP ${id}</div>
      <div class="profile-tags">${tagTrue}${tagPred}</div>
    </div>

    <div style="font-size:11px;color:var(--muted);margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px">Probability Distribution</div>
    <div class="prob-bar">
      <div style="background:${COLORS.SEG_A};width:${pa}%">${pa>10?'A '+pa+'%':''}</div>
      <div style="background:${COLORS.SEG_B};width:${pb}%">${pb>10?'B '+pb+'%':''}</div>
      <div style="background:${COLORS.SEG_C};width:${pc}%">${pc>10?'C '+pc+'%':''}</div>
    </div>

    <h3>Key Features vs Segment Ranges</h3>
    <div class="feat-legend">
      <span class="legend-item"><span class="swatch" style="background:var(--a);opacity:0.55"></span>SEG_A IQR</span>
      <span class="legend-item"><span class="swatch" style="background:var(--b);opacity:0.55"></span>SEG_B IQR</span>
      <span class="legend-item"><span class="swatch" style="background:var(--c);opacity:0.55"></span>SEG_C IQR</span>
      <span class="legend-item"><span class="swatch" style="background:#000"></span>This doctor</span>
    </div>
    <div>${featRows}</div>

    <div class="explanation">
      <div style="font-weight:700;color:var(--pfd);margin-bottom:6px">Why this prediction?</div>
      ${buildExplanation(d)}
    </div>
  `;
}

/* Search */
document.getElementById('search').addEventListener('input', e => {
  const q = e.target.value.trim();
  const result = document.getElementById('search_result');
  if (!q) { result.textContent = ''; return; }
  if (ALL[q]) {
    result.textContent = '';
    showProfile(q);
  } else {
    result.textContent = `No HCP found with ID "${q}"`;
  }
});

/* Initial render */
plotTernary(); plot3D(); plotScatter();
</script>
</body></html>'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--label-col", default="ATSEG_first")
    p.add_argument("--id-col", default="NUEVO_ID")
    p.add_argument("--output-html", default="hcp_dashboard.html")
    p.add_argument("--output-csv", default="hcp_predictions.csv")
    p.add_argument("--output-ambiguous", default="ambiguous_hcps.csv")
    p.add_argument("--ambiguous-threshold", type=float, default=0.5)
    p.add_argument("--max-html-points", type=int, default=4000)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--exclude-cols", nargs="*",
                   default=["WEEK_ID_first", "WEEK_ID_last", "WEEK_ID_count"])
    args = p.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  Shape: {df.shape}")
    print(f"  Label column: '{args.label_col}'")
    print(f"  ID column:    '{args.id_col}'")

    if args.label_col not in df.columns:
        sys.exit(f"  ERROR: column '{args.label_col}' not found")
    if args.id_col not in df.columns:
        sys.exit(f"  ERROR: column '{args.id_col}' not found")

    df[args.label_col] = df[args.label_col].astype(str)
    is_labeled = df[args.label_col].isin(VALID_LABELS).values
    n_labeled = is_labeled.sum()
    n_unlabeled = (~is_labeled).sum()
    print(f"\n  Labeled (ground truth): {n_labeled:>6,}")
    print(f"  Unlabeled (to score):   {n_unlabeled:>6,}")
    if n_labeled == 0:
        sys.exit("  ERROR: no HCPs with SEG_A/SEG_B/SEG_C labels")

    print(f"\n  Label distribution:")
    dist = {}
    for lab in VALID_LABELS:
        n = (df[args.label_col] == lab).sum()
        dist[lab] = n / n_labeled
        print(f"    {lab}: {n:>6,} ({dist[lab]*100:5.1f}%)")

    # Save key feature copies BEFORE any transformation (for the explorer)
    raw_feature_values = {}
    for fkey, _, _ in KEY_FEATURES:
        if fkey in df.columns:
            raw_feature_values[fkey] = df[fkey].values.copy()
        else:
            print(f"  WARNING: key feature '{fkey}' not found — skipping in explorer")

    # Build feature matrix
    print("\nGenerating derived features...")
    df = add_ratios(df)
    drop_cols = set([args.label_col, args.id_col] + list(args.exclude_cols))
    feat_cols = [c for c in df.columns
                 if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  Total features used: {len(feat_cols)}")

    X_all = df[feat_cols].fillna(0).values
    y_all = df[args.label_col].values
    ids_all = df[args.id_col].values

    X_lab = X_all[is_labeled]
    y_lab = y_all[is_labeled]
    X_unlab = X_all[~is_labeled]

    # OOF predictions on labeled
    print(f"\n[1/4] Out-of-fold CV on labeled HCPs ({args.n_folds} folds)...")
    P_A_lab, P_B_lab, P_C_lab = oof_predictions(X_lab, y_lab, n_splits=args.n_folds)
    pred_lab = decide(P_A_lab, P_C_lab)

    acc = (pred_lab == y_lab).mean()
    bal_acc = balanced_accuracy_score(y_lab, pred_lab)
    mask_C = y_lab == "SEG_C"
    recall_C = (pred_lab[mask_C] == "SEG_C").mean() if mask_C.sum() > 0 else 0
    c_lost_pct = (((y_lab == "SEG_C") & (pred_lab == "SEG_A")).sum() /
                  max(mask_C.sum(), 1))

    print(f"\n  Accuracy:           {acc:.4f}")
    print(f"  Balanced accuracy:  {bal_acc:.4f}")
    print(f"  SEG_C recall:       {recall_C:.4f}")
    print(f"  SEG_C → A losses:   {c_lost_pct:.4f}")
    print("\n  Confusion matrix:")
    cm = confusion_matrix(y_lab, pred_lab, labels=list(VALID_LABELS))
    print(pd.DataFrame(cm,
        index=[f"{l}_true" for l in VALID_LABELS],
        columns=[f"{l}_pred" for l in VALID_LABELS]))

    # Final model on all labeled, score unlabeled
    if n_unlabeled > 0:
        print(f"\n[2/4] Training final model on {n_labeled:,} labeled HCPs...")
        m1, m2 = fit_ordinal(X_lab, y_lab)
        print(f"      Scoring {n_unlabeled:,} unlabeled HCPs...")
        P_A_unlab, P_B_unlab, P_C_unlab = predict_ordinal(m1, m2, X_unlab)
        pred_unlab = decide(P_A_unlab, P_C_unlab)
    else:
        P_A_unlab = P_B_unlab = P_C_unlab = np.array([])
        pred_unlab = np.array([])

    # Reassemble in original order
    P_A = np.zeros(len(df)); P_B = np.zeros(len(df)); P_C = np.zeros(len(df))
    pred = np.empty(len(df), dtype=object)
    P_A[is_labeled] = P_A_lab; P_B[is_labeled] = P_B_lab; P_C[is_labeled] = P_C_lab
    pred[is_labeled] = pred_lab
    if n_unlabeled > 0:
        P_A[~is_labeled] = P_A_unlab; P_B[~is_labeled] = P_B_unlab; P_C[~is_labeled] = P_C_unlab
        pred[~is_labeled] = pred_unlab

    max_p = np.maximum.reduce([P_A, P_B, P_C])
    ambiguous = max_p < args.ambiguous_threshold

    # ----- Segment statistics for explainer -----
    print("\n[3/4] Computing segment statistics for explanation engine...")
    df_lab = df[is_labeled]
    available_features = [(k,n,d) for k,n,d in KEY_FEATURES if k in raw_feature_values]
    df_lab_keys = pd.DataFrame({k: raw_feature_values[k][is_labeled] for k,_,_ in available_features})
    df_lab_keys[args.label_col] = y_lab
    seg_stats = compute_segment_stats(df_lab_keys, args.label_col,
                                       [k for k,_,_ in available_features])

    # ----- CSV outputs -----
    print("\n[4/4] Writing outputs...")
    out_df = pd.DataFrame({
        args.id_col: ids_all,
        "true_segment": y_all,
        "was_labeled": is_labeled,
        "predicted_segment": pred,
        "P_A": P_A, "P_B": P_B, "P_C": P_C,
        "max_prob": max_p,
        "ambiguous": ambiguous,
    })
    out_df.to_csv(args.output_csv, index=False)
    print(f"  ✓ {args.output_csv}  ({len(out_df):,} rows)")

    amb_df = out_df[out_df["ambiguous"]].sort_values("max_prob")
    amb_df.to_csv(args.output_ambiguous, index=False)
    print(f"  ✓ {args.output_ambiguous}  ({len(amb_df):,} HCPs flagged for review)")

    # ----- Build viz subset (50/50 labeled/unlabeled when possible) -----
    rng = np.random.default_rng(0)
    if len(df) > args.max_html_points:
        keep_idx = []
        n_lab_target = args.max_html_points // 2
        n_unlab_target = args.max_html_points - n_lab_target
        # stratified within labeled
        for cls in VALID_LABELS:
            cls_idx = np.where(y_all == cls)[0]
            n_keep = int(n_lab_target * (len(cls_idx) / max(n_labeled, 1)))
            keep_idx.extend(rng.choice(cls_idx, size=min(n_keep, len(cls_idx)), replace=False))
        # unlabeled
        unlab_idx = np.where(~is_labeled)[0]
        if len(unlab_idx) > 0:
            keep_idx.extend(rng.choice(unlab_idx,
                size=min(n_unlab_target, len(unlab_idx)), replace=False))
        keep_idx = np.array(keep_idx)
    else:
        keep_idx = np.arange(len(df))

    # log-transformed UC_TRX, ORAL_TRX for the interpretable scatter
    uc_trx = raw_feature_values.get("UC_TRX_sum", np.zeros(len(df)))
    oral_trx = raw_feature_values.get("ORAL_TRX_sum", np.zeros(len(df)))
    log_uc = np.log1p(np.maximum(uc_trx, 0))
    log_oral = np.log1p(np.maximum(oral_trx, 0))

    viz = {
        "hcp_id":     [str(x) for x in ids_all[keep_idx]],
        "P_A":        [round(float(x),4) for x in P_A[keep_idx]],
        "P_B":        [round(float(x),4) for x in P_B[keep_idx]],
        "P_C":        [round(float(x),4) for x in P_C[keep_idx]],
        "true":       [str(x) for x in y_all[keep_idx]],
        "pred":       [str(x) for x in pred[keep_idx]],
        "is_labeled": [bool(x) for x in is_labeled[keep_idx]],
        "log_uc_trx": [round(float(x),3) for x in log_uc[keep_idx]],
        "log_oral_trx":[round(float(x),3) for x in log_oral[keep_idx]],
    }

    # ----- Full doctor lookup map (compact) -----
    print("  Building per-doctor lookup table...")
    all_lookup = {}
    feat_keys = [k for k,_,_ in available_features]
    feat_arrays = [raw_feature_values[k] for k in feat_keys]
    for i, hid in enumerate(ids_all):
        all_lookup[str(hid)] = {
            "pa": round(float(P_A[i]), 4),
            "pb": round(float(P_B[i]), 4),
            "pc": round(float(P_C[i]), 4),
            "t":  str(y_all[i]) if is_labeled[i] else "",
            "p":  str(pred[i]),
            "is_l": bool(is_labeled[i]),
            "f":  [round(float(arr[i]), 3) for arr in feat_arrays],
        }

    payload = {
        "viz": viz,
        "all": all_lookup,
        "features": [{"key":k, "name":n, "desc":d} for k,n,d in available_features],
        "segment_stats": seg_stats,
        "metrics": {
            "n_labeled": int(n_labeled),
            "n_unlabeled": int(n_unlabeled),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "recall_C": float(recall_C),
            "c_lost_pct": float(c_lost_pct),
            "dist_A": float(dist.get("SEG_A", 0)),
            "dist_B": float(dist.get("SEG_B", 0)),
            "dist_C": float(dist.get("SEG_C", 0)),
        },
    }

    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", json.dumps(payload))
    Path(args.output_html).write_text(html, encoding="utf-8")
    size_mb = Path(args.output_html).stat().st_size / 1024 / 1024
    print(f"  ✓ {args.output_html}  ({size_mb:.2f} MB · {len(keep_idx):,} viz points · {len(all_lookup):,} searchable HCPs)")
    print(f"\nDone. Open {args.output_html} in any modern browser.")


if __name__ == "__main__":
    main()