"""
build_probability_visualization_v4.py
=====================================
HCP segmentation dashboard with TWO models compared side-by-side:

  MODEL 1 — "Argmax Baseline"
    XGBClassifier(multi:softprob, balanced sample_weight)
    Decision: argmax(P_A, P_B, P_C)
    Optimizes for: overall accuracy
    Result: ~64% accuracy / ~49% recall_C / ~24% C-A losses

  MODEL 2 — "Business-Calibrated Ordinal" (the capstone's chosen model)
    Two-stage XGBoost ordinal: P(>=B) and P(>=C), with sample_weight=2 for C
    Decision: P(A)>=0.70 -> A;  elif P(C)>=0.30 -> C;  else B
    Optimizes for: minimizing the catastrophic C->A error
    Result: ~58% accuracy / ~57% recall_C / ~17% C-A losses

The dashboard has TWO tabs:
  Tab 1 — Overview: explains both models, shows side-by-side comparison
  Tab 2 — Probability Map + Doctor Explorer: with a live model toggle

USAGE:
    python build_probability_visualization_v4.py --input doctors_aggregated.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

try:
    import xgboost as xgb
except ImportError:
    sys.exit("xgboost not installed.  Run: pip install xgboost")


VALID_LABELS = ("SEG_A", "SEG_B", "SEG_C")

# Hyperparameters of the BUSINESS-CALIBRATED ORDINAL model (capstone final)
ORDINAL_PARAMS = dict(
    n_estimators=800, max_depth=5, learning_rate=0.04,
    subsample=0.85, colsample_bytree=0.7, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
    eval_metric="logloss", random_state=42, n_jobs=-1,
)
THR_A, THR_C = 0.70, 0.30
SEG_C_SAMPLE_WEIGHT = 2.0

# Hyperparameters of the ARGMAX BASELINE (matches user's reference code)
ARGMAX_PARAMS = dict(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    objective="multi:softprob", num_class=3,
    random_state=42, n_jobs=-1, eval_metric="mlogloss",
)

# Features used for the per-doctor profile in the explorer
KEY_FEATURES = [
    ("UC_TRX_sum",    "UC prescriptions (TRx)",     "Total Rx for ulcerative colitis"),
    ("UC_NRX_sum",    "UC new prescriptions (NRx)", "Patients newly starting UC therapy"),
    ("ORAL_TRX_sum",  "Oral therapy Rx",            "Oral UC treatments — Velsipity's category"),
    ("IL23_TRX_sum",  "IL-23 inhibitor Rx",         "Competitor biologics for UC"),
    ("DETAILS_sum",   "Sales detail visits",        "Pfizer rep visits to this HCP"),
    ("TOTAL_TRX_sum", "Total prescription volume",  "Overall prescribing activity"),
]


# ============================== MODEL 2: ORDINAL ==============================
def fit_ordinal(X_train, y_train):
    y_geq_b = (y_train != "SEG_A").astype(int)
    y_geq_c = (y_train == "SEG_C").astype(int)
    m1 = xgb.XGBClassifier(**ORDINAL_PARAMS); m1.fit(X_train, y_geq_b)
    sw = np.where(y_geq_c == 1, SEG_C_SAMPLE_WEIGHT, 1.0)
    m2 = xgb.XGBClassifier(**ORDINAL_PARAMS); m2.fit(X_train, y_geq_c, sample_weight=sw)
    return m1, m2


def predict_ordinal(m1, m2, X):
    p_b = m1.predict_proba(X)[:, 1]
    p_c = np.minimum(m2.predict_proba(X)[:, 1], p_b)
    P_A, P_B, P_C = 1 - p_b, p_b - p_c, p_c
    s = P_A + P_B + P_C
    P_A, P_B, P_C = P_A / s, P_B / s, P_C / s
    pred = np.where(P_A >= THR_A, "SEG_A",
           np.where(P_C >= THR_C, "SEG_C", "SEG_B"))
    return P_A, P_B, P_C, pred


# ============================== MODEL 1: ARGMAX ==============================
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


# ============================== OOF for both ==============================
def oof_both(X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n = len(y)
    out = {
        "ord": {"P_A": np.zeros(n), "P_B": np.zeros(n), "P_C": np.zeros(n),
                "pred": np.empty(n, dtype=object)},
        "amx": {"P_A": np.zeros(n), "P_B": np.zeros(n), "P_C": np.zeros(n),
                "pred": np.empty(n, dtype=object)},
    }
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"    fold {fold}/{n_splits}: ordinal...", end=" ", flush=True)
        m1, m2 = fit_ordinal(X[tr], y[tr])
        pa, pb, pc, pr = predict_ordinal(m1, m2, X[va])
        out["ord"]["P_A"][va] = pa; out["ord"]["P_B"][va] = pb
        out["ord"]["P_C"][va] = pc; out["ord"]["pred"][va] = pr

        print("argmax...", flush=True)
        m, le = fit_argmax(X[tr], y[tr])
        pa, pb, pc, pr = predict_argmax(m, le, X[va])
        out["amx"]["P_A"][va] = pa; out["amx"]["P_B"][va] = pb
        out["amx"]["P_C"][va] = pc; out["amx"]["pred"][va] = pr
    return out


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


def metrics_for(y_true, pred):
    """Compute key metrics for a model."""
    acc = float((pred == y_true).mean())
    bal_acc = float(balanced_accuracy_score(y_true, pred))
    mask_C = y_true == "SEG_C"
    recall_C = float((pred[mask_C] == "SEG_C").mean()) if mask_C.sum() > 0 else 0.0
    c_lost_pct = float(((y_true == "SEG_C") & (pred == "SEG_A")).sum() / max(mask_C.sum(), 1))
    cm = confusion_matrix(y_true, pred, labels=list(VALID_LABELS))
    return {
        "accuracy": acc, "balanced_accuracy": bal_acc,
        "recall_C": recall_C, "c_lost_pct": c_lost_pct,
        "confusion_matrix": cm.tolist(),
    }


def compute_segment_stats(df_lab, label_col, key_feature_cols):
    stats = {}
    for feat in key_feature_cols:
        if feat not in df_lab.columns:
            continue
        per_seg = {}
        for seg in VALID_LABELS:
            vals = df_lab.loc[df_lab[label_col] == seg, feat].values
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
  --m1:#7c4dff;  /* argmax model accent */
  --m2:#0093D0;  /* ordinal model accent */
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

/* Segment cards */
.seg-card{padding:14px;border-radius:8px;border-left:5px solid;background:#f8fafc}
.seg-A{border-left-color:var(--a)}
.seg-B{border-left-color:var(--b)}
.seg-C{border-left-color:var(--c)}
.seg-card .name{font-weight:700;font-size:14px;margin-bottom:2px}
.seg-card .pct{font-size:11px;color:var(--muted);margin-bottom:6px}
.seg-card .desc{font-size:12px}
.seg-card .action{font-size:11px;color:var(--muted);margin-top:4px;font-style:italic}

/* Cost asymmetry */
.cost-row{display:grid;grid-template-columns:30px 1fr auto;gap:10px;padding:10px;
  border-radius:6px;margin-bottom:6px;font-size:13px;align-items:center}
.cost-bad{background:#fdebec}.cost-warn{background:#fdf6e8}.cost-ok{background:#eaf6ed}
.cost-icon{font-size:18px;text-align:center}
.cost-tag{font-weight:700;font-size:11px;padding:2px 8px;border-radius:4px;color:#fff}
.cost-tag.bad{background:var(--bad)}.cost-tag.warn{background:var(--warn)}
.cost-tag.ok{background:var(--ok)}

/* Model comparison cards */
.model-card{padding:16px;border-radius:8px;border:2px solid;background:#f8fafc;position:relative}
.model-card.m1{border-color:var(--m1)}
.model-card.m2{border-color:var(--m2)}
.model-card .badge{position:absolute;top:-10px;right:14px;padding:3px 10px;
  font-size:10px;font-weight:700;color:#fff;border-radius:4px;letter-spacing:.5px}
.m1 .badge{background:var(--m1)}.m2 .badge{background:var(--m2)}
.model-card h3{margin-top:0;font-size:14px;text-transform:none;letter-spacing:0}
.model-card .subt{font-size:11px;color:var(--muted);margin-bottom:10px;font-style:italic}
.model-metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:10px}
.model-metrics .m{background:#fff;border:1px solid var(--border);border-radius:6px;padding:8px}
.model-metrics .m .l{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.model-metrics .m .v{font-size:18px;font-weight:700;color:var(--pfd);margin-top:2px}

/* Confusion matrix */
.cm-table{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}
.cm-table th, .cm-table td{padding:8px;text-align:center;border:1px solid var(--border)}
.cm-table th{background:#f4f6fa;font-weight:600;color:var(--pfd)}
.cm-table td.diag{font-weight:700}
.cm-table .row-label{background:#f4f6fa;font-weight:600;color:var(--pfd);text-align:left;padding-left:12px}
.cm-table .col-label{font-size:10px;text-transform:uppercase;letter-spacing:.5px}

/* Trade-off arrows */
.tradeoff{display:grid;grid-template-columns:1fr auto 1fr;gap:14px;align-items:center;
  background:#fffbe8;border:1px solid #f0d780;border-radius:8px;padding:14px;margin-top:14px}
.tradeoff .arrow{font-size:20px;color:#cc7a00;font-weight:700}
.tradeoff .col{font-size:13px}
.tradeoff strong{color:var(--pfd)}

/* Plot containers */
.plot{width:100%;height:460px}

/* Toggles */
.controls{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;align-items:center}
.ctrl-label{font-size:12px;color:var(--muted);margin-right:4px}
.ctrl-btn{background:#fff;border:1px solid var(--border);border-radius:6px;
  padding:6px 12px;font-size:12px;cursor:pointer;color:var(--text);font-family:inherit}
.ctrl-btn:hover{border-color:var(--pfizer)}
.ctrl-btn.active{background:var(--pfizer);color:#fff;border-color:var(--pfizer)}

/* Big model toggle (live) */
.model-toggle{background:#fff;border:1px solid var(--border);border-radius:10px;
  padding:14px 18px;margin-bottom:14px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.model-toggle .label{font-size:13px;font-weight:600;color:var(--pfd)}
.model-toggle .desc{font-size:11px;color:var(--muted)}
.model-btn{padding:10px 16px;border:2px solid var(--border);border-radius:8px;background:#fff;
  cursor:pointer;font-size:13px;font-family:inherit;display:flex;flex-direction:column;
  align-items:flex-start;gap:2px;min-width:200px;transition:all 0.15s}
.model-btn:hover{border-color:var(--pfizer)}
.model-btn .name{font-weight:700;color:var(--pfd)}
.model-btn .sub{font-size:10px;color:var(--muted)}
.model-btn.active.m1{border-color:var(--m1);background:#f3efff}
.model-btn.active.m2{border-color:var(--m2);background:#e8f4fb}
.model-btn.active .name{color:#000}

/* Legend */
.legend-row{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-bottom:8px}
.legend-item{display:inline-flex;align-items:center;gap:6px}
.swatch{width:12px;height:12px;border-radius:3px;display:inline-block}

/* Info boxes */
.info{background:#fffbe8;border:1px solid #f0d780;border-radius:8px;
  padding:10px 14px;font-size:12px;color:#6b5500;margin-bottom:14px}
.info.tip{background:#eaf3fb;border-color:#b8d8ed;color:#0a4775}

/* Metrics row */
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.metric{background:#f8fafc;border:1px solid var(--border);border-radius:8px;padding:10px 12px}
.metric .label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.metric .value{font-size:20px;font-weight:700;color:var(--pfd);margin-top:2px}

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
.feat-bar-marker{position:absolute;top:-2px;width:3px;height:22px;background:#000;
  border-radius:1px;box-shadow:0 0 0 2px #fff;z-index:5}
.feat-legend{display:flex;gap:10px;font-size:10px;color:var(--muted);margin:4px 0 8px}

.explanation{background:#f0f7fc;border-left:3px solid var(--pfizer);padding:12px 14px;
  border-radius:4px;font-size:13px;line-height:1.55;margin-top:14px}

footer{text-align:center;font-size:11px;color:var(--muted);margin-top:18px;padding:14px}
</style></head><body>

<header>
  <h1>HCP Segmentation Dashboard — Velsipity</h1>
  <p>Two-model comparison: Argmax baseline vs. Business-calibrated ordinal · Capstone Project, Tec de Monterrey × Pfizer</p>
</header>

<div class="tabs">
  <button class="tab active" data-tab="overview">1 · Overview & Model Comparison</button>
  <button class="tab" data-tab="explore">2 · Probability Map & Doctor Explorer</button>
</div>

<!-- ================== TAB 1: OVERVIEW ================== -->
<div class="tab-content active" id="tab-overview">

  <div class="panel">
    <h2>The Business Problem</h2>
    <p>Pfizer launched <strong>Velsipity</strong>, an oral therapy for ulcerative colitis (FDA-approved Oct 2023). The commercial team needs to know — for each of <strong id="kpi_total">—</strong> healthcare providers (HCPs) — how likely they are to prescribe, so sales force time goes where it has the most impact.</p>
    <p>The original segmentation came from interviewing <strong>205 HCPs</strong> and was extrapolated to <strong id="kpi_labeled">—</strong> doctors. Another <strong id="kpi_unlabeled">—</strong> remained unlabeled (<em>untyped</em>). The capstone goal: build a model that scales the segmentation accurately to the entire population.</p>
  </div>

  <div class="panel">
    <h2>The Three Segments (Ordinal: A &lt; B &lt; C)</h2>
    <div class="grid3" style="margin-top:8px">
      <div class="seg-card seg-A">
        <div class="name" style="color:var(--a)">SEG_A</div>
        <div class="pct"><span id="kpi_a_pct">—</span> · Will NOT prescribe</div>
        <div class="desc">Doctors with low UC prescribing activity and weak alignment with oral-therapy patterns.</div>
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
      <strong>Why ordinal matters:</strong> the segments are not three unrelated classes. Confusing SEG_A with SEG_B is much smaller error than confusing SEG_A with SEG_C, because the underlying scale is "propensity to prescribe."
    </div>
  </div>

  <div class="panel">
    <h2>Cost of Errors — Why Plain Accuracy Is the Wrong Metric</h2>
    <p>Some classification errors are far more expensive to the business than others. The chosen model is calibrated to minimize the worst kind first.</p>
    <div class="cost-row cost-bad">
      <div class="cost-icon">✕</div>
      <div><strong>Predicting SEG_A when true is SEG_C</strong> — A real prescriber was abandoned. Direct revenue loss.</div>
      <div class="cost-tag bad">CATASTROPHIC</div>
    </div>
    <div class="cost-row cost-warn">
      <div class="cost-icon">!</div>
      <div><strong>Predicting SEG_C when true is SEG_A</strong> — Sales force visits a non-prescriber. Time wasted, but recoverable.</div>
      <div class="cost-tag warn">COSTLY</div>
    </div>
    <div class="cost-row cost-ok">
      <div class="cost-icon">~</div>
      <div><strong>Any error involving SEG_B</strong> — SEG_B is itself a residual / ambiguous bucket; errors here are inherent.</div>
      <div class="cost-tag ok">ACCEPTABLE</div>
    </div>
  </div>

  <!-- Two models side by side -->
  <div class="panel">
    <h2>Two Models, Two Philosophies</h2>
    <p>We compare two XGBoost-based approaches that differ fundamentally in <em>what they optimize for</em>. The trade-off between them is the heart of this project.</p>

    <div class="grid2" style="margin-top:14px">
      <!-- Model 1: Argmax -->
      <div class="model-card m1">
        <span class="badge">MODEL 1</span>
        <h3 style="color:var(--m1)">Argmax Baseline</h3>
        <div class="subt">XGBoost multi-class · class_weight balanced · argmax decision</div>
        <p><strong>Decision rule:</strong> assigns the segment with the highest probability — winner takes all.</p>
        <p><strong>What it optimizes:</strong> overall classification accuracy. Treats all errors equally.</p>
        <div class="model-metrics">
          <div class="m"><div class="l">Accuracy</div><div class="v" id="amx_acc">—</div></div>
          <div class="m"><div class="l">Balanced Acc</div><div class="v" id="amx_ba">—</div></div>
          <div class="m"><div class="l">Recall SEG_C</div><div class="v" id="amx_rc">—</div></div>
          <div class="m"><div class="l" style="color:var(--bad)">SEG_C → A losses</div><div class="v" id="amx_lost" style="color:var(--bad)">—</div></div>
        </div>
        <h3>Confusion Matrix</h3>
        <div id="cm_amx"></div>
      </div>

      <!-- Model 2: Ordinal -->
      <div class="model-card m2">
        <span class="badge">MODEL 2 ⭐ (CHOSEN)</span>
        <h3 style="color:var(--m2)">Business-Calibrated Ordinal</h3>
        <div class="subt">Two-stage XGBoost (P≥B then P≥C) · sample_weight=2.0 for SEG_C · custom thresholds</div>
        <p><strong>Decision rule:</strong> if P(A) ≥ 0.70 → SEG_A; else if P(C) ≥ 0.30 → SEG_C; else SEG_B.</p>
        <p><strong>What it optimizes:</strong> minimizing the catastrophic SEG_C → SEG_A error, even at the cost of overall accuracy.</p>
        <div class="model-metrics">
          <div class="m"><div class="l">Accuracy</div><div class="v" id="ord_acc">—</div></div>
          <div class="m"><div class="l">Balanced Acc</div><div class="v" id="ord_ba">—</div></div>
          <div class="m"><div class="l">Recall SEG_C</div><div class="v" id="ord_rc">—</div></div>
          <div class="m"><div class="l" style="color:var(--ok)">SEG_C → A losses</div><div class="v" id="ord_lost" style="color:var(--ok)">—</div></div>
        </div>
        <h3>Confusion Matrix</h3>
        <div id="cm_ord"></div>
      </div>
    </div>

    <div class="tradeoff">
      <div class="col"><strong>Argmax Baseline</strong><br>Higher overall accuracy but loses many real prescribers — every missed SEG_C is a doctor who keeps prescribing competitor brands, unreached.</div>
      <div class="arrow">⇄</div>
      <div class="col"><strong>Business-Calibrated Ordinal ⭐</strong><br>Lower accuracy, but cuts catastrophic SEG_C → SEG_A losses by ~half. Sales force visits a few extra SEG_A doctors (recoverable) to make sure no real prescriber is missed.</div>
    </div>

    <div class="info tip" style="margin-top:14px">
      <strong>Why the chosen model is "worse" on accuracy and yet better for the business:</strong> in pharma, every SEG_C lost as SEG_A is a recurring revenue loss for as long as that doctor keeps prescribing. Every SEG_A visited as SEG_C is a one-time cost the rep can de-prioritize after a few visits. The asymmetry of those two costs is exactly what the chosen thresholds (0.70 / 0.30) encode.
    </div>
  </div>

  <div class="panel">
    <h2>Why This Problem Is Hard (Both Models Hit a Ceiling)</h2>
    <p>The internal Pfizer report acknowledges <strong>98.6% feature-space overlap</strong> between SEG_A and SEG_C, <strong>35.8% of cases inherently difficult</strong>, and <strong>SEG_B as a residual ambiguous category</strong>. Labels were extrapolated from 205 interviewed doctors; ~17% are inconsistent with observed prescribing behavior. After 10 strategy iterations, all approaches converged near a <strong>~58–60% balanced-accuracy ceiling</strong>. Beyond that, model improvements run into the noise floor of the labels themselves.</p>
    <p>The model's value is therefore not just the prediction — it's the <strong>probability</strong>, which lets the business reason about confidence, identify ambiguous cases for manual review, and prioritize within segments.</p>
  </div>

</div>

<!-- ================== TAB 2: PROBABILITY MAP + EXPLORER ================== -->
<div class="tab-content" id="tab-explore">

  <!-- LIVE MODEL TOGGLE -->
  <div class="model-toggle">
    <div>
      <div class="label">Active model:</div>
      <div class="desc">Toggle to see how predictions and probabilities change.</div>
    </div>
    <button class="model-btn m1" data-model="amx">
      <span class="name">Argmax Baseline</span>
      <span class="sub">Higher accuracy · more SEG_C losses</span>
    </button>
    <button class="model-btn m2 active" data-model="ord">
      <span class="name">Business-Calibrated Ordinal ⭐</span>
      <span class="sub">Lower accuracy · fewer SEG_C losses</span>
    </button>
    <div style="margin-left:auto">
      <div style="font-size:11px;color:var(--muted)">Live metrics</div>
      <div style="font-size:13px;font-weight:600;color:var(--pfd)">
        Acc <span id="live_acc">—</span> · Recall_C <span id="live_rc">—</span> · C→A <span id="live_lost">—</span>
      </div>
    </div>
  </div>

  <div class="info">
    <strong>How to read this view.</strong> Each dot is one doctor. Position = the model's probability distribution over the three segments. Doctors near a vertex are confidently classified; near the center are ambiguous. Click any dot in the bottom scatter to see why that specific doctor got those probabilities.
  </div>

  <!-- ===== Section A: Probability Maps ===== -->
  <div class="panel">
    <h2>Probability Maps</h2>
    <div class="controls">
      <span class="ctrl-label">Color by:</span>
      <button class="ctrl-btn active" data-color="true">Original label</button>
      <button class="ctrl-btn" data-color="pred">Model prediction</button>
      <button class="ctrl-btn" data-color="confidence">Confidence (max P)</button>
    </div>
    <div class="controls">
      <span class="ctrl-label">Show:</span>
      <button class="ctrl-btn active" data-show="all">All HCPs</button>
      <button class="ctrl-btn" data-show="labeled">With ground truth</button>
      <button class="ctrl-btn" data-show="unlabeled">No ground truth (untyped)</button>
    </div>
    <div class="legend-row">
      <span class="legend-item"><span class="swatch" style="background:var(--a)"></span>SEG_A</span>
      <span class="legend-item"><span class="swatch" style="background:var(--b)"></span>SEG_B</span>
      <span class="legend-item"><span class="swatch" style="background:var(--c)"></span>SEG_C</span>
      <span class="legend-item"><span class="swatch" style="background:var(--u)"></span>No ground truth (only in "Original label" mode)</span>
    </div>
    <div class="info tip">
      <strong>About "No ground truth" doctors:</strong> these <span id="info_unlab">—</span> HCPs never had an original label — they're the <em>untyped</em> population the model must classify. They appear gray only in <em>"Original label"</em> mode (no truth to color); in <em>"Model prediction"</em> mode they show their predicted color, just like any other doctor.
    </div>
    <div class="grid2">
      <div>
        <h3>Probability Simplex (Ternary)</h3>
        <p style="font-size:12px;color:var(--muted);margin-bottom:8px">Mathematically canonical view. Vertices = pure SEG_A/B/C. Center = maximum ambiguity.</p>
        <div id="plot_ternary" class="plot"></div>
      </div>
      <div>
        <h3>3D Scatter — P(A), P(B), P(C)</h3>
        <p style="font-size:12px;color:var(--muted);margin-bottom:8px">Same data, three explicit axes. All points lie on the plane P(A)+P(B)+P(C)=1.</p>
        <div id="plot_3d" class="plot"></div>
      </div>
    </div>
  </div>

  <!-- ===== Section B: Doctor Explorer ===== -->
  <div class="panel">
    <h2>Doctor Explorer</h2>
    <p style="font-size:13px;color:var(--muted)">Click any dot in the scatter (or search by HCP ID) to see why the active model assigned its probabilities. The right panel compares the doctor's metrics against typical ranges for each segment.</p>
    <div class="explorer" style="margin-top:10px">
      <div>
        <input type="text" class="search-box" id="search" placeholder="Search by HCP ID — or click a dot below">
        <div id="search_result" style="font-size:12px;color:var(--muted);margin-bottom:10px"></div>
        <h3>UC Rx vs Oral Rx (top discriminating features)</h3>
        <p style="font-size:11px;color:var(--muted);margin-bottom:6px">Axes are log(1+x). UC_TRX has the strongest signal (AUC 0.73). Click any point.</p>
        <div id="plot_scatter" class="plot" style="height:430px"></div>
        <div class="legend-row" style="margin-top:6px">
          <span class="legend-item"><span class="swatch" style="background:var(--a)"></span>Predicted SEG_A</span>
          <span class="legend-item"><span class="swatch" style="background:var(--b)"></span>Predicted SEG_B</span>
          <span class="legend-item"><span class="swatch" style="background:var(--c)"></span>Predicted SEG_C</span>
        </div>
      </div>
      <div>
        <h3>Doctor Profile</h3>
        <div id="profile" class="profile">
          <div class="profile-empty">Click a dot in the scatter, or search by HCP ID, to see a detailed profile and explanation.</div>
        </div>
      </div>
    </div>
  </div>

</div>

<footer>
  Capstone Project · Tec de Monterrey × Pfizer Global Commercial Analytics · 5-fold cross-validation (out-of-fold predictions on labeled HCPs); forward scoring on untyped HCPs
</footer>

<script>
const DATA = __DATA_PLACEHOLDER__;
const COLORS = {SEG_A:'#2e6cb0', SEG_B:'#f0a830', SEG_C:'#c83a3a', UNLABELED:'#9aa6b4'};
const M = DATA.metrics;
let ACTIVE_MODEL = 'ord';   // 'ord' = ordinal, 'amx' = argmax
let colorMode = 'true';
let showMode  = 'all';

/* ---------- Fill static overview metrics ---------- */
document.getElementById('kpi_total').textContent     = (M.n_labeled + M.n_unlabeled).toLocaleString();
document.getElementById('kpi_labeled').textContent   = M.n_labeled.toLocaleString();
document.getElementById('kpi_unlabeled').textContent = M.n_unlabeled.toLocaleString();
document.getElementById('kpi_a_pct').textContent     = (M.dist_A*100).toFixed(1) + '%';
document.getElementById('kpi_b_pct').textContent     = (M.dist_B*100).toFixed(1) + '%';
document.getElementById('kpi_c_pct').textContent     = (M.dist_C*100).toFixed(1) + '%';

['ord','amx'].forEach(mod => {
  const mm = M[mod];
  const pref = mod === 'ord' ? 'ord' : 'amx';
  document.getElementById(pref+'_acc').textContent  = (mm.accuracy*100).toFixed(1) + '%';
  document.getElementById(pref+'_ba').textContent   = (mm.balanced_accuracy*100).toFixed(1) + '%';
  document.getElementById(pref+'_rc').textContent   = (mm.recall_C*100).toFixed(1) + '%';
  document.getElementById(pref+'_lost').textContent = (mm.c_lost_pct*100).toFixed(1) + '%';
});

document.getElementById('info_unlab').textContent = M.n_unlabeled.toLocaleString();

/* ---------- Confusion matrix renderer ---------- */
function renderCM(containerId, cm){
  const labels = ['SEG_A','SEG_B','SEG_C'];
  let html = '<table class="cm-table"><thead><tr><th></th>';
  html += '<th colspan="3" class="col-label">Predicted</th></tr><tr><th></th>';
  labels.forEach(l => html += `<th>${l}</th>`);
  html += '</tr></thead><tbody>';
  cm.forEach((row, i) => {
    html += `<tr><td class="row-label">${labels[i]}</td>`;
    row.forEach((v, j) => {
      const isDiag = i === j;
      const total = row.reduce((a,b)=>a+b, 0);
      const pct = total > 0 ? (v/total*100).toFixed(0) : 0;
      const bg = isDiag ? 'background:#eaf6ed' : (v > 50 ? 'background:#fdebec' : '');
      html += `<td class="${isDiag?'diag':''}" style="${bg}">${v.toLocaleString()}<br><span style="color:var(--muted);font-size:10px">${pct}%</span></td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  html += '<div style="font-size:10px;color:var(--muted);margin-top:4px">Rows = actual segment · Columns = predicted segment · % = row-normalized (recall)</div>';
  document.getElementById(containerId).innerHTML = html;
}
renderCM('cm_amx', M.amx.confusion_matrix);
renderCM('cm_ord', M.ord.confusion_matrix);

/* ---------- Active-model accessors ---------- */
const VIZ = DATA.viz;       // shared base info for plotted doctors
const ALL = DATA.all;       // full lookup
const FEATS = DATA.features;
const STATS = DATA.segment_stats;

function activeViz(field){ return VIZ[ACTIVE_MODEL][field]; }
function buildHover(){
  return VIZ.hcp_id.map((_,i) => {
    const t = VIZ.is_labeled[i] ? VIZ.true[i] : '(untyped)';
    const PA = activeViz('P_A')[i], PB = activeViz('P_B')[i], PC = activeViz('P_C')[i];
    const PR = activeViz('pred')[i];
    return `HCP ${VIZ.hcp_id[i]}<br>Original: <b>${t}</b> · Predicted: <b>${PR}</b><br>` +
           `P(A)=${(PA*100).toFixed(1)}% · P(B)=${(PB*100).toFixed(1)}% · P(C)=${(PC*100).toFixed(1)}%`;
  });
}

function getMask(){
  const lb = VIZ.is_labeled;
  if (showMode === 'all') return lb.map(()=>true);
  if (showMode === 'labeled') return lb;
  return lb.map(x => !x);
}
function getColors(mode){
  if (mode === 'true')
    return VIZ.true.map((s,i) => VIZ.is_labeled[i] ? (COLORS[s]||COLORS.UNLABELED) : COLORS.UNLABELED);
  if (mode === 'pred')
    return activeViz('pred').map(s => COLORS[s]);
  // confidence
  return activeViz('P_A').map((_,i) => Math.max(activeViz('P_A')[i], activeViz('P_B')[i], activeViz('P_C')[i]));
}
function filt(arr, mask){ return arr.filter((_,i) => mask[i]); }

/* ---------- Plots ---------- */
function plotTernary(){
  const m = getMask(), c = getColors(colorMode), isC = colorMode === 'confidence';
  const hover = buildHover();
  Plotly.react('plot_ternary', [{
    type: 'scatterternary', mode: 'markers',
    a: filt(activeViz('P_A'), m), b: filt(activeViz('P_B'), m), c: filt(activeViz('P_C'), m),
    text: filt(hover, m), hoverinfo: 'text',
    marker: {
      size: 5, color: filt(c, m),
      colorscale: isC ? 'Viridis' : undefined,
      cmin: isC ? 0.33 : undefined, cmax: isC ? 1 : undefined,
      showscale: isC, opacity: 0.7,
      line: {width: 0.3, color: 'rgba(0,0,0,0.15)'}
    }
  }], {
    ternary: {sum: 1,
      aaxis: {title: 'P(SEG_A)', min: 0},
      baxis: {title: 'P(SEG_B)', min: 0},
      caxis: {title: 'P(SEG_C)', min: 0},
      bgcolor: '#fafbfd'},
    margin: {l: 50, r: 30, t: 20, b: 30}, showlegend: false
  }, {responsive: true, displaylogo: false});
}
function plot3D(){
  const m = getMask(), c = getColors(colorMode), isC = colorMode === 'confidence';
  const hover = buildHover();
  Plotly.react('plot_3d', [{
    type: 'scatter3d', mode: 'markers',
    x: filt(activeViz('P_A'), m), y: filt(activeViz('P_B'), m), z: filt(activeViz('P_C'), m),
    text: filt(hover, m), hoverinfo: 'text',
    marker: {
      size: 3, color: filt(c, m),
      colorscale: isC ? 'Viridis' : undefined,
      cmin: isC ? 0.33 : undefined, cmax: isC ? 1 : undefined,
      opacity: 0.8
    }
  }], {
    scene: {
      xaxis: {title: 'P(A)', range: [0,1]},
      yaxis: {title: 'P(B)', range: [0,1]},
      zaxis: {title: 'P(C)', range: [0,1]},
      camera: {eye: {x: 1.4, y: 1.4, z: 1}}
    },
    margin: {l: 0, r: 0, t: 0, b: 0}
  }, {responsive: true, displaylogo: false});
}
function plotScatter(){
  const colors = activeViz('pred').map(s => COLORS[s]);
  const hover = buildHover();
  Plotly.react('plot_scatter', [{
    type: 'scattergl', mode: 'markers',
    x: VIZ.log_uc_trx, y: VIZ.log_oral_trx,
    text: hover, hoverinfo: 'text', customdata: VIZ.hcp_id,
    marker: {size: 6, color: colors, opacity: 0.6,
             line: {width: 0.3, color: 'rgba(0,0,0,0.2)'}}
  }], {
    xaxis: {title: 'log(1 + UC_TRX_sum)', gridcolor: '#eef2f7'},
    yaxis: {title: 'log(1 + ORAL_TRX_sum)', gridcolor: '#eef2f7'},
    margin: {l: 50, r: 20, t: 10, b: 40},
    plot_bgcolor: '#fafbfd', hovermode: 'closest'
  }, {responsive: true, displaylogo: false});
  document.getElementById('plot_scatter').on('plotly_click', ev => {
    showProfile(ev.points[0].customdata);
  });
}

function rerenderAll(){
  plotTernary(); plot3D(); plotScatter();
  // refresh live metric strip
  const mm = M[ACTIVE_MODEL];
  document.getElementById('live_acc').textContent  = (mm.accuracy*100).toFixed(1) + '%';
  document.getElementById('live_rc').textContent   = (mm.recall_C*100).toFixed(1) + '%';
  document.getElementById('live_lost').textContent = (mm.c_lost_pct*100).toFixed(1) + '%';
  // re-render currently selected profile if any
  if (currentProfileId) showProfile(currentProfileId);
}

/* ---------- Tab switcher ---------- */
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('tab-' + t.dataset.tab).classList.add('active');
    if (t.dataset.tab === 'explore') { rerenderAll(); }
  });
});

/* ---------- Model toggle (live) ---------- */
document.querySelectorAll('.model-btn').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('.model-btn').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    ACTIVE_MODEL = b.dataset.model;
    rerenderAll();
  });
});

/* ---------- Color/show controls ---------- */
document.querySelectorAll('.ctrl-btn[data-color]').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('.ctrl-btn[data-color]').forEach(x => x.classList.remove('active'));
  b.classList.add('active'); colorMode = b.dataset.color; plotTernary(); plot3D();
}));
document.querySelectorAll('.ctrl-btn[data-show]').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('.ctrl-btn[data-show]').forEach(x => x.classList.remove('active'));
  b.classList.add('active'); showMode = b.dataset.show; plotTernary(); plot3D();
}));

/* ---------- Doctor profile ---------- */
let currentProfileId = null;

function fmtNum(v){
  if (v === null || v === undefined) return '—';
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10)  return v.toFixed(1);
  return v.toFixed(2);
}

function classifyFeatureValue(v, statsForFeat){
  const r = {};
  for (const seg of ['SEG_A','SEG_B','SEG_C']){
    const s = statsForFeat[seg];
    if (v < s.p25) r[seg] = 'below';
    else if (v > s.p75) r[seg] = 'above';
    else r[seg] = 'within';
  }
  return r;
}
function closestSegment(v, statsForFeat){
  let best=null, bd=Infinity;
  for (const seg of ['SEG_A','SEG_B','SEG_C']){
    const d = Math.abs(v - statsForFeat[seg].median);
    if (d < bd){ bd = d; best = seg; }
  }
  return best;
}

function buildExplanation(doctor){
  const featDescriptions = [];
  FEATS.forEach((f, i) => {
    const v = doctor.f[i];
    const stats = STATS[f.key];
    if (!stats) return;
    const cls = classifyFeatureValue(v, stats);
    let segLabel;
    if (cls.SEG_A === 'within' && cls.SEG_C === 'within')
      segLabel = 'overlaps both SEG_A and SEG_C ranges (ambiguous zone)';
    else if (cls.SEG_C === 'within' || cls.SEG_C === 'above')
      segLabel = `is in the typical range of SEG_C (median: ${fmtNum(stats.SEG_C.median)})`;
    else if (cls.SEG_A === 'within' || cls.SEG_A === 'below')
      segLabel = `is in the typical range of SEG_A (median: ${fmtNum(stats.SEG_A.median)})`;
    else
      segLabel = `falls in the SEG_B range (median: ${fmtNum(stats.SEG_B.median)})`;
    featDescriptions.push(`<strong>${f.name}</strong> = ${fmtNum(v)} — ${segLabel}.`);
  });

  const probs = {SEG_A: doctor.pa, SEG_B: doctor.pb, SEG_C: doctor.pc};
  const topSeg = Object.keys(probs).reduce((a,b) => probs[a] > probs[b] ? a : b);
  const maxP = Math.max(doctor.pa, doctor.pb, doctor.pc);

  let lead;
  if (maxP >= 0.70)
    lead = `The model is <strong>highly confident</strong> this HCP belongs to <strong style="color:${COLORS[topSeg]}">${topSeg}</strong> (P = ${(maxP*100).toFixed(0)}%).`;
  else if (maxP >= 0.50)
    lead = `The model leans toward <strong style="color:${COLORS[topSeg]}">${topSeg}</strong> (P = ${(maxP*100).toFixed(0)}%), with meaningful uncertainty.`;
  else
    lead = `This HCP is in the <strong>ambiguous zone</strong> — no single segment exceeds 50% probability. Best guess: <strong style="color:${COLORS[topSeg]}">${topSeg}</strong> (P = ${(maxP*100).toFixed(0)}%). Manual review recommended.`;

  let ruleNote;
  if (ACTIVE_MODEL === 'ord'){
    if (doctor.pa >= 0.70)
      ruleNote = `<strong>Decision rule (ordinal):</strong> classified as <strong>SEG_A</strong> because P(A) = ${(doctor.pa*100).toFixed(0)}% ≥ 70% threshold.`;
    else if (doctor.pc >= 0.30)
      ruleNote = `<strong>Decision rule (ordinal):</strong> classified as <strong>SEG_C</strong> because P(C) = ${(doctor.pc*100).toFixed(0)}% ≥ 30% threshold (low bar by design — losing a real prescriber is the most costly error).`;
    else
      ruleNote = `<strong>Decision rule (ordinal):</strong> defaulted to <strong>SEG_B</strong> because neither P(A) ≥ 70% nor P(C) ≥ 30% — not enough signal in either direction.`;
  } else {
    ruleNote = `<strong>Decision rule (argmax):</strong> classified as <strong>${doctor.p}</strong> because it had the highest probability (${(maxP*100).toFixed(0)}%) — winner takes all.`;
  }

  return `<p>${lead}</p>
<p style="margin-top:8px">${ruleNote}</p>
<p style="margin-top:10px"><strong>Feature-by-feature:</strong></p>
<ul style="margin-top:4px;margin-left:20px">${featDescriptions.map(d=>`<li>${d}</li>`).join('')}</ul>`;
}

function buildFeatureRow(f, value, stats){
  const maxRef = Math.max(stats.SEG_A.p75, stats.SEG_B.p75, stats.SEG_C.p75) * 1.3 + 0.01;
  const bandPct = seg => {
    const s = stats[seg];
    const left  = Math.max(0, Math.min(100, s.p25 / maxRef * 100));
    const right = Math.max(0, Math.min(100, s.p75 / maxRef * 100));
    return {left, width: Math.max(0, right - left)};
  };
  const bA = bandPct('SEG_A'), bB = bandPct('SEG_B'), bC = bandPct('SEG_C');
  const markerPct = Math.max(0, Math.min(100, value / maxRef * 100));
  return `
    <div class="feat-row">
      <div class="feat-name">${f.name}<span class="desc">${f.desc}</span></div>
      <div class="feat-value">${fmtNum(value)}</div>
      <div>
        <div class="feat-bar-wrap">
          <div style="position:absolute;top:0;left:${bA.left}%;width:${bA.width}%;height:100%;background:${COLORS.SEG_A};opacity:0.55"></div>
          <div style="position:absolute;top:0;left:${bB.left}%;width:${bB.width}%;height:100%;background:${COLORS.SEG_B};opacity:0.55"></div>
          <div style="position:absolute;top:0;left:${bC.left}%;width:${bC.width}%;height:100%;background:${COLORS.SEG_C};opacity:0.55"></div>
          <div class="feat-bar-marker" style="left:calc(${markerPct}% - 1.5px)" title="This doctor: ${fmtNum(value)}"></div>
        </div>
      </div>
    </div>`;
}

function showProfile(hcpId){
  const id = String(hcpId);
  currentProfileId = id;
  const d = ALL[id];
  if (!d){
    document.getElementById('profile').innerHTML = '<div class="profile-empty">HCP not found.</div>';
    return;
  }
  // Active-model probabilities and prediction
  const m = d.m[ACTIVE_MODEL];
  const docForExplain = {pa: m.pa, pb: m.pb, pc: m.pc, p: m.p, f: d.f};

  const tagTrue = d.il
    ? `<span class="tag tag-true">Original: ${d.t}</span>`
    : '<span class="tag tag-true">No ground truth (untyped)</span>';
  const tagPred = `<span class="tag tag-pred">Predicted (${ACTIVE_MODEL==='ord'?'ordinal':'argmax'}): ${m.p}</span>`;

  const pa = (m.pa*100).toFixed(0), pb = (m.pb*100).toFixed(0), pc = (m.pc*100).toFixed(0);

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
    <div style="font-size:11px;color:var(--muted);margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px">Probability Distribution (active model)</div>
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
      ${buildExplanation(docForExplain)}
    </div>
  `;
}

/* ---------- Search ---------- */
document.getElementById('search').addEventListener('input', e => {
  const q = e.target.value.trim();
  const result = document.getElementById('search_result');
  if (!q){ result.textContent = ''; return; }
  if (ALL[q]){ result.textContent = ''; showProfile(q); }
  else { result.textContent = `No HCP found with ID "${q}"`; }
});

/* ---------- Initial render ---------- */
rerenderAll();
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

    # Save raw key feature values BEFORE feature engineering (for explorer/profile)
    raw_feature_values = {}
    for fkey, _, _ in KEY_FEATURES:
        if fkey in df.columns:
            raw_feature_values[fkey] = df[fkey].values.copy()

    print("\nGenerating derived features...")
    df = add_ratios(df)
    drop_cols = set([args.label_col, args.id_col] + list(args.exclude_cols))
    feat_cols = [c for c in df.columns
                 if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  Total features: {len(feat_cols)}")

    X_all = df[feat_cols].fillna(0).values
    y_all = df[args.label_col].values
    ids_all = df[args.id_col].values

    X_lab = X_all[is_labeled]
    y_lab = y_all[is_labeled]
    X_unlab = X_all[~is_labeled]

    # ----- OOF for both models on labeled -----
    print(f"\n[1/4] Out-of-fold CV on labeled HCPs ({args.n_folds} folds, BOTH models)...")
    oof = oof_both(X_lab, y_lab, n_splits=args.n_folds)

    metrics = {}
    for mod, friendly in [("ord", "Business-Calibrated Ordinal"), ("amx", "Argmax Baseline")]:
        m = metrics_for(y_lab, oof[mod]["pred"])
        metrics[mod] = m
        print(f"\n  --- {friendly} ---")
        print(f"  Accuracy:           {m['accuracy']:.4f}")
        print(f"  Balanced accuracy:  {m['balanced_accuracy']:.4f}")
        print(f"  Recall SEG_C:       {m['recall_C']:.4f}")
        print(f"  SEG_C → A losses:   {m['c_lost_pct']:.4f}")
        print("  Confusion matrix:")
        cm = np.array(m["confusion_matrix"])
        print(pd.DataFrame(cm,
            index=[f"{l}_true" for l in VALID_LABELS],
            columns=[f"{l}_pred" for l in VALID_LABELS]))

    # ----- Train final models on all labeled, score unlabeled -----
    if n_unlabeled > 0:
        print(f"\n[2/4] Training final models on {n_labeled:,} labeled HCPs...")
        m1, m2 = fit_ordinal(X_lab, y_lab)
        m_amx, le_amx = fit_argmax(X_lab, y_lab)
        print(f"      Scoring {n_unlabeled:,} unlabeled HCPs with both models...")
        ord_unlab = predict_ordinal(m1, m2, X_unlab)
        amx_unlab = predict_argmax(m_amx, le_amx, X_unlab)
    else:
        ord_unlab = (np.array([]),)*4
        amx_unlab = (np.array([]),)*4

    # ----- Reassemble full vectors -----
    n = len(df)
    full = {}
    for mod, oof_d, unlab_d in [("ord", oof["ord"], ord_unlab),
                                ("amx", oof["amx"], amx_unlab)]:
        PA = np.zeros(n); PB = np.zeros(n); PC = np.zeros(n)
        pred = np.empty(n, dtype=object)
        if mod == "ord":
            PA[is_labeled] = oof_d["P_A"]; PB[is_labeled] = oof_d["P_B"]
            PC[is_labeled] = oof_d["P_C"]; pred[is_labeled] = oof_d["pred"]
        else:
            PA[is_labeled] = oof_d["P_A"]; PB[is_labeled] = oof_d["P_B"]
            PC[is_labeled] = oof_d["P_C"]; pred[is_labeled] = oof_d["pred"]
        if n_unlabeled > 0:
            PA[~is_labeled] = unlab_d[0]; PB[~is_labeled] = unlab_d[1]
            PC[~is_labeled] = unlab_d[2]; pred[~is_labeled] = unlab_d[3]
        full[mod] = {"P_A": PA, "P_B": PB, "P_C": PC, "pred": pred}

    # ----- Segment statistics for explainer -----
    print("\n[3/4] Computing segment statistics for explanation engine...")
    available_features = [(k,n_,d_) for k,n_,d_ in KEY_FEATURES if k in raw_feature_values]
    df_lab_keys = pd.DataFrame({k: raw_feature_values[k][is_labeled] for k,_,_ in available_features})
    df_lab_keys[args.label_col] = y_lab
    seg_stats = compute_segment_stats(df_lab_keys, args.label_col,
                                       [k for k,_,_ in available_features])

    # ----- CSVs -----
    print("\n[4/4] Writing outputs...")
    out_df = pd.DataFrame({
        args.id_col: ids_all,
        "true_segment": y_all,
        "was_labeled": is_labeled,
        # Ordinal model (chosen)
        "ord_pred": full["ord"]["pred"],
        "ord_P_A": full["ord"]["P_A"],
        "ord_P_B": full["ord"]["P_B"],
        "ord_P_C": full["ord"]["P_C"],
        # Argmax baseline
        "amx_pred": full["amx"]["pred"],
        "amx_P_A": full["amx"]["P_A"],
        "amx_P_B": full["amx"]["P_B"],
        "amx_P_C": full["amx"]["P_C"],
    })
    # Use ordinal max-prob as the "ambiguous" criterion (it's the chosen model)
    max_p_ord = np.maximum.reduce([full["ord"]["P_A"], full["ord"]["P_B"], full["ord"]["P_C"]])
    out_df["max_prob_ord"] = max_p_ord
    out_df["ambiguous"] = max_p_ord < args.ambiguous_threshold
    out_df.to_csv(args.output_csv, index=False)
    print(f"  ✓ {args.output_csv}  ({len(out_df):,} rows · both models)")

    amb_df = out_df[out_df["ambiguous"]].sort_values("max_prob_ord")
    amb_df.to_csv(args.output_ambiguous, index=False)
    print(f"  ✓ {args.output_ambiguous}  ({len(amb_df):,} HCPs flagged for review)")

    # ----- Build viz subset -----
    rng = np.random.default_rng(0)
    if len(df) > args.max_html_points:
        keep_idx = []
        n_lab_target = args.max_html_points // 2
        n_unlab_target = args.max_html_points - n_lab_target
        for cls in VALID_LABELS:
            cls_idx = np.where(y_all == cls)[0]
            n_keep = int(n_lab_target * (len(cls_idx) / max(n_labeled, 1)))
            keep_idx.extend(rng.choice(cls_idx, size=min(n_keep, len(cls_idx)), replace=False))
        unlab_idx = np.where(~is_labeled)[0]
        if len(unlab_idx) > 0:
            keep_idx.extend(rng.choice(unlab_idx, size=min(n_unlab_target, len(unlab_idx)), replace=False))
        keep_idx = np.array(keep_idx)
    else:
        keep_idx = np.arange(len(df))

    uc_trx = raw_feature_values.get("UC_TRX_sum", np.zeros(n))
    oral_trx = raw_feature_values.get("ORAL_TRX_sum", np.zeros(n))
    log_uc = np.log1p(np.maximum(uc_trx, 0))
    log_oral = np.log1p(np.maximum(oral_trx, 0))

    viz = {
        "hcp_id":     [str(x) for x in ids_all[keep_idx]],
        "true":       [str(x) for x in y_all[keep_idx]],
        "is_labeled": [bool(x) for x in is_labeled[keep_idx]],
        "log_uc_trx": [round(float(x), 3) for x in log_uc[keep_idx]],
        "log_oral_trx":[round(float(x), 3) for x in log_oral[keep_idx]],
        "ord": {
            "P_A":  [round(float(x), 3) for x in full["ord"]["P_A"][keep_idx]],
            "P_B":  [round(float(x), 3) for x in full["ord"]["P_B"][keep_idx]],
            "P_C":  [round(float(x), 3) for x in full["ord"]["P_C"][keep_idx]],
            "pred": [str(x) for x in full["ord"]["pred"][keep_idx]],
        },
        "amx": {
            "P_A":  [round(float(x), 3) for x in full["amx"]["P_A"][keep_idx]],
            "P_B":  [round(float(x), 3) for x in full["amx"]["P_B"][keep_idx]],
            "P_C":  [round(float(x), 3) for x in full["amx"]["P_C"][keep_idx]],
            "pred": [str(x) for x in full["amx"]["pred"][keep_idx]],
        },
    }

    # ----- Full doctor lookup -----
    print("  Building per-doctor lookup table...")
    feat_keys = [k for k,_,_ in available_features]
    feat_arrays = [raw_feature_values[k] for k in feat_keys]
    all_lookup = {}
    for i, hid in enumerate(ids_all):
        all_lookup[str(hid)] = {
            "t":  str(y_all[i]) if is_labeled[i] else "",
            "il": bool(is_labeled[i]),
            "f":  [round(float(arr[i]), 1) for arr in feat_arrays],
            "m": {
                "ord": {
                    "pa": round(float(full["ord"]["P_A"][i]), 3),
                    "pb": round(float(full["ord"]["P_B"][i]), 3),
                    "pc": round(float(full["ord"]["P_C"][i]), 3),
                    "p":  str(full["ord"]["pred"][i]),
                },
                "amx": {
                    "pa": round(float(full["amx"]["P_A"][i]), 3),
                    "pb": round(float(full["amx"]["P_B"][i]), 3),
                    "pc": round(float(full["amx"]["P_C"][i]), 3),
                    "p":  str(full["amx"]["pred"][i]),
                },
            },
        }

    payload = {
        "viz": viz,
        "all": all_lookup,
        "features": [{"key":k, "name":n_, "desc":d_} for k,n_,d_ in available_features],
        "segment_stats": seg_stats,
        "metrics": {
            "n_labeled":   int(n_labeled),
            "n_unlabeled": int(n_unlabeled),
            "dist_A": float(dist.get("SEG_A", 0)),
            "dist_B": float(dist.get("SEG_B", 0)),
            "dist_C": float(dist.get("SEG_C", 0)),
            "ord": metrics["ord"],
            "amx": metrics["amx"],
        },
    }

    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", json.dumps(payload))
    Path(args.output_html).write_text(html, encoding="utf-8")
    size_mb = Path(args.output_html).stat().st_size / 1024 / 1024
    print(f"  ✓ {args.output_html}  ({size_mb:.2f} MB · {len(keep_idx):,} viz points · {len(all_lookup):,} searchable HCPs)")
    print(f"\nDone. Open {args.output_html} in any modern browser.")


if __name__ == "__main__":
    main()