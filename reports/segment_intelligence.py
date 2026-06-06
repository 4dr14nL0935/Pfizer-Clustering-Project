"""
hcp_segmentation_final_v3.py
============================
FINAL Capstone Deliverable — HCP Segmentation for Velsipity

MODEL: Business-Calibrated Ordinal XGBoost + Dominance Rule (RECOMMENDED)
  Two-stage XGBoost: P(≥B|X) and P(≥C|X)
  Decision cascade:
    P(A) ≥ 0.70 → SEG_A
    P(C) ≥ 0.30 → SEG_C
    P(C) > P(B) → SEG_C  (dominance rule)
    else        → SEG_B

WHAT'S NEW IN v3:
  ✦ New "Conversion Strategy" tab — actionable B→C movement plan,
    identifying doctors and the engagement gaps holding them back
  ✦ Simplified CI presentation — per-doctor only, with clear explanation
  ✦ Improved business framing throughout
  ✦ New 3D view toggle to color untyped HCPs by predicted segment
  ✦ Dedicated explanations of SHAP, Confidence Intervals, and Confidence
  ✦ Removed huge threshold sensitivity table (replaced with model explainer)

USAGE:
    python hcp_segmentation_final_v3.py --input doctors_aggregated.csv

Tec de Monterrey × Pfizer Global Commercial Analytics
"""

import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except ImportError:
    sys.exit("xgboost not installed. Run: pip install xgboost")

# -- Path bootstrap so `import src` resolves under `python reports/...` --
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

# -- Shared core: single source of truth for config, features, model & intel --
from src import (
    SEED, VALID_LABELS, THR_A, THR_C,
    KEY_FEATURES, N_SHAP_TOP, N_CONVERSION_CANDIDATES, SEG_COLORS,
    add_features, select_features,
    fit_ordinal, predict_ordinal, fit_argmax, predict_argmax,
    get_shap_values, top_shap_per_doctor,
    oof_with_ci, metrics_for, compute_segment_stats,
    compute_segment_centroids, find_prototype_doctors,
    compute_business_levers, compute_churn_risks, generate_executive_insights,
)

def compute_conversion_strategy(ids_all, pred_ord, PA, PB, PC,
                                  is_labeled, y_all, raw_fv, key_features,
                                  n_candidates=N_CONVERSION_CANDIDATES):
    """
    Identify predicted-SEG_B doctors with highest P_C as conversion candidates.
    For each, compute feature-level gaps vs SEG_C median to surface what's
    'holding them back' from being classified as C.

    Returns:
      candidates: list of dicts with HCP info, gaps, and a top action
      aggregate:  dict with cross-candidate statistics
      c_medians:  reference values per feature
    """
    # SEG_C median on labeled doctors only (the ground truth reference)
    c_mask = is_labeled & (y_all == "SEG_C")
    c_medians = {k: float(np.median(raw_fv[k][c_mask])) for k, _, _, _ in key_features
                 if k in raw_fv}

    # Predicted-B doctors
    pb_idx = np.where(pred_ord == "SEG_B")[0]
    # Sort by P_C descending (closer to C = better candidates)
    pb_sorted = pb_idx[np.argsort(-PC[pb_idx])]
    # Cap at n_candidates
    cand_idx = pb_sorted[:n_candidates]

    feat_meta = {k: (name, desc, actionable) for k, name, desc, actionable
                 in key_features if k in raw_fv}

    candidates = []
    for i in cand_idx:
        gaps = {}
        for k in c_medians:
            val = float(raw_fv[k][i])
            med = c_medians[k]
            gap = val - med  # negative = below C median
            gaps[k] = {
                "value": round(val, 2),
                "c_median": round(med, 2),
                "gap": round(gap, 2),
                "below": gap < 0,
            }
        # Top action: most-negative gap among ACTIONABLE features
        actionable_gaps = [(k, gaps[k]["gap"]) for k in gaps
                           if feat_meta[k][2]]
        actionable_gaps.sort(key=lambda x: x[1])  # most negative first
        top_action = None
        if actionable_gaps and actionable_gaps[0][1] < 0:
            k, g = actionable_gaps[0]
            top_action = {
                "feature": k,
                "feature_name": feat_meta[k][0],
                "gap": round(g, 2),
                "current": gaps[k]["value"],
                "target": gaps[k]["c_median"],
            }
        candidates.append({
            "id": str(ids_all[i]),
            "pa": round(float(PA[i]), 3),
            "pb": round(float(PB[i]), 3),
            "pc": round(float(PC[i]), 3),
            "true": str(y_all[i]) if is_labeled[i] else None,
            "gaps": gaps,
            "top_action": top_action,
        })

    # Aggregate: % of candidates below C median per feature
    aggregate = {"n_candidates": len(candidates), "features": []}
    for k in c_medians:
        n_below = sum(1 for c in candidates if c["gaps"][k]["below"])
        aggregate["features"].append({
            "key": k,
            "name": feat_meta[k][0],
            "actionable": feat_meta[k][2],
            "pct_below_C_median": round(n_below / max(len(candidates), 1) * 100, 1),
            "c_median": round(c_medians[k], 2),
        })
    aggregate["features"].sort(key=lambda x: -x["pct_below_C_median"])

    return candidates, aggregate, c_medians


# ═══════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ═══════════════════════════════════════════════════════════════

HTML_TEMPLATE = r'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>HCP Segmentation — Velsipity · Pfizer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#f5f7fb;--card:#fff;--border:#e3e8f0;--text:#1a2330;--muted:#6b7c93;
--pfizer:#0093D0;--pfd:#00255D;--a:#2e6cb0;--b:#f0a830;--c:#c83a3a;--u:#9aa6b4;
--ok:#2a8b4a;--warn:#cc7a00;--bad:#b22a2a}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
background:var(--bg);color:var(--text);padding:18px;line-height:1.5}
header{background:linear-gradient(135deg,var(--pfd),var(--pfizer));color:#fff;
padding:18px 24px;border-radius:10px;margin-bottom:14px}
header h1{font-size:22px;margin-bottom:4px}header p{font-size:13px;opacity:.92}
.tabs{display:flex;gap:4px;margin-bottom:14px;border-bottom:2px solid var(--border);flex-wrap:wrap}
.tab{padding:10px 18px;background:none;border:none;font-size:14px;font-weight:500;
color:var(--muted);cursor:pointer;border-bottom:3px solid transparent;margin-bottom:-2px;font-family:inherit}
.tab.active{color:var(--pfd);border-bottom-color:var(--pfizer);font-weight:600}
.tab-content{display:none}.tab-content.active{display:block}
.panel{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:18px;margin-bottom:14px}
.panel h2{font-size:16px;color:var(--pfd);margin-bottom:8px}
.panel h3{font-size:13px;color:var(--pfd);margin:14px 0 6px;text-transform:uppercase;letter-spacing:.5px}
.panel p{font-size:13px;margin-bottom:8px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
@media(max-width:1100px){.grid2,.grid3{grid-template-columns:1fr}}
.seg-card{padding:14px;border-radius:8px;border-left:5px solid;background:#f8fafc}
.seg-A{border-left-color:var(--a)}.seg-B{border-left-color:var(--b)}.seg-C{border-left-color:var(--c)}
.seg-card .name{font-weight:700;font-size:14px}.seg-card .pct{font-size:11px;color:var(--muted);margin:2px 0 6px}
.seg-card .desc{font-size:12px}
.cost-row{display:grid;grid-template-columns:30px 1fr auto;gap:10px;padding:10px;border-radius:6px;margin-bottom:6px;font-size:13px;align-items:center}
.cost-bad{background:#fdebec}.cost-warn{background:#fdf6e8}.cost-ok{background:#eaf6ed}
.cost-tag{font-weight:700;font-size:11px;padding:2px 8px;border-radius:4px;color:#fff}
.cost-tag.bad{background:var(--bad)}.cost-tag.warn{background:var(--warn)}.cost-tag.ok{background:var(--ok)}
.model-card{padding:16px;border-radius:8px;border:2px solid;background:#f8fafc;position:relative}
.model-card .badge{position:absolute;top:-10px;right:14px;padding:3px 10px;font-size:10px;font-weight:700;color:#fff;border-radius:4px}
.mm{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:10px}
.mm .m{background:#fff;border:1px solid var(--border);border-radius:6px;padding:8px}
.mm .m .l{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.mm .m .v{font-size:18px;font-weight:700;color:var(--pfd);margin-top:2px}
.cm-table{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}
.cm-table th,.cm-table td{padding:8px;text-align:center;border:1px solid var(--border)}
.cm-table th{background:#f4f6fa;font-weight:600;color:var(--pfd)}
.cm-table td.diag{font-weight:700}
.cm-table .rl{background:#f4f6fa;font-weight:600;color:var(--pfd);text-align:left;padding-left:12px}
.plot{width:100%;height:460px}
.controls{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;align-items:center}
.ctrl-label{font-size:12px;color:var(--muted);margin-right:4px}
.ctrl-btn{background:#fff;border:1px solid var(--border);border-radius:6px;padding:6px 12px;font-size:12px;cursor:pointer;font-family:inherit}
.ctrl-btn.active{background:var(--pfizer);color:#fff;border-color:var(--pfizer)}
.legend-row{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-bottom:8px}
.legend-item{display:inline-flex;align-items:center;gap:6px}
.swatch{width:12px;height:12px;border-radius:3px;display:inline-block}
.info{background:#fffbe8;border:1px solid #f0d780;border-radius:8px;padding:10px 14px;font-size:12px;color:#6b5500;margin-bottom:14px}
.info.tip{background:#eaf3fb;border-color:#b8d8ed;color:#0a4775}
.info.success{background:#eaf6ed;border-color:#a8d4b3;color:#1e5a32}
.explorer{display:grid;grid-template-columns:1.1fr 1fr;gap:14px}
@media(max-width:1100px){.explorer{grid-template-columns:1fr}}
.search-box{width:100%;padding:10px 12px;border:1px solid var(--border);border-radius:6px;font-size:14px;font-family:inherit;margin-bottom:10px}
.search-box:focus{outline:none;border-color:var(--pfizer)}
.profile{min-height:300px}
.profile-empty{text-align:center;color:var(--muted);padding:60px 20px;font-size:13px;font-style:italic}
.profile-header{padding-bottom:10px;border-bottom:1px solid var(--border);margin-bottom:12px}
.profile-id{font-size:18px;font-weight:700;color:var(--pfd)}
.profile-tags{display:flex;gap:8px;margin-top:6px;flex-wrap:wrap}
.tag{padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}
.tag-true{background:#eef2f7;color:var(--text)}.tag-pred{background:var(--pfizer);color:#fff}
.tag-conf{background:#f0f0f0;color:var(--muted)}
.prob-bar{display:flex;height:28px;border-radius:4px;overflow:hidden;background:#e8edf3;margin-top:6px}
.prob-bar>div{display:flex;align-items:center;justify-content:center;color:#fff;font-size:11px;font-weight:700}
.ci-row{display:grid;grid-template-columns:50px 1fr 70px;gap:8px;align-items:center;font-size:11px;margin-top:3px}
.ci-bar-wrap{position:relative;height:14px;background:#f0f3f8;border-radius:3px}
.ci-fill{position:absolute;top:0;height:100%;border-radius:3px;opacity:.4}
.ci-point{position:absolute;top:-1px;width:3px;height:16px;border-radius:1px}
.ci-label{font-weight:600;color:var(--pfd);text-align:right}
.ci-val{font-size:10px;color:var(--muted)}
.shap-title{font-size:13px;font-weight:700;color:var(--pfd);margin:14px 0 6px}
.shap-row{display:grid;grid-template-columns:140px 1fr 50px;gap:6px;align-items:center;padding:4px 0;border-bottom:1px solid #f4f6fa;font-size:12px}
.shap-fname{font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.shap-bar-wrap{position:relative;height:16px}
.shap-bar{position:absolute;top:0;height:100%;border-radius:2px}
.shap-bar.pos{background:var(--c);opacity:.7}.shap-bar.neg{background:var(--a);opacity:.7}
.shap-val{font-size:11px;font-weight:600;text-align:right}
.feat-row{display:grid;grid-template-columns:140px 55px 1fr;gap:8px;align-items:center;padding:6px 0;border-bottom:1px solid #f0f3f8;font-size:12px}
.feat-name{font-weight:600}.feat-name .desc{font-weight:400;color:var(--muted);font-size:10px;display:block}
.feat-value{font-weight:700;text-align:right;color:var(--pfd);font-size:13px}
.feat-bar-wrap{position:relative;height:16px;background:#f0f3f8;border-radius:3px}
.feat-bar-marker{position:absolute;top:-2px;width:3px;height:20px;background:#000;border-radius:1px;box-shadow:0 0 0 2px #fff;z-index:5}
.explanation{background:#f0f7fc;border-left:3px solid var(--pfizer);padding:12px 14px;border-radius:4px;font-size:13px;line-height:1.55;margin-top:14px}
.exp-block{background:#fff;border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:14px}
.exp-block h3{margin-top:0;color:var(--pfd);text-transform:none;letter-spacing:0;font-size:15px}
.exp-block .icon{display:inline-block;width:28px;height:28px;background:var(--pfizer);color:#fff;border-radius:50%;text-align:center;line-height:28px;font-weight:700;margin-right:8px;font-size:13px}
.conv-table{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}
.conv-table th,.conv-table td{padding:8px;text-align:left;border-bottom:1px solid var(--border)}
.conv-table th{background:#f4f6fa;font-weight:600;color:var(--pfd);text-transform:uppercase;font-size:10px;letter-spacing:.5px}
.conv-table tr:hover{background:#f4f8fc;cursor:pointer}
.conv-table tr.selected{background:#eaf3fb}
.conv-table .pc-cell{font-weight:700;color:var(--c)}
.conv-table .num{text-align:right;font-variant-numeric:tabular-nums}
.act-pill{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600}
.act-pill.actionable{background:#eaf6ed;color:var(--ok)}
.act-pill.signal{background:#eef2f7;color:var(--muted)}
.gap-bar{display:inline-flex;align-items:center;gap:6px;font-size:11px}
.gap-bar .bar{height:6px;background:#f0f3f8;border-radius:3px;width:80px;position:relative;overflow:hidden}
.gap-bar .fill{height:100%;border-radius:3px}
.gap-bar .fill.below{background:var(--bad)}.gap-bar .fill.above{background:var(--ok)}
footer{text-align:center;font-size:11px;color:var(--muted);margin-top:18px;padding:14px}
</style></head><body>
<header>
<h1>HCP Segmentation — Velsipity</h1>
<p>Business-Calibrated Ordinal Model + Dominance Rule · Tec de Monterrey × Pfizer Global Commercial Analytics</p>
</header>
<div class="tabs">
<button class="tab active" data-tab="overview">1 · Business & Model</button>
<button class="tab" data-tab="explore">2 · Doctor Explorer</button>
<button class="tab" data-tab="howitworks">3 · Understanding the Model</button>
<button class="tab" data-tab="conversion">4 · Conversion Strategy (B → C)</button>
</div>

<!-- ══════ TAB 1 — Business & Model ══════ -->
<div class="tab-content active" id="tab-overview">
<div class="panel">
<h2>The Business Problem</h2>
<p>Pfizer markets Velsipity, an oral therapy for moderately-to-severely active ulcerative colitis (UC), approved by the FDA in October 2023. The UC market is projected to grow from ~USD 9.5B today to ~USD 14.8B by 2032, with strong competition from biologics and JAK inhibitors.</p>
<p>To prioritize sales effort, Pfizer interviewed 205 HCPs to assign them to actionable segments. That manual segmentation was extended through internal labeling to <strong id="kpi_lab">—</strong> HCPs, but <strong id="kpi_unl">—</strong> HCPs remain untyped. The total addressable network is <strong id="kpi_total">—</strong> HCPs.</p>
<p><strong>Goal:</strong> deploy a model that scales this segmentation to every HCP — including those never interviewed — so sales and marketing can decide who to engage, who to educate, and who to skip.</p>
</div>

<div class="panel">
<h2>The Three Segments</h2>
<div class="grid3">
<div class="seg-card seg-A"><div class="name" style="color:var(--a)">SEG_A — Will NOT prescribe</div>
<div class="pct" id="kpi_a"></div><div class="desc">Low UC activity. Don't waste sales effort here.</div></div>
<div class="seg-card seg-B"><div class="name" style="color:var(--b)">SEG_B — Neutral / undecided</div>
<div class="pct" id="kpi_b"></div><div class="desc">Ambiguous middle ground. Educate and persuade.</div></div>
<div class="seg-card seg-C"><div class="name" style="color:var(--c)">SEG_C — Will prescribe</div>
<div class="pct" id="kpi_c"></div><div class="desc">Active UC prescribers. Top priority — never miss.</div></div>
</div></div>

<div class="panel">
<h2>The Model in Plain Language</h2>
<p>We trained an <strong>ordinal XGBoost classifier</strong> — a two-stage system that treats the segments as ordered (A &lt; B &lt; C, by prescribing intent) rather than three unrelated categories. Stage 1 asks "is this doctor at least B?". Stage 2 asks "is this doctor at least C?". The model combines both answers into a final assignment.</p>

<h3>Why we don't just pick the highest probability</h3>
<p>A standard model would assign whatever class has the highest probability. The problem: missing a real prescriber (true SEG_C classified as SEG_A) is a <strong>catastrophic, recurring revenue loss</strong>, while visiting a non-prescriber (true SEG_A classified as SEG_C) is a one-time wasted visit. These errors are not equal — but a naive model treats them as if they were.</p>

<div class="cost-row cost-bad"><div>✕</div><div><strong>C classified as A</strong> — Real prescriber abandoned. Recurring revenue lost across many patients.</div><div class="cost-tag bad">CATASTROPHIC</div></div>
<div class="cost-row cost-warn"><div>!</div><div><strong>A classified as C</strong> — Rep visits a non-prescriber. One-time cost, fully recoverable.</div><div class="cost-tag warn">COSTLY</div></div>
<div class="cost-row cost-ok"><div>~</div><div><strong>Any B error</strong> — B is inherently ambiguous; errors there are expected.</div><div class="cost-tag ok">ACCEPTABLE</div></div>

<h3>Our decision logic</h3>
<p>To reflect this asymmetric cost, the model uses thresholds, not argmax:</p>
<ol style="margin-left:20px;font-size:13px;line-height:1.7">
<li><strong>If P(A) ≥ 70%</strong> → classify as SEG_A. We demand high confidence before abandoning a doctor.</li>
<li><strong>Else if P(C) ≥ 30%</strong> → classify as SEG_C. A low bar to flag a prescriber.</li>
<li><strong>Else if P(C) &gt; P(B)</strong> → classify as SEG_C. Dominance rule: when the model itself thinks C is more likely than B, we don't default to the ambiguous category.</li>
<li><strong>Otherwise</strong> → SEG_B (residual, ambiguous).</li>
</ol>
<div class="info success" style="margin-top:12px;margin-bottom:0">
The dominance rule rescued <strong>87 real prescribers</strong> who were being labeled B despite the model giving them higher C probability — pure free recall gain at zero cost.
</div>
</div>

<div class="panel">
<h2>Recommended Model vs Baseline</h2>
<div class="grid2">
<div class="model-card" style="border-color:var(--pfizer)">
<span class="badge" style="background:var(--pfizer)">RECOMMENDED ★</span>
<h3 style="color:var(--pfd);margin-top:0;text-transform:none;letter-spacing:0">Ordinal + Dominance Rule</h3>
<p style="font-size:11px;color:var(--muted)">Two-stage XGBoost · thresholds + dominance · weighted for C</p>
<p>Optimized for <strong>minimizing catastrophic C→A errors</strong> and finding prescribers — even at the cost of slightly lower headline accuracy.</p>
<div class="mm">
<div class="m"><div class="l">Balanced Accuracy</div><div class="v" id="ord_ba">—</div></div>
<div class="m"><div class="l">Recall SEG_C</div><div class="v" id="ord_rc">—</div></div>
<div class="m"><div class="l" style="color:var(--ok)">C→A losses</div><div class="v" id="ord_cl" style="color:var(--ok)">—</div></div>
<div class="m"><div class="l">Overall Accuracy</div><div class="v" id="ord_ac">—</div></div>
</div><h3>Confusion Matrix (5-fold OOF)</h3><div id="cm_ord"></div></div>
<div class="model-card" style="border-color:#7c4dff">
<span class="badge" style="background:#7c4dff">BASELINE</span>
<h3 style="color:#7c4dff;margin-top:0;text-transform:none;letter-spacing:0">Argmax (Standard ML)</h3>
<p style="font-size:11px;color:var(--muted)">Multi-class XGBoost · class_weight balanced · argmax</p>
<p>Higher headline accuracy, but treats every error as equal — and ends up <strong>losing 25% of real prescribers</strong> as SEG_A.</p>
<div class="mm">
<div class="m"><div class="l">Balanced Accuracy</div><div class="v" id="amx_ba">—</div></div>
<div class="m"><div class="l">Recall SEG_C</div><div class="v" id="amx_rc">—</div></div>
<div class="m"><div class="l" style="color:var(--bad)">C→A losses</div><div class="v" id="amx_cl" style="color:var(--bad)">—</div></div>
<div class="m"><div class="l">Overall Accuracy</div><div class="v" id="amx_ac">—</div></div>
</div><h3>Confusion Matrix (5-fold OOF)</h3><div id="cm_amx"></div></div>
</div>
<div class="info tip" style="margin-top:14px;margin-bottom:0">
<strong>How to read this trade-off:</strong> the argmax baseline reaches higher overall accuracy by aggressively classifying as SEG_A — but it abandons 537 real prescribers along the way. The recommended model finds 1,325 prescribers (61.8% recall) while only losing 355 as A (16.6%). For Pfizer, every saved prescriber translates to recurring Velsipity revenue.
</div>
</div>

<div class="panel">
<h2>Validation Methodology</h2>
<p><strong>5-fold stratified out-of-fold cross-validation.</strong> The 11,899 labeled HCPs are split into 5 groups. For each fold, 80% of the data trains the model and the remaining 20% is predicted — rotating so every doctor receives exactly one prediction from a model that never saw them during training. This is methodologically equivalent to a held-out test set without sacrificing training data.</p>
<p><strong>Why no separate test set:</strong> with only ~2,144 SEG_C labels (the critical class), holding out 20% would meaningfully hurt model quality. OOF avoids this trade-off while remaining fully honest about generalization performance.</p>
</div>
</div>

<!-- ══════ TAB 2 — Doctor Explorer ══════ -->
<div class="tab-content" id="tab-explore">
<div class="info tip"><strong>How to use:</strong> click any dot in the scatter plot or search by HCP ID. The right panel shows the doctor's predicted segment, probability distribution with confidence interval, SHAP explanation, and feature comparison.</div>

<div class="panel">
<h2>Probability Maps</h2>
<div class="controls">
<span class="ctrl-label">Color by:</span>
<button class="ctrl-btn active" data-color="true">True label (gray = untyped)</button>
<button class="ctrl-btn" data-color="pred">Predicted (incl. untyped)</button>
<button class="ctrl-btn" data-color="errors">Show errors</button>
<button class="ctrl-btn" data-color="confidence">Confidence</button>
</div>
<div class="legend-row">
<span class="legend-item"><span class="swatch" style="background:var(--a)"></span>SEG_A</span>
<span class="legend-item"><span class="swatch" style="background:var(--b)"></span>SEG_B</span>
<span class="legend-item"><span class="swatch" style="background:var(--c)"></span>SEG_C</span>
<span class="legend-item"><span class="swatch" style="background:var(--u)"></span>Untyped (in "True label" mode)</span>
</div>
<div class="info tip" style="margin-bottom:10px"><strong>How to read this view:</strong> each dot's <em>position</em> shows where the model placed that doctor (its predicted probabilities P(A), P(B), P(C)). The dot's <em>color</em> depends on the mode you select — see Tab 3 for full explanations of each mode.</div>
<div class="grid2">
<div><h3>Ternary Simplex</h3><div id="plot_ternary" class="plot"></div></div>
<div><h3>3D Scatter</h3><div id="plot_3d" class="plot"></div></div>
</div></div>

<div class="panel">
<h2>Doctor Profile Explorer</h2>
<div class="explorer">
<div>
<input type="text" class="search-box" id="search" placeholder="Search by HCP ID...">
<div id="search_result" style="font-size:12px;color:var(--muted);margin-bottom:8px"></div>
<h3>UC Rx vs Oral Rx — colored by prediction</h3>
<div id="plot_scatter" class="plot" style="height:420px"></div>
</div>
<div><h3>Selected Doctor</h3>
<div id="profile" class="profile">
<div class="profile-empty">Click a dot or search by HCP ID to see the full breakdown.</div>
</div></div>
</div></div>
</div>

<!-- ══════ TAB 3 — Understanding the Model ══════ -->
<div class="tab-content" id="tab-howitworks">
<div class="info tip">This tab explains the three explanation tools used in the Doctor Explorer: <strong>SHAP values</strong>, <strong>Confidence Intervals</strong>, and the <strong>Confidence</strong> view.</div>

<div class="exp-block">
<h3><span class="icon">1</span>What is SHAP?</h3>
<p><strong>SHAP (SHapley Additive exPlanations)</strong> answers the question: "For this specific doctor, which features drove the model's prediction, and in which direction?"</p>
<p>SHAP is grounded in game theory. It treats each feature as a "player" contributing to the final prediction, and computes how much each feature pushed the prediction up or down relative to the average. For Velsipity, our SHAP values explain the model's <strong>P(C)</strong> score — the probability of being a prescriber.</p>
<p style="background:#f8fafc;padding:10px;border-radius:6px;font-size:12px"><strong>Reading SHAP bars in the Doctor Profile:</strong></p>
<ul style="margin-left:20px;font-size:13px;line-height:1.7">
<li><span style="color:var(--c);font-weight:700">Red bar (positive value):</span> this feature is pushing the doctor TOWARD SEG_C. Example: high UC_TRX → red bar → strong prescriber signal.</li>
<li><span style="color:var(--a);font-weight:700">Blue bar (negative value):</span> this feature is pushing AWAY from SEG_C. Example: zero DETAILS visits → blue bar → low engagement reduces P(C).</li>
<li><strong>Length</strong> = strength of effect. Long bars are the dominant drivers; short bars are minor adjustments.</li>
</ul>
<p style="margin-top:10px"><strong>Business value:</strong> SHAP makes every prediction defensible. If a sales rep asks "why is doctor 12345 classified as B?", we can answer with the top 5 features driving that exact decision.</p>
</div>

<div class="exp-block">
<h3><span class="icon">2</span>What are Confidence Intervals (CI)?</h3>
<p>Our model is trained 5 times — once per fold of the cross-validation. Each of the 5 trained models gives slightly different probabilities for any given doctor. The <strong>95% Confidence Interval</strong> shows the range of these 5 estimates: roughly speaking, "where the true probability is likely to fall."</p>
<p style="background:#f8fafc;padding:10px;border-radius:6px;font-size:12px"><strong>Reading the CI bars in the Doctor Profile:</strong></p>
<ul style="margin-left:20px;font-size:13px;line-height:1.7">
<li><strong>Narrow interval (e.g. 0.85–0.91):</strong> the 5 models agree. The prediction is <span style="color:var(--ok);font-weight:600">stable</span>. Safe to act on.</li>
<li><strong>Wide interval (e.g. 0.30–0.75):</strong> the 5 models disagree. The doctor is <span style="color:var(--bad);font-weight:600">on the edge</span>. Worth human review before committing budget.</li>
</ul>
<p style="margin-top:10px"><strong>Business value:</strong> CI tells you which predictions to trust and which to verify. A stable C with narrow CI is a great priority. A C with wide CI may flip to B next quarter — treat with care.</p>
</div>

<div class="exp-block">
<h3><span class="icon">3</span>What is the "Confidence" color mode?</h3>
<p>In the probability maps (Tab 2), the <strong>Confidence</strong> button colors each dot by the <strong>maximum of P(A), P(B), P(C)</strong> — that is, "how sure is the model of whatever prediction it made?"</p>
<p style="background:#f8fafc;padding:10px;border-radius:6px;font-size:12px"><strong>Reading the Confidence map:</strong></p>
<ul style="margin-left:20px;font-size:13px;line-height:1.7">
<li><strong>Bright yellow / green:</strong> high confidence (≥70%). These doctors are deep in one corner of the probability simplex. Safe to commit resources.</li>
<li><strong>Dark blue / purple:</strong> low confidence (around 33–50%). These doctors live near the center, where all three classes are roughly equally likely. Borderline cases.</li>
</ul>
<p style="margin-top:10px"><strong>Difference from CI:</strong> Confidence asks "how peaked is the model's belief?" — a single number per doctor. CI asks "how reliable is that belief across re-trainings?" — a range. The two are complementary: a doctor can have <em>high confidence</em> in their prediction but <em>wide CI</em> if the model is consistently bold but the 5 folds disagree about how bold.</p>
</div>

<div class="exp-block" style="background:#f0f7fc;border-color:var(--pfizer)">
<h3 style="color:var(--pfd)">Putting it all together</h3>
<p style="font-size:13px">For any predicted SEG_C doctor, ask three questions:</p>
<ol style="margin-left:20px;font-size:13px;line-height:1.8">
<li><strong>SHAP — Why?</strong> What features made the model classify this person as C? Are those features something we believe in (high UC_TRX, frequent details) or anomalies?</li>
<li><strong>CI — How stable?</strong> Is the prediction robust across folds, or is it fragile?</li>
<li><strong>Confidence — How sure?</strong> Is the model strongly betting on C, or just barely choosing C over the alternatives?</li>
</ol>
<p style="font-size:13px;margin-top:8px">A doctor that scores well on all three (sensible SHAP drivers, narrow CI, high confidence) is a high-quality prescriber lead. A doctor that scores well on one but not the others deserves a closer human look.</p>
</div>
</div>

<!-- ══════ TAB 4 — Conversion Strategy (B → C) ══════ -->
<div class="tab-content" id="tab-conversion">
<div class="panel">
<h2>From SEG_B to SEG_C: A Conversion Strategy</h2>
<p>SEG_B is the residual, ambiguous segment — doctors the model is uncertain about. But within SEG_B, some doctors lean strongly toward becoming prescribers (high P(C)) while others lean toward inactivity (high P(A)). The first group is a <strong>conversion opportunity</strong>: doctors who are not yet C but show signals of becoming C with the right engagement.</p>
<p>This tab identifies the top <strong id="conv_n">—</strong> B-predicted doctors ranked by P(C), and analyzes <strong>what's holding them back</strong> from being classified as C — using the same features the model relies on.</p>
</div>

<div class="panel">
<h2>What's Holding B Doctors Back?</h2>
<p>For each conversion candidate, we compare their feature values against the <strong>SEG_C median</strong> — the typical prescriber profile. Features where the candidate is below the C median are gaps to close. We separate gaps into two categories:</p>
<div class="grid2" style="margin-top:10px">
<div class="seg-card" style="border-left-color:var(--ok)">
<div class="name" style="color:var(--ok)">Actionable gaps (Pfizer-controlled)</div>
<div class="desc" style="margin-top:6px">DETAILS_sum (rep visits), SAMPLES_sum. Pfizer can directly increase these through targeted sales-force allocation.</div>
</div>
<div class="seg-card" style="border-left-color:var(--muted)">
<div class="name" style="color:var(--muted)">Signal gaps (prescribing behavior)</div>
<div class="desc" style="margin-top:6px">UC_TRX, UC_NRX, ORAL_TRX. These reflect the doctor's existing patient base and habits. Not directly controllable, but indicate strategic fit.</div>
</div>
</div>

<h3>Aggregate view: across all candidates</h3>
<p>The chart below shows the percentage of conversion candidates whose value is below the SEG_C median for each key feature. Higher percentages = more common gaps = bigger lever for conversion campaigns.</p>
<div id="conv_aggregate_plot" style="width:100%;height:340px"></div>
</div>

<div class="panel">
<h2>Top Conversion Candidates</h2>
<p style="font-size:12px;color:var(--muted)">Click any row to see that doctor's full gap analysis. Sorted by P(C) descending.</p>
<div style="max-height:520px;overflow-y:auto;border:1px solid var(--border);border-radius:6px">
<table class="conv-table" id="conv_table">
<thead><tr>
<th>HCP ID</th><th class="num">P(B)</th><th class="num">P(C)</th><th>Top Action</th><th class="num">Current</th><th class="num">Target (C median)</th><th>Gap</th>
</tr></thead>
<tbody id="conv_tbody"></tbody>
</table>
</div>
</div>

<div class="panel">
<h2>Selected Candidate — Detail</h2>
<div id="conv_detail">
<div class="profile-empty">Click any row in the table above to see the full gap breakdown.</div>
</div>
</div>
</div>

<footer>Capstone Project · Tec de Monterrey × Pfizer Global Commercial Analytics · 5-fold stratified OOF · TreeSHAP · Cross-fold 95% CI · v3 (Conversion Strategy)</footer>

<script>
const D=__DATA_PLACEHOLDER__;
const COL={SEG_A:'#2e6cb0',SEG_B:'#f0a830',SEG_C:'#c83a3a',UNLABELED:'#9aa6b4'};
const M=D.metrics;let colorMode='true';

// Fill overview KPIs
document.getElementById('kpi_total').textContent=(M.n_labeled+M.n_unlabeled).toLocaleString();
document.getElementById('kpi_lab').textContent=M.n_labeled.toLocaleString();
document.getElementById('kpi_unl').textContent=M.n_unlabeled.toLocaleString();
document.getElementById('kpi_a').textContent=(M.dist_A*100).toFixed(1)+'% of labeled ('+Math.round(M.n_labeled*M.dist_A).toLocaleString()+' HCPs)';
document.getElementById('kpi_b').textContent=(M.dist_B*100).toFixed(1)+'% of labeled ('+Math.round(M.n_labeled*M.dist_B).toLocaleString()+' HCPs)';
document.getElementById('kpi_c').textContent=(M.dist_C*100).toFixed(1)+'% of labeled ('+Math.round(M.n_labeled*M.dist_C).toLocaleString()+' HCPs)';

['ord','amx'].forEach(k=>{const m=M[k];
document.getElementById(k+'_ba').textContent=(m.balanced_accuracy*100).toFixed(1)+'%';
document.getElementById(k+'_rc').textContent=(m.recall_C*100).toFixed(1)+'%';
document.getElementById(k+'_cl').textContent=(m.c_lost_pct*100).toFixed(1)+'%';
document.getElementById(k+'_ac').textContent=(m.accuracy*100).toFixed(1)+'%';});

function renderCM(id,cm){const L=['SEG_A','SEG_B','SEG_C'];
let h='<table class="cm-table"><thead><tr><th></th>';L.forEach(l=>h+='<th>'+l+'</th>');
h+='</tr></thead><tbody>';cm.forEach((r,i)=>{h+='<tr><td class="rl">'+L[i]+'</td>';
r.forEach((v,j)=>{const d=i===j,t=r.reduce((a,b)=>a+b,0),p=t>0?(v/t*100).toFixed(0):0;
h+='<td class="'+(d?'diag':'')+'" style="'+(d?'background:#eaf6ed':'')+'">'+v.toLocaleString()+
'<br><span style="color:var(--muted);font-size:10px">'+p+'%</span></td>';});h+='</tr>';});
h+='</tbody></table>';document.getElementById(id).innerHTML=h;}
renderCM('cm_ord',M.ord.confusion_matrix);renderCM('cm_amx',M.amx.confusion_matrix);

// Viz data
const V=D.viz,ALL=D.all,FEATS=D.features,STATS=D.segment_stats,FN=D.feat_names;
const CONV=D.conversion;

function bh(){return V.hcp_id.map((_,i)=>{const t=V.is_labeled[i]?V.true_seg[i]:'(untyped)';
return 'HCP '+V.hcp_id[i]+'<br>True: <b>'+t+'</b> Pred: <b>'+V.pred[i]+'</b><br>'+
'P(A)='+(V.P_A[i]*100).toFixed(1)+'% P(B)='+(V.P_B[i]*100).toFixed(1)+'% P(C)='+(V.P_C[i]*100).toFixed(1)+'%';});}

function gc(){
  if(colorMode==='true')return V.true_seg.map((s,i)=>V.is_labeled[i]?(COL[s]||COL.UNLABELED):COL.UNLABELED);
  if(colorMode==='pred')return V.pred.map(s=>COL[s]||COL.UNLABELED);
  if(colorMode==='errors')return V.true_seg.map((s,i)=>{
    if(!V.is_labeled[i])return'rgba(200,200,200,0.15)';
    return V.pred[i]===s?'rgba(200,200,200,0.25)':COL[s];});
  return V.P_A.map((_,i)=>Math.max(V.P_A[i],V.P_B[i],V.P_C[i]));
}

function gsize(){if(colorMode!=='errors')return 5;
return V.true_seg.map((s,i)=>(!V.is_labeled[i]||V.pred[i]===s)?3:9);}

function plotT(){const c=gc(),isC=colorMode==='confidence',h=bh(),sz=gsize();
Plotly.react('plot_ternary',[{type:'scatterternary',mode:'markers',a:V.P_A,b:V.P_B,c:V.P_C,
text:h,hoverinfo:'text',marker:{size:sz,color:c,colorscale:isC?'Viridis':undefined,
cmin:isC?.33:undefined,cmax:isC?1:undefined,showscale:isC,opacity:.8}}],
{ternary:{sum:1,aaxis:{title:'P(A)',min:0},baxis:{title:'P(B)',min:0},caxis:{title:'P(C)',min:0},
bgcolor:'#fafbfd'},margin:{l:50,r:30,t:20,b:30},showlegend:false},{responsive:true,displaylogo:false});}

function plot3(){const c=gc(),isC=colorMode==='confidence',h=bh(),sz=gsize();
Plotly.react('plot_3d',[{type:'scatter3d',mode:'markers',x:V.P_A,y:V.P_B,z:V.P_C,
text:h,hoverinfo:'text',marker:{size:colorMode==='errors'?sz.map(s=>s*.6):3,color:c,colorscale:isC?'Viridis':undefined,opacity:.8}}],
{scene:{xaxis:{title:'P(A)',range:[0,1]},yaxis:{title:'P(B)',range:[0,1]},zaxis:{title:'P(C)',range:[0,1]},
camera:{eye:{x:1.4,y:1.4,z:1}}},margin:{l:0,r:0,t:0,b:0}},{responsive:true,displaylogo:false});}

function plotS(){const c=V.pred.map(s=>COL[s]),h=bh();
Plotly.react('plot_scatter',[{type:'scattergl',mode:'markers',x:V.log_uc,y:V.log_oral,
text:h,hoverinfo:'text',customdata:V.hcp_id,marker:{size:6,color:c,opacity:.6}}],
{xaxis:{title:'log(1+UC_TRX)',gridcolor:'#eef2f7'},yaxis:{title:'log(1+ORAL_TRX)',gridcolor:'#eef2f7'},
margin:{l:50,r:20,t:10,b:40},plot_bgcolor:'#fafbfd',hovermode:'closest'},{responsive:true,displaylogo:false});
document.getElementById('plot_scatter').on('plotly_click',ev=>{showProfile(ev.points[0].customdata);});}

function rerender(){plotT();plot3();plotS();}

// Tab switching
document.querySelectorAll('.tab').forEach(t=>t.addEventListener('click',()=>{
document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
document.querySelectorAll('.tab-content').forEach(x=>x.classList.remove('active'));
t.classList.add('active');document.getElementById('tab-'+t.dataset.tab).classList.add('active');
if(t.dataset.tab==='explore')rerender();
if(t.dataset.tab==='conversion')renderConversion();
}));

document.querySelectorAll('.ctrl-btn[data-color]').forEach(b=>b.addEventListener('click',()=>{
document.querySelectorAll('.ctrl-btn[data-color]').forEach(x=>x.classList.remove('active'));
b.classList.add('active');colorMode=b.dataset.color;plotT();plot3();}));

// ─── Doctor Profile ───
function fmt(v){if(v==null)return'—';if(Math.abs(v)>=100)return v.toFixed(0);if(Math.abs(v)>=10)return v.toFixed(1);return v.toFixed(2);}

function showProfile(id){id=String(id);const d=ALL[id];
if(!d){document.getElementById('profile').innerHTML='<div class="profile-empty">HCP not found.</div>';return;}
const pa=(d.pa*100).toFixed(0),pb=(d.pb*100).toFixed(0),pc=(d.pc*100).toFixed(0);
const maxP=Math.max(d.pa,d.pb,d.pc);
const confL=maxP>=.7?'High':maxP>=.5?'Moderate':'⚠ Low';
const confC=maxP>=.7?'var(--ok)':maxP>=.5?'var(--warn)':'var(--bad)';
const tagT=d.il?'<span class="tag tag-true">True: '+d.t+'</span>':'<span class="tag tag-true">Untyped</span>';
const tagP='<span class="tag tag-pred">Predicted: '+d.p+'</span>';
const tagC='<span class="tag tag-conf" style="color:'+confC+'">'+confL+' ('+Math.round(maxP*100)+'%)</span>';

// CI bars (simplified, no big table)
const ciWidth=d.ci.hi.map((h,i)=>h-d.ci.lo[i]);
const maxCIW=Math.max(...ciWidth);
const ciStability=maxCIW<0.15?'Stable':maxCIW<0.30?'Moderate':'⚠ Unstable';
const ciStabColor=maxCIW<0.15?'var(--ok)':maxCIW<0.30?'var(--warn)':'var(--bad)';
let ciH='<div style="font-size:11px;color:var(--muted);margin:8px 0 4px">95% CONFIDENCE INTERVAL (across 5 folds)</div>'+
'<div style="font-size:10px;color:var(--muted);margin-bottom:6px">Stability: <span style="color:'+ciStabColor+';font-weight:700">'+ciStability+'</span> — see Tab 3 for what this means.</div>';
['A','B','C'].forEach((s,i)=>{const lo=d.ci.lo[i],hi=d.ci.hi[i],mn=d.ci.mean[i],c=COL['SEG_'+s];
ciH+='<div class="ci-row"><div class="ci-label">P('+s+')</div><div class="ci-bar-wrap">'+
'<div class="ci-fill" style="left:'+(lo*100)+'%;width:'+((hi-lo)*100)+'%;background:'+c+'"></div>'+
'<div class="ci-point" style="left:calc('+(mn*100)+'% - 1.5px);background:'+c+'"></div></div>'+
'<div class="ci-val">'+(lo*100).toFixed(0)+'–'+(hi*100).toFixed(0)+'%</div></div>';});

// SHAP
let shH='';const sh=d.shap||[];
if(sh.length){const mx=Math.max(...sh.map(s=>Math.abs(s[1])))||1;
shH='<div style="display:flex;gap:16px;font-size:10px;color:var(--muted);margin-bottom:4px">'+
'<span class="legend-item"><span class="swatch" style="background:var(--c);opacity:.7"></span>Pushes toward C</span>'+
'<span class="legend-item"><span class="swatch" style="background:var(--a);opacity:.7"></span>Pushes away from C</span></div>';
sh.forEach(([fi,val])=>{const fn=FN[fi]||'f'+fi;const pct=Math.min(Math.abs(val)/mx*100,100);
const pos=val>=0;const bs=pos?'left:50%;width:'+pct/2+'%':'left:'+(50-pct/2)+'%;width:'+pct/2+'%';
shH+='<div class="shap-row"><div class="shap-fname" title="'+fn+'">'+fn+'</div>'+
'<div class="shap-bar-wrap"><div style="position:absolute;left:50%;top:0;width:1px;height:100%;background:#ddd"></div>'+
'<div class="shap-bar '+(pos?'pos':'neg')+'" style="'+bs+'"></div></div>'+
'<div class="shap-val" style="color:'+(pos?'var(--c)':'var(--a)')+'">'+(val>=0?'+':'')+val.toFixed(3)+'</div></div>';});}

// Feature rows
let fH='';FEATS.forEach((f,i)=>{const st=STATS[f.key];if(!st)return;const v=d.f[i];
const mx=Math.max(st.SEG_A.p75,st.SEG_B.p75,st.SEG_C.p75)*1.3+.01;
const mp=Math.max(0,Math.min(100,v/mx*100));
const bar=(seg)=>{const s=st[seg];return'position:absolute;top:0;left:'+Math.max(0,s.p25/mx*100)+'%;width:'+
Math.max(0,(s.p75-s.p25)/mx*100)+'%;height:100%;background:'+COL[seg]+';opacity:.5';};
fH+='<div class="feat-row"><div class="feat-name">'+f.name+'<span class="desc">'+f.desc+'</span></div>'+
'<div class="feat-value">'+fmt(v)+'</div><div><div class="feat-bar-wrap">'+
'<div style="'+bar('SEG_A')+'"></div><div style="'+bar('SEG_B')+'"></div><div style="'+bar('SEG_C')+'"></div>'+
'<div class="feat-bar-marker" style="left:calc('+mp+'% - 1.5px)"></div></div></div></div>';});

let expl='';
if(d.p==='SEG_A')expl='Classified as <strong>SEG_A</strong> because P(A) = '+pa+'% ≥ 70% threshold.';
else if(d.p==='SEG_C'){
  if(parseFloat(pc)>=30)expl='Classified as <strong>SEG_C</strong> because P(C) = '+pc+'% ≥ 30% threshold.';
  else expl='Classified as <strong>SEG_C</strong> via dominance rule: P(C) = '+pc+'% > P(B) = '+pb+'%. The model favors C over B even at low absolute probability.';
}
else expl='Classified as <strong>SEG_B</strong> (ambiguous): neither P(A) ≥ 70% nor P(C) ≥ 30%, and P(B) ≥ P(C).';

document.getElementById('profile').innerHTML=
'<div class="profile-header"><div class="profile-id">HCP '+id+'</div><div class="profile-tags">'+tagT+tagP+tagC+'</div></div>'+
'<div style="font-size:11px;color:var(--muted);margin-top:10px">PROBABILITY DISTRIBUTION</div>'+
'<div class="prob-bar"><div style="background:'+COL.SEG_A+';width:'+pa+'%">'+(pa>10?'A '+pa+'%':'')+'</div>'+
'<div style="background:'+COL.SEG_B+';width:'+pb+'%">'+(pb>10?'B '+pb+'%':'')+'</div>'+
'<div style="background:'+COL.SEG_C+';width:'+pc+'%">'+(pc>10?'C '+pc+'%':'')+'</div></div>'+ciH+
'<div class="shap-title">SHAP — What drove this prediction?</div>'+shH+
'<h3>Key Features vs Segment Ranges</h3>'+
'<div style="display:flex;gap:10px;font-size:10px;color:var(--muted);margin-bottom:4px">'+
'<span class="legend-item"><span class="swatch" style="background:var(--a);opacity:.5"></span>A range</span>'+
'<span class="legend-item"><span class="swatch" style="background:var(--b);opacity:.5"></span>B range</span>'+
'<span class="legend-item"><span class="swatch" style="background:var(--c);opacity:.5"></span>C range</span>'+
'<span class="legend-item"><span class="swatch" style="background:#000"></span>This doctor</span></div>'+fH+
'<div class="explanation"><strong>Decision rule:</strong> '+expl+'</div>';}

document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.trim();
const r=document.getElementById('search_result');if(!q){r.textContent='';return;}
if(ALL[q]){r.textContent='';showProfile(q);}else r.textContent='Not found: "'+q+'"';});

// ─── Conversion Strategy ───
function renderConversion(){
  if(!CONV||!CONV.candidates)return;
  document.getElementById('conv_n').textContent=CONV.candidates.length.toLocaleString();

  // Aggregate plot
  const agg=CONV.aggregate.features;
  const x=agg.map(f=>f.name);
  const y=agg.map(f=>f.pct_below_C_median);
  const colors=agg.map(f=>f.actionable?'#2a8b4a':'#9aa6b4');
  Plotly.react('conv_aggregate_plot',[{
    type:'bar',x:x,y:y,marker:{color:colors},
    text:y.map(v=>v.toFixed(0)+'%'),textposition:'outside',
    hovertemplate:'%{x}<br>%{y:.1f}% below C median<extra></extra>'
  }],{
    margin:{l:50,r:20,t:20,b:80},
    yaxis:{title:'% of candidates below C median',range:[0,110],gridcolor:'#eef2f7'},
    xaxis:{tickangle:-25},plot_bgcolor:'#fafbfd',
    annotations:[{xref:'paper',yref:'paper',x:0,y:1.05,text:'Green = actionable (Pfizer-controlled) · Gray = signal only',showarrow:false,font:{size:10,color:'#6b7c93'}}]
  },{responsive:true,displaylogo:false});

  // Table
  const tbody=document.getElementById('conv_tbody');
  let h='';
  CONV.candidates.forEach((c,idx)=>{
    const ta=c.top_action;
    const actHtml=ta?'<span class="act-pill actionable">↑ '+ta.feature_name+'</span>':'<span class="act-pill signal">No clear action</span>';
    const cur=ta?fmt(ta.current):'—';
    const tgt=ta?fmt(ta.target):'—';
    const gap=ta?ta.gap.toFixed(2):'—';
    h+='<tr data-idx="'+idx+'" onclick="showConvDetail('+idx+')">'+
       '<td><strong>'+c.id+'</strong></td>'+
       '<td class="num">'+(c.pb*100).toFixed(0)+'%</td>'+
       '<td class="num pc-cell">'+(c.pc*100).toFixed(0)+'%</td>'+
       '<td>'+actHtml+'</td>'+
       '<td class="num">'+cur+'</td>'+
       '<td class="num">'+tgt+'</td>'+
       '<td class="num" style="color:var(--bad)">'+gap+'</td>'+
       '</tr>';
  });
  tbody.innerHTML=h;
}

window.showConvDetail=function(idx){
  document.querySelectorAll('#conv_table tbody tr').forEach(t=>t.classList.remove('selected'));
  document.querySelector('#conv_table tbody tr[data-idx="'+idx+'"]').classList.add('selected');
  const c=CONV.candidates[idx];
  const fullDoc=ALL[c.id];

  let gapsHtml='<h3 style="margin-top:0">Feature gaps vs SEG_C median</h3>';
  const feats=CONV.aggregate.features;
  feats.forEach(fm=>{
    const g=c.gaps[fm.key];if(!g)return;
    const pill=fm.actionable?'<span class="act-pill actionable">Actionable</span>':'<span class="act-pill signal">Signal</span>';
    const gapColor=g.below?'var(--bad)':'var(--ok)';
    const gapPct=Math.min(100,Math.abs(g.gap)/(Math.max(g.c_median,1)+0.01)*100);
    gapsHtml+='<div style="display:grid;grid-template-columns:180px 90px 1fr 90px;gap:10px;align-items:center;padding:8px 0;border-bottom:1px solid #f0f3f8;font-size:12px">'+
      '<div><strong>'+fm.name+'</strong> '+pill+'</div>'+
      '<div class="num" style="font-weight:700">'+fmt(g.value)+'</div>'+
      '<div class="gap-bar"><div class="bar"><div class="fill '+(g.below?'below':'above')+'" style="width:'+gapPct+'%"></div></div>'+
      '<span style="color:'+gapColor+';font-weight:600">'+(g.gap>0?'+':'')+g.gap.toFixed(2)+'</span></div>'+
      '<div class="num" style="color:var(--muted)">target: '+fmt(g.c_median)+'</div>'+
      '</div>';
  });

  // Plain-English recommendation
  const ta=c.top_action;
  let recHtml='';
  if(ta){
    recHtml='<div class="info success" style="margin-top:14px"><strong>Recommended action:</strong> Increase '+ta.feature_name+' from '+fmt(ta.current)+' toward '+fmt(ta.target)+' (gap of '+ta.gap.toFixed(2)+'). This is a Pfizer-controllable lever — direct sales-force allocation can close this gap.</div>';
  }else{
    recHtml='<div class="info" style="margin-top:14px"><strong>Note:</strong> No actionable engagement gap identified — this doctor\'s gap is in prescribing behavior (signal), not in Pfizer engagement. They may need a different approach: medical education, KOL referral, or patient-pathway intervention.</div>';
  }

  document.getElementById('conv_detail').innerHTML=
    '<div class="profile-header"><div class="profile-id">HCP '+c.id+'</div>'+
    '<div class="profile-tags">'+
    '<span class="tag tag-pred">Predicted: SEG_B</span>'+
    '<span class="tag" style="background:var(--c);color:#fff">P(C) = '+(c.pc*100).toFixed(0)+'%</span>'+
    (c.true?'<span class="tag tag-true">True (validation): '+c.true+'</span>':'<span class="tag tag-true">Untyped</span>')+
    '</div></div>'+
    gapsHtml+recHtml;
};

rerender();
</script></body></html>'''


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# SEGMENT INTELLIGENCE — static HTML tab (centroids_v2 content,
# rendered server-side and injected into the v3 template)
# ═══════════════════════════════════════════════════════════════
_SEV_COLOR = {"info": "#0070BF", "warning": "#F47B20",
              "danger": "#C0392B", "success": "#00A651"}


def _esc(x):
    """HTML-escape a value for safe insertion into the report."""
    return (str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build_intel_tab_html(insights, centroids_df, protos_df, levers_df,
                         at_risk_df, near_conv_df, key_features, id_col):
    """Render the Segment Intelligence tab as self-contained static HTML."""
    P = []
    P.append('<div class="tab-content" id="tab-intel">')
    P.append('<div style="max-width:1200px;margin:0 auto;padding:18px 8px;'
             'font-family:Inter,Segoe UI,sans-serif;color:#1E293B">')
    P.append('<h2 style="color:#003B71">Segment Intelligence</h2>')
    P.append('<p style="color:#535554;font-size:14px">Centroid analysis, prototype '
             'doctors, business levers, churn risk and conversion opportunities — '
             'all computed from the Business-Calibrated Ordinal predictions.</p>')

    # ----- Executive insights -----
    P.append('<h3 style="color:#003B71;margin-top:22px">Executive Insights</h3>')
    P.append('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px">')
    for ins in insights:
        c = _SEV_COLOR.get(ins.get("severity", "info"), "#0070BF")
        P.append(
            f'<div style="border-left:5px solid {c};background:#fff;border-radius:8px;'
            f'padding:12px 14px;box-shadow:0 1px 4px rgba(0,0,0,.08)">'
            f'<div style="font-weight:700;color:{c};margin-bottom:4px">{ins.get("icon","")} {_esc(ins.get("title",""))}</div>'
            f'<div style="font-size:13px;line-height:1.5">{ins.get("body","")}</div></div>'
        )
    P.append('</div>')

    def _table(title, df, cols, max_rows=15, headers=None):
        """Render a DataFrame as a styled static HTML table."""
        P.append(f'<h3 style="color:#003B71;margin-top:26px">{title}</h3>')
        if df is None or len(df) == 0:
            P.append('<p style="color:#535554;font-size:13px">No records.</p>')
            return
        headers = headers or cols
        P.append('<div style="overflow-x:auto"><table style="border-collapse:collapse;'
                 'width:100%;font-size:12.5px;background:#fff;border-radius:8px;overflow:hidden;'
                 'box-shadow:0 1px 4px rgba(0,0,0,.06)">')
        P.append('<thead><tr style="background:#003B71;color:#fff">'
                 + "".join(f'<th style="padding:8px 10px;text-align:left">{_esc(h)}</th>' for h in headers)
                 + '</tr></thead><tbody>')
        for ri, (_, row) in enumerate(df.head(max_rows).iterrows()):
            bg = "#F4F7FB" if ri % 2 else "#fff"
            P.append(f'<tr style="background:{bg}">' + "".join(
                f'<td style="padding:7px 10px;border-bottom:1px solid #eef2f7">{_esc(row[c]) if c in row else ""}</td>'
                for c in cols) + '</tr>')
        P.append('</tbody></table></div>')

    # ----- Segment centroids (key features only) -----
    P.append('<h3 style="color:#003B71;margin-top:26px">Segment Centroids '
             '(median behavioral profile)</h3>')
    key_cols = [k for k, *_ in key_features if k in centroids_df.columns]
    if len(centroids_df) and key_cols:
        P.append('<div style="overflow-x:auto"><table style="border-collapse:collapse;'
                 'width:100%;font-size:12.5px;background:#fff;border-radius:8px;overflow:hidden">')
        P.append('<thead><tr style="background:#003B71;color:#fff">'
                 '<th style="padding:8px 10px;text-align:left">Feature</th>'
                 + "".join(f'<th style="padding:8px 10px">{seg}</th>' for seg in centroids_df.index)
                 + '</tr></thead><tbody>')
        for ci, col in enumerate(key_cols):
            bg = "#F4F7FB" if ci % 2 else "#fff"
            cells = "".join(f'<td style="padding:7px 10px;text-align:center;border-bottom:1px solid #eef2f7">'
                            f'{centroids_df.loc[seg, col]:.2f}</td>' for seg in centroids_df.index)
            P.append(f'<tr style="background:{bg}"><td style="padding:7px 10px;border-bottom:1px solid #eef2f7">{col}</td>{cells}</tr>')
        P.append('</tbody></table></div>')

    # ----- Business levers -----
    if levers_df is not None and len(levers_df):
        _table("Business Levers — SEG_B &rarr; SEG_C (biggest gaps)", levers_df,
               ["feature", "SEG_B_median", "SEG_C_median", "absolute_gap", "interpretation"],
               max_rows=12,
               headers=["Feature", "SEG_B median", "SEG_C median", "Gap (C−B)", "Interpretation"])

    # ----- Prototype doctors -----
    if protos_df is not None and len(protos_df):
        proto_cols = ["segment", id_col, "ord_P_A", "ord_P_B", "ord_P_C", "distance_to_centroid"]
        _table("Prototype Doctors (most representative HCP per segment)", protos_df,
               [c for c in proto_cols if c in protos_df.columns], max_rows=5)

    # ----- Churn risk -----
    _table(f"SEG_C Churn Risk — {0 if at_risk_df is None else len(at_risk_df):,} flagged (top by risk)",
           at_risk_df,
           [c for c in [id_col, "ord_P_C", "margin_C_over_B", "risk_score", "recommended_action"]
            if at_risk_df is not None and c in at_risk_df.columns],
           max_rows=15)

    # ----- Near-conversion -----
    _table(f"Near-Conversion SEG_B — {0 if near_conv_df is None else len(near_conv_df):,} candidates (top by score)",
           near_conv_df,
           [c for c in [id_col, "ord_P_C", "distance_to_C_centroid", "conversion_score", "recommended_action"]
            if near_conv_df is not None and c in near_conv_df.columns],
           max_rows=15)

    P.append('</div></div>')
    return "\n".join(P)



def main():
    """Command-line entry point: run the full pipeline and write outputs."""
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--label-col", default="ATSEG_first")
    p.add_argument("--id-col", default="NUEVO_ID")
    p.add_argument("--output-html", default="hcp_dashboard_final_v3.html")
    p.add_argument("--output-csv", default="hcp_predictions_final_v3.csv")
    p.add_argument("--max-html-points", type=int, default=4000)
    p.add_argument("--n-folds", type=int, default=5)
    args = p.parse_args()

    print("=" * 65)
    print("  HCP SEGMENTATION — FINAL CAPSTONE MODEL (v3)")
    print("  Ordinal XGBoost + Dominance + Conversion Strategy")
    print("=" * 65)

    df = pd.read_csv(args.input)
    print(f"\n{args.input}: {df.shape[0]:,} rows x {df.shape[1]} cols")
    df[args.label_col] = df[args.label_col].astype(str)
    is_labeled = df[args.label_col].isin(VALID_LABELS).values
    n_lab = int(is_labeled.sum()); n_unl = int((~is_labeled).sum())
    print(f"   Labeled: {n_lab:,}  |  Unlabeled: {n_unl:,}")
    dist = {}
    for lab in VALID_LABELS:
        n = (df[args.label_col] == lab).sum()
        dist[lab] = n / n_lab
        print(f"     {lab}: {n:>6,} ({dist[lab]*100:.1f}%)")

    raw_fv = {}
    for fk, _, _, _ in KEY_FEATURES:
        if fk in df.columns: raw_fv[fk] = df[fk].values.copy()

    print("\nFeature engineering...")
    df = add_features(df)
    exclude = [args.label_col, args.id_col, "ATSEG", "SEGMENT", "PROB_SEG",
               "CONFIDENCE", "CLUSTER", "WEEK_ID_first", "WEEK_ID_last", "WEEK_ID_count"]
    feat_cols = select_features(df, exclude)
    print(f"   Final features: {len(feat_cols)}")

    X_all = df[feat_cols].fillna(0).values
    y_all = df[args.label_col].values
    ids_all = df[args.id_col].values
    X_lab = X_all[is_labeled]; y_lab = y_all[is_labeled]

    print(f"\n5-fold OOF + cross-fold CI...")
    oof, ci_lo, ci_hi, fold_models = oof_with_ci(X_lab, y_lab, n_splits=args.n_folds)

    metrics = {}
    for key, name in [("ord","Ordinal + dominance (recommended)"), ("amx","Argmax (baseline)")]:
        m = metrics_for(y_lab, oof[key]["pred"], name)
        metrics[key] = m
        print(f"\n  {name}:")
        print(f"    Accuracy={m['accuracy']:.4f}  Bal.Acc={m['balanced_accuracy']:.4f}  "
              f"Recall_C={m['recall_C']:.4f}  C->A={m['c_lost_pct']:.4f}")

    print(f"\nTraining final ordinal model on all {n_lab:,} labeled...")
    final_m1, final_m2 = fit_ordinal(X_lab, y_lab)
    final_amx, final_le = fit_argmax(X_lab, y_lab)

    PA, PB, PC, pred_ord = predict_ordinal(final_m1, final_m2, X_all)
    PA[is_labeled] = oof["ord"]["P_A"]
    PB[is_labeled] = oof["ord"]["P_B"]
    PC[is_labeled] = oof["ord"]["P_C"]
    pred_ord[is_labeled] = oof["ord"]["pred"]

    amx_PA, amx_PB, amx_PC, pred_amx = predict_argmax(final_amx, final_le, X_all)
    pred_amx[is_labeled] = oof["amx"]["pred"]

    print("\nComputing SHAP (TreeSHAP)...")
    shap_vals = get_shap_values(final_m2, X_all)
    top_shap = top_shap_per_doctor(shap_vals)

    ci_lo_full = np.zeros((len(df), 3)); ci_hi_full = np.zeros((len(df), 3))
    ci_lo_full[is_labeled] = ci_lo; ci_hi_full[is_labeled] = ci_hi
    if n_unl > 0:
        print(f"\nComputing CI for {n_unl:,} unlabeled...")
        X_unl = X_all[~is_labeled]
        cf_unl = np.zeros((n_unl, args.n_folds, 3))
        for fi, (fm1, fm2) in enumerate(fold_models):
            pa_f, pb_f, pc_f, _ = predict_ordinal(fm1, fm2, X_unl)
            cf_unl[:, fi, :] = np.column_stack([pa_f, pb_f, pc_f])
        um = cf_unl.mean(axis=1); us = cf_unl.std(axis=1)
        ci_lo_full[~is_labeled] = np.clip(um - 1.96*us, 0, 1)
        ci_hi_full[~is_labeled] = np.clip(um + 1.96*us, 0, 1)

    print("\nComputing B->C conversion strategy...")
    conv_candidates, conv_aggregate, c_medians = compute_conversion_strategy(
        ids_all, pred_ord, PA, PB, PC, is_labeled, y_all, raw_fv, KEY_FEATURES
    )
    print(f"   Identified {len(conv_candidates):,} conversion candidates")
    print(f"   Top features below C median (across candidates):")
    for fm in conv_aggregate["features"][:3]:
        tag = "[ACTIONABLE]" if fm["actionable"] else "[signal]"
        print(f"     {tag:14s} {fm['name']:24s} {fm['pct_below_C_median']:>5.1f}% below median")

    avail_feats = [(k,n_,d_) for k,n_,d_,_ in KEY_FEATURES if k in raw_fv]
    df_lab_keys = pd.DataFrame({k: raw_fv[k][is_labeled] for k,_,_ in avail_feats})
    df_lab_keys[args.label_col] = y_lab
    seg_stats = compute_segment_stats(df_lab_keys, args.label_col, [k for k,_,_ in avail_feats])

    print("\nWriting outputs...")
    out_df = pd.DataFrame({
        args.id_col: ids_all, "true_segment": y_all, "was_labeled": is_labeled,
        "predicted_segment": pred_ord,
        "P_A": np.round(PA, 4), "P_B": np.round(PB, 4), "P_C": np.round(PC, 4),
        "CI_A_lo": np.round(ci_lo_full[:,0],4), "CI_A_hi": np.round(ci_hi_full[:,0],4),
        "CI_B_lo": np.round(ci_lo_full[:,1],4), "CI_B_hi": np.round(ci_hi_full[:,1],4),
        "CI_C_lo": np.round(ci_lo_full[:,2],4), "CI_C_hi": np.round(ci_hi_full[:,2],4),
        "confidence": np.round(np.maximum.reduce([PA, PB, PC]), 4),
        "argmax_pred": pred_amx,
        "dominance_rule_applied": np.where(
            (PA < THR_A) & (PC < THR_C) & (PC > PB), True, False
        ),
    })
    out_df.to_csv(args.output_csv, index=False)
    print(f"   {args.output_csv} ({len(out_df):,} rows)")

    print("   Building HTML dashboard...")
    rng = np.random.default_rng(0)
    max_pts = args.max_html_points
    if len(df) > max_pts:
        ki = []
        nt = max_pts // 2
        for cls in VALID_LABELS:
            ci_ = np.where(y_all == cls)[0]
            nk = int(nt * (len(ci_) / max(n_lab,1)))
            ki.extend(rng.choice(ci_, size=min(nk, len(ci_)), replace=False))
        ui = np.where(~is_labeled)[0]
        if len(ui): ki.extend(rng.choice(ui, size=min(max_pts-len(ki), len(ui)), replace=False))
        ki = np.array(ki)
    else:
        ki = np.arange(len(df))

    uc = raw_fv.get("UC_TRX_sum", np.zeros(len(df)))
    oral = raw_fv.get("ORAL_TRX_sum", np.zeros(len(df)))

    viz = {
        "hcp_id": [str(x) for x in ids_all[ki]],
        "true_seg": [str(x) for x in y_all[ki]],
        "is_labeled": [bool(x) for x in is_labeled[ki]],
        "log_uc": [round(float(np.log1p(max(x,0))),3) for x in uc[ki]],
        "log_oral": [round(float(np.log1p(max(x,0))),3) for x in oral[ki]],
        "P_A": [round(float(x),3) for x in PA[ki]],
        "P_B": [round(float(x),3) for x in PB[ki]],
        "P_C": [round(float(x),3) for x in PC[ki]],
        "pred": [str(x) for x in pred_ord[ki]],
    }

    feat_arrays = [raw_fv[k] for k,_,_ in avail_feats]
    all_lookup = {}
    for i, hid in enumerate(ids_all):
        all_lookup[str(hid)] = {
            "t": str(y_all[i]) if is_labeled[i] else "",
            "il": bool(is_labeled[i]),
            "p": str(pred_ord[i]),
            "pa": round(float(PA[i]),3), "pb": round(float(PB[i]),3), "pc": round(float(PC[i]),3),
            "f": [round(float(a[i]),1) for a in feat_arrays],
            "shap": top_shap[i],
            "ci": {"lo": [round(float(ci_lo_full[i,j]),3) for j in range(3)],
                   "hi": [round(float(ci_hi_full[i,j]),3) for j in range(3)],
                   "mean": [round(float(PA[i]),3), round(float(PB[i]),3), round(float(PC[i]),3)]},
        }

    payload = {
        "viz": viz, "all": all_lookup,
        "features": [{"key":k,"name":n_,"desc":d_} for k,n_,d_ in avail_feats],
        "segment_stats": seg_stats, "feat_names": feat_cols,
        "conversion": {"candidates": conv_candidates, "aggregate": conv_aggregate},
        "metrics": {"n_labeled": n_lab, "n_unlabeled": n_unl,
                    "dist_A": float(dist["SEG_A"]), "dist_B": float(dist["SEG_B"]),
                    "dist_C": float(dist["SEG_C"]),
                    "ord": metrics["ord"], "amx": metrics["amx"]},
    }

    # ── Segment Intelligence (from centroids_v2), computed on ordinal preds ──
    print("\nComputing segment intelligence (centroids, levers, churn)...")
    full_ord = {"pred": pred_ord, "P_A": PA, "P_B": PB, "P_C": PC}
    df_pred = df.copy()
    df_pred["__pred"] = pred_ord
    key_cols_raw = [k for k, *_ in KEY_FEATURES]
    centroids_df, centroids_scaled, X_scaled, _ = compute_segment_centroids(
        df_pred, "__pred", feat_cols, args.id_col)
    protos_df = find_prototype_doctors(
        df_pred, "__pred", args.id_col, X_scaled, centroids_scaled,
        full_ord, feat_cols, key_cols_raw)
    levers_df = compute_business_levers(centroids_df, feat_cols)
    at_risk_df, near_conv_df = compute_churn_risks(
        df_pred, "__pred", args.id_col, X_scaled, centroids_scaled, full_ord)
    insights = generate_executive_insights(
        centroids_df, levers_df, at_risk_df, near_conv_df, protos_df, n_lab, dist)
    intel_html = build_intel_tab_html(
        insights, centroids_df, protos_df, levers_df,
        at_risk_df, near_conv_df, KEY_FEATURES, args.id_col)

    # ── Inject the Segment Intelligence tab into the v3 template ──
    template = HTML_TEMPLATE.replace(
        '<button class="tab" data-tab="conversion">4 \u00b7 Conversion Strategy (B \u2192 C)</button>',
        '<button class="tab" data-tab="conversion">4 \u00b7 Conversion Strategy (B \u2192 C)</button>\n'
        '<button class="tab" data-tab="intel">5 \u00b7 Segment Intelligence</button>')
    template = template.replace("<footer>Capstone Project", intel_html + "\n<footer>Capstone Project")

    html = template.replace("__DATA_PLACEHOLDER__", json.dumps(payload))
    Path(args.output_html).write_text(html, encoding="utf-8")
    sz = Path(args.output_html).stat().st_size / 1024 / 1024
    print(f"   {args.output_html} ({sz:.1f} MB)")

    m = metrics["ord"]
    print(f"\n{'='*65}")
    print(f"  FINAL MODEL SUMMARY (v3)")
    print(f"     Model:            Ordinal + Dominance + Conversion Strategy")
    print(f"     Decision:         P(A)>={THR_A}->A | P(C)>={THR_C}->C | P(C)>P(B)->C | else B")
    print(f"     Balanced Acc:     {m['balanced_accuracy']:.4f}")
    print(f"     Recall SEG_C:     {m['recall_C']:.4f}")
    print(f"     C->A losses:      {m['c_lost_pct']:.4f}")
    print(f"     Conversion cands: {len(conv_candidates):,}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()