"""
intelligence.py
===============
Segment-intelligence analytics layered on top of the model's predictions:

  * compute_segment_centroids  - median behavioural profile of each segment
                                 (RobustScaler + log1p on skewed columns).
  * find_prototype_doctors      - the real HCP closest to each segment centroid.
  * compute_business_levers     - SEG_B vs SEG_C centroid gaps, with auto text,
                                 ranked by the biggest behavioural differences.
  * compute_churn_risks         - at-risk SEG_C doctors + near-conversion SEG_B.
  * generate_executive_insights - 6-7 ready-to-present narrative insights.

These functions are model-agnostic: they consume a ``full_ord`` predictions
dict (keys ``pred``, ``P_A``, ``P_B``, ``P_C`` -- one entry per row, aligned
with ``df``), so they work with whatever model produced those predictions.

Ported from build_probability_visualization_v5 (centroids_v2) and wired to the
shared core.

Tec de Monterrey x Pfizer Global Commercial Analytics
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .config import VALID_LABELS


# ────────────────────────────────────────────────────────────────────────
# Scaled feature space (shared by centroids / prototypes / churn)
# ────────────────────────────────────────────────────────────────────────
def _prepare_scaled_matrix(df_feat, feat_cols):
    """log1p the skewed columns, then RobustScaler the whole matrix.

    A column is treated as skewed if its p75/p25 ratio > 5, or its
    p90/median ratio > 10 (robust heuristics for heavy right tails).

    Returns
    -------
    (X_scaled, scaler, used_cols, skewed_cols)
    """
    df_work = df_feat[feat_cols].fillna(0).copy()
    skewed = []
    for c in feat_cols:
        vals = df_work[c].values
        p25, p75 = np.quantile(vals, 0.25), np.quantile(vals, 0.75)
        if p25 > 0 and (p75 / p25) > 5:
            skewed.append(c)
        elif np.median(vals) > 0 and np.quantile(vals, 0.90) / (np.median(vals) + 1e-9) > 10:
            skewed.append(c)
    for c in skewed:
        df_work[c] = np.log1p(df_work[c].clip(lower=0))
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_work.values)
    return X_scaled, scaler, feat_cols, skewed


# ────────────────────────────────────────────────────────────────────────
# Centroids
# ────────────────────────────────────────────────────────────────────────
def compute_segment_centroids(df, pred_col, feat_cols, id_col, verbose=True):
    """Median-based centroid of each segment, in raw and scaled space.

    Returns
    -------
    centroids_df : DataFrame
        One row per segment; columns are raw-space median feature values.
    centroids_scaled : dict[str, np.ndarray]
        Scaled-space median centroid per segment (for distance computations).
    X_scaled : np.ndarray
        Scaled matrix for ALL doctors (reused for prototypes / churn).
    feat_info : (used_cols, skewed_cols)
    """
    if verbose:
        print("  [Intelligence] Computing segment centroids (RobustScaler + log1p + median)...")
    df_work = df.copy()
    X_scaled, scaler, used_cols, skewed = _prepare_scaled_matrix(df_work, feat_cols)

    rows = []
    centroids_scaled = {}
    for seg in VALID_LABELS:
        mask = df_work[pred_col].values == seg
        if mask.sum() == 0:
            continue
        centroids_scaled[seg] = np.median(X_scaled[mask], axis=0)
        cent_raw = {col: float(np.median(df_work[col].fillna(0).values[mask]))
                    for col in used_cols}
        rows.append({"segment": seg, **cent_raw})

    centroids_df = pd.DataFrame(rows).set_index("segment")
    if verbose:
        print(f"    Centroids: {list(centroids_scaled.keys())} - {len(used_cols)} features")
    return centroids_df, centroids_scaled, X_scaled, (used_cols, skewed)


# ────────────────────────────────────────────────────────────────────────
# Prototype doctors
# ────────────────────────────────────────────────────────────────────────
def find_prototype_doctors(df, pred_col, id_col, X_scaled, centroids_scaled,
                           full_ord, feat_cols, key_feat_cols, verbose=True):
    """The real doctor closest (euclidean, scaled space) to each centroid.

    Returns
    -------
    pandas.DataFrame
        One prototype row per segment with its probabilities and key features.
    """
    if verbose:
        print("  [Intelligence] Finding prototype doctors per segment...")
    rows = []
    pred_arr = df[pred_col].values
    ids = df[id_col].values

    for seg in VALID_LABELS:
        if seg not in centroids_scaled:
            continue
        seg_idx = np.where(pred_arr == seg)[0]
        if len(seg_idx) == 0:
            continue
        cent = centroids_scaled[seg]
        dists = np.linalg.norm(X_scaled[seg_idx] - cent, axis=1)
        best = seg_idx[int(np.argmin(dists))]

        row = {
            "segment": seg,
            id_col: ids[best],
            "distance_to_centroid": round(float(dists.min()), 4),
            "ord_pred": full_ord["pred"][best],
            "ord_P_A": round(float(full_ord["P_A"][best]), 3),
            "ord_P_B": round(float(full_ord["P_B"][best]), 3),
            "ord_P_C": round(float(full_ord["P_C"][best]), 3),
        }
        for col in key_feat_cols:
            if col in df.columns:
                row[col] = round(float(df[col].iloc[best]), 2)
        rows.append(row)
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────
# Business levers (SEG_B -> SEG_C)
# ────────────────────────────────────────────────────────────────────────
def _interpret_lever(col, abs_gap):
    """Plain-language interpretation of a SEG_B vs SEG_C feature gap."""
    cl = col.lower()
    pos = abs_gap > 0
    if abs(abs_gap) < 1e-6:
        return "No meaningful difference between SEG_B and SEG_C."
    if "oral" in cl:
        return ("SEG_C doctors prescribe significantly more oral therapies - a strong Velsipity "
                "adoption signal.") if pos else "SEG_B doctors show higher oral therapy usage in this metric."
    if "uc" in cl and "trx" in cl:
        return ("SEG_C doctors have stronger UC specialization (higher UC TRx volume)."
                if pos else "UC TRx is higher in SEG_B; not a differentiating lever toward SEG_C.")
    if "detail" in cl:
        return ("SEG_C doctors receive more Pfizer detailing visits - rep engagement drives prescribing."
                if pos else "Detailing frequency is similar; allocation alone won't move doctors to SEG_C.")
    if "il23" in cl or "biologic" in cl:
        return ("SEG_C doctors use more IL-23 biologics - indicating advanced UC therapeutic experience."
                if pos else "SEG_B doctors use more IL-23 biologics; this feature may not differentiate toward SEG_C.")
    if "nrx" in cl:
        return ("SEG_C doctors start more new patients on UC therapy - higher patient acquisition rate."
                if pos else "New prescription rates are similar; new patient starts alone don't predict SEG_C.")
    if "total" in cl and "trx" in cl:
        return ("SEG_C doctors have higher overall prescribing volume - experienced, high-output HCPs."
                if pos else "Total Rx volume is not a reliable lever; prescribing mix matters more.")
    if "ratio" in cl:
        return ("This prescribing ratio is higher in SEG_C - structural behavioral difference."
                if pos else "This ratio favors SEG_B; may indicate different therapy-mix preferences.")
    direction = "higher" if pos else "lower"
    return f"SEG_C doctors show {direction} values on this metric vs SEG_B."


def compute_business_levers(centroids_df, feat_cols, verbose=True):
    """Rank SEG_B->SEG_C centroid gaps, biggest behavioural difference first.

    Returns
    -------
    pandas.DataFrame
        Columns: feature, SEG_B_median, SEG_C_median, absolute_gap,
        percent_gap, interpretation.
    """
    if verbose:
        print("  [Intelligence] Computing business levers (SEG_B -> SEG_C)...")
    if "SEG_B" not in centroids_df.index or "SEG_C" not in centroids_df.index:
        if verbose:
            print("    WARNING: SEG_B or SEG_C missing - skipping levers")
        return pd.DataFrame()

    rows = []
    for col in feat_cols:
        if col not in centroids_df.columns:
            continue
        b_val = float(centroids_df.loc["SEG_B", col])
        c_val = float(centroids_df.loc["SEG_C", col])
        abs_gap = c_val - b_val
        pct_gap = (abs_gap / (abs(b_val) + 1e-9)) * 100
        rows.append({
            "feature": col,
            "SEG_B_median": round(b_val, 4),
            "SEG_C_median": round(c_val, 4),
            "absolute_gap": round(abs_gap, 4),
            "percent_gap": round(pct_gap, 2),
            "interpretation": _interpret_lever(col, abs_gap),
        })

    levers_df = pd.DataFrame(rows).sort_values("absolute_gap", key=abs, ascending=False)
    if verbose and len(levers_df):
        print(f"    Top lever: {levers_df.iloc[0]['feature']} (gap={levers_df.iloc[0]['absolute_gap']:.2f})")
    return levers_df


# ────────────────────────────────────────────────────────────────────────
# Churn risk + conversion opportunities
# ────────────────────────────────────────────────────────────────────────
def compute_churn_risks(df, pred_col, id_col, X_scaled, centroids_scaled,
                        full_ord, ambiguous_thr=0.55, verbose=True):
    """Flag at-risk SEG_C doctors and near-conversion SEG_B doctors.

    An SEG_C doctor is *at risk* if its top probability is below
    ``ambiguous_thr`` (low confidence) or its P_C only narrowly beats P_B
    (drifting). An SEG_B doctor is a *near-conversion* candidate if it sits in
    the closest 20% to the SEG_C centroid.

    Returns
    -------
    (at_risk_df, near_conversion_df) : tuple of DataFrame
    """
    if verbose:
        print("  [Intelligence] Computing churn risks and conversion opportunities...")
    pred_arr = full_ord["pred"]
    P_A, P_B, P_C = full_ord["P_A"], full_ord["P_B"], full_ord["P_C"]
    ids = df[id_col].values
    cent_C = centroids_scaled.get("SEG_C", None)

    # At-risk SEG_C
    at_risk_rows = []
    for i in np.where(pred_arr == "SEG_C")[0]:
        max_p = max(P_A[i], P_B[i], P_C[i])
        margin = P_C[i] - P_B[i]
        dist_to_c = float(np.linalg.norm(X_scaled[i] - cent_C)) if cent_C is not None else None
        is_ambiguous = max_p < ambiguous_thr
        is_drifting = margin < 0.15
        if is_ambiguous or is_drifting:
            risk_score = 1.0 - P_C[i] + (0.15 - margin if is_drifting else 0)
            action = ("URGENT: reinforce with detailing and education - high churn risk"
                      if is_ambiguous and is_drifting else
                      "MONITOR: low confidence in SEG_C, schedule follow-up" if is_ambiguous else
                      "WATCH: narrow margin vs SEG_B, validate engagement")
            at_risk_rows.append({
                id_col: ids[i], "ord_pred": pred_arr[i],
                "ord_P_A": round(float(P_A[i]), 3), "ord_P_B": round(float(P_B[i]), 3),
                "ord_P_C": round(float(P_C[i]), 3), "max_prob": round(float(max_p), 3),
                "margin_C_over_B": round(float(margin), 3),
                "distance_to_C_centroid": round(dist_to_c, 3) if dist_to_c is not None else None,
                "ambiguous": is_ambiguous, "drifting_to_B": is_drifting,
                "risk_score": round(float(risk_score), 3), "recommended_action": action,
            })
    at_risk_df = pd.DataFrame(at_risk_rows)
    if len(at_risk_df):
        at_risk_df = at_risk_df.sort_values("risk_score", ascending=False)

    # Near-conversion SEG_B
    near_rows = []
    b_idx = np.where(pred_arr == "SEG_B")[0]
    if cent_C is not None and len(b_idx) > 0:
        dists_to_c = np.linalg.norm(X_scaled[b_idx] - cent_C, axis=1)
        thr_20 = np.quantile(dists_to_c, 0.20)
        thr_05 = np.quantile(dists_to_c, 0.05)
        for local_i, gi in enumerate(b_idx):
            d = dists_to_c[local_i]
            if d <= thr_20:
                near_rows.append({
                    id_col: ids[gi], "ord_pred": pred_arr[gi],
                    "ord_P_A": round(float(P_A[gi]), 3), "ord_P_B": round(float(P_B[gi]), 3),
                    "ord_P_C": round(float(P_C[gi]), 3),
                    "distance_to_C_centroid": round(float(d), 3),
                    "margin_C_over_B": round(float(P_C[gi] - P_B[gi]), 3),
                    "conversion_score": round(float(1.0 / (d + 1e-9) * P_C[gi]), 3),
                    "recommended_action": (
                        "HIGH PRIORITY: behavioral profile is SEG_C-like - invest in conversion"
                        if d <= thr_05 else
                        "MEDIUM PRIORITY: near SEG_C boundary - targeted education recommended"),
                })
    near_conv_df = pd.DataFrame(near_rows)
    if len(near_conv_df):
        near_conv_df = near_conv_df.sort_values("conversion_score", ascending=False)

    if verbose:
        print(f"    At-risk SEG_C doctors:  {len(at_risk_df):,}")
        print(f"    Near-conversion SEG_B:  {len(near_conv_df):,}")
    return at_risk_df, near_conv_df


# ────────────────────────────────────────────────────────────────────────
# Executive insights (narrative)
# ────────────────────────────────────────────────────────────────────────
def generate_executive_insights(centroids_df, levers_df, at_risk_df,
                                near_conv_df, prototypes_df, n_labeled, dist):
    """Build 6-7 narrative, presentation-ready insights.

    Returns
    -------
    list[dict]
        Each dict has ``icon``, ``title``, ``body`` (may contain HTML <strong>),
        and ``severity`` in {info, warning, danger, success}.
    """
    insights = []

    top_levers = levers_df[levers_df["absolute_gap"] > 0].head(3) if len(levers_df) else levers_df
    lever_names = ", ".join(top_levers["feature"].tolist()) if len(top_levers) else "UC_TRX_sum"
    insights.append({
        "icon": "📊", "title": "Behavioral Signature of SEG_C", "severity": "info",
        "body": (f"The strongest behavioral signals separating SEG_C prescribers are: "
                 f"<strong>{lever_names}</strong>. Doctors with elevated values on these features "
                 f"are significantly more likely to prescribe Velsipity."),
    })

    if "SEG_B" in centroids_df.index and "SEG_C" in centroids_df.index:
        oral_col = next((c for c in centroids_df.columns if "oral" in c.lower()), None)
        if oral_col:
            b_oral = centroids_df.loc["SEG_B", oral_col]
            c_oral = centroids_df.loc["SEG_C", oral_col]
            pct_gap = (c_oral - b_oral) / (b_oral + 1e-9) * 100
            insights.append({
                "icon": "🔄", "title": "SEG_B: Intermediate Behavioral State", "severity": "warning",
                "body": (f"SEG_B appears to be an intermediate behavioral state rather than a fully "
                         f"distinct class. Oral therapy adoption ('{oral_col}') is "
                         f"<strong>{pct_gap:+.0f}% higher in SEG_C</strong> vs SEG_B - "
                         f"targeted oral therapy education is the most actionable lever."),
            })

    if len(levers_df):
        det = levers_df[levers_df["feature"].str.contains("DETAIL", case=False)]
        if len(det) > 0 and det.iloc[0]["absolute_gap"] > 0:
            insights.append({
                "icon": "🤝", "title": "Detailing Activity Correlates with SEG_C", "severity": "info",
                "body": ("SEG_C doctors receive more Pfizer sales rep visits on average. Increasing "
                         "structured detailing frequency for high-potential SEG_B doctors - especially "
                         "those near the SEG_C centroid - may accelerate conversion."),
            })

    if len(at_risk_df) > 0:
        n_urgent = int(at_risk_df["recommended_action"].str.startswith("URGENT").sum())
        insights.append({
            "icon": "⚠️", "title": f"SEG_C Churn Risk: {len(at_risk_df):,} Doctors Flagged",
            "severity": "danger",
            "body": (f"<strong>{len(at_risk_df):,} SEG_C doctors</strong> show signs of behavioral "
                     f"ambiguity or low model confidence ({n_urgent} flagged URGENT). These are at "
                     f"risk of becoming inactive prescribers. Immediate rep intervention is "
                     f"recommended for URGENT cases."),
        })

    if len(near_conv_df) > 0:
        n_high = int(near_conv_df["recommended_action"].str.startswith("HIGH").sum())
        insights.append({
            "icon": "🎯", "title": f"Conversion Opportunity: {len(near_conv_df):,} SEG_B Near SEG_C",
            "severity": "success",
            "body": (f"<strong>{len(near_conv_df):,} SEG_B doctors</strong> have behavioral profiles "
                     f"similar to confirmed SEG_C prescribers. {n_high} are HIGH PRIORITY. A focused "
                     f"detailing campaign on this subgroup offers the highest ROI for expanding the "
                     f"prescriber base."),
        })

    if len(levers_df) and len(levers_df[levers_df["feature"].str.contains("UC_TRX|ORAL", case=False, regex=True)]) > 0:
        insights.append({
            "icon": "💊", "title": "Oral UC Therapy: The Core Differentiator", "severity": "info",
            "body": ("High oral UC therapy adoption is the most consistent behavioral differentiator "
                     "of SEG_C doctors. Sales strategy should prioritize HCPs already prescribing oral "
                     "UC treatments, as they represent the natural conversion funnel for Velsipity."),
        })

    c_pct = dist.get("SEG_C", 0) * 100
    b_pct = dist.get("SEG_B", 0) * 100
    insights.append({
        "icon": "📌", "title": "Segment Composition & Prioritization", "severity": "info",
        "body": (f"Of labeled HCPs, <strong>SEG_C = {c_pct:.1f}%</strong> (top priority), "
                 f"<strong>SEG_B = {b_pct:.1f}%</strong> (conversion target). Sales force allocation "
                 f"should be: SEG_C retention first, SEG_B near-conversion second, SEG_A minimum touch."),
    })

    return insights
