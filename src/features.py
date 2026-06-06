"""
features.py
===========
Data loading and feature engineering for the HCP segmentation pipeline.

This module is the single definition of how raw aggregated data becomes the
numeric feature matrix the models consume. It exposes TWO feature-selection
paths on purpose, because the two families of deliverables historically used
different ones and we must not silently change results that were already
reported:

  * select_features()        -> pruned set (drops quasi-constant and highly
                                correlated columns). Used by the HTML reports.
  * basic_feature_columns()  -> all numeric columns minus IDs/labels/WEEK_ID.
                                Used by the Streamlit dashboard.

Tec de Monterrey x Pfizer Global Commercial Analytics
"""

import numpy as np
import pandas as pd

from .config import VALID_LABELS, LABEL_COL, ID_COL, DATA_EXCLUDE


# ────────────────────────────────────────────────────────────────────────
# Loading
# ────────────────────────────────────────────────────────────────────────
def load_data(path, label_col=LABEL_COL, clean_zero_label=True):
    """Read the aggregated HCP CSV and normalise the label column.

    Parameters
    ----------
    path : str
        Path to ``doctors_aggregated.csv`` (one row per HCP).
    label_col : str
        Name of the segment column.
    clean_zero_label : bool
        If True, the placeholder ``"0"`` used for untyped HCPs is converted
        to NaN so that ``df[label_col].isin(VALID_LABELS)`` cleanly separates
        labeled from unlabeled doctors.

    Returns
    -------
    pandas.DataFrame
    """
    df = pd.read_csv(path)
    df[label_col] = df[label_col].astype(str)
    if clean_zero_label:
        df[label_col] = df[label_col].replace("0", np.nan)
    return df


def split_labeled(df, label_col=LABEL_COL):
    """Return a boolean mask marking the rows whose label is a valid segment."""
    return df[label_col].isin(VALID_LABELS).values


# ────────────────────────────────────────────────────────────────────────
# Feature engineering
# ────────────────────────────────────────────────────────────────────────
def add_features(df, verbose=True):
    """Add engineered ratio, diversity, engagement and log features.

    The engineered features capture *relative* prescribing behaviour (ratios),
    breadth of activity (diversity / concentration), total engagement effort,
    and compressed magnitudes (log1p) for the high-variance volume columns.
    Every operation is guarded so missing source columns are skipped silently.

    Parameters
    ----------
    df : pandas.DataFrame
        Aggregated, one-row-per-HCP frame.
    verbose : bool
        Print how many derived features were added.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with the new columns appended.
    """
    out = df.copy()
    eps = 1e-6
    n = 0

    # Pairwise ratios (behaviour relative to a denominator).
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
            n += 1

    # Brand diversity: how many distinct brands the HCP prescribes.
    bc = [c for c in out.columns if c.startswith("BRAND") and "_TRX_sum" in c]
    if bc:
        out["brand_diversity"] = (out[bc] > 0).sum(axis=1)
        n += 1

    # Brand concentration: share of the single most-prescribed brand.
    bn = [c for c in out.columns if c.startswith("BRAND") and "_NRX_sum" in c]
    if bn:
        t = out[bn].sum(axis=1)
        out["brand_concentration"] = out[bn].max(axis=1) / (t + eps)
        n += 1

    # Total engagement: sum of every promotional lever.
    ec = [c for c in out.columns if any(e in c for e in
          ["DETAILS_sum", "SAMPLES_sum", "RTE_sum", "SPK_sum",
           "COPAY_sum", "DIRECTMAIL_sum"])]
    if ec:
        out["total_engagement"] = out[ec].sum(axis=1)
        n += 1

    # Claims breadth and volume.
    clm = [c for c in out.columns if c.startswith("N_CLM") and "_sum" in c]
    if clm:
        out["claims_diversity"] = (out[clm] > 0).sum(axis=1)
        out["total_claims"] = out[clm].sum(axis=1)
        n += 2

    # Log-compressed magnitudes for the heavy-tailed volume columns.
    for col in ["UC_TRX_sum", "ORAL_TRX_sum", "IL23_TRX_sum",
                "TOTAL_TRX_sum", "UC_NRX_sum", "DETAILS_sum"]:
        if col in out.columns:
            out[f"log_{col}"] = np.log1p(np.maximum(out[col], 0))
            n += 1

    if verbose:
        print(f"  {n} derived features added")
    return out


# ────────────────────────────────────────────────────────────────────────
# Feature selection — two paths (see module docstring)
# ────────────────────────────────────────────────────────────────────────
def select_features(df, exclude=DATA_EXCLUDE,
                     quasi_constant_thr=0.98, corr_thr=0.95):
    """Pruned feature list used by the HTML reports.

    Drops, in order:
      1. Any column whose name contains an excluded substring (IDs, labels,
         alternative segment encodings) -- case-insensitive.
      2. Non-numeric columns.
      3. Quasi-constant columns (a single value covers >= ``quasi_constant_thr``
         of the rows).
      4. One of each pair of columns correlated above ``corr_thr`` (keeps the
         higher-variance one).

    Returns
    -------
    list[str]
        Ordered list of selected feature column names.
    """
    drop = set()
    for col in df.columns:
        cu = col.upper()
        if any(p.upper() in cu for p in exclude):
            drop.add(col)

    feat = [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

    # Drop quasi-constant columns.
    qc = [c for c in feat
          if df[c].value_counts(normalize=True, dropna=False).iloc[0] >= quasi_constant_thr]
    feat = [c for c in feat if c not in qc]

    # Drop one column from each highly correlated pair.
    if len(feat) > 1:
        corr = df[feat].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = set()
        for col in upper.columns:
            for c in upper.index[upper[col] > corr_thr]:
                to_drop.add(c if df[col].var() >= df[c].var() else col)
        feat = [c for c in feat if c not in to_drop]

    return feat


def basic_feature_columns(df, id_col=ID_COL, label_col=LABEL_COL):
    """Unpruned feature list used by the Streamlit dashboard.

    Keeps every numeric column except the identifier, the label, and any raw
    ``WEEK_ID*`` columns. No quasi-constant or correlation pruning is applied,
    matching the dashboard's original behaviour exactly.

    Returns
    -------
    list[str]
    """
    drop_cols = {id_col, label_col}
    drop_cols |= {c for c in df.columns if c.startswith("WEEK_ID")}
    return [c for c in df.columns
            if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]


# ────────────────────────────────────────────────────────────────────────
# Descriptive statistics
# ────────────────────────────────────────────────────────────────────────
def compute_segment_stats(df_lab, label_col, feat_keys):
    """Per-segment p25 / median / p75 for each requested feature.

    Used to draw the reference bands ("where does a typical SEG_x doctor sit?")
    in the Doctor Explorer.

    Returns
    -------
    dict
        ``{feature: {segment: {"p25": ..., "median": ..., "p75": ...}}}``
    """
    stats = {}
    for f in feat_keys:
        if f not in df_lab.columns:
            continue
        per_seg = {}
        for seg in VALID_LABELS:
            vals = df_lab.loc[df_lab[label_col] == seg, f].values
            per_seg[seg] = {
                "p25": float(np.quantile(vals, 0.25)),
                "median": float(np.median(vals)),
                "p75": float(np.quantile(vals, 0.75)),
            }
        stats[f] = per_seg
    return stats
