"""
model.py
========
The two models compared in the capstone, plus cross-validation and SHAP.

  MODEL 1 - Argmax baseline
      A single multiclass XGBoost with balanced sample weights.
      Decision = argmax(P_A, P_B, P_C). Optimises raw accuracy.

  MODEL 2 - Business-calibrated ordinal  (the chosen model)
      Two cascading binary classifiers:
          m1 estimates P(>=B)   (i.e. "not SEG_A")
          m2 estimates P(>=C)   (i.e. "is SEG_C"), with SEG_C up-weighted.
      Class probabilities are recovered as:
          P_A = 1 - P(>=B) ;  P_B = P(>=B) - P(>=C) ;  P_C = P(>=C)
      Decision cascade (thresholds + dominance rule):
          P(A) >= THR_A          -> SEG_A
          P(C) >= THR_C          -> SEG_C
          P(C) >  P(B)           -> SEG_C   (dominance rule, v3 change)
          otherwise              -> SEG_B
      Optimises for catching SEG_C and avoiding the catastrophic C->A error.

Tec de Monterrey x Pfizer Global Commercial Analytics
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from .config import (
    ORDINAL_PARAMS, ARGMAX_PARAMS,
    THR_A, THR_C, SEG_C_WEIGHT, SEED, N_SHAP_TOP,
)


# ────────────────────────────────────────────────────────────────────────
# Model 2 — Business-calibrated ordinal
# ────────────────────────────────────────────────────────────────────────
def fit_ordinal(X, y):
    """Train the two cascading binary stages of the ordinal model.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray of str
        Labels in ``{"SEG_A", "SEG_B", "SEG_C"}``.

    Returns
    -------
    (m1, m2) : tuple of fitted XGBClassifier
        m1 predicts P(>=B); m2 predicts P(>=C) with SEG_C up-weighted.
    """
    m1 = xgb.XGBClassifier(**ORDINAL_PARAMS)
    m1.fit(X, (y != "SEG_A").astype(int))

    sw = np.where(y == "SEG_C", SEG_C_WEIGHT, 1.0)
    m2 = xgb.XGBClassifier(**ORDINAL_PARAMS)
    m2.fit(X, (y == "SEG_C").astype(int), sample_weight=sw)
    return m1, m2


def predict_ordinal(m1, m2, X):
    """Apply the decision cascade (thresholds + dominance rule).

    Returns
    -------
    (P_A, P_B, P_C, pred) : tuple of np.ndarray
        Normalised class probabilities and the final string predictions.
    """
    pb = m1.predict_proba(X)[:, 1]
    # P(>=C) can never exceed P(>=B); clip to keep the ordinal ordering valid.
    pc = np.minimum(m2.predict_proba(X)[:, 1], pb)

    PA = 1 - pb
    PB = pb - pc
    PC = pc
    s = PA + PB + PC + 1e-9       # renormalise to a proper distribution
    PA, PB, PC = PA / s, PB / s, PC / s

    pred = np.where(PA >= THR_A, "SEG_A",
           np.where(PC >= THR_C, "SEG_C",
           np.where(PC > PB,     "SEG_C", "SEG_B")))
    return PA, PB, PC, pred


# ────────────────────────────────────────────────────────────────────────
# Model 1 — Argmax baseline
# ────────────────────────────────────────────────────────────────────────
def fit_argmax(X, y):
    """Train the balanced multiclass baseline.

    Returns
    -------
    (model, label_encoder) : tuple
    """
    le = LabelEncoder()
    ye = le.fit_transform(y)
    sw = compute_sample_weight("balanced", y=ye)
    m = xgb.XGBClassifier(**ARGMAX_PARAMS)
    m.fit(X, ye, sample_weight=sw)
    return m, le


def predict_argmax(m, le, X):
    """Predict with the baseline; decision = argmax of class probabilities.

    Returns
    -------
    (P_A, P_B, P_C, pred) : tuple of np.ndarray
    """
    pr = m.predict_proba(X)
    cl = list(le.classes_)
    ia, ib, ic = cl.index("SEG_A"), cl.index("SEG_B"), cl.index("SEG_C")
    pred = le.inverse_transform(pr.argmax(axis=1)).astype(object)
    return pr[:, ia], pr[:, ib], pr[:, ic], pred


# ────────────────────────────────────────────────────────────────────────
# Out-of-fold cross-validation (+ cross-fold confidence intervals)
# ────────────────────────────────────────────────────────────────────────
def oof_with_ci(X, y, n_splits=5, seed=SEED):
    """Honest 5-fold out-of-fold predictions for both models.

    For the ordinal model it additionally re-scores *all* rows with every
    fold's model, so the spread across folds yields a 95% confidence interval
    on each doctor's class probabilities.

    Returns
    -------
    out : dict
        ``{"ord": {...}, "amx": {...}}`` each with keys P_A, P_B, P_C, pred
        (out-of-fold, one prediction per labeled row).
    ci_lo, ci_hi : np.ndarray of shape (n, 3)
        Lower / upper 95% bounds on (P_A, P_B, P_C) from cross-fold spread.
    fold_models : list of (m1, m2)
        The per-fold ordinal models (handy for counterfactual re-scoring).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n = len(y)
    out = {m: {"P_A": np.zeros(n), "P_B": np.zeros(n), "P_C": np.zeros(n),
               "pred": np.empty(n, dtype=object)} for m in ["ord", "amx"]}
    cf_probs = np.zeros((n, n_splits, 3))
    fold_models = []

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        Xtr, Xva, ytr = X[tr], X[va], y[tr]

        print(f"    Fold {fold}/{n_splits}: ordinal...", end=" ", flush=True)
        om1, om2 = fit_ordinal(Xtr, ytr)
        pa, pb, pc, pr = predict_ordinal(om1, om2, Xva)
        out["ord"]["P_A"][va] = pa
        out["ord"]["P_B"][va] = pb
        out["ord"]["P_C"][va] = pc
        out["ord"]["pred"][va] = pr
        fold_models.append((om1, om2))

        # Re-score everyone with this fold's model for the CI.
        pa_all, pb_all, pc_all, _ = predict_ordinal(om1, om2, X)
        cf_probs[:, fold - 1, :] = np.column_stack([pa_all, pb_all, pc_all])

        print("argmax...", flush=True)
        am, ale = fit_argmax(Xtr, ytr)
        pa, pb, pc, pr = predict_argmax(am, ale, Xva)
        out["amx"]["P_A"][va] = pa
        out["amx"]["P_B"][va] = pb
        out["amx"]["P_C"][va] = pc
        out["amx"]["pred"][va] = pr

    cf_mean = cf_probs.mean(axis=1)
    cf_std = cf_probs.std(axis=1)
    ci_lo = np.clip(cf_mean - 1.96 * cf_std, 0, 1)
    ci_hi = np.clip(cf_mean + 1.96 * cf_std, 0, 1)
    return out, ci_lo, ci_hi, fold_models


# ────────────────────────────────────────────────────────────────────────
# SHAP (TreeSHAP via the native booster)
# ────────────────────────────────────────────────────────────────────────
def get_shap_values(model, X):
    """Return per-row SHAP contributions (drops the bias column).

    Parameters
    ----------
    model : fitted XGBClassifier
        Typically the P(>=C) stage (``m2``), since SEG_C drives the business.
    X : np.ndarray

    Returns
    -------
    np.ndarray of shape (n_rows, n_features)
    """
    dm = xgb.DMatrix(X)
    contribs = model.get_booster().predict(dm, pred_contribs=True)
    return contribs[:, :-1]      # last column is the bias term


def top_shap_per_doctor(shap_vals, top_n=N_SHAP_TOP):
    """For each doctor, the ``top_n`` features by absolute SHAP value.

    Returns
    -------
    list[list[[int, float]]]
        One list per row; each entry is ``[feature_index, signed_shap]``.
    """
    result = []
    for i in range(shap_vals.shape[0]):
        vals = shap_vals[i]
        idx = np.argsort(np.abs(vals))[::-1][:top_n]
        result.append([[int(j), round(float(vals[j]), 4)] for j in idx])
    return result
