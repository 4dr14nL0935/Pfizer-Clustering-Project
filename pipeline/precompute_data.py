"""
precompute_data.py
==================
One-shot script that runs the full HCP segmentation pipeline (training,
5-fold CV, SHAP, counterfactual scenarios, conversion strategy, segment
medians, +1 visit simulation) and saves all the results to a single
pickle file.  The output is then loaded by `pfizer_dashboard.py` so the
dashboard renders instantly with NO training on every page load.

Run once:
    python precompute_data.py

Output:
    precomputed_results.pkl   ←  loaded by pfizer_dashboard.py
"""

import pickle
import sys
import io
import time
from pathlib import Path

# Force UTF-8 on Windows so unicode arrows / Greek letters print cleanly
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# ─────────────────────────────────────────────────────────────────────────
# CONFIG — keep in sync with dashboard.py
# ─────────────────────────────────────────────────────────────────────────
# -- Path bootstrap so `import src` resolves under `python pipeline/...` --
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

# -- Shared core: single source of truth for config, features & model --
from src import (
    SEED, VALID_LABELS, THR_A, THR_C, SEG_C_WEIGHT,
    ORDINAL_PARAMS, ARGMAX_PARAMS, KEY_FEATURES,
    N_SHAP_TOP, N_CONVERSION_CANDIDATES,
    add_features, fit_ordinal, predict_ordinal, fit_argmax, predict_argmax,
    get_shap_values, top_shap_per_doctor,
)
DATASET_NAME = "doctors_aggregated"
OUTPUT_FILE = str(_REPO_ROOT / "precomputed_results.pkl")


# ─────────────────────────────────────────────────────────────────────────
# Feature engineering — same as dashboard.add_features
# ─────────────────────────────────────────────────────────────────────────
def add_features(df):
    """Engineered features -- delegates to the shared src/ core."""
    from src import add_features as _add_features
    return _add_features(df, verbose=False)


def fit_ordinal(X, y):
    """Two-stage ordinal model -- delegates to src/."""
    from src import fit_ordinal as _fit_ordinal
    return _fit_ordinal(X, y)


def predict_ordinal(m1, m2, X):
    """Decision cascade (thresholds + dominance) -- delegates to src/."""
    from src import predict_ordinal as _predict_ordinal
    return _predict_ordinal(m1, m2, X)


def fit_argmax(X, y):
    """Argmax baseline -- delegates to src/."""
    from src import fit_argmax as _fit_argmax
    return _fit_argmax(X, y)


def predict_argmax(m, le, X):
    """Argmax baseline prediction -- delegates to src/."""
    from src import predict_argmax as _predict_argmax
    return _predict_argmax(m, le, X)


def get_shap_values(model, X):
    """TreeSHAP contributions -- delegates to src/."""
    from src import get_shap_values as _gsv
    return _gsv(model, X)


def top_shap_per_doctor(shap_vals, top_n=N_SHAP_TOP):
    """Top-N SHAP features per doctor -- delegates to src/."""
    from src import top_shap_per_doctor as _tsp
    return _tsp(shap_vals, top_n)


def weighted_balanced_accuracy(y_true, y_pred, class_weights=None):
    """Balanced accuracy with SEG_C up-weighted."""
    if class_weights is None:
        class_weights = {"SEG_A": 1.0, "SEG_B": 1.0, "SEG_C": float(SEG_C_WEIGHT)}
    classes = sorted(set(y_true))
    recalls = {}
    for cls in classes:
        mask = y_true == cls
        recalls[cls] = float((y_pred[mask] == cls).mean()) if mask.sum() else 0.0
    total_w = sum(class_weights.get(c, 1.0) for c in classes)
    if total_w == 0:
        return 0.0
    return sum(recalls[c] * class_weights.get(c, 1.0) for c in classes) / total_w


def metrics_for(y_true, pred):
    """Headline metrics (WBA, balanced accuracy, recall_C, C->A loss)."""
    wba = float(weighted_balanced_accuracy(y_true, pred))
    bal_acc = float(balanced_accuracy_score(y_true, pred))
    mask_C = y_true == "SEG_C"
    recall_C = float((pred[mask_C] == "SEG_C").mean()) if mask_C.sum() > 0 else 0.0
    c_lost_pct = float(((y_true == "SEG_C") & (pred == "SEG_A")).sum()
                        / max(mask_C.sum(), 1))
    cm = confusion_matrix(y_true, pred, labels=list(VALID_LABELS))
    return {
        "accuracy": wba,
        "balanced_accuracy": bal_acc,
        "recall_C": recall_C,
        "c_lost_pct": c_lost_pct,
        "confusion_matrix": cm,
    }


# ─────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────
def load_data(dataset_name):
    """Load the input CSV and normalise the label column."""
    df = pd.read_csv(_REPO_ROOT / "data" / "processed" / f"{dataset_name}.csv")
    df["ATSEG_first"] = df["ATSEG_first"].replace("0", np.nan)
    return df


def main():
    """Command-line entry point: run the full pipeline and write outputs."""
    t_start = time.time()
    print("=" * 65)
    print(f"  PRECOMPUTE — {DATASET_NAME}.csv → {OUTPUT_FILE}")
    print("=" * 65)

    # ── Load + feature-engineer ──
    print("\n[1/8] Loading data...")
    df_raw = load_data(DATASET_NAME)
    print(f"  Shape: {df_raw.shape}")

    df = add_features(df_raw.copy())

    is_labeled = df["ATSEG_first"].isin(VALID_LABELS).values
    y_all = df["ATSEG_first"].fillna("").astype(str).values
    ids_all = df["NUEVO_ID"].astype(str).values

    drop_cols = {"NUEVO_ID", "ATSEG_first"}
    drop_cols |= {c for c in df.columns if c.startswith("WEEK_ID")}
    feat_cols = [c for c in df.columns
                 if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    # quasi-constant
    qc_drop = [c for c in feat_cols
               if df[c].value_counts(normalize=True, dropna=False).iloc[0] >= 0.98]
    feat_cols = [c for c in feat_cols if c not in qc_drop]
    # high correlation
    if len(feat_cols) > 1:
        corr = df[feat_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        corr_drop = set()
        for col in upper.columns:
            for c in upper.index[upper[col] > 0.95]:
                corr_drop.add(c if df[col].var() >= df[c].var() else col)
        feat_cols = [c for c in feat_cols if c not in corr_drop]

    X_all = df[feat_cols].fillna(0).values
    X_lab = X_all[is_labeled]
    y_lab = y_all[is_labeled]
    X_unlab = X_all[~is_labeled]
    n_labeled = int(is_labeled.sum())
    n_unlabeled = int((~is_labeled).sum())
    print(f"  Features after filtering: {len(feat_cols)}")
    print(f"  Labeled: {n_labeled:,}  Unlabeled: {n_unlabeled:,}")

    # ── OOF CV + per-fold CI bounds ──
    print("\n[2/8] 5-fold OOF CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    n = len(y_lab)
    oof = {
        "ord": {"P_A": np.zeros(n), "P_B": np.zeros(n), "P_C": np.zeros(n),
                 "pred": np.empty(n, dtype=object)},
        "amx": {"P_A": np.zeros(n), "P_B": np.zeros(n), "P_C": np.zeros(n),
                 "pred": np.empty(n, dtype=object)},
    }
    cf_probs = np.zeros((n, 5, 3))
    for fold_i, (tr, va) in enumerate(skf.split(X_lab, y_lab), 1):
        print(f"  fold {fold_i}/5  ...", end=" ", flush=True)
        m1, m2 = fit_ordinal(X_lab[tr], y_lab[tr])
        pa, pb, pc, pr = predict_ordinal(m1, m2, X_lab[va])
        oof["ord"]["P_A"][va] = pa; oof["ord"]["P_B"][va] = pb
        oof["ord"]["P_C"][va] = pc; oof["ord"]["pred"][va] = pr
        pa_all, pb_all, pc_all, _ = predict_ordinal(m1, m2, X_lab)
        cf_probs[:, fold_i-1, :] = np.column_stack([pa_all, pb_all, pc_all])

        m, le = fit_argmax(X_lab[tr], y_lab[tr])
        pa, pb, pc, pr = predict_argmax(m, le, X_lab[va])
        oof["amx"]["P_A"][va] = pa; oof["amx"]["P_B"][va] = pb
        oof["amx"]["P_C"][va] = pc; oof["amx"]["pred"][va] = pr
        print("done")

    cf_mean = cf_probs.mean(axis=1)
    cf_std  = cf_probs.std(axis=1)
    ci_lo_lab = np.clip(cf_mean - 1.96 * cf_std, 0, 1)
    ci_hi_lab = np.clip(cf_mean + 1.96 * cf_std, 0, 1)

    # ── Final models + SHAP ──
    print("\n[3/8] Training final models + SHAP...")
    final_m1, final_m2 = fit_ordinal(X_lab, y_lab)
    if n_unlabeled > 0:
        ord_unlab = predict_ordinal(final_m1, final_m2, X_unlab)
        final_m_amx, final_le_amx = fit_argmax(X_lab, y_lab)
        amx_unlab = predict_argmax(final_m_amx, final_le_amx, X_unlab)
    else:
        ord_unlab = amx_unlab = (np.array([]),) * 4
    shap_lab = get_shap_values(final_m2, X_lab)
    top_shap_lab = top_shap_per_doctor(shap_lab, top_n=N_SHAP_TOP)

    # ── Reassemble full vectors ──
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

    ci_full_lo = np.zeros((len(df), 3))
    ci_full_hi = np.zeros((len(df), 3))
    ci_full_lo[is_labeled] = ci_lo_lab
    ci_full_hi[is_labeled] = ci_hi_lab

    shap_full = [None] * len(df)
    lab_indices = np.where(is_labeled)[0]
    for k, idx in enumerate(lab_indices):
        shap_full[idx] = top_shap_lab[k]

    metrics = {}
    for mod in ["ord", "amx"]:
        metrics[mod] = metrics_for(y_lab, oof[mod]["pred"])

    avail_keys = [(k, n_, d_) for k, n_, d_, _ in KEY_FEATURES if k in df.columns]
    avail_keys_full = [(k, n_, d_, a) for k, n_, d_, a in KEY_FEATURES if k in df.columns]
    raw_fv = {k: df[k].values.copy() for k, _, _, _ in KEY_FEATURES if k in df.columns}

    R = {
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
        "available_features_full": avail_keys_full,
        "raw_fv": raw_fv,
        "ci_lo": ci_full_lo,
        "ci_hi": ci_full_hi,
        "shap_top": shap_full,
    }

    # ── Segment medians (top 6 features) ──
    print("\n[4/8] Segment medians...")
    feats_full = [(k, n_) for k, n_, _, _ in avail_keys_full][:6]
    seg_medians = {}
    for seg in VALID_LABELS:
        mask = df["ATSEG_first"] == seg
        seg_medians[seg] = [
            float(df.loc[mask, k].median()) if mask.sum() > 0 else 0.0
            for k, _ in feats_full
        ]

    # ── Counterfactual scenarios (10 of them + 0..10 sweep) ──
    print("\n[5/8] Counterfactual scenarios (this is the slow one)...")
    pred_all_ord = full["ord"]["pred"]
    PC_all_ord = full["ord"]["P_C"]
    seg_b_idx = np.where(pred_all_ord == "SEG_B")[0]
    n_seg_b = int(len(seg_b_idx))

    c_mask = is_labeled & (y_all == "SEG_C")
    details_c_med = float(np.median(df_raw["DETAILS_sum"].values[c_mask])) if c_mask.sum() > 0 else 0.0
    samples_c_med = float(np.median(df_raw["SAMPLES_sum"].values[c_mask])) if c_mask.sum() > 0 else 0.0

    def _simulate(details_delta=None, details_target=None,
                   samples_delta=None, samples_target=None):
        """Re-score doctors after a simulated engagement intervention."""
        df_sim = df_raw.copy()
        sb_rows = df_sim.index[seg_b_idx]
        if details_delta is not None:
            df_sim.loc[sb_rows, "DETAILS_sum"] = df_sim.loc[sb_rows, "DETAILS_sum"] + details_delta
        if details_target is not None:
            df_sim.loc[sb_rows, "DETAILS_sum"] = df_sim.loc[sb_rows, "DETAILS_sum"].clip(lower=details_target)
        if samples_delta is not None:
            df_sim.loc[sb_rows, "SAMPLES_sum"] = df_sim.loc[sb_rows, "SAMPLES_sum"] + samples_delta
        if samples_target is not None:
            df_sim.loc[sb_rows, "SAMPLES_sum"] = df_sim.loc[sb_rows, "SAMPLES_sum"].clip(lower=samples_target)
        df_sim = add_features(df_sim)
        X_sim = df_sim.reindex(columns=feat_cols, fill_value=0).fillna(0).values
        PA_s, PB_s, PC_s, pred_s = predict_ordinal(final_m1, final_m2, X_sim[seg_b_idx])
        return PA_s, PB_s, PC_s, pred_s

    scenarios = [
        ("DETAILS", "+1 visit",                  dict(details_delta=1)),
        ("DETAILS", "+2 visits",                 dict(details_delta=2)),
        ("DETAILS", "+3 visits",                 dict(details_delta=3)),
        ("DETAILS", "+5 visits",                 dict(details_delta=5)),
        ("DETAILS", f"→ SEG_C median (≈{details_c_med:.0f})",
         dict(details_target=details_c_med)),
        ("SAMPLES", "+1 sample",                 dict(samples_delta=1)),
        ("SAMPLES", "+2 samples",                dict(samples_delta=2)),
        ("COMBINED", "+1 visit + 1 sample",      dict(details_delta=1, samples_delta=1)),
        ("COMBINED", "+2 visits + 2 samples",    dict(details_delta=2, samples_delta=2)),
        ("COMBINED", "Both → SEG_C medians",
         dict(details_target=details_c_med, samples_target=samples_c_med)),
    ]
    rows = []
    for group, label, spec in scenarios:
        _, _, PC_s, pred_s = _simulate(**spec)
        rows.append({
            "Group": group, "Scenario": label,
            "B→C": int((pred_s == "SEG_C").sum()),
            "B stays": int((pred_s == "SEG_B").sum()),
            "B→A": int((pred_s == "SEG_A").sum()),
            "Total B": n_seg_b,
            "% to C": round(int((pred_s == "SEG_C").sum()) / max(n_seg_b, 1) * 100, 1),
            "Mean P(C) after": round(float(PC_s.mean()), 4),
        })
        print(f"  scenario '{label}' → B→C={rows[-1]['B→C']}")
    summary_df = pd.DataFrame(rows)

    # +1 visit detail + diminishing returns
    print("  +1 visit detail + diminishing returns sweep...")
    _, _, PC_plus1, pred_plus1 = _simulate(details_delta=1)
    flippers_mask = pred_plus1 == "SEG_C"
    flippers_idx = seg_b_idx[flippers_mask]
    stayers_mask = pred_plus1 == "SEG_B"
    stayers_idx = seg_b_idx[stayers_mask]
    pc_base_seg_b = PC_all_ord[seg_b_idx]

    dim_returns = []
    for delta in range(0, 11):
        if delta == 0:
            n_flip = 0
        else:
            _, _, _, pred_d = _simulate(details_delta=delta)
            n_flip = int((pred_d == "SEG_C").sum())
        dim_returns.append({"Added visits": delta, "B→C cumulative": n_flip})
    dim_df = pd.DataFrame(dim_returns)
    dim_df["Incremental"] = dim_df["B→C cumulative"].diff().fillna(0).astype(int)

    CF = {
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

    # ── Conversion strategy ──
    print("\n[6/8] Conversion strategy candidates...")
    feat_meta = {k: (n_, d_, a) for (k, n_, d_, a) in avail_keys_full}
    c_mask_lab = is_labeled & (y_all == "SEG_C")
    c_medians = {k: float(np.median(raw_fv[k][c_mask_lab]))
                  for k in raw_fv if c_mask_lab.sum() > 0}

    pb_sorted = seg_b_idx[np.argsort(-PC_all_ord[seg_b_idx])]
    cand_idx = pb_sorted[:N_CONVERSION_CANDIDATES]

    cand_rows = []
    for i in cand_idx:
        row = {
            "HCP_ID":   ids_all[i],
            "True":     y_all[i] if is_labeled[i] else "Unlabeled",
            "P(A)":     round(float(full["ord"]["P_A"][i]), 3),
            "P(B)":     round(float(full["ord"]["P_B"][i]), 3),
            "P(C)":     round(float(full["ord"]["P_C"][i]), 3),
        }
        worst_gap = 0.0
        worst_feat = None
        for k in c_medians:
            v = float(raw_fv[k][i])
            gap = v - c_medians[k]
            row[f"{feat_meta[k][0]}"] = round(v, 2)
            if feat_meta[k][2] and gap < worst_gap:
                worst_gap = gap
                worst_feat = feat_meta[k][0]
        row["Top action"] = f"↑ {worst_feat}" if worst_feat else "—"
        row["Action gap"] = round(worst_gap, 2) if worst_feat else 0.0
        cand_rows.append(row)
    candidates_df = pd.DataFrame(cand_rows)

    agg_rows = []
    for k in c_medians:
        n_below = sum(1 for i in cand_idx if float(raw_fv[k][i]) < c_medians[k])
        agg_rows.append({
            "Feature":    feat_meta[k][0],
            "Code":       k,
            "Actionable": feat_meta[k][2],
            "% below SEG_C median": round(n_below / max(len(cand_idx), 1) * 100, 1),
            "SEG_C median": round(c_medians[k], 2),
        })
    aggregate_df = (pd.DataFrame(agg_rows)
                     .sort_values("% below SEG_C median", ascending=False)
                     .reset_index(drop=True))

    # ── +1 visit applied only to predicted SEG_B (segment counts) ──
    print("\n[7/8] +1 visit on SEG_B → full-population segment counts...")
    df_sim = df_raw.copy()
    df_sim.loc[df_sim.index[seg_b_idx], "DETAILS_sum"] = (
        df_sim.loc[df_sim.index[seg_b_idx], "DETAILS_sum"] + 1
    )
    df_sim = add_features(df_sim)
    X_sim = df_sim.reindex(columns=feat_cols, fill_value=0).fillna(0).values
    _, _, _, pred_seg_b_after = predict_ordinal(final_m1, final_m2, X_sim[seg_b_idx])
    pred_after = pred_all_ord.copy()
    pred_after[seg_b_idx] = pred_seg_b_after
    plus1_all = {
        "SEG_A": int((pred_after == "SEG_A").sum()),
        "SEG_B": int((pred_after == "SEG_B").sum()),
        "SEG_C": int((pred_after == "SEG_C").sum()),
        "total": int(len(pred_after)),
        "n_seg_b_baseline": int(len(seg_b_idx)),
        "b_to_c": int((pred_seg_b_after == "SEG_C").sum()),
        "b_to_a": int((pred_seg_b_after == "SEG_A").sum()),
        "b_stays": int((pred_seg_b_after == "SEG_B").sum()),
    }

    # ── Bundle + save ──
    print("\n[8/8] Writing pickle...")
    data = {
        "R": R,
        "CF": CF,
        "conversion_strategy": {
            "candidates_df": candidates_df,
            "aggregate_df": aggregate_df,
            "c_medians": c_medians,
        },
        "plus1_visit_all": plus1_all,
        "segment_medians": {
            "feats_full": feats_full,
            "seg_meds": seg_medians,
        },
        "config": {
            "SEED": SEED,
            "THR_A": THR_A,
            "THR_C": THR_C,
            "SEG_C_WEIGHT": SEG_C_WEIGHT,
            "KEY_FEATURES": KEY_FEATURES,
            "N_SHAP_TOP": N_SHAP_TOP,
            "DATASET_NAME": DATASET_NAME,
        },
        "df_raw": df_raw,
    }
    out_path = Path(OUTPUT_FILE)
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = out_path.stat().st_size / 1024 / 1024
    elapsed = time.time() - t_start

    print(f"\n  ✓ Saved {out_path}  ({size_mb:.1f} MB)")
    print(f"  Total time: {elapsed:.1f} s")
    print("=" * 65)


if __name__ == "__main__":
    main()
