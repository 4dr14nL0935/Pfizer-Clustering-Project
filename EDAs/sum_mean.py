"""
================================================================================
ENSEMBLE TWO-STAGE + REJECT OPTION — sums vs means  (v3)
================================================================================

Improvements over previous two-stage script:

  1. ENSEMBLE in each stage: average of calibrated probabilities from
     XGBoost + Random Forest + ExtraTrees. Different model families make
     different errors; averaging usually adds 1-2 points of balanced acc.

  2. AUTOMATIC THRESHOLD TUNING on the calibration holdout (not on test!).
     For each candidate threshold, balanced accuracy is computed on the
     validation set, and the threshold that maximizes it is selected.

  3. METRICS PRINTED LARGE AND VISIBLE on every figure panel.

For each dataset (sums, means) reports baseline + two-stage no-reject +
two-stage with tuned reject threshold. Per model: accuracy, balanced acc,
F1 macro, F1 per class, recall per class, full confusion matrix.

Outputs:
  panel_<dataset>.png          (6-panel figure with metrics on every panel)
  metrics_<dataset>.csv
  operating_curve_<dataset>.csv
  summary_all_metrics.csv

Requires: pandas, numpy, scikit-learn, imbalanced-learn, xgboost,
          matplotlib, seaborn
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (balanced_accuracy_score, accuracy_score,
                             confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# =====================================================================
# CONFIG
# =====================================================================
DATASETS = {
    'sums':  'data/processed/doctors_sums.csv',
    'means': 'data/processed/doctors_means.csv',
}
DROP_COLS = ['NUEVO_ID', 'ATSEG_first',
             'WEEK_ID_first', 'WEEK_ID_last', 'WEEK_ID_count']
LABELS = ['SEG_A', 'SEG_B', 'SEG_C']
RANDOM_STATE = 42

XGB_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.08,
                  subsample=0.85, colsample_bytree=0.85,
                  random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss')
RF_PARAMS  = dict(n_estimators=400, max_depth=15, min_samples_leaf=5,
                  class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
ET_PARAMS  = dict(n_estimators=400, max_depth=15, min_samples_leaf=5,
                  class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)


# =====================================================================
# 1. LOAD + PREPROCESS
# =====================================================================
def load_and_prep(path):
    df = pd.read_csv(path)
    df = df[df['ATSEG_first'] != '0'].copy()
    feat = [c for c in df.columns if c not in DROP_COLS]
    X = df[feat].copy()
    y = df['ATSEG_first'].copy()
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(
            X[col].fillna('MISSING').astype(str))
    X = X.fillna(0)
    X.columns = (X.columns
                 .str.replace(r'[\[\]<>(),]', '_', regex=True)
                 .str.replace(r'\s+', '_', regex=True)
                 .str.replace(r'_+', '_', regex=True)
                 .str.strip('_'))
    return X, y


def _wrap_prefit(base):
    """sklearn 1.6+ removed cv='prefit'; use FrozenEstimator instead."""
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(FrozenEstimator(base), method='isotonic')
    except ImportError:
        return CalibratedClassifierCV(base, cv='prefit', method='isotonic')


# =====================================================================
# 2. BASELINE  (single-stage XGB + SMOTE; reference)
# =====================================================================
def baseline_single_stage(X_tr, y_tr, X_te):
    Xs, ys = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr, y_tr)
    le = LabelEncoder()
    ys_enc = le.fit_transform(ys)
    sw = compute_sample_weight('balanced', y=ys_enc)
    m = XGBClassifier(objective='multi:softprob', num_class=3,
                      **{**XGB_PARAMS, 'eval_metric': 'mlogloss'})
    m.fit(Xs, ys_enc, sample_weight=sw)
    return le.inverse_transform(m.predict(X_te))


# =====================================================================
# 3. ENSEMBLE BINARY STAGE (XGB + RF + ExtraTrees, all calibrated)
# =====================================================================
def fit_ensemble_binary(X_fit, y_fit_bin, X_cal, y_cal_bin):
    Xs, ys = SMOTE(random_state=RANDOM_STATE).fit_resample(X_fit, y_fit_bin)
    bases = [
        XGBClassifier(objective='binary:logistic', **XGB_PARAMS),
        RandomForestClassifier(**RF_PARAMS),
        ExtraTreesClassifier(**ET_PARAMS),
    ]
    cals = []
    for b in bases:
        b.fit(Xs, ys)
        c = _wrap_prefit(b)
        c.fit(X_cal, y_cal_bin)
        cals.append(c)
    return cals


def ensemble_proba_pos(cals, X):
    """Mean of P(positive) across the calibrated members."""
    return np.mean([c.predict_proba(X)[:, 1] for c in cals], axis=0)


def train_two_stage_ensemble(X_tr, y_tr):
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=RANDOM_STATE, stratify=y_tr)

    print("    Stage 1 ensemble (A vs not-A)...")
    s1 = fit_ensemble_binary(
        X_fit, (y_fit == 'SEG_A').astype(int),
        X_cal, (y_cal == 'SEG_A').astype(int))

    print("    Stage 2 ensemble (B vs C, on B+C subset)...")
    mfit = y_fit.isin(['SEG_B', 'SEG_C'])
    mcal = y_cal.isin(['SEG_B', 'SEG_C'])
    s2 = fit_ensemble_binary(
        X_fit[mfit], (y_fit[mfit] == 'SEG_B').astype(int),
        X_cal[mcal], (y_cal[mcal] == 'SEG_B').astype(int))

    return s1, s2, X_cal, y_cal


def two_stage_proba(s1, s2, X):
    p_A = ensemble_proba_pos(s1, X)
    p_B_given_notA = ensemble_proba_pos(s2, X)
    p_B = (1 - p_A) * p_B_given_notA
    p_C = (1 - p_A) * (1 - p_B_given_notA)
    return np.column_stack([p_A, p_B, p_C])


def predict_with_reject(probs, threshold):
    preds = np.array(LABELS)[probs.argmax(axis=1)]
    max_p = probs.max(axis=1)
    return np.where(max_p < threshold, 'REVIEW', preds), max_p


# =====================================================================
# 4. THRESHOLD TUNING ON VALIDATION (NOT ON TEST)
# =====================================================================
def tune_threshold_on_val(s1, s2, X_val, y_val,
                          thresholds=np.arange(0.30, 0.71, 0.025)):
    val_probs = two_stage_proba(s1, s2, X_val)
    best = (0.0, -1.0, 0.0)  # (thr, balanced_acc, pct_review)
    for thr in thresholds:
        y_pred, _ = predict_with_reject(val_probs, thr)
        keep = y_pred != 'REVIEW'
        if keep.sum() < 50:
            continue
        ba = balanced_accuracy_score(np.asarray(y_val)[keep], y_pred[keep])
        if ba > best[1]:
            best = (thr, ba, (1 - keep.mean()) * 100)
    return best  # (best_threshold, val_balanced_acc, val_pct_review)


# =====================================================================
# 5. METRICS
# =====================================================================
def compute_metrics(y_true, y_pred, label='model'):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    keep = y_pred != 'REVIEW'
    y_t, y_p = y_true[keep], y_pred[keep]
    if keep.sum() == 0:
        return {'model': label}
    rep = classification_report(y_t, y_p, labels=LABELS,
                                output_dict=True, zero_division=0)
    out = {
        'model':        label,
        'n_total':      len(y_true),
        'n_kept':       int(keep.sum()),
        'pct_review':   round((1 - keep.mean()) * 100, 2),
        'accuracy':     round(accuracy_score(y_t, y_p) * 100, 2),
        'balanced_acc': round(balanced_accuracy_score(y_t, y_p) * 100, 2),
        'f1_macro':     round(rep['macro avg']['f1-score'] * 100, 2),
        'recall_macro': round(rep['macro avg']['recall']   * 100, 2),
    }
    for c in LABELS:
        out[f'recall_{c}']    = round(rep[c]['recall']    * 100, 2)
        out[f'f1_{c}']        = round(rep[c]['f1-score']  * 100, 2)
        out[f'precision_{c}'] = round(rep[c]['precision'] * 100, 2)
    return out


def operating_curve(y_true, probs, thresholds):
    rows = []
    for thr in thresholds:
        y_pred, _ = predict_with_reject(probs, thr)
        m = compute_metrics(y_true, y_pred)
        m['threshold'] = round(thr, 3)
        rows.append(m)
    return pd.DataFrame(rows)


# =====================================================================
# 6. PER-DATASET PIPELINE
# =====================================================================
def run_dataset(name, path):
    print(f"\n{'='*72}\n  DATASET: {name}  ({path})\n{'='*72}")
    X, y = load_and_prep(path)
    print(f"  HCPs: {X.shape[0]:,}  |  features: {X.shape[1]}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    print("\n  [1/4] Baseline single-stage XGB + SMOTE...")
    y_pred_base = baseline_single_stage(X_tr, y_tr, X_te)

    print("\n  [2/4] Training two-stage ENSEMBLE (XGB + RF + ExtraTrees)...")
    s1, s2, X_val, y_val = train_two_stage_ensemble(X_tr, y_tr)

    probs_te = two_stage_proba(s1, s2, X_te)
    y_pred_2s, _ = predict_with_reject(probs_te, threshold=0.0)

    print("\n  [3/4] Tuning reject threshold on validation set...")
    best_thr, val_ba, val_pct = tune_threshold_on_val(s1, s2, X_val, y_val)
    print(f"        best threshold = {best_thr:.3f}  "
          f"(val BalAcc = {val_ba*100:.2f}%, val %review = {val_pct:.1f}%)")

    y_pred_2s_thr, _ = predict_with_reject(probs_te, best_thr)

    print("\n  [4/4] Computing metrics + operating curve...")
    op = operating_curve(y_te, probs_te, np.arange(0.30, 0.71, 0.05))
    op.to_csv(f'operating_curve_{name}.csv', index=False)

    metrics = pd.DataFrame([
        compute_metrics(y_te, y_pred_base,    'baseline_xgb_smote'),
        compute_metrics(y_te, y_pred_2s,      'two_stage_ensemble'),
        compute_metrics(y_te, y_pred_2s_thr,  f'two_stage_ens_thr_{best_thr:.2f}'),
    ])
    metrics.to_csv(f'metrics_{name}.csv', index=False)
    print('\n  ' + metrics.to_string(index=False).replace('\n', '\n  '))

    return dict(
        name=name, n_features=X.shape[1],
        y_te=y_te, probs=probs_te,
        y_pred_base=y_pred_base, y_pred_2s=y_pred_2s, y_pred_2s_thr=y_pred_2s_thr,
        cm_base=confusion_matrix(y_te, y_pred_base, labels=LABELS),
        cm_2s=confusion_matrix(y_te, y_pred_2s, labels=LABELS),
        cm_2s_thr=confusion_matrix(
            np.asarray(y_te)[y_pred_2s_thr != 'REVIEW'],
            y_pred_2s_thr[y_pred_2s_thr != 'REVIEW'], labels=LABELS),
        op=op, metrics=metrics, best_thr=best_thr,
    )


# =====================================================================
# 7. PLOTTING WITH BIG VISIBLE METRICS
# =====================================================================
def _metric_subtitle(m):
    """Big readable metric line for each panel."""
    return (f"Acc {m['accuracy']:.1f}%   |   "
            f"BalAcc {m['balanced_acc']:.1f}%   |   "
            f"F1 {m['f1_macro']:.1f}%   |   "
            f"Recall(C) {m['recall_SEG_C']:.1f}%")


def plot_dataset_panel(res, save_path):
    name = res['name']
    metrics = res['metrics']
    m_base = metrics.iloc[0]; m_2s = metrics.iloc[1]; m_2s_thr = metrics.iloc[2]

    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.6, 1.6, 1.0],
                          hspace=0.55, wspace=0.30)

    # ---- ROW 1: three confusion matrices ----
    cm_specs = [
        (res['cm_base'],  m_base,   'Blues',
         'Baseline\nsingle-stage XGB + SMOTE', '#1f77b4'),
        (res['cm_2s'],    m_2s,     'Purples',
         'Two-stage Ensemble\n(no reject)', '#7b3f99'),
        (res['cm_2s_thr'], m_2s_thr, 'Greens',
         f'Two-stage Ensemble\n(reject thr = {res["best_thr"]:.2f})', '#1e8449'),
    ]
    for col, (cm, m, cmap, title, color) in enumerate(cm_specs):
        ax = fig.add_subplot(gs[0, col])
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=LABELS, yticklabels=LABELS, ax=ax,
                    annot_kws={'size': 16}, cbar=False)
        ax.set_title(title, fontsize=13, fontweight='bold', color=color)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        # Big metric line under the matrix
        ax.text(0.5, -0.32, _metric_subtitle(m),
                transform=ax.transAxes, ha='center', fontsize=12,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='#f4f4f4', edgecolor=color, lw=1.5))

    # ---- ROW 2 LEFT: per-class metrics bar chart ----
    ax_pc = fig.add_subplot(gs[1, 0])
    metric_names = ['recall', 'f1', 'precision']
    for i, mname in enumerate(metric_names):
        vals = [m_2s_thr[f'{mname}_{c}'] for c in LABELS]
        ax_pc.bar(np.arange(3) + i * 0.27, vals, 0.27,
                  label=mname.capitalize())
        for j, v in enumerate(vals):
            ax_pc.text(j + i * 0.27, v + 1, f'{v:.1f}',
                       ha='center', fontsize=9, fontweight='bold')
    ax_pc.set_xticks(np.arange(3) + 0.27)
    ax_pc.set_xticklabels(LABELS)
    ax_pc.set_ylim(0, 105); ax_pc.set_ylabel('%')
    ax_pc.set_title(f'Per-class metrics — Two-stage @ thr={res["best_thr"]:.2f}',
                    fontsize=12, fontweight='bold')
    ax_pc.legend(loc='upper right'); ax_pc.grid(axis='y', alpha=0.3)

    # ---- ROW 2 MIDDLE: operating curve ----
    ax_op = fig.add_subplot(gs[1, 1])
    op = res['op']
    ax_op.plot(op['pct_review'], op['balanced_acc'], 'o-',
               color='#7b3f99', lw=2, markersize=7)
    ax_op.axhline(m_base['balanced_acc'], color='#1f77b4', ls='--',
                  label=f"Baseline ({m_base['balanced_acc']:.1f}%)")
    ax_op.axvline((1 - m_2s_thr['n_kept'] / m_2s_thr['n_total']) * 100,
                  color='#1e8449', ls=':',
                  label=f"Selected thr ({m_2s_thr['balanced_acc']:.1f}%)")
    for _, r in op.iterrows():
        ax_op.annotate(f"{r['threshold']:.2f}",
                       (r['pct_review'], r['balanced_acc']),
                       fontsize=7, xytext=(4, 4), textcoords='offset points')
    ax_op.set_xlabel('% sent to manual REVIEW')
    ax_op.set_ylabel('Balanced Accuracy on auto-classified (%)')
    ax_op.set_title('Operating Curve', fontsize=12, fontweight='bold')
    ax_op.grid(alpha=0.3); ax_op.legend(fontsize=9)

    # ---- ROW 2 RIGHT: per-class recall vs threshold ----
    ax_rc = fig.add_subplot(gs[1, 2])
    op = res['op']
    ax_rc.plot(op['threshold'], op['recall_SEG_A'], 'o-',
               label='Recall SEG_A', color='#3498db')
    ax_rc.plot(op['threshold'], op['recall_SEG_B'], 's-',
               label='Recall SEG_B', color='#f39c12')
    ax_rc.plot(op['threshold'], op['recall_SEG_C'], '^-',
               label='Recall SEG_C', color='#e74c3c')
    ax_rc.set_xlabel('Confidence threshold')
    ax_rc.set_ylabel('Per-class recall on kept HCPs (%)')
    ax_rc.set_title('Which classes get rejected most?',
                    fontsize=12, fontweight='bold')
    ax_rc.grid(alpha=0.3); ax_rc.legend(fontsize=9)

    # ---- ROW 3: full metrics table as text ----
    ax_tab = fig.add_subplot(gs[2, :]); ax_tab.axis('off')
    cols = ['model', 'pct_review', 'accuracy', 'balanced_acc',
            'f1_macro', 'recall_SEG_A', 'recall_SEG_B', 'recall_SEG_C']
    table_data = metrics[cols].values
    headers = ['Model', '% Review', 'Accuracy', 'Bal Acc',
               'F1 macro', 'Recall A', 'Recall B', 'Recall C']
    table = ax_tab.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         colColours=['#34495e'] * len(headers))
    table.auto_set_font_size(False); table.set_fontsize(11)
    table.scale(1, 2.0)
    for k in range(len(headers)):
        table[0, k].set_text_props(color='white', fontweight='bold')
    ax_tab.set_title('Full metrics summary', fontsize=13, fontweight='bold',
                     pad=10)

    fig.suptitle(f'Dataset: {name}  ({res["n_features"]} features)  —  '
                 f'Two-stage Ensemble Pipeline',
                 fontsize=16, fontweight='bold', y=0.99)
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.show()


# =====================================================================
# MAIN
# =====================================================================
def main():
    results = {}
    for name, path in DATASETS.items():
        results[name] = run_dataset(name, path)
        plot_dataset_panel(results[name], f'panel_{name}.png')

    # Combined summary CSV
    rows = []
    for ds_name, res in results.items():
        m = res['metrics'].copy()
        m.insert(0, 'dataset', ds_name)
        rows.append(m)
    final = pd.concat(rows, ignore_index=True)
    final.to_csv('summary_all_metrics.csv', index=False)

    print(f"\n{'='*72}\n  FINAL SUMMARY (both datasets)\n{'='*72}")
    cols = ['dataset', 'model', 'pct_review', 'accuracy', 'balanced_acc',
            'f1_macro', 'recall_SEG_A', 'recall_SEG_B', 'recall_SEG_C']
    print(final[cols].to_string(index=False))
    print('\nSaved: panel_sums.png, panel_means.png, '
          'metrics_*.csv, operating_curve_*.csv, summary_all_metrics.csv')


if __name__ == '__main__':
    main()