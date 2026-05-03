"""
================================================================================
Random Forest vs XGBoost — Segmentación HCPs (doctors_aggregated.csv)
================================================================================

Same pipeline as the original 4-panel comparison, but extended to:
  - Compute accuracy, balanced accuracy, F1 (macro + per class),
    recall (per class), precision (per class) for both models.
  - Display all metric values directly on the figure (no need to
    look at the console).

Plots are in English.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, balanced_accuracy_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


# ----------------------------------------------------------------------
# 1. LOAD + PREPROCESS
# ----------------------------------------------------------------------
df = pd.read_csv('data/processed/doctors_aggregated.csv')
df = df[df['ATSEG_first'] != '0'].copy()
print(f"Labeled HCPs: {len(df):,}")
print(df['ATSEG_first'].value_counts(normalize=True).round(3))

drop_cols = ['NUEVO_ID', 'ATSEG_first',
             'WEEK_ID_first', 'WEEK_ID_last', 'WEEK_ID_count']
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].copy()
y = df['ATSEG_first'].copy()

for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].fillna('MISSING').astype(str))
X = X.fillna(0)

# XGBoost rejects [, ], <, etc. in feature names
X.columns = (X.columns
             .str.replace(r'[\[\]<>(),]', '_', regex=True)
             .str.replace(r'\s+', '_', regex=True)
             .str.replace(r'_+', '_', regex=True)
             .str.strip('_'))

LABELS = ['SEG_A', 'SEG_B', 'SEG_C']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


# ----------------------------------------------------------------------
# 2. TRAIN MODELS
# ----------------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                            class_weight='balanced',
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

le_y = LabelEncoder()
y_train_enc = le_y.fit_transform(y_train)
sw = compute_sample_weight('balanced', y=y_train_enc)

xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                    objective='multi:softprob', num_class=3,
                    random_state=42, n_jobs=-1, eval_metric='mlogloss')
xgb.fit(X_train, y_train_enc, sample_weight=sw)
y_pred_xgb = le_y.inverse_transform(xgb.predict(X_test))


# ----------------------------------------------------------------------
# 3. METRICS
# ----------------------------------------------------------------------
def full_metrics(y_true, y_pred, name):
    rep = classification_report(y_true, y_pred, labels=LABELS,
                                output_dict=True, zero_division=0)
    out = {
        'model':        name,
        'accuracy':     round(accuracy_score(y_true, y_pred) * 100, 2),
        'balanced_acc': round(balanced_accuracy_score(y_true, y_pred) * 100, 2),
        'f1_macro':     round(rep['macro avg']['f1-score'] * 100, 2),
        'recall_macro': round(rep['macro avg']['recall']   * 100, 2),
    }
    for c in LABELS:
        out[f'recall_{c}']    = round(rep[c]['recall']    * 100, 2)
        out[f'f1_{c}']        = round(rep[c]['f1-score']  * 100, 2)
        out[f'precision_{c}'] = round(rep[c]['precision'] * 100, 2)
    return out


m_rf  = full_metrics(y_test, y_pred_rf,  'Random Forest')
m_xgb = full_metrics(y_test, y_pred_xgb, 'XGBoost')

metrics_df = pd.DataFrame([m_rf, m_xgb])
print('\n' + '=' * 78)
print('METRIC SUMMARY')
print('=' * 78)
print(metrics_df.to_string(index=False))
metrics_df.to_csv('metrics_aggregated.csv', index=False)

cm_rf  = confusion_matrix(y_test, y_pred_rf,  labels=LABELS)
cm_xgb = confusion_matrix(y_test, y_pred_xgb, labels=LABELS)
acc_rf_per  = cm_rf.diagonal()  / cm_rf.sum(axis=1)  * 100
acc_xgb_per = cm_xgb.diagonal() / cm_xgb.sum(axis=1) * 100


# ----------------------------------------------------------------------
# 4. PLOT
# ----------------------------------------------------------------------
def metric_banner(m):
    return (f"Accuracy {m['accuracy']:.1f}%   |   "
            f"Balanced Acc {m['balanced_acc']:.1f}%   |   "
            f"F1 macro {m['f1_macro']:.1f}%   |   "
            f"Recall macro {m['recall_macro']:.1f}%")


fig = plt.figure(figsize=(18, 13))
gs = fig.add_gridspec(3, 2, height_ratios=[1.6, 1.3, 1.1],
                      hspace=0.55, wspace=0.25)

# --- Row 1: confusion matrices with metric banner under each ---
specs = [
    (cm_rf,  m_rf,  'Blues',  'Random Forest with Class Balancing', '#1f77b4'),
    (cm_xgb, m_xgb, 'Greens', 'XGBoost with Sample Weights',        '#2ca02c'),
]
for col, (cm, m, cmap, title, color) in enumerate(specs):
    ax = fig.add_subplot(gs[0, col])
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=LABELS, yticklabels=LABELS, ax=ax,
                annot_kws={'size': 16}, cbar=True)
    ax.set_title(title, fontsize=14, fontweight='bold', color=color)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.text(0.5, -0.28, metric_banner(m),
            transform=ax.transAxes, ha='center',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='#f4f4f4', edgecolor=color, lw=1.5))

# --- Row 2 left: per-class accuracy bars ---
ax_acc = fig.add_subplot(gs[1, 0])
x = np.arange(len(LABELS)); w = 0.35
b1 = ax_acc.bar(x - w/2, acc_rf_per,  w, label='Random Forest', color='#3498db')
b2 = ax_acc.bar(x + w/2, acc_xgb_per, w, label='XGBoost',       color='#2ecc71')
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax_acc.annotate(f'{h:.1f}%',
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=10, fontweight='bold')
ax_acc.set_title('Per-Segment Accuracy', fontweight='bold', fontsize=12)
ax_acc.set_ylabel('Accuracy (%)')
ax_acc.set_xticks(x); ax_acc.set_xticklabels(LABELS)
ax_acc.set_ylim(0, 100); ax_acc.legend(); ax_acc.grid(axis='y', alpha=0.3)

# --- Row 2 right: prediction distribution ---
ax_dist = fig.add_subplot(gs[1, 1])
real_counts = pd.Series(y_test).value_counts().reindex(LABELS).values
rf_counts   = pd.Series(y_pred_rf).value_counts().reindex(LABELS).fillna(0).values
xgb_counts  = pd.Series(y_pred_xgb).value_counts().reindex(LABELS).fillna(0).values
w = 0.27
ax_dist.bar(x - w, real_counts, w, label='Actual (ATSEG)',   color='gray')
ax_dist.bar(x,     rf_counts,   w, label='Random Forest',    color='#3498db')
ax_dist.bar(x + w, xgb_counts,  w, label='XGBoost',          color='#2ecc71')
ax_dist.set_title('Prediction Distribution', fontweight='bold', fontsize=12)
ax_dist.set_ylabel('Number of HCPs')
ax_dist.set_xticks(x); ax_dist.set_xticklabels(LABELS)
ax_dist.legend(); ax_dist.grid(axis='y', alpha=0.3)

# --- Row 3: full metrics table ---
ax_tab = fig.add_subplot(gs[2, :]); ax_tab.axis('off')
display_cols = ['model', 'accuracy', 'balanced_acc', 'f1_macro',
                'recall_SEG_A', 'recall_SEG_B', 'recall_SEG_C',
                'f1_SEG_A', 'f1_SEG_B', 'f1_SEG_C']
headers = ['Model', 'Accuracy', 'Bal Acc', 'F1 macro',
           'Recall A', 'Recall B', 'Recall C',
           'F1 A', 'F1 B', 'F1 C']
table_data = metrics_df[display_cols].values
table = ax_tab.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colColours=['#34495e'] * len(headers))
table.auto_set_font_size(False); table.set_fontsize(11)
table.scale(1, 2.2)
for k in range(len(headers)):
    table[0, k].set_text_props(color='white', fontweight='bold')
ax_tab.set_title('Full Metrics Summary', fontsize=13, fontweight='bold', pad=10)

fig.suptitle('Random Forest vs XGBoost — doctors_aggregated.csv',
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('rf_vs_xgb_with_metrics.png', dpi=150, bbox_inches='tight')
plt.show()

print('\nSaved: rf_vs_xgb_with_metrics.png, metrics_aggregated.csv')