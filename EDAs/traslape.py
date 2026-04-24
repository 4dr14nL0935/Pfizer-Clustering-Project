"""
================================================================================
VISUALIZACIONES DE OVERLAP — versión corregida (pandas 3.0 compatible)
================================================================================
Fix: el bug 'KeyError: seg' ocurre porque en pandas 3.0+ groupby().apply()
elimina la columna de agrupación del resultado. Reemplazado por un sampling
manual por clase que es más robusto.

USO:
  python generate_overlap_visualizations.py
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Intentar XGBoost; si no está, usar sklearn como fallback
try:
    from xgboost import XGBClassifier
    USE_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    USE_XGBOOST = False
    print('⚠ xgboost no disponible, usando GradientBoostingClassifier')

# ============================================================================
# CONFIG
# ============================================================================
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

INPUT_FILE = os.path.join(DATA_DIR, 'doctors_aggregated.csv')

COLORS = {'SEG_A': '#2ecc71', 'SEG_B': '#3498db', 'SEG_C': '#e74c3c'}
SEG_ORDER = ['SEG_A', 'SEG_B', 'SEG_C']


def balanced_sample(df, class_col, n_per_class=1500, seed=42):
    """Sampling balanceado robusto (reemplaza el groupby.apply problemático)."""
    parts = []
    for cls in df[class_col].unique():
        sub = df[df[class_col] == cls]
        n = min(n_per_class, len(sub))
        parts.append(sub.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def _feature_family(name):
    """
    Extrae la 'familia' de una feature removiendo el sufijo de agregación.
    Ej: 'ORAL_TRX_mean' -> 'ORAL_TRX', 'ORAL_TRX_max' -> 'ORAL_TRX'
    """
    for suffix in ['_sum', '_mean', '_max', '_min', '_std', '_first',
                   '_last', '_count', '_median']:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def select_top_uncorrelated(X, importance_series, n=2, max_corr=0.6,
                             min_unique=20, distinct_families=True):
    """
    Selecciona las top-N features por importancia con estos filtros:
      1. Continuas (n_unique >= min_unique) — descarta binarias
      2. De familias distintas (ej. no elegir ORAL_TRX_mean Y ORAL_TRX_max)
      3. Correlación absoluta entre ellas < max_corr
    """
    selected = []
    selected_families = set()
    for feat in importance_series.index:
        if feat not in X.columns:
            continue
        if X[feat].nunique() < min_unique:
            continue
        fam = _feature_family(feat)
        if distinct_families and fam in selected_families:
            continue
        if len(selected) > 0:
            try:
                corrs = [abs(X[feat].corr(X[s])) for s in selected]
                corrs = [c if not np.isnan(c) else 0 for c in corrs]
                if max(corrs) >= max_corr:
                    continue
            except Exception:
                continue
        selected.append(feat)
        selected_families.add(fam)
        if len(selected) >= n:
            break
    return selected


# ============================================================================
# 1. CARGA + MODELO
# ============================================================================
def load_and_train():
    print('Cargando datos...')
    df = pd.read_csv(INPUT_FILE)

    rename_map = {}
    for col in df.columns:
        clean = col
        clean = clean.replace('[', '_').replace(']', '_')
        clean = clean.replace('<', '_lt_').replace('>', '_gt_')
        while '__' in clean:
            clean = clean.replace('__', '_')
        clean = clean.rstrip('_')
        if clean != col:
            rename_map[col] = clean
    df = df.rename(columns=rename_map)

    labeled = df[df['ATSEG_first'].isin(SEG_ORDER)].copy()

    exclude = ['NUEVO_ID', 'ATSEG_first', 'WEEK_ID_first', 'WEEK_ID_last',
               'WEEK_ID_count', 'TERRITORY']
    num_cols = labeled.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in num_cols if c not in exclude]

    X = labeled[features].fillna(0).replace([np.inf, -np.inf], 0).reset_index(drop=True)
    y = labeled['ATSEG_first'].reset_index(drop=True)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    sw = np.ones(len(y_tr))
    sw[y_tr == 1] = 1.5
    sw[y_tr == 2] = 2.5

    print(f'Entrenando modelo ({"XGBoost" if USE_XGBOOST else "GradientBoosting"})...')
    if USE_XGBOOST:
        model = XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.06,
            min_child_weight=7, subsample=0.9, colsample_bytree=0.6,
            gamma=0.5, reg_alpha=0.1, reg_lambda=1,
            random_state=42, eval_metric='mlogloss', verbosity=0
        )
        model.fit(X_tr, y_tr, sample_weight=sw)
    else:
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.06,
            subsample=0.9, random_state=42
        )
        model.fit(X_tr, y_tr, sample_weight=sw)

    print(f'  HCPs: {len(X):,}   Features: {len(features)}')
    return X, y, X_te, y_te, model, le, features


# ============================================================================
# VIZ 1: SCATTER + CONTORNOS KDE
# ============================================================================
def viz1_scatter_overlap(X, y, model, features, output_path):
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    # FIX: top-2 no correlacionadas (evita que salgan mean y sum de la misma columna)
    top2 = select_top_uncorrelated(X, imp, n=2, max_corr=0.7)
    f1, f2 = top2[0], top2[1]

    df_plot = pd.DataFrame({
        f1: X[f1].values,
        f2: X[f2].values,
        'seg': y.values
    })

    # FIX: sampling manual balanceado (antes era groupby.apply que rompe en pandas 3.0)
    if len(df_plot) > 4500:
        df_plot = balanced_sample(df_plot, 'seg', n_per_class=1500)

    fig, ax = plt.subplots(figsize=(11, 8))

    for seg in SEG_ORDER:
        sub = df_plot[df_plot['seg'] == seg]
        n_full = (y == seg).sum()
        ax.scatter(sub[f1], sub[f2], c=COLORS[seg], alpha=0.25, s=20,
                   edgecolors='none', label=f'{seg} (n={n_full:,})')

    for seg in SEG_ORDER:
        sub = df_plot[df_plot['seg'] == seg]
        if len(sub) < 20:
            continue
        try:
            sns.kdeplot(x=sub[f1].values, y=sub[f2].values, ax=ax, color=COLORS[seg],
                        levels=[0.2, 0.5], linewidths=2.5, alpha=0.95)
        except Exception as e:
            print(f'  ⚠ KDE falló para {seg}: {e}')

    x_lo = np.percentile(df_plot[f1], 1)
    x_hi = np.percentile(df_plot[f1], 99)
    y_lo = np.percentile(df_plot[f2], 1)
    y_hi = np.percentile(df_plot[f2], 99)
    pad_x = max((x_hi - x_lo) * 0.05, 0.01)
    pad_y = max((y_hi - y_lo) * 0.05, 0.01)
    ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
    ax.set_ylim(y_lo - pad_y, y_hi + pad_y)

    ax.set_xlabel(f1, fontsize=12, fontweight='bold')
    ax.set_ylabel(f2, fontsize=12, fontweight='bold')
    ax.set_title(
        'Overlap de segmentos en las 2 features más importantes\n'
        'Contornos: 50% y 80% de la densidad de cada clase',
        fontsize=13, fontweight='bold', pad=15
    )
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓ Guardado: {output_path}  [features: {f1}, {f2}]')
    return f1, f2


# ============================================================================
# VIZ 2: GRID 2x3 DE VIOLINES
# ============================================================================
def viz2_violin_grid(X, y, model, features, output_path):
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    # Top-6 no correlacionadas (evita duplicados tipo mean/sum de la misma columna)
    top6 = select_top_uncorrelated(X, imp, n=6, max_corr=0.7)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for ax, feat in zip(axes.flatten(), top6):
        data_plot = pd.DataFrame({feat: X[feat].values, 'seg': y.values})
        p98 = np.percentile(data_plot[feat], 98)
        if p98 > data_plot[feat].min():
            data_plot.loc[data_plot[feat] > p98, feat] = p98

        sns.violinplot(data=data_plot, x='seg', y=feat, ax=ax,
                       order=SEG_ORDER,
                       palette=[COLORS[s] for s in SEG_ORDER],
                       inner='quartile', linewidth=1.2, cut=0,
                       hue='seg', legend=False)
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('valor (clip p98)', fontsize=9)
        ax.grid(axis='y', alpha=0.2)

    fig.suptitle(
        'Ninguna feature individual separa las 3 clases\n'
        'Top 6 features por importancia del modelo',
        fontsize=14, fontweight='bold', y=1.00
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓ Guardado: {output_path}')


# ============================================================================
# VIZ 3: HISTOGRAMA DE CONFIANZA
# ============================================================================
def viz3_confidence_hist(X_te, y_te, model, output_path):
    proba = model.predict_proba(X_te)
    y_pred = model.predict(X_te)
    max_p = proba.max(axis=1)
    correct = (y_pred == y_te)

    fig, ax = plt.subplots(figsize=(11, 6))

    bins = np.linspace(0.33, 1.0, 35)
    ax.hist(max_p[correct], bins=bins, alpha=0.75, color='#27ae60',
            label=f'Aciertos (n={correct.sum():,})',
            edgecolor='white', linewidth=0.5)
    ax.hist(max_p[~correct], bins=bins, alpha=0.75, color='#c0392b',
            label=f'Errores (n={(~correct).sum():,})',
            edgecolor='white', linewidth=0.5)

    ax.axvspan(0.33, 0.45, alpha=0.08, color='red')
    ax.axvspan(0.45, 0.65, alpha=0.08, color='orange')
    ax.axvspan(0.65, 1.00, alpha=0.08, color='green')

    for thr in [0.45, 0.65]:
        ax.axvline(thr, color='black', linestyle='--', alpha=0.6, linewidth=1.5)

    ymax = ax.get_ylim()[1]
    ax.text(0.39, ymax * 0.92, 'Escalar\n(baja conf.)', ha='center', fontsize=10,
            fontweight='bold', color='#8B0000')
    ax.text(0.55, ymax * 0.92, 'Revisar\n(media conf.)', ha='center', fontsize=10,
            fontweight='bold', color='#D35400')
    ax.text(0.82, ymax * 0.92, 'Auto-asignar\n(alta conf.)', ha='center', fontsize=10,
            fontweight='bold', color='#1E8449')

    ax.set_xlabel('Confianza del modelo (probabilidad máxima)',
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Número de HCPs', fontsize=12, fontweight='bold')
    ax.set_title(
        'Los errores se concentran en baja confianza → estrategia de escalamiento\n'
        'El modelo "sabe" cuándo duda: base de confidence thresholds',
        fontsize=13, fontweight='bold', pad=12
    )
    ax.legend(fontsize=11, loc='center left', bbox_to_anchor=(1.01, 0.5),
              framealpha=0.95)
    ax.grid(alpha=0.2)

    high = max_p > 0.65
    mid = (max_p > 0.45) & (max_p <= 0.65)
    low = max_p <= 0.45

    print('\n--- Métricas por zona de confianza (para tu slide) ---')
    print(f'{"Zona":<22} {"Cobertura":>10} {"Accuracy":>10}')
    for name, mask in [('ALTA (>0.65)', high),
                       ('MEDIA (0.45-0.65)', mid),
                       ('BAJA (<=0.45)', low)]:
        if mask.sum() > 0:
            acc = correct[mask].mean()
            pct = mask.mean() * 100
            print(f'  {name:<20} {pct:>8.1f}% {acc:>9.1%}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓ Guardado: {output_path}')


# ============================================================================
# MÉTRICAS DE OVERLAP
# ============================================================================
def overlap_metrics(X, y, model, features):
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    top20 = imp.index[:20].tolist()

    Xs = X[top20].copy()
    Xs = (Xs - Xs.mean()) / Xs.std().replace(0, 1)

    centroids = {s: Xs[y == s].mean().values for s in SEG_ORDER}
    stds = {s: Xs[y == s].std().mean() for s in SEG_ORDER}

    print('\n--- Distancia entre centroides (top-20 features estandarizadas) ---')
    print('Regla informal: d/σ_pooled < 2 ⇒ clases MUY solapadas')
    print(f'{"Par":<20} {"Distancia":>12} {"d/σ_pooled":>14}')
    for a, b in [('SEG_A', 'SEG_B'), ('SEG_A', 'SEG_C'), ('SEG_B', 'SEG_C')]:
        d = np.linalg.norm(centroids[a] - centroids[b])
        pooled_std = (stds[a] + stds[b]) / 2
        d_norm = d / pooled_std if pooled_std > 0 else np.nan
        print(f'  {a} <-> {b:<10} {d:>10.2f}   {d_norm:>12.2f}')


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print('=' * 72)
    print(' VISUALIZACIONES DE OVERLAP PARA PRESENTACIÓN CAPSTONE')
    print('=' * 72)

    X, y, X_te, y_te, model, le, features = load_and_train()

    print('\n[1/3] Scatter + contornos de densidad...')
    viz1_scatter_overlap(X, y, model, features,
                         os.path.join(RESULTS_DIR, 'viz1_overlap_scatter.png'))

    print('\n[2/3] Violin grid de top-6 features...')
    viz2_violin_grid(X, y, model, features,
                     os.path.join(RESULTS_DIR, 'viz2_violin_grid.png'))

    print('\n[3/3] Histograma de confianza (aciertos vs errores)...')
    viz3_confidence_hist(X_te, y_te, model,
                         os.path.join(RESULTS_DIR, 'viz3_confidence_hist.png'))

    overlap_metrics(X, y, model, features)

    print('\n' + '=' * 72)
    print(' ✅ LISTO — 3 PNGs en /results + métricas en consola')
    print('=' * 72)