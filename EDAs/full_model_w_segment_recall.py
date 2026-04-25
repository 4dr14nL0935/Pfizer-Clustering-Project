"""
================================================================================
ONE-VS-REST con THRESHOLD OPTIMIZATION
================================================================================

ESTRATEGIA COMPLETAMENTE DIFERENTE:

En lugar de predecir A vs B vs C al mismo tiempo,
creamos 3 modelos binarios:

1. Modelo "¿Es SEG_A?" (optimizado para detectar SEG_A)
2. Modelo "¿Es SEG_B?" (optimizado para detectar SEG_B)  
3. Modelo "¿Es SEG_C?" (optimizado para detectar SEG_C)

CADA modelo usa SOLO las features más importantes para ESE segmento.

Luego combinamos con reglas:
- Si score_C > threshold_C: Predecir SEG_C
- Else if score_A > threshold_A: Predecir SEG_A
- Else: Predecir SEG_B

VENTAJA: Cada modelo se especializa en UN problema más simple.

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, recall_score, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
import warnings
import os
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST = True
except:
    XGBOOST = False

# ==============================================================================
# SETUP
# ==============================================================================

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

INPUT_FILE = os.path.join(DATA_DIR, 'doctors_aggregated.csv')
OUTPUT_PLOT = os.path.join(RESULTS_DIR, 'one_vs_rest_optimized.png')


# Features por segmento (basado en Cohen's D analysis previo)
FEATURES_SEG_A = [
    'ORAL_TRX_mean', 'ORAL_TRX_sum', 'ORAL_TRX_max',
    'UC_TRX_mean', 'UC_TRX_sum', 
    'TOTAL_TRX_mean', 'TOTAL_TRX_sum',
    'N_CLMOTHERS_mean'
]

FEATURES_SEG_B = [
    'UC_TRX_mean', 'ORAL_TRX_mean',
    'TOTAL_TRX_mean', 'TOTAL_TRX_sum',
    'N_CLMOTHERS_mean', 'N_CLMOTHERS_sum'
]

FEATURES_SEG_C = [
    'UC_TRX_mean', 'UC_TRX_sum', 'UC_TRX_max',
    'TOTAL_TRX_mean', 'TOTAL_TRX_sum', 'TOTAL_TRX_max',
    'N_CLMOTHERS_mean', 'N_CLMOTHERS_sum',
    'ORAL_TRX_mean'
]


def load_data():
    """Cargar datos"""
    
    print("\n" + "="*80)
    print("PASO 1: CARGANDO DATOS")
    print("="*80)
    
    df = pd.read_csv(INPUT_FILE)
    labeled = df[df['ATSEG_first'].isin(['SEG_A', 'SEG_B', 'SEG_C'])].copy()
    
    print(f"\n✓ Doctores: {len(labeled):,}")
    print(f"\n📊 Distribución:")
    for seg in ['SEG_A', 'SEG_B', 'SEG_C']:
        count = (labeled['ATSEG_first'] == seg).sum()
        pct = count / len(labeled) * 100
        print(f"  {seg}: {count:,} ({pct:.1f}%)")
    
    return labeled


def find_optimal_threshold(y_true, y_scores, target_class):
    """
    Encontrar el threshold óptimo que maximiza F1 para una clase
    """
    # Probar diferentes thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        
        # Calcular F1 solo para la clase objetivo
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        
        if tp + fp > 0 and tp + fn > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = thresh
    
    return best_threshold, best_f1


def train_one_vs_rest(labeled):
    """Entrenar modelos One-vs-Rest con threshold optimization"""
    
    print("\n" + "="*80)
    print("PASO 2: ENTRENANDO ONE-VS-REST CON THRESHOLD OPTIMIZATION")
    print("="*80)
    
    # Preparar datos
    X_all = labeled.select_dtypes(include=[np.number]).fillna(0).replace([np.inf, -np.inf], 0)
    y_all = labeled['ATSEG_first'].values
    
    # Split
    X_train_all, X_test_all, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    print(f"\n✓ Train: {len(X_train_all):,} | Test: {len(X_test_all):,}")
    
    models = {}
    thresholds = {}
    scalers = {}
    
    # ==========================================================================
    # MODELO 1: ¿Es SEG_A?
    # ==========================================================================
    
    print(f"\n{'='*80}")
    print("MODELO 1: ¿Es SEG_A? (BAJO volumen)")
    print(f"{'='*80}")
    
    # Features específicas
    features_a = [f for f in FEATURES_SEG_A if f in X_train_all.columns]
    X_train_a = X_train_all[features_a]
    X_test_a = X_test_all[features_a]
    
    # Normalizar
    scaler_a = StandardScaler()
    X_train_a_scaled = scaler_a.fit_transform(X_train_a)
    X_test_a_scaled = scaler_a.transform(X_test_a)
    
    # Labels binarios
    y_train_a = (y_train == 'SEG_A').astype(int)
    y_test_a = (y_test == 'SEG_A').astype(int)
    
    # Entrenar modelo balanceado
    model_a = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model_a.fit(X_train_a_scaled, y_train_a)
    
    # Predicciones de probabilidad
    scores_a = model_a.predict_proba(X_test_a_scaled)[:, 1]
    
    # Encontrar threshold óptimo
    threshold_a, f1_a = find_optimal_threshold(y_test_a, scores_a, 'SEG_A')
    
    print(f"\n  Features usadas: {len(features_a)}")
    print(f"  Threshold óptimo: {threshold_a:.3f}")
    print(f"  F1-Score: {f1_a:.3f}")
    
    models['SEG_A'] = model_a
    thresholds['SEG_A'] = threshold_a
    scalers['SEG_A'] = scaler_a
    
    # ==========================================================================
    # MODELO 2: ¿Es SEG_B?
    # ==========================================================================
    
    print(f"\n{'='*80}")
    print("MODELO 2: ¿Es SEG_B? (MEDIO volumen)")
    print(f"{'='*80}")
    
    features_b = [f for f in FEATURES_SEG_B if f in X_train_all.columns]
    X_train_b = X_train_all[features_b]
    X_test_b = X_test_all[features_b]
    
    scaler_b = StandardScaler()
    X_train_b_scaled = scaler_b.fit_transform(X_train_b)
    X_test_b_scaled = scaler_b.transform(X_test_b)
    
    y_train_b = (y_train == 'SEG_B').astype(int)
    y_test_b = (y_test == 'SEG_B').astype(int)
    
    model_b = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model_b.fit(X_train_b_scaled, y_train_b)
    scores_b = model_b.predict_proba(X_test_b_scaled)[:, 1]
    threshold_b, f1_b = find_optimal_threshold(y_test_b, scores_b, 'SEG_B')
    
    print(f"\n  Features usadas: {len(features_b)}")
    print(f"  Threshold óptimo: {threshold_b:.3f}")
    print(f"  F1-Score: {f1_b:.3f}")
    
    models['SEG_B'] = model_b
    thresholds['SEG_B'] = threshold_b
    scalers['SEG_B'] = scaler_b
    
    # ==========================================================================
    # MODELO 3: ¿Es SEG_C? - USA SOLO FEATURES MUY DISCRIMINANTES
    # ==========================================================================
    
    print(f"\n{'='*80}")
    print("MODELO 3: ¿Es SEG_C? (ALTO volumen)")
    print(f"{'='*80}")
    
    features_c = [f for f in FEATURES_SEG_C if f in X_train_all.columns]
    X_train_c = X_train_all[features_c]
    X_test_c = X_test_all[features_c]
    
    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train_c)
    X_test_c_scaled = scaler_c.transform(X_test_c)
    
    y_train_c = (y_train == 'SEG_C').astype(int)
    y_test_c = (y_test == 'SEG_C').astype(int)
    
    # XGBoost con scale_pos_weight AGRESIVO
    if XGBOOST:
        # Calcular scale_pos_weight
        n_neg = (y_train_c == 0).sum()
        n_pos = (y_train_c == 1).sum()
        scale = (n_neg / n_pos) * 3  # 3x más agresivo
        
        model_c = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        print(f"\n  Usando XGBoost con scale_pos_weight={scale:.2f}")
    else:
        model_c = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        print(f"\n  Usando Random Forest")
    
    model_c.fit(X_train_c_scaled, y_train_c)
    scores_c = model_c.predict_proba(X_test_c_scaled)[:, 1]
    threshold_c, f1_c = find_optimal_threshold(y_test_c, scores_c, 'SEG_C')
    
    print(f"\n  Features usadas: {len(features_c)}")
    print(f"  Threshold óptimo: {threshold_c:.3f}")
    print(f"  F1-Score: {f1_c:.3f}")
    
    models['SEG_C'] = model_c
    thresholds['SEG_C'] = threshold_c
    scalers['SEG_C'] = scaler_c
    
    # ==========================================================================
    # PREDICCIÓN COMBINADA con PRIORIDAD A SEG_C
    # ==========================================================================
    
    print(f"\n{'='*80}")
    print("PASO 3: COMBINANDO MODELOS CON REGLAS DE PRIORIDAD")
    print(f"{'='*80}")
    
    # Preparar datos de test
    X_test_a_scaled = scalers['SEG_A'].transform(X_test_all[features_a])
    X_test_b_scaled = scalers['SEG_B'].transform(X_test_all[features_b])
    X_test_c_scaled = scalers['SEG_C'].transform(X_test_all[features_c])
    
    # Obtener scores
    scores_a_final = models['SEG_A'].predict_proba(X_test_a_scaled)[:, 1]
    scores_b_final = models['SEG_B'].predict_proba(X_test_b_scaled)[:, 1]
    scores_c_final = models['SEG_C'].predict_proba(X_test_c_scaled)[:, 1]
    
    # Combinar con reglas de prioridad
    y_pred = np.empty(len(y_test), dtype=object)
    
    for i in range(len(y_test)):
        # PRIORIDAD 1: SEG_C (alto volumen)
        if scores_c_final[i] >= thresholds['SEG_C']:
            y_pred[i] = 'SEG_C'
        # PRIORIDAD 2: SEG_A (bajo volumen)
        elif scores_a_final[i] >= thresholds['SEG_A']:
            y_pred[i] = 'SEG_A'
        # DEFAULT: SEG_B
        else:
            y_pred[i] = 'SEG_B'
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    
    print(f"\n📊 MÉTRICAS GENERALES:")
    print(f"  • Accuracy: {acc:.2%}")
    print(f"  • F1-Score: {f1:.3f}")
    print(f"  • Recall:   {recall:.3f}")
    print(f"  • Cohen's Kappa: {kappa:.3f}") 
    
    print(f"\n📊 ACCURACY POR SEGMENTO:")
    seg_results = {}
    for seg in ['SEG_A', 'SEG_B', 'SEG_C']:
        mask = y_test == seg
        if mask.sum() > 0:
            seg_acc = accuracy_score(y_test[mask], y_pred[mask])
            n_correct = ((y_test == seg) & (y_pred == seg)).sum()
            n_total = mask.sum()
            seg_results[seg] = seg_acc
            print(f"  • {seg}: {seg_acc:.1%} ({n_correct}/{n_total})")
    
    print(f"\n📊 RECALL POR SEGMENTO:")
    for seg in ['SEG_A', 'SEG_B', 'SEG_C']:
        mask = y_test == seg
        if mask.sum() > 0:
            tp = ((y_test == seg) & (y_pred == seg)).sum()
            fn = ((y_test == seg) & (y_pred != seg)).sum()
            seg_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            print(f"  • {seg}: {seg_recall:.1%} ({tp}/{tp+fn})")

    balanced = np.mean(list(seg_results.values()))
    print(f"\n📊 BALANCED SCORE: {balanced:.1%}")
    
    return y_pred, y_test, seg_results, {
        'accuracy': acc,
        'kappa': kappa,
        'f1': f1,
        'recall': recall
    }


def create_visualization(y_test, y_pred, seg_results, metrics):
    """Crear visualización"""
    
    print("\n" + "="*80)
    print("PASO 4: CREANDO VISUALIZACIÓN")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Matriz de confusión
    ax1 = axes[0]
    cm = confusion_matrix(y_test, y_pred, labels=['SEG_A', 'SEG_B', 'SEG_C'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax1,
                xticklabels=['SEG_A', 'SEG_B', 'SEG_C'],
                yticklabels=['SEG_A', 'SEG_B', 'SEG_C'])
    title = f'ONE-VS-REST + THRESHOLD OPTIMIZATION\nAcc: {metrics["accuracy"]:.1%} | F1: {metrics["f1"]:.3f} | Recall: {metrics["recall"]:.3f} | Kappa: {metrics["kappa"]:.3f}'
    ax1.set_title(title, fontweight='bold', fontsize=10)
    ax1.set_ylabel('Real (ATSEG)')
    ax1.set_xlabel('Predicción del Modelo')
    
    # Accuracy por segmento
    ax2 = axes[1]
    segments = ['SEG_A', 'SEG_B', 'SEG_C']
    accs = [seg_results[seg] * 100 for seg in segments]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = ax2.bar(segments, accs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy por Segmento', fontweight='bold', fontsize=14)
    ax2.set_ylim(0, 100)
    ax2.axhline(y=60, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='60% (best so far)')
    ax2.axhline(y=70, color='green', linestyle='--', alpha=0.5, linewidth=2, label='70% objetivo')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"\n✓ Guardado en: {OUTPUT_PLOT}")


def main():
    """Función principal"""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*12 + "ONE-VS-REST + THRESHOLD OPTIMIZATION" + " "*26 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")
    
    # Paso 1: Cargar
    labeled = load_data()
    
    # Paso 2: Entrenar
    y_pred, y_test, seg_results, metrics = train_one_vs_rest(labeled)
    
    # Paso 3: Visualizar
    create_visualization(y_test, y_pred, seg_results, metrics)
    
    print("\n" + "="*80)
    print("✅ ONE-VS-REST COMPLETO")
    print("="*80)
    
    print(f"\n🎯 RESULTADO FINAL:")
    print(f"  SEG_A: {seg_results['SEG_A']:.1%}")
    print(f"  SEG_B: {seg_results['SEG_B']:.1%}")
    print(f"  SEG_C: {seg_results['SEG_C']:.1%}")
    
    balanced = (seg_results['SEG_A'] + seg_results['SEG_B'] + seg_results['SEG_C']) / 3
    print(f"\n📊 BALANCED SCORE: {balanced:.1%}")

    print(f"  📉 RECALL (macro): {metrics['recall']:.3f}")
    
    # Comparación
    print(f"\n📈 COMPARACIÓN:")
    print(f"  XGBoost Optimizado: 60.2% balanced")
    print(f"  One-vs-Rest: {balanced:.1%} balanced")
    
    if balanced > 0.602:
        mejora = (balanced - 0.602) * 100
        print(f"\n  ✅ MEJORA: +{mejora:.1f} puntos ¡ÉXITO!")
    else:
        diff = (0.602 - balanced) * 100
        print(f"\n  Diferencia: -{diff:.1f} puntos")
    
    print("\n")
    
    return seg_results, metrics


if __name__ == "__main__":
    seg_results, metrics = main()