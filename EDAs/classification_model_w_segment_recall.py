"""
================================================================================
🏆 MEJOR MODELO - XGBoost Optimizado (60.2% Balanced Accuracy)
================================================================================

Este es el modelo con MEJOR rendimiento después de probar 10+ técnicas diferentes.

RESULTADO:
  SEG_A: 78.1%
  SEG_B: 50.3%
  SEG_C: 52.2%
  BALANCED: 60.2%

HIPERPARÁMETROS OPTIMIZADOS:
  n_estimators: 400
  max_depth: 3
  learning_rate: 0.06
  min_child_weight: 7
  subsample: 0.9
  colsample_bytree: 0.6
  gamma: 0.5
  reg_alpha: 0.1
  reg_lambda: 1
  sample_weights: SEG_A=1.0, SEG_B=1.5, SEG_C=2.5

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, recall_score, confusion_matrix
import warnings
import os
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST = True
except:
    XGBOOST = False
    print("❌ ERROR: XGBoost no disponible")
    print("   Instala con: pip install xgboost")
    exit()

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

INPUT_FILE = os.path.join(DATA_DIR, 'doctors_aggregated.csv')
OUTPUT_PLOT = os.path.join(RESULTS_DIR, 'best_model_60_2.png')

def load_data():
    """Cargar y preparar datos"""
    
    print("\n" + "="*80)
    print("PASO 1: CARGANDO DATOS")
    print("="*80)
    
    df = pd.read_csv(INPUT_FILE)
    
    # CRÍTICO: Limpiar nombres de columnas para XGBoost
    print(f"\n🔧 Limpiando nombres de columnas...")
    rename_map = {}
    for col in df.columns:
        clean_col = col
        clean_col = clean_col.replace('[', '_')
        clean_col = clean_col.replace(']', '_')
        clean_col = clean_col.replace('<', '_lt_')
        clean_col = clean_col.replace('>', '_gt_')
        
        while '__' in clean_col:
            clean_col = clean_col.replace('__', '_')
        
        clean_col = clean_col.rstrip('_')
        
        if clean_col != col:
            rename_map[col] = clean_col
    
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"   ✓ {len(rename_map)} columnas renombradas para XGBoost")
    
    labeled = df[df['ATSEG_first'].isin(['SEG_A', 'SEG_B', 'SEG_C'])].copy()
    
    print(f"\n✓ Doctores totales: {len(labeled):,}")
    
    # Distribución
    print(f"\n📊 Distribución de clases:")
    for seg in ['SEG_A', 'SEG_B', 'SEG_C']:
        count = (labeled['ATSEG_first'] == seg).sum()
        pct = count / len(labeled) * 100
        print(f"  {seg}: {count:,} ({pct:.1f}%)")
    
    # Seleccionar features numéricas
    exclude_cols = ['NUEVO_ID', 'ATSEG_first', 'WEEK_ID_first', 'WEEK_ID_last', 
                    'WEEK_ID_count', 'TERRITORY']
    numeric_cols = labeled.select_dtypes(include=[np.number]).columns.tolist()
    features = [col for col in numeric_cols if col not in exclude_cols]
    
    # Preparar X, y
    X = labeled[features].fillna(0).replace([np.inf, -np.inf], 0)
    y = labeled['ATSEG_first']
    
    print(f"✓ Features: {len(features)}")
    
    return X, y


def train_best_model(X, y):
    """Entrenar el mejor modelo (XGBoost Optimizado)"""
    
    print("\n" + "="*80)
    print("PASO 2: ENTRENANDO XGBoost OPTIMIZADO")
    print("="*80)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split con estratificación
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n✓ Train: {len(X_train):,} doctores")
    print(f"✓ Test:  {len(X_test):,} doctores")
    
    # Sample weights para balancear clases
    print(f"\n🔧 Configurando sample weights...")
    sample_weights = np.ones(len(y_train))
    sample_weights[y_train == 0] = 1.0   # SEG_A
    sample_weights[y_train == 1] = 1.5   # SEG_B
    sample_weights[y_train == 2] = 2.5   # SEG_C
    
    print(f"  SEG_A: 1.0")
    print(f"  SEG_B: 1.5")
    print(f"  SEG_C: 2.5")
    
    # Crear modelo con hiperparámetros optimizados
    print(f"\n🔧 Entrenando XGBoost con hiperparámetros optimizados...")
    
    model = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.06,
        min_child_weight=7,
        subsample=0.9,
        colsample_bytree=0.6,
        gamma=0.5,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0
    )
    
    # Entrenar
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    print(f"  ✓ Modelo entrenado exitosamente")
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Decode labels
    y_pred_decoded = le.inverse_transform(y_pred)
    y_test_decoded = le.inverse_transform(y_test)
    
    # Calcular métricas
    acc = accuracy_score(y_test_decoded, y_pred_decoded)
    kappa = cohen_kappa_score(y_test_decoded, y_pred_decoded)
    f1 = f1_score(y_test_decoded, y_pred_decoded, average='macro')
    recall = recall_score(y_test_decoded, y_pred_decoded, average='macro')
    
    print(f"\n" + "="*80)
    print("PASO 3: RESULTADOS")
    print("="*80)
    
    print(f"\n📊 MÉTRICAS GENERALES:")
    print(f"  • Accuracy:      {acc:.2%}")
    print(f"  • F1-Score:      {f1:.3f}")
    print(f"  • Recall:        {recall:.3f}")
    print(f"  • Cohen's Kappa: {kappa:.3f}")
    
    print(f"\n📊 ACCURACY POR SEGMENTO:")
    seg_results = {}
    for seg in ['SEG_A', 'SEG_B', 'SEG_C']:
        mask = y_test_decoded == seg
        if mask.sum() > 0:
            seg_acc = accuracy_score(y_test_decoded[mask], y_pred_decoded[mask])
            n_correct = ((y_test_decoded == seg) & (y_pred_decoded == seg)).sum()
            n_total = mask.sum()
            seg_results[seg] = seg_acc
            print(f"  • {seg}: {seg_acc:.1%} ({n_correct}/{n_total})")
    
    print(f"\n📊 RECALL POR SEGMENTO:")
    for seg in ['SEG_A', 'SEG_B', 'SEG_C']:
        mask = y_test_decoded == seg
        if mask.sum() > 0:
            true_positive = ((y_test_decoded == seg) & (y_pred_decoded == seg)).sum()
            actual_total = (y_test_decoded == seg).sum()
            seg_recall = true_positive / actual_total
            print(f"  • {seg}: {seg_recall:.1%} ({true_positive}/{actual_total})")
    
    balanced = np.mean(list(seg_results.values()))
    print(f"\n🎯 BALANCED ACCURACY: {balanced:.1%}")
    
    # Feature importance (top 20)
    print(f"\n📊 TOP 20 FEATURES MÁS IMPORTANTES:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    for idx, row in feature_importance.iterrows():
        print(f"  {row['feature']:<40} {row['importance']:.4f}")
    
    return y_pred_decoded, y_test_decoded, seg_results, {
        'accuracy': acc,
        'kappa': kappa,
        'f1': f1,
        'recall': recall
    }, model

def create_visualization(y_test, y_pred, seg_results, metrics):
    """Crear visualización de resultados"""
    
    print(f"\n" + "="*80)
    print("PASO 4: CREANDO VISUALIZACIÓN")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Matriz de confusión
    ax1 = axes[0]
    cm = confusion_matrix(y_test, y_pred, labels=['SEG_A', 'SEG_B', 'SEG_C'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax1,
                xticklabels=['SEG_A', 'SEG_B', 'SEG_C'],
                yticklabels=['SEG_A', 'SEG_B', 'SEG_C'],
                cbar_kws={'label': 'Número de Doctores'})
    
    balanced = (seg_results['SEG_A'] + seg_results['SEG_B'] + seg_results['SEG_C']) / 3
    title = f'🏆 MEJOR MODELO - XGBoost Optimizado\nBalanced Acc: {balanced:.1%} | Overall Acc: {metrics["accuracy"]:.1%} | Kappa: {metrics["kappa"]:.3f}'
    ax1.set_title(title, fontweight='bold', fontsize=11, pad=15)
    ax1.set_ylabel('Segmento Real (ATSEG)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Predicción del Modelo', fontsize=11, fontweight='bold')
    
    # Accuracy por segmento
    ax2 = axes[1]
    segments = ['SEG_A', 'SEG_B', 'SEG_C']
    accs = [seg_results[seg] * 100 for seg in segments]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = ax2.bar(segments, accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy por Segmento', fontweight='bold', fontsize=14, pad=15)
    ax2.set_ylim(0, 100)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='50% baseline')
    ax2.axhline(y=balanced, color='green', linestyle='--', alpha=0.7, linewidth=2, 
                label=f'Balanced: {balanced:.1f}%')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)
    
    # Etiquetas en barras
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualización guardada: {OUTPUT_PLOT}")


def save_model(model, filename='best_model_xgboost.json'):
    """Guardar modelo entrenado"""
    
    output_path = os.path.join(RESULTS_DIR, filename)
    model.save_model(output_path)
    print(f"✓ Modelo guardado: {output_path}")


def main():
    """Función principal"""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "🏆 MEJOR MODELO - XGBoost Optimizado 🏆" + " "*21 + "║")
    print("╚" + "="*78 + "╝")
    
    # Cargar datos
    X, y = load_data()
    
    # Entrenar modelo
    y_pred, y_test, seg_results, metrics, model = train_best_model(X, y)
    
    # Crear visualización
    create_visualization(y_test, y_pred, seg_results, metrics)
    
    # Guardar modelo
    print(f"\n" + "="*80)
    print("PASO 5: GUARDANDO MODELO")
    print("="*80 + "\n")
    save_model(model)
    
    # Resumen final
    balanced = (seg_results['SEG_A'] + seg_results['SEG_B'] + seg_results['SEG_C']) / 3
    
    print("\n" + "="*80)
    print("✅ PROCESO COMPLETO")
    print("="*80)
    
    print(f"\n🎯 RESULTADO FINAL:")
    print(f"  {'Segmento':<10} {'Accuracy':<12} {'Doctores Correctos'}")
    print(f"  {'-'*50}")
    for seg in ['SEG_A', 'SEG_B', 'SEG_C']:
        mask = y_test == seg
        n_correct = ((y_test == seg) & (y_pred == seg)).sum()
        n_total = mask.sum()
        print(f"  {seg:<10} {seg_results[seg]:<11.1%} {n_correct}/{n_total}")
    
    print(f"\n  🎯 BALANCED ACCURACY: {balanced:.1%}")
    print(f"  📊 OVERALL ACCURACY:  {metrics['accuracy']:.1%}")
    print(f"  📈 F1-SCORE (macro):  {metrics['f1']:.3f}")
    print(f"  🔢 COHEN'S KAPPA:     {metrics['kappa']:.3f}")
    print(f"  📉 RECALL (macro):    {metrics['recall']:.3f}")
    
    print(f"\n📁 ARCHIVOS GENERADOS:")
    print(f"  • Visualización: {OUTPUT_PLOT}")
    print(f"  • Modelo:        {os.path.join(RESULTS_DIR, 'best_model_xgboost.json')}")
    
    print("\n" + "="*80)
    print("🏆 Este es el MEJOR modelo después de probar 10+ técnicas diferentes")
    print("="*80 + "\n")
    
    return seg_results, metrics, model


if __name__ == "__main__":
    seg_results, metrics, model = main()