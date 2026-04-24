"""
================================================================================
ANÁLISIS DE CALIDAD DE ATSEG: ¿Los errores del modelo tienen sentido?
================================================================================

OBJETIVO:
Identificar doctores que el modelo SIEMPRE clasifica mal.
Si el modelo consistentemente predice algo diferente de ATSEG,
puede que ATSEG tenga errores.

ANÁLISIS:
1. Entrenar modelo con K-Fold Cross-Validation
2. Identificar doctores que SIEMPRE se clasifican mal
3. Analizar sus características
4. Ver si tienen más sentido en la clase predicha

================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import warnings
import os
warnings.filterwarnings('ignore')

# ==============================================================================
# SETUP
# ==============================================================================

BASE_DIR = os.getcwd()
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

INPUT_FILE_GEO = os.path.join(PROCESSED_DIR, 'doctors_aggregated_with_geo.csv')
INPUT_FILE = os.path.join(PROCESSED_DIR, 'doctors_aggregated.csv')

if os.path.exists(INPUT_FILE_GEO):
    INPUT = INPUT_FILE_GEO
else:
    INPUT = INPUT_FILE

OUTPUT_CSV = os.path.join(RESULTS_DIR, 'atseg_quality_analysis.csv')


def load_data():
    """Cargar datos"""
    
    print("\n" + "="*80)
    print("CARGANDO DATOS")
    print("="*80)
    
    df = pd.read_csv(INPUT)
    labeled = df[df['ATSEG_first'].isin(['SEG_A', 'SEG_B', 'SEG_C'])].copy()
    
    print(f"\n✓ Doctores: {len(labeled):,}")
    
    # Features
    exclude_cols = ['NUEVO_ID', 'ATSEG_first', 'WEEK_ID_first', 'WEEK_ID_last', 
                    'WEEK_ID_count', 'TERRITORY']
    numeric_cols = labeled.select_dtypes(include=[np.number]).columns.tolist()
    features = [col for col in numeric_cols if col not in exclude_cols]
    
    X = labeled[features].fillna(0).replace([np.inf, -np.inf], 0)
    y = labeled['ATSEG_first']
    doctor_ids = labeled['NUEVO_ID'].values
    
    # Encode
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le, doctor_ids, labeled


def cross_validate_predictions(X, y, doctor_ids, n_splits=5):
    """K-Fold CV para obtener predicciones de TODOS los doctores"""
    
    print("\n" + "="*80)
    print(f"CROSS-VALIDATION ({n_splits}-FOLD)")
    print("="*80)
    
    # Inicializar arrays
    all_predictions = np.zeros(len(y))
    all_probabilities = np.zeros((len(y), 3))
    prediction_counts = np.zeros(len(y))
    
    # K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n⏳ Fold {fold}/{n_splits}...")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Normalizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Sample weights
        sample_weights = np.ones(len(y_train))
        sample_weights[y_train == 2] = 2.5
        sample_weights[y_train == 1] = 1.5
        
        # Modelo optimizado
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
            eval_metric='mlogloss'
        )
        
        model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        
        # Predicciones
        y_pred = model.predict(X_val_scaled)
        y_proba = model.predict_proba(X_val_scaled)
        
        # Guardar
        all_predictions[val_idx] = y_pred
        all_probabilities[val_idx] = y_proba
        prediction_counts[val_idx] += 1
    
    print("\n✓ Cross-validation completo")
    
    return all_predictions, all_probabilities


def analyze_misclassifications(y_true, y_pred, y_proba, doctor_ids, le, df):
    """Analizar doctores mal clasificados"""
    
    print("\n" + "="*80)
    print("ANÁLISIS DE CLASIFICACIONES INCORRECTAS")
    print("="*80)
    
    # Identificar errores
    misclassified_mask = y_true != y_pred
    
    print(f"\n📊 RESUMEN:")
    print(f"  Total doctores: {len(y_true):,}")
    print(f"  Correctos: {(~misclassified_mask).sum():,} ({(~misclassified_mask).mean():.1%})")
    print(f"  Incorrectos: {misclassified_mask.sum():,} ({misclassified_mask.mean():.1%})")
    
    # Análisis por segmento
    print(f"\n📊 ERRORES POR SEGMENTO REAL (ATSEG):")
    for seg_idx, seg_name in enumerate(['SEG_A', 'SEG_B', 'SEG_C']):
        seg_mask = y_true == seg_idx
        seg_errors = misclassified_mask & seg_mask
        
        if seg_mask.sum() > 0:
            error_rate = seg_errors.sum() / seg_mask.sum()
            print(f"  {seg_name}: {seg_errors.sum():,}/{seg_mask.sum():,} ({error_rate:.1%} error)")
    
    # Crear DataFrame de errores
    misclass_df = pd.DataFrame({
        'NUEVO_ID': doctor_ids[misclassified_mask],
        'ATSEG': le.inverse_transform(y_true[misclassified_mask]),
        'Predicted': le.inverse_transform(y_pred[misclassified_mask].astype(int)),
        'Prob_SEG_A': y_proba[misclassified_mask, 0],
        'Prob_SEG_B': y_proba[misclassified_mask, 1],
        'Prob_SEG_C': y_proba[misclassified_mask, 2],
        'Max_Prob': y_proba[misclassified_mask].max(axis=1)
    })
    
    # Unir con features
    misclass_df = misclass_df.merge(
        df[['NUEVO_ID', 'ORAL_TRX_mean', 'UC_TRX_mean', 'TOTAL_TRX_mean', 
            'N_CLMOTHERS_mean', 'IL23_TRX_mean']],
        on='NUEVO_ID',
        how='left'
    )
    
    # Ordenar por confianza (los más seguros que están mal)
    misclass_df = misclass_df.sort_values('Max_Prob', ascending=False)
    
    print(f"\n📊 TOP 20 DOCTORES MAL CLASIFICADOS (Alta Confianza):")
    print("-" * 130)
    print(f"{'NUEVO_ID':<12} {'ATSEG':<10} {'Predicted':<12} {'Confidence':<12} {'ORAL_TRX':<12} {'UC_TRX':<12} {'OTHERS':<12}")
    print("-" * 130)
    
    for _, row in misclass_df.head(20).iterrows():
        print(f"{row['NUEVO_ID']:<12} {row['ATSEG']:<10} {row['Predicted']:<12} "
              f"{row['Max_Prob']:<12.1%} {row['ORAL_TRX_mean']:<12.2f} "
              f"{row['UC_TRX_mean']:<12.2f} {row['N_CLMOTHERS_mean']:<12.2f}")
    
    # Guardar CSV completo
    misclass_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Lista completa guardada en: {OUTPUT_CSV}")
    
    # Estadísticas
    print(f"\n📊 CONFUSIONES MÁS COMUNES:")
    confusion_summary = misclass_df.groupby(['ATSEG', 'Predicted']).size().reset_index(name='Count')
    confusion_summary = confusion_summary.sort_values('Count', ascending=False)
    
    for _, row in confusion_summary.iterrows():
        pct = row['Count'] / misclassified_mask.sum() * 100
        print(f"  {row['ATSEG']} → {row['Predicted']}: {row['Count']:,} casos ({pct:.1f}% de errores)")
    
    return misclass_df


def main():
    """Función principal"""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "ANÁLISIS DE CALIDAD DE ATSEG" + " "*28 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")
    
    # Cargar datos
    X, y, le, doctor_ids, df = load_data()
    
    # Cross-validation
    y_pred, y_proba = cross_validate_predictions(X, y, doctor_ids, n_splits=5)
    
    # Analizar errores
    misclass_df = analyze_misclassifications(y, y_pred, y_proba, doctor_ids, le, df)
    
    print("\n" + "="*80)
    print("✅ ANÁLISIS COMPLETO")
    print("="*80)
    
    print("\n💡 INTERPRETACIÓN:")
    print("  • Si hay doctores con >80% confianza mal clasificados:")
    print("    → Puede que ATSEG tenga errores")
    print("  • Si las confusiones son sistemáticas (e.g., siempre SEG_B → SEG_C):")
    print("    → Los segmentos pueden estar mal definidos")
    print("  • Si los doctores mal clasificados tienen features 'en el medio':")
    print("    → Es overlap natural, no error de ATSEG")
    
    print(f"\n📄 Revisa: {OUTPUT_CSV}")
    print("   Busca doctores con alta confianza (>70%) mal clasificados")
    print("   Verifica si sus features tienen más sentido en la clase predicha")
    
    print("\n")
    
    return misclass_df


if __name__ == "__main__":
    misclass_df = main()