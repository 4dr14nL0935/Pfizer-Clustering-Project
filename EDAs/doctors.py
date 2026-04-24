"""
==============================================================================
DOCTOR DATA AGGREGATION SCRIPT
==============================================================================

INPUT:  data/raw/data_processed.csv (desde donde ejecutas el comando)
OUTPUT: data/processed/doctors_aggregated.csv

==============================================================================
"""

import pandas as pd
import numpy as np
import os
import sys

# ==============================================================================
# PATHS - USA EL DIRECTORIO DONDE EJECUTAS EL COMANDO, NO DONDE ESTÁ EL SCRIPT
# ==============================================================================

# USA getcwd() en lugar de __file__ 
# Esto significa: usa el directorio desde donde ejecutaste "python ..."
BASE_DIR = os.getcwd()  # ¡ESTO ES LO IMPORTANTE!

# Define rutas relativas desde BASE_DIR
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Crea directorios si no existen
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Rutas de archivos
INPUT_FILE = os.path.join(RAW_DATA_DIR, 'data_processed.csv')
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, 'doctors_aggregated.csv')


# ==============================================================================
# STEP 1: LOAD DATA
# ==============================================================================

def load_data(filepath):
    """Cargar datos semanales de HCP"""
    
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Verifica si el archivo existe
    if not os.path.exists(filepath):
        print(f"\n❌ ERROR: ¡Archivo no encontrado!")
        print(f"\n   Buscando en: {filepath}")
        print(f"\n   Directorio actual (donde ejecutaste el comando): {os.getcwd()}")
        print(f"\n   SOLUCIÓN:")
        print(f"   1. Asegúrate de estar en: C:\\Users\\Ana.Gaby\\Pfizer-Clustering-Project")
        print(f"   2. Tu archivo debe estar en: data\\raw\\data_processed.csv")
        print(f"   3. Ejecuta desde la raíz del proyecto, NO desde EDAs")
        print(f"\n   ESTRUCTURA CORRECTA:")
        print(f"   Pfizer-Clustering-Project\\")
        print(f"   ├── data\\")
        print(f"   │   └── raw\\")
        print(f"   │       └── data_processed.csv  ← AQUÍ")
        print(f"   └── EDAs\\")
        print(f"       └── (tus scripts pueden estar aquí)")
        sys.exit(1)
    
    print(f"✓ Archivo encontrado: {filepath}")
    df = pd.read_csv(filepath)
    
    print(f"✓ Cargados: {len(df):,} filas × {len(df.columns)} columnas")
    print(f"✓ Doctores únicos: {df['NUEVO_ID'].nunique():,}")
    print(f"✓ Promedio filas por doctor: {len(df) / df['NUEVO_ID'].nunique():.1f}")
    
    return df


# ==============================================================================
# STEP 2: AGGREGATION RULES
# ==============================================================================

def create_aggregation_rules(df):
    """Define cómo agregar cada columna"""
    
    print("\n" + "="*80)
    print("DEFINIENDO REGLAS DE AGREGACIÓN")
    print("="*80)
    
    agg_rules = {}
    
    # REGLA 1: Volúmenes de prescripción
    prescription_cols = [col for col in df.columns 
                        if any(x in col for x in ['_TRX', '_NRX', '_NBRX', 'N_CLM'])]
    
    print(f"\n1. COLUMNAS DE PRESCRIPCIÓN ({len(prescription_cols)})")
    print(f"   Agregación: Sum + Mean + Max")
    
    for col in prescription_cols:
        agg_rules[col] = ['sum', 'mean', 'max']
    
    # REGLA 2: Métricas de engagement
    engagement_cols = ['RTE', 'SAMPLES', 'COPAY', 'DIRECTMAIL', 'SPK', 
                      'DETAILS', 'ENGAGEMENT_SCORE']
    engagement_cols = [c for c in engagement_cols if c in df.columns]
    
    print(f"\n2. COLUMNAS DE ENGAGEMENT ({len(engagement_cols)})")
    print(f"   Agregación: Sum + Mean")
    
    for col in engagement_cols:
        agg_rules[col] = ['sum', 'mean']
    
    # REGLA 3: Índices y ratios
    index_cols = [col for col in df.columns 
                 if any(x in col for x in ['GIDX', 'RATIO', 'SHARE'])]
    
    print(f"\n3. COLUMNAS DE ÍNDICES/RATIOS ({len(index_cols)})")
    print(f"   Agregación: Mean + Max")
    
    for col in index_cols:
        agg_rules[col] = ['mean', 'max']
    
    # REGLA 4: Rolling sums
    rolling_cols = [col for col in df.columns 
                   if 'SUM' in col and col not in prescription_cols]
    
    print(f"\n4. ROLLING SUMS ({len(rolling_cols)})")
    print(f"   Agregación: Max + Mean")
    
    for col in rolling_cols:
        agg_rules[col] = ['max', 'mean']
    
    # REGLA 5: Demográficos
    demo_cols = [col for col in df.columns 
                if any(x in col for x in ['SPEC_', 'STATE_', 'STS_', 'ATSEG', '(', ']'])]
    
    print(f"\n5. COLUMNAS DEMOGRÁFICAS ({len(demo_cols)})")
    print(f"   Agregación: First (valor constante)")
    
    for col in demo_cols:
        agg_rules[col] = 'first'
    
    # REGLA 6: Tiempo
    if 'WEEK_ID' in df.columns:
        agg_rules['WEEK_ID'] = ['first', 'last', 'count']
        print(f"\n6. COLUMNA DE TIEMPO")
        print(f"   Agregación: First + Last + Count")
    
    print(f"\n{'='*80}")
    print(f"TOTAL COLUMNAS A AGREGAR: {len(agg_rules)}")
    print(f"{'='*80}")
    
    return agg_rules


# ==============================================================================
# STEP 3: PERFORM AGGREGATION
# ==============================================================================

def aggregate_by_doctor(df, agg_rules):
    """Agrupar por doctor y aplicar reglas"""
    
    print("\n" + "="*80)
    print("REALIZANDO AGREGACIÓN")
    print("="*80)
    
    print("\nAgrupando por NUEVO_ID (doctor)...")
    
    aggregated = df.groupby('NUEVO_ID').agg(agg_rules)
    
    print("✓ Agregación completa")
    print("✓ Aplanando nombres de columnas...")
    
    if isinstance(aggregated.columns, pd.MultiIndex):
        aggregated.columns = ['_'.join(col).strip() if isinstance(col, tuple) 
                             else col 
                             for col in aggregated.columns.values]
    
    aggregated.reset_index(inplace=True)
    aggregated = aggregated.fillna(0)
    
    print("\n" + "="*80)
    print("RESULTADOS DE AGREGACIÓN")
    print("="*80)
    print(f"Original:   {len(df):,} filas × {len(df.columns)} columnas")
    print(f"Agregado:   {len(aggregated):,} filas × {len(aggregated.columns)} columnas")
    print(f"\nReducción:  {len(df)/len(aggregated):.1f}x menos filas")
    
    return aggregated


# ==============================================================================
# STEP 4: SAVE
# ==============================================================================

def save_aggregated_data(aggregated_df, output_filepath):
    """Guardar datos agregados"""
    
    print("\n" + "="*80)
    print("GUARDANDO RESULTADOS")
    print("="*80)
    
    aggregated_df.to_csv(output_filepath, index=False)
    
    print(f"✓ Guardado en: {output_filepath}")
    print(f"✓ Tamaño: {len(aggregated_df):,} filas × {len(aggregated_df.columns)} columnas")


# ==============================================================================
# STEP 5: DISPLAY
# ==============================================================================

def display_results(aggregated_df):
    """Mostrar muestra"""
    
    print("\n" + "="*80)
    print("MUESTRA (Primeros 3 Doctores)")
    print("="*80)
    
    key_cols = ['NUEVO_ID', 'TOTAL_TRX_sum', 'TOTAL_TRX_mean', 
                'ENGAGEMENT_SCORE_sum', 'ATSEG_first']
    
    available_cols = [col for col in key_cols if col in aggregated_df.columns]
    
    if available_cols:
        print(aggregated_df[available_cols].head(3))
    else:
        print(aggregated_df.iloc[:3, :5])


# ==============================================================================
# MAIN
# ==============================================================================

def main(input_filepath, output_filepath):
    """Función principal"""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "AGREGACIÓN DE DATOS DE DOCTORES" + " "*27 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")
    
    df = load_data(input_filepath)
    agg_rules = create_aggregation_rules(df)
    aggregated_df = aggregate_by_doctor(df, agg_rules)
    save_aggregated_data(aggregated_df, output_filepath)
    display_results(aggregated_df)
    
    print("\n" + "="*80)
    print("✅ ¡AGREGACIÓN COMPLETA!")
    print("="*80)
    print(f"\nArchivo guardado: {output_filepath}")
    print("\n")
    
    return aggregated_df


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("INFORMACIÓN DE RUTAS")
    print("="*80)
    print(f"\nDirectorio actual (donde ejecutaste el comando):")
    print(f"  {os.getcwd()}")
    print(f"\nBuscando archivo en:")
    print(f"  {INPUT_FILE}")
    print(f"\nGuardará resultado en:")
    print(f"  {OUTPUT_FILE}")
    print(f"\n¿Existe el archivo? {os.path.exists(INPUT_FILE)}")
    
    if not os.path.exists(INPUT_FILE):
        print("\n" + "="*80)
        print("⚠️  INSTRUCCIONES")
        print("="*80)
        print("\n1. Abre tu terminal/cmd")
        print("2. Ve al directorio del proyecto:")
        print("   cd C:\\Users\\Ana.Gaby\\Pfizer-Clustering-Project")
        print("\n3. Ejecuta el script desde ahí:")
        print("   python EDAs\\aggregate_doctors.py")
        print("   (o donde sea que esté el script)")
        print("\n4. La estructura debe ser:")
        print("   Pfizer-Clustering-Project\\  ← EJECUTA DESDE AQUÍ")
        print("   ├── data\\")
        print("   │   └── raw\\")
        print("   │       └── data_processed.csv")
        print("   └── EDAs\\")
        print("       └── aggregate_doctors.py")
        print("\n")
    else:
        # Run the aggregation
        aggregated_data = main(INPUT_FILE, OUTPUT_FILE)
        
        print(f"\n{'='*80}")
        print(f"✓ ¡Éxito! Siguiente paso: ejecuta train_classification_model.py")
        print(f"{'='*80}\n")