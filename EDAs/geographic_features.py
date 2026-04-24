"""
================================================================================
AGREGAR FEATURES GEOGRÁFICOS BASADOS EN ZIP CODES
================================================================================

Este script:
1. Extrae ZIP codes de la columna de dirección
2. Crea features geográficos (ZIP3, ZIP2, densidad)
3. Agrega al dataset agregado
4. Re-entrena el modelo con features geográficos

AUTOR: Pfizer Data Science Team
FECHA: March 2026
================================================================================
"""

import pandas as pd
import numpy as np
import re
import os

# Paths
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

INPUT_RAW = os.path.join(RAW_DATA_DIR, 'data_processed.csv')
INPUT_AGGREGATED = os.path.join(PROCESSED_DATA_DIR, 'doctors_aggregated.csv')
OUTPUT_WITH_GEO = os.path.join(PROCESSED_DATA_DIR, 'doctors_aggregated_with_geo.csv')


# ==============================================================================
# PASO 1: EXTRAER ZIP CODES
# ==============================================================================

def extract_zip_code(address):
    """
    Extrae ZIP code de una dirección
    
    Busca patrones:
    - 12345
    - 12345-6789
    
    Returns:
    --------
    str : ZIP code o None
    """
    if pd.isna(address):
        return None
    
    # Patrón para ZIP codes de 5 dígitos (con o sin +4)
    pattern = r'\b(\d{5})(?:-\d{4})?\b'
    match = re.search(pattern, str(address))
    
    if match:
        return match.group(1)  # Retorna solo los primeros 5 dígitos
    
    return None


def create_zip_hierarchy(zip_code):
    """
    Crea jerarquía de ZIP codes según Caliper
    
    ZIP: 12345
    ├─ ZIP3: 123 (área regional ~100-200 millas)
    ├─ ZIP2: 12  (estado/región grande)
    └─ ZIP1: 1   (división nacional)
    
    Returns:
    --------
    dict : {'zip5': '12345', 'zip3': '123', 'zip2': '12', 'zip1': '1'}
    """
    if pd.isna(zip_code) or zip_code is None:
        return {'zip5': None, 'zip3': None, 'zip2': None, 'zip1': None}
    
    zip_str = str(zip_code).zfill(5)  # Asegura 5 dígitos con ceros a la izquierda
    
    return {
        'zip5': zip_str,      # 12345
        'zip3': zip_str[:3],  # 123
        'zip2': zip_str[:2],  # 12
        'zip1': zip_str[:1]   # 1
    }


# ==============================================================================
# PASO 2: CREAR FEATURES GEOGRÁFICOS
# ==============================================================================

def create_geographic_features(df_raw, df_aggregated):
    """
    Crea features geográficos para cada doctor
    
    Features creados:
    1. ZIP codes (ZIP5, ZIP3, ZIP2, ZIP1)
    2. Densidad de doctores por zona
    3. Concentración de prescripciones por zona
    4. Features de competencia local
    
    Parameters:
    -----------
    df_raw : pd.DataFrame
        Dataset original con columna de dirección
    df_aggregated : pd.DataFrame
        Dataset agregado
        
    Returns:
    --------
    pd.DataFrame : Dataset agregado con features geográficos
    """
    
    print("="*80)
    print("CREANDO FEATURES GEOGRÁFICOS")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # 1. BUSCAR COLUMNA DE DIRECCIÓN
    # -------------------------------------------------------------------------
    
    print("\n1. Buscando columna de dirección...")
    
    # Posibles nombres de columna de dirección
    address_col_options = ['ADDRESS', 'ADDR', 'DIRECCION', 'DIRECTION', 
                          'STREET', 'LOCATION', 'address', 'addr']
    
    address_col = None
    for col in df_raw.columns:
        if any(opt.lower() in col.lower() for opt in address_col_options):
            address_col = col
            break
    
    if address_col is None:
        print("\n⚠️  No se encontró columna de dirección en el dataset")
        print(f"   Columnas disponibles: {df_raw.columns.tolist()[:10]}...")
        print("\n   SOLUCIÓN:")
        print("   1. Verifica que tu dataset tenga una columna de dirección")
        print("   2. O proporciona ZIP codes en una columna separada")
        print("\n   El script continuará SIN features geográficos")
        return df_aggregated
    
    print(f"✓ Columna de dirección encontrada: '{address_col}'")
    
    # -------------------------------------------------------------------------
    # 2. EXTRAER ZIP CODES
    # -------------------------------------------------------------------------
    
    print("\n2. Extrayendo ZIP codes...")
    
    # Toma solo una fila por doctor para extraer ZIP
    df_one_per_doctor = df_raw.groupby('NUEVO_ID').first().reset_index()
    
    # Extrae ZIP codes
    df_one_per_doctor['ZIP5'] = df_one_per_doctor[address_col].apply(extract_zip_code)
    
    # Cuenta cuántos tienen ZIP
    zip_count = df_one_per_doctor['ZIP5'].notna().sum()
    total = len(df_one_per_doctor)
    
    print(f"✓ ZIP codes extraídos: {zip_count:,} de {total:,} doctores ({zip_count/total*100:.1f}%)")
    
    if zip_count == 0:
        print("\n⚠️  No se pudieron extraer ZIP codes")
        print("   Verifica el formato de la columna de dirección")
        print(f"   Ejemplo de valores: {df_one_per_doctor[address_col].head(3).tolist()}")
        return df_aggregated
    
    # -------------------------------------------------------------------------
    # 3. CREAR JERARQUÍA DE ZIP CODES
    # -------------------------------------------------------------------------
    
    print("\n3. Creando jerarquía de ZIP codes (ZIP5, ZIP3, ZIP2, ZIP1)...")
    
    zip_hierarchy = df_one_per_doctor['ZIP5'].apply(create_zip_hierarchy)
    
    df_one_per_doctor['ZIP5'] = zip_hierarchy.apply(lambda x: x['zip5'])
    df_one_per_doctor['ZIP3'] = zip_hierarchy.apply(lambda x: x['zip3'])
    df_one_per_doctor['ZIP2'] = zip_hierarchy.apply(lambda x: x['zip2'])
    df_one_per_doctor['ZIP1'] = zip_hierarchy.apply(lambda x: x['zip1'])
    
    print(f"✓ Jerarquía creada")
    print(f"   Ejemplo ZIP5: {df_one_per_doctor['ZIP5'].dropna().iloc[0] if zip_count > 0 else 'N/A'}")
    print(f"   → ZIP3: {df_one_per_doctor['ZIP3'].dropna().iloc[0] if zip_count > 0 else 'N/A'}")
    print(f"   → ZIP2: {df_one_per_doctor['ZIP2'].dropna().iloc[0] if zip_count > 0 else 'N/A'}")
    print(f"   → ZIP1: {df_one_per_doctor['ZIP1'].dropna().iloc[0] if zip_count > 0 else 'N/A'}")
    
    # -------------------------------------------------------------------------
    # 4. FEATURES DE DENSIDAD GEOGRÁFICA
    # -------------------------------------------------------------------------
    
    print("\n4. Calculando densidad de doctores por zona...")
    
    # Merge con datos agregados para obtener métricas
    df_with_zip = df_one_per_doctor[['NUEVO_ID', 'ZIP5', 'ZIP3', 'ZIP2', 'ZIP1']].merge(
        df_aggregated[['NUEVO_ID', 'TOTAL_TRX_sum', 'ATSEG_first']],
        on='NUEVO_ID',
        how='left'
    )
    
    # 4a. Densidad de doctores por ZIP3
    zip3_counts = df_with_zip.groupby('ZIP3').size()
    df_with_zip['DOCTORS_IN_ZIP3'] = df_with_zip['ZIP3'].map(zip3_counts)
    
    # 4b. Densidad de doctores por ZIP2
    zip2_counts = df_with_zip.groupby('ZIP2').size()
    df_with_zip['DOCTORS_IN_ZIP2'] = df_with_zip['ZIP2'].map(zip2_counts)
    
    # 4c. Prescripciones promedio en ZIP3
    zip3_avg_trx = df_with_zip.groupby('ZIP3')['TOTAL_TRX_sum'].mean()
    df_with_zip['AVG_TRX_IN_ZIP3'] = df_with_zip['ZIP3'].map(zip3_avg_trx)
    
    # 4d. Prescripciones totales en ZIP3
    zip3_total_trx = df_with_zip.groupby('ZIP3')['TOTAL_TRX_sum'].sum()
    df_with_zip['TOTAL_TRX_IN_ZIP3'] = df_with_zip['ZIP3'].map(zip3_total_trx)
    
    # 4e. % de cada segmento en ZIP3
    zip3_seg_pct = df_with_zip.groupby(['ZIP3', 'ATSEG_first']).size() / df_with_zip.groupby('ZIP3').size()
    
    for seg in ['SEG_A', 'SEG_B', 'SEG_C']:
        df_with_zip[f'PCT_{seg}_IN_ZIP3'] = df_with_zip.apply(
            lambda row: zip3_seg_pct.get((row['ZIP3'], seg), 0) if pd.notna(row['ZIP3']) else 0,
            axis=1
        )
    
    print(f"✓ Features de densidad creados:")
    print(f"   - DOCTORS_IN_ZIP3: Número de doctores en mismo ZIP3")
    print(f"   - DOCTORS_IN_ZIP2: Número de doctores en mismo ZIP2")
    print(f"   - AVG_TRX_IN_ZIP3: Prescripciones promedio en ZIP3")
    print(f"   - TOTAL_TRX_IN_ZIP3: Prescripciones totales en ZIP3")
    print(f"   - PCT_SEG_X_IN_ZIP3: % de cada segmento en ZIP3")
    
    # -------------------------------------------------------------------------
    # 5. ONE-HOT ENCODING PARA ZIP3 MÁS COMUNES
    # -------------------------------------------------------------------------
    
    print("\n5. Creando one-hot encoding para ZIP3 más comunes...")
    
    # Toma los 20 ZIP3 más comunes
    top_zip3 = df_with_zip['ZIP3'].value_counts().head(20).index.tolist()
    
    for zip3 in top_zip3:
        df_with_zip[f'IS_ZIP3_{zip3}'] = (df_with_zip['ZIP3'] == zip3).astype(int)
    
    print(f"✓ One-hot encoding creado para top 20 ZIP3")
    print(f"   Top 5 ZIP3: {top_zip3[:5]}")
    
    # -------------------------------------------------------------------------
    # 6. MERGE CON DATASET AGREGADO
    # -------------------------------------------------------------------------
    
    print("\n6. Agregando features geográficos al dataset...")
    
    # Columnas a agregar
    geo_cols = ['NUEVO_ID', 'ZIP5', 'ZIP3', 'ZIP2', 'ZIP1',
                'DOCTORS_IN_ZIP3', 'DOCTORS_IN_ZIP2',
                'AVG_TRX_IN_ZIP3', 'TOTAL_TRX_IN_ZIP3',
                'PCT_SEG_A_IN_ZIP3', 'PCT_SEG_B_IN_ZIP3', 'PCT_SEG_C_IN_ZIP3']
    
    # Agrega columnas de one-hot
    geo_cols += [f'IS_ZIP3_{z}' for z in top_zip3]
    
    df_with_geo_features = df_with_zip[geo_cols]
    
    # Merge con dataset agregado
    df_final = df_aggregated.merge(df_with_geo_features, on='NUEVO_ID', how='left')
    
    # Llena NaN con 0 para features geográficos
    geo_feature_cols = [col for col in df_final.columns if col not in df_aggregated.columns]
    df_final[geo_feature_cols] = df_final[geo_feature_cols].fillna(0)
    
    print(f"✓ Features agregados: {len(geo_feature_cols)} nuevas columnas")
    print(f"   Dataset original: {len(df_aggregated.columns)} columnas")
    print(f"   Dataset con geo: {len(df_final.columns)} columnas")
    
    # -------------------------------------------------------------------------
    # 7. RESUMEN
    # -------------------------------------------------------------------------
    
    print("\n" + "="*80)
    print("RESUMEN DE FEATURES GEOGRÁFICOS")
    print("="*80)
    
    print(f"\n✓ Total nuevas features: {len(geo_feature_cols)}")
    print(f"\nCategorías:")
    print(f"  - ZIP hierarchy (ZIP5, ZIP3, ZIP2, ZIP1): 4 features")
    print(f"  - Densidad (doctors in area): 2 features")
    print(f"  - Prescripciones en zona: 2 features")
    print(f"  - Distribución de segmentos: 3 features")
    print(f"  - One-hot ZIP3: {len(top_zip3)} features")
    
    print(f"\nEstadísticas de densidad:")
    print(f"  Doctores por ZIP3:")
    print(f"    - Mínimo: {df_final['DOCTORS_IN_ZIP3'].min():.0f}")
    print(f"    - Promedio: {df_final['DOCTORS_IN_ZIP3'].mean():.1f}")
    print(f"    - Máximo: {df_final['DOCTORS_IN_ZIP3'].max():.0f}")
    
    return df_final


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Función principal"""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "AGREGANDO FEATURES GEOGRÁFICOS" + " "*33 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")
    
    # Cargar datos
    print("Cargando datos...")
    
    if not os.path.exists(INPUT_RAW):
        print(f"❌ ERROR: No se encuentra {INPUT_RAW}")
        print("   Necesitas el dataset RAW con la columna de dirección")
        return
    
    if not os.path.exists(INPUT_AGGREGATED):
        print(f"❌ ERROR: No se encuentra {INPUT_AGGREGATED}")
        print("   Ejecuta aggregate_doctors.py primero")
        return
    
    df_raw = pd.read_csv(INPUT_RAW)
    df_aggregated = pd.read_csv(INPUT_AGGREGATED)
    
    print(f"✓ Dataset raw cargado: {len(df_raw):,} filas")
    print(f"✓ Dataset agregado cargado: {len(df_aggregated):,} filas")
    
    # Crear features geográficos
    df_with_geo = create_geographic_features(df_raw, df_aggregated)
    
    # Guardar
    print("\n" + "="*80)
    print("GUARDANDO DATASET CON FEATURES GEOGRÁFICOS")
    print("="*80)
    
    df_with_geo.to_csv(OUTPUT_WITH_GEO, index=False)
    
    print(f"\n✓ Guardado en: {OUTPUT_WITH_GEO}")
    print(f"✓ Tamaño: {len(df_with_geo):,} filas × {len(df_with_geo.columns)} columnas")
    
    print("\n" + "="*80)
    print("✅ ¡COMPLETO!")
    print("="*80)
    print("\nSiguiente paso:")
    print("  1. Usa 'doctors_aggregated_with_geo.csv' para entrenar el modelo")
    print("  2. Compara accuracy con y sin features geográficos")
    print("\n")
    
    return df_with_geo


if __name__ == "__main__":
    main()