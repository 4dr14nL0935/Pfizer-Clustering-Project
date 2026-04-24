"""
================================================================================
AGREGAR FEATURES GEOGRÁFICOS USANDO TERRITORIOS (STATE_1 a STATE_8)
================================================================================

Este script:
1. Identifica el territorio principal de cada doctor
2. Calcula densidad de doctores por territorio
3. Calcula métricas de prescripciones por territorio
4. Agrega features geográficos al dataset
5. Crea visualizaciones

AUTOR: Pfizer Data Science Team
FECHA: March 2026
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
BASE_DIR = os.getcwd()
RAW_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'VELSIPITY_AFF_MULTIPLI_prepared_prepared_prepared.csv')
AGG_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'doctors_aggregated.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'doctors_aggregated_with_geo.csv')
OUTPUT_PLOT = os.path.join(BASE_DIR, 'results', 'geographic_analysis.png')

os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)


print("\n" + "="*80)
print("🗺️  AGREGANDO FEATURES GEOGRÁFICOS (TERRITORIOS)")
print("="*80)


# ==============================================================================
# PASO 1: CARGAR DATOS
# ==============================================================================

print("\n📂 PASO 1: Cargando datos...")

if not os.path.exists(RAW_FILE):
    print(f"❌ ERROR: No encuentro {RAW_FILE}")
    exit()

if not os.path.exists(AGG_FILE):
    print(f"❌ ERROR: No encuentro {AGG_FILE}")
    print("   Ejecuta primero: aggregate_doctors.py")
    exit()

df_raw = pd.read_csv(RAW_FILE)
df_agg = pd.read_csv(AGG_FILE)

print(f"✓ Dataset raw: {len(df_raw):,} filas")
print(f"✓ Dataset agregado: {len(df_agg):,} doctores")


# ==============================================================================
# PASO 2: IDENTIFICAR TERRITORIO DE CADA DOCTOR
# ==============================================================================

print("\n🔍 PASO 2: Identificando territorio de cada doctor...")

# Las columnas STATE son binarias (0 o 1)
state_cols = ['STATE_1', 'STATE_2', 'STATE_3', 'STATE_4', 'STATE_5', 
              'STATE_6', 'STATE_7', 'STATE_8', 'STS_OTHER_STS']

# Verificar que existan
existing_state_cols = [col for col in state_cols if col in df_raw.columns]

if len(existing_state_cols) == 0:
    print("❌ No se encontraron columnas STATE en el dataset")
    exit()

print(f"✓ Columnas de territorio encontradas: {len(existing_state_cols)}")
print(f"  {existing_state_cols}")

# Tomar una fila por doctor
df_territory = df_raw.groupby('NUEVO_ID')[existing_state_cols].first().reset_index()

# Identificar el territorio principal (donde tiene 1)
def get_main_territory(row):
    """Identifica el territorio principal del doctor"""
    for col in existing_state_cols:
        if row[col] == 1:
            # Extraer número del territorio (STATE_1 → 1)
            if 'OTHER' in col:
                return 'OTHER'
            else:
                return col.replace('STATE_', '').replace('STS_', '')
    return 'UNKNOWN'

df_territory['TERRITORY'] = df_territory[existing_state_cols].apply(get_main_territory, axis=1)

# Contar doctores por territorio
territory_counts = df_territory['TERRITORY'].value_counts()

print(f"\n✓ Distribución de doctores por territorio:")
for territory, count in territory_counts.items():
    pct = count / len(df_territory) * 100
    print(f"  Territorio {territory}: {count:,} doctores ({pct:.1f}%)")


# ==============================================================================
# PASO 3: CALCULAR FEATURES GEOGRÁFICOS
# ==============================================================================

print("\n📊 PASO 3: Calculando features geográficos por territorio...")

# Merge con datos agregados para obtener métricas
df_with_territory = df_territory[['NUEVO_ID', 'TERRITORY']].merge(
    df_agg[['NUEVO_ID', 'TOTAL_TRX_sum', 'UC_TRX_sum', 'ATSEG_first']], 
    on='NUEVO_ID', 
    how='left'
)

# 3.1 Densidad de doctores por territorio
territory_doctor_counts = df_with_territory.groupby('TERRITORY').size()
df_with_territory['DOCTORS_IN_TERRITORY'] = df_with_territory['TERRITORY'].map(territory_doctor_counts)

# 3.2 Prescripciones promedio en el territorio
territory_avg_trx = df_with_territory.groupby('TERRITORY')['TOTAL_TRX_sum'].mean()
df_with_territory['AVG_TRX_IN_TERRITORY'] = df_with_territory['TERRITORY'].map(territory_avg_trx)

territory_avg_uc = df_with_territory.groupby('TERRITORY')['UC_TRX_sum'].mean()
df_with_territory['AVG_UC_TRX_IN_TERRITORY'] = df_with_territory['TERRITORY'].map(territory_avg_uc)

# 3.3 Prescripciones totales en el territorio
territory_total_trx = df_with_territory.groupby('TERRITORY')['TOTAL_TRX_sum'].sum()
df_with_territory['TOTAL_TRX_IN_TERRITORY'] = df_with_territory['TERRITORY'].map(territory_total_trx)

# 3.4 Porcentaje de cada segmento en el territorio
for seg in ['SEG_A', 'SEG_B', 'SEG_C']:
    seg_counts = df_with_territory[df_with_territory['ATSEG_first'] == seg].groupby('TERRITORY').size()
    total_counts = df_with_territory.groupby('TERRITORY').size()
    seg_pct = (seg_counts / total_counts).fillna(0)
    df_with_territory[f'PCT_{seg}_IN_TERRITORY'] = df_with_territory['TERRITORY'].map(seg_pct).fillna(0)

# 3.5 Ranking del territorio (por volumen total de Rx)
territory_ranking = territory_total_trx.rank(ascending=False, method='dense')
df_with_territory['TERRITORY_RANK'] = df_with_territory['TERRITORY'].map(territory_ranking)

# 3.6 One-hot encoding para cada territorio
for territory in territory_counts.index:
    df_with_territory[f'IS_TERRITORY_{territory}'] = (df_with_territory['TERRITORY'] == territory).astype(int)

print(f"✓ Features creados:")
print(f"   • Densidad: DOCTORS_IN_TERRITORY")
print(f"   • Prescripciones: AVG_TRX_IN_TERRITORY, TOTAL_TRX_IN_TERRITORY")
print(f"   • UC específico: AVG_UC_TRX_IN_TERRITORY")
print(f"   • Distribución: PCT_SEG_A/B/C_IN_TERRITORY")
print(f"   • Ranking: TERRITORY_RANK (1 = territorio con más Rx)")
print(f"   • One-hot: {len(territory_counts)} territorios")


# ==============================================================================
# PASO 4: MERGE CON DATASET AGREGADO
# ==============================================================================

print("\n🔗 PASO 4: Agregando features al dataset...")

# Seleccionar columnas geográficas
geo_cols = ['NUEVO_ID', 'TERRITORY', 'DOCTORS_IN_TERRITORY',
            'AVG_TRX_IN_TERRITORY', 'AVG_UC_TRX_IN_TERRITORY',
            'TOTAL_TRX_IN_TERRITORY', 'TERRITORY_RANK',
            'PCT_SEG_A_IN_TERRITORY', 'PCT_SEG_B_IN_TERRITORY', 
            'PCT_SEG_C_IN_TERRITORY']

# Agregar columnas de one-hot
geo_cols += [f'IS_TERRITORY_{t}' for t in territory_counts.index]

df_geo_features = df_with_territory[geo_cols]

# Merge con dataset agregado
df_final = df_agg.merge(df_geo_features, on='NUEVO_ID', how='left')

# Llenar NaN con 0
new_cols = [c for c in df_final.columns if c not in df_agg.columns]
df_final[new_cols] = df_final[new_cols].fillna(0)

print(f"✓ Dataset original: {len(df_agg.columns)} columnas")
print(f"✓ Dataset con geo: {len(df_final.columns)} columnas")
print(f"✓ Nuevas columnas: {len(new_cols)}")


# ==============================================================================
# PASO 5: GUARDAR
# ==============================================================================

print("\n💾 PASO 5: Guardando dataset...")

df_final.to_csv(OUTPUT_FILE, index=False)

print(f"✓ Guardado en: {OUTPUT_FILE}")
print(f"  {len(df_final):,} filas × {len(df_final.columns)} columnas")


# ==============================================================================
# PASO 6: CREAR GRÁFICAS
# ==============================================================================

print("\n📊 PASO 6: Creando gráficas de verificación...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('FEATURES GEOGRÁFICOS: Análisis por Territorio', 
             fontsize=16, fontweight='bold', y=1.02)

# GRÁFICA 1: Doctores por territorio
ax1 = axes[0, 0]
territory_counts.plot(kind='bar', ax=ax1, color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_title('Doctores por Territorio', fontweight='bold', fontsize=11)
ax1.set_xlabel('Territorio')
ax1.set_ylabel('Número de Doctores')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# Agregar valores encima de barras
for i, (territory, count) in enumerate(territory_counts.items()):
    ax1.text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=9)

# GRÁFICA 2: Prescripciones promedio por territorio
ax2 = axes[0, 1]
territory_avg_trx.sort_values(ascending=False).plot(kind='bar', ax=ax2, 
                                                     color='coral', alpha=0.7, edgecolor='black')
ax2.set_title('Rx Promedio por Territorio', fontweight='bold', fontsize=11)
ax2.set_xlabel('Territorio')
ax2.set_ylabel('Prescripciones Promedio')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

# GRÁFICA 3: Volumen total por territorio
ax3 = axes[0, 2]
territory_total_trx.sort_values(ascending=False).plot(kind='bar', ax=ax3, 
                                                       color='green', alpha=0.7, edgecolor='black')
ax3.set_title('Volumen Total de Rx por Territorio', fontweight='bold', fontsize=11)
ax3.set_xlabel('Territorio')
ax3.set_ylabel('Prescripciones Totales')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(axis='y', alpha=0.3)

# GRÁFICA 4: Densidad promedio por segmento
ax4 = axes[1, 0]
df_labeled = df_final[df_final['ATSEG_first'].isin(['SEG_A', 'SEG_B', 'SEG_C'])]
if len(df_labeled) > 0:
    density_by_seg = df_labeled.groupby('ATSEG_first')['DOCTORS_IN_TERRITORY'].mean()
    bars = ax4.bar(density_by_seg.index, density_by_seg.values, 
                   color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax4.set_title('Densidad Promedio por Segmento', fontweight='bold', fontsize=11)
    ax4.set_xlabel('Segmento ATSEG')
    ax4.set_ylabel('Doctores en Territorio (promedio)')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
else:
    ax4.text(0.5, 0.5, 'No hay datos\nde ATSEG', 
            ha='center', va='center', fontsize=14, transform=ax4.transAxes)

# GRÁFICA 5: Rx promedio en territorio por segmento
ax5 = axes[1, 1]
if len(df_labeled) > 0:
    trx_by_seg = df_labeled.groupby('ATSEG_first')['AVG_TRX_IN_TERRITORY'].mean()
    bars = ax5.bar(trx_by_seg.index, trx_by_seg.values, 
                   color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax5.set_title('Rx Promedio en Territorio por Segmento', fontweight='bold', fontsize=11)
    ax5.set_xlabel('Segmento ATSEG')
    ax5.set_ylabel('Rx en Territorio (promedio)')
    ax5.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
else:
    ax5.text(0.5, 0.5, 'No hay datos\nde ATSEG', 
            ha='center', va='center', fontsize=14, transform=ax5.transAxes)

# GRÁFICA 6: Distribución de segmentos por territorio
ax6 = axes[1, 2]
if len(df_labeled) > 0:
    # Top 5 territorios por volumen
    top_5_territories = territory_total_trx.sort_values(ascending=False).head(5).index
    
    seg_data = []
    for territory in top_5_territories:
        df_terr = df_labeled[df_labeled['TERRITORY'] == territory]
        if len(df_terr) > 0:
            counts = df_terr['ATSEG_first'].value_counts()
            seg_data.append({
                'Territory': str(territory),
                'SEG_A': counts.get('SEG_A', 0),
                'SEG_B': counts.get('SEG_B', 0),
                'SEG_C': counts.get('SEG_C', 0)
            })
    
    if seg_data:
        df_seg = pd.DataFrame(seg_data).set_index('Territory')
        df_seg.plot(kind='bar', stacked=True, ax=ax6,
                   color=['#2ecc71', '#3498db', '#e74c3c'], 
                   alpha=0.7, edgecolor='black')
        ax6.set_title('Segmentos en Top 5 Territorios', fontweight='bold', fontsize=11)
        ax6.set_xlabel('Territorio')
        ax6.set_ylabel('Número de Doctores')
        ax6.legend(title='Segmento', loc='upper right')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(axis='y', alpha=0.3)
else:
    ax6.text(0.5, 0.5, 'No hay datos\nde ATSEG', 
            ha='center', va='center', fontsize=14, transform=ax6.transAxes)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')

print(f"✓ Gráficas guardadas en: {OUTPUT_PLOT}")


# ==============================================================================
# PASO 7: RESUMEN FINAL
# ==============================================================================

print("\n" + "="*80)
print("✅ RESUMEN FINAL")
print("="*80)

total = len(df_final)
with_territory = df_final['TERRITORY'].notna().sum()

print(f"\n📮 TERRITORIOS:")
print(f"   Total doctores: {total:,}")
print(f"   Con territorio: {with_territory:,} ({with_territory/total*100:.1f}%)")
print(f"   Territorios únicos: {df_final['TERRITORY'].nunique()}")

print(f"\n🗺️  TOP 5 TERRITORIOS:")
for i, (territory, count) in enumerate(territory_counts.head(5).items(), 1):
    avg_trx = territory_avg_trx.get(territory, 0)
    print(f"   {i}. Territorio {territory}: {count:,} doctores, {avg_trx:.1f} Rx promedio")

print(f"\n📊 ESTADÍSTICAS:")
print(f"   Densidad mínima: {df_final['DOCTORS_IN_TERRITORY'].min():.0f} doctores/territorio")
print(f"   Densidad promedio: {df_final['DOCTORS_IN_TERRITORY'].mean():.1f} doctores/territorio")
print(f"   Densidad máxima: {df_final['DOCTORS_IN_TERRITORY'].max():.0f} doctores/territorio")

if len(df_labeled) > 0:
    print(f"\n🎯 INSIGHTS POR SEGMENTO:")
    for seg in ['SEG_A', 'SEG_B', 'SEG_C']:
        df_seg = df_labeled[df_labeled['ATSEG_first'] == seg]
        if len(df_seg) > 0:
            avg_density = df_seg['DOCTORS_IN_TERRITORY'].mean()
            avg_trx_in_terr = df_seg['AVG_TRX_IN_TERRITORY'].mean()
            print(f"   {seg}:")
            print(f"     - Densidad promedio en territorio: {avg_density:.0f} doctores")
            print(f"     - Rx promedio en su territorio: {avg_trx_in_terr:.1f}")

print(f"\n📁 ARCHIVOS GENERADOS:")
print(f"   • Dataset: {OUTPUT_FILE}")
print(f"   • Gráficas: {OUTPUT_PLOT}")

print(f"\n🎯 PRÓXIMO PASO:")
print(f"   1. Abre la imagen: {OUTPUT_PLOT}")
print(f"   2. Usa 'doctors_aggregated_with_geo.csv' para re-entrenar el modelo")
print(f"   3. Compara accuracy CON vs SIN features geográficos")

print("\n")