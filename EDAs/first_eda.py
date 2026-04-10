"""
EDA COMPLETO - VELSIPITY HCP SEGMENTATION
Análisis exploratorio robusto con visualizaciones que SÍ funcionan

Autor: Ana Gabriela
Fecha: Abril 2026
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin display
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("\n" + "="*80)
print("EDA COMPLETO - VELSIPITY HCP SEGMENTATION")
print("="*80)

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================
print("\n[1/7] Cargando datos...")
df = pd.read_csv('data/raw/VELSIPITY_AFF_MULTIPLI_prepared_prepared_prepared.csv')
print(f"✓ Dataset cargado: {len(df):,} filas × {len(df.columns)} columnas")

# ============================================================================
# 2. ANÁLISIS DE IDs ÚNICOS Y CLASIFICACIÓN
# ============================================================================
print("\n[2/7] Analizando IDs únicos y clasificación...")

total_ids = df['NUEVO_ID'].nunique()
print(f"✓ Total IDs únicos (NUEVO_ID): {total_ids:,}")

# Contar cuántos tienen ATSEG
ids_con_atseg = df[df['ATSEG'].notna()]['NUEVO_ID'].nunique()
ids_sin_atseg = total_ids - ids_con_atseg

print(f"✓ IDs clasificados (con ATSEG): {ids_con_atseg:,} ({ids_con_atseg/total_ids*100:.1f}%)")
print(f"✓ IDs sin clasificar (sin ATSEG): {ids_sin_atseg:,} ({ids_sin_atseg/total_ids*100:.1f}%)")

# Distribución por segmento
seg_counts = df.groupby('NUEVO_ID')['ATSEG'].first().value_counts()
print("\nDistribución por segmento:")
for seg, count in seg_counts.items():
    if pd.notna(seg):
        print(f"  {seg}: {count:,} ({count/total_ids*100:.1f}%)")

# ============================================================================
# 3. ANÁLISIS DE FECHAS (WEEK_ID)
# ============================================================================
print("\n[3/7] Analizando fechas (WEEK_ID)...")

df['WEEK_ID'] = pd.to_datetime(df['WEEK_ID'])
print(f"✓ Rango de fechas: {df['WEEK_ID'].min().strftime('%Y-%m-%d')} a {df['WEEK_ID'].max().strftime('%Y-%m-%d')}")

total_weeks = df['WEEK_ID'].nunique()
print(f"✓ Total semanas únicas: {total_weeks}")

# Distribución temporal
df['Year'] = df['WEEK_ID'].dt.year
df['Month'] = df['WEEK_ID'].dt.month
df['Quarter'] = df['WEEK_ID'].dt.quarter

print("\nDistribución por año:")
year_dist = df.groupby('Year').size()
for year, count in year_dist.items():
    print(f"  {year}: {count:,} registros")

# ============================================================================
# 4. AGREGAR A NIVEL HCP
# ============================================================================
print("\n[4/7] Agregando datos a nivel HCP...")

# Crear dataset agregado
numeric_cols = df.select_dtypes(include=[np.number]).columns
agg_dict = {}

for col in numeric_cols:
    if col not in ['NUEVO_ID', 'YEAR', 'QTR', 'YEAR_QTR', 'Month', 'Quarter']:
        if 'GIDX' in col:
            agg_dict[col] = 'mean'
        else:
            agg_dict[col] = 'sum'

# Agregar ATSEG y columnas categóricas
agg_dict['ATSEG'] = 'first'
for col in ['SPEC_GE', 'SPEC_GPFM', 'SPEC_IM', 'SPEC_NRP']:
    if col in df.columns:
        agg_dict[col] = 'first'

hcp_df = df.groupby('NUEVO_ID').agg(agg_dict).reset_index()
print(f"✓ Dataset HCP creado: {len(hcp_df):,} HCPs")

# ============================================================================
# 5. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================================
print("\n[5/7] Calculando estadísticas descriptivas...")

key_vars = ['UC_TRX', 'ORAL_TRX', 'IL23_TRX', 'BRAND1_TRX', 'BRAND2_TRX', 
            'N_CLMOTHERS', 'DETAILS', 'SAMPLES']

desc_stats = []
for var in key_vars:
    if var in hcp_df.columns:
        col = hcp_df[var]
        zeros_pct = (col == 0).sum() / len(col) * 100
        desc_stats.append({
            'Variable': var,
            'Mean': col.mean(),
            'Median': col.median(),
            'Std': col.std(),
            'Min': col.min(),
            'Max': col.max(),
            'Zeros_pct': zeros_pct,
            'Skew': col.skew()
        })

desc_df = pd.DataFrame(desc_stats)
print("\nEstadísticas clave:")
print(desc_df.to_string(index=False))

# ============================================================================
# 6. GENERAR VISUALIZACIONES
# ============================================================================
print("\n[6/7] Generando visualizaciones...")

import os
os.makedirs('EDAS/figures', exist_ok=True)

# --- FIGURA 1: Distribución de ATSEG ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
seg_dist = hcp_df['ATSEG'].value_counts()
colors = ['#0033A0', '#00A3E0', '#6E3FA3', '#B0BEC5']
axes[0].pie(seg_dist.values, labels=seg_dist.index, autopct='%1.1f%%', 
           colors=colors, startangle=90)
axes[0].set_title('Distribución de Segmentos (HCPs)', fontsize=14, fontweight='bold')

# Bar chart - Clasificados vs No clasificados
labels = list(seg_dist.index)
values = list(seg_dist.values)
axes[1].bar(range(len(labels)), values, color=colors[:len(labels)])
axes[1].set_xticks(range(len(labels)))
axes[1].set_xticklabels(labels, rotation=45)
axes[1].set_ylabel('Cantidad de HCPs')
axes[1].set_title('HCPs por Segmento', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('EDAS/figures/01_segmentacion.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figura 1: Segmentación guardada")

# --- FIGURA 2: Perfiles por Segmento ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

profile_vars = ['UC_TRX', 'BRAND1_TRX', 'N_CLMOTHERS', 'DETAILS']
for idx, var in enumerate(profile_vars):
    if var in hcp_df.columns:
        data_by_seg = []
        labels_seg = []
        for seg in ['SEG_A', 'SEG_B', 'SEG_C']:
            seg_data = hcp_df[hcp_df['ATSEG'] == seg][var]
            if len(seg_data) > 0:
                data_by_seg.append(seg_data)
                labels_seg.append(seg)
        
        bp = axes[idx].boxplot(data_by_seg, labels=labels_seg, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#0033A0', '#00A3E0', '#6E3FA3']):
            patch.set_facecolor(color)
        
        axes[idx].set_title(f'{var} por Segmento', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(var)
        axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('EDAS/figures/02_perfiles_segmento.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figura 2: Perfiles por segmento guardada")

# --- FIGURA 3: Distribución Temporal ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Registros por mes
monthly_counts = df.groupby([df['WEEK_ID'].dt.to_period('M'), 'ATSEG']).size().unstack(fill_value=0)
monthly_counts.index = monthly_counts.index.to_timestamp()

for seg, color in [('SEG_A', '#0033A0'), ('SEG_B', '#00A3E0'), ('SEG_C', '#6E3FA3')]:
    if seg in monthly_counts.columns:
        axes[0].plot(monthly_counts.index, monthly_counts[seg], label=seg, 
                    color=color, linewidth=2, marker='o', markersize=4)

axes[0].set_title('Registros por Mes y Segmento', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Fecha')
axes[0].set_ylabel('Cantidad de Registros')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# UC_TRX promedio por trimestre
if 'UC_TRX' in df.columns:
    df['Quarter_Period'] = df['WEEK_ID'].dt.to_period('Q')
    quarterly_avg = df.groupby(['Quarter_Period', 'ATSEG'])['UC_TRX'].mean().unstack(fill_value=0)
    quarterly_avg.index = quarterly_avg.index.astype(str)
    
    x = range(len(quarterly_avg))
    width = 0.25
    
    for i, (seg, color) in enumerate([('SEG_A', '#0033A0'), ('SEG_B', '#00A3E0'), ('SEG_C', '#6E3FA3')]):
        if seg in quarterly_avg.columns:
            axes[1].bar([xi + i*width for xi in x], quarterly_avg[seg], 
                       width, label=seg, color=color)
    
    axes[1].set_xticks([xi + width for xi in x])
    axes[1].set_xticklabels(quarterly_avg.index, rotation=45)
    axes[1].set_title('UC_TRX Promedio por Trimestre', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('UC_TRX Promedio')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('EDAS/figures/03_analisis_temporal.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figura 3: Análisis temporal guardado")

# --- FIGURA 4: Correlaciones ---
corr_vars = ['UC_TRX', 'ORAL_TRX', 'IL23_TRX', 'BRAND1_TRX', 'BRAND2_TRX', 'N_CLMOTHERS']
available_vars = [v for v in corr_vars if v in hcp_df.columns]

if len(available_vars) > 1:
    corr_matrix = hcp_df[available_vars].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
               center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    ax.set_title('Matriz de Correlación - Variables Clave', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('EDAS/figures/04_correlaciones.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Figura 4: Correlaciones guardada")

# --- FIGURA 5: Sparsity ---
sparsity_data = []
for col in hcp_df.select_dtypes(include=[np.number]).columns:
    if col != 'NUEVO_ID':
        zero_pct = (hcp_df[col] == 0).sum() / len(hcp_df) * 100
        if zero_pct > 20:
            sparsity_data.append({'Variable': col, 'Zeros_pct': zero_pct})

sparsity_df = pd.DataFrame(sparsity_data).sort_values('Zeros_pct', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(12, 8))
colors_sparse = ['#D0021B' if x > 90 else '#E87722' if x > 75 else '#FFB74D' if x > 50 else '#00875A' 
                for x in sparsity_df['Zeros_pct']]
ax.barh(sparsity_df['Variable'], sparsity_df['Zeros_pct'], color=colors_sparse)
ax.set_xlabel('% de Ceros')
ax.set_title('Top 15 Variables con Mayor Sparsity', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('EDAS/figures/05_sparsity.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figura 5: Sparsity guardada")

# ============================================================================
# 7. GENERAR REPORTE HTML
# ============================================================================
print("\n[7/7] Generando reporte HTML...")

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Completo - Velsipity HCP Segmentation</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            color: #1a2a3a;
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #001A4D 0%, #0033A0 50%, #0079C1 100%);
            color: white;
            padding: 60px 40px 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 42px;
            font-weight: 900;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 18px;
            opacity: 0.9;
        }}
        .badge {{
            display: inline-block;
            background: rgba(255,255,255,0.15);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            margin-top: 20px;
            font-weight: 600;
            letter-spacing: 1px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            text-align: center;
        }}
        .stat-card .number {{
            font-size: 36px;
            font-weight: 800;
            color: #0033A0;
            margin-bottom: 8px;
        }}
        .stat-card .label {{
            font-size: 12px;
            color: #5A6A7A;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }}
        .section {{
            background: white;
            border-radius: 12px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}
        .section h2 {{
            font-size: 24px;
            font-weight: 800;
            color: #001A4D;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 3px solid #0033A0;
        }}
        .section h3 {{
            font-size: 18px;
            font-weight: 700;
            color: #0033A0;
            margin: 25px 0 15px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th {{
            background: #E8F4FA;
            padding: 12px;
            text-align: left;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            color: #0033A0;
            border-bottom: 2px solid #B4D8E7;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #F0F3F6;
            font-size: 13px;
        }}
        tr:hover {{
            background: #FAFCFE;
        }}
        .highlight {{
            background: #E8F4FA;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #0033A0;
            margin: 20px 0;
        }}
        .highlight strong {{
            color: #0033A0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .footer {{
            text-align: center;
            padding: 40px;
            color: #5A6A7A;
            font-size: 12px;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }}
        @media (max-width: 768px) {{
            .grid-2 {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>

<div class="header">
    <h1>EDA Completo</h1>
    <p>Velsipity HCP Segmentation - Análisis Exploratorio de Datos</p>
    <div class="badge">PFIZER DATA SCIENCE · CAPSTONE 2026</div>
</div>

<div class="container">

    <!-- ESTADÍSTICAS GENERALES -->
    <div class="stats-grid">
        <div class="stat-card">
            <div class="number">{len(df):,}</div>
            <div class="label">Total Registros</div>
        </div>
        <div class="stat-card">
            <div class="number">{total_ids:,}</div>
            <div class="label">IDs Únicos (HCPs)</div>
        </div>
        <div class="stat-card">
            <div class="number">{total_weeks}</div>
            <div class="label">Semanas de Datos</div>
        </div>
        <div class="stat-card">
            <div class="number">{len(df.columns)}</div>
            <div class="label">Variables</div>
        </div>
        <div class="stat-card">
            <div class="number">{ids_con_atseg:,}</div>
            <div class="label">IDs Clasificados</div>
        </div>
        <div class="stat-card">
            <div class="number">{ids_sin_atseg:,}</div>
            <div class="label">IDs Sin Clasificar</div>
        </div>
    </div>

    <!-- SECCIÓN 1: IDs y Clasificación -->
    <div class="section">
        <h2>📋 Análisis de IDs Únicos y Clasificación</h2>
        
        <div class="highlight">
            <strong>Total de IDs únicos (NUEVO_ID):</strong> {total_ids:,} HCPs<br>
            <strong>IDs clasificados (con ATSEG):</strong> {ids_con_atseg:,} ({ids_con_atseg/total_ids*100:.1f}%)<br>
            <strong>IDs sin clasificar (sin ATSEG):</strong> {ids_sin_atseg:,} ({ids_sin_atseg/total_ids*100:.1f}%)
        </div>
        
        <h3>Distribución por Segmento</h3>
        <table>
            <thead>
                <tr>
                    <th>Segmento</th>
                    <th>Cantidad</th>
                    <th>% del Total</th>
                </tr>
            </thead>
            <tbody>
"""

for seg, count in seg_counts.items():
    if pd.notna(seg):
        html += f"""
                <tr>
                    <td><strong>{seg}</strong></td>
                    <td>{count:,}</td>
                    <td>{count/total_ids*100:.1f}%</td>
                </tr>
"""

html += f"""
            </tbody>
        </table>
        
        <img src="figures/01_segmentacion.png" alt="Distribución de Segmentos">
    </div>

    <!-- SECCIÓN 2: Análisis Temporal -->
    <div class="section">
        <h2>📅 Análisis de Fechas (WEEK_ID)</h2>
        
        <div class="highlight">
            <strong>Rango de fechas:</strong> {df['WEEK_ID'].min().strftime('%Y-%m-%d')} a {df['WEEK_ID'].max().strftime('%Y-%m-%d')}<br>
            <strong>Total semanas únicas:</strong> {total_weeks}<br>
            <strong>Duración:</strong> {(df['WEEK_ID'].max() - df['WEEK_ID'].min()).days} días
        </div>
        
        <h3>Distribución por Año</h3>
        <table>
            <thead>
                <tr>
                    <th>Año</th>
                    <th>Cantidad de Registros</th>
                </tr>
            </thead>
            <tbody>
"""

for year, count in year_dist.items():
    html += f"""
                <tr>
                    <td>{int(year)}</td>
                    <td>{count:,}</td>
                </tr>
"""

html += """
            </tbody>
        </table>
        
        <img src="figures/03_analisis_temporal.png" alt="Análisis Temporal">
    </div>

    <!-- SECCIÓN 3: Perfiles por Segmento -->
    <div class="section">
        <h2>📊 Perfiles por Segmento</h2>
        <p>Distribución de variables clave agregadas a nivel HCP por segmento (SEG_A, SEG_B, SEG_C).</p>
        <img src="figures/02_perfiles_segmento.png" alt="Perfiles por Segmento">
    </div>

    <!-- SECCIÓN 4: Estadísticas Descriptivas -->
    <div class="section">
        <h2>📝 Estadísticas Descriptivas - Variables Clave</h2>
        <table>
            <thead>
                <tr>
                    <th>Variable</th>
                    <th>Media</th>
                    <th>Mediana</th>
                    <th>Desv. Std</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>% Ceros</th>
                    <th>Skewness</th>
                </tr>
            </thead>
            <tbody>
"""

for _, row in desc_df.iterrows():
    html += f"""
                <tr>
                    <td><strong>{row['Variable']}</strong></td>
                    <td>{row['Mean']:.2f}</td>
                    <td>{row['Median']:.2f}</td>
                    <td>{row['Std']:.2f}</td>
                    <td>{row['Min']:.2f}</td>
                    <td>{row['Max']:.2f}</td>
                    <td>{row['Zeros_pct']:.1f}%</td>
                    <td>{row['Skew']:.2f}</td>
                </tr>
"""

html += """
            </tbody>
        </table>
    </div>

    <!-- SECCIÓN 5: Correlaciones -->
    <div class="section">
        <h2>🔗 Matriz de Correlación</h2>
        <p>Correlaciones entre las variables clave del dataset agregado a nivel HCP.</p>
        <img src="figures/04_correlaciones.png" alt="Correlaciones">
    </div>

    <!-- SECCIÓN 6: Sparsity -->
    <div class="section">
        <h2>🕳️ Análisis de Sparsity</h2>
        <p>Variables con mayor porcentaje de valores en cero (sparsity > 20%).</p>
        <img src="figures/05_sparsity.png" alt="Sparsity">
    </div>

</div>

<div class="footer">
    <p><strong>Pfizer Data Science Team</strong> · Tec de Monterrey Capstone Project 2026</p>
    <p>"Breakthroughs that change patients' lives"</p>
    <p>Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>

</body>
</html>
"""

# Guardar HTML
with open('EDAS/first_EDA.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n{'='*80}")
print("✓ EDA COMPLETO GENERADO EXITOSAMENTE")
print(f"{'='*80}")
print(f"\nArchivos generados:")
print(f"  - EDAS/first_EDA.html (Reporte principal)")
print(f"  - EDAS/figures/01_segmentacion.png")
print(f"  - EDAS/figures/02_perfiles_segmento.png")
print(f"  - EDAS/figures/03_analisis_temporal.png")
print(f"  - EDAS/figures/04_correlaciones.png")
print(f"  - EDAS/figures/05_sparsity.png")
print(f"\nAbre EDAS/first_EDA.html en tu navegador para ver el reporte completo.\n")