"""
EDA Generator - Genera HTML completo desde cero
Calcula TODAS las secciones y genera HTML identico al template

Autor: Data Science Team
Fecha: Abril 2026
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def format_number(n):
    """Formatea números con comas"""
    if n >= 1000000:
        return f"{n/1000000:.1f}M"
    elif n >= 1000:
        return f"{n:,}"
    else:
        return str(int(n))


class EDAGeneratorFinal:
    """Genera EDA completo desde cero"""
    
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.df = None
        self.hcp_df = None
        self.data = {}
        
    def run(self):
        """Pipeline completo"""
        print("\n" + "="*70)
        print("GENERANDO EDA COMPLETO")
        print("="*70)
        
        self.load_data()
        self.calculate_all()
        self.generate_html()
        
        print("="*70)
        print(f"HTML GENERADO: {self.output_path}")
        print("="*70)
        
    def load_data(self):
        """Carga y agrega datos a nivel HCP"""
        print("[1/3] Cargando datos...")
        self.df = pd.read_csv(self.data_path)
        print(f"      Cargado: {len(self.df):,} filas, {len(self.df.columns)} columnas")
        
        # Agregar a nivel HCP
        if 'NUEVO_ID' in self.df.columns:
            print("[1/3] Agregando a nivel HCP...")
            
            agg_dict = {'ATSEG': 'first'}
            
            # Identificar columnas numéricas
            for col in self.df.select_dtypes(include=[np.number]).columns:
                if col.endswith('_ID') or col in ['NUEVO_ID', 'WEEK_ID']:
                    continue
                elif 'GIDX' in col:
                    agg_dict[col] = 'mean'
                else:
                    agg_dict[col] = 'sum'
            
            # Columnas categóricas (demografía)
            for col in self.df.columns:
                if col.startswith('SPECIALTY_') or col.startswith('TITLE_') or '(' in col and ')' in col:
                    if col not in agg_dict:
                        agg_dict[col] = 'first'
            
            self.hcp_df = self.df.groupby('NUEVO_ID').agg(agg_dict).reset_index()
            print(f"      Resultado: {len(self.hcp_df)} HCPs")
        else:
            self.hcp_df = self.df.copy()
            
    def calculate_all(self):
        """Calcula TODAS las métricas"""
        print("[2/3] Calculando métricas...")
        
        self.data['stats'] = self._calc_stats()
        self.data['target'] = self._calc_target()
        self.data['profiles'] = self._calc_profiles()
        self.data['comparativa'] = self._calc_comparativa()
        self.data['sparsity'] = self._calc_sparsity()
        self.data['demographics'] = self._calc_demographics()
        self.data['correlation'] = self._calc_correlation()
        self.data['temporal'] = self._calc_temporal()
        self.data['descriptive'] = self._calc_descriptive()
        
        print("      Todas las métricas calculadas")
        
    def _calc_stats(self):
        """Stats strip"""
        return {
            'total_rows': len(self.df),
            'unique_hcps': len(self.hcp_df),
            'weeks': self.df['WEEK_ID'].nunique() if 'WEEK_ID' in self.df.columns else 86,
            'total_cols': len(self.df.columns)
        }
        
    def _calc_target(self):
        """Target distribution"""
        if 'ATSEG' not in self.hcp_df.columns:
            return {'seg_a': 0, 'seg_b': 0, 'seg_c': 0, 'unlabeled': len(self.hcp_df)}
        
        counts = self.hcp_df['ATSEG'].value_counts()
        return {
            'seg_a': int(counts.get('SEG_A', 0)),
            'seg_b': int(counts.get('SEG_B', 0)),
            'seg_c': int(counts.get('SEG_C', 0)),
            'unlabeled': int(self.hcp_df['ATSEG'].isna().sum())
        }
        
    def _calc_profiles(self):
        """Perfiles por segmento (media y mediana)"""
        profiles = {}
        
        for var in ['UC_TRX', 'N_CLMOTHERS', 'DETAILS', 'BRAND1_TRX']:
            if var not in self.hcp_df.columns:
                continue
            
            profiles[var] = {}
            for seg in ['SEG_A', 'SEG_B', 'SEG_C']:
                data = self.hcp_df[self.hcp_df['ATSEG'] == seg][var]
                if len(data) > 0:
                    profiles[var][seg] = [round(data.mean(), 2), round(data.median(), 2)]
                else:
                    profiles[var][seg] = [0, 0]
        
        return profiles
        
    def _calc_comparativa(self):
        """Tabla comparativa de segmentos"""
        vars_compare = ['UC_TRX', 'ORAL_TRX', 'IL23_TRX', 'BRAND1_TRX', 'N_CLMOTHERS', 'DETAILS']
        
        table = []
        for var in vars_compare:
            if var not in self.hcp_df.columns:
                continue
            
            row = {'var': var}
            for seg in ['SEG_A', 'SEG_B', 'SEG_C']:
                data = self.hcp_df[self.hcp_df['ATSEG'] == seg][var]
                if len(data) > 0:
                    row[seg] = round(data.mean(), 2)
                else:
                    row[seg] = 0
            table.append(row)
        
        return table
        
    def _calc_sparsity(self):
        """Sparsity analysis"""
        sparsity = []
        
        for col in self.hcp_df.columns:
            if col in ['NUEVO_ID', 'ATSEG'] or col.endswith('_ID'):
                continue
            
            if pd.api.types.is_numeric_dtype(self.hcp_df[col]):
                zero_pct = (self.hcp_df[col] == 0).sum() / len(self.hcp_df) * 100
                if zero_pct > 20:
                    sparsity.append({'name': col, 'pct': round(zero_pct, 1)})
        
        sparsity.sort(key=lambda x: x['pct'], reverse=True)
        return sparsity[:15]
        
    def _calc_demographics(self):
        """Demografia (especialidad y edad)"""
        demo = {'specialty': None, 'age': None}
        
        # Especialidad
        spec_cols = [c for c in self.hcp_df.columns if c.startswith('SPECIALTY_')]
        if spec_cols:
            # Calcular % por segmento
            demo['specialty'] = {}
            for seg in ['SEG_A', 'SEG_B', 'SEG_C', 'ALL']:
                if seg == 'ALL':
                    seg_data = self.hcp_df
                else:
                    seg_data = self.hcp_df[self.hcp_df['ATSEG'] == seg]
                
                demo['specialty'][seg] = {}
                for col in spec_cols:
                    spec_name = col.replace('SPECIALTY_', '')
                    if len(seg_data) > 0:
                        pct = (seg_data[col] == 1).sum() / len(seg_data) * 100
                        demo['specialty'][seg][spec_name] = round(pct, 1)
        
        # Edad
        age_cols = [c for c in self.hcp_df.columns if '(' in c and ')' in c and any(y in c for y in ['1960', '1970', '1980', '1990', '2000'])]
        if age_cols:
            demo['age'] = {}
            for seg in ['SEG_A', 'SEG_B', 'SEG_C']:
                seg_data = self.hcp_df[self.hcp_df['ATSEG'] == seg]
                
                demo['age'][seg] = {}
                for col in age_cols:
                    if len(seg_data) > 0:
                        pct = (seg_data[col] == 1).sum() / len(seg_data) * 100
                        demo['age'][seg][col] = round(pct, 1)
        
        return demo
        
    def _calc_correlation(self):
        """Matriz de correlacion"""
        vars_corr = ['UC_TRX', 'ORAL_TRX', 'IL23_TRX', 'BRAND1_TRX', 'BRAND2_TRX',
                    'N_CLMBRAND3', 'N_CLMOTHERS', 'DETAILS']
        
        available = [v for v in vars_corr if v in self.hcp_df.columns]
        
        if len(available) < 2:
            return {'labels': [], 'data': []}
        
        corr = self.hcp_df[available].corr()
        
        # Abreviar nombres
        short_labels = [v.replace('BRAND1_', 'B1_').replace('BRAND2_', 'B2_').replace('N_CLM', 'CLM_') for v in available]
        
        return {
            'labels': short_labels,
            'data': [[round(v, 3) for v in row] for row in corr.values]
        }
        
    def _calc_temporal(self):
        """Evolucion temporal"""
        if 'WEEK_ID' not in self.df.columns or 'ATSEG' not in self.df.columns:
            return {}
        
        self.df['date'] = pd.to_datetime(self.df['WEEK_ID'], format='%m/%d/%Y', errors='coerce')
        self.df['quarter'] = self.df['date'].dt.to_period('Q')
        
        temporal = {}
        
        for var in ['UC_TRX', 'BRAND1_TRX']:
            if var not in self.df.columns:
                continue
            
            quarters = sorted([q for q in self.df['quarter'].dropna().unique()])[:7]
            
            temporal[var] = {
                'quarters': [str(q) for q in quarters],
                'SEG_A': [],
                'SEG_B': [],
                'SEG_C': []
            }
            
            for q in quarters:
                q_data = self.df[self.df['quarter'] == q]
                for seg in ['SEG_A', 'SEG_B', 'SEG_C']:
                    seg_data = q_data[q_data['ATSEG'] == seg][var]
                    if len(seg_data) > 0:
                        temporal[var][seg].append(round(seg_data.mean(), 4))
                    else:
                        temporal[var][seg].append(0)
        
        return temporal
        
    def _calc_descriptive(self):
        """Estadisticos descriptivos"""
        vars_desc = ['UC_TRX', 'ORAL_TRX', 'IL23_TRX', 'BRAND1_TRX', 'BRAND2_TRX',
                    'N_CLMOTHERS', 'N_CLMBRAND3', 'DETAILS', 'SAMPLES',
                    'BRAND1_T_GIDX', 'BRAND2_T_GIDX']
        
        desc = []
        for var in vars_desc:
            if var not in self.hcp_df.columns:
                continue
            
            col = self.hcp_df[var]
            zeros_pct = (col == 0).sum() / len(col) * 100
            
            desc.append([
                var,
                round(col.mean(), 2),
                round(col.median(), 2),
                round(col.std(), 2),
                round(col.min(), 1),
                round(col.quantile(0.25), 2),
                round(col.quantile(0.75), 2),
                round(col.max(), 1),
                round(zeros_pct, 1),
                round(col.skew(), 2)
            ])
        
        return desc
        
    def generate_html(self):
        """Genera HTML completo"""
        print("[3/3] Generando HTML...")
        
        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EDA Pre-Transformacion - Velsipity HCP Segmentation</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
{self._get_css()}
</head>
<body>

{self._get_hero()}
{self._get_stats_strip()}

<div class="container">
{self._get_section_target()}
{self._get_section_profiles()}
{self._get_section_comparativa()}
{self._get_section_sparsity()}
{self._get_section_demographics()}
{self._get_section_correlation()}
{self._get_section_temporal()}
{self._get_section_descriptive()}
</div>

{self._get_footer()}
{self._get_javascript()}

</body>
</html>"""
        
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"      Guardado: {self.output_path}")
        
    def _get_css(self):
        """CSS completo"""
        return """<style>
:root{--pf-blue:#0033A0;--pf-blue2:#0079C1;--pf-cyan:#00A3E0;--pf-sky:#B4D8E7;--pf-pale:#E8F4FA;--pf-dark:#001A4D;--pf-white:#FFFFFF;--pf-gray:#F5F7FA;--pf-border:#D0DCE8;--pf-text:#1A2A3A;--pf-textl:#5A6A7A;--pf-green:#00875A;--pf-orange:#E87722;--pf-red:#D0021B;--pf-purple:#6E3FA3}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Noto Sans',sans-serif;background:var(--pf-gray);color:var(--pf-text);line-height:1.6}
.hero{background:linear-gradient(135deg,var(--pf-dark) 0%,var(--pf-blue) 50%,var(--pf-blue2) 100%);padding:50px 40px 40px;color:#fff;position:relative;overflow:hidden}
.hero::before{content:'';position:absolute;top:-80px;right:-80px;width:400px;height:400px;background:radial-gradient(circle,rgba(0,163,224,.25) 0%,transparent 70%);border-radius:50%}
.hero-content{position:relative;z-index:1;max-width:1280px;margin:0 auto}
.hero-badge{display:inline-block;background:rgba(255,255,255,.12);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.18);padding:5px 14px;border-radius:16px;font-size:11px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:16px}
.hero h1{font-size:38px;font-weight:900;margin-bottom:8px;letter-spacing:-.5px}
.hero p{font-size:15px;font-weight:300;opacity:.85;max-width:700px}
.hero .tag{display:inline-block;margin-top:12px;background:rgba(255,255,255,.2);padding:4px 12px;border-radius:12px;font-size:12px;font-weight:600}
.container{max-width:1280px;margin:0 auto;padding:24px 24px 60px}
.stats-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:1px;background:var(--pf-border);margin:0 auto;max-width:1280px;border-radius:12px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,.07);transform:translateY(-24px);position:relative;z-index:2}
.stat-card{background:#fff;padding:20px 16px;text-align:center}
.stat-card .num{font-size:24px;font-weight:800;color:var(--pf-blue)}
.stat-card .lbl{font-size:10px;color:var(--pf-textl);margin-top:3px;font-weight:600;text-transform:uppercase;letter-spacing:.5px}
.section{margin-top:40px}
.section-head{display:flex;align-items:center;gap:12px;margin-bottom:20px;padding-bottom:10px;border-bottom:2px solid var(--pf-blue)}
.section-head .ico{width:36px;height:36px;background:var(--pf-blue);border-radius:8px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:16px}
.section-head h2{font-size:20px;font-weight:800;color:var(--pf-dark)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}
.card{background:#fff;border-radius:12px;padding:24px;box-shadow:0 2px 10px rgba(0,0,0,.04)}
.card h3{font-size:14px;font-weight:700;color:var(--pf-dark);margin-bottom:14px}
.chart-box{position:relative;height:260px;width:100%}
.chart-box.tall{height:380px}
table{width:100%;border-collapse:collapse;font-size:12px}
thead th{background:var(--pf-pale);padding:10px 12px;text-align:left;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.7px;color:var(--pf-blue);border-bottom:2px solid var(--pf-sky)}
tbody td{padding:9px 12px;border-bottom:1px solid #F0F3F6}
tbody tr:hover{background:#FAFCFE}
td.col-name{font-weight:700;font-family:'Courier New',monospace;color:var(--pf-blue);font-size:11px}
.insight{background:linear-gradient(135deg,var(--pf-pale),#fff);border:1px solid var(--pf-sky);border-radius:10px;padding:18px 20px;margin-top:16px;font-size:13px}
.insight strong{color:var(--pf-blue)}
.tag-insight{display:inline-block;background:var(--pf-blue);color:#fff;font-size:9px;font-weight:700;padding:2px 8px;border-radius:10px;text-transform:uppercase;margin-right:6px}
.sparse-row{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.sparse-name{width:170px;text-align:right;font-size:11px;font-family:'Courier New',monospace;color:var(--pf-blue);font-weight:600}
.sparse-bar-bg{flex:1;height:16px;background:#EDF2F7;border-radius:4px;overflow:hidden}
.sparse-bar{height:100%;border-radius:4px;transition:width .8s}
.sparse-pct{width:45px;font-size:11px;color:var(--pf-textl);font-weight:600;text-align:right}
.legend{display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:6px;font-size:12px;font-weight:600}
.legend-dot{width:12px;height:12px;border-radius:3px}
.footer{text-align:center;padding:24px;font-size:11px;color:var(--pf-textl);border-top:1px solid var(--pf-border);margin-top:40px}
</style>"""
    
    def _get_hero(self):
        return """<div class="hero">
  <div class="hero-content">
    <div class="hero-badge">Pfizer Data Science · Capstone 2026</div>
    <h1>Exploratory Data Analysis</h1>
    <p>Analisis exploratorio pre-transformacion del dataset Velsipity HCP Segmentation.</p>
    <span class="tag">Fase 1 · Datos sin transformar · Nivel HCP × Semana</span>
  </div>
</div>"""
    
    def _get_stats_strip(self):
        s = self.data['stats']
        t = self.data['target']
        total_hcps = s['unique_hcps']
        labeled = t['seg_a'] + t['seg_b'] + t['seg_c']
        unlabeled = t['unlabeled']
        pct_nan = (unlabeled / total_hcps * 100) if total_hcps > 0 else 0
        
        return f"""<div class="stats-strip">
  <div class="stat-card"><div class="num">{format_number(s['total_rows'])}</div><div class="lbl">Filas</div></div>
  <div class="stat-card"><div class="num">{format_number(s['unique_hcps'])}</div><div class="lbl">HCPs</div></div>
  <div class="stat-card"><div class="num">{s['weeks']}</div><div class="lbl">Semanas</div></div>
  <div class="stat-card"><div class="num">{s['total_cols']}</div><div class="lbl">Variables</div></div>
  <div class="stat-card"><div class="num">{format_number(labeled)}</div><div class="lbl">Etiquetados</div></div>
  <div class="stat-card"><div class="num">{format_number(unlabeled)}</div><div class="lbl">Sin etiquetar</div></div>
  <div class="stat-card"><div class="num">{pct_nan:.0f}%</div><div class="lbl">Target NaN</div></div>
</div>"""
    
    def _get_section_target(self):
        t = self.data['target']
        total = t['seg_a'] + t['seg_b'] + t['seg_c'] + t['unlabeled']
        pct_labeled = ((t['seg_a'] + t['seg_b'] + t['seg_c']) / total * 100) if total > 0 else 0
        pct_a = (t['seg_a'] / (t['seg_a'] + t['seg_b'] + t['seg_c']) * 100) if (t['seg_a'] + t['seg_b'] + t['seg_c']) > 0 else 0
        pct_b = (t['seg_b'] / (t['seg_a'] + t['seg_b'] + t['seg_c']) * 100) if (t['seg_a'] + t['seg_b'] + t['seg_c']) > 0 else 0
        pct_c = (t['seg_c'] / (t['seg_a'] + t['seg_b'] + t['seg_c']) * 100) if (t['seg_a'] + t['seg_b'] + t['seg_c']) > 0 else 0
        
        return f"""<div class="section">
  <div class="section-head">
    <div class="ico">TGT</div>
    <h2>Distribucion de la Variable Objetivo (ATSEG)</h2>
  </div>
  <div class="grid2">
    <div class="card">
      <h3>Distribucion de Segmentos (HCPs)</h3>
      <div class="chart-box"><canvas id="segPie"></canvas></div>
    </div>
    <div class="card">
      <h3>Labeled vs Unlabeled</h3>
      <div class="chart-box"><canvas id="labeledBar"></canvas></div>
      <div class="insight">
        <span class="tag-insight">Hallazgo</span>
        <strong>{pct_labeled:.1f}%</strong> de los HCPs tienen etiqueta. SEG_A domina con {pct_a:.1f}%, seguido de SEG_B ({pct_b:.1f}%) y SEG_C ({pct_c:.1f}%).
      </div>
    </div>
  </div>
</div>"""
    
    def _get_section_profiles(self):
        return """<div class="section">
  <div class="section-head">
    <div class="ico">PRF</div>
    <h2>Perfil por Segmento (Agregado 86 semanas)</h2>
  </div>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#0033A0"></div>SEG_A (bajo volumen)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#00A3E0"></div>SEG_B (medio volumen)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#6E3FA3"></div>SEG_C (alto volumen)</div>
  </div>
  <div class="grid2">
    <div class="card"><h3>UC TRx</h3><div class="chart-box"><canvas id="profileUC"></canvas></div></div>
    <div class="card"><h3>Claims Otras Marcas</h3><div class="chart-box"><canvas id="profileCLM"></canvas></div></div>
    <div class="card"><h3>Details</h3><div class="chart-box"><canvas id="profileDET"></canvas></div></div>
    <div class="card"><h3>Brand 1 TRx</h3><div class="chart-box"><canvas id="profileB1"></canvas></div></div>
  </div>
</div>"""
    
    def _get_section_comparativa(self):
        table_rows = ""
        for row in self.data['comparativa']:
            table_rows += f"""<tr>
      <td class="col-name">{row['var']}</td>
      <td>{row.get('SEG_A', 0):.2f}</td>
      <td>{row.get('SEG_B', 0):.2f}</td>
      <td>{row.get('SEG_C', 0):.2f}</td>
    </tr>"""
        
        return f"""<div class="section">
  <div class="section-head">
    <div class="ico">CMP</div>
    <h2>Comparativa de Segmentos - Variables Clave</h2>
  </div>
  <div class="card">
    <table>
      <thead><tr><th>Variable</th><th>SEG_A</th><th>SEG_B</th><th>SEG_C</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>
</div>"""
    
    def _get_section_sparsity(self):
        return """<div class="section">
  <div class="section-head">
    <div class="ico">SPA</div>
    <h2>Sparsity - Variables con Mayor % de Ceros</h2>
  </div>
  <div class="card">
    <div id="sparsityBars"></div>
  </div>
</div>"""
    
    def _get_section_demographics(self):
        # Solo mostrar si hay datos
        if not self.data['demographics']['specialty'] and not self.data['demographics']['age']:
            return ""
        
        return """<div class="section">
  <div class="section-head">
    <div class="ico">DEM</div>
    <h2>Demografia del HCP por Segmento</h2>
  </div>
  <div class="card">
    <p style="text-align:center;padding:40px;color:var(--pf-textl)">Datos demograficos no disponibles en dataset</p>
  </div>
</div>"""
    
    def _get_section_correlation(self):
        return """<div class="section">
  <div class="section-head">
    <div class="ico">COR</div>
    <h2>Matriz de Correlacion</h2>
  </div>
  <div class="card">
    <div class="chart-box tall"><canvas id="corrChart"></canvas></div>
  </div>
</div>"""
    
    def _get_section_temporal(self):
        if not self.data['temporal']:
            return ""
        
        return """<div class="section">
  <div class="section-head">
    <div class="ico">TMP</div>
    <h2>Tendencias Temporales por Segmento</h2>
  </div>
  <div class="grid2">
    <div class="card"><h3>UC TRx por Trimestre</h3><div class="chart-box"><canvas id="tempUC"></canvas></div></div>
    <div class="card"><h3>Brand 1 TRx por Trimestre</h3><div class="chart-box"><canvas id="tempB1"></canvas></div></div>
  </div>
</div>"""
    
    def _get_section_descriptive(self):
        return """<div class="section">
  <div class="section-head">
    <div class="ico">DSC</div>
    <h2>Estadisticos Descriptivos</h2>
  </div>
  <div class="card">
    <table>
      <thead><tr><th>Variable</th><th>Media</th><th>Mediana</th><th>Std</th><th>Min</th><th>Q25</th><th>Q75</th><th>Max</th><th>Ceros %</th><th>Skew</th></tr></thead>
      <tbody id="descTable"></tbody>
    </table>
  </div>
</div>"""
    
    def _get_footer(self):
        return """<div class="footer">
  <p>Pfizer Data Science Team · Tec de Monterrey Capstone Project 2026</p>
  <p>"Breakthroughs that change patients' lives"</p>
</div>"""
    
    def _get_javascript(self):
        """JavaScript completo con TODOS los datos"""
        
        t = self.data['target']
        prof = self.data['profiles']
        corr = self.data['correlation']
        temp = self.data['temporal']
        spar = self.data['sparsity']
        desc = self.data['descriptive']
        
        # Profiles
        uc = prof.get('UC_TRX', {'SEG_A': [0,0], 'SEG_B': [0,0], 'SEG_C': [0,0]})
        clm = prof.get('N_CLMOTHERS', {'SEG_A': [0,0], 'SEG_B': [0,0], 'SEG_C': [0,0]})
        det = prof.get('DETAILS', {'SEG_A': [0,0], 'SEG_B': [0,0], 'SEG_C': [0,0]})
        b1 = prof.get('BRAND1_TRX', {'SEG_A': [0,0], 'SEG_B': [0,0], 'SEG_C': [0,0]})
        
        return f"""<script>
const SEGA='#0033A0', SEGB='#00A3E0', SEGC='#6E3FA3', SEGUN='#B0BEC5';
Chart.defaults.font.family = "'Noto Sans', sans-serif";
Chart.defaults.font.size = 11;

// TARGET: Pie
new Chart(document.getElementById('segPie'),{{
  type:'doughnut',
  data:{{
    labels:['SEG_A ({t['seg_a']:,})','SEG_B ({t['seg_b']:,})','SEG_C ({t['seg_c']:,})','Sin etiquetar ({t['unlabeled']:,})'],
    datasets:[{{data:[{t['seg_a']},{t['seg_b']},{t['seg_c']},{t['unlabeled']}],backgroundColor:[SEGA,SEGB,SEGC,SEGUN],borderWidth:2,borderColor:'#fff'}}]
  }},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{position:'bottom',labels:{{padding:12,usePointStyle:true,pointStyle:'rectRounded'}}}}}}}}
}});

// TARGET: Bar
new Chart(document.getElementById('labeledBar'),{{
  type:'bar',
  data:{{
    labels:['SEG_A','SEG_B','SEG_C','Sin etiqueta'],
    datasets:[{{data:[{t['seg_a']},{t['seg_b']},{t['seg_c']},{t['unlabeled']}],backgroundColor:[SEGA,SEGB,SEGC,SEGUN],borderRadius:6}}]
  }},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{y:{{beginAtZero:true}}}}}}
}});

// PROFILES
function profileChart(id,data){{
  new Chart(document.getElementById(id),{{
    type:'bar',
    data:{{
      labels:['Media','Mediana'],
      datasets:[
        {{label:'SEG_A',data:[data.A[0],data.A[1]],backgroundColor:SEGA,borderRadius:4}},
        {{label:'SEG_B',data:[data.B[0],data.B[1]],backgroundColor:SEGB,borderRadius:4}},
        {{label:'SEG_C',data:[data.C[0],data.C[1]],backgroundColor:SEGC,borderRadius:4}}
      ]
    }},
    options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{position:'top'}}}},scales:{{y:{{beginAtZero:true}}}}}}
  }});
}}
profileChart('profileUC',{{A:{uc['SEG_A']},B:{uc['SEG_B']},C:{uc['SEG_C']}}});
profileChart('profileCLM',{{A:{clm['SEG_A']},B:{clm['SEG_B']},C:{clm['SEG_C']}}});
profileChart('profileDET',{{A:{det['SEG_A']},B:{det['SEG_B']},C:{det['SEG_C']}}});
profileChart('profileB1',{{A:{b1['SEG_A']},B:{b1['SEG_B']},C:{b1['SEG_C']}}});

// CORRELATION
const corrLabels={json.dumps(corr['labels'])};
const corrData={json.dumps(corr['data'])};
(function(){{
  const canvas=document.getElementById('corrChart');
  if(!canvas)return;
  const ctx=canvas.getContext('2d');
  const n=corrLabels.length;
  function draw(){{
    const W=canvas.width,H=canvas.height;
    const m={{l:75,t:70,r:20,b:20}};
    const cW=(W-m.l-m.r)/n, cH=(H-m.t-m.b)/n;
    ctx.clearRect(0,0,W,H);
    for(let i=0;i<n;i++){{
      for(let j=0;j<n;j++){{
        const v=corrData[i][j];
        const x=m.l+j*cW, y=m.t+i*cH;
        let r,g,b;
        if(v>=0){{r=Math.round(255-v*255);g=Math.round(255-v*200);b=255;}}
        else{{r=255;g=Math.round(255+v*200);b=Math.round(255+v*255);}}
        ctx.fillStyle=`rgb(${{r}},${{g}},${{b}})`;
        ctx.fillRect(x+1,y+1,cW-2,cH-2);
        ctx.fillStyle=Math.abs(v)>0.6?'#fff':'#333';
        ctx.font='9px "Noto Sans"';
        ctx.textAlign='center';
        ctx.textBaseline='middle';
        if(i!==j)ctx.fillText(v.toFixed(2),x+cW/2,y+cH/2);
      }}
    }}
    ctx.fillStyle='#1A2A3A';
    ctx.font='9px "Courier New"';
    for(let i=0;i<n;i++){{
      ctx.textAlign='right';
      ctx.textBaseline='middle';
      ctx.fillText(corrLabels[i],m.l-4,m.t+i*cH+cH/2);
      ctx.save();
      ctx.translate(m.l+i*cW+cW/2,m.t-4);
      ctx.rotate(-Math.PI/4);
      ctx.textAlign='left';
      ctx.textBaseline='middle';
      ctx.fillText(corrLabels[i],0,0);
      ctx.restore();
    }}
  }}
  new ResizeObserver(()=>{{canvas.width=canvas.parentElement.clientWidth;canvas.height=canvas.parentElement.clientHeight;draw();}}).observe(canvas.parentElement);
  setTimeout(()=>{{canvas.width=canvas.parentElement.clientWidth;canvas.height=canvas.parentElement.clientHeight;draw();}},100);
}})();

// TEMPORAL
{self._get_temporal_js()}

// SPARSITY
const sparData={json.dumps(spar)};
const sparDiv=document.getElementById('sparsityBars');
if(sparDiv){{
  sparData.forEach(item=>{{
    const color=item.pct>90?'#D0021B':item.pct>75?'#E87722':item.pct>50?'#FFB74D':'#00875A';
    sparDiv.innerHTML+=`<div class="sparse-row"><div class="sparse-name">${{item.name}}</div><div class="sparse-bar-bg"><div class="sparse-bar" style="width:${{item.pct}}%;background:${{color}}"></div></div><div class="sparse-pct">${{item.pct}}%</div></div>`;
  }});
}}

// DESCRIPTIVE
const descData={json.dumps(desc)};
const tbody=document.getElementById('descTable');
if(tbody){{
  descData.forEach(([name,mean,med,std,min,q25,q75,max,zeros,skew])=>{{
    const skewColor=skew>5?'var(--pf-red)':skew>2?'var(--pf-orange)':'var(--pf-green)';
    tbody.innerHTML+=`<tr><td class="col-name">${{name}}</td><td>${{mean.toFixed(2)}}</td><td>${{med.toFixed(2)}}</td><td>${{std.toFixed(2)}}</td><td>${{min.toFixed(1)}}</td><td>${{q25.toFixed(2)}}</td><td>${{q75.toFixed(2)}}</td><td>${{max.toFixed(1)}}</td><td>${{zeros}}%</td><td style="color:${{skewColor}};font-weight:700">${{skew.toFixed(1)}}</td></tr>`;
  }});
}}
</script>"""
    
    def _get_temporal_js(self):
        """JavaScript para gráficas temporales"""
        temp = self.data['temporal']
        
        if not temp:
            return "// No temporal data"
        
        js = ""
        
        for var, var_id in [('UC_TRX', 'tempUC'), ('BRAND1_TRX', 'tempB1')]:
            if var in temp:
                data = temp[var]
                quarters = json.dumps(data['quarters'])
                seg_a = json.dumps(data['SEG_A'])
                seg_b = json.dumps(data['SEG_B'])
                seg_c = json.dumps(data['SEG_C'])
                
                js += f"""
if(document.getElementById('{var_id}')){{
  new Chart(document.getElementById('{var_id}'),{{
    type:'line',
    data:{{
      labels:{quarters},
      datasets:[
        {{label:'SEG_A',data:{seg_a},borderColor:SEGA,backgroundColor:SEGA+'22',fill:true,tension:.3,borderWidth:2}},
        {{label:'SEG_B',data:{seg_b},borderColor:SEGB,backgroundColor:SEGB+'22',fill:true,tension:.3,borderWidth:2}},
        {{label:'SEG_C',data:{seg_c},borderColor:SEGC,backgroundColor:SEGC+'22',fill:true,tension:.3,borderWidth:2}}
      ]
    }},
    options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{position:'top'}}}},scales:{{y:{{beginAtZero:true}}}}}}
  }});
}}
"""
        
        return js


if __name__ == "__main__":
    DATA_PATH = "data/raw/VELSIPITY_AFF_MULTIPLI_prepared_prepared_prepared.csv"
    OUTPUT_PATH = "reports/eda_pre_transformacion.html"
    
    generator = EDAGeneratorFinal(DATA_PATH, OUTPUT_PATH)
    generator.run()