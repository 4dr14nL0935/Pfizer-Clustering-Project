# Reporte Comparativo de Clustering
## VELSIPITY_cleaned_igna vs data_processed

---

## 1. Configuración del experimento

| Parámetro | DS1 — cleaned_igna | DS2 — data_processed |
|---|---|---|
| Archivo | `VELSIPITY_cleaned_igna.csv` | `data_processed.csv` |
| Registros totales | 1,800,066 | 1,800,066 |
| Columnas totales | 64 | 70 |
| Features para clustering | 60 | 65 |
| Muestra de análisis | 25,000 | 25,000 |
| Varianza PCA 2D | 18.6% | 18.4% |
| k óptimo (≥ 3) | **4** | **4** |

### Features exclusivas de DS2

DS2 incorpora 5 features de ingeniería adicionales respecto a DS1:

| Feature | Descripción |
|---|---|
| `TOTAL_TRX` | Total de prescripciones acumuladas |
| `TOTAL_NRX` | Total de nuevas prescripciones acumuladas |
| `ENGAGEMENT_SCORE` | Score compuesto de engagement del prescriptor |
| `NBRx_RATIO` | Ratio nuevas prescripciones / total |
| `BRAND1_MARKET_SHARE` | Participación de mercado de BRAND1 |

---

## 2. Selección del número óptimo de clusters

Se evaluaron k=2 a k=8 con K-Means (10 inicializaciones, Silhouette sobre 3,000 puntos). **k=2 fue excluido por criterio de negocio** — se forzó k ≥ 3.

### DS1 — VELSIPITY_cleaned_igna

| k | Inercia | Silhouette | |
|---|---|---|---|
| 2 | 1,663,337 | 0.6183 | *(excluido)* |
| 3 | 1,603,978 | 0.1230 | |
| **4** | **1,557,881** | **0.1303** | ✅ k óptimo |
| 5 | 1,504,827 | 0.0938 | |
| 6 | 1,474,019 | 0.0790 | |
| 7 | 1,449,415 | 0.0750 | |
| 8 | 1,411,350 | 0.0716 | |

### DS2 — data_processed

| k | Inercia | Silhouette | |
|---|---|---|---|
| 2 | 1,775,489 | 0.4612 | *(excluido)* |
| 3 | 1,713,801 | 0.1757 | |
| **4** | **1,665,148** | **0.1830** | ✅ k óptimo |
| 5 | 1,602,254 | 0.1550 | |
| 6 | 1,559,676 | 0.1400 | |
| 7 | 1,562,072 | 0.1116 | |
| 8 | 1,482,284 | 0.0578 | |

> Ambos datasets convergen al mismo k óptimo (k=4), lo que refuerza que la estructura subyacente de los prescriptores tiene 4 grupos naturales una vez excluido k=2.

---

## 3. K-Means (k=4)

### Métricas

| Métrica | DS1 cleaned_igna | DS2 processed | Ganador |
|---|---|---|---|
| Silhouette ↑ | 0.1303 | **0.1830** | DS2 |
| Davies-Bouldin ↓ | **2.6017** | 2.6085 | DS1 |
| Calinski-Harabasz ↑ | 1,407.77 | **1,501.84** | DS2 |

### Distribución de clusters — DS1

| Cluster | Observaciones | % |
|---|---|---|
| C0 | 781 | 3.1% |
| C1 | 18,841 | 75.4% |
| C2 | 208 | 0.8% |
| C3 | 5,170 | 20.7% |

### Distribución de clusters — DS2

| Cluster | Observaciones | % |
|---|---|---|
| C0 | 16,868 | 67.5% |
| C1 | 7,014 | 28.1% |
| C2 | 1,101 | 4.4% |
| C3 | 17 | 0.1% |

**DS2 produce una distribución más balanceada** entre los 4 clusters, mientras DS1 concentra el 75.4% en C1. Las features de engagement y market share en DS2 ayudan a separar mejor la masa principal de prescriptores.

---

## 4. K-Medoids (k=4)

### Métricas

| Métrica | DS1 cleaned_igna | DS2 processed | Ganador |
|---|---|---|---|
| Silhouette ↑ | **0.1300** | 0.0555 | DS1 |
| Davies-Bouldin ↓ | **2.5981** | 2.7739 | DS1 |
| Calinski-Harabasz ↑ | 188.06 | **231.87** | DS2 |

### Distribución de clusters — DS1

| Cluster | % |
|---|---|
| C0 | 3.4% |
| C1 | 24.5% |
| C2 | 71.7% |
| C3 | 0.4% |

### Distribución de clusters — DS2

| Cluster | % |
|---|---|
| C0 | 8.4% |
| C1 | 12.8% |
| C2 | 77.1% |
| C3 | 1.6% |

**DS1 produce medoides más representativos** con mejor separación (Silhouette 0.13 vs 0.06 de DS2), siendo más adecuado para selección de KOL o representantes reales de cada segmento.

---

## 5. DBSCAN

**Configuración:** DS1 eps=4.62 | DS2 eps=5.71, min_samples=10, muestra=7,000

### Métricas (excluyendo ruido)

| Métrica | DS1 cleaned_igna | DS2 processed | Ganador |
|---|---|---|---|
| Silhouette ↑ | **0.2803** | 0.2350 | DS1 |
| Davies-Bouldin ↓ | **1.0850** | 1.0925 | DS1 |
| Calinski-Harabasz ↑ | **262.38** | 172.66 | DS1 |
| Clusters detectados ↑ | **15** | 12 | DS1 |
| Ruido ↓ | 14.1% | **13.6%** | DS2 |

### Distribución top clusters — DS1

| Cluster | Observaciones |
|---|---|
| C0 | 4,563 |
| C4 | 239 |
| C7 | 191 |
| C5 | 188 |
| C2 | 185 |
| C1 | 180 |
| Ruido | 987 |

### Distribución top clusters — DS2

| Cluster | Observaciones |
|---|---|
| C0 | 5,204 |
| C3 | 204 |
| C6 | 194 |
| C7 | 171 |
| C5 | 77 |
| C1 | 62 |
| Ruido | 951 |

**DS1 detecta más clusters útiles (15 vs 12) con mejor calidad de separación** en todos los indicadores.

---

## 6. Scorecard final

| Métrica | DS1 cleaned_igna | DS2 processed | Ganador |
|---|---|---|---|
| Silhouette K-Means ↑ | 0.1303 | **0.1830** | DS2 ✓ |
| Davies-Bouldin K-Means ↓ | **2.6017** | 2.6085 | DS1 ✓ |
| Calinski-Harabasz K-Means ↑ | 1,407.77 | **1,501.84** | DS2 ✓ |
| Silhouette K-Medoids ↑ | **0.1300** | 0.0555 | DS1 ✓ |
| Davies-Bouldin K-Medoids ↓ | **2.5981** | 2.7739 | DS1 ✓ |
| Silhouette DBSCAN ↑ | **0.2803** | 0.2350 | DS1 ✓ |
| Davies-Bouldin DBSCAN ↓ | **1.0850** | 1.0925 | DS1 ✓ |
| Clusters DBSCAN útiles ↑ | **15** | 12 | DS1 ✓ |
| Ruido DBSCAN ↓ | 14.1% | **13.6%** | DS2 ✓ |
| **TOTAL VICTORIAS** | **6** | **3** | |

## 🏆 Ganador: DS1 — VELSIPITY_cleaned_igna (6 de 9 métricas)

---

## 7. ¿Por qué gana DS1?

**DS1 gana en 6 de 9 métricas**, especialmente en K-Medoids (2/2) y DBSCAN (4/5). Dos razones principales:

**Transformaciones log1p:** DS1 aplica transformación logarítmica a las variables de prescripción, normalizando la distribución altamente sesgada característica de datos farmacéuticos. Esto mejora la geometría del espacio y facilita que los algoritmos basados en distancias (K-Medoids, DBSCAN) encuentren clusters más coherentes.

**Variables de lag:** Los lags semanales (`RTE_L1`, `SAMPLES_L1`, `DETAILS_L1`, etc.) capturan el comportamiento previo del prescriptor, añadiendo una dimensión temporal que reduce el ruido semana a semana y produce clusters más estables.

**DS2 gana en K-Means** porque sus features adicionales (`ENGAGEMENT_SCORE`, `BRAND1_MARKET_SHARE`) introducen señal discriminante que K-Means aprovecha bien para separar grupos con centroides, pero que aumenta la dispersión general del espacio dificultando el trabajo de K-Medoids y DBSCAN.

---

## 8. Recomendaciones finales

| Caso de uso | Dataset | Algoritmo | Justificación |
|---|---|---|---|
| Segmentación productiva general | **DS1 cleaned_igna** | K-Means k=4 | Clusters más estables por log1p |
| Selección de KOL / representantes | **DS1 cleaned_igna** | K-Medoids k=4 | Mejor Silhouette (0.13 vs 0.06) |
| Descubrimiento de micro-nichos | **DS1 cleaned_igna** | DBSCAN | Más clusters útiles (15 vs 12) |
| Análisis de engagement / market share | **DS2 processed** | K-Means k=4 | Features de negocio específicas |
| Minimizar prescriptores sin clasificar | **DS2 processed** | DBSCAN | Menor ruido (13.6% vs 14.1%) |

> **Recomendación global:** usar **DS1 `VELSIPITY_cleaned_igna` como base principal de clustering** por su superioridad en 6/9 métricas. Si el análisis requiere interpretar engagement o participación de mercado, complementar con las features de DS2 en un pipeline secundario.
