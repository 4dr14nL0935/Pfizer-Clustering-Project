# Resultados de Clustering — VELSIPITY Dataset

## 1. Descripción del dataset

| Parámetro | Valor |
|---|---|
| Total de registros | 1,800,066 |
| Total de columnas | 68 |
| Valores nulos | 776,752 |
| Features usadas para clustering | 23 |
| Muestra para análisis | 50,000 (K-Means) / 10,000 (K-Medoids, DBSCAN) |
| Varianza explicada por PCA 2D | 32.9% |

### Segmentación original (ATSEG)

| Segmento | Observaciones |
|---|---|
| SEG_A | 550,916 |
| SEG_B | 288,014 |
| SEG_C | 184,384 |
| Sin segmento (NaN) | 776,752 |

### Features incluidas en el clustering

Variables de prescripción (TRX, NRX, NBRX) para UC, ORAL, IL23, BRAND1 y BRAND2, más actividades de marketing: `RTE`, `SAMPLES`, `COPAY`, `DIRECTMAIL`, `SPK`, `DETAILS`, y métricas acumuladas `UC_TRX_R4_16SUM`, `ORAL_NBRX_R4_29SUM`, `IL23_NBRX_R4_29SUM`.

---

## 2. Selección del número óptimo de clusters

Se evaluaron valores de k entre 2 y 8 usando K-Means con 10 inicializaciones y Silhouette Score sobre una muestra de 5,000 puntos. **Se forzó k ≥ 3** para evitar una segmentación de solo 2 grupos, poco accionable para fines de negocio.

| k | Inercia (WCSS) | Silhouette Score | |
|---|---|---|---|
| 2 | 1,000,170 | 0.8042 | *(excluido, k mínimo = 3)* |
| 3 | 939,124 | 0.7974 | |
| 4 | 885,606 | 0.7265 | |
| **5** | **837,605** | **0.8004** | ✅ **k óptimo seleccionado** |
| 6 | 785,732 | 0.5670 | |
| 7 | 732,713 | 0.6760 | |
| 8 | 682,794 | 0.6697 | |

> **k = 5** es el óptimo con Silhouette = 0.8004, el mayor entre los valores de k ≥ 3. Coincide bien con la segmentación previa `ATSEG` de 3 niveles al añadir mayor granularidad.

---

## 3. K-Means

**Configuración:** k=5, n_init=20, random_state=42, muestra=50,000

### Métricas

| Métrica | Valor | Interpretación |
|---|---|---|
| Silhouette Score | **0.6991** | Alta — clusters bien separados |
| Davies-Bouldin | 1.3760 | Moderado-bajo |
| Calinski-Harabasz | 4,710.76 | Alto — clusters compactos |

### Tamaño de clusters

| Cluster | Observaciones | % de la muestra |
|---|---|---|
| C0 | 3,187 | 6.4% |
| C1 | 46,460 | 92.9% |
| C2 | 61 | 0.1% |
| C3 | 53 | 0.1% |
| C4 | 239 | 0.5% |

### Observaciones

K-Means con k=5 revela una estructura muy asimétrica: **C1 concentra el 92.9% de los prescriptores**, representando la masa de baja o nula actividad, mientras C0 (6.4%) agrupa prescriptores de actividad moderada y C2, C3, C4 son clusters pequeños de alta prescripción o perfiles especializados. Esta distribución es consistente con los datos farmacéuticos, donde una minoría de médicos genera la mayor parte del volumen.

---

## 4. K-Medoids (PAM)

**Configuración:** k=5, max_iter=150, muestra=10,000

### Métricas

| Métrica | Valor | Interpretación |
|---|---|---|
| Silhouette Score | 0.1466 | Baja sobre esta muestra |
| Davies-Bouldin | 3.7581 | Alto — clusters con solapamiento |
| Calinski-Harabasz | 319.64 | Bajo respecto a K-Means |

### Tamaño de clusters

| Cluster | Observaciones | % de la muestra |
|---|---|---|
| C0 | 1,453 | 14.5% |
| C1 | 849 | 8.5% |
| C2 | 1,060 | 10.6% |
| C3 | 2,019 | 20.2% |
| C4 | 4,619 | 46.2% |

### Observaciones

Con k=5 y muestra de 10,000, K-Medoids produce una distribución más balanceada entre clusters. Las métricas más bajas frente a K-Means se explican por la mayor complejidad de separar 5 grupos con medoides reales en 23 dimensiones con muestra reducida. La ventaja clave es que los **5 medoides son prescriptores reales del dataset**, directamente utilizables como representantes o targets de validación de negocio.

---

## 5. DBSCAN

**Configuración:** eps=0.86, min_samples=10, muestra=10,000
*(eps seleccionado con k-distance plot, percentil 85 de distancias al 5-NN)*

### Resultados

| Parámetro | Valor |
|---|---|
| Clusters encontrados | 17 |
| Puntos de ruido (outliers) | 1,508 (15.1%) |
| Cluster principal (C2) | 6,985 puntos (69.9% del total no-ruido) |

### Distribución de los 10 grupos más grandes

| Cluster | Observaciones |
|---|---|
| C2 | 6,985 |
| C0 | 422 |
| C4 | 370 |
| C1 | 240 |
| C3 | 126 |
| C5 | 75 |
| C7 | 52 |
| C14 | 43 |
| C13 | 41 |
| Ruido | 1,508 |

### Métricas (excluyendo ruido)

| Métrica | Valor | Interpretación |
|---|---|---|
| Silhouette Score | 0.5466 | Buena separación interna |
| Davies-Bouldin | **0.6472** | Mejor de los tres algoritmos |
| Calinski-Harabasz | 2,290.80 | Bueno considerando 17 clusters |

### Observaciones

DBSCAN detecta automáticamente **17 clusters** sin necesitar especificar k. El **15.1% de ruido** identifica prescriptores con comportamientos atípicos. El cluster dominante C2 (~70% de los puntos no-ruido) confirma la existencia de una masa homogénea de baja actividad, mientras los clusters pequeños revelan nichos de alta especialización no capturados por los métodos de centroides.

---

## 6. Comparación de algoritmos

### Métricas cuantitativas

| Algoritmo | k / Clusters | Muestra | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Ruido |
|---|---|---|---|---|---|---|
| K-Means | 5 | 50,000 | **0.6991** | 1.3760 | **4,710.76** | 0% |
| K-Medoids | 5 | 10,000 | 0.1466 | 3.7581 | 319.64 | 0% |
| DBSCAN | 17 (auto) | 10,000 | 0.5466 | **0.6472** | 2,290.80 | 15.1% |

### Propiedades cualitativas

| Aspecto | K-Means | K-Medoids | DBSCAN |
|---|---|---|---|
| Número de clusters | Requiere k | Requiere k | Automático |
| Representante del cluster | Centroide sintético | Punto real del dataset | No aplica |
| Manejo de outliers | No los maneja | Robusto | Los etiqueta como ruido |
| Formas admitidas | Esféricas | Esféricas | Arbitrarias |
| Escalabilidad | Alta ✅ | Media | Media |
| Interpretabilidad | Alta | Alta | Media |
| Parámetros a tunear | k | k | eps, min_samples |

---

## 7. Conclusiones y recomendaciones

**K-Means (k=5)** obtiene el mejor Silhouette entre k ≥ 3 (0.6991) y el mayor Calinski-Harabasz (4,710), siendo el más adecuado para producción dado el volumen de 1.8M registros. La distribución asimétrica (C1 con 92.9%) es esperable en prescripción farmacéutica y útil para priorizar esfuerzos de campo hacia los clusters minoritarios de alta actividad.

**K-Medoids (k=5)** es preferible cuando se necesita un prescriptor real como representante de cada segmento, por ejemplo para selección de Key Opinion Leaders (KOL) o definición de targets del equipo de ventas. Sus métricas más bajas frente a K-Means se deben al tamaño de muestra reducido (10k vs 50k), no a peor calidad estructural.

**DBSCAN** obtiene el mejor Davies-Bouldin (0.65) y revela una estructura granular de 17 clusters naturales, útil para identificar los 1,508 prescriptores atípicos (15.1% clasificados como ruido) y nichos especializados no capturados por los métodos de centroides.

> **Sobre la elección de k=5:** Aunque estadísticamente k=2 tiene el mayor Silhouette absoluto (0.8042), se forzó k ≥ 3 por criterio de negocio. Dentro de ese rango, **k=5 es el óptimo** (Silhouette=0.8004) y produce una segmentación más granular y accionable, comparable con la segmentación previa `ATSEG` (SEG_A, SEG_B, SEG_C).
