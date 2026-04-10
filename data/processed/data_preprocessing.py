"""
Data Preprocessing - Pfizer HCP Segmentation Project
Este script realiza el procesamiento y limpieza de los datos.

Transformaciones incluidas:
- Manejo de valores faltantes
- Eliminación de features irrelevantes
- Feature engineering
- Normalización y escalado
- Manejo de outliers

Autor: Data Science Team
Fecha: Abril 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Clase para preprocesar datos de HCP para segmentación"""
    
    def __init__(self, input_path, output_path='data/processed/data_processed.csv'):
        """
        Inicializa el preprocesador
        
        Args:
            input_path (str): Ruta al archivo CSV raw
            output_path (str): Ruta donde guardar datos procesados
        """
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.df_processed = None
        self.processing_log = []
        
    def log(self, message):
        """Registra un mensaje en el log de procesamiento"""
        timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.processing_log.append(log_message)
        
    def load_data(self):
        """Carga los datos raw"""
        self.log("📊 Cargando datos raw...")
        self.df = pd.read_csv(self.input_path)
        self.df_processed = self.df.copy()
        self.log(f"✅ Datos cargados: {self.df.shape[0]:,} filas, {self.df.shape[1]} columnas")
        
    def handle_id_columns(self):
        """Identifica y maneja columnas de ID"""
        self.log("\n🔑 Procesando columnas de identificación...")
        
        # Identificar columnas de ID
        id_columns = [col for col in self.df_processed.columns if col.endswith('_ID') or col in ['NUEVO_ID', 'WEEK_ID']]
        
        self.log(f"   Columnas de ID encontradas: {len(id_columns)}")
        self.log(f"   Estas columnas se mantendrán pero no se usarán para modelado")
        
        # Guardar para referencia posterior
        self.id_columns = id_columns
        
    def handle_missing_values(self):
        """Maneja valores faltantes de manera inteligente"""
        self.log("\n🔍 Manejando valores faltantes...")
        
        # Analizar valores faltantes antes
        missing_before = self.df_processed.isnull().sum().sum()
        self.log(f"   Valores faltantes antes: {missing_before:,}")
        
        # Estrategia 1: Para columnas numéricas con pocas faltantes (<5%), imputar con mediana
        numeric_cols = self.df_processed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in self.id_columns:
                continue
                
            missing_pct = (self.df_processed[col].isnull().sum() / len(self.df_processed)) * 100
            
            if 0 < missing_pct < 5:
                median_value = self.df_processed[col].median()
                self.df_processed[col].fillna(median_value, inplace=True)
                self.log(f"   ✓ {col}: Imputado con mediana ({missing_pct:.2f}% faltantes)")
            elif missing_pct >= 5 and missing_pct < 30:
                # Para más del 5% pero menos del 30%, usar 0 (asumiendo que significa "sin actividad")
                self.df_processed[col].fillna(0, inplace=True)
                self.log(f"   ✓ {col}: Imputado con 0 ({missing_pct:.2f}% faltantes)")
            elif missing_pct >= 30:
                self.log(f"   ⚠️ {col}: {missing_pct:.2f}% faltantes - considerar eliminar")
        
        # Estrategia 2: Para columnas categóricas, crear categoría 'Unknown'
        categorical_cols = self.df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            missing_pct = (self.df_processed[col].isnull().sum() / len(self.df_processed)) * 100
            if missing_pct > 0:
                self.df_processed[col].fillna('Unknown', inplace=True)
                self.log(f"   ✓ {col}: Categoría 'Unknown' creada ({missing_pct:.2f}% faltantes)")
        
        # Analizar valores faltantes después
        missing_after = self.df_processed.isnull().sum().sum()
        self.log(f"   Valores faltantes después: {missing_after:,}")
        self.log(f"   Reducción: {missing_before - missing_after:,} valores imputados")
        
    def remove_constant_features(self):
        """Elimina features con valores constantes"""
        self.log("\n🗑️ Eliminando features constantes...")
        
        constant_cols = []
        for col in self.df_processed.columns:
            if col in self.id_columns or col == 'SEGMENT':
                continue
            if self.df_processed[col].nunique() == 1:
                constant_cols.append(col)
        
        if constant_cols:
            self.df_processed.drop(columns=constant_cols, inplace=True)
            self.log(f"   ✓ Eliminadas {len(constant_cols)} columnas constantes")
            self.log(f"   Columnas eliminadas: {', '.join(constant_cols[:10])}")
        else:
            self.log("   ✓ No se encontraron columnas constantes")
            
    def remove_high_correlation(self, threshold=0.95):
        """Elimina features altamente correlacionadas"""
        self.log(f"\n🔗 Eliminando features con correlación > {threshold}...")
        
        # Seleccionar solo columnas numéricas (excluyendo IDs y SEGMENT)
        numeric_cols = [col for col in self.df_processed.select_dtypes(include=[np.number]).columns 
                       if col not in self.id_columns and col != 'SEGMENT']
        
        if len(numeric_cols) < 2:
            self.log("   ⚠️ No hay suficientes columnas numéricas para analizar correlación")
            return
        
        # Calcular matriz de correlación
        corr_matrix = self.df_processed[numeric_cols].corr().abs()
        
        # Encontrar pares altamente correlacionados
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > threshold)]
        
        if to_drop:
            self.df_processed.drop(columns=to_drop, inplace=True)
            self.log(f"   ✓ Eliminadas {len(to_drop)} columnas altamente correlacionadas")
            if len(to_drop) <= 20:
                self.log(f"   Columnas eliminadas: {', '.join(to_drop)}")
        else:
            self.log(f"   ✓ No se encontraron columnas con correlación > {threshold}")
            
    def handle_outliers(self, method='clip', n_std=3):
        """
        Maneja outliers en variables numéricas
        
        Args:
            method (str): 'clip' para recortar, 'remove' para eliminar filas
            n_std (int): Número de desviaciones estándar para definir outliers
        """
        self.log(f"\n📊 Manejando outliers (método: {method}, {n_std} std)...")
        
        numeric_cols = [col for col in self.df_processed.select_dtypes(include=[np.number]).columns 
                       if col not in self.id_columns and col != 'SEGMENT']
        
        outliers_count = 0
        
        for col in numeric_cols:
            mean = self.df_processed[col].mean()
            std = self.df_processed[col].std()
            
            lower_bound = mean - (n_std * std)
            upper_bound = mean + (n_std * std)
            
            # Contar outliers
            col_outliers = ((self.df_processed[col] < lower_bound) | 
                          (self.df_processed[col] > upper_bound)).sum()
            
            if col_outliers > 0:
                outliers_count += col_outliers
                
                if method == 'clip':
                    self.df_processed[col] = self.df_processed[col].clip(lower=lower_bound, upper=upper_bound)
        
        self.log(f"   ✓ Outliers detectados y tratados: {outliers_count:,}")
        
    def feature_engineering(self):
        """Crea nuevas features relevantes para segmentación"""
        self.log("\n🔧 Creando nuevas features...")
        
        # Feature 1: Total TRx (prescripciones totales)
        trx_cols = [col for col in self.df_processed.columns if 'TRX' in col and col not in self.id_columns]
        if trx_cols:
            self.df_processed['TOTAL_TRX'] = self.df_processed[trx_cols].sum(axis=1)
            self.log(f"   ✓ Creada: TOTAL_TRX (suma de {len(trx_cols)} columnas)")
        
        # Feature 2: Total NRx (nuevas prescripciones)
        nrx_cols = [col for col in self.df_processed.columns if 'NRX' in col and col not in self.id_columns]
        if nrx_cols:
            self.df_processed['TOTAL_NRX'] = self.df_processed[nrx_cols].sum(axis=1)
            self.log(f"   ✓ Creada: TOTAL_NRX (suma de {len(nrx_cols)} columnas)")
        
        # Feature 3: Engagement Score (combinación de diferentes tipos de engagement)
        engagement_cols = [col for col in self.df_processed.columns 
                          if any(x in col for x in ['DETAIL', 'SAMPLE', 'SPEAKER', 'MAIL', 'COPAY'])]
        if engagement_cols:
            self.df_processed['ENGAGEMENT_SCORE'] = self.df_processed[engagement_cols].sum(axis=1)
            self.log(f"   ✓ Creada: ENGAGEMENT_SCORE (suma de {len(engagement_cols)} columnas)")
        
        # Feature 4: Ratio NBRx (new-to-brand vs total prescriptions)
        if 'TOTAL_TRX' in self.df_processed.columns and 'BRAND1_NBRX' in self.df_processed.columns:
            self.df_processed['NBRx_RATIO'] = np.where(
                self.df_processed['TOTAL_TRX'] > 0,
                self.df_processed['BRAND1_NBRX'] / self.df_processed['TOTAL_TRX'],
                0
            )
            self.log("   ✓ Creada: NBRx_RATIO (BRAND1_NBRX / TOTAL_TRX)")
        
        # Feature 5: Market Share (proporción de prescripciones de Brand 1)
        if 'UC_TRX' in self.df_processed.columns and 'ORAL_TRX' in self.df_processed.columns:
            total_market = self.df_processed['UC_TRX'] + self.df_processed['ORAL_TRX']
            self.df_processed['BRAND1_MARKET_SHARE'] = np.where(
                total_market > 0,
                self.df_processed['UC_TRX'] / total_market,
                0
            )
            self.log("   ✓ Creada: BRAND1_MARKET_SHARE")
            
    def normalize_features(self, method='robust'):
        """
        Normaliza features numéricas
        
        Args:
            method (str): 'standard' o 'robust'
        """
        self.log(f"\n📏 Normalizando features (método: {method})...")
        
        # Seleccionar columnas a normalizar
        cols_to_normalize = [col for col in self.df_processed.select_dtypes(include=[np.number]).columns 
                            if col not in self.id_columns and col != 'SEGMENT']
        
        if not cols_to_normalize:
            self.log("   ⚠️ No hay columnas para normalizar")
            return
        
        # Aplicar escalador
        if method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        # Guardar nombres de columnas normalizadas para referencia
        self.normalized_columns = cols_to_normalize
        
        # Normalizar
        self.df_processed[cols_to_normalize] = scaler.fit_transform(
            self.df_processed[cols_to_normalize]
        )
        
        self.log(f"   ✓ Normalizadas {len(cols_to_normalize)} columnas")
        
    def save_processed_data(self):
        """Guarda los datos procesados"""
        self.log(f"\n💾 Guardando datos procesados...")
        
        # Crear directorio si no existe
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar
        self.df_processed.to_csv(self.output_path, index=False)
        
        self.log(f"   ✓ Datos guardados en: {self.output_path}")
        self.log(f"   Dimensiones finales: {self.df_processed.shape[0]:,} filas, {self.df_processed.shape[1]} columnas")
        
    def save_processing_log(self, log_path='reports/preprocessing_log.txt'):
        """Guarda el log de procesamiento"""
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("PFIZER HCP SEGMENTATION - LOG DE PROCESAMIENTO DE DATOS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Archivo de entrada: {self.input_path}\n")
            f.write(f"Archivo de salida: {self.output_path}\n\n")
            f.write("="*70 + "\n\n")
            
            for log_entry in self.processing_log:
                f.write(log_entry + "\n")
                
            f.write("\n" + "="*70 + "\n")
            f.write("PROCESAMIENTO COMPLETADO\n")
            f.write("="*70 + "\n")
        
        self.log(f"\n📝 Log guardado en: {log_path}")
        
    def run_full_preprocessing(self, 
                              remove_constants=True,
                              remove_correlations=True,
                              correlation_threshold=0.95,
                              handle_outliers_flag=True,
                              outlier_method='clip',
                              create_features=True,
                              normalize=True,
                              normalization_method='robust'):
        """
        Ejecuta el pipeline completo de procesamiento
        
        Args:
            remove_constants (bool): Eliminar features constantes
            remove_correlations (bool): Eliminar features altamente correlacionadas
            correlation_threshold (float): Umbral de correlación
            handle_outliers_flag (bool): Manejar outliers
            outlier_method (str): Método para outliers ('clip' o 'remove')
            create_features (bool): Crear nuevas features
            normalize (bool): Normalizar features
            normalization_method (str): Método de normalización ('standard' o 'robust')
        """
        print("\n" + "="*70)
        print("🚀 INICIANDO PROCESAMIENTO DE DATOS")
        print("="*70)
        
        # 1. Cargar datos
        self.load_data()
        
        # 2. Manejar columnas de ID
        self.handle_id_columns()
        
        # 3. Manejar valores faltantes
        self.handle_missing_values()
        
        # 4. Eliminar features constantes
        if remove_constants:
            self.remove_constant_features()
        
        # 5. Eliminar features altamente correlacionadas
        if remove_correlations:
            self.remove_high_correlation(threshold=correlation_threshold)
        
        # 6. Manejar outliers
        if handle_outliers_flag:
            self.handle_outliers(method=outlier_method)
        
        # 7. Feature engineering
        if create_features:
            self.feature_engineering()
        
        # 8. Normalizar
        if normalize:
            self.normalize_features(method=normalization_method)
        
        # 9. Guardar datos procesados
        self.save_processed_data()
        
        # 10. Guardar log
        self.save_processing_log()
        
        print("\n" + "="*70)
        print("✅ PROCESAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"\n📊 Datos procesados guardados en: {self.output_path}")
        print(f"📝 Log guardado en: reports/preprocessing_log.txt")
        
        return self.df_processed


if __name__ == "__main__":
    # Configurar rutas
    INPUT_PATH = "data/raw/VELSIPITY_AFF_MULTIPLI_prepared_prepared_prepared.csv"
    OUTPUT_PATH = "data/processed/data_processed.csv"
    
    # Crear preprocesador
    preprocessor = DataPreprocessor(INPUT_PATH, OUTPUT_PATH)
    
    # Ejecutar procesamiento completo
    df_processed = preprocessor.run_full_preprocessing(
        remove_constants=True,
        remove_correlations=True,
        correlation_threshold=0.95,
        handle_outliers_flag=True,
        outlier_method='clip',
        create_features=True,
        normalize=True,
        normalization_method='robust'
    )