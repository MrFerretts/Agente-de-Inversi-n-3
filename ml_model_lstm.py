"""
LSTM DEEP LEARNING MODULE - Advanced Time Series Prediction
Sistema de Deep Learning con redes neuronales LSTM para predecir movimientos de precio
Captura patrones temporales que otros modelos no pueden detectar

FIX 2026-03-17:
  - Features sincronizados con ml_model.py (~15 features, no 23)
  - Arquitectura reducida (64→32→16) para Railway (menos RAM)
  - Error handling en predict()
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    TENSORFLOW_AVAILABLE = True
except:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow no disponible. Instala: pip install tensorflow")


class LSTMTradingModel:
    """
    Modelo LSTM (Long Short-Term Memory) para predección de series temporales
    
    LSTM es ideal para trading porque:
    1. Tiene memoria de largo plazo (recuerda patrones de días/semanas atrás)
    2. Captura dependencias temporales complejas
    3. Maneja secuencias de longitud variable
    4. Detecta patrones no lineales
    """
    
    def __init__(self, 
                 prediction_days: int = 5,
                 lookback_window: int = 20,
                 threshold: float = 2.0):
        """
        Args:
            prediction_days: Días hacia adelante para predecir
            lookback_window: Cuántos días de historia usar (ventana temporal)
            threshold: % mínimo de cambio para considerar "subida"
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow no está instalado. Ejecuta: pip install tensorflow")
        
        self.prediction_days = prediction_days
        self.lookback_window = lookback_window
        self.threshold = threshold
        
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_names = []
        self.is_trained = False
        self.training_date = None
        self.model_metrics = {}
        self.training_history = None
        
    def create_sequences(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea secuencias temporales para LSTM
        
        Ejemplo: Si lookback_window=20
        X[0] = [datos día 0-19] → y[0] = sube/baja día 24
        X[1] = [datos día 1-20] → y[1] = sube/baja día 25
        ...
        
        Args:
            data: Datos de features (escalados)
            labels: Etiquetas (0 o 1)
        
        Returns:
            X (secuencias), y (labels)
        """
        X, y = [], []
        
        for i in range(self.lookback_window, len(data) - self.prediction_days):
            # Secuencia de lookback_window días
            X.append(data[i-self.lookback_window:i])
            # Label es del día futuro
            y.append(labels[i + self.prediction_days - 1])
        
        return np.array(X), np.array(y)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara features para LSTM — SINCRONIZADOS con ml_model.py (~15 features).

        Mismos features que el ensemble ML para consistencia entre modelos:
          RSI_14, Dist_SMA20, Dist_SMA50, SMA_Cross_50_200, ATR_14_Norm,
          HV_20, RVOL_20, Return_1D, Return_5D, Return_20D,
          Position_in_Range_20D, MACD_Hist, ADX, StochRSI, BB_Width
        """
        data = df.copy()
        features = pd.DataFrame(index=data.index)

        # RSI 14 normalizado a [0, 1]
        if 'RSI' in data.columns:
            features['RSI_14'] = data['RSI'] / 100
        elif 'RSI_14' in data.columns:
            features['RSI_14'] = data['RSI_14'] / 100
        else:
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / (loss + 1e-9)
            features['RSI_14'] = (100 - (100 / (1 + rs))) / 100

        # Distancias a SMAs
        sma20 = data['SMA20'] if 'SMA20' in data.columns else data['Close'].rolling(20).mean()
        sma50 = data['SMA50'] if 'SMA50' in data.columns else data['Close'].rolling(50).mean()
        features['Dist_SMA20'] = (data['Close'] - sma20) / (sma20 + 1e-9)
        features['Dist_SMA50'] = (data['Close'] - sma50) / (sma50 + 1e-9)

        # Golden/Death cross
        sma200 = data['SMA200'] if 'SMA200' in data.columns else data['Close'].rolling(200).mean()
        features['SMA_Cross'] = ((sma50 > sma200).astype(float) -
                                  (sma50 < sma200).astype(float))

        # ATR normalizado
        if 'ATR' in data.columns:
            features['ATR_Norm'] = data['ATR'] / (data['Close'] + 1e-9)
        elif 'ATR_14' in data.columns:
            features['ATR_Norm'] = data['ATR_14'] / (data['Close'] + 1e-9)
        else:
            hl = data['High'] - data['Low']
            hc = (data['High'] - data['Close'].shift()).abs()
            lc = (data['Low'] - data['Close'].shift()).abs()
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            features['ATR_Norm'] = tr.ewm(alpha=1/14, adjust=False).mean() / (data['Close'] + 1e-9)

        # Volatilidad histórica 20d
        returns = data['Returns'] if 'Returns' in data.columns else data['Close'].pct_change()
        features['HV_20'] = returns.rolling(20).std() * np.sqrt(252)

        # RVOL 20
        if 'RVOL' in data.columns:
            features['RVOL_20'] = data['RVOL']
        elif 'RVOL_20' in data.columns:
            features['RVOL_20'] = data['RVOL_20']
        else:
            avg_vol = data['Volume'].rolling(20).mean()
            features['RVOL_20'] = data['Volume'] / (avg_vol + 1e-9)

        # Retornos multi-escala
        features['Return_1D'] = data['Close'].pct_change(1)
        features['Return_5D'] = data['Close'].pct_change(5)
        features['Return_20D'] = data['Close'].pct_change(20)

        # Posición en rango 20D
        high_20 = data['High'].rolling(20).max()
        low_20 = data['Low'].rolling(20).min()
        features['Position_Range'] = (data['Close'] - low_20) / (high_20 - low_20 + 1e-9)

        # MACD Hist normalizado
        if 'MACD_Hist' in data.columns:
            features['MACD_Hist'] = data['MACD_Hist'] / (data['Close'] + 1e-9)

        # ADX normalizado
        if 'ADX' in data.columns:
            features['ADX'] = data['ADX'] / 100

        # StochRSI
        if 'StochRSI' in data.columns:
            features['StochRSI'] = data['StochRSI']

        # BB Width normalizado
        if 'BB_Width' in data.columns:
            features['BB_Width'] = data['BB_Width'] / (data['Close'] + 1e-9)
        elif 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
            features['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / (data['Close'] + 1e-9)

        return features.dropna()
    
    def build_model(self, input_shape: Tuple) -> Sequential:
        """
        Construye arquitectura LSTM
        
        Arquitectura:
        1. LSTM Layer 1 (128 units) - Captura patrones de largo plazo
        2. Dropout (0.2) - Previene overfitting
        3. LSTM Layer 2 (64 units) - Refina patrones
        4. Dropout (0.2)
        5. LSTM Layer 3 (32 units) - Extrae features finales
        6. Dropout (0.2)
        7. Dense (16 units) - Combinación
        8. Output (1 unit, sigmoid) - Probabilidad
        
        Args:
            input_shape: (lookback_window, n_features)
        
        Returns:
            Modelo compilado
        """
        # Arquitectura reducida para Railway (menos RAM, menos overfitting)
        # Original: 128→64→32 (~300K params) → Ahora: 64→32→16 (~50K params)
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),

            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),

            Dense(16, activation='relu'),
            Dropout(0.2),

            Dense(1, activation='sigmoid')
        ])
        
        # Compilar con optimizer Adam
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def train(self, data: pd.DataFrame, 
              epochs: int = 50,
              batch_size: int = 32,
              validation_split: float = 0.2) -> Dict:
        """
        Entrena el modelo LSTM
        
        Args:
            data: DataFrame con indicadores
            epochs: Número de épocas (ciclos de entrenamiento)
            batch_size: Tamaño de batch
            validation_split: % de datos para validación
        
        Returns:
            Dict con métricas
        """
        print("\n" + "="*70)
        print("🧠 ENTRENANDO MODELO LSTM (DEEP LEARNING)")
        print("="*70 + "\n")
        
        # ====================================================================
        # PREPARAR DATOS
        # ====================================================================
        
        print("📊 Preparando datos temporales...")
        
        # Crear features
        features_df = self.prepare_features(data)
        self.feature_names = list(features_df.columns)
        
        # ====================================================================
        # CREAR ETIQUETAS (TARGET) — SIN DATA LEAKAGE
        # ====================================================================
        # CORRECCIÓN: Igual que en ml_model.py.
        # Future_Return[t] = (Close[t+N] - Close[t]) / Close[t]
        # Los features del día t no contienen información de días futuros.

        df_with_labels = data.loc[features_df.index].copy()
        df_with_labels['Future_Return'] = (
            df_with_labels['Close'].shift(-self.prediction_days) /
            df_with_labels['Close'] - 1
        ) * 100
        df_with_labels['Target'] = (df_with_labels['Future_Return'] > self.threshold).astype(int)
        
        # Alinear features y labels
        valid_idx = features_df.index.intersection(df_with_labels.dropna().index)
        features_array = features_df.loc[valid_idx].values
        labels_array = df_with_labels.loc[valid_idx, 'Target'].values
        
        if len(features_array) < self.lookback_window + self.prediction_days + 50:
            raise ValueError(f"Datos insuficientes. Necesitas al menos {self.lookback_window + self.prediction_days + 50} días.")
        
        print(f"✅ Datos preparados: {len(features_array)} días")
        print(f"   - Features: {features_array.shape[1]}")
        print(f"   - Lookback window: {self.lookback_window} días")
        print(f"   - Clase 1 (subida): {labels_array.sum()} ({labels_array.sum()/len(labels_array)*100:.1f}%)")
        print(f"   - Clase 0 (no subida): {len(labels_array)-labels_array.sum()} ({(len(labels_array)-labels_array.sum())/len(labels_array)*100:.1f}%)")
        
        # ====================================================================
        # SPLIT TEMPORAL PREVIO AL SCALING — CRÍTICO
        # ====================================================================
        # CORRECCIÓN: El scaler debe ajustarse (fit) SOLO con datos de train.
        # Si hacemos fit_transform sobre todos los datos, el scaler "ve"
        # los valores máximos y mínimos del test set, filtrando información
        # del futuro hacia el pasado (scaling leakage).
        #
        # Orden correcto:
        #   1. Dividir en train/test por tiempo
        #   2. scaler.fit(train) — aprende estadísticas solo del pasado
        #   3. scaler.transform(train) y scaler.transform(test) por separado

        split_raw = int(len(features_array) * 0.8)
        features_train_raw = features_array[:split_raw]
        features_test_raw  = features_array[split_raw:]
        labels_train_raw   = labels_array[:split_raw]
        labels_test_raw    = labels_array[split_raw:]

        print("\n🔢 Normalizando datos (scaler ajustado solo con train)...")
        # fit SOLO sobre train — el test nunca influye en la normalización
        features_train_scaled = self.scaler.fit_transform(features_train_raw)
        features_test_scaled  = self.scaler.transform(features_test_raw)

        # ====================================================================
        # CREAR SECUENCIAS
        # ====================================================================

        print(f"\n⏱️ Creando secuencias temporales (ventana de {self.lookback_window} días)...")
        X_train, y_train = self.create_sequences(features_train_scaled, labels_train_raw)
        X_test,  y_test  = self.create_sequences(features_test_scaled,  labels_test_raw)

        print(f"✅ Secuencias creadas:")
        print(f"   Train: {X_train.shape[0]} secuencias | Shape: {X_train.shape}")
        print(f"   Test:  {X_test.shape[0]} secuencias  | Shape: {X_test.shape}")
        
        # ====================================================================
        # CONSTRUIR MODELO
        # ====================================================================
        
        print("\n🏗️ Construyendo arquitectura LSTM...")
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        print("\n📋 Arquitectura del modelo:")
        self.model.summary()
        
        # ====================================================================
        # CALLBACKS (para mejor entrenamiento)
        # ====================================================================
        
        callbacks = [
            # Early stopping - detiene si no mejora
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate si se estanca
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # ====================================================================
        # ENTRENAR
        # ====================================================================
        
        print(f"\n🚀 Entrenando LSTM ({epochs} épocas)...")
        print("   (Esto puede tardar 2-5 minutos)\n")
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_history = history.history
        
        print("\n✅ Entrenamiento completado!")
        
        # ====================================================================
        # EVALUAR
        # ====================================================================
        
        print("\n📊 Evaluando modelo en test set...")
        
        # Predicciones
        y_pred_proba = self.model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc_roc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_roc = 0.5
        
        # Guardar métricas
        self.model_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': X_train.shape[2],
            'lookback_window': self.lookback_window,
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'features': self.feature_names
        }
        
        self.is_trained = True
        self.training_date = datetime.now()
        
        # ====================================================================
        # MOSTRAR RESULTADOS
        # ====================================================================
        
        print("\n" + "="*70)
        print("🏆 RESULTADOS FINALES - LSTM")
        print("="*70)
        print(f"Accuracy:    {accuracy*100:>6.1f}%")
        print(f"Precision:   {precision*100:>6.1f}%")
        print(f"Recall:      {recall*100:>6.1f}%")
        print(f"F1-Score:    {f1*100:>6.1f}%")
        print(f"AUC-ROC:     {auc_roc:>6.3f}")
        print("="*70)
        
        print(f"\n📉 Loss final:")
        print(f"   Train: {history.history['loss'][-1]:.4f}")
        print(f"   Val:   {history.history['val_loss'][-1]:.4f}")
        
        print(f"\n⏱️ Épocas entrenadas: {len(history.history['loss'])}/{epochs}")
        
        # Análisis de overfitting
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        
        if val_loss > train_loss * 1.5:
            print("\n⚠️ Posible overfitting detectado (val_loss >> train_loss)")
            print("   Considera reducir epochs o agregar más datos")
        elif abs(val_loss - train_loss) < 0.01:
            print("\n✅ Buen balance entre train y validation")
        
        return self.model_metrics
    
    def predict(self, data: pd.DataFrame) -> Dict:
        """
        Predice con LSTM usando datos recientes
        
        Args:
            data: DataFrame con indicadores (debe tener al menos lookback_window días)
        
        Returns:
            Dict con predicción
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        
        # Preparar features
        try:
            features_df = self.prepare_features(data)
        except Exception as e:
            raise ValueError(f"Error preparando features: {e}")

        if len(features_df) < self.lookback_window:
            raise ValueError(f"Necesitas al menos {self.lookback_window} días de datos")

        # Tomar últimos lookback_window días
        recent_data = features_df.iloc[-self.lookback_window:].values

        # Verificar NaN antes de escalar
        if np.isnan(recent_data).any():
            # Rellenar NaN con la media de la columna
            col_means = np.nanmean(recent_data, axis=0)
            for j in range(recent_data.shape[1]):
                mask = np.isnan(recent_data[:, j])
                recent_data[mask, j] = col_means[j]

        # Verificar shape matches scaler
        if recent_data.shape[1] != self.scaler.n_features_in_:
            raise ValueError(
                f"Feature count mismatch: got {recent_data.shape[1]}, "
                f"expected {self.scaler.n_features_in_}"
            )

        # Normalizar
        recent_scaled = self.scaler.transform(recent_data)

        # Reshape para LSTM: (1, lookback_window, n_features)
        X_pred = recent_scaled.reshape(1, self.lookback_window, -1)

        # Predecir
        prob_up = self.model.predict(X_pred, verbose=0)[0][0]
        prob_down = 1 - prob_up
        pred_class = int(prob_up > 0.5)
        
        # Calcular confianza
        confidence = max(prob_up, prob_down)
        
        if confidence > 0.75:
            confidence_level = "MUY ALTA"
        elif confidence > 0.65:
            confidence_level = "ALTA"
        elif confidence > 0.55:
            confidence_level = "MEDIA"
        else:
            confidence_level = "BAJA"
        
        # Recomendación
        if prob_up > 0.75:
            recommendation = "COMPRA FUERTE"
        elif prob_up > 0.60:
            recommendation = "COMPRA"
        elif prob_up < 0.25:
            recommendation = "VENTA FUERTE"
        elif prob_up < 0.40:
            recommendation = "VENTA"
        else:
            recommendation = "MANTENER"
        
        return {
            'probability_up': float(prob_up),
            'probability_down': float(prob_down),
            'predicted_class': pred_class,
            'recommendation': recommendation,
            'confidence': float(confidence),
            'confidence_level': confidence_level,
            'prediction_days': self.prediction_days,
            'threshold': self.threshold,
            'model_accuracy': self.model_metrics['accuracy'],
            'model_type': 'LSTM',
            'lookback_window': self.lookback_window
        }
    
    def save_model(self, filepath: str):
        """Guarda modelo LSTM"""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        
        # Guardar modelo Keras
        model_path = filepath.replace('.pkl', '_keras.h5')
        self.model.save(model_path)
        
        # Guardar metadata
        metadata = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_metrics': self.model_metrics,
            'training_date': self.training_date,
            'prediction_days': self.prediction_days,
            'lookback_window': self.lookback_window,
            'threshold': self.threshold,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Modelo LSTM guardado:")
        print(f"   - Keras model: {model_path}")
        print(f"   - Metadata: {filepath}")
    
    def load_model(self, filepath: str):
        """Carga modelo LSTM"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        
        # Cargar modelo Keras
        model_path = filepath.replace('.pkl', '_keras.h5')
        self.model = keras.models.load_model(model_path)
        
        # Cargar metadata
        with open(filepath, 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler = metadata['scaler']
        self.feature_names = metadata['feature_names']
        self.model_metrics = metadata['model_metrics']
        self.training_date = metadata['training_date']
        self.prediction_days = metadata['prediction_days']
        self.lookback_window = metadata['lookback_window']
        self.threshold = metadata['threshold']
        self.training_history = metadata['training_history']
        self.is_trained = True
        
        print(f"✅ Modelo LSTM cargado: {filepath}")
        print(f"   Entrenado: {self.training_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Accuracy: {self.model_metrics['accuracy']*100:.1f}%")


# ============================================================================
# FUNCIÓN PARA STREAMLIT
# ============================================================================

def train_lstm_model(ticker: str, data_processed: pd.DataFrame,
                     prediction_days: int = 5,
                     lookback_window: int = 20,
                     epochs: int = 50) -> LSTMTradingModel:
    """
    Entrena modelo LSTM para un ticker
    
    Args:
        ticker: Símbolo
        data_processed: DataFrame con indicadores
        prediction_days: Días a predecir
        lookback_window: Ventana temporal
        epochs: Épocas de entrenamiento
    
    Returns:
        Modelo entrenado
    """
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow no disponible")
        print("   Instala con: pip install tensorflow")
        return None
    
    print(f"\n{'='*70}")
    print(f"🧠 ENTRENANDO LSTM (DEEP LEARNING) PARA {ticker}")
    print(f"{'='*70}\n")
    
    model = LSTMTradingModel(
        prediction_days=prediction_days,
        lookback_window=lookback_window,
        threshold=2.0
    )
    
    try:
        metrics = model.train(data_processed, epochs=epochs, batch_size=32)
        return model
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
