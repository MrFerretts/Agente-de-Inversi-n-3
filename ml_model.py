"""
ADVANCED MACHINE LEARNING MODULE - Ensemble Predictor
Sistema avanzado que combina Random Forest + XGBoost + Logistic Regression
Mayor precisión y robustez que el modelo básico
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost no disponible. Usando solo RF + LR")


class AdvancedTradingMLModel:
    """
    Modelo avanzado de ML con ensemble de múltiples algoritmos
    - Random Forest (robusto, no requiere normalización)
    - XGBoost (gradient boosting, alta precisión)
    - Logistic Regression (baseline, rápido)
    
    Usa voting ensemble para combinar predicciones
    """
    
    def __init__(self, prediction_days: int = 5, threshold: float = 2.0):
        """
        Args:
            prediction_days: Días hacia adelante para predecir
            threshold: % mínimo de cambio para considerar "subida"
        """
        self.prediction_days = prediction_days
        self.threshold = threshold
        self.ensemble_model = None
        self.rf_model = None
        self.xgb_model = None
        self.lr_model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.is_trained = False
        self.training_date = None
        self.model_metrics = {}
        self.individual_predictions = {}
        
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features avanzados para mejor precisión
        
        Args:
            df: DataFrame con indicadores básicos
        
        Returns:
            DataFrame con features adicionales
        """
        data = df.copy()
        
        # ====================================================================
        # FEATURES DE MOMENTUM AVANZADO
        # ====================================================================
        
        # RSI en múltiples timeframes
        for period in [7, 14, 21]:
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
            loss = -delta.where(delta < 0, 0).ewm(alpha=1/period, adjust=False).mean()
            rs = gain / (loss + 1e-9)
            data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # Divergencia RSI (feature explícito)
        data['RSI_Divergence'] = (data['Close'].pct_change(20) < 0) & (data['RSI'].diff(20) > 0)
        data['RSI_Divergence'] = data['RSI_Divergence'].astype(int)
        
        # ====================================================================
        # FEATURES DE TENDENCIA AVANZADOS
        # ====================================================================
        
        # Distancia a múltiples SMAs
        for period in [10, 20, 50, 100, 200]:
            if f'SMA{period}' not in data.columns:
                data[f'SMA{period}'] = data['Close'].rolling(period).mean()
            data[f'Dist_SMA{period}'] = (data['Close'] / data[f'SMA{period}'] - 1) * 100
        
        # Cruce de medias (Golden Cross / Death Cross)
        data['SMA_Cross_50_200'] = ((data['SMA50'] > data['SMA200']).astype(int) -
                                     (data['SMA50'] < data['SMA200']).astype(int))
        
        # ====================================================================
        # FEATURES DE VOLATILIDAD AVANZADOS
        # ====================================================================
        
        # ATR normalizado en múltiples periodos
        for period in [7, 14, 21]:
            high_low = data['High'] - data['Low']
            high_close = (data['High'] - data['Close'].shift()).abs()
            low_close = (data['Low'] - data['Close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            data[f'ATR_{period}'] = true_range.ewm(alpha=1/period, adjust=False).mean()
            data[f'ATR_{period}_Norm'] = data[f'ATR_{period}'] / data['Close']
        
        # Volatilidad histórica
        for period in [10, 20, 30]:
            data[f'HV_{period}'] = data['Returns'].rolling(period).std() * np.sqrt(252) * 100
        
        # Ratio de volatilidad (actual vs histórica)
        data['Vol_Ratio'] = data['HV_20'] / (data['HV_20'].rolling(50).mean() + 1e-9)
        
        # ====================================================================
        # FEATURES DE VOLUMEN AVANZADOS
        # ====================================================================
        
        # RVOL en múltiples periodos
        for period in [10, 20, 30]:
            data[f'Avg_Vol_{period}'] = data['Volume'].rolling(period).mean()
            data[f'RVOL_{period}'] = data['Volume'] / (data[f'Avg_Vol_{period}'] + 1e-9)
        
        # Volumen en rangos de precio (¿compran arriba o abajo?)
        data['Vol_High_Range'] = (data['Close'] > data['Low'] + (data['High'] - data['Low']) * 0.66).astype(int)
        
        # ====================================================================
        # FEATURES DE PRICE ACTION
        # ====================================================================
        
        # Retornos en múltiples timeframes
        for period in [1, 3, 5, 10, 20, 30]:
            data[f'Return_{period}D'] = data['Close'].pct_change(period) * 100
        
        # Máximo y mínimo de N días
        for period in [5, 10, 20]:
            data[f'High_{period}D'] = data['High'].rolling(period).max()
            data[f'Low_{period}D'] = data['Low'].rolling(period).min()
            data[f'Range_{period}D'] = (data[f'High_{period}D'] - data[f'Low_{period}D']) / data['Close'] * 100
        
        # Posición en el rango
        data['Position_in_Range_20D'] = ((data['Close'] - data['Low_20D']) / 
                                          (data['High_20D'] - data['Low_20D'] + 1e-9))
        
        # ====================================================================
        # FEATURES DE PATRONES (SIMPLIFICADOS)
        # ====================================================================
        
        # Doji (indecisión)
        body = (data['Close'] - data['Open']).abs()
        range_total = data['High'] - data['Low']
        data['Is_Doji'] = (body / (range_total + 1e-9) < 0.1).astype(int)
        
        # Vela grande (momentum fuerte)
        data['Is_Large_Candle'] = (body / (range_total + 1e-9) > 0.7).astype(int)
        
        # ====================================================================
        # FEATURES DE INTERACCIÓN (COMBINACIONES)
        # ====================================================================
        
        # RSI * ADX (momentum confirmado por fuerza)
        data['RSI_x_ADX'] = data['RSI'] * data['ADX'] / 100
        
        # RVOL * Price_Change (volumen confirma movimiento)
        data['RVOL_x_Return'] = data['RVOL'] * data['Return_1D']
        
        # BB_Width * ATR (compresión de volatilidad)
        data['BB_ATR_Ratio'] = data['BB_Width'] / (data['ATR'] / data['Close'])
        
        return data
    
    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara datos con features avanzados
        
        Args:
            data: DataFrame con indicadores técnicos básicos
        
        Returns:
            X (features), y (labels)
        """
        print("🔬 Creando features avanzados...")
        df = self.create_advanced_features(data)
        
        # ====================================================================
        # CREAR ETIQUETAS (TARGET) — SIN DATA LEAKAGE
        # ====================================================================
        # CORRECCIÓN CRÍTICA: El target debe medir el retorno FUTURO
        # desde el cierre del día t hasta el cierre del día t+N.
        #
        # ANTES (con leakage):
        #   pct_change(N) calcula (Close[t] - Close[t-N]) / Close[t-N]
        #   shift(-N) desplaza ese valor al día t-N, mezclando información
        #   del día t (que está en los features) con el label.
        #
        # AHORA (correcto):
        #   Future_Return[t] = (Close[t+N] - Close[t]) / Close[t]
        #   El modelo aprende: "dados los indicadores del día t,
        #   ¿el precio subirá X% en los próximos N días?"
        #   Los features del día t NO contienen información de Close[t+N].

        df['Future_Return'] = (
            df['Close'].shift(-self.prediction_days) / df['Close'] - 1
        ) * 100

        # Etiqueta binaria: 1 si sube más del threshold, 0 si no
        df['Target'] = (df['Future_Return'] > self.threshold).astype(int)
        
        # ====================================================================
        # SELECCIONAR FEATURES
        # ====================================================================
        
        feature_columns = [
            # RSI en múltiples timeframes
            'RSI_7', 'RSI_14', 'RSI_21',
            'RSI_Divergence',
            
            # Posición vs SMAs
            'Dist_SMA10', 'Dist_SMA20', 'Dist_SMA50', 'Dist_SMA100', 'Dist_SMA200',
            'SMA_Cross_50_200',
            
            # Volatilidad
            'ATR_7_Norm', 'ATR_14_Norm', 'ATR_21_Norm',
            'HV_10', 'HV_20', 'HV_30',
            'Vol_Ratio',
            
            # Volumen
            'RVOL_10', 'RVOL_20', 'RVOL_30',
            'Vol_High_Range',
            
            # Returns
            'Return_1D', 'Return_3D', 'Return_5D', 'Return_10D', 'Return_20D', 'Return_30D',
            
            # Rangos de precio
            'Range_5D', 'Range_10D', 'Range_20D',
            'Position_in_Range_20D',
            
            # Patrones
            'Is_Doji', 'Is_Large_Candle',
            
            # Interacciones
            'RSI_x_ADX', 'RVOL_x_Return', 'BB_ATR_Ratio',
            
            # Indicadores originales importantes
            'MACD_Hist', 'ADX', 'StochRSI', 'BB_Width'
        ]
        
        # Filtrar columnas que existen
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        print(f"✅ Features preparados: {len(feature_columns)} features")
        
        # Remover NaN
        df_clean = df[feature_columns + ['Target']].dropna()
        
        X = df_clean[feature_columns]
        y = df_clean['Target']
        
        return X, y
    
    def train(self, data: pd.DataFrame, test_size: float = 0.2, 
             optimize_hyperparameters: bool = False) -> Dict:
        """
        Entrena ensemble de modelos
        
        Args:
            data: DataFrame con indicadores
            test_size: Proporción para test
            optimize_hyperparameters: Si hacer GridSearch (tarda más)
        
        Returns:
            Dict con métricas
        """
        print("\n" + "="*70)
        print("🚀 ENTRENANDO MODELO AVANZADO (ENSEMBLE)")
        print("="*70 + "\n")
        
        X, y = self.prepare_training_data(data)
        
        if len(X) < 100:
            raise ValueError(f"Datos insuficientes. Mínimo 100, tienes {len(X)}")
        
        print(f"📊 Dataset: {len(X)} muestras")
        print(f"   - Clase 1 (subida): {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
        print(f"   - Clase 0 (no subida): {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")
        
        # ====================================================================
        # SPLIT TEMPORAL — SIN SHUFFLE
        # ====================================================================
        # CORRECCIÓN: Nunca usar shuffle=True en time series.
        # El test set SIEMPRE es el 20% más reciente (el "futuro" del modelo).
        # Esto simula exactamente cómo se usará el modelo en producción:
        # entrenado con datos viejos, evaluado en datos nuevos que no vio.

        split_idx = int(len(X) * (1 - test_size))
        X_train = X.iloc[:split_idx]
        X_test  = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test  = y.iloc[split_idx:]

        print(f"   Train: {len(X_train)} muestras (hasta índice {split_idx})")
        print(f"   Test:  {len(X_test)} muestras (últimas — datos más recientes)")
        
        # ====================================================================
        # ENTRENAR MODELOS INDIVIDUALES
        # ====================================================================
        
        print("\n🌲 1/3 Entrenando Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        rf_score = self.rf_model.score(X_test, y_test)
        print(f"   ✅ RF Accuracy: {rf_score*100:.1f}%")
        
        if XGBOOST_AVAILABLE:
            print("\n🚀 2/3 Entrenando XGBoost...")
            self.xgb_model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )
            self.xgb_model.fit(X_train, y_train)
            xgb_score = self.xgb_model.score(X_test, y_test)
            print(f"   ✅ XGB Accuracy: {xgb_score*100:.1f}%")
        
        print("\n📊 3/3 Entrenando Logistic Regression...")
        # Normalizar para LR
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.lr_model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        self.lr_model.fit(X_train_scaled, y_train)
        lr_score = self.lr_model.score(X_test_scaled, y_test)
        print(f"   ✅ LR Accuracy: {lr_score*100:.1f}%")
        
        # ====================================================================
        # CREAR ENSEMBLE
        # ====================================================================
        
        print("\n🎭 Creando Ensemble (Voting)...")
        
        estimators = [
            ('rf', self.rf_model),
            ('lr', self.lr_model)
        ]
        
        if XGBOOST_AVAILABLE:
            estimators.insert(1, ('xgb', self.xgb_model))
        
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Usa probabilidades
            n_jobs=-1
        )
        
        # Entrenar ensemble (necesita datos normalizados para LR)
        X_train_for_ensemble = X_train.copy()
        X_test_for_ensemble = X_test.copy()
        
        self.ensemble_model.fit(X_train_for_ensemble, y_train)
        
        print("   ✅ Ensemble entrenado!")
        
        # ====================================================================
        # EVALUAR ENSEMBLE
        # ====================================================================
        
        print("\n📈 Evaluando ensemble...")
        
        y_pred = self.ensemble_model.predict(X_test_for_ensemble)
        y_pred_proba = self.ensemble_model.predict_proba(X_test_for_ensemble)[:, 1]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc_roc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_roc = 0.5
        
        # Feature importance (promedio de RF y XGB)
        rf_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.rf_model.feature_importances_
        })
        
        if XGBOOST_AVAILABLE:
            xgb_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.xgb_model.feature_importances_
            })
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': (rf_importance['importance'].values + xgb_importance['importance'].values) / 2
            }).sort_values('importance', ascending=False)
        else:
            self.feature_importance = rf_importance.sort_values('importance', ascending=False)
        
        # ====================================================================
        # VALIDACIÓN CRUZADA TEMPORAL (Walk-Forward)
        # ====================================================================
        # CORRECCIÓN CRÍTICA: cross_val_score con cv=3 sin TimeSeriesSplit
        # mezcla datos del futuro en los folds de entrenamiento.
        #
        # TimeSeriesSplit garantiza que cada fold de validación siempre
        # contiene datos MÁS RECIENTES que el fold de entrenamiento.
        # Esto replica exactamente cómo funciona el modelo en producción real.

        print("\n🔄 Validación cruzada Walk-Forward (TimeSeriesSplit, 5 folds)...")
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(
            self.ensemble_model,
            X_train_for_ensemble,
            y_train,
            cv=tscv,          # ← Walk-forward, no aleatorio
            scoring='accuracy',
            n_jobs=-1
        )
        print(f"   Walk-Forward CV: {cv_scores.mean()*100:.1f}% (±{cv_scores.std()*100:.1f}%)")
        print(f"   Scores por fold: {[f'{s*100:.1f}%' for s in cv_scores]}")
        
        # Métricas
        self.model_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': len(X.columns),
            'features': list(X.columns),
            'rf_accuracy': rf_score,
            'lr_accuracy': lr_score
        }
        
        if XGBOOST_AVAILABLE:
            self.model_metrics['xgb_accuracy'] = xgb_score
        
        self.is_trained = True
        self.training_date = datetime.now()
        
        # Imprimir resultados
        print("\n" + "="*70)
        print("🏆 RESULTADOS DEL ENSEMBLE")
        print("="*70)
        print(f"{'Modelo':<20} {'Accuracy':<15} {'Status'}")
        print("-"*70)
        print(f"{'Random Forest':<20} {rf_score*100:>6.1f}%")
        if XGBOOST_AVAILABLE:
            print(f"{'XGBoost':<20} {xgb_score*100:>6.1f}%")
        print(f"{'Logistic Regression':<20} {lr_score*100:>6.1f}%")
        print("-"*70)
        print(f"{'ENSEMBLE (Voting)':<20} {accuracy*100:>6.1f}%        {'🏆 MEJOR' if accuracy > max(rf_score, lr_score) else ''}")
        print("="*70)
        
        print(f"\n📊 Métricas Ensemble:")
        print(f"   Precision: {precision*100:.1f}%")
        print(f"   Recall: {recall*100:.1f}%")
        print(f"   F1-Score: {f1*100:.1f}%")
        print(f"   AUC-ROC: {auc_roc:.3f}")
        print(f"   Cross-Val: {cv_scores.mean()*100:.1f}% (±{cv_scores.std()*100:.1f}%)")
        
        print(f"\n🔥 Top 5 Features:")
        for idx, row in self.feature_importance.head(5).iterrows():
            print(f"   {row['feature']:<30s} {row['importance']*100:>5.1f}%")
        
        return self.model_metrics
    
    def predict(self, data: pd.DataFrame) -> Dict:
        """
        Predice con ensemble
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        
        # Preparar features
        df = self.create_advanced_features(data)
        
        feature_columns = self.model_metrics['features']
        X_current = df[feature_columns].iloc[[-1]]
        
        # Predicción del ensemble
        proba = self.ensemble_model.predict_proba(X_current)[0]
        pred_class = self.ensemble_model.predict(X_current)[0]
        
        prob_down = proba[0]
        prob_up = proba[1]
        
        # Predicciones individuales
        rf_proba = self.rf_model.predict_proba(X_current)[0][1]
        lr_proba = self.lr_model.predict_proba(self.scaler.transform(X_current))[0][1]
        
        individual_preds = {
            'rf': rf_proba,
            'lr': lr_proba
        }
        
        if XGBOOST_AVAILABLE:
            xgb_proba = self.xgb_model.predict_proba(X_current)[0][1]
            individual_preds['xgb'] = xgb_proba
        
        # Calcular confianza
        confidence = max(prob_up, prob_down)
        
        # Acuerdo entre modelos (mayor acuerdo = mayor confianza real)
        model_probs = list(individual_preds.values())
        agreement = 1 - (np.std(model_probs) / 0.5)  # Normalizado
        
        if confidence > 0.75 and agreement > 0.8:
            confidence_level = "MUY ALTA"
        elif confidence > 0.70:
            confidence_level = "ALTA"
        elif confidence > 0.60:
            confidence_level = "MEDIA"
        else:
            confidence_level = "BAJA"
        
        # Recomendación
        if prob_up > 0.70:
            recommendation = "COMPRA FUERTE"
        elif prob_up > 0.60:
            recommendation = "COMPRA"
        elif prob_up < 0.30:
            recommendation = "VENTA FUERTE"
        elif prob_up < 0.40:
            recommendation = "VENTA"
        else:
            recommendation = "MANTENER"
        
        return {
            'probability_up': prob_up,
            'probability_down': prob_down,
            'predicted_class': pred_class,
            'recommendation': recommendation,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'model_agreement': agreement,
            'individual_predictions': individual_preds,
            'prediction_days': self.prediction_days,
            'threshold': self.threshold,
            'model_accuracy': self.model_metrics['accuracy'],
            'ensemble_used': True
        }
    
    # Métodos de utilidad (iguales al modelo básico)
    def get_feature_importance(self) -> pd.DataFrame:
        if self.feature_importance is None:
            raise ValueError("Modelo no entrenado")
        return self.feature_importance
    
    def save_model(self, filepath: str):
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        
        model_data = {
            'ensemble_model': self.ensemble_model,
            'rf_model': self.rf_model,
            'xgb_model': self.xgb_model,
            'lr_model': self.lr_model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'model_metrics': self.model_metrics,
            'training_date': self.training_date,
            'prediction_days': self.prediction_days,
            'threshold': self.threshold
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Modelo ensemble guardado: {filepath}")
    
    def load_model(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.ensemble_model = model_data['ensemble_model']
        self.rf_model = model_data['rf_model']
        self.xgb_model = model_data.get('xgb_model')
        self.lr_model = model_data['lr_model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data['feature_importance']
        self.model_metrics = model_data['model_metrics']
        self.training_date = model_data['training_date']
        self.prediction_days = model_data['prediction_days']
        self.threshold = model_data['threshold']
        self.is_trained = True
        
        print(f"✅ Modelo ensemble cargado: {filepath}")
        print(f"   Entrenado: {self.training_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Accuracy: {self.model_metrics['accuracy']*100:.1f}%")


# ============================================================================
# FUNCIÓN PARA USAR EN STREAMLIT
# ============================================================================

def train_advanced_ml_model(ticker: str, data_processed: pd.DataFrame, 
                           prediction_days: int = 5) -> AdvancedTradingMLModel:
    """
    Entrena modelo avanzado (ensemble)
    """
    print(f"\n{'='*70}")
    print(f"🤖 ENTRENANDO MODELO AVANZADO (ENSEMBLE) PARA {ticker}")
    print(f"{'='*70}\n")
    
    model = AdvancedTradingMLModel(prediction_days=prediction_days, threshold=2.0)
    
    try:
        metrics = model.train(data_processed, test_size=0.2)
        return model
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# FUNCIONES DE COMPATIBILIDAD PARA IMPORTS ANTERIORES
# ============================================================================

# Crear alias para que el código que importa TradingMLModel funcione
TradingMLModel = AdvancedTradingMLModel


def train_ml_model_for_ticker(ticker: str, data_processed: pd.DataFrame, 
                              prediction_days: int = 5) -> AdvancedTradingMLModel:
    """
    Wrapper para compatibilidad con código anterior
    Llama a train_advanced_ml_model con el nombre antiguo
    
    Args:
        ticker: Símbolo del activo
        data_processed: DataFrame con indicadores
        prediction_days: Días a predecir
    
    Returns:
        Modelo entrenado
    """
    return train_advanced_ml_model(ticker, data_processed, prediction_days)


def get_ml_prediction(model: AdvancedTradingMLModel, data_processed: pd.DataFrame) -> Dict:
    """
    Obtiene predicción del modelo
    
    Args:
        model: Modelo entrenado
        data_processed: DataFrame con indicadores
    
    Returns:
        Dict con predicción o None
    """
    if model is None or not model.is_trained:
        return None
    
    try:
        prediction = model.predict(data_processed)
        return prediction
    except Exception as e:
        print(f"❌ Error en predicción: {str(e)}")
        return None


def format_ml_output(prediction: Dict, ticker: str) -> str:
    """
    Formatea output de ML para Streamlit
    
    Args:
        prediction: Dict con predicción
        ticker: Símbolo del activo
    
    Returns:
        String formateado en markdown
    """
    if prediction is None:
        return "⚠️ No hay predicción disponible"
    
    # Emoji según recomendación
    if "COMPRA" in prediction['recommendation']:
        emoji = "🟢"
    elif "VENTA" in prediction['recommendation']:
        emoji = "🔴"
    else:
        emoji = "🟡"
    
    output = f"""
## 🤖 Predicción Machine Learning - {ticker}

### Probabilidades
- **📈 Subida en {prediction['prediction_days']} días:** {prediction['probability_up']*100:.1f}%
- **📉 Bajada en {prediction['prediction_days']} días:** {prediction['probability_down']*100:.1f}%

### Recomendación
{emoji} **{prediction['recommendation']}**

### Confianza del Modelo
- **Nivel:** {prediction['confidence_level']}
- **Score:** {prediction['confidence']*100:.1f}%
- **Accuracy del modelo:** {prediction['model_accuracy']*100:.1f}%
"""
    
    # Si es ensemble, mostrar predicciones individuales
    if prediction.get('individual_predictions'):
        output += "\n### 🔬 Predicciones de Modelos Individuales\n"
        
        indiv = prediction['individual_predictions']
        
        if 'rf' in indiv:
            output += f"- **Random Forest:** {indiv['rf']*100:.1f}%\n"
        if 'xgb' in indiv:
            output += f"- **XGBoost:** {indiv['xgb']*100:.1f}%\n"
        if 'lr' in indiv:
            output += f"- **Logistic Regression:** {indiv['lr']*100:.1f}%\n"
        
        agreement = prediction.get('model_agreement', 0)
        if agreement > 0:
            output += "\n"
            if agreement > 0.85:
                output += f"✅ **Alto acuerdo** entre modelos ({agreement*100:.0f}%) - Señal muy confiable"
            elif agreement > 0.70:
                output += f"ℹ️ Acuerdo moderado entre modelos ({agreement*100:.0f}%)"
            else:
                output += f"⚠️ Bajo acuerdo entre modelos ({agreement*100:.0f}%) - Precaución"
    
    output += "\n\n### Interpretación\n"
    
    if prediction['probability_up'] > 0.70:
        output += "✅ El modelo tiene **alta confianza** en una subida. Las condiciones técnicas favorecen posiciones largas."
    elif prediction['probability_up'] > 0.60:
        output += "↗️ El modelo sugiere **sesgo alcista moderado**. Considerar entrada con stops ajustados."
    elif prediction['probability_up'] < 0.30:
        output += "❌ El modelo tiene **alta confianza** en una bajada. Evitar posiciones largas."
    elif prediction['probability_up'] < 0.40:
        output += "↘️ El modelo sugiere **sesgo bajista**. Considerar reducir exposición."
    else:
        output += "↔️ El modelo es **neutral**. Esperar señal más clara antes de operar."
    
    if prediction['confidence_level'] == "BAJA":
        output += "\n\n⚠️ **Nota:** Confianza baja. Combinar con análisis técnico tradicional."
    
    return output
