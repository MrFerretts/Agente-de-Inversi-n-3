"""
State Manager - Sistema de caché inteligente para terminal quant
Evita recálculos innecesarios y optimiza performance
"""

import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import hashlib
import pickle


class StateManager:
    """Gestor de estado con caché inteligente por TTL"""
    
    def __init__(self, cache_ttl_seconds: int = 300):  # 5 minutos default
        self.cache_ttl = cache_ttl_seconds
        if 'cache_store' not in st.session_state:
            st.session_state.cache_store = {}
        if 'analysis_cache' not in st.session_state:
            st.session_state.analysis_cache = {}
    
    def _generate_key(self, symbol: str, data_type: str, **kwargs) -> str:
        """Genera clave única para el caché basada en parámetros"""
        params_str = f"{symbol}_{data_type}_" + "_".join([f"{k}={v}" for k, v in sorted(kwargs.items())])
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def get_cached_data(self, symbol: str, data_type: str, **kwargs) -> Optional[Any]:
        """Recupera datos del caché si no han expirado"""
        key = self._generate_key(symbol, data_type, **kwargs)
        
        if key in st.session_state.cache_store:
            cached_item = st.session_state.cache_store[key]
            
            # Verificar si el caché sigue válido
            if datetime.now() - cached_item['timestamp'] < timedelta(seconds=self.cache_ttl):
                return cached_item['data']
            else:
                # Caché expirado, eliminar
                del st.session_state.cache_store[key]
        
        return None
    
    def set_cached_data(self, symbol: str, data_type: str, data: Any, **kwargs):
        """Almacena datos en el caché con timestamp"""
        key = self._generate_key(symbol, data_type, **kwargs)
        st.session_state.cache_store[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def invalidate_cache(self, symbol: Optional[str] = None):
        """Invalida el caché (todo o de un símbolo específico)"""
        if symbol is None:
            st.session_state.cache_store = {}
        else:
            # Eliminar solo entradas del símbolo
            keys_to_delete = [k for k in st.session_state.cache_store.keys() if symbol in k]
            for k in keys_to_delete:
                del st.session_state.cache_store[k]
    
    def get_analysis_cache(self, symbol: str) -> Optional[Dict]:
        """Recupera análisis técnico cacheado"""
        return st.session_state.analysis_cache.get(symbol)
    
    def set_analysis_cache(self, symbol: str, analysis: Dict):
        """Guarda análisis técnico en caché"""
        st.session_state.analysis_cache[symbol] = {
            'data': analysis,
            'timestamp': datetime.now()
        }
    
    def get_cache_stats(self) -> Dict:
        """Retorna estadísticas del caché"""
        total_items = len(st.session_state.cache_store)
        valid_items = sum(
            1 for item in st.session_state.cache_store.values()
            if datetime.now() - item['timestamp'] < timedelta(seconds=self.cache_ttl)
        )
        
        return {
            'total_items': total_items,
            'valid_items': valid_items,
            'expired_items': total_items - valid_items,
            'cache_ttl': self.cache_ttl
        }


class DataProcessor:
    """Procesador central de datos con indicadores pre-calculados"""
    
    @staticmethod
    def prepare_full_analysis(data: pd.DataFrame, analyzer) -> pd.DataFrame:
        """
        Calcula TODOS los indicadores de una vez y los almacena en el DataFrame
        Evita recalcular en cada pestaña
        """
        df = data.copy()
        
        # Medias móviles
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # Bandas de Bollinger
        std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['SMA20'] + (std * 2)
        df['BB_Lower'] = df['SMA20'] - (std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close']
        
        # RSI y Stochastic RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Stochastic RSI
        low_rsi = df['RSI'].rolling(14).min()
        high_rsi = df['RSI'].rolling(14).max()
        df['StochRSI'] = (df['RSI'] - low_rsi) / (high_rsi - low_rsi + 1e-9)
        
        # MACD
        df['MACD_Line'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.ewm(alpha=1/14, adjust=False).mean()
        
        # ADX
        up = df['High'].diff()
        down = df['Low'].shift() - df['Low']
        plus_dm = pd.Series(0, index=df.index)
        minus_dm = pd.Series(0, index=df.index)
        plus_dm[(up > down) & (up > 0)] = up[(up > down) & (up > 0)]
        minus_dm[(down > up) & (down > 0)] = down[(down > up) & (down > 0)]
        
        smoothed_plus_dm = plus_dm.ewm(alpha=1/14, adjust=False).mean()
        smoothed_minus_dm = minus_dm.ewm(alpha=1/14, adjust=False).mean()
        
        plus_di = 100 * (smoothed_plus_dm / (df['ATR'] + 1e-9))
        minus_di = 100 * (smoothed_minus_dm / (df['ATR'] + 1e-9))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
        df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()
        
        # Volumen relativo
        df['Avg_Volume'] = df['Volume'].rolling(20).mean()
        df['RVOL'] = df['Volume'] / (df['Avg_Volume'] + 1e-9)
        
        # Retornos
        df['Returns'] = df['Close'].pct_change()
        df['Returns_Cum'] = (1 + df['Returns']).cumprod()
        
        return df
    
    @staticmethod
    def get_latest_signals(df: pd.DataFrame) -> Dict:
        """Extrae las últimas señales del DataFrame procesado"""
        if df is None or df.empty:
            return {}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        
        return {
            'price': latest['Close'],
            'price_change_pct': ((latest['Close'] - prev['Close']) / prev['Close']) * 100,
            'rsi': latest['RSI'],
            'stoch_rsi': latest['StochRSI'],
            'macd_hist': latest['MACD_Hist'],
            'macd_trend': 'BULLISH' if latest['MACD_Hist'] > 0 else 'BEARISH',
            'adx': latest['ADX'],
            'atr': latest['ATR'],
            'rvol': latest['RVOL'],
            'bb_position': 'UPPER' if latest['Close'] >= latest['BB_Upper'] * 0.99 else 'LOWER' if latest['Close'] <= latest['BB_Lower'] * 1.01 else 'MIDDLE',
            'trend': 'BULLISH' if latest['Close'] > latest['SMA50'] else 'BEARISH',
            'trend_strength': 'STRONG' if latest['ADX'] > 25 else 'WEAK'
        }
