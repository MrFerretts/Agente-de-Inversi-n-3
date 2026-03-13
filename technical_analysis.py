"""
Módulo de Análisis Técnico - Versión Maestro Quant
VERSIÓN PROFESIONAL - 2026-02-12
Sistema de scoring multifactorial que usa TODOS los indicadores
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

"""
MÓDULO DE DETECCIÓN DE DIVERGENCIAS
Sistema avanzado para detectar divergencias RSI-Precio y MACD-Precio
Estas son señales MUY potentes de reversión de tendencia

Copiar este código y agregarlo a technical_analysis.py
"""

import pandas as pd
from typing import Dict, List, Tuple


def detect_divergences(data: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Detecta divergencias entre precio e indicadores (RSI y MACD)
    
    Las divergencias son señales de REVERSIÓN muy potentes:
    - Divergencia Alcista: Precio baja pero indicador sube → Posible rebote
    - Divergencia Bajista: Precio sube pero indicador baja → Posible caída
    
    Args:
        data: DataFrame con columnas Close, RSI, MACD_Hist
        lookback: Número de períodos a analizar (default 20 días)
    
    Returns:
        Dict con tipos de divergencias detectadas y detalles
    """
    divergences = {
        'rsi_bullish': False,
        'rsi_bearish': False,
        'macd_bullish': False,
        'macd_bearish': False,
        'signals': [],
        'score_impact': 0,
        'details': {}
    }
    
    if len(data) < lookback + 5:
        return divergences
    
    # Tomar ventana de análisis
    window = data.tail(lookback)
    
    # ====================================================================
    # ANÁLISIS DE TENDENCIAS
    # ====================================================================
    
    # Precios
    precio_inicio = window['Close'].iloc[0]
    precio_fin = window['Close'].iloc[-1]
    precio_cambio = ((precio_fin - precio_inicio) / precio_inicio) * 100
    
    # RSI
    rsi_inicio = window['RSI'].iloc[0]
    rsi_fin = window['RSI'].iloc[-1]
    rsi_cambio = rsi_fin - rsi_inicio
    
    # MACD Histogram
    macd_inicio = window['MACD_Hist'].iloc[0]
    macd_fin = window['MACD_Hist'].iloc[-1]
    macd_cambio = macd_fin - macd_inicio
    
    # ====================================================================
    # DETECCIÓN DE DIVERGENCIAS RSI
    # ====================================================================
    
    # Umbrales para considerar cambio significativo
    MIN_PRECIO_CAMBIO = 2.0  # 2% mínimo de movimiento de precio
    MIN_RSI_CAMBIO = 5.0     # 5 puntos mínimo de cambio en RSI
    
    # DIVERGENCIA ALCISTA RSI
    # Precio hace mínimos más bajos, pero RSI hace mínimos más altos
    if precio_cambio < -MIN_PRECIO_CAMBIO and rsi_cambio > MIN_RSI_CAMBIO:
        divergences['rsi_bullish'] = True
        divergences['score_impact'] += 20
        
        signal = (
            f"⚡ DIVERGENCIA ALCISTA RSI detectada: "
            f"Precio cayó {abs(precio_cambio):.1f}% pero RSI subió {rsi_cambio:.1f} puntos. "
            f"Señal de posible REVERSIÓN alcista."
        )
        divergences['signals'].append(signal)
        
        divergences['details']['rsi_bullish'] = {
            'precio_cambio': precio_cambio,
            'rsi_cambio': rsi_cambio,
            'strength': 'FUERTE' if abs(precio_cambio) > 5 and rsi_cambio > 10 else 'MODERADA'
        }
    
    # DIVERGENCIA BAJISTA RSI
    # Precio hace máximos más altos, pero RSI hace máximos más bajos
    elif precio_cambio > MIN_PRECIO_CAMBIO and rsi_cambio < -MIN_RSI_CAMBIO:
        divergences['rsi_bearish'] = True
        divergences['score_impact'] -= 20
        
        signal = (
            f"⚡ DIVERGENCIA BAJISTA RSI detectada: "
            f"Precio subió {precio_cambio:.1f}% pero RSI cayó {abs(rsi_cambio):.1f} puntos. "
            f"Señal de posible REVERSIÓN bajista."
        )
        divergences['signals'].append(signal)
        
        divergences['details']['rsi_bearish'] = {
            'precio_cambio': precio_cambio,
            'rsi_cambio': rsi_cambio,
            'strength': 'FUERTE' if precio_cambio > 5 and abs(rsi_cambio) > 10 else 'MODERADA'
        }
    
    # ====================================================================
    # DETECCIÓN DE DIVERGENCIAS MACD
    # ====================================================================
    
    MIN_MACD_CAMBIO = 0.5  # Cambio mínimo en MACD histogram
    
    # DIVERGENCIA ALCISTA MACD
    # Precio baja pero MACD sube (menos bajista)
    if precio_cambio < -MIN_PRECIO_CAMBIO and macd_cambio > MIN_MACD_CAMBIO:
        divergences['macd_bullish'] = True
        divergences['score_impact'] += 15
        
        signal = (
            f"📊 DIVERGENCIA ALCISTA MACD detectada: "
            f"Precio cayó {abs(precio_cambio):.1f}% pero MACD mejoró. "
            f"Momentum bajista perdiendo fuerza."
        )
        divergences['signals'].append(signal)
        
        divergences['details']['macd_bullish'] = {
            'precio_cambio': precio_cambio,
            'macd_cambio': macd_cambio,
            'strength': 'FUERTE' if abs(precio_cambio) > 5 and macd_cambio > 1.0 else 'MODERADA'
        }
    
    # DIVERGENCIA BAJISTA MACD
    # Precio sube pero MACD baja (menos alcista)
    elif precio_cambio > MIN_PRECIO_CAMBIO and macd_cambio < -MIN_MACD_CAMBIO:
        divergences['macd_bearish'] = True
        divergences['score_impact'] -= 15
        
        signal = (
            f"📊 DIVERGENCIA BAJISTA MACD detectada: "
            f"Precio subió {precio_cambio:.1f}% pero MACD empeoró. "
            f"Momentum alcista perdiendo fuerza."
        )
        divergences['signals'].append(signal)
        
        divergences['details']['macd_bearish'] = {
            'precio_cambio': precio_cambio,
            'macd_cambio': macd_cambio,
            'strength': 'FUERTE' if precio_cambio > 5 and abs(macd_cambio) > 1.0 else 'MODERADA'
        }
    
    # ====================================================================
    # DIVERGENCIA OCULTA (Hidden Divergence) - AVANZADO
    # ====================================================================
    # Solo si NO hay divergencia regular
    
    if not any([divergences['rsi_bullish'], divergences['rsi_bearish'], 
                divergences['macd_bullish'], divergences['macd_bearish']]):
        
        # Buscar mínimos y máximos locales
        window_prices = window['Close'].values
        window_rsi = window['RSI'].values
        
        # Encontrar últimos 2 mínimos en precio
        price_lows = find_local_minima(window_prices, order=5)
        rsi_lows = find_local_minima(window_rsi, order=5)
        
        # Divergencia oculta alcista: Precio hace mínimos más ALTOS pero RSI hace mínimos más BAJOS
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            if (window_prices[price_lows[-1]] > window_prices[price_lows[-2]] and
                window_rsi[rsi_lows[-1]] < window_rsi[rsi_lows[-2]]):
                
                divergences['score_impact'] += 10
                divergences['signals'].append(
                    "🔍 DIVERGENCIA OCULTA ALCISTA: Continuación de tendencia alcista probable"
                )
    
    return divergences


def find_local_minima(data: list, order: int = 5) -> List[int]:
    """
    Encuentra mínimos locales en una serie de datos
    
    Args:
        data: Lista de valores
        order: Número de puntos a cada lado para comparar
    
    Returns:
        Lista de índices donde hay mínimos locales
    """
    minima = []
    
    for i in range(order, len(data) - order):
        # Verificar si es un mínimo local
        is_min = True
        for j in range(1, order + 1):
            if data[i] >= data[i - j] or data[i] >= data[i + j]:
                is_min = False
                break
        
        if is_min:
            minima.append(i)
    
    return minima


def find_local_maxima(data: list, order: int = 5) -> List[int]:
    """
    Encuentra máximos locales en una serie de datos
    
    Args:
        data: Lista de valores
        order: Número de puntos a cada lado para comparar
    
    Returns:
        Lista de índices donde hay máximos locales
    """
    maxima = []
    
    for i in range(order, len(data) - order):
        # Verificar si es un máximo local
        is_max = True
        for j in range(1, order + 1):
            if data[i] <= data[i - j] or data[i] <= data[i + j]:
                is_max = False
                break
        
        if is_max:
            maxima.append(i)
    
    return maxima


def analyze_divergence_strength(divergence_details: Dict) -> str:
    """
    Analiza la fuerza de una divergencia detectada
    
    Args:
        divergence_details: Detalles de la divergencia
    
    Returns:
        String describiendo la fuerza: DÉBIL, MODERADA, FUERTE, MUY FUERTE
    """
    if not divergence_details:
        return "N/A"
    
    # Combinar criterios
    precio_cambio = abs(divergence_details.get('precio_cambio', 0))
    
    if 'rsi_cambio' in divergence_details:
        indicador_cambio = abs(divergence_details['rsi_cambio'])
        
        if precio_cambio > 8 and indicador_cambio > 15:
            return "MUY FUERTE ⚡⚡⚡"
        elif precio_cambio > 5 and indicador_cambio > 10:
            return "FUERTE ⚡⚡"
        elif precio_cambio > 3 and indicador_cambio > 7:
            return "MODERADA ⚡"
        else:
            return "DÉBIL"
    
    elif 'macd_cambio' in divergence_details:
        macd_cambio = abs(divergence_details['macd_cambio'])
        
        if precio_cambio > 8 and macd_cambio > 1.5:
            return "MUY FUERTE ⚡⚡⚡"
        elif precio_cambio > 5 and macd_cambio > 1.0:
            return "FUERTE ⚡⚡"
        elif precio_cambio > 3 and macd_cambio > 0.5:
            return "MODERADA ⚡"
        else:
            return "DÉBIL"
    
    return "N/A"


# ============================================================================
# FUNCIÓN DE INTEGRACIÓN CON EL SISTEMA EXISTENTE
# ============================================================================

def integrate_divergences_into_scoring(data: pd.DataFrame, current_score: int, 
                                      buy_signals: List, sell_signals: List,
                                      neutral_signals: List) -> Tuple[int, List, List, List]:
    """
    Integra el análisis de divergencias en el sistema de scoring existente
    
    Args:
        data: DataFrame con indicadores
        current_score: Score actual del sistema
        buy_signals: Lista de señales de compra
        sell_signals: Lista de señales de venta
        neutral_signals: Lista de señales neutrales
    
    Returns:
        Tupla (nuevo_score, buy_signals, sell_signals, neutral_signals)
    """
    # Detectar divergencias
    divergences = detect_divergences(data, lookback=20)
    
    # Aplicar impacto al score
    new_score = current_score + divergences['score_impact']
    
    # Agregar señales a las listas correspondientes
    for signal in divergences['signals']:
        if divergences['score_impact'] > 0:
            buy_signals.append(signal)
        elif divergences['score_impact'] < 0:
            sell_signals.append(signal)
        else:
            neutral_signals.append(signal)
    
    # Si hay divergencia fuerte, agregar nota adicional
    if any([divergences['rsi_bullish'], divergences['macd_bullish']]):
        if divergences.get('details'):
            for div_type, details in divergences['details'].items():
                if 'bullish' in div_type:
                    strength = analyze_divergence_strength(details)
                    neutral_signals.append(
                        f"💡 Fuerza de divergencia alcista: {strength}"
                    )
    
    if any([divergences['rsi_bearish'], divergences['macd_bearish']]):
        if divergences.get('details'):
            for div_type, details in divergences['details'].items():
                if 'bearish' in div_type:
                    strength = analyze_divergence_strength(details)
                    neutral_signals.append(
                        f"💡 Fuerza de divergencia bajista: {strength}"
                    )
    
    return new_score, buy_signals, sell_signals, neutral_signals


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Análisis técnico avanzado con detección de régimen y gestión de riesgo"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.indicators = config.get('TECHNICAL_INDICATORS', {})

    # --- CÁLCULOS MATEMÁTICOS BASE ---

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calcula el ATR (Average True Range) usando EMA - CORREGIDO
        Usa el método de suavizado de Wilder (EMA) en lugar de SMA
        """
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        # CORREGIDO: Usar EMA en lugar de SMA (método Wilder)
        return true_range.ewm(alpha=1/period, adjust=False).mean()

    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calcula el ADX (Average Directional Index) usando EMA - CORREGIDO
        Usa el método de suavizado de Wilder (EMA) en lugar de SMA
        """
        df = data.copy()
        df['up'] = df['High'].diff()
        df['down'] = df['Low'].shift() - df['Low']
        df['+dm'] = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0)
        df['-dm'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0)
        
        atr = self.calculate_atr(data, period)
        
        # CORREGIDO: Usar EMA en lugar de SMA para suavizar +DM y -DM
        smoothed_plus_dm = pd.Series(df['+dm']).ewm(alpha=1/period, adjust=False).mean()
        smoothed_minus_dm = pd.Series(df['-dm']).ewm(alpha=1/period, adjust=False).mean()
        
        df['+di'] = 100 * (smoothed_plus_dm / (atr + 1e-9))
        df['-di'] = 100 * (smoothed_minus_dm / (atr + 1e-9))
        df['dx'] = 100 * np.abs(df['+di'] - df['-di']) / (df['+di'] + df['-di'] + 1e-9)
        
        # CORREGIDO: Usar EMA en lugar de SMA para suavizar DX
        return df['dx'].ewm(alpha=1/period, adjust=False).mean()

    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Cálculo estándar de RSI usando EMA (método Wilder) - CORREGIDO
        """
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # CORREGIDO: Usar EMA en lugar de SMA (método Wilder estándar)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_stoch_rsi(self, rsi_series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calcula el Stochastic RSI (0 a 1)
        """
        low_rsi = rsi_series.rolling(window=period).min()
        high_rsi = rsi_series.rolling(window=period).max()
        stoch_rsi = (rsi_series - low_rsi) / (high_rsi - low_rsi + 1e-9)
        return stoch_rsi

    # --- MOTOR DE ANÁLISIS PRINCIPAL ---

    def analyze_asset(self, data: pd.DataFrame, symbol: str) -> Dict:
        """
        Análisis completo de un activo
        Ahora con todos los indicadores calculados correctamente
        """
        if data is None or data.empty or len(data) < 30:
            logger.warning(f"Datos insuficientes para {symbol}")
            return {}

        try:
            # Precio y medias móviles
            current_price = data['Close'].iloc[-1]
            sma_short = data['Close'].rolling(window=20).mean().iloc[-1]
            sma_long = data['Close'].rolling(window=50).mean().iloc[-1]
            
            # Indicadores técnicos (ahora corregidos)
            rsi = self.calculate_rsi(data).iloc[-1]
            atr = self.calculate_atr(data).iloc[-1]
            adx = self.calculate_adx(data).iloc[-1]

            # Bandas de Bollinger
            std_dev = data['Close'].rolling(window=20).std().iloc[-1]
            bb_upper = sma_short + (std_dev * 2)
            bb_lower = sma_short - (std_dev * 2)
            
            # MACD
            ema_fast = data['Close'].ewm(span=12, adjust=False).mean()
            ema_slow = data['Close'].ewm(span=26, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = macd_line - signal_line
            
            # Volumen relativo
            avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            rvol = current_volume / (avg_volume + 1e-9)

            # Stochastic RSI
            rsi_series = self.calculate_rsi(data)
            stoch_rsi = self.calculate_stoch_rsi(rsi_series).iloc[-1]
            
            # Régimen de mercado
            regime = "TRENDING" if adx > 25 else "RANGING"
            
            # ===================================================================
            # 🔥 NUEVO SISTEMA DE SEÑALES PROFESIONAL
            # ===================================================================
            signals = self._generate_signals_professional(
                data=data,
                price={
                    'current': current_price,
                    'sma_short': sma_short,
                    'sma_long': sma_long
                },
                indicators={
                    'rsi': rsi,
                    'stoch_rsi': stoch_rsi,
                    'rvol': rvol,
                    'macd_hist': macd_hist.iloc[-1],
                    'prev_macd_hist': macd_hist.iloc[-2] if len(macd_hist) > 1 else 0,
                    'adx': adx,
                    'atr': atr,
                    'bb_upper': bb_upper,
                    'bb_lower': bb_lower,
                    'regime': regime
                }
            )

            return {
                'symbol': symbol,
                'price': {
                    'current': current_price,
                    'sma_short': sma_short,
                    'sma_long': sma_long,
                    'change_pct': data['Close'].pct_change().iloc[-1] * 100
                },
                'indicators': {
                    'rsi': rsi,
                    'stoch_rsi': stoch_rsi,
                    'rvol': rvol,
                    'macd': macd_line.iloc[-1],
                    'macd_hist': macd_hist.iloc[-1],
                    'volatility': (atr / current_price),
                    'adx': adx,
                    'atr': atr,
                    'bb_upper': bb_upper,
                    'bb_lower': bb_lower,
                    'regime': regime
                },
                'signals': signals
            }
        except Exception as e:
            logger.error(f"Error crítico en TechnicalAnalyzer para {symbol}: {str(e)}")
            return {}

    def _generate_signals_professional(self, data: pd.DataFrame, price: dict, indicators: dict) -> Dict:
        """
        ===================================================================
        🧠 SISTEMA DE SCORING PROFESIONAL MULTIFACTORIAL
        ===================================================================
        Usa TODOS los indicadores con pesos calibrados
        Score máximo: ±100 puntos
        """
        score = 0
        buy_signals = []
        sell_signals = []
        neutral_signals = []
        
        # ===================================================================
        # 1️⃣ TENDENCIA (Peso: 30 puntos)
        # ===================================================================
        if price['current'] > price['sma_short'] > price['sma_long']:
            score += 30
            buy_signals.append("✅ Tendencia alcista fuerte (Precio > SMA20 > SMA50)")
        elif price['current'] < price['sma_short'] < price['sma_long']:
            score -= 30
            sell_signals.append("❌ Tendencia bajista fuerte (Precio < SMA20 < SMA50)")
        elif price['current'] > price['sma_short']:
            score += 15
            buy_signals.append("⚠️ Precio por encima de SMA20 (tendencia corta alcista)")
        else:
            score -= 15
            sell_signals.append("⚠️ Precio por debajo de SMA20 (tendencia corta bajista)")
        
        # ===================================================================
        # 2️⃣ MOMENTUM (Peso: 25 puntos) - RSI + Stoch RSI
        # ===================================================================
        rsi = indicators['rsi']
        stoch = indicators['stoch_rsi']
        
        # Sobreventa extrema (COMPRA)
        if rsi < 30 and stoch < 0.2:
            score += 25
            buy_signals.append(f"🔥 Sobreventa EXTREMA (RSI:{rsi:.1f} | StochRSI:{stoch:.2f})")
        elif rsi < 40 and stoch < 0.3:
            score += 15
            buy_signals.append(f"⚡ Sobreventa moderada (RSI:{rsi:.1f})")
        
        # Sobrecompra extrema (VENTA)
        elif rsi > 70 and stoch > 0.8:
            score -= 25
            sell_signals.append(f"🔥 Sobrecompra EXTREMA (RSI:{rsi:.1f} | StochRSI:{stoch:.2f})")
        elif rsi > 60 and stoch > 0.7:
            score -= 15
            sell_signals.append(f"⚡ Sobrecompra moderada (RSI:{rsi:.1f})")
        
        # Zona neutral (sin impacto)
        else:
            neutral_signals.append(f"↔️ RSI neutral ({rsi:.1f})")
        
        # ===================================================================
        # 3️⃣ FUERZA DIRECCIONAL (Peso: 20 puntos) - ADX
        # ===================================================================
        adx = indicators['adx']
        
        if adx > 40:
            # Tendencia MUY fuerte - Amplificar señal existente
            multiplier = 1.4 if score > 0 else 1.4
            score = int(score * multiplier)
            (buy_signals if score > 0 else sell_signals).append(f"💪 Tendencia EXTREMA (ADX:{adx:.1f})")
        elif adx > 25:
            # Tendencia fuerte - Amplificar moderadamente
            multiplier = 1.2 if score > 0 else 1.2
            score = int(score * multiplier)
            (buy_signals if score > 0 else sell_signals).append(f"📈 Tendencia confirmada (ADX:{adx:.1f})")
        elif adx < 20:
            # Mercado lateral - PENALIZAR agresividad
            score = int(score * 0.4)
            neutral_signals.append(f"⚠️ Mercado LATERAL - Evitar trades (ADX:{adx:.1f})")
        
        # ===================================================================
        # 4️⃣ CONFIRMACIÓN MACD (Peso: 15 puntos)
        # ===================================================================
        macd_hist = indicators['macd_hist']
        prev_macd = indicators['prev_macd_hist']
        
        # MACD cruzando al alza (momentum positivo)
        if macd_hist > 0 and macd_hist > prev_macd:
            score += 15
            buy_signals.append(f"📊 MACD alcista (Hist:{macd_hist:.3f})")
        # MACD cruzando a la baja (momentum negativo)
        elif macd_hist < 0 and macd_hist < prev_macd:
            score -= 15
            sell_signals.append(f"📊 MACD bajista (Hist:{macd_hist:.3f})")
        
        # ===================================================================
        # 5️⃣ VOLUMEN INSTITUCIONAL (Peso: 10 puntos) - RVOL
        # ===================================================================
        rvol = indicators['rvol']
        
        # Solo amplifica señales de COMPRA si hay volumen fuerte
        if rvol > 2.0 and score > 0:
            score += 10
            buy_signals.append(f"🚀 Volumen INSTITUCIONAL (RVOL:{rvol:.1f}x)")
        elif rvol > 1.5 and score > 0:
            score += 5
            buy_signals.append(f"📦 Volumen superior al promedio (RVOL:{rvol:.1f}x)")
        elif rvol < 0.5:
            neutral_signals.append(f"💤 Volumen bajo - Falta convicción (RVOL:{rvol:.1f}x)")
        
        # ===================================================================
        # 6️⃣ COMPRESIÓN BOLLINGER (Peso: Extra si aplica)
        # ===================================================================
        bb_width = (indicators['bb_upper'] - indicators['bb_lower']) / price['current']
        
        # Si las bandas están MUY comprimidas (< 5%), posible ruptura inminente
        if bb_width < 0.05:
            neutral_signals.append(f"⚡ Compresión Bollinger ({bb_width*100:.1f}%) - Ruptura inminente")
        
        # Si el precio toca banda inferior (sobreventa visual)
        if price['current'] <= indicators['bb_lower'] * 1.01:
            score += 5
            buy_signals.append("🎯 Precio en banda inferior (Bollinger)")
        # Si el precio toca banda superior (sobrecompra visual)
        elif price['current'] >= indicators['bb_upper'] * 0.99:
            score -= 5
            sell_signals.append("🎯 Precio en banda superior (Bollinger)")
        
        # ===================================================================
        # 7️⃣ DETECCIÓN DE DIVERGENCIAS (±20 puntos)
        # ===================================================================
        score, buy_signals, sell_signals, neutral_signals = integrate_divergences_into_scoring(
            data=data,
            current_score=score,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            neutral_signals=neutral_signals
        )
        
        # ===================================================================
        # 8️⃣ CLASIFICACIÓN FINAL (cambiar número)
        # ===================================================================
        
        if score >= 60:
            recommendation = 'COMPRA FUERTE'
            confidence = "MUY ALTA"
        elif score >= 30:
            recommendation = 'COMPRA'
            confidence = "ALTA" if adx > 25 else "MEDIA"
        elif score <= -60:
            recommendation = 'VENTA FUERTE'
            confidence = "MUY ALTA"
        elif score <= -30:
            recommendation = 'VENTA'
            confidence = "ALTA" if adx > 25 else "MEDIA"
        else:
            recommendation = 'MANTENER'
            confidence = "BAJA"

        return {
            'recommendation': recommendation,
            'score': score,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'neutral_signals': neutral_signals,
            'confidence': confidence
        }

    def compare_assets(self, analyses: Dict) -> pd.DataFrame:
        """Genera tabla comparativa para el log de agent.py"""
        comparison = []
        for symbol, analysis in analyses.items():
            if not analysis: continue
            comparison.append({
                'Symbol': symbol,
                'Price': analysis['price']['current'],
                'Change %': analysis['price']['change_pct'],
                'RSI': analysis['indicators']['rsi'],
                'Score': analysis['signals']['score'],
                'Recommendation': analysis['signals']['recommendation']
            })
        return pd.DataFrame(comparison).sort_values('Score', ascending=False)

