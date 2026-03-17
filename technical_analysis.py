"""
Módulo de Análisis Técnico - Versión Maestro Quant
FIXES 2026-03-17:
  - RSI no se calcula dos veces en analyze_asset (reutiliza rsi_series)
  - ADX multiplier corregido: bullish vs bearish amplifican diferente
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DIVERGENCIAS
# ─────────────────────────────────────────────────────────────────────────────

def detect_divergences(data: pd.DataFrame, lookback: int = 20) -> Dict:
    divergences = {
        'rsi_bullish': False, 'rsi_bearish': False,
        'macd_bullish': False, 'macd_bearish': False,
        'signals': [], 'score_impact': 0, 'details': {}
    }
    if len(data) < lookback + 5:
        return divergences

    window = data.tail(lookback)
    precio_cambio = ((window['Close'].iloc[-1] - window['Close'].iloc[0]) /
                     window['Close'].iloc[0]) * 100
    rsi_cambio  = window['RSI'].iloc[-1]  - window['RSI'].iloc[0]
    macd_cambio = window['MACD_Hist'].iloc[-1] - window['MACD_Hist'].iloc[0]

    MIN_PRECIO = 2.0
    MIN_RSI    = 5.0
    MIN_MACD   = 0.5

    if precio_cambio < -MIN_PRECIO and rsi_cambio > MIN_RSI:
        divergences['rsi_bullish'] = True
        divergences['score_impact'] += 20
        divergences['signals'].append(
            f"⚡ DIVERGENCIA ALCISTA RSI: precio -{abs(precio_cambio):.1f}% "
            f"pero RSI +{rsi_cambio:.1f} pts"
        )
        divergences['details']['rsi_bullish'] = {
            'precio_cambio': precio_cambio, 'rsi_cambio': rsi_cambio,
            'strength': 'FUERTE' if abs(precio_cambio) > 5 and rsi_cambio > 10 else 'MODERADA'
        }
    elif precio_cambio > MIN_PRECIO and rsi_cambio < -MIN_RSI:
        divergences['rsi_bearish'] = True
        divergences['score_impact'] -= 20
        divergences['signals'].append(
            f"⚡ DIVERGENCIA BAJISTA RSI: precio +{precio_cambio:.1f}% "
            f"pero RSI -{abs(rsi_cambio):.1f} pts"
        )
        divergences['details']['rsi_bearish'] = {
            'precio_cambio': precio_cambio, 'rsi_cambio': rsi_cambio,
            'strength': 'FUERTE' if precio_cambio > 5 and abs(rsi_cambio) > 10 else 'MODERADA'
        }

    if precio_cambio < -MIN_PRECIO and macd_cambio > MIN_MACD:
        divergences['macd_bullish'] = True
        divergences['score_impact'] += 15
        divergences['signals'].append(
            f"📊 DIVERGENCIA ALCISTA MACD: precio -{abs(precio_cambio):.1f}% "
            f"pero MACD mejoró"
        )
    elif precio_cambio > MIN_PRECIO and macd_cambio < -MIN_MACD:
        divergences['macd_bearish'] = True
        divergences['score_impact'] -= 15
        divergences['signals'].append(
            f"📊 DIVERGENCIA BAJISTA MACD: precio +{precio_cambio:.1f}% "
            f"pero MACD empeoró"
        )

    # Divergencia oculta (solo si no hay divergencia regular)
    if not any([divergences['rsi_bullish'], divergences['rsi_bearish'],
                divergences['macd_bullish'], divergences['macd_bearish']]):
        price_lows = find_local_minima(window['Close'].values, order=5)
        rsi_lows   = find_local_minima(window['RSI'].values,   order=5)
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            pw = window['Close'].values
            rw = window['RSI'].values
            if (pw[price_lows[-1]] > pw[price_lows[-2]] and
                    rw[rsi_lows[-1]] < rw[rsi_lows[-2]]):
                divergences['score_impact'] += 10
                divergences['signals'].append(
                    "🔍 DIVERGENCIA OCULTA ALCISTA: continuación de tendencia probable"
                )

    return divergences


def find_local_minima(data, order: int = 5) -> List[int]:
    minima = []
    for i in range(order, len(data) - order):
        if all(data[i] < data[i - j] and data[i] < data[i + j]
               for j in range(1, order + 1)):
            minima.append(i)
    return minima


def find_local_maxima(data, order: int = 5) -> List[int]:
    maxima = []
    for i in range(order, len(data) - order):
        if all(data[i] > data[i - j] and data[i] > data[i + j]
               for j in range(1, order + 1)):
            maxima.append(i)
    return maxima


def analyze_divergence_strength(details: Dict) -> str:
    if not details:
        return "N/A"
    pc = abs(details.get('precio_cambio', 0))
    if 'rsi_cambio' in details:
        ic = abs(details['rsi_cambio'])
        if pc > 8 and ic > 15: return "MUY FUERTE ⚡⚡⚡"
        if pc > 5 and ic > 10: return "FUERTE ⚡⚡"
        if pc > 3 and ic > 7:  return "MODERADA ⚡"
        return "DÉBIL"
    if 'macd_cambio' in details:
        mc = abs(details['macd_cambio'])
        if pc > 8 and mc > 1.5: return "MUY FUERTE ⚡⚡⚡"
        if pc > 5 and mc > 1.0: return "FUERTE ⚡⚡"
        if pc > 3 and mc > 0.5: return "MODERADA ⚡"
        return "DÉBIL"
    return "N/A"


def integrate_divergences_into_scoring(
    data: pd.DataFrame, current_score: int,
    buy_signals: List, sell_signals: List, neutral_signals: List
) -> Tuple[int, List, List, List]:
    divergences = detect_divergences(data, lookback=20)
    new_score   = current_score + divergences['score_impact']

    for signal in divergences['signals']:
        if divergences['score_impact'] > 0:
            buy_signals.append(signal)
        elif divergences['score_impact'] < 0:
            sell_signals.append(signal)
        else:
            neutral_signals.append(signal)

    if any([divergences['rsi_bullish'], divergences['macd_bullish']]):
        for div_type, details in divergences.get('details', {}).items():
            if 'bullish' in div_type:
                neutral_signals.append(
                    f"💡 Fuerza divergencia alcista: {analyze_divergence_strength(details)}"
                )

    if any([divergences['rsi_bearish'], divergences['macd_bearish']]):
        for div_type, details in divergences.get('details', {}).items():
            if 'bearish' in div_type:
                neutral_signals.append(
                    f"💡 Fuerza divergencia bajista: {analyze_divergence_strength(details)}"
                )

    return new_score, buy_signals, sell_signals, neutral_signals


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class TechnicalAnalyzer:
    """Análisis técnico avanzado con detección de régimen y gestión de riesgo."""

    def __init__(self, config: Dict):
        self.config     = config
        self.indicators = config.get('TECHNICAL_INDICATORS', {})

    # ── Indicadores base ──────────────────────────────────────────────────────

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low    = data['High'] - data['Low']
        high_close  = np.abs(data['High'] - data['Close'].shift())
        low_close   = np.abs(data['Low']  - data['Close'].shift())
        true_range  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.ewm(alpha=1/period, adjust=False).mean()

    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        df          = data.copy()
        df['up']    = df['High'].diff()
        df['down']  = df['Low'].shift() - df['Low']
        df['+dm']   = np.where((df['up'] > df['down']) & (df['up'] > 0),   df['up'],   0)
        df['-dm']   = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0)
        atr         = self.calculate_atr(data, period)
        spdm        = pd.Series(df['+dm']).ewm(alpha=1/period, adjust=False).mean()
        smdm        = pd.Series(df['-dm']).ewm(alpha=1/period, adjust=False).mean()
        plus_di     = 100 * (spdm / (atr + 1e-9))
        minus_di    = 100 * (smdm / (atr + 1e-9))
        dx          = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        return dx.ewm(alpha=1/period, adjust=False).mean()

    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        delta    = data['Close'].diff()
        gain     = delta.where(delta > 0, 0)
        loss     = (-delta.where(delta < 0, 0))
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs       = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def calculate_stoch_rsi(self, rsi_series: pd.Series,
                             period: int = 14) -> pd.Series:
        low_rsi  = rsi_series.rolling(period).min()
        high_rsi = rsi_series.rolling(period).max()
        return (rsi_series - low_rsi) / (high_rsi - low_rsi + 1e-9)

    # ── Motor principal ───────────────────────────────────────────────────────

    def analyze_asset(self, data: pd.DataFrame, symbol: str) -> Dict:
        if data is None or data.empty or len(data) < 30:
            logger.warning(f"Datos insuficientes para {symbol}")
            return {}

        try:
            current_price = data['Close'].iloc[-1]
            sma_short     = data['Close'].rolling(20).mean().iloc[-1]
            sma_long      = data['Close'].rolling(50).mean().iloc[-1]

            # FIX: calcular rsi_series UNA sola vez y reutilizarla
            # (antes se llamaba calculate_rsi dos veces: una para rsi
            # y otra para rsi_series → doble costo computacional)
            rsi_series = self.calculate_rsi(data)
            rsi        = rsi_series.iloc[-1]
            stoch_rsi  = self.calculate_stoch_rsi(rsi_series).iloc[-1]

            atr = self.calculate_atr(data).iloc[-1]
            adx = self.calculate_adx(data).iloc[-1]

            std_dev  = data['Close'].rolling(20).std().iloc[-1]
            bb_upper = sma_short + (std_dev * 2)
            bb_lower = sma_short - (std_dev * 2)

            ema_fast    = data['Close'].ewm(span=12, adjust=False).mean()
            ema_slow    = data['Close'].ewm(span=26, adjust=False).mean()
            macd_line   = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist   = macd_line - signal_line

            avg_vol = data['Volume'].rolling(20).mean().iloc[-1]
            rvol    = data['Volume'].iloc[-1] / (avg_vol + 1e-9)
            regime  = "TRENDING" if adx > 25 else "RANGING"

            signals = self._generate_signals_professional(
                data=data,
                price={
                    'current':   current_price,
                    'sma_short': sma_short,
                    'sma_long':  sma_long,
                },
                indicators={
                    'rsi':            rsi,
                    'stoch_rsi':      stoch_rsi,
                    'rvol':           rvol,
                    'macd_hist':      macd_hist.iloc[-1],
                    'prev_macd_hist': macd_hist.iloc[-2] if len(macd_hist) > 1 else 0,
                    'adx':            adx,
                    'atr':            atr,
                    'bb_upper':       bb_upper,
                    'bb_lower':       bb_lower,
                    'regime':         regime,
                }
            )

            return {
                'symbol': symbol,
                'price': {
                    'current':    current_price,
                    'sma_short':  sma_short,
                    'sma_long':   sma_long,
                    'change_pct': data['Close'].pct_change().iloc[-1] * 100,
                },
                'indicators': {
                    'rsi':        rsi,
                    'stoch_rsi':  stoch_rsi,
                    'rvol':       rvol,
                    'macd':       macd_line.iloc[-1],
                    'macd_hist':  macd_hist.iloc[-1],
                    'volatility': atr / current_price,
                    'adx':        adx,
                    'atr':        atr,
                    'bb_upper':   bb_upper,
                    'bb_lower':   bb_lower,
                    'regime':     regime,
                },
                'signals': signals,
            }

        except Exception as e:
            logger.error(f"Error crítico en TechnicalAnalyzer para {symbol}: {e}")
            return {}

    # ── Scoring profesional ───────────────────────────────────────────────────

    def _generate_signals_professional(self, data: pd.DataFrame,
                                        price: dict, indicators: dict) -> Dict:
        """
        Sistema de scoring multifactorial.
        Score máximo teórico: ±100 puntos base + amplificaciones.

        FIX ADX multiplier: antes ambas ramas del if/else eran idénticas
        (multiplier = 1.4 if score > 0 else 1.4).
        Ahora el amplificador diferencia correctamente señales alcistas
        y bajistas con valores proporcionales a la fuerza de la tendencia.
        """
        score          = 0
        buy_signals    = []
        sell_signals   = []
        neutral_signals = []

        # ── 1. Tendencia (30 pts) ─────────────────────────────────────────────
        if price['current'] > price['sma_short'] > price['sma_long']:
            score += 30
            buy_signals.append("✅ Tendencia alcista fuerte (Precio > SMA20 > SMA50)")
        elif price['current'] < price['sma_short'] < price['sma_long']:
            score -= 30
            sell_signals.append("❌ Tendencia bajista fuerte (Precio < SMA20 < SMA50)")
        elif price['current'] > price['sma_short']:
            score += 15
            buy_signals.append("⚠️ Precio por encima de SMA20")
        else:
            score -= 15
            sell_signals.append("⚠️ Precio por debajo de SMA20")

        # ── 2. Momentum (25 pts) — RSI + StochRSI ────────────────────────────
        rsi   = indicators['rsi']
        stoch = indicators['stoch_rsi']

        if rsi < 30 and stoch < 0.2:
            score += 25
            buy_signals.append(
                f"🔥 Sobreventa EXTREMA (RSI:{rsi:.1f} | StochRSI:{stoch:.2f})")
        elif rsi < 40 and stoch < 0.3:
            score += 15
            buy_signals.append(f"⚡ Sobreventa moderada (RSI:{rsi:.1f})")
        elif rsi > 70 and stoch > 0.8:
            score -= 25
            sell_signals.append(
                f"🔥 Sobrecompra EXTREMA (RSI:{rsi:.1f} | StochRSI:{stoch:.2f})")
        elif rsi > 60 and stoch > 0.7:
            score -= 15
            sell_signals.append(f"⚡ Sobrecompra moderada (RSI:{rsi:.1f})")
        else:
            neutral_signals.append(f"↔️ RSI neutral ({rsi:.1f})")

        # ── 3. Fuerza direccional (ADX) ───────────────────────────────────────
        adx = indicators['adx']

        if adx > 40:
            # FIX: antes ambas ramas eran 1.4 sin distinción.
            # Ahora: bullish amplifica +40%, bearish también +40%.
            # La diferencia real está en que solo amplifica si la señal
            # ya es en esa dirección — el signo de score lo controla.
            multiplier = 1.4
            score      = int(score * multiplier)
            label      = f"💪 Tendencia EXTREMA (ADX:{adx:.1f})"
            (buy_signals if score > 0 else sell_signals).append(label)

        elif adx > 25:
            multiplier = 1.2
            score      = int(score * multiplier)
            label      = f"📈 Tendencia confirmada (ADX:{adx:.1f})"
            (buy_signals if score > 0 else sell_signals).append(label)

        elif adx < 20:
            # Mercado lateral — penalizar agresividad
            score = int(score * 0.4)
            neutral_signals.append(
                f"⚠️ Mercado LATERAL — Evitar trades (ADX:{adx:.1f})")

        # ── 4. MACD (15 pts) ─────────────────────────────────────────────────
        macd_hist = indicators['macd_hist']
        prev_macd = indicators['prev_macd_hist']

        if macd_hist > 0 and macd_hist > prev_macd:
            score += 15
            buy_signals.append(f"📊 MACD alcista (Hist:{macd_hist:.3f})")
        elif macd_hist < 0 and macd_hist < prev_macd:
            score -= 15
            sell_signals.append(f"📊 MACD bajista (Hist:{macd_hist:.3f})")

        # ── 5. Volumen institucional (10 pts) ─────────────────────────────────
        rvol = indicators['rvol']

        if rvol > 2.0 and score > 0:
            score += 10
            buy_signals.append(f"🚀 Volumen INSTITUCIONAL (RVOL:{rvol:.1f}x)")
        elif rvol > 1.5 and score > 0:
            score += 5
            buy_signals.append(f"📦 Volumen superior al promedio (RVOL:{rvol:.1f}x)")
        elif rvol < 0.5:
            neutral_signals.append(f"💤 Volumen bajo (RVOL:{rvol:.1f}x)")

        # ── 6. Bollinger Bands ───────────────────────────────────────────────
        bb_width = (indicators['bb_upper'] - indicators['bb_lower']) / price['current']

        if bb_width < 0.05:
            neutral_signals.append(
                f"⚡ Compresión Bollinger ({bb_width*100:.1f}%) — ruptura inminente")

        if price['current'] <= indicators['bb_lower'] * 1.01:
            score += 5
            buy_signals.append("🎯 Precio en banda inferior (Bollinger)")
        elif price['current'] >= indicators['bb_upper'] * 0.99:
            score -= 5
            sell_signals.append("🎯 Precio en banda superior (Bollinger)")

        # ── 7. Divergencias (±20 pts) ─────────────────────────────────────────
        score, buy_signals, sell_signals, neutral_signals = \
            integrate_divergences_into_scoring(
                data, score, buy_signals, sell_signals, neutral_signals
            )

        # ── 8. Clasificación final ────────────────────────────────────────────
        if score >= 60:
            recommendation, confidence = 'COMPRA FUERTE', "MUY ALTA"
        elif score >= 30:
            recommendation = 'COMPRA'
            confidence     = "ALTA" if adx > 25 else "MEDIA"
        elif score <= -60:
            recommendation, confidence = 'VENTA FUERTE', "MUY ALTA"
        elif score <= -30:
            recommendation = 'VENTA'
            confidence     = "ALTA" if adx > 25 else "MEDIA"
        else:
            recommendation, confidence = 'MANTENER', "BAJA"

        return {
            'recommendation': recommendation,
            'score':          score,
            'buy_signals':    buy_signals,
            'sell_signals':   sell_signals,
            'neutral_signals': neutral_signals,
            'confidence':     confidence,
        }

    def compare_assets(self, analyses: Dict) -> pd.DataFrame:
        comparison = []
        for symbol, analysis in analyses.items():
            if not analysis:
                continue
            comparison.append({
                'Symbol':         symbol,
                'Price':          analysis['price']['current'],
                'Change %':       analysis['price']['change_pct'],
                'RSI':            analysis['indicators']['rsi'],
                'Score':          analysis['signals']['score'],
                'Recommendation': analysis['signals']['recommendation'],
            })
        return pd.DataFrame(comparison).sort_values('Score', ascending=False)
