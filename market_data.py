"""
Módulo de Datos de Mercado (Versión Quant con Macro y Sentimiento)
Integra precios, datos fundamentales y "Sensores Macro" (VIX, Bonos).
ACTUALIZADO: 2026-03-17 - Cache en get_market_regime + descarga paralela
"""

import time
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CACHE GLOBAL para get_market_regime
# Evita descargar 6 meses de datos en cada scan (cada 5 min = 288 veces/día)
# ─────────────────────────────────────────────────────────────────────────────
_regime_cache = {"data": None, "ts": None}
_REGIME_TTL_SECONDS = 1800  # 30 minutos


class MarketDataFetcher:
    """Gestor de datos de mercado y contexto macroeconómico"""

    def __init__(self, config: Dict):
        self.config = config

    # ─────────────────────────────────────────────────────────────────────────
    # SENSOR DE SENTIMIENTO Y MACRO — con caché de 30 min
    # ─────────────────────────────────────────────────────────────────────────

    def get_market_regime(self) -> Dict:
        """
        Analiza el 'clima' global del mercado (Macro + Sentimiento).
        Retorna: 'RISK_ON' (Alcista), 'RISK_OFF' (Bajista/Pánico) o 'NEUTRAL'.

        FIX: Cachea el resultado 30 minutos para evitar descargar 6 meses
        de datos de 4 tickers en cada scan (288 veces/día).
        """
        # Verificar caché
        if (_regime_cache["ts"] is not None and
                time.time() - _regime_cache["ts"] < _REGIME_TTL_SECONDS):
            return _regime_cache["data"]

        try:
            tickers = ['^VIX', '^TNX', 'SPY', 'BTC-USD']
            data = yf.download(tickers, period='6mo', progress=False)['Close']
            data = data.ffill().dropna()

            current_vix = data['^VIX'].iloc[-1]
            spy_price   = data['SPY'].iloc[-1]
            spy_sma200  = data['SPY'].rolling(120).mean().iloc[-1]
            btc_price   = data['BTC-USD'].iloc[-1]
            btc_sma50   = data['BTC-USD'].rolling(50).mean().iloc[-1]

            macro_score = 0
            if current_vix < 20:   macro_score += 1
            elif current_vix > 30: macro_score -= 2
            if spy_price > spy_sma200: macro_score += 1
            else:                      macro_score -= 1
            if btc_price > btc_sma50:  macro_score += 1

            regime = "NEUTRAL"
            if macro_score >= 2:   regime = "RISK_ON"
            elif macro_score <= -1: regime = "RISK_OFF"

            result = {
                'regime':      regime,
                'vix':         current_vix,
                'spy_trend':   'BULLISH' if spy_price > spy_sma200 else 'BEARISH',
                'macro_score': macro_score,
                'description': f"Mercado en modo {regime} (VIX: {current_vix:.1f})"
            }

            # Guardar en caché
            _regime_cache["data"] = result
            _regime_cache["ts"]   = time.time()
            return result

        except Exception as e:
            logger.error(f"Error obteniendo régimen de mercado: {e}")
            return {'regime': 'NEUTRAL', 'vix': 0, 'description': 'Error en datos macro'}

    # ─────────────────────────────────────────────────────────────────────────
    # MÉTODOS ESTÁNDAR
    # ─────────────────────────────────────────────────────────────────────────

    def get_stock_data(self, symbol: str, period: str = '1mo') -> Optional[pd.DataFrame]:
        try:
            ticker = yf.Ticker(symbol)
            data   = ticker.history(period=period, auto_adjust=True)
            if data.empty:
                return None
            data['Symbol']  = symbol
            data['Returns'] = data['Close'].pct_change()
            return data
        except Exception as e:
            logger.error(f"Error en {symbol}: {e}")
            return None

    def get_crypto_data(self, symbol: str, period: str = '1mo') -> Optional[pd.DataFrame]:
        return self.get_stock_data(symbol, period)

    def get_forex_data(self, symbol: str, period: str = '1mo') -> Optional[pd.DataFrame]:
        return self.get_stock_data(symbol, period)

    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = yf.Ticker(symbol)
            return ticker.fast_info.last_price
        except Exception:
            try:
                data = yf.Ticker(symbol).history(period='1d')
                if not data.empty:
                    return data['Close'].iloc[-1]
            except Exception:
                return None
        return None

    def get_market_info(self, symbol: str) -> Dict:
        try:
            i = yf.Ticker(symbol).info
            return {
                'symbol':     symbol,
                'name':       i.get('shortName', symbol),
                'sector':     i.get('sector', 'N/A'),
                'pe_ratio':   i.get('trailingPE', 0),
                'beta':       i.get('beta', 1),
                'market_cap': i.get('marketCap', 0),
            }
        except Exception:
            return {}

    def get_portfolio_data(self, symbols: List[str], period: str = '3mo') -> Dict:
        """
        FIX: Descarga masiva en paralelo con yf.download en lugar de loop secuencial.
        Con 21+ activos es significativamente más rápido.
        """
        if not symbols:
            return {}

        try:
            # Descarga masiva paralela
            raw = yf.download(
                symbols,
                period=period,
                auto_adjust=True,
                progress=False,
                group_by='ticker',
                threads=True,
            )

            data_dict = {}

            if len(symbols) == 1:
                # yf.download con 1 ticker no agrupa por ticker
                sym = symbols[0]
                if not raw.empty:
                    raw['Symbol']  = sym
                    raw['Returns'] = raw['Close'].pct_change()
                    data_dict[sym] = raw
            else:
                for sym in symbols:
                    try:
                        df = raw[sym].copy()
                        df.dropna(how='all', inplace=True)
                        if not df.empty:
                            df['Symbol']  = sym
                            df['Returns'] = df['Close'].pct_change()
                            data_dict[sym] = df
                    except Exception:
                        # Fallback individual si falla el símbolo
                        single = self.get_stock_data(sym, period)
                        if single is not None:
                            data_dict[sym] = single

            return data_dict

        except Exception as e:
            logger.warning(f"Descarga masiva falló ({e}), usando fallback secuencial")
            data_dict = {}
            for symbol in symbols:
                df = self.get_stock_data(symbol, period)
                if df is not None:
                    data_dict[symbol] = df
            return data_dict

    def check_price_alerts(self, symbol: str, threshold: float = 5.0) -> Dict:
        try:
            df = self.get_stock_data(symbol, period='5d')
            if df is None or len(df) < 2:
                return {'alert': False}
            curr = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-2]
            pct  = ((curr - prev) / prev) * 100
            return {
                'symbol':         symbol,
                'alert':          abs(pct) >= threshold,
                'current_price':  curr,
                'previous_price': prev,
                'change_pct':     pct,
                'direction':      'UP' if pct > 0 else 'DOWN',
            }
        except Exception:
            return {'alert': False}

    def get_premarket_data(self, symbols: List[str]) -> Dict:
        """Obtiene datos de pre-market (4:00 AM - 9:30 AM EST)."""
        premarket_dict = {}

        for symbol in symbols:
            try:
                info = yf.Ticker(symbol).info
                premarket_price = info.get('preMarketPrice')
                regular_price   = info.get('regularMarketPrice')
                prev_close      = info.get('previousClose', 0)
                current_price   = premarket_price if premarket_price else regular_price

                if not current_price or not prev_close:
                    continue

                gap_pct = (current_price - prev_close) / prev_close * 100

                premarket_dict[symbol] = {
                    'price':        current_price,
                    'prev_close':   prev_close,
                    'gap_pct':      gap_pct,
                    'change_pct':   gap_pct,
                    'volume':       info.get('preMarketVolume', 0),
                    'is_premarket': bool(premarket_price),
                    'timestamp':    datetime.now(),
                }
            except Exception as e:
                logger.error(f"Error pre-market {symbol}: {e}")
                continue

        return premarket_dict
