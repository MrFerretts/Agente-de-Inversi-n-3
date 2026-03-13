"""
Módulo de Datos de Mercado (Versión Quant con Macro y Sentimiento)
Integra precios, datos fundamentales y "Sensores Macro" (VIX, Bonos).
ACTUALIZADO: 2026-02-12 - Agregado soporte PRE-MARKET
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataFetcher:
    """Gestor de datos de mercado y contexto macroeconómico"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    # --- NUEVO: SENSOR DE SENTIMIENTO Y MACRO ---
    def get_market_regime(self) -> Dict:
        """
        Analiza el 'clima' global del mercado (Macro + Sentimiento).
        Retorna: 'RISK_ON' (Alcista), 'RISK_OFF' (Bajista/Pánico) o 'NEUTRAL'.
        """
        try:
            # 1. Obtener datos de referencia global
            tickers = ['^VIX', '^TNX', 'SPY', 'BTC-USD']
            data = yf.download(tickers, period='6mo', progress=False)['Close']
            
            # Limpieza básica por si faltan datos
            data = data.ffill().dropna()
            
            # 2. Análisis del VIX (Índice de Miedo)
            # VIX > 30 = Pánico extremo | VIX < 20 = Complacencia
            current_vix = data['^VIX'].iloc[-1]
            vix_ma = data['^VIX'].rolling(20).mean().iloc[-1]
            
            # 3. Tendencia del Mercado (SPY y BTC)
            spy_price = data['SPY'].iloc[-1]
            spy_sma200 = data['SPY'].rolling(120).mean().iloc[-1] # Usamos 120 días como proxy rápido
            
            btc_price = data['BTC-USD'].iloc[-1]
            btc_sma50 = data['BTC-USD'].rolling(50).mean().iloc[-1]

            # 4. Lógica de Semáforo (Score Macro)
            macro_score = 0
            
            # Evaluar VIX (Miedo)
            if current_vix < 20: macro_score += 1     # Calma
            elif current_vix > 30: macro_score -= 2   # Pánico
            
            # Evaluar Tendencia SP500 (Salud Económica)
            if spy_price > spy_sma200: macro_score += 1
            else: macro_score -= 1
            
            # Evaluar Momentum de Crypto (Apetito de riesgo especulativo)
            if btc_price > btc_sma50: macro_score += 1
            
            # Determinar Régimen
            regime = "NEUTRAL"
            if macro_score >= 2: regime = "RISK_ON"   # Mercado favorable
            elif macro_score <= -1: regime = "RISK_OFF" # Mercado peligroso (Cash is king)
            
            return {
                'regime': regime,
                'vix': current_vix,
                'spy_trend': 'BULLISH' if spy_price > spy_sma200 else 'BEARISH',
                'macro_score': macro_score,
                'description': f"Mercado en modo {regime} (VIX: {current_vix:.1f})"
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo régimen de mercado: {e}")
            # En caso de error, asumimos neutralidad defensiva
            return {'regime': 'NEUTRAL', 'vix': 0, 'description': 'Error en datos macro'}

    # --- MÉTODOS ESTÁNDAR (Optimizados) ---

    def get_stock_data(self, symbol: str, period: str = '1mo') -> Optional[pd.DataFrame]:
        try:
            ticker = yf.Ticker(symbol)
            # Forzamos auto_adjust=True para corregir dividendos/splits
            data = ticker.history(period=period, auto_adjust=True)
            
            if data.empty: return None
            
            data['Symbol'] = symbol
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
            # Método rápido: fast_info (más eficiente que .info completo)
            return ticker.fast_info.last_price
        except:
            # Fallback al método lento
            try:
                data = ticker.history(period='1d')
                if not data.empty: return data['Close'].iloc[-1]
            except:
                return None
            return None

    def get_market_info(self, symbol: str) -> Dict:
        """Información fundamental básica"""
        try:
            t = yf.Ticker(symbol)
            i = t.info
            return {
                'symbol': symbol,
                'name': i.get('shortName', symbol),
                'sector': i.get('sector', 'N/A'),
                'pe_ratio': i.get('trailingPE', 0),
                'beta': i.get('beta', 1),
                'market_cap': i.get('marketCap', 0)
            }
        except:
            return {}

    def get_portfolio_data(self, symbols: List[str], period: str = '3mo') -> Dict:
        """Descarga masiva optimizada (Threads)"""
        data_dict = {}
        for symbol in symbols:
            df = self.get_stock_data(symbol, period)
            if df is not None:
                data_dict[symbol] = df
        return data_dict
    
    def check_price_alerts(self, symbol: str, threshold: float = 5.0) -> Dict:
        try:
            df = self.get_stock_data(symbol, period='5d')
            if df is None or len(df) < 2: return {'alert': False}
            
            curr = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-2]
            pct = ((curr - prev) / prev) * 100
            
            return {
                'symbol': symbol,
                'alert': abs(pct) >= threshold,
                'current_price': curr,
                'previous_price': prev,
                'change_pct': pct,
                'direction': 'UP' if pct > 0 else 'DOWN'
            }
        except:
            return {'alert': False}
    
    # ===================================================================
    # 🌅 NUEVO: MÉTODO PRE-MARKET
    # ===================================================================
    def get_premarket_data(self, symbols: List[str]) -> Dict:
        """
        Obtiene datos de pre-market (4:00 AM - 9:30 AM EST)
        Devuelve precios actuales y gaps vs cierre anterior
        
        Returns:
            Dict con estructura:
            {
                'AAPL': {
                    'price': 275.50,
                    'prev_close': 275.23,
                    'gap_pct': 0.10,
                    'volume': 125000,
                    'is_premarket': True
                }
            }
        """
        premarket_dict = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # Intentar obtener info completa
                try:
                    info = ticker.info
                    
                    # Precio pre-market (si está disponible)
                    premarket_price = info.get('preMarketPrice')
                    regular_price = info.get('regularMarketPrice')
                    prev_close = info.get('previousClose', 0)
                    
                    # Usar pre-market si existe, sino regular market
                    current_price = premarket_price if premarket_price else regular_price
                    
                    if not current_price or not prev_close:
                        continue
                    
                    # Calcular gap
                    gap_pct = ((current_price - prev_close) / prev_close * 100)
                    
                    # Volumen pre-market
                    premarket_volume = info.get('preMarketVolume', 0)
                    
                    premarket_dict[symbol] = {
                        'price': current_price,
                        'prev_close': prev_close,
                        'gap_pct': gap_pct,
                        'change_pct': gap_pct,
                        'volume': premarket_volume,
                        'is_premarket': bool(premarket_price),
                        'timestamp': datetime.now()
                    }
                    
                except Exception as e:
                    # Fallback: usar solo precio actual
                    try:
                        current_price = self.get_current_price(symbol)
                        if current_price:
                            premarket_dict[symbol] = {
                                'price': current_price,
                                'prev_close': 0,
                                'gap_pct': 0,
                                'change_pct': 0,
                                'volume': 0,
                                'is_premarket': False,
                                'timestamp': datetime.now()
                            }
                    except:
                        continue
                
            except Exception as e:
                logger.error(f"Error pre-market {symbol}: {e}")
                continue
        
        return premarket_dict
