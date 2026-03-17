"""
╔══════════════════════════════════════════════════════════════════╗
║         PATO QUANT — MOTOR DE TRADING AUTÓNOMO                  ║
║                                                                  ║
║  El agente decide SOLO cuándo entrar y salir.                   ║
║  Usa Alpaca Paper Trading (dinero ficticio).                    ║
║                                                                  ║
║  Lógica de decisión:                                            ║
║    → Score técnico + ML + régimen de mercado                    ║
║    → Tamaño de posición: % del portafolio según volatilidad     ║
║    → Stop loss / Take profit: dinámicos basados en ATR          ║
║    → Gestión de riesgo: drawdown máximo, exposición total       ║
║                                                                  ║
║  Se integra en scheduler.py como job adicional.                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import pytz
import requests

logger = logging.getLogger("PatoQuant.Trader")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE RIESGO
# Todos los parámetros ajustables sin tocar la lógica
# ─────────────────────────────────────────────────────────────────────────────

RISK_CONFIG = {
    # ── Tamaño de posición ────────────────────────────────────────────────────
    "position_pct":        0.05,    # 5% del portafolio por trade
    "max_position_pct":    0.10,    # Nunca más del 10% en un solo activo
    "min_position_usd":    50.0,    # Mínimo $50 por trade (evitar comisiones)

    # ── Stop Loss / Take Profit dinámicos (basados en ATR) ────────────────────
    "stop_loss_atr_mult":  2.0,     # Stop = precio_entrada - (ATR * 2.0)
    "take_profit_atr_mult": 4.0,    # TP   = precio_entrada + (ATR * 4.0) → R:R 1:2
    "trailing_stop":       True,    # Activar trailing stop
    "trailing_atr_mult":   1.5,     # Trailing = precio_max - (ATR * 1.5)

    # ── Filtros de entrada ────────────────────────────────────────────────────
    "min_score":           55,      # Score técnico mínimo para comprar
    "min_adx":             20,      # Tendencia mínima confirmada
    "max_rsi_entry":       72,      # No comprar en sobrecompra extrema
    "min_rsi_entry":       25,      # No comprar en caída libre
    "require_market_open": True,    # Solo operar en horario regular

    # ── Gestión de riesgo del portafolio ──────────────────────────────────────
    "max_drawdown_pct":    0.15,    # Pausar si portafolio cae 15% desde pico
    "max_daily_loss_pct":  0.05,    # Pausar si pierde 5% en el día
    "max_total_exposure":  0.80,    # Máximo 80% del portafolio invertido

    # ── Cooldown ─────────────────────────────────────────────────────────────
    "trade_cooldown_hours": 4,      # Esperar 4h antes de re-entrar en mismo ticker
    "max_trades_per_day":  10,      # Máximo 10 operaciones por día

    # ── Correlación sectorial ────────────────────────────────────────────────
    "max_positions_per_sector": 3,  # Máximo 3 posiciones en el mismo sector
}


# ─────────────────────────────────────────────────────────────────────────────
# MAPA DE SECTORES — Evita concentración en activos correlacionados
# ─────────────────────────────────────────────────────────────────────────────

SECTOR_MAP = {
    # Tech
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "GOOG": "tech",
    "META": "tech", "NVDA": "tech", "AMD": "tech", "INTC": "tech",
    "CRM": "tech", "ORCL": "tech", "ADBE": "tech", "CSCO": "tech",
    "AVGO": "tech", "QCOM": "tech", "MU": "tech", "AMAT": "tech",
    "PLTR": "tech", "SNOW": "tech", "NET": "tech", "DDOG": "tech",
    # E-commerce / Internet
    "AMZN": "ecommerce", "SHOP": "ecommerce", "MELI": "ecommerce",
    "BABA": "ecommerce", "JD": "ecommerce", "EBAY": "ecommerce",
    # Fintech / Finanzas
    "V": "fintech", "MA": "fintech", "PYPL": "fintech", "SQ": "fintech",
    "COIN": "fintech", "JPM": "fintech", "GS": "fintech", "BAC": "fintech",
    "WFC": "fintech", "C": "fintech", "SOFI": "fintech",
    # Auto / EV
    "TSLA": "auto", "F": "auto", "GM": "auto", "RIVN": "auto",
    "LCID": "auto", "NIO": "auto",
    # Media / Entertainment
    "DIS": "media", "NFLX": "media", "CMCSA": "media", "WBD": "media",
    "SPOT": "media", "ROKU": "media",
    # Healthcare
    "JNJ": "health", "UNH": "health", "PFE": "health", "ABBV": "health",
    "MRK": "health", "LLY": "health", "TMO": "health",
    # Energy
    "XOM": "energy", "CVX": "energy", "COP": "energy", "OXY": "energy",
    "USO": "energy", "XLE": "energy",
    # Crypto
    "BTC-USD": "crypto", "ETH-USD": "crypto", "SOL-USD": "crypto",
    "XRP-USD": "crypto", "ADA-USD": "crypto", "DOGE-USD": "crypto",
    "AVAX-USD": "crypto", "DOT-USD": "crypto", "MATIC-USD": "crypto",
    "MSTR": "crypto",
    # Commodities
    "GC=F": "commodity", "SI=F": "commodity", "CL=F": "commodity",
    "GLD": "commodity", "SLV": "commodity",
    # Aerospace / Defense
    "RKLB": "aerospace", "BA": "aerospace", "LMT": "aerospace",
    "RTX": "aerospace", "NOC": "aerospace",
    # ETFs sectoriales
    "SPY": "index_etf", "QQQ": "index_etf", "IWM": "index_etf",
    "DIA": "index_etf", "VOO": "index_etf",
    "XLF": "sector_etf", "XLV": "sector_etf", "XLU": "sector_etf",
    "XLK": "sector_etf", "XLE": "sector_etf", "XLI": "sector_etf",
    # Cybersecurity
    "OKTA": "cybersec", "CRWD": "cybersec", "ZS": "cybersec",
    "PANW": "cybersec", "FTNT": "cybersec",
}


def get_sector(ticker: str) -> str:
    """Retorna el sector del ticker. Default: 'other'."""
    return SECTOR_MAP.get(ticker, "other")


# ─────────────────────────────────────────────────────────────────────────────
# CLIENTE ALPACA
# ─────────────────────────────────────────────────────────────────────────────

class AlpacaClient:
    """
    Wrapper simple para la API de Alpaca.
    Usa paper trading por defecto.
    """

    BASE_URL = "https://paper-api.alpaca.markets"

    def __init__(self):
        self.api_key    = os.getenv("ALPACA_API_KEY", "")
        self.api_secret = os.getenv("ALPACA_API_SECRET", "")

        if not self.api_key or not self.api_secret:
            raise ValueError("ALPACA_API_KEY y ALPACA_API_SECRET requeridos")

        self.headers = {
            "APCA-API-KEY-ID":     self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type":        "application/json",
        }

    def _get(self, endpoint: str) -> Dict:
        r = requests.get(f"{self.BASE_URL}{endpoint}",
                         headers=self.headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def _post(self, endpoint: str, data: Dict) -> Dict:
        r = requests.post(f"{self.BASE_URL}{endpoint}",
                          headers=self.headers, json=data, timeout=10)
        r.raise_for_status()
        return r.json()

    def _delete(self, endpoint: str) -> bool:
        r = requests.delete(f"{self.BASE_URL}{endpoint}",
                            headers=self.headers, timeout=10)
        return r.status_code in (200, 204)

    # ── Cuenta ────────────────────────────────────────────────────────────────

    def get_account(self) -> Dict:
        return self._get("/v2/account")

    def get_portfolio_value(self) -> float:
        acc = self.get_account()
        return float(acc.get("portfolio_value", 0))

    def get_buying_power(self) -> float:
        acc = self.get_account()
        return float(acc.get("buying_power", 0))

    def get_equity(self) -> float:
        acc = self.get_account()
        return float(acc.get("equity", 0))

    # ── Posiciones ────────────────────────────────────────────────────────────

    def get_positions(self) -> List[Dict]:
        return self._get("/v2/positions")

    def get_position(self, symbol: str) -> Optional[Dict]:
        try:
            return self._get(f"/v2/positions/{symbol}")
        except Exception:
            return None

    def close_position(self, symbol: str) -> bool:
        try:
            self._delete(f"/v2/positions/{symbol}")
            logger.info(f"✅ Posición cerrada: {symbol}")
            return True
        except Exception as e:
            logger.error(f"❌ Error cerrando {symbol}: {e}")
            return False

    def close_all_positions(self) -> bool:
        try:
            self._delete("/v2/positions")
            logger.info("✅ Todas las posiciones cerradas")
            return True
        except Exception as e:
            logger.error(f"❌ Error cerrando todas: {e}")
            return False

    # ── Órdenes ───────────────────────────────────────────────────────────────

    def submit_order(self, symbol: str, qty: float,
                     side: str, order_type: str = "market",
                     time_in_force: str = "day",
                     limit_price: float = None,
                     stop_price: float = None) -> Optional[Dict]:
        """
        Envía una orden a Alpaca.
        side: 'buy' | 'sell'
        order_type: 'market' | 'limit' | 'stop' | 'stop_limit'
        """
        data = {
            "symbol":        symbol,
            "qty":           str(round(qty, 4)),
            "side":          side,
            "type":          order_type,
            "time_in_force": time_in_force,
        }
        if limit_price:
            data["limit_price"] = str(round(limit_price, 2))
        if stop_price:
            data["stop_price"] = str(round(stop_price, 2))

        try:
            order = self._post("/v2/orders", data)
            logger.info(
                f"📋 Orden enviada: {side.upper()} {qty:.4f} {symbol} "
                f"@ {order_type} | ID: {order.get('id','?')[:8]}"
            )
            return order
        except Exception as e:
            logger.error(f"❌ Error enviando orden {side} {symbol}: {e}")
            return None

    def get_orders(self, status: str = "open") -> List[Dict]:
        return self._get(f"/v2/orders?status={status}")

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._delete(f"/v2/orders/{order_id}")
            return True
        except Exception:
            return False

    def cancel_all_orders(self):
        try:
            self._delete("/v2/orders")
        except Exception:
            pass

    # ── Precio actual ─────────────────────────────────────────────────────────

    def get_latest_price(self, symbol: str) -> Optional[float]:
        try:
            data = self._get(f"/v2/stocks/{symbol}/quotes/latest")
            quote = data.get("quote", {})
            # Usar mid-price si está disponible
            ask = float(quote.get("ap", 0))
            bid = float(quote.get("bp", 0))
            if ask > 0 and bid > 0:
                return (ask + bid) / 2
            return ask or bid or None
        except Exception:
            try:
                data = self._get(f"/v2/stocks/{symbol}/trades/latest")
                return float(data["trade"]["p"])
            except Exception:
                return None


# ─────────────────────────────────────────────────────────────────────────────
# CEREBRO DE DECISIÓN
# ─────────────────────────────────────────────────────────────────────────────

class TradingBrain:
    """
    Decide si comprar, mantener o vender basándose en:
    - Score técnico del scanner
    - Predicción ML (si disponible)
    - Régimen de mercado
    - Gestión de riesgo del portafolio
    """

    def __init__(self, alpaca: AlpacaClient):
        self.alpaca        = alpaca
        self.trade_history: List[Dict] = []  # Log de operaciones del día
        self.daily_pnl     = 0.0
        self.peak_equity   = None
        self.last_trade_time: Dict[str, datetime] = {}

    # ── Decisión de entrada ───────────────────────────────────────────────────

    def should_buy(self, ticker: str, scan_result: Dict,
                   ml_result: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Evalúa si el sistema debe comprar este ticker.
        Retorna (True/False, razón).
        """
        score   = float(scan_result.get("score", 0))
        rsi     = float(scan_result.get("rsi", 50))
        adx     = float(scan_result.get("adx", 0))
        rvol    = float(scan_result.get("rvol", 1))
        rec     = scan_result.get("recommendation", "")

        # ── 1. Filtros básicos ────────────────────────────────────────────────

        if score < RISK_CONFIG["min_score"]:
            return False, f"Score insuficiente ({score:.0f} < {RISK_CONFIG['min_score']})"

        if rsi > RISK_CONFIG["max_rsi_entry"]:
            return False, f"RSI sobrecomprado ({rsi:.1f})"

        if rsi < RISK_CONFIG["min_rsi_entry"]:
            return False, f"RSI en caída libre ({rsi:.1f})"

        if adx < RISK_CONFIG["min_adx"]:
            return False, f"Tendencia débil (ADX {adx:.1f})"

        if rec not in ("COMPRA", "COMPRA FUERTE"):
            return False, f"Señal no alcista ({rec})"

        # ── 2. Cooldown por ticker ────────────────────────────────────────────

        if ticker in self.last_trade_time:
            hours_since = (datetime.now() - self.last_trade_time[ticker]).seconds / 3600
            if hours_since < RISK_CONFIG["trade_cooldown_hours"]:
                return False, f"Cooldown activo ({hours_since:.1f}h < {RISK_CONFIG['trade_cooldown_hours']}h)"

        # ── 3. ¿Ya tenemos posición en este ticker? ───────────────────────────

        pos = self.alpaca.get_position(ticker)
        if pos:
            return False, "Ya hay posición abierta"

        # ── 4. Límite de trades diarios ───────────────────────────────────────

        today_trades = [t for t in self.trade_history
                        if t.get("date") == datetime.now().strftime("%Y-%m-%d")]
        if len(today_trades) >= RISK_CONFIG["max_trades_per_day"]:
            return False, f"Límite diario alcanzado ({RISK_CONFIG['max_trades_per_day']})"

        # ── 5. Verificar drawdown del portafolio ──────────────────────────────

        ok, reason = self._check_portfolio_health()
        if not ok:
            return False, reason

        # ── 5b. Límite de posiciones por sector (anti-correlación) ───────────

        sector = get_sector(ticker)
        positions = self.alpaca.get_positions()
        sector_count = sum(
            1 for p in positions
            if get_sector(p.get("symbol", "")) == sector
        )
        max_per_sector = RISK_CONFIG["max_positions_per_sector"]
        if sector_count >= max_per_sector:
            return False, (
                f"Sector '{sector}' saturado ({sector_count}/{max_per_sector} posiciones)"
            )

        # ── 6. Filtro ML (bonus de confianza si disponible) ───────────────────

        ml_confidence = 0.0
        if ml_result and ml_result.get("probability_up"):
            ml_confidence = float(ml_result["probability_up"])
            # Si ML predice bajada con alta confianza, bloqueamos la entrada
            if ml_confidence < 0.35:
                return False, f"ML predice bajada ({ml_confidence*100:.0f}% prob subida)"

        # ── 7. Score compuesto final ──────────────────────────────────────────

        composite = score
        if ml_confidence > 0.6:
            composite += 10  # Bonus si ML confirma
        if rvol > 2.0:
            composite += 5   # Bonus si hay volumen institucional

        if composite < RISK_CONFIG["min_score"]:
            return False, f"Score compuesto insuficiente ({composite:.0f})"

        return True, f"Score={composite:.0f} | RSI={rsi:.1f} | ADX={adx:.1f} | ML={ml_confidence*100:.0f}%"

    # ── Tamaño de posición ────────────────────────────────────────────────────

    def calculate_position_size(self, ticker: str, price: float,
                                  atr: float) -> Tuple[float, float, float]:
        """
        Calcula tamaño de posición usando volatilidad (ATR).

        Método: Risk-based position sizing
          - Arriesgar N% del portafolio por trade
          - Stop loss = ATR * multiplicador
          - Qty = (portafolio * risk_pct) / stop_distance

        Retorna: (qty, stop_loss_price, take_profit_price)
        """
        portfolio_value = self.alpaca.get_portfolio_value()

        # Riesgo en dólares = 5% del portafolio
        risk_dollars = portfolio_value * RISK_CONFIG["position_pct"]

        # Stop distance basado en ATR
        stop_distance   = atr * RISK_CONFIG["stop_loss_atr_mult"]
        profit_distance = atr * RISK_CONFIG["take_profit_atr_mult"]

        # Evitar divisiones por cero
        if stop_distance < 0.01:
            stop_distance = price * 0.03  # Fallback: 3% del precio

        # Número de acciones que podemos comprar arriesgando risk_dollars
        qty = risk_dollars / stop_distance

        # Verificar que no exceda max_position_pct del portafolio
        max_value = portfolio_value * RISK_CONFIG["max_position_pct"]
        qty = min(qty, max_value / price)

        # Verificar mínimo
        if qty * price < RISK_CONFIG["min_position_usd"]:
            qty = RISK_CONFIG["min_position_usd"] / price

        # Verificar buying power
        buying_power = self.alpaca.get_buying_power()
        if qty * price > buying_power * 0.95:
            qty = (buying_power * 0.95) / price

        # Redondear a 4 decimales para fractional shares
        qty = max(round(qty, 4), 0.0001)

        stop_loss   = round(price - stop_distance,   2)
        take_profit = round(price + profit_distance, 2)

        return qty, stop_loss, take_profit

    # ── Decisión de salida ────────────────────────────────────────────────────

    def should_sell(self, position: Dict, scan_result: Dict) -> Tuple[bool, str]:
        """
        Evalúa si cerrar una posición existente.
        Usa trailing stop dinámico + señales técnicas.
        """
        symbol      = position.get("symbol", "")
        entry_price = float(position.get("avg_entry_price", 0))
        current_val = float(position.get("current_price", 0))
        unrealized  = float(position.get("unrealized_plpc", 0)) * 100  # en %
        qty         = float(position.get("qty", 0))

        score = float(scan_result.get("score", 0))
        rsi   = float(scan_result.get("rsi", 50))
        rec   = scan_result.get("recommendation", "")

        # Recuperar ATR del scan para trailing stop
        atr = float(scan_result.get("atr", current_val * 0.02))

        # ── Stop loss dinámico ────────────────────────────────────────────────
        stop_distance = atr * RISK_CONFIG["stop_loss_atr_mult"]
        stop_price    = entry_price - stop_distance

        if current_val <= stop_price:
            return True, f"Stop loss activado (precio {current_val:.2f} ≤ stop {stop_price:.2f})"

        # ── Take profit ───────────────────────────────────────────────────────
        tp_distance = atr * RISK_CONFIG["take_profit_atr_mult"]
        tp_price    = entry_price + tp_distance

        if current_val >= tp_price:
            return True, f"Take profit alcanzado (+{unrealized:.1f}% | TP {tp_price:.2f})"

        # ── Señal técnica bajista fuerte ──────────────────────────────────────
        if rec in ("VENTA", "VENTA FUERTE") and score <= -40:
            return True, f"Señal bajista fuerte (Score {score:.0f})"

        # ── RSI sobrecomprado extremo ─────────────────────────────────────────
        if rsi > 78 and unrealized > 5:
            return True, f"RSI sobrecomprado ({rsi:.1f}) con ganancia ({unrealized:.1f}%)"

        # ── Pérdida máxima tolerada ───────────────────────────────────────────
        max_loss_pct = -RISK_CONFIG["stop_loss_atr_mult"] * 100 * (atr / entry_price)
        if unrealized < max_loss_pct * 1.5:  # 50% peor que el stop calculado
            return True, f"Pérdida excesiva ({unrealized:.1f}%)"

        return False, f"Mantener (Score {score:.0f} | {unrealized:+.1f}%)"

    # ── Salud del portafolio ──────────────────────────────────────────────────

    def _check_portfolio_health(self) -> Tuple[bool, str]:
        """Verifica que el portafolio no esté en drawdown excesivo."""
        try:
            equity = self.alpaca.get_equity()

            # Inicializar pico si es la primera vez
            if self.peak_equity is None:
                self.peak_equity = equity
            else:
                self.peak_equity = max(self.peak_equity, equity)

            # Drawdown actual
            if self.peak_equity > 0:
                drawdown = (self.peak_equity - equity) / self.peak_equity
                if drawdown > RISK_CONFIG["max_drawdown_pct"]:
                    return False, f"Drawdown máximo alcanzado ({drawdown*100:.1f}%)"

            # Exposición total
            positions   = self.alpaca.get_positions()
            total_invested = sum(
                float(p.get("market_value", 0)) for p in positions
            )
            exposure = total_invested / equity if equity > 0 else 0

            if exposure > RISK_CONFIG["max_total_exposure"]:
                return False, f"Exposición máxima alcanzada ({exposure*100:.1f}%)"

            return True, "OK"

        except Exception as e:
            logger.warning(f"⚠️ No se pudo verificar salud del portafolio: {e}")
            return True, "OK (sin datos)"

    def log_trade(self, ticker: str, action: str, qty: float,
                  price: float, reason: str):
        """Registra operación en el historial."""
        self.last_trade_time[ticker] = datetime.now()
        self.trade_history.append({
            "date":   datetime.now().strftime("%Y-%m-%d"),
            "time":   datetime.now().strftime("%H:%M:%S"),
            "ticker": ticker,
            "action": action,
            "qty":    qty,
            "price":  price,
            "reason": reason,
        })


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR PRINCIPAL DE TRADING
# ─────────────────────────────────────────────────────────────────────────────

class AutonomousTrader:
    """
    Motor de trading autónomo.
    Se llama desde scheduler.py después de cada scan.

    Flujo:
      1. Revisar posiciones abiertas → ¿cerrar?
      2. Revisar resultados del scan → ¿abrir nuevas?
      3. Log de todo en Supabase
    """

    def __init__(self, db, notifier, perf_tracker=None):
        self.db            = db
        self.notifier      = notifier
        self.perf_tracker  = perf_tracker

        try:
            self.alpaca = AlpacaClient()
            self.brain  = TradingBrain(self.alpaca)
            self.active = True
            logger.info("💰 Motor de Trading Autónomo inicializado (Paper Trading)")
            logger.info(f"   Capital por trade: {RISK_CONFIG['position_pct']*100:.0f}% del portafolio")
            logger.info(f"   Stop/TP: {RISK_CONFIG['stop_loss_atr_mult']}x ATR / {RISK_CONFIG['take_profit_atr_mult']}x ATR")
        except Exception as e:
            self.active = False
            logger.error(f"❌ No se pudo inicializar Alpaca: {e}")

    # ── Job principal ─────────────────────────────────────────────────────────

    def run(self, scan_results: List[Dict], ml_models: Dict = None):
        """
        Ejecuta el ciclo completo de trading.
        Llamar después de cada scan del scheduler.

        Args:
            scan_results: Lista de resultados del scan actual
            ml_models:    Dict {ticker: model} para predicciones
        """
        if not self.active:
            logger.debug("🔒 Trader inactivo (Alpaca no configurado)")
            return

        logger.info("💰 Iniciando ciclo de trading autónomo...")

        try:
            # ── Paso 1: Gestionar posiciones existentes ───────────────────────
            self._manage_open_positions(scan_results)

            # ── Paso 2: Buscar nuevas entradas ────────────────────────────────
            self._find_new_entries(scan_results, ml_models or {})

            # ── Paso 3: Resumen ───────────────────────────────────────────────
            self._log_portfolio_summary()

        except Exception as e:
            logger.error(f"❌ Error en ciclo de trading: {e}")
            logger.debug(traceback.format_exc())

    def _manage_open_positions(self, scan_results: List[Dict]):
        """Revisa posiciones abiertas y decide si cerrarlas."""
        positions = self.alpaca.get_positions()

        if not positions:
            return

        logger.info(f"   Revisando {len(positions)} posiciones abiertas...")

        # Crear dict de scan_results por ticker para búsqueda rápida
        scan_map = {r["ticker"]: r for r in scan_results}

        for position in positions:
            ticker  = position.get("symbol", "")
            qty     = float(position.get("qty", 0))
            pnl_pct = float(position.get("unrealized_plpc", 0)) * 100
            price   = float(position.get("current_price", 0))

            scan = scan_map.get(ticker)
            if not scan:
                logger.debug(f"   {ticker}: sin datos de scan, manteniendo")
                continue

            should_close, reason = self.brain.should_sell(position, scan)

            if should_close:
                logger.info(f"   🔴 CERRAR {ticker}: {reason}")
                order = self.alpaca.submit_order(
                    symbol=ticker, qty=qty, side="sell"
                )
                if order:
                    self.brain.log_trade(ticker, "SELL", qty, price, reason)
                    self._notify_trade(ticker, "VENTA", qty, price, pnl_pct, reason)
                    self._save_trade_db(ticker, "SELL", qty, price, pnl_pct, reason)

                    # Registrar en performance tracker
                    if self.perf_tracker:
                        entry = float(position.get("avg_entry_price", 0))
                        pnl_usd = (price - entry) * qty
                        self.perf_tracker.record_trade(
                            ticker=ticker, action="SELL", qty=qty,
                            entry_price=entry, exit_price=price,
                            pnl=pnl_usd, reason=reason,
                        )
            else:
                logger.info(f"   🟡 MANTENER {ticker}: {reason}")

    def _find_new_entries(self, scan_results: List[Dict], ml_models: Dict):
        """Busca oportunidades de entrada en los resultados del scan."""

        # Ordenar por score descendente
        candidates = sorted(
            scan_results,
            key=lambda x: float(x.get("score", 0)),
            reverse=True
        )

        entries_this_cycle = 0

        for result in candidates:
            ticker = result.get("ticker", "")
            score  = float(result.get("score", 0))
            price  = float(result.get("price", 0))
            atr    = float(result.get("atr", price * 0.02))

            if score < RISK_CONFIG["min_score"]:
                break  # Ordenados por score, podemos parar

            if price <= 0:
                continue

            # Obtener predicción ML si hay modelo entrenado
            ml_result = None
            if ticker in ml_models:
                try:
                    ml_result = ml_models[ticker].predict_latest(result)
                except Exception:
                    pass

            # ¿Debemos comprar?
            should_buy, reason = self.brain.should_buy(ticker, result, ml_result)

            if should_buy:
                qty, stop_loss, take_profit = self.brain.calculate_position_size(
                    ticker, price, atr
                )

                logger.info(
                    f"   🟢 COMPRAR {ticker}: {reason}\n"
                    f"      Qty: {qty:.4f} | Precio: ${price:.2f} | "
                    f"SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}"
                )

                order = self.alpaca.submit_order(
                    symbol=ticker, qty=qty, side="buy"
                )

                if order:
                    self.brain.log_trade(ticker, "BUY", qty, price, reason)
                    self._notify_trade(ticker, "COMPRA", qty, price, 0, reason,
                                       stop_loss=stop_loss, take_profit=take_profit)
                    self._save_trade_db(ticker, "BUY", qty, price, 0, reason,
                                        stop_loss=stop_loss, take_profit=take_profit)
                    entries_this_cycle += 1

        if entries_this_cycle == 0:
            logger.info("   Sin nuevas entradas en este ciclo")
        else:
            logger.info(f"   ✅ {entries_this_cycle} nuevas posiciones abiertas")

    def _log_portfolio_summary(self):
        """Log del estado del portafolio."""
        try:
            account   = self.alpaca.get_account()
            equity    = float(account.get("equity", 0))
            cash      = float(account.get("cash", 0))
            positions = self.alpaca.get_positions()

            total_pnl = sum(float(p.get("unrealized_pl", 0)) for p in positions)
            total_pnl_pct = sum(
                float(p.get("unrealized_plpc", 0)) * 100 for p in positions
            ) / max(len(positions), 1)

            logger.info(
                f"\n{'─'*50}\n"
                f"💼 PORTAFOLIO PAPER TRADING\n"
                f"   Equity:     ${equity:>10.2f}\n"
                f"   Cash:       ${cash:>10.2f}\n"
                f"   Posiciones: {len(positions)}\n"
                f"   PnL abierto: ${total_pnl:>+8.2f} ({total_pnl_pct:+.1f}%)\n"
                f"{'─'*50}"
            )
        except Exception as e:
            logger.warning(f"⚠️ No se pudo obtener resumen: {e}")

    # ── Notificaciones ────────────────────────────────────────────────────────

    def _notify_trade(self, ticker: str, action: str, qty: float,
                       price: float, pnl_pct: float, reason: str,
                       stop_loss: float = None, take_profit: float = None):
        """Envía notificación Telegram de la operación."""
        emoji = "🟢" if action == "COMPRA" else "🔴"
        pnl_str = f" | PnL: {pnl_pct:+.1f}%" if action == "VENTA" else ""

        msg = (
            f"{emoji} *PAPER TRADE — {action}*\n"
            f"📈 Ticker: *{ticker}*\n"
            f"💲 Precio: ${price:.2f}\n"
            f"📦 Qty: {qty:.4f} acciones\n"
            f"💰 Valor: ${qty*price:.2f}{pnl_str}\n"
        )

        if stop_loss and take_profit:
            msg += (
                f"🛡️ Stop Loss: ${stop_loss:.2f}\n"
                f"🎯 Take Profit: ${take_profit:.2f}\n"
            )

        msg += f"\n📊 Razón: {reason}"

        try:
            self.notifier.send_telegram(msg, ticker, f"trade_{action.lower()}")
        except Exception:
            pass

    # ── Persistencia ──────────────────────────────────────────────────────────

    def _save_trade_db(self, ticker: str, action: str, qty: float,
                        price: float, pnl_pct: float, reason: str,
                        stop_loss: float = None, take_profit: float = None):
        """Guarda la operación en Supabase."""
        try:
            import json
            message = json.dumps({
                "action":      action,
                "qty":         qty,
                "price":       price,
                "pnl_pct":     pnl_pct,
                "stop_loss":   stop_loss,
                "take_profit": take_profit,
                "reason":      reason,
            })
            self.db.save_alert(ticker, f"trade_{action.lower()}", message, "alpaca")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo guardar trade en BD: {e}")

    # ── Estado del trader ─────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Retorna estado actual para el dashboard."""
        if not self.active:
            return {"active": False, "error": "Alpaca no configurado"}

        try:
            account   = self.alpaca.get_account()
            positions = self.alpaca.get_positions()

            return {
                "active":           True,
                "mode":             "PAPER TRADING",
                "equity":           float(account.get("equity", 0)),
                "cash":             float(account.get("cash", 0)),
                "buying_power":     float(account.get("buying_power", 0)),
                "open_positions":   len(positions),
                "positions":        positions,
                "trades_today":     len([
                    t for t in self.brain.trade_history
                    if t.get("date") == datetime.now().strftime("%Y-%m-%d")
                ]),
                "trade_history":    self.brain.trade_history[-20:],
            }
        except Exception as e:
            return {"active": False, "error": str(e)}
