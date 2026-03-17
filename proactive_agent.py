"""
╔══════════════════════════════════════════════════════════════════╗
║         PATO QUANT — AGENTE PROACTIVO DE OPORTUNIDADES          ║
║                                                                  ║
║  Este agente corre en background y:                             ║
║    → Escanea el S&P500, crypto y ETFs globales                  ║
║    → Detecta volumen anómalo, momentum, rupturas                ║
║    → Consulta a Groq qué activos están trending                 ║
║    → Agrega automáticamente los mejores a la watchlist          ║
║    → Elimina los que ya no tienen oportunidad                   ║
║    → Mantiene máximo 50 activos                                 ║
║                                                                  ║
║  Se integra en scheduler.py — no correr solo.                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import logging
import requests
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
import pytz

from core.circuit_breaker import CircuitBreaker

logger = logging.getLogger("PatoQuant.Agent")

# Timeout para yfinance — evita que Railway se cuelgue
_YF_TIMEOUT = 5  # segundos

# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSO DE ACTIVOS
# El agente escanea estos pools para encontrar oportunidades
# ─────────────────────────────────────────────────────────────────────────────

# S&P 500 — los 50 más líquidos y representativos
SP500_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "XOM", "LLY", "JNJ", "PG", "MA", "HD", "CVX",
    "MRK", "ABBV", "KO", "PEP", "AVGO", "COST", "WMT", "BAC", "TMO",
    "CSCO", "ACN", "MCD", "NKE", "TXN", "NEE", "DHR", "AMGN", "CRM",
    "LIN", "PM", "ORCL", "IBM", "GS", "CAT", "RTX", "INTU", "QCOM",
    "SPGI", "HON", "UPS", "LOW", "AMD",
]

# Crypto — los más líquidos
CRYPTO_UNIVERSE = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "ADA-USD", "AVAX-USD", "DOGE-USD", "DOT-USD", "MATIC-USD",
    "LINK-USD", "UNI-USD", "ATOM-USD", "LTC-USD", "BCH-USD",
]

# ETFs globales y sectoriales
ETF_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "USO", "TLT",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
    "ARKK", "ARKG", "VNQ", "EEM", "EFA", "VEA", "KWEB",
]

# Activos de alto momentum / meme / tendencia
MOMENTUM_UNIVERSE = [
    "COIN", "MSTR", "PLTR", "RKLB", "SMCI", "ARM", "IONQ",
    "RIVN", "LCID", "HOOD", "SOFI", "UPST", "AFRM", "BILL",
    "DIS", "NFLX", "SPOT", "UBER", "LYFT", "ABNB", "DASH",
]

FULL_UNIVERSE = list(set(
    SP500_UNIVERSE + CRYPTO_UNIVERSE + ETF_UNIVERSE + MOMENTUM_UNIVERSE
))


# ─────────────────────────────────────────────────────────────────────────────
# AGENTE PROACTIVO
# ─────────────────────────────────────────────────────────────────────────────

class ProactiveAgent:
    """
    Agente que decide solo qué activos merecen estar en la watchlist.

    Criterios para AGREGAR:
      1. Volumen anómalo (RVOL > 2.5x en las últimas 2 sesiones)
      2. Momentum fuerte (precio > SMA20 > SMA50, ADX > 25)
      3. Ruptura técnica (precio cruza resistencia de 20 días)
      4. Trending según Groq (noticias/mercado global)
      5. Correlación alta con activos fuertes de la watchlist actual

    Criterios para ELIMINAR:
      - Score técnico < -30 por 3 scans consecutivos
      - Sin volumen relevante por 5 días (RVOL < 0.5)
      - Ya no está entre los 50 mejores del universo
    """

    def __init__(self,
                 watchlist_path: str = "data/watchlist.json",
                 max_watchlist_size: int = 50,
                 groq_api_key: str = ""):

        self.watchlist_path = watchlist_path
        self.max_size = max_watchlist_size
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")

        # Contadores de señales negativas por ticker
        self.weak_signal_count: Dict[str, int] = {}

        # Cache de scores del universo (para no re-descargar todo)
        self.universe_cache: Dict[str, Dict] = {}
        self.cache_timestamp: Optional[datetime] = None
        self.cache_ttl_minutes = 30

        # Circuit breakers
        self.yf_breaker = CircuitBreaker("yfinance", failure_threshold=10, cooldown_seconds=300)
        self.groq_breaker = CircuitBreaker("groq", failure_threshold=3, cooldown_seconds=600)

        logger.info("🤖 Agente Proactivo inicializado")
        logger.info(f"   Universo total: {len(FULL_UNIVERSE)} activos")
        logger.info(f"   Máximo watchlist: {self.max_size}")

    # ── Watchlist I/O ─────────────────────────────────────────────────────────

    def load_watchlist(self) -> Dict:
        """Carga la watchlist actual desde JSON."""
        if Path(self.watchlist_path).exists():
            with open(self.watchlist_path, "r") as f:
                return json.load(f)
        return {"stocks": [], "crypto": []}

    def save_watchlist(self, watchlist: Dict):
        """Guarda la watchlist en JSON."""
        Path(self.watchlist_path).parent.mkdir(exist_ok=True)
        with open(self.watchlist_path, "w") as f:
            json.dump(watchlist, f, indent=2)

    def get_all_tickers(self) -> List[str]:
        """Retorna lista plana de todos los tickers actuales."""
        wl = self.load_watchlist()
        return wl.get("stocks", []) + wl.get("crypto", [])

    def get_watchlist_size(self) -> int:
        return len(self.get_all_tickers())

    # ── Scoring rápido de un ticker ───────────────────────────────────────────

    def _quick_score(self, ticker: str) -> Optional[Dict]:
        """
        Descarga datos y calcula score rápido sin usar TechnicalAnalyzer completo.
        Optimizado para escanear el universo completo en poco tiempo.
        """
        if not self.yf_breaker.can_execute():
            return None

        try:
            df = yf.download(ticker, period="3mo", interval="1d",
                             progress=False, auto_adjust=True,
                             timeout=_YF_TIMEOUT)

            if df is None or len(df) < 20:
                return None

            self.yf_breaker.record_success()

            close = df["Close"].squeeze()
            volume = df["Volume"].squeeze()
            high = df["High"].squeeze()
            low = df["Low"].squeeze()

            # ── Indicadores básicos ───────────────────────────────────────────
            sma20 = close.rolling(20).mean()
            sma50 = close.rolling(50).mean() if len(close) >= 50 else sma20
            avg_vol = volume.rolling(20).mean()

            current_price = float(close.iloc[-1])
            current_vol   = float(volume.iloc[-1])
            prev_vol      = float(volume.iloc[-2]) if len(volume) > 1 else current_vol
            avg_vol_val   = float(avg_vol.iloc[-1]) + 1e-9

            rvol_today = current_vol / avg_vol_val
            rvol_prev  = prev_vol / avg_vol_val
            rvol_max   = max(rvol_today, rvol_prev)

            # Momentum: distancia a SMAs
            sma20_val = float(sma20.iloc[-1])
            sma50_val = float(sma50.iloc[-1]) if len(close) >= 50 else sma20_val

            dist_sma20 = (current_price - sma20_val) / sma20_val * 100
            dist_sma50 = (current_price - sma50_val) / sma50_val * 100

            # RSI rápido (14)
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / (loss + 1e-9)
            rsi   = float((100 - 100 / (1 + rs)).iloc[-1])

            # Ruptura: precio vs máximo de 20 días (excluyendo hoy)
            high_20 = high.iloc[-21:-1].max() if len(high) > 21 else high.max()
            breakout = current_price > float(high_20) * 1.005  # 0.5% sobre resistencia

            # Retorno 5 días
            ret_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) > 6 else 0

            # ── Score compuesto ───────────────────────────────────────────────
            score = 0

            # Volumen anómalo (máx 35 pts)
            if rvol_max >= 3.0:   score += 35
            elif rvol_max >= 2.5: score += 25
            elif rvol_max >= 2.0: score += 15
            elif rvol_max >= 1.5: score += 5

            # Momentum / tendencia (máx 30 pts)
            if current_price > sma20_val > sma50_val:
                score += 30
            elif current_price > sma20_val:
                score += 15

            # Ruptura técnica (máx 20 pts)
            if breakout:
                score += 20

            # RSI saludable — no sobrecomprado (máx 10 pts)
            if 40 < rsi < 65:  score += 10
            elif rsi < 35:     score += 5   # Sobreventa = rebote potencial

            # Retorno reciente positivo (máx 5 pts)
            if ret_5d > 3:    score += 5
            elif ret_5d > 1:  score += 2

            return {
                "ticker":    ticker,
                "price":     current_price,
                "score":     score,
                "rvol":      round(rvol_max, 2),
                "rsi":       round(rsi, 1),
                "dist_sma20": round(dist_sma20, 2),
                "breakout":  breakout,
                "ret_5d":    round(ret_5d, 2),
                "is_crypto": ticker.endswith("-USD"),
            }

        except Exception as e:
            self.yf_breaker.record_failure()
            logger.debug(f"Quick score falló para {ticker}: {e}")
            return None

    # ── Scan del universo completo ────────────────────────────────────────────

    def scan_universe(self) -> pd.DataFrame:
        """
        Escanea todo el universo de activos y retorna DataFrame con scores.
        Usa caché para no re-descargar si fue reciente.
        """
        # Verificar caché
        if (self.cache_timestamp and
            (datetime.now() - self.cache_timestamp).seconds < self.cache_ttl_minutes * 60
            and self.universe_cache):
            logger.info("📦 Usando caché del universo")
            return pd.DataFrame(list(self.universe_cache.values()))

        logger.info(f"🌍 Escaneando universo completo: {len(FULL_UNIVERSE)} activos...")
        results = []

        for i, ticker in enumerate(FULL_UNIVERSE):
            result = self._quick_score(ticker)
            if result:
                results.append(result)
                self.universe_cache[ticker] = result

            # Log de progreso cada 20 activos
            if (i + 1) % 20 == 0:
                logger.info(f"   Progreso: {i+1}/{len(FULL_UNIVERSE)}")

            # Pequeña pausa para no saturar yfinance
            if i % 10 == 0:
                time.sleep(0.5)

        self.cache_timestamp = datetime.now()
        df = pd.DataFrame(results).sort_values("score", ascending=False)
        logger.info(f"✅ Universo escaneado: {len(df)} activos válidos")
        return df

    # ── Consulta a Groq: activos trending ─────────────────────────────────────

    def get_trending_from_groq(self, current_watchlist: List[str]) -> List[str]:
        """
        Pregunta a Groq qué activos están tendiendo hoy en el mercado global.
        Retorna lista de tickers recomendados para considerar.
        """
        if not self.groq_api_key:
            logger.warning("⚠️ Sin GROQ_API_KEY — omitiendo trending")
            return []

        if not self.groq_breaker.can_execute():
            logger.info("🔌 Groq circuit breaker abierto — omitiendo trending")
            return []

        try:
            tz = pytz.timezone("America/New_York")
            today = datetime.now(tz).strftime("%Y-%m-%d %A")

            prompt = f"""Hoy es {today}. Eres un quant trader experto.

Basándote en tu conocimiento del mercado global, dime exactamente 10 tickers
(acciones, crypto o ETFs del mercado estadounidense) que probablemente tengan
momentum, volumen anómalo o catalizadores técnicos HOY o esta semana.

Watchlist actual (no repitas estos): {', '.join(current_watchlist[:15])}

Responde ÚNICAMENTE con un JSON array de tickers, sin explicación.
Formato exacto: ["TICKER1", "TICKER2", ...]
Solo tickers válidos de Yahoo Finance."""

            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "Eres un experto en mercados financieros. Responde solo con JSON válido."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 200,
            }

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15,
            )

            if not response.ok:
                self.groq_breaker.record_failure()
                logger.warning(f"⚠️ Groq error: {response.status_code}")
                return []

            self.groq_breaker.record_success()
            content = response.json()["choices"][0]["message"]["content"].strip()

            # Parsear JSON de la respuesta
            import re
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                tickers = json.loads(match.group())
                # Validar que sean strings y uppercase
                valid = [t.upper().strip() for t in tickers if isinstance(t, str) and len(t) <= 10]
                logger.info(f"🤖 Groq sugiere: {', '.join(valid)}")
                return valid

        except Exception as e:
            self.groq_breaker.record_failure()
            logger.warning(f"⚠️ Error consultando Groq para trending: {e}")

        return []

    # ── Detectar correlaciones ────────────────────────────────────────────────

    def get_correlated_opportunities(self, strong_tickers: List[str]) -> List[str]:
        """
        Encuentra activos del universo que están correlacionados con los
        activos fuertes de la watchlist actual.
        """
        if not strong_tickers:
            return []

        try:
            # Descargar precios de activos fuertes
            prices_strong = {}
            for ticker in strong_tickers[:5]:  # Máximo 5 para no tardar
                df = yf.download(ticker, period="1mo", interval="1d",
                                 progress=False, auto_adjust=True)
                if not df.empty:
                    prices_strong[ticker] = df["Close"].squeeze()

            if not prices_strong:
                return []

            # Candidatos a correlacionar (sector ETFs y momentum)
            candidates = ETF_UNIVERSE + MOMENTUM_UNIVERSE
            correlated = []

            for ticker in candidates:
                if ticker in strong_tickers:
                    continue
                try:
                    df = yf.download(ticker, period="1mo", interval="1d",
                                     progress=False, auto_adjust=True)
                    if df.empty:
                        continue

                    candidate_ret = df["Close"].squeeze().pct_change().dropna()

                    for strong_ticker, strong_prices in prices_strong.items():
                        strong_ret = strong_prices.pct_change().dropna()
                        common = candidate_ret.index.intersection(strong_ret.index)

                        if len(common) < 10:
                            continue

                        corr = candidate_ret.loc[common].corr(strong_ret.loc[common])

                        if corr > 0.75:  # Alta correlación positiva
                            correlated.append(ticker)
                            break

                except Exception:
                    continue

            logger.info(f"🔗 Correlacionados encontrados: {', '.join(correlated[:5])}")
            return correlated[:5]

        except Exception as e:
            logger.warning(f"⚠️ Error en correlaciones: {e}")
            return []

    # ── Decisión: qué agregar y qué quitar ───────────────────────────────────

    def decide_watchlist_changes(self,
                                  universe_df: pd.DataFrame,
                                  current_tickers: List[str],
                                  trending_tickers: List[str],
                                  correlated_tickers: List[str]) -> Tuple[List[str], List[str]]:
        """
        Decide qué tickers agregar y cuáles eliminar.

        Returns:
            (to_add, to_remove)
        """
        to_add    = []
        to_remove = []

        current_set = set(current_tickers)

        # ── CANDIDATOS A AGREGAR ──────────────────────────────────────────────

        # 1. Top 10 del universo por score que no estén en watchlist
        top_universe = universe_df[
            ~universe_df["ticker"].isin(current_set)
        ].head(10)

        for _, row in top_universe.iterrows():
            if row["score"] >= 50:  # Solo si el score es relevante
                to_add.append({
                    "ticker": row["ticker"],
                    "reason": f"Score universo: {row['score']} | RVOL: {row['rvol']}x",
                    "priority": row["score"],
                })

        # 2. Trending de Groq
        for ticker in trending_tickers:
            if ticker not in current_set and ticker not in [t["ticker"] for t in to_add]:
                # Verificar que existe en yfinance
                row = universe_df[universe_df["ticker"] == ticker]
                score = int(row["score"].iloc[0]) if not row.empty else 30
                to_add.append({
                    "ticker": ticker,
                    "reason": f"Trending según Groq | Score: {score}",
                    "priority": score + 10,  # Bonus por trending
                })

        # 3. Correlacionados
        for ticker in correlated_tickers:
            if ticker not in current_set and ticker not in [t["ticker"] for t in to_add]:
                row = universe_df[universe_df["ticker"] == ticker]
                score = int(row["score"].iloc[0]) if not row.empty else 25
                to_add.append({
                    "ticker": ticker,
                    "reason": f"Alta correlación con activos fuertes | Score: {score}",
                    "priority": score,
                })

        # Ordenar candidatos por prioridad
        to_add = sorted(to_add, key=lambda x: x["priority"], reverse=True)

        # ── CANDIDATOS A ELIMINAR ─────────────────────────────────────────────

        for ticker in current_tickers:
            row = universe_df[universe_df["ticker"] == ticker]

            if row.empty:
                # No encontrado en el universo
                self.weak_signal_count[ticker] = self.weak_signal_count.get(ticker, 0) + 1
                if self.weak_signal_count[ticker] >= 3:
                    to_remove.append({
                        "ticker": ticker,
                        "reason": "No encontrado en universo por 3 ciclos consecutivos"
                    })
                continue

            score = float(row["score"].iloc[0])
            rvol  = float(row["rvol"].iloc[0])

            # Score muy negativo persistente
            if score < 20:
                self.weak_signal_count[ticker] = self.weak_signal_count.get(ticker, 0) + 1
                if self.weak_signal_count.get(ticker, 0) >= 3:
                    to_remove.append({
                        "ticker": ticker,
                        "reason": f"Score bajo ({score}) por 3 ciclos | RVOL: {rvol}x"
                    })
            else:
                # Reset contador si se recuperó
                self.weak_signal_count[ticker] = 0

            # Sin volumen por mucho tiempo
            if rvol < 0.4:
                to_remove.append({
                    "ticker": ticker,
                    "reason": f"Sin volumen relevante (RVOL: {rvol}x)"
                })

        return to_add, to_remove

    # ── Aplicar cambios a la watchlist ────────────────────────────────────────

    def apply_changes(self,
                       to_add: List[Dict],
                       to_remove: List[Dict]) -> Dict:
        """
        Aplica los cambios a watchlist.json respetando el límite de 50 activos.
        Retorna resumen de cambios realizados.
        """
        watchlist = self.load_watchlist()
        current_stocks = watchlist.get("stocks", [])
        current_crypto = watchlist.get("crypto", [])
        current_all    = set(current_stocks + current_crypto)

        added   = []
        removed = []

        # 1. Eliminar primero para hacer espacio
        tickers_to_remove = {r["ticker"] for r in to_remove}
        current_stocks = [t for t in current_stocks if t not in tickers_to_remove]
        current_crypto = [t for t in current_crypto if t not in tickers_to_remove]
        removed = [r for r in to_remove if r["ticker"] in current_all]

        # Actualizar set
        current_all = set(current_stocks + current_crypto)

        # 2. Agregar nuevos respetando límite
        for candidate in to_add:
            ticker = candidate["ticker"]
            total  = len(current_stocks) + len(current_crypto)

            if total >= self.max_size:
                logger.info(f"   Límite de {self.max_size} activos alcanzado. Deteniendo.")
                break

            if ticker in current_all:
                continue

            # Clasificar como crypto o stock
            if ticker.endswith("-USD"):
                current_crypto.append(ticker)
            else:
                current_stocks.append(ticker)

            current_all.add(ticker)
            added.append(candidate)
            logger.info(f"   ✅ Agregado: {ticker} — {candidate['reason']}")

        # 3. Guardar
        watchlist["stocks"] = current_stocks
        watchlist["crypto"] = current_crypto
        self.save_watchlist(watchlist)

        summary = {
            "added":        [a["ticker"] for a in added],
            "removed":      [r["ticker"] for r in removed],
            "added_reasons":   {a["ticker"]: a["reason"] for a in added},
            "removed_reasons": {r["ticker"]: r["reason"] for r in removed},
            "total_after":  len(current_stocks) + len(current_crypto),
            "timestamp":    datetime.now().isoformat(),
        }

        return summary

    # ── Job principal ─────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """
        Ejecuta el ciclo completo del agente:
          1. Escanear universo
          2. Consultar Groq por trending
          3. Detectar correlaciones
          4. Decidir cambios
          5. Aplicar cambios
          6. Retornar resumen

        Este método lo llama scheduler.py automáticamente.
        """
        logger.info("\n🤖 AGENTE PROACTIVO — INICIANDO CICLO")
        logger.info("─" * 50)

        current_tickers = self.get_all_tickers()
        current_size    = len(current_tickers)
        logger.info(f"   Watchlist actual: {current_size} activos")

        # 1. Scan del universo
        universe_df = self.scan_universe()

        if universe_df.empty:
            logger.warning("⚠️ Universo vacío — abortando ciclo")
            return {}

        # 2. Activos fuertes actuales (para correlación)
        strong_in_watchlist = []
        for ticker in current_tickers:
            row = universe_df[universe_df["ticker"] == ticker]
            if not row.empty and float(row["score"].iloc[0]) >= 50:
                strong_in_watchlist.append(ticker)

        # 3. Trending de Groq
        trending = self.get_trending_from_groq(current_tickers)

        # 4. Correlaciones
        correlated = self.get_correlated_opportunities(strong_in_watchlist)

        # 5. Decidir cambios
        to_add, to_remove = self.decide_watchlist_changes(
            universe_df, current_tickers, trending, correlated
        )

        logger.info(f"   Candidatos a agregar: {len(to_add)}")
        logger.info(f"   Candidatos a eliminar: {len(to_remove)}")

        # 6. Aplicar
        summary = self.apply_changes(to_add, to_remove)

        # Log resumen
        if summary.get("added"):
            logger.info(f"✅ Agregados: {', '.join(summary['added'])}")
        if summary.get("removed"):
            logger.info(f"🗑️  Eliminados: {', '.join(summary['removed'])}")

        logger.info(f"📋 Watchlist final: {summary['total_after']} activos")
        logger.info("─" * 50)

        return summary
