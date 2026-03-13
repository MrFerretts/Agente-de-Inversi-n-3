"""
╔══════════════════════════════════════════════════════════════════╗
║           PATO QUANT — SCHEDULER AUTÓNOMO                       ║
║                                                                  ║
║  Este archivo es el CEREBRO del sistema.                        ║
║  Corre 24/7 en background, sin necesidad de abrir Streamlit.    ║
║                                                                  ║
║  Qué hace:                                                       ║
║    → Cada 5 min: descarga precios y calcula indicadores         ║
║    → Corre modelos ML automáticamente                           ║
║    → Guarda resultados en SQLite                                ║
║    → Envía alertas por email/Telegram si detecta señales        ║
║    → Log completo de todo lo que hace                           ║
║                                                                  ║
║  Cómo correrlo:                                                  ║
║    python scheduler.py                                           ║
║                                                                  ║
║  Para correrlo en background (Linux/Mac):                        ║
║    nohup python scheduler.py > logs/scheduler.log 2>&1 &        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import logging
import sqlite3
import smtplib
import threading
import traceback
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import pytz
import schedule

# ── Ajuste de path para importar tus módulos existentes ──────────────────────
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from market_data import MarketDataFetcher
from technical_analysis import TechnicalAnalyzer
from core.state_manager import DataProcessor

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN CENTRAL
# Edita esta sección con tus valores reales.
# En producción usa variables de entorno (ver abajo).
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # ── Intervalos ────────────────────────────────────────────────────────────
    "scan_interval_minutes": 5,        # Cada cuánto escanear (mín 1)
    "ml_retrain_hours": 24,            # Cada cuánto re-entrenar modelos ML
    "market_hours_only": True,         # Solo correr en horario de mercado US

    # ── Umbrales de alerta ────────────────────────────────────────────────────
    "alert_score_threshold": 60,       # Score mínimo para alertar (|score| >= N)
    "alert_ml_probability": 0.72,      # Probabilidad ML mínima para alertar
    "alert_rvol_threshold": 2.0,       # Volumen relativo mínimo para alertar

    # ── Base de datos ─────────────────────────────────────────────────────────
    "db_path": "data/scheduler.db",    # SQLite local
    "results_retention_days": 30,      # Borrar resultados >30 días

    # ── Email (usa variables de entorno en producción) ─────────────────────────
    "email_enabled": bool(os.getenv("EMAIL_SENDER")),
    "email_sender": os.getenv("EMAIL_SENDER", ""),
    "email_password": os.getenv("EMAIL_PASSWORD", ""),
    "email_recipient": os.getenv("EMAIL_RECIPIENT", ""),
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,

    # ── Telegram (opcional) ───────────────────────────────────────────────────
    "telegram_enabled": bool(os.getenv("TELEGRAM_TOKEN")),
    "telegram_token": os.getenv("TELEGRAM_TOKEN", ""),
    "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),

    # ── Timezone ──────────────────────────────────────────────────────────────
    "timezone": "America/New_York",
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

Path("logs").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/scheduler.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("PatoQuant")


# ─────────────────────────────────────────────────────────────────────────────
# BASE DE DATOS — SQLite
# ─────────────────────────────────────────────────────────────────────────────

class Database:
    """
    Gestiona toda la persistencia del scheduler.
    Tablas:
      - scan_results   : resultado de cada escaneo por ticker
      - alerts_sent    : historial de alertas enviadas
      - watchlist      : tickers a monitorear (sincronizado con watchlist.json)
      - ml_signals     : predicciones ML por ticker
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Crea las tablas si no existen."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS scan_results (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker      TEXT NOT NULL,
                    timestamp   TEXT NOT NULL,
                    price       REAL,
                    change_pct  REAL,
                    score       INTEGER,
                    recommendation TEXT,
                    rsi         REAL,
                    adx         REAL,
                    macd_hist   REAL,
                    rvol        REAL,
                    atr         REAL,
                    regime      TEXT,
                    raw_json    TEXT
                );

                CREATE TABLE IF NOT EXISTS alerts_sent (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker      TEXT NOT NULL,
                    timestamp   TEXT NOT NULL,
                    alert_type  TEXT,
                    message     TEXT,
                    channel     TEXT
                );

                CREATE TABLE IF NOT EXISTS watchlist (
                    ticker      TEXT PRIMARY KEY,
                    category    TEXT DEFAULT 'stock',
                    added_at    TEXT DEFAULT (datetime('now')),
                    active      INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS ml_signals (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker          TEXT NOT NULL,
                    timestamp       TEXT NOT NULL,
                    prob_up         REAL,
                    prob_down       REAL,
                    recommendation  TEXT,
                    confidence      REAL,
                    model_accuracy  REAL
                );

                CREATE INDEX IF NOT EXISTS idx_scan_ticker_ts
                    ON scan_results(ticker, timestamp);
                CREATE INDEX IF NOT EXISTS idx_alerts_ticker
                    ON alerts_sent(ticker, timestamp);
            """)
        logger.info(f"✅ Base de datos lista: {self.db_path}")

    # ── Scan results ──────────────────────────────────────────────────────────

    def save_scan_result(self, ticker: str, result: Dict):
        """Guarda el resultado de un escaneo."""
        ts = datetime.now(pytz.timezone(CONFIG["timezone"])).isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO scan_results
                    (ticker, timestamp, price, change_pct, score, recommendation,
                     rsi, adx, macd_hist, rvol, atr, regime, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker, ts,
                result.get("price"),
                result.get("change_pct"),
                result.get("score"),
                result.get("recommendation"),
                result.get("rsi"),
                result.get("adx"),
                result.get("macd_hist"),
                result.get("rvol"),
                result.get("atr"),
                result.get("regime"),
                json.dumps(result),
            ))

    def get_latest_results(self, limit: int = 50) -> pd.DataFrame:
        """Retorna los últimos N resultados ordenados por score."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT ticker, timestamp, price, change_pct, score,
                       recommendation, rsi, adx, rvol, regime
                FROM scan_results
                WHERE timestamp >= datetime('now', '-1 hour')
                ORDER BY ABS(score) DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    def get_ticker_history(self, ticker: str, days: int = 7) -> pd.DataFrame:
        """Historial de un ticker específico."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT timestamp, price, score, rsi, adx, rvol
                FROM scan_results
                WHERE ticker = ?
                  AND timestamp >= datetime('now', ?)
                ORDER BY timestamp ASC
            """, (ticker, f"-{days} days")).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    def cleanup_old_data(self, days: int):
        """Borra registros antiguos para mantener la BD liviana."""
        with self._connect() as conn:
            deleted = conn.execute("""
                DELETE FROM scan_results
                WHERE timestamp < datetime('now', ?)
            """, (f"-{days} days",)).rowcount
        if deleted:
            logger.info(f"🗑️  Limpieza: {deleted} registros eliminados (>{days} días)")

    # ── Watchlist ─────────────────────────────────────────────────────────────

    def get_watchlist(self) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ticker, category FROM watchlist WHERE active=1"
            ).fetchall()
        return [dict(r) for r in rows]

    def sync_watchlist_from_json(self, json_path: str = "data/watchlist.json"):
        """
        Sincroniza la BD con tu watchlist.json existente.
        Llámalo al arrancar el scheduler.
        """
        if not os.path.exists(json_path):
            logger.warning(f"⚠️  {json_path} no encontrado. Usa la watchlist por defecto.")
            return

        with open(json_path, "r") as f:
            data = json.load(f)

        stocks = data.get("stocks", [])
        crypto = data.get("crypto", [])

        with self._connect() as conn:
            for ticker in stocks:
                conn.execute("""
                    INSERT OR IGNORE INTO watchlist (ticker, category) VALUES (?, 'stock')
                """, (ticker,))
            for ticker in crypto:
                conn.execute("""
                    INSERT OR IGNORE INTO watchlist (ticker, category) VALUES (?, 'crypto')
                """, (ticker,))

        total = len(stocks) + len(crypto)
        logger.info(f"📋 Watchlist sincronizada: {total} activos")

    # ── Alertas ───────────────────────────────────────────────────────────────

    def save_alert(self, ticker: str, alert_type: str, message: str, channel: str):
        ts = datetime.now(pytz.timezone(CONFIG["timezone"])).isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO alerts_sent (ticker, timestamp, alert_type, message, channel)
                VALUES (?, ?, ?, ?, ?)
            """, (ticker, ts, alert_type, message, channel))

    def was_alert_sent_recently(self, ticker: str, alert_type: str,
                                 cooldown_minutes: int = 30) -> bool:
        """Evita spam: retorna True si ya se envió esta alerta recientemente."""
        since = (datetime.now() - timedelta(minutes=cooldown_minutes)).isoformat()
        with self._connect() as conn:
            count = conn.execute("""
                SELECT COUNT(*) FROM alerts_sent
                WHERE ticker=? AND alert_type=? AND timestamp > ?
            """, (ticker, alert_type, since)).fetchone()[0]
        return count > 0

    # ── ML Signals ────────────────────────────────────────────────────────────

    def save_ml_signal(self, ticker: str, prediction: Dict):
        ts = datetime.now(pytz.timezone(CONFIG["timezone"])).isoformat()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO ml_signals
                    (ticker, timestamp, prob_up, prob_down, recommendation,
                     confidence, model_accuracy)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker, ts,
                prediction.get("probability_up"),
                prediction.get("probability_down"),
                prediction.get("recommendation"),
                prediction.get("confidence"),
                prediction.get("model_accuracy"),
            ))


# ─────────────────────────────────────────────────────────────────────────────
# NOTIFICACIONES
# ─────────────────────────────────────────────────────────────────────────────

class Notifier:
    """Envía alertas por email y/o Telegram."""

    def __init__(self, db: Database):
        self.db = db

    def send_email(self, subject: str, body_html: str, ticker: str, alert_type: str):
        if not CONFIG["email_enabled"]:
            return

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = CONFIG["email_sender"]
            msg["To"] = CONFIG["email_recipient"]
            msg.attach(MIMEText(body_html, "html"))

            with smtplib.SMTP(CONFIG["smtp_server"], CONFIG["smtp_port"]) as server:
                server.starttls()
                server.login(CONFIG["email_sender"], CONFIG["email_password"])
                server.send_message(msg)

            self.db.save_alert(ticker, alert_type, subject, "email")
            logger.info(f"📧 Email enviado: {subject}")

        except Exception as e:
            logger.error(f"❌ Error enviando email: {e}")

    def send_telegram(self, message: str, ticker: str, alert_type: str):
        if not CONFIG["telegram_enabled"]:
            return

        try:
            import requests
            url = f"https://api.telegram.org/bot{CONFIG['telegram_token']}/sendMessage"
            payload = {
                "chat_id": CONFIG["telegram_chat_id"],
                "text": message,
                "parse_mode": "Markdown",
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.ok:
                self.db.save_alert(ticker, alert_type, message, "telegram")
                logger.info(f"📱 Telegram enviado: {ticker} — {alert_type}")
            else:
                logger.warning(f"⚠️  Telegram falló: {response.text}")
        except Exception as e:
            logger.error(f"❌ Error Telegram: {e}")

    def notify(self, ticker: str, result: Dict, alert_type: str):
        """
        Punto de entrada único para notificaciones.
        Verifica cooldown antes de enviar.
        """
        if self.db.was_alert_sent_recently(ticker, alert_type, cooldown_minutes=30):
            logger.debug(f"🔕 Alerta omitida (cooldown): {ticker} — {alert_type}")
            return

        tz = pytz.timezone(CONFIG["timezone"])
        now_str = datetime.now(tz).strftime("%H:%M:%S %Z")
        score = result.get("score", 0)
        rec = result.get("recommendation", "—")
        price = result.get("price", 0)
        rsi = result.get("rsi", 0)
        rvol = result.get("rvol", 0)

        # ── Telegram ──────────────────────────────────────────────────────────
        emoji = "🟢" if score > 0 else "🔴"
        tg_msg = (
            f"{emoji} *{ticker}* — {alert_type.upper()}\n"
            f"⏰ {now_str}\n"
            f"💲 Precio: ${price:.2f}\n"
            f"📊 Score: {score}/100 — {rec}\n"
            f"📈 RSI: {rsi:.1f} | RVOL: {rvol:.2f}x"
        )
        self.send_telegram(tg_msg, ticker, alert_type)

        # ── Email ─────────────────────────────────────────────────────────────
        subject = f"🦆 Pato Quant | {ticker} — {rec} (Score: {score})"
        html = f"""
        <html><body style="font-family:Arial,sans-serif;background:#111;color:#eee;padding:20px">
        <h2 style="color:#00e676">🦆 Pato Quant — Alerta Automática</h2>
        <table style="border-collapse:collapse;width:100%">
          <tr><td style="padding:8px;color:#aaa">Ticker</td>
              <td style="padding:8px;font-weight:bold;font-size:1.4em">{ticker}</td></tr>
          <tr><td style="padding:8px;color:#aaa">Precio</td>
              <td style="padding:8px">${price:.2f}</td></tr>
          <tr><td style="padding:8px;color:#aaa">Score</td>
              <td style="padding:8px;color:{'#00e676' if score>0 else '#ff5252'};font-weight:bold">
              {score}/100</td></tr>
          <tr><td style="padding:8px;color:#aaa">Señal</td>
              <td style="padding:8px;font-weight:bold">{rec}</td></tr>
          <tr><td style="padding:8px;color:#aaa">RSI</td>
              <td style="padding:8px">{rsi:.1f}</td></tr>
          <tr><td style="padding:8px;color:#aaa">RVOL</td>
              <td style="padding:8px">{rvol:.2f}x</td></tr>
          <tr><td style="padding:8px;color:#aaa">Hora</td>
              <td style="padding:8px">{now_str}</td></tr>
        </table>
        <p style="color:#555;font-size:0.8em;margin-top:20px">
          Enviado automáticamente por Pato Quant Scheduler.<br>
          No constituye asesoría financiera.
        </p>
        </body></html>
        """
        self.send_email(subject, html, ticker, alert_type)


# ─────────────────────────────────────────────────────────────────────────────
# MARKET HOURS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def is_market_open() -> bool:
    """
    Retorna True si el mercado de EE.UU. está abierto ahora mismo.
    Horario regular: Lun-Vie 09:30–16:00 ET.
    No contempla días festivos (mejora futura).
    """
    tz = pytz.timezone("America/New_York")
    now = datetime.now(tz)

    if now.weekday() >= 5:  # sábado=5, domingo=6
        return False

    market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)

    return market_open <= now <= market_close


def is_pre_market() -> bool:
    """Pre-market: 04:00–09:30 ET."""
    tz = pytz.timezone("America/New_York")
    now = datetime.now(tz)
    if now.weekday() >= 5:
        return False
    pre_open  = now.replace(hour=4,  minute=0,  second=0, microsecond=0)
    pre_close = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    return pre_open <= now < pre_close


# ─────────────────────────────────────────────────────────────────────────────
# SCANNER JOB — el trabajo principal
# ─────────────────────────────────────────────────────────────────────────────

class QuantScheduler:
    """
    Orquesta todos los jobs del scheduler.
    """

    def __init__(self):
        # Cargar config de la app (igual que app.py)
        self._load_app_config()

        self.db       = Database(CONFIG["db_path"])
        self.notifier = Notifier(self.db)
        self.fetcher  = MarketDataFetcher(self.api_config)
        self.analyzer = TechnicalAnalyzer(self.technical_config)
        self.ml_models: Dict = {}  # {ticker: model}

        # Sincronizar watchlist desde JSON al arrancar
        self.db.sync_watchlist_from_json()

        # Stats de sesión
        self.stats = {
            "scans_completed": 0,
            "alerts_sent": 0,
            "errors": 0,
            "started_at": datetime.now().isoformat(),
        }

        logger.info("🦆 Pato Quant Scheduler inicializado")
        logger.info(f"   Intervalo: cada {CONFIG['scan_interval_minutes']} min")
        logger.info(f"   Solo horario de mercado: {CONFIG['market_hours_only']}")

    def _load_app_config(self):
        """Carga la misma config que usa app.py."""
        try:
            from config import API_CONFIG, PORTFOLIO_CONFIG, TECHNICAL_INDICATORS
            self.api_config       = API_CONFIG
            self.portfolio_config = PORTFOLIO_CONFIG
            self.technical_config = TECHNICAL_INDICATORS
        except ImportError:
            logger.warning("⚠️  config.py no encontrado. Usando defaults.")
            self.api_config       = {}
            self.portfolio_config = {"stocks": ["AAPL", "MSFT"], "crypto": ["BTC-USD"]}
            self.technical_config = {}

    # ── Carga de modelos ML ───────────────────────────────────────────────────

    def load_ml_models(self):
        """
        Intenta cargar modelos ML pre-entrenados desde disco.
        Si no existen, los entrena en el primer ciclo.
        """
        try:
            import pickle
            models_dir = Path("data/models")
            if not models_dir.exists():
                logger.info("📂 No hay modelos guardados. Se entrenarán en el primer ciclo.")
                return

            for model_file in models_dir.glob("*.pkl"):
                ticker = model_file.stem
                with open(model_file, "rb") as f:
                    self.ml_models[ticker] = pickle.load(f)
                logger.info(f"🤖 Modelo cargado: {ticker}")

        except Exception as e:
            logger.error(f"❌ Error cargando modelos: {e}")

    def train_and_save_model(self, ticker: str, data: pd.DataFrame):
        """Entrena y persiste modelo ML para un ticker."""
        try:
            import pickle
            from ml_model import AdvancedTradingMLModel

            Path("data/models").mkdir(parents=True, exist_ok=True)

            logger.info(f"🎓 Entrenando modelo ML para {ticker}...")
            model = AdvancedTradingMLModel(prediction_days=5, threshold=2.0)
            metrics = model.train(data, test_size=0.2)

            if model.is_trained:
                self.ml_models[ticker] = model
                model_path = f"data/models/{ticker}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                logger.info(
                    f"✅ Modelo {ticker} guardado — "
                    f"Accuracy: {metrics.get('accuracy', 0)*100:.1f}%"
                )
        except Exception as e:
            logger.warning(f"⚠️  No se pudo entrenar modelo para {ticker}: {e}")

    # ── Job principal: escanear todos los activos ─────────────────────────────

    def run_scan(self):
        """
        Job principal. Se ejecuta cada N minutos.
        Descarga datos → calcula indicadores → evalúa alertas → guarda en BD.
        """
        # Verificar horario de mercado
        if CONFIG["market_hours_only"] and not is_market_open() and not is_pre_market():
            logger.debug("🔒 Mercado cerrado. Scan omitido.")
            return

        watchlist = self.db.get_watchlist()
        if not watchlist:
            logger.warning("⚠️  Watchlist vacía. Agrega tickers en data/watchlist.json")
            return

        tz_str = CONFIG["timezone"]
        now_str = datetime.now(pytz.timezone(tz_str)).strftime("%H:%M:%S %Z")
        logger.info(f"🔍 Iniciando scan — {now_str} — {len(watchlist)} activos")

        results = []
        errors  = 0

        for item in watchlist:
            ticker   = item["ticker"]
            category = item["category"]

            try:
                result = self._analyze_ticker(ticker, category)
                if result:
                    results.append(result)
                    self.db.save_scan_result(ticker, result)
                    self._evaluate_alerts(ticker, result)

            except Exception as e:
                errors += 1
                self.stats["errors"] += 1
                logger.error(f"❌ Error en {ticker}: {e}")
                logger.debug(traceback.format_exc())

        # Resumen del scan
        if results:
            df = pd.DataFrame(results).sort_values("score", ascending=False)
            top3 = df.head(3)[["ticker", "score", "recommendation", "price"]].to_string(index=False)
            logger.info(f"\n{'─'*50}\n📊 TOP 3 PICKS:\n{top3}\n{'─'*50}")

        self.stats["scans_completed"] += 1
        logger.info(
            f"✅ Scan #{self.stats['scans_completed']} completado — "
            f"{len(results)} analizados, {errors} errores"
        )

        # Limpieza periódica (cada 100 scans)
        if self.stats["scans_completed"] % 100 == 0:
            self.db.cleanup_old_data(CONFIG["results_retention_days"])

    def _analyze_ticker(self, ticker: str, category: str) -> Optional[Dict]:
        """
        Analiza un solo ticker. Retorna dict con todos los indicadores.
        Reutiliza tus clases existentes sin modificarlas.
        """
        # 1. Descargar datos
        period = "3mo" if category == "stock" else "1mo"
        raw_data = self.fetcher.get_stock_data(ticker, period=period)

        if raw_data is None or raw_data.empty or len(raw_data) < 30:
            logger.warning(f"⚠️  Datos insuficientes para {ticker}")
            return None

        # 2. Calcular indicadores (usando tu DataProcessor existente)
        data_processed = DataProcessor.prepare_full_analysis(raw_data, self.analyzer)

        # 3. Análisis técnico (usando tu TechnicalAnalyzer existente)
        analysis = self.analyzer.analyze_asset(data_processed, ticker)

        if not analysis:
            return None

        # 4. Señales actuales
        signals = DataProcessor.get_latest_signals(data_processed)

        # 5. Predicción ML (si hay modelo entrenado)
        ml_result = self._get_ml_prediction(ticker, data_processed)

        # 6. Construir resultado limpio y serializable
        result = {
            "ticker":         ticker,
            "category":       category,
            "price":          round(float(signals.get("price", 0)), 4),
            "change_pct":     round(float(signals.get("price_change_pct", 0)), 4),
            "score":          int(analysis["signals"].get("score", 0)),
            "recommendation": analysis["signals"].get("recommendation", "MANTENER"),
            "rsi":            round(float(signals.get("rsi", 0)), 2),
            "adx":            round(float(signals.get("adx", 0)), 2),
            "macd_hist":      round(float(signals.get("macd_hist", 0)), 6),
            "rvol":           round(float(signals.get("rvol", 0)), 2),
            "atr":            round(float(signals.get("atr", 0)), 4),
            "regime":         signals.get("trend", "UNKNOWN"),
            "trend_strength": signals.get("trend_strength", "WEAK"),
            "ml_prob_up":     ml_result.get("probability_up") if ml_result else None,
            "ml_rec":         ml_result.get("recommendation") if ml_result else None,
            "scanned_at":     datetime.now(pytz.timezone(CONFIG["timezone"])).isoformat(),
        }

        logger.info(
            f"   {ticker:8s} | ${result['price']:>8.2f} | "
            f"Score: {result['score']:>4d} | {result['recommendation']:<15s} | "
            f"RSI: {result['rsi']:>5.1f} | RVOL: {result['rvol']:>4.2f}x"
        )

        return result

    def _get_ml_prediction(self, ticker: str, data: pd.DataFrame) -> Optional[Dict]:
        """Retorna predicción ML si hay modelo disponible."""
        if ticker not in self.ml_models:
            return None
        try:
            model = self.ml_models[ticker]
            return model.predict(data)
        except Exception as e:
            logger.debug(f"ML predict falló para {ticker}: {e}")
            return None

    # ── Evaluación de alertas ─────────────────────────────────────────────────

    def _evaluate_alerts(self, ticker: str, result: Dict):
        """
        Evalúa si el resultado merece una alerta.
        Solo envía si supera umbrales Y no está en cooldown.
        """
        score  = result.get("score", 0)
        rvol   = result.get("rvol", 0)
        ml_up  = result.get("ml_prob_up")

        # ── Alerta por Score técnico alto ─────────────────────────────────────
        threshold = CONFIG["alert_score_threshold"]
        if score >= threshold:
            self.notifier.notify(ticker, result, "COMPRA_TECNICA")
            self.stats["alerts_sent"] += 1

        elif score <= -threshold:
            self.notifier.notify(ticker, result, "VENTA_TECNICA")
            self.stats["alerts_sent"] += 1

        # ── Alerta por ML ─────────────────────────────────────────────────────
        if ml_up is not None:
            ml_threshold = CONFIG["alert_ml_probability"]
            if ml_up >= ml_threshold:
                self.notifier.notify(ticker, result, "ML_COMPRA")
            elif ml_up <= (1 - ml_threshold):
                self.notifier.notify(ticker, result, "ML_VENTA")

        # ── Alerta por volumen anómalo ────────────────────────────────────────
        if rvol >= CONFIG["alert_rvol_threshold"] and abs(score) >= 30:
            self.notifier.notify(ticker, result, "VOLUMEN_ANOMALO")

    # ── Job de re-entrenamiento ML ────────────────────────────────────────────

    def run_ml_retrain(self):
        """
        Re-entrena todos los modelos ML con datos frescos.
        Corre cada N horas (configurable).
        """
        logger.info("🎓 Iniciando re-entrenamiento de modelos ML...")
        watchlist = self.db.get_watchlist()

        for item in watchlist:
            ticker = item["ticker"]
            try:
                data = self.fetcher.get_stock_data(ticker, period="1y")
                if data is not None and len(data) >= 200:
                    data_processed = DataProcessor.prepare_full_analysis(data, self.analyzer)
                    self.train_and_save_model(ticker, data_processed)
                else:
                    logger.warning(f"⚠️  {ticker}: datos insuficientes para entrenar")
            except Exception as e:
                logger.error(f"❌ ML retrain falló para {ticker}: {e}")

        logger.info("✅ Re-entrenamiento completado")

    # ── Reporte de status ─────────────────────────────────────────────────────

    def print_status(self):
        """Imprime estado del scheduler en el log."""
        uptime = datetime.now() - datetime.fromisoformat(self.stats["started_at"])
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes = remainder // 60

        logger.info(
            f"\n{'═'*50}\n"
            f"🦆 PATO QUANT SCHEDULER STATUS\n"
            f"   Uptime:       {hours}h {minutes}m\n"
            f"   Scans:        {self.stats['scans_completed']}\n"
            f"   Alertas:      {self.stats['alerts_sent']}\n"
            f"   Errores:      {self.stats['errors']}\n"
            f"   Modelos ML:   {len(self.ml_models)}\n"
            f"   Mercado:      {'🟢 ABIERTO' if is_market_open() else '🔴 CERRADO'}\n"
            f"{'═'*50}"
        )

    # ── Setup de todos los jobs ───────────────────────────────────────────────

    def setup_schedule(self):
        """Registra todos los jobs en el scheduler."""
        interval = CONFIG["scan_interval_minutes"]

        # Scan principal
        schedule.every(interval).minutes.do(self.run_scan)

        # Re-entrenamiento ML
        retrain_hours = CONFIG["ml_retrain_hours"]
        schedule.every(retrain_hours).hours.do(self.run_ml_retrain)

        # Status cada hora
        schedule.every(1).hours.do(self.print_status)

        # Limpieza diaria a las 2am
        schedule.every().day.at("02:00").do(
            lambda: self.db.cleanup_old_data(CONFIG["results_retention_days"])
        )

        logger.info(f"📅 Jobs programados:")
        logger.info(f"   → Scan: cada {interval} minutos")
        logger.info(f"   → ML retrain: cada {retrain_hours} horas")
        logger.info(f"   → Limpieza BD: diaria a las 02:00")

    # ── Loop principal ────────────────────────────────────────────────────────

    def run(self):
        """
        Arranca el scheduler. Este método nunca termina (loop infinito).
        Maneja Ctrl+C limpiamente.
        """
        logger.info("\n" + "═"*50)
        logger.info("🚀 PATO QUANT SCHEDULER ARRANCANDO")
        logger.info("═"*50)

        # Cargar modelos ML existentes
        self.load_ml_models()

        # Configurar jobs
        self.setup_schedule()

        # Primer scan inmediato al arrancar
        logger.info("⚡ Ejecutando primer scan...")
        self.run_scan()

        # Loop principal
        logger.info(f"⏳ Scheduler corriendo. Próximo scan en {CONFIG['scan_interval_minutes']} min.")
        logger.info("   Presiona Ctrl+C para detener.\n")

        try:
            while True:
                schedule.run_pending()
                time.sleep(30)  # Verificar jobs cada 30 segundos

        except KeyboardInterrupt:
            logger.info("\n🛑 Scheduler detenido por el usuario.")
            self.print_status()
        except Exception as e:
            logger.critical(f"💥 Error crítico en el loop principal: {e}")
            logger.critical(traceback.format_exc())
            raise


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scheduler = QuantScheduler()
    scheduler.run()
