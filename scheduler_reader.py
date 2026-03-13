"""
╔══════════════════════════════════════════════════════════════════╗
║         PATO QUANT — SCHEDULER READER                           ║
║                                                                  ║
║  Este módulo lo importas en app.py para leer los resultados     ║
║  que el scheduler guardó en SQLite.                             ║
║                                                                  ║
║  Tu Streamlit deja de calcular y solo muestra resultados.       ║
║                                                                  ║
║  Uso en app.py:                                                  ║
║    from scheduler_reader import SchedulerReader                  ║
║    reader = SchedulerReader()                                    ║
║    df = reader.get_latest_scan()                                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
import json


DB_PATH = "data/scheduler.db"


class SchedulerReader:
    """
    Interfaz de solo-lectura para que Streamlit consuma
    los datos generados por el scheduler autónomo.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._available = Path(db_path).exists()

    @property
    def is_available(self) -> bool:
        """True si la BD del scheduler existe y tiene datos."""
        return self._available and Path(self.db_path).exists()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Resultados del scanner ─────────────────────────────────────────────

    def get_latest_scan(self, max_age_minutes: int = 10) -> pd.DataFrame:
        """
        Retorna el último escaneo completo.
        Si los datos tienen más de max_age_minutes, avisa que están desactualizados.

        Returns:
            DataFrame con todos los activos del último scan, ordenados por |score|
        """
        if not self.is_available:
            return pd.DataFrame()

        try:
            with self._connect() as conn:
                rows = conn.execute("""
                    SELECT
                        ticker, timestamp, price, change_pct, score,
                        recommendation, rsi, adx, macd_hist, rvol,
                        atr, regime
                    FROM scan_results
                    WHERE timestamp >= datetime('now', ?)
                    ORDER BY ABS(score) DESC
                """, (f"-{max_age_minutes} minutes",)).fetchall()

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame([dict(r) for r in rows])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df

        except Exception as e:
            return pd.DataFrame()

    def get_ticker_latest(self, ticker: str) -> Optional[Dict]:
        """
        Retorna el análisis más reciente de un ticker específico.
        """
        if not self.is_available:
            return None

        try:
            with self._connect() as conn:
                row = conn.execute("""
                    SELECT *
                    FROM scan_results
                    WHERE ticker = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (ticker,)).fetchone()

            if row:
                result = dict(row)
                if result.get("raw_json"):
                    result.update(json.loads(result["raw_json"]))
                return result
            return None

        except Exception:
            return None

    def get_ticker_history(self, ticker: str, days: int = 7) -> pd.DataFrame:
        """
        Historial de scores e indicadores de un ticker.
        Útil para graficar cómo evoluciona el score en el tiempo.
        """
        if not self.is_available:
            return pd.DataFrame()

        try:
            with self._connect() as conn:
                rows = conn.execute("""
                    SELECT timestamp, price, score, rsi, adx, rvol
                    FROM scan_results
                    WHERE ticker = ?
                      AND timestamp >= datetime('now', ?)
                    ORDER BY timestamp ASC
                """, (ticker, f"-{days} days")).fetchall()

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame([dict(r) for r in rows])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df

        except Exception:
            return pd.DataFrame()

    def get_top_picks(self, n: int = 5, direction: str = "buy") -> pd.DataFrame:
        """
        Retorna los N mejores picks del último scan.

        Args:
            n: Número de picks
            direction: 'buy' (score > 0) | 'sell' (score < 0) | 'all'
        """
        df = self.get_latest_scan()
        if df.empty:
            return pd.DataFrame()

        if direction == "buy":
            df = df[df["score"] > 0].head(n)
        elif direction == "sell":
            df = df[df["score"] < 0].sort_values("score").head(n)
        else:
            df = df.head(n)

        return df

    # ── Status del scheduler ───────────────────────────────────────────────

    def get_scheduler_status(self) -> Dict:
        """
        Retorna información del estado del scheduler.
        Útil para mostrar en el sidebar de Streamlit.
        """
        if not self.is_available:
            return {
                "running": False,
                "message": "⚠️ Scheduler no iniciado. Corre: python scheduler.py",
                "last_scan": None,
                "total_scans": 0,
                "data_age_minutes": None,
            }

        try:
            with self._connect() as conn:
                # Último scan
                last_row = conn.execute("""
                    SELECT MAX(timestamp) as last_ts, COUNT(DISTINCT timestamp) as total
                    FROM scan_results
                """).fetchone()

                # Número de alertas hoy
                alerts_today = conn.execute("""
                    SELECT COUNT(*) as cnt FROM alerts_sent
                    WHERE timestamp >= date('now')
                """).fetchone()["cnt"]

            last_ts_str = last_row["last_ts"] if last_row else None
            total_scans = last_row["total"] if last_row else 0

            if last_ts_str:
                last_ts = pd.to_datetime(last_ts_str)
                age_minutes = (datetime.utcnow() - last_ts.replace(tzinfo=None)).total_seconds() / 60
                is_fresh = age_minutes < 10
                message = (
                    f"🟢 Activo — último scan hace {age_minutes:.0f} min"
                    if is_fresh
                    else f"🟡 Datos de hace {age_minutes:.0f} min — ¿scheduler corriendo?"
                )
            else:
                age_minutes = None
                is_fresh = False
                message = "🔴 Sin datos aún"

            return {
                "running": is_fresh,
                "message": message,
                "last_scan": last_ts_str,
                "total_scans": total_scans,
                "data_age_minutes": age_minutes,
                "alerts_today": alerts_today,
            }

        except Exception as e:
            return {
                "running": False,
                "message": f"❌ Error leyendo BD: {e}",
                "last_scan": None,
                "total_scans": 0,
                "data_age_minutes": None,
            }

    # ── ML Signals ────────────────────────────────────────────────────────

    def get_ml_signals(self, max_age_hours: int = 1) -> pd.DataFrame:
        """
        Retorna las últimas predicciones ML guardadas por el scheduler.
        """
        if not self.is_available:
            return pd.DataFrame()

        try:
            with self._connect() as conn:
                rows = conn.execute("""
                    SELECT ticker, timestamp, prob_up, prob_down,
                           recommendation, confidence, model_accuracy
                    FROM ml_signals
                    WHERE timestamp >= datetime('now', ?)
                    ORDER BY prob_up DESC
                """, (f"-{max_age_hours} hours",)).fetchall()

            if not rows:
                return pd.DataFrame()

            return pd.DataFrame([dict(r) for r in rows])

        except Exception:
            return pd.DataFrame()

    # ── Alertas recientes ─────────────────────────────────────────────────

    def get_recent_alerts(self, hours: int = 24) -> pd.DataFrame:
        """
        Retorna alertas de las últimas N horas.
        Útil para mostrar en Streamlit un log de señales.
        """
        if not self.is_available:
            return pd.DataFrame()

        try:
            with self._connect() as conn:
                rows = conn.execute("""
                    SELECT ticker, timestamp, alert_type, message, channel
                    FROM alerts_sent
                    WHERE timestamp >= datetime('now', ?)
                    ORDER BY timestamp DESC
                """, (f"-{hours} hours",)).fetchall()

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame([dict(r) for r in rows])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df

        except Exception:
            return pd.DataFrame()
