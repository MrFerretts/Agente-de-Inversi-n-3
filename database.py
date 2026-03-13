"""
╔══════════════════════════════════════════════════════════════════╗
║         PATO QUANT — DATABASE MANAGER (SUPABASE)                ║
║                                                                  ║
║  Reemplaza SQLite por PostgreSQL en Supabase.                   ║
║  Tanto scheduler.py (Railway) como dashboard.py                 ║
║  (Streamlit Cloud) usan este mismo módulo.                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import logging
from datetime import datetime
from typing import Optional, List, Dict
import pandas as pd

logger = logging.getLogger("PatoQuant.DB")

# ─────────────────────────────────────────────────────────────────────────────
# CONEXIÓN
# ─────────────────────────────────────────────────────────────────────────────

def get_connection():
    """
    Retorna conexión a Supabase PostgreSQL.
    Lee SUPABASE_DB_URL desde variables de entorno.
    """
    try:
        import psycopg2
        url = os.getenv("SUPABASE_DB_URL", "")
        if not url:
            raise ValueError("SUPABASE_DB_URL no configurada")
        conn = psycopg2.connect(url, connect_timeout=10)
        return conn
    except ImportError:
        raise ImportError("Instala psycopg2-binary: pip install psycopg2-binary")
    except Exception as e:
        logger.error(f"❌ Error conectando a Supabase: {e}")
        raise


def query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Ejecuta SELECT y retorna DataFrame. Retorna vacío si falla."""
    try:
        conn = get_connection()
        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        return df
    except Exception as e:
        logger.warning(f"⚠️ Query falló: {e}")
        return pd.DataFrame()


def execute(sql: str, params: tuple = ()):
    """Ejecuta INSERT/UPDATE/DELETE."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"❌ Execute falló: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# INICIALIZAR TABLAS
# ─────────────────────────────────────────────────────────────────────────────

def init_tables():
    """
    Crea las tablas en Supabase si no existen.
    Correr una sola vez al arrancar el scheduler.
    """
    tables = [
        # Resultados de cada scan
        """
        CREATE TABLE IF NOT EXISTS scan_results (
            id          SERIAL PRIMARY KEY,
            ticker      VARCHAR(20)  NOT NULL,
            timestamp   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            price       NUMERIC(12,4),
            change_pct  NUMERIC(8,4),
            score       NUMERIC(8,2),
            recommendation VARCHAR(30),
            rsi         NUMERIC(6,2),
            adx         NUMERIC(6,2),
            rvol        NUMERIC(6,3),
            macd        NUMERIC(10,6),
            bb_position NUMERIC(6,4),
            volume      BIGINT,
            raw_data    JSONB
        )
        """,

        # Alertas enviadas
        """
        CREATE TABLE IF NOT EXISTS alerts_sent (
            id          SERIAL PRIMARY KEY,
            ticker      VARCHAR(20)  NOT NULL,
            timestamp   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            alert_type  VARCHAR(50),
            message     TEXT,
            channel     VARCHAR(20)
        )
        """,

        # Watchlist activa
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            id          SERIAL PRIMARY KEY,
            ticker      VARCHAR(20)  NOT NULL UNIQUE,
            asset_type  VARCHAR(10)  NOT NULL DEFAULT 'stock',
            added_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            added_by    VARCHAR(20)  NOT NULL DEFAULT 'user',
            is_active   BOOLEAN      NOT NULL DEFAULT TRUE
        )
        """,

        # Señales ML
        """
        CREATE TABLE IF NOT EXISTS ml_signals (
            id          SERIAL PRIMARY KEY,
            ticker      VARCHAR(20)  NOT NULL,
            timestamp   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            signal      VARCHAR(10),
            probability NUMERIC(6,4),
            model_type  VARCHAR(30),
            features    JSONB
        )
        """,

        # Historial de cambios de watchlist por el agente
        """
        CREATE TABLE IF NOT EXISTS agent_log (
            id          SERIAL PRIMARY KEY,
            timestamp   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            action      VARCHAR(10)  NOT NULL,
            ticker      VARCHAR(20)  NOT NULL,
            reason      TEXT,
            score       NUMERIC(8,2)
        )
        """,

        # Índice para queries rápidas
        """
        CREATE INDEX IF NOT EXISTS idx_scan_ticker_ts
        ON scan_results (ticker, timestamp DESC)
        """,

        """
        CREATE INDEX IF NOT EXISTS idx_alerts_ts
        ON alerts_sent (timestamp DESC)
        """,
    ]

    for sql in tables:
        try:
            execute(sql)
        except Exception as e:
            logger.warning(f"⚠️ Tabla ya existe o error: {e}")

    logger.info("✅ Tablas Supabase listas")


# ─────────────────────────────────────────────────────────────────────────────
# OPERACIONES DE WATCHLIST
# ─────────────────────────────────────────────────────────────────────────────

def get_watchlist() -> List[Dict]:
    """Retorna lista de tickers activos."""
    df = query_df("""
        SELECT ticker, asset_type, added_at, added_by
        FROM watchlist
        WHERE is_active = TRUE
        ORDER BY added_at ASC
    """)
    return df.to_dict("records") if not df.empty else []


def get_watchlist_tickers() -> List[str]:
    """Retorna solo los tickers activos."""
    df = query_df("SELECT ticker FROM watchlist WHERE is_active = TRUE")
    return df["ticker"].tolist() if not df.empty else []


def add_ticker(ticker: str, asset_type: str = "stock", added_by: str = "agent"):
    """Agrega ticker a watchlist. Ignora si ya existe."""
    execute("""
        INSERT INTO watchlist (ticker, asset_type, added_by)
        VALUES (%s, %s, %s)
        ON CONFLICT (ticker) DO UPDATE SET is_active = TRUE
    """, (ticker.upper(), asset_type, added_by))


def remove_ticker(ticker: str):
    """Desactiva ticker (soft delete)."""
    execute("""
        UPDATE watchlist SET is_active = FALSE
        WHERE ticker = %s
    """, (ticker.upper(),))


def sync_watchlist_from_json(json_path: str = "data/watchlist.json"):
    """
    Sincroniza watchlist.json → Supabase.
    Corre al arrancar el scheduler para importar la lista actual.
    """
    import json
    from pathlib import Path

    if not Path(json_path).exists():
        logger.warning(f"⚠️ {json_path} no encontrado")
        return

    with open(json_path) as f:
        data = json.load(f)

    stocks = data.get("stocks", [])
    crypto = data.get("crypto", [])

    for ticker in stocks:
        add_ticker(ticker, "stock", "json_sync")
    for ticker in crypto:
        add_ticker(ticker, "crypto", "json_sync")

    logger.info(f"✅ Watchlist sincronizada: {len(stocks)} stocks + {len(crypto)} crypto")


# ─────────────────────────────────────────────────────────────────────────────
# OPERACIONES DE SCAN
# ─────────────────────────────────────────────────────────────────────────────

def save_scan_result(ticker: str, data: Dict):
    """Guarda resultado de un scan individual."""
    import json
    execute("""
        INSERT INTO scan_results
            (ticker, price, change_pct, score, recommendation,
             rsi, adx, rvol, macd, bb_position, volume, raw_data)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        ticker,
        data.get("price"),
        data.get("change_pct"),
        data.get("score"),
        data.get("recommendation"),
        data.get("rsi"),
        data.get("adx"),
        data.get("rvol"),
        data.get("macd"),
        data.get("bb_position"),
        data.get("volume"),
        json.dumps(data.get("extra", {})),
    ))


def get_latest_scan() -> pd.DataFrame:
    """Retorna el scan más reciente de cada ticker."""
    return query_df("""
        SELECT DISTINCT ON (ticker)
            ticker, timestamp, price, change_pct, score,
            recommendation, rsi, adx, rvol
        FROM scan_results
        WHERE timestamp >= NOW() - INTERVAL '2 hours'
        ORDER BY ticker, timestamp DESC
    """)


def get_ticker_history(ticker: str, days: int = 3) -> pd.DataFrame:
    """Retorna historial de scores de un ticker."""
    return query_df("""
        SELECT timestamp, score, rsi, rvol, price
        FROM scan_results
        WHERE ticker = %s
          AND timestamp >= NOW() - INTERVAL '%s days'
        ORDER BY timestamp ASC
    """, (ticker, days))


def get_top_picks(n: int = 5) -> pd.DataFrame:
    """Retorna los mejores N picks del último scan."""
    return query_df(f"""
        SELECT DISTINCT ON (ticker)
            ticker, timestamp, price, change_pct,
            score, recommendation, rsi, adx, rvol
        FROM scan_results
        WHERE timestamp >= NOW() - INTERVAL '2 hours'
          AND score > 0
        ORDER BY ticker, timestamp DESC
        LIMIT {n}
    """)


# ─────────────────────────────────────────────────────────────────────────────
# OPERACIONES DE ALERTAS
# ─────────────────────────────────────────────────────────────────────────────

def save_alert(ticker: str, alert_type: str, message: str, channel: str = "telegram"):
    """Guarda alerta enviada."""
    execute("""
        INSERT INTO alerts_sent (ticker, alert_type, message, channel)
        VALUES (%s, %s, %s, %s)
    """, (ticker, alert_type, message, channel))


def get_recent_alerts(hours: int = 24) -> pd.DataFrame:
    """Retorna alertas de las últimas N horas."""
    return query_df("""
        SELECT ticker, timestamp, alert_type, message, channel
        FROM alerts_sent
        WHERE timestamp >= NOW() - INTERVAL '%s hours'
        ORDER BY timestamp DESC
        LIMIT 50
    """, (hours,))


def alert_cooldown_ok(ticker: str, alert_type: str, minutes: int = 30) -> bool:
    """Retorna True si no se ha enviado esta alerta recientemente."""
    df = query_df("""
        SELECT COUNT(*) as cnt
        FROM alerts_sent
        WHERE ticker = %s
          AND alert_type = %s
          AND timestamp >= NOW() - INTERVAL '%s minutes'
    """, (ticker, alert_type, minutes))

    if df.empty:
        return True
    return int(df["cnt"].iloc[0]) == 0


# ─────────────────────────────────────────────────────────────────────────────
# OPERACIONES DEL AGENTE
# ─────────────────────────────────────────────────────────────────────────────

def log_agent_action(action: str, ticker: str, reason: str, score: float = 0):
    """Registra acción del agente (ADD/REMOVE)."""
    execute("""
        INSERT INTO agent_log (action, ticker, reason, score)
        VALUES (%s, %s, %s, %s)
    """, (action.upper(), ticker, reason, score))


def get_agent_log(limit: int = 20) -> pd.DataFrame:
    """Retorna historial de acciones del agente."""
    return query_df(f"""
        SELECT timestamp, action, ticker, reason, score
        FROM agent_log
        ORDER BY timestamp DESC
        LIMIT {limit}
    """)


# ─────────────────────────────────────────────────────────────────────────────
# STATUS DEL SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

def get_scheduler_status() -> Dict:
    """Retorna estado actual del scheduler."""
    df = query_df("""
        SELECT
            MAX(timestamp)            AS last_scan,
            COUNT(DISTINCT timestamp) AS total_scans,
            COUNT(DISTINCT ticker)    AS tickers_scanned
        FROM scan_results
    """)

    if df.empty or df["last_scan"].iloc[0] is None:
        return {
            "running": False,
            "status":  "inactive",
            "label":   "Sin datos",
            "last_scan": None,
            "age_min": None,
            "total_scans": 0,
        }

    last_ts  = pd.to_datetime(df["last_scan"].iloc[0])
    total    = int(df["total_scans"].iloc[0])
    tickers  = int(df["tickers_scanned"].iloc[0])

    # Calcular age en minutos (manejar timezone)
    now = pd.Timestamp.utcnow().tz_localize(None)
    if last_ts.tzinfo is not None:
        last_ts = last_ts.tz_convert(None)
    age_min = (now - last_ts).total_seconds() / 60

    if age_min < 8:
        status = "active"
        label  = f"Activo · hace {age_min:.0f} min"
    elif age_min < 30:
        status = "waiting"
        label  = f"Esperando · hace {age_min:.0f} min"
    else:
        status = "inactive"
        label  = f"Inactivo · hace {age_min:.0f} min"

    return {
        "running":        status == "active",
        "status":         status,
        "label":          label,
        "last_scan":      str(df["last_scan"].iloc[0]),
        "age_min":        round(age_min, 1),
        "total_scans":    total,
        "tickers_scanned": tickers,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LIMPIEZA
# ─────────────────────────────────────────────────────────────────────────────

def cleanup_old_data(days: int = 7):
    """Elimina datos más viejos de N días para mantener la BD limpia."""
    execute("""
        DELETE FROM scan_results
        WHERE timestamp < NOW() - INTERVAL '%s days'
    """, (days,))

    execute("""
        DELETE FROM alerts_sent
        WHERE timestamp < NOW() - INTERVAL '%s days'
    """, (days,))

    logger.info(f"✅ Limpieza completada — datos > {days} días eliminados")
