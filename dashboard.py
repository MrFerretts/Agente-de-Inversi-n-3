"""
╔══════════════════════════════════════════════════════════════════╗
║         PATO QUANT — DASHBOARD AUTÓNOMO                         ║
║                                                                  ║
║  App de Streamlit separada que muestra en tiempo real           ║
║  todo lo que el scheduler y el agente están haciendo.           ║
║                                                                  ║
║  Cómo correrla localmente:                                       ║
║    streamlit run dashboard.py                                    ║
║                                                                  ║
║  En Streamlit Cloud:                                             ║
║    Main file path: dashboard.py                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import time
import json
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pytz
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pato Quant | Dashboard",
    page_icon="🦆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Dark terminal financiero, limpio y profesional
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e2e8f0;
}

/* Header principal */
.dash-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 0 1rem 0;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 1.5rem;
}
.dash-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #f8fafc;
    letter-spacing: -0.02em;
}
.dash-subtitle {
    font-size: 0.75rem;
    color: #64748b;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 2px;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
}
.status-active   { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.status-inactive { background: #2d1b1b; color: #f87171; border: 1px solid #7f1d1d; }
.status-waiting  { background: #1c1917; color: #f59e0b; border: 1px solid #78350f; }

/* Metric cards */
.metric-row { display: flex; gap: 12px; margin-bottom: 1.2rem; }
.metric-card {
    flex: 1;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #334155; }
.metric-label {
    font-size: 0.68rem;
    color: #64748b;
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    color: #f1f5f9;
    line-height: 1;
}
.metric-delta {
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 4px;
}
.delta-up   { color: #4ade80; }
.delta-down { color: #f87171; }
.delta-neu  { color: #94a3b8; }

/* Score pills */
.score-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
}
.score-strong-buy  { background: #052e16; color: #4ade80; }
.score-buy         { background: #0c2a1a; color: #86efac; }
.score-hold        { background: #1c1917; color: #94a3b8; }
.score-sell        { background: #2d1515; color: #fca5a5; }
.score-strong-sell { background: #3b0000; color: #f87171; }

/* Top picks cards */
.pick-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: border-color 0.15s, transform 0.15s;
}
.pick-card:hover { border-color: #22d3ee; transform: translateX(2px); }
.pick-ticker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem;
    font-weight: 600;
    color: #f1f5f9;
}
.pick-price {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 2px;
}
.pick-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.2rem;
    font-weight: 600;
}

/* Tabla de activos */
.asset-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
}
.asset-table th {
    padding: 8px 12px;
    text-align: left;
    color: #64748b;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid #1e293b;
    font-weight: 500;
}
.asset-table td {
    padding: 9px 12px;
    border-bottom: 1px solid #0f172a;
    color: #cbd5e1;
}
.asset-table tr:hover td { background: #0f172a; color: #f1f5f9; }
.ticker-cell { color: #38bdf8; font-weight: 600; }

/* Alert log */
.alert-item {
    padding: 8px 12px;
    border-left: 3px solid;
    border-radius: 0 6px 6px 0;
    margin-bottom: 6px;
    background: #0f172a;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
}
.alert-buy    { border-color: #4ade80; }
.alert-sell   { border-color: #f87171; }
.alert-agent  { border-color: #818cf8; }
.alert-vol    { border-color: #f59e0b; }
.alert-time   { color: #475569; font-size: 0.68rem; }
.alert-msg    { color: #cbd5e1; margin-top: 2px; }

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.8rem;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e293b;
}

/* No data state */
.no-data {
    text-align: center;
    padding: 2rem;
    color: #475569;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    border: 1px dashed #1e293b;
    border-radius: 10px;
}

/* Streamlit overrides */
[data-testid="stMetric"] { display: none; }
div.block-container { padding-top: 1rem; padding-bottom: 1rem; }
hr { border-color: #1e293b; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONEXIÓN A BASE DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = "data/scheduler.db"

def get_db():
    """Retorna conexión a SQLite del scheduler."""
    if not Path(DB_PATH).exists():
        return None
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Ejecuta query y retorna DataFrame. Retorna vacío si no hay BD."""
    conn = get_db()
    if conn is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def scheduler_status() -> dict:
    """Retorna estado actual del scheduler."""
    df = query("""
        SELECT MAX(timestamp) as last_ts,
               COUNT(DISTINCT timestamp) as total_scans
        FROM scan_results
    """)

    if df.empty or df["last_ts"].iloc[0] is None:
        return {
            "running": False,
            "status": "inactive",
            "label": "Sin datos",
            "last_scan": None,
            "age_min": None,
            "total_scans": 0,
        }

    last_ts_str = df["last_ts"].iloc[0]
    total = int(df["total_scans"].iloc[0])

    try:
        last_ts = pd.to_datetime(last_ts_str).replace(tzinfo=None)
        age_min = (datetime.utcnow() - last_ts).total_seconds() / 60

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
            "running": status == "active",
            "status": status,
            "label": label,
            "last_scan": last_ts_str,
            "age_min": age_min,
            "total_scans": total,
        }
    except Exception:
        return {"running": False, "status": "inactive",
                "label": "Error", "last_scan": None,
                "age_min": None, "total_scans": 0}


def load_watchlist() -> list:
    path = Path("data/watchlist.json")
    if path.exists():
        with open(path) as f:
            d = json.load(f)
        return d.get("stocks", []) + d.get("crypto", [])
    return []


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE FORMATO
# ─────────────────────────────────────────────────────────────────────────────

def score_class(score) -> str:
    try:
        s = float(score)
    except Exception:
        return "score-hold"
    if s >= 60:  return "score-strong-buy"
    if s >= 30:  return "score-buy"
    if s >= -30: return "score-hold"
    if s >= -60: return "score-sell"
    return "score-strong-sell"

def score_label(score) -> str:
    try:
        s = float(score)
    except Exception:
        return "—"
    if s >= 60:  return "COMPRA FUERTE"
    if s >= 30:  return "COMPRA"
    if s >= -30: return "MANTENER"
    if s >= -60: return "VENTA"
    return "VENTA FUERTE"

def change_color(val) -> str:
    try:
        return "delta-up" if float(val) >= 0 else "delta-down"
    except Exception:
        return "delta-neu"

def fmt_ts(ts_str: Optional[str]) -> str:
    if not ts_str:
        return "—"
    try:
        dt = pd.to_datetime(ts_str)
        tz = pytz.timezone("America/Monterrey")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.utc)
        return dt.astimezone(tz).strftime("%H:%M:%S")
    except Exception:
        return ts_str[:19] if ts_str else "—"


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

tz_mty = pytz.timezone("America/Monterrey")
now_str = datetime.now(tz_mty).strftime("%a %d %b %Y · %H:%M:%S CST")
status  = scheduler_status()

# Status class
scls = {"active": "status-active",
        "waiting": "status-waiting",
        "inactive": "status-inactive"}.get(status["status"], "status-inactive")
sdot = {"active": "●", "waiting": "◐", "inactive": "○"}.get(status["status"], "○")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dash-header">
  <div>
    <div class="dash-title">🦆 Pato Quant · Dashboard Autónomo</div>
    <div class="dash-subtitle">{now_str}</div>
  </div>
  <div class="status-badge {scls}">{sdot} {status['label']}</div>
</div>
""", unsafe_allow_html=True)

# ── Metric cards ──────────────────────────────────────────────────────────────
wl = load_watchlist()

df_latest = query("""
    SELECT ticker, timestamp, price, change_pct, score, recommendation, rsi, adx, rvol
    FROM scan_results
    WHERE timestamp >= datetime('now', '-60 minutes')
    ORDER BY ABS(score) DESC
""")

df_alerts_today = query("""
    SELECT COUNT(*) as cnt FROM alerts_sent
    WHERE timestamp >= date('now')
""")

alerts_count = int(df_alerts_today["cnt"].iloc[0]) if not df_alerts_today.empty else 0
top_score    = int(df_latest["score"].max()) if not df_latest.empty else 0
top_ticker   = df_latest.iloc[0]["ticker"] if not df_latest.empty else "—"

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="metric-label">Scheduler</div>
    <div class="metric-value" style="font-size:1rem;margin-top:4px">{status['label']}</div>
    <div class="metric-delta delta-neu">Scans totales: {status['total_scans']}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Watchlist</div>
    <div class="metric-value">{len(wl)}</div>
    <div class="metric-delta delta-neu">de 50 activos máximo</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Mejor score</div>
    <div class="metric-value" style="color:#4ade80">{top_score}</div>
    <div class="metric-delta delta-neu">{top_ticker}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Alertas hoy</div>
    <div class="metric-value">{alerts_count}</div>
    <div class="metric-delta delta-neu">señales detectadas</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Último scan</div>
    <div class="metric-value" style="font-size:1.1rem;margin-top:4px">{fmt_ts(status['last_scan'])}</div>
    <div class="metric-delta delta-neu">hora Monterrey</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT EN 3 COLUMNAS
# ─────────────────────────────────────────────────────────────────────────────

col_left, col_mid, col_right = st.columns([2.2, 1.4, 1.4])

# ══════════════════════════════════════════════════════════════════
# COLUMNA IZQUIERDA — Tabla completa de activos
# ══════════════════════════════════════════════════════════════════
with col_left:
    st.markdown('<div class="section-header">Todos los activos · último scan</div>',
                unsafe_allow_html=True)

    if df_latest.empty:
        st.markdown("""
        <div class="no-data">
          Sin datos aún.<br>
          El scheduler empieza a las 9:30 AM ET.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Construir tabla HTML
        rows_html = ""
        for _, row in df_latest.iterrows():
            sc   = float(row.get("score", 0))
            chg  = float(row.get("change_pct", 0))
            scls_pill = score_class(sc)
            slbl      = score_label(sc)
            chg_cls   = "delta-up" if chg >= 0 else "delta-down"
            chg_sign  = "+" if chg >= 0 else ""

            rows_html += f"""
            <tr>
              <td class="ticker-cell">{row['ticker']}</td>
              <td>${float(row.get('price',0)):.2f}</td>
              <td class="{chg_cls}">{chg_sign}{chg:.2f}%</td>
              <td><span class="score-pill {scls_pill}">{sc:+.0f}</span></td>
              <td style="color:#94a3b8">{slbl}</td>
              <td>{float(row.get('rsi',0)):.1f}</td>
              <td>{float(row.get('adx',0)):.1f}</td>
              <td>{float(row.get('rvol',0)):.2f}x</td>
              <td style="color:#475569;font-size:0.7rem">{fmt_ts(row.get('timestamp',''))}</td>
            </tr>"""

        st.markdown(f"""
        <table class="asset-table">
          <thead>
            <tr>
              <th>Ticker</th><th>Precio</th><th>Cambio</th>
              <th>Score</th><th>Señal</th><th>RSI</th>
              <th>ADX</th><th>RVOL</th><th>Hora</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        # Gráfica de scores
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Distribución de scores · último scan</div>',
                    unsafe_allow_html=True)

        df_chart = df_latest.sort_values("score", ascending=True).tail(20)
        colors   = ["#4ade80" if s >= 30 else "#f87171" if s <= -30 else "#94a3b8"
                    for s in df_chart["score"]]

        fig = go.Figure(go.Bar(
            x=df_chart["score"],
            y=df_chart["ticker"],
            orientation="h",
            marker_color=colors,
            text=df_chart["score"].apply(lambda s: f"{s:+.0f}"),
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10, color="#94a3b8"),
        ))
        fig.update_layout(
            plot_bgcolor="#0a0a0f",
            paper_bgcolor="#0a0a0f",
            font=dict(family="IBM Plex Mono", color="#64748b", size=10),
            height=max(280, len(df_chart) * 22),
            margin=dict(l=0, r=30, t=10, b=10),
            xaxis=dict(
                showgrid=True, gridcolor="#1e293b",
                zeroline=True, zerolinecolor="#334155",
                tickfont=dict(size=9),
            ),
            yaxis=dict(showgrid=False, tickfont=dict(size=10, color="#94a3b8")),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════
# COLUMNA CENTRAL — Top 5 picks + scores en el tiempo
# ══════════════════════════════════════════════════════════════════
with col_mid:

    # Top 5 picks
    st.markdown('<div class="section-header">Top 5 picks · agente</div>',
                unsafe_allow_html=True)

    if df_latest.empty:
        st.markdown('<div class="no-data">Sin picks aún</div>', unsafe_allow_html=True)
    else:
        top5 = df_latest[df_latest["score"] > 0].head(5)

        if top5.empty:
            st.markdown('<div class="no-data">Sin picks alcistas en este scan</div>',
                        unsafe_allow_html=True)
        else:
            for rank, (_, row) in enumerate(top5.iterrows(), 1):
                sc   = float(row.get("score", 0))
                chg  = float(row.get("change_pct", 0))
                chg_sign = "+" if chg >= 0 else ""
                chg_cls  = "#4ade80" if chg >= 0 else "#f87171"
                sc_color = "#4ade80" if sc >= 60 else "#86efac" if sc >= 30 else "#94a3b8"

                st.markdown(f"""
                <div class="pick-card">
                  <div>
                    <div style="display:flex;align-items:center;gap:8px">
                      <span style="color:#475569;font-family:'IBM Plex Mono',monospace;
                                   font-size:0.72rem">#{rank}</span>
                      <span class="pick-ticker">{row['ticker']}</span>
                    </div>
                    <div class="pick-price">
                      ${float(row.get('price',0)):.2f}
                      <span style="color:{chg_cls};margin-left:6px">
                        {chg_sign}{chg:.2f}%
                      </span>
                    </div>
                    <div style="margin-top:4px;font-size:0.68rem;
                                color:#475569;font-family:'IBM Plex Mono',monospace">
                      RSI {float(row.get('rsi',0)):.0f} ·
                      RVOL {float(row.get('rvol',0)):.2f}x ·
                      ADX {float(row.get('adx',0)):.0f}
                    </div>
                  </div>
                  <div class="pick-score" style="color:{sc_color}">{sc:+.0f}</div>
                </div>
                """, unsafe_allow_html=True)

    # Score en el tiempo — ticker seleccionado
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Score en el tiempo</div>',
                unsafe_allow_html=True)

    ticker_options = wl if wl else ["AAPL"]
    selected = st.selectbox(
        "Seleccionar activo",
        ticker_options,
        label_visibility="collapsed",
    )

    df_hist = query("""
        SELECT timestamp, score, rsi, rvol
        FROM scan_results
        WHERE ticker = ?
          AND timestamp >= datetime('now', '-3 days')
        ORDER BY timestamp ASC
    """, (selected,))

    if df_hist.empty:
        st.markdown(f'<div class="no-data">Sin historial para {selected}</div>',
                    unsafe_allow_html=True)
    else:
        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
        df_hist["score"]     = df_hist["score"].astype(float)

        fig2 = go.Figure()

        # Zona de colores de fondo
        fig2.add_hrect(y0=60, y1=100,  fillcolor="#052e16", opacity=0.3, line_width=0)
        fig2.add_hrect(y0=30, y1=60,   fillcolor="#0c2a1a", opacity=0.2, line_width=0)
        fig2.add_hrect(y0=-30, y1=30,  fillcolor="#1c1917", opacity=0.2, line_width=0)
        fig2.add_hrect(y0=-60, y1=-30, fillcolor="#2d1515", opacity=0.2, line_width=0)
        fig2.add_hrect(y0=-100, y1=-60, fillcolor="#3b0000", opacity=0.3, line_width=0)

        fig2.add_trace(go.Scatter(
            x=df_hist["timestamp"],
            y=df_hist["score"],
            mode="lines+markers",
            line=dict(color="#38bdf8", width=2),
            marker=dict(size=5, color="#38bdf8"),
            name="Score",
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.06)",
        ))

        fig2.add_hline(y=0, line_color="#334155", line_width=1)

        fig2.update_layout(
            plot_bgcolor="#0a0a0f",
            paper_bgcolor="#0a0a0f",
            font=dict(family="IBM Plex Mono", color="#64748b", size=9),
            height=200,
            margin=dict(l=0, r=0, t=10, b=10),
            showlegend=False,
            xaxis=dict(showgrid=False, tickfont=dict(size=8)),
            yaxis=dict(
                showgrid=True, gridcolor="#1e293b",
                range=[-110, 110],
                tickfont=dict(size=8),
            ),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════
# COLUMNA DERECHA — Log de alertas + cambios de watchlist
# ══════════════════════════════════════════════════════════════════
with col_right:

    st.markdown('<div class="section-header">Alertas recientes · 24h</div>',
                unsafe_allow_html=True)

    df_alerts = query("""
        SELECT ticker, timestamp, alert_type, message
        FROM alerts_sent
        WHERE timestamp >= datetime('now', '-24 hours')
        ORDER BY timestamp DESC
        LIMIT 30
    """)

    if df_alerts.empty:
        st.markdown('<div class="no-data">Sin alertas en las últimas 24h</div>',
                    unsafe_allow_html=True)
    else:
        for _, alert in df_alerts.iterrows():
            atype = str(alert.get("alert_type", "")).upper()
            msg   = str(alert.get("message", ""))[:80]
            ts    = fmt_ts(alert.get("timestamp"))

            if "COMPRA" in atype or "BUY" in atype:
                cls = "alert-buy"
            elif "VENTA" in atype or "SELL" in atype:
                cls = "alert-sell"
            elif "AGENT" in atype or "WATCHLIST" in atype:
                cls = "alert-agent"
            else:
                cls = "alert-vol"

            st.markdown(f"""
            <div class="alert-item {cls}">
              <div style="display:flex;justify-content:space-between">
                <span style="color:#e2e8f0;font-weight:500">{alert.get('ticker','')}</span>
                <span class="alert-time">{ts}</span>
              </div>
              <div class="alert-msg">{atype.replace('_',' ')}</div>
              <div style="color:#475569;font-size:0.68rem;margin-top:2px">{msg}</div>
            </div>
            """, unsafe_allow_html=True)

    # Cambios del agente
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Cambios del agente · watchlist</div>',
                unsafe_allow_html=True)

    df_agent = query("""
        SELECT timestamp, message
        FROM alerts_sent
        WHERE alert_type = 'watchlist_update'
        ORDER BY timestamp DESC
        LIMIT 5
    """)

    if df_agent.empty:
        st.markdown('<div class="no-data">El agente aún no ha modificado la watchlist</div>',
                    unsafe_allow_html=True)
    else:
        for _, row in df_agent.iterrows():
            ts  = fmt_ts(row.get("timestamp"))
            msg = str(row.get("message", ""))

            # Extraer tickers agregados/eliminados del mensaje
            lines = [l.strip() for l in msg.split("\n") if "•" in l]
            preview = " · ".join(lines[:4]) if lines else msg[:80]

            st.markdown(f"""
            <div class="alert-item alert-agent">
              <div class="alert-time">{ts}</div>
              <div class="alert-msg" style="margin-top:4px">{preview}</div>
            </div>
            """, unsafe_allow_html=True)

    # Watchlist actual
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Watchlist actual</div>',
                unsafe_allow_html=True)

    if wl:
        chips_html = " ".join([
            f'<span style="display:inline-block;padding:2px 8px;margin:2px;'
            f'background:#0f172a;border:1px solid #1e293b;border-radius:12px;'
            f'font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;'
            f'color:#38bdf8">{t}</span>'
            for t in sorted(wl)
        ])
        st.markdown(chips_html, unsafe_allow_html=True)
    else:
        st.markdown('<div class="no-data">Sin tickers</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-REFRESH
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
col_ref1, col_ref2, col_ref3 = st.columns([2, 1, 2])
with col_ref2:
    auto = st.toggle("🔄 Auto-refresh (60s)", value=False)
    if auto:
        time.sleep(60)
        st.rerun()
