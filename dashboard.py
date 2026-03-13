"""
PATO QUANT — DASHBOARD AUTÓNOMO (Supabase version)
Lee datos desde Supabase PostgreSQL — funciona desde cualquier servidor.
"""

import time
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import pytz
import os
import streamlit as st

st.set_page_config(
    page_title="Pato Quant | Dashboard",
    page_icon="🦆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONEXIÓN
# ─────────────────────────────────────────────────────────────────────────────

def get_db_url() -> str:
    try:
        return st.secrets["SUPABASE_DB_URL"]
    except Exception:
        return os.getenv("SUPABASE_DB_URL", "")

@st.cache_resource
def get_engine():
    from sqlalchemy import create_engine
    url = get_db_url()
    if not url:
        return None
    return create_engine(url, pool_size=3, max_overflow=5,
                         pool_timeout=10, pool_pre_ping=True)

def query(sql: str, params: dict = {}) -> pd.DataFrame:
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            return pd.read_sql_query(text(sql), conn, params=params)
    except Exception:
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; background:#0a0a0f; color:#e2e8f0; }
.dash-header { display:flex; align-items:center; justify-content:space-between; padding:1.2rem 0 1rem 0; border-bottom:1px solid #1e293b; margin-bottom:1.5rem; }
.dash-title  { font-family:'IBM Plex Mono',monospace; font-size:1.4rem; font-weight:600; color:#f8fafc; letter-spacing:-0.02em; }
.dash-sub    { font-size:0.75rem; color:#64748b; font-family:'IBM Plex Mono',monospace; margin-top:2px; }
.status-badge { display:inline-flex; align-items:center; gap:6px; padding:4px 12px; border-radius:20px; font-size:0.72rem; font-family:'IBM Plex Mono',monospace; font-weight:500; }
.status-active   { background:#052e16; color:#4ade80; border:1px solid #166534; }
.status-inactive { background:#2d1b1b; color:#f87171; border:1px solid #7f1d1d; }
.status-waiting  { background:#1c1917; color:#f59e0b; border:1px solid #78350f; }
.metric-row { display:flex; gap:12px; margin-bottom:1.2rem; }
.metric-card { flex:1; background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:1rem 1.2rem; }
.metric-label { font-size:0.68rem; color:#64748b; font-family:'IBM Plex Mono',monospace; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px; }
.metric-value { font-size:1.6rem; font-weight:600; font-family:'IBM Plex Mono',monospace; color:#f1f5f9; line-height:1; }
.metric-delta { font-size:0.72rem; font-family:'IBM Plex Mono',monospace; margin-top:4px; }
.delta-up { color:#4ade80; } .delta-down { color:#f87171; } .delta-neu { color:#94a3b8; }
.score-pill { display:inline-block; padding:2px 10px; border-radius:12px; font-family:'IBM Plex Mono',monospace; font-size:0.78rem; font-weight:600; }
.score-strong-buy  { background:#052e16; color:#4ade80; }
.score-buy         { background:#0c2a1a; color:#86efac; }
.score-hold        { background:#1c1917; color:#94a3b8; }
.score-sell        { background:#2d1515; color:#fca5a5; }
.score-strong-sell { background:#3b0000; color:#f87171; }
.pick-card { background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:1rem; margin-bottom:8px; display:flex; align-items:center; justify-content:space-between; transition:border-color 0.15s,transform 0.15s; }
.pick-card:hover { border-color:#22d3ee; transform:translateX(2px); }
.pick-ticker { font-family:'IBM Plex Mono',monospace; font-size:1rem; font-weight:600; color:#f1f5f9; }
.pick-price  { font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:#94a3b8; margin-top:2px; }
.asset-table { width:100%; border-collapse:collapse; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; }
.asset-table th { padding:8px 12px; text-align:left; color:#64748b; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.06em; border-bottom:1px solid #1e293b; font-weight:500; }
.asset-table td { padding:9px 12px; border-bottom:1px solid #0f172a; color:#cbd5e1; }
.asset-table tr:hover td { background:#0f172a; color:#f1f5f9; }
.ticker-cell { color:#38bdf8; font-weight:600; }
.alert-item { padding:8px 12px; border-left:3px solid; border-radius:0 6px 6px 0; margin-bottom:6px; background:#0f172a; font-size:0.78rem; font-family:'IBM Plex Mono',monospace; }
.alert-buy   { border-color:#4ade80; } .alert-sell  { border-color:#f87171; }
.alert-agent { border-color:#818cf8; } .alert-vol   { border-color:#f59e0b; }
.section-header { font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.8rem; padding-bottom:6px; border-bottom:1px solid #1e293b; }
.no-data { text-align:center; padding:2rem; color:#475569; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; border:1px dashed #1e293b; border-radius:10px; }
div.block-container { padding-top:1rem; padding-bottom:1rem; }
hr { border-color:#1e293b; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# VERIFICAR CONEXIÓN
# ─────────────────────────────────────────────────────────────────────────────

if not get_db_url():
    st.error("⚠️ SUPABASE_DB_URL no configurada en Secrets.")
    st.code('SUPABASE_DB_URL = "postgresql://postgres.xxxx:PASSWORD@..."')
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def score_class(s) -> str:
    try: s = float(s)
    except: return "score-hold"
    if s >= 60:  return "score-strong-buy"
    if s >= 30:  return "score-buy"
    if s >= -30: return "score-hold"
    if s >= -60: return "score-sell"
    return "score-strong-sell"

def score_label(s) -> str:
    try: s = float(s)
    except: return "—"
    if s >= 60:  return "COMPRA FUERTE"
    if s >= 30:  return "COMPRA"
    if s >= -30: return "MANTENER"
    if s >= -60: return "VENTA"
    return "VENTA FUERTE"

def fmt_ts(ts) -> str:
    if ts is None: return "—"
    try:
        dt = pd.to_datetime(ts)
        tz = pytz.timezone("America/Monterrey")
        if dt.tzinfo is None: dt = dt.replace(tzinfo=pytz.utc)
        return dt.astimezone(tz).strftime("%H:%M:%S")
    except: return str(ts)[:19]

# ─────────────────────────────────────────────────────────────────────────────
# CARGAR DATOS (cache 60s)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_data():
    latest = query("""
        SELECT DISTINCT ON (ticker) ticker, timestamp, price, change_pct,
               score, recommendation, rsi, adx, rvol
        FROM scan_results
        WHERE timestamp >= NOW() - INTERVAL '2 hours'
        ORDER BY ticker, timestamp DESC
    """)

    top5 = pd.DataFrame()
    if not latest.empty:
        top5 = latest[latest["score"] > 0].sort_values("score", ascending=False).head(5)

    alerts = query("""
        SELECT ticker, timestamp, alert_type, message
        FROM alerts_sent
        WHERE timestamp >= NOW() - INTERVAL '24 hours'
        ORDER BY timestamp DESC LIMIT 30
    """)

    agent_log = query("""
        SELECT timestamp, action, ticker, reason, score
        FROM agent_log ORDER BY timestamp DESC LIMIT 20
    """)

    status_df = query("""
        SELECT MAX(timestamp) AS last_scan,
               COUNT(DISTINCT timestamp) AS total_scans
        FROM scan_results
    """)

    alerts_today = query("SELECT COUNT(*) AS cnt FROM alerts_sent WHERE timestamp >= CURRENT_DATE")

    watchlist = query("SELECT ticker FROM watchlist WHERE is_active = TRUE ORDER BY added_at ASC")

    return latest, top5, alerts, agent_log, status_df, alerts_today, watchlist

df_latest, df_top5, df_alerts, df_agent, df_status_raw, df_alerts_today, df_wl = load_data()

# Calcular status
def compute_status(df):
    if df.empty or df["last_scan"].iloc[0] is None:
        return {"status": "inactive", "label": "Sin datos", "total_scans": 0, "last_scan": None}
    last_ts = pd.to_datetime(df["last_scan"].iloc[0])
    total   = int(df["total_scans"].iloc[0])
    now     = pd.Timestamp.utcnow().tz_localize(None)
    if last_ts.tzinfo is not None: last_ts = last_ts.tz_convert(None)
    age     = (now - last_ts).total_seconds() / 60
    if age < 8:   return {"status":"active",   "label":f"Activo · hace {age:.0f} min",   "total_scans":total, "last_scan":df["last_scan"].iloc[0]}
    if age < 30:  return {"status":"waiting",  "label":f"Esperando · hace {age:.0f} min","total_scans":total, "last_scan":df["last_scan"].iloc[0]}
    return             {"status":"inactive", "label":f"Inactivo · hace {age:.0f} min","total_scans":total, "last_scan":df["last_scan"].iloc[0]}

status     = compute_status(df_status_raw)
wl_tickers = df_wl["ticker"].tolist() if not df_wl.empty else []
alerts_cnt = int(df_alerts_today["cnt"].iloc[0]) if not df_alerts_today.empty else 0
top_score  = int(df_latest["score"].max()) if not df_latest.empty else 0
top_ticker = df_latest.sort_values("score", ascending=False).iloc[0]["ticker"] if not df_latest.empty else "—"

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

tz_mty  = pytz.timezone("America/Monterrey")
now_str = datetime.now(tz_mty).strftime("%a %d %b %Y · %H:%M:%S CST")
scls    = {"active":"status-active","waiting":"status-waiting","inactive":"status-inactive"}.get(status["status"],"status-inactive")
sdot    = {"active":"●","waiting":"◐","inactive":"○"}.get(status["status"],"○")

st.markdown(f"""
<div class="dash-header">
  <div>
    <div class="dash-title">🦆 Pato Quant · Dashboard Autónomo</div>
    <div class="dash-sub">{now_str}</div>
  </div>
  <div class="status-badge {scls}">{sdot} {status['label']}</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="metric-label">Scheduler</div>
    <div class="metric-value" style="font-size:1rem;margin-top:4px">{status['label']}</div>
    <div class="metric-delta delta-neu">Scans totales: {status['total_scans']}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Watchlist</div>
    <div class="metric-value">{len(wl_tickers)}</div>
    <div class="metric-delta delta-neu">de 50 máximo</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Mejor score</div>
    <div class="metric-value" style="color:#4ade80">{top_score}</div>
    <div class="metric-delta delta-neu">{top_ticker}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Alertas hoy</div>
    <div class="metric-value">{alerts_cnt}</div>
    <div class="metric-delta delta-neu">señales</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Último scan</div>
    <div class="metric-value" style="font-size:1.1rem;margin-top:4px">{fmt_ts(status['last_scan'])}</div>
    <div class="metric-delta delta-neu">hora Monterrey</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMNAS
# ─────────────────────────────────────────────────────────────────────────────

col_left, col_mid, col_right = st.columns([2.2, 1.4, 1.4])

# ═══ IZQUIERDA ═══
with col_left:
    st.markdown('<div class="section-header">Todos los activos · último scan</div>', unsafe_allow_html=True)

    if df_latest.empty:
        st.markdown('<div class="no-data">Sin datos aún.<br>El scheduler corre a las 9:30 AM ET.</div>', unsafe_allow_html=True)
    else:
        rows = ""
        for _, row in df_latest.sort_values("score", ascending=False).iterrows():
            sc  = float(row.get("score", 0))
            chg = float(row.get("change_pct", 0))
            cc  = "delta-up" if chg >= 0 else "delta-down"
            cs  = "+" if chg >= 0 else ""
            rows += f"""<tr>
              <td class="ticker-cell">{row['ticker']}</td>
              <td>${float(row.get('price',0)):.2f}</td>
              <td class="{cc}">{cs}{chg:.2f}%</td>
              <td><span class="score-pill {score_class(sc)}">{sc:+.0f}</span></td>
              <td style="color:#94a3b8">{score_label(sc)}</td>
              <td>{float(row.get('rsi',0)):.1f}</td>
              <td>{float(row.get('adx',0)):.1f}</td>
              <td>{float(row.get('rvol',0)):.2f}x</td>
              <td style="color:#475569;font-size:0.7rem">{fmt_ts(row.get('timestamp'))}</td>
            </tr>"""
        st.markdown(f"""
        <table class="asset-table">
          <thead><tr><th>Ticker</th><th>Precio</th><th>Cambio</th><th>Score</th>
          <th>Señal</th><th>RSI</th><th>ADX</th><th>RVOL</th><th>Hora</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Distribución de scores</div>', unsafe_allow_html=True)
        df_c   = df_latest.sort_values("score", ascending=True).tail(20)
        colors = ["#4ade80" if s >= 30 else "#f87171" if s <= -30 else "#94a3b8" for s in df_c["score"]]
        fig = go.Figure(go.Bar(
            x=df_c["score"], y=df_c["ticker"], orientation="h",
            marker_color=colors,
            text=df_c["score"].apply(lambda s: f"{s:+.0f}"),
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10, color="#94a3b8"),
        ))
        fig.update_layout(
            plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f",
            font=dict(family="IBM Plex Mono", color="#64748b", size=10),
            height=max(280, len(df_c)*22), margin=dict(l=0,r=30,t=10,b=10),
            xaxis=dict(showgrid=True,gridcolor="#1e293b",zeroline=True,zerolinecolor="#334155"),
            yaxis=dict(showgrid=False,tickfont=dict(size=10,color="#94a3b8")),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

# ═══ CENTRO ═══
with col_mid:
    st.markdown('<div class="section-header">Top 5 picks</div>', unsafe_allow_html=True)
    if df_top5.empty:
        st.markdown('<div class="no-data">Sin picks aún</div>', unsafe_allow_html=True)
    else:
        for rank, (_, row) in enumerate(df_top5.iterrows(), 1):
            sc  = float(row.get("score",0))
            chg = float(row.get("change_pct",0))
            cs  = "+" if chg >= 0 else ""
            cc  = "#4ade80" if chg >= 0 else "#f87171"
            sc_clr = "#4ade80" if sc >= 60 else "#86efac" if sc >= 30 else "#94a3b8"
            st.markdown(f"""
            <div class="pick-card">
              <div>
                <div style="display:flex;align-items:center;gap:8px">
                  <span style="color:#475569;font-family:'IBM Plex Mono',monospace;font-size:0.72rem">#{rank}</span>
                  <span class="pick-ticker">{row['ticker']}</span>
                </div>
                <div class="pick-price">${float(row.get('price',0)):.2f}
                  <span style="color:{cc};margin-left:6px">{cs}{chg:.2f}%</span>
                </div>
                <div style="margin-top:4px;font-size:0.68rem;color:#475569;font-family:'IBM Plex Mono',monospace">
                  RSI {float(row.get('rsi',0)):.0f} · RVOL {float(row.get('rvol',0)):.2f}x · ADX {float(row.get('adx',0)):.0f}
                </div>
              </div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:1.2rem;font-weight:600;color:{sc_clr}">{sc:+.0f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Score en el tiempo</div>', unsafe_allow_html=True)
    selected = st.selectbox("Activo", wl_tickers if wl_tickers else ["—"], label_visibility="collapsed")

    if selected and selected != "—":
        df_hist = query("""
            SELECT timestamp, score FROM scan_results
            WHERE ticker = :ticker AND timestamp >= NOW() - INTERVAL '3 days'
            ORDER BY timestamp ASC
        """, {"ticker": selected})

        if df_hist.empty:
            st.markdown(f'<div class="no-data">Sin historial para {selected}</div>', unsafe_allow_html=True)
        else:
            df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
            df_hist["score"]     = df_hist["score"].astype(float)
            fig2 = go.Figure()
            fig2.add_hrect(y0=60,  y1=100,  fillcolor="#052e16", opacity=0.3, line_width=0)
            fig2.add_hrect(y0=30,  y1=60,   fillcolor="#0c2a1a", opacity=0.2, line_width=0)
            fig2.add_hrect(y0=-30, y1=30,   fillcolor="#1c1917", opacity=0.2, line_width=0)
            fig2.add_hrect(y0=-60, y1=-30,  fillcolor="#2d1515", opacity=0.2, line_width=0)
            fig2.add_hrect(y0=-100,y1=-60,  fillcolor="#3b0000", opacity=0.3, line_width=0)
            fig2.add_trace(go.Scatter(
                x=df_hist["timestamp"], y=df_hist["score"],
                mode="lines+markers", line=dict(color="#38bdf8",width=2),
                marker=dict(size=5,color="#38bdf8"),
                fill="tozeroy", fillcolor="rgba(56,189,248,0.06)",
            ))
            fig2.add_hline(y=0, line_color="#334155", line_width=1)
            fig2.update_layout(
                plot_bgcolor="#0a0a0f", paper_bgcolor="#0a0a0f",
                font=dict(family="IBM Plex Mono",color="#64748b",size=9),
                height=200, margin=dict(l=0,r=0,t=10,b=10), showlegend=False,
                xaxis=dict(showgrid=False,tickfont=dict(size=8)),
                yaxis=dict(showgrid=True,gridcolor="#1e293b",range=[-110,110],tickfont=dict(size=8)),
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

# ═══ DERECHA ═══
with col_right:
    st.markdown('<div class="section-header">Alertas · 24h</div>', unsafe_allow_html=True)
    if df_alerts.empty:
        st.markdown('<div class="no-data">Sin alertas en 24h</div>', unsafe_allow_html=True)
    else:
        for _, alert in df_alerts.iterrows():
            atype = str(alert.get("alert_type","")).upper()
            msg   = str(alert.get("message",""))[:80]
            ts    = fmt_ts(alert.get("timestamp"))
            cls   = ("alert-buy"   if any(x in atype for x in ["COMPRA","BUY"]) else
                     "alert-sell"  if any(x in atype for x in ["VENTA","SELL"]) else
                     "alert-agent" if any(x in atype for x in ["AGENT","WATCHLIST"]) else
                     "alert-vol")
            st.markdown(f"""
            <div class="alert-item {cls}">
              <div style="display:flex;justify-content:space-between">
                <span style="color:#e2e8f0;font-weight:500">{alert.get('ticker','')}</span>
                <span style="color:#475569;font-size:0.68rem">{ts}</span>
              </div>
              <div style="color:#cbd5e1;margin-top:2px">{atype.replace('_',' ')}</div>
              <div style="color:#475569;font-size:0.68rem;margin-top:2px">{msg}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Acciones del agente</div>', unsafe_allow_html=True)
    if df_agent.empty:
        st.markdown('<div class="no-data">El agente aún no modificó la watchlist</div>', unsafe_allow_html=True)
    else:
        for _, row in df_agent.iterrows():
            action = str(row.get("action",""))
            ticker = str(row.get("ticker",""))
            reason = str(row.get("reason",""))[:60]
            ts     = fmt_ts(row.get("timestamp"))
            clr    = "#4ade80" if action == "ADD" else "#f87171"
            icon   = "✅" if action == "ADD" else "🗑️"
            st.markdown(f"""
            <div class="alert-item alert-agent">
              <div style="display:flex;justify-content:space-between">
                <span style="color:{clr};font-weight:600">{icon} {action} {ticker}</span>
                <span style="color:#475569;font-size:0.68rem">{ts}</span>
              </div>
              <div style="color:#64748b;font-size:0.7rem;margin-top:2px">{reason}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Watchlist actual</div>', unsafe_allow_html=True)
    if wl_tickers:
        chips = " ".join([
            f'<span style="display:inline-block;padding:2px 8px;margin:2px;background:#0f172a;'
            f'border:1px solid #1e293b;border-radius:12px;font-family:\'IBM Plex Mono\',monospace;'
            f'font-size:0.72rem;color:#38bdf8">{t}</span>'
            for t in sorted(wl_tickers)
        ])
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.markdown('<div class="no-data">Sin tickers</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-REFRESH
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
_, col_c, _ = st.columns([2,1,2])
with col_c:
    if st.toggle("🔄 Auto-refresh 60s", value=False):
        time.sleep(60)
        st.cache_data.clear()
        st.rerun()
