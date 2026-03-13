"""
╔══════════════════════════════════════════════════════════════════╗
║     PATO QUANT — INTEGRACIÓN STREAMLIT ↔ SCHEDULER             ║
║                                                                  ║
║  Pega este código en tu app.py para que Streamlit muestre       ║
║  los resultados del scheduler en lugar de calcularlos.          ║
╚══════════════════════════════════════════════════════════════════╝

INSTRUCCIONES:
  1. Copia scheduler.py y scheduler_reader.py a la raíz de tu proyecto.
  2. Agrega los bloques de abajo en las secciones indicadas de app.py.
  3. Corre el scheduler por separado: python scheduler.py
  4. Streamlit automáticamente mostrará los datos del scheduler.
"""

# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 1: Agregar al inicio de app.py (junto a los otros imports)
# ─────────────────────────────────────────────────────────────────────────────

from scheduler_reader import SchedulerReader

# Inicializar reader (solo lectura, no calcula nada)
if "scheduler_reader" not in st.session_state:
    st.session_state.scheduler_reader = SchedulerReader()

scheduler_reader = st.session_state.scheduler_reader


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 2: Agregar en el SIDEBAR (muestra status del scheduler)
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.header("🤖 Scheduler Autónomo")

status = scheduler_reader.get_scheduler_status()
st.sidebar.markdown(status["message"])

if status["last_scan"]:
    st.sidebar.caption(f"Último scan: {status['last_scan'][:19]}")
    st.sidebar.caption(f"Alertas hoy: {status.get('alerts_today', 0)}")

if not status["running"]:
    st.sidebar.code("python scheduler.py", language="bash")


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 3: Reemplaza el Scanner Multi-Activo (Tab 5) con esto
# ─────────────────────────────────────────────────────────────────────────────

with tab5:
    st.header("🔍 Scanner Multi-Activo — Modo Autónomo")

    status = scheduler_reader.get_scheduler_status()

    # ── Status del scheduler ──────────────────────────────────────────────
    if status["running"]:
        age = status.get("data_age_minutes", 0)
        st.success(f"✅ Scheduler activo — datos de hace {age:.0f} min")
    else:
        st.warning(
            "⚠️ El scheduler no está corriendo. "
            "Los datos mostrados pueden estar desactualizados.\n\n"
            "Para activarlo: `python scheduler.py`"
        )

    # ── Tabla de resultados ───────────────────────────────────────────────
    df_scan = scheduler_reader.get_latest_scan(max_age_minutes=60)

    if not df_scan.empty:
        # Colorear recomendación
        def color_rec(val):
            if "COMPRA FUERTE" in str(val): return "background-color: #1b5e20; color: white"
            if "COMPRA" in str(val):        return "background-color: #2e7d32; color: white"
            if "VENTA FUERTE" in str(val):  return "background-color: #b71c1c; color: white"
            if "VENTA" in str(val):         return "background-color: #c62828; color: white"
            return "background-color: #424242; color: white"

        display_cols = [
            "ticker", "price", "change_pct", "score",
            "recommendation", "rsi", "adx", "rvol"
        ]
        df_display = df_scan[display_cols].copy()
        df_display.columns = [
            "Ticker", "Precio", "Cambio %", "Score",
            "Señal", "RSI", "ADX", "RVOL"
        ]

        styled = df_display.style.applymap(color_rec, subset=["Señal"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # ── Top picks rápido ──────────────────────────────────────────────
        st.markdown("---")
        col_buy, col_sell = st.columns(2)

        with col_buy:
            st.subheader("🟢 Top Compras")
            top_buy = scheduler_reader.get_top_picks(n=3, direction="buy")
            if not top_buy.empty:
                for _, row in top_buy.iterrows():
                    st.metric(
                        label=row["ticker"],
                        value=f"${row['price']:.2f}",
                        delta=f"Score: {row['score']}"
                    )

        with col_sell:
            st.subheader("🔴 Top Ventas")
            top_sell = scheduler_reader.get_top_picks(n=3, direction="sell")
            if not top_sell.empty:
                for _, row in top_sell.iterrows():
                    st.metric(
                        label=row["ticker"],
                        value=f"${row['price']:.2f}",
                        delta=f"Score: {row['score']}"
                    )

    else:
        st.info(
            "📭 No hay resultados recientes.\n\n"
            "Asegúrate de que el scheduler esté corriendo: `python scheduler.py`"
        )

    # ── Historial de alertas ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔔 Alertas Recientes (últimas 24h)")
    df_alerts = scheduler_reader.get_recent_alerts(hours=24)
    if not df_alerts.empty:
        st.dataframe(
            df_alerts[["timestamp", "ticker", "alert_type", "message"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.caption("Sin alertas en las últimas 24 horas.")


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 4: Auto-refresh de Streamlit (opcional)
# Muestra datos frescos del scheduler cada N segundos
# sin recalcular nada — solo lee la BD.
# ─────────────────────────────────────────────────────────────────────────────

import time as _time

st.sidebar.markdown("---")
auto_refresh = st.sidebar.toggle("🔄 Auto-refresh (30s)", value=False)

if auto_refresh:
    _time.sleep(30)
    st.rerun()
