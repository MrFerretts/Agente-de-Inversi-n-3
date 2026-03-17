"""
╔══════════════════════════════════════════════════════════════════╗
║         PATO QUANT — MOTOR DE BACKTESTING                       ║
║                                                                  ║
║  Testea la estrategia sobre 5 años de datos históricos.         ║
║  Compara: Scoring Técnico vs ML vs Buy & Hold                   ║
║                                                                  ║
║  Cómo correr:                                                    ║
║    python backtest.py                                            ║
║    python backtest.py --tickers AAPL MSFT NVDA --years 5        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import sys
import argparse
import warnings
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("Backtest")

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from core.state_manager import DataProcessor
from technical_analysis import TechnicalAnalyzer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "TSLA", "COIN", "OKTA", "USO", "V",
]

BT_CONFIG = {
    "initial_capital":    10_000.0,
    "position_pct":       0.10,       # 10% del capital por trade
    "commission_pct":     0.001,      # 0.1% comisión por lado
    "slippage_pct":       0.0005,     # 0.05% slippage por ejecución
    "stop_loss_atr_mult": 2.0,
    "take_profit_atr_mult": 4.0,

    # Filtros de entrada técnica
    "min_score_tech":     40,         # Score mínimo para señal de compra
    "min_score_sell":    -30,         # Score máximo para señal de venta

    # Filtros ML
    "min_ml_prob":        0.60,       # Probabilidad mínima de subida

    # Warm-up: días necesarios para calcular todos los indicadores
    "warmup_days":        60,
}

# ─────────────────────────────────────────────────────────────────────────────
# DESCARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def download_data(tickers: List[str], years: int = 5) -> Dict[str, pd.DataFrame]:
    """Descarga datos históricos de yfinance."""
    end   = datetime.now()
    start = end - timedelta(days=years * 365 + 90)  # +90 para warm-up

    logger.info(f"Descargando {len(tickers)} tickers ({years} años)...")
    data = {}

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df is not None and len(df) >= BT_CONFIG["warmup_days"] + 30:
                # Fix yfinance MultiIndex columns (v1.2+)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df["Returns"] = df["Close"].pct_change()
                data[ticker]  = df
                logger.info(f"  ✅ {ticker}: {len(df)} días")
            else:
                logger.warning(f"  ⚠️  {ticker}: datos insuficientes")
        except Exception as e:
            logger.error(f"  ❌ {ticker}: {e}")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR DE SIMULACIÓN
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Simula trades día a día sobre datos históricos.
    Aplica comisiones, slippage y gestión de riesgo idéntica al trader real.
    """

    def __init__(self, initial_capital: float = 10_000.0):
        self.initial_capital = initial_capital
        self.analyzer        = TechnicalAnalyzer({})

    def _apply_transaction_cost(self, price: float, side: str) -> float:
        """Aplica slippage y comisión al precio de ejecución."""
        slip = price * BT_CONFIG["slippage_pct"]
        comm = price * BT_CONFIG["commission_pct"]
        if side == "buy":
            return price + slip + comm
        else:
            return price - slip - comm

    def run_technical(self, ticker: str,
                       df_full: pd.DataFrame) -> pd.DataFrame:
        """
        Backtesting de la estrategia de scoring técnico.
        Señal de compra: score >= min_score_tech
        Señal de venta:  score <= min_score_sell  O  stop/TP activado
        """
        # Calcular indicadores completos
        df = DataProcessor.prepare_full_analysis(df_full, self.analyzer)
        df = df.dropna().copy()

        # Calcular score diario
        scores = []
        for i in range(len(df)):
            try:
                window = df.iloc[max(0, i - BT_CONFIG["warmup_days"]):i + 1]
                if len(window) < 30:
                    scores.append(0)
                    continue
                analysis = self.analyzer.analyze_asset(window, ticker)
                scores.append(analysis.get("signals", {}).get("score", 0) if analysis else 0)
            except Exception:
                scores.append(0)

        df["score"] = scores

        # Simular trades
        capital    = self.initial_capital
        position   = 0.0     # shares held
        entry_price = 0.0
        stop_loss  = 0.0
        take_profit = 0.0
        trades     = []
        equity     = []

        for i, (idx, row) in enumerate(df.iterrows()):
            price = float(row["Close"])
            score = float(row["score"])
            atr   = float(row.get("ATR", price * 0.02))

            current_value = capital + position * price
            equity.append({"date": idx, "equity": current_value,
                           "price": price, "score": score})

            # Gestión de posición abierta
            if position > 0:
                # Stop loss
                if price <= stop_loss:
                    sell_price = self._apply_transaction_cost(price, "sell")
                    pnl = (sell_price - entry_price) * position
                    capital += position * sell_price
                    trades.append({
                        "ticker": ticker, "entry_date": entry_date,
                        "exit_date": idx, "entry": entry_price,
                        "exit": sell_price, "qty": position,
                        "pnl": pnl, "reason": "stop_loss",
                        "pnl_pct": (sell_price / entry_price - 1) * 100,
                    })
                    position = 0.0

                # Take profit
                elif price >= take_profit:
                    sell_price = self._apply_transaction_cost(price, "sell")
                    pnl = (sell_price - entry_price) * position
                    capital += position * sell_price
                    trades.append({
                        "ticker": ticker, "entry_date": entry_date,
                        "exit_date": idx, "entry": entry_price,
                        "exit": sell_price, "qty": position,
                        "pnl": pnl, "reason": "take_profit",
                        "pnl_pct": (sell_price / entry_price - 1) * 100,
                    })
                    position = 0.0

                # Señal bajista fuerte
                elif score <= BT_CONFIG["min_score_sell"]:
                    sell_price = self._apply_transaction_cost(price, "sell")
                    pnl = (sell_price - entry_price) * position
                    capital += position * sell_price
                    trades.append({
                        "ticker": ticker, "entry_date": entry_date,
                        "exit_date": idx, "entry": entry_price,
                        "exit": sell_price, "qty": position,
                        "pnl": pnl, "reason": "signal_sell",
                        "pnl_pct": (sell_price / entry_price - 1) * 100,
                    })
                    position = 0.0

            # Nueva entrada
            elif position == 0 and score >= BT_CONFIG["min_score_tech"]:
                buy_price = self._apply_transaction_cost(price, "buy")
                trade_capital = capital * BT_CONFIG["position_pct"]
                qty = trade_capital / buy_price

                if qty * buy_price <= capital:
                    position    = qty
                    capital    -= qty * buy_price
                    entry_price = buy_price
                    entry_date  = idx
                    stop_loss   = buy_price - atr * BT_CONFIG["stop_loss_atr_mult"]
                    take_profit = buy_price + atr * BT_CONFIG["take_profit_atr_mult"]

        # Cerrar posición al final si queda abierta
        if position > 0:
            last_price = float(df["Close"].iloc[-1])
            sell_price = self._apply_transaction_cost(last_price, "sell")
            pnl = (sell_price - entry_price) * position
            capital += position * sell_price
            trades.append({
                "ticker": ticker, "entry_date": entry_date,
                "exit_date": df.index[-1], "entry": entry_price,
                "exit": sell_price, "qty": position,
                "pnl": pnl, "reason": "end_of_period",
                "pnl_pct": (sell_price / entry_price - 1) * 100,
            })

        equity_df = pd.DataFrame(equity).set_index("date")
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        return equity_df, trades_df

    def run_ml(self, ticker: str,
                df_full: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Backtesting Walk-Forward del modelo ML.

        En lugar de un solo split 60/40, divide los datos en ventanas
        rolling donde el modelo se re-entrena periódicamente con datos
        pasados y se testea en datos futuros que nunca ha visto.

        Ventanas:
          - Train: 252 días (1 año) de datos pasados
          - Test:  63 días (1 trimestre) futuros
          - El modelo se re-entrena cada 63 días (avanza la ventana)

        Esto replica exactamente cómo se usaría en producción:
        el modelo NUNCA ve datos futuros durante el entrenamiento.
        """
        try:
            from ml_model import AdvancedTradingMLModel
        except ImportError:
            logger.warning("ml_model.py no encontrado, omitiendo backtesting ML")
            return pd.DataFrame(), pd.DataFrame()

        df = DataProcessor.prepare_full_analysis(df_full, self.analyzer)
        df = df.dropna().copy()

        TRAIN_DAYS = 252   # 1 año de training
        TEST_DAYS  = 63    # 1 trimestre de test (re-entrena cada trimestre)
        MIN_TRAIN  = 150

        if len(df) < TRAIN_DAYS + TEST_DAYS:
            logger.warning(f"  {ticker}: datos insuficientes para walk-forward ML backtest")
            return pd.DataFrame(), pd.DataFrame()

        capital     = self.initial_capital
        position    = 0.0
        entry_price = 0.0
        stop_loss   = 0.0
        take_profit = 0.0
        trades      = []
        equity      = []
        n_windows   = 0

        # Walk-Forward: avanzar ventana de training+test
        start = TRAIN_DAYS
        while start + 30 < len(df):  # Al menos 30 días de test
            end = min(start + TEST_DAYS, len(df))
            df_train = df.iloc[:start]
            df_test  = df.iloc[start:end]

            if len(df_train) < MIN_TRAIN:
                start = end
                continue

            # Re-entrenar modelo con datos hasta 'start'
            try:
                model = AdvancedTradingMLModel(prediction_days=5, threshold=2.0)
                model.train(df_train, test_size=0.2)
                n_windows += 1
                logger.info(
                    f"  🔄 Walk-Forward ventana {n_windows}: "
                    f"train={len(df_train)}d, test={len(df_test)}d"
                )
            except Exception as e:
                logger.warning(f"  ⚠️  ML train falló ventana {n_windows}: {e}")
                start = end
                continue

            # Simular trades en ventana de test
            for i in range(30, len(df_test)):
                window = df_test.iloc[max(0, i - 60):i + 1]
                row    = df_test.iloc[i]
                price  = float(row["Close"])
                atr    = float(row.get("ATR", price * 0.02))
                idx    = df_test.index[i]

                current_value = capital + position * price
                equity.append({"date": idx, "equity": current_value, "price": price})

                try:
                    pred    = model.predict(window)
                    ml_prob = float(pred.get("probability_up", 0.5))
                except Exception:
                    ml_prob = 0.5

                if position > 0:
                    if price <= stop_loss:
                        sell_price = self._apply_transaction_cost(price, "sell")
                        pnl = (sell_price - entry_price) * position
                        capital += position * sell_price
                        trades.append({
                            "ticker": ticker, "entry_date": entry_date,
                            "exit_date": idx, "entry": entry_price,
                            "exit": sell_price, "qty": position,
                            "pnl": pnl, "reason": "stop_loss",
                            "pnl_pct": (sell_price / entry_price - 1) * 100,
                            "ml_prob": ml_prob, "wf_window": n_windows,
                        })
                        position = 0.0

                    elif price >= take_profit:
                        sell_price = self._apply_transaction_cost(price, "sell")
                        pnl = (sell_price - entry_price) * position
                        capital += position * sell_price
                        trades.append({
                            "ticker": ticker, "entry_date": entry_date,
                            "exit_date": idx, "entry": entry_price,
                            "exit": sell_price, "qty": position,
                            "pnl": pnl, "reason": "take_profit",
                            "pnl_pct": (sell_price / entry_price - 1) * 100,
                            "ml_prob": ml_prob, "wf_window": n_windows,
                        })
                        position = 0.0

                    elif ml_prob < 0.35:
                        sell_price = self._apply_transaction_cost(price, "sell")
                        pnl = (sell_price - entry_price) * position
                        capital += position * sell_price
                        trades.append({
                            "ticker": ticker, "entry_date": entry_date,
                            "exit_date": idx, "entry": entry_price,
                            "exit": sell_price, "qty": position,
                            "pnl": pnl, "reason": "ml_sell",
                            "pnl_pct": (sell_price / entry_price - 1) * 100,
                            "ml_prob": ml_prob, "wf_window": n_windows,
                        })
                        position = 0.0

                elif position == 0 and ml_prob >= BT_CONFIG["min_ml_prob"]:
                    buy_price     = self._apply_transaction_cost(price, "buy")
                    trade_capital = capital * BT_CONFIG["position_pct"]
                    qty           = trade_capital / buy_price

                    if qty * buy_price <= capital:
                        position    = qty
                        capital    -= qty * buy_price
                        entry_price = buy_price
                        entry_date  = idx
                        stop_loss   = buy_price - atr * BT_CONFIG["stop_loss_atr_mult"]
                        take_profit = buy_price + atr * BT_CONFIG["take_profit_atr_mult"]

            # Avanzar ventana
            start = end

        # Cierre final
        if position > 0:
            last_price = float(df["Close"].iloc[-1])
            sell_price = self._apply_transaction_cost(last_price, "sell")
            pnl = (sell_price - entry_price) * position
            capital += position * sell_price
            trades.append({
                "ticker": ticker, "entry_date": entry_date,
                "exit_date": df.index[-1], "entry": entry_price,
                "exit": sell_price, "qty": position,
                "pnl": pnl, "reason": "end_of_period",
                "pnl_pct": (sell_price / entry_price - 1) * 100,
                "ml_prob": 0.5, "wf_window": n_windows,
            })

        logger.info(f"  ✅ Walk-Forward completado: {n_windows} ventanas de re-entrenamiento")

        equity_df = pd.DataFrame(equity).set_index("date") if equity else pd.DataFrame()
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        return equity_df, trades_df


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(equity_df: pd.DataFrame,
                    trades_df: pd.DataFrame,
                    initial_capital: float) -> Dict:
    """Calcula todas las métricas de performance."""
    if equity_df.empty:
        return {}

    eq = equity_df["equity"].values
    final_equity = eq[-1]

    # ── Retorno total ─────────────────────────────────────────────────────────
    total_return_pct = (final_equity / initial_capital - 1) * 100

    # ── CAGR ─────────────────────────────────────────────────────────────────
    n_years = len(eq) / 252
    cagr = ((final_equity / initial_capital) ** (1 / max(n_years, 0.1)) - 1) * 100

    # ── Sharpe ratio (anualizado, rf=0) ───────────────────────────────────────
    daily_returns = pd.Series(eq).pct_change().dropna()
    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-9)) * np.sqrt(252)

    # ── Sortino ratio ─────────────────────────────────────────────────────────
    downside = daily_returns[daily_returns < 0].std()
    sortino  = (daily_returns.mean() / (downside + 1e-9)) * np.sqrt(252)

    # ── Max Drawdown ──────────────────────────────────────────────────────────
    rolling_max = pd.Series(eq).cummax()
    drawdown    = (pd.Series(eq) - rolling_max) / rolling_max
    max_dd      = drawdown.min() * 100

    # ── Calmar ratio ─────────────────────────────────────────────────────────
    calmar = cagr / abs(max_dd + 1e-9)

    # ── Métricas de trades ────────────────────────────────────────────────────
    if not trades_df.empty and "pnl" in trades_df.columns:
        wins     = trades_df[trades_df["pnl"] > 0]
        losses   = trades_df[trades_df["pnl"] <= 0]
        n_trades = len(trades_df)
        win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0

        avg_win  = wins["pnl_pct"].mean()   if not wins.empty   else 0
        avg_loss = losses["pnl_pct"].mean() if not losses.empty else 0
        rr_ratio = abs(avg_win / avg_loss)  if avg_loss != 0    else 0

        # Profit factor
        gross_profit = wins["pnl"].sum()   if not wins.empty   else 0
        gross_loss   = abs(losses["pnl"].sum()) if not losses.empty else 1
        profit_factor = gross_profit / gross_loss

        # Duración media de trade
        if "entry_date" in trades_df.columns and "exit_date" in trades_df.columns:
            trades_df["duration"] = (
                pd.to_datetime(trades_df["exit_date"]) -
                pd.to_datetime(trades_df["entry_date"])
            ).dt.days
            avg_duration = trades_df["duration"].mean()
        else:
            avg_duration = 0

        exit_reasons = trades_df["reason"].value_counts().to_dict() if "reason" in trades_df.columns else {}
    else:
        n_trades = win_rate = avg_win = avg_loss = 0
        rr_ratio = profit_factor = avg_duration = 0
        exit_reasons = {}

    return {
        "total_return_pct": round(total_return_pct, 2),
        "cagr":             round(cagr, 2),
        "sharpe":           round(sharpe, 3),
        "sortino":          round(sortino, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "calmar":           round(calmar, 3),
        "n_trades":         n_trades,
        "win_rate":         round(win_rate, 1),
        "avg_win_pct":      round(avg_win, 2),
        "avg_loss_pct":     round(avg_loss, 2),
        "rr_ratio":         round(rr_ratio, 2),
        "profit_factor":    round(profit_factor, 2),
        "avg_duration_days": round(avg_duration, 1),
        "final_equity":     round(final_equity, 2),
        "exit_reasons":     exit_reasons,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORTE VISUAL
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(results: Dict, output_path: str = "backtest_report.png"):
    """
    Genera el reporte visual completo con todas las métricas.
    """
    fig = plt.figure(figsize=(20, 24), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(5, 3, figure=fig,
                            hspace=0.45, wspace=0.35,
                            top=0.95, bottom=0.04,
                            left=0.06, right=0.97)

    DARK   = "#0d1117"
    PANEL  = "#161b22"
    BORDER = "#30363d"
    GREEN  = "#3fb950"
    RED    = "#f85149"
    BLUE   = "#58a6ff"
    AMBER  = "#d29922"
    PURPLE = "#bc8cff"
    GRAY   = "#8b949e"
    WHITE  = "#e6edf3"

    def panel_bg(ax):
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
            spine.set_linewidth(0.5)
        ax.tick_params(colors=GRAY, labelsize=8)
        ax.xaxis.label.set_color(GRAY)
        ax.yaxis.label.set_color(GRAY)

    strategies = {
        "Technical": {"color": BLUE,   "label": "Scoring Técnico"},
        "ML":        {"color": PURPLE, "label": "Machine Learning"},
        "BuyHold":   {"color": AMBER,  "label": "Buy & Hold"},
    }

    # ── TÍTULO ────────────────────────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor(PANEL)
    for spine in ax_title.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.5)
    ax_title.text(0.5, 0.65, "🦆 Pato Quant — Backtest Report",
                  ha="center", va="center", color=WHITE,
                  fontsize=22, fontweight="bold", transform=ax_title.transAxes)
    ax_title.text(0.5, 0.25,
                  f"Capital inicial: ${BT_CONFIG['initial_capital']:,.0f}  |  "
                  f"Comisión: {BT_CONFIG['commission_pct']*100:.1f}%  |  "
                  f"Slippage: {BT_CONFIG['slippage_pct']*100:.2f}%  |  "
                  f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                  ha="center", va="center", color=GRAY,
                  fontsize=11, transform=ax_title.transAxes)
    ax_title.axis("off")

    # ── EQUITY CURVES ─────────────────────────────────────────────────────────
    ax_eq = fig.add_subplot(gs[1, :])
    panel_bg(ax_eq)
    ax_eq.set_title("Equity Curve — Capital Acumulado", color=WHITE,
                    fontsize=13, pad=10)

    has_data = False
    for strat_key, info in strategies.items():
        eq_combined = []
        for ticker_results in results.values():
            eq_df = ticker_results.get(f"equity_{strat_key.lower()}")
            if eq_df is not None and not eq_df.empty:
                eq_combined.append(eq_df["equity"])

        if eq_combined:
            combined = pd.concat(eq_combined).groupby(level=0).mean()
            combined = combined.sort_index()
            norm     = combined / combined.iloc[0] * BT_CONFIG["initial_capital"]
            ax_eq.plot(norm.index, norm.values,
                       color=info["color"], linewidth=1.5,
                       label=info["label"], alpha=0.9)
            ax_eq.fill_between(norm.index, BT_CONFIG["initial_capital"],
                                norm.values,
                                alpha=0.06, color=info["color"])
            has_data = True

    if has_data:
        ax_eq.axhline(y=BT_CONFIG["initial_capital"], color=GRAY,
                      linewidth=0.8, linestyle="--", alpha=0.5)
        ax_eq.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax_eq.legend(loc="upper left", fontsize=9,
                     facecolor=DARK, edgecolor=BORDER,
                     labelcolor=WHITE)
        ax_eq.grid(True, alpha=0.15, color=BORDER)

    # ── MÉTRICAS RESUMEN por estrategia ───────────────────────────────────────
    metric_keys = ["total_return_pct", "cagr", "sharpe", "max_drawdown_pct",
                   "win_rate", "profit_factor", "n_trades", "rr_ratio"]
    metric_labels = ["Retorno total", "CAGR", "Sharpe ratio", "Max Drawdown",
                     "Win rate", "Profit factor", "Nº trades", "R:R ratio"]

    strat_list = list(strategies.keys())
    for col_idx, strat_key in enumerate(strat_list):
        ax_m = fig.add_subplot(gs[2, col_idx])
        panel_bg(ax_m)
        info = strategies[strat_key]
        ax_m.set_title(info["label"], color=info["color"], fontsize=12, pad=8)
        ax_m.axis("off")

        # Agregar métricas de todos los tickers
        agg_metrics = _aggregate_metrics(results, strat_key.lower())

        y = 0.92
        for mk, ml in zip(metric_keys, metric_labels):
            val = agg_metrics.get(mk, "N/A")

            # Colorear según positivo/negativo
            if mk == "total_return_pct":
                c = GREEN if (isinstance(val, (int,float)) and val > 0) else RED
                val_str = f"{val:+.1f}%" if isinstance(val, (int,float)) else str(val)
            elif mk == "cagr":
                c = GREEN if (isinstance(val, (int,float)) and val > 0) else RED
                val_str = f"{val:+.1f}%" if isinstance(val, (int,float)) else str(val)
            elif mk == "sharpe":
                c = GREEN if (isinstance(val, (int,float)) and val > 1) else (
                    AMBER if (isinstance(val, (int,float)) and val > 0) else RED)
                val_str = f"{val:.2f}" if isinstance(val, (int,float)) else str(val)
            elif mk == "max_drawdown_pct":
                c = GREEN if (isinstance(val, (int,float)) and val > -10) else (
                    AMBER if (isinstance(val, (int,float)) and val > -20) else RED)
                val_str = f"{val:.1f}%" if isinstance(val, (int,float)) else str(val)
            elif mk == "win_rate":
                c = GREEN if (isinstance(val, (int,float)) and val > 50) else (
                    AMBER if (isinstance(val, (int,float)) and val > 40) else RED)
                val_str = f"{val:.1f}%" if isinstance(val, (int,float)) else str(val)
            elif mk == "profit_factor":
                c = GREEN if (isinstance(val, (int,float)) and val > 1.5) else (
                    AMBER if (isinstance(val, (int,float)) and val > 1) else RED)
                val_str = f"{val:.2f}" if isinstance(val, (int,float)) else str(val)
            else:
                c = WHITE
                val_str = f"{val:.1f}" if isinstance(val, (int,float)) else str(val)

            ax_m.text(0.05, y, ml, color=GRAY,    fontsize=9,  transform=ax_m.transAxes)
            ax_m.text(0.95, y, val_str, color=c, fontsize=10,
                      fontweight="bold", ha="right", transform=ax_m.transAxes)
            ax_m.axhline(y=(y - 0.045) * ax_m.get_ylim()[1] if ax_m.get_ylim()[1] != 0 else 0,
                         color=BORDER, linewidth=0.3, alpha=0.5)
            y -= 0.11

    # ── DISTRIBUCIÓN DE PnL por trade ─────────────────────────────────────────
    for col_idx, strat_key in enumerate(strat_list):
        ax_hist = fig.add_subplot(gs[3, col_idx])
        panel_bg(ax_hist)
        info = strategies[strat_key]
        ax_hist.set_title(f"Distribución PnL — {info['label']}",
                          color=WHITE, fontsize=10, pad=6)

        all_trades = _get_all_trades(results, strat_key.lower())
        if all_trades is not None and not all_trades.empty and "pnl_pct" in all_trades.columns:
            pnl = all_trades["pnl_pct"].dropna()
            positive = pnl[pnl >= 0]
            negative = pnl[pnl < 0]

            if not positive.empty:
                ax_hist.hist(positive, bins=20, color=GREEN, alpha=0.7,
                             edgecolor=DARK, linewidth=0.3)
            if not negative.empty:
                ax_hist.hist(negative, bins=20, color=RED, alpha=0.7,
                             edgecolor=DARK, linewidth=0.3)

            ax_hist.axvline(x=0, color=GRAY, linewidth=0.8, linestyle="--")
            ax_hist.axvline(x=pnl.mean(), color=info["color"],
                            linewidth=1.2, linestyle="-",
                            label=f"Media: {pnl.mean():.2f}%")
            ax_hist.legend(fontsize=8, facecolor=DARK,
                           edgecolor=BORDER, labelcolor=WHITE)
            ax_hist.set_xlabel("PnL por trade (%)", color=GRAY, fontsize=8)
            ax_hist.grid(True, alpha=0.1, color=BORDER)
        else:
            ax_hist.text(0.5, 0.5, "Sin trades", ha="center", va="center",
                         color=GRAY, fontsize=10, transform=ax_hist.transAxes)

    # ── TABLA COMPARATIVA FINAL ───────────────────────────────────────────────
    ax_table = fig.add_subplot(gs[4, :])
    panel_bg(ax_table)
    ax_table.set_title("Comparativa por Ticker", color=WHITE,
                        fontsize=12, pad=8)
    ax_table.axis("off")

    col_headers = ["Ticker", "Retorno Técnico", "Retorno ML",
                   "Buy & Hold", "Mejor", "Sharpe Técnico", "Sharpe ML"]
    rows = []
    for ticker in sorted(results.keys()):
        tr = results[ticker]
        ret_tech = _get_metric(tr, "technical", "total_return_pct")
        ret_ml   = _get_metric(tr, "ml",        "total_return_pct")
        ret_bh   = _get_metric(tr, "buyhold",   "total_return_pct")
        sh_tech  = _get_metric(tr, "technical", "sharpe")
        sh_ml    = _get_metric(tr, "ml",        "sharpe")

        vals  = {"Tech": ret_tech, "ML": ret_ml, "BH": ret_bh}
        best  = max(vals, key=lambda k: vals[k] if isinstance(vals[k], (int,float)) else -999)

        rows.append([
            ticker,
            f"{ret_tech:+.1f}%" if isinstance(ret_tech, float) else "N/A",
            f"{ret_ml:+.1f}%"   if isinstance(ret_ml,   float) else "N/A",
            f"{ret_bh:+.1f}%"   if isinstance(ret_bh,   float) else "N/A",
            best,
            f"{sh_tech:.2f}"    if isinstance(sh_tech, float) else "N/A",
            f"{sh_ml:.2f}"      if isinstance(sh_ml,   float) else "N/A",
        ])

    if rows:
        tbl = ax_table.table(
            cellText=rows,
            colLabels=col_headers,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.6)

        for (row, col), cell in tbl.get_celld().items():
            cell.set_facecolor(PANEL if row % 2 == 0 else "#1c2128")
            cell.set_edgecolor(BORDER)
            cell.set_linewidth(0.5)

            if row == 0:
                cell.set_facecolor("#21262d")
                cell.set_text_props(color=WHITE, fontweight="bold")
            else:
                text = cell.get_text().get_text()
                if "+" in text:
                    cell.set_text_props(color=GREEN)
                elif text.startswith("-"):
                    cell.set_text_props(color=RED)
                elif col == 4:  # Columna "Mejor"
                    color_map = {"Tech": BLUE, "ML": PURPLE, "BH": AMBER}
                    cell.set_text_props(
                        color=color_map.get(text, WHITE), fontweight="bold"
                    )
                else:
                    cell.set_text_props(color=GRAY)

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=DARK, edgecolor="none")
    plt.close()
    logger.info(f"✅ Reporte guardado: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _buyhold_equity(df: pd.DataFrame,
                     initial_capital: float) -> pd.DataFrame:
    """Calcula la equity curve de Buy & Hold."""
    df = df.dropna(subset=["Close"]).copy()
    norm  = df["Close"] / df["Close"].iloc[0]
    eq    = norm * initial_capital
    return pd.DataFrame({"equity": eq, "price": df["Close"]})


def _aggregate_metrics(results: Dict, strat: str) -> Dict:
    """Promedia métricas de todos los tickers para una estrategia."""
    all_m = [r.get(f"metrics_{strat}", {}) for r in results.values()
             if r.get(f"metrics_{strat}")]
    if not all_m:
        return {}
    keys = all_m[0].keys()
    out  = {}
    for k in keys:
        vals = [m[k] for m in all_m
                if isinstance(m.get(k), (int, float))]
        if vals:
            out[k] = round(np.mean(vals), 3)
    return out


def _get_all_trades(results: Dict, strat: str) -> Optional[pd.DataFrame]:
    dfs = [r.get(f"trades_{strat}") for r in results.values()
           if r.get(f"trades_{strat}") is not None]
    dfs = [d for d in dfs if not d.empty]
    return pd.concat(dfs) if dfs else None


def _get_metric(ticker_result: Dict, strat: str, metric: str):
    return ticker_result.get(f"metrics_{strat}", {}).get(metric, "N/A")


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(tickers: List[str], years: int = 5,
                 output: str = "backtest_report.png") -> Dict:
    """
    Corre el backtesting completo y genera el reporte.
    """
    logger.info("=" * 60)
    logger.info("🦆 PATO QUANT — BACKTESTING INICIANDO")
    logger.info(f"   Tickers: {', '.join(tickers)}")
    logger.info(f"   Período: {years} años")
    logger.info(f"   Capital: ${BT_CONFIG['initial_capital']:,.0f}")
    logger.info("=" * 60)

    data    = download_data(tickers, years)
    engine  = BacktestEngine(BT_CONFIG["initial_capital"])
    results = {}

    for ticker, df in data.items():
        logger.info(f"\n📊 Backtesting {ticker}...")
        ticker_result = {}

        # ── Técnico ───────────────────────────────────────────────────────────
        logger.info(f"  [1/3] Estrategia técnica...")
        try:
            eq_tech, tr_tech = engine.run_technical(ticker, df)
            ticker_result["equity_technical"] = eq_tech
            ticker_result["trades_technical"] = tr_tech
            ticker_result["metrics_technical"] = compute_metrics(
                eq_tech, tr_tech, BT_CONFIG["initial_capital"]
            )
            m = ticker_result["metrics_technical"]
            logger.info(
                f"  ✅ Retorno: {m.get('total_return_pct',0):+.1f}% | "
                f"Sharpe: {m.get('sharpe',0):.2f} | "
                f"Trades: {m.get('n_trades',0)} | "
                f"Win rate: {m.get('win_rate',0):.1f}%"
            )
        except Exception as e:
            logger.error(f"  ❌ Error técnico: {e}")

        # ── ML ────────────────────────────────────────────────────────────────
        logger.info(f"  [2/3] Estrategia ML...")
        try:
            eq_ml, tr_ml = engine.run_ml(ticker, df)
            ticker_result["equity_ml"] = eq_ml
            ticker_result["trades_ml"] = tr_ml
            ticker_result["metrics_ml"] = compute_metrics(
                eq_ml, tr_ml, BT_CONFIG["initial_capital"]
            )
            m = ticker_result["metrics_ml"]
            logger.info(
                f"  ✅ Retorno: {m.get('total_return_pct',0):+.1f}% | "
                f"Sharpe: {m.get('sharpe',0):.2f} | "
                f"Trades: {m.get('n_trades',0)} | "
                f"Win rate: {m.get('win_rate',0):.1f}%"
            )
        except Exception as e:
            logger.error(f"  ❌ Error ML: {e}")

        # ── Buy & Hold ────────────────────────────────────────────────────────
        logger.info(f"  [3/3] Buy & Hold benchmark...")
        try:
            eq_bh = _buyhold_equity(df, BT_CONFIG["initial_capital"])
            ticker_result["equity_buyhold"] = eq_bh
            ticker_result["trades_buyhold"] = pd.DataFrame()
            ticker_result["metrics_buyhold"] = compute_metrics(
                eq_bh, pd.DataFrame(), BT_CONFIG["initial_capital"]
            )
            m = ticker_result["metrics_buyhold"]
            logger.info(
                f"  ✅ Retorno B&H: {m.get('total_return_pct',0):+.1f}% | "
                f"Sharpe: {m.get('sharpe',0):.2f}"
            )
        except Exception as e:
            logger.error(f"  ❌ Error B&H: {e}")

        results[ticker] = ticker_result

    # ── Resumen global ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESUMEN GLOBAL")
    logger.info("=" * 60)
    for strat in ["technical", "ml", "buyhold"]:
        agg = _aggregate_metrics(results, strat)
        if agg:
            label = {"technical": "Técnico ", "ml": "ML      ",
                     "buyhold":   "Buy&Hold"}[strat]
            logger.info(
                f"  {label} → Retorno: {agg.get('total_return_pct',0):+.1f}% | "
                f"Sharpe: {agg.get('sharpe',0):.2f} | "
                f"MaxDD: {agg.get('max_drawdown_pct',0):.1f}% | "
                f"Win rate: {agg.get('win_rate',0):.1f}%"
            )

    # ── Generar reporte ───────────────────────────────────────────────────────
    logger.info(f"\n🖼️  Generando reporte visual...")
    report_path = generate_report(results, output)
    logger.info(f"\n✅ Backtesting completado. Reporte: {report_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pato Quant Backtester")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                        help="Lista de tickers a backtestear")
    parser.add_argument("--years",   type=int,  default=5,
                        help="Años de historial (default: 5)")
    parser.add_argument("--output",  default="backtest_report.png",
                        help="Ruta del reporte de salida")
    parser.add_argument("--capital", type=float, default=10_000.0,
                        help="Capital inicial (default: 10000)")
    args = parser.parse_args()

    BT_CONFIG["initial_capital"] = args.capital
    run_backtest(args.tickers, args.years, args.output)
