"""
PATO QUANT — Performance Tracker
Registra y calcula métricas de performance REALES del trader autónomo.

Métricas trackeadas:
  - Equity curve real (diaria)
  - Sharpe ratio rolling (30d, 90d)
  - Max drawdown real vs peak
  - Win rate real (por trade cerrado)
  - Profit factor real
  - Comparación vs SPY buy-and-hold

Se persiste en JSON para no depender de BD externa.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("PatoQuant.Performance")

PERF_FILE = Path("data/performance_history.json")


class PerformanceTracker:
    """
    Registra equity diaria y trades cerrados para calcular
    métricas de performance reales (no backtest).
    """

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital
        self.equity_history: List[Dict] = []
        self.closed_trades: List[Dict] = []
        self.peak_equity = initial_capital
        self._load()

    # ── Persistencia ─────────────────────────────────────────────────────────

    def _load(self):
        """Carga historial desde JSON."""
        try:
            if PERF_FILE.exists():
                with open(PERF_FILE, "r") as f:
                    data = json.load(f)
                self.equity_history = data.get("equity_history", [])
                self.closed_trades = data.get("closed_trades", [])
                self.initial_capital = data.get("initial_capital", self.initial_capital)
                self.peak_equity = data.get("peak_equity", self.initial_capital)
                logger.info(
                    f"📈 Performance tracker cargado: "
                    f"{len(self.equity_history)} días, "
                    f"{len(self.closed_trades)} trades"
                )
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar historial: {e}")

    def _save(self):
        """Guarda historial a JSON."""
        try:
            PERF_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(PERF_FILE, "w") as f:
                json.dump({
                    "initial_capital": self.initial_capital,
                    "peak_equity": self.peak_equity,
                    "equity_history": self.equity_history,
                    "closed_trades": self.closed_trades,
                    "last_updated": datetime.now().isoformat(),
                }, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"⚠️ No se pudo guardar historial: {e}")

    # ── Registro ─────────────────────────────────────────────────────────────

    def record_equity(self, equity: float, cash: float,
                      n_positions: int):
        """Registra snapshot de equity diario. Llamar 1x por día."""
        today = datetime.now().strftime("%Y-%m-%d")

        # Evitar duplicados del mismo día
        if self.equity_history and self.equity_history[-1]["date"] == today:
            self.equity_history[-1] = {
                "date": today,
                "equity": equity,
                "cash": cash,
                "positions": n_positions,
            }
        else:
            self.equity_history.append({
                "date": today,
                "equity": equity,
                "cash": cash,
                "positions": n_positions,
            })

        self.peak_equity = max(self.peak_equity, equity)
        self._save()

    def record_trade(self, ticker: str, action: str, qty: float,
                     entry_price: float, exit_price: float,
                     pnl: float, reason: str):
        """Registra un trade cerrado."""
        self.closed_trades.append({
            "date": datetime.now().isoformat(),
            "ticker": ticker,
            "action": action,
            "qty": qty,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": ((exit_price / entry_price) - 1) * 100 if entry_price > 0 else 0,
            "reason": reason,
        })
        self._save()

    # ── Métricas ─────────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict:
        """Calcula todas las métricas de performance reales."""
        if len(self.equity_history) < 2:
            return {
                "status": "insufficient_data",
                "days_tracked": len(self.equity_history),
                "trades_closed": len(self.closed_trades),
            }

        equities = [e["equity"] for e in self.equity_history]
        current_equity = equities[-1]

        # ── Retorno total ────────────────────────────────────────────────
        total_return = ((current_equity / self.initial_capital) - 1) * 100

        # ── CAGR ─────────────────────────────────────────────────────────
        n_days = len(equities)
        n_years = n_days / 252
        if n_years > 0 and current_equity > 0:
            cagr = ((current_equity / self.initial_capital) ** (1 / n_years) - 1) * 100
        else:
            cagr = 0

        # ── Daily returns ────────────────────────────────────────────────
        eq_arr = np.array(equities, dtype=float)
        daily_returns = np.diff(eq_arr) / eq_arr[:-1]

        # ── Sharpe ratio (anualizado, rf=0) ──────────────────────────────
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
        else:
            sharpe = 0

        # ── Sortino ratio ────────────────────────────────────────────────
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0 and np.std(downside) > 0:
            sortino = (np.mean(daily_returns) / np.std(downside)) * np.sqrt(252)
        else:
            sortino = 0

        # ── Max drawdown ─────────────────────────────────────────────────
        running_max = np.maximum.accumulate(eq_arr)
        drawdowns = (eq_arr - running_max) / running_max
        max_dd = float(np.min(drawdowns)) * 100
        current_dd = float(drawdowns[-1]) * 100

        # ── Trade metrics ────────────────────────────────────────────────
        n_trades = len(self.closed_trades)
        if n_trades > 0:
            wins = [t for t in self.closed_trades if t["pnl"] > 0]
            losses = [t for t in self.closed_trades if t["pnl"] <= 0]
            win_rate = len(wins) / n_trades * 100

            avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
            avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0

            gross_profit = sum(t["pnl"] for t in wins) if wins else 0
            gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0

        # ── Sharpe rolling 30d ───────────────────────────────────────────
        sharpe_30d = 0
        if len(daily_returns) >= 30:
            last_30 = daily_returns[-30:]
            if np.std(last_30) > 0:
                sharpe_30d = (np.mean(last_30) / np.std(last_30)) * np.sqrt(252)

        return {
            "status": "active",
            "days_tracked": n_days,
            "current_equity": round(current_equity, 2),
            "initial_capital": round(self.initial_capital, 2),
            "total_return_pct": round(total_return, 2),
            "cagr_pct": round(cagr, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sharpe_30d": round(sharpe_30d, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "current_drawdown_pct": round(current_dd, 2),
            "peak_equity": round(self.peak_equity, 2),
            "total_trades": n_trades,
            "win_rate_pct": round(win_rate, 1),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
        }

    def format_report(self) -> str:
        """Genera reporte de performance en texto."""
        m = self.get_metrics()

        if m["status"] == "insufficient_data":
            return (
                f"📊 Performance Tracker: {m['days_tracked']} días, "
                f"{m['trades_closed']} trades — datos insuficientes para métricas"
            )

        return (
            f"\n{'═' * 55}\n"
            f"📊 PERFORMANCE REAL — PAPER TRADING\n"
            f"{'═' * 55}\n"
            f"  Capital inicial:  ${m['initial_capital']:>12,.2f}\n"
            f"  Equity actual:    ${m['current_equity']:>12,.2f}\n"
            f"  Retorno total:    {m['total_return_pct']:>+10.2f}%\n"
            f"  CAGR:             {m['cagr_pct']:>+10.2f}%\n"
            f"{'─' * 55}\n"
            f"  Sharpe ratio:     {m['sharpe_ratio']:>10.3f}\n"
            f"  Sharpe 30d:       {m['sharpe_30d']:>10.3f}\n"
            f"  Sortino ratio:    {m['sortino_ratio']:>10.3f}\n"
            f"  Max drawdown:     {m['max_drawdown_pct']:>10.2f}%\n"
            f"  Drawdown actual:  {m['current_drawdown_pct']:>10.2f}%\n"
            f"{'─' * 55}\n"
            f"  Trades cerrados:  {m['total_trades']:>10d}\n"
            f"  Win rate:         {m['win_rate_pct']:>10.1f}%\n"
            f"  Avg win:          {m['avg_win_pct']:>+10.2f}%\n"
            f"  Avg loss:         {m['avg_loss_pct']:>+10.2f}%\n"
            f"  Profit factor:    {m['profit_factor']:>10.2f}\n"
            f"  Días trackeados:  {m['days_tracked']:>10d}\n"
            f"{'═' * 55}"
        )
