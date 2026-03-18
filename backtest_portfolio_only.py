"""
Test rápido: Solo corre la comparación Momentum vs Buy & Hold a nivel portafolio.
Sin backtests individuales (técnico/ML) para ir rápido.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backtest import (
    download_data, run_portfolio_momentum, run_portfolio_buyhold,
    BT_CONFIG, logger
)

TICKERS_25 = [
    # Tech giants
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    # Semis & cloud
    "AMD", "AVGO", "CRM", "ADBE", "ORCL", "NFLX",
    # Fintech / payments
    "V", "MA", "PYPL",
    # E-commerce / gig
    "SHOP", "UBER", "ABNB", "COIN",
    # Traditional
    "JPM", "COST", "UNH", "XOM",
]

def main():
    BT_CONFIG["initial_capital"] = 10_000.0

    for years in [3, 5]:
        logger.info("\n" + "=" * 70)
        logger.info(f"BACKTEST PORTAFOLIO: {len(TICKERS_25)} tickers, {years} AÑOS")
        logger.info("=" * 70)

        data = download_data(TICKERS_25, years)
        logger.info(f"Tickers válidos: {len(data)}")

        # Momentum rotation
        eq, tr, m = run_portfolio_momentum(
            data, BT_CONFIG["initial_capital"],
            rebalance_days=20, top_n=5,
        )
        if m:
            logger.info(
                f"  MOMENTUM:  Retorno {m.get('total_return_pct',0):+.1f}% | "
                f"CAGR {m.get('cagr',0):+.1f}% | "
                f"Sharpe {m.get('sharpe',0):.2f} | "
                f"MaxDD {m.get('max_drawdown_pct',0):.1f}% | "
                f"Trades {m.get('n_trades',0)} | "
                f"WR {m.get('win_rate',0):.0f}%"
            )

        # Benchmark
        eq_bh, m_bh = run_portfolio_buyhold(data, BT_CONFIG["initial_capital"])
        if m_bh:
            logger.info(
                f"  BUY&HOLD:  Retorno {m_bh.get('total_return_pct',0):+.1f}% | "
                f"CAGR {m_bh.get('cagr',0):+.1f}% | "
                f"Sharpe {m_bh.get('sharpe',0):.2f} | "
                f"MaxDD {m_bh.get('max_drawdown_pct',0):.1f}%"
            )

        if m and m_bh:
            diff = m['total_return_pct'] - m_bh['total_return_pct']
            emoji = "🏆" if diff > 0 else "📊"
            logger.info(f"  {emoji} Diferencia: {diff:+.1f}%")

if __name__ == "__main__":
    main()
