"""
PATO QUANT — CONFIG
Lee todas las credenciales desde variables de entorno.
Este archivo SÍ se puede subir a GitHub — no contiene ninguna credencial real.

Las credenciales van en:
  - Railway: panel de Variables
  - Local:   archivo .env (nunca subir a GitHub)
"""

import os

# ============= ACTIVOS =============
PORTFOLIO_CONFIG = {
    'stocks': ['MSFT', 'AMZN', 'TSLA', 'V', 'DIS', 'COIN', 'NVDA', 'AAPL', 'GOOGL'],
    'crypto': ['BTC-USD'],
    'etfs':   [],
    'forex':  ['GC=F']
}

# ============= INDICADORES TÉCNICOS =============
TECHNICAL_INDICATORS = {
    'sma_short':      20,
    'sma_long':       50,
    'ema_period':     12,
    'rsi_period':     14,
    'rsi_oversold':   30,
    'rsi_overbought': 70,
    'macd_fast':      12,
    'macd_slow':      26,
    'macd_signal':    9,
    'bb_period':      20,
    'bb_std':         2
}

# ============= API KEYS =============
API_CONFIG = {
    'groq_api_key':       os.getenv('GROQ_API_KEY', ''),
    'gemini_api_key':     os.getenv('GEMINI_API_KEY', ''),
    'alpha_vantage':      os.getenv('ALPHA_VANTAGE_KEY', ''),
    'coinmarketcap':      os.getenv('COINMARKETCAP_KEY', ''),
    'binance': {
        'api_key':    os.getenv('BINANCE_API_KEY', ''),
        'api_secret': os.getenv('BINANCE_API_SECRET', ''),
    }
}

# ============= ALPACA =============
ALPACA = {
    'api_key':      os.getenv('ALPACA_API_KEY', ''),
    'api_secret':   os.getenv('ALPACA_API_SECRET', ''),
    'paper_trading': True
}

# ============= NOTIFICACIONES =============
NOTIFICATIONS = {
    'email': {
        'enabled':     bool(os.getenv('EMAIL_SENDER')),
        'smtp_server': 'smtp.gmail.com',
        'smtp_port':   587,
        'sender':      os.getenv('EMAIL_SENDER', ''),
        'password':    os.getenv('EMAIL_PASSWORD', ''),
        'recipient':   os.getenv('EMAIL_RECIPIENT', ''),
    },
    'telegram': {
        'enabled':   bool(os.getenv('TELEGRAM_TOKEN')),
        'bot_token': os.getenv('TELEGRAM_TOKEN', ''),
        'chat_id':   os.getenv('TELEGRAM_CHAT_ID', ''),
    },
    'console': {
        'enabled': True
    }
}
