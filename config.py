"""
Configuration file for the Trading Algorithm AI.
Contains all configurable parameters for the trading system.
"""

# General Configuration
CONFIG = {
    "starting_capital": 1000,  # Starting capital in USD
    "risk_profile": "medium_to_high",  # Options: low, low_to_medium, medium, medium_to_high, high
    "target_markets": ["stocks", "crypto", "options"],  # Markets to trade
    "execution_mode": "signal_only",  # Options: signal_only, auto_trade
}

# Data Sources Configuration
DATA_SOURCES = {
    "stocks": {
        "provider": "polygon",
        "api_key": "NDwYagSyL0_hShJ9yiFUIgdcabZpBwRU",  # Polygon API key
        "base_url": "https://api.polygon.io",
        "endpoints": {
            "chart": "/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_}/{to_}",  # <-- fix here
            "insights": "/v3/reference/tickers/{ticker}",
            "holders": "/v3/reference/ownership/institutional",
            "analyst": None
        }
    },
    "crypto": {
        "provider": "polygon",  # Changed to polygon
        "api_key": "NDwYagSyL0_hShJ9yiFUIgdcabZpBwRU",  # Polygon API key
        "base_url": "https://api.polygon.io",
        "endpoints": {
            "chart": "/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_}/{to_}",  # <-- fix here
            "insights": "/v1/meta/cryptos/{ticker}",
            "historical": "/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_}/{to_}",
            "ohlc": "/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_}/{to_}"
        }
    },
    "options": {
        "provider": "polygon",
        "api_key": "NDwYagSyL0_hShJ9yiFUIgdcabZpBwRU",  # Polygon API key
        "base_url": "https://api.polygon.io",
        "endpoints": {
            "options_contracts": "/v3/reference/options/contracts",
            "options_aggregates": "/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_}/{to_}",
            "last_quote": "/v2/last/nbbo/{ticker}"
        }
    }
}

# Technical Indicators Configuration
INDICATORS = {
    "trend": {
        "sma": {"periods": [5, 13, 50]},  # Shorter periods for faster signals
        "ema": {"periods": [8, 13, 21]},  # More responsive EMAs
        "macd": {"fast_period": 8, "slow_period": 21, "signal_period": 5},  # Optimized for faster signals
        "parabolic_sar": {"acceleration": 0.025, "maximum": 0.2}
    },
    "momentum": {
        "rsi": {"period": 10, "overbought": 75, "oversold": 25},  # More aggressive thresholds
        "stochastic": {"k_period": 8, "d_period": 3, "overbought": 85, "oversold": 15},
        "cci": {"period": 15, "overbought": 120, "oversold": -120}
    },
    "volatility": {
        "bollinger_bands": {"period": 15, "std_dev": 2.5},
        "atr": {"period": 10}
    },
    "volume": {
        "obv": {},
        "volume_sma": {"period": 15},
        "money_flow_index": {"period": 14}
    }
}

# Options Trading Configuration
OPTIONS_CONFIG = {
    "strategy": "momentum",  # Options strategy type
    "min_volume": 100,      # Minimum option contract volume
    "min_open_interest": 500,  # Minimum open interest
    "delta_range": {
        "calls": [0.3, 0.7],  # Delta range for calls
        "puts": [-0.7, -0.3]  # Delta range for puts
    },
    "dte_range": [10, 45],    # Days to expiration range
    "position_sizing": {
        "max_position_size": 0.1,  # Maximum position size as fraction of portfolio
        "risk_per_trade": 0.02     # Maximum risk per trade
    }
}

# Risk Management Configuration
RISK_MANAGEMENT = {
    "max_position_size": 0.15,     # Reduced from 0.2
    "max_drawdown": 0.10,          # Reduced from 0.15
    "daily_loss_limit": 0.03,      # Reduced from 0.05
    "initial_position_size": 0.05,  # Reduced from 0.1
    "risk_per_trade": 0.02,        # Maximum risk per trade as fraction of portfolio (2%)
    "stop_loss": {
        "technical": True,
        "atr_multiplier": 1.5,      # Tightened from 2
        "trailing": True,
        "time_based": True,
        "max_days": 3               # Reduced from 5
    }
}

# Portfolio Management Configuration
PORTFOLIO = {
    "initial_allocation": {
        "stocks": 0.4,
        "crypto": 0.4,
        "options": 0.2
    },
    "rebalance_frequency": "weekly",  # Options: daily, weekly, monthly
    "correlation_threshold": 0.7  # Maximum correlation allowed between positions
}

# Signal Classification Configuration
SIGNAL_CLASSIFICATION = {
    "strong": {
        "min_indicators": 4,        # Increased from 3
        "position_size_multiplier": 1.0
    },
    "moderate": {
        "min_indicators": 3,        # Increased from 2
        "position_size_multiplier": 0.6  # Reduced from 0.7
    },
    "weak": {
        "min_indicators": 2,        # Increased from 1
        "position_size_multiplier": 0.2  # Reduced from 0.3
    }
}

# Backtesting Configuration
BACKTEST = {
    "start_date": "2023-01-01",
    "end_date": "2025-05-01",
    "max_duration_days": 365,  # Maximum backtest duration in days
    "commission": {
        "stocks": 0.001,  # 0.1% commission
        "crypto": 0.002,  # 0.2% commission
        "options": 0.005  # 0.5% commission
    },
    "slippage": 0.001  # 0.1% slippage
}

# Logging Configuration
LOGGING = {
    "level": "INFO",  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    "file": "trading_algo.log",
    "console": True
}
