"""
Configuration file for the Trading Algorithm AI.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

CONFIG = {
    "polygon_api_key": os.getenv("POLYGON_API_KEY"),
    "paper_trading": os.getenv("PAPER_TRADING", "True").lower() == "true",
    "risk_per_trade": float(os.getenv("RISK_PER_TRADE", "0.02")),
    "max_position_size": float(os.getenv("MAX_POSITION_SIZE", "0.1")),
    "stop_loss_pct": float(os.getenv("STOP_LOSS_PCT", "0.02")),
    "take_profit_pct": float(os.getenv("TAKE_PROFIT_PCT", "0.04")),
    "use_ml_features": os.getenv("USE_ML_FEATURES", "True").lower() == "true",
    "starting_capital": float(os.getenv("STARTING_CAPITAL", "10000")),
    "buffer_size": int(os.getenv("BUFFER_SIZE", "100")),
    "min_data_points": int(os.getenv("MIN_DATA_POINTS", "50")),
    "symbols": {
        "stocks": os.getenv("STOCK_SYMBOLS", "").split(",")
    }
}

# Validate required configuration
if not CONFIG["polygon_api_key"]:
    raise ValueError("Polygon API key is required")
