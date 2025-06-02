import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

def load_env_variables():
    """Load environment variables from .env file"""
    try:
        # Load .env file
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        load_dotenv(env_path)
        
        # Validate required variables
        required_vars = ['POLYGON_API_KEY']
        for var in required_vars:
            if not os.getenv(var):
                logger.error(f"Missing required environment variable: {var}")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Error loading environment variables: {e}")
        return False

def get_config():
    """Get configuration from environment variables"""
    return {
        'polygon_api_key': os.getenv('POLYGON_API_KEY'),
        'paper_trading': os.getenv('PAPER_TRADING', 'True').lower() == 'true',
        'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.02')),
        'starting_capital': float(os.getenv('STARTING_CAPITAL', '10000')),
        'model_path': os.getenv('MODEL_PATH', 'models'),
        'sequence_length': int(os.getenv('SEQUENCE_LENGTH', '60')),
        'prediction_threshold': float(os.getenv('PREDICTION_THRESHOLD', '0.7')),
        'symbols': {
            'stocks': os.getenv('STOCK_SYMBOLS', '').split(','),
            'crypto': os.getenv('CRYPTO_SYMBOLS', '').split(',')
        }
    }
