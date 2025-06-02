"""
Main entry point for the trading application
"""
import os
import sys
import logging
import time
from datetime import datetime
import pandas as pd
from threading import Event

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from trading.ml.model import TradingModel
from trading.execution.trader import Trader
from trading.data.yahoo_data_feed import YahooDataFeed
from trading.data.live_data_feed import PolygonWebSocket
from trading.indicators import add_indicators
from trading.strategy import DelayedDataStrategy
from config import CONFIG
from trading.utils.market_hours import MarketHours

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingApp:
    def __init__(self):
        # Validate configuration
        if not CONFIG["polygon_api_key"]:
            raise ValueError("Polygon API key is not configured")
            
        self.trader = Trader(CONFIG)
        self.model = TradingModel()
        self.running = Event()
        self.last_prices = {}
        
        # Remove YahooDataFeed initialization
        self.data_client = None
        
        # Initialize data buffer with proper dtypes
        self.data_buffer = {}
        self.dtype_dict = {
            'timestamp': 'datetime64[ns]',
            'price': 'float64',
            'size': 'int64'
        }
        
        for symbol in CONFIG["symbols"]["stocks"]:
            self.data_buffer[symbol] = pd.DataFrame({
                'timestamp': pd.Series(dtype='datetime64[ns]'),
                'price': pd.Series(dtype='float64'),
                'size': pd.Series(dtype='int64')
            })
        
        # Initialize websocket with validated config
        self.ws = PolygonWebSocket(
            api_key=CONFIG["polygon_api_key"],
            symbols=CONFIG["symbols"]["stocks"],
            on_data_callback=self.handle_data
        )
        
        # Initialize trading strategy
        self.strategy = DelayedDataStrategy(CONFIG)
        
        # Show initial setup with more details
        logger.info("\n=== Trading System Configuration ===")
        logger.info(f"Data Source: Polygon.io Websocket")
        logger.info(f"Mode: {'ML-enabled' if CONFIG['use_ml_features'] else 'Simple Moving Average'}")
        logger.info(f"Trading: {'Paper' if CONFIG['paper_trading'] else 'Live'}")
        logger.info(f"Starting Capital: ${CONFIG['starting_capital']:,.2f}")
        logger.info(f"Risk Per Trade: {CONFIG['risk_per_trade']*100}%")
        
        self.min_data_points = CONFIG["min_data_points"]
        self.running.set()  # Set running state at initialization

    def analyze_price_action(self, symbol, current_price):
        """Analyze price movement using simple logic when ML is disabled"""
        if symbol not in self.last_prices:
            self.last_prices[symbol] = current_price
            return None
        
        price_change = (current_price - self.last_prices[symbol]) / self.last_prices[symbol]
        self.last_prices[symbol] = current_price
        
        if price_change > 0.01:  # 1% up
            return "BUY"
        elif price_change < -0.01:  # 1% down
            return "SELL"
        return None

    def analyze_trading_signals(self, symbol, data):
        """Analyze both ML and technical signals"""
        df = add_indicators(data)
        signals = []
        
        # ML Signal
        if CONFIG["use_ml_features"]:
            prediction = self.model.predict(df)
            if prediction > 0.7:
                signals.append(("ML", "BUY", 0.7))
            elif prediction < 0.3:
                signals.append(("ML", "SELL", 0.7))
        
        # Technical Signals
        latest = df.iloc[-1]
        
        # Moving Average Crossover
        if latest['MA_short'] > latest['MA_long']:
            signals.append(("MA", "BUY", 0.5))
        elif latest['MA_short'] < latest['MA_long']:
            signals.append(("MA", "SELL", 0.5))
        
        # RSI Signals
        if latest['RSI'] < 30:
            signals.append(("RSI", "BUY", 0.6))
        elif latest['RSI'] > 70:
            signals.append(("RSI", "SELL", 0.6))
        
        return signals

    def calculate_position_size(self, symbol, price):
        """Calculate position size based on risk management"""
        account_info = self.trader.get_account_info()
        risk_amount = account_info['total_value'] * CONFIG['risk_per_trade']
        max_position = account_info['total_value'] * CONFIG['max_position_size']
        
        # Calculate shares based on risk
        stop_loss = price * (1 - CONFIG['stop_loss_pct'])
        risk_per_share = price - stop_loss
        shares = int(risk_amount / risk_per_share)
        
        # Limit by max position size
        max_shares = int(max_position / price)
        return min(shares, max_shares)

    def handle_data(self, data):
        """Handle incoming market data"""
        try:
            symbol = data["symbol"]
            
            # Create DataFrame with correct dtypes
            new_data = pd.DataFrame([{
                'timestamp': pd.Timestamp(data['timestamp']),
                'price': float(data['price']),
                'size': int(data['size'])
            }]).astype(self.dtype_dict)
            
            # Concatenate with proper dtypes
            self.data_buffer[symbol] = pd.concat(
                [self.data_buffer[symbol], new_data],
                ignore_index=True
            ).tail(100)
            
            logger.info(f"\n[{symbol}] New Data Point:")
            logger.info(f"Price: ${data['price']:.2f} | Volume: {data['size']:,}")
            
            # Trading logic
            if len(self.data_buffer[symbol]) >= self.min_data_points:
                # Analyze trends considering data delay
                trend_metrics = self.strategy.analyze_trend(self.data_buffer[symbol])
                
                # Only trade if trend is strong enough to overcome delay
                if trend_metrics['trend_strength'] > 0.02:  # 2% minimum trend
                    base_shares = self.calculate_position_size(symbol, data['price'])
                    adjusted_shares = self.strategy.get_position_size(trend_metrics, base_shares)
                    
                    if trend_metrics['trend_direction'] > 0:
                        logger.info(f"🚀 Strong Uptrend Detected: BUY {adjusted_shares} {symbol}")
                        self.trader.place_order(symbol, adjusted_shares, "BUY")
                    elif trend_metrics['trend_direction'] < 0:
                        logger.info(f"📉 Strong Downtrend Detected: SELL {adjusted_shares} {symbol}")
                        self.trader.place_order(symbol, adjusted_shares, "SELL")
                
        except Exception as e:
            logger.error(f"Error handling data: {e}")

    def run(self):
        """Run the trading application"""
        try:
            logger.info("\n=== Trading Algorithm Starting ===")
            
            # Check market status
            is_open, market_status = MarketHours.is_market_open()
            logger.info(f"Market Status: {market_status}")
            
            if not is_open and not CONFIG.get('trade_extended_hours', False):
                logger.warning("Market is closed. Start during market hours (9:30 AM - 4:00 PM ET)")
                return
            
            # Initialize connection with shorter timeout and REST fallback
            connection_timeout = 10
            start_time = time.time()
            
            logger.info("Initializing data feed...")
            self.ws.connect()
            
            # Wait for initial data
            data_timeout = 30
            start_time = time.time()
            
            while not self.ws.is_connected() and time.time() - start_time < data_timeout:
                time.sleep(1)
            
            if not self.ws.is_connected():
                logger.warning("No live data available - check market hours or data subscription")
                return
                
            logger.info("Successfully receiving market data")
            
            # Main trading loop
            while self.running.is_set():
                if not self.ws.is_connected():
                    logger.warning("Data connection lost - attempting to reconnect...")
                    self.ws.connect()
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("\n=== Shutting Down Trading System ===")
            self.running.clear()
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            self.running.clear()
        finally:
            if self.ws:
                self.ws.disconnect()

def main():
    app = TradingApp()
    app.run()

if __name__ == "__main__":
    main()
