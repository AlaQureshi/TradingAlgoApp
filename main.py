import os
from trading.data.live_data_feed import PolygonWebSocket
from trading.ml.model import TradingModel
from trading.execution.trader import Trader
import pandas as pd
import logging
from datetime import datetime, timedelta
import numpy as np
from utils.env_loader import load_env_variables, get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingSystem:
    def __init__(self, config):
        self.config = config
        
        # Initialize components
        try:
            self.model = TradingModel.load(self.config['model_path'])
        except:
            logger.warning("Could not load existing model, initializing new one")
            self.model = TradingModel()
            
        self.trader = Trader(self.config)
        self.data_buffer = {}
        
        for symbol in self.config['symbols']:
            self.data_buffer[symbol] = pd.DataFrame()
        
        # Initialize WebSocket
        self.ws = PolygonWebSocket(
            api_key=self.config['polygon_api_key'],
            symbols=self.config['symbols'],
            on_data_callback=self.on_data_received
        )
    
    def start(self):
        """Start the trading system"""
        logger.info("Starting trading system...")
        self.ws.connect()
    
    def on_data_received(self, data):
        """Handle incoming market data"""
        try:
            symbol = data['symbol']
            
            # Update data buffer
            new_row = pd.DataFrame([data])
            if symbol in self.data_buffer:
                self.data_buffer[symbol] = pd.concat([self.data_buffer[symbol], new_row]).tail(1000)
                
                # Generate prediction if we have enough data
                if len(self.data_buffer[symbol]) >= self.model.sequence_length:
                    prediction = self._generate_prediction(symbol)
                    self._handle_prediction(symbol, prediction, data['price'])
                    
        except Exception as e:
            logger.error(f"Error processing data: {e}")
    
    def _generate_prediction(self, symbol):
        """Generate trading signals using the ML model"""
        try:
            X = self.model.prepare_data(self.data_buffer[symbol])
            return self.model.predict(X[-1:])
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            return None
    
    def _handle_prediction(self, symbol, prediction, current_price):
        """Handle model predictions and execute trades"""
        if prediction is None:
            return
            
        threshold = 0.7
        current_positions = self.trader.get_positions()
        
        try:
            if prediction > threshold and symbol not in current_positions:
                # Strong buy signal
                self.trader.place_order(
                    symbol=symbol,
                    quantity=self._calculate_position_size(current_price),
                    side='BUY'
                )
            elif prediction < (1 - threshold) and symbol in current_positions:
                # Strong sell signal
                position = current_positions[symbol]
                self.trader.place_order(
                    symbol=symbol,
                    quantity=abs(position['quantity']),
                    side='SELL'
                )
        except Exception as e:
            logger.error(f"Error handling prediction: {e}")
    
    def _calculate_position_size(self, price):
        """Calculate position size based on risk management"""
        account = self.trader.get_account_info()
        risk_per_trade = 0.02  # 2% risk per trade
        position_value = account['total_value'] * risk_per_trade
        return round(position_value / price, 2)

def main():
    # Load environment variables
    if not load_env_variables():
        logger.error("Failed to load environment variables")
        return

    # Get configuration
    config = get_config()
    
    # Initialize trading system with config
    system = TradingSystem(config)
    system.start()

if __name__ == "__main__":
    main()
