import os
from trading.data.live_data_feed import PolygonWebSocket
from trading.ml.model import TradingModel
from signal_generator import SignalGenerator
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
        self.signal_generator = SignalGenerator()
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
                
                # Process data and generate signals
                processed_data_buffer = self.signal_generator.calculate_indicators(self.data_buffer[symbol])
                self.data_buffer[symbol] = self.signal_generator.generate_signals(processed_data_buffer)

                # Generate prediction if we have enough data
                if len(self.data_buffer[symbol]) >= self.model.sequence_length: # Keep previous guard for now
                    signal = self._get_latest_signal(symbol)
                    self._handle_prediction(symbol, signal, data['price'])
                    
        except Exception as e:
            logger.error(f"Error processing data: {e}")
    
    def _get_latest_signal(self, symbol):
        """Retrieves the latest signal from the data buffer."""
        try:
            if not self.data_buffer[symbol].empty and 'signal_strength' in self.data_buffer[symbol].columns:
                # Assuming 'signal_strength' is the column name from SignalGenerator
                latest_signal = self.data_buffer[symbol]['signal_strength'].iloc[-1]
                return latest_signal
            else:
                logger.warning(f"No signal or signal_strength column found for {symbol}. Returning neutral signal.")
                return 0.5 # Neutral signal
        except Exception as e:
            logger.error(f"Error retrieving latest signal for {symbol}: {e}")
            return 0.5 # Neutral signal in case of error
    
    def _handle_prediction(self, symbol, signal, current_price):
        """Handle trading signals and execute trades"""
        if signal is None: # Should ideally be a string like 'neutral' if no signal
            logger.warning(f"Received None signal for {symbol}, taking no action.")
            return

        # Ensure signal is a string, as expected by the new logic
        if not isinstance(signal, str):
            logger.warning(f"Received non-string signal '{signal}' for {symbol}. Expected string (e.g., 'strong_bullish'). Taking no action.")
            # Or, convert known numerical signals to string representations if applicable
            # For now, strict check and no action.
            return
            
        current_positions = self.trader.get_positions()
        symbol_in_positions = any(pos['symbol'] == symbol for pos in current_positions) # Alpaca returns list of positions

        try:
            logger.info(f"Received signal for {symbol}: {signal}, Current price: {current_price}")

            if signal in ['strong_bullish', 'moderate_bullish']:
                if not symbol_in_positions:
                    quantity = self._calculate_position_size(current_price)
                    if quantity > 0:
                        logger.info(f"Placing BUY order for {symbol}, Quantity: {quantity}")
                        self.trader.place_order(
                            symbol=symbol,
                            quantity=quantity,
                            side='BUY'
                        )
                    else:
                        logger.info(f"Calculated quantity is 0 for {symbol}, not placing BUY order.")
                else:
                    logger.info(f"Already have a position for {symbol}, not placing BUY order despite {signal} signal.")

            elif signal in ['strong_bearish', 'moderate_bearish']:
                if symbol_in_positions:
                    # Find the specific position to get its quantity
                    position_qty = 0
                    for pos in current_positions:
                        if pos['symbol'] == symbol:
                            position_qty = abs(float(pos['qty'])) # Ensure quantity is positive float
                            break

                    if position_qty > 0:
                        logger.info(f"Placing SELL order for {symbol}, Quantity: {position_qty}")
                        self.trader.place_order(
                            symbol=symbol,
                            quantity=position_qty,
                            side='SELL'
                        )
                    else:
                        logger.info(f"No quantity to sell for {symbol}, not placing SELL order.")
                else:
                    logger.info(f"No position for {symbol}, not placing SELL order despite {signal} signal.")

            elif signal in ['neutral', 'weak_bullish', 'weak_bearish']:
                logger.info(f"Neutral or weak signal '{signal}' for {symbol}. No trading action taken.")
                # Future: Consider closing existing positions on 'neutral' or weak opposing signals if strategy dictates.

            else:
                logger.warning(f"Unknown signal type '{signal}' for {symbol}. No action taken.")

        except Exception as e:
            logger.error(f"Error handling signal '{signal}' for {symbol}: {e}")
    
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
