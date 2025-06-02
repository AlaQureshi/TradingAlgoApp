"""
Main execution module for the Trading Algorithm AI.
Integrates data collection, signal generation, risk management, and portfolio management.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time

# Import modules
from data_collector import DataCollector
from signal_generator import SignalGenerator
from risk_manager import RiskManager
from portfolio_manager import PortfolioManager
from config import CONFIG, DATA_SOURCES, RISK_MANAGEMENT, PORTFOLIO, BACKTEST

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_algo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('trading_algo')

class TradingAlgorithm:
    """
    Main trading algorithm class that integrates all components.
    """
    
    def __init__(self):
        """Initialize the trading algorithm with all required components."""
        self.config = CONFIG
        self.portfolio_value = CONFIG['starting_capital']
        
        # Initialize components
        self.data_collector = DataCollector()
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager(portfolio_value=self.portfolio_value)
        self.portfolio_manager = PortfolioManager(portfolio_value=self.portfolio_value)
        
        # Track watchlist and active trades
        self.watchlist = {
            'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'crypto': ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano'],
            'options': []  # Will be populated based on stock signals
        }
        
        self.active_trades = {}
        self.signals_history = {}
        
        logger.info("Trading Algorithm initialized")
    
    def run(self):
        """
        Main execution method for the trading algorithm.
        """
        logger.info("Starting Trading Algorithm execution")
        
        try:
            # Process each asset class
            for asset_class in self.config['target_markets']:
                logger.info(f"Processing {asset_class}")
                
                # Process watchlist for this asset class
                for symbol in self.watchlist[asset_class]:
                    self._process_symbol(symbol, asset_class)
            
            # Update portfolio and check for rebalancing
            self._update_portfolio()
            
            logger.info("Trading Algorithm execution completed")
            
        except Exception as e:
            logger.error(f"Error in Trading Algorithm execution: {str(e)}")
    
    def _process_symbol(self, symbol, asset_class):
        """
        Process a single symbol for trading signals.
        
        Args:
            symbol (str): Asset symbol
            asset_class (str): Asset class (stocks, crypto, options)
        """
        logger.info(f"Processing {symbol} ({asset_class})")
        
        try:
            # Collect data
            data = self._collect_data(symbol, asset_class)
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return
            
            # Generate signals
            signals = self._generate_signals(symbol, data)
            
            # Store signals history
            self.signals_history[symbol] = signals.iloc[-1].to_dict()
            
            # Check for trading opportunities
            self._check_trading_signals(symbol, asset_class, signals, data)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
    
    def _collect_data(self, symbol, asset_class):
        """
        Collect market data for a symbol.
        
        Args:
            symbol (str): Asset symbol
            asset_class (str): Asset class (stocks, crypto, options)
            
        Returns:
            pandas.DataFrame: Market data
        """
        # Use backtest date range from config
        start_date = BACKTEST['start_date']
        end_date = BACKTEST['end_date']

        if asset_class == 'stocks':
            return self.data_collector.get_stock_data(symbol, start_date, end_date)
        
        elif asset_class == 'crypto':
            # For Polygon, symbol should be in format 'X:BTCUSD'
            polygon_symbol = symbol if symbol.startswith('X:') else f"X:{symbol.upper()}USD"
            return self.data_collector.get_crypto_data(polygon_symbol, start_date, end_date)
        
        elif asset_class == 'options':
            return self.data_collector.get_options_data(symbol)
        
        return pd.DataFrame()
    
    def _generate_signals(self, symbol, data):
        """
        Generate trading signals for a symbol.
        
        Args:
            symbol (str): Asset symbol
            data (pandas.DataFrame): Market data
            
        Returns:
            pandas.DataFrame: Data with signals
        """
        # Calculate technical indicators
        data_with_indicators = self.signal_generator.calculate_indicators(data)
        
        # Generate signals
        signals = self.signal_generator.generate_signals(data_with_indicators)
        
        logger.info(f"Generated signals for {symbol}: {signals['signal_strength'].iloc[-1]}")
        
        return signals
    
    def _check_trading_signals(self, symbol, asset_class, signals, data):
        """
        Check for trading signals and execute trades if appropriate.
        
        Args:
            symbol (str): Asset symbol
            asset_class (str): Asset class (stocks, crypto, options)
            signals (pandas.DataFrame): Data with signals
            data (pandas.DataFrame): Original market data
        """
        # Get the latest signal
        latest_signal = signals.iloc[-1]
        
        # Get the latest price
        latest_price = latest_signal['close']
        
        # Get the signal strength factor
        signal_strength = latest_signal['position_size_factor']
        
        # Get volatility (ATR)
        volatility = latest_signal.get('atr', latest_price * 0.02)  # Default to 2% if ATR not available
        
        # Check if we already have a position in this symbol
        existing_position = self.risk_manager.get_position(symbol)
        
        if existing_position:
            # Update existing position
            self._update_position(symbol, asset_class, latest_price, signal_strength, volatility, existing_position)
        else:
            # Check for new position
            self._check_new_position(symbol, asset_class, latest_price, signal_strength, volatility)
    
    def _update_position(self, symbol, asset_class, price, signal_strength, volatility, position):
        """
        Update an existing position based on new signals.
        
        Args:
            symbol (str): Asset symbol
            asset_class (str): Asset class (stocks, crypto, options)
            price (float): Current price
            signal_strength (float): Signal strength factor
            volatility (float): Asset volatility
            position (dict): Existing position details
        """
        # Check if signal has reversed
        current_direction = 1 if signal_strength > 0 else -1 if signal_strength < 0 else 0
        position_direction = 1 if position['direction'] == 'long' else -1
        
        # If signal has reversed strongly, exit position
        if current_direction * position_direction < 0 and abs(signal_strength) > 0.5:
            logger.info(f"Signal reversed for {symbol}, exiting position")
            self._exit_position(symbol, price, "Signal Reversal")
            return
        
        # Check stop loss
        if self.risk_manager.check_stop_loss(position, price, datetime.now()):
            logger.info(f"Stop loss triggered for {symbol}")
            self._exit_position(symbol, price, "Stop Loss")
            return
        
        # Update trailing stop if applicable
        self.risk_manager.update_stop_loss(position, price)
    
    def _check_new_position(self, symbol, asset_class, price, signal_strength, volatility):
        """
        Check if a new position should be opened based on signals.
        
        Args:
            symbol (str): Asset symbol
            asset_class (str): Asset class (stocks, crypto, options)
            price (float): Current price
            signal_strength (float): Signal strength factor
            volatility (float): Asset volatility
        """
        # Skip if signal is not strong enough
        if abs(signal_strength) < 0.3:
            logger.info(f"Signal not strong enough for {symbol}: {signal_strength}")
            return
        
        # Check available cash for this asset class
        available_cash = self.portfolio_manager.get_asset_class_available_cash(asset_class)
        
        if available_cash <= 0:
            logger.info(f"No available cash for {asset_class}")
            return
        
        # Calculate position size
        position = self.risk_manager.calculate_position_size(symbol, signal_strength, price, volatility)
        
        if not position:
            logger.info(f"No valid position calculated for {symbol}")
            return
        
        # Add asset class to position
        position['asset_class'] = asset_class
        
        # Check if position value exceeds available cash
        if position['position_value'] > available_cash:
            # Scale down position to fit available cash
            scale_factor = available_cash / position['position_value']
            position['quantity'] *= scale_factor
            position['position_value'] *= scale_factor
            position['risk_amount'] *= scale_factor
            logger.info(f"Scaled down position for {symbol} to fit available cash")
        
        # Execute trade (in signal-only mode, just log the signal)
        if self.config['execution_mode'] == 'signal_only':
            logger.info(f"SIGNAL: {position['direction']} {symbol} ({asset_class}), "
                       f"quantity: {position['quantity']:.4f}, "
                       f"price: ${price:.2f}, "
                       f"value: ${position['position_value']:.2f}")
            
            # In signal-only mode, still track the position for portfolio management
            self.risk_manager.add_position(position)
            self.portfolio_manager.add_position(position)
            
        else:
            # In auto-trade mode, execute the trade (not implemented in this version)
            logger.info(f"Would execute {position['direction']} trade for {symbol}")
            
            # Add position to tracking
            self.risk_manager.add_position(position)
            self.portfolio_manager.add_position(position)
    
    def _exit_position(self, symbol, price, reason):
        """
        Exit an existing position.
        
        Args:
            symbol (str): Asset symbol
            price (float): Current price
            reason (str): Reason for exit
        """
        # Get position details
        position = self.risk_manager.get_position(symbol)
        
        if not position:
            logger.warning(f"Attempted to exit non-existent position for {symbol}")
            return
        
        # Calculate P&L
        entry_price = position['entry_price']
        quantity = position['quantity']
        direction_multiplier = 1 if position['direction'] == 'long' else -1
        
        pnl = (price - entry_price) * quantity * direction_multiplier
        pnl_percentage = pnl / position['position_value'] if position['position_value'] > 0 else 0
        
        # Log the exit
        logger.info(f"EXIT: {position['direction']} {symbol}, "
                   f"quantity: {quantity:.4f}, "
                   f"entry: ${entry_price:.2f}, "
                   f"exit: ${price:.2f}, "
                   f"P&L: ${pnl:.2f} ({pnl_percentage:.2%}), "
                   f"reason: {reason}")
        
        # In signal-only mode, just log the exit
        if self.config['execution_mode'] == 'signal_only':
            # Remove position from tracking
            self.risk_manager.remove_position(symbol)
            self.portfolio_manager.remove_position(symbol)
            
            # Update portfolio value
            self.portfolio_value += pnl
            self.risk_manager.update_portfolio_value(self.portfolio_value)
            self.portfolio_manager.update_portfolio_value(self.portfolio_value)
            
        else:
            # In auto-trade mode, execute the exit (not implemented in this version)
            logger.info(f"Would execute exit for {symbol}")
            
            # Remove position from tracking
            self.risk_manager.remove_position(symbol)
            self.portfolio_manager.remove_position(symbol)
            
            # Update portfolio value
            self.portfolio_value += pnl
            self.risk_manager.update_portfolio_value(self.portfolio_value)
            self.portfolio_manager.update_portfolio_value(self.portfolio_value)
    
    def _update_portfolio(self):
        """
        Update portfolio status and check for rebalancing.
        """
        # Check if rebalancing is needed
        if self.portfolio_manager.check_rebalance_needed():
            logger.info("Portfolio rebalancing needed")
            
            # Calculate rebalance actions
            actions = self.portfolio_manager.calculate_rebalance_actions()
            
            # Log rebalance actions
            for asset_class, action in actions.items():
                logger.info(f"Rebalance {asset_class}: {action['action']} ${abs(action['adjustment']):.2f}")
            
            # In a real implementation, would execute rebalance trades here
        
        # Log portfolio status
        logger.info(f"Portfolio value: ${self.portfolio_value:.2f}")
        
        for asset_class in self.config['target_markets']:
            allocation = self.portfolio_manager.get_current_allocation(asset_class)
            logger.info(f"{asset_class} allocation: {allocation:.2%}")
    
    def get_signals(self):
        """
        Get the latest trading signals.
        
        Returns:
            dict: Latest signals by symbol
        """
        return self.signals_history
    
    def get_portfolio_status(self):
        """
        Get the current portfolio status.
        
        Returns:
            dict: Portfolio status including value and positions
        """
        positions = self.risk_manager.get_all_positions()
        
        return {
            'portfolio_value': self.portfolio_value,
            'positions': positions,
            'cash': self.portfolio_manager.get_available_cash()
        }

# Example usage
if __name__ == "__main__":
    # Initialize trading algorithm
    algo = TradingAlgorithm()
    
    # Run the algorithm once
    algo.run()
    
    # Get the latest signals
    signals = algo.get_signals()
    print("\nLatest Signals:")
    for symbol, signal in signals.items():
        print(f"{symbol}: {signal.get('signal_strength', 'neutral')}")
    
    # Get portfolio status
    status = algo.get_portfolio_status()
    print(f"\nPortfolio Value: ${status['portfolio_value']:.2f}")
    print(f"Available Cash: ${status['cash']:.2f}")
    print(f"Positions: {len(status['positions'])}")
    
    # In a real implementation, would run in a loop with appropriate timing
    # while True:
    #     algo.run()
    #     time.sleep(3600)  # Run hourly
