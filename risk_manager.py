"""
Risk management module for the Trading Algorithm AI.
Handles position sizing, stop loss calculation, and risk metrics.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Import configuration
from config import RISK_MANAGEMENT, CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('risk_manager')

class RiskManager:
    """
    Manages risk for the trading algorithm, including position sizing,
    stop loss calculation, and portfolio risk metrics.
    """
    
    def __init__(self, portfolio_value=None):
        """
        Initialize the risk manager with configuration settings.
        
        Args:
            portfolio_value (float, optional): Current portfolio value. Defaults to starting capital.
        """
        self.config = RISK_MANAGEMENT
        self.portfolio_value = portfolio_value if portfolio_value is not None else CONFIG['starting_capital']
        self.positions = {}  # Dictionary to track current positions
        self.daily_pnl = 0  # Track daily profit/loss
        logger.info(f"RiskManager initialized with portfolio value: ${self.portfolio_value}")
    
    def calculate_position_size(self, symbol, signal_strength, price, volatility):
        """
        Calculate the appropriate position size based on signal strength and risk parameters.
        
        Args:
            symbol (str): Asset symbol
            signal_strength (float): Signal strength factor (-1.0 to 1.0)
            price (float): Current asset price
            volatility (float): Asset volatility (e.g., ATR)
            
        Returns:
            dict: Position details including size, value, and risk metrics
        """
        # Skip if signal strength is zero or very weak
        if abs(signal_strength) < 0.1:
            return None
        
        # Determine direction (long/short)
        direction = 1 if signal_strength > 0 else -1
        
        # Base position size as percentage of portfolio
        base_size = self.config['initial_position_size']
        
        # Adjust based on signal strength
        adjusted_size = base_size * abs(signal_strength)
        
        # Ensure position size doesn't exceed maximum
        position_size = min(adjusted_size, self.config['max_position_size'])
        
        # Calculate position value
        position_value = self.portfolio_value * position_size
        
        # Calculate quantity based on price
        quantity = position_value / price
        
        # Calculate stop loss price
        stop_loss_price = self._calculate_stop_loss(price, volatility, direction)
        
        # Calculate risk per share
        risk_per_share = abs(price - stop_loss_price)
        
        # Calculate total risk amount
        risk_amount = risk_per_share * quantity
        
        # Calculate risk as percentage of portfolio
        risk_percentage = risk_amount / self.portfolio_value
        
        # Position details
        position = {
            'symbol': symbol,
            'direction': 'long' if direction > 0 else 'short',
            'quantity': quantity,
            'entry_price': price,
            'position_value': position_value,
            'stop_loss': stop_loss_price,
            'risk_amount': risk_amount,
            'risk_percentage': risk_percentage,
            'entry_time': datetime.now(),
            'max_days': self.config['stop_loss']['max_days'] if self.config['stop_loss']['time_based'] else None
        }
        
        logger.info(f"Calculated position for {symbol}: {position['direction']}, quantity: {position['quantity']:.4f}, risk: {position['risk_percentage']:.2%}")
        return position
    
    def _calculate_stop_loss(self, price, volatility, direction):
        """
        Calculate stop loss price based on volatility and direction.
        
        Args:
            price (float): Current asset price
            volatility (float): Asset volatility (e.g., ATR)
            direction (int): Trade direction (1 for long, -1 for short)
            
        Returns:
            float: Stop loss price
        """
        if not self.config['stop_loss']['technical']:
            # If technical stop loss is disabled, use a fixed percentage
            stop_percentage = 0.05  # 5% default stop loss
            return price * (1 - direction * stop_percentage)
        
        # Use ATR for stop loss calculation
        atr_multiplier = self.config['stop_loss']['atr_multiplier']
        stop_distance = volatility * atr_multiplier
        
        # Calculate stop loss price based on direction
        if direction > 0:
            # Long position: stop loss below entry price
            stop_loss = price - stop_distance
        else:
            # Short position: stop loss above entry price
            stop_loss = price + stop_distance
        
        return stop_loss
    
    def update_stop_loss(self, position, current_price):
        """
        Update stop loss for a position, implementing trailing stops if enabled.
        
        Args:
            position (dict): Position details
            current_price (float): Current asset price
            
        Returns:
            dict: Updated position with new stop loss
        """
        # Skip if trailing stops are disabled
        if not self.config['stop_loss']['trailing']:
            return position
        
        # Update stop loss for long positions
        if position['direction'] == 'long' and current_price > position['entry_price']:
            # Calculate potential new stop loss
            price_gain = current_price - position['entry_price']
            trail_percentage = 0.5  # Trail by 50% of the gain
            new_stop = position['entry_price'] + (price_gain * trail_percentage)
            
            # Only update if new stop is higher than current stop
            if new_stop > position['stop_loss']:
                position['stop_loss'] = new_stop
                logger.info(f"Updated trailing stop for {position['symbol']} to {position['stop_loss']:.4f}")
        
        # Update stop loss for short positions
        elif position['direction'] == 'short' and current_price < position['entry_price']:
            # Calculate potential new stop loss
            price_gain = position['entry_price'] - current_price
            trail_percentage = 0.5  # Trail by 50% of the gain
            new_stop = position['entry_price'] - (price_gain * trail_percentage)
            
            # Only update if new stop is lower than current stop
            if new_stop < position['stop_loss']:
                position['stop_loss'] = new_stop
                logger.info(f"Updated trailing stop for {position['symbol']} to {position['stop_loss']:.4f}")
        
        return position
    
    def check_stop_loss(self, position, current_price, current_time=None):
        """
        Check if a position has hit its stop loss or time-based exit criteria.
        
        Args:
            position (dict): Position details
            current_price (float): Current asset price
            current_time (datetime, optional): Current time for time-based exits
            
        Returns:
            bool: True if stop loss is triggered, False otherwise
        """
        # Check price-based stop loss
        if position['direction'] == 'long' and current_price <= position['stop_loss']:
            logger.info(f"Stop loss triggered for long position in {position['symbol']} at {current_price:.4f}")
            return True
        
        if position['direction'] == 'short' and current_price >= position['stop_loss']:
            logger.info(f"Stop loss triggered for short position in {position['symbol']} at {current_price:.4f}")
            return True
        
        # Check time-based stop loss if enabled
        if self.config['stop_loss']['time_based'] and current_time and position['max_days']:
            entry_time = position['entry_time']
            days_held = (current_time - entry_time).days
            
            if days_held >= position['max_days']:
                logger.info(f"Time-based exit triggered for {position['symbol']} after {days_held} days")
                return True
        
        return False
    
    def check_portfolio_risk(self):
        """
        Check if portfolio risk limits have been exceeded.
        
        Returns:
            bool: True if trading should continue, False if risk limits exceeded
        """
        # Calculate total portfolio risk
        total_risk = sum(position['risk_percentage'] for position in self.positions.values())
        
        # Check if total risk exceeds maximum allowed
        if total_risk > self.config['max_drawdown']:
            logger.warning(f"Portfolio risk limit exceeded: {total_risk:.2%} > {self.config['max_drawdown']:.2%}")
            return False
        
        # Check daily loss limit
        if self.daily_pnl < 0 and abs(self.daily_pnl) / self.portfolio_value > self.config['daily_loss_limit']:
            logger.warning(f"Daily loss limit exceeded: {abs(self.daily_pnl) / self.portfolio_value:.2%} > {self.config['daily_loss_limit']:.2%}")
            return False
        
        return True
    
    def update_portfolio_value(self, new_value):
        """
        Update the current portfolio value.
        
        Args:
            new_value (float): New portfolio value
        """
        old_value = self.portfolio_value
        self.portfolio_value = new_value
        
        # Calculate daily P&L
        self.daily_pnl = new_value - old_value
        
        logger.info(f"Portfolio value updated to ${new_value:.2f}, daily P&L: ${self.daily_pnl:.2f}")
    
    def reset_daily_pnl(self):
        """Reset the daily P&L counter."""
        self.daily_pnl = 0
        logger.info("Daily P&L reset")
    
    def add_position(self, position):
        """
        Add a new position to the tracking dictionary.
        
        Args:
            position (dict): Position details
        """
        self.positions[position['symbol']] = position
        logger.info(f"Added position for {position['symbol']}")
    
    def remove_position(self, symbol):
        """
        Remove a position from the tracking dictionary.
        
        Args:
            symbol (str): Symbol of the position to remove
            
        Returns:
            dict: The removed position, or None if not found
        """
        if symbol in self.positions:
            position = self.positions.pop(symbol)
            logger.info(f"Removed position for {symbol}")
            return position
        
        logger.warning(f"Attempted to remove non-existent position for {symbol}")
        return None
    
    def get_position(self, symbol):
        """
        Get position details for a specific symbol.
        
        Args:
            symbol (str): Symbol to look up
            
        Returns:
            dict: Position details, or None if not found
        """
        return self.positions.get(symbol)
    
    def get_all_positions(self):
        """
        Get all current positions.
        
        Returns:
            dict: All positions
        """
        return self.positions
    
    def calculate_kelly_criterion(self, win_rate, win_loss_ratio):
        """
        Calculate optimal position size using the Kelly Criterion.
        
        Args:
            win_rate (float): Probability of winning (0.0 to 1.0)
            win_loss_ratio (float): Ratio of average win to average loss
            
        Returns:
            float: Optimal position size as a fraction of portfolio
        """
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Limit Kelly to a reasonable range
        kelly = max(0, min(kelly, self.config['max_position_size']))
        
        # Apply a fractional Kelly for more conservative sizing
        fractional_kelly = kelly * 0.5  # Use half Kelly
        
        return fractional_kelly

# Example usage
if __name__ == "__main__":
    # Initialize risk manager
    risk_manager = RiskManager(portfolio_value=100)
    
    # Calculate position size for a sample trade
    position = risk_manager.calculate_position_size(
        symbol="AAPL",
        signal_strength=0.8,  # Strong bullish signal
        price=150.0,
        volatility=5.0  # ATR value
    )
    
    print("Sample position:")
    print(position)
    
    # Add position to tracking
    if position:
        risk_manager.add_position(position)
    
    # Update stop loss with new price
    if position:
        updated_position = risk_manager.update_stop_loss(position, 160.0)
        print("\nUpdated position with trailing stop:")
        print(updated_position)
    
    # Check if stop loss is triggered
    if position:
        is_stopped = risk_manager.check_stop_loss(position, 140.0)
        print(f"\nStop loss triggered: {is_stopped}")
