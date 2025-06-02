"""
Portfolio management module for the Trading Algorithm AI.
Handles asset allocation, rebalancing, and correlation management.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Import configuration
from config import PORTFOLIO, CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('portfolio_manager')

class PortfolioManager:
    """
    Manages portfolio allocation, rebalancing, and correlation analysis
    for the trading algorithm.
    """
    
    def __init__(self, portfolio_value=None):
        """
        Initialize the portfolio manager with configuration settings.
        
        Args:
            portfolio_value (float, optional): Current portfolio value. Defaults to starting capital.
        """
        self.config = PORTFOLIO
        self.portfolio_value = portfolio_value if portfolio_value is not None else CONFIG['starting_capital']
        self.allocations = self.config['initial_allocation'].copy()
        self.positions = {}  # Dictionary to track current positions
        self.last_rebalance = datetime.now()
        logger.info(f"PortfolioManager initialized with portfolio value: ${self.portfolio_value}")
    
    def get_target_allocation(self, asset_class):
        """
        Get the target allocation for a specific asset class.
        
        Args:
            asset_class (str): Asset class (e.g., 'stocks', 'crypto', 'options')
            
        Returns:
            float: Target allocation as a fraction of portfolio
        """
        return self.allocations.get(asset_class, 0.0)
    
    def get_current_allocation(self, asset_class):
        """
        Calculate the current allocation for a specific asset class.
        
        Args:
            asset_class (str): Asset class (e.g., 'stocks', 'crypto', 'options')
            
        Returns:
            float: Current allocation as a fraction of portfolio
        """
        # Sum the value of all positions in the asset class
        asset_value = sum(
            position['position_value'] 
            for position in self.positions.values() 
            if position['asset_class'] == asset_class
        )
        
        # Calculate allocation as a fraction of portfolio
        current_allocation = asset_value / self.portfolio_value if self.portfolio_value > 0 else 0.0
        
        return current_allocation
    
    def check_rebalance_needed(self, current_time=None):
        """
        Check if portfolio rebalancing is needed based on time or allocation drift.
        
        Args:
            current_time (datetime, optional): Current time for time-based rebalancing
            
        Returns:
            bool: True if rebalancing is needed, False otherwise
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Check time-based rebalancing
        if self.config['rebalance_frequency'] == 'daily':
            days_since_rebalance = (current_time - self.last_rebalance).days
            if days_since_rebalance >= 1:
                logger.info("Daily rebalance check triggered")
                return True
        
        elif self.config['rebalance_frequency'] == 'weekly':
            days_since_rebalance = (current_time - self.last_rebalance).days
            if days_since_rebalance >= 7:
                logger.info("Weekly rebalance check triggered")
                return True
        
        elif self.config['rebalance_frequency'] == 'monthly':
            days_since_rebalance = (current_time - self.last_rebalance).days
            if days_since_rebalance >= 30:
                logger.info("Monthly rebalance check triggered")
                return True
        
        # Check allocation drift
        for asset_class in self.allocations:
            target = self.allocations[asset_class]
            current = self.get_current_allocation(asset_class)
            
            # If allocation has drifted more than 10% from target, rebalance
            if abs(current - target) / target > 0.1:
                logger.info(f"Allocation drift detected for {asset_class}: {current:.2%} vs target {target:.2%}")
                return True
        
        return False
    
    def calculate_rebalance_actions(self):
        """
        Calculate actions needed to rebalance the portfolio.
        
        Returns:
            dict: Rebalance actions by asset class
        """
        actions = {}
        
        for asset_class in self.allocations:
            target = self.allocations[asset_class]
            current = self.get_current_allocation(asset_class)
            
            # Calculate target value
            target_value = self.portfolio_value * target
            
            # Calculate current value
            current_value = self.portfolio_value * current
            
            # Calculate adjustment needed
            adjustment = target_value - current_value
            
            actions[asset_class] = {
                'target_allocation': target,
                'current_allocation': current,
                'target_value': target_value,
                'current_value': current_value,
                'adjustment': adjustment,
                'action': 'buy' if adjustment > 0 else 'sell'
            }
        
        logger.info(f"Rebalance actions calculated: {actions}")
        return actions
    
    def update_allocations(self, new_allocations):
        """
        Update target allocations for asset classes.
        
        Args:
            new_allocations (dict): New target allocations by asset class
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Validate that allocations sum to 1.0
        if abs(sum(new_allocations.values()) - 1.0) > 0.001:
            logger.error(f"Invalid allocations: sum is {sum(new_allocations.values())}, should be 1.0")
            return False
        
        # Update allocations
        self.allocations = new_allocations.copy()
        logger.info(f"Updated target allocations: {self.allocations}")
        
        # Mark as rebalanced
        self.last_rebalance = datetime.now()
        
        return True
    
    def calculate_correlation(self, price_data):
        """
        Calculate correlation matrix for assets in the portfolio.
        
        Args:
            price_data (dict): Dictionary of price DataFrames by symbol
            
        Returns:
            pandas.DataFrame: Correlation matrix
        """
        # Extract close prices for each asset
        close_prices = {}
        
        for symbol, df in price_data.items():
            if 'close' in df.columns:
                close_prices[symbol] = df['close']
        
        # Create DataFrame with all close prices
        if close_prices:
            price_df = pd.DataFrame(close_prices)
            
            # Calculate correlation matrix
            correlation_matrix = price_df.corr()
            
            logger.info(f"Calculated correlation matrix for {len(close_prices)} assets")
            return correlation_matrix
        
        logger.warning("No price data available for correlation calculation")
        return pd.DataFrame()
    
    def check_correlation_limits(self, correlation_matrix):
        """
        Check if any assets exceed correlation limits.
        
        Args:
            correlation_matrix (pandas.DataFrame): Correlation matrix
            
        Returns:
            list: Pairs of assets that exceed correlation limits
        """
        high_correlation_pairs = []
        threshold = self.config['correlation_threshold']
        
        # Check each pair of assets
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                symbol1 = correlation_matrix.columns[i]
                symbol2 = correlation_matrix.columns[j]
                correlation = correlation_matrix.iloc[i, j]
                
                # Check if correlation exceeds threshold
                if abs(correlation) > threshold:
                    high_correlation_pairs.append({
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'correlation': correlation
                    })
        
        if high_correlation_pairs:
            logger.warning(f"Found {len(high_correlation_pairs)} pairs with correlation > {threshold}")
        
        return high_correlation_pairs
    
    def adjust_for_correlation(self, positions, correlation_matrix):
        """
        Adjust position sizes based on correlation to reduce systemic risk.
        
        Args:
            positions (dict): Current positions
            correlation_matrix (pandas.DataFrame): Correlation matrix
            
        Returns:
            dict: Adjusted positions
        """
        # Skip if no correlation data or positions
        if correlation_matrix.empty or not positions:
            return positions
        
        adjusted_positions = positions.copy()
        threshold = self.config['correlation_threshold']
        
        # Check each position against others for high correlation
        for symbol1, position1 in positions.items():
            if symbol1 not in correlation_matrix.columns:
                continue
            
            # Count highly correlated positions with the same direction
            high_correlation_count = 0
            
            for symbol2, position2 in positions.items():
                if symbol1 == symbol2 or symbol2 not in correlation_matrix.columns:
                    continue
                
                correlation = correlation_matrix.loc[symbol1, symbol2]
                
                # Check if correlation is high and positions have same direction
                if abs(correlation) > threshold and position1['direction'] == position2['direction']:
                    high_correlation_count += 1
            
            # Adjust position size if highly correlated with multiple others
            if high_correlation_count > 1:
                # Reduce position size based on number of correlations
                reduction_factor = 1.0 / (1.0 + (high_correlation_count * 0.2))
                
                # Apply reduction to position value and quantity
                adjusted_positions[symbol1]['position_value'] *= reduction_factor
                adjusted_positions[symbol1]['quantity'] *= reduction_factor
                
                logger.info(f"Adjusted position for {symbol1} due to high correlation (factor: {reduction_factor:.2f})")
        
        return adjusted_positions
    
    def add_position(self, position):
        """
        Add a new position to the tracking dictionary.
        
        Args:
            position (dict): Position details including asset_class
        """
        if 'asset_class' not in position:
            logger.error(f"Cannot add position for {position['symbol']}: missing asset_class")
            return
        
        self.positions[position['symbol']] = position
        logger.info(f"Added position for {position['symbol']} in {position['asset_class']}")
    
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
    
    def update_portfolio_value(self, new_value):
        """
        Update the current portfolio value.
        
        Args:
            new_value (float): New portfolio value
        """
        self.portfolio_value = new_value
        logger.info(f"Portfolio value updated to ${new_value:.2f}")
    
    def get_available_cash(self):
        """
        Calculate available cash based on portfolio value and current positions.
        
        Returns:
            float: Available cash
        """
        # Sum the value of all positions
        total_position_value = sum(position['position_value'] for position in self.positions.values())
        
        # Calculate available cash
        available_cash = self.portfolio_value - total_position_value
        
        return max(0, available_cash)
    
    def get_asset_class_available_cash(self, asset_class):
        """
        Calculate available cash for a specific asset class based on target allocation.
        
        Args:
            asset_class (str): Asset class (e.g., 'stocks', 'crypto', 'options')
            
        Returns:
            float: Available cash for the asset class
        """
        # Get target allocation for the asset class
        target_allocation = self.allocations.get(asset_class, 0.0)
        
        # Calculate target value
        target_value = self.portfolio_value * target_allocation
        
        # Calculate current value
        current_value = sum(
            position['position_value'] 
            for position in self.positions.values() 
            if position['asset_class'] == asset_class
        )
        
        # Calculate available cash
        available_cash = target_value - current_value
        
        return max(0, available_cash)

# Example usage
if __name__ == "__main__":
    # Initialize portfolio manager
    portfolio_manager = PortfolioManager(portfolio_value=100)
    
    # Add sample positions
    portfolio_manager.add_position({
        'symbol': 'AAPL',
        'asset_class': 'stocks',
        'direction': 'long',
        'quantity': 0.5,
        'entry_price': 150.0,
        'position_value': 75.0
    })
    
    portfolio_manager.add_position({
        'symbol': 'BTC',
        'asset_class': 'crypto',
        'direction': 'long',
        'quantity': 0.002,
        'entry_price': 25000.0,
        'position_value': 50.0
    })
    
    # Check current allocations
    for asset_class in ['stocks', 'crypto', 'options']:
        current = portfolio_manager.get_current_allocation(asset_class)
        target = portfolio_manager.get_target_allocation(asset_class)
        print(f"{asset_class}: Current {current:.2%}, Target {target:.2%}")
    
    # Calculate rebalance actions
    actions = portfolio_manager.calculate_rebalance_actions()
    print("\nRebalance actions:")
    for asset_class, action in actions.items():
        print(f"{asset_class}: {action['action']} ${abs(action['adjustment']):.2f}")
    
    # Check available cash
    available_cash = portfolio_manager.get_available_cash()
    print(f"\nAvailable cash: ${available_cash:.2f}")
    
    # Check available cash by asset class
    for asset_class in ['stocks', 'crypto', 'options']:
        available = portfolio_manager.get_asset_class_available_cash(asset_class)
        print(f"{asset_class} available cash: ${available:.2f}")
