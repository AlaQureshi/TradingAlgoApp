"""
Backtesting module for the Trading Algorithm AI.
Tests the algorithm on historical data and optimizes parameters.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import modules
from data_collector import DataCollector
from signal_generator import SignalGenerator
from risk_manager import RiskManager
from portfolio_manager import PortfolioManager
from trading_algorithm import TradingAlgorithm
from config import CONFIG, BACKTEST, INDICATORS, RISK_MANAGEMENT, SIGNAL_CLASSIFICATION, OPTIONS_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('backtest')

class Backtester:
    """
    Backtests the trading algorithm on historical data and optimizes parameters.
    """
    
    def __init__(self):
        """Initialize the backtester with configuration settings."""
        self.config = BACKTEST
        self.data_collector = DataCollector()
        self.results = {}
        # Initialize portfolio
        self.portfolio = {
            'cash': CONFIG['starting_capital'],
            'positions': {},
            'history': [],
            'trades': []
        }
        logger.info("Backtester initialized")
    
    def run_backtest(self, start_date=None, end_date=None, symbols=None):
        """
        Run a backtest of the trading algorithm.
        
        Args:
            start_date (str, optional): Start date for backtest (YYYY-MM-DD)
            end_date (str, optional): End date for backtest (YYYY-MM-DD)
            symbols (dict, optional): Dictionary of symbols by asset class
            
        Returns:
            dict: Backtest results
        """
        # Use configured dates if not provided
        self.start_date = start_date or self.config['start_date']
        self.end_date = end_date or self.config['end_date']
        
        # Convert dates to datetime for validation
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        # Validate date range
        if end_dt <= start_dt:
            logger.error("End date must be after start date")
            return None
        
        # Reset portfolio at the start of each backtest
        self.portfolio = {
            'cash': CONFIG['starting_capital'],
            'positions': {},
            'history': [],
            'trades': []
        }
        
        # Default symbols if not provided
        if not symbols:
            symbols = {
                'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                'crypto': ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano'],
                'options': []  # Options will be derived from stocks
            }
        logger.info(f"Starting backtest from {self.start_date} to {self.end_date}")
        
        # Initialize components
        signal_generator = SignalGenerator()
        risk_manager = RiskManager(portfolio_value=CONFIG['starting_capital'])
        portfolio_manager = PortfolioManager(portfolio_value=CONFIG['starting_capital'])
        
        # Collect historical data
        historical_data = self._collect_historical_data(symbols, self.start_date, self.end_date)
        
        if not historical_data:
            logger.error("No historical data available for backtest")
            return None
        
        # Get all dates in the backtest period
        all_dates = self._get_all_dates(historical_data)
        
        # Run the backtest day by day with progress bar
        with tqdm(total=len(all_dates), desc="Backtesting") as pbar:
            for current_date in all_dates:
                # Skip dates outside our range
                if current_date < start_dt or current_date > end_dt:
                    continue
                    
                # Process each asset class
                for asset_class in symbols:
                    # Process each symbol in this asset class
                    for symbol in symbols[asset_class]:
                        # Skip if no data for this symbol
                        if symbol not in historical_data or asset_class not in historical_data[symbol]:
                            continue
                        
                        # Get data up to current date
                        symbol_data = historical_data[symbol][asset_class]
                        current_data = symbol_data[symbol_data.index <= current_date]
                        
                        if len(current_data) < 20:  # Need enough data for indicators
                            continue
                        
                        # Generate signals
                        data_with_indicators = signal_generator.calculate_indicators(current_data)
                        signals = signal_generator.generate_signals(data_with_indicators)
                        
                        # Check trading signals
                        self._check_trading_signals(symbol, asset_class, signals, current_data)
                        
                        # Get the latest signal
                        latest_signal = signals.iloc[-1]
                        
                        # Get the latest price
                        latest_price = latest_signal['close']
                        
                        # Get the signal strength factor
                        signal_strength = latest_signal['position_size_factor']
                        
                        # Get volatility (ATR)
                        volatility = latest_signal.get('atr', latest_price * 0.02)
                        
                        # Check existing positions
                        if symbol in self.portfolio['positions']:
                            # Update existing position
                            position = self.portfolio['positions'][symbol]
                            
                            # Check if signal has reversed
                            current_direction = 1 if signal_strength > 0 else -1 if signal_strength < 0 else 0
                            position_direction = 1 if position['direction'] == 'long' else -1
                            
                            # If signal has reversed strongly, exit position
                            if current_direction * position_direction < 0 and abs(signal_strength) > 0.5:
                                self._exit_position(self.portfolio, symbol, latest_price, current_date, "Signal Reversal")
                                continue
                            
                            # Check stop loss
                            if position['direction'] == 'long' and latest_price <= position['stop_loss']:
                                self._exit_position(self.portfolio, symbol, latest_price, current_date, "Stop Loss")
                                continue
                            
                            if position['direction'] == 'short' and latest_price >= position['stop_loss']:
                                self._exit_position(self.portfolio, symbol, latest_price, current_date, "Stop Loss")
                                continue
                            
                            # Update trailing stop if applicable
                            if position['direction'] == 'long' and latest_price > position['entry_price']:
                                # Calculate potential new stop loss
                                price_gain = latest_price - position['entry_price']
                                trail_percentage = 0.5  # Trail by 50% of the gain
                                new_stop = position['entry_price'] + (price_gain * trail_percentage)
                                
                                # Only update if new stop is higher than current stop
                                if new_stop > position['stop_loss']:
                                    self.portfolio['positions'][symbol]['stop_loss'] = new_stop
                            
                            elif position['direction'] == 'short' and latest_price < position['entry_price']:
                                # Calculate potential new stop loss
                                price_gain = position['entry_price'] - latest_price
                                trail_percentage = 0.5  # Trail by 50% of the gain
                                new_stop = position['entry_price'] - (price_gain * trail_percentage)
                                
                                # Only update if new stop is lower than current stop
                                if new_stop < position['stop_loss']:
                                    self.portfolio['positions'][symbol]['stop_loss'] = new_stop
                        
                        else:
                            # Check for new position
                            if abs(signal_strength) >= 0.3:  # Only take strong enough signals
                                # Calculate position size
                                position = risk_manager.calculate_position_size(
                                    symbol, signal_strength, latest_price, volatility
                                )
                                
                                if position:
                                    # Add asset class to position
                                    position['asset_class'] = asset_class
                                    
                                    # Check if we have enough cash
                                    if position['position_value'] <= self.portfolio['cash']:
                                        # Enter new position
                                        self._enter_position(self.portfolio, position, latest_price, current_date)
                
                # Update progress bar
                pbar.update(1)
                
                # Force portfolio update after each day
                self._update_portfolio_value(self.portfolio, current_date, historical_data, symbols)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(self.portfolio, CONFIG['starting_capital'])
        
        # Store results
        self.results = {
            'portfolio': self.portfolio,
            'metrics': metrics
        }
        
        logger.info(f"Backtest completed. Final portfolio value: ${self.portfolio['cash']:.2f}")
        logger.info(f"Performance metrics: {metrics}")
        
        return self.results
    
    def _collect_historical_data(self, symbols, start_date, end_date):
        """
        Collect historical data for all symbols.
        
        Args:
            symbols (dict): Dictionary of symbols by asset class
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            
        Returns:
            dict: Historical data by symbol and asset class
        """
        historical_data = {}
        
        # Convert dates to datetime
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Add buffer for indicator calculation
        buffer_start = start_dt - timedelta(days=50)
        buffer_start_str = buffer_start.strftime('%Y-%m-%d')
        
        logger.info(f"Collecting historical data from {buffer_start_str} to {end_date}")
        
        # Collect data for each asset class
        for asset_class in symbols:
            for symbol in symbols[asset_class]:
                logger.info(f"Collecting data for {symbol} ({asset_class})")
                
                try:
                    if asset_class == 'stocks':
                        data = self.data_collector.get_stock_data(
                            symbol, 
                            buffer_start_str,
                            end_date
                        )
                    
                    elif asset_class == 'crypto':
                        data = self.data_collector.get_crypto_data(
                            symbol,
                            buffer_start_str,
                            end_date
                        )
                    
                    elif asset_class == 'options':
                        # Get underlying stock data first
                        stock_data = self.data_collector.get_stock_data(
                            symbol, 
                            buffer_start_str,
                            end_date
                        )
                        
                        if not stock_data.empty:
                            # Get options chain data
                            options_data = self.data_collector.get_options_chain(
                                symbol,
                                buffer_start_str,
                                end_date,
                                OPTIONS_CONFIG['dte_range'],
                                OPTIONS_CONFIG['delta_range']
                            )
                            
                            if not options_data.empty:
                                if symbol not in historical_data:
                                    historical_data[symbol] = {}
                                historical_data[symbol]['options'] = options_data
                                historical_data[symbol]['underlying'] = stock_data
                                logger.info(f"Collected options data for {symbol}")
                    
                    # Filter to the required date range
                    if not data.empty:
                        data = data.sort_index()
                        data = data[data.index >= buffer_start]
                        data = data[data.index <= end_dt]
                        if symbol not in historical_data:
                            historical_data[symbol] = {}
                        historical_data[symbol][asset_class] = data
                        logger.info(f"Collected {len(data)} data points for {symbol}")
                    else:
                        logger.warning(f"No data available for {symbol}")
                
                except Exception as e:
                    logger.error(f"Error collecting data for {symbol}: {str(e)}")
        
        return historical_data
    
    def _get_all_dates(self, historical_data):
        """
        Get all unique dates across all historical data.
        
        Args:
            historical_data (dict): Historical data by symbol and asset class
            
        Returns:
            list: Sorted list of all unique dates
        """
        all_dates = set()
        
        for symbol, asset_classes in historical_data.items():
            for asset_class, data in asset_classes.items():
                all_dates.update(data.index)
        
        return sorted(list(all_dates))
    
    def _enter_position(self, portfolio, position, price, date):
        """
        Enter a new position in the backtest.
        
        Args:
            portfolio (dict): Portfolio state
            position (dict): Position details
            price (float): Entry price
            date (datetime): Entry date
        """
        symbol = position['symbol']
        quantity = position['quantity']
        position_value = position['position_value']
        
        # Apply commission
        commission = position_value * self.config['commission'].get(position['asset_class'], 0.001)
        position_value += commission
        
        # Check if we have enough cash
        if position_value > portfolio['cash']:
            return
        
        # Update cash
        portfolio['cash'] -= position_value
        
        # Add position to portfolio
        portfolio['positions'][symbol] = position
        
        # Record trade
        portfolio['trades'].append({
            'symbol': symbol,
            'action': 'buy' if position['direction'] == 'long' else 'sell',
            'quantity': quantity,
            'price': price,
            'value': position_value,
            'commission': commission,
            'date': date
        })
    
    def _exit_position(self, portfolio, symbol, price, date, reason):
        """
        Exit an existing position in the backtest.
        
        Args:
            portfolio (dict): Portfolio state
            symbol (str): Symbol to exit
            price (float): Exit price
            date (datetime): Exit date
            reason (str): Reason for exit
        """
        if symbol not in portfolio['positions']:
            return
        
        position = portfolio['positions'][symbol]
        quantity = position['quantity']
        entry_price = position['entry_price']
        direction = position['direction']
        
        # Calculate position value
        position_value = quantity * price
        
        # Apply commission
        commission = position_value * self.config['commission'].get(position['asset_class'], 0.001)
        
        # Calculate P&L
        if direction == 'long':
            pnl = (price - entry_price) * quantity - commission
        else:  # short
            pnl = (entry_price - price) * quantity - commission
        
        # Update cash
        portfolio['cash'] += position_value - commission
        
        # Remove position from portfolio
        del portfolio['positions'][symbol]
        
        # Record trade
        portfolio['trades'].append({
            'symbol': symbol,
            'action': 'sell' if direction == 'long' else 'buy',
            'quantity': quantity,
            'price': price,
            'value': position_value,
            'commission': commission,
            'pnl': pnl,
            'date': date,
            'reason': reason
        })
    
    def _calculate_performance_metrics(self, portfolio, initial_capital):
        """
        Calculate performance metrics for the backtest.
        
        Args:
            portfolio (dict): Portfolio state
            initial_capital (float): Initial capital
            
        Returns:
            dict: Performance metrics
        """
        # Extract portfolio history
        history = pd.DataFrame(portfolio['history'])
        
        if history.empty:
            return {}
        
        # Calculate returns
        history['daily_return'] = history['portfolio_value'].pct_change()
        
        # Calculate metrics
        total_days = len(history)
        trading_days_per_year = 252;
        
        # Total return
        total_return = (history['portfolio_value'].iloc[-1] / initial_capital) - 1;
        
        # Annualized return
        years = total_days / trading_days_per_year;
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0;
        
        # Volatility
        daily_volatility = history['daily_return'].std();
        annualized_volatility = daily_volatility * (trading_days_per_year ** 0.5);
        
        # Sharpe ratio (assuming risk-free rate of 0.02)
        risk_free_rate = 0.02;
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0;
        
        # Maximum drawdown
        history['cumulative_return'] = (1 + history['daily_return']).cumprod();
        history['cumulative_max'] = history['cumulative_return'].cummax();
        history['drawdown'] = (history['cumulative_max'] - history['cumulative_return']) / history['cumulative_max'];
        max_drawdown = history['drawdown'].max();
        
        # Win rate
        trades = pd.DataFrame(portfolio['trades']);
        if not trades.empty and 'pnl' in trades.columns:
            winning_trades = trades[trades['pnl'] > 0];
            win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0;
            
            # Average win/loss
            avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0;
            losing_trades = trades[trades['pnl'] <= 0];
            avg_loss = abs(losing_trades['pnl'].mean()) if not losing_trades.empty else 0;
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0;
            
            # Profit factor
            gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0;
            gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0;
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0;
        else:
            win_rate = 0;
            win_loss_ratio = 0;
            profit_factor = 0;
        
        # Compile metrics
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'total_trades': len(trades) if not trades.empty else 0
        };
        
        return metrics;
    
    def plot_results(self, save_path=None):
        """
        Plot the backtest results.
        
        Args:
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        if not self.results or 'portfolio' not in self.results:
            logger.error("No backtest results available to plot")
            return None
        
        portfolio = self.results['portfolio']
        
        # Extract portfolio history
        history = pd.DataFrame(portfolio['history'])
        
        if history.empty:
            logger.error("No portfolio history available to plot")
            return None
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        
        # Plot portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(history['date'], history['portfolio_value'])
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        
        # Plot drawdown
        if 'drawdown' in history.columns:
            plt.subplot(2, 1, 2)
            plt.fill_between(history['date'], 0, history['drawdown'], color='red', alpha=0.3)
            plt.title('Portfolio Drawdown')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
        
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return plt.gcf()
    
    def optimize_parameters(self, param_grid, symbols=None):
        """
        Optimize algorithm parameters using grid search.
        
        Args:
            param_grid (dict): Dictionary of parameters to optimize
            symbols (dict, optional): Dictionary of symbols by asset class
            
        Returns:
            dict: Optimization results
        """
        logger.info("Starting parameter optimization")
        
        # Default symbols if not provided
        if not symbols:
            symbols = {
                'stocks': ['AAPL', 'MSFT', 'GOOGL'],  # Reduced set for faster optimization
                'crypto': ['bitcoin', 'ethereum'],
                'options': []
            }
        
        # Track best parameters and results
        best_sharpe = -float('inf')
        best_params = {}
        all_results = []
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        # Test each parameter combination
        for params in tqdm(param_combinations, desc="Optimizing"):
            # Update configuration with these parameters
            self._update_config(params)
            
            # Run backtest with these parameters
            results = self.run_backtest(symbols=symbols)
            
            if results and 'metrics' in results:
                metrics = results['metrics']
                sharpe = metrics.get('sharpe_ratio', -float('inf'))
                
                # Track all results
                all_results.append({
                    'params': params,
                    'metrics': metrics
                })
                
                # Update best parameters if this is better
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
                    logger.info(f"New best parameters found: Sharpe = {sharpe:.4f}")
        
        # Restore original configuration
        self._restore_config()
        
        # Return optimization results
        optimization_results = {
            'best_params': best_params,
            'best_sharpe': best_sharpe,
            'all_results': all_results
        }
        
        logger.info(f"Optimization completed. Best Sharpe ratio: {best_sharpe:.4f}")
        
        return optimization_results
    
    def _generate_param_combinations(self, param_grid):
        """
        Generate all combinations of parameters for grid search.
        
        Args:
            param_grid (dict): Dictionary of parameters to optimize
            
        Returns:
            list: List of parameter dictionaries
        """
        # Helper function to generate combinations
        def _generate_combinations(grid, current_idx=0, current_params={}):
            if current_idx == len(param_names):
                return [current_params.copy()]
            
            param_name = param_names[current_idx]
            param_values = grid[param_name]
            
            combinations = []
            for value in param_values:
                current_params[param_name] = value
                combinations.extend(_generate_combinations(grid, current_idx + 1, current_params))
            
            return combinations
        
        # Get parameter names
        param_names = list(param_grid.keys())
        
        # Generate all combinations
        return _generate_combinations(param_grid)
    
    def _update_config(self, params):
        """
        Update configuration with new parameters.
        
        Args:
            params (dict): New parameters
        """
        # Store original configuration for later restoration
        self._original_config = {
            'INDICATORS': {k: v.copy() for k, v in INDICATORS.items()},
            'RISK_MANAGEMENT': RISK_MANAGEMENT.copy(),
            'SIGNAL_CLASSIFICATION': SIGNAL_CLASSIFICATION.copy()
        }
        
        # Update configuration with new parameters
        for param_name, value in params.items():
            # Parse parameter name to determine which config to update
            parts = param_name.split('.')
            
            if len(parts) == 3:
                config_name, section, key = parts
                
                if config_name == 'INDICATORS':
                    INDICATORS[section][key] = value
                elif config_name == 'RISK_MANAGEMENT':
                    if section == 'stop_loss':
                        RISK_MANAGEMENT['stop_loss'][key] = value
                    else:
                        RISK_MANAGEMENT[key] = value
                elif config_name == 'SIGNAL_CLASSIFICATION':
                    SIGNAL_CLASSIFICATION[section][key] = value
    
    def _restore_config(self):
        """Restore original configuration after optimization."""
        if hasattr(self, '_original_config'):
            # Restore original configuration
            for k, v in self._original_config['INDICATORS'].items():
                INDICATORS[k] = v.copy()
            
            for k, v in self._original_config['RISK_MANAGEMENT'].items():
                RISK_MANAGEMENT[k] = v
            
            for k, v in self._original_config['SIGNAL_CLASSIFICATION'].items():
                SIGNAL_CLASSIFICATION[k] = v

    def _check_trading_signals(self, symbol, asset_class, signals, current_data):
        """Enhanced signal checking with options support."""
        # Get the latest signal
        latest_signal = signals.iloc[-1]
        
        # Additional technical filters
        if not self._validate_signal(latest_signal):
            return
        
        if asset_class == 'options':
            self._process_options_signals(symbol, latest_signal, current_data)
        else:
            # Process stocks and crypto
            signal_strength = latest_signal['position_size_factor']
            latest_price = latest_signal['close']
            volatility = latest_signal.get('atr', latest_price * 0.02)
            
            # Check if signal is strong enough
            if abs(signal_strength) >= 0.4:  # Higher threshold for non-options
                # Calculate position size
                position = self._calculate_position(
                    symbol,
                    asset_class,
                    signal_strength,
                    latest_price,
                    volatility,
                    current_data
                )
                
                if position:
                    self._enter_position(
                        self.portfolio,
                        position,
                        latest_price,
                        latest_signal.name  # Use timestamp as date
                    )

    def _calculate_position(self, symbol, asset_class, signal_strength, price, volatility, data):
        """Calculate position details based on signal and risk parameters."""
        direction = 'long' if signal_strength > 0 else 'short'
        
        # Calculate base position size using risk parameters
        risk_per_trade = RISK_MANAGEMENT['risk_per_trade']
        portfolio_value = self.portfolio['cash'] + sum(
            pos['position_value'] for pos in self.portfolio['positions'].values()
        )
        max_position_value = portfolio_value * RISK_MANAGEMENT['max_position_size']
        
        # Calculate stop loss level
        atr = volatility
        stop_distance = atr * RISK_MANAGEMENT['stop_loss']['atr_multiplier']
        stop_loss = price - stop_distance if direction == 'long' else price + stop_distance
        
        # Calculate position size based on stop loss
        risk_amount = portfolio_value * risk_per_trade
        price_risk = abs(price - stop_loss)
        quantity = risk_amount / price_risk
        position_value = quantity * price
        
        # Adjust if position value exceeds max position size
        if position_value > max_position_value:
            quantity = max_position_value / price
            position_value = max_position_value
        
        # Create position object
        position = {
            'symbol': symbol,
            'asset_class': asset_class,
            'direction': direction,
            'quantity': quantity,
            'entry_price': price,
            'stop_loss': stop_loss,
            'position_value': position_value,
            'risk_amount': risk_amount
        }
        
        return position
    
    def _validate_signal(self, signal):
        """Additional signal validation."""
        # Minimum strength threshold
        if abs(signal['position_size_factor']) < 0.4:  # Increased from 0.3
            return False

        # Volume confirmation
        if signal['volume'] < signal['volume_sma']:
            return False

        # Trend confirmation
        if not (
            (signal['position_size_factor'] > 0 and 
             signal['close'] > signal['sma_50'] and 
             signal['macd_histogram'] > 0) or
            (signal['position_size_factor'] < 0 and 
             signal['close'] < signal['sma_50'] and 
             signal['macd_histogram'] < 0)
        ):
            return False

        return True

    def _process_options_signals(self, symbol, signal, data):
        """Process signals for options trading."""
        if abs(signal['position_size_factor']) >= 0.5:  # Strong signal threshold
            direction = 'calls' if signal['position_size_factor'] > 0 else 'puts'
            
            # Select appropriate options contracts
            viable_contracts = self._filter_options_contracts(
                data['options'],
                direction,
                OPTIONS_CONFIG['delta_range'][direction],
                OPTIONS_CONFIG['dte_range']
            )
            
            if not viable_contracts.empty:
                # Select best contract based on liquidity and price
                best_contract = self._select_best_contract(viable_contracts)
                if best_contract is not None:
                    self._enter_options_position(best_contract, signal['position_size_factor'])

    def _update_portfolio_value(self, portfolio, current_date, historical_data, symbols):
        """Update portfolio value and history at the end of each day."""
        portfolio_value = portfolio['cash']
        
        # Calculate positions value
        for symbol, position in list(portfolio['positions'].items()):
            # Find latest price for the symbol
            price_found = False
            for asset_class in symbols:
                if symbol in symbols[asset_class]:
                    if symbol in historical_data and asset_class in historical_data[symbol]:
                        symbol_data = historical_data[symbol][asset_class]
                        if current_date in symbol_data.index:
                            latest_price = symbol_data.loc[current_date, 'close']
                            position_value = position['quantity'] * latest_price
                            portfolio_value += position_value
                            price_found = True
                            break
            
            # If no price found, use last known price or exit position
            if not price_found:
                if 'last_price' in position:
                    portfolio_value += position['quantity'] * position['last_price']
                else:
                    logger.warning(f"No price data for {symbol} on {current_date}, exiting position")
                    self._exit_position(portfolio, symbol, position['entry_price'], current_date, "No Data")
        
        # Record portfolio value history
        portfolio['history'].append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'cash': portfolio['cash'],
            'positions_count': len(portfolio['positions'])
        })

# Example usage
if __name__ == "__main__":
    # Initialize backtester
    backtester = Backtester()
    
    # Run a simple backtest
    results = backtester.run_backtest()
    
    if results:
        # Plot results
        backtester.plot_results(save_path=r"C:\Users\alaqu\Desktop\CompSci\TradingAlgoApp\backtest_results.png")
        
        # Print performance metrics
        metrics = results['metrics']
        print("\nPerformance Metrics:")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Total Trades: {metrics['total_trades']}")
    
    # Example parameter optimization (commented out as it can be time-consuming)
    """
    param_grid = {
        'INDICATORS.trend.sma.periods[0]': [10, 20, 30],
        'INDICATORS.momentum.rsi.period': [7, 14, 21],
        'RISK_MANAGEMENT.max_position_size': [0.1, 0.2, 0.3],
        'RISK_MANAGEMENT.stop_loss.atr_multiplier': [1.5, 2.0, 2.5]
    }
    
    optimization_results = backtester.optimize_parameters(param_grid)
    
    print("\nBest Parameters:")
    for param, value in optimization_results['best_params'].items():
        print(f"{param}: {value}")
    """
