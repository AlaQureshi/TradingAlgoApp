"""
Simplified backtesting module for the Trading Algorithm AI.
Provides a more efficient version for quicker results.
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

# Import modules
from data_collector import DataCollector
from signal_generator import SignalGenerator
from risk_manager import RiskManager
from portfolio_manager import PortfolioManager
from config import CONFIG, BACKTEST

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backtest_simplified.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('backtest_simplified')

class SimplifiedBacktester:
    """
    A simplified version of the backtester for quicker results.
    """
    
    def __init__(self):
        """Initialize the simplified backtester."""
        self.config = BACKTEST
        self.data_collector = DataCollector()
        self.results = {}
        logger.info("SimplifiedBacktester initialized")
    
    def run_backtest(self, start_date=None, end_date=None, symbols=None):
        """
        Run a simplified backtest with reduced scope for quicker results.
        
        Args:
            start_date (str, optional): Start date for backtest (YYYY-MM-DD)
            end_date (str, optional): End date for backtest (YYYY-MM-DD)
            symbols (dict, optional): Dictionary of symbols by asset class
            
        Returns:
            dict: Backtest results
        """
        # Use configured dates if not provided
        start_date = start_date or self.config['start_date']
        end_date = end_date or self.config['end_date']
        
        # Use a smaller symbol set for faster processing
        if not symbols:
            symbols = {
                'stocks': ['AAPL', 'MSFT'],  # Reduced set
                'crypto': ['bitcoin'],       # Reduced set
                'options': []                # Skip options for simplicity
            }
        
        logger.info(f"Starting simplified backtest from {start_date} to {end_date}")
        
        # Initialize portfolio
        initial_capital = CONFIG['starting_capital']
        portfolio_value = initial_capital
        portfolio = {
            'cash': initial_capital,
            'positions': {},
            'history': [],
            'trades': []
        }
        
        # Initialize components
        signal_generator = SignalGenerator()
        risk_manager = RiskManager(portfolio_value=initial_capital)
        portfolio_manager = PortfolioManager(portfolio_value=initial_capital)
        
        # Collect historical data (with reduced date range if needed)
        # Calculate a shorter date range if needed
        if start_date and end_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            days_diff = (end_dt - start_dt).days
            
            # If more than 90 days, reduce to last 90 days for speed
            if days_diff > 90:
                start_dt = end_dt - timedelta(days=90)
                start_date = start_dt.strftime('%Y-%m-%d')
                logger.info(f"Reduced backtest period to 90 days: {start_date} to {end_date}")
        
        historical_data = self._collect_historical_data(symbols, start_date, end_date)
        
        if not historical_data:
            logger.error("No historical data available for backtest")
            return None
        
        # Get all dates in the backtest period
        all_dates = self._get_all_dates(historical_data)
        
        # Run the backtest day by day
        logger.info(f"Running simplified backtest over {len(all_dates)} trading days")
        
        for current_date in all_dates:
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
                    
                    # Get the latest signal
                    latest_signal = signals.iloc[-1]
                    
                    # Get the latest price
                    latest_price = latest_signal['close']
                    
                    # Get the signal strength factor
                    signal_strength = latest_signal['position_size_factor']
                    
                    # Get volatility (ATR)
                    volatility = latest_signal.get('atr', latest_price * 0.02)
                    
                    # Check existing positions
                    if symbol in portfolio['positions']:
                        # Update existing position
                        position = portfolio['positions'][symbol]
                        
                        # Check if signal has reversed
                        current_direction = 1 if signal_strength > 0 else -1 if signal_strength < 0 else 0
                        position_direction = 1 if position['direction'] == 'long' else -1
                        
                        # If signal has reversed strongly, exit position
                        if current_direction * position_direction < 0 and abs(signal_strength) > 0.5:
                            self._exit_position(portfolio, symbol, latest_price, current_date, "Signal Reversal")
                            continue
                        
                        # Check stop loss
                        if position['direction'] == 'long' and latest_price <= position['stop_loss']:
                            self._exit_position(portfolio, symbol, latest_price, current_date, "Stop Loss")
                            continue
                        
                        if position['direction'] == 'short' and latest_price >= position['stop_loss']:
                            self._exit_position(portfolio, symbol, latest_price, current_date, "Stop Loss")
                            continue
                        
                        # Update trailing stop if applicable
                        if position['direction'] == 'long' and latest_price > position['entry_price']:
                            # Calculate potential new stop loss
                            price_gain = latest_price - position['entry_price']
                            trail_percentage = 0.5  # Trail by 50% of the gain
                            new_stop = position['entry_price'] + (price_gain * trail_percentage)
                            
                            # Only update if new stop is higher than current stop
                            if new_stop > position['stop_loss']:
                                portfolio['positions'][symbol]['stop_loss'] = new_stop
                        
                        elif position['direction'] == 'short' and latest_price < position['entry_price']:
                            # Calculate potential new stop loss
                            price_gain = position['entry_price'] - latest_price
                            trail_percentage = 0.5  # Trail by 50% of the gain
                            new_stop = position['entry_price'] - (price_gain * trail_percentage)
                            
                            # Only update if new stop is lower than current stop
                            if new_stop < position['stop_loss']:
                                portfolio['positions'][symbol]['stop_loss'] = new_stop
                    
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
                                if position['position_value'] <= portfolio['cash']:
                                    # Enter new position
                                    self._enter_position(portfolio, position, latest_price, current_date)
            
            # Calculate portfolio value at end of day
            portfolio_value = portfolio['cash']
            for symbol, position in portfolio['positions'].items():
                # Get the latest price for this symbol
                for asset_class in symbols:
                    if symbol in symbols[asset_class]:
                        symbol_data = historical_data[symbol][asset_class]
                        if current_date in symbol_data.index:
                            latest_price = symbol_data.loc[current_date, 'close']
                            position_value = position['quantity'] * latest_price
                            portfolio_value += position_value
                            break
            
            # Record portfolio value history
            portfolio['history'].append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': portfolio['cash'],
                'positions_count': len(portfolio['positions'])
            })
            
            # Update risk manager and portfolio manager
            risk_manager.update_portfolio_value(portfolio_value)
            portfolio_manager.update_portfolio_value(portfolio_value)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(portfolio, initial_capital)
        
        # Store results
        self.results = {
            'portfolio': portfolio,
            'metrics': metrics
        }
        
        logger.info(f"Simplified backtest completed. Final portfolio value: ${portfolio_value:.2f}")
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
        
        try:
            for asset_class in symbols:
                for symbol in symbols[asset_class]:
                    logger.info(f"Collecting data for {symbol} ({asset_class})")
                    
                    try:
                        if asset_class == 'stocks':
                            data = self.data_collector.get_stock_data(symbol, start_date, end_date)
                        elif asset_class == 'crypto':
                            data = self.data_collector.get_crypto_data(symbol, start_date, end_date)
                        else:
                            continue
                        
                        # Verify required columns exist
                        required_columns = ['open', 'high', 'low', 'close', 'volume']
                        if not data.empty and all(col in data.columns for col in required_columns):
                            if symbol not in historical_data:
                                historical_data[symbol] = {}
                            historical_data[symbol][asset_class] = data
                            logger.info(f"Collected {len(data)} data points for {symbol}")
                        else:
                            logger.warning(f"Missing required columns for {symbol}")
                            
                    except Exception as e:
                        logger.error(f"Error collecting data for {symbol}: {str(e)}")
                        continue
                        
            return historical_data
            
        except Exception as e:
            logger.error(f"Error in data collection: {str(e)}")
            return {}
    
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
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            except Exception as e:
                logger.error(f"Error saving plot to {save_path}: {str(e)}")
                # Try saving to current directory instead
                fallback_path = os.path.join(os.getcwd(), "backtest_results.png")
                try:
                    plt.savefig(fallback_path)
                    logger.info(f"Plot saved to fallback location: {fallback_path}")
                except Exception as e2:
                    logger.error(f"Error saving plot to fallback location: {str(e2)}")
        
        return plt.gcf()

# Example usage
if __name__ == "__main__":
    # Initialize backtester
    backtester = SimplifiedBacktester()
    
    # Run a simplified backtest
    results = backtester.run_backtest()
    
    if results:
        # Plot results
        backtester.plot_results(save_path=r"C:\Users\alaqu\Desktop\CompSci\TradingAlgoApp\simplified_backtest_results.png")
        
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
