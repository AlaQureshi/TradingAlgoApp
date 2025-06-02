#!/usr/bin/env python3

import sys
import os
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.append(r'C:\Users\alaqu\Desktop\CompSci\TradingAlgoApp\updatedcode')

# Import the backtester
from backtester import Backtester

def main():
    print("Starting backtesting process...")
    
    # Set up output paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    chart_path = os.path.join(results_dir, "backtest_results.png")
    metrics_path = os.path.join(results_dir, "backtest_metrics.txt")
    optimization_path = os.path.join(results_dir, "optimization_results.txt")

    # Initialize backtester
    backtester = Backtester()
    
    # Run backtest
    print("Running backtest on historical data...")
    results = backtester.run_backtest()
    
    if not results:
        print("Backtesting failed to produce results.")
        return
    
    # Plot and save results
    print("Generating performance charts...")
    backtester.plot_results(save_path=chart_path)
    
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
    
    # Save metrics to file
    with open(metrics_path, 'w') as f:
        f.write("Trading Algorithm AI - Backtest Results\n")
        f.write("======================================\n\n")
        f.write(f"Total Return: {metrics['total_return']:.2%}\n")
        f.write(f"Annualized Return: {metrics['annualized_return']:.2%}\n")
        f.write(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}\n")
        f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}\n")
        f.write(f"Max Drawdown: {metrics['max_drawdown']:.2%}\n")
        f.write(f"Win Rate: {metrics['win_rate']:.2%}\n")
        f.write(f"Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}\n")
        f.write(f"Profit Factor: {metrics['profit_factor']:.2f}\n")
        f.write(f"Total Trades: {metrics['total_trades']}\n")
    
    print(f"\nBacktest results saved to: {metrics_path}")
    print(f"Performance chart saved to: {chart_path}")
    
    # Run parameter optimization (limited scope for demonstration)
    print("\nRunning parameter optimization...")
    param_grid = {
        'INDICATORS.trend.sma.periods[0]': [10, 20],
        'INDICATORS.momentum.rsi.period': [7, 14],
        'RISK_MANAGEMENT.max_position_size': [0.1, 0.2],
        'RISK_MANAGEMENT.stop_loss.atr_multiplier': [1.5, 2.0]
    }
    
    # Use a smaller symbol set for optimization to speed up the process
    optimization_symbols = {
        'stocks': ['AAPL', 'MSFT'],
        'crypto': ['bitcoin'],
        'options': []
    }
    
    optimization_results = backtester.optimize_parameters(param_grid, symbols=optimization_symbols)
    
    # Save optimization results
    with open(optimization_path, 'w') as f:
        f.write("Trading Algorithm AI - Parameter Optimization Results\n")
        f.write("=================================================\n\n")
        f.write(f"Best Sharpe Ratio: {optimization_results['best_sharpe']:.4f}\n\n")
        f.write("Best Parameters:\n")
        for param, value in optimization_results['best_params'].items():
            f.write(f"{param}: {value}\n")
    
    print(f"\nOptimization results saved to: {optimization_path}")
    print("Backtesting and optimization completed successfully.")

if __name__ == "__main__":
    main()
