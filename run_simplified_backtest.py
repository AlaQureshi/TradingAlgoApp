#!/usr/bin/env python3

import sys
import os
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.append(r'C:\Users\alaqu\Desktop\CompSci\TradingAlgoApp\updatedcode')

# Import the simplified backtester
from simplified_backtester import SimplifiedBacktester

def main():
    print("Starting simplified backtesting process...")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up output paths
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    chart_path = os.path.join(results_dir, "simplified_backtest_results.png")
    metrics_path = os.path.join(results_dir, "simplified_backtest_metrics.txt")
    
    # Initialize simplified backtester
    backtester = SimplifiedBacktester()
    
    # Run simplified backtest
    print("Running simplified backtest on historical data...")
    results = backtester.run_backtest()
    
    if not results:
        print("Simplified backtesting failed to produce results.")
        return
    
    # Plot and save results
    print("Generating performance charts...")
    try:
        backtester.plot_results(save_path=chart_path)
    except Exception as e:
        print(f"Error saving performance chart: {str(e)}")
    
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
    try:
        with open(metrics_path, 'w') as f:
            f.write("Trading Algorithm AI - Simplified Backtest Results\n")
            f.write("==============================================\n\n")
            f.write(f"Total Return: {metrics['total_return']:.2%}\n")
            f.write(f"Annualized Return: {metrics['annualized_return']:.2%}\n")
            f.write(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}\n")
            f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}\n")
            f.write(f"Max Drawdown: {metrics['max_drawdown']:.2%}\n")
            f.write(f"Win Rate: {metrics['win_rate']:.2%}\n")
            f.write(f"Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}\n")
            f.write(f"Profit Factor: {metrics['profit_factor']:.2f}\n")
            f.write(f"Total Trades: {metrics['total_trades']}\n")
        print(f"\nSimplified backtest results saved to: {metrics_path}")
        print(f"Performance chart saved to: {chart_path}")
    except Exception as e:
        print(f"Error saving metrics: {str(e)}")
    
    print("Simplified backtesting completed successfully.")

if __name__ == "__main__":
    main()
