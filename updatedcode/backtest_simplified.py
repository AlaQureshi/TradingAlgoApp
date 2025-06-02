import pandas as pd

class SimplifiedBacktester:
    def __init__(self, assets, start_date, end_date, data_collector, logger):
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        self.data_collector = data_collector
        self.logger = logger

    def run_backtest(self):
        try:
            self.logger.info(f"Starting simplified backtest from {self.start_date} to {self.end_date}")
            
            # Initialize empty dictionary to store collected data
            collected_data = {}
            
            # Collect data for each asset
            for symbol, asset_type in self.assets.items():
                self.logger.info(f"Collecting data for {symbol} ({asset_type})")
                
                if asset_type == "stocks":
                    data = self.data_collector.get_stock_data(symbol, self.start_date, self.end_date)
                elif asset_type == "crypto":
                    data = self.data_collector.get_crypto_data(symbol, self.start_date, self.end_date)
                
                if not data.empty:
                    collected_data[symbol] = data
                else:
                    self.logger.warning(f"No data available for {symbol}")
            
            if not collected_data:
                raise ValueError("No historical data available for backtest")
                
            # Continue with backtest using collected_data
            # ...existing code...
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            return None