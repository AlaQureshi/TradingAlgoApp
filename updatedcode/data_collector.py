"""
Main data collection module for the Trading Algorithm AI.
Handles fetching and processing data from various sources.
"""

import sys
import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta, date # Added date for better date handling
import logging

# REMOVE THIS LINE: sys.path.append('/opt/.manus/.sandbox-runtime')
# This was for the missing 'data_api' module and is no longer relevant if we're not using it.

# Import configuration
from config import DATA_SOURCES, CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_collector')

class DataCollector:
    """
    Handles data collection from various sources for the trading algorithm.
    Supports stocks, cryptocurrencies, and options data (all via Polygon.io).
    """

    def __init__(self):
        """Initialize the data collector with configuration settings."""
        self.config = DATA_SOURCES
        self.api_keys = {}
        self.setup_api_keys() # Call this after self.config is set
        logger.info("DataCollector initialized")

    def setup_api_keys(self):
        """Set up API keys from environment or config."""
        polygon_api_key_env = os.environ.get('POLYGON_API_KEY', None)
        polygon_api_key_config_stocks = self.config.get('stocks', {}).get('api_key', None)
        polygon_api_key_config_options = self.config.get('options', {}).get('api_key', None)
        polygon_api_key_config_crypto = self.config.get('crypto', {}).get('api_key', None)

        self.api_keys = {
            'polygon': polygon_api_key_env or polygon_api_key_config_stocks or polygon_api_key_config_options or polygon_api_key_config_crypto
        }

        if not self.api_keys['polygon']:
            logger.warning("Polygon is configured as a provider, but POLYGON_API_KEY was not found in environment or config. Polygon.io requests will likely fail.")
        logger.info("API keys configured")

    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch stock data using Polygon.io API.
        'start_date' and 'end_date' are expected in 'YYYY-MM-DD' format.
        """
        # Check if stocks configuration is for Polygon
        stock_config = self.config.get('stocks', {})
        if stock_config.get('provider') != 'polygon':
            logger.error("Stock provider in config is not 'polygon'")
            return pd.DataFrame()

        if not self.api_keys['polygon']:
            logger.error("Polygon API key is required but not configured")
            return pd.DataFrame()

        try:
            base_url = stock_config['base_url']
            endpoint_template = stock_config['endpoints']['chart']
            
            # Format dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Set up API parameters
            url = f"{base_url}{endpoint_template.format(ticker=symbol.upper(), multiplier=1, timespan='day', from_=start_dt.strftime('%Y-%m-%d'), to_=end_dt.strftime('%Y-%m-%d'))}"
            params = {'apiKey': self.api_keys['polygon'], 'adjusted': 'true', 'sort': 'asc', 'limit': 50000}

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get('resultsCount', 0) == 0 or not data.get('results'):
                logger.warning(f"No stock data found for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(data['results'])
            df.rename(columns={
                'o': 'open', 
                'h': 'high', 
                'l': 'low', 
                'c': 'close', 
                'v': 'volume', 
                't': 'timestamp'
            }, inplace=True)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]

            logger.info(f"Successfully fetched {len(df)} stock data points for {symbol} from Polygon.io")
            return df

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error fetching Polygon stock data for {symbol}: {http_err} - Response: {response.text if 'response' in locals() and hasattr(response, 'text') else 'N/A'}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"General error fetching Polygon stock data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_stock_insights(self, symbol):
        """
        Fetch stock insights (ticker details) using Polygon.io API.
        """
        stock_config = self.config.get('stocks', {})
        if stock_config.get('provider') != 'polygon':
            logger.error(f"Stock provider in config is '{stock_config.get('provider')}', not 'polygon'. This method requires Polygon for insights.")
            return {}
        if not self.api_keys['polygon']:
            logger.error("Polygon API key is required for stock insights but not configured.")
            return {}

        try:
            base_url = stock_config['base_url']
            # Uses the 'insights' key which should map to Polygon's ticker_details endpoint
            endpoint_template = stock_config['endpoints']['insights']
            url = f"{base_url}{endpoint_template.format(ticker=symbol.upper())}"
            params = {'apiKey': self.api_keys['polygon']}

            logger.debug(f"Requesting Polygon stock insights: {url}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get('status') == 'OK' and 'results' in data:
                logger.info(f"Successfully fetched stock insights for {symbol} from Polygon.io")
                return data['results']
            else:
                logger.warning(f"Could not fetch stock insights for {symbol} from Polygon.io. Status: {data.get('status', 'N/A')}, Response: {data}")
                return {}

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error fetching Polygon stock insights for {symbol}: {http_err} - Response: {response.text if 'response' in locals() and hasattr(response, 'text') else 'N/A'}")
            return {}
        except Exception as e:
            logger.error(f"General error fetching Polygon stock insights for {symbol}: {str(e)}")
            return {}

    def get_crypto_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch cryptocurrency data using Polygon.io API.
        'symbol' should be in Polygon format, e.g., 'X:BTCUSD'
        """
        crypto_config = self.config.get('crypto', {})
        if crypto_config.get('provider') != 'polygon':
            logger.error("Crypto provider in config is not 'polygon'")
            return pd.DataFrame()

        if not self.api_keys['polygon']:
            logger.error("Polygon API key is required but not configured")
            return pd.DataFrame()

        try:
            base_url = crypto_config['base_url']
            endpoint_template = crypto_config['endpoints']['chart']

            # Format dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            url = f"{base_url}{endpoint_template.format(ticker=symbol.upper(), multiplier=1, timespan='day', from_=start_dt.strftime('%Y-%m-%d'), to_=end_dt.strftime('%Y-%m-%d'))}"
            params = {'apiKey': self.api_keys['polygon'], 'adjusted': 'true', 'sort': 'asc', 'limit': 50000}

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get('resultsCount', 0) == 0 or not data.get('results'):
                logger.warning(f"No crypto data found for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(data['results'])
            df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                't': 'timestamp'
            }, inplace=True)

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]

            logger.info(f"Successfully fetched {len(df)} crypto data points for {symbol} from Polygon.io")
            return df

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error fetching Polygon crypto data for {symbol}: {http_err} - Response: {response.text if 'response' in locals() and hasattr(response, 'text') else 'N/A'}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"General error fetching Polygon crypto data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_options_data(self, underlying_symbol, expiration_date=None, strike=None):
        """
        Fetch options contract list using Polygon API.
        Returns price data for the first contract found if successful.
        """
        options_config = self.config.get('options', {})
        if options_config.get('provider') != 'polygon':
            logger.error(f"Options provider in config is '{options_config.get('provider')}', not 'polygon'. This method is for Polygon.")
            return pd.DataFrame()
        if not self.api_keys['polygon']:
            logger.error("Polygon API key is required for options data but not configured.")
            return pd.DataFrame()

        try:
            base_url = options_config['base_url']
            endpoint_template = options_config['endpoints']['options_contracts'] # Assumes 'options_contracts' key in config
            url = f"{base_url}{endpoint_template}"
            params = {
                'underlying_ticker': underlying_symbol.upper(),
                'apiKey': self.api_keys['polygon'],
                'limit': 100 # Fetch a list of contracts
            }
            if expiration_date: params['expiration_date'] = expiration_date
            if strike: params['strike_price'] = strike

            logger.debug(f"Requesting Polygon options contracts: {url} for {underlying_symbol}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get('status') == 'OK' and 'results' in data and data['results']:
                contracts_df = pd.DataFrame(data['results'])
                logger.info(f"Successfully fetched {len(contracts_df)} options contracts for {underlying_symbol} from Polygon.io")
                if not contracts_df.empty and 'ticker' in contracts_df.columns:
                    option_ticker = contracts_df.iloc[0]['ticker']
                    logger.info(f"Fetching price data for the first option contract: {option_ticker}")
                    return self.get_options_price_data(option_ticker, days=30) # Default 30 days for example
                return contracts_df
            else:
                logger.warning(f"No options contracts found for {underlying_symbol} from Polygon.io. Status: {data.get('status', 'N/A')}, Response: {data}")
                return pd.DataFrame()

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error fetching Polygon options contracts for {underlying_symbol}: {http_err} - Response: {response.text if 'response' in locals() and hasattr(response, 'text') else 'N/A'}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"General error fetching Polygon options data for {underlying_symbol}: {str(e)}")
            return pd.DataFrame()

    def get_options_price_data(self, option_ticker, days=30):
        """
        Fetch historical price data for a specific options contract using Polygon.
        """
        options_config = self.config.get('options', {})
        if options_config.get('provider') != 'polygon':
            logger.error(f"Options provider in config is '{options_config.get('provider')}', not 'polygon' for price data.")
            return pd.DataFrame()
        if not self.api_keys['polygon']:
            logger.error("Polygon API key is required for options price data but not configured.")
            return pd.DataFrame()

        try:
            base_url = options_config['base_url']
            endpoint_template = options_config['endpoints']['options_aggregates'] # Assumes 'options_aggregates' key in config

            end_dt = datetime.combine(date.today(), datetime.min.time())
            start_dt = end_dt - timedelta(days=days)
            from_date_str = start_dt.strftime('%Y-%m-%d')
            to_date_str = end_dt.strftime('%Y-%m-%d')

            multiplier = 1
            timespan = 'day'

            url = f"{base_url}{endpoint_template.format(ticker=option_ticker, multiplier=multiplier, timespan=timespan, from_=from_date_str, to_=to_date_str)}"
            params = {'apiKey': self.api_keys['polygon'], 'adjusted': 'true', 'sort': 'asc', 'limit': 50000}

            logger.debug(f"Requesting Polygon options price data: {url}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get('resultsCount', 0) > 0 and 'results' in data:
                df = pd.DataFrame(data['results'])
                df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume', 't': 'timestamp'}, inplace=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume']]
                logger.info(f"Successfully fetched {len(df)} options price data points for {option_ticker}")
                return df
            else:
                logger.warning(f"No price data found for options contract {option_ticker} from Polygon.io. Response: {data}")
                return pd.DataFrame()

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error fetching Polygon options price data for {option_ticker}: {http_err} - Response: {response.text if 'response' in locals() and hasattr(response, 'text') else 'N/A'}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"General error fetching Polygon options price data for {option_ticker}: {str(e)}")
            return pd.DataFrame()

    def get_options_chain(self, symbol, start_date, end_date, dte_range, delta_range):
        """Collect options chain data from Polygon."""
        try:
            base_url = DATA_SOURCES['options']['base_url']
            api_key = DATA_SOURCES['options']['api_key']
            
            # Get options contracts
            contracts_endpoint = DATA_SOURCES['options']['endpoints']['options_contracts']
            contracts_url = f"{base_url}{contracts_endpoint}"
            
            params = {
                'underlying_ticker': symbol,
                'contract_type': 'call,put',
                'expired': False,
                'limit': 1000,
                'apiKey': api_key
            }
            
            response = requests.get(contracts_url, params=params)
            if response.status_code == 200:
                contracts_data = response.json()
                
                # Filter contracts based on DTE and delta
                valid_contracts = self._filter_contracts(
                    contracts_data['results'],
                    dte_range,
                    delta_range
                )
                
                # Get historical data for valid contracts
                options_data = self._get_contracts_history(
                    valid_contracts,
                    start_date,
                    end_date
                )
                
                return options_data
                
        except Exception as e:
            logger.error(f"Error fetching options data: {str(e)}")
            return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Ensure your config.py has DATA_SOURCES["stocks"]["provider"] = "polygon"
    # and the API key is available via config or environment variable.
    print("Testing DataCollector with Polygon for stocks...")
    collector = DataCollector()

    stock_data = collector.get_stock_data('AAPL', interval='1d', range='1mo')
    if not stock_data.empty:
        print("\nAAPL Stock Data Sample (Polygon):")
        print(stock_data.head())
    else:
        print("\nFailed to get AAPL stock data from Polygon.")

    # stock_insights = collector.get_stock_insights('MSFT')
    # if stock_insights:
    #     print("\nMSFT Stock Insights Sample (Polygon):")
    #     # Print a snippet of the insights
    #     print(json.dumps(stock_insights, indent=2, default=str)[:500] + "...")
    # else:
    #     print("\nFailed to get MSFT stock insights from Polygon.")