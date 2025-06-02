import requests
import logging
import time
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class PolygonRESTClient:
    def __init__(self, api_key, symbols):
        self.api_key = api_key
        self.symbols = symbols
        self.base_url = "https://api.polygon.io/v2"
        self.logger = logging.getLogger(__name__)
        self.session = self._setup_session()
        self.last_request_time = 0
        self.min_request_interval = 12  # Free tier: 5 requests per minute

    def _setup_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
        return session

    def _rate_limit(self):
        """Implement rate limiting for free tier"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def get_latest_price(self, symbol):
        """Get the latest price using daily open/close endpoint (free tier)"""
        try:
            self._rate_limit()
            
            # Get previous trading day
            today = datetime.now()
            date = (today - timedelta(days=1)).strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/aggs/ticker/{symbol}/prev"
            params = {
                'apiKey': self.api_key,
                'adjusted': 'true'
            }
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data['resultsCount'] > 0:
                    result = data['results'][0]
                    return {
                        'symbol': symbol,
                        'price': result['c'],  # Closing price
                        'size': result['v'],   # Volume
                        'timestamp': datetime.now()
                    }
            else:
                self.logger.warning(f"Error fetching {symbol}: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error getting price data for {symbol}: {e}")
        return None

    def get_all_latest_trades(self):
        """Get latest prices for all symbols"""
        trades = []
        for symbol in self.symbols:
            trade = self.get_latest_price(symbol)
            if trade:
                trades.append(trade)
                self.logger.info(f"Retrieved {symbol} price: ${trade['price']:.2f}")
        return trades
