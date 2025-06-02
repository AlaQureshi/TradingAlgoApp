import yfinance as yf
import logging
from datetime import datetime

class YahooDataFeed:
    def __init__(self, symbols):
        self.symbols = symbols
        self.logger = logging.getLogger(__name__)

    def get_latest_price(self, symbol):
        """Get the latest price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d')
            
            if not data.empty:
                return {
                    'symbol': symbol,
                    'price': float(data['Close'].iloc[-1]),
                    'size': int(data['Volume'].iloc[-1]),
                    'timestamp': datetime.now()
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting data for {symbol}: {e}")
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
