import logging
from datetime import datetime
import yfinance as yf

class Trader:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.paper_trading = config.get('paper_trading', True)
        self.positions = {}
        self.starting_capital = config.get('starting_capital', 10000)
        self.current_capital = self.starting_capital
    
    def get_current_price(self, symbol):
        """Get current price from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d')
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
        return None

    def place_order(self, symbol, quantity, side, order_type='MARKET'):
        """Simulate order placement in paper trading mode"""
        try:
            price = self.get_current_price(symbol)
            
            if not price:
                self.logger.error(f"Could not get price for {symbol}")
                return None
            
            # Create simulated order
            order = {
                'symbol': symbol,
                'quantity': quantity,
                'side': side,
                'type': order_type,
                'executed_price': price,
                'timestamp': datetime.now()
            }
            
            # Update positions for paper trading
            self._update_paper_positions(order)
            
            self.logger.info(f"Order placed: {order}")
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    def _update_paper_positions(self, order):
        """Update paper trading positions"""
        symbol = order['symbol']
        qty = order['quantity'] * (1 if order['side'] == 'BUY' else -1)
        
        if symbol in self.positions:
            self.positions[symbol]['quantity'] += qty
            if self.positions[symbol]['quantity'] == 0:
                del self.positions[symbol]
        else:
            self.positions[symbol] = {
                'quantity': qty,
                'entry_price': order['executed_price']
            }
    
    def get_positions(self):
        """Get current positions"""
        return self.positions.copy()

    def get_account_info(self):
        """Get simulated account information"""
        total_value = sum(
            pos['quantity'] * self.get_current_price(sym) 
            for sym, pos in self.positions.items()
        )
        return {
            'total_value': total_value,
            'positions': len(self.positions)
        }
