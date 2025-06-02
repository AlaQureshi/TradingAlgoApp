from datetime import datetime, time
import pytz

class MarketHours:
    @staticmethod
    def is_market_open():
        """Check if US market is currently open"""
        et_tz = pytz.timezone('US/Eastern')
        now = datetime.now(et_tz)
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False, "Market is closed (Weekend)"
            
        market_open = time(9, 30)  # 9:30 AM ET
        market_close = time(16, 0)  # 4:00 PM ET
        current_time = now.time()
        
        if market_open <= current_time < market_close:
            return True, "Market is open"
        elif time(4, 0) <= current_time < market_open:
            return True, "Pre-market hours"
        elif market_close <= current_time < time(20, 0):
            return True, "After-hours trading"
        
        return False, "Market is closed"
