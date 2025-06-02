import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DelayedDataStrategy:
    def __init__(self, config):
        self.config = config
        self.min_holding = config.get('MIN_HOLDING_PERIOD', 30)
        self.delay = config.get('DELAY_MINUTES', 15)
        
    def analyze_trend(self, data):
        """Analyze longer-term trends suitable for delayed data"""
        df = data.copy()
        
        # Add momentum indicators
        df['trend'] = df['price'].diff(self.min_holding).fillna(0)
        df['volume_ma'] = df['size'].rolling(window=20).mean()
        df['volume_trend'] = df['size'] > df['volume_ma']
        
        # Detect strong trends that persist beyond data delay
        strong_trend = (df['trend'].abs() > df['price'].std() * 1.5) & (df['volume_trend'])
        trend_direction = np.sign(df['trend'])
        
        return {
            'trend_strength': abs(df['trend'].iloc[-1]),
            'trend_direction': trend_direction.iloc[-1],
            'volume_support': df['volume_trend'].iloc[-1],
            'persistence': strong_trend.sum() / len(strong_trend)
        }
    
    def get_position_size(self, trend_metrics, base_size):
        """Scale position size based on trend strength"""
        if not self.config.get('POSITION_SCALING', False):
            return base_size
            
        scale_factor = min(trend_metrics['trend_strength'] * 
                         trend_metrics['persistence'], 1.5)
        return int(base_size * scale_factor)
