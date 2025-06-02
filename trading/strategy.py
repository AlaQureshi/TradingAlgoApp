import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DelayedDataStrategy:
    def __init__(self, config):
        self.config = config
        self.min_holding = config.get('MIN_HOLDING_PERIOD', 30)
        self.delay = config.get('DELAY_MINUTES', 15)
        
    def analyze_trend(self, data):
        """Analyze using minute-level data"""
        df = data.copy()
        
        # Add candlestick patterns
        df['body'] = df['price'] - df['open']
        df['upper_wick'] = df['high'] - df.apply(lambda x: max(x['open'], x['price']), axis=1)
        df['lower_wick'] = df.apply(lambda x: min(x['open'], x['price']), axis=1) - df['low']
        
        # Calculate momentum using OHLC
        df['true_range'] = df.apply(
            lambda x: max([
                x['high'] - x['low'],
                abs(x['high'] - x['price'].shift(1)),
                abs(x['low'] - x['price'].shift(1))
            ]),
            axis=1
        )
        
        return {
            'trend_strength': df['true_range'].mean() / df['price'].mean(),
            'trend_direction': np.sign(df['body'].rolling(5).mean().iloc[-1]),
            'volume_support': df['size'] > df['size'].rolling(20).mean(),
            'volatility': df['true_range'].std() / df['price'].mean()
        }
    
    def get_position_size(self, trend_metrics, base_size):
        """Scale position size based on trend strength"""
        if not self.config.get('POSITION_SCALING', False):
            return base_size
            
        scale_factor = min(trend_metrics['trend_strength'] * 
                         trend_metrics['persistence'], 1.5)
        return int(base_size * scale_factor)
