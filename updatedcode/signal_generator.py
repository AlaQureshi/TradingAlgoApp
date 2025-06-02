"""
Signal generator module for the Trading Algorithm AI.
Handles technical analysis and signal generation based on market data.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Import configuration
from config import INDICATORS, SIGNAL_CLASSIFICATION

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('signal_generator')

class SignalGenerator:
    """
    Generates trading signals based on technical analysis of market data.
    Supports multiple indicators and signal classification.
    """
    
    def __init__(self):
        """Initialize the signal generator with configuration settings."""
        self.indicators_config = INDICATORS
        self.signal_config = SIGNAL_CLASSIFICATION
        logger.info("SignalGenerator initialized")
    
    def calculate_indicators(self, df):
        """
        Calculate technical indicators for the given data.
        
        Args:
            df (pandas.DataFrame): Market data with OHLCV columns
            
        Returns:
            pandas.DataFrame: Original data with added indicator columns
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate trend indicators
        self._calculate_moving_averages(result)
        self._calculate_macd(result)
        self._calculate_parabolic_sar(result)
        
        # Calculate momentum indicators
        self._calculate_rsi(result)
        self._calculate_stochastic(result)
        self._calculate_cci(result)
        
        # Calculate volatility indicators
        self._calculate_bollinger_bands(result)
        self._calculate_atr(result)
        
        # Calculate volume indicators
        self._calculate_obv(result)
        self._calculate_volume_sma(result)
        
        logger.info("Technical indicators calculated successfully")
        return result
    
    def generate_signals(self, df):
        """
        Generate trading signals based on calculated indicators.
        
        Args:
            df (pandas.DataFrame): Market data with indicators
            
        Returns:
            pandas.DataFrame: Data with added signal columns
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Generate individual indicator signals
        self._generate_ma_signals(result)
        self._generate_macd_signals(result)
        self._generate_rsi_signals(result)
        self._generate_stochastic_signals(result)
        self._generate_bollinger_signals(result)
        self._generate_volume_signals(result)
        
        # Combine signals to determine overall signal strength
        self._classify_signal_strength(result)
        
        logger.info("Trading signals generated successfully")
        return result
    
    def _calculate_moving_averages(self, df):
        """Calculate Simple Moving Averages (SMA) and Exponential Moving Averages (EMA)."""
        # Calculate SMAs
        for period in self.indicators_config['trend']['sma']['periods']:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Calculate EMAs
        for period in self.indicators_config['trend']['ema']['periods']:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    def _calculate_macd(self, df):
        """Calculate Moving Average Convergence Divergence (MACD)."""
        fast_period = self.indicators_config['trend']['macd']['fast_period']
        slow_period = self.indicators_config['trend']['macd']['slow_period']
        signal_period = self.indicators_config['trend']['macd']['signal_period']
        
        # Calculate MACD line
        df['macd_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
        df['macd_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
        df['macd_line'] = df['macd_fast'] - df['macd_slow']
        
        # Calculate signal line
        df['macd_signal'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        
        # Clean up intermediate columns
        df.drop(['macd_fast', 'macd_slow'], axis=1, inplace=True)
    
    def _calculate_parabolic_sar(self, df):
        """Calculate Parabolic SAR."""
        acceleration = self.indicators_config['trend']['parabolic_sar']['acceleration']
        maximum = self.indicators_config['trend']['parabolic_sar']['maximum']
        
        # Initialize columns
        df['psar'] = np.nan
        df['psar_trend'] = np.nan
        
        # Simplified implementation (for a full implementation, consider using a library like ta-lib)
        # This is a basic implementation and may not match exactly with other implementations
        
        # Find initial trend
        if len(df) < 2:
            return
        
        # Determine initial trend (up or down)
        initial_trend = 1 if df['close'].iloc[1] > df['close'].iloc[0] else -1
        
        # Set initial values
        if initial_trend == 1:
            # Uptrend
            psar = df['low'].iloc[0]
            ep = df['high'].iloc[1]  # Extreme point
        else:
            # Downtrend
            psar = df['high'].iloc[0]
            ep = df['low'].iloc[1]  # Extreme point
        
        af = acceleration  # Acceleration factor
        
        # Calculate PSAR for each period
        for i in range(2, len(df)):
            # Previous PSAR
            prev_psar = psar
            
            # Calculate PSAR
            psar = prev_psar + af * (ep - prev_psar)
            
            # Ensure PSAR doesn't go beyond the previous two periods' extremes
            if initial_trend == 1:
                # Uptrend - PSAR can't be above the previous two lows
                psar = min(psar, df['low'].iloc[i-1], df['low'].iloc[i-2])
            else:
                # Downtrend - PSAR can't be below the previous two highs
                psar = max(psar, df['high'].iloc[i-1], df['high'].iloc[i-2])
            
            # Check for trend reversal
            if (initial_trend == 1 and df['low'].iloc[i] < psar) or \
               (initial_trend == -1 and df['high'].iloc[i] > psar):
                # Trend reversal
                initial_trend *= -1
                psar = ep
                
                if initial_trend == 1:
                    ep = df['high'].iloc[i]
                else:
                    ep = df['low'].iloc[i]
                
                af = acceleration
            else:
                # No trend reversal, update extreme point if needed
                if initial_trend == 1 and df['high'].iloc[i] > ep:
                    ep = df['high'].iloc[i]
                    af = min(af + acceleration, maximum)
                elif initial_trend == -1 and df['low'].iloc[i] < ep:
                    ep = df['low'].iloc[i]
                    af = min(af + acceleration, maximum)
            
            # Store values
            df.loc[df.index[i], 'psar'] = psar
            df.loc[df.index[i], 'psar_trend'] = initial_trend
    
    def _calculate_rsi(self, df):
        """Calculate Relative Strength Index (RSI)."""
        period = self.indicators_config['momentum']['rsi']['period']
        
        # Calculate price changes
        delta = df['close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
    def _calculate_stochastic(self, df):
        """Calculate Stochastic Oscillator."""
        k_period = self.indicators_config['momentum']['stochastic']['k_period']
        d_period = self.indicators_config['momentum']['stochastic']['d_period']
        
        # Calculate %K
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        
        # Calculate %D (SMA of %K)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
    
    def _calculate_cci(self, df):
        """Calculate Commodity Channel Index (CCI)."""
        period = self.indicators_config['momentum']['cci']['period']
        
        # Calculate typical price
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate SMA of typical price
        df['tp_sma'] = df['tp'].rolling(window=period).mean()
        
        # Calculate mean deviation
        df['tp_md'] = df['tp'].rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        # Calculate CCI
        df['cci'] = (df['tp'] - df['tp_sma']) / (0.015 * df['tp_md'])
        
        # Clean up intermediate columns
        df.drop(['tp', 'tp_sma', 'tp_md'], axis=1, inplace=True)
    
    def _calculate_bollinger_bands(self, df):
        """Calculate Bollinger Bands."""
        period = self.indicators_config['volatility']['bollinger_bands']['period']
        std_dev = self.indicators_config['volatility']['bollinger_bands']['std_dev']
        
        # Calculate middle band (SMA)
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        df['bb_std'] = df['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
        
        # Calculate bandwidth and %B
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Clean up intermediate columns
        df.drop(['bb_std'], axis=1, inplace=True)
    
    def _calculate_atr(self, df):
        """Calculate Average True Range (ATR)."""
        period = self.indicators_config['volatility']['atr']['period']
        
        # Calculate true range
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        df['atr'] = df['tr'].rolling(window=period).mean()
        
        # Clean up intermediate columns
        df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1, inplace=True)
    
    def _calculate_obv(self, df):
        """Calculate On-Balance Volume (OBV)."""
        # Initialize OBV
        df['obv'] = 0
        
        # Calculate OBV
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1] - df['volume'].iloc[i]
            else:
                df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1]
    
    def _calculate_volume_sma(self, df):
        """Calculate Volume Simple Moving Average."""
        period = self.indicators_config['volume']['volume_sma']['period']
        
        # Calculate volume SMA
        df['volume_sma'] = df['volume'].rolling(window=period).mean()
    
    def _generate_ma_signals(self, df):
        """Generate signals based on moving averages."""
        # Initialize signal columns
        df['signal_ma_cross'] = 0
        
        # Check for SMA crossovers (short-term vs. medium-term)
        short_term = self.indicators_config['trend']['sma']['periods'][0]  # e.g., 20
        medium_term = self.indicators_config['trend']['sma']['periods'][1]  # e.g., 50
        
        # Generate crossover signals
        for i in range(1, len(df)):
            # Bullish crossover: short-term SMA crosses above medium-term SMA
            if df[f'sma_{short_term}'].iloc[i-1] <= df[f'sma_{medium_term}'].iloc[i-1] and \
               df[f'sma_{short_term}'].iloc[i] > df[f'sma_{medium_term}'].iloc[i]:
                df.loc[df.index[i], 'signal_ma_cross'] = 1
            
            # Bearish crossover: short-term SMA crosses below medium-term SMA
            elif df[f'sma_{short_term}'].iloc[i-1] >= df[f'sma_{medium_term}'].iloc[i-1] and \
                 df[f'sma_{short_term}'].iloc[i] < df[f'sma_{medium_term}'].iloc[i]:
                df.loc[df.index[i], 'signal_ma_cross'] = -1
        
        # Price relative to moving averages
        df['signal_ma_position'] = 0
        
        # Bullish: price above both short and medium-term SMAs
        df.loc[(df['close'] > df[f'sma_{short_term}']) & 
               (df['close'] > df[f'sma_{medium_term}']), 'signal_ma_position'] = 1
        
        # Bearish: price below both short and medium-term SMAs
        df.loc[(df['close'] < df[f'sma_{short_term}']) & 
               (df['close'] < df[f'sma_{medium_term}']), 'signal_ma_position'] = -1
    
    def _generate_macd_signals(self, df):
        """Generate signals based on MACD."""
        # Initialize signal column
        df['signal_macd'] = 0
        
        # Generate crossover signals
        for i in range(1, len(df)):
            # Bullish crossover: MACD line crosses above signal line
            if df['macd_line'].iloc[i-1] <= df['macd_signal'].iloc[i-1] and \
               df['macd_line'].iloc[i] > df['macd_signal'].iloc[i]:
                df.loc[df.index[i], 'signal_macd'] = 1
            
            # Bearish crossover: MACD line crosses below signal line
            elif df['macd_line'].iloc[i-1] >= df['macd_signal'].iloc[i-1] and \
                 df['macd_line'].iloc[i] < df['macd_signal'].iloc[i]:
                df.loc[df.index[i], 'signal_macd'] = -1
        
        # MACD histogram direction
        df['signal_macd_hist'] = 0
        
        # Bullish: histogram is positive and increasing
        df.loc[(df['macd_histogram'] > 0) & 
               (df['macd_histogram'] > df['macd_histogram'].shift(1)), 'signal_macd_hist'] = 1
        
        # Bearish: histogram is negative and decreasing
        df.loc[(df['macd_histogram'] < 0) & 
               (df['macd_histogram'] < df['macd_histogram'].shift(1)), 'signal_macd_hist'] = -1
    
    def _generate_rsi_signals(self, df):
        """Generate signals based on RSI."""
        # Initialize signal column
        df['signal_rsi'] = 0
        
        # Get overbought and oversold thresholds
        overbought = self.indicators_config['momentum']['rsi']['overbought']
        oversold = self.indicators_config['momentum']['rsi']['oversold']
        
        # Bullish: RSI crosses above oversold level
        df.loc[(df['rsi'] > oversold) & 
               (df['rsi'].shift(1) <= oversold), 'signal_rsi'] = 1
        
        # Bearish: RSI crosses below overbought level
        df.loc[(df['rsi'] < overbought) & 
               (df['rsi'].shift(1) >= overbought), 'signal_rsi'] = -1
    
    def _generate_stochastic_signals(self, df):
        """Generate signals based on Stochastic Oscillator."""
        # Initialize signal column
        df['signal_stoch'] = 0
        
        # Get overbought and oversold thresholds
        overbought = self.indicators_config['momentum']['stochastic']['overbought']
        oversold = self.indicators_config['momentum']['stochastic']['oversold']
        
        # Bullish: %K crosses above %D in oversold territory
        df.loc[(df['stoch_k'] > df['stoch_d']) & 
               (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1)) & 
               (df['stoch_k'] < oversold), 'signal_stoch'] = 1
        
        # Bearish: %K crosses below %D in overbought territory
        df.loc[(df['stoch_k'] < df['stoch_d']) & 
               (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1)) & 
               (df['stoch_k'] > overbought), 'signal_stoch'] = -1
    
    def _generate_bollinger_signals(self, df):
        """Generate signals based on Bollinger Bands."""
        # Initialize signal column
        df['signal_bb'] = 0
        
        # Bullish: price touches or crosses below lower band
        df.loc[df['close'] <= df['bb_lower'], 'signal_bb'] = 1
        
        # Bearish: price touches or crosses above upper band
        df.loc[df['close'] >= df['bb_upper'], 'signal_bb'] = -1
        
        # Additional signal: Bollinger Band squeeze (potential breakout)
        df['signal_bb_squeeze'] = 0
        
        # Detect Bollinger Band squeeze (bandwidth contraction)
        df.loc[df['bb_bandwidth'] < df['bb_bandwidth'].rolling(window=20).mean() * 0.8, 'signal_bb_squeeze'] = 1
    
    def _generate_volume_signals(self, df):
        """Generate signals based on volume indicators."""
        # Initialize signal column
        df['signal_volume'] = 0
        
        # Bullish: volume increasing with price increase
        df.loc[(df['close'] > df['close'].shift(1)) & 
               (df['volume'] > df['volume'].shift(1)), 'signal_volume'] = 1
        
        # Bearish: volume increasing with price decrease
        df.loc[(df['close'] < df['close'].shift(1)) & 
               (df['volume'] > df['volume'].shift(1)), 'signal_volume'] = -1
        
        # OBV trend
        df['signal_obv'] = 0
        
        # Calculate OBV SMA
        df['obv_sma'] = df['obv'].rolling(window=20).mean()
        
        # Bullish: OBV above its SMA
        df.loc[df['obv'] > df['obv_sma'], 'signal_obv'] = 1
        
        # Bearish: OBV below its SMA
        df.loc[df['obv'] < df['obv_sma'], 'signal_obv'] = -1
        
        # Clean up intermediate columns
        df.drop(['obv_sma'], axis=1, inplace=True)
    
    def _classify_signal_strength(self, df):
        """Classify overall signal strength based on individual signals."""
        # Initialize signal strength columns
        df['bullish_signals'] = 0
        df['bearish_signals'] = 0
        df['signal_strength'] = 'neutral'
        df['position_size_factor'] = 0.0
        
        # Count bullish signals
        signal_columns = [col for col in df.columns if col.startswith('signal_')]
        
        for col in signal_columns:
            df['bullish_signals'] += (df[col] == 1).astype(int)
            df['bearish_signals'] += (df[col] == -1).astype(int)
        
        # Determine signal strength based on number of aligned signals
        strong_min = self.signal_config['strong']['min_indicators']
        moderate_min = self.signal_config['moderate']['min_indicators']
        weak_min = self.signal_config['weak']['min_indicators']
        
        # Strong bullish signal
        df.loc[df['bullish_signals'] >= strong_min, 'signal_strength'] = 'strong_bullish'
        df.loc[df['bullish_signals'] >= strong_min, 'position_size_factor'] = self.signal_config['strong']['position_size_multiplier']
        
        # Moderate bullish signal
        df.loc[(df['bullish_signals'] >= moderate_min) & 
               (df['bullish_signals'] < strong_min), 'signal_strength'] = 'moderate_bullish'
        df.loc[(df['bullish_signals'] >= moderate_min) & 
               (df['bullish_signals'] < strong_min), 'position_size_factor'] = self.signal_config['moderate']['position_size_multiplier']
        
        # Weak bullish signal
        df.loc[(df['bullish_signals'] >= weak_min) & 
               (df['bullish_signals'] < moderate_min), 'signal_strength'] = 'weak_bullish'
        df.loc[(df['bullish_signals'] >= weak_min) & 
               (df['bullish_signals'] < moderate_min), 'position_size_factor'] = self.signal_config['weak']['position_size_multiplier']
        
        # Strong bearish signal
        df.loc[df['bearish_signals'] >= strong_min, 'signal_strength'] = 'strong_bearish'
        df.loc[df['bearish_signals'] >= strong_min, 'position_size_factor'] = -self.signal_config['strong']['position_size_multiplier']
        
        # Moderate bearish signal
        df.loc[(df['bearish_signals'] >= moderate_min) & 
               (df['bearish_signals'] < strong_min), 'signal_strength'] = 'moderate_bearish'
        df.loc[(df['bearish_signals'] >= moderate_min) & 
               (df['bearish_signals'] < strong_min), 'position_size_factor'] = -self.signal_config['moderate']['position_size_multiplier']
        
        # Weak bearish signal
        df.loc[(df['bearish_signals'] >= weak_min) & 
               (df['bearish_signals'] < moderate_min), 'signal_strength'] = 'weak_bearish'
        df.loc[(df['bearish_signals'] >= weak_min) & 
               (df['bearish_signals'] < moderate_min), 'position_size_factor'] = -self.signal_config['weak']['position_size_multiplier']

# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = {
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(100, 5, 100),
        'volume': np.random.normal(1000000, 200000, 100)
    }
    df = pd.DataFrame(data, index=dates)
    
    # Ensure high is the highest and low is the lowest
    for i in range(len(df)):
        values = [df['open'].iloc[i], df['close'].iloc[i]]
        df.loc[df.index[i], 'high'] = max(values) + abs(np.random.normal(0, 1))
        df.loc[df.index[i], 'low'] = min(values) - abs(np.random.normal(0, 1))
    
    # Initialize signal generator
    generator = SignalGenerator()
    
    # Calculate indicators
    df_with_indicators = generator.calculate_indicators(df)
    
    # Generate signals
    df_with_signals = generator.generate_signals(df_with_indicators)
    
    # Print results
    print("Sample data with signals:")
    print(df_with_signals[['close', 'signal_strength', 'position_size_factor']].tail())
