import pandas as pd
import numpy as np

def add_indicators(df):
    """Add technical indicators to DataFrame"""
    # Moving Averages
    df['MA_short'] = df['price'].rolling(window=20).mean()
    df['MA_long'] = df['price'].rolling(window=50).mean()
    
    # RSI
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility
    df['volatility'] = df['price'].rolling(window=20).std()
    
    return df
