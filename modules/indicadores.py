import pandas as pd
import numpy as np
import ta
from typing import Dict, Any

class TechnicalIndicators:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def calculate_all_indicators(self) -> pd.DataFrame:
        df = self.df.copy()
        
        df = self._add_moving_averages(df)
        df = self._add_macd(df)
        df = self._add_bollinger_bands(df)
        df = self._add_rsi(df)
        df = self._add_stochastic(df)
        df = self._add_williams_r(df)
        df = self._add_atr(df)
        df = self._add_volatility(df)
        df = self._add_volume_indicators(df)
        df = self._add_custom_indicators(df)
        
        return df
    
    def _add_moving_averages(self, df):
        df['SMA_5'] = ta.trend.sma_indicator(df['close'], window=5)
        df['SMA_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
        
        df['EMA_5'] = ta.trend.ema_indicator(df['close'], window=5)
        df['EMA_10'] = ta.trend.ema_indicator(df['close'], window=10)
        df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)
        
        return df
    
    def _add_macd(self, df):
        df['MACD'] = ta.trend.macd_diff(df['close'])
        df['MACD_signal'] = ta.trend.macd_signal(df['close'])
        df['MACD_histogram'] = ta.trend.macd(df['close'])
        
        return df
    
    def _add_bollinger_bands(self, df):
        df['BB_upper'] = ta.volatility.bollinger_hband(df['close'])
        df['BB_middle'] = ta.volatility.bollinger_mavg(df['close'])
        df['BB_lower'] = ta.volatility.bollinger_lband(df['close'])
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['close'] - df['BB_lower']) / df['BB_width']
        
        return df
    
    def _add_rsi(self, df):
        df['RSI'] = ta.momentum.rsi(df['close'])
        df['RSI_oversold'] = (df['RSI'] < 30).astype(int)
        df['RSI_overbought'] = (df['RSI'] > 70).astype(int)
        
        return df
    
    def _add_stochastic(self, df):
        df['Stoch_K'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['Stoch_D'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
        
        return df
    
    def _add_williams_r(self, df):
        df['Williams_R'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        return df
    
    def _add_atr(self, df):
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        df['ATR_percent'] = (df['ATR'] / df['close']) * 100
        
        return df
    
    def _add_volatility(self, df):
        df['price_change'] = df['close'].pct_change()
        df['volatility_5'] = df['price_change'].rolling(5).std()
        df['volatility_20'] = df['price_change'].rolling(20).std()
        
        return df
    
    def _add_volume_indicators(self, df):
        if 'volume' in df.columns and not df['volume'].isna().all():
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        else:
            df['volume_sma'] = 0
            df['volume_ratio'] = 1
        
        return df
    
    def _add_custom_indicators(self, df):
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        df['price_position'] = (df['close'] - df['low'].rolling(20).min()) / \
                              (df['high'].rolling(20).max() - df['low'].rolling(20).min())
        
        df['trend_short'] = np.where(df['close'] > df['EMA_5'], 1, 
                           np.where(df['close'] < df['EMA_5'], -1, 0))
        
        df['MA_convergence'] = (df['EMA_5'] - df['EMA_20']) / df['EMA_20']
        
        return df
    
    def get_feature_importance_score(self, df):
        features = {}
        
        latest = df.iloc[-1]
        
        rsi_score = 0
        if latest['RSI'] < 30:
            rsi_score = 2
        elif latest['RSI'] < 40:
            rsi_score = 1
        elif latest['RSI'] > 70:
            rsi_score = -2
        elif latest['RSI'] > 60:
            rsi_score = -1
        
        features['rsi_score'] = rsi_score
        
        macd_score = 0
        if latest['MACD'] > latest['MACD_signal']:
            macd_score = 1
        else:
            macd_score = -1
        
        features['macd_score'] = macd_score
        
        bb_score = 0
        bb_pos = latest['BB_position']
        if bb_pos < 0.2:
            bb_score = 1
        elif bb_pos > 0.8:
            bb_score = -1
        
        features['bb_score'] = bb_score
        
        return features
