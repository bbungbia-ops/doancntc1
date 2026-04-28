"""
Module 4: Feature Engineering
==============================
Tạo các đặc trưng (features) từ dữ liệu giá và sentiment
cho Machine Learning models.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import ta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Tạo features cho ML models:
    1. Technical Indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
    2. Sentiment Features (mean, std, MA, momentum)
    3. Lag Features (price lags, sentiment lags)
    4. Volatility Features
    """

    def __init__(self):
        self.feature_columns = []

    def add_technical_indicators(self, df):
        """
        Tính toán các chỉ số kỹ thuật (Technical Indicators).
        
        Args:
            df: DataFrame chứa OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame với các cột indicator mới
        """
        df = df.copy()
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # ---- Simple Moving Averages (SMA) ----
        for period in config.SMA_PERIODS:
            df[f'sma_{period}'] = ta.trend.sma_indicator(close, window=period)
            # Tỷ lệ giá so với SMA
            df[f'price_sma_{period}_ratio'] = close / df[f'sma_{period}']
        
        # ---- Exponential Moving Averages (EMA) ----
        for period in config.EMA_PERIODS:
            df[f'ema_{period}'] = ta.trend.ema_indicator(close, window=period)
        
        # ---- RSI (Relative Strength Index) ----
        df['rsi'] = ta.momentum.rsi(close, window=config.RSI_PERIOD)
        
        # ---- MACD ----
        macd = ta.trend.MACD(
            close,
            window_slow=config.MACD_SLOW,
            window_fast=config.MACD_FAST,
            window_sign=config.MACD_SIGNAL
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # ---- Bollinger Bands ----
        bollinger = ta.volatility.BollingerBands(
            close,
            window=config.BOLLINGER_PERIOD,
            window_dev=config.BOLLINGER_STD
        )
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ---- Average True Range (ATR) ----
        df['atr'] = ta.volatility.average_true_range(high, low, close, window=14)
        
        # ---- Stochastic Oscillator ----
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ---- Volume Features ----
        df['volume_sma_20'] = ta.trend.sma_indicator(volume, window=20)
        df['volume_ratio'] = volume / df['volume_sma_20']
        
        # ---- On-Balance Volume (OBV) ----
        df['obv'] = ta.volume.on_balance_volume(close, volume)
        
        # ---- Daily Returns & Volatility ----
        df['daily_return'] = close.pct_change()
        df['volatility_7'] = df['daily_return'].rolling(window=7).std()
        df['volatility_14'] = df['daily_return'].rolling(window=14).std()
        df['volatility_30'] = df['daily_return'].rolling(window=30).std()
        
        # ---- Momentum ----
        df['momentum_7'] = close.pct_change(periods=7)
        df['momentum_14'] = close.pct_change(periods=14)
        df['momentum_30'] = close.pct_change(periods=30)
        
        logger.info(f"Đã thêm {len([c for c in df.columns if c not in ['Open','High','Low','Close','Volume']])} technical indicators")
        
        return df

    def add_sentiment_features(self, price_df, daily_sentiment_df):
        """
        Merge sentiment features vào dữ liệu giá.
        
        Args:
            price_df: DataFrame giá (index = Date)
            daily_sentiment_df: DataFrame sentiment theo ngày
            
        Returns:
            pd.DataFrame: DataFrame đã merge
        """
        df = price_df.copy()
        
        # Đảm bảo index là datetime
        df.index = pd.to_datetime(df.index)
        daily_sentiment_df.index = pd.to_datetime(daily_sentiment_df.index)
        
        # Merge sentiment vào price data
        sentiment_cols = ['sentiment_mean', 'sentiment_std', 'sentiment_min',
                         'sentiment_max', 'news_count', 'vader_mean',
                         'finbert_mean', 'confidence_mean']
        
        available_cols = [c for c in sentiment_cols if c in daily_sentiment_df.columns]
        
        for col in available_cols:
            df[col] = daily_sentiment_df[col].reindex(df.index)
        
        # Fill missing sentiment với forward fill rồi backward fill
        for col in available_cols:
            df[col] = df[col].ffill().bfill()
            # Fill remaining NaN với 0
            df[col] = df[col].fillna(0)
        
        # ---- Sentiment Moving Averages ----
        if 'sentiment_mean' in df.columns:
            for period in config.SENTIMENT_MA_PERIODS:
                df[f'sentiment_ma_{period}'] = df['sentiment_mean'].rolling(window=period).mean()
            
            # Sentiment Momentum
            df['sentiment_momentum'] = df['sentiment_mean'].diff(periods=3)
            
            # Sentiment Acceleration
            df['sentiment_acceleration'] = df['sentiment_momentum'].diff()
            
            # Sentiment Volatility (rolling std)
            df['sentiment_volatility'] = df['sentiment_mean'].rolling(window=7).std()
            
            # Sentiment-Price Divergence
            df['price_return_7d'] = df['Close'].pct_change(periods=7)
            df['sent_price_divergence'] = df['sentiment_mean'] - df['price_return_7d']
        
        logger.info(f"Đã thêm sentiment features vào dữ liệu giá")
        
        return df

    def add_lag_features(self, df):
        """
        Thêm lag features (giá và sentiment quá khứ).
        
        Args:
            df: DataFrame đã có price và sentiment features
            
        Returns:
            pd.DataFrame: DataFrame với lag features
        """
        df = df.copy()
        
        # Price lag features
        for lag in range(1, config.PRICE_LAG_DAYS + 1):
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'return_lag_{lag}'] = df['daily_return'].shift(lag) if 'daily_return' in df.columns else df['Close'].pct_change().shift(lag)
        
        # Sentiment lag features
        if 'sentiment_mean' in df.columns:
            for lag in range(1, config.SENTIMENT_LAG_DAYS + 1):
                df[f'sentiment_lag_{lag}'] = df['sentiment_mean'].shift(lag)
        
        # Volume lag
        df['volume_lag_1'] = df['Volume'].shift(1)
        df['volume_change'] = df['Volume'].pct_change()
        
        logger.info(f"Đã thêm lag features")
        
        return df

    def add_time_features(self, df):
        """
        Thêm time-based features.
        
        Args:
            df: DataFrame với DatetimeIndex
            
        Returns:
            pd.DataFrame: DataFrame với time features
        """
        df = df.copy()
        
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        
        logger.info("Đã thêm time features")
        
        return df

    def build_features(self, price_df, daily_sentiment_df):
        """
        Pipeline xây dựng toàn bộ features.
        
        Args:
            price_df: DataFrame dữ liệu giá (đã có labels)
            daily_sentiment_df: DataFrame sentiment theo ngày
            
        Returns:
            pd.DataFrame: DataFrame hoàn chỉnh với tất cả features
        """
        logger.info("=" * 60)
        logger.info("BẮT ĐẦU FEATURE ENGINEERING")
        logger.info("=" * 60)
        
        # Bước 1: Technical Indicators
        df = self.add_technical_indicators(price_df)
        
        # Bước 2: Sentiment Features
        df = self.add_sentiment_features(df, daily_sentiment_df)
        
        # Bước 3: Lag Features
        df = self.add_lag_features(df)
        
        # Bước 4: Time Features
        df = self.add_time_features(df)
        
        # Bước 5: Loại bỏ NaN rows (do rolling windows)
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        
        # Lưu danh sách feature columns
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'label', 'price_change']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        logger.info("=" * 60)
        logger.info("HOÀN TẤT FEATURE ENGINEERING")
        logger.info(f"  - Tổng số features: {len(self.feature_columns)}")
        logger.info(f"  - Số mẫu: {len(df)} (loại bỏ {dropped} dòng NaN)")
        logger.info(f"  - Features: {self.feature_columns[:10]}...")
        logger.info("=" * 60)
        
        # Lưu dữ liệu đã xử lý
        output_path = os.path.join(config.PROCESSED_DATA_DIR, "features_data.csv")
        df.to_csv(output_path)
        logger.info(f"Đã lưu features data tại: {output_path}")
        
        return df

    def get_feature_columns(self):
        """Trả về danh sách feature columns."""
        return self.feature_columns


if __name__ == "__main__":
    import yfinance as yf
    
    # Test feature engineering
    print("Tải dữ liệu test...")
    df = yf.download("BTC-USD", start="2024-01-01", end="2024-06-30")
    
    fe = FeatureEngineer()
    df_features = fe.add_technical_indicators(df)
    print(f"\nSố features: {len(df_features.columns)}")
    print(f"Columns: {list(df_features.columns)}")
    print(f"\nHead:\n{df_features.tail()}")
