"""
Live Pipeline - Chạy trực tiếp trên Streamlit Cloud
Không phụ thuộc vào file CSV từ main.py
"""
import os, sys, logging, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Hỗ trợ cả flat (GitHub) và nested (local) structure
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.dirname(_THIS_DIR))
import config

logger = logging.getLogger(__name__)


def collect_news_live():
    """Thu thập tin tức từ RSS feeds."""
    try:
        from data_collector import NewsDataCollector
    except ImportError:
        from src.data_collector import NewsDataCollector
    collector = NewsDataCollector()
    news_df = collector.collect_all_news()
    return news_df


def preprocess_news_live(news_df):
    """Tiền xử lý tin tức."""
    try:
        from preprocessing import TextPreprocessor
    except ImportError:
        from src.preprocessing import TextPreprocessor
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_news_dataframe(news_df)


def analyze_sentiment_live(news_df):
    """Phân tích sentiment dùng VADER + FinBERT fallback."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    # Thêm crypto lexicon
    crypto_lexicon = {
        'bullish': 3.0, 'bearish': -3.0, 'moon': 2.5, 'dump': -2.5,
        'pump': 2.0, 'crash': -3.5, 'rally': 2.5, 'surge': 2.5,
        'plunge': -3.0, 'soar': 2.5, 'tank': -2.5, 'breakout': 2.0,
        'breakdown': -2.0, 'hodl': 1.5, 'fud': -2.0, 'fomo': 1.0,
        'hack': -3.0, 'adoption': 2.0, 'ban': -3.0, 'etf': 1.5,
        'approval': 2.5, 'rejection': -2.5, 'ath': 3.0, 'dip': -1.0,
        'correction': -1.5, 'recovery': 2.0, 'liquidation': -2.5,
        'growth': 2.0, 'decline': -2.0,
    }
    analyzer.lexicon.update(crypto_lexicon)

    # FinBERT fallback function
    def finbert_fallback(text):
        text_lower = text.lower()
        pos_words = {'surge','soar','rally','bull','bullish','gain','growth',
                     'high','record','adoption','approve','approval','profit',
                     'moon','pump','breakout','recover','recovery','increase',
                     'rise','rising','positive','strong','success','inflow'}
        neg_words = {'crash','plunge','bear','bearish','drop','decline','loss',
                     'low','hack','scam','ban','crackdown','lawsuit','fraud',
                     'dump','tank','bankruptcy','liquidation','selloff','sell-off',
                     'fear','panic','negative','weak','risk','concern','warning'}
        words = set(text_lower.split())
        pc = len(words & pos_words)
        nc = len(words & neg_words)
        total = pc + nc
        if total == 0:
            return 0.0
        return (pc - nc) / total

    results = []
    for _, row in news_df.iterrows():
        text = row.get('content', row.get('title', ''))
        if not isinstance(text, str) or not text.strip():
            text = ''

        # VADER
        vader_scores = analyzer.polarity_scores(text) if text else {'compound': 0.0}
        vader_score = vader_scores['compound']

        # FinBERT fallback
        finbert_score = finbert_fallback(text) if text else 0.0

        # Ensemble
        ensemble_score = 0.3 * vader_score + 0.7 * finbert_score

        if ensemble_score > 0.2:
            label = 'positive'
        elif ensemble_score < -0.2:
            label = 'negative'
        else:
            label = 'neutral'

        results.append({
            'title': row.get('title', ''),
            'ensemble_score': round(ensemble_score, 4),
            'vader_score': round(vader_score, 4),
            'finbert_score': round(finbert_score, 4),
            'sentiment_label': label,
            'published': row.get('published', datetime.now()),
        })

    sentiment_df = pd.DataFrame(results)

    # Daily sentiment
    sentiment_df['date'] = pd.to_datetime(sentiment_df['published'], errors='coerce').dt.date
    daily = sentiment_df.groupby('date').agg({
        'ensemble_score': ['mean', 'std', 'min', 'max', 'count'],
        'vader_score': 'mean',
        'finbert_score': 'mean',
    })
    daily.columns = ['sentiment_mean','sentiment_std','sentiment_min',
                     'sentiment_max','news_count','vader_mean','finbert_mean']
    daily['sentiment_std'] = daily['sentiment_std'].fillna(0)
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    return sentiment_df, daily


def build_features_live(price_df, daily_sentiment):
    """Feature engineering nhẹ cho cloud."""
    import ta

    df = price_df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # Labels
    df['price_change'] = close.pct_change(1).shift(-1)
    threshold = config.PRICE_CHANGE_THRESHOLD
    conditions = [df['price_change'] < -threshold, df['price_change'] > threshold]
    df['label'] = np.select(conditions, [0, 2], default=1)

    # Technical indicators
    for p in [7, 14, 30]:
        df[f'sma_{p}'] = ta.trend.sma_indicator(close, window=p)
        df[f'price_sma_{p}_ratio'] = close / df[f'sma_{p}']
    for p in [12, 26]:
        df[f'ema_{p}'] = ta.trend.ema_indicator(close, window=p)

    df['rsi'] = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    df['atr'] = ta.volatility.average_true_range(high, low, close, window=14)
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['volume_sma_20'] = ta.trend.sma_indicator(volume, window=20)
    df['volume_ratio'] = volume / df['volume_sma_20']
    df['obv'] = ta.volume.on_balance_volume(close, volume)

    df['daily_return'] = close.pct_change()
    for w in [7, 14, 30]:
        df[f'volatility_{w}'] = df['daily_return'].rolling(window=w).std()
        df[f'momentum_{w}'] = close.pct_change(periods=w)

    # Sentiment features
    df.index = pd.to_datetime(df.index)
    if daily_sentiment is not None:
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
        for col in ['sentiment_mean','sentiment_std','sentiment_min',
                     'sentiment_max','news_count','vader_mean','finbert_mean']:
            if col in daily_sentiment.columns:
                df[col] = daily_sentiment[col].reindex(df.index)
                df[col] = df[col].ffill().bfill().fillna(0)
        if 'sentiment_mean' in df.columns:
            for p in [3, 7]:
                df[f'sentiment_ma_{p}'] = df['sentiment_mean'].rolling(window=p).mean()
            df['sentiment_momentum'] = df['sentiment_mean'].diff(periods=3)
            df['sentiment_acceleration'] = df['sentiment_momentum'].diff()
            df['sentiment_volatility'] = df['sentiment_mean'].rolling(window=7).std()
            df['price_return_7d'] = close.pct_change(periods=7)
            df['sent_price_divergence'] = df['sentiment_mean'] - df['price_return_7d']

    # Lag features
    for lag in range(1, 8):
        df[f'close_lag_{lag}'] = close.shift(lag)
        df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)
    if 'sentiment_mean' in df.columns:
        for lag in range(1, 4):
            df[f'sentiment_lag_{lag}'] = df['sentiment_mean'].shift(lag)
    df['volume_lag_1'] = volume.shift(1)
    df['volume_change'] = volume.pct_change()

    # Time features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)

    df = df.dropna()
    exclude = ['Open','High','Low','Close','Volume','label','price_change']
    feature_cols = [c for c in df.columns if c not in exclude]
    return df, feature_cols


def train_models_live(features_df, feature_cols):
    """Train RF + XGBoost (skip LSTM for cloud)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import xgboost as xgb

    n = len(features_df)
    test_start = int(n * 0.8)
    val_start = int(n * 0.7)

    X = features_df[feature_cols].values
    y = features_df['label'].values.astype(int)

    X_train, y_train = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:test_start], y[val_start:test_start]
    X_test, y_test = X[test_start:], y[test_start:]

    # Random Forest (lighter config)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # XGBoost (lighter config)
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        use_label_encoder=False, eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    # Ensemble (simple voting)
    rf_proba = rf.predict_proba(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)
    ensemble_proba = 0.5 * rf_proba + 0.5 * xgb_proba
    ensemble_pred = np.argmax(ensemble_proba, axis=1)

    rows = []
    for name, pred in [('Random Forest', rf_pred), ('XGBoost', xgb_pred), ('Ensemble', ensemble_pred)]:
        rows.append({
            'Model': name,
            'Accuracy': round(accuracy_score(y_test, pred), 4),
            'Precision': round(precision_score(y_test, pred, average='macro', zero_division=0), 4),
            'Recall': round(recall_score(y_test, pred, average='macro', zero_division=0), 4),
            'F1 (macro)': round(f1_score(y_test, pred, average='macro', zero_division=0), 4),
            'F1 (weighted)': round(f1_score(y_test, pred, average='weighted', zero_division=0), 4),
        })

    comparison_df = pd.DataFrame(rows)

    # Return data for backtesting
    test_dates = features_df.index[test_start:]
    test_prices = features_df['Close'].values[test_start:]
    test_sentiment = features_df['sentiment_mean'].values[test_start:] \
        if 'sentiment_mean' in features_df.columns else np.zeros(len(test_dates))

    return comparison_df, ensemble_pred, test_dates, test_prices, test_sentiment, rf, xgb_model


def run_backtesting_live(predictions, dates, prices, sentiment_scores):
    """Chạy backtesting từ predictions."""
    tc = config.TRADING_CONFIG
    initial_capital = tc['initial_capital']
    position_size = tc['position_size']
    stop_loss = tc['stop_loss']
    take_profit = tc['take_profit']
    commission = tc['commission']

    # Generate signals
    n = len(predictions)
    signals = []
    for i in range(n):
        pred = predictions[i]
        sent = sentiment_scores[i] if i < len(sentiment_scores) else 0
        if pred == 2 and sent > 0.2:
            sig = 'BUY'
        elif pred == 0 and sent < -0.2:
            sig = 'SELL'
        else:
            sig = 'HOLD'
        signals.append(sig)

    # Simulate
    cash = initial_capital
    holdings = 0.0
    entry_price = 0.0
    history = []

    for i in range(n):
        price = prices[i]
        signal = signals[i]
        if price <= 0:
            continue

        # Stop-loss / Take-profit
        if holdings > 0 and entry_price > 0:
            pnl = (price - entry_price) / entry_price
            if pnl <= -stop_loss or pnl >= take_profit:
                cash += holdings * price * (1 - commission)
                holdings = 0.0
                entry_price = 0.0

        if signal == 'BUY' and holdings == 0:
            invest = cash * position_size
            holdings = invest / price * (1 - commission)
            cash -= invest
            entry_price = price
        elif signal == 'SELL' and holdings > 0:
            cash += holdings * price * (1 - commission)
            holdings = 0.0
            entry_price = 0.0

        pv = cash + holdings * price
        history.append({
            'date': dates[i] if i < len(dates) else i,
            'equity': pv,
            'price': price,
        })

    equity_df = pd.DataFrame(history)
    if len(equity_df) > 0:
        equity_df['peak'] = equity_df['equity'].expanding().max()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        first_price = equity_df['price'].iloc[0]
        equity_df['buy_hold_equity'] = initial_capital * (equity_df['price'] / first_price)
    return equity_df
