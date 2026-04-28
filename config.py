"""
Cấu hình toàn hệ thống cho dự án Giao dịch Crypto tự động.
Tập trung tất cả các tham số, đường dẫn, và hyperparameters.
"""

import os
from datetime import datetime, timedelta

# ============================================
# ĐƯỜNG DẪN
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(DATA_DIR, "models")

# Tạo thư mục nếu chưa tồn tại
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================
# CẤU HÌNH DỮ LIỆU
# ============================================
# Đồng tiền điện tử
CRYPTO_SYMBOL = "BTC-USD"
CRYPTO_NAME = "Bitcoin"

# Khoảng thời gian lấy dữ liệu
END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")  # 2 năm

# ============================================
# CẤU HÌNH THU THẬP TIN TỨC
# ============================================
NEWS_SOURCES = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "cryptonews": "https://cryptonews.com/news/feed/",
}

# Số lượng tin tối đa mỗi nguồn
MAX_NEWS_PER_SOURCE = 100

# ============================================
# CẤU HÌNH SENTIMENT ANALYSIS
# ============================================
# Trọng số ensemble VADER + FinBERT
VADER_WEIGHT = 0.3
FINBERT_WEIGHT = 0.7

# FinBERT model
FINBERT_MODEL_NAME = "ProsusAI/finbert"

# Sentiment thresholds
SENTIMENT_POSITIVE_THRESHOLD = 0.2
SENTIMENT_NEGATIVE_THRESHOLD = -0.2

# ============================================
# CẤU HÌNH FEATURE ENGINEERING
# ============================================
# Technical Indicators
SMA_PERIODS = [7, 14, 30]
EMA_PERIODS = [12, 26]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Sentiment aggregation windows
SENTIMENT_MA_PERIODS = [3, 7]

# Lag features
PRICE_LAG_DAYS = 7
SENTIMENT_LAG_DAYS = 3

# ============================================
# CẤU HÌNH LSTM MODEL
# ============================================
LSTM_CONFIG = {
    "sequence_length": 30,       # Số ngày lookback
    "lstm_units_1": 128,         # Số units layer 1
    "lstm_units_2": 64,          # Số units layer 2
    "dropout_rate": 0.3,         # Dropout rate
    "dense_units": 32,           # Dense layer units
    "learning_rate": 0.001,      # Learning rate
    "batch_size": 32,            # Batch size
    "epochs": 100,               # Số epochs
    "patience": 15,              # Early stopping patience
}

# ============================================
# CẤU HÌNH RANDOM FOREST
# ============================================
RF_CONFIG = {
    "n_estimators": 500,         # Số cây
    "max_depth": 15,             # Độ sâu tối đa
    "min_samples_split": 5,      # Min samples để split
    "min_samples_leaf": 2,       # Min samples ở leaf
    "class_weight": "balanced",  # Cân bằng lớp
    "random_state": 42,
    "n_jobs": -1,                # Sử dụng tất cả CPU
}

# ============================================
# CẤU HÌNH XGBOOST
# ============================================
XGB_CONFIG = {
    "n_estimators": 500,         # Số cây
    "max_depth": 8,              # Độ sâu tối đa
    "learning_rate": 0.05,       # Learning rate
    "subsample": 0.8,            # Tỷ lệ sample
    "colsample_bytree": 0.8,    # Tỷ lệ feature mỗi cây
    "reg_alpha": 0.1,            # L1 regularization
    "reg_lambda": 1.0,           # L2 regularization
    "random_state": 42,
    "use_label_encoder": False,
    "eval_metric": "mlogloss",
}

# ============================================
# CẤU HÌNH STACKING ENSEMBLE
# ============================================
ENSEMBLE_CONFIG = {
    "cv_folds": 5,               # Số fold cho TimeSeriesSplit
    "meta_learner": "logistic",  # Meta-learner: logistic / ridge
}

# ============================================
# CẤU HÌNH TRADING STRATEGY
# ============================================
TRADING_CONFIG = {
    "initial_capital": 10000,    # Vốn ban đầu (USD)
    "position_size": 0.3,        # Phần trăm vốn mỗi lệnh
    "stop_loss": 0.05,           # Stop-loss 5%
    "take_profit": 0.10,         # Take-profit 10%
    "commission": 0.001,         # Phí giao dịch 0.1%
}

# ============================================
# CẤU HÌNH TRAIN/TEST SPLIT
# ============================================
TEST_SIZE = 0.2                  # 20% cho test
VALIDATION_SIZE = 0.1            # 10% cho validation

# Label encoding: 0=DOWN, 1=NEUTRAL, 2=UP
LABEL_MAP = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
PRICE_CHANGE_THRESHOLD = 0.005   # 0.5% threshold cho UP/DOWN
