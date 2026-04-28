"""
Module 5: Machine Learning Models (LSTM, Random Forest, XGBoost)
================================================================
Ba thuật toán mạnh kết hợp cho dự đoán xu hướng giá crypto.
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# MODEL 1: LSTM (Long Short-Term Memory)
# ============================================================
class LSTMModel:
    """
    LSTM Model cho dự đoán xu hướng giá crypto.
    
    Kiến trúc:
    - LSTM Layer 1 (128 units) -> Dropout(0.3)
    - LSTM Layer 2 (64 units) -> Dropout(0.3)
    - Dense Layer (32 units, ReLU)
    - Output Layer (3 units, Softmax) -> [DOWN, NEUTRAL, UP]
    
    Ưu điểm:
    - Bắt được temporal dependencies trong chuỗi thời gian
    - Nhớ các pattern dài hạn qua cơ chế gates
    """

    def __init__(self, input_shape=None, lstm_config=None):
        self.config = lstm_config or config.LSTM_CONFIG
        self.input_shape = input_shape
        self.model = None
        self.history = None

    def build_model(self, input_shape):
        """
        Xây dựng kiến trúc LSTM.
        
        Args:
            input_shape: (sequence_length, n_features)
        """
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        
        self.input_shape = input_shape
        
        model = Sequential([
            # LSTM Layer 1
            LSTM(
                units=self.config['lstm_units_1'],
                return_sequences=True,
                input_shape=input_shape
            ),
            Dropout(self.config['dropout_rate']),
            BatchNormalization(),
            
            # LSTM Layer 2
            LSTM(
                units=self.config['lstm_units_2'],
                return_sequences=False
            ),
            Dropout(self.config['dropout_rate']),
            BatchNormalization(),
            
            # Dense Layers
            Dense(self.config['dense_units'], activation='relu'),
            Dropout(0.2),
            
            # Output Layer: 3 classes (DOWN, NEUTRAL, UP)
            Dense(3, activation='softmax')
        ])
        
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info(f"LSTM Model built: input_shape={input_shape}")
        model.summary(print_fn=logger.info)
        
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Huấn luyện LSTM model.
        
        Args:
            X_train: Training sequences (n_samples, seq_len, n_features)
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
        """
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.config['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        logger.info("Bắt đầu huấn luyện LSTM...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Hoàn tất huấn luyện LSTM!")
        return self.history

    def predict(self, X):
        """Dự đoán nhãn."""
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện!")
        proba = self.model.predict(X, verbose=0)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        """Dự đoán xác suất cho mỗi lớp."""
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện!")
        return self.model.predict(X, verbose=0)

    def save(self, filepath=None):
        """Lưu model."""
        if filepath is None:
            filepath = os.path.join(config.MODEL_DIR, "lstm_model.keras")
        self.model.save(filepath)
        logger.info(f"LSTM model saved: {filepath}")

    def load(self, filepath=None):
        """Tải model."""
        import tensorflow as tf
        if filepath is None:
            filepath = os.path.join(config.MODEL_DIR, "lstm_model.keras")
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"LSTM model loaded: {filepath}")


# ============================================================
# MODEL 2: Random Forest Classifier
# ============================================================
class RandomForestModel:
    """
    Random Forest Model cho dự đoán xu hướng giá.
    
    Cấu hình:
    - 500 trees
    - max_depth=15
    - class_weight='balanced' (xử lý imbalanced data)
    
    Ưu điểm:
    - Robust với noise và outliers
    - Không cần feature scaling
    - Cho feature importance rankings
    """

    def __init__(self, rf_config=None):
        self.config = rf_config or config.RF_CONFIG
        self.model = RandomForestClassifier(**self.config)
        self.feature_importance = None

    def train(self, X_train, y_train):
        """
        Huấn luyện Random Forest.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels
        """
        logger.info("Bắt đầu huấn luyện Random Forest...")
        self.model.fit(X_train, y_train)
        
        self.feature_importance = self.model.feature_importances_
        
        logger.info(f"Hoàn tất huấn luyện Random Forest!")
        logger.info(f"  - Training Accuracy: {self.model.score(X_train, y_train):.4f}")
        
        return self

    def predict(self, X):
        """Dự đoán nhãn."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Dự đoán xác suất."""
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names=None):
        """
        Trả về feature importance ranking.
        
        Args:
            feature_names: Danh sách tên features
            
        Returns:
            pd.DataFrame: DataFrame feature importance sorted
        """
        if self.feature_importance is None:
            return None
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df

    def save(self, filepath=None):
        """Lưu model."""
        if filepath is None:
            filepath = os.path.join(config.MODEL_DIR, "random_forest_model.pkl")
        joblib.dump(self.model, filepath)
        logger.info(f"Random Forest model saved: {filepath}")

    def load(self, filepath=None):
        """Tải model."""
        if filepath is None:
            filepath = os.path.join(config.MODEL_DIR, "random_forest_model.pkl")
        self.model = joblib.load(filepath)
        logger.info(f"Random Forest model loaded: {filepath}")


# ============================================================
# MODEL 3: XGBoost Classifier
# ============================================================
class XGBoostModel:
    """
    XGBoost Model cho dự đoán xu hướng giá.
    
    Cấu hình:
    - learning_rate=0.05, n_estimators=500
    - max_depth=8
    - L1 + L2 regularization
    
    Ưu điểm:
    - State-of-the-art cho dữ liệu dạng bảng
    - Gradient boosting giảm bias hiệu quả
    - Built-in regularization chống overfitting
    """

    def __init__(self, xgb_config=None):
        self.config = xgb_config or config.XGB_CONFIG
        self.model = xgb.XGBClassifier(**self.config)
        self.feature_importance = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Huấn luyện XGBoost.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        logger.info("Bắt đầu huấn luyện XGBoost...")
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=50
        )
        
        self.feature_importance = self.model.feature_importances_
        
        logger.info(f"Hoàn tất huấn luyện XGBoost!")
        logger.info(f"  - Training Accuracy: {self.model.score(X_train, y_train):.4f}")
        
        return self

    def predict(self, X):
        """Dự đoán nhãn."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Dự đoán xác suất."""
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names=None):
        """Trả về feature importance ranking."""
        if self.feature_importance is None:
            return None
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df

    def save(self, filepath=None):
        """Lưu model."""
        if filepath is None:
            filepath = os.path.join(config.MODEL_DIR, "xgboost_model.pkl")
        joblib.dump(self.model, filepath)
        logger.info(f"XGBoost model saved: {filepath}")

    def load(self, filepath=None):
        """Tải model."""
        if filepath is None:
            filepath = os.path.join(config.MODEL_DIR, "xgboost_model.pkl")
        self.model = joblib.load(filepath)
        logger.info(f"XGBoost model loaded: {filepath}")


# ============================================================
# TRANING & EVALUATION UTILITIES
# ============================================================
def prepare_data_splits(df, feature_columns, target_col='label',
                       test_size=None, val_size=None):
    """
    Chia dữ liệu train/val/test theo thời gian (time-based split).
    
    QUAN TRỌNG: Không dùng random split cho time series!
    
    Args:
        df: DataFrame đầy đủ features
        feature_columns: Danh sách tên features
        target_col: Tên cột target
        test_size: Tỷ lệ test set
        val_size: Tỷ lệ validation set
        
    Returns:
        dict: {X_train, y_train, X_val, y_val, X_test, y_test, 
               train_index, val_index, test_index}
    """
    test_size = test_size or config.TEST_SIZE
    val_size = val_size or config.VALIDATION_SIZE
    
    n = len(df)
    test_start = int(n * (1 - test_size))
    val_start = int(n * (1 - test_size - val_size))
    
    # Features & Labels
    X = df[feature_columns].values
    y = df[target_col].values.astype(int)
    
    # Time-based split
    X_train = X[:val_start]
    y_train = y[:val_start]
    
    X_val = X[val_start:test_start]
    y_val = y[val_start:test_start]
    
    X_test = X[test_start:]
    y_test = y[test_start:]
    
    logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    logger.info(f"Train period: {df.index[0].strftime('%Y-%m-%d')} -> {df.index[val_start-1].strftime('%Y-%m-%d')}")
    logger.info(f"Val period: {df.index[val_start].strftime('%Y-%m-%d')} -> {df.index[test_start-1].strftime('%Y-%m-%d')}")
    logger.info(f"Test period: {df.index[test_start].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'train_index': df.index[:val_start],
        'val_index': df.index[val_start:test_start],
        'test_index': df.index[test_start:],
    }


def evaluate_model(model_name, y_true, y_pred):
    """
    Đánh giá hiệu suất model.
    
    Args:
        model_name: Tên model
        y_true: Nhãn thực
        y_pred: Nhãn dự đoán
        
    Returns:
        dict: Các metrics đánh giá
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    report = classification_report(
        y_true, y_pred,
        target_names=['DOWN', 'NEUTRAL', 'UP'],
        output_dict=True
    )
    
    logger.info(f"\n{'='*50}")
    logger.info(f"ĐÁNH GIÁ: {model_name}")
    logger.info(f"{'='*50}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1-Score (weighted): {f1:.4f}")
    logger.info(f"\nClassification Report:")
    logger.info(classification_report(y_true, y_pred, 
                                      target_names=['DOWN', 'NEUTRAL', 'UP']))
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'report': report,
    }


if __name__ == "__main__":
    print("Models module loaded successfully!")
    print("Available models: LSTMModel, RandomForestModel, XGBoostModel")
