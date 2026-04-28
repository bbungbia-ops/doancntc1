"""
Module 6: Stacking Ensemble
=============================
Kết hợp LSTM + Random Forest + XGBoost bằng phương pháp Stacking.
Meta-Learner: Logistic Regression
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.models import LSTMModel, RandomForestModel, XGBoostModel, evaluate_model
from src.preprocessing import PricePreprocessor

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StackingEnsemble:
    """
    Stacking Ensemble: LSTM + Random Forest + XGBoost
    
    Kiến trúc 2 tầng:
    
    Level 0 (Base Models):
        - LSTM: Bắt temporal patterns từ sequences
        - Random Forest: Bắt non-linear relationships  
        - XGBoost: Gradient boosting cho structured data
    
    Level 1 (Meta-Learner):
        - Logistic Regression: Học cách kết hợp tối ưu output 3 models
        - Input: Predictions (probabilities) từ 3 base models
        - Output: Final prediction [DOWN, NEUTRAL, UP]
    
    Sử dụng TimeSeriesSplit để tạo out-of-fold predictions,
    tránh data leakage trong time series.
    """

    def __init__(self, ensemble_config=None):
        self.config = ensemble_config or config.ENSEMBLE_CONFIG
        
        # Base models
        self.lstm_model = LSTMModel()
        self.rf_model = RandomForestModel()
        self.xgb_model = XGBoostModel()
        
        # Meta-learner
        if self.config['meta_learner'] == 'logistic':
            self.meta_model = LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42,
            )
        else:
            self.meta_model = RidgeClassifier(alpha=1.0)
        
        # Scalers
        self.price_preprocessor = PricePreprocessor()
        
        # State
        self.is_trained = False
        self.base_model_scores = {}
        self.ensemble_score = {}

    def _prepare_lstm_data(self, X, y=None):
        """
        Chuẩn bị dữ liệu dạng sequences cho LSTM.
        
        Args:
            X: Features array
            y: Labels array
            
        Returns:
            tuple: (X_seq, y_seq) hoặc X_seq
        """
        seq_len = config.LSTM_CONFIG['sequence_length']
        
        # Scale features cho LSTM
        X_scaled = self.price_preprocessor.scale_features(X, fit=(y is not None))
        
        if y is not None:
            X_seq, y_seq = self.price_preprocessor.create_sequences(
                X_scaled, y, sequence_length=seq_len
            )
            return X_seq, y_seq
        else:
            X_seq = []
            for i in range(seq_len, len(X_scaled)):
                X_seq.append(X_scaled[i - seq_len:i])
            return np.array(X_seq)

    def train(self, X_train, y_train, X_val, y_val, feature_names=None):
        """
        Huấn luyện toàn bộ Stacking Ensemble.
        
        Quy trình:
        1. Tạo out-of-fold predictions cho training data
        2. Train base models trên toàn bộ training data
        3. Train meta-learner trên out-of-fold predictions
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Tên các features
        """
        logger.info("=" * 60)
        logger.info("BẮT ĐẦU HUẤN LUYỆN STACKING ENSEMBLE")
        logger.info("=" * 60)
        
        n_classes = 3  # DOWN, NEUTRAL, UP
        
        # ============================================
        # BƯỚC 1: Tạo Out-of-Fold Predictions
        # ============================================
        logger.info("\n--- Bước 1: Tạo Out-of-Fold Predictions ---")
        
        # Khởi tạo arrays cho meta-features
        oof_rf_proba = np.zeros((len(X_train), n_classes))
        oof_xgb_proba = np.zeros((len(X_train), n_classes))
        oof_lstm_proba = np.zeros((len(X_train), n_classes))
        
        # TimeSeriesSplit cho cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config['cv_folds'])
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            logger.info(f"\n--- Fold {fold + 1}/{self.config['cv_folds']} ---")
            
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]
            
            # Random Forest - OOF predictions
            rf_fold = RandomForestModel()
            rf_fold.train(X_fold_train, y_fold_train)
            oof_rf_proba[val_idx] = rf_fold.predict_proba(X_fold_val)
            
            # XGBoost - OOF predictions
            xgb_fold = XGBoostModel()
            xgb_fold.train(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
            oof_xgb_proba[val_idx] = xgb_fold.predict_proba(X_fold_val)
            
            # LSTM - OOF predictions
            try:
                lstm_fold = LSTMModel()
                X_lstm_train, y_lstm_train = self._prepare_lstm_data(X_fold_train, y_fold_train)
                X_lstm_val = self._prepare_lstm_data(X_fold_val)
                
                if len(X_lstm_train) > 0 and len(X_lstm_val) > 0:
                    y_lstm_val = y_fold_val[config.LSTM_CONFIG['sequence_length']:]
                    lstm_fold.train(X_lstm_train, y_lstm_train)
                    lstm_proba = lstm_fold.predict_proba(X_lstm_val)
                    
                    # Align LSTM predictions (chiều ngắn hơn do sequence)
                    seq_len = config.LSTM_CONFIG['sequence_length']
                    lstm_aligned = np.zeros((len(X_fold_val), n_classes))
                    lstm_aligned[:, 1] = 1.0  # Default to NEUTRAL
                    lstm_aligned[seq_len:seq_len + len(lstm_proba)] = lstm_proba
                    oof_lstm_proba[val_idx] = lstm_aligned
                else:
                    oof_lstm_proba[val_idx, 1] = 1.0  # Default NEUTRAL
            except Exception as e:
                logger.warning(f"LSTM fold {fold+1} lỗi: {e}. Dùng default predictions.")
                oof_lstm_proba[val_idx, 1] = 1.0
        
        # ============================================
        # BƯỚC 2: Train Base Models trên toàn bộ data
        # ============================================
        logger.info("\n--- Bước 2: Train Base Models (Full Training Data) ---")
        
        # Train Random Forest
        logger.info("\n🌲 Training Random Forest...")
        self.rf_model.train(X_train, y_train)
        rf_val_pred = self.rf_model.predict(X_val)
        self.base_model_scores['Random Forest'] = evaluate_model(
            "Random Forest", y_val, rf_val_pred
        )
        
        # Train XGBoost
        logger.info("\n⚡ Training XGBoost...")
        self.xgb_model.train(X_train, y_train, X_val, y_val)
        xgb_val_pred = self.xgb_model.predict(X_val)
        self.base_model_scores['XGBoost'] = evaluate_model(
            "XGBoost", y_val, xgb_val_pred
        )
        
        # Train LSTM
        logger.info("\n🧠 Training LSTM...")
        try:
            X_lstm_train, y_lstm_train = self._prepare_lstm_data(X_train, y_train)
            X_lstm_val = self._prepare_lstm_data(X_val)
            y_lstm_val = y_val[config.LSTM_CONFIG['sequence_length']:]
            
            if len(X_lstm_train) > 0:
                self.lstm_model.train(X_lstm_train, y_lstm_train,
                                     X_lstm_val[:len(y_lstm_val)], y_lstm_val)
                lstm_val_pred = self.lstm_model.predict(X_lstm_val[:len(y_lstm_val)])
                self.base_model_scores['LSTM'] = evaluate_model(
                    "LSTM", y_lstm_val, lstm_val_pred
                )
        except Exception as e:
            logger.warning(f"Lỗi LSTM training: {e}")
        
        # Feature importance
        if feature_names is not None:
            rf_importance = self.rf_model.get_feature_importance(feature_names)
            xgb_importance = self.xgb_model.get_feature_importance(feature_names)
            
            logger.info("\n📊 Top 10 Features (Random Forest):")
            if rf_importance is not None:
                logger.info(f"\n{rf_importance.head(10).to_string()}")
            
            logger.info("\n📊 Top 10 Features (XGBoost):")
            if xgb_importance is not None:
                logger.info(f"\n{xgb_importance.head(10).to_string()}")
        
        # ============================================
        # BƯỚC 3: Train Meta-Learner
        # ============================================
        logger.info("\n--- Bước 3: Train Meta-Learner ---")
        
        # Tạo meta-features từ OOF predictions
        meta_train = np.hstack([oof_rf_proba, oof_xgb_proba, oof_lstm_proba])
        
        # Loại bỏ các dòng có toàn bộ zero (từ initial folds)
        valid_mask = meta_train.sum(axis=1) > 0
        meta_train_valid = meta_train[valid_mask]
        y_train_valid = y_train[valid_mask]
        
        # Train meta-learner
        self.meta_model.fit(meta_train_valid, y_train_valid)
        
        logger.info("Meta-Learner đã được huấn luyện!")
        
        # Đánh giá trên validation set
        meta_val_pred = self.predict(X_val)
        self.ensemble_score = evaluate_model("Stacking Ensemble", y_val, meta_val_pred)
        
        self.is_trained = True
        
        logger.info("\n" + "=" * 60)
        logger.info("HOÀN TẤT HUẤN LUYỆN STACKING ENSEMBLE")
        logger.info("=" * 60)
        
        # So sánh kết quả
        self._print_comparison()
        
        return self

    def predict(self, X):
        """
        Dự đoán với Stacking Ensemble.
        
        Args:
            X: Features array
            
        Returns:
            np.ndarray: Predicted labels
        """
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X):
        """Dự đoán xác suất."""
        meta_features = self._get_meta_features(X)
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_features)
        else:
            # RidgeClassifier doesn't have predict_proba
            decisions = self.meta_model.decision_function(meta_features)
            from scipy.special import softmax
            return softmax(decisions, axis=1)

    def _get_meta_features(self, X):
        """
        Tạo meta-features từ base model predictions.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Meta-features (stacked probabilities)
        """
        # Random Forest predictions
        rf_proba = self.rf_model.predict_proba(X)
        
        # XGBoost predictions
        xgb_proba = self.xgb_model.predict_proba(X)
        
        # LSTM predictions
        n_classes = 3
        try:
            X_lstm = self._prepare_lstm_data(X)
            if len(X_lstm) > 0:
                lstm_proba_raw = self.lstm_model.predict_proba(X_lstm)
                
                # Align LSTM predictions
                seq_len = config.LSTM_CONFIG['sequence_length']
                lstm_proba = np.zeros((len(X), n_classes))
                lstm_proba[:, 1] = 1.0  # Default NEUTRAL
                lstm_proba[seq_len:seq_len + len(lstm_proba_raw)] = lstm_proba_raw
            else:
                lstm_proba = np.zeros((len(X), n_classes))
                lstm_proba[:, 1] = 1.0
        except Exception:
            lstm_proba = np.zeros((len(X), n_classes))
            lstm_proba[:, 1] = 1.0
        
        # Stack all probabilities
        return np.hstack([rf_proba, xgb_proba, lstm_proba])

    def _print_comparison(self):
        """In bảng so sánh kết quả các models."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 SO SÁNH KẾT QUẢ GIỮA CÁC MODELS")
        logger.info("=" * 60)
        
        comparison = []
        for name, scores in self.base_model_scores.items():
            comparison.append({
                'Model': name,
                'Accuracy': f"{scores['accuracy']:.4f}",
                'F1-Score': f"{scores['f1_score']:.4f}",
            })
        
        if self.ensemble_score:
            comparison.append({
                'Model': '🏆 Stacking Ensemble',
                'Accuracy': f"{self.ensemble_score['accuracy']:.4f}",
                'F1-Score': f"{self.ensemble_score['f1_score']:.4f}",
            })
        
        comparison_df = pd.DataFrame(comparison)
        logger.info(f"\n{comparison_df.to_string(index=False)}")

    def save_all_models(self):
        """Lưu tất cả models."""
        logger.info("Đang lưu tất cả models...")
        
        self.rf_model.save()
        self.xgb_model.save()
        
        try:
            self.lstm_model.save()
        except Exception as e:
            logger.warning(f"Không thể lưu LSTM: {e}")
        
        meta_path = os.path.join(config.MODEL_DIR, "meta_learner.pkl")
        joblib.dump(self.meta_model, meta_path)
        
        logger.info("Đã lưu tất cả models thành công!")

    def load_all_models(self):
        """Tải tất cả models."""
        logger.info("Đang tải tất cả models...")
        
        self.rf_model.load()
        self.xgb_model.load()
        
        try:
            self.lstm_model.load()
        except Exception as e:
            logger.warning(f"Không thể tải LSTM: {e}")
        
        meta_path = os.path.join(config.MODEL_DIR, "meta_learner.pkl")
        self.meta_model = joblib.load(meta_path)
        
        self.is_trained = True
        logger.info("Đã tải tất cả models thành công!")

    def get_model_comparison(self):
        """
        Trả về bảng so sánh kết quả.
        
        Returns:
            pd.DataFrame: Comparison table
        """
        rows = []
        for name, scores in self.base_model_scores.items():
            rows.append({
                'Model': name,
                'Accuracy': scores['accuracy'],
                'F1-Score': scores['f1_score'],
            })
        
        if self.ensemble_score:
            rows.append({
                'Model': 'Stacking Ensemble',
                'Accuracy': self.ensemble_score['accuracy'],
                'F1-Score': self.ensemble_score['f1_score'],
            })
        
        return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Stacking Ensemble module loaded successfully!")
    print("Components: LSTM + Random Forest + XGBoost -> Meta-Learner (Logistic Regression)")
