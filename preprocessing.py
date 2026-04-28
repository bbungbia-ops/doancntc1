"""
Module 2: Tiền xử lý dữ liệu (Preprocessing)
===============================================
Làm sạch văn bản và chuẩn hóa dữ liệu số.
"""

import re
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Tiền xử lý văn bản tin tức."""

    def __init__(self):
        # Stopwords cơ bản cho tiếng Anh (không cần NLTK download)
        self.stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'out', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but',
            'and', 'or', 'if', 'while', 'about', 'up', 'down', 'this',
            'that', 'these', 'those', 'it', 'its', 'i', 'me', 'my', 'we',
            'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they',
            'them', 'their', 'what', 'which', 'who', 'whom',
        }

    def clean_text(self, text):
        """
        Làm sạch văn bản.
        
        Args:
            text: Chuỗi văn bản cần làm sạch
            
        Returns:
            str: Văn bản đã được làm sạch
        """
        if not isinstance(text, str) or not text.strip():
            return ''
        
        # 1. Loại bỏ HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # 2. Loại bỏ URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # 3. Loại bỏ mentions (@username) và hashtags
        text = re.sub(r'@\w+', '', text)
        # Giữ lại text sau # vì có thể chứa thông tin hữu ích
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # 4. Loại bỏ ký tự đặc biệt nhưng giữ dấu câu cơ bản
        text = re.sub(r'[^\w\s.,!?;:\'-]', ' ', text)
        
        # 5. Loại bỏ số đứng một mình (giữ số gắn với text như "Bitcoin2.0")
        text = re.sub(r'\b\d+\b', '', text)
        
        # 6. Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def preprocess_for_sentiment(self, text):
        """
        Tiền xử lý text cho Sentiment Analysis.
        Giữ lại nhiều ngữ cảnh hơn so với clean_text thông thường.
        
        Args:
            text: Văn bản gốc
            
        Returns:
            str: Văn bản đã tiền xử lý cho sentiment
        """
        if not isinstance(text, str) or not text.strip():
            return ''
        
        # Chỉ loại bỏ HTML và URLs, giữ lại ngữ cảnh
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Giới hạn độ dài cho FinBERT (max 512 tokens)
        words = text.split()
        if len(words) > 400:
            text = ' '.join(words[:400])
        
        return text

    def preprocess_news_dataframe(self, df):
        """
        Tiền xử lý toàn bộ DataFrame tin tức.
        
        Args:
            df: DataFrame chứa cột 'content'
            
        Returns:
            pd.DataFrame: DataFrame đã được tiền xử lý
        """
        logger.info(f"Đang tiền xử lý {len(df)} tin tức...")
        
        df = df.copy()
        
        # Tạo cột cleaned content
        df['cleaned_content'] = df['content'].apply(self.clean_text)
        
        # Tạo cột cho sentiment analysis (giữ ngữ cảnh nhiều hơn)
        df['sentiment_text'] = df['content'].apply(self.preprocess_for_sentiment)
        
        # Loại bỏ các dòng rỗng
        df = df[df['cleaned_content'].str.len() > 10]
        
        logger.info(f"Sau tiền xử lý: {len(df)} tin tức hợp lệ.")
        return df


class PricePreprocessor:
    """Tiền xử lý và chuẩn hóa dữ liệu giá."""

    def __init__(self):
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = StandardScaler()
        self._fitted = False

    def create_labels(self, df, threshold=None):
        """
        Tạo nhãn (labels) dựa trên biến động giá.
        
        Args:
            df: DataFrame chứa cột 'Close'
            threshold: Ngưỡng phần trăm thay đổi cho UP/DOWN
            
        Returns:
            pd.DataFrame: DataFrame với cột 'label' mới
        """
        threshold = threshold or config.PRICE_CHANGE_THRESHOLD
        
        df = df.copy()
        
        # Tính % thay đổi giá ngày hôm sau (target)
        df['price_change'] = df['Close'].pct_change(1).shift(-1)
        
        # Gán nhãn: 0=DOWN, 1=NEUTRAL, 2=UP
        conditions = [
            df['price_change'] < -threshold,  # DOWN
            df['price_change'] > threshold,    # UP
        ]
        choices = [0, 2]  # DOWN=0, UP=2
        df['label'] = np.select(conditions, choices, default=1)  # NEUTRAL=1
        
        # Loại bỏ dòng cuối cùng (không có label)
        df = df.dropna(subset=['price_change'])
        
        logger.info(f"Phân bố nhãn: DOWN={sum(df['label']==0)}, "
                    f"NEUTRAL={sum(df['label']==1)}, UP={sum(df['label']==2)}")
        
        return df

    def scale_prices(self, df, fit=True):
        """
        Chuẩn hóa giá về [0, 1] cho LSTM.
        
        Args:
            df: DataFrame chứa cột giá
            fit: True nếu fit scaler mới, False nếu dùng scaler đã fit
            
        Returns:
            np.ndarray: Dữ liệu đã chuẩn hóa
        """
        price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in price_columns if col in df.columns]
        
        if fit:
            scaled = self.price_scaler.fit_transform(df[available_cols])
            self._fitted = True
        else:
            scaled = self.price_scaler.transform(df[available_cols])
        
        return scaled

    def scale_features(self, features, fit=True):
        """
        Chuẩn hóa features cho tree-based models.
        
        Args:
            features: DataFrame hoặc array features
            fit: True nếu fit scaler mới
            
        Returns:
            np.ndarray: Features đã chuẩn hóa
        """
        if fit:
            return self.feature_scaler.fit_transform(features)
        else:
            return self.feature_scaler.transform(features)

    def create_sequences(self, data, labels, sequence_length=None):
        """
        Tạo sequences cho LSTM input.
        
        Args:
            data: Array dữ liệu đã chuẩn hóa
            labels: Array nhãn
            sequence_length: Độ dài mỗi sequence
            
        Returns:
            tuple: (X, y) - sequences và labels tương ứng
        """
        seq_len = sequence_length or config.LSTM_CONFIG['sequence_length']
        X, y = [], []
        
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len:i])
            y.append(labels[i])
        
        return np.array(X), np.array(y)


def preprocess_pipeline(price_df, news_df):
    """
    Pipeline tiền xử lý hoàn chỉnh.
    
    Args:
        price_df: DataFrame dữ liệu giá
        news_df: DataFrame dữ liệu tin tức
        
    Returns:
        tuple: (processed_price_df, processed_news_df)
    """
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU")
    logger.info("=" * 60)
    
    # Tiền xử lý tin tức
    text_preprocessor = TextPreprocessor()
    processed_news = text_preprocessor.preprocess_news_dataframe(news_df)
    
    # Tiền xử lý giá
    price_preprocessor = PricePreprocessor()
    processed_price = price_preprocessor.create_labels(price_df)
    
    logger.info("HOÀN TẤT TIỀN XỬ LÝ")
    
    return processed_price, processed_news, price_preprocessor


if __name__ == "__main__":
    # Test preprocessing
    sample_text = """
    <p>Bitcoin surges to $100,000! 🚀🚀🚀 
    Check out https://example.com for more @cryptoanalyst
    #BTC #cryptocurrency market is booming today!</p>
    """
    
    preprocessor = TextPreprocessor()
    print("Original:", sample_text)
    print("Cleaned:", preprocessor.clean_text(sample_text))
    print("For Sentiment:", preprocessor.preprocess_for_sentiment(sample_text))
