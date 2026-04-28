"""
Module 3: Sentiment Analysis Ensemble (VADER + FinBERT)
========================================================
Kết hợp VADER (rule-based) và FinBERT (transformer-based)
để phân tích tâm lý thị trường crypto từ tin tức.
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VADERAnalyzer:
    """
    Phân tích tâm lý sử dụng VADER (Valence Aware Dictionary and sEntiment Reasoner).
    
    VADER là thuật toán rule-based, hoạt động tốt với:
    - Văn bản social media, tin tức ngắn
    - Emoji, viết tắt, slang
    - Xử lý nhanh, không cần GPU
    """

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Thêm từ vựng crypto-specific vào VADER lexicon
        crypto_lexicon = {
            'bullish': 3.0,
            'bearish': -3.0,
            'moon': 2.5,
            'dump': -2.5,
            'pump': 2.0,
            'crash': -3.5,
            'rally': 2.5,
            'surge': 2.5,
            'plunge': -3.0,
            'soar': 2.5,
            'tank': -2.5,
            'breakout': 2.0,
            'breakdown': -2.0,
            'hodl': 1.5,
            'fud': -2.0,
            'fomo': 1.0,
            'whale': 0.5,
            'rug pull': -4.0,
            'scam': -3.5,
            'hack': -3.0,
            'adoption': 2.0,
            'regulation': -0.5,
            'ban': -3.0,
            'etf': 1.5,
            'approval': 2.5,
            'rejection': -2.5,
            'all-time high': 3.0,
            'ath': 3.0,
            'dip': -1.0,
            'correction': -1.5,
            'recovery': 2.0,
            'accumulation': 1.5,
            'distribution': -1.0,
            'liquidation': -2.5,
            'defi': 1.0,
            'institutional': 1.5,
            'mainstream': 1.5,
            'volatile': -0.5,
            'stable': 0.5,
            'growth': 2.0,
            'decline': -2.0,
            'outperform': 2.0,
            'underperform': -2.0,
        }
        self.analyzer.lexicon.update(crypto_lexicon)
        logger.info("VADER Analyzer đã khởi tạo (với crypto lexicon)")

    def analyze(self, text):
        """
        Phân tích sentiment của một đoạn text.
        
        Args:
            text: Chuỗi văn bản cần phân tích
            
        Returns:
            dict: {compound, positive, negative, neutral}
        """
        if not isinstance(text, str) or not text.strip():
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        scores = self.analyzer.polarity_scores(text)
        return {
            'compound': scores['compound'],     # Điểm tổng hợp [-1, 1]
            'positive': scores['pos'],          # Xác suất positive
            'negative': scores['neg'],          # Xác suất negative
            'neutral': scores['neu'],           # Xác suất neutral
        }

    def analyze_batch(self, texts):
        """Phân tích sentiment cho danh sách texts."""
        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results


class FinBERTAnalyzer:
    """
    Phân tích tâm lý sử dụng FinBERT (Financial BERT).
    
    FinBERT là model transformer pre-trained trên dữ liệu tài chính,
    hiểu sâu ngữ cảnh và thuật ngữ tài chính/crypto.
    
    Model: ProsusAI/finbert
    """

    def __init__(self, model_name=None):
        self.model_name = model_name or config.FINBERT_MODEL_NAME
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Tải FinBERT model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            
            logger.info(f"Đang tải FinBERT model: {self.model_name}...")
            
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                return_all_scores=True,
                truncation=True,
                max_length=512,
            )
            
            logger.info("FinBERT model đã tải thành công!")
            
        except Exception as e:
            logger.warning(f"Không thể tải FinBERT: {e}")
            logger.warning("Sẽ sử dụng fallback sentiment method.")
            self.pipeline = None

    def analyze(self, text):
        """
        Phân tích sentiment với FinBERT.
        
        Args:
            text: Văn bản cần phân tích
            
        Returns:
            dict: {compound, positive, negative, neutral}
        """
        if not isinstance(text, str) or not text.strip():
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        if self.pipeline is None:
            return self._fallback_analyze(text)
        
        try:
            # Giới hạn độ dài input
            text = text[:512]
            result = self.pipeline(text)[0]
            
            # FinBERT trả về: positive, negative, neutral
            scores = {item['label'].lower(): item['score'] for item in result}
            
            positive = scores.get('positive', 0.0)
            negative = scores.get('negative', 0.0)
            neutral = scores.get('neutral', 0.0)
            
            # Tính compound score từ FinBERT: positive - negative
            compound = positive - negative
            
            return {
                'compound': compound,
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
            }
            
        except Exception as e:
            logger.warning(f"Lỗi FinBERT: {e}")
            return self._fallback_analyze(text)

    def _fallback_analyze(self, text):
        """
        Phương pháp fallback khi FinBERT không khả dụng.
        Sử dụng TextBlob hoặc từ điển đơn giản.
        """
        text_lower = text.lower()
        
        positive_words = {
            'surge', 'soar', 'rally', 'bull', 'bullish', 'gain', 'growth',
            'high', 'record', 'breakthrough', 'adoption', 'approve', 'approval',
            'profit', 'moon', 'pump', 'breakout', 'recover', 'recovery',
            'increase', 'rise', 'rising', 'up', 'positive', 'strong',
            'milestone', 'success', 'institutional', 'inflow', 'accumulation',
        }
        
        negative_words = {
            'crash', 'plunge', 'bear', 'bearish', 'drop', 'decline', 'loss',
            'low', 'hack', 'scam', 'ban', 'crackdown', 'lawsuit', 'fraud',
            'dump', 'tank', 'bankruptcy', 'liquidation', 'sell-off', 'selloff',
            'fear', 'panic', 'negative', 'weak', 'risk', 'concern', 'warning',
            'regulation', 'restrict', 'penalty', 'fine', 'stolen', 'rug pull',
        }
        
        words = set(text_lower.split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        total = pos_count + neg_count
        
        if total == 0:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        positive = pos_count / total
        negative = neg_count / total
        compound = (pos_count - neg_count) / total
        
        return {
            'compound': compound,
            'positive': positive,
            'negative': negative,
            'neutral': max(0, 1., - positive - negative),
        }

    def analyze_batch(self, texts, batch_size=16):
        """Phân tích sentiment cho danh sách texts theo batch."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                results.append(self.analyze(text))
        return results


class SentimentEnsemble:
    """
    Ensemble VADER + FinBERT cho phân tích tâm lý thị trường.
    
    Kết hợp:
    - VADER (w=0.3): Nhanh, tốt với slang/emoji
    - FinBERT (w=0.7): Hiểu ngữ cảnh tài chính sâu
    
    Weighted Average: final_score = w1*VADER + w2*FinBERT
    """

    def __init__(self, vader_weight=None, finbert_weight=None):
        self.vader_weight = vader_weight or config.VADER_WEIGHT
        self.finbert_weight = finbert_weight or config.FINBERT_WEIGHT
        
        # Khởi tạo analyzers
        logger.info("Khởi tạo Sentiment Ensemble...")
        self.vader = VADERAnalyzer()
        self.finbert = FinBERTAnalyzer()
        
        logger.info(f"Trọng số: VADER={self.vader_weight}, FinBERT={self.finbert_weight}")

    def analyze(self, text):
        """
        Phân tích sentiment bằng ensemble method.
        
        Args:
            text: Văn bản cần phân tích
            
        Returns:
            dict: {
                ensemble_score, vader_score, finbert_score,
                sentiment_label, confidence
            }
        """
        # Phân tích từng model
        vader_result = self.vader.analyze(text)
        finbert_result = self.finbert.analyze(text)
        
        # Weighted Average Ensemble
        ensemble_score = (
            self.vader_weight * vader_result['compound'] +
            self.finbert_weight * finbert_result['compound']
        )
        
        # Xác định label
        if ensemble_score > config.SENTIMENT_POSITIVE_THRESHOLD:
            label = 'positive'
        elif ensemble_score < config.SENTIMENT_NEGATIVE_THRESHOLD:
            label = 'negative'
        else:
            label = 'neutral'
        
        # Confidence: mức độ đồng thuận giữa 2 models
        agreement = 1.0 - abs(vader_result['compound'] - finbert_result['compound']) / 2.0
        
        return {
            'ensemble_score': round(ensemble_score, 4),
            'vader_score': round(vader_result['compound'], 4),
            'finbert_score': round(finbert_result['compound'], 4),
            'sentiment_label': label,
            'confidence': round(agreement, 4),
        }

    def analyze_dataframe(self, df, text_column='sentiment_text'):
        """
        Phân tích sentiment cho toàn bộ DataFrame.
        
        Args:
            df: DataFrame chứa cột text
            text_column: Tên cột text cần phân tích
            
        Returns:
            pd.DataFrame: DataFrame với các cột sentiment mới
        """
        logger.info(f"Đang phân tích sentiment cho {len(df)} tin tức...")
        
        df = df.copy()
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Sentiment"):
            text = row.get(text_column, row.get('content', ''))
            result = self.analyze(text)
            results.append(result)
        
        # Thêm kết quả vào DataFrame
        result_df = pd.DataFrame(results)
        for col in result_df.columns:
            df[col] = result_df[col].values
        
        logger.info(f"Phân bố sentiment: "
                    f"Positive={sum(df['sentiment_label']=='positive')}, "
                    f"Negative={sum(df['sentiment_label']=='negative')}, "
                    f"Neutral={sum(df['sentiment_label']=='neutral')}")
        
        return df

    def aggregate_daily_sentiment(self, df, date_column='published'):
        """
        Tổng hợp sentiment theo ngày.
        
        Args:
            df: DataFrame đã có sentiment scores
            date_column: Tên cột ngày
            
        Returns:
            pd.DataFrame: Sentiment trung bình theo ngày
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df['date'] = df[date_column].dt.date
        
        daily_sentiment = df.groupby('date').agg({
            'ensemble_score': ['mean', 'std', 'min', 'max', 'count'],
            'vader_score': 'mean',
            'finbert_score': 'mean',
            'confidence': 'mean',
        })
        
        # Flatten multi-level columns
        daily_sentiment.columns = [
            'sentiment_mean', 'sentiment_std', 'sentiment_min',
            'sentiment_max', 'news_count',
            'vader_mean', 'finbert_mean', 'confidence_mean'
        ]
        
        # Fill NaN std với 0
        daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_std'].fillna(0)
        
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
        daily_sentiment = daily_sentiment.sort_index()
        
        logger.info(f"Tổng hợp sentiment cho {len(daily_sentiment)} ngày.")
        
        return daily_sentiment


def analyze_sentiment(news_df):
    """
    Pipeline phân tích sentiment hoàn chỉnh.
    
    Args:
        news_df: DataFrame tin tức đã tiền xử lý
        
    Returns:
        tuple: (sentiment_df, daily_sentiment_df)
    """
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU PHÂN TÍCH TÂM LÝ THỊ TRƯỜNG")
    logger.info("=" * 60)
    
    ensemble = SentimentEnsemble()
    
    # Phân tích từng tin
    sentiment_df = ensemble.analyze_dataframe(news_df)
    
    # Tổng hợp theo ngày
    daily_sentiment = ensemble.aggregate_daily_sentiment(sentiment_df)
    
    # Lưu kết quả
    sentiment_path = os.path.join(config.PROCESSED_DATA_DIR, "sentiment_results.csv")
    sentiment_df.to_csv(sentiment_path, index=False, encoding='utf-8-sig')
    
    daily_path = os.path.join(config.PROCESSED_DATA_DIR, "daily_sentiment.csv")
    daily_sentiment.to_csv(daily_path)
    
    logger.info("=" * 60)
    logger.info("HOÀN TẤT PHÂN TÍCH TÂM LÝ")
    logger.info(f"  - Sentiment trung bình: {daily_sentiment['sentiment_mean'].mean():.4f}")
    logger.info(f"  - Đã lưu kết quả tại: {sentiment_path}")
    logger.info("=" * 60)
    
    return sentiment_df, daily_sentiment


if __name__ == "__main__":
    # Test sentiment analysis
    test_texts = [
        "Bitcoin surges to new all-time high as institutional adoption accelerates!",
        "Crypto exchange hacked, millions stolen in major security breach",
        "Bitcoin trades sideways amid market uncertainty and regulatory concerns",
        "Ethereum ETF approved! Massive bullish signal for crypto market",
        "China bans cryptocurrency mining, market crashes 20%",
    ]
    
    ensemble = SentimentEnsemble()
    
    print("\n" + "=" * 80)
    print("TEST SENTIMENT ANALYSIS ENSEMBLE")
    print("=" * 80)
    
    for text in test_texts:
        result = ensemble.analyze(text)
        print(f"\n📰 {text[:70]}...")
        print(f"   Ensemble: {result['ensemble_score']:+.4f} | "
              f"VADER: {result['vader_score']:+.4f} | "
              f"FinBERT: {result['finbert_score']:+.4f} | "
              f"Label: {result['sentiment_label']} | "
              f"Confidence: {result['confidence']:.4f}")
