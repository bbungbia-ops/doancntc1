"""
Module 1: Thu thập dữ liệu (Data Collector)
=============================================
Thu thập dữ liệu giá crypto từ Yahoo Finance và tin tức từ các nguồn RSS.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PriceDataCollector:
    """Thu thập dữ liệu giá lịch sử từ Yahoo Finance."""

    def __init__(self, symbol=None, start_date=None, end_date=None):
        self.symbol = symbol or config.CRYPTO_SYMBOL
        self.start_date = start_date or config.START_DATE
        self.end_date = end_date or config.END_DATE

    def fetch_price_data(self):
        """
        Tải dữ liệu giá lịch sử (OHLCV) từ Yahoo Finance.
        
        Returns:
            pd.DataFrame: DataFrame chứa Open, High, Low, Close, Volume
        """
        logger.info(f"Đang tải dữ liệu giá {self.symbol} từ {self.start_date} đến {self.end_date}...")
        
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(start=self.start_date, end=self.end_date, interval="1d")
            
            if df.empty:
                logger.warning("Không có dữ liệu giá. Thử dùng phương thức download...")
                df = yf.download(self.symbol, start=self.start_date, end=self.end_date)
            
            # Chuẩn hóa tên cột
            df.index.name = 'Date'
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df = df.dropna()
            
            logger.info(f"Đã tải thành công {len(df)} ngày dữ liệu giá.")
            return df
            
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu giá: {e}")
            raise

    def save_price_data(self, df=None):
        """Lưu dữ liệu giá ra CSV."""
        if df is None:
            df = self.fetch_price_data()
        
        filepath = os.path.join(config.RAW_DATA_DIR, f"price_{self.symbol.replace('-', '_')}.csv")
        df.to_csv(filepath)
        logger.info(f"Đã lưu dữ liệu giá tại: {filepath}")
        return filepath


class NewsDataCollector:
    """Thu thập tin tức crypto từ nhiều nguồn."""

    def __init__(self):
        self.news_sources = config.NEWS_SOURCES
        self.max_news = config.MAX_NEWS_PER_SOURCE
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    def fetch_rss_news(self, source_name, rss_url):
        """
        Thu thập tin tức từ RSS feed.
        
        Args:
            source_name: Tên nguồn tin
            rss_url: URL RSS feed
            
        Returns:
            list: Danh sách dict chứa thông tin tin tức
        """
        logger.info(f"Đang thu thập tin từ {source_name}...")
        news_list = []
        
        try:
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:self.max_news]:
                news_item = {
                    'source': source_name,
                    'title': entry.get('title', ''),
                    'description': self._clean_html(entry.get('summary', entry.get('description', ''))),
                    'link': entry.get('link', ''),
                    'published': self._parse_date(entry.get('published', entry.get('updated', ''))),
                }
                
                # Kết hợp title + description làm content chính
                news_item['content'] = f"{news_item['title']}. {news_item['description']}"
                news_list.append(news_item)
            
            logger.info(f"Thu thập được {len(news_list)} tin từ {source_name}")
            
        except Exception as e:
            logger.warning(f"Lỗi khi thu thập từ {source_name}: {e}")
        
        return news_list

    def fetch_cryptopanic_news(self):
        """
        Thu thập tin tức từ CryptoPanic (free tier, không cần API key).
        Scrape headlines từ trang chính.
        """
        logger.info("Đang thu thập tin từ CryptoPanic...")
        news_list = []
        url = "https://cryptopanic.com/news/"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Tìm các tiêu đề tin tức
                articles = soup.find_all('a', class_='news-cell')
                if not articles:
                    articles = soup.find_all('div', class_='news-row')
                
                for article in articles[:self.max_news]:
                    title_tag = article.find('span', class_='title-text') or article
                    title = title_tag.get_text(strip=True) if title_tag else ''
                    
                    if title:
                        news_list.append({
                            'source': 'cryptopanic',
                            'title': title,
                            'description': '',
                            'content': title,
                            'link': '',
                            'published': datetime.now().strftime('%Y-%m-%d'),
                        })
                
                logger.info(f"Thu thập được {len(news_list)} tin từ CryptoPanic")
        except Exception as e:
            logger.warning(f"Lỗi khi thu thập từ CryptoPanic: {e}")
        
        return news_list

    def collect_all_news(self):
        """
        Thu thập tin tức từ tất cả các nguồn.
        
        Returns:
            pd.DataFrame: DataFrame chứa tất cả tin tức
        """
        all_news = []
        
        # Thu thập từ RSS feeds
        for source_name, rss_url in self.news_sources.items():
            news = self.fetch_rss_news(source_name, rss_url)
            all_news.extend(news)
            time.sleep(1)  # Tránh rate limiting
        
        # Thu thập từ CryptoPanic
        crypto_panic_news = self.fetch_cryptopanic_news()
        all_news.extend(crypto_panic_news)
        
        if not all_news:
            logger.warning("Không thu thập được tin tức nào. Sử dụng dữ liệu mẫu...")
            all_news = self._generate_sample_news()
        
        df = pd.DataFrame(all_news)
        
        # Đảm bảo cột published có format đúng
        df['published'] = pd.to_datetime(df['published'], errors='coerce')
        df = df.dropna(subset=['content'])
        df = df.drop_duplicates(subset=['title'])
        df = df.sort_values('published', ascending=False)
        
        logger.info(f"Tổng cộng thu thập được {len(df)} tin tức.")
        return df

    def save_news_data(self, df=None):
        """Lưu dữ liệu tin tức ra CSV."""
        if df is None:
            df = self.collect_all_news()
        
        filepath = os.path.join(config.RAW_DATA_DIR, "news_data.csv")
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"Đã lưu dữ liệu tin tức tại: {filepath}")
        return filepath

    def _clean_html(self, text):
        """Loại bỏ HTML tags."""
        if not text:
            return ''
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text(strip=True)

    def _parse_date(self, date_str):
        """Chuyển đổi chuỗi ngày về format chuẩn."""
        if not date_str:
            return datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Thử các format phổ biến
            for fmt in ['%a, %d %b %Y %H:%M:%S %z', '%Y-%m-%dT%H:%M:%S%z',
                        '%Y-%m-%d %H:%M:%S', '%a, %d %b %Y %H:%M:%S GMT']:
                try:
                    return datetime.strptime(date_str.strip(), fmt).strftime('%Y-%m-%d')
                except ValueError:
                    continue
            
            # Fallback: dùng pandas
            return pd.to_datetime(date_str).strftime('%Y-%m-%d')
        except Exception:
            return datetime.now().strftime('%Y-%m-%d')

    def _generate_sample_news(self):
        """
        Tạo dữ liệu tin tức mẫu cho mục đích demo/test.
        Trong thực tế sẽ được thay thế bằng dữ liệu thật.
        """
        logger.info("Tạo dữ liệu tin tức mẫu...")
        
        sample_headlines = [
            # Positive
            ("Bitcoin surges past $100,000 as institutional adoption accelerates", "positive"),
            ("Ethereum ETF approval drives massive crypto inflows", "positive"),
            ("Major banks adopt blockchain technology for cross-border payments", "positive"),
            ("Bitcoin mining revenue hits all-time high amid bull market", "positive"),
            ("Crypto market cap reaches new record as mainstream adoption grows", "positive"),
            ("DeFi total value locked surpasses $200 billion milestone", "positive"),
            ("BlackRock increases Bitcoin ETF holdings significantly", "positive"),
            ("Bitcoin hash rate reaches record high, network stronger than ever", "positive"),
            ("Institutional investors pour billions into crypto funds", "positive"),
            ("Major retailer announces Bitcoin payment acceptance", "positive"),
            ("Bitcoin whale accumulation signals continued bull trend", "positive"),
            ("Ethereum staking yields attract traditional finance investors", "positive"),
            ("Crypto adoption in emerging markets shows exponential growth", "positive"),
            ("SEC approves multiple spot Bitcoin ETF applications", "positive"),
            ("Layer 2 solutions drive Ethereum scalability breakthrough", "positive"),
            
            # Negative
            ("Crypto exchange hacked, millions in Bitcoin stolen", "negative"),
            ("China announces new crackdown on cryptocurrency trading", "negative"),
            ("Bitcoin drops 15% amid global market uncertainty", "negative"),
            ("SEC files lawsuit against major crypto exchange", "negative"),
            ("Crypto lender files for bankruptcy affecting thousands", "negative"),
            ("Regulatory fears cause massive crypto market selloff", "negative"),
            ("Bitcoin miners face profitability crisis as prices decline", "negative"),
            ("Major stablecoin depegs causing market panic", "negative"),
            ("Crypto scam losses exceed $10 billion this year", "negative"),
            ("Federal Reserve hawkish stance pressures Bitcoin price", "negative"),
            ("Bitcoin faces resistance as bears dominate short-term outlook", "negative"),
            ("Crypto market liquidations hit $500 million in 24 hours", "negative"),
            ("Government proposes strict crypto taxation framework", "negative"),
            ("Major crypto project rug pull affects thousands of investors", "negative"),
            ("Bitcoin correlation with tech stocks increases during downturn", "negative"),
            
            # Neutral
            ("Bitcoin trades sideways as market awaits Federal Reserve decision", "neutral"),
            ("Crypto market volume remains steady amid consolidation", "neutral"),
            ("New blockchain project launches with focus on interoperability", "neutral"),
            ("Bitcoin halving event expected to impact mining economics", "neutral"),
            ("Analysts divided on short-term crypto market direction", "neutral"),
            ("Cryptocurrency regulation framework under development", "neutral"),
            ("Bitcoin network processes record number of transactions", "neutral"),
            ("Crypto industry undergoes consolidation phase", "neutral"),
            ("New stablecoin regulation proposal released for comment", "neutral"),
            ("Bitcoin dominance holds steady as altcoins trade mixed", "neutral"),
        ]
        
        news_list = []
        base_date = datetime.now()
        
        for i, (headline, sentiment_label) in enumerate(sample_headlines):
            # Phân bố tin tức qua nhiều ngày
            days_ago = i % 60  # Phân bố trong 60 ngày
            pub_date = (base_date - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            
            news_list.append({
                'source': 'sample_data',
                'title': headline,
                'description': headline,
                'content': headline,
                'link': '',
                'published': pub_date,
            })
        
        # Nhân bản dữ liệu để có nhiều mẫu hơn cho các ngày khác nhau
        extended_news = []
        for item in news_list:
            extended_news.append(item)
            # Tạo thêm bản sao với ngày khác nhau
            for offset in [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]:
                new_item = item.copy()
                orig_date = datetime.strptime(item['published'], '%Y-%m-%d')
                new_date = orig_date - timedelta(days=offset)
                new_item['published'] = new_date.strftime('%Y-%m-%d')
                extended_news.append(new_item)
        
        return extended_news


def collect_all_data():
    """
    Thu thập toàn bộ dữ liệu cần thiết.
    
    Returns:
        tuple: (price_df, news_df)
    """
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU THU THẬP DỮ LIỆU")
    logger.info("=" * 60)
    
    # Thu thập dữ liệu giá
    price_collector = PriceDataCollector()
    price_df = price_collector.fetch_price_data()
    price_collector.save_price_data(price_df)
    
    # Thu thập tin tức
    news_collector = NewsDataCollector()
    news_df = news_collector.collect_all_news()
    news_collector.save_news_data(news_df)
    
    logger.info("=" * 60)
    logger.info("HOÀN TẤT THU THẬP DỮ LIỆU")
    logger.info(f"  - Dữ liệu giá: {len(price_df)} ngày")
    logger.info(f"  - Tin tức: {len(news_df)} bài")
    logger.info("=" * 60)
    
    return price_df, news_df


if __name__ == "__main__":
    price_df, news_df = collect_all_data()
    print("\nDữ liệu giá (5 dòng cuối):")
    print(price_df.tail())
    print(f"\nTin tức (5 dòng đầu):")
    print(news_df.head())
