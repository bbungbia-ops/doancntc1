"""
Main Entry Point - Pipeline Chạy Toàn bộ Hệ thống
====================================================
Chạy: python main.py

Pipeline:
1. Thu thập dữ liệu (giá + tin tức)
2. Tiền xử lý dữ liệu
3. Phân tích Sentiment (VADER + FinBERT Ensemble)
4. Feature Engineering (Technical Indicators + Sentiment)
5. Huấn luyện Models (LSTM + Random Forest + XGBoost)
6. Stacking Ensemble (Meta-Learner)
7. Sinh tín hiệu giao dịch
8. Backtesting & Đánh giá
"""

import os
import sys
import logging
import warnings
import time
import numpy as np
import pandas as pd

import config

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.DATA_DIR, 'pipeline.log'), encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def run_pipeline():
    """
    Chạy toàn bộ pipeline từ thu thập dữ liệu đến backtesting.
    """
    start_time = time.time()
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║   🚀 CRYPTO SENTIMENT TRADING STRATEGY                     ║
    ║   📊 Powered by: VADER + FinBERT + LSTM + RF + XGBoost     ║
    ║   🏆 Stacking Ensemble Meta-Learner                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # ============================================================
    # BƯỚC 1: THU THẬP DỮ LIỆU
    # ============================================================
    logger.info("\n" + "🔷" * 30)
    logger.info("BƯỚC 1: THU THẬP DỮ LIỆU")
    logger.info("🔷" * 30)
    
    from src.data_collector import collect_all_data
    price_df, news_df = collect_all_data()
    
    # ============================================================
    # BƯỚC 2: TIỀN XỬ LÝ
    # ============================================================
    logger.info("\n" + "🔷" * 30)
    logger.info("BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU")
    logger.info("🔷" * 30)
    
    from src.preprocessing import preprocess_pipeline
    processed_price, processed_news, price_preprocessor = preprocess_pipeline(price_df, news_df)
    
    # ============================================================
    # BƯỚC 3: PHÂN TÍCH SENTIMENT
    # ============================================================
    logger.info("\n" + "🔷" * 30)
    logger.info("BƯỚC 3: PHÂN TÍCH TÂM LÝ THỊ TRƯỜNG")
    logger.info("🔷" * 30)
    
    from src.sentiment_analyzer import analyze_sentiment
    sentiment_df, daily_sentiment = analyze_sentiment(processed_news)
    
    # ============================================================
    # BƯỚC 4: FEATURE ENGINEERING
    # ============================================================
    logger.info("\n" + "🔷" * 30)
    logger.info("BƯỚC 4: FEATURE ENGINEERING")
    logger.info("🔷" * 30)
    
    from src.feature_engineering import FeatureEngineer
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.build_features(processed_price, daily_sentiment)
    feature_columns = feature_engineer.get_feature_columns()
    
    logger.info(f"Features DataFrame shape: {features_df.shape}")
    logger.info(f"Số lượng features: {len(feature_columns)}")
    
    # ============================================================
    # BƯỚC 5 & 6: HUẤN LUYỆN MODELS & STACKING ENSEMBLE
    # ============================================================
    logger.info("\n" + "🔷" * 30)
    logger.info("BƯỚC 5 & 6: HUẤN LUYỆN MODELS & STACKING ENSEMBLE")
    logger.info("🔷" * 30)
    
    from src.models import prepare_data_splits
    from src.ensemble import StackingEnsemble
    from src.backtesting import ModelEvaluator
    
    # Chia dữ liệu
    data_splits = prepare_data_splits(features_df, feature_columns)
    
    X_train = data_splits['X_train']
    y_train = data_splits['y_train']
    X_val = data_splits['X_val']
    y_val = data_splits['y_val']
    X_test = data_splits['X_test']
    y_test = data_splits['y_test']
    
    # Huấn luyện Stacking Ensemble
    ensemble = StackingEnsemble()
    ensemble.train(X_train, y_train, X_val, y_val, feature_names=feature_columns)
    
    # Đánh giá trên test set
    logger.info("\n--- Đánh giá trên Test Set ---")
    
    evaluator = ModelEvaluator()
    
    # Đánh giá từng model đơn lẻ trên test set
    rf_test_pred = ensemble.rf_model.predict(X_test)
    evaluator.evaluate("Random Forest", y_test, rf_test_pred)
    
    xgb_test_pred = ensemble.xgb_model.predict(X_test)
    evaluator.evaluate("XGBoost", y_test, xgb_test_pred)
    
    # LSTM test evaluation
    try:
        seq_len = config.LSTM_CONFIG['sequence_length']
        X_test_scaled = price_preprocessor.scale_features(X_test, fit=False)
        X_test_seq = []
        for i in range(seq_len, len(X_test_scaled)):
            X_test_seq.append(X_test_scaled[i - seq_len:i])
        X_test_seq = np.array(X_test_seq)
        
        if len(X_test_seq) > 0:
            lstm_test_pred = ensemble.lstm_model.predict(X_test_seq)
            y_test_lstm = y_test[seq_len:]
            evaluator.evaluate("LSTM", y_test_lstm, lstm_test_pred)
    except Exception as e:
        logger.warning(f"Không thể đánh giá LSTM trên test set: {e}")
    
    # Đánh giá ensemble trên test set
    ensemble_test_pred = ensemble.predict(X_test)
    evaluator.evaluate("Stacking Ensemble", y_test, ensemble_test_pred)
    
    # So sánh models
    comparison_df = evaluator.compare_models()
    comparison_path = os.path.join(config.PROCESSED_DATA_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    
    # Lưu models
    ensemble.save_all_models()
    
    # ============================================================
    # BƯỚC 7: SINH TÍN HIỆU GIAO DỊCH
    # ============================================================
    logger.info("\n" + "🔷" * 30)
    logger.info("BƯỚC 7: SINH TÍN HIỆU GIAO DỊCH")
    logger.info("🔷" * 30)
    
    from src.trading_strategy import TradingSignalGenerator, PortfolioSimulator
    
    # Lấy sentiment cho test period
    test_dates = data_splits['test_index']
    test_sentiment = features_df.loc[test_dates, 'sentiment_mean'].values \
        if 'sentiment_mean' in features_df.columns \
        else np.zeros(len(test_dates))
    test_prices = features_df.loc[test_dates, 'Close'].values
    
    # Sinh signals
    signal_generator = TradingSignalGenerator()
    signals_df = signal_generator.generate_signals(
        predictions=ensemble_test_pred,
        sentiment_scores=test_sentiment,
        dates=test_dates,
        prices=test_prices
    )
    
    # Lưu signals
    signals_path = os.path.join(config.PROCESSED_DATA_DIR, "trading_signals.csv")
    signals_df.to_csv(signals_path, index=False)
    
    # ============================================================
    # BƯỚC 8: BACKTESTING
    # ============================================================
    logger.info("\n" + "🔷" * 30)
    logger.info("BƯỚC 8: BACKTESTING")
    logger.info("🔷" * 30)
    
    from src.backtesting import Backtester
    
    # Mô phỏng portfolio
    simulator = PortfolioSimulator()
    portfolio_df = simulator.simulate(signals_df)
    performance = simulator.print_performance_report()
    
    # Backtesting chi tiết
    backtester = Backtester(portfolio_df, simulator.trades)
    report = backtester.generate_full_report()
    backtester.get_trade_analysis()
    
    # ============================================================
    # KẾT QUẢ TỔNG HỢP
    # ============================================================
    elapsed_time = time.time() - start_time
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                   📊 KẾT QUẢ TỔNG HỢP                     ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  ⏱️  Thời gian chạy:    {elapsed_time:>8.1f} giây                    ║
    ║  📰 Số tin tức:         {len(news_df):>8d}                         ║
    ║  📈 Số ngày giá:        {len(price_df):>8d}                         ║
    ║  🔧 Số features:        {len(feature_columns):>8d}                         ║
    ║  📊 Số mẫu train:       {len(X_train):>8d}                         ║
    ║  📊 Số mẫu test:        {len(X_test):>8d}                         ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  💰 Vốn ban đầu:    ${config.TRADING_CONFIG['initial_capital']:>10,.2f}                    ║
    ║  💰 Giá trị cuối:   ${performance['final_value']:>10,.2f}                    ║
    ║  📈 Tổng lợi nhuận: {performance['total_return']:>10.2f}%                    ║
    ║  📊 Buy & Hold:     {performance['buy_hold_return']:>10.2f}%                    ║
    ║  📐 Sharpe Ratio:   {performance['sharpe_ratio']:>10.4f}                    ║
    ║  📉 Max Drawdown:   {performance['max_drawdown']:>10.2f}%                    ║
    ║  ✅ Win Rate:       {performance['win_rate']:>10.2f}%                    ║
    ╚══════════════════════════════════════════════════════════════╝
    
    📂 Kết quả đã được lưu tại: {config.PROCESSED_DATA_DIR}
    📊 Để xem Dashboard: streamlit run visualization/dashboard.py
    """)
    
    return {
        'features_df': features_df,
        'ensemble': ensemble,
        'signals_df': signals_df,
        'portfolio_df': portfolio_df,
        'performance': performance,
        'comparison_df': comparison_df,
        'evaluator': evaluator,
    }


if __name__ == "__main__":
    results = run_pipeline()
