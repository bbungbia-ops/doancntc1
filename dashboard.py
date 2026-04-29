"""
Dashboard trực quan hóa kết quả
=================================
Dashboard Streamlit hiển thị kết quả phân tích và giao dịch.
Chạy: streamlit run visualization/dashboard.py
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def fetch_live_price_data(symbol="BTC-USD", period="2y"):
    """
    Tải dữ liệu giá trực tiếp từ Yahoo Finance khi không có file CSV.
    Tính thêm các chỉ số kỹ thuật cơ bản.
    """
    import yfinance as yf
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    
    if df.empty:
        return None
    
    # Tính các chỉ số kỹ thuật cơ bản
    # SMA
    df['sma_7'] = df['Close'].rolling(window=7).mean()
    df['sma_14'] = df['Close'].rolling(window=14).mean()
    df['sma_30'] = df['Close'].rolling(window=30).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    return df


def create_price_sentiment_chart(price_df, daily_sentiment_df, signals_df=None):
    """
    Tạo biểu đồ giá + sentiment + trading signals.
    
    Args:
        price_df: DataFrame giá
        daily_sentiment_df: DataFrame sentiment theo ngày
        signals_df: DataFrame trading signals
        
    Returns:
        plotly.Figure
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.40, 0.20, 0.20, 0.20],
        subplot_titles=[
            f'📈 Giá {config.CRYPTO_NAME} ({config.CRYPTO_SYMBOL})',
            '📊 Sentiment Score',
            '📊 RSI (14)',
            '📊 Volume'
        ]
    )
    
    # ---- Chart 1: Price with Candlestick ----
    fig.add_trace(
        go.Candlestick(
            x=price_df.index,
            open=price_df['Open'],
            high=price_df['High'],
            low=price_df['Low'],
            close=price_df['Close'],
            name='Price',
            increasing_line_color='#00C853',
            decreasing_line_color='#FF1744',
        ),
        row=1, col=1
    )
    
    # SMA overlays
    if 'sma_7' in price_df.columns:
        fig.add_trace(go.Scatter(
            x=price_df.index, y=price_df['sma_7'],
            name='SMA 7', line=dict(color='#FFA726', width=1),
            opacity=0.7
        ), row=1, col=1)
    
    if 'sma_30' in price_df.columns:
        fig.add_trace(go.Scatter(
            x=price_df.index, y=price_df['sma_30'],
            name='SMA 30', line=dict(color='#42A5F5', width=1),
            opacity=0.7
        ), row=1, col=1)
    
    # Bollinger Bands
    if 'bb_upper' in price_df.columns:
        fig.add_trace(go.Scatter(
            x=price_df.index, y=price_df['bb_upper'],
            name='BB Upper', line=dict(color='gray', width=1, dash='dot'),
            opacity=0.3
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=price_df.index, y=price_df['bb_lower'],
            name='BB Lower', line=dict(color='gray', width=1, dash='dot'),
            fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
            opacity=0.3
        ), row=1, col=1)
    
    # Trading Signals
    if signals_df is not None:
        buy_signals = signals_df[signals_df['signal'].isin(['BUY', 'STRONG_BUY'])]
        sell_signals = signals_df[signals_df['signal'].isin(['SELL', 'STRONG_SELL'])]
        
        if len(buy_signals) > 0:
            fig.add_trace(go.Scatter(
                x=buy_signals['date'], y=buy_signals['price'],
                mode='markers', name='BUY',
                marker=dict(symbol='triangle-up', size=12, color='#00E676',
                           line=dict(width=1, color='darkgreen'))
            ), row=1, col=1)
        
        if len(sell_signals) > 0:
            fig.add_trace(go.Scatter(
                x=sell_signals['date'], y=sell_signals['price'],
                mode='markers', name='SELL',
                marker=dict(symbol='triangle-down', size=12, color='#FF1744',
                           line=dict(width=1, color='darkred'))
            ), row=1, col=1)
    
    # ---- Chart 2: Sentiment ----
    if daily_sentiment_df is not None and len(daily_sentiment_df) > 0:
        colors = ['#00C853' if s > 0 else '#FF1744' if s < 0 else '#FFA726'
                 for s in daily_sentiment_df['sentiment_mean']]
        
        fig.add_trace(go.Bar(
            x=daily_sentiment_df.index,
            y=daily_sentiment_df['sentiment_mean'],
            name='Sentiment',
            marker_color=colors,
            opacity=0.7
        ), row=2, col=1)
        
        # Sentiment MA
        if 'sentiment_ma_3' in price_df.columns:
            fig.add_trace(go.Scatter(
                x=price_df.index, y=price_df['sentiment_ma_3'],
                name='Sentiment MA(3)',
                line=dict(color='#FFA726', width=2)
            ), row=2, col=1)
    
    # ---- Chart 3: RSI ----
    if 'rsi' in price_df.columns:
        fig.add_trace(go.Scatter(
            x=price_df.index, y=price_df['rsi'],
            name='RSI(14)', line=dict(color='#AB47BC', width=1.5)
        ), row=3, col=1)
        
        # Overbought/Oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red",
                     opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green",
                     opacity=0.5, row=3, col=1)
    
    # ---- Chart 4: Volume ----
    if 'Volume' in price_df.columns:
        colors_vol = ['#00C853' if price_df['Close'].iloc[i] >= price_df['Open'].iloc[i]
                     else '#FF1744' for i in range(len(price_df))]
        fig.add_trace(go.Bar(
            x=price_df.index, y=price_df['Volume'],
            name='Volume', marker_color=colors_vol, opacity=0.5
        ), row=4, col=1)
    
    fig.update_layout(
        title=f'🚀 Crypto Trading Dashboard - {config.CRYPTO_NAME}',
        template='plotly_dark',
        height=1000,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
    )
    
    return fig


def create_equity_curve_chart(equity_df):
    """
    Tạo biểu đồ equity curve.
    
    Args:
        equity_df: DataFrame từ backtester
        
    Returns:
        plotly.Figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=['💰 Equity Curve', '📉 Drawdown']
    )
    
    # Equity curve - Strategy
    fig.add_trace(go.Scatter(
        x=equity_df['date'], y=equity_df['equity'],
        name='Strategy', fill='tozeroy',
        line=dict(color='#00E676', width=2),
        fillcolor='rgba(0, 230, 118, 0.1)'
    ), row=1, col=1)
    
    # Buy & Hold comparison
    if 'buy_hold_equity' in equity_df.columns:
        fig.add_trace(go.Scatter(
            x=equity_df['date'], y=equity_df['buy_hold_equity'],
            name='Buy & Hold',
            line=dict(color='#FFA726', width=1.5, dash='dash')
        ), row=1, col=1)
    
    # Drawdown
    fig.add_trace(go.Scatter(
        x=equity_df['date'], y=equity_df['drawdown'],
        name='Drawdown', fill='tozeroy',
        line=dict(color='#FF1744', width=1),
        fillcolor='rgba(255, 23, 68, 0.3)'
    ), row=2, col=1)
    
    fig.update_layout(
        title='💰 Equity Curve & Drawdown Analysis',
        template='plotly_dark',
        height=600,
        showlegend=True,
    )
    
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    return fig


def create_model_comparison_chart(comparison_df):
    """
    Tạo biểu đồ so sánh models.
    
    Args:
        comparison_df: DataFrame so sánh models
        
    Returns:
        plotly.Figure
    """
    fig = go.Figure()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 (macro)', 'F1 (weighted)']
    available_metrics = [m for m in metrics if m in comparison_df.columns]
    
    colors = ['#00E676', '#42A5F5', '#FFA726', '#AB47BC', '#FF1744']
    
    for i, metric in enumerate(available_metrics):
        fig.add_trace(go.Bar(
            name=metric,
            x=comparison_df['Model'],
            y=comparison_df[metric],
            text=comparison_df[metric].round(4),
            textposition='auto',
            marker_color=colors[i % len(colors)],
            opacity=0.8
        ))
    
    fig.update_layout(
        title='📊 So sánh Hiệu suất các Models',
        template='plotly_dark',
        barmode='group',
        height=500,
        xaxis_title='Model',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1]),
    )
    
    return fig


def create_sentiment_distribution_chart(sentiment_df):
    """
    Tạo biểu đồ phân bố sentiment.
    
    Args:
        sentiment_df: DataFrame với sentiment scores
        
    Returns:
        plotly.Figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Phân bố Ensemble Score', 'VADER vs FinBERT']
    )
    
    # Histogram sentiment score
    fig.add_trace(go.Histogram(
        x=sentiment_df['ensemble_score'],
        nbinsx=50,
        name='Ensemble Score',
        marker_color='#42A5F5',
        opacity=0.7
    ), row=1, col=1)
    
    # VADER vs FinBERT scatter
    if 'vader_score' in sentiment_df.columns and 'finbert_score' in sentiment_df.columns:
        fig.add_trace(go.Scatter(
            x=sentiment_df['vader_score'],
            y=sentiment_df['finbert_score'],
            mode='markers',
            name='VADER vs FinBERT',
            marker=dict(
                color=sentiment_df['ensemble_score'],
                colorscale='RdYlGn',
                size=5,
                opacity=0.6,
                colorbar=dict(title='Score')
            )
        ), row=1, col=2)
    
    fig.update_layout(
        title='🎭 Phân tích Sentiment Distribution',
        template='plotly_dark',
        height=400,
    )
    
    fig.update_xaxes(title_text='Ensemble Score', row=1, col=1)
    fig.update_xaxes(title_text='VADER Score', row=1, col=2)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    fig.update_yaxes(title_text='FinBERT Score', row=1, col=2)
    
    return fig


def create_confusion_matrix_chart(cm, title="Confusion Matrix"):
    """
    Tạo biểu đồ confusion matrix.
    
    Args:
        cm: Confusion matrix array
        title: Tiêu đề
        
    Returns:
        plotly.Figure
    """
    labels = ['DOWN', 'NEUTRAL', 'UP']
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cm,
        texttemplate='%{text}',
        colorscale='Blues',
        showscale=True,
    ))
    
    fig.update_layout(
        title=f'🔲 {title}',
        template='plotly_dark',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400,
        width=500,
    )
    
    return fig


def run_streamlit_dashboard():
    """
    Chạy Streamlit Dashboard.
    Tự động tải dữ liệu và chạy pipeline trực tiếp, không phụ thuộc file CSV.
    Sử dụng: streamlit run visualization/dashboard.py
    """
    try:
        import streamlit as st
    except ImportError:
        print("Cần cài đặt streamlit: pip install streamlit")
        return

    from visualization.live_pipeline import (
        collect_news_live, preprocess_news_live, analyze_sentiment_live,
        build_features_live, train_models_live, run_backtesting_live
    )

    st.set_page_config(
        page_title="Crypto Sentiment Trading Dashboard",
        page_icon="🚀",
        layout="wide"
    )

    st.title("🚀 Crypto Sentiment Trading Dashboard")
    st.markdown("**Chiến lược giao dịch tự động dựa trên Phân tích Tâm lý Thị trường**")

    # Sidebar
    st.sidebar.header("⚙️ Cấu hình")
    st.sidebar.info(f"""
    **Symbol**: {config.CRYPTO_SYMBOL}  
    **Period**: {config.START_DATE} → {config.END_DATE}  
    **Models**: RF + XGBoost  
    **Sentiment**: VADER + FinBERT  
    """)

    # ---- Cached data loading functions ----
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_price_data():
        return fetch_live_price_data(config.CRYPTO_SYMBOL)

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_sentiment_data():
        news_df = collect_news_live()
        if news_df is not None and len(news_df) > 0:
            processed = preprocess_news_live(news_df)
            return analyze_sentiment_live(processed)
        return None, None

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_features_and_models(_price_df, _daily_sentiment):
        features_df, feature_cols = build_features_live(_price_df, _daily_sentiment)
        comparison_df, predictions, test_dates, test_prices, test_sentiment, _, _ = \
            train_models_live(features_df, feature_cols)
        backtest_df = run_backtesting_live(predictions, test_dates, test_prices, test_sentiment)
        return features_df, comparison_df, backtest_df

    try:
        # Bước 1: Tải giá
        with st.spinner("🚀 Đang tải dữ liệu trực tiếp từ Yahoo Finance..."):
            price_df = load_price_data()
        if price_df is None:
            st.error("Không thể tải dữ liệu giá. Vui lòng thử lại sau.")
            return
        st.success(f"✅ Đã tải {len(price_df)} ngày dữ liệu giá {config.CRYPTO_NAME}!")

        # Bước 2: Sentiment
        with st.spinner("🎭 Đang phân tích tâm lý thị trường..."):
            sentiment_df, daily_sentiment = load_sentiment_data()

        # Bước 3: Features + Models + Backtesting
        with st.spinner("🤖 Đang huấn luyện models và chạy backtesting..."):
            features_df, comparison_df, backtest_df = \
                load_features_and_models(price_df, daily_sentiment)

        st.sidebar.success("✅ Chế độ LIVE — Tất cả dữ liệu được tải và xử lý trực tiếp.\n"
                          "Dữ liệu được cache 1 giờ.")

        # ---- DISPLAY ----
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Tổng quan", "🎭 Sentiment", "🤖 Models", "💰 Backtesting"
        ])

        with tab1:
            st.subheader("📈 Biểu đồ giá và chỉ số kỹ thuật")
            chart_df = features_df if features_df is not None else price_df
            fig = create_price_sentiment_chart(chart_df, daily_sentiment)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                latest_price = price_df['Close'].iloc[-1]
                st.metric("Giá hiện tại", f"${latest_price:,.2f}")
            with col2:
                change = price_df['Close'].pct_change().iloc[-1] * 100
                st.metric("Thay đổi 24h", f"{change:+.2f}%")
            with col3:
                st.metric("Số ngày dữ liệu", len(price_df))
            with col4:
                if daily_sentiment is not None and len(daily_sentiment) > 0:
                    avg_sent = daily_sentiment['sentiment_mean'].mean()
                    st.metric("Sentiment TB", f"{avg_sent:+.4f}")

        with tab2:
            st.subheader("🎭 Phân tích Tâm lý Thị trường")
            if sentiment_df is not None and len(sentiment_df) > 0:
                fig = create_sentiment_distribution_chart(sentiment_df)
                st.plotly_chart(fig, use_container_width=True)
                display_cols = ['title', 'ensemble_score', 'vader_score',
                               'finbert_score', 'sentiment_label']
                available_cols = [c for c in display_cols if c in sentiment_df.columns]
                st.dataframe(sentiment_df[available_cols].head(20))
            else:
                st.warning("⚠️ Không thể thu thập tin tức. Kiểm tra kết nối mạng.")

        with tab3:
            st.subheader("🤖 So sánh Models")
            if comparison_df is not None and len(comparison_df) > 0:
                fig = create_model_comparison_chart(comparison_df)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(comparison_df)
            else:
                st.warning("⚠️ Không thể huấn luyện models.")

        with tab4:
            st.subheader("💰 Kết quả Backtesting")
            if backtest_df is not None and len(backtest_df) > 0:
                fig = create_equity_curve_chart(backtest_df)
                st.plotly_chart(fig, use_container_width=True)

                # Performance summary
                initial = config.TRADING_CONFIG['initial_capital']
                final = backtest_df['equity'].iloc[-1]
                total_return = (final / initial - 1) * 100
                max_dd = backtest_df['drawdown'].min() if 'drawdown' in backtest_df.columns else 0

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Vốn ban đầu", f"${initial:,.2f}")
                with c2:
                    st.metric("Giá trị cuối", f"${final:,.2f}")
                with c3:
                    st.metric("Lợi nhuận", f"{total_return:+.2f}%")
                with c4:
                    st.metric("Max Drawdown", f"{max_dd:.2f}%")
            else:
                st.warning("⚠️ Không thể chạy backtesting.")

    except Exception as e:
        st.error(f"Lỗi: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    run_streamlit_dashboard()
