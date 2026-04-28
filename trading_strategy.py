"""
Module 7: Trading Strategy
============================
Chiến lược giao dịch tự động dựa trên kết hợp
dự đoán ML + Sentiment Analysis.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingSignalGenerator:
    """
    Sinh tín hiệu giao dịch (BUY/SELL/HOLD) dựa trên:
    1. Dự đoán của Stacking Ensemble (UP/DOWN/NEUTRAL)
    2. Sentiment Score từ VADER + FinBERT ensemble
    3. Ngưỡng Confidence
    
    Quy tắc:
    - BUY: Model dự đoán UP + Sentiment > +0.2
    - SELL: Model dự đoán DOWN + Sentiment < -0.2
    - HOLD: Tất cả trường hợp khác
    """

    def __init__(self, trading_config=None):
        self.config = trading_config or config.TRADING_CONFIG
        self.signals = None

    def generate_signals(self, predictions, sentiment_scores, dates=None, prices=None):
        """
        Sinh tín hiệu giao dịch.
        
        Args:
            predictions: Array dự đoán (0=DOWN, 1=NEUTRAL, 2=UP)
            sentiment_scores: Array điểm sentiment [-1, 1]
            dates: DatetimeIndex
            prices: Array giá Close
            
        Returns:
            pd.DataFrame: DataFrame chứa signals
        """
        n = len(predictions)
        
        signals = pd.DataFrame({
            'date': dates if dates is not None else range(n),
            'prediction': predictions,
            'prediction_label': [config.LABEL_MAP[p] for p in predictions],
            'sentiment': sentiment_scores[:n] if len(sentiment_scores) >= n else 
                        np.concatenate([sentiment_scores, np.zeros(n - len(sentiment_scores))]),
            'price': prices[:n] if prices is not None else np.zeros(n),
        })
        
        # Sinh signals
        signals['signal'] = 'HOLD'
        
        # BUY: Prediction = UP + Sentiment positive
        buy_condition = (
            (signals['prediction'] == 2) &  # UP
            (signals['sentiment'] > config.SENTIMENT_POSITIVE_THRESHOLD)
        )
        signals.loc[buy_condition, 'signal'] = 'BUY'
        
        # SELL: Prediction = DOWN + Sentiment negative
        sell_condition = (
            (signals['prediction'] == 0) &  # DOWN
            (signals['sentiment'] < config.SENTIMENT_NEGATIVE_THRESHOLD)
        )
        signals.loc[sell_condition, 'signal'] = 'SELL'
        
        # Strong BUY: Cả prediction UP và sentiment rất positive
        strong_buy = (
            (signals['prediction'] == 2) &
            (signals['sentiment'] > 0.5)
        )
        signals.loc[strong_buy, 'signal'] = 'STRONG_BUY'
        
        # Strong SELL: Cả prediction DOWN và sentiment rất negative
        strong_sell = (
            (signals['prediction'] == 0) &
            (signals['sentiment'] < -0.5)
        )
        signals.loc[strong_sell, 'signal'] = 'STRONG_SELL'
        
        # Signal strength (từ 0 đến 1)
        signals['signal_strength'] = abs(signals['sentiment']) * 0.5 + \
            (signals['prediction'].map({0: 0.5, 1: 0.0, 2: 0.5}))
        
        self.signals = signals
        
        # Thống kê
        signal_counts = signals['signal'].value_counts()
        logger.info(f"\n📊 Phân bố tín hiệu giao dịch:")
        for signal, count in signal_counts.items():
            emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡', 
                    'STRONG_BUY': '🟢🟢', 'STRONG_SELL': '🔴🔴'}.get(signal, '⚪')
            logger.info(f"  {emoji} {signal}: {count} ({count/len(signals)*100:.1f}%)")
        
        return signals


class PortfolioSimulator:
    """
    Mô phỏng danh mục đầu tư (Portfolio Simulation).
    
    Features:
    - Position sizing (quản lý kích thước lệnh)
    - Stop-loss & Take-profit
    - Commission fees
    - Equity tracking
    """

    def __init__(self, trading_config=None):
        self.config = trading_config or config.TRADING_CONFIG
        self.initial_capital = self.config['initial_capital']
        self.position_size = self.config['position_size']
        self.stop_loss = self.config['stop_loss']
        self.take_profit = self.config['take_profit']
        self.commission = self.config['commission']
        
        # State
        self.portfolio = None
        self.trades = []

    def simulate(self, signals_df):
        """
        Mô phỏng giao dịch dựa trên signals.
        
        Args:
            signals_df: DataFrame chứa cột 'signal' và 'price'
            
        Returns:
            pd.DataFrame: Portfolio history
        """
        logger.info("=" * 60)
        logger.info("BẮT ĐẦU MÔ PHỎNG GIAO DỊCH")
        logger.info(f"  Vốn ban đầu: ${self.initial_capital:,.2f}")
        logger.info(f"  Position size: {self.position_size*100}%")
        logger.info(f"  Stop-loss: {self.stop_loss*100}%")
        logger.info(f"  Take-profit: {self.take_profit*100}%")
        logger.info("=" * 60)
        
        cash = self.initial_capital
        holdings = 0.0  # Số lượng crypto đang nắm giữ
        entry_price = 0.0  # Giá mua vào
        
        portfolio_history = []
        self.trades = []
        
        for idx, row in signals_df.iterrows():
            price = row['price']
            signal = row['signal']
            date = row.get('date', idx)
            
            if price <= 0:
                continue
            
            # Kiểm tra Stop-loss / Take-profit nếu đang giữ lệnh
            if holdings > 0 and entry_price > 0:
                pnl_pct = (price - entry_price) / entry_price
                
                # Stop-loss triggered
                if pnl_pct <= -self.stop_loss:
                    sell_value = holdings * price * (1 - self.commission)
                    cash += sell_value
                    self.trades.append({
                        'date': date, 'type': 'STOP_LOSS',
                        'price': price, 'amount': holdings,
                        'value': sell_value, 'pnl_pct': pnl_pct
                    })
                    logger.info(f"  🛑 STOP-LOSS @ ${price:.2f} (P/L: {pnl_pct*100:+.2f}%)")
                    holdings = 0.0
                    entry_price = 0.0
                    
                # Take-profit triggered
                elif pnl_pct >= self.take_profit:
                    sell_value = holdings * price * (1 - self.commission)
                    cash += sell_value
                    self.trades.append({
                        'date': date, 'type': 'TAKE_PROFIT',
                        'price': price, 'amount': holdings,
                        'value': sell_value, 'pnl_pct': pnl_pct
                    })
                    logger.info(f"  🎯 TAKE-PROFIT @ ${price:.2f} (P/L: {pnl_pct*100:+.2f}%)")
                    holdings = 0.0
                    entry_price = 0.0
            
            # Xử lý tín hiệu BUY
            if signal in ['BUY', 'STRONG_BUY'] and holdings == 0:
                # Tính số tiền đầu tư
                invest = cash * self.position_size
                if signal == 'STRONG_BUY':
                    invest = cash * min(self.position_size * 1.5, 0.5)
                
                buy_amount = invest / price * (1 - self.commission)
                cash -= invest
                holdings = buy_amount
                entry_price = price
                
                self.trades.append({
                    'date': date, 'type': signal,
                    'price': price, 'amount': buy_amount,
                    'value': invest, 'pnl_pct': 0
                })
                
            # Xử lý tín hiệu SELL
            elif signal in ['SELL', 'STRONG_SELL'] and holdings > 0:
                sell_value = holdings * price * (1 - self.commission)
                pnl_pct = (price - entry_price) / entry_price if entry_price > 0 else 0
                cash += sell_value
                
                self.trades.append({
                    'date': date, 'type': signal,
                    'price': price, 'amount': holdings,
                    'value': sell_value, 'pnl_pct': pnl_pct
                })
                
                holdings = 0.0
                entry_price = 0.0
            
            # Tính portfolio value
            portfolio_value = cash + holdings * price
            portfolio_history.append({
                'date': date,
                'cash': cash,
                'holdings': holdings,
                'holdings_value': holdings * price,
                'portfolio_value': portfolio_value,
                'signal': signal,
                'price': price,
            })
        
        self.portfolio = pd.DataFrame(portfolio_history)
        
        return self.portfolio

    def get_performance_metrics(self):
        """
        Tính các metrics hiệu suất.
        
        Returns:
            dict: Performance metrics
        """
        if self.portfolio is None or len(self.portfolio) == 0:
            return {}
        
        portfolio = self.portfolio
        initial = self.initial_capital
        final = portfolio['portfolio_value'].iloc[-1]
        
        # Total Return
        total_return = (final - initial) / initial
        
        # Daily Returns
        daily_returns = portfolio['portfolio_value'].pct_change().dropna()
        
        # Sharpe Ratio (annualized, assuming 365 trading days for crypto)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
        else:
            sharpe_ratio = 0
        
        # Max Drawdown
        peak = portfolio['portfolio_value'].expanding().max()
        drawdown = (portfolio['portfolio_value'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win Rate
        if len(self.trades) > 0:
            profitable_trades = [t for t in self.trades 
                               if t['type'] in ['SELL', 'STRONG_SELL', 'TAKE_PROFIT', 'STOP_LOSS']
                               and t['pnl_pct'] > 0]
            total_closed = [t for t in self.trades 
                          if t['type'] in ['SELL', 'STRONG_SELL', 'TAKE_PROFIT', 'STOP_LOSS']]
            win_rate = len(profitable_trades) / len(total_closed) if total_closed else 0
        else:
            win_rate = 0
        
        # Buy & Hold comparison
        if len(portfolio) > 0:
            first_price = portfolio['price'].iloc[0]
            last_price = portfolio['price'].iloc[-1]
            buy_hold_return = (last_price - first_price) / first_price if first_price > 0 else 0
        else:
            buy_hold_return = 0
        
        metrics = {
            'initial_capital': initial,
            'final_value': round(final, 2),
            'total_return': round(total_return * 100, 2),
            'buy_hold_return': round(buy_hold_return * 100, 2),
            'excess_return': round((total_return - buy_hold_return) * 100, 2),
            'sharpe_ratio': round(sharpe_ratio, 4),
            'max_drawdown': round(max_drawdown * 100, 2),
            'total_trades': len(self.trades),
            'win_rate': round(win_rate * 100, 2),
            'avg_trade_pnl': round(np.mean([t['pnl_pct'] for t in self.trades 
                                            if t['type'] not in ['BUY', 'STRONG_BUY']]) * 100, 2) 
                            if self.trades else 0,
        }
        
        return metrics

    def print_performance_report(self):
        """In báo cáo hiệu suất."""
        metrics = self.get_performance_metrics()
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 BÁO CÁO HIỆU SUẤT GIAO DỊCH")
        logger.info("=" * 60)
        logger.info(f"  💰 Vốn ban đầu:     ${metrics['initial_capital']:>12,.2f}")
        logger.info(f"  💰 Giá trị cuối:     ${metrics['final_value']:>12,.2f}")
        logger.info(f"  📈 Tổng lợi nhuận:   {metrics['total_return']:>11.2f}%")
        logger.info(f"  📊 Buy & Hold:       {metrics['buy_hold_return']:>11.2f}%")
        logger.info(f"  ✨ Excess Return:    {metrics['excess_return']:>11.2f}%")
        logger.info(f"  📐 Sharpe Ratio:     {metrics['sharpe_ratio']:>11.4f}")
        logger.info(f"  📉 Max Drawdown:     {metrics['max_drawdown']:>11.2f}%")
        logger.info(f"  🔄 Tổng giao dịch:  {metrics['total_trades']:>11d}")
        logger.info(f"  ✅ Win Rate:         {metrics['win_rate']:>11.2f}%")
        logger.info(f"  📊 Avg Trade P/L:    {metrics['avg_trade_pnl']:>11.2f}%")
        logger.info("=" * 60)
        
        return metrics


if __name__ == "__main__":
    # Test trading strategy
    np.random.seed(42)
    n = 100
    
    test_predictions = np.random.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
    test_sentiment = np.random.uniform(-1, 1, size=n)
    test_prices = np.cumsum(np.random.randn(n) * 100) + 50000
    test_prices = np.maximum(test_prices, 1000)
    
    generator = TradingSignalGenerator()
    signals = generator.generate_signals(test_predictions, test_sentiment, prices=test_prices)
    
    simulator = PortfolioSimulator()
    portfolio = simulator.simulate(signals)
    simulator.print_performance_report()
