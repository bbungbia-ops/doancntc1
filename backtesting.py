"""
Module 8: Backtesting & Evaluation
====================================
Đánh giá hiệu suất chiến lược giao dịch và các models.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Đánh giá hiệu suất dự đoán của các models.
    
    Metrics:
    - Accuracy
    - Precision (per class)
    - Recall (per class)
    - F1-Score (macro, weighted)
    - Confusion Matrix
    """

    def __init__(self):
        self.results = {}

    def evaluate(self, model_name, y_true, y_pred, y_proba=None):
        """
        Đánh giá một model.
        
        Args:
            model_name: Tên model
            y_true: Nhãn thực
            y_pred: Nhãn dự đoán
            y_proba: Xác suất dự đoán (optional)
            
        Returns:
            dict: Evaluation metrics
        """
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(
                y_true, y_pred,
                target_names=['DOWN', 'NEUTRAL', 'UP'],
                output_dict=True,
                zero_division=0
            ),
        }
        
        self.results[model_name] = metrics
        
        # Print report
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 ĐÁNH GIÁ MODEL: {model_name}")
        logger.info(f"{'='*60}")
        logger.info(f"  Accuracy:          {metrics['accuracy']:.4f}")
        logger.info(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        logger.info(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
        logger.info(f"  F1-Score (macro):  {metrics['f1_macro']:.4f}")
        logger.info(f"  F1-Score (weighted): {metrics['f1_weighted']:.4f}")
        logger.info(f"\n  Confusion Matrix:")
        logger.info(f"  {metrics['confusion_matrix']}")
        logger.info(f"\n  Classification Report:")
        logger.info(classification_report(y_true, y_pred, 
                                          target_names=['DOWN', 'NEUTRAL', 'UP'],
                                          zero_division=0))
        
        return metrics

    def compare_models(self):
        """
        So sánh tất cả models đã đánh giá.
        
        Returns:
            pd.DataFrame: Bảng so sánh
        """
        if not self.results:
            logger.warning("Chưa có kết quả đánh giá nào!")
            return pd.DataFrame()
        
        rows = []
        for name, metrics in self.results.items():
            rows.append({
                'Model': name,
                'Accuracy': round(metrics['accuracy'], 4),
                'Precision': round(metrics['precision_macro'], 4),
                'Recall': round(metrics['recall_macro'], 4),
                'F1 (macro)': round(metrics['f1_macro'], 4),
                'F1 (weighted)': round(metrics['f1_weighted'], 4),
            })
        
        comparison_df = pd.DataFrame(rows)
        comparison_df = comparison_df.sort_values('F1 (weighted)', ascending=False)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 BẢNG SO SÁNH CÁC MODELS")
        logger.info(f"{'='*60}")
        logger.info(f"\n{comparison_df.to_string(index=False)}")
        
        return comparison_df


class Backtester:
    """
    Backtesting Engine - Kiểm tra chiến lược trên dữ liệu lịch sử.
    
    Tính toán:
    - Equity Curve
    - Drawdown Curve
    - Trade Log
    - Rolling Metrics
    """

    def __init__(self, portfolio_df, trades, initial_capital=None):
        """
        Args:
            portfolio_df: DataFrame từ PortfolioSimulator
            trades: List of trade dicts
            initial_capital: Vốn ban đầu
        """
        self.portfolio = portfolio_df
        self.trades = trades
        self.initial_capital = initial_capital or config.TRADING_CONFIG['initial_capital']

    def compute_equity_curve(self):
        """
        Tính equity curve và drawdown.
        
        Returns:
            pd.DataFrame: DataFrame với equity, drawdown, returns
        """
        df = self.portfolio.copy()
        
        # Equity
        df['equity'] = df['portfolio_value']
        df['equity_return'] = df['equity'].pct_change()
        
        # Cumulative return
        df['cumulative_return'] = (df['equity'] / self.initial_capital - 1) * 100
        
        # Drawdown
        df['peak'] = df['equity'].expanding().max()
        df['drawdown'] = (df['equity'] - df['peak']) / df['peak'] * 100
        
        # Buy & Hold comparison
        if 'price' in df.columns:
            first_price = df['price'].iloc[0]
            df['buy_hold_equity'] = self.initial_capital * (df['price'] / first_price)
            df['buy_hold_return'] = (df['buy_hold_equity'] / self.initial_capital - 1) * 100
        
        return df

    def compute_rolling_metrics(self, window=30):
        """
        Tính rolling metrics (theo window ngày).
        
        Args:
            window: Số ngày cho rolling window
            
        Returns:
            pd.DataFrame: Rolling metrics
        """
        df = self.portfolio.copy()
        daily_returns = df['portfolio_value'].pct_change()
        
        rolling_df = pd.DataFrame({
            'date': df['date'],
            'rolling_mean_return': daily_returns.rolling(window).mean() * 100,
            'rolling_volatility': daily_returns.rolling(window).std() * np.sqrt(365) * 100,
            'rolling_sharpe': (daily_returns.rolling(window).mean() / 
                             daily_returns.rolling(window).std()) * np.sqrt(365),
        })
        
        return rolling_df

    def get_trade_analysis(self):
        """
        Phân tích chi tiết các giao dịch.
        
        Returns:
            dict: Trade analysis metrics
        """
        if not self.trades:
            return {}
        
        # Phân loại trades
        buy_trades = [t for t in self.trades if t['type'] in ['BUY', 'STRONG_BUY']]
        sell_trades = [t for t in self.trades 
                      if t['type'] in ['SELL', 'STRONG_SELL', 'TAKE_PROFIT', 'STOP_LOSS']]
        
        if not sell_trades:
            return {'total_trades': len(self.trades), 'no_closed_trades': True}
        
        pnls = [t['pnl_pct'] for t in sell_trades]
        
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p <= 0]
        
        analysis = {
            'total_trades': len(self.trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(sell_trades) * 100 if sell_trades else 0,
            'avg_win': np.mean(winning) * 100 if winning else 0,
            'avg_loss': np.mean(losing) * 100 if losing else 0,
            'max_win': max(pnls) * 100 if pnls else 0,
            'max_loss': min(pnls) * 100 if pnls else 0,
            'profit_factor': abs(sum(winning) / sum(losing)) if losing and sum(losing) != 0 else float('inf'),
            'stop_loss_count': sum(1 for t in sell_trades if t['type'] == 'STOP_LOSS'),
            'take_profit_count': sum(1 for t in sell_trades if t['type'] == 'TAKE_PROFIT'),
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 PHÂN TÍCH GIAO DỊCH")
        logger.info(f"{'='*60}")
        logger.info(f"  Tổng giao dịch:        {analysis['total_trades']}")
        logger.info(f"  Lệnh mua:              {analysis['buy_trades']}")
        logger.info(f"  Lệnh bán:              {analysis['sell_trades']}")
        logger.info(f"  ✅ Giao dịch thắng:    {analysis['winning_trades']}")
        logger.info(f"  ❌ Giao dịch thua:     {analysis['losing_trades']}")
        logger.info(f"  📊 Win Rate:           {analysis['win_rate']:.1f}%")
        logger.info(f"  📈 Avg Win:            {analysis['avg_win']:+.2f}%")
        logger.info(f"  📉 Avg Loss:           {analysis['avg_loss']:+.2f}%")
        logger.info(f"  🏆 Max Win:            {analysis['max_win']:+.2f}%")
        logger.info(f"  💀 Max Loss:           {analysis['max_loss']:+.2f}%")
        logger.info(f"  📐 Profit Factor:      {analysis['profit_factor']:.2f}")
        logger.info(f"  🛑 Stop-Loss count:    {analysis['stop_loss_count']}")
        logger.info(f"  🎯 Take-Profit count:  {analysis['take_profit_count']}")
        
        return analysis

    def generate_full_report(self):
        """
        Tạo báo cáo backtesting hoàn chỉnh.
        
        Returns:
            dict: Full backtesting report
        """
        equity = self.compute_equity_curve()
        rolling = self.compute_rolling_metrics()
        trade_analysis = self.get_trade_analysis()
        
        if len(self.portfolio) > 0:
            final_value = self.portfolio['portfolio_value'].iloc[-1]
            total_return = (final_value / self.initial_capital - 1) * 100
            
            daily_returns = self.portfolio['portfolio_value'].pct_change().dropna()
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365) \
                if daily_returns.std() > 0 else 0
            
            max_dd = equity['drawdown'].min()
        else:
            final_value = self.initial_capital
            total_return = 0
            sharpe = 0
            max_dd = 0
        
        report = {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_value': round(final_value, 2),
                'total_return_pct': round(total_return, 2),
                'sharpe_ratio': round(sharpe, 4),
                'max_drawdown_pct': round(max_dd, 2),
            },
            'trade_analysis': trade_analysis,
            'equity_curve': equity,
            'rolling_metrics': rolling,
        }
        
        # Save report
        report_path = os.path.join(config.PROCESSED_DATA_DIR, "backtest_report.csv")
        equity.to_csv(report_path, index=False)
        logger.info(f"\nĐã lưu báo cáo backtest tại: {report_path}")
        
        return report


if __name__ == "__main__":
    print("Backtesting module loaded successfully!")
    print("Components: ModelEvaluator, Backtester")
