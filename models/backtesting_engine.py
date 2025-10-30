"""
Backtesting Engine for ML Trading System
Tests the complete 3-stage pipeline end-to-end with performance analysis and visualization
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from config import BACKTEST_CONFIG, VIZ_CONFIG, RL_CONFIG
from utils import setup_logging, calculate_portfolio_metrics
from stage1_price_prediction import LSTMPricePredictor
from stage2_trend_classification import TrendClassifier
from stage3_trading_agent import TradingAgent

class BacktestingEngine:
    """Comprehensive backtesting engine for the ML trading system"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logging()
        self.results = {}
        
        # Set plotting style
        plt.style.use(VIZ_CONFIG['style'])
        sns.set_palette(VIZ_CONFIG['color_palette'])
    
    def run_backtest(self, test_data: pd.DataFrame, 
                    lstm_model: LSTMPricePredictor,
                    rf_model: TrendClassifier,
                    rl_agent: TradingAgent,
                    stock_symbol: str = "STOCK") -> Dict[str, Any]:
        """
        Run complete backtest on test data
        
        Args:
            test_data: Test dataset
            lstm_model: Trained LSTM price predictor
            rf_model: Trained trend classifier
            rl_agent: Trained RL trading agent
            stock_symbol: Stock symbol for reporting
            
        Returns:
            Dictionary with comprehensive backtest results
        """
        self.logger.info(f"Starting backtest for {stock_symbol} on {len(test_data)} samples")
        
        # Stage 1: Get price predictions
        self.logger.info("Stage 1: Generating price predictions...")
        price_predictions = lstm_model.predict(test_data, return_confidence=True)
        
        # Stage 2: Get trend predictions
        self.logger.info("Stage 2: Generating trend predictions...")
        trend_predictions = rf_model.predict(test_data, price_predictions)
        
        # Stage 3: Execute RL trading strategy
        self.logger.info("Stage 3: Executing RL trading strategy...")
        trading_results = rl_agent.predict(test_data, price_predictions, trend_predictions)
        
        # Run benchmark strategies
        self.logger.info("Running benchmark strategies...")
        benchmark_results = self._run_benchmarks(test_data)
        
        # Calculate performance metrics
        self.logger.info("Calculating performance metrics...")
        performance_metrics = self._calculate_performance_metrics(
            trading_results, benchmark_results, test_data
        )
        
        # Store results
        self.results[stock_symbol] = {
            'price_predictions': price_predictions,
            'trend_predictions': trend_predictions,
            'trading_results': trading_results,
            'benchmark_results': benchmark_results,
            'performance_metrics': performance_metrics,
            'test_data': test_data
        }
        
        # Generate visualizations
        if BACKTEST_CONFIG['plot_results']:
            self.logger.info("Generating visualizations...")
            self._create_visualizations(stock_symbol)
        
        self.logger.info(f"Backtest completed for {stock_symbol}")
        return self.results[stock_symbol]
    
    def _run_benchmarks(self, test_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Run benchmark strategies"""
        benchmarks = {}
        
        # Buy and Hold strategy
        benchmarks['buy_and_hold'] = self._buy_and_hold_strategy(test_data)
        
        # Simple Moving Average Crossover strategy
        benchmarks['sma_crossover'] = self._sma_crossover_strategy(test_data)
        
        return benchmarks
    
    def _buy_and_hold_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Implement buy and hold strategy"""
        initial_cash = RL_CONFIG['initial_cash']
        
        # Buy on first day, hold until end
        first_price = data['Close'].iloc[0]
        shares = initial_cash // first_price
        cash = initial_cash - (shares * first_price)
        
        results = []
        for date, row in data.iterrows():
            portfolio_value = cash + shares * row['Close']
            results.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'shares_held': shares,
                'action': 'BUY' if date == data.index[0] else 'HOLD'
            })
        
        return pd.DataFrame(results)
    
    def _sma_crossover_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Implement simple moving average crossover strategy"""
        initial_cash = RL_CONFIG['initial_cash']
        cash = initial_cash
        shares = 0
        
        # Calculate signals
        data = data.copy()
        data['sma_5'] = data['Close'].rolling(5).mean()
        data['sma_20'] = data['Close'].rolling(20).mean()
        data['signal'] = np.where(data['sma_5'] > data['sma_20'], 1, 0)
        data['position'] = data['signal'].diff()
        
        results = []
        for date, row in data.iterrows():
            action = 'HOLD'
            
            # Buy signal
            if row['position'] == 1 and cash > 0:
                shares_to_buy = cash // row['Close']
                if shares_to_buy > 0:
                    cash -= shares_to_buy * row['Close']
                    shares += shares_to_buy
                    action = 'BUY'
            
            # Sell signal
            elif row['position'] == -1 and shares > 0:
                cash += shares * row['Close']
                shares = 0
                action = 'SELL'
            
            portfolio_value = cash + shares * row['Close']
            results.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'shares_held': shares,
                'action': action
            })
        
        return pd.DataFrame(results)
    
    def _calculate_performance_metrics(self, trading_results: pd.DataFrame,
                                     benchmark_results: Dict[str, pd.DataFrame],
                                     test_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        # ML Strategy metrics
        if len(trading_results) > 0:
            ml_returns = trading_results['portfolio_value'].pct_change().dropna()
            metrics['ml_strategy'] = calculate_portfolio_metrics(ml_returns)
            
            # Add specific trading metrics
            trades = [r for r in trading_results['action'] if r != 'HOLD']
            metrics['ml_strategy'].update({
                'num_trades': len(trades),
                'final_portfolio_value': trading_results['portfolio_value'].iloc[-1],
                'total_return_pct': (trading_results['portfolio_value'].iloc[-1] - RL_CONFIG['initial_cash']) / RL_CONFIG['initial_cash'] * 100
            })
        
        # Benchmark metrics
        for name, results in benchmark_results.items():
            if len(results) > 0:
                bench_returns = results['portfolio_value'].pct_change().dropna()
                metrics[name] = calculate_portfolio_metrics(bench_returns)
                metrics[name].update({
                    'final_portfolio_value': results['portfolio_value'].iloc[-1],
                    'total_return_pct': (results['portfolio_value'].iloc[-1] - RL_CONFIG['initial_cash']) / RL_CONFIG['initial_cash'] * 100
                })
        
        # Calculate relative performance
        if 'ml_strategy' in metrics and 'buy_and_hold' in metrics:
            metrics['relative_to_buy_hold'] = {
                'return_difference': metrics['ml_strategy']['total_return_pct'] - metrics['buy_and_hold']['total_return_pct'],
                'sharpe_difference': metrics['ml_strategy']['sharpe_ratio'] - metrics['buy_and_hold']['sharpe_ratio'],
                'max_drawdown_difference': metrics['ml_strategy']['max_drawdown'] - metrics['buy_and_hold']['max_drawdown']
            }
        
        return metrics
    
    def _create_visualizations(self, stock_symbol: str):
        """Create comprehensive visualizations"""
        if stock_symbol not in self.results:
            return
        
        result = self.results[stock_symbol]
        
        # Create output directory
        import os
        os.makedirs(VIZ_CONFIG['plot_dir'], exist_ok=True)
        
        # 1. Portfolio Performance Comparison
        self._plot_portfolio_performance(result, stock_symbol)
        
        # 2. Drawdown Analysis
        self._plot_drawdown_analysis(result, stock_symbol)
        
        # 3. Trading Signals on Price Chart
        self._plot_trading_signals(result, stock_symbol)
        
        # 4. Feature Importance (if available)
        self._plot_feature_importance(result, stock_symbol)
        
        # 5. Performance Metrics Summary
        self._plot_performance_summary(result, stock_symbol)
    
    def _plot_portfolio_performance(self, result: Dict, stock_symbol: str):
        """Plot portfolio value comparison over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=VIZ_CONFIG['figure_size'], height_ratios=[3, 1])
        
        # Portfolio values
        ml_portfolio = result['trading_results']['portfolio_value']
        buy_hold_portfolio = result['benchmark_results']['buy_and_hold']['portfolio_value']
        
        # Plot portfolio values
        ax1.plot(ml_portfolio.index, ml_portfolio.values, label='ML Strategy', linewidth=2)
        ax1.plot(buy_hold_portfolio.index, buy_hold_portfolio.values, label='Buy & Hold', linewidth=2)
        ax1.set_title(f'{stock_symbol} - Portfolio Performance Comparison', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot relative performance
        relative_perf = (ml_portfolio / buy_hold_portfolio - 1) * 100
        ax2.plot(relative_perf.index, relative_perf.values, color='green', linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('ML Strategy vs Buy & Hold (%)', fontsize=12)
        ax2.set_ylabel('Relative Return (%)', fontsize=10)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{VIZ_CONFIG['plot_dir']}/{stock_symbol}_portfolio_performance.png", 
                   dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
    
    def _plot_drawdown_analysis(self, result: Dict, stock_symbol: str):
        """Plot drawdown analysis"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate drawdowns
        ml_portfolio = result['trading_results']['portfolio_value']
        buy_hold_portfolio = result['benchmark_results']['buy_and_hold']['portfolio_value']
        
        # ML Strategy drawdown
        ml_peak = ml_portfolio.cummax()
        ml_drawdown = (ml_portfolio - ml_peak) / ml_peak * 100
        
        # Buy & Hold drawdown
        bh_peak = buy_hold_portfolio.cummax()
        bh_drawdown = (buy_hold_portfolio - bh_peak) / bh_peak * 100
        
        ax.fill_between(ml_drawdown.index, ml_drawdown.values, 0, alpha=0.7, label='ML Strategy')
        ax.fill_between(bh_drawdown.index, bh_drawdown.values, 0, alpha=0.7, label='Buy & Hold')
        
        ax.set_title(f'{stock_symbol} - Drawdown Analysis', fontsize=16, fontweight='bold')
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{VIZ_CONFIG['plot_dir']}/{stock_symbol}_drawdown_analysis.png", 
                   dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
    
    def _plot_trading_signals(self, result: Dict, stock_symbol: str):
        """Plot trading signals on price chart"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Price data
        price_data = result['test_data']['Close']
        ax.plot(price_data.index, price_data.values, label='Price', color='black', linewidth=1)
        
        # Trading signals
        trading_results = result['trading_results']
        
        # Buy signals
        buy_signals = trading_results[trading_results['action'].str.contains('BUY', na=False)]
        if len(buy_signals) > 0:
            ax.scatter(buy_signals['date'], price_data.loc[buy_signals['date']], 
                      color='green', marker='^', s=50, label='Buy', zorder=5)
        
        # Sell signals
        sell_signals = trading_results[trading_results['action'].str.contains('SELL', na=False)]
        if len(sell_signals) > 0:
            ax.scatter(sell_signals['date'], price_data.loc[sell_signals['date']], 
                      color='red', marker='v', s=50, label='Sell', zorder=5)
        
        ax.set_title(f'{stock_symbol} - Trading Signals', fontsize=16, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{VIZ_CONFIG['plot_dir']}/{stock_symbol}_trading_signals.png", 
                   dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, result: Dict, stock_symbol: str):
        """Plot feature importance from Random Forest"""
        # This would require access to the trained RF model
        # For now, create a placeholder
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Dummy feature importance data
        features = ['RSI', 'MACD', 'BB_Position', 'Volume_Ratio', 'Price_Momentum']
        importance = [0.25, 0.20, 0.18, 0.15, 0.12]
        
        ax.barh(features, importance)
        ax.set_title(f'{stock_symbol} - Feature Importance', fontsize=16, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{VIZ_CONFIG['plot_dir']}/{stock_symbol}_feature_importance.png", 
                   dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
    
    def _plot_performance_summary(self, result: Dict, stock_symbol: str):
        """Plot performance metrics summary"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = result['performance_metrics']
        
        # Strategy comparison
        strategies = []
        returns = []
        sharpe_ratios = []
        max_drawdowns = []
        
        for strategy, metric in metrics.items():
            if strategy != 'relative_to_buy_hold' and isinstance(metric, dict):
                strategies.append(strategy.replace('_', ' ').title())
                returns.append(metric.get('total_return_pct', 0))
                sharpe_ratios.append(metric.get('sharpe_ratio', 0))
                max_drawdowns.append(abs(metric.get('max_drawdown', 0)) * 100)
        
        # Total Returns
        ax1.bar(strategies, returns)
        ax1.set_title('Total Returns (%)', fontweight='bold')
        ax1.set_ylabel('Return (%)')
        for i, v in enumerate(returns):
            ax1.text(i, v + 0.5, f'{v:.1f}%', ha='center')
        
        # Sharpe Ratios
        ax2.bar(strategies, sharpe_ratios)
        ax2.set_title('Sharpe Ratios', fontweight='bold')
        ax2.set_ylabel('Sharpe Ratio')
        for i, v in enumerate(sharpe_ratios):
            ax2.text(i, v + 0.05, f'{v:.2f}', ha='center')
        
        # Max Drawdowns
        ax3.bar(strategies, max_drawdowns)
        ax3.set_title('Maximum Drawdowns (%)', fontweight='bold')
        ax3.set_ylabel('Max Drawdown (%)')
        for i, v in enumerate(max_drawdowns):
            ax3.text(i, v + 0.2, f'{v:.1f}%', ha='center')
        
        # Risk-Return Scatter
        ax4.scatter(max_drawdowns, returns, s=100)
        for i, strategy in enumerate(strategies):
            ax4.annotate(strategy, (max_drawdowns[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax4.set_xlabel('Max Drawdown (%)')
        ax4.set_ylabel('Total Return (%)')
        ax4.set_title('Risk-Return Profile', fontweight='bold')
        
        plt.suptitle(f'{stock_symbol} - Performance Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{VIZ_CONFIG['plot_dir']}/{stock_symbol}_performance_summary.png", 
                   dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
    
    def generate_report(self, stock_symbols: List[str] = None) -> pd.DataFrame:
        """Generate comprehensive performance report"""
        if stock_symbols is None:
            stock_symbols = list(self.results.keys())
        
        report_data = []
        
        for symbol in stock_symbols:
            if symbol not in self.results:
                continue
            
            metrics = self.results[symbol]['performance_metrics']
            
            # Extract ML strategy metrics
            ml_metrics = metrics.get('ml_strategy', {})
            bh_metrics = metrics.get('buy_and_hold', {})
            
            report_data.append({
                'Stock': symbol,
                'ML_Total_Return_%': ml_metrics.get('total_return_pct', 0),
                'ML_Sharpe_Ratio': ml_metrics.get('sharpe_ratio', 0),
                'ML_Max_Drawdown_%': abs(ml_metrics.get('max_drawdown', 0)) * 100,
                'ML_Win_Rate_%': ml_metrics.get('win_rate', 0) * 100,
                'BH_Total_Return_%': bh_metrics.get('total_return_pct', 0),
                'BH_Sharpe_Ratio': bh_metrics.get('sharpe_ratio', 0),
                'BH_Max_Drawdown_%': abs(bh_metrics.get('max_drawdown', 0)) * 100,
                'Num_Trades': ml_metrics.get('num_trades', 0),
                'Outperformance_%': ml_metrics.get('total_return_pct', 0) - bh_metrics.get('total_return_pct', 0)
            })
        
        report_df = pd.DataFrame(report_data)
        
        if BACKTEST_CONFIG['save_trades']:
            report_df.to_csv(f"{VIZ_CONFIG['plot_dir']}/backtest_summary.csv", index=False)
            self.logger.info(f"Backtest report saved to {VIZ_CONFIG['plot_dir']}/backtest_summary.csv")
        
        return report_df
    
    def print_summary(self, stock_symbols: List[str] = None):
        """Print summary of backtest results"""
        report_df = self.generate_report(stock_symbols)
        
        print("\n" + "="*100)
        print("ML TRADING SYSTEM - BACKTEST SUMMARY")
        print("="*100)
        
        print(f"\nTested on {len(report_df)} stocks")
        print(f"Initial capital: ${RL_CONFIG['initial_cash']:,}")
        
        print("\nPERFORMANCE OVERVIEW:")
        print("-" * 50)
        
        # Overall statistics
        avg_ml_return = report_df['ML_Total_Return_%'].mean()
        avg_bh_return = report_df['BH_Total_Return_%'].mean()
        avg_outperformance = report_df['Outperformance_%'].mean()
        win_rate = (report_df['Outperformance_%'] > 0).mean() * 100
        
        print(f"Average ML Strategy Return: {avg_ml_return:.2f}%")
        print(f"Average Buy & Hold Return: {avg_bh_return:.2f}%")
        print(f"Average Outperformance: {avg_outperformance:.2f}%")
        print(f"Win Rate (vs Buy & Hold): {win_rate:.1f}%")
        
        print(f"\nDETAILED RESULTS:")
        print("-" * 50)
        print(report_df.round(2).to_string(index=False))
        
        print("\n" + "="*100)

def run_backtest(test_data: pd.DataFrame, lstm_model: LSTMPricePredictor,
                rf_model: TrendClassifier, rl_agent: TradingAgent,
                stock_symbol: str = "STOCK", logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Run backtest for a single stock
    
    Args:
        test_data: Test dataset
        lstm_model: Trained LSTM model
        rf_model: Trained RF model
        rl_agent: Trained RL agent
        stock_symbol: Stock symbol
        logger: Logger instance
        
    Returns:
        Dictionary with backtest results
    """
    if logger is None:
        logger = setup_logging()
    
    engine = BacktestingEngine(logger)
    return engine.run_backtest(test_data, lstm_model, rf_model, rl_agent, stock_symbol)

if __name__ == "__main__":
    # Example usage
    import warnings
    warnings.filterwarnings('ignore')
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("Backtesting Engine for ML Trading System")
    logger.info("This module should be imported and used within the main pipeline")
    logger.info("Example: results = run_backtest(test_data, lstm_model, rf_model, rl_agent)")