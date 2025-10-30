"""
Configuration file for ML Trading System
Contains all hyperparameters and settings for the 3-stage pipeline
"""

import numpy as np

# Random Seeds for Reproducibility
RANDOM_SEEDS = {
    'numpy': 42,
    'tensorflow': 42,
    'sklearn': 42,
    'stable_baselines': 42
}

# Data Configuration
DATA_CONFIG = {
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'min_data_points': 1000,  # Minimum data points required
}

# LSTM Configuration (Stage 1)
LSTM_CONFIG = {
    'sequence_length': 60,
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'lstm_units': [128, 64, 32],
    'dropout_rate': 0.2,
    'patience': 15,  # Early stopping patience
    'min_delta': 0.0001,  # Minimum change for early stopping
    'ensemble_size': 5,  # Number of models for ensemble prediction
    'scaler_type': 'MinMaxScaler',  # 'MinMaxScaler' or 'StandardScaler'
    'validation_split': 0.2,
    'shuffle': False,  # No shuffling for time series
}

# Feature Configuration for LSTM
LSTM_FEATURES = [
    'Close', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
    'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'Volume_MA', 'ROC', 'Historical_Volatility'
]

# Random Forest Configuration (Stage 2)
RF_CONFIG = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    'trend_threshold': 0.5,  # % threshold for UP/DOWN classification
    'cv_folds': 5,  # Cross-validation folds
}

# Feature Configuration for Random Forest
RF_FEATURES = [
    'predicted_price', 'price_confidence', 'Close', 'Volume',
    'SMA_5', 'SMA_10', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal',
    'BB_Position', 'Volume_Ratio', 'ROC', 'Historical_Volatility',
    'price_momentum_1d', 'price_momentum_5d', 'volume_momentum'
]

# RL Agent Configuration (Stage 3)
RL_CONFIG = {
    'initial_cash': 100000,
    'transaction_cost': 0.001,  # 0.1% commission
    'max_position_size': 0.3,   # Max 30% of portfolio in single position
    'min_trade_amount': 1000,   # Minimum trade amount
    'learning_rate': 0.0003,
    'total_timesteps': 100000,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'policy_kwargs': {
        'net_arch': [64, 64],
        'activation_fn': 'tanh'
    }
}

# Action Space Configuration
ACTION_CONFIG = {
    'actions': {
        0: {'name': 'HOLD', 'position_change': 0.0},
        1: {'name': 'BUY_10', 'position_change': 0.1},
        2: {'name': 'BUY_25', 'position_change': 0.25},
        3: {'name': 'BUY_50', 'position_change': 0.5},
        4: {'name': 'BUY_100', 'position_change': 1.0},
        5: {'name': 'SELL_25', 'position_change': -0.25},
        6: {'name': 'SELL_50', 'position_change': -0.5},
        7: {'name': 'SELL_75', 'position_change': -0.75},
        8: {'name': 'SELL_100', 'position_change': -1.0},
    },
    'n_actions': 9
}

# State Space Configuration
STATE_CONFIG = {
    'state_features': [
        'predicted_price', 'predicted_trend_down', 'predicted_trend_sideways', 'predicted_trend_up',
        'trend_confidence', 'current_position_ratio', 'cash_ratio', 'current_price',
        'unrealized_pnl_ratio', 'rsi', 'macd', 'bb_position', 'current_drawdown',
        'portfolio_volatility', 'sharpe_ratio', 'price_momentum', 'volume_momentum',
        'days_in_position', 'profit_taking_signal', 'stop_loss_signal'
    ],
    'state_size': 20
}

# Risk Management Configuration
RISK_CONFIG = {
    'max_drawdown_threshold': 0.15,  # 15% max drawdown
    'stop_loss_threshold': 0.05,     # 5% stop loss
    'profit_taking_threshold': 0.1,   # 10% profit taking
    'volatility_lookback': 20,        # Days for volatility calculation
    'sharpe_lookback': 60,           # Days for Sharpe ratio calculation
    'max_consecutive_losses': 5,      # Max consecutive losing trades
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    'benchmark_strategies': ['buy_and_hold', 'sma_crossover'],
    'performance_metrics': [
        'total_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown',
        'win_rate', 'avg_win_loss_ratio', 'calmar_ratio', 'sortino_ratio',
        'num_trades', 'avg_trade_duration'
    ],
    'plot_results': True,
    'save_trades': True
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': 'trading_system.log',
    'max_bytes': 10485760,  # 10MB
    'backup_count': 5
}

# Model Saving Configuration
MODEL_CONFIG = {
    'save_models': True,
    'model_dir': 'models',
    'lstm_model_name': 'lstm_price_predictor.h5',
    'rf_model_name': 'rf_trend_classifier.pkl',
    'rl_model_name': 'rl_trading_agent',
    'scaler_name': 'feature_scaler.pkl'
}

# Visualization Configuration
VIZ_CONFIG = {
    'figure_size': (15, 10),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'save_plots': True,
    'plot_dir': 'plots'
}

# Supported Stock Symbols (for testing)
SUPPORTED_STOCKS = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOGL', 'META', 'SPY', 'QQQ']

# Data Validation Configuration
VALIDATION_CONFIG = {
    'required_columns': [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
        'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal',
        'BB_Upper', 'BB_Middle', 'BB_Lower'
    ],
    'min_data_quality': 0.95,  # 95% data completeness required
    'outlier_threshold': 3.0,  # Standard deviations for outlier detection
}