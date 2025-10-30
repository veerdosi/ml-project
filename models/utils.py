"""
Utility functions for ML Trading System
Contains helper functions for data processing, validation, and common operations
"""

import pandas as pd
import numpy as np
import logging
import os
import pickle
from typing import Tuple, List, Dict, Optional, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_CONFIG, LSTM_CONFIG, LSTM_FEATURES, RF_FEATURES,
    VALIDATION_CONFIG, LOGGING_CONFIG, MODEL_CONFIG, RANDOM_SEEDS
)

def setup_logging() -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.FileHandler(f"logs/{LOGGING_CONFIG['filename']}"),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger

def set_random_seeds():
    """Set random seeds for reproducibility"""
    np.random.seed(RANDOM_SEEDS['numpy'])
    try:
        import tensorflow as tf
        tf.random.set_seed(RANDOM_SEEDS['tensorflow'])
    except ImportError:
        pass

def validate_data(data: pd.DataFrame, logger: Optional[logging.Logger] = None) -> bool:
    """
    Validate input data quality and structure
    
    Args:
        data: Input DataFrame
        logger: Logger instance
        
    Returns:
        bool: True if data is valid
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Check required columns
    missing_cols = set(VALIDATION_CONFIG['required_columns']) - set(data.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    # Check data completeness
    completeness = 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
    if completeness < VALIDATION_CONFIG['min_data_quality']:
        logger.error(f"Data completeness {completeness:.2%} below threshold {VALIDATION_CONFIG['min_data_quality']:.2%}")
        return False
    
    # Check minimum data points
    if len(data) < DATA_CONFIG['min_data_points']:
        logger.error(f"Insufficient data points: {len(data)} < {DATA_CONFIG['min_data_points']}")
        return False
    
    # Check for infinite values
    if np.isinf(data.select_dtypes(include=[np.number])).any().any():
        logger.error("Data contains infinite values")
        return False
    
    logger.info(f"Data validation passed. Shape: {data.shape}, Completeness: {completeness:.2%}")
    return True

def split_data(data: pd.DataFrame, logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets chronologically
    
    Args:
        data: Input DataFrame with datetime index
        logger: Logger instance
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    n = len(data)
    train_end = int(n * DATA_CONFIG['train_ratio'])
    val_end = train_end + int(n * DATA_CONFIG['val_ratio'])
    
    train_data = data.iloc[:train_end].copy()
    val_data = data.iloc[train_end:val_end].copy()
    test_data = data.iloc[val_end:].copy()
    
    logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data

def create_sequences(data: np.ndarray, sequence_length: int, target_col_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction
    
    Args:
        data: Input data array
        sequence_length: Length of input sequences
        target_col_idx: Index of target column
        
    Returns:
        Tuple of (X, y) arrays
    """
    X, y = [], []
    
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i, target_col_idx])
    
    return np.array(X), np.array(y)

def prepare_lstm_data(data: pd.DataFrame, features: List[str], target: str = 'Close',
                     sequence_length: int = None, scaler_type: str = None) -> Dict[str, Any]:
    """
    Prepare data for LSTM training
    
    Args:
        data: Input DataFrame
        features: List of feature column names
        target: Target column name
        sequence_length: Length of input sequences
        scaler_type: Type of scaler to use
        
    Returns:
        Dictionary containing prepared data and scaler
    """
    if sequence_length is None:
        sequence_length = LSTM_CONFIG['sequence_length']
    if scaler_type is None:
        scaler_type = LSTM_CONFIG['scaler_type']
    
    # Select features
    feature_data = data[features].copy()
    
    # Initialize scaler
    if scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Scale features
    scaled_data = scaler.fit_transform(feature_data)
    
    # Create sequences
    target_idx = features.index(target)
    X, y = create_sequences(scaled_data, sequence_length, target_idx)
    
    return {
        'X': X,
        'y': y,
        'scaler': scaler,
        'feature_names': features,
        'target_name': target
    }

def create_trend_labels(returns: pd.Series, threshold: float = None) -> pd.Series:
    """
    Create trend labels for classification
    
    Args:
        returns: Series of price returns
        threshold: Threshold for trend classification
        
    Returns:
        Series of trend labels (0: DOWN, 1: SIDEWAYS, 2: UP)
    """
    if threshold is None:
        threshold = RF_CONFIG['trend_threshold']
    
    labels = pd.Series(index=returns.index, dtype=int)
    labels[returns > threshold] = 2  # UP
    labels[returns < -threshold] = 0  # DOWN
    labels[(returns >= -threshold) & (returns <= threshold)] = 1  # SIDEWAYS
    
    return labels

def calculate_technical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional technical features for the models
    
    Args:
        data: Input DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional features
    """
    df = data.copy()
    
    # Price momentum features
    df['price_momentum_1d'] = df['Close'].pct_change(1) * 100
    df['price_momentum_5d'] = df['Close'].pct_change(5) * 100
    
    # Volume features
    df['volume_momentum'] = df['Volume'].pct_change(5) * 100
    df['volume_ratio'] = df['Volume'] / df['Volume_MA']
    
    # Bollinger Band position
    df['bb_position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Price relative to moving averages
    df['price_vs_sma_5'] = (df['Close'] - df['SMA_5']) / df['SMA_5'] * 100
    df['price_vs_sma_20'] = (df['Close'] - df['SMA_20']) / df['SMA_20'] * 100
    
    # MACD features
    df['macd_histogram_momentum'] = df['MACD_Histogram'].diff()
    df['macd_signal_cross'] = np.where(df['MACD'] > df['MACD_Signal'], 1, 0)
    
    return df

def calculate_portfolio_metrics(returns: pd.Series, trading_days: int = 252) -> Dict[str, float]:
    """
    Calculate portfolio performance metrics
    
    Args:
        returns: Series of portfolio returns
        trading_days: Number of trading days per year
        
    Returns:
        Dictionary of performance metrics
    """
    if len(returns) == 0:
        return {}
    
    # Total return
    total_return = (1 + returns).prod() - 1
    
    # Annualized return
    periods = len(returns) / trading_days
    annualized_return = (1 + total_return) ** (1 / periods) - 1 if periods > 0 else 0
    
    # Volatility
    volatility = returns.std() * np.sqrt(trading_days)
    
    # Sharpe ratio
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Sortino ratio
    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(trading_days) if len(negative_returns) > 0 else 0
    sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
    
    # Win rate
    win_rate = (returns > 0).mean() if len(returns) > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'sortino_ratio': sortino_ratio,
        'win_rate': win_rate
    }

def save_model(model: Any, filepath: str, logger: Optional[logging.Logger] = None):
    """
    Save model to disk
    
    Args:
        model: Model object to save
        filepath: Path to save the model
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    try:
        if hasattr(model, 'save'):  # Keras model
            model.save(filepath)
        else:  # Scikit-learn model or other
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        logger.info(f"Model saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model to {filepath}: {e}")

def load_model(filepath: str, logger: Optional[logging.Logger] = None) -> Any:
    """
    Load model from disk
    
    Args:
        filepath: Path to load the model from
        logger: Logger instance
        
    Returns:
        Loaded model object
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        if filepath.endswith('.h5'):  # Keras model
            from tensorflow.keras.models import load_model
            model = load_model(filepath)
        else:  # Pickle file
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {filepath}: {e}")
        return None

def get_time_series_split(n_samples: int, n_splits: int = 5) -> TimeSeriesSplit:
    """
    Get TimeSeriesSplit for cross-validation
    
    Args:
        n_samples: Number of samples
        n_splits: Number of splits
        
    Returns:
        TimeSeriesSplit object
    """
    return TimeSeriesSplit(n_splits=n_splits, test_size=n_samples // (n_splits + 1))

def detect_outliers(data: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    """
    Detect and handle outliers using Z-score method
    
    Args:
        data: Input DataFrame
        threshold: Z-score threshold for outlier detection
        
    Returns:
        DataFrame with outliers handled
    """
    if threshold is None:
        threshold = VALIDATION_CONFIG['outlier_threshold']
    
    df = data.copy()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers = z_scores > threshold
        if outliers.any():
            # Replace outliers with median
            df.loc[outliers, col] = df[col].median()
    
    return df

def create_model_directory():
    """Create directory for saving models"""
    os.makedirs(MODEL_CONFIG['model_dir'], exist_ok=True)

def get_model_path(model_name: str) -> str:
    """Get full path for model file"""
    return os.path.join(MODEL_CONFIG['model_dir'], model_name)

class DataProcessor:
    """Class for processing and preparing data for ML models"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.scalers = {}
        
    def fit_scalers(self, train_data: pd.DataFrame, features: List[str]):
        """Fit scalers on training data"""
        for feature_set_name, feature_list in [('lstm', LSTM_FEATURES), ('rf', RF_FEATURES)]:
            if feature_set_name == 'rf' and 'predicted_price' in feature_list:
                continue  # Skip RF features that require Stage 1 output
                
            available_features = [f for f in feature_list if f in train_data.columns]
            if available_features:
                scaler = MinMaxScaler()
                self.scalers[feature_set_name] = scaler.fit(train_data[available_features])
    
    def transform_features(self, data: pd.DataFrame, feature_set: str, features: List[str]) -> np.ndarray:
        """Transform features using fitted scaler"""
        if feature_set not in self.scalers:
            raise ValueError(f"Scaler for {feature_set} not fitted")
        
        available_features = [f for f in features if f in data.columns]
        return self.scalers[feature_set].transform(data[available_features])
    
    def inverse_transform_target(self, scaled_target: np.ndarray, feature_set: str = 'lstm') -> np.ndarray:
        """Inverse transform target variable"""
        if feature_set not in self.scalers:
            raise ValueError(f"Scaler for {feature_set} not fitted")
        
        # Assuming Close price is the first feature
        dummy_data = np.zeros((len(scaled_target), self.scalers[feature_set].n_features_in_))
        dummy_data[:, 0] = scaled_target.flatten()
        
        return self.scalers[feature_set].inverse_transform(dummy_data)[:, 0]