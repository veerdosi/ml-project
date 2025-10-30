"""
Stage 1: LSTM Price Prediction Model
Predicts next day's closing price using historical data and technical indicators
"""

import pandas as pd
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from config import LSTM_CONFIG, LSTM_FEATURES, MODEL_CONFIG, RANDOM_SEEDS
from utils import (
    setup_logging, set_random_seeds, validate_data, create_sequences,
    prepare_lstm_data, save_model, get_model_path, create_model_directory
)

class LSTMPricePredictor:
    """LSTM model for stock price prediction"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logging()
        self.models = []  # Ensemble of models
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
        # Set random seeds
        set_random_seeds()
        
        # Create model directory
        create_model_directory()
    
    def _build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(
                LSTM_CONFIG['lstm_units'][0], 
                return_sequences=True, 
                input_shape=input_shape,
                name='lstm_1'
            ),
            Dropout(LSTM_CONFIG['dropout_rate']),
            BatchNormalization(),
            
            LSTM(
                LSTM_CONFIG['lstm_units'][1], 
                return_sequences=True,
                name='lstm_2'
            ),
            Dropout(LSTM_CONFIG['dropout_rate']),
            BatchNormalization(),
            
            LSTM(
                LSTM_CONFIG['lstm_units'][2], 
                return_sequences=False,
                name='lstm_3'
            ),
            Dropout(LSTM_CONFIG['dropout_rate']),
            
            Dense(16, activation='relu', name='dense_1'),
            Dense(1, name='output')  # Single output: predicted price
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=LSTM_CONFIG['learning_rate']),
            loss='huber',  # Robust to outliers
            metrics=['mae', 'mse']
        )
        
        return model
    
    def _prepare_callbacks(self, model_name: str) -> List:
        """Prepare training callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=LSTM_CONFIG['patience'],
                min_delta=LSTM_CONFIG['min_delta'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=LSTM_CONFIG['patience'] // 2,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=get_model_path(f"{model_name}.h5"),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, float]:
        """
        Train ensemble of LSTM models
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info("Starting LSTM training...")
        
        # Validate data
        if not validate_data(train_data, self.logger):
            raise ValueError("Training data validation failed")
        if not validate_data(val_data, self.logger):
            raise ValueError("Validation data validation failed")
        
        # Prepare data
        train_prepared = prepare_lstm_data(
            train_data, 
            LSTM_FEATURES, 
            target='Close',
            sequence_length=LSTM_CONFIG['sequence_length'],
            scaler_type=LSTM_CONFIG['scaler_type']
        )
        
        # Store scaler and feature names
        self.scaler = train_prepared['scaler']
        self.feature_names = train_prepared['feature_names']
        
        # Prepare validation data using same scaler
        val_features = val_data[LSTM_FEATURES].copy()
        val_scaled = self.scaler.transform(val_features)
        target_idx = LSTM_FEATURES.index('Close')
        X_val, y_val = create_sequences(
            val_scaled, 
            LSTM_CONFIG['sequence_length'], 
            target_idx
        )
        
        X_train, y_train = train_prepared['X'], train_prepared['y']
        
        self.logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        self.logger.info(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
        
        # Train ensemble of models
        training_metrics = []
        
        for i in range(LSTM_CONFIG['ensemble_size']):
            self.logger.info(f"Training model {i+1}/{LSTM_CONFIG['ensemble_size']}")
            
            # Set different random seed for each model
            tf.random.set_seed(RANDOM_SEEDS['tensorflow'] + i)
            
            # Build model
            model = self._build_model((X_train.shape[1], X_train.shape[2]))
            
            # Prepare callbacks
            callbacks = self._prepare_callbacks(f"lstm_model_{i}")
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=LSTM_CONFIG['epochs'],
                batch_size=LSTM_CONFIG['batch_size'],
                callbacks=callbacks,
                verbose=0
            )
            
            # Store model
            self.models.append(model)
            
            # Record best metrics
            best_epoch = np.argmin(history.history['val_loss'])
            metrics = {
                'model_idx': i,
                'best_epoch': best_epoch,
                'train_loss': history.history['loss'][best_epoch],
                'val_loss': history.history['val_loss'][best_epoch],
                'train_mae': history.history['mae'][best_epoch],
                'val_mae': history.history['val_mae'][best_epoch]
            }
            training_metrics.append(metrics)
            
            self.logger.info(f"Model {i+1} - Best Epoch: {best_epoch}, Val Loss: {metrics['val_loss']:.6f}")
        
        self.is_trained = True
        
        # Save scaler
        import pickle
        with open(get_model_path(MODEL_CONFIG['scaler_name']), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Calculate ensemble metrics
        ensemble_metrics = self._calculate_ensemble_metrics(training_metrics)
        
        self.logger.info("LSTM training completed successfully")
        return ensemble_metrics
    
    def _calculate_ensemble_metrics(self, individual_metrics: List[Dict]) -> Dict[str, float]:
        """Calculate ensemble performance metrics"""
        metrics_df = pd.DataFrame(individual_metrics)
        
        return {
            'ensemble_size': len(individual_metrics),
            'avg_train_loss': metrics_df['train_loss'].mean(),
            'std_train_loss': metrics_df['train_loss'].std(),
            'avg_val_loss': metrics_df['val_loss'].mean(),
            'std_val_loss': metrics_df['val_loss'].std(),
            'avg_val_mae': metrics_df['val_mae'].mean(),
            'best_val_loss': metrics_df['val_loss'].min(),
            'best_model_idx': metrics_df.loc[metrics_df['val_loss'].idxmin(), 'model_idx']
        }
    
    def predict(self, data: pd.DataFrame, return_confidence: bool = True) -> pd.DataFrame:
        """
        Make predictions using ensemble of models
        
        Args:
            data: Input data for prediction
            return_confidence: Whether to return prediction confidence
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.logger.info(f"Making predictions on {len(data)} samples")
        
        # Prepare data
        features = data[LSTM_FEATURES].copy()
        scaled_data = self.scaler.transform(features)
        
        # Create sequences
        target_idx = LSTM_FEATURES.index('Close')
        X, y_actual = create_sequences(
            scaled_data, 
            LSTM_CONFIG['sequence_length'], 
            target_idx
        )
        
        if len(X) == 0:
            self.logger.warning("Not enough data for sequence creation")
            return pd.DataFrame()
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred.flatten())
        
        predictions = np.array(predictions)
        
        # Calculate ensemble statistics
        ensemble_pred = np.mean(predictions, axis=0)
        pred_std = np.std(predictions, axis=0)
        
        # Inverse transform predictions (approximate)
        # Create dummy array for inverse transform
        dummy_data = np.zeros((len(ensemble_pred), len(LSTM_FEATURES)))
        dummy_data[:, target_idx] = ensemble_pred
        pred_prices = self.scaler.inverse_transform(dummy_data)[:, target_idx]
        
        # Inverse transform actual values
        dummy_actual = np.zeros((len(y_actual), len(LSTM_FEATURES)))
        dummy_actual[:, target_idx] = y_actual
        actual_prices = self.scaler.inverse_transform(dummy_actual)[:, target_idx]
        
        # Create results DataFrame
        # Align with original data (skip first sequence_length rows)
        start_idx = LSTM_CONFIG['sequence_length']
        result_dates = data.index[start_idx:start_idx + len(pred_prices)]
        
        results = pd.DataFrame({
            'date': result_dates,
            'actual_price': actual_prices,
            'predicted_price': pred_prices,
        })
        
        if return_confidence:
            # Calculate confidence (inverse of standard deviation)
            results['prediction_std'] = pred_std
            results['prediction_confidence'] = 1 / (pred_std + 1e-8)  # Add small epsilon
        
        return results
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: Test dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(test_data, return_confidence=True)
        
        if len(predictions) == 0:
            return {}
        
        actual = predictions['actual_price'].values
        predicted = predictions['predicted_price'].values
        
        # Calculate metrics
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predicted)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Direction accuracy (whether we predicted the direction correctly)
        actual_direction = np.sign(np.diff(actual))
        predicted_direction = np.sign(np.diff(predicted))
        direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'prediction_samples': len(predictions)
        }
        
        self.logger.info(f"Evaluation metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}, MAPE={mape:.2f}%")
        
        return metrics
    
    def save_models(self):
        """Save all trained models"""
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        for i, model in enumerate(self.models):
            model_path = get_model_path(f"lstm_model_{i}.h5")
            model.save(model_path)
            self.logger.info(f"Saved model {i} to {model_path}")
    
    def load_models(self, model_dir: str = None):
        """Load pre-trained models"""
        if model_dir is None:
            model_dir = MODEL_CONFIG['model_dir']
        
        import os
        self.models = []
        
        # Load scaler
        scaler_path = get_model_path(MODEL_CONFIG['scaler_name'])
        if os.path.exists(scaler_path):
            import pickle
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.feature_names = LSTM_FEATURES
        
        # Load models
        for i in range(LSTM_CONFIG['ensemble_size']):
            model_path = get_model_path(f"lstm_model_{i}.h5")
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                self.models.append(model)
                self.logger.info(f"Loaded model {i} from {model_path}")
        
        if self.models:
            self.is_trained = True
            self.logger.info(f"Loaded {len(self.models)} models successfully")
        else:
            self.logger.warning("No models found to load")

def train_lstm_model(train_data: pd.DataFrame, val_data: pd.DataFrame, 
                    logger: Optional[logging.Logger] = None) -> LSTMPricePredictor:
    """
    Train LSTM price prediction model
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        logger: Logger instance
        
    Returns:
        Trained LSTMPricePredictor instance
    """
    if logger is None:
        logger = setup_logging()
    
    predictor = LSTMPricePredictor(logger)
    
    try:
        # Train the model
        training_metrics = predictor.train(train_data, val_data)
        
        # Save models
        predictor.save_models()
        
        logger.info("LSTM model training completed successfully")
        logger.info(f"Training metrics: {training_metrics}")
        
        return predictor
        
    except Exception as e:
        logger.error(f"Error during LSTM training: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    import warnings
    warnings.filterwarnings('ignore')
    
    # Setup logging
    logger = setup_logging()
    
    # This would be replaced with actual data loading
    logger.info("LSTM Price Predictor - Stage 1")
    logger.info("This module should be imported and used within the main pipeline")
    logger.info("Example: predictor = train_lstm_model(train_data, val_data)")