"""
Stage 2: Random Forest Trend Classification
Classifies next day's trend direction using predictions from Stage 1 and technical indicators
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import pickle
import warnings
warnings.filterwarnings('ignore')

from config import RF_CONFIG, RF_FEATURES, MODEL_CONFIG, RANDOM_SEEDS
from utils import (
    setup_logging, set_random_seeds, validate_data, create_trend_labels,
    calculate_technical_features, get_time_series_split, save_model, 
    get_model_path, create_model_directory
)

class TrendClassifier:
    """Random Forest classifier for trend prediction"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logging()
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        self.class_names = ['DOWN', 'SIDEWAYS', 'UP']
        
        # Set random seeds
        set_random_seeds()
        
        # Create model directory
        create_model_directory()
    
    def _prepare_features(self, data: pd.DataFrame, price_predictions: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepare features for trend classification
        
        Args:
            data: Input data with technical indicators
            price_predictions: Price predictions from Stage 1
            
        Returns:
            DataFrame with prepared features
        """
        df = data.copy()
        
        # Add technical features
        df = calculate_technical_features(df)
        
        # Add price predictions if available
        if price_predictions is not None:
            # Align predictions with data
            aligned_predictions = price_predictions.set_index('date').reindex(df.index)
            df['predicted_price'] = aligned_predictions['predicted_price']
            df['price_confidence'] = aligned_predictions.get('prediction_confidence', 1.0)
            
            # Calculate prediction-based features
            df['predicted_return'] = (df['predicted_price'] - df['Close']) / df['Close'] * 100
            df['prediction_error'] = np.abs(df['predicted_price'] - df['Close']) / df['Close'] * 100
        else:
            # Use placeholder values for training without Stage 1 predictions
            df['predicted_price'] = df['Close']
            df['price_confidence'] = 1.0
            df['predicted_return'] = 0.0
            df['prediction_error'] = 0.0
        
        return df
    
    def _create_labels(self, data: pd.DataFrame) -> pd.Series:
        """
        Create trend labels for classification
        
        Args:
            data: Input data with price information
            
        Returns:
            Series of trend labels
        """
        # Calculate next day returns
        next_day_return = (data['Close'].shift(-1) - data['Close']) / data['Close'] * 100
        
        # Create labels
        labels = create_trend_labels(next_day_return, RF_CONFIG['trend_threshold'])
        
        return labels
    
    def _select_features(self, data: pd.DataFrame) -> List[str]:
        """
        Select available features from the feature list
        
        Args:
            data: Input DataFrame
            
        Returns:
            List of available feature names
        """
        available_features = []
        for feature in RF_FEATURES:
            if feature in data.columns:
                available_features.append(feature)
            else:
                self.logger.warning(f"Feature {feature} not found in data")
        
        return available_features
    
    def train(self, train_data: pd.DataFrame, price_predictions_train: pd.DataFrame = None,
             optimize_hyperparameters: bool = True) -> Dict[str, float]:
        """
        Train Random Forest trend classifier
        
        Args:
            train_data: Training dataset
            price_predictions_train: Price predictions for training data from Stage 1
            optimize_hyperparameters: Whether to perform hyperparameter optimization
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info("Starting Random Forest training...")
        
        # Validate data
        if not validate_data(train_data, self.logger):
            raise ValueError("Training data validation failed")
        
        # Prepare features
        prepared_data = self._prepare_features(train_data, price_predictions_train)
        
        # Create labels
        labels = self._create_labels(prepared_data)
        
        # Remove rows with NaN labels (last row typically)
        mask = ~labels.isna()
        prepared_data = prepared_data[mask]
        labels = labels[mask]
        
        # Select available features
        self.feature_names = self._select_features(prepared_data)
        
        if not self.feature_names:
            raise ValueError("No valid features found for training")
        
        X = prepared_data[self.feature_names].fillna(0)  # Fill any remaining NaNs
        y = labels
        
        self.logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        self.logger.info(f"Features used: {self.feature_names}")
        self.logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=RF_CONFIG['n_estimators'],
            max_depth=RF_CONFIG['max_depth'],
            min_samples_split=RF_CONFIG['min_samples_split'],
            min_samples_leaf=RF_CONFIG['min_samples_leaf'],
            max_features=RF_CONFIG['max_features'],
            class_weight=RF_CONFIG['class_weight'],
            random_state=RF_CONFIG['random_state'],
            n_jobs=RF_CONFIG['n_jobs']
        )
        
        # Hyperparameter optimization
        if optimize_hyperparameters:
            self.logger.info("Performing hyperparameter optimization...")
            best_params = self._optimize_hyperparameters(X_scaled, y)
            self.model.set_params(**best_params)
        
        # Train final model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)
        
        metrics = self._calculate_metrics(y, y_pred, y_proba)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.logger.info("Top 10 feature importances:")
        for _, row in feature_importance.head(10).iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save model and scaler
        self._save_model()
        
        self.logger.info("Random Forest training completed successfully")
        return metrics
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Optimize hyperparameters using GridSearchCV with TimeSeriesSplit
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary of best parameters
        """
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [10, 20, 30],
            'min_samples_leaf': [5, 10, 15],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Use TimeSeriesSplit for cross-validation
        tscv = get_time_series_split(len(X), RF_CONFIG['cv_folds'])
        
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(
                class_weight=RF_CONFIG['class_weight'],
                random_state=RF_CONFIG['random_state'],
                n_jobs=1  # Reduce parallelism for grid search
            ),
            param_grid=param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=2,  # Limited parallelism
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.logger.info(f"Best hyperparameters: {grid_search.best_params_}")
        self.logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def predict(self, data: pd.DataFrame, price_predictions: pd.DataFrame = None) -> pd.DataFrame:
        """
        Make trend predictions
        
        Args:
            data: Input data for prediction
            price_predictions: Price predictions from Stage 1
            
        Returns:
            DataFrame with trend predictions and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.logger.info(f"Making trend predictions on {len(data)} samples")
        
        # Prepare features
        prepared_data = self._prepare_features(data, price_predictions)
        
        # Select features and handle missing values
        X = prepared_data[self.feature_names].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'date': data.index,
            'predicted_trend': y_pred,
            'prob_down': y_proba[:, 0],
            'prob_sideways': y_proba[:, 1],
            'prob_up': y_proba[:, 2],
        })
        
        # Calculate trend confidence (max probability)
        results['trend_confidence'] = np.max(y_proba, axis=1)
        
        # Add actual trend labels if possible
        try:
            actual_labels = self._create_labels(data)
            results['actual_trend'] = actual_labels
        except:
            # Can't create actual labels (e.g., for future predictions)
            pass
        
        return results
    
    def evaluate(self, test_data: pd.DataFrame, price_predictions_test: pd.DataFrame = None) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: Test dataset
            price_predictions_test: Price predictions for test data from Stage 1
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        predictions = self.predict(test_data, price_predictions_test)
        
        if 'actual_trend' not in predictions.columns:
            self.logger.warning("Cannot evaluate without actual trend labels")
            return {}
        
        # Remove rows with NaN actual labels
        mask = ~predictions['actual_trend'].isna()
        predictions = predictions[mask]
        
        if len(predictions) == 0:
            return {}
        
        y_true = predictions['actual_trend'].values
        y_pred = predictions['predicted_trend'].values
        y_proba = predictions[['prob_down', 'prob_sideways', 'prob_up']].values
        
        metrics = self._calculate_metrics(y_true, y_pred, y_proba)
        
        self.logger.info(f"Evaluation metrics: Accuracy={metrics['accuracy']:.4f}, "
                        f"Macro F1={metrics['macro_f1']:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics"""
        from sklearn.metrics import f1_score, precision_score, recall_score, log_loss
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        macro_precision = precision_score(y_true, y_pred, average='macro')
        macro_recall = recall_score(y_true, y_pred, average='macro')
        
        # Log loss
        try:
            logloss = log_loss(y_true, y_proba)
        except:
            logloss = np.nan
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Class-specific metrics
        class_report = classification_report(y_true, y_pred, 
                                           target_names=self.class_names,
                                           output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'log_loss': logloss,
            'confusion_matrix': cm.tolist(),
            'class_report': class_report,
            'prediction_samples': len(y_true)
        }
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def _save_model(self):
        """Save trained model and scaler"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        # Save model
        model_path = get_model_path(MODEL_CONFIG['rf_model_name'])
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        scaler_path = get_model_path('rf_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature names
        features_path = get_model_path('rf_features.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self):
        """Load pre-trained model and scaler"""
        try:
            # Load model
            model_path = get_model_path(MODEL_CONFIG['rf_model_name'])
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            scaler_path = get_model_path('rf_scaler.pkl')
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load feature names
            features_path = get_model_path('rf_features.pkl')
            with open(features_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            
            self.is_trained = True
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

def train_trend_classifier(train_data: pd.DataFrame, price_predictions_train: pd.DataFrame = None,
                          logger: Optional[logging.Logger] = None) -> TrendClassifier:
    """
    Train Random Forest trend classifier
    
    Args:
        train_data: Training dataset
        price_predictions_train: Price predictions for training data from Stage 1
        logger: Logger instance
        
    Returns:
        Trained TrendClassifier instance
    """
    if logger is None:
        logger = setup_logging()
    
    classifier = TrendClassifier(logger)
    
    try:
        # Train the model
        training_metrics = classifier.train(train_data, price_predictions_train)
        
        logger.info("Random Forest trend classifier training completed successfully")
        logger.info(f"Training metrics: {training_metrics}")
        
        return classifier
        
    except Exception as e:
        logger.error(f"Error during Random Forest training: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    import warnings
    warnings.filterwarnings('ignore')
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("Random Forest Trend Classifier - Stage 2")
    logger.info("This module should be imported and used within the main pipeline")
    logger.info("Example: classifier = train_trend_classifier(train_data, price_predictions)")