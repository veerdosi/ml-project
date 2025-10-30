"""
Main Pipeline Orchestrator for ML Trading System
Coordinates the complete 3-stage pipeline execution
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
import argparse
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import all components
from config import SUPPORTED_STOCKS, DATA_CONFIG, LOGGING_CONFIG
from utils import (
    setup_logging, set_random_seeds, validate_data, split_data,
    calculate_technical_features, create_model_directory
)
from stage1_price_prediction import train_lstm_model, LSTMPricePredictor
from stage2_trend_classification import train_trend_classifier, TrendClassifier
from stage3_trading_agent import train_trading_agent, TradingAgent
from backtesting_engine import BacktestingEngine

class MLTradingPipeline:
    """Main pipeline orchestrator for the ML trading system"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logging()
        self.results = {}
        
        # Set random seeds for reproducibility
        set_random_seeds()
        
        # Create necessary directories
        create_model_directory()
        os.makedirs('logs', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        
        self.logger.info("ML Trading Pipeline initialized")
    
    def load_data(self, data_path: str, stock_symbol: str) -> pd.DataFrame:
        """
        Load and validate stock data
        
        Args:
            data_path: Path to data file
            stock_symbol: Stock symbol
            
        Returns:
            Validated DataFrame
        """
        try:
            # Load data
            if data_path.endswith('.csv'):
                data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            elif data_path.endswith('.parquet'):
                data = pd.read_parquet(data_path)
            else:
                raise ValueError("Unsupported file format. Use .csv or .parquet")
            
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Sort by date
            data = data.sort_index()
            
            # Add technical features if missing
            data = calculate_technical_features(data)
            
            # Validate data
            if not validate_data(data, self.logger):
                raise ValueError("Data validation failed")
            
            self.logger.info(f"Loaded data for {stock_symbol}: {len(data)} samples, "
                           f"date range: {data.index[0]} to {data.index[-1]}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data for {stock_symbol}: {e}")
            raise
    
    def run_single_stock(self, data: pd.DataFrame, stock_symbol: str,
                        train_models: bool = True, load_existing: bool = False) -> Dict[str, Any]:
        """
        Run complete pipeline for a single stock
        
        Args:
            data: Stock data with technical indicators
            stock_symbol: Stock symbol
            train_models: Whether to train new models
            load_existing: Whether to load existing models
            
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info(f"Starting pipeline for {stock_symbol}")
        self.logger.info("="*50)
        
        try:
            # Split data
            self.logger.info("Step 1: Splitting data...")
            train_data, val_data, test_data = split_data(data, self.logger)
            
            # Initialize models
            lstm_model = LSTMPricePredictor(self.logger)
            rf_model = TrendClassifier(self.logger)
            rl_agent = TradingAgent(self.logger)
            
            # Load or train models
            if load_existing:
                self.logger.info("Loading existing models...")
                try:
                    lstm_model.load_models()
                    rf_model.load_model()
                    rl_agent.load_model()
                    self.logger.info("Successfully loaded existing models")
                except Exception as e:
                    self.logger.warning(f"Could not load existing models: {e}")
                    self.logger.info("Falling back to training new models")
                    train_models = True
            
            if train_models:
                # Stage 1: Train LSTM Price Predictor
                self.logger.info("Step 2: Training LSTM Price Predictor (Stage 1)...")
                lstm_model = train_lstm_model(train_data, val_data, self.logger)
                
                # Get predictions for training trend classifier
                self.logger.info("Generating price predictions for training data...")
                price_predictions_train = lstm_model.predict(train_data, return_confidence=True)
                price_predictions_val = lstm_model.predict(val_data, return_confidence=True)
                
                # Stage 2: Train Trend Classifier
                self.logger.info("Step 3: Training Random Forest Trend Classifier (Stage 2)...")
                rf_model = train_trend_classifier(train_data, price_predictions_train, self.logger)
                
                # Get trend predictions for training RL agent
                self.logger.info("Generating trend predictions for training data...")
                trend_predictions_train = rf_model.predict(train_data, price_predictions_train)
                trend_predictions_val = rf_model.predict(val_data, price_predictions_val)
                
                # Stage 3: Train RL Trading Agent
                self.logger.info("Step 4: Training RL Trading Agent (Stage 3)...")
                rl_agent = train_trading_agent(
                    train_data, val_data,
                    price_predictions_train, price_predictions_val,
                    trend_predictions_train, trend_predictions_val,
                    self.logger
                )
            
            # Backtesting
            self.logger.info("Step 5: Running Backtest...")
            backtesting_engine = BacktestingEngine(self.logger)
            backtest_results = backtesting_engine.run_backtest(
                test_data, lstm_model, rf_model, rl_agent, stock_symbol
            )
            
            # Store results
            pipeline_results = {
                'stock_symbol': stock_symbol,
                'data_info': {
                    'total_samples': len(data),
                    'train_samples': len(train_data),
                    'val_samples': len(val_data),
                    'test_samples': len(test_data),
                    'date_range': (data.index[0], data.index[-1])
                },
                'models': {
                    'lstm_model': lstm_model,
                    'rf_model': rf_model,
                    'rl_agent': rl_agent
                },
                'backtest_results': backtest_results
            }
            
            self.results[stock_symbol] = pipeline_results
            
            self.logger.info(f"Pipeline completed successfully for {stock_symbol}")
            self.logger.info("="*50)
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed for {stock_symbol}: {e}")
            raise
    
    def run_multiple_stocks(self, data_configs: List[Dict[str, str]],
                           train_models: bool = True, load_existing: bool = False) -> Dict[str, Any]:
        """
        Run pipeline for multiple stocks
        
        Args:
            data_configs: List of dicts with 'symbol' and 'data_path' keys
            train_models: Whether to train new models
            load_existing: Whether to load existing models
            
        Returns:
            Dictionary with results for all stocks
        """
        self.logger.info(f"Starting pipeline for {len(data_configs)} stocks")
        
        all_results = {}
        
        for config in data_configs:
            stock_symbol = config['symbol']
            data_path = config['data_path']
            
            try:
                # Load data
                data = self.load_data(data_path, stock_symbol)
                
                # Run pipeline
                results = self.run_single_stock(data, stock_symbol, train_models, load_existing)
                all_results[stock_symbol] = results
                
            except Exception as e:
                self.logger.error(f"Failed to process {stock_symbol}: {e}")
                continue
        
        # Generate summary report
        self.logger.info("Generating summary report...")
        self._generate_summary_report(all_results)
        
        return all_results
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate summary report for multiple stocks"""
        backtesting_engine = BacktestingEngine(self.logger)
        
        # Extract backtest results
        for symbol, result in results.items():
            if 'backtest_results' in result:
                backtesting_engine.results[symbol] = result['backtest_results']
        
        # Generate and print summary
        backtesting_engine.print_summary()
    
    def predict_single_stock(self, data: pd.DataFrame, stock_symbol: str,
                           model_dir: str = None) -> Dict[str, pd.DataFrame]:
        """
        Make predictions on new data using pre-trained models
        
        Args:
            data: New stock data
            stock_symbol: Stock symbol
            model_dir: Directory containing trained models
            
        Returns:
            Dictionary with predictions from all stages
        """
        self.logger.info(f"Making predictions for {stock_symbol}")
        
        # Load models
        lstm_model = LSTMPricePredictor(self.logger)
        rf_model = TrendClassifier(self.logger)
        rl_agent = TradingAgent(self.logger)
        
        try:
            lstm_model.load_models(model_dir)
            rf_model.load_model()
            rl_agent.load_model()
        except Exception as e:
            raise ValueError(f"Could not load models: {e}")
        
        # Make predictions
        price_predictions = lstm_model.predict(data, return_confidence=True)
        trend_predictions = rf_model.predict(data, price_predictions)
        trading_decisions = rl_agent.predict(data, price_predictions, trend_predictions)
        
        return {
            'price_predictions': price_predictions,
            'trend_predictions': trend_predictions,
            'trading_decisions': trading_decisions
        }

def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description='ML Trading System Pipeline')
    parser.add_argument('--mode', choices=['train', 'predict', 'backtest'], 
                       default='train', help='Mode to run the pipeline')
    parser.add_argument('--stock', type=str, help='Stock symbol')
    parser.add_argument('--data_path', type=str, help='Path to data file')
    parser.add_argument('--config_file', type=str, help='Path to configuration file with multiple stocks')
    parser.add_argument('--load_existing', action='store_true', 
                       help='Load existing models instead of training')
    parser.add_argument('--no_train', action='store_true', 
                       help='Skip training (only for predict/backtest modes)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Initialize pipeline
    pipeline = MLTradingPipeline(logger)
    
    if args.mode == 'train':
        if args.config_file:
            # Multiple stocks from config file
            import json
            with open(args.config_file, 'r') as f:
                config_data = json.load(f)
            
            data_configs = config_data.get('stocks', [])
            pipeline.run_multiple_stocks(data_configs, 
                                       train_models=not args.no_train,
                                       load_existing=args.load_existing)
        
        elif args.stock and args.data_path:
            # Single stock
            data = pipeline.load_data(args.data_path, args.stock)
            pipeline.run_single_stock(data, args.stock,
                                    train_models=not args.no_train,
                                    load_existing=args.load_existing)
        else:
            logger.error("Please provide either --config_file or both --stock and --data_path")
            sys.exit(1)
    
    elif args.mode == 'predict':
        if not (args.stock and args.data_path):
            logger.error("Predict mode requires both --stock and --data_path")
            sys.exit(1)
        
        data = pipeline.load_data(args.data_path, args.stock)
        predictions = pipeline.predict_single_stock(data, args.stock)
        
        # Save predictions
        for pred_type, pred_data in predictions.items():
            output_file = f"predictions_{args.stock}_{pred_type}.csv"
            pred_data.to_csv(output_file)
            logger.info(f"Saved {pred_type} to {output_file}")
    
    else:  # backtest mode
        logger.error("Backtest mode not implemented in CLI. Use training mode with backtest results.")
        sys.exit(1)

# Example usage functions
def run_full_pipeline_example():
    """Example of running the complete pipeline"""
    logger = setup_logging()
    pipeline = MLTradingPipeline(logger)
    
    # Example configuration for multiple stocks
    data_configs = [
        {'symbol': 'AAPL', 'data_path': 'data/AAPL_with_indicators.csv'},
        {'symbol': 'MSFT', 'data_path': 'data/MSFT_with_indicators.csv'},
        {'symbol': 'NVDA', 'data_path': 'data/NVDA_with_indicators.csv'},
    ]
    
    # Run pipeline
    results = pipeline.run_multiple_stocks(data_configs, train_models=True)
    
    return results

def run_single_stock_example():
    """Example of running pipeline for a single stock"""
    logger = setup_logging()
    pipeline = MLTradingPipeline(logger)
    
    # Load data (replace with actual path)
    data = pipeline.load_data('data/AAPL_with_indicators.csv', 'AAPL')
    
    # Run pipeline
    results = pipeline.run_single_stock(data, 'AAPL', train_models=True)
    
    return results

if __name__ == "__main__":
    # Run CLI if arguments provided, otherwise run example
    if len(sys.argv) > 1:
        main()
    else:
        print("ML Trading System Pipeline")
        print("=" * 50)
        print("No arguments provided. Running in example mode.")
        print("\nTo use the CLI, run:")
        print("python main.py --mode train --stock AAPL --data_path data/AAPL.csv")
        print("python main.py --mode train --config_file config/stocks.json")
        print("python main.py --mode predict --stock AAPL --data_path data/AAPL_new.csv")
        print("\nFor more options, run: python main.py --help")
        
        # Set up basic logging for example
        logger = setup_logging()
        logger.info("Example mode - Pipeline ready for use")
        logger.info("Please provide data files and run with appropriate arguments")
        logger.info("Expected data format: CSV with OHLCV and technical indicators")