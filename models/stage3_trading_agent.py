"""
Stage 3: Reinforcement Learning Trading Agent
Makes trading decisions using predictions from Stages 1 & 2
"""

import pandas as pd
import numpy as np
import logging
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from typing import Dict, List, Optional, Tuple, Any
import pickle
import warnings
warnings.filterwarnings('ignore')

from config import (
    RL_CONFIG, ACTION_CONFIG, STATE_CONFIG, RISK_CONFIG, 
    MODEL_CONFIG, RANDOM_SEEDS
)
from utils import (
    setup_logging, set_random_seeds, validate_data, calculate_portfolio_metrics,
    get_model_path, create_model_directory
)

class TradingEnvironment(gym.Env):
    """Custom trading environment for reinforcement learning"""
    
    def __init__(self, data: pd.DataFrame, price_predictions: pd.DataFrame,
                 trend_predictions: pd.DataFrame, initial_cash: float = None,
                 logger: Optional[logging.Logger] = None):
        super(TradingEnvironment, self).__init__()
        
        self.logger = logger or logging.getLogger(__name__)
        self.data = data.copy()
        self.price_predictions = price_predictions.set_index('date') if 'date' in price_predictions.columns else price_predictions
        self.trend_predictions = trend_predictions.set_index('date') if 'date' in trend_predictions.columns else trend_predictions
        
        # Environment parameters
        self.initial_cash = initial_cash or RL_CONFIG['initial_cash']
        self.transaction_cost = RL_CONFIG['transaction_cost']
        self.max_position_size = RL_CONFIG['max_position_size']
        self.min_trade_amount = RL_CONFIG['min_trade_amount']
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(ACTION_CONFIG['n_actions'])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(STATE_CONFIG['state_size'],), 
            dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.max_steps = len(data) - 1
        self.cash = self.initial_cash
        self.shares_held = 0
        self.portfolio_value = self.initial_cash
        self.peak_portfolio_value = self.initial_cash
        
        # Trading history
        self.trades = []
        self.portfolio_history = []
        self.returns_history = []
        
        # Risk management
        self.consecutive_losses = 0
        self.days_in_position = 0
        
        # Align predictions with data
        self._align_predictions()
    
    def _align_predictions(self):
        """Align prediction data with main data"""
        # Reindex predictions to match data
        self.price_predictions = self.price_predictions.reindex(self.data.index)
        self.trend_predictions = self.trend_predictions.reindex(self.data.index)
        
        # Fill missing values
        self.price_predictions = self.price_predictions.fillna(method='ffill').fillna(method='bfill')
        self.trend_predictions = self.trend_predictions.fillna(method='ffill').fillna(method='bfill')
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 60  # Start after lookback period
        self.cash = self.initial_cash
        self.shares_held = 0
        self.portfolio_value = self.initial_cash
        self.peak_portfolio_value = self.initial_cash
        
        # Clear history
        self.trades = []
        self.portfolio_history = []
        self.returns_history = []
        
        # Reset risk management
        self.consecutive_losses = 0
        self.days_in_position = 0
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        if self.current_step >= len(self.data):
            return np.zeros(STATE_CONFIG['state_size'], dtype=np.float32)
        
        current_date = self.data.index[self.current_step]
        current_price = self.data.loc[current_date, 'Close']
        
        # Get predictions
        predicted_price = self.price_predictions.loc[current_date, 'predicted_price'] if current_date in self.price_predictions.index else current_price
        price_confidence = self.price_predictions.loc[current_date, 'prediction_confidence'] if current_date in self.price_predictions.index else 0.5
        
        # Get trend predictions
        trend_probs = [0.33, 0.33, 0.34]  # Default equal probabilities
        trend_confidence = 0.5
        if current_date in self.trend_predictions.index:
            trend_probs = [
                self.trend_predictions.loc[current_date, 'prob_down'],
                self.trend_predictions.loc[current_date, 'prob_sideways'],
                self.trend_predictions.loc[current_date, 'prob_up']
            ]
            trend_confidence = self.trend_predictions.loc[current_date, 'trend_confidence']
        
        # Portfolio state
        total_value = self.cash + self.shares_held * current_price
        position_ratio = (self.shares_held * current_price) / total_value if total_value > 0 else 0
        cash_ratio = self.cash / total_value if total_value > 0 else 1
        
        # Unrealized P&L
        if self.shares_held > 0:
            unrealized_pnl = (current_price * self.shares_held) - (self.shares_held * self._get_avg_entry_price())
            unrealized_pnl_ratio = unrealized_pnl / (self.shares_held * self._get_avg_entry_price()) if self.shares_held > 0 else 0
        else:
            unrealized_pnl_ratio = 0
        
        # Technical indicators
        rsi = self.data.loc[current_date, 'RSI'] / 100.0  # Normalize to 0-1
        macd = self.data.loc[current_date, 'MACD']
        bb_position = self._calculate_bb_position(current_date)
        
        # Risk metrics
        current_drawdown = self._calculate_current_drawdown()
        portfolio_volatility = self._calculate_portfolio_volatility()
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Price momentum
        price_momentum = self._calculate_price_momentum(current_date)
        volume_momentum = self._calculate_volume_momentum(current_date)
        
        # Risk signals
        profit_taking_signal = 1.0 if unrealized_pnl_ratio > RISK_CONFIG['profit_taking_threshold'] else 0.0
        stop_loss_signal = 1.0 if unrealized_pnl_ratio < -RISK_CONFIG['stop_loss_threshold'] else 0.0
        
        # Construct observation
        observation = np.array([
            predicted_price / current_price,  # Normalized predicted price
            trend_probs[0],  # Prob down
            trend_probs[1],  # Prob sideways 
            trend_probs[2],  # Prob up
            trend_confidence,
            position_ratio,
            cash_ratio,
            current_price / 1000.0,  # Normalized price
            unrealized_pnl_ratio,
            rsi,
            macd / 100.0,  # Normalized MACD
            bb_position,
            current_drawdown,
            portfolio_volatility,
            sharpe_ratio / 5.0,  # Normalized Sharpe
            price_momentum,
            volume_momentum,
            self.days_in_position / 100.0,  # Normalized days
            profit_taking_signal,
            stop_loss_signal
        ], dtype=np.float32)
        
        # Replace any NaN or inf values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute trading action and return new state"""
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0, True, {}
        
        current_date = self.data.index[self.current_step]
        current_price = self.data.loc[current_date, 'Close']
        
        # Store previous portfolio value
        prev_portfolio_value = self.cash + self.shares_held * current_price
        
        # Execute action
        reward = self._execute_action(action, current_price, current_date)
        
        # Update portfolio tracking
        new_portfolio_value = self.cash + self.shares_held * current_price
        self.portfolio_value = new_portfolio_value
        self.peak_portfolio_value = max(self.peak_portfolio_value, new_portfolio_value)
        
        # Calculate return
        if prev_portfolio_value > 0:
            portfolio_return = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
            self.returns_history.append(portfolio_return)
        
        # Store portfolio history
        self.portfolio_history.append({
            'date': current_date,
            'portfolio_value': new_portfolio_value,
            'cash': self.cash,
            'shares_held': self.shares_held,
            'price': current_price
        })
        
        # Update position tracking
        if self.shares_held > 0:
            self.days_in_position += 1
        else:
            self.days_in_position = 0
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = (self.current_step >= self.max_steps or 
                new_portfolio_value <= 0.1 * self.initial_cash)  # Stop if lost 90%
        
        # Get new observation
        obs = self._get_observation()
        
        # Additional info
        info = {
            'portfolio_value': new_portfolio_value,
            'cash': self.cash,
            'shares_held': self.shares_held,
            'action': action,
            'current_price': current_price
        }
        
        return obs, reward, done, info
    
    def _execute_action(self, action: int, current_price: float, current_date) -> float:
        """Execute trading action and return reward"""
        action_info = ACTION_CONFIG['actions'][action]
        position_change = action_info['position_change']
        
        # Calculate current portfolio value
        current_portfolio_value = self.cash + self.shares_held * current_price
        
        # Calculate shares to trade
        if position_change > 0:  # Buy
            max_shares_to_buy = self.cash // current_price
            target_cash_to_use = current_portfolio_value * position_change
            shares_to_buy = min(max_shares_to_buy, target_cash_to_use // current_price)
            shares_to_buy = max(0, shares_to_buy)
            
            if shares_to_buy * current_price >= self.min_trade_amount:
                self._execute_buy(shares_to_buy, current_price, current_date)
        
        elif position_change < 0:  # Sell
            shares_to_sell = int(self.shares_held * abs(position_change))
            shares_to_sell = min(shares_to_sell, self.shares_held)
            
            if shares_to_sell > 0:
                self._execute_sell(shares_to_sell, current_price, current_date)
        
        # Calculate reward
        reward = self._calculate_reward(current_price, current_date)
        
        return reward
    
    def _execute_buy(self, shares: int, price: float, date):
        """Execute buy order"""
        if shares <= 0:
            return
        
        total_cost = shares * price
        transaction_cost = total_cost * self.transaction_cost
        total_cost_with_fees = total_cost + transaction_cost
        
        if self.cash >= total_cost_with_fees:
            self.cash -= total_cost_with_fees
            self.shares_held += shares
            
            # Record trade
            self.trades.append({
                'date': date,
                'action': 'BUY',
                'shares': shares,
                'price': price,
                'total_cost': total_cost_with_fees,
                'cash_after': self.cash,
                'shares_after': self.shares_held
            })
    
    def _execute_sell(self, shares: int, price: float, date):
        """Execute sell order"""
        if shares <= 0 or shares > self.shares_held:
            return
        
        total_revenue = shares * price
        transaction_cost = total_revenue * self.transaction_cost
        total_revenue_after_fees = total_revenue - transaction_cost
        
        # Calculate P&L for this trade
        avg_entry_price = self._get_avg_entry_price()
        pnl = (price - avg_entry_price) * shares if avg_entry_price > 0 else 0
        
        self.cash += total_revenue_after_fees
        self.shares_held -= shares
        
        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Record trade
        self.trades.append({
            'date': date,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'total_revenue': total_revenue_after_fees,
            'pnl': pnl,
            'cash_after': self.cash,
            'shares_after': self.shares_held
        })
    
    def _calculate_reward(self, current_price: float, current_date) -> float:
        """Calculate reward for current step"""
        # Portfolio return component
        current_portfolio_value = self.cash + self.shares_held * current_price
        if len(self.portfolio_history) > 0:
            prev_portfolio_value = self.portfolio_history[-1]['portfolio_value']
            portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        else:
            portfolio_return = 0
        
        # Base reward from portfolio return
        reward = portfolio_return * 100
        
        # Risk penalties
        drawdown_penalty = 0
        if self.peak_portfolio_value > 0:
            current_drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
            if current_drawdown > RISK_CONFIG['max_drawdown_threshold']:
                drawdown_penalty = current_drawdown * 50
        
        # Consecutive losses penalty
        consecutive_loss_penalty = min(self.consecutive_losses * 2, 10)
        
        # Sharpe bonus
        sharpe_bonus = 0
        if len(self.returns_history) >= 20:
            sharpe_ratio = self._calculate_sharpe_ratio()
            if sharpe_ratio > 1.5:
                sharpe_bonus = 5
        
        # Position holding bonus (encourage some persistence)
        position_bonus = 0
        if 5 <= self.days_in_position <= 20:  # Sweet spot for holding
            position_bonus = 1
        
        # Final reward
        final_reward = reward - drawdown_penalty - consecutive_loss_penalty + sharpe_bonus + position_bonus
        
        return final_reward
    
    def _get_avg_entry_price(self) -> float:
        """Calculate average entry price for current position"""
        if self.shares_held == 0:
            return 0
        
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        if not buy_trades:
            return self.data.iloc[0]['Close']  # Fallback
        
        total_cost = sum(t['shares'] * t['price'] for t in buy_trades)
        total_shares = sum(t['shares'] for t in buy_trades)
        
        return total_cost / total_shares if total_shares > 0 else 0
    
    def _calculate_bb_position(self, date) -> float:
        """Calculate Bollinger Band position"""
        try:
            price = self.data.loc[date, 'Close']
            bb_upper = self.data.loc[date, 'BB_Upper']
            bb_lower = self.data.loc[date, 'BB_Lower']
            return (price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        except:
            return 0.5
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown"""
        if self.peak_portfolio_value == 0:
            return 0
        current_value = self.cash + self.shares_held * self.data.iloc[self.current_step]['Close']
        return (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility"""
        if len(self.returns_history) < 2:
            return 0
        recent_returns = self.returns_history[-20:]  # Last 20 periods
        return np.std(recent_returns) * np.sqrt(252)  # Annualized
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.returns_history) < 2:
            return 0
        returns = np.array(self.returns_history)
        if np.std(returns) == 0:
            return 0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
    
    def _calculate_price_momentum(self, date) -> float:
        """Calculate price momentum"""
        try:
            current_idx = self.data.index.get_loc(date)
            if current_idx >= 5:
                current_price = self.data.iloc[current_idx]['Close']
                past_price = self.data.iloc[current_idx - 5]['Close']
                return (current_price - past_price) / past_price
        except:
            pass
        return 0
    
    def _calculate_volume_momentum(self, date) -> float:
        """Calculate volume momentum"""
        try:
            current_idx = self.data.index.get_loc(date)
            if current_idx >= 5:
                current_volume = self.data.iloc[current_idx]['Volume']
                past_volume = self.data.iloc[current_idx - 5]['Volume']
                return (current_volume - past_volume) / past_volume if past_volume > 0 else 0
        except:
            pass
        return 0
    
    def get_trading_log(self) -> pd.DataFrame:
        """Get complete trading log"""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio value history"""
        if not self.portfolio_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.portfolio_history)

class TradingAgent:
    """RL Trading Agent using PPO"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logging()
        self.model = None
        self.env = None
        self.is_trained = False
        
        # Set random seeds
        set_random_seeds()
        
        # Create model directory
        create_model_directory()
    
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame,
             price_predictions_train: pd.DataFrame, price_predictions_val: pd.DataFrame,
             trend_predictions_train: pd.DataFrame, trend_predictions_val: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the RL trading agent
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            price_predictions_train: Price predictions for training data
            price_predictions_val: Price predictions for validation data
            trend_predictions_train: Trend predictions for training data
            trend_predictions_val: Trend predictions for validation data
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info("Starting RL agent training...")
        
        # Create training environment
        train_env = TradingEnvironment(
            train_data, price_predictions_train, trend_predictions_train,
            RL_CONFIG['initial_cash'], self.logger
        )
        
        # Create validation environment
        val_env = TradingEnvironment(
            val_data, price_predictions_val, trend_predictions_val,
            RL_CONFIG['initial_cash'], self.logger
        )
        
        # Wrap environments
        train_env = DummyVecEnv([lambda: train_env])
        val_env = DummyVecEnv([lambda: val_env])
        
        # Create PPO model
        self.model = PPO(
            'MlpPolicy',
            train_env,
            learning_rate=RL_CONFIG['learning_rate'],
            n_steps=RL_CONFIG['n_steps'],
            batch_size=RL_CONFIG['batch_size'],
            n_epochs=RL_CONFIG['n_epochs'],
            gamma=RL_CONFIG['gamma'],
            gae_lambda=RL_CONFIG['gae_lambda'],
            clip_range=RL_CONFIG['clip_range'],
            ent_coef=RL_CONFIG['ent_coef'],
            vf_coef=RL_CONFIG['vf_coef'],
            max_grad_norm=RL_CONFIG['max_grad_norm'],
            policy_kwargs=RL_CONFIG['policy_kwargs'],
            tensorboard_log="./tensorboard_logs/",
            verbose=1
        )
        
        # Setup callbacks
        eval_callback = EvalCallback(
            val_env, 
            best_model_save_path=get_model_path(""),
            log_path="./logs/",
            eval_freq=10000,
            deterministic=True,
            render=False
        )
        
        # Train the model
        self.logger.info(f"Training for {RL_CONFIG['total_timesteps']} timesteps...")
        self.model.learn(
            total_timesteps=RL_CONFIG['total_timesteps'],
            callback=eval_callback
        )
        
        self.is_trained = True
        
        # Save model
        model_path = get_model_path(MODEL_CONFIG['rl_model_name'])
        self.model.save(model_path)
        self.logger.info(f"Model saved to {model_path}")
        
        # Evaluate on validation set
        val_metrics = self.evaluate(val_data, price_predictions_val, trend_predictions_val)
        
        self.logger.info("RL agent training completed successfully")
        
        return {
            'training_timesteps': RL_CONFIG['total_timesteps'],
            'validation_metrics': val_metrics
        }
    
    def predict(self, data: pd.DataFrame, price_predictions: pd.DataFrame,
               trend_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Make trading decisions using trained agent
        
        Args:
            data: Input data
            price_predictions: Price predictions from Stage 1
            trend_predictions: Trend predictions from Stage 2
            
        Returns:
            DataFrame with trading decisions and portfolio state
        """
        if not self.is_trained:
            raise ValueError("Agent must be trained before making predictions")
        
        self.logger.info(f"Making trading decisions on {len(data)} samples")
        
        # Create environment
        env = TradingEnvironment(
            data, price_predictions, trend_predictions,
            RL_CONFIG['initial_cash'], self.logger
        )
        
        # Run episode
        obs = env.reset()
        done = False
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        
        # Get trading log and portfolio history
        trading_log = env.get_trading_log()
        portfolio_history = env.get_portfolio_history()
        
        # Combine into single result
        if len(trading_log) > 0 and len(portfolio_history) > 0:
            # Merge trading log with portfolio history
            result = portfolio_history.merge(
                trading_log[['date', 'action', 'shares', 'price']], 
                on='date', how='left'
            )
            result['action'] = result['action'].fillna('HOLD')
        else:
            result = portfolio_history.copy()
            result['action'] = 'HOLD'
            result['shares'] = 0
        
        return result
    
    def evaluate(self, test_data: pd.DataFrame, price_predictions: pd.DataFrame,
                trend_predictions: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate agent performance on test data
        
        Args:
            test_data: Test dataset
            price_predictions: Price predictions from Stage 1
            trend_predictions: Trend predictions from Stage 2
            
        Returns:
            Dictionary with performance metrics
        """
        # Get trading results
        results = self.predict(test_data, price_predictions, trend_predictions)
        
        if len(results) == 0:
            return {}
        
        # Calculate returns
        initial_value = RL_CONFIG['initial_cash']
        final_value = results['portfolio_value'].iloc[-1]
        returns = results['portfolio_value'].pct_change().dropna()
        
        # Calculate metrics using utils
        metrics = calculate_portfolio_metrics(returns)
        
        # Add additional metrics
        metrics.update({
            'initial_portfolio_value': initial_value,
            'final_portfolio_value': final_value,
            'total_return_pct': (final_value - initial_value) / initial_value * 100,
            'num_trades': len([r for r in results['action'] if r != 'HOLD']),
            'max_shares_held': results['shares_held'].max(),
            'avg_cash_ratio': (results['cash'] / results['portfolio_value']).mean()
        })
        
        self.logger.info(f"Performance: Total Return {metrics['total_return_pct']:.2f}%, "
                        f"Sharpe {metrics['sharpe_ratio']:.2f}, Max DD {metrics['max_drawdown']:.2%}")
        
        return metrics
    
    def load_model(self, model_path: str = None):
        """Load pre-trained model"""
        if model_path is None:
            model_path = get_model_path(MODEL_CONFIG['rl_model_name'])
        
        try:
            self.model = PPO.load(model_path)
            self.is_trained = True
            self.logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

def train_trading_agent(train_data: pd.DataFrame, val_data: pd.DataFrame,
                       price_predictions_train: pd.DataFrame, price_predictions_val: pd.DataFrame,
                       trend_predictions_train: pd.DataFrame, trend_predictions_val: pd.DataFrame,
                       logger: Optional[logging.Logger] = None) -> TradingAgent:
    """
    Train RL trading agent
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        price_predictions_train: Price predictions for training data
        price_predictions_val: Price predictions for validation data
        trend_predictions_train: Trend predictions for training data
        trend_predictions_val: Trend predictions for validation data
        logger: Logger instance
        
    Returns:
        Trained TradingAgent instance
    """
    if logger is None:
        logger = setup_logging()
    
    agent = TradingAgent(logger)
    
    try:
        # Train the agent
        training_metrics = agent.train(
            train_data, val_data,
            price_predictions_train, price_predictions_val,
            trend_predictions_train, trend_predictions_val
        )
        
        logger.info("RL trading agent training completed successfully")
        logger.info(f"Training metrics: {training_metrics}")
        
        return agent
        
    except Exception as e:
        logger.error(f"Error during RL agent training: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    import warnings
    warnings.filterwarnings('ignore')
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("RL Trading Agent - Stage 3")
    logger.info("This module should be imported and used within the main pipeline")
    logger.info("Example: agent = train_trading_agent(train_data, val_data, price_pred, trend_pred)")