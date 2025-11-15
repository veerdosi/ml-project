# Technical Indicators Guide - 7 Year Historical Data

## Overview

This document provides a comprehensive guide to the **80+ technical indicators** created for the 7-year historical stock dataset. These indicators are designed to capture price patterns, momentum, volatility, volume dynamics, and trend characteristics.

---

## Table of Contents

1. [Moving Averages](#moving-averages)
2. [Momentum Indicators](#momentum-indicators)
3. [Volatility Indicators](#volatility-indicators)
4. [Volume Indicators](#volume-indicators)
5. [Trend Strength Indicators](#trend-strength-indicators)
6. [Pattern Recognition Features](#pattern-recognition-features)
7. [Time-Based Features](#time-based-features)
8. [Lagged Features](#lagged-features)
9. [Usage Examples](#usage-examples)

---

## Moving Averages

Moving averages smooth price data to identify trends and potential support/resistance levels.

### Simple Moving Averages (SMA)

| Feature | Description | Period | Use Case |
|---------|-------------|--------|----------|
| `SMA_5` | 5-day simple moving average | 5 days | Very short-term trend |
| `SMA_10` | 10-day simple moving average | 10 days | Short-term trend |
| `SMA_20` | 20-day simple moving average | 20 days | Common swing trading reference |
| `SMA_50` | 50-day simple moving average | 50 days | Intermediate trend indicator |
| `SMA_100` | 100-day simple moving average | 100 days | Long-term trend indicator |
| `SMA_200` | 200-day simple moving average | 200 days | Major long-term trend (bull/bear market) |

**Interpretation:**
- Price above MA = Uptrend
- Price below MA = Downtrend
- Longer periods = Less noise, more lag

### Exponential Moving Averages (EMA)

| Feature | Description | Period | Use Case |
|---------|-------------|--------|----------|
| `EMA_12` | 12-day exponential moving average | 12 days | MACD fast line component |
| `EMA_26` | 26-day exponential moving average | 26 days | MACD slow line component |
| `EMA_50` | 50-day exponential moving average | 50 days | More responsive than SMA_50 |

**Interpretation:**
- EMAs give more weight to recent prices
- More responsive to price changes than SMA
- Better for short-term trading signals

### Moving Average Crossovers

| Feature | Description | Components | Signal |
|---------|-------------|------------|--------|
| `MA_Cross_50_200` | Golden/Death Cross indicator | SMA_50 - SMA_200 | Positive = Golden Cross (bullish) |
| `MA_Cross_20_50` | Short/Intermediate crossover | SMA_20 - SMA_50 | Positive = Bullish crossover |

**Interpretation:**
- **Golden Cross**: 50-day MA crosses above 200-day MA (major bullish signal)
- **Death Cross**: 50-day MA crosses below 200-day MA (major bearish signal)

---

## Momentum Indicators

Momentum indicators measure the speed and strength of price movements.

### Relative Strength Index (RSI)

| Feature | Description | Period | Range |
|---------|-------------|--------|-------|
| `RSI_14` | Relative Strength Index | 14 days | 0-100 |
| `RSI_Overbought` | Binary flag for overbought condition | - | 0 or 1 |
| `RSI_Oversold` | Binary flag for oversold condition | - | 0 or 1 |

**Interpretation:**
- **RSI > 70**: Overbought (potential sell signal)
- **RSI < 30**: Oversold (potential buy signal)
- **RSI = 50**: Neutral momentum
- **Divergence**: Price makes new high/low but RSI doesn't confirm

### MACD (Moving Average Convergence Divergence)

| Feature | Description | Formula |
|---------|-------------|---------|
| `MACD` | MACD line | EMA(12) - EMA(26) |
| `MACD_signal` | Signal line | 9-day EMA of MACD |
| `MACD_hist` | MACD histogram | MACD - Signal |

**Interpretation:**
- **MACD > Signal**: Bullish momentum
- **MACD < Signal**: Bearish momentum
- **Histogram expanding**: Momentum strengthening
- **Zero line crossover**: Trend change confirmation

### Stochastic Oscillator

| Feature | Description | Period | Range |
|---------|-------------|--------|-------|
| `Stoch_K` | Fast stochastic | 14 days | 0-100 |
| `Stoch_D` | Slow stochastic (smoothed K) | 3 days | 0-100 |

**Interpretation:**
- **> 80**: Overbought zone
- **< 20**: Oversold zone
- **K crosses above D**: Bullish signal
- **K crosses below D**: Bearish signal

### Williams %R

| Feature | Description | Period | Range |
|---------|-------------|--------|-------|
| `Williams_R` | Williams Percent Range | 14 days | -100 to 0 |

**Interpretation:**
- **> -20**: Overbought
- **< -80**: Oversold
- Inverse of Stochastic (plotted upside down)

### Commodity Channel Index (CCI)

| Feature | Description | Period | Typical Range |
|---------|-------------|--------|---------------|
| `CCI_20` | Commodity Channel Index | 20 days | -200 to +200 |

**Interpretation:**
- **> +100**: Overbought, strong uptrend
- **< -100**: Oversold, strong downtrend
- **Between -100 and +100**: Normal trading range
- Identifies cyclical trends and extremes

### Rate of Change (ROC)

| Feature | Description | Period | Unit |
|---------|-------------|--------|------|
| `ROC_10` | 10-day rate of change | 10 days | Percentage |
| `ROC_20` | 20-day rate of change | 20 days | Percentage |

**Interpretation:**
- **Positive**: Price increasing
- **Negative**: Price decreasing
- **Magnitude**: Speed of change

---

## Volatility Indicators

Volatility indicators measure the magnitude of price fluctuations.

### Bollinger Bands

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `BB_upper` | Upper Bollinger Band | SMA(20) + 2×STD |
| `BB_middle` | Middle Bollinger Band | SMA(20) |
| `BB_lower` | Lower Bollinger Band | SMA(20) - 2×STD |
| `BB_width` | Band width (volatility measure) | (Upper - Lower) / Middle |
| `BB_pct` | %B - Price position in bands | (Close - Lower) / (Upper - Lower) |

**Interpretation:**
- **Narrow bands**: Low volatility (potential breakout)
- **Wide bands**: High volatility
- **Price at upper band**: Overbought
- **Price at lower band**: Oversold
- **BB_pct > 1**: Price above upper band (very overbought)
- **BB_pct < 0**: Price below lower band (very oversold)

### Average True Range (ATR)

| Feature | Description | Period | Use Case |
|---------|-------------|--------|----------|
| `ATR_14` | Average True Range | 14 days | Volatility measurement, stop-loss placement |

**Interpretation:**
- **High ATR**: High volatility, wider stops needed
- **Low ATR**: Low volatility, tighter stops possible
- Used for position sizing and risk management

### Historical Volatility

| Feature | Description | Period | Annualized |
|---------|-------------|--------|------------|
| `HV_20` | 20-day historical volatility | 20 days | Yes (×√252) |
| `HV_50` | 50-day historical volatility | 50 days | Yes (×√252) |

**Interpretation:**
- Standard deviation of returns
- Higher values = More volatile
- Used for option pricing and risk assessment

### Keltner Channels

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `KC_upper` | Upper Keltner Channel | EMA(20) + 2×ATR(10) |
| `KC_lower` | Lower Keltner Channel | EMA(20) - 2×ATR(10) |

**Interpretation:**
- Similar to Bollinger Bands but uses ATR
- Price breaking above: Bullish
- Price breaking below: Bearish
- Inside channels: Range-bound market

---

## Volume Indicators

Volume indicators analyze trading activity to confirm price movements.

### Volume Moving Averages

| Feature | Description | Period | Use Case |
|---------|-------------|--------|----------|
| `Volume_MA_20` | 20-day volume average | 20 days | Identify abnormal volume |
| `Volume_MA_50` | 50-day volume average | 50 days | Long-term volume trend |

**Interpretation:**
- **Volume > Average**: Strong participation
- **Volume < Average**: Weak participation
- High volume confirms price moves

### Volume Rate of Change

| Feature | Description | Period |
|---------|-------------|--------|
| `Volume_ROC_10` | 10-day volume ROC | 10 days |
| `Volume_ROC_20` | 20-day volume ROC | 20 days |

**Interpretation:**
- **Positive**: Volume increasing
- **Negative**: Volume decreasing
- Confirms momentum changes

### On-Balance Volume (OBV)

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `OBV` | On-Balance Volume | Cumulative volume (add on up days, subtract on down days) |
| `OBV_MA_20` | 20-day OBV moving average | Smoothed OBV trend |

**Interpretation:**
- **OBV rising**: Accumulation (buying pressure)
- **OBV falling**: Distribution (selling pressure)
- **OBV divergence**: Leading indicator of trend change

### Volume Weighted Average Price (VWAP)

| Feature | Description | Use Case |
|---------|-------------|----------|
| `VWAP` | Volume-weighted average price | Intraday benchmark, institutional reference |
| `Price_VWAP_Ratio` | Price relative to VWAP | Price strength indicator |

**Interpretation:**
- **Price > VWAP**: Above average trading price (bullish)
- **Price < VWAP**: Below average trading price (bearish)
- Institutional traders use as execution benchmark

### Money Flow Index (MFI)

| Feature | Description | Period | Range |
|---------|-------------|--------|-------|
| `MFI_14` | Money Flow Index | 14 days | 0-100 |

**Interpretation:**
- Volume-weighted RSI
- **MFI > 80**: Overbought
- **MFI < 20**: Oversold
- Identifies buying/selling pressure with volume

---

## Trend Strength Indicators

These indicators measure the strength of a trend, not its direction.

### Average Directional Index (ADX)

| Feature | Description | Period | Range |
|---------|-------------|--------|-------|
| `ADX_14` | Average Directional Index | 14 days | 0-100 |
| `Plus_DI` | Plus Directional Indicator (+DI) | 14 days | 0-100 |
| `Minus_DI` | Minus Directional Indicator (-DI) | 14 days | 0-100 |

**Interpretation:**
- **ADX < 25**: Weak trend (range-bound)
- **ADX 25-50**: Strong trend
- **ADX > 50**: Very strong trend
- **+DI > -DI**: Bullish trend
- **-DI > +DI**: Bearish trend

### Aroon Indicator

| Feature | Description | Period | Range |
|---------|-------------|--------|-------|
| `Aroon_Up` | Aroon Up | 25 days | 0-100 |
| `Aroon_Down` | Aroon Down | 25 days | 0-100 |
| `Aroon_Oscillator` | Aroon Up - Aroon Down | 25 days | -100 to +100 |

**Interpretation:**
- **Aroon Up > 70**: Strong uptrend
- **Aroon Down > 70**: Strong downtrend
- **Aroon Oscillator > 0**: Bullish
- **Aroon Oscillator < 0**: Bearish

### Supertrend

| Feature | Description | Parameters | Values |
|---------|-------------|------------|--------|
| `Supertrend` | Supertrend line | Period=10, Multiplier=3 | Price level |
| `Supertrend_Direction` | Trend direction | - | 1 (up) or -1 (down) |

**Interpretation:**
- **Direction = 1**: Uptrend (price above Supertrend)
- **Direction = -1**: Downtrend (price below Supertrend)
- Trend-following indicator with dynamic support/resistance

---

## Pattern Recognition Features

### Price Position Relative to Moving Averages

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `Price_to_SMA20` | Distance from 20-day SMA | (Close - SMA_20) / SMA_20 |
| `Price_to_SMA50` | Distance from 50-day SMA | (Close - SMA_50) / SMA_50 |
| `Price_to_SMA200` | Distance from 200-day SMA | (Close - SMA_200) / SMA_200 |

**Interpretation:**
- **Positive**: Price above MA (bullish)
- **Negative**: Price below MA (bearish)
- **Magnitude**: How far from MA (extreme values = potential reversal)

### Price Action Patterns

| Feature | Description | Values |
|---------|-------------|--------|
| `Higher_High` | Today's high > Yesterday's high | 0 or 1 |
| `Lower_Low` | Today's low < Yesterday's low | 0 or 1 |
| `Daily_Range` | Intraday range relative to price | (High - Low) / Low |
| `High_Low_Ratio` | High/Low ratio | High / Low |

**Interpretation:**
- **Higher_High = 1**: Bullish momentum
- **Lower_Low = 1**: Bearish momentum
- **Large Daily_Range**: High volatility day
- **High_Low_Ratio near 1**: Small range (low volatility)

---

## Time-Based Features

Capture seasonal and cyclical patterns.

| Feature | Description | Range | Use Case |
|---------|-------------|-------|----------|
| `Day_of_Week` | Day of the week | 0-6 (Monday=0) | Day-of-week effects |
| `Month` | Month of the year | 1-12 | Seasonal patterns |
| `Quarter` | Quarter of the year | 1-4 | Quarterly earnings cycles |
| `Day_of_Month` | Day of the month | 1-31 | Monthly patterns (options expiry, etc.) |

**Interpretation:**
- **Day_of_Week**: Monday effect, Friday patterns
- **Month**: January effect, "Sell in May" patterns
- **Quarter**: Earnings season impact
- **Day_of_Month**: End-of-month flows

---

## Lagged Features

Historical values to capture temporal dependencies.

### Lagged Prices

| Feature | Description | Lag Period |
|---------|-------------|------------|
| `Close_lag_1` | Yesterday's close | 1 day |
| `Close_lag_2` | Close from 2 days ago | 2 days |
| `Close_lag_3` | Close from 3 days ago | 3 days |
| `Close_lag_5` | Close from 5 days ago | 5 days |
| `Close_lag_10` | Close from 10 days ago | 10 days |
| `Close_lag_20` | Close from 20 days ago | 20 days |

### Lagged Returns

| Feature | Description | Period |
|---------|-------------|--------|
| `Return_lag_1` | 1-day return | 1 day |
| `Return_lag_5` | 5-day return | 5 days |
| `Return_lag_10` | 10-day return | 10 days |
| `Return_lag_20` | 20-day return | 20 days |

### Lagged Volume

| Feature | Description | Lag Period |
|---------|-------------|------------|
| `Volume_lag_1` | Yesterday's volume | 1 day |
| `Volume_lag_5` | Volume from 5 days ago | 5 days |
| `Volume_lag_10` | Volume from 10 days ago | 10 days |

**Interpretation:**
- Captures price momentum and mean reversion
- Useful for time-series modeling (LSTM, etc.)
- Identifies patterns in recent history

---

## Usage Examples

### Example 1: Identify Overbought/Oversold Conditions

```python
# Stocks that are overbought
overbought = total_data[
    (total_data['RSI_14'] > 70) & 
    (total_data['MFI_14'] > 80) &
    (total_data['BB_pct'] > 0.8)
]

# Stocks that are oversold
oversold = total_data[
    (total_data['RSI_14'] < 30) & 
    (total_data['MFI_14'] < 20) &
    (total_data['BB_pct'] < 0.2)
]
```

### Example 2: Identify Strong Trends

```python
# Strong uptrends
strong_uptrend = total_data[
    (total_data['ADX_14'] > 25) &
    (total_data['Plus_DI'] > total_data['Minus_DI']) &
    (total_data['Supertrend_Direction'] == 1)
]

# Strong downtrends
strong_downtrend = total_data[
    (total_data['ADX_14'] > 25) &
    (total_data['Minus_DI'] > total_data['Plus_DI']) &
    (total_data['Supertrend_Direction'] == -1)
]
```

### Example 3: Feature Selection for ML Model

```python
# Select top momentum and trend features
momentum_features = [
    'RSI_14', 'MACD', 'MACD_hist', 'Stoch_K', 
    'Williams_R', 'CCI_20', 'ROC_10', 'ROC_20'
]

trend_features = [
    'ADX_14', 'Plus_DI', 'Minus_DI', 
    'Aroon_Oscillator', 'Supertrend_Direction'
]

volatility_features = [
    'ATR_14', 'BB_width', 'BB_pct', 'HV_20'
]

volume_features = [
    'OBV', 'MFI_14', 'Volume_ROC_10', 'Price_VWAP_Ratio'
]

all_features = momentum_features + trend_features + volatility_features + volume_features
X = total_data[all_features]
y = total_data['Next_Close']
```

### Example 4: Create Trading Signals

```python
# Golden Cross signal
total_data['Golden_Cross'] = (
    (total_data['MA_Cross_50_200'] > 0) & 
    (total_data['MA_Cross_50_200'].shift(1) <= 0)
).astype(int)

# MACD bullish crossover
total_data['MACD_Buy'] = (
    (total_data['MACD'] > total_data['MACD_signal']) & 
    (total_data['MACD'].shift(1) <= total_data['MACD_signal'].shift(1))
).astype(int)

# Combine signals
total_data['Buy_Signal'] = (
    (total_data['Golden_Cross'] == 1) | 
    (total_data['MACD_Buy'] == 1) |
    (total_data['RSI_Oversold'] == 1)
).astype(int)
```

### Example 5: Correlation Analysis

```python
# Check correlation between indicators
indicators = ['RSI_14', 'MACD', 'ADX_14', 'MFI_14', 'BB_pct']
correlation_matrix = total_data[indicators].corr()

# Plot heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Technical Indicator Correlation Matrix')
plt.show()
```

---

## Best Practices

### 1. **Avoid Overfitting**
- Don't use all 80+ features simultaneously
- Use feature selection techniques (correlation, importance, PCA)
- Cross-validate your models

### 2. **Consider Multi-Collinearity**
- Many indicators are correlated (e.g., different MAs)
- Remove highly correlated features (correlation > 0.9)
- Use domain knowledge to select best representatives

### 3. **Normalization**
- Different indicators have different scales
- Normalize/standardize features before ML modeling
- Consider robust scalers for outliers

### 4. **Lookback Period**
- Some indicators need warm-up period (e.g., SMA_200 needs 200 days)
- First ~200 rows will have NaN values
- Already handled by `dropna()` in the code

### 5. **Regime Changes**
- Market conditions change (bull/bear, volatile/calm)
- Consider splitting data by market regime
- Use regime-specific features or models

### 6. **Walk-Forward Validation**
- Use time-series cross-validation
- Never train on future data (data leakage)
- Respect temporal ordering

---

## References

### Books
- "Technical Analysis of the Financial Markets" by John J. Murphy
- "Evidence-Based Technical Analysis" by David Aronson

### Online Resources
- [Investopedia Technical Indicators](https://www.investopedia.com/terms/t/technicalindicator.asp)
- [TA-Lib Documentation](https://ta-lib.org/)
- [TradingView Indicators](https://www.tradingview.com/scripts/)

---

## Support

For questions or issues with these indicators, please refer to:
- Notebook: `stock_price_prediction_regression.ipynb`
- Sections: 1.5 - 1.9 (Enhanced Technical Indicators)
- Variables: `total_data`, `enhanced_stock_data`

**Last Updated**: November 15, 2025  
**Data Coverage**: 2016-2025 (7+ years)  
**Stocks**: AAPL, MSFT, NFLX, NVDA
