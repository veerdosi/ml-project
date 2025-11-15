# Technical Indicators Quick Reference Cheat Sheet

## 🚀 Quick Start

```python
# Load the notebook and run cells 1-9 (Sections 1.5 - 1.9)
# Main variables created:
total_data              # Combined dataset (all stocks)
enhanced_stock_data     # Dictionary: {'AAPL': df, 'MSFT': df, ...}
```

---

## 📊 Feature Categories (80+ indicators)

### 1️⃣ Moving Averages (11 features)
```
SMA: 5, 10, 20, 50, 100, 200
EMA: 12, 26, 50
Crossovers: MA_Cross_50_200, MA_Cross_20_50
```

### 2️⃣ Momentum (12 features)
```
RSI_14, RSI_Overbought, RSI_Oversold
MACD, MACD_signal, MACD_hist
Stoch_K, Stoch_D
Williams_R, CCI_20
ROC_10, ROC_20
```

### 3️⃣ Volatility (10 features)
```
BB_upper, BB_middle, BB_lower, BB_width, BB_pct
ATR_14
HV_20, HV_50
KC_upper, KC_lower
```

### 4️⃣ Volume (9 features)
```
Volume_MA_20, Volume_MA_50
Volume_ROC_10, Volume_ROC_20
OBV, OBV_MA_20
VWAP, Price_VWAP_Ratio
MFI_14
```

### 5️⃣ Trend (8 features)
```
ADX_14, Plus_DI, Minus_DI
Aroon_Up, Aroon_Down, Aroon_Oscillator
Supertrend, Supertrend_Direction
```

### 6️⃣ Patterns (7 features)
```
Price_to_SMA20, Price_to_SMA50, Price_to_SMA200
Higher_High, Lower_Low
Daily_Range, High_Low_Ratio
```

### 7️⃣ Time (4 features)
```
Day_of_Week, Month, Quarter, Day_of_Month
```

### 8️⃣ Lagged (13 features)
```
Close_lag: 1, 2, 3, 5, 10, 20
Return_lag: 1, 5, 10, 20
Volume_lag: 1, 5, 10
```

### 9️⃣ Target (2 features)
```
Next_Close, Target_Return
```

---

## 🎯 Common Use Cases

### Overbought/Oversold Detection
```python
# Overbought
overbought = total_data[
    (total_data['RSI_14'] > 70) & 
    (total_data['BB_pct'] > 0.8)
]

# Oversold
oversold = total_data[
    (total_data['RSI_14'] < 30) & 
    (total_data['BB_pct'] < 0.2)
]
```

### Trend Identification
```python
# Strong Uptrend
uptrend = total_data[
    (total_data['ADX_14'] > 25) &
    (total_data['Close'] > total_data['SMA_50']) &
    (total_data['Supertrend_Direction'] == 1)
]

# Strong Downtrend
downtrend = total_data[
    (total_data['ADX_14'] > 25) &
    (total_data['Close'] < total_data['SMA_50']) &
    (total_data['Supertrend_Direction'] == -1)
]
```

### Volatility Analysis
```python
# High Volatility
high_vol = total_data[total_data['ATR_14'] > total_data['ATR_14'].quantile(0.75)]

# Low Volatility (potential breakout)
low_vol = total_data[total_data['BB_width'] < total_data['BB_width'].quantile(0.25)]
```

### Volume Confirmation
```python
# Strong buying pressure
buying = total_data[
    (total_data['OBV'] > total_data['OBV_MA_20']) &
    (total_data['MFI_14'] > 50)
]

# Strong selling pressure
selling = total_data[
    (total_data['OBV'] < total_data['OBV_MA_20']) &
    (total_data['MFI_14'] < 50)
]
```

---

## 🔔 Key Signal Combinations

### Golden Cross (Bullish)
```python
golden_cross = (
    (total_data['SMA_50'] > total_data['SMA_200']) &
    (total_data['SMA_50'].shift(1) <= total_data['SMA_200'].shift(1))
)
```

### Death Cross (Bearish)
```python
death_cross = (
    (total_data['SMA_50'] < total_data['SMA_200']) &
    (total_data['SMA_50'].shift(1) >= total_data['SMA_200'].shift(1))
)
```

### MACD Bullish Crossover
```python
macd_buy = (
    (total_data['MACD'] > total_data['MACD_signal']) &
    (total_data['MACD'].shift(1) <= total_data['MACD_signal'].shift(1))
)
```

### RSI Divergence (Manual Check)
```python
# Price makes new high but RSI doesn't = Bearish divergence
# Price makes new low but RSI doesn't = Bullish divergence
```

---

## 📈 Indicator Interpretation Guide

| Indicator | Bullish | Bearish | Neutral |
|-----------|---------|---------|---------|
| **RSI** | < 30 (oversold) | > 70 (overbought) | 40-60 |
| **MACD** | > 0 and rising | < 0 and falling | Near 0 |
| **ADX** | > 25 (any direction) | > 25 (any direction) | < 25 |
| **Stochastic** | < 20 (oversold) | > 80 (overbought) | 20-80 |
| **BB %B** | < 0 (below lower) | > 1 (above upper) | 0.2-0.8 |
| **MFI** | < 20 | > 80 | 40-60 |
| **Aroon** | Up > 70, Down < 30 | Down > 70, Up < 30 | Both 30-70 |

---

## 🛠️ Feature Selection Tips

### For Trend Following Models
```python
features = [
    'SMA_50', 'SMA_200', 'MA_Cross_50_200',
    'ADX_14', 'Aroon_Oscillator',
    'Supertrend_Direction'
]
```

### For Mean Reversion Models
```python
features = [
    'RSI_14', 'BB_pct', 'MFI_14',
    'Stoch_K', 'Williams_R',
    'Price_to_SMA20'
]
```

### For Momentum Models
```python
features = [
    'MACD', 'MACD_hist', 'ROC_10', 'ROC_20',
    'RSI_14', 'CCI_20',
    'Return_lag_1', 'Return_lag_5'
]
```

### For Volatility Breakout Models
```python
features = [
    'ATR_14', 'BB_width', 'HV_20',
    'Volume_ROC_10', 'Daily_Range',
    'Supertrend_Direction'
]
```

---

## 💡 Pro Tips

### 1. Combine Multiple Timeframes
```python
# Short-term (5-20), Mid-term (50), Long-term (200)
signal = (
    (total_data['SMA_5'] > total_data['SMA_20']) &   # Short uptrend
    (total_data['SMA_50'] > total_data['SMA_200'])   # Long uptrend
)
```

### 2. Use Volume for Confirmation
```python
# Price move with volume confirmation
valid_breakout = (
    (total_data['Close'] > total_data['BB_upper']) &
    (total_data['Volume'] > total_data['Volume_MA_20'] * 1.5)
)
```

### 3. Wait for Trend Strength
```python
# Only trade when trend is strong
strong_signal = (
    (total_data['RSI_14'] > 70) &      # Overbought
    (total_data['ADX_14'] > 25)        # But strong trend
)
```

### 4. Multiple Indicator Confluence
```python
# All indicators agree
confluence = (
    (total_data['RSI_14'] < 30) &                    # Oversold
    (total_data['BB_pct'] < 0) &                     # Below lower BB
    (total_data['MFI_14'] < 20) &                    # Low money flow
    (total_data['Price_to_SMA20'] < -0.05)           # 5% below MA
)
```

---

## ⚠️ Common Pitfalls

1. **Don't use all features** → Feature selection is crucial
2. **Avoid look-ahead bias** → Never use future data
3. **Check for multicollinearity** → Remove highly correlated features
4. **Normalize features** → Scale before ML modeling
5. **Respect temporal order** → Use TimeSeriesSplit for CV
6. **Account for transaction costs** → Real trading has fees
7. **Test across market regimes** → Bull, bear, sideways markets

---

## 📚 Data Access Patterns

### Get specific stock
```python
aapl = enhanced_stock_data['AAPL']
```

### Filter by date
```python
recent = total_data.loc['2024-01-01':]
```

### Get specific features
```python
X = total_data[['RSI_14', 'MACD', 'ADX_14']]
y = total_data['Next_Close']
```

### Filter by ticker
```python
aapl_only = total_data[total_data['Ticker'] == 'AAPL']
```

---

## 🔗 Quick Links

- **Full Documentation**: `TECHNICAL_INDICATORS_GUIDE.md`
- **Notebook**: `stock_price_prediction_regression.ipynb` (Sections 1.5-1.9)
- **Data Files**: `*_minute_data_all_20251107(in).csv`

---

## 📞 Need Help?

```python
# Show all columns
print(total_data.columns.tolist())

# Check data types
print(total_data.dtypes)

# Summary statistics
print(total_data.describe())

# Check for missing values
print(total_data.isnull().sum())
```

---

**Created**: November 15, 2025  
**Data**: 2016-2025 (7 years)  
**Stocks**: AAPL, MSFT, NFLX, NVDA  
**Total Indicators**: 80+
