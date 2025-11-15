# Enhanced Stock Analysis with 7 Years Historical Data

## 📊 Overview

This project implements **80+ technical indicators** on 7 years of minute-level historical stock data (2016-2025) for machine learning and quantitative analysis.

### Stocks Analyzed
- **AAPL** - Apple Inc.
- **MSFT** - Microsoft Corporation  
- **NFLX** - Netflix Inc.
- **NVDA** - NVIDIA Corporation

### Data Specifications
- **Time Period**: January 2016 - November 2025 (7+ years)
- **Frequency**: Minute-level → Aggregated to Daily
- **Total Records**: ~8,000+ daily bars (combined)
- **Features**: 80+ technical indicators per stock

---

## 🚀 Quick Start

### Step 1: Open the Notebook
```bash
jupyter notebook stock_price_prediction_regression.ipynb
```

### Step 2: Run Initial Cells
Execute cells in **Sections 1 through 1.9** to:
1. Import libraries
2. Load 7-year historical minute data
3. Aggregate to daily bars
4. Calculate 80+ technical indicators
5. Visualize key indicators

### Step 3: Access the Data
```python
# Combined dataset (all stocks)
total_data              # Main DataFrame

# Individual stock datasets
enhanced_stock_data['AAPL']
enhanced_stock_data['MSFT']
enhanced_stock_data['NFLX']
enhanced_stock_data['NVDA']
```

---

## 📁 Project Structure

```
ml-project/
│
├── stock_price_prediction_regression.ipynb  # Main notebook with indicators
├── TECHNICAL_INDICATORS_GUIDE.md            # Comprehensive indicator documentation
├── INDICATORS_CHEAT_SHEET.md               # Quick reference guide
├── README_INDICATORS.md                    # This file
│
├── AAPL_minute_data_all_20251107(in).csv   # Historical data files
├── MSFT_minute_data_all_20251107(in).csv
├── NFLX_minute_data_all_20251107 1(in).csv
├── NVDA_minute_data_all_20251107(in).csv
│
└── models/                                  # Additional models (optional)
```

---

## 📈 Technical Indicators Implemented

### Momentum Indicators (12)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Williams %R
- Commodity Channel Index (CCI)
- Rate of Change (ROC)

### Trend Indicators (8)
- ADX (Average Directional Index)
- Aroon Indicator
- Supertrend
- Directional Indicators (+DI, -DI)

### Volatility Indicators (10)
- Bollinger Bands (with %B and Width)
- Average True Range (ATR)
- Historical Volatility
- Keltner Channels

### Volume Indicators (9)
- On-Balance Volume (OBV)
- Volume Weighted Average Price (VWAP)
- Money Flow Index (MFI)
- Volume Moving Averages
- Volume Rate of Change

### Moving Averages (11)
- Simple Moving Averages (5, 10, 20, 50, 100, 200)
- Exponential Moving Averages (12, 26, 50)
- Golden/Death Cross indicators

### Pattern Recognition (7)
- Price position relative to MAs
- Higher Highs / Lower Lows
- Daily Range metrics
- Price-to-VWAP ratio

### Time Features (4)
- Day of week
- Month
- Quarter  
- Day of month

### Lagged Features (13)
- Historical prices (1, 2, 3, 5, 10, 20 days)
- Historical returns (1, 5, 10, 20 days)
- Historical volume (1, 5, 10 days)

### Target Variables (2)
- Next day's closing price
- Next day's return percentage

**Total: 80+ Features**

---

## 📚 Documentation

### 1. TECHNICAL_INDICATORS_GUIDE.md
**Comprehensive guide** covering:
- Detailed explanation of each indicator
- Mathematical formulas
- Interpretation guidelines
- Trading signals
- Usage examples
- Best practices

### 2. INDICATORS_CHEAT_SHEET.md
**Quick reference** for:
- Feature list by category
- Common use cases
- Signal combinations
- Feature selection tips
- Code snippets

### 3. Notebook Sections 1.5-1.9
**Implementation details**:
- Data loading functions
- Indicator calculation functions
- Feature engineering pipeline
- Data quality checks
- Visualizations

---

## 💻 Usage Examples

### Example 1: Load and Explore Data
```python
# Run notebook sections 1.5-1.9 first, then:

# Check available stocks
print(enhanced_stock_data.keys())

# Get AAPL data
aapl = enhanced_stock_data['AAPL']

# View first rows
print(aapl.head())

# Summary statistics
print(aapl[['Close', 'RSI_14', 'MACD', 'ADX_14']].describe())
```

### Example 2: Identify Trading Signals
```python
# Find oversold conditions
oversold = total_data[
    (total_data['RSI_14'] < 30) & 
    (total_data['MFI_14'] < 20) &
    (total_data['BB_pct'] < 0.2)
]

# Find strong uptrends
uptrends = total_data[
    (total_data['ADX_14'] > 25) &
    (total_data['Supertrend_Direction'] == 1) &
    (total_data['Close'] > total_data['SMA_50'])
]
```

### Example 3: Prepare Data for ML
```python
# Select features
features = [
    'RSI_14', 'MACD', 'MACD_hist',
    'BB_pct', 'ATR_14',
    'ADX_14', 'MFI_14',
    'Volume_ROC_10', 'OBV',
    'Close_lag_1', 'Close_lag_5'
]

# Create feature matrix and target
X = total_data[features]
y = total_data['Next_Close']

# Remove any remaining NaN
mask = ~(X.isnull().any(axis=1) | y.isnull())
X = X[mask]
y = y[mask]

# Train/test split (maintain temporal order)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
```

### Example 4: Visualize Indicators
```python
import matplotlib.pyplot as plt

# Get recent data for one stock
recent = enhanced_stock_data['AAPL'].tail(252)  # Last year

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Price with Moving Averages
axes[0].plot(recent.index, recent['Close'], label='Close')
axes[0].plot(recent.index, recent['SMA_20'], label='SMA 20')
axes[0].plot(recent.index, recent['SMA_50'], label='SMA 50')
axes[0].legend()
axes[0].set_title('AAPL Price with Moving Averages')

# RSI
axes[1].plot(recent.index, recent['RSI_14'])
axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5)
axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5)
axes[1].set_title('RSI (14)')

# MACD
axes[2].plot(recent.index, recent['MACD'], label='MACD')
axes[2].plot(recent.index, recent['MACD_signal'], label='Signal')
axes[2].bar(recent.index, recent['MACD_hist'], alpha=0.3)
axes[2].legend()
axes[2].set_title('MACD')

plt.tight_layout()
plt.show()
```

---

## 🎯 Use Cases

### 1. Machine Learning Models
- **Regression**: Predict next day's closing price
- **Classification**: Predict price direction (up/down)
- **Time Series**: LSTM/GRU models with sequential data

### 2. Algorithmic Trading
- **Trend Following**: Use MA crossovers, ADX, Supertrend
- **Mean Reversion**: Use RSI, Bollinger Bands, Stochastic
- **Momentum Trading**: Use MACD, ROC, MFI

### 3. Technical Analysis
- **Backtesting**: Test strategies on 7 years of data
- **Pattern Recognition**: Identify chart patterns automatically
- **Risk Management**: Use ATR for position sizing

### 4. Research & Education
- **Indicator Comparison**: Evaluate effectiveness of different indicators
- **Feature Importance**: Determine which indicators matter most
- **Market Regime Analysis**: Study bull/bear market characteristics

---

## 🔬 Feature Engineering Notes

### Data Preprocessing
1. **Loading**: Minute data loaded from CSV files
2. **Aggregation**: Converted to daily OHLCV bars
3. **Indicators**: 80+ technical indicators calculated
4. **Cleaning**: NaN rows removed (from indicator warm-up periods)

### Important Considerations
- **Lookback Period**: First ~200 rows may have NaN (for SMA_200)
- **Data Leakage**: Never use future data in features
- **Temporal Order**: Always maintain chronological sequence
- **Normalization**: Scale features before ML modeling

### Variables Created
```python
minute_data           # Dict: Original minute-level data
daily_stock_data      # Dict: Daily OHLCV (before indicators)
enhanced_stock_data   # Dict: Daily with all 80+ indicators
total_data            # DataFrame: Combined all stocks
```

---

## 📊 Data Statistics

### Coverage
- **Start Date**: January 4, 2016
- **End Date**: November 7, 2025
- **Total Days**: ~2,500 trading days per stock
- **Total Records**: ~10,000 combined rows

### Data Quality
- ✅ No missing values (after preprocessing)
- ✅ Consistent timestamps
- ✅ Clean OHLCV data
- ✅ All indicators validated

---

## 🛠️ Technical Requirements

### Python Libraries
```python
pandas>=1.3.0          # Data manipulation
numpy>=1.21.0          # Numerical operations
matplotlib>=3.4.0      # Visualization
seaborn>=0.11.0        # Statistical plotting
scikit-learn>=0.24.0   # Machine learning
datetime               # Date handling
warnings               # Warning suppression
```

### Hardware Requirements
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 500MB for data files
- **CPU**: Any modern processor

---

## 🚨 Important Notes

### Data Integrity
- Data is from **minute-level historical records**
- Aggregated to daily bars for analysis
- All calculations performed on clean data
- No forward-looking bias in features

### Limitations
- **Survivorship Bias**: Only includes current stocks
- **Corporate Actions**: May not account for all splits/dividends
- **Market Hours**: Only regular trading hours included
- **Data Gaps**: Holidays and non-trading days removed

### Best Practices
1. **Always validate** signals with multiple indicators
2. **Use proper position sizing** based on volatility (ATR)
3. **Implement risk management** (stop-losses, take-profits)
4. **Backtest thoroughly** before live trading
5. **Consider transaction costs** in strategy evaluation

---

## 📖 Learning Path

### Beginner
1. Start with `INDICATORS_CHEAT_SHEET.md`
2. Run notebook sections 1.5-1.9
3. Explore `total_data` with basic filtering
4. Try simple moving average strategies

### Intermediate
1. Read `TECHNICAL_INDICATORS_GUIDE.md`
2. Combine multiple indicators for signals
3. Build simple classification models
4. Backtest basic strategies

### Advanced
1. Implement custom indicators
2. Build ensemble ML models
3. Develop automated trading systems
4. Optimize parameters and portfolio

---

## 🤝 Contributing

### To Add New Indicators
1. Add calculation function in Section 1.7
2. Integrate into `create_enhanced_features()`
3. Document in `TECHNICAL_INDICATORS_GUIDE.md`
4. Add to cheat sheet

### To Add New Stocks
1. Place minute data CSV in project root
2. Add to `data_files` dictionary
3. Run data loading and processing cells
4. New stock will be included in `total_data`

---

## 📞 Support & Questions

### Common Issues

**Q: "I see NaN values in my data"**  
A: Run the preprocessing cells again. NaN values should be removed automatically.

**Q: "How do I add more stocks?"**  
A: Add CSV files and update the `data_files` dictionary in Section 1.5.

**Q: "Which indicators should I use for prediction?"**  
A: Start with RSI, MACD, ADX, ATR, and volume indicators. See cheat sheet for combinations.

**Q: "The notebook is slow"**  
A: Processing 7 years of data takes time. Consider reducing date range or using fewer features.

---

## 📜 License

See `LICENSE` file for details.

---

## 🎓 References

### Books
- "Technical Analysis of the Financial Markets" - John Murphy
- "Evidence-Based Technical Analysis" - David Aronson
- "Quantitative Trading" - Ernest Chan

### Online Resources
- [Investopedia Technical Analysis](https://www.investopedia.com/technical-analysis-4689657)
- [TA-Lib Documentation](https://ta-lib.org/)
- [Scikit-learn Time Series](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

---

## ✅ Checklist for Colleagues

- [ ] Read this README
- [ ] Review `INDICATORS_CHEAT_SHEET.md` for quick reference
- [ ] Open `stock_price_prediction_regression.ipynb`
- [ ] Run cells in Sections 1.5 through 1.9
- [ ] Verify `total_data` and `enhanced_stock_data` are created
- [ ] Check `TECHNICAL_INDICATORS_GUIDE.md` for indicator details
- [ ] Explore data with provided examples
- [ ] Build your first model/strategy!

---

**Last Updated**: November 15, 2025  
**Version**: 1.0  
**Maintainer**: Your Team

Happy Trading! 📈🚀
