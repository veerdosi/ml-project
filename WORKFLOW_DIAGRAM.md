# Technical Indicators Workflow

## 📊 Data Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: CSV FILES (7 Years)                   │
│  AAPL, MSFT, NFLX, NVDA - Minute-level data (2016-2025)       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               SECTION 1.5: Load Minute Data                     │
│  • Parse timestamps                                             │
│  • Create indexed DataFrames                                    │
│  • Validate data coverage                                       │
│  Output: minute_data (dict)                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│          SECTION 1.6: Aggregate to Daily OHLCV                  │
│  • Resample to daily frequency                                  │
│  • Open (first), High (max), Low (min)                         │
│  • Close (last), Volume (sum)                                   │
│  Output: daily_stock_data (dict)                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│     SECTION 1.7: Calculate Technical Indicators (80+)           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. Moving Averages (11)                                  │  │
│  │    SMA: 5,10,20,50,100,200 | EMA: 12,26,50             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 2. Momentum Indicators (12)                              │  │
│  │    RSI, MACD, Stochastic, Williams %R, CCI, ROC         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 3. Volatility Indicators (10)                            │  │
│  │    Bollinger Bands, ATR, Historical Vol, Keltner        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 4. Volume Indicators (9)                                 │  │
│  │    OBV, VWAP, MFI, Volume MA, Volume ROC               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 5. Trend Indicators (8)                                  │  │
│  │    ADX, Aroon, Supertrend, Directional Indicators       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 6. Pattern Features (7)                                  │  │
│  │    Price/MA ratios, Higher/Lower patterns, Ranges       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 7. Time Features (4)                                     │  │
│  │    Day of Week, Month, Quarter, Day of Month            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 8. Lagged Features (13)                                  │  │
│  │    Price lags, Return lags, Volume lags                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 9. Target Variables (2)                                  │  │
│  │    Next_Close, Target_Return                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Output: enhanced_stock_data (dict)                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           SECTION 1.8: Data Quality & Validation                │
│  • Check for missing values                                     │
│  • Display summary statistics                                   │
│  • Visualize key indicators                                     │
│  • Validate feature calculations                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              SECTION 1.9: Optional Export                       │
│  • Save to CSV (optional)                                       │
│  • Preserve for future use                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT: READY TO USE                         │
│                                                                  │
│  total_data (DataFrame)                                         │
│  • ~10,000 rows (all stocks combined)                          │
│  • 80+ technical indicator features                            │
│  • Clean, validated data                                        │
│  • Ready for ML/Trading                                         │
│                                                                  │
│  enhanced_stock_data (Dictionary)                               │
│  • Individual stock DataFrames                                  │
│  • Keys: 'AAPL', 'MSFT', 'NFLX', 'NVDA'                       │
│  • Each with 80+ features                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Usage Patterns

### Pattern 1: Machine Learning
```
total_data → Feature Selection → Train/Test Split → Model Training → Evaluation
```

### Pattern 2: Trading Strategy
```
enhanced_stock_data → Signal Generation → Backtesting → Performance Analysis
```

### Pattern 3: Technical Analysis
```
enhanced_stock_data → Indicator Analysis → Pattern Recognition → Trade Decisions
```

### Pattern 4: Research
```
total_data → Statistical Analysis → Feature Importance → Insights
```

---

## 📊 Feature Categories Breakdown

```
MOVING AVERAGES (11)
├── Simple MA: SMA_5, SMA_10, SMA_20, SMA_50, SMA_100, SMA_200
├── Exponential MA: EMA_12, EMA_26, EMA_50
└── Crossovers: MA_Cross_50_200, MA_Cross_20_50

MOMENTUM (12)
├── RSI: RSI_14, RSI_Overbought, RSI_Oversold
├── MACD: MACD, MACD_signal, MACD_hist
├── Stochastic: Stoch_K, Stoch_D
├── Other: Williams_R, CCI_20, ROC_10, ROC_20

VOLATILITY (10)
├── Bollinger Bands: BB_upper, BB_middle, BB_lower, BB_width, BB_pct
├── ATR: ATR_14
├── Historical Vol: HV_20, HV_50
└── Keltner: KC_upper, KC_lower

VOLUME (9)
├── Volume MA: Volume_MA_20, Volume_MA_50
├── Volume ROC: Volume_ROC_10, Volume_ROC_20
├── OBV: OBV, OBV_MA_20
├── VWAP: VWAP, Price_VWAP_Ratio
└── MFI: MFI_14

TREND (8)
├── ADX System: ADX_14, Plus_DI, Minus_DI
├── Aroon: Aroon_Up, Aroon_Down, Aroon_Oscillator
└── Supertrend: Supertrend, Supertrend_Direction

PATTERNS (7)
├── Price Position: Price_to_SMA20, Price_to_SMA50, Price_to_SMA200
├── Price Action: Higher_High, Lower_Low
└── Ranges: Daily_Range, High_Low_Ratio

TIME (4)
└── Calendar: Day_of_Week, Month, Quarter, Day_of_Month

LAGGED (13)
├── Price Lags: Close_lag_1, Close_lag_2, Close_lag_3, Close_lag_5, Close_lag_10, Close_lag_20
├── Return Lags: Return_lag_1, Return_lag_5, Return_lag_10, Return_lag_20
└── Volume Lags: Volume_lag_1, Volume_lag_5, Volume_lag_10

TARGET (2)
└── Prediction: Next_Close, Target_Return
```

---

## 🔄 Data Flow Example

```python
# Step 1: Load Data
minute_data = load_minute_data('AAPL.csv', 'AAPL')
# Result: ~500K rows of minute data

# Step 2: Aggregate
daily_data = aggregate_to_daily(minute_data)
# Result: ~2,500 rows of daily data

# Step 3: Create Features
enhanced_data = create_enhanced_features(daily_data, 'AAPL')
# Result: ~2,500 rows × 80+ columns

# Step 4: Use Data
X = enhanced_data[feature_list]
y = enhanced_data['Next_Close']
# Ready for modeling!
```

---

## 📈 Indicator Categories by Purpose

### For Trend Following:
```
✓ Moving Averages (all)
✓ ADX System
✓ Supertrend
✓ Aroon
```

### For Mean Reversion:
```
✓ RSI
✓ Bollinger Bands
✓ Stochastic
✓ Williams %R
```

### For Momentum Trading:
```
✓ MACD
✓ ROC
✓ CCI
✓ MFI
```

### For Volatility Strategies:
```
✓ ATR
✓ Bollinger Bands Width
✓ Historical Volatility
✓ Keltner Channels
```

---

## 🎓 Learning Path

```
BEGINNER
   │
   ├─→ Read INDICATORS_CHEAT_SHEET.md
   ├─→ Run notebook sections 1.5-1.9
   ├─→ Explore total_data
   └─→ Try simple MA strategies
   
INTERMEDIATE
   │
   ├─→ Read TECHNICAL_INDICATORS_GUIDE.md
   ├─→ Combine multiple indicators
   ├─→ Build classification models
   └─→ Backtest strategies
   
ADVANCED
   │
   ├─→ Create custom indicators
   ├─→ Build ensemble models
   ├─→ Optimize parameters
   └─→ Deploy trading systems
```

---

## 📚 Documentation Structure

```
DOCUMENTATION
│
├── IMPLEMENTATION_SUMMARY.md
│   └── What was created and why
│
├── README_INDICATORS.md
│   ├── Quick start
│   ├── Project structure
│   ├── Usage examples
│   └── Troubleshooting
│
├── TECHNICAL_INDICATORS_GUIDE.md
│   ├── Detailed indicator explanations
│   ├── Mathematical formulas
│   ├── Interpretation guidelines
│   └── Trading signals
│
├── INDICATORS_CHEAT_SHEET.md
│   ├── Quick reference tables
│   ├── Common patterns
│   ├── Code snippets
│   └── Pro tips
│
└── WORKFLOW_DIAGRAM.md (this file)
    ├── Visual workflow
    ├── Data flow
    └── Feature breakdown
```

---

## 🚀 Quick Commands

### View Data Structure
```python
# See all columns
print(total_data.columns.tolist())

# Check data shape
print(total_data.shape)

# Summary statistics
print(total_data.describe())
```

### Filter Data
```python
# By ticker
aapl = total_data[total_data['Ticker'] == 'AAPL']

# By date
recent = total_data.loc['2024-01-01':]

# By condition
overbought = total_data[total_data['RSI_14'] > 70]
```

### Feature Selection
```python
# Momentum features
momentum = ['RSI_14', 'MACD', 'Stoch_K', 'ROC_10']

# Trend features
trend = ['ADX_14', 'Supertrend_Direction', 'MA_Cross_50_200']

# Combine
features = momentum + trend
X = total_data[features]
```

---

## ✅ Validation Checklist

Before using the data, verify:

- [ ] All CSV files loaded successfully
- [ ] Date ranges cover 2016-2025
- [ ] No missing values in final dataset
- [ ] All 80+ features calculated
- [ ] Indicator values in expected ranges (e.g., RSI 0-100)
- [ ] total_data variable exists
- [ ] enhanced_stock_data variable exists

---

**Last Updated**: November 15, 2025  
**Purpose**: Visual guide to technical indicators implementation  
**Audience**: Team members and collaborators
