# ALGO2 — ML-Powered Crypto Futures Trading Bot

> A production-grade algorithmic trading system combining supervised machine learning, reinforcement learning, and multi-timeframe regime filtering to trade BTC and ETH perpetual futures on Binance.

---

## Live Status

![Status](https://img.shields.io/badge/status-paper%20trading-yellow)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Currently running:** Paper trading on Binance Futures Demo (24/7 VPS deployment)
**Assets:** BTC/USDT and ETH/USDT perpetual futures
**Signal frequency:** Every hour at candle close

---

## Performance — Backtest Results

Test period: **November 2025 → March 2026** (bear market, BTC -20%, ETH -26%)

| Metric | BTC | ETH |
|---|---|---|
| Bot return | **+139.7%** | **+295.8%** |
| Buy & hold | -19.8% | -25.9% |
| Alpha | +159.5% | +321.7% |
| Sharpe ratio | 56.2 | 61.4 |
| Max drawdown | -4.75% | -6.79% |
| Win rate | 74.0% | 73.6% |
| Total trades | 123 | 129 |

> **Note:** Backtest results are on a held-out test set never seen during training. Paper trading is the current validation phase before live deployment.

---

## How It Works

The system uses a three-layer decision pipeline:

```
Live Data (Binance API)
        ↓
Feature Engineering (31 signals across 1h / 4h / 1d)
        ↓
Supervised Model → P(sell)  P(hold)  P(buy)
        ↓
Regime Filter (macro trend gate)
        ↓
RL Agent (PPO) → direction + position size
        ↓
Risk Gate (threshold + stop-loss + drawdown)
        ↓
Order Execution (Binance Futures)
        ↓
Trade Logging + Monitoring
```

### Layer 1 — Feature Engineering (31 features)

| Category | Features |
|---|---|
| Returns | 1h, 3h, 6h, 12h, 24h percentage change |
| Trend | EMA crossovers (9/21/50), MACD, price vs EMA |
| Momentum | RSI (7/14/21), Rate of Change |
| Volatility | Bollinger Bands, ATR (normalized) |
| Volume | OBV z-score, volume ratio |
| Multi-timeframe | 4h RSI, 4h trend, daily bull/bear regime |

### Layer 2 — Supervised Signal Model

- **BTC:** LightGBM classifier — ROC-AUC **0.721** on holdout test
- **ETH:** XGBoost classifier — ROC-AUC **0.722** on holdout test
- **Output:** Three probabilities per candle — P(sell), P(hold), P(buy)
- **Target:** 3-class classification (BUY / HOLD / SELL) on 6-hour horizon
- **Training:** 17,400+ hourly candles, strict time-based train/val/test split

### Layer 3 — Regime Filter

Prevents trading against the macro trend — the single highest-impact component:

```python
# Only long in confirmed bullish conditions
if prob_buy >= 0.55 and (trend_4h == 1 or bull_regime == 1):
    enter_long()

# Only short in confirmed bearish conditions
if prob_sell >= 0.65 and (trend_4h == 0 and bull_regime == 0):
    enter_short()
```

Without this filter: **-47% backtest**. With it: **+139% backtest**.

### Layer 4 — RL Position Sizing (PPO Agent)

- **Framework:** Stable-Baselines3, Proximal Policy Optimization
- **Observation:** 38-dimensional vector (features + probabilities + position state)
- **Actions:** 8 discrete — Hold, Long/Short at 25%/50%/75%, Close
- **Training:** 500,000 environment steps with randomized episode starts
- **Key learned behavior:** Sizes positions dynamically based on signal confidence — not a fixed percentage

### Layer 5 — Risk Management

| Control | BTC | ETH |
|---|---|---|
| Stop loss | 2% | 3% |
| Take profit | 4% | 6% |
| Max hold | 6 hours | 6 hours |
| Max drawdown limit | 15% portfolio | 15% portfolio |
| Min buy confidence | 0.55 | 0.50 |
| Min sell confidence | 0.65 | 0.60 |

---

## Project Structure

```
ALGO2/
├── src/
│   ├── paper_trading.py      # Main loop — runs 24/7, hourly signals
│   ├── features.py           # Feature engineering pipeline
│   ├── signals.py            # Supervised model + RL agent inference
│   └── executor.py           # Binance Futures order placement
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_backtesting.ipynb
│   └── 05_rl_agent.ipynb
├── data/
│   ├── models/               # Saved XGBoost / LightGBM models
│   ├── rl/                   # Saved PPO agents
│   ├── processed/            # Feature matrices + scalers
│   └── raw/                  # Raw OHLCV CSVs
├── logs/
│   ├── paper_trading.log     # Hourly cycle logs
│   └── trades.csv            # Every trade entry/exit
├── .env                      # API keys (never commit)
├── .gitignore
└── README.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Data ingestion | `ccxt`, `yfinance` |
| Feature engineering | `pandas`, `numpy` |
| Signal model | `xgboost`, `lightgbm`, `scikit-learn` |
| RL agent | `stable-baselines3` (PPO) |
| Backtesting | Custom engine with realistic fees + slippage |
| Execution | `ccxt` → Binance Futures API |
| Scheduling | `schedule` (hourly) |
| Infrastructure | Ubuntu 22.04 VPS, `systemd` |

---

## Development Phases

| Phase | Description | Status |
|---|---|---|
| 1 — Data | OHLCV collection, multi-timeframe pipeline | ✅ Complete |
| 2 — Features | 31 engineered features, regime detection | ✅ Complete |
| 3 — Model | XGBoost/LightGBM 3-class classifier | ✅ Complete |
| 4 — Backtest | Walk-forward validation, fee simulation | ✅ Complete |
| 5 — RL Agent | PPO position sizing, 500k step training | ✅ Complete |
| 6 — Paper Trading | Live signals on Binance Demo, 24/7 VPS | 🟡 Running |
| 7 — Live Trading | Real capital, small size, scale on results | ⏳ Pending |

---

## Setup

### Prerequisites

- Python 3.11
- Binance Futures Demo account (free at [testnet.binancefuture.com](https://testnet.binancefuture.com))
- VPS with Ubuntu 22.04 (optional, for 24/7 running)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/algo2.git
cd algo2

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
BINANCE_API_KEY=your_demo_api_key
BINANCE_API_SECRET=your_demo_api_secret
TESTNET=true
```

> Get free demo API keys from [testnet.binancefuture.com](https://testnet.binancefuture.com) — no real money required.

### Run the bot

```bash
python src/paper_trading.py
```

The bot will:
1. Load all models
2. Connect to Binance Demo
3. Run one cycle immediately
4. Schedule hourly cycles at `:01`
5. Log everything to `logs/`

---

## Running the Notebooks

To reproduce the full training pipeline:

```bash
# Install Jupyter
pip install jupyterlab

# Launch
jupyter lab
```

Run notebooks in order: `01` → `02` → `03` → `04` → `05`

Each notebook is self-contained with explanations for every cell.

---

## Monitoring

```bash
# Watch live logs
tail -f logs/paper_trading.log

# Check trade history
cat logs/trades.csv

# Run dashboard
python logs/dashboard.py
```

Sample log output when a signal fires:

```
2026-03-27 08:01:23 | INFO | Trading cycle started at 2026-03-27 08:01
2026-03-27 08:01:26 | INFO |   Signal: SHORT  size=75% | P(sell)=0.681 P(hold)=0.214 P(buy)=0.105
2026-03-27 08:01:26 | INFO |   SHORT ETH $3,750.00 (75%) at $2,134.50
```

---

## Deploying to VPS (24/7)

```bash
# Upload project
scp -r ./algo2 root@YOUR_VPS_IP:/root/algo2

# SSH into VPS
ssh root@YOUR_VPS_IP

# Setup environment
cd /root/algo2
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create systemd service for auto-restart
nano /etc/systemd/system/tradingbot.service
systemctl enable tradingbot
systemctl start tradingbot

# Monitor
tail -f /root/algo2/logs/paper_trading.log
```

Full VPS deployment guide in [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Key Design Decisions

**Why no LLM?**
LLMs have no edge on OHLCV data. A properly trained gradient boosting model with AUC 0.72 on financial time series outperforms any LLM-based signal generation. LLMs are useful for sentiment analysis of news text — a planned future addition via FinBERT.

**Why reinforcement learning for sizing?**
Fixed position sizing leaves money on the table. The RL agent learned that in downtrends, short positions deserve higher conviction sizing (75%) while uncertain signals warrant smaller exposure (25%). This adaptive behavior cannot be captured by rules.

**Why the regime filter matters so much?**
Without it: -47% backtest. With it: +139%. The model generates correct directional signals but acts on them at the wrong time without macro context. The regime filter ensures the bot only longs in bullish conditions and only shorts in bearish conditions — eliminating the single largest source of losses.

**Why futures over spot?**
Futures allow shorting — profiting from both up and down moves. In the test period (bear market), short PnL was $9,712 vs long PnL of $4,261 for BTC. A spot-only bot would have missed more than half its returns.

---

## Risk Disclaimer

This software is for educational and research purposes. Past backtest performance does not guarantee future results. Cryptocurrency trading involves substantial risk of loss. Never trade with money you cannot afford to lose. Always start with paper trading and validate thoroughly before deploying real capital.

The authors are not financial advisors. Nothing in this repository constitutes financial advice.

---

## Roadmap

- [ ] FinBERT news sentiment as additional features (Phase 2.5)
- [ ] Walk-forward retraining pipeline (monthly model updates)
- [ ] Streamlit monitoring dashboard
- [ ] Multi-asset expansion (SOL, BNB)
- [ ] Live trading deployment (after 4 weeks profitable paper trading)
- [ ] Portfolio-level risk management across assets

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

Built from scratch across 7 development phases — data pipeline, feature engineering, supervised ML, backtesting engine, reinforcement learning, and live deployment.

