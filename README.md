# Deep Reinforcement Learning Portfolio Management

Production-ready project for multi-asset portfolio management with a PPO-style Actor-Critic architecture in PyTorch. The system downloads real market data from Yahoo Finance, engineers technical indicators, trains a portfolio policy over historical data, and compares the learned strategy against strong baselines.

## 1. Project Overview

This repository builds a full DRL workflow for portfolio allocation:

- Multi-asset market simulation from historical OHLCV data
- Technical feature engineering: returns, RSI, MACD, Bollinger Bands, moving averages, volatility, momentum
- Selectable encoder backbone:
  - `LSTM`
  - `CNN1D`
  - `Transformer Encoder`
- PPO-style clipped Actor-Critic for stable policy optimization
- Reward with transaction cost and volatility penalty
- Full evaluation and benchmark comparison

The actor outputs target portfolio weights over all risky assets plus a cash bucket, and the critic estimates the state value.

## 2. Model Architecture

### Input State

At each timestep, the state contains:

- Rolling lookback window of market features for all assets
- Current portfolio weights
- Cash ratio

Feature set is configurable in `configs/config.yaml`.

### Encoder

Choose via:

```yaml
model:
  encoder_type: "lstm"        # or cnn / transformer
```

### Actor

- Shared latent representation
- Dirichlet policy over portfolio weights
- Sampled weights always sum to 1
- Deterministic inference uses the mean of the Dirichlet distribution

### Critic

- Scalar state-value head

### Reward

Default reward:

```text
R_t = log(1 + portfolio_return - kappa * transaction_cost) - lambda * portfolio_variance
```

Optional Sharpe-style approximation:

```yaml
environment:
  reward_mode: "sharpe"
```

## 3. Folder Structure

```text
project/
│── configs/
│   └── config.yaml
│── data/
│   ├── raw/
│   └── processed/
│── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── env/
│   │   └── portfolio_env.py
│   ├── models/
│   │   ├── encoders.py
│   │   ├── actor.py
│   │   ├── critic.py
│   │   └── actor_critic.py
│   ├── agents/
│   │   └── trainer.py
│   ├── baselines/
│   │   ├── buy_hold.py
│   │   ├── markowitz.py
│   │   └── random_strategy.py
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── plotting.py
│   │   ├── logger.py
│   │   └── seed.py
│   └── inference.py
│── outputs/
│   ├── models/
│   ├── figures/
│   └── reports/
│── train.py
│── evaluate.py
│── requirements.txt
│── README.md
```

## 4. Installation

Create a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## 5. How To Train

```bash
python train.py
```

Training pipeline:

1. Download or load cached Yahoo Finance data
2. Build technical indicators and covariance matrices
3. Split time series into train/validation/test
4. Train PPO-style Actor-Critic
5. Save checkpoints and TensorBoard logs
6. Save processed dataset to `data/processed/`

### TensorBoard

```bash
tensorboard --logdir outputs/tensorboard
```

Tracked logs:

- Reward
- Actor loss
- Critic loss
- Entropy
- Validation Sharpe
- Learning rate

## 6. How To Evaluate

```bash
python evaluate.py
```

Evaluation will:

- Load the saved processed dataset
- Load the best trained model
- Backtest on the test split
- Compare against:
  - Buy and Hold Equal Weight
  - Mean-Variance Markowitz
  - Random Allocation
- Save reports and figures

## 7. Default Tickers

Configured in `configs/config.yaml`:

```yaml
data:
  tickers:
    - AAPL
    - MSFT
    - GOOGL
    - AMZN
    - TSLA
    - META
    - NVDA
    - JPM
    - XOM
    - NFLX
```

You can replace or extend this list with any Yahoo Finance tickers.

## 8. Switching Encoder Type

Update:

```yaml
model:
  encoder_type: "transformer"
```

Supported values:

- `lstm`
- `cnn`
- `transformer`

## 9. Config Highlights

Important sections in `configs/config.yaml`:

- `data`: tickers, dates, split ratios, cache directories
- `features`: feature list, normalization, covariance window
- `environment`: fee rate, slippage, reward coefficients
- `model`: encoder type and dimensions
- `training`: PPO hyperparameters, early stopping, checkpointing
- `evaluation`: benchmark settings and report output paths

## 10. Metrics

The evaluation report includes:

1. Total Return %
2. CAGR
3. Sharpe Ratio
4. Sortino Ratio
5. Max Drawdown
6. Volatility
7. Calmar Ratio
8. Win Rate
9. Final Portfolio Value
10. Turnover
11. Avg Holding Period

## 11. Visualizations

Saved to `outputs/figures/`:

- `training_reward_curve.png`
- `loss_curve.png`
- `equity_curve_test.png`
- `drawdown_chart.png`
- `weight_heatmap.png`
- `baseline_comparison.png`
- `rolling_sharpe.png`

## 12. Robustness Features

The code handles:

- Yahoo download retries
- Empty or failed ticker downloads
- Missing values and forward/backward filling
- NaN indicators
- Covariance regularization
- Zero-division protection in metrics
- Gradient clipping
- Mixed precision on CUDA
- Early stopping
- Checkpoint resume

## 13. Troubleshooting Yahoo Finance Errors

If Yahoo fails for some tickers:

- The loader retries automatically
- Failed tickers are skipped with logs
- Raw CSV cache in `data/raw/` reduces repeated requests

If you want a clean data refresh:

1. Delete the corresponding files in `data/raw/`
2. Run `python train.py` again

If too many tickers fail:

- Check internet connectivity
- Try fewer tickers
- Make sure ticker symbols are valid on Yahoo Finance

## 14. Train / Validation / Test Split

Default split:

- Train: 70%
- Validation: 15%
- Test: 15%

No shuffle is used because this is time-series data.

Optional walk-forward settings are included in config for future extension.

## 15. Example Workflow

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
```

Expected artifacts:

- Model checkpoints in `outputs/models/`
- Processed arrays in `data/processed/`
- TensorBoard logs in `outputs/tensorboard/`
- Comparison CSV / JSON in `outputs/reports/`
- Charts in `outputs/figures/`

## 16. Notes On Production Use

This project is built to be clean, modular, and reproducible for research and extension. For live deployment, you would typically add:

- Broker execution adapters
- Live risk constraints
- Intraday data ingestion
- More advanced transaction-cost models
- Regime filters and macro signals
- Hyperparameter sweeps

## 17. Command Summary

```bash
python train.py
python evaluate.py
```

That is enough to run the end-to-end research pipeline after installing dependencies.
