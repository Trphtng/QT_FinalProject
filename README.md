# Deep Reinforcement Learning Portfolio Management

Production-ready project for multi-asset portfolio management with a PPO-style Actor-Critic architecture in PyTorch. The system downloads real market data from Yahoo Finance or Vietnam market data via Vnstock, engineers technical indicators, trains a portfolio policy over historical data, and compares the learned strategy against strong baselines.

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

Supported market data providers:

- `yahoo`: US/global equities and ETFs through Yahoo Finance
- `vnstock`: Vietnam equities, ETFs, and indices through Vnstock

## 5. How To Train

```bash
python train.py
```

Train with the bundled Vietnam configuration:

```bash
python train.py --config configs/config_vn30.yaml
```

Training pipeline:

1. Download or load cached market data from the configured provider
2. Reuse the cached processed dataset when the active data/feature config has not changed
3. Otherwise build technical indicators and covariance matrices and save them to the processed cache
4. Split time series into train/validation/test or walk-forward folds
5. Train PPO-style Actor-Critic
6. Save checkpoints and TensorBoard logs
7. Save processed dataset to the configured `processed_dir`

Current default training budget in both bundled configs:

- `total_epochs: 50`
- `max_rollout_steps_per_fold: 250`
- Walk-forward training enabled

Data loading notes:

- Raw ticker downloads are cached in the configured `raw_dir`
- Processed tensors are cached in the configured `processed_dir`
- Multi-ticker downloads run in parallel via `data.max_workers`

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

Evaluate the Vietnam experiment:

```bash
python evaluate.py --config configs/config_vn30.yaml
```

Evaluate only the PPO model without baseline comparisons:

```bash
python evaluate.py --config configs/config.yaml --rl-only
python evaluate.py --config configs/config_vn30.yaml --rl-only
```

Evaluation will:

- Load the saved processed dataset
- If walk-forward is enabled, automatically select the best completed fold from `outputs/reports/walk_forward_summary.json`
- If that summary file is not available yet, fall back to the most recently trained fold checkpoint
- Otherwise load the default checkpoint from `training.checkpoint_path`
- Backtest on the matching test split for the selected fold
- Compare against:
  - Buy and Hold Equal Weight
  - Mean-Variance Markowitz
  - Random Allocation
- Save reports and figures

With `--rl-only`, evaluation will:

- Load the trained PPO model
- Backtest only that model on the test split
- Save PPO metrics and PPO-only charts without benchmark comparison tables

## 7. Default Tickers

Configured in `configs/config.yaml`:

```yaml
data:
  provider: "yahoo"
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

## 8. Vietnam Dataset Support

The repository now includes a Vietnam-market configuration at `configs/config_vn30.yaml`.

Default VN sample basket:

```yaml
data:
  provider: "vnstock"
  vn_source: "VCI"
  tickers:
    - FPT
    - VCB
    - HPG
    - MBB
    - ACB
    - SSI
    - MWG
    - VNM
    - TCB
    - VPB
```

Notes:

- `vn_source` defaults to `VCI`, which is the richer local data source in the Vnstock docs.
- You can swap in other Vietnam tickers, ETFs such as `E1VFVN30`, or indices such as `VNINDEX` and `VN30`.
- The VN config uses separate cache and processed directories: `data/raw_vn30/` and `data/processed_vn30/`.

To build a different Vietnam dataset, copy `configs/config_vn30.yaml` and edit only:

- `data.tickers`
- `data.start_date` / `data.end_date`
- output paths under `data`, `training`, and `evaluation`

## 9. Switching Encoder Type

Update:

```yaml
model:
  encoder_type: "transformer"
```

Supported values:

- `lstm`
- `cnn`
- `transformer`

## 10. Config Highlights

Important sections in `configs/config.yaml`:

- `data`: provider, tickers, dates, split ratios, cache directories
- `data.max_workers`: parallel download workers for multi-ticker ingestion
- `features`: feature list, normalization, covariance window
- `environment`: fee rate, slippage, reward coefficients
- `model`: encoder type and dimensions
- `training`: PPO hyperparameters, early stopping, checkpointing, `total_epochs`, and `max_rollout_steps_per_fold`
- `evaluation`: benchmark settings and report output paths

## 11. Metrics

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

## 12. Visualizations

Saved to the configured evaluation figure directory:

- `training_reward_curve.png`
- `loss_curve.png`
- `equity_curve_test.png`
- `drawdown_chart.png`
- `weight_heatmap.png`
- `baseline_comparison.png`
- `rolling_sharpe.png`

Default evaluation figure directories:

- US/global config: `outputs/figures/us/`
- Vietnam config: `outputs/figures/vn30/`

## 13. Robustness Features

The code handles:

- Yahoo download retries
- Vnstock download retries
- Empty or failed ticker downloads
- Missing values and forward/backward filling
- NaN indicators
- Covariance regularization
- Zero-division protection in metrics
- Gradient clipping
- Mixed precision on CUDA
- Early stopping
- Checkpoint resume

## 14. Troubleshooting Data Download Errors

If a provider fails for some tickers:

- The loader retries automatically
- Failed tickers are skipped with logs
- Raw CSV cache in the configured `raw_dir` reduces repeated requests
- Processed dataset cache in the configured `processed_dir` reduces repeated feature engineering

If you want a clean data refresh:

1. Delete the corresponding files in the configured `raw_dir`
2. Run `python train.py` again

If too many Yahoo tickers fail:

- Check internet connectivity
- Try fewer tickers
- Make sure ticker symbols are valid on Yahoo Finance

If too many Vietnam tickers fail:

- Check that `data.provider` is `vnstock`
- Verify ticker codes are valid on HOSE/HNX/UPCoM or as Vietnam indices/ETFs
- Try switching `vn_source` between `VCI` and `KBS` if your environment blocks one source

## 15. Train / Validation / Test Split

Default split:

- Train: 70%
- Validation: 15%
- Test: 15%

No shuffle is used because this is time-series data.

Walk-forward training is enabled in the bundled configs. Each completed fold writes or updates `outputs/reports/walk_forward_summary.json`, which `evaluate.py` uses to pick the best available fold automatically.

## 16. Example Workflow

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
```

Vietnam workflow:

```bash
pip install -r requirements.txt
python train.py --config configs/config_vn30.yaml
python evaluate.py --config configs/config_vn30.yaml
```

Two-dataset PPO-only evaluation workflow:

```bash
python train.py --config configs/config.yaml
python evaluate.py --config configs/config.yaml --rl-only

python train.py --config configs/config_vn30.yaml
python evaluate.py --config configs/config_vn30.yaml --rl-only
```

This gives you a clean evaluation section for the two datasets:

- US/global dataset: metrics in `outputs/reports/metrics_summary.json`
- Vietnam dataset: metrics in `outputs/reports/metrics_summary_vn30.json`
- Separate PPO charts in `outputs/figures/us/` and `outputs/figures/vn30/`

Expected artifacts:

- Model checkpoints in `outputs/models/`
- Processed arrays in `data/processed/`
- TensorBoard logs in `outputs/tensorboard/`
- Walk-forward fold summary in `outputs/reports/walk_forward_summary.json`
- Comparison CSV / JSON in `outputs/reports/`
- Charts in `outputs/figures/`

## 17. Notes On Production Use

This project is built to be clean, modular, and reproducible for research and extension. For live deployment, you would typically add:

- Broker execution adapters
- Live risk constraints
- Intraday data ingestion
- More advanced transaction-cost models
- Regime filters and macro signals
- Hyperparameter sweeps

## 18. Command Summary

```bash
python train.py
python evaluate.py
python train.py --config configs/config_vn30.yaml
python evaluate.py --config configs/config_vn30.yaml
python evaluate.py --config configs/config.yaml --rl-only
python evaluate.py --config configs/config_vn30.yaml --rl-only
```

That is enough to run the end-to-end research pipeline after installing dependencies.
