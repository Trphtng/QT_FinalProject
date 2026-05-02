"""Feature engineering and dataset preparation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass
class DataBundle:
    dates: list[str]
    tickers: list[str]
    feature_names: list[str]
    features: np.ndarray
    returns: np.ndarray
    covariances: np.ndarray
    prices: np.ndarray
    market_regime: np.ndarray

    def save(self, output_dir: str) -> None:
        """Persist processed arrays and metadata."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output / "dataset.npz",
            features=self.features.astype(np.float32),
            returns=self.returns.astype(np.float32),
            covariances=self.covariances.astype(np.float32),
            prices=self.prices.astype(np.float32),
            market_regime=self.market_regime.astype(np.float32),
        )
        metadata = {
            "dates": self.dates,
            "tickers": self.tickers,
            "feature_names": self.feature_names,
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, output_dir: str) -> "DataBundle":
        """Load processed arrays and metadata."""
        output = Path(output_dir)
        arrays = np.load(output / "dataset.npz")
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        return cls(
            dates=metadata["dates"],
            tickers=metadata["tickers"],
            feature_names=metadata["feature_names"],
            features=arrays["features"],
            returns=arrays["returns"],
            covariances=arrays["covariances"],
            prices=arrays["prices"],
            market_regime=arrays["market_regime"],
        )


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=1).mean()


def compute_macd(series: pd.Series) -> pd.DataFrame:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return pd.DataFrame({"MACD": macd, "MACDSignal": signal, "MACDHist": hist})


def compute_bollinger(series: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mean = series.rolling(period).mean()
    std = series.rolling(period).std()
    return pd.DataFrame(
        {
            "BBUpper": mean + n_std * std,
            "BBMiddle": mean,
            "BBLower": mean - n_std * std,
        }
    )


def normalize_features_by_train_split(features: np.ndarray, train_end: int) -> np.ndarray:
    train_end = max(1, min(train_end, features.shape[0]))
    train_slice = features[:train_end]
    mean = np.nanmean(train_slice, axis=0, keepdims=True)
    std = np.nanstd(train_slice, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (features - mean) / std


def engineer_features(raw_df: pd.DataFrame, feature_cfg: dict, data_cfg: dict) -> DataBundle:
    """Create model-ready tensors from long-form OHLCV data."""
    feature_frames: list[pd.DataFrame] = []
    min_history = int(data_cfg.get("min_history", 120))
    market_regime_cfg = feature_cfg.get("market_regime", {})

    for ticker, group in raw_df.groupby("Ticker"):
        group = group.sort_values("Date").copy()
        group["Return"] = group["Close"].pct_change().replace([np.inf, -np.inf], 0.0)
        group["LogReturn"] = np.log1p(group["Return"]).replace([np.inf, -np.inf], 0.0)
        group["ATR14"] = compute_atr(group["High"], group["Low"], group["Close"], period=14)
        group["RSI14"] = compute_rsi(group["Close"], period=14)
        group = pd.concat([group, compute_macd(group["Close"])], axis=1)
        group = pd.concat([group, compute_bollinger(group["Close"])], axis=1)
        group["SMA10"] = group["Close"].rolling(10).mean()
        group["SMA20"] = group["Close"].rolling(20).mean()
        group["EMA20"] = group["Close"].ewm(span=20, adjust=False).mean()
        group["Volatility20"] = group["Return"].rolling(20).std().fillna(0.0)
        group["Return5"] = group["Close"].pct_change(5)
        group["Return20"] = group["Close"].pct_change(20)
        group["Momentum10"] = group["Close"].pct_change(10)
        group["RealizedVol20"] = group["LogReturn"].rolling(20).std().fillna(0.0) * np.sqrt(20.0)
        group["Momentum20"] = group["Close"].pct_change(20)
        local_peak = group["Close"].rolling(60, min_periods=1).max()
        group["DrawdownLocalPeak"] = 1.0 - group["Close"] / local_peak.replace(0.0, np.nan)

        if market_regime_cfg.get("enabled", True):
            fast = int(market_regime_cfg.get("ma_fast", 20))
            slow = int(market_regime_cfg.get("ma_slow", 100))
            fast_ma = group["Close"].rolling(fast).mean()
            slow_ma = group["Close"].rolling(slow).mean()
            group["MarketRegime"] = (fast_ma > slow_ma).astype(float)
        else:
            group["MarketRegime"] = 0.0

        group = group.ffill().bfill()
        if len(group) < min_history:
            LOGGER.warning("Skipping %s because it has less than %s rows", ticker, min_history)
            continue
        feature_frames.append(group)

    if not feature_frames:
        raise RuntimeError("No ticker left after feature engineering.")

    df = pd.concat(feature_frames, axis=0)
    market_returns = (
        df.pivot(index="Date", columns="Ticker", values="Return").sort_index().fillna(0.0).mean(axis=1)
    )
    df = df.join(market_returns.rename("MarketReturn"), on="Date")
    df["RollingCorrMarket20"] = (
        df.groupby("Ticker", group_keys=False)[["Return", "MarketReturn"]]
        .apply(lambda g: g["Return"].rolling(20, min_periods=5).corr(g["MarketReturn"]))
        .reset_index(level=0, drop=True)
    )
    df = df.ffill().bfill()
    feature_names = list(feature_cfg["include_columns"])

    wide_feature_dict: dict[str, pd.DataFrame] = {}
    for feature in feature_names:
        wide_feature_dict[feature] = (
            df.pivot(index="Date", columns="Ticker", values=feature).sort_index().ffill().bfill()
        )

    close_wide = df.pivot(index="Date", columns="Ticker", values="Close").sort_index().ffill().bfill()
    returns_wide = df.pivot(index="Date", columns="Ticker", values="Return").sort_index().fillna(0.0)
    regime_wide = df.pivot(index="Date", columns="Ticker", values="MarketRegime").sort_index().ffill().bfill()

    aligned_dates = close_wide.index
    aligned_tickers = close_wide.columns.tolist()
    n_dates = len(aligned_dates)
    n_assets = len(aligned_tickers)
    n_features = len(feature_names)

    features = np.zeros((n_dates, n_assets, n_features), dtype=np.float32)
    for idx, feature in enumerate(feature_names):
        features[:, :, idx] = wide_feature_dict[feature].loc[aligned_dates, aligned_tickers].to_numpy(dtype=np.float32)

    if feature_cfg.get("normalize_features", True):
        train_end = int(n_dates * float(data_cfg.get("train_ratio", 0.7)))
        features = normalize_features_by_train_split(features, train_end=train_end)

    returns = returns_wide.loc[aligned_dates, aligned_tickers].to_numpy(dtype=np.float32)
    prices = close_wide.loc[aligned_dates, aligned_tickers].to_numpy(dtype=np.float32)
    market_regime = regime_wide.loc[aligned_dates, aligned_tickers].mean(axis=1).to_numpy(dtype=np.float32)

    cov_window = int(feature_cfg.get("covariance_window", 20))
    covariances = np.zeros((n_dates, n_assets, n_assets), dtype=np.float32)
    for t in range(n_dates):
        start = max(0, t - cov_window + 1)
        window_returns = returns[start : t + 1]
        if window_returns.shape[0] < 2:
            covariances[t] = np.eye(n_assets, dtype=np.float32) * 1e-6
        else:
            cov = np.cov(window_returns.T)
            cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
            cov += np.eye(n_assets) * 1e-6
            covariances[t] = cov.astype(np.float32)

    bundle = DataBundle(
        dates=[date.strftime("%Y-%m-%d") for date in aligned_dates],
        tickers=aligned_tickers,
        feature_names=feature_names,
        features=np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0),
        returns=np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0),
        covariances=covariances,
        prices=np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0),
        market_regime=np.nan_to_num(market_regime, nan=0.0, posinf=0.0, neginf=0.0),
    )
    return bundle


def time_series_split(
    n_steps: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, tuple[int, int]]:
    """Create non-shuffled train/val/test boundaries."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train/val/test ratios must sum to 1.0")

    train_end = int(n_steps * train_ratio)
    val_end = train_end + int(n_steps * val_ratio)
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n_steps),
    }


def walk_forward_splits(n_steps: int, cfg: dict) -> list[dict[str, tuple[int, int]]]:
    """Build optional walk-forward splits."""
    wf_cfg = cfg.get("walk_forward", {})
    if not wf_cfg.get("enabled", False):
        return []

    train_window = int(wf_cfg["train_window"])
    val_window = int(wf_cfg["val_window"])
    test_window = int(wf_cfg["test_window"])
    step_size = int(wf_cfg["step_size"])
    splits: list[dict[str, tuple[int, int]]] = []

    start = 0
    while start + train_window + val_window + test_window <= n_steps:
        train_end = start + train_window
        val_end = train_end + val_window
        test_end = val_end + test_window
        splits.append(
            {
                "train": (start, train_end),
                "val": (train_end, val_end),
                "test": (val_end, test_end),
            }
        )
        start += step_size
    return splits
