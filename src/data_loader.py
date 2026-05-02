"""Data loading and caching utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import yfinance as yf

from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass
class DownloadConfig:
    tickers: List[str]
    start_date: str
    end_date: str | None
    interval: str
    raw_dir: str
    use_cache: bool
    max_retries: int
    retry_delay_seconds: int


class YahooDataLoader:
    """Download OHLCV data from Yahoo Finance with caching and retries."""

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, config: DownloadConfig) -> None:
        self.config = config
        self.raw_dir = Path(config.raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> pd.DataFrame:
        """Return a long-format dataframe indexed by Date and Ticker."""
        frames: list[pd.DataFrame] = []
        failed: list[str] = []
        for ticker in self.config.tickers:
            try:
                frames.append(self._load_single_ticker(ticker))
            except Exception as exc:  # pragma: no cover - defensive logging
                failed.append(ticker)
                LOGGER.exception("Failed to load %s: %s", ticker, exc)

        if not frames:
            raise RuntimeError("No ticker could be downloaded from Yahoo Finance.")

        combined = pd.concat(frames, axis=0).sort_values(["Date", "Ticker"])
        if failed:
            LOGGER.warning("Skipped tickers due to download issues: %s", failed)
        return combined.reset_index(drop=True)

    def _load_single_ticker(self, ticker: str) -> pd.DataFrame:
        cache_path = self.raw_dir / f"{ticker}.csv"
        if self.config.use_cache and cache_path.exists():
            LOGGER.info("Loading cached raw data for %s", ticker)
            df = pd.read_csv(cache_path, parse_dates=["Date"])
        else:
            df = self._download_with_retry(ticker)
            df.to_csv(cache_path, index=False)

        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Ticker {ticker} missing columns: {missing_cols}")

        df["Ticker"] = ticker
        df = df.dropna(subset=["Close"]).copy()
        df["Volume"] = df["Volume"].fillna(0.0)
        return df

    def _download_with_retry(self, ticker: str) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                LOGGER.info("Downloading %s from Yahoo Finance (attempt %s)", ticker, attempt)
                df = yf.download(
                    ticker,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    interval=self.config.interval,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                if df.empty:
                    raise ValueError(f"Yahoo returned empty data for {ticker}")

                df = df.reset_index()
                df.columns = [str(col[0] if isinstance(col, tuple) else col) for col in df.columns]
                if "Adj Close" in df.columns and "Close" not in df.columns:
                    df["Close"] = df["Adj Close"]
                return df
            except Exception as exc:  # pragma: no cover - retry path
                last_error = exc
                LOGGER.warning("Yahoo download failed for %s: %s", ticker, exc)
                time.sleep(self.config.retry_delay_seconds)
        raise RuntimeError(f"Exceeded retries for {ticker}") from last_error


def create_download_config(data_cfg: dict) -> DownloadConfig:
    """Build a typed download config from the yaml dictionary."""
    return DownloadConfig(
        tickers=list(data_cfg["tickers"]),
        start_date=data_cfg["start_date"],
        end_date=data_cfg.get("end_date"),
        interval=data_cfg["interval"],
        raw_dir=data_cfg["raw_dir"],
        use_cache=bool(data_cfg.get("use_cache", True)),
        max_retries=int(data_cfg.get("max_retries", 3)),
        retry_delay_seconds=int(data_cfg.get("retry_delay_seconds", 2)),
    )


def available_tickers(df: pd.DataFrame) -> Iterable[str]:
    """Yield tickers that survived ingestion."""
    return sorted(df["Ticker"].unique().tolist())
