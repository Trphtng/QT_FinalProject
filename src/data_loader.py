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
    provider: str
    tickers: List[str]
    start_date: str
    end_date: str | None
    interval: str
    raw_dir: str
    use_cache: bool
    max_retries: int
    retry_delay_seconds: int
    vn_source: str


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


class VNStockDataLoader:
    """Download OHLCV data for the Vietnam market via vnstock with caching and retries."""

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, config: DownloadConfig) -> None:
        self.config = config
        self.raw_dir = Path(config.raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.quote_cls = self._resolve_quote_class()

    def load(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        failed: list[str] = []
        for ticker in self.config.tickers:
            try:
                frames.append(self._load_single_ticker(ticker))
            except Exception as exc:  # pragma: no cover - defensive logging
                failed.append(ticker)
                LOGGER.exception("Failed to load %s from vnstock: %s", ticker, exc)

        if not frames:
            raise RuntimeError("No ticker could be downloaded from vnstock.")

        combined = pd.concat(frames, axis=0).sort_values(["Date", "Ticker"])
        if failed:
            LOGGER.warning("Skipped VN tickers due to download issues: %s", failed)
        return combined.reset_index(drop=True)

    def _resolve_quote_class(self):
        try:
            from vnstock import Quote

            return Quote
        except ImportError:
            try:
                from vnstock_data import Quote

                return Quote
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImportError(
                    "vnstock is required for provider='vnstock'. Install it with `pip install vnstock`."
                ) from exc

    def _load_single_ticker(self, ticker: str) -> pd.DataFrame:
        cache_path = self.raw_dir / f"{ticker}.csv"
        if self.config.use_cache and cache_path.exists():
            LOGGER.info("Loading cached raw VN data for %s", ticker)
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
        interval = self._normalize_interval(self.config.interval)
        source = self.config.vn_source
        for attempt in range(1, self.config.max_retries + 1):
            try:
                LOGGER.info("Downloading %s from vnstock source=%s (attempt %s)", ticker, source, attempt)
                quote = self.quote_cls(symbol=ticker, source=source)
                df = quote.history(
                    start=self.config.start_date,
                    end=self.config.end_date,
                    interval=interval,
                )
                if df is None or df.empty:
                    raise ValueError(f"vnstock returned empty data for {ticker}")
                normalized = self._normalize_vnstock_frame(df)
                if normalized.empty:
                    raise ValueError(f"vnstock normalization produced empty data for {ticker}")
                return normalized
            except Exception as exc:  # pragma: no cover - retry path
                last_error = exc
                LOGGER.warning("vnstock download failed for %s: %s", ticker, exc)
                time.sleep(self.config.retry_delay_seconds)
        raise RuntimeError(f"Exceeded retries for {ticker}") from last_error

    @staticmethod
    def _normalize_interval(interval: str) -> str:
        mapping = {
            "1d": "1D",
            "1wk": "1W",
            "1mo": "1M",
            "1h": "1H",
            "60m": "1H",
            "30m": "30m",
            "15m": "15m",
            "5m": "5m",
            "1m": "1m",
        }
        return mapping.get(str(interval).strip(), str(interval).strip())

    @staticmethod
    def _normalize_vnstock_frame(df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        for column in df.columns:
            lower_col = str(column).strip().lower()
            if lower_col == "time":
                rename_map[column] = "Date"
            elif lower_col == "date":
                rename_map[column] = "Date"
            elif lower_col == "open":
                rename_map[column] = "Open"
            elif lower_col == "high":
                rename_map[column] = "High"
            elif lower_col == "low":
                rename_map[column] = "Low"
            elif lower_col == "close":
                rename_map[column] = "Close"
            elif lower_col == "volume":
                rename_map[column] = "Volume"

        normalized = df.rename(columns=rename_map).copy()
        if "Date" not in normalized.columns:
            raise ValueError("vnstock dataframe is missing a time/date column.")

        normalized["Date"] = pd.to_datetime(normalized["Date"])
        keep_columns = [col for col in ["Date", "Open", "High", "Low", "Close", "Volume"] if col in normalized.columns]
        normalized = normalized[keep_columns].sort_values("Date").reset_index(drop=True)
        return normalized


def build_data_loader(config: DownloadConfig) -> YahooDataLoader | VNStockDataLoader:
    provider = config.provider.lower()
    if provider == "yahoo":
        return YahooDataLoader(config)
    if provider == "vnstock":
        return VNStockDataLoader(config)
    raise ValueError(f"Unsupported data provider: {config.provider}")


def create_download_config(data_cfg: dict) -> DownloadConfig:
    """Build a typed download config from the yaml dictionary."""
    return DownloadConfig(
        provider=str(data_cfg.get("provider", "yahoo")),
        tickers=list(data_cfg["tickers"]),
        start_date=data_cfg["start_date"],
        end_date=data_cfg.get("end_date"),
        interval=data_cfg["interval"],
        raw_dir=data_cfg["raw_dir"],
        use_cache=bool(data_cfg.get("use_cache", True)),
        max_retries=int(data_cfg.get("max_retries", 3)),
        retry_delay_seconds=int(data_cfg.get("retry_delay_seconds", 2)),
        vn_source=str(data_cfg.get("vn_source", "VCI")),
    )


def available_tickers(df: pd.DataFrame) -> Iterable[str]:
    """Yield tickers that survived ingestion."""
    return sorted(df["Ticker"].unique().tolist())
