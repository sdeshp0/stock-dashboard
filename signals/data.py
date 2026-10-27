"""Streamlit-facing data loading. Prices come from the nightly snapshot first
when it covers the request and is current (daily bars only change once a day,
so a live download would return the same thing), then from a live fetch, and
finally from the snapshot again as a fallback for when Yahoo Finance or
Wikipedia are unreachable - e.g. rate-limited on a shared IP like Streamlit
Community Cloud. The snapshot is produced by scripts/build_snapshot.py on a
schedule (see .github/workflows/update_snapshot.yml), published to the repo's
`data` branch, and downloaded into data/ by the app at most once an hour.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from signals.data_core import (
    BENCHMARK_TICKER,
    SNAPSHOT_PERIODS,
    download_snapshot,
    fetch_prices,
    fetch_sp500,
    slice_snapshot,
    snapshot_is_fresh,
)

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SNAPSHOT_CONSTITUENTS = DATA_DIR / "snapshot_constituents.csv"
SNAPSHOT_PRICES = DATA_DIR / "snapshot_prices.parquet"
SNAPSHOT_META = DATA_DIR / "snapshot_meta.json"

# Plain cached versions of the live fetchers (no fallback) - used directly by
# the safe wrappers below, and available on their own if ever needed.
load_sp500 = st.cache_data(ttl=24 * 3600, show_spinner=False)(fetch_sp500)
load_prices = st.cache_data(ttl=3600, show_spinner=False)(fetch_prices)


@st.cache_resource(ttl=3600, show_spinner=False)
def sync_snapshot() -> None:
    """Pull the latest published snapshot into data/, at most once an hour per
    server process. If the download fails, whatever is already in data/ (if
    anything) keeps being used, and with nothing there the app just goes live."""
    try:
        if download_snapshot(DATA_DIR):
            log.info("Downloaded a new data snapshot.")
    except Exception as e:
        log.warning("Snapshot download failed (%s); using the local copy if any.", e)


def snapshot_asof() -> str | None:
    sync_snapshot()
    if not SNAPSHOT_META.exists():
        return None
    try:
        return json.loads(SNAPSHOT_META.read_text()).get("as_of")
    except Exception:
        return None


def _read_snapshot_constituents() -> pd.DataFrame:
    return pd.read_csv(SNAPSHOT_CONSTITUENTS)


@st.cache_resource(show_spinner=False, max_entries=1)
def _snapshot_wide(mtime: float) -> pd.DataFrame:
    """The whole snapshot, read once per file version (keyed on mtime, so a
    freshly downloaded nightly snapshot is picked up)."""
    return pd.read_parquet(SNAPSHOT_PRICES)  # MultiIndex columns: (ticker, field)


def _read_snapshot_prices(tickers: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    wide = _snapshot_wide(SNAPSHOT_PRICES.stat().st_mtime)
    have = {t for t, _ in wide.columns} & set(tickers)
    return {t: wide[t].dropna(how="all") for t in have}


def _fresh_snapshot_prices(tickers: tuple[str, ...], period: str):
    """(prices, missing, last_bar) from the snapshot if it can serve this request
    and is up to date with the last completed trading day; otherwise None."""
    sync_snapshot()
    if period not in SNAPSHOT_PERIODS or not SNAPSHOT_PRICES.exists():
        return None
    try:
        wide = _snapshot_wide(SNAPSHOT_PRICES.stat().st_mtime)
    except Exception as e:
        log.warning("Snapshot unreadable (%s); going live.", e)
        return None
    last_bar = wide.index.max().date()
    if not snapshot_is_fresh(last_bar, datetime.now(timezone.utc)):
        return None
    prices, missing = slice_snapshot(wide, tickers, period)
    if not prices:
        return None
    return prices, missing, last_bar


@st.cache_data(ttl=3600, show_spinner=False)
def load_benchmark_close(period: str = "1y") -> tuple[pd.Series | None, str, str | None]:
    """The market benchmark's Close series, for beta. Reuses load_prices_safe
    (and therefore its live/snapshot fallback) for a single ticker."""
    try:
        prices, source, asof = load_prices_safe((BENCHMARK_TICKER,), period)
    except Exception as e:
        log.warning("Benchmark (%s) unavailable (%s); Beta will be blank.", BENCHMARK_TICKER, e)
        return None, "unavailable", None
    df = prices.get(BENCHMARK_TICKER)
    if df is None:
        return None, "unavailable", None
    return df["Close"], source, asof


@st.cache_data(ttl=3600, show_spinner=False)
def load_sp500_safe() -> tuple[pd.DataFrame, str, str | None]:
    """Returns (constituents, source, as_of). source is 'live' or 'snapshot'."""
    try:
        return load_sp500(), "live", None
    except Exception as e:
        log.warning("Live S&P 500 list failed (%s); trying snapshot.", e)
        sync_snapshot()
        if not SNAPSHOT_CONSTITUENTS.exists():
            raise
        return _read_snapshot_constituents(), "snapshot", snapshot_asof()


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices_safe(tickers: tuple[str, ...], period: str = "1y"):
    """Returns (prices, source, as_of). source is:
    - 'nightly': the snapshot is current and covers the request (as_of is the
      last close it holds). Any requested tickers it lacks are fetched live and
      merged in; if that fails they just come back missing.
    - 'live': fetched from Yahoo just now (as_of is None).
    - 'fallback': live failed, so an out-of-date snapshot is being served."""
    fresh = _fresh_snapshot_prices(tickers, period)
    if fresh is not None:
        prices, missing, last_bar = fresh
        if missing:
            try:
                prices = {**prices, **load_prices(tuple(missing), period)}
            except Exception as e:
                log.warning("Live fetch for %d non-snapshot ticker(s) failed: %s", len(missing), e)
        return prices, "nightly", f"{last_bar:%Y-%m-%d} close"
    try:
        return load_prices(tickers, period), "live", None
    except Exception as e:
        log.warning("Live price download failed (%s); trying snapshot.", e)
        sync_snapshot()
        if not SNAPSHOT_PRICES.exists():
            raise
        prices = _read_snapshot_prices(tickers)
        if not prices:
            raise RuntimeError("No matching tickers found in the snapshot either.") from e
        return prices, "fallback", snapshot_asof()


@st.cache_data(ttl=24 * 3600, show_spinner=False, max_entries=50)
def load_long_history(ticker: str) -> pd.DataFrame:
    """All available daily history for one ticker, for the seasonality view.
    Always live (the snapshot only holds a year), and cached for a day since
    monthly returns barely move intraday. Raises if Yahoo returns nothing."""
    return fetch_prices((ticker,), "max")[ticker]
