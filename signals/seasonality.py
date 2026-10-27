"""Monthly-return seasonality. Pure pandas (no Streamlit), pandas-3 safe.

A month's return is close-to-close: its last close vs the previous month's last
close, on adjusted prices. The first month of the data has no previous close,
so it gets no return. A month still in progress (its last bar is before the
month's last weekday) is flagged as partial: it shows as month-to-date in the
grid but is left out of the per-month stats.
"""
import pandas as pd

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
STAT_COLS = ["Avg", "Median", "% Up", "Years"]


def monthly_returns(close: pd.Series) -> tuple[pd.Series, bool]:
    """(returns indexed by month-end date, whether the last month is partial).
    Months with no bars (gaps in the data) and the month after a gap come back
    NaN rather than spanning the gap."""
    close = close.dropna()
    if close.empty:
        return pd.Series(dtype=float), False
    month_close = close.resample("ME").last()
    rets = month_close.pct_change().iloc[1:]
    last_bar = close.index[-1]
    partial = last_bar.normalize() < last_bar + pd.offsets.BMonthEnd(0)
    return rets, bool(partial)


def seasonality_grid(rets: pd.Series, years: int | None = None) -> pd.DataFrame:
    """Months (Jan..Dec) as rows, calendar years as columns. `years` keeps only
    the last N calendar years (counting the current, possibly partial, one)."""
    rets = rets.dropna()
    df = pd.DataFrame({"year": rets.index.year, "month": rets.index.month, "ret": rets.to_numpy()})
    grid = df.pivot(index="month", columns="year", values="ret").reindex(range(1, 13))
    if years is not None:
        grid = grid[grid.columns[-years:]]
    grid.index = MONTHS
    grid.columns = [int(c) for c in grid.columns]
    return grid


def month_stats(grid: pd.DataFrame, partial: tuple[int, int] | None = None) -> pd.DataFrame:
    """Avg / Median / % Up / Years per month row. `partial` is the (year,
    month) of an in-progress month to leave out."""
    full = grid.copy()
    if partial is not None and partial[0] in full.columns:
        full.loc[MONTHS[partial[1] - 1], partial[0]] = float("nan")
    return pd.DataFrame({
        "Avg": full.mean(axis=1),
        "Median": full.median(axis=1),
        "% Up": (full > 0).sum(axis=1) / full.count(axis=1),
        "Years": full.count(axis=1),
    })


def year_totals(grid: pd.DataFrame) -> pd.Series:
    """Each column's compounded return over the months it has. A year whose
    first months predate the data (or the current year) is therefore partial."""
    return (1 + grid).prod(min_count=1) - 1


def month_outlook(stats: pd.DataFrame, month: int) -> dict | None:
    """Summary of one calendar month (1-12) for a one-line callout, or None
    if there's no completed history for it."""
    row = stats.loc[MONTHS[month - 1]]
    if not row["Years"]:
        return None
    n = int(row["Years"])
    return {"month": MONTHS[month - 1], "avg": row["Avg"], "median": row["Median"],
            "up": round(row["% Up"] * n), "years": n}
