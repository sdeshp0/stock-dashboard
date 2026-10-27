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


# ---- Within-year shape and calendar effects (daily data) ----

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
TURN_WINDOW = 5  # trading days either side of month end for the turn-of-month view


def seasonal_paths(close: pd.Series) -> pd.DataFrame:
    """Each calendar year's cumulative return by trading day of the year,
    measured from the previous year's last close (so day 1 includes the first
    session's move). Rows are trading day 1, 2, ...; columns are years. The
    first year of data is dropped, since it has no prior year-end close."""
    close = close.dropna()
    year_end = close.groupby(close.index.year).last()
    paths = {}
    for year, s in close.groupby(close.index.year):
        if year - 1 not in year_end.index:
            continue
        paths[int(year)] = (s / year_end[year - 1] - 1).reset_index(drop=True)
    out = pd.DataFrame(paths)
    out.index = out.index + 1
    out.index.name = "day"
    return out


def typical_path(paths: pd.DataFrame, exclude: int | None = None) -> pd.DataFrame:
    """Median and interquartile band across years for each trading day, plus
    how many years reach that day. `exclude` leaves out a year (the current,
    unfinished one). Median rather than mean so one extreme year doesn't
    reshape the curve."""
    p = paths.drop(columns=[exclude], errors="ignore")
    return pd.DataFrame({"median": p.median(axis=1), "q25": p.quantile(0.25, axis=1),
                         "q75": p.quantile(0.75, axis=1), "years": p.count(axis=1)})


def _daily(close: pd.Series) -> pd.DataFrame:
    """Daily returns with each day's position in its month: `from_start` (1 =
    first session) and `from_end` (-1 = last session). If the data starts
    mid-month, that first month is dropped (its first sessions are missing),
    and an unfinished last month has no `from_end`, since its real last
    session hasn't happened yet."""
    close = close.dropna()
    df = pd.DataFrame({"ret": close.pct_change()}, index=close.index).iloc[1:]
    key = df.index.to_period("M")
    if len(df) and close.index[0].to_period("M") == key[0]:
        df = df[key != key[0]]
    key = df.index.to_period("M")
    df["from_start"] = df.groupby(key).cumcount() + 1
    df["from_end"] = df.groupby(key).cumcount(ascending=False) * -1 - 1
    if len(df):
        _, partial = monthly_returns(close)
        if partial:
            df.loc[key == key[-1], "from_end"] = pd.NA
    return df


def _ret_stats(g) -> pd.DataFrame:
    return pd.DataFrame({"Avg": g.mean(), "% Up": g.apply(lambda s: (s > 0).mean()),
                         "Days": g.count()})


def turn_of_month(close: pd.Series, window: int = TURN_WINDOW) -> tuple[pd.DataFrame, float]:
    """(stats by offset, average of all days). Offsets run -window..-1 for the
    last sessions of a month (T-1 = last day) and +1..+window for the first
    sessions of the next (T+1 = first day). Stats are Avg, % Up, Days."""
    df = _daily(close)
    if df.empty:
        return pd.DataFrame(columns=["Avg", "% Up", "Days"]), float("nan")
    end = df.loc[df["from_end"] >= -window]
    start = df.loc[df["from_start"] <= window]
    offs = pd.concat([pd.Series(end["ret"].to_numpy(), index=end["from_end"].astype(int)),
                      pd.Series(start["ret"].to_numpy(), index=start["from_start"].astype(int))])
    order = list(range(-window, 0)) + list(range(1, window + 1))
    stats = _ret_stats(offs.groupby(level=0)).reindex(order)
    stats.index = [f"T{o:+d}" for o in order]
    return stats, float(df["ret"].mean())


def weekday_stats(close: pd.Series) -> pd.DataFrame:
    """Avg, % Up and Days of daily returns by weekday (Mon..Fri)."""
    df = _daily(close)
    stats = _ret_stats(df["ret"].groupby(df.index.dayofweek)).reindex(range(5))
    stats.index = WEEKDAYS
    return stats


def last_n_years(close: pd.Series, years: int | None) -> pd.Series:
    """Trim daily closes to the last `years` calendar years, keeping the
    previous year's last close as the base for the first year's returns."""
    close = close.dropna()
    if years is None or close.empty:
        return close
    first_year = close.index[-1].year - years + 1
    prior = close[close.index.year < first_year]
    keep = close[close.index.year >= first_year]
    return pd.concat([prior.iloc[-1:], keep]) if len(prior) else keep
