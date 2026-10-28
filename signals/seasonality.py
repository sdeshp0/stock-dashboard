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


# ---- Forward outlook from today's calendar date ----

FORWARD_HORIZONS = (5, 10, 20, 60)  # trading sessions


def forward_from_date(close: pd.Series, horizons=FORWARD_HORIZONS,
                      as_of: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How the next `h` sessions went in past years, starting from the same
    calendar date as `as_of` (default: the last bar). Each year enters at the
    close of its last session on or before that date - like buying at today's
    close - and the current year is left out, since its window hasn't happened.
    Returns (by_year: years x horizons, summary: horizons x stats)."""
    close = close.dropna()
    cols = [f"{h}d" for h in horizons]
    if close.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=FORWARD_STATS)
    as_of = as_of or close.index[-1]
    idx, vals = close.index, close.to_numpy()
    rows = {}
    for year in sorted(set(idx.year)):
        if year >= as_of.year:
            continue
        day = min(as_of.day, 28) if as_of.month == 2 else as_of.day  # Feb 29 -> 28
        anchor = pd.Timestamp(year=year, month=as_of.month, day=day)
        pos = idx.searchsorted(anchor, side="right") - 1
        if pos < 0 or anchor - idx[pos] > pd.Timedelta(days=7):
            continue  # data starts after this date (or a long gap before it)
        rows[year] = [vals[pos + h] / vals[pos] - 1 if pos + h < len(vals) else float("nan")
                      for h in horizons]
    by_year = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    by_year.index.name = "Year"
    return by_year, summarize_returns(by_year)


FORWARD_STATS = ["Avg", "Median", "% Up", "Best", "Worst", "Years"]


def summarize_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column Avg / Median / % Up / Best / Worst / Years, one row per column."""
    return pd.DataFrame({"Avg": df.mean(), "Median": df.median(),
                         "% Up": (df > 0).sum() / df.count(), "Best": df.max(),
                         "Worst": df.min(), "Years": df.count()})[FORWARD_STATS]


# ---- Event windows ----

EVENT_ORDER = ["Day before a holiday", "Day after a holiday", "Options-expiry week",
               "Week after options expiry", "Quarter's last 5 days", "Quarter's first 5 days",
               "All days"]


def third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    return first + pd.Timedelta(days=(4 - first.dayofweek) % 7 + 14)


def event_flags(close: pd.Series) -> pd.DataFrame:
    """Daily returns (as in _daily) flagged with the calendar event each session
    falls in. Market holidays are read from the data itself: a weekday with no
    session between two sessions is taken as a holiday. Options expiry is the
    third Friday of the month (the session before it if that Friday is a holiday)."""
    df = _daily(close)
    if df.empty:
        return df
    sessions = close.dropna().index
    have = set(sessions)
    first, last = sessions[0], sessions[-1]
    nxt = df.index + pd.offsets.BDay(1)
    prv = df.index - pd.offsets.BDay(1)
    df["pre_holiday"] = [(n not in have) and n < last for n in nxt]
    df["post_holiday"] = [(p not in have) and p > first for p in prv]

    # Options expiry sessions, then the Mon-Fri week each one falls in.
    opex = []
    for p in pd.period_range(first, last, freq="M"):
        tf = third_friday(p.year, p.month)
        pos = sessions.searchsorted(tf, side="right") - 1
        if pos >= 0 and tf - sessions[pos] < pd.Timedelta(days=4):
            opex.append(sessions[pos])
    week = df.index.to_period("W-FRI")
    opex_weeks = {d.to_period("W-FRI") for d in opex}
    df["opex_week"] = week.isin(opex_weeks)
    df["post_opex_week"] = week.isin({w + 1 for w in opex_weeks})

    q_end = df.index.month.isin([3, 6, 9, 12])
    df["quarter_end"] = q_end & (df["from_end"] >= -5).fillna(False).to_numpy(dtype=bool)
    df["quarter_start"] = df.index.month.isin([1, 4, 7, 10]) & (df["from_start"] <= 5)
    return df


def event_windows(close: pd.Series) -> pd.DataFrame:
    """Avg daily return, % Up and Days for each calendar event, plus all days."""
    df = event_flags(close)
    if df.empty:
        return pd.DataFrame(columns=["Avg", "% Up", "Days"])
    groups = {"Day before a holiday": "pre_holiday", "Day after a holiday": "post_holiday",
              "Options-expiry week": "opex_week", "Week after options expiry": "post_opex_week",
              "Quarter's last 5 days": "quarter_end", "Quarter's first 5 days": "quarter_start"}
    out = {}
    for label, col in groups.items():
        r = df.loc[df[col], "ret"]
        out[label] = {"Avg": r.mean(), "% Up": (r > 0).mean() if len(r) else float("nan"),
                      "Days": len(r)}
    out["All days"] = {"Avg": df["ret"].mean(), "% Up": (df["ret"] > 0).mean(), "Days": len(df)}
    return pd.DataFrame.from_dict(out, orient="index").loc[EVENT_ORDER]


def earnings_reaction_day(sessions: pd.DatetimeIndex, when: pd.Timestamp) -> pd.Timestamp | None:
    """The session that first trades on an earnings release at local time
    `when`: a release from noon on (after the close) is traded the next
    session, an earlier one (before the open) the same day."""
    day = when.normalize()
    pos = sessions.searchsorted(day, side="right" if when.hour >= 12 else "left")
    return sessions[pos] if pos < len(sessions) else None


def earnings_moves(close: pd.Series, release_times, pre: int = 5,
                   post: int = 20) -> pd.DataFrame:
    """One row per past earnings release: the `pre` sessions before the
    reaction day, the reaction day itself (close-to-close), and the `post`
    sessions after it. Windows that run past the data are NaN."""
    close = close.dropna()
    idx, vals = close.index, close.to_numpy()
    rows = {}
    for when in sorted(release_times):
        day = earnings_reaction_day(idx, when)
        if day is None:
            continue
        r = idx.get_loc(day)
        if r - 1 - pre < 0:
            continue
        rows[day] = {
            f"{pre} days before": vals[r - 1] / vals[r - 1 - pre] - 1,
            "Reaction day": vals[r] / vals[r - 1] - 1,
            f"{post} days after": vals[r + post] / vals[r] - 1 if r + post < len(vals) else float("nan"),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "Reaction day"
    return out


# ---- Cross-ticker calendar ----

def month_avg_row(close: pd.Series, years: int | None,
                  benchmark: pd.Series | None = None) -> pd.Series:
    """Average return for each calendar month (completed months, last `years`
    calendar years) - in excess of the benchmark's return in the same month
    when one is given. Indexed Jan..Dec, plus 'Years' (most years any month has)."""
    rets, partial = monthly_returns(close)
    if benchmark is not None:
        b, _ = monthly_returns(benchmark)
        rets = (rets - b).dropna()
    if rets.empty:
        return pd.Series(float("nan"), index=MONTHS + ["Years"])
    grid = seasonality_grid(rets, years)
    last = rets.index[-1]
    stats = month_stats(grid, (last.year, last.month) if partial else None)
    return pd.concat([stats["Avg"], pd.Series({"Years": stats["Years"].max()})])
