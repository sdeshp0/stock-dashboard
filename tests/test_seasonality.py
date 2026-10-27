import numpy as np
import pandas as pd
import pytest

from signals.seasonality import (MONTHS, WEEKDAYS, last_n_years, month_outlook, month_stats,
                                 monthly_returns, seasonal_paths, seasonality_grid, turn_of_month,
                                 typical_path, weekday_stats, year_totals)


def _closes(month_end_closes: dict[str, float], extra_days: int = 3) -> pd.Series:
    """Daily business-day closes where each month's last bar is the given
    close, with a few earlier bars at other prices that must be ignored."""
    parts = []
    for month, close in month_end_closes.items():
        days = pd.bdate_range(end=pd.Timestamp(month) + pd.offsets.BMonthEnd(0),
                              periods=extra_days + 1)
        parts.append(pd.Series([close * 1.5] * extra_days + [close], index=days))
    return pd.concat(parts)


def test_close_to_close_and_first_month_dropped():
    s = _closes({"2024-01": 100, "2024-02": 110, "2024-03": 99})
    rets, partial = monthly_returns(s)
    assert list(rets.index.strftime("%Y-%m")) == ["2024-02", "2024-03"]
    assert rets.iloc[0] == pytest.approx(0.10)
    assert rets.iloc[1] == pytest.approx(-0.10)
    assert partial is False


def test_partial_month_detected():
    s = _closes({"2024-01": 100, "2024-02": 110})
    s = pd.concat([s, pd.Series([121.0], index=[pd.Timestamp("2024-03-05")])])
    rets, partial = monthly_returns(s)
    assert partial is True
    assert rets.iloc[-1] == pytest.approx(0.10)  # month-to-date


def test_gap_month_is_nan_not_spanned():
    s = _closes({"2024-01": 100, "2024-02": 110, "2024-04": 121})
    rets, _ = monthly_returns(s)
    assert np.isnan(rets["2024-03"].iloc[0])
    assert np.isnan(rets["2024-04"].iloc[0])


def test_grid_shape_and_year_limit():
    idx = pd.date_range("2019-01-31", "2024-12-31", freq="ME")
    rets = pd.Series(0.01, index=idx)
    grid = seasonality_grid(rets)
    assert list(grid.index) == MONTHS
    assert list(grid.columns) == [2019, 2020, 2021, 2022, 2023, 2024]
    assert list(seasonality_grid(rets, years=3).columns) == [2022, 2023, 2024]


def test_grid_fills_missing_months_with_nan():
    rets = pd.Series([0.02, -0.01], index=pd.to_datetime(["2024-03-31", "2024-11-30"]))
    grid = seasonality_grid(rets)
    assert grid.loc["Mar", 2024] == pytest.approx(0.02)
    assert grid[2024].count() == 2


def test_month_stats_excludes_partial_month():
    rets = pd.Series([0.04, -0.02, 0.01, 0.10],
                     index=pd.to_datetime(["2021-03-31", "2022-03-31", "2023-03-31", "2024-03-31"]))
    grid = seasonality_grid(rets)
    stats = month_stats(grid, partial=(2024, 3))
    mar = stats.loc["Mar"]
    assert mar["Avg"] == pytest.approx(0.01)
    assert mar["Median"] == pytest.approx(0.01)
    assert mar["% Up"] == pytest.approx(2 / 3)
    assert mar["Years"] == 3
    assert grid.loc["Mar", 2024] == pytest.approx(0.10)  # grid itself untouched
    assert stats.loc["Jan", "Years"] == 0


def test_year_totals_compound():
    rets = pd.Series([0.10, -0.10], index=pd.to_datetime(["2024-01-31", "2024-02-29"]))
    totals = year_totals(seasonality_grid(rets))
    assert totals[2024] == pytest.approx(1.1 * 0.9 - 1)


def test_month_outlook():
    rets = pd.Series([0.04, -0.02, 0.01],
                     index=pd.to_datetime(["2021-03-31", "2022-03-31", "2023-03-31"]))
    stats = month_stats(seasonality_grid(rets))
    out = month_outlook(stats, 3)
    assert out == {"month": "Mar", "avg": pytest.approx(0.01), "median": pytest.approx(0.01),
                   "up": 2, "years": 3}
    assert month_outlook(stats, 1) is None


def _daily_closes(start: str, end: str, rets_by_date=None) -> pd.Series:
    """Business-day closes starting at 100, flat unless a date's return is given."""
    idx = pd.bdate_range(start, end)
    r = pd.Series(0.0, index=idx)
    for d, v in (rets_by_date or {}).items():
        r[pd.Timestamp(d)] = v
    r.iloc[0] = 0.0
    return 100 * (1 + r).cumprod()


def test_seasonal_paths_from_prior_year_end():
    s = _daily_closes("2023-06-01", "2025-03-31", {"2024-01-01": 0.10, "2025-01-01": -0.05})
    paths = seasonal_paths(s)
    assert list(paths.columns) == [2024, 2025]  # 2023 has no prior year-end close
    assert paths.loc[1, 2024] == pytest.approx(0.10)  # day 1 includes the first session's move
    assert paths.loc[1, 2025] == pytest.approx(-0.05)
    assert paths[2025].count() == len(pd.bdate_range("2025-01-01", "2025-03-31"))


def test_typical_path_excludes_year_and_uses_median():
    paths = pd.DataFrame({2022: [0.01, 0.02], 2023: [0.03, 0.04], 2024: [0.50, 0.60],
                          2025: [9.0, 9.0]}, index=[1, 2])
    t = typical_path(paths, exclude=2025)
    assert t.loc[1, "median"] == pytest.approx(0.03)
    assert t.loc[2, "years"] == 3


def test_turn_of_month_offsets():
    # Feb 2024: last session Thu 29th. Mar 2024: first session Fri 1st.
    s = _daily_closes("2024-01-31", "2024-03-29", {"2024-02-29": 0.02, "2024-03-01": 0.01,
                                                   "2024-02-28": -0.01})
    stats, avg_all = turn_of_month(s)
    assert stats.loc["T-1", "Avg"] == pytest.approx(0.01)  # Feb 29 and Mar 29 (0)
    assert stats.loc["T-2", "Avg"] == pytest.approx(-0.005)
    assert stats.loc["T+1", "Avg"] == pytest.approx(0.005)  # Feb 1 (0) and Mar 1
    assert stats.loc["T+1", "Days"] == 2
    assert stats.loc["T-1", "% Up"] == pytest.approx(0.5)
    assert avg_all == pytest.approx((0.02 + 0.01 - 0.01) / len(pd.bdate_range("2024-02-01", "2024-03-29")))


def test_turn_of_month_skips_partial_first_and_last_month():
    s = _daily_closes("2024-01-15", "2024-03-12", {"2024-01-31": 0.05, "2024-03-12": 0.03})
    stats, _ = turn_of_month(s)
    assert stats.loc["T-1", "Days"] == 1  # only Feb: Jan starts mid-month, Mar is unfinished
    assert stats.loc["T+1", "Days"] == 2  # Feb 1 and Mar 1 (Jan dropped)


def test_weekday_stats():
    s = _daily_closes("2023-12-29", "2024-01-31", {"2024-01-08": 0.02, "2024-01-15": -0.01})
    w = weekday_stats(s)
    assert list(w.index) == WEEKDAYS
    assert w.loc["Mon", "Avg"] == pytest.approx(0.01 / 5)  # 5 Mondays in Jan (Dec 29 is the base)
    assert w.loc["Mon", "% Up"] == pytest.approx(1 / 5)
    assert w.loc["Tue", "Avg"] == 0


def test_last_n_years_keeps_prior_year_end_base():
    s = _daily_closes("2020-01-01", "2024-06-28")
    t = last_n_years(s, 2)
    assert t.index[0] == pd.Timestamp("2022-12-30")
    assert t.index[1] == pd.Timestamp("2023-01-02")
    assert last_n_years(s, None).equals(s)
