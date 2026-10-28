import numpy as np
import pandas as pd
import pytest

from signals.seasonality import (MONTHS, WEEKDAYS, earnings_moves, earnings_reaction_day,
                                 event_flags, event_windows, forward_from_date, last_n_years,
                                 month_avg_row, month_outlook, month_stats, monthly_returns,
                                 seasonal_paths, seasonality_grid, third_friday, turn_of_month,
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


def test_forward_from_date_enters_on_or_before_same_date():
    # Oct 28 2023 is a Saturday -> enter at Fri Oct 27's close; 2024 enters Mon Oct 28.
    s = _daily_closes("2022-12-30", "2025-10-28", {"2023-10-30": 0.10, "2024-10-29": -0.05})
    by_year, summary = forward_from_date(s, horizons=(1, 5))
    assert list(by_year.index) == [2023, 2024]  # 2025 is the as-of year: excluded
    assert by_year.loc[2023, "1d"] == pytest.approx(0.10)
    assert by_year.loc[2024, "1d"] == pytest.approx(-0.05)
    assert summary.loc["1d", "% Up"] == pytest.approx(0.5)
    assert summary.loc["5d", "Worst"] == pytest.approx(-0.05)
    assert summary.loc["1d", "Years"] == 2


def test_forward_from_date_skips_years_before_data():
    s = _daily_closes("2024-11-15", "2025-10-28")
    by_year, _ = forward_from_date(s, horizons=(5,))
    assert by_year.empty  # data starts after Oct 28 2024


def test_third_friday():
    assert third_friday(2024, 3) == pd.Timestamp("2024-03-15")
    assert third_friday(2024, 11) == pd.Timestamp("2024-11-15")
    assert third_friday(2026, 5) == pd.Timestamp("2026-05-15")


def test_event_flags_holidays_and_opex():
    s = _daily_closes("2024-06-28", "2024-07-31")
    s = s.drop(pd.Timestamp("2024-07-04"))  # Independence Day
    df = event_flags(s)
    assert df.loc["2024-07-03", "pre_holiday"]
    assert df.loc["2024-07-05", "post_holiday"]
    assert df["pre_holiday"].sum() == 1  # Jul 31 is the last bar, not a holiday
    assert df.loc["2024-07-15":"2024-07-19", "opex_week"].all()  # Jul 19 = 3rd Friday
    assert not df.loc["2024-07-22", "opex_week"]
    assert df.loc["2024-07-22":"2024-07-26", "post_opex_week"].all()


def test_event_windows_quarter_and_all_days():
    s = _daily_closes("2024-05-31", "2024-07-31", {"2024-06-28": 0.02, "2024-07-01": 0.01})
    ev = event_windows(s)
    assert ev.loc["Quarter's last 5 days", "Days"] == 5
    assert ev.loc["Quarter's last 5 days", "Avg"] == pytest.approx(0.02 / 5)
    assert ev.loc["Quarter's first 5 days", "Avg"] == pytest.approx(0.01 / 5)
    assert ev.index[-1] == "All days"


def test_earnings_reaction_day():
    sessions = pd.bdate_range("2024-04-22", "2024-04-30")
    assert earnings_reaction_day(sessions, pd.Timestamp("2024-04-25 16:30")) == pd.Timestamp("2024-04-26")
    assert earnings_reaction_day(sessions, pd.Timestamp("2024-04-25 07:00")) == pd.Timestamp("2024-04-25")
    assert earnings_reaction_day(sessions, pd.Timestamp("2024-04-26 16:30")) == pd.Timestamp("2024-04-29")
    assert earnings_reaction_day(sessions, pd.Timestamp("2024-04-30 16:30")) is None


def test_earnings_moves_windows():
    s = _daily_closes("2024-03-01", "2024-05-31", {"2024-04-22": 0.05, "2024-04-26": 0.08,
                                                   "2024-05-06": -0.02})
    moves = earnings_moves(s, [pd.Timestamp("2024-04-25 16:05")], pre=5, post=20)
    row = moves.loc[pd.Timestamp("2024-04-26")]
    assert row["Reaction day"] == pytest.approx(0.08)
    assert row["5 days before"] == pytest.approx(0.05)  # Apr 18 close -> Apr 25 close
    assert row["20 days after"] == pytest.approx(-0.02)
    late = earnings_moves(s, [pd.Timestamp("2024-05-20 16:05")], pre=5, post=20)
    assert np.isnan(late.iloc[0]["20 days after"])


def test_month_avg_row_vs_benchmark():
    s = _closes({"2023-12": 100, "2024-01": 110, "2024-02": 110})
    b = _closes({"2023-12": 100, "2024-01": 105, "2024-02": 115.5})
    raw = month_avg_row(s, None)
    rel = month_avg_row(s, None, benchmark=b)
    assert raw["Jan"] == pytest.approx(0.10)
    assert rel["Jan"] == pytest.approx(0.05)
    assert rel["Feb"] == pytest.approx(-0.10)
    assert rel["Years"] == 1
