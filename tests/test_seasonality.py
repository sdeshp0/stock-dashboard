import numpy as np
import pandas as pd
import pytest

from signals.seasonality import (MONTHS, month_outlook, month_stats, monthly_returns,
                                 seasonality_grid, year_totals)


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
