"""S&P 500 Signal Dashboard - single-page Streamlit app.
Run: streamlit run app.py
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from signals.backtest import (
    BacktestParams, SIGNAL_GROUPS, equity_curve, get_signal_events, simulate_trades, summarize_trades,
)
from signals.data import (load_benchmark_close, load_earnings_dates, load_long_histories, load_long_history,
                          load_prices_safe, load_sp500_safe)
from signals.data_core import BENCHMARK_TICKER
from signals.indicators import compute_indicators
from signals.optimizer import HOLD_GRID, RANK_METRICS, optimize
from signals.seasonality import (FORWARD_HORIZONS, MONTHS, STAT_COLS, earnings_moves, event_windows,
                                 forward_from_date, last_n_years, month_avg_row, month_outlook,
                                 month_stats, monthly_returns, seasonal_paths, seasonality_grid,
                                 summarize_returns, turn_of_month, typical_path, weekday_stats,
                                 year_totals)
from signals.summary import (CONDITIONS, GROUPS, PRESETS, add_sector_relative_beta, add_summary,
                             build_table, condition_mask)
from signals.universes import COUNTRY_ETFS, SECTOR_ETFS

st.set_page_config(page_title="S&P 500 Signal Dashboard", page_icon="📈", layout="wide")
st.title("📈 S&P 500 Signal Dashboard")
st.caption("Technical-indicator screener")

PERIODS = {"6 months": "6mo", "1 year": "1y", "2 years": "2y", "5 years": "5y",
          "10 years": "10y", "Max": "max"}
OVERLAY_OPTIONS = {"SMA 10": "SMA_10", "SMA 50": "SMA_50", "SMA 100": "SMA_100",
                   "SMA 150": "SMA_150", "SMA 200": "SMA_200", "EMA 9": "EMA_9",
                   "EMA 18": "EMA_18", "VWAP": "VWAP", "Support": "Support",
                   "Resistance": "Resistance"}
DEFAULT_OVERLAYS = ["SMA 50", "SMA 200", "VWAP", "Support", "Resistance"]
PANEL_ORDER = ["Heikin Ashi", "RSI", "MACD", "ATR", "CMF", "CCI"]
OVERLAY_STYLE = {
    "SMA_10": dict(color="#1abc9c"), "SMA_50": dict(color="#f39c12"),
    "SMA_100": dict(color="#8e44ad"), "SMA_150": dict(color="#16a085"),
    "SMA_200": dict(color="#3498db"), "EMA_9": dict(color="#d35400", dash="dot"),
    "EMA_18": dict(color="#2c3e50", dash="dot"), "VWAP": dict(color="#e67e22", dash="dot"),
    "Support": dict(color="#2ecc71", dash="dash"), "Resistance": dict(color="#e74c3c", dash="dash"),
}


@st.cache_data(ttl=3600, show_spinner=False)
def get_table(tickers: tuple[str, ...], lookback: int, period: str, trend_sma: int):
    prices, source, asof = load_prices_safe(tickers, period)
    benchmark_close, bench_source, _bench_asof = load_benchmark_close(period)
    table = build_table(prices, lookback, benchmark_close, trend_sma)
    return table, source, asof, bench_source


@st.cache_data(ttl=3600, show_spinner=False, max_entries=20)
def run_optimizer(df, rsi_low, rsi_high, window, train_frac, min_trades, rank_by,
                  include_triples, stop_loss_pct, cost_bps):
    return optimize(df, rsi_low, rsi_high, window, train_frac, min_trades, rank_by,
                    include_triples, stop_loss_pct, cost_bps)


@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_data(ticker: str, tickers: tuple[str, ...], period: str):
    prices, source, asof = load_prices_safe(tickers, period)
    return compute_indicators(prices[ticker]), source, asof


def build_chart(df, ticker: str, rsi_low: int, rsi_high: int,
                overlays: list[str], panels: list[str]) -> go.Figure:
    panels = [p for p in PANEL_ORDER if p in panels]  # canonical order regardless of pick order
    n_panels = len(panels)
    heights = [0.55] + [0.45 / n_panels] * n_panels if n_panels else [1.0]
    fig = make_subplots(rows=1 + n_panels, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=heights)

    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
                                 close=df["Close"], name=ticker), row=1, col=1)
    for label in overlays:
        col = OVERLAY_OPTIONS[label]
        style = OVERLAY_STYLE.get(col, {})
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=label,
                                 line=dict(width=1.5, **style)), row=1, col=1)

    row = 2
    if "Heikin Ashi" in panels:
        fig.add_trace(go.Candlestick(x=df.index, open=df["HA_Open"], high=df["HA_High"],
                                     low=df["HA_Low"], close=df["HA_Close"], name="Heikin Ashi",
                                     increasing=dict(line=dict(color="#2ecc71"), fillcolor="#2ecc71"),
                                     decreasing=dict(line=dict(color="#e74c3c"), fillcolor="#e74c3c")),
                      row=row, col=1)
        for direction, symbol, color, y_col in ((1, "triangle-up", "#2ecc71", "HA_Low"),
                                                (-1, "triangle-down", "#e74c3c", "HA_High")):
            flips = df[df["HA_x"] == direction]
            pad = (df["HA_High"] - df["HA_Low"]).median() * 0.5  # nudge markers off the wicks
            y = flips[y_col] - pad if direction == 1 else flips[y_col] + pad
            fig.add_trace(go.Scatter(x=flips.index, y=y, mode="markers",
                                     name=f"HA turned {'up' if direction == 1 else 'down'}",
                                     marker=dict(symbol=symbol, color=color, size=9)),
                          row=row, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=row, col=1)
        row += 1
    if "RSI" in panels:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#9b59b6")),
                      row=row, col=1)
        fig.add_hline(y=rsi_high, line_dash="dot", line_color="#e74c3c", row=row, col=1)
        fig.add_hline(y=rsi_low, line_dash="dot", line_color="#2ecc71", row=row, col=1)
        row += 1
    if "MACD" in panels:
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="MACD hist",
                             marker_color="#95a5a6"), row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(width=1)),
                      row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", line=dict(width=1)),
                      row=row, col=1)
        row += 1
    if "ATR" in panels:
        fig.add_trace(go.Scatter(x=df.index, y=df["ATR"], name="ATR (14d)",
                                 line=dict(color="#7f8c8d")), row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ATR_SMA"], name="ATR avg (20d)",
                                 line=dict(color="#bdc3c7", dash="dot")), row=row, col=1)
        row += 1
    if "CMF" in panels:
        fig.add_trace(go.Scatter(x=df.index, y=df["CMF"], name="CMF (20d)",
                                 line=dict(color="#16a085"), fill="tozeroy"), row=row, col=1)
        fig.add_hline(y=0, line_color="#7f8c8d", row=row, col=1)
        row += 1
    if "CCI" in panels:
        fig.add_trace(go.Scatter(x=df.index, y=df["CCI"], name="CCI (20d)",
                                 line=dict(color="#2980b9")), row=row, col=1)
        fig.add_hline(y=0, line_color="#7f8c8d", row=row, col=1)
        fig.add_hline(y=100, line_dash="dot", line_color="#e74c3c", row=row, col=1)
        fig.add_hline(y=-100, line_dash="dot", line_color="#2ecc71", row=row, col=1)
        row += 1

    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(height=420 + 180 * n_panels, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.05))
    return fig


EARNINGS_PRE = [1, 5, 10]    # sessions before the reaction day (run-up)
EARNINGS_POST = [5, 20, 60]  # sessions after it (post-earnings drift)
SEASONALITY_YEARS = {"5 years": 5, "10 years": 10, "20 years": 20, "All": None}


def _heat(v, cap: float, center: float = 0.0) -> str:
    """Green/red cell shading, stronger the further v is from center (full at cap)."""
    if pd.isna(v) or v == center:
        return ""
    alpha = 0.08 + 0.6 * min(abs(v - center) / cap, 1.0)
    rgb = "46,204,113" if v > center else "231,76,60"
    return f"background-color: rgba({rgb},{alpha:.2f})"


def render_seasonality(ticker: str, fallback_df: pd.DataFrame) -> None:
    try:
        with st.spinner(f"Loading full price history for {ticker}…"):
            close = load_long_history(ticker)["Close"]
    except Exception as e:
        close = fallback_df["Close"]
        st.caption(f"⚠️ Couldn't load {ticker}'s full history ({e}), so this uses only the "
                   f"{len(close)} days loaded for the chart above.")
    rets, partial = monthly_returns(close)
    if rets.dropna().empty:
        st.info(f"Not enough history for {ticker} to compute monthly returns.")
        return

    years_label = st.radio("Years shown", list(SEASONALITY_YEARS), index=1, horizontal=True,
                           key="season_years")
    grid = seasonality_grid(rets, SEASONALITY_YEARS[years_label])
    last = rets.index[-1]
    partial_cell = (last.year, last.month) if partial else None
    stats = month_stats(grid, partial_cell)

    # One-line takeaway for the month the data ends in.
    out = month_outlook(stats, last.month)
    if out:
        line = (f"**{out['month']}** has averaged **{out['avg']:+.1%}** for {ticker} "
                f"(median {out['median']:+.1%}), up in {out['up']} of {out['years']} "
                f"year{'s' if out['years'] != 1 else ''}.")
        if partial:
            line += f" This {out['month']} so far: **{rets.iloc[-1]:+.1%}**."
        st.markdown(line)
    daily = last_n_years(close, SEASONALITY_YEARS[years_label])
    render_forward_outlook(daily, ticker)

    # Heatmap table: month x year, then per-month stats, with a calendar-year row underneath.
    year_cols = [str(y) for y in grid.columns]
    table = grid.set_axis(year_cols, axis=1).join(stats)
    table.loc["Year"] = pd.concat([year_totals(grid).set_axis(year_cols),
                                   pd.Series(float("nan"), index=STAT_COLS)])
    vals = grid.abs().stack()
    cap = max(float(vals.quantile(0.9)), 0.01) if len(vals) else 0.05
    totals = table.loc["Year", year_cols].abs().dropna()
    year_cap = max(float(totals.quantile(0.9)), 0.01) if len(totals) else 0.2
    # Shade from the numbers but show pre-formatted text, so gaps render blank instead of "None".
    heat_cols = year_cols + ["Avg", "Median"]
    css = pd.DataFrame("", index=table.index, columns=table.columns)
    css.loc[MONTHS, heat_cols] = table.loc[MONTHS, heat_cols].map(lambda v: _heat(v, cap))
    css.loc["Year", year_cols] = [_heat(v, year_cap) + "; font-weight: bold"
                                  for v in table.loc["Year", year_cols]]
    css.loc[MONTHS, "% Up"] = [_heat(v, 0.5, center=0.5) for v in table.loc[MONTHS, "% Up"]]
    text = pd.DataFrame("", index=table.index, columns=table.columns)
    for cols, spec in ((heat_cols, "{:+.1%}"), (["% Up"], "{:.0%}"), (["Years"], "{:.0f}")):
        # Round tiny values to 0 first so they don't print as "-0.0%".
        text[cols] = table[cols].map(lambda v, spec=spec: "" if pd.isna(v) else
                                     spec.format(0.0 if abs(v) < 5e-4 else v))
    if partial_cell and str(partial_cell[0]) in year_cols:
        text.loc[MONTHS[partial_cell[1] - 1], str(partial_cell[0])] += "*"
    styled = text.style.apply(lambda _: css, axis=None)
    st.dataframe(styled, width="stretch", height=35 * (len(table) + 1) + 3)
    notes = [("Close-to-close % return per calendar month on adjusted prices (splits and "
              "dividends included). Avg, Median and % Up are over the years shown, using "
              "completed months only.")]
    if partial:
        notes.append(f"{MONTHS[last.month - 1]} {last.year} is month-to-date (marked *).")
    notes.append("The Year row compounds the months shown in each column, so the first year "
                 "of data and the current year are partial.")
    st.caption(" ".join(notes))

    # Average month bar chart, with the median as a marker.
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in stats["Avg"].fillna(0)]
    custom = list(zip(stats["Median"], stats["% Up"], stats["Years"]))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=MONTHS, y=stats["Avg"], name="Average", marker_color=colors,
                         customdata=custom,
                         hovertemplate="%{x}: avg %{y:+.2%}<br>median %{customdata[0]:+.2%}"
                                       "<br>up %{customdata[1]:.0%} of %{customdata[2]} years"
                                       "<extra></extra>"))
    fig.add_trace(go.Scatter(x=MONTHS, y=stats["Median"], name="Median", mode="markers",
                             marker=dict(symbol="diamond", size=9, color="#34495e"),
                             hoverinfo="skip"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis=dict(tickformat=".1%", title="Return"),
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.12),
                      title=dict(text=f"Average month, {years_label.lower()}", x=0, font=dict(size=14)))
    fig.add_hline(y=0, line_color="#7f8c8d")
    st.plotly_chart(fig, width="stretch")

    render_typical_year(daily, ticker, years_label)
    render_calendar_effects(daily, years_label)
    render_event_windows(daily, ticker, years_label)

    st.download_button(f"📥 Download {ticker} monthly returns",
                       table.to_csv(index_label="Month").encode("utf-8"),
                       file_name=f"{ticker}_monthly_returns.csv", mime="text/csv")


def _bar_colors(values) -> list[str]:
    return ["#2ecc71" if v > 0 else "#e74c3c" for v in pd.Series(values).fillna(0)]


def render_typical_year(close: pd.Series, ticker: str, years_label: str) -> None:
    """Median cumulative path through the year across completed years, with an
    interquartile band, and the current year drawn on top."""
    paths = seasonal_paths(close)
    this_year = close.index[-1].year
    typ = typical_path(paths, exclude=this_year)
    # Drop the ragged tail where only a year or two have that many sessions.
    typ = typ[typ["years"] >= max(2, int(typ["years"].max()) // 2)] if len(typ) else typ
    if typ.empty:
        st.caption("Not enough complete years for a typical-year path.")
        return
    # Trading day n plotted at the n-th weekday of the current year: close enough to
    # read months off the axis (holidays shift it by a few days at most).
    n_days = int(max(typ.index.max(), paths[this_year].count() if this_year in paths else 0))
    x_of = pd.Series(pd.bdate_range(f"{this_year}-01-01", periods=n_days), index=range(1, n_days + 1))
    x = x_of[typ.index]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=typ["q75"], line=dict(width=0), hoverinfo="skip",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=typ["q25"], line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(52,152,219,.15)", name="Middle half of years",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=typ["median"], name="Typical year (median)",
                             line=dict(color="#3498db", width=2), customdata=list(typ.index),
                             hovertemplate="day %{customdata}: %{y:+.1%}<extra>median</extra>"))
    if this_year in paths:
        cur = paths[this_year].dropna()
        fig.add_trace(go.Scatter(x=x_of[cur.index], y=cur, name=str(this_year),
                                 line=dict(color="#e67e22", width=2.5), customdata=list(cur.index),
                                 hovertemplate="day %{customdata}: %{y:+.1%}<extra>"
                                               f"{this_year}</extra>"))
    fig.add_hline(y=0, line_color="#7f8c8d")
    fig.update_layout(height=370, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis=dict(tickformat=".0%", title="Return since prior year-end"),
                      xaxis=dict(tickformat="%b", dtick="M1"),
                      legend=dict(orientation="h", x=0, y=-0.12),
                      title=dict(text=f"Typical year, {years_label.lower()}", x=0, font=dict(size=14)))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Each year's return since the previous year's last close, by trading day. The "
               f"blue line is the median across {int(typ['years'].max())} completed years and the "
               f"band covers the middle half of them; {this_year} is drawn on top.")


def render_calendar_effects(close: pd.Series, years_label: str) -> None:
    """Average daily return around month end, and by weekday."""
    tom, avg_all = turn_of_month(close)
    wd = weekday_stats(close)
    if tom["Days"].fillna(0).sum() == 0:
        return
    hover = "%{x}: avg %{y:+.3%}<br>up %{customdata[0]:.0%} of %{customdata[1]} days<extra></extra>"
    c1, c2 = st.columns(2)
    for col, stats, title in ((c1, tom, "Around month end"), (c2, wd, "By weekday")):
        fig = go.Figure(go.Bar(x=list(stats.index), y=stats["Avg"], marker_color=_bar_colors(stats["Avg"]),
                               customdata=list(zip(stats["% Up"], stats["Days"])),
                               hovertemplate=hover, showlegend=False))
        fig.add_hline(y=0, line_color="#7f8c8d")
        fig.add_hline(y=avg_all, line_dash="dot", line_color="#34495e",
                      annotation_text="all days", annotation_position="top right")
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10),
                          yaxis=dict(tickformat=".2%", title="Avg daily return"),
                          xaxis=dict(type="category"),
                          title=dict(text=f"{title}, {years_label.lower()}", x=0, font=dict(size=14)))
        col.plotly_chart(fig, width="stretch")
    st.caption("Average one-day return. Left: the last five sessions of each month (T-5 to T-1, "
               "T-1 = last day) and the first five of the next (T+1 = first day). Right: by "
               "weekday. The dotted line is the average of all days. These effects are small "
               "next to daily noise and trading costs, and have faded since they became widely "
               "known - read them as a description of the past, not a signal.")


def _fmt(v, spec: str) -> str:
    """Blank for NaN, and a value that rounds to zero prints as zero, not '-0.0%'."""
    if pd.isna(v):
        return ""
    out, zero = spec.format(v), spec.format(0.0)
    return zero if out.lstrip("+-") == zero.lstrip("+-") else out


def _heat_frame(values: pd.DataFrame, specs: dict[str, str], heat: dict[str, tuple[float, float]]):
    """Pre-formatted text (blank for NaN) with green/red shading from the
    numbers. `specs` maps column -> format, `heat` maps column -> (cap, center)."""
    text = pd.DataFrame("", index=values.index, columns=values.columns)
    css = pd.DataFrame("", index=values.index, columns=values.columns)
    for c, spec in specs.items():
        text[c] = values[c].map(lambda v, spec=spec: _fmt(v, spec))
    for c, (cap, center) in heat.items():
        css[c] = [_heat(v, cap, center) for v in values[c]]
    return text.style.apply(lambda _: css, axis=None)


def _cap(values, floor: float) -> float:
    vals = pd.DataFrame(values).abs().stack().dropna()
    return max(float(vals.quantile(0.9)), floor) if len(vals) else floor


def render_forward_outlook(close: pd.Series, ticker: str) -> None:
    """How the next 5/10/20/60 sessions went in past years from today's date."""
    by_year, summary = forward_from_date(close)
    if by_year.empty:
        return
    # Built by hand: strftime's "%-d" (no leading zero) is Linux/macOS-only and fails on Windows.
    as_of = close.index[-1]
    date = f"{as_of:%b} {as_of.day}"
    st.markdown(f"**From {date}: how the next few weeks went in past years**")
    summary.index = [f"Next {h} days" for h in FORWARD_HORIZONS]
    ret_cols = ["Avg", "Median", "Best", "Worst"]
    cap = _cap(summary[ret_cols], 0.01)
    st.dataframe(_heat_frame(summary, {**{c: "{:+.1%}" for c in ret_cols}, "% Up": "{:.0%}",
                                       "Years": "{:.0f}"},
                             {**{c: (cap, 0.0) for c in ret_cols}, "% Up": (0.5, 0.5)}),
                 width="stretch", height=35 * (len(summary) + 1) + 3)
    if st.toggle("Show each year", key="season_fwd_years"):
        yr = by_year.sort_index(ascending=False)
        yr.index = yr.index.astype(str)
        st.dataframe(_heat_frame(yr, dict.fromkeys(yr.columns, "{:+.1%}"),
                                 {c: (_cap(yr[c], 0.01), 0.0) for c in yr.columns}),
                     width="stretch")
    st.caption(f"Each past year enters at the close of its last session on or before "
               f"{date} and holds for that many trading sessions. 'Best' and 'Worst' "
               f"are the single best and worst years, which says more about risk than the average.")


def render_event_windows(close: pd.Series, ticker: str, years_label: str) -> None:
    """Average daily return around recurring calendar events, and around
    earnings releases for tickers that have them."""
    ev = event_windows(close)
    ev = ev[ev["Days"] > 0]  # e.g. no holidays inside a short history
    if len(ev) < 2:
        return
    st.markdown(f"**Around recurring events, {years_label.lower()}**")
    all_avg = ev.loc["All days", "Avg"]
    c1, c2 = st.columns([3, 2])
    fig = go.Figure(go.Bar(y=list(ev.index[:-1])[::-1], x=ev["Avg"].iloc[:-1][::-1], orientation="h",
                           marker_color=_bar_colors(ev["Avg"].iloc[:-1][::-1]),
                           customdata=list(zip(ev["% Up"].iloc[:-1][::-1], ev["Days"].iloc[:-1][::-1])),
                           hovertemplate="%{y}: avg %{x:+.3%}<br>up %{customdata[0]:.0%} of "
                                         "%{customdata[1]} days<extra></extra>"))
    fig.add_vline(x=0, line_color="#7f8c8d")
    fig.add_vline(x=all_avg, line_dash="dot", line_color="#34495e",
                  annotation_text="all days", annotation_position="top")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(tickformat=".2%", title="Avg daily return"))
    c1.plotly_chart(fig, width="stretch")
    # Shaded relative to the all-days row, which (being the center) stays unshaded.
    c2.dataframe(_heat_frame(ev, {"Avg": "{:+.3%}", "% Up": "{:.0%}", "Days": "{:.0f}"},
                             {"Avg": (_cap(ev["Avg"] - all_avg, 0.0005), all_avg),
                              "% Up": (0.1, ev.loc["All days", "% Up"])}),
                 width="stretch", height=35 * (len(ev) + 1) + 3)
    st.caption("Average one-day return on sessions in each window. Holidays are read from the "
               "price data (a weekday with no trading between two sessions). Options expiry is the "
               "third Friday of each month; its week is that Monday to Friday.")

    dates = load_earnings_dates(ticker)
    if dates is None:
        st.caption(f"Couldn't load earnings dates for {ticker} right now.")
        return
    if not dates:
        return  # ETFs and anything else without an earnings calendar
    st.markdown("**Around earnings**")
    c1, c2 = st.columns(2)
    pre = c1.radio("Days before", EARNINGS_PRE, index=1, horizontal=True, key="season_earn_pre")
    post = c2.radio("Days after", EARNINGS_POST, index=1, horizontal=True, key="season_earn_post")
    moves = earnings_moves(close, dates, pre=pre, post=post)
    if moves.empty:
        st.caption("No earnings releases within the years shown.")
        return
    summ = summarize_returns(moves).rename(columns={"Years": "Releases"})
    ret_cols = ["Avg", "Median", "Best", "Worst"]
    st.dataframe(_heat_frame(summ, {**{c: "{:+.1%}" for c in ret_cols}, "% Up": "{:.0%}",
                                    "Releases": "{:.0f}"},
                             {**{c: (_cap(summ[ret_cols], 0.01), 0.0) for c in ret_cols},
                              "% Up": (0.5, 0.5)}),
                 width="stretch", height=35 * (len(summ) + 1) + 3)
    if st.toggle("Show each release", key="season_earn_each"):
        each = moves.sort_index(ascending=False)
        each.index = each.index.strftime("%Y-%m-%d")
        st.dataframe(_heat_frame(each, dict.fromkeys(each.columns, "{:+.1%}"),
                                 {c: (_cap(each[c], 0.01), 0.0) for c in each.columns}),
                     width="stretch")
    st.caption("The reaction day is the first session to trade on the release (the same day for a "
               "release before the open, the next day for one after the close). 'Days before' runs "
               "up to the close before it, 'days after' from its close.")


def render_sector_calendar() -> None:
    """Average return by calendar month for each sector ETF, raw or vs SPY."""
    st.caption("Average return in each calendar month for the sector ETFs, to see which have "
               "tended to lead at each point in the year.")
    c1, c2 = st.columns(2)
    years_label = c1.radio("Years", list(SEASONALITY_YEARS), index=1, horizontal=True,
                           key="sector_cal_years")
    vs_spy = c2.radio("Show", ["Raw return", "vs SPY"], horizontal=True, key="sector_cal_mode") == "vs SPY"
    tickers = tuple(sorted(SECTOR_ETFS)) + (BENCHMARK_TICKER,)
    try:
        with st.spinner("Loading sector ETF history…"):
            hist = load_long_histories(tickers)
    except Exception as e:
        st.error(f"Couldn't load sector ETF history: {e}")
        return
    bench = hist[BENCHMARK_TICKER]["Close"] if BENCHMARK_TICKER in hist else None
    if vs_spy and bench is None:
        st.warning("SPY history unavailable, so showing raw returns.")
        vs_spy = False
    rows = {f"{t} {SECTOR_ETFS[t]}": month_avg_row(hist[t]["Close"], SEASONALITY_YEARS[years_label],
                                                   bench if vs_spy else None)
            for t in sorted(SECTOR_ETFS) if t in hist}
    if not rows:
        st.info("No sector ETF history came back.")
        return
    table = pd.DataFrame(rows).T
    if not vs_spy and bench is not None:
        table.loc[f"{BENCHMARK_TICKER} S&P 500"] = month_avg_row(bench, SEASONALITY_YEARS[years_label])
    last = max(h.index[-1] for h in hist.values())
    now, nxt = MONTHS[last.month - 1], MONTHS[last.month % 12]
    st.dataframe(_heat_frame(table, {**dict.fromkeys(MONTHS, "{:+.1%}"), "Years": "{:.0f}"},
                             {m: (_cap(table[MONTHS], 0.005), 0.0) for m in MONTHS})
                 .set_properties(subset=[now, nxt], **{"font-weight": "bold"}),
                 width="stretch", height=35 * (len(table) + 1) + 3)
    ranked = table[nxt].dropna().sort_values(ascending=False)
    if len(ranked) >= 3:
        what = "vs SPY" if vs_spy else "on average"
        st.markdown(f"Historically strongest in **{nxt}** ({what}): "
                    + ", ".join(f"{k.split()[0]} {v:+.1%}" for k, v in ranked.head(3).items())
                    + ". Weakest: "
                    + ", ".join(f"{k.split()[0]} {v:+.1%}" for k, v in ranked.tail(3)[::-1].items()) + ".")
    st.caption(f"{'Each ETF minus SPY in the same month' if vs_spy else 'Raw monthly returns'}, "
               f"averaged over completed months in the {years_label.lower()} window; {now} and {nxt} "
               "are in bold. Years = most years any month has (XLRE and XLC are younger funds).")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Universe")
    mode = st.radio("Select tickers by",
                    ["S&P 500 sectors", "US Sector ETFs", "Country ETFs", "Custom list"],
                    label_visibility="collapsed")
    meta = None
    if mode == "S&P 500 sectors":
        try:
            meta, meta_source, meta_asof = load_sp500_safe()
        except Exception as e:
            st.error(f"Could not load the S&P 500 list (live or snapshot): {e}")
            st.stop()
        if meta_source == "snapshot":
            st.caption(f"⚠️ Using a cached constituent list from {meta_asof or 'an earlier run'} "
                      "(Wikipedia unreachable).")
        sectors = sorted(meta["GICS Sector"].unique())
        chosen = st.multiselect("Sectors", sectors, default=["Information Technology"])
        tickers = tuple(sorted(meta.loc[meta["GICS Sector"].isin(chosen), "Ticker"]))
    elif mode == "US Sector ETFs":
        chosen = st.multiselect("Sector ETFs", list(SECTOR_ETFS.values()),
                                default=list(SECTOR_ETFS.values()))
        tickers = tuple(sorted(t for t, label in SECTOR_ETFS.items() if label in chosen))
        meta = pd.DataFrame({"Ticker": list(SECTOR_ETFS), "Security": list(SECTOR_ETFS.values())})
    elif mode == "Country ETFs":
        chosen = st.multiselect("Country ETFs", list(COUNTRY_ETFS.values()),
                                default=list(COUNTRY_ETFS.values()))
        tickers = tuple(sorted(t for t, label in COUNTRY_ETFS.items() if label in chosen))
        meta = pd.DataFrame({"Ticker": list(COUNTRY_ETFS), "Security": list(COUNTRY_ETFS.values())})
    else:
        raw = st.text_area("Comma-separated tickers", "AAPL, MSFT, NVDA, AMZN, TSLA, JPM")
        tickers = tuple(sorted({t.strip().upper().replace(".", "-") for t in raw.split(",") if t.strip()}))

    period_label = st.selectbox("History length", list(PERIODS), index=1,
                                help="How much price history to fetch. Longer history needed for "
                                     "e.g. SMA 200 to have a full warm-up, but is slower to load "
                                     "and, if Yahoo is unreachable, the offline snapshot only has 1 year.")
    period = PERIODS[period_label]

    st.header("Signals")
    rsi_low, rsi_high = st.slider("RSI oversold / overbought", 5, 95, (30, 70))
    lookback = st.slider("Crossover lookback (days)", 1, 10, 3,
                         help="A crossover counts if it happened within this many recent days.")
    trend_sma = int(st.number_input(
        "Trend SMA length (days)", min_value=5, max_value=250, value=150, step=5,
        help="The SMA used for the days-above/below counts, their highlights and screens, "
             "and the above/below headline numbers. Independent of the fixed SMA150 "
             "crossover signal."))
    days_above_min = int(st.number_input(
        f"Min days above SMA{trend_sma} (X)", min_value=1, max_value=1000, value=20, step=1,
        help=f"Highlights tickers whose close has been above the {trend_sma}-day SMA for at "
             "least this many trading days in a row, and drives the 'Above trend SMA for X+ "
             "days' screen. Counts can't exceed the history loaded after the SMA's warm-up "
             "(about 250 trading days in 1 year, minus the SMA length)."))
    days_below_min = int(st.number_input(
        f"Min days below SMA{trend_sma} (Y)", min_value=1, max_value=1000, value=20, step=1,
        help=f"Same as X, for closes below the {trend_sma}-day SMA. Drives the 'Below trend "
             "SMA for Y+ days' screen."))
    preset_category = st.selectbox("Screen category", list(GROUPS))
    preset = st.selectbox("Screen", GROUPS[preset_category])
    conditions = st.multiselect(
        "Build your own screen", list(CONDITIONS), placeholder="Add conditions…",
        help="Combine any of these signals. Applied on top of the Screen above (leave Screen "
             "on Overall → All to use only these). Crossovers count within the crossover "
             "lookback; RSI uses the oversold/overbought slider; the trend-SMA conditions "
             "use X and Y.")
    match = "all"
    if len(conditions) > 1:
        match = "all" if st.radio("Match", ["All conditions", "Any condition"], horizontal=True,
                                  label_visibility="collapsed") == "All conditions" else "any"
    query = st.text_input("Filter by signal text (optional)",
                          placeholder="e.g. breakout, VWAP, RSI high",
                          help="Further narrows the table to rows whose Signals text contains this.")

    st.header("Chart")
    overlays = st.multiselect("Overlay lines", list(OVERLAY_OPTIONS), default=DEFAULT_OVERLAYS)
    panels = st.multiselect("Lower panels", PANEL_ORDER, default=["RSI", "MACD"])

if not tickers:
    st.info("Pick at least one sector or enter some tickers.")
    st.stop()
if len(tickers) > 150 or period in ("5y", "10y", "max"):
    st.caption(f"{len(tickers)} tickers, {period_label.lower()} of history selected - "
              "the first load may take a little while.")

# ---------------- Data ----------------
try:
    with st.spinner(f"Loading {len(tickers)} tickers…"):
        table, data_source, data_asof, bench_source = get_table(tickers, lookback, period, trend_sma)
except Exception as e:
    st.error(f"Data load failed (live and snapshot both unavailable): {e}")
    st.stop()

if data_source == "nightly":
    st.caption(f"Prices from the nightly snapshot ({data_asof}), the latest completed trading day.")
elif data_source == "fallback":
    st.warning(f"⚠️ Live price data is unavailable right now (Yahoo Finance may be rate-limiting "
              f"or unreachable). Showing a cached snapshot from **{data_asof or 'an earlier run'}**.")
if bench_source == "unavailable":
    st.caption("⚠️ Benchmark (SPY) data is unavailable right now, so the Beta column will be blank.")

missing = sorted(set(tickers) - set(table.index))
if missing:
    with st.expander(f"⚠️ {len(missing)} ticker(s) returned no data"):
        st.code(", ".join(missing))

table = add_summary(table, rsi_low, rsi_high, days_above_min, days_below_min, trend_sma)
if meta is not None:
    join_cols = [c for c in ("Security", "GICS Sector") if c in meta.columns]
    table = table.join(meta.set_index("Ticker")[join_cols])
table = add_sector_relative_beta(table)  # NaN if no sector info (custom ticker list) or too few peers

view = table[PRESETS[preset](table, rsi_low, rsi_high)
             & condition_mask(table, conditions, rsi_low, rsi_high, match)]
if query:
    view = view[view["Signals"].str.contains(query, case=False, na=False, regex=False)]

# ---------------- Summary metrics ----------------
r1c1, r1c2, r1c3, r1c4 = st.columns(4)
r1c1.metric("Tickers shown", f"{len(view)} / {len(table)}")
r1c2.metric("Bullish bias", int((table["Bias"] == "Bullish").sum()))
r1c3.metric("Bearish bias", int((table["Bias"] == "Bearish").sum()))
r1c4.metric("Above VWAP", f"{(table['Close'] > table['VWAP']).mean():.0%}")

r2c1, r2c2, r2c3, r2c4 = st.columns(4)
r2c1.metric("Resistance breakouts", int((table["Resistance_x"] > 0).sum()))
r2c2.metric("Support breakdowns", int((table["Support_x"] < 0).sum()))
r2c3.metric("RSI oversold", int((table["RSI"] < rsi_low).sum()))
r2c4.metric("RSI overbought", int((table["RSI"] > rsi_high).sum()))

r3c1, r3c2, r3c3, r3c4 = st.columns(4)
r3c1.metric(f"Above SMA{trend_sma}", int((table["Close"] > table["Trend_SMA"]).sum()),
            help=f"Tickers closing above their {trend_sma}-day SMA on the latest bar, of any "
                 f"duration. {int(table['SMA_hold_above'].sum())} of them for {days_above_min}+ days.")
r3c2.metric(f"Below SMA{trend_sma}", int((table["Close"] < table["Trend_SMA"]).sum()),
            help=f"Tickers closing below their {trend_sma}-day SMA on the latest bar, of any "
                 f"duration. {int(table['SMA_hold_below'].sum())} of them for {days_below_min}+ days. "
                 "Tickers without enough history for the SMA count in neither.")

st.caption(f"Latest bar: {table['Date'].max():%Y-%m-%d}. Prices are split/dividend adjusted; "
          "data via Yahoo Finance, cached for 1 hour. VWAP is a 20-day rolling volume-weighted "
          "average (an approximation - true VWAP needs intraday data). Support/resistance are "
          "20-day Donchian-style channel levels. Beta is trailing, vs. SPY, over up to the last "
          "252 trading days (fewer if less history is available; blank if there isn't enough "
          "overlapping data to estimate it). 'Beta vs Sector' compares it to a leave-one-out "
          "average of its sector peers *currently loaded in this table* (blank with a custom "
          "ticker list, or with fewer than 3 sector peers loaded).")

# ---------------- Table ----------------
X = {1: "↑ Bull", -1: "↓ Bear", 0: "–"}
RX = {1: "↑ Breakout", -1: "↓ Back below", 0: "–"}
SX = {1: "↑ Bounce", -1: "↓ Breakdown", 0: "–"}
AX = {1: "↑ Expanding", -1: "↓ Contracting", 0: "–"}
show = view.assign(
    MACD_x=view["MACD_x"].map(X), SMA_x=view["SMA_x"].map(X), CCI_x=view["CCI_x"].map(X),
    VWAP_x=view["VWAP_x"].map(X), EMA_x=view["EMA_x"].map(X), SMA150_x=view["SMA150_x"].map(X),
    CMF_x=view["CMF_x"].map(X),
    Resistance_x=view["Resistance_x"].map(RX), Support_x=view["Support_x"].map(SX),
    ATR_x=view["ATR_x"].map(AX), HA_x=view["HA_x"].map(X),
)
cols = ["Security", "GICS Sector", "Close", "Ret_1d", "Ret_5d", "Ret_1m", "RSI",
        "MACD_x", "SMA_x", "CCI_x", "VWAP_x", "EMA_x", "CMF_x", "HA_x",
        "Support", "Resistance", "VWAP", "SMA_150", "ATR", "CMF",
        "Support_x", "Resistance_x", "SMA150_x", "Trend_SMA", "Days_Above_SMA", "Days_Below_SMA",
        "ATR_x",
        "Streak", "Vol_Index", "Beta", "Sector Beta", "Beta vs Sector",
        "From_52w_High", "Above_SMA200", "Bias", "Signals"]
if trend_sma == 150:
    cols.remove("Trend_SMA")  # same numbers as the SMA150 column
show = show[[c for c in cols if c in show.columns]].rename(columns={
    "Trend_SMA": f"SMA{trend_sma}", "Days_Above_SMA": f"Days > SMA{trend_sma}",
    "Days_Below_SMA": f"Days < SMA{trend_sma}",
    "Ret_1d": "1d", "Ret_5d": "5d", "Ret_1m": "1m", "MACD_x": "MACD", "SMA_x": "SMA 10/50",
    "CCI_x": "CCI", "VWAP_x": "VWAP x", "EMA_x": "EMA 9/18", "SMA_150": "SMA150",
    "SMA150_x": "SMA150 x", "Support_x": "Support x", "Resistance_x": "Resistance x",
    "ATR_x": "ATR x", "CMF_x": "CMF x", "HA_x": "HA x", "Vol_Index": "Vol vs 50d", "From_52w_High": "vs 52w high",
    "Above_SMA200": "> SMA200", "GICS Sector": "Sector"})


def _rsi_style(v):
    if pd.isna(v):
        return ""
    if v > rsi_high:
        return "background-color: rgba(231,76,60,.35)"
    return "background-color: rgba(46,204,113,.35)" if v < rsi_low else ""


def _days_above_style(v):
    if pd.isna(v) or v < days_above_min:
        return ""
    return "background-color: rgba(46,204,113,.45); font-weight: bold"


def _days_below_style(v):
    if pd.isna(v) or v < days_below_min:
        return ""
    return "background-color: rgba(231,76,60,.45); font-weight: bold"


def _ret_style(v):
    return "" if pd.isna(v) else ("color: #2ecc71" if v > 0 else "color: #e74c3c" if v < 0 else "")


def _row_style(row):
    color = {"Bullish": "background-color: rgba(46,204,113,.12)",
             "Bearish": "background-color: rgba(231,76,60,.12)"}.get(row.get("Bias"), "")
    return [color] * len(row)


fmt = {"Close": "{:.2f}", "1d": "{:+.2%}", "5d": "{:+.2%}", "1m": "{:+.2%}", "RSI": "{:.0f}",
       "Support": "{:.2f}", "Resistance": "{:.2f}", "VWAP": "{:.2f}", "SMA150": "{:.2f}",
       "ATR": "{:.2f}", "Beta": "{:.2f}", "Sector Beta": "{:.2f}", "Beta vs Sector": "{:.2f}x",
       "CMF": "{:.2f}", "Vol vs 50d": "{:.2f}x", "vs 52w high": "{:+.1%}", "Streak": "{:+d}",
       f"SMA{trend_sma}": "{:.2f}", f"Days > SMA{trend_sma}": "{:.0f}",
       f"Days < SMA{trend_sma}": "{:.0f}"}
styled = (show.style.apply(_row_style, axis=1)
          .map(_rsi_style, subset=["RSI"])
          .map(_days_above_style, subset=[f"Days > SMA{trend_sma}"])
          .map(_days_below_style, subset=[f"Days < SMA{trend_sma}"]).map(_ret_style, subset=["1d", "5d", "1m"])
          .format({k: v for k, v in fmt.items() if k in show.columns}))

st.subheader("Signals")
st.caption("Rows are tinted by overall Bias (net of all crossovers + streak). "
          f"'Days > SMA{trend_sma}' is highlighted green at {days_above_min}+ closes in a row "
          f"above the {trend_sma}-day SMA, and 'Days < SMA{trend_sma}' red at {days_below_min}+ "
          f"below it (both blank until the SMA has {trend_sma} days of data). "
          "Use the Screen preset or the signal-text filter in the sidebar to narrow further.")
event = st.dataframe(styled, on_select="rerun", selection_mode="single-row",
                     width="stretch", height=430)
st.download_button("📥 Download CSV", show.reset_index().to_csv(index=False).encode("utf-8"),
                   file_name="sp500_signals.csv", mime="text/csv")

# ---------------- Drill-down ----------------
rows = event.selection.rows
if rows:
    sel = show.index[rows[0]]
    label = f"{sel} - {show.loc[sel, 'Security']}" if "Security" in show.columns else sel
    st.subheader(label)
    try:
        df_full, chart_source, chart_asof = get_ticker_data(sel, tickers, period)
    except Exception as e:
        st.error(f"Could not load chart data for {sel}: {e}")
    else:
        if chart_source == "fallback":
            st.caption(f"⚠️ Chart is from the cached snapshot ({chart_asof or 'an earlier run'}), "
                      f"limited to ~1 year of history regardless of the History length setting.")
        if overlays or panels:
            st.plotly_chart(build_chart(df_full, sel, rsi_low, rsi_high, overlays, panels),
                            width="stretch")
        else:
            st.info("No overlay lines or panels selected - pick some under Chart in the sidebar.")
        st.download_button(f"📥 Download all data for {sel}",
                           df_full.reset_index().rename(columns={"index": "Date"})
                           .to_csv(index=False).encode("utf-8"),
                           file_name=f"{sel}_data.csv", mime="text/csv")

        # Expensive (full-history download), so only built while the expander is open.
        season = st.expander(f"📅 Seasonality for {sel}", key="season_exp", on_change="rerun")
        with season:
            if season.open:
                render_seasonality(sel, df_full)

        # ---------------- Backtest ----------------
        with st.expander(f"📊 Backtest a signal on {sel}"):
            st.caption("Single-ticker historical simulation over the loaded history, not "
                      "investment advice. Trade counts can be small, and thresholds you can "
                      "freely tune elsewhere in this app (RSI levels, lookback) make it easy "
                      "to overfit a backtest without realizing it - treat this as exploratory.")

            b1, b2 = st.columns(2)
            bt_cat = b1.selectbox("Signal category", list(SIGNAL_GROUPS), key="bt_cat")
            bt_signal = b2.selectbox("Signal", SIGNAL_GROUPS[bt_cat], key="bt_signal")

            b3, b4, b5 = st.columns(3)
            bt_dir_label = b3.selectbox("Direction", ["Both", "Long only", "Short only"], key="bt_dir")
            bt_exit_label = b4.selectbox("Exit rule", ["Fixed holding period", "Until opposite signal"],
                                        key="bt_exit")
            bt_stop_pct = b5.number_input("Stop loss %  (0 = off)", 0.0, 50.0, 0.0, 0.5, key="bt_stop")

            bt_hold = st.slider("Holding period (days)", 1, 60, 10, key="bt_hold",
                                help="The exit duration when using 'Fixed holding period'. Also "
                                     "applied as a maximum hold cap when using 'Until opposite "
                                     "signal', so a trade can't run forever if the signal never "
                                     "reverses.")
            bt_cost = st.number_input("Round-trip cost (bps, 0 = off)", 0.0, 200.0, 0.0, 5.0, key="bt_cost")

            events = get_signal_events(bt_signal, df_full, rsi_low, rsi_high)
            params = BacktestParams(
                direction={"Both": "both", "Long only": "long", "Short only": "short"}[bt_dir_label],
                use_signal_exit=(bt_exit_label == "Until opposite signal"),
                hold_days=bt_hold, stop_loss_pct=bt_stop_pct / 100, cost_bps=bt_cost,
            )
            trades = simulate_trades(df_full, events, params)
            stats = summarize_trades(trades, df_full)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Trades", stats["n_trades"])
            m2.metric("Win rate", f"{stats['win_rate']:.0%}" if stats["n_trades"] else "–")
            m3.metric("Avg return / trade", f"{stats['avg_return']:+.2%}" if stats["n_trades"] else "–")
            m4.metric("Max drawdown", f"{stats['max_drawdown']:.2%}" if stats["n_trades"] else "–")

            m5, m6 = st.columns(2)
            m5.metric("Total compounded return", f"{stats['total_compounded_return']:+.2%}"
                     if stats["n_trades"] else "–")
            m6.metric(f"Buy & hold over same {period_label.lower()}", f"{stats['buy_and_hold_return']:+.2%}")

            if stats["n_trades"] == 0:
                st.info("This signal didn't fire (with these settings) over the selected history - "
                       "try a different signal, a longer History length, or a wider RSI band.")
            else:
                curve = equity_curve(trades)
                bh_curve = df_full["Close"] / df_full["Close"].iloc[0] - 1
                st.caption("Trade-sequence equity (steps only on trade exits) vs. continuously-held "
                          "buy-and-hold over the same period - the two aren't on the same time "
                          "basis, so treat this as a rough visual comparison, not an exact overlay.")
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=bh_curve.index, y=bh_curve, name="Buy & hold",
                                            line=dict(color="#95a5a6", dash="dot")))
                fig_bt.add_trace(go.Scatter(x=curve.index, y=curve, name=f"{bt_signal} strategy",
                                            mode="lines+markers", line=dict(color="#2ecc71")))
                fig_bt.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10),
                                     yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_bt, width="stretch")

                st.dataframe(
                    trades.style.format({"entry_price": "{:.2f}", "exit_price": "{:.2f}",
                                        "gross_return": "{:+.2%}", "net_return": "{:+.2%}"}),
                    width="stretch", height=250)
                st.download_button(f"📥 Download trade log ({bt_signal})",
                                   trades.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{sel}_{bt_signal.replace(' ', '_')}_trades.csv",
                                   mime="text/csv")

        # ---------------- Signal optimizer ----------------
        with st.expander(f"🔎 Find the best signals for {sel}"):
            st.caption(
                "Backtests every signal, and every pair (optionally triple) of signals firing in the "
                "same direction within the crossover lookback, long-only, short-only and both ways, "
                f"under both exit rules, at holding periods of {', '.join(map(str, HOLD_GRID))} days. "
                "Uses the stop loss and cost set in the backtest above. Settings are picked on the "
                "first part of the history (train) and re-checked unchanged on the rest (test). "
                "Searching this many variants of one ticker's history always finds something that "
                "looked great in hindsight - trust rows whose test results hold up, with a "
                "reasonable number of trades, and prefer longer History lengths.")
            o1, o2, o3, o4 = st.columns(4)
            opt_rank = o1.selectbox("Rank by", list(RANK_METRICS), key="opt_rank",
                                    help="Also the metric used to pick each row's best holding period.")
            opt_min_trades = int(o2.number_input("Min trades (train)", 2, 100, 8, 1, key="opt_min",
                                                 help="Rows need at least this many train-period "
                                                      "trades at their holding period."))
            opt_train = o3.slider("Train share of history", 0.5, 0.9, 0.7, 0.05, key="opt_train")
            opt_min_win = o4.slider("Min train win rate", 0.0, 1.0, 0.0, 0.05, key="opt_min_win")
            opt_triples = st.checkbox("Include 3-signal combinations", key="opt_triples")

            opt_args = (rsi_low, rsi_high, lookback, opt_train, opt_min_trades, opt_rank,
                        opt_triples, bt_stop_pct / 100, bt_cost)
            if st.button("Run optimizer", key="opt_run"):
                st.session_state["opt_for"] = (sel, period, opt_args)
            if st.session_state.get("opt_for") != (sel, period, opt_args):
                st.info("Press Run optimizer to search (takes a few seconds; re-run after changing "
                        "any setting).")
            else:
                with st.spinner("Backtesting signal combinations…"):
                    opt_results, opt_curves = run_optimizer(df_full, *opt_args)
                opt_view = opt_results[opt_results["Train win rate"].fillna(0) >= opt_min_win]
                split_date = df_full.index[int(len(df_full) * opt_train)]
                st.caption(f"Train: {df_full.index[0]:%Y-%m-%d} to {split_date:%Y-%m-%d} · "
                          f"Test: {split_date:%Y-%m-%d} to {df_full.index[-1]:%Y-%m-%d} · "
                          f"{len(opt_view)} of {len(opt_results)} rows shown. 'Holds up' means the "
                          "same settings still made money in the test period.")
                if opt_view.empty:
                    st.info("No signal reached the minimum trade count and win rate on this "
                            "history - lower Min trades / Min train win rate, or load a longer "
                            "History length.")
                else:
                    pct = "{:+.1%}"
                    opt_event = st.dataframe(
                        opt_view.drop(columns="Legs").style.format(
                            {"Train win rate": "{:.0%}", "Test win rate": "{:.0%}",
                             "Train return": pct, "Test return": pct}, na_rep="–"),
                        on_select="rerun", selection_mode="single-row", key="opt_table",
                        width="stretch", height=350)
                    st.download_button("📥 Download optimizer results",
                                       opt_view.to_csv(index=False).encode("utf-8"),
                                       file_name=f"{sel}_signal_optimizer.csv", mime="text/csv")
                    opt_rows = opt_event.selection.rows
                    pick = opt_view.iloc[opt_rows[0] if opt_rows else 0]
                    curve = opt_curves[(opt_curves["Signals"] == pick["Signals"])
                                       & (opt_curves["Direction"] == pick["Direction"])
                                       & (opt_curves["Exit"] == pick["Exit"])]
                    st.markdown(f"**Train results by holding period: {pick['Signals']} · "
                                f"{pick['Direction']} · {pick['Exit']}**"
                                + ("" if opt_rows else " (top row - click a row to switch)"))
                    enough = curve["Trades"] >= opt_min_trades
                    fig_h = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_h.add_trace(go.Bar(
                        x=curve["Hold"].astype(str), y=curve["Compounded return"], name="Compounded return",
                        marker_color=["#2ecc71" if h == pick["Best hold"] else
                                      ("#95a5a6" if ok else "#d5d8dc")
                                      for h, ok in zip(curve["Hold"], enough)]),
                        secondary_y=False)
                    fig_h.add_trace(go.Scatter(x=curve["Hold"].astype(str), y=curve["Win rate"],
                                               name="Win rate", mode="lines+markers",
                                               line=dict(color="#2980b9")), secondary_y=True)
                    fig_h.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10),
                                        legend=dict(orientation="h", y=1.15),
                                        xaxis=dict(title="Holding period (days)", type="category"))
                    fig_h.update_yaxes(tickformat=".0%", secondary_y=False, title="Return")
                    fig_h.update_yaxes(tickformat=".0%", secondary_y=True, title="Win rate",
                                       range=[0, 1], dtick=0.25, showgrid=False)
                    st.plotly_chart(fig_h, width="stretch")
                    st.caption("Green = the picked holding period (best once each period is averaged "
                               "with its neighbours, so a lone spike doesn't win). Light bars had "
                               "fewer than the minimum trades.")
else:
    st.caption("👆 Click a row to see its chart.")

# ---------------- Sector calendar ----------------
sector_cal = st.expander("🗓️ Sector rotation calendar", key="sector_cal_exp", on_change="rerun")
with sector_cal:
    if sector_cal.open:
        render_sector_calendar()
