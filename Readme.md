# S&P 500 Signal Dashboard

A Streamlit app that screens S&P 500 stocks for technical signals — RSI,
SMA/MACD/CCI crossovers, streaks, and distance from 52-week highs — with a
sortable table and a per-ticker drill-down chart.

**Not investment advice.** This is an educational/demo project.

## Features

- Four ways to pick tickers: S&P 500 by GICS sector, the 11 US Sector SPDR ETFs,
  a curated set of 16 major single-country ETFs, or a custom comma-separated list
- **Configurable history length** for the live fetch (6 months to max available) -
  longer history gives indicators like SMA 200 a fuller warm-up, at the cost of a
  slower first load
- Indicators: SMA (10/50/100/150/200), EMA (9/18), Wilder RSI, MACD, ATR, CCI, CMF,
  20-day support/resistance (Donchian-style channel), and a 20-day rolling VWAP - every
  one of them except RSI and ATR has its own crossover/breakout signal (ATR gets
  a separate "volatility expanding/contracting" signal instead, since it's not directional),
  plus Heikin Ashi candles with a direction-change signal (candles turning green or red)
- Adjustable RSI thresholds and crossover lookback window
- **Days above / below a trend SMA**: how many trading days in a row each ticker has
  closed above, or below, an SMA whose length you pick in the sidebar ("Trend SMA
  length", default 150). Two more settings, X ("Min days above") and Y ("Min days
  below"), both default 20, highlight tickers at or past those counts (green above,
  red below) and drive the "Above trend SMA for X+ days" / "Below trend SMA for Y+
  days" screens. The headline numbers include how many tickers are currently above
  and below that SMA
- An overall **Bias** (Bullish/Bearish/Neutral) per ticker, computed as the net of all
  crossover events plus any 2+ day streak - rows are tinted accordingly in the table
- Preset screens, grouped into categories (Overall, RSI, Trend crossovers, Price
  levels, Volatility) in a two-level dropdown so the list stays easy to scan. The
  screens are a fixed menu defined in `signals/summary.py`, not generated from the data
- **Build your own screen**: pick any number of single conditions (RSI oversold, MACD
  crossed up, above SMA200, resistance breakout, above trend SMA for X+ days, ...) and
  keep rows matching all of them, or any of them. It stacks on top of the preset
  screen, so leave that on Overall → All to use only your own conditions. This
  replaces the old one-item "Combo screens" category (its "uptrend pullback" screen
  is now "Above SMA200" + "RSI oversold")
- A free-text filter that narrows the table to rows whose signal summary contains a
  given word (e.g. "breakout")
- Click any row for a **configurable candlestick chart** - pick which overlay lines
  (any SMA/EMA, VWAP, support, resistance) and which lower panels (Heikin Ashi, RSI,
  MACD, ATR, CMF, CCI) to show; the current default set is a sensible starting point, not the
  only option
- **Beta** vs. SPY (the market benchmark) - trailing, from daily returns over
  up to the last 252 trading days, shown as a column in the screener table -
  plus **Beta vs Sector**, comparing it to a leave-one-out average of its
  sector peers currently loaded in the table (only meaningful with the S&P
  500 sector picker and at least 3 other sector peers loaded)
- **CSV export** of either the current screener table, or the complete indicator
  history for a single selected ticker (every column, every date) - the same data
  driving its chart
- **Seasonality** for the selected ticker (stocks and ETFs alike) - a heatmap table
  of close-to-close monthly returns (months as rows, years as columns) with each
  month's average, median and share of up years, a calendar-year row, a one-line
  summary of how the current month has usually gone, an average-month bar
  chart, a "typical year" path with the current year overlaid, and
  turn-of-month and day-of-week return charts. It loads the ticker's full history on its own, independent of the
  History length setting
- **Backtesting** for the selected ticker - pick any crossover signal, the RSI
  mean-reversion rule, or the overall Bias score, choose direction (long/short/both),
  an exit rule (fixed holding period or until the opposite signal fires, with the
  holding period doubling as a safety cap either way), and an optional stop-loss and
  trading cost, then see the trade log, summary stats, and an equity curve vs.
  buy-and-hold
- Data cached for 1 hour (prices) / 24 hours (S&P 500 constituent list)

## Project structure

```
.
├── app.py                          # Streamlit entrypoint (single page)
├── signals/
│   ├── data_core.py                 # Pure fetch functions (no Streamlit) - Wikipedia + yfinance
│   ├── data.py                      # Streamlit-cached wrappers: snapshot-first, live, fallback
│   ├── indicators.py                # SMA, Wilder RSI, MACD, CCI, CMF, support/resistance, VWAP
│   ├── summary.py                   # Per-ticker table, signal text, Bias, preset screens
│   ├── backtest.py                  # Event-based backtest engine (signal -> trade log -> stats)
│   ├── optimizer.py                 # Per-ticker signal/combination/holding-period sweep
│   ├── seasonality.py               # Monthly grid + stats, typical-year path, calendar effects
│   └── universes.py                 # Fixed ETF ticker lists (US sectors, countries)
├── scripts/
│   └── build_snapshot.py            # Builds the offline fallback snapshot (run manually or by CI)
├── .github/workflows/
│   └── update_snapshot.yml          # Scheduled job that refreshes the snapshot on the `data` branch
├── data/                            # Not in git: the app downloads the snapshot here at runtime
│   ├── snapshot_constituents.csv    # Fallback S&P 500 list (generated)
│   ├── snapshot_prices.parquet      # Fallback 1y prices (generated)
│   └── snapshot_meta.json           # Snapshot run time + last bar date (generated)
├── tests/
│   ├── test_indicators.py           # Unit tests for the indicator math
│   └── test_backtest.py             # Unit tests for the backtest engine
├── requirements.txt                 # Pinned dependencies - used for local installs and deploy
├── requirements-dev.txt             # requirements.txt plus dev-only tools (pytest, ruff)
├── check_env.py                      # Standalone script to sanity-check the environment
└── .streamlit/config.toml           # Theme
```

## Setup

Requires Python 3.12. Create a virtual environment with Python's built-in
`venv` and install the dependencies with `pip`.

Windows (PowerShell):

```powershell
py --list                        # confirm 3.12 is installed
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

If 3.12 isn't installed, get it from [python.org](https://www.python.org/downloads/)
or, on Windows, `winget install Python.Python.3.12` (then open a new terminal).
If PowerShell says running scripts is disabled when you activate, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

### Verify the environment

```bash
python check_env.py
```

Checks that the Wikipedia constituent scrape and a batched Yahoo Finance
download both work before you run the app — useful after any dependency
upgrade, since this project previously broke silently when yfinance/pandas
changed their return shapes.

## Running

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). Choose
sectors or paste tickers in the sidebar, adjust RSI/lookback if you like,
and click any row in the table to see its chart.

## Testing

```bash
pytest -q
```

Covers the indicator math directly (RSI bounds, crossover detection, streak
counting, SMA/MACD correctness) without needing network access or Streamlit.

## Data sources

- **Constituents:** scraped from the [Wikipedia S&P 500 list](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
  Tickers with dots (e.g. `BRK.B`) are normalized to Yahoo's dash format (`BRK-B`).
- **Prices:** [Yahoo Finance](https://finance.yahoo.com/) via `yfinance`, downloaded
  in batches of 100 tickers to avoid one request per symbol.

Both sources can rate-limit or go down; see
[Nightly snapshot](#nightly-snapshot-and-offline-fallback) for how the app
avoids most live downloads and what it falls back to.

## Indicator notes

- **Support / Resistance**: the 20-day high/low of the *prior* bars (today's own bar is
  excluded), so a close above resistance or below support is a genuine breakout/breakdown,
  not just an artifact of today being a new high or low.
- **VWAP**: true VWAP resets every trading day and needs intraday (tick or minute) data,
  which this project doesn't have. What's shown is a **rolling 20-day volume-weighted
  average price** using daily bars - a common approximation, but not the same number
  you'd see on an intraday VWAP chart. Treat it as a volume-aware moving average.
- **Bias**: a simple net count of directional crossover events (MACD, SMA 10/50, CCI,
  VWAP, support, resistance, EMA 9/18, price vs SMA 150, CMF, Heikin Ashi) plus the current streak if it's
  2+ days. It's a quick-glance label for sorting/highlighting, not a weighted or
  backtested signal.
- **Days above / below the trend SMA**: consecutive closes above (or below) the
  chosen SMA, counted back from the latest bar - 0 on the side the close isn't on,
  blank until the SMA has its full warm-up. Because the count starts only once the SMA
  exists, it's capped by the history loaded minus the SMA length (about 100 days for
  SMA 150 with the default 1 year); pick a longer History length for larger
  thresholds. The trend SMA is separate from the fixed SMA150 crossover signal. It's
  a sustained state rather than a fresh event, so it's highlighted and screenable but
  *not* added to Bias.
- **CMF (Chaikin Money Flow)**: 20-day, from the money-flow multiplier
  (`((Close-Low)-(High-Close))/(High-Low)`) weighted by volume - positive means
  the close is sitting toward the top of its daily range on heavier volume
  (buying pressure/accumulation), negative means the bottom (selling
  pressure/distribution). Its signal is a zero-cross, the same treatment CCI
  already gets, and it's directional, so it's included in Bias.
- **Heikin Ashi**: smoothed candles computed from the regular OHLC bars. HA close
  is the bar's `(Open+High+Low+Close)/4`; HA open is the midpoint of the previous HA
  candle's body (the first bar is seeded with its own open/close midpoint, which
  washes out within a few bars); HA high/low are the real high/low extended to cover
  the HA body. The signal ("HA x") fires when the HA candle changes color: up when
  HA close moves above HA open, down when it drops back to or below it (a doji
  after a green candle counts as a turn down). It's directional, so it's included
  in Bias, has its own screens and conditions, and can be backtested. Pick "Heikin
  Ashi" under Lower panels to see the HA candles in their own panel, with ▲/▼ markers
  on the bars where the direction changed.
- **ATR (Average True Range)**: Wilder-smoothed, 14-day window - a volatility
  *magnitude*, not a direction, so it doesn't factor into Bias. Instead it gets its own
  signal: ATR crossing its own 20-day average, flagged as "volatility expanding" or
  "contracting." That can accompany a move in either direction, which is exactly why
  it's kept separate from the bullish/bearish scoring.
- **Beta**: the slope of the stock's daily returns regressed against SPY's daily
  returns (`Cov(stock, SPY) / Var(SPY)`), using the closes aligned first and returns
  computed *after* aligning (not the other way around, which could silently distort
  a return across a data gap). Uses up to the trailing 252 trading days, or however
  many overlapping days are available if less history is loaded; blank if there
  are fewer than 40 overlapping days to estimate from. SPY is fetched the same way
  as every other ticker, including the live/snapshot fallback - `build_snapshot.py`
  includes SPY in the snapshot for exactly that reason. Beta is a *market-relative*
  risk measure only - it says nothing about a stock's own idiosyncratic volatility
  (that's closer to what ATR captures), and it isn't wired into Bias or backtesting
  at this point. A **sector-adjusted** variant is now implemented as **Beta vs
  Sector**: each ticker's Beta divided by a *leave-one-out* average of its
  sector peers' Beta (excluding its own value, so a ticker can't dominate its
  own comparison group in a small sample). This only uses sector peers
  currently loaded in the table - not the full 500 - and needs at least 3
  other peers with a valid Beta in that sector before showing a value;
  otherwise it's blank. It's also unavailable with a custom (non-sector)
  ticker list, since there's no sector to compare against. Of the sector-
  adjustment approaches discussed (beta vs. a sector index instead of SPY,
  peer-relative beta, or Vasicek-style shrinkage toward the sector mean),
  this is the peer-relative version - the other two remain unimplemented.

## Seasonality

Below the chart for the selected ticker is a seasonality section built from
**monthly close-to-close returns**: each month's last close vs the previous
month's last close, on adjusted prices (so splits and dividends don't show up
as fake moves).

- **Full history, one ticker**: the section fetches all available daily history
  (`period="max"`) for just the selected ticker, cached for 24 hours, so it
  works the same whatever History length is picked in the sidebar and for any
  ticker - S&P 500 stocks, the sector and country ETFs, or a custom list. If
  that fetch fails it falls back to the history already loaded for the chart
  and says so (with the 1-year default that's only one year of months).
- **Heatmap table**: months as rows, calendar years as columns (last 5, 10, 20
  years, or all), shaded green/red by size. To the right, each month's
  **Avg**, **Median**, **% Up** (share of years it rose) and **Years** (how
  many years of data that is). A **Year** row underneath compounds each
  column's months into a calendar-year return.
- The **current month** is shown month-to-date (marked `*`) but is left out of
  Avg / Median / % Up, so a half-finished month doesn't skew them. The first
  month of data has no prior close, so it has no return; the first year and the
  current year are therefore partial in the Year row.
- A one-line **this-month summary** (e.g. "Oct has averaged +0.6% ..., up in 4
  of 9 years") and an **average-month bar chart** (average as bars, median as
  diamonds, win rate on hover) give the quick read. The table downloads as CSV.

Under that, using the same Years shown window:

- **Typical year**: each year's cumulative return since the previous year's
  last close, by trading day. The blue line is the median across completed
  years (median, so one extreme year can't reshape it), the shaded band covers
  the middle half of years, and the current year is drawn on top in orange -
  so you can see both *when* within a month moves tend to happen and whether
  this year is tracking its usual shape.
- **Around month end**: average one-day return for the last five sessions of
  each month (T-5 .. T-1, T-1 = last day) and the first five of the next
  (T+1 = first day), against the average of all days. If the data starts
  mid-month that month is skipped, and the unfinished current month has no
  "last day" yet, so it only counts towards T+ days.
- **By weekday**: average one-day return and share of up days, Monday to
  Friday.

The turn-of-month and weekday effects are well documented historically but
small next to daily noise and trading costs, and have faded since they became
widely known; they're descriptive, not trade signals.

With 10 or so data points per month, a month that looks "strong" can easily
be chance - especially as twelve months are compared at once. Treat it as a
starting point for a question, not a signal on its own.

## Backtesting

For the ticker selected in the drill-down chart, an expander below it lets you
simulate trades off any signal already computed for that ticker:

- **Signal**: any crossover column (MACD, SMA 10/50, CCI, VWAP, support,
  resistance, EMA 9/18, SMA150, CMF, Heikin Ashi), RSI mean-reversion (oversold -> long,
  overbought -> short, using the same thresholds set in the sidebar), or the
  overall Bias score (a trade opens the day Bias *changes into* Bullish or
  Bearish, not on every day it stays there).
- **Entries** always execute at the **next bar's open** after the signal fired
  at the prior bar's close - using that bar's own close would be look-ahead
  bias (you can't act on information before it exists).
- **Exit**: fixed holding period, or "until the opposite signal fires" - and
  in that second mode, the holding period still applies as a maximum cap, so
  a trade can't run forever if the signal simply never reverses.
- **Stop-loss** (optional) is checked every bar and can end a trade early
  under either exit rule.
- **Direction**: long-only, short-only, or both (a bullish event opens a
  long, a bearish event opens a short). Only one position is held at a time -
  a new signal while already in a trade is ignored until that trade closes.
- Output: a trade log (downloadable as CSV), summary stats (win rate, average
  return, max drawdown, total compounded return), and an equity curve
  compared against simply buying and holding over the same period.

### Finding the best signals (optimizer)

A second expander under the backtest, **Find the best signals**, runs the same
engine as a brute-force sweep for the selected ticker, only when you press
**Run optimizer** (a few seconds; results are cached per ticker and settings):

- **Candidates**: every crossover signal (including Heikin Ashi) and RSI
  mean-reversion on their own, every pair, and optionally every triple. Overall
  Bias is left out, since it's built from the other crossovers plus the streak
  and would mostly double-count them. A combination fires when all its signals have fired in the same
  direction within the crossover lookback window, on the bar the last one fires.
- Each candidate is tested **long-only, short-only and both ways**, under both
  exit rules, at holding periods of 1, 2, 3, 5, 7, 10, 15, 20, 30, 40 and 60
  days, with the stop loss and cost set in the backtest.
- **Train / test split**: settings are picked on the first part of the history
  (70% by default) and re-run unchanged on the rest. "Holds up" is ✓ when those
  same settings still made money in the test period.
- **Best holding period**: each holding period's score is first smoothed with its
  neighbours (a median of three), so a lone lucky spike can't win, then the best
  raw value next to that plateau is picked. Clicking a row charts train return
  and win rate by holding period, with the pick highlighted.
- Rows need a minimum number of train trades (default 8) and can be filtered by
  a minimum train win rate; rank by compounded return or win rate.

The top of this table is biased upward by construction: it's the best of a few
thousand variants of one ticker's history. Treat it as a way to see which
signals are worth a closer look, weight the test columns over the train ones,
and use a longer History length so both periods have enough trades.

**Read this before trusting a result.** This is a single-ticker historical
simulation with everything that implies:
- Trade counts are often small (a handful to a few dozen over a year), which
  isn't a statistically reliable sample - the win rate and average return
  swing a lot with just one or two trades.
- The RSI thresholds and crossover lookback are the same ones you can freely
  drag around elsewhere in the sidebar. Tuning them until a backtest looks
  good is a textbook way to overfit to noise rather than find something real.
- No slippage is modeled beyond the optional flat basis-point cost you set
  yourself; a market order in practice may fill worse than the bar's open.
- The equity curve compounds trade-by-trade (it only steps on exit dates),
  while buy-and-hold is continuously invested every calendar day - the two
  are plotted together for a visual sense of scale, not as directly
  comparable time series.
- None of this is investment advice, and past performance in a backtest is
  not a guarantee of anything going forward.

## Nightly snapshot (and offline fallback)

Daily bars only change once a day, after the close, so re-downloading a year
of prices from Yahoo on every cold start mostly returns data the repo already
has. Yahoo and Wikipedia can also throttle or block shared IPs, which
Streamlit Community Cloud uses. So the app reads prices from a nightly
snapshot first, goes live only when the snapshot can't answer, and falls back
to the snapshot again if live fails:

- `signals/data_core.py` holds the plain fetch functions (no Streamlit
  dependency), so the same code path is used by the app and by the snapshot
  script. It also has the staleness check (`last_trading_day()`: the latest
  weekday whose 16:00 ET close has passed).
- `scripts/build_snapshot.py` fetches the current constituent list, one year of
  prices for every constituent **plus the SPY benchmark and the fixed US Sector /
  Country ETF universes** from `signals/universes.py`, and writes them to
  `data/snapshot_constituents.csv`, `data/snapshot_prices.parquet`, and
  `data/snapshot_meta.json`. The ETF universes aren't scraped from anywhere -
  they're the same hardcoded lists the sidebar presets use - so only their
  *prices* need a fallback, not a constituent list. It refuses to write
  anything (and exits non-zero, failing the Action) if fewer than 480
  constituents or under 95% of tickers come back, so a partly rate-limited
  run can't replace a good snapshot with a thin one.
- The snapshot is **not committed to `main`**. The workflow publishes it to
  the repo's `data` branch as a single commit with no parent, force-pushed
  over the previous one each night, so the branch only ever holds the latest
  copy and git history doesn't grow (committing a ~7 MB parquet file to
  `main` every night used to add a new copy to history each time).
  `signals/data.py` downloads it from
  `https://raw.githubusercontent.com/sdeshp0/stock-dashboard/data/` into
  `data/` at most once an hour per server process (`sync_snapshot()`), and
  only when `snapshot_meta.json` has changed. If the download fails, it keeps
  using whatever is already in `data/`; with nothing there the app simply
  goes live. Set the `SNAPSHOT_URL` environment variable to point it at a
  fork's `data` branch.
- `signals/data.py` exposes `load_sp500_safe()` / `load_prices_safe()` /
  `load_benchmark_close()`, which return `(data, source, as_of)`. For prices:
  1. **`nightly`** - the history length is 6 months or 1 year, the snapshot's
     last bar is on or after the last completed trading day, and it holds at
     least one requested ticker. Requested tickers it lacks (e.g. in a custom
     list) are fetched live and merged in. The app shows a small caption with
     the close date.
  2. **`live`** - anything else goes to Yahoo, as before.
  3. **`fallback`** - live failed, so the snapshot is served even though it's
     out of date, with a warning banner.

  Exchange holidays aren't modelled: on one, the snapshot looks a day stale
  and the app does a live fetch, which is slower but returns the same data.
  The constituent list is still live-first, with the snapshot as fallback.
- `.github/workflows/update_snapshot.yml` runs the script on weekdays at
  22:00 UTC (after the US close year-round) and again at 06:00 UTC the next
  morning, publishing the refreshed files to the `data` branch only when the
  prices or constituents changed. GitHub tends to start scheduled runs a few hours late, and Yahoo
  sometimes hasn't published the day's bar by the first run, so the second
  run fills that gap before the next open. Either can also be triggered
  manually from the **Actions** tab (`workflow_dispatch`). If both fail, the
  app simply goes back to live-first until a run succeeds.

**Before your first deploy** (e.g. on a fork), trigger the workflow once
manually from the Actions tab so the `data` branch exists. Until it does,
the app just runs live-first with no fallback. To work offline locally, run
`python scripts/build_snapshot.py`, which writes the files straight into
`data/` (they're gitignored).

**If you already have a snapshot deployed from before the ETF presets were
added**, it won't have the sector/country ETF tickers in it yet (or SPY, if
it predates Beta). Trigger the Action manually once to refresh it - until then, the offline
fallback simply won't have those tickers to serve, the same as if Yahoo were
down for any other ticker not yet in the snapshot.

## Deploying to Streamlit Community Cloud

1. Push the repo to GitHub, including `requirements.txt`, and make sure the
   `data` branch exists (see the step above).
2. Go to [share.streamlit.io](https://share.streamlit.io), click **New app**,
   and pick this repo/branch with `app.py` as the main file.
3. Under **Advanced settings**, set the Python version to match what you
   tested locally (3.12; use 3.11 if 3.12 isn't offered).
4. Deploy. Community Cloud installs from `requirements.txt` automatically.
5. Keep an eye on the first cold load - it reinstalls dependencies and starts
   fresh each time the app wakes from sleeping, and Yahoo is more likely to
   throttle a shared cloud IP than your own machine. That's exactly what the
   snapshot fallback above is for.
6. There are no secrets to configure yet. If you add an API key later (e.g.
   an alternative data provider), put it in the app's **Secrets** panel and
   read it via `st.secrets`, and never commit it.

The filesystem on Community Cloud is ephemeral - nothing the app writes at
runtime persists across restarts, which is why the snapshot is built by the
workflow and downloaded on each start rather than generated by the app itself.
Because the snapshot no longer lands on `main`, the nightly refresh doesn't
trigger a redeploy either; a running app picks it up within the hour.

## Known limitations / next steps

- No footer/disclaimer in the UI itself yet (documented here instead).
- `Ret_1m` uses a 21-trading-day convention rather than a calendar month.
- `main`'s history still holds the old snapshot copies committed before the
  move to the `data` branch (about 33 MB). Removing them means rewriting
  history with `git filter-repo` and force-pushing, which changes every
  commit ID, so it's left as an optional one-time cleanup.
- Large ticker selections, or a long history length (5y/10y/max), take longer on
  first load, before caching kicks in.
- When live data is used, it's all-or-nothing per request: if the live fetch
  returns data for *some* tickers it's used as-is (those tickers just show
  up in the "no data" expander); the stale snapshot only kicks in when the
  live fetch fails completely. The snapshot itself is always ~1 year of
  history, so 2y and longer lookbacks are always live, and in a fallback the
  chart is limited to that year regardless of the History length setting.
- During market hours, the nightly snapshot shows the previous close rather
  than Yahoo's in-progress bar for today. That's deliberate: signals don't
  shift around while the session is open.
- **The offline snapshot covers S&P 500 constituents, SPY, and the US
  Sector/Country ETF universes** - any *custom* ticker list still has no
  offline fallback, since there's no way to know in advance which tickers
  someone might type in. If Yahoo is unreachable, a custom list shows the
  same "data load failed" error as before; the three preset-based modes now
  fall back gracefully.
- "Beta vs Sector" doesn't apply to the ETF universes (each ETF has no GICS
  sector of its own in this app) or custom lists - it's blank there by
  design, not a bug.

## License

For personal/educational use.
