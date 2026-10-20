# Option Lab

Turn a signal log (`trade_logs.csv`) or a fixed daily entry time into multi-leg option strategies (buy/sell ITM,
credit spreads, straddles, condors ...) priced from your 1-minute option-chain CSVs, with charges, slippage,
linked exits, expiry roll rules, a year-by-month report and a library of saved strategies.

## Run
    pip install -r requirements.txt
    streamlit run app.py

## Pages
- **Strategy Builder**: entries (CSV or time-based), legs, costs, results, trades. Save from the sidebar.
- **Saved Strategies**: one card per strategy with name, net profit, max drawdown, avg profit per trade and the equity
  curve. **Load** opens it in the builder with its legs, settings, signal CSV and results.

## Data folder
    option chain data/
        week/        <year folders>/NIFTY_YYYY-MM-DD_1m.csv     required (current-week contract)
        next_week/   <year folders>/NIFTY_YYYY-MM-DD_1m.csv     optional - add later, no code change needed
Paste the full path in the **Option data folder** box at the top of the builder sidebar. It is remembered for next time
(stored in `.option_lab_settings.json`; the `OPTION_LAB_DATA` environment variable also works). A leg that uses "Next week" with no `next_week/` folder stops the run with
"Next week data not found". After adding the folder press **Rescan folders**.

## Speed
The first time a day's CSV is read, its minute bars are cached as a binary file in `.cache/days/` (about 0.8 MB per day,
so roughly 1 GB for 1,300 days). Later runs load that instead of parsing the CSV: a 150-trade, 4-leg backtest dropped
from about 25 s to about 0.2 s with identical results. The cache is keyed on each CSV's size and modified time, so
changed files are rebuilt automatically. Delete `.cache/` at any time to reclaim the space.

## Entries
- **Upload trade log CSV**: required `trade_type` (LONG/SHORT), `entry_time`, `exit_time` (same day).
  Optional `exit_reason` (TIME/SL/EOD/REVERSAL), `sl_price`, `entry_price`, `exit_price`. See `sample_trade_log.csv`.
- **Time-based entries**: pick entry time, exit time (after entry, same day), weekdays and a date range. No CSV.
  **Max trades per day** (default 1): when every leg has been closed early (stop, target or linked exit) the strategy
  can enter again after a wait, up to N trades a day, with a cut-off time for new entries.
  Intraday only. Overnight holding is not supported because the files cannot tell us the expiry date reliably.

## Linked exits
Set **Linked exit** on a leg that has a Stop loss (or Take profit). When it exits on its own stop, every other leg that
is still open is closed at the close of that same minute (exit reason LINKED).

## Saved strategies
Saved under `saved_strategies/<name>/` next to `app.py` (override with the `OPTION_LAB_SAVE_DIR` environment variable):
`strategy.json` (legs, settings, entry setup, summary), `signals.csv` (CSV strategies), `results_trades.csv`,
`results_legs.csv`. Saving under an existing name replaces it.

## Files
- `app.py`            entry point and navigation
- `views/builder.py`  builder page
- `views/saved.py`    saved strategies page
- `engine.py`         data access, expiry calendar, charges, leg simulation, backtest loop (no Streamlit)
- `reporting.py`      KPIs, equity curve, year x month table, cost bars
- `storage.py`        save / load / list / delete strategies (no Streamlit)
- `shared.py`         session-state helpers shared by both pages
