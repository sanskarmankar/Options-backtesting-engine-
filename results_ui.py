"""
results_ui.py - the result panels shared by the Strategy Builder (one strategy) and the Strategy Combiner (several).
Every function takes the trades table (engine.run_backtest -> trades_out, or several saved logs stacked together).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics as A
import engine as E
import reporting as R
from shared import kpi, pn

MC_METHODS = {"shuffle": "Shuffle the order of the same trades", "bootstrap": "Resample trades with replacement"}
MC_UNITS = {"trade": "Per trade", "day": "Per day"}


def _fmt_pct(v):
    return "-" if pd.isna(v) else f"{100 * v:.1f}%"


# --------------------------------------------------------------------------- headline numbers
def kpi_block(t: pd.DataFrame, capital: float, margin: float | None = None, n_signals: int | None = None):
    """Two rows of KPI tiles + CAGR (on capital, and on margin when one is given)."""
    k_ = R.kpis(t, capital)
    cols = st.columns(6)
    kpi(cols[0], "Net profit", R.inr(k_["net"], sign=True), pn(k_["net"]))
    kpi(cols[1], "Gross profit (before costs)", R.inr(k_["gross"], sign=True), pn(k_["gross"]))
    kpi(cols[2], "Total costs", R.inr(k_["costs"]), R.YELLOW)
    kpi(cols[3], "Win rate", f"{k_['win_rate']:.1f}%")
    kpi(cols[4], "Profit factor", "-" if pd.isna(k_["profit_factor"]) else f"{k_['profit_factor']:.2f}")
    kpi(cols[5], "Max drawdown", f"{R.inr(-k_['max_dd'])} ({k_['max_dd_pct']:.1f}%)", R.RED)
    cols2 = st.columns(5)
    kpi(cols2[0], "Avg profit per winning trade", R.inr(k_["avg_win"], sign=True), R.GREEN)
    kpi(cols2[1], "Avg loss per losing trade", R.inr(k_["avg_loss"]), R.RED)
    kpi(cols2[2], "Avg profit per trade", R.inr(k_["avg_trade"], sign=True), pn(k_["avg_trade"]))
    kpi(cols2[3], "Best trade", R.inr(k_["best"], sign=True), pn(k_["best"]))
    kpi(cols2[4], "Worst trade", R.inr(k_["worst"], sign=True), pn(k_["worst"]))

    first, last = pd.to_datetime(t["entry_time"]).min(), pd.to_datetime(t["exit_time"]).max()
    c_cap = A.cagr(k_["net"], capital, first, last)
    cols3 = st.columns(3)
    kpi(cols3[0], f"CAGR on capital (Rs {R.inr(capital)})", _fmt_pct(c_cap), pn(c_cap) if pd.notna(c_cap) else R.MUTED)
    if margin:
        c_mar = A.cagr(k_["net"], margin, first, last)
        kpi(cols3[1], f"CAGR on margin (Rs {R.inr(margin)})", _fmt_pct(c_mar), pn(c_mar) if pd.notna(c_mar) else R.MUTED)
        kpi(cols3[2], "Max drawdown % of margin", f"{100 * k_['max_dd'] / margin:.1f}%", R.RED)
    years = (last - first).days / 365.25
    entries = f" from {n_signals} scheduled entries" if n_signals else ""
    st.caption(f"{k_['trades']} trades{entries} over {years:.1f} years. "
               + (f"Costs are {k_['costs_pct_gross']:.0f}% of gross P&L. " if pd.notna(k_["costs_pct_gross"]) else "")
               + "CAGR = growth rate per year of the capital (or margin) if the profit had been earned over this period"
               + ("; it is shaky for periods under a year." if years < 1 else "."))


# --------------------------------------------------------------------------- charts
@st.cache_data(show_spinner="Reading NIFTY daily prices (once, then cached)...")
def nifty_daily(week_folder: str, stamp: str) -> pd.DataFrame:
    store = E.ChainStore(week_folder, "week", skip_dirs=("next_week",), cache_dir=E.DEFAULT_CACHE_DIR)
    return E.build_spot_daily(store)


def buy_and_hold(t: pd.DataFrame, week_folder: str, base: float):
    """NIFTY bought at the open of the first trade day with `base` Rs and held: daily P&L in Rs at each close.
    -> (x timestamps, y Rs, stats dict) or None when there is no price data for the period."""
    files = E._scan_files(week_folder, ("next_week",))
    daily = nifty_daily(week_folder, E._files_stamp(files))
    first, last = pd.to_datetime(t["entry_time"]).min(), pd.to_datetime(t["exit_time"]).max()
    d = daily[(daily["date"] >= first.date()) & (daily["date"] <= last.date())]
    if d.empty:
        return None
    entry = float(d["open"].iloc[0])
    y = base * (d["close"].to_numpy(float) / entry - 1)
    x = pd.to_datetime(d["date"]) + pd.Timedelta(hours=15, minutes=30)
    x = pd.concat([pd.Series([first]), pd.Series(x)], ignore_index=True)
    y = np.concatenate([[0.0], y])
    dd = float((np.maximum.accumulate(y) - y).max())
    return x, y, dict(net=float(y[-1]), max_dd=dd, cagr=A.cagr(float(y[-1]), base, first, last),
                      entry=entry, exit=float(d["close"].iloc[-1]))


def charts_block(t: pd.DataFrame, key: str = "", base: float | None = None, base_label: str = "capital",
                 week_folder: str | None = None):
    """Equity curve (optionally with NIFTY buy and hold on the same money), year x month table and cost bars.
    base = the money the strategy needs (margin if known, else capital); buy and hold invests the same amount."""
    bench = None
    if week_folder and base:
        if st.checkbox(f"Show NIFTY buy and hold (same Rs {R.inr(base)} as the {base_label})", key=f"{key}show_bh"):
            bh = buy_and_hold(t, week_folder, base)
            if bh is None:
                st.info("No NIFTY prices found for this period in the data folder.")
            else:
                bx, by, bs = bh
                bench = (bx, by, "NIFTY buy & hold")
    st.plotly_chart(R.equity_fig(t, bench=bench), width="stretch", key=f"{key}equity")
    if bench is not None:
        net = float(t["net"].sum())
        dd = R.max_drawdown(t.sort_values("exit_time")["net"])
        first, last = pd.to_datetime(t["entry_time"]).min(), pd.to_datetime(t["exit_time"]).max()
        cs = A.cagr(net, base, first, last)
        cmp = pd.DataFrame({
            "": ["Strategy", "NIFTY buy & hold"],
            "Net P&L": [R.inr(net, sign=True), R.inr(bs["net"], sign=True)],
            f"Return on {base_label}": [f"{100 * net / base:.1f}%", f"{100 * bs['net'] / base:.1f}%"],
            "CAGR": [_fmt_pct(cs), _fmt_pct(bs["cagr"])],
            "Max drawdown": [f"{R.inr(-dd)} ({100 * dd / base:.1f}%)", f"{R.inr(-bs['max_dd'])} ({100 * bs['max_dd'] / base:.1f}%)"]})
        st.dataframe(cmp, hide_index=True, width="stretch")
        st.caption(f"Buy and hold puts Rs {R.inr(base)} into NIFTY at the open of the first trade day "
                   f"({bs['entry']:,.0f}) and holds to the last day ({bs['exit']:,.0f}), valued at each day's close. "
                   f"Index level only: no dividends, costs or futures roll. The strategy line is closed trades only.")
    st.markdown("#### Year by month (net P&L, Rs)")
    st.markdown(R.year_month_html(t), unsafe_allow_html=True)
    st.caption("Max drawdown is measured inside each year, starting from that year's first trade. "
               "Charges = STT + GST + exchange + SEBI + stamp. Brokerage and slippage are separate.")
    show_profit = st.checkbox("Include total profit bar (it dwarfs the costs when profit is large)", value=True,
                              key=f"{key}show_profit")
    st.plotly_chart(R.cost_fig(t, show_profit), width="stretch")


# --------------------------------------------------------------------------- weekday / days-to-expiry tables
_ROWS = [("No. of trades", "Trades", lambda v: f"{int(v)}"),
         ("Avg profit per trade", "Avg profit per trade", lambda v: R.inr(v, sign=True)),
         ("Profit factor", "Profit factor", lambda v: "inf" if np.isinf(v) else f"{v:.2f}"),
         ("Worst trade", "Worst trade", lambda v: R.inr(v)),
         ("Best trade", "Best trade", lambda v: R.inr(v, sign=True)),
         ("Win rate", "Win rate %", lambda v: f"{v:.1f}%")]


def _day_table(stats: pd.DataFrame, columns: list):
    data = {"": [r[0] for r in _ROWS]}
    for c in columns:
        if c in stats.index:
            data[c] = [("-" if pd.isna(stats.loc[c, key]) else f(stats.loc[c, key])) for _, key, f in _ROWS]
        else:
            data[c] = ["0"] + ["-"] * (len(_ROWS) - 1)
    st.dataframe(pd.DataFrame(data), hide_index=True, width="stretch")


def day_tables(t: pd.DataFrame, dte_key: pd.Series | None, calendar_label: str = ""):
    """One table per weekday (Mon-Sat) and one per days to expiry (0-6). dte_key aligns with t's index."""
    st.markdown("#### Results by day")
    st.markdown("**By weekday**")
    wd = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    _day_table(A.group_stats(t, A.weekday_keys(t), wd), wd)

    st.markdown("**By days to expiry** " + (f"(expiry calendar: {calendar_label})" if calendar_label else ""))
    if dte_key is None or not dte_key.notna().any():
        st.info("Days to expiry is not available for these results. Run the backtest again with the data folder loaded.")
    else:
        missing = int(dte_key.isna().sum())
        key = dte_key.astype("Int64").map(lambda v: str(v) if pd.notna(v) else None)
        cols = [str(i) for i in range(0, 7)]
        _day_table(A.group_stats(t, key, cols), cols)
        beyond = int((key.dropna().astype(int) > 6).sum())
        notes = []
        if beyond:
            notes.append(f"{beyond} trade(s) were more than 6 days from expiry and are not in the table.")
        if missing:
            notes.append(f"{missing} trade(s) have no days-to-expiry value (older save) and are left out.")
        st.caption("0 = expiry day, 1 = one calendar day before expiry, and so on. " + " ".join(notes))
    st.caption("Each column uses only the trades of that day. Profit factor = gross profit / gross loss. "
               "'-' = no trades on that day.")


# --------------------------------------------------------------------------- validation
def validation_block(t: pd.DataFrame, capital: float, key: str = "", pf_mult: float | None = None,
                     show_sweep: bool = True):
    """Monte Carlo max drawdown, rolling returns and (optionally) the premium-filter sweep."""
    ss = st.session_state
    for k, v in (("mc_sims", 5000), ("mc_method", "shuffle"), ("mc_unit", "trade"), ("mc_seed", 0)):
        ss.setdefault(f"{key}{k}", v)
    K = lambda n: f"{key}{n}"

    st.markdown("#### Monte Carlo max drawdown")
    st.caption("Your real equity curve is one lucky or unlucky ordering of the same trades. This re-orders them "
               "(or resamples them) many times and shows how deep the drawdown could have been.")
    m1, m2, m3, m4 = st.columns([1, 1.6, 0.9, 1])
    m1.number_input("Number of simulations", min_value=100, max_value=100_000, step=500, key=K("mc_sims"))
    m2.radio("Method", list(MC_METHODS), format_func=MC_METHODS.get, key=K("mc_method"),
             help="Shuffle keeps the exact trades, so total profit never changes, only the path. Resampling draws "
                  "trades with replacement, so profit changes too and the result is a bit more pessimistic.")
    m3.radio("Unit", list(MC_UNITS), format_func=MC_UNITS.get, key=K("mc_unit"),
             help="Per day adds up the trades of each day first, which keeps same-day re-entries together.")
    m4.number_input("Random seed (0 = random)", min_value=0, max_value=1_000_000, step=1, key=K("mc_seed"))
    sig = (len(t), round(float(t["net"].sum()), 2))
    if st.button("Run simulation", type="primary", key=K("mc_run")):
        try:
            with st.spinner(f"Running {int(ss[K('mc_sims')]):,} simulations..."):
                out = A.monte_carlo_drawdown(A.pnl_series(t, ss[K("mc_unit")]), int(ss[K("mc_sims")]),
                                             ss[K("mc_method")], int(ss[K("mc_seed")]) or None)
            ss[K("mc")] = dict(out=out, key=sig, capital=capital, unit=ss[K("mc_unit")])
        except ValueError as ex:
            ss.pop(K("mc"), None)
            st.error(str(ex))

    mc = ss.get(K("mc"))
    if mc is not None:
        if mc["key"] != sig:
            st.warning("The results changed since this simulation ran. Press Run simulation again.")
        out, mcap = mc["out"], mc["capital"]
        dd = out["dd"]
        p50, p95, p99 = (float(np.percentile(dd, q)) for q in (50, 95, 99))
        kc = st.columns(5)
        kpi(kc[0], "Median max drawdown", R.inr(-p50), R.RED)
        kpi(kc[1], "95th percentile", R.inr(-p95), R.RED)
        kpi(kc[2], "99th percentile", R.inr(-p99), R.RED)
        kpi(kc[3], "Worst simulated", R.inr(-float(dd.max())), R.RED)
        kpi(kc[4], "Actual history", R.inr(-out["actual_dd"]), R.RED)
        st.caption(f"{out['n_sims']:,} simulations of {out['n']} {'days' if mc['unit'] == 'day' else 'trades'} "
                   f"({MC_METHODS[out['method']].lower()}). In 95% of them the max drawdown stayed within "
                   f"Rs {p95:,.0f} ({100 * p95 / mcap:.1f}% of capital). Your real drawdown was worse than "
                   f"{out['actual_pct']:.0f}% of the simulated orderings.")
        tbl = A.mc_table(out, mcap)
        st.dataframe(pd.DataFrame({"Case": tbl["Case"],
                                   "Max drawdown (Rs)": [f"-{v:,.0f}" for v in tbl["Rs"]],
                                   "% of capital": [f"-{v:.1f}%" for v in tbl["Pct"]]}),
                     hide_index=True, width="stretch")
        fig = go.Figure(go.Histogram(x=dd, nbinsx=60, marker_color="#e5484d", opacity=0.8))
        fig.add_vline(x=out["actual_dd"], line_dash="dash", annotation_text="actual", annotation_position="top")
        fig.add_vline(x=p95, line_dash="dot", annotation_text="95th", annotation_position="top")
        fig.update_layout(title="Distribution of simulated max drawdown", xaxis_title="Max drawdown (Rs)",
                          yaxis_title="Simulations", height=340, margin=dict(l=10, r=10, t=50, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("Plan your capital and position size around the 95th or 99th percentile, not the actual history. "
                   "Shuffling assumes trades are independent, and it cannot show losing streaks worse than anything "
                   "in your data, so treat it as a minimum cushion.")

    st.divider()
    st.markdown("#### Rolling returns")
    rr = A.rolling_returns(t)
    fmt = lambda v: "-" if pd.isna(v) else f"{v:,.0f}"
    st.dataframe(pd.DataFrame({
        "Rolling horizon": rr["Rolling horizon"],
        "Min return (Rs)": [fmt(v) for v in rr["Min return (Rs)"]],
        "Avg return (Rs)": [fmt(v) for v in rr["Avg return (Rs)"]],
        "Max return (Rs)": [fmt(v) for v in rr["Max return (Rs)"]],
        "% positive windows": ["-" if pd.isna(v) else f"{v:.1f}%" for v in rr["% positive windows"]],
        "Windows": rr["Windows"]}), hide_index=True, width="stretch")
    st.caption("Net P&L in Rs over every rolling calendar window of that length that fits inside your data "
               "(a new window ends each day, so windows overlap and are not independent). '-' means your data is "
               "shorter than that horizon. A strategy that is positive in nearly every 1-year window is far more "
               "convincing than one whose profit comes from a single stretch.")

    if not show_sweep:
        return
    st.divider()
    st.markdown("#### Premium filter: what each multiplier would have done")
    ratio = A.premium_ratio(t)
    if ratio.empty:
        st.caption("Turn on the premium filter (tab 1) and run again. Set the multiplier to 0 to record the "
                   "premium-to-average-move ratio on every trade without skipping any; this table then shows the "
                   "result at each multiplier from that single run.")
        return
    base = float(pf_mult or 0.0)
    q = ratio.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    st.caption(f"Premium / average move on the {len(ratio)} trades: 10th pct {q.iloc[0]:.2f}, 25th {q.iloc[1]:.2f}, "
               f"median {q.iloc[2]:.2f}, 75th {q.iloc[3]:.2f}, 90th {q.iloc[4]:.2f}. "
               f"A multiplier above the 90th percentile leaves almost no trades.")
    sw = A.premium_sweep(t, base_mult=base)
    st.dataframe(pd.DataFrame({
        "Multiplier": [f"{m:g}x" if m else "no filter" for m in sw["Multiplier"]],
        "Trades": sw["Trades"],
        "Net P&L (Rs)": [f"{v:,.0f}" for v in sw["Net P&L (Rs)"]],
        "Profit factor": ["-" if pd.isna(v) else f"{v:.2f}" for v in sw["Profit factor"]],
        "Win rate": ["-" if pd.isna(v) else f"{v:.1f}%" for v in sw["Win rate %"]],
        "Avg per trade (Rs)": ["-" if pd.isna(v) else f"{v:,.0f}" for v in sw["Avg per trade (Rs)"]]}),
        hide_index=True, width="stretch")
    st.caption(f"Built from one run (multiplier {base:g}x). Rows below {base:g}x are not shown because those "
               f"trades were never run. Look for a smooth improvement as the multiplier rises, with enough "
               f"trades left (30+). One magic value with few trades is curve fitting. Exact when re-entry is off.")


# --------------------------------------------------------------------------- strategy shape (payoff at expiry)
def payoff_panel(legs, side: str, lot_size: int, key: str = ""):
    """Illustrative expiry payoff of the legs with hypothetical prices: chart, bias, max profit / loss, breakevens."""
    import payoff as P
    a = P.analyse(legs, side, lot_size)
    if a is None:
        st.caption("No leg applies to this side.")
        return

    def money(v, side_):
        if v is not None:
            return R.inr(v, sign=True)
        return "Unlimited" + (" (index up)" if side_ == "up" else " (index down)")

    mp_side = "up" if a["up"] == "profit" else "down"
    ml_side = "up" if a["up"] == "loss" else "down"
    c = st.columns(5)
    color = R.YELLOW if a["bias"] == "Neutral" else R.GREEN if "bullish" in a["bias"] else R.RED
    kpi(c[0], "Strategy type", a["bias"], color)
    kpi(c[1], "Structure", a["name"])
    kpi(c[2], "Max profit", money(a["max_profit"], mp_side), R.GREEN)
    kpi(c[3], "Max loss", money(a["max_loss"], ml_side), R.RED)
    prem = a["net_premium"]
    kpi(c[4], "Net premium", f"{R.inr(abs(prem))} {'credit' if prem >= 0 else 'debit'}", R.GREEN if prem >= 0 else R.RED)

    x, y = a["x"], a["y"]
    fig = go.Figure()
    fig.add_scatter(x=x, y=np.where(y >= 0, y, np.nan), mode="lines", line=dict(color=R.GREEN, width=2.5),
                    fill="tozeroy", fillcolor="rgba(34,197,94,0.22)", name="Profit",
                    hovertemplate="Index %{x:,.0f}<br>P&L at expiry: Rs %{y:,.0f}<extra></extra>")
    fig.add_scatter(x=x, y=np.where(y <= 0, y, np.nan), mode="lines", line=dict(color=R.RED, width=2.5),
                    fill="tozeroy", fillcolor="rgba(239,68,68,0.22)", name="Loss",
                    hovertemplate="Index %{x:,.0f}<br>P&L at expiry: Rs %{y:,.0f}<extra></extra>")
    fig.add_hline(y=0, line=dict(color=R.MUTED, width=1))
    fig.add_vline(x=a["spot"], line=dict(color="#94a3b8", width=1, dash="dot"),
                  annotation_text=f"Entry {a['spot']:,.0f}", annotation_position="top")
    for b in a["breakevens"]:
        fig.add_vline(x=b, line=dict(color=R.YELLOW, width=1, dash="dash"),
                      annotation_text=f"BE {b:,.0f}", annotation_position="bottom")
    for p in a["positions"]:
        fig.add_scatter(x=[p["strike"]], y=[0], mode="markers", showlegend=False,
                        marker=dict(symbol="triangle-up" if p["sign"] > 0 else "triangle-down", size=11,
                                    color=R.GREEN if p["sign"] > 0 else R.RED),
                        hovertemplate=f"{p['label']} @ {p['strike']:,} (premium {p['premium']:g})<extra></extra>")
    fig.update_layout(template="plotly_dark", height=330, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      xaxis=dict(title="Index at expiry", showgrid=False),
                      yaxis=dict(title="P&L (Rs)", gridcolor="rgba(148,163,184,0.15)", zeroline=False))
    st.plotly_chart(fig, width="stretch", key=f"payoff_{key}{side}")
    be = ", ".join(f"{b:,.0f}" for b in a["breakevens"]) or "none in range"
    legs_txt = "; ".join(f"{p['label']} @ {p['strike']:,} for {p['premium']:g}" for p in a["positions"])
    st.caption(f"{a['bias_long']}. Breakeven(s): {be}. Legs: {legs_txt}. "
               f"Hypothetical numbers for illustration only: index {P.SPOT:,.0f}, volatility {P.IV:.0%}, "
               f"{P.DAYS} days to expiry, lot size {lot_size}. Payoff if every leg is held to expiry; stops, targets, "
               f"linked exits and next-week legs are not modelled.")
