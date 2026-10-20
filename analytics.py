"""
analytics.py - robustness checks on a finished backtest: Monte Carlo max drawdown and rolling returns.
No Streamlit imports, so it can be tested / scripted headless. Input is the `trades_out` DataFrame from
engine.run_backtest (needs columns: exit_time, date, net).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLL_HORIZONS = [("1 month", 1), ("3 months", 3), ("6 months", 6),
                 ("1 year", 12), ("2 years", 24), ("4 years", 48)]

PERCENTILES = [50, 75, 90, 95, 99]


def pnl_series(trades: pd.DataFrame, unit: str = "trade") -> np.ndarray:
    """Net P&L in time order, one value per trade or per trading day."""
    t = trades.sort_values("exit_time")             # same ordering as the report's equity curve and drawdown
    if unit == "day":
        return t.groupby("date")["net"].sum().sort_index().to_numpy(float)
    return t["net"].to_numpy(float)


def max_drawdown(pnl) -> float:
    """Largest peak-to-trough fall of cumulative P&L, in Rs (positive number). The curve starts at 0."""
    eq = np.concatenate([[0.0], np.cumsum(np.asarray(pnl, float))])
    return float((np.maximum.accumulate(eq) - eq).max())


def monte_carlo_drawdown(pnl, n_sims: int = 5000, method: str = "shuffle", seed: int | None = None,
                         max_cells: int = 2_000_000) -> dict:
    """
    Build n_sims alternative P&L sequences and measure the max drawdown of each.
      shuffle   : the SAME trades in a random order (total profit is identical, only the path changes)
      bootstrap : draw trades with replacement (total profit changes too)
    Returns dict(dd=array of simulated max drawdowns (Rs, positive), final=array of final P&L, actual_dd, actual_pct, ...).
    actual_pct = % of simulated sequences whose drawdown was SMALLER than the real one (high = history was unlucky).
    """
    pnl = np.asarray(pnl, float)
    n = len(pnl)
    if n < 5:
        raise ValueError("Need at least 5 trades to run a Monte Carlo simulation.")
    if method not in ("shuffle", "bootstrap"):
        raise ValueError(f"Unknown method '{method}'.")
    n_sims = int(n_sims)
    rng = np.random.default_rng(seed)
    dd = np.empty(n_sims)
    final = np.empty(n_sims)
    chunk = max(1, int(max_cells // n))
    for s in range(0, n_sims, chunk):
        m = min(chunk, n_sims - s)
        if method == "bootstrap":
            idx = rng.integers(0, n, size=(m, n))
        else:
            idx = np.argsort(rng.random((m, n)), axis=1)
        paths = np.cumsum(pnl[idx], axis=1)
        eq = np.concatenate([np.zeros((m, 1)), paths], axis=1)
        dd[s:s + m] = (np.maximum.accumulate(eq, axis=1) - eq).max(axis=1)
        final[s:s + m] = paths[:, -1]
    actual = max_drawdown(pnl)
    return dict(dd=dd, final=final, actual_dd=actual, actual_pct=float((dd < actual).mean() * 100),
                n=n, n_sims=n_sims, method=method)


def mc_table(out: dict, capital: float) -> pd.DataFrame:
    """Percentile table of simulated max drawdown, in Rs and as % of capital."""
    dd = out["dd"]
    rows = [(f"{p}th percentile" if p != 50 else "Median", float(np.percentile(dd, p))) for p in PERCENTILES]
    rows.append((f"Worst of {out['n_sims']:,}", float(dd.max())))
    rows.append(("Actual history", out["actual_dd"]))
    return pd.DataFrame([dict(Case=k, Rs=v, Pct=100 * v / capital if capital else np.nan) for k, v in rows])


def rolling_returns(trades: pd.DataFrame, horizons=ROLL_HORIZONS) -> pd.DataFrame:
    """
    Rolling-window P&L in Rs. Every calendar day is a window end; the window covers the previous N months,
    and only windows that fit fully inside your data are used. Days with no trades count as 0.
    Columns: Rolling horizon, Min return (Rs), Avg return (Rs), Max return (Rs), % positive windows, Windows.
    """
    cols = ["Rolling horizon", "Min return (Rs)", "Avg return (Rs)", "Max return (Rs)", "% positive windows", "Windows"]
    if trades.empty:
        return pd.DataFrame(columns=cols)
    daily = trades.assign(_d=pd.to_datetime(trades["date"])).groupby("_d")["net"].sum()
    idx = pd.date_range(daily.index.min() - pd.Timedelta(days=1), daily.index.max(), freq="D")
    cum = daily.reindex(idx, fill_value=0.0).cumsum()
    ends = idx[1:]
    rows = []
    for label, months in horizons:
        starts = ends - pd.DateOffset(months=months)
        ok = np.asarray(starts >= idx[0])
        if not ok.any():
            rows.append([label, np.nan, np.nan, np.nan, np.nan, 0])
            continue
        v = cum.reindex(ends[ok]).to_numpy() - cum.reindex(starts[ok]).to_numpy()
        rows.append([label, float(v.min()), float(v.mean()), float(v.max()), float(100 * (v > 0).mean()), int(ok.sum())])
    return pd.DataFrame(rows, columns=cols)


SWEEP_MULTS = [0.0, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.3, 1.5, 2.0]


def premium_ratio(trades: pd.DataFrame) -> pd.Series:
    """entry_premium / avg_move per trade (only trades where the premium filter measured an average move)."""
    if "avg_move" not in trades or "entry_premium" not in trades:
        return pd.Series(dtype=float)
    t = trades[trades["avg_move"].notna() & (trades["avg_move"] > 0)]
    return (t["entry_premium"] / t["avg_move"]).rename("ratio")


def premium_sweep(trades: pd.DataFrame, mults=SWEEP_MULTS, base_mult: float = 0.0) -> pd.DataFrame:
    """
    What the result would have been at each multiplier, from ONE run, by keeping only trades whose
    entry_premium >= mult x avg_move. Only multipliers >= the one the run used are valid (lower ones were never traded).
    Exact when re-entry is off; with re-entry the later entries of a day could differ slightly.
    """
    ratio = premium_ratio(trades)
    t = trades.loc[ratio.index]
    rows = []
    for m in mults:
        if m < base_mult - 1e-9:
            continue
        k = t[ratio >= m]
        net = k["net"]
        win, loss = net[net > 0].sum(), -net[net < 0].sum()
        rows.append(dict(Multiplier=m, Trades=len(k), **{
            "Net P&L (Rs)": float(net.sum()),
            "Profit factor": float(win / loss) if loss > 0 else (np.nan if len(k) == 0 else np.inf),
            "Win rate %": float(100 * (net > 0).mean()) if len(k) else np.nan,
            "Avg per trade (Rs)": float(net.mean()) if len(k) else np.nan}))
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------- breakdown by weekday / days to expiry
GROUP_COLS = ["Trades", "Net P&L", "Win rate %", "Profit factor", "Avg profit per trade", "Avg win", "Avg loss",
              "Best trade", "Worst trade", "Max drawdown", "% of total profit"]


def group_stats(trades: pd.DataFrame, keys: pd.Series, order=None) -> pd.DataFrame:
    """
    The same headline stats for each group of trades (e.g. each weekday, or each days-to-expiry value).
    `keys` has the same index as `trades`. Max drawdown is the drawdown you would have had trading ONLY that group
    (closed trades in exit-time order). Index = group, in `order` when given.
    """
    t = trades.assign(_k=keys).dropna(subset=["_k"]).sort_values("exit_time")
    total = float(t["net"].sum())
    rows = {}
    for k, g in t.groupby("_k", sort=False):
        net = g["net"]
        win, loss = net[net > 0].sum(), -net[net < 0].sum()
        rows[k] = {"Trades": len(g), "Net P&L": float(net.sum()), "Win rate %": float(100 * (net > 0).mean()),
                   "Profit factor": float(win / loss) if loss > 0 else (np.inf if win > 0 else np.nan),
                   "Avg profit per trade": float(net.mean()),
                   "Avg win": float(net[net > 0].mean()) if (net > 0).any() else 0.0,
                   "Avg loss": float(net[net < 0].mean()) if (net < 0).any() else 0.0,
                   "Best trade": float(net.max()), "Worst trade": float(net.min()),
                   "Max drawdown": max_drawdown(net.to_numpy()),
                   "% of total profit": float(100 * net.sum() / total) if total else np.nan}
    out = pd.DataFrame.from_dict(rows, orient="index", columns=GROUP_COLS)
    if order is not None:
        out = out.reindex([o for o in order if o in out.index])
    else:
        out = out.sort_index()
    return out


def weekday_keys(trades: pd.DataFrame) -> pd.Series:
    names = np.array(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    return pd.Series(names[pd.to_datetime(trades["date"]).dt.dayofweek.to_numpy()], index=trades.index)


# --------------------------------------------------------------------------- combining saved strategies
def combine_period(frames: dict, common_period: bool = True):
    """frames = {name: trades_df}. With common_period only the dates every strategy covers are kept.
    -> (filtered frames, (first_date, last_date))"""
    lo = max(min(f["date"]) for f in frames.values())
    hi = min(max(f["date"]) for f in frames.values())
    if not common_period:
        lo, hi = min(min(f["date"]) for f in frames.values()), max(max(f["date"]) for f in frames.values())
    return {n: f[(f["date"] >= lo) & (f["date"] <= hi)] for n, f in frames.items()}, (lo, hi)


def daily_pnl(frames: dict) -> pd.DataFrame:
    """Net P&L per calendar day, one column per strategy (0 on days a strategy did not trade)."""
    cols = {n: f.groupby("date")["net"].sum() for n, f in frames.items()}
    d = pd.DataFrame(cols).fillna(0.0).sort_index()
    d.index = pd.to_datetime(d.index)
    return d


def cagr(net: float, base: float, first, last) -> float:
    """Compound annual growth rate of `base` (capital or margin, Rs) after earning `net` between two dates.
    nan when the base is 0 or the period is under a week; -1 (= -100%) if the loss wipes out the base."""
    days = (pd.Timestamp(last) - pd.Timestamp(first)).days
    if not base or base <= 0 or days < 7:
        return float("nan")
    end = base + net
    if end <= 0:
        return -1.0
    return float((end / base) ** (365.25 / days) - 1)
