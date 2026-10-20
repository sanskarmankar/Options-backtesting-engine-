"""
optimizer.py - grid search over strategy parameters (no Streamlit).

A Dim is one varied parameter with its list of values. Every combination of values is applied on top of the builder's
current strategy, checked, then backtested (in parallel worker processes) and reduced to a row of metrics.

Value conventions per leg dimension:
    strike  "ATM" / "OTM2" / "ITM1" ...
    sl, tp  float (value in the leg's unit) or None = switched off
    trail   (x, y) or None = no trailing stop
    prem    float lower bound, (low, high) for 'between', or None = no premium condition
    iv      same as prem, for the IV condition
"""
from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import datetime as dt
import itertools
import math
import multiprocessing as mp
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd

import analytics as A
import engine as E

MAX_COMBOS = 80000
WARN_COMBOS = 500
TIME_KINDS = ("entry", "exit", "max_trades", "cooldown", "last_entry")


# =========================================================================== parsing "20, 30, 40-60/10"
def _time(s: str) -> dt.time:
    h, m = s.strip().split(":")[:2]
    return dt.time(int(h), int(m))


def parse_values(text: str, kind: str) -> list:
    """kind = 'time' | 'int' | 'float'. Accepts single values and ranges 'from-to/step' (step in minutes for times)."""
    out = []
    for tok in [t.strip() for t in str(text).replace(";", ",").split(",") if t.strip()]:
        try:
            if "/" in tok:
                rng, step = tok.split("/", 1)
                a, b = rng.split("-", 1)
                if kind == "time":
                    a, b, step = _time(a), _time(b), int(step)
                    if step <= 0:
                        raise ValueError
                    m, end = a.hour * 60 + a.minute, b.hour * 60 + b.minute
                    while m <= end:
                        out.append(dt.time(m // 60, m % 60))
                        m += step
                else:
                    a, b, step = float(a), float(b), float(step)
                    n = int(math.floor((b - a) / step + 1e-9))
                    if step <= 0 or n < 0 or n > 1000:
                        raise ValueError
                    out += [a + i * step for i in range(n + 1)]
            else:
                out.append(_time(tok) if kind == "time" else float(tok))
        except (ValueError, IndexError):
            raise ValueError(f"Can't read '{tok}'. Use values like {'09:20' if kind == 'time' else '30'} or a range "
                             f"like {'09:15-10:00/15' if kind == 'time' else '20-60/10'}.")
    if kind == "int":
        out = [int(round(v)) for v in out]
    elif kind == "float":
        out = [round(v, 4) for v in out]
    return sorted(set(out))


# =========================================================================== dimensions and combinations
@dataclass
class Dim:
    label: str
    kind: str                  # one of TIME_KINDS, or strike / sl / tp / trail / prem
    values: list
    leg: int | None = None     # leg index for leg dimensions


def fmt_value(kind: str, v) -> str:
    if v is None:
        return "off"
    if isinstance(v, dt.time):
        return v.strftime("%H:%M")
    if isinstance(v, tuple):
        return ("-" if kind in ("prem", "iv") else "/").join(f"{x:g}" for x in v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def count(dims) -> int:
    return int(np.prod([len(d.values) for d in dims])) if dims else 0


def _set(lg: E.Leg, kind: str, v) -> E.Leg:
    if kind == "strike":
        return dataclasses.replace(lg, strike=v)
    if kind in ("sl", "tp"):
        return dataclasses.replace(lg, **({f"{kind}_on": False} if v is None else {f"{kind}_on": True, f"{kind}_val": v}))
    if kind == "trail":
        return dataclasses.replace(lg, trail_on=False) if v is None else \
            dataclasses.replace(lg, trail_on=True, trail_x=v[0], trail_y=v[1])
    if kind in ("prem", "iv"):
        if v is None:
            return dataclasses.replace(lg, **{f"{kind}_on": False})
        if isinstance(v, tuple):
            return dataclasses.replace(lg, **{f"{kind}_on": True, f"{kind}_val": v[0], f"{kind}_val2": v[1]})
        return dataclasses.replace(lg, **{f"{kind}_on": True, f"{kind}_val": v})
    raise ValueError(kind)


def apply(base_legs, base_time: dict, follow: dict, dims, values):
    """-> (legs, time) for one combination. follow = {leg index: leader index}: followers copy the leader's values."""
    legs = [dataclasses.replace(lg) for lg in base_legs]
    time = dict(base_time)
    for d, v in zip(dims, values):
        if d.leg is None:
            time[d.kind] = v
            continue
        for j in [d.leg] + [j for j, k in follow.items() if k == d.leg]:
            legs[j] = _set(legs[j], d.kind, v)
    return legs, time


def check(legs, time: dict, mode: str) -> str | None:
    """Reason the combination cannot run (or makes no sense), or None."""
    try:
        E.check_legs(legs)
    except ValueError as ex:
        return str(ex).split(":", 1)[-1].strip()
    if mode == "time":
        if time["exit"] <= time["entry"]:
            return "exit time not after entry time"
        if int(time["max_trades"]) > 1:
            if time["last_entry"] <= time["entry"]:
                return "'no new entries after' not later than entry time (re-entry impossible)"
    return None


def _effective(legs, time: dict, mode: str):
    """What actually drives the backtest. Settings that are switched off or have no effect are blanked, so two
    combinations that only differ in such a setting give the same key (and are run once)."""
    eff_legs = []
    for lg in legs:
        d = dataclasses.asdict(lg)
        if not lg.sl_on:
            d["sl_val"] = d["sl_unit"] = None
        if not lg.tp_on:
            d["tp_val"] = d["tp_unit"] = None
        if not lg.trail_on:
            d["trail_x"] = d["trail_y"] = d["trail_unit"] = None
        if not lg.prem_on:
            d["prem_op"] = d["prem_val"] = d["prem_val2"] = d["prem_skip"] = None
        elif lg.prem_op != "between":
            d["prem_val2"] = None
        if not lg.iv_on:
            d["iv_src"] = d["iv_op"] = d["iv_val"] = d["iv_val2"] = d["iv_skip"] = None
        elif lg.iv_op != "between":
            d["iv_val2"] = None
        if lg.link == "off":
            d["group"] = None
        eff_legs.append(tuple(sorted(d.items())))
    if mode != "time":
        t = None
    elif int(time["max_trades"]) <= 1:
        t = (time["entry"], time["exit"], 1)                          # cooldown / last entry unused
    else:
        t = (time["entry"], time["exit"], int(time["max_trades"]), int(time["cooldown"]),
             min(time["last_entry"], time["exit"]))                 # any cut-off at/after the exit acts the same
    return tuple(eff_legs), t


DUPLICATE = "same backtest as another combination (the varied setting has no effect there)"


def expand(dims, base_legs, base_time, follow, mode):
    """-> (valid [(values, legs, time)], {reason: count of skipped}). Invalid and duplicate combinations are skipped."""
    n = count(dims)
    if n > MAX_COMBOS:
        raise ValueError(f"{n:,} combinations is more than the limit of {MAX_COMBOS:,}. Use fewer values.")
    ok, bad, seen = [], {}, set()
    for values in itertools.product(*[d.values for d in dims]):
        legs, time = apply(base_legs, base_time, follow, dims, values)
        why = check(legs, time, mode)
        if why is None:
            key = _effective(legs, time, mode)
            if key in seen:
                why = DUPLICATE
            else:
                seen.add(key)
        if why:
            bad[why] = bad.get(why, 0) + 1
        else:
            ok.append((values, legs, time))
    return ok, bad


# =========================================================================== metrics of one backtest
METRICS = ["trades", "net", "return_pct", "cagr_pct", "max_dd", "ret_dd", "pf", "win_rate", "avg_trade"]


def metrics(t: pd.DataFrame, capital: float) -> dict:
    if t.empty:
        return dict(trades=0, net=0.0, return_pct=0.0, cagr_pct=np.nan, max_dd=0.0, ret_dd=np.nan, pf=np.nan,
                    win_rate=np.nan, avg_trade=np.nan)
    t = t.sort_values("exit_time")
    net = t["net"]
    total = float(net.sum())
    win, loss = float(net[net > 0].sum()), float(-net[net < 0].sum())
    dd = A.max_drawdown(net.to_numpy())
    c = A.cagr(total, capital, t["entry_time"].min(), t["exit_time"].max())
    return dict(trades=int(len(t)), net=total, return_pct=100 * total / capital if capital else np.nan,
                cagr_pct=100 * c if pd.notna(c) else np.nan, max_dd=dd,
                ret_dd=total / dd if dd > 0 else (np.inf if total > 0 else np.nan),
                pf=win / loss if loss > 0 else (np.inf if win > 0 else np.nan),
                win_rate=float(100 * (net > 0).mean()), avg_trade=float(net.mean()))


# =========================================================================== workers
_W: dict = {}


def _init(root, expiries, csv_signals, settings):
    _W.clear()
    _W.update(stores=E.open_stores(root), cal=E.ExpiryCalendar(expiries) if expiries else None,
              csv=csv_signals, s=settings, sig_cache={}, move_cache={})


def _signals(time: dict):
    s = _W["s"]
    if s["mode"] != "time":
        sig = _W["csv"]
    else:
        key = (time["entry"], time["exit"])
        if key not in _W["sig_cache"]:
            _W["sig_cache"][key] = E.generate_time_signals(_W["stores"].week, time["entry"], time["exit"],
                                                           time["weekdays"], time["start"], time["end"])
        sig = _W["sig_cache"][key]
    if s["dte_sel"] and _W["cal"] is not None and not sig.empty:
        sig = E.filter_by_dte(sig, _W["cal"], s["dte_sel"])
    return sig


def _run(job):
    idx, legs, time = job
    s = _W["s"]
    try:
        sig = _signals(time)
        if sig.empty:
            return idx, dict(metrics(pd.DataFrame(), s["capital"]), error="no entries")
        reentry = None
        if s["mode"] == "time" and int(time["max_trades"]) > 1:
            reentry = dict(max_per_day=int(time["max_trades"]), cooldown=int(time["cooldown"]),
                           last_entry=time["last_entry"])
        pf = ms = None
        if s.get("pf") and s["mode"] == "time":
            pf = E.PremiumFilter(True, s["pf"]["metric"], s["pf"]["lookback"], s["pf"]["mult"])
            key = (time["entry"], time["exit"])
            if key not in _W["move_cache"]:
                _W["move_cache"][key] = E.build_move_table(_W["stores"].week, time["entry"], time["exit"])
            sel = s["dte_sel"]
            ms = E.MoveStats(_W["move_cache"][key], pf.metric, pf.lookback, calendar=_W["cal"],
                             dte_only=sel[0] if len(sel) == 1 else None)
        tr, _lg, _sk = E.run_backtest(sig, legs, s["cc"], _W["stores"], _W["cal"], lot_size=s["lot_size"],
                                      exit_mode=s["exit_mode"], fixed_exit=s["fixed_exit"], reentry=reentry,
                                      prem_filter=pf, move_stats=ms)
        return idx, dict(metrics(tr, s["capital"]), error=None)
    except Exception as ex:                                    # one bad combination must not kill the whole grid
        return idx, dict(metrics(pd.DataFrame(), s["capital"]), error=f"{type(ex).__name__}: {ex}")


def run_grid(jobs, root, expiries, csv_signals, settings, workers: int = 1):
    """Yield (idx, metrics) as each job finishes. jobs = [(idx, legs, time)]."""
    if workers <= 1:
        _init(root, expiries, csv_signals, settings)
        for j in jobs:
            yield _run(j)
        return
    ctx = mp.get_context("spawn")
    ex = cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init,
                                initargs=(root, expiries, csv_signals, settings))
    try:
        futs = [ex.submit(_run, j) for j in jobs]
        for f in cf.as_completed(futs):
            yield f.result()
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


# =========================================================================== ranking
RANK_METRICS = {                       # label: (column, higher is better)
    "Avg profit per trade": ("avg_trade", True),
    "Returns": ("return_pct", True),
    "No. of trades": ("trades", True),
    "Profit factor": ("pf", True),
    "Win rate": ("win_rate", True),
    "Max drawdown": ("max_dd", False),
    "Return / drawdown": ("ret_dd", True),
}


def custom_score(df: pd.DataFrame, chosen: list) -> pd.DataFrame:
    """
    Equal-weighted score: every chosen metric becomes its percentile rank among the combinations (0 = worst,
    100 = best), then the ranks are averaged. Rupee returns and trade counts live on very different scales, so ranking
    first stops any one of them dominating. -> df with 'Score' and one 'rank: <metric>' column per metric.
    """
    out = df.copy()
    cols = []
    for m in chosen:
        col, higher = RANK_METRICS[m]
        r = out[col].replace([np.inf, -np.inf], [1e300, -1e300]).rank(pct=True, ascending=higher) * 100
        out[f"rank: {m}"] = r.fillna(0.0)
        cols.append(f"rank: {m}")
    out["Score"] = out[cols].mean(axis=1) if cols else np.nan
    return out.sort_values("Score", ascending=False)
