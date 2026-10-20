"""
engine.py - data access, expiry calendar, charges, multi-leg simulation and backtest loop.
No Streamlit imports here, so everything can be tested / scripted headless.

Data layout expected (same CSV format as your sample file):
    <root>/week/<any sub-folders>/NIFTY_YYYY-MM-DD_1m.csv        current-week contract (required)
    <root>/next_week/<any sub-folders>/NIFTY_YYYY-MM-DD_1m.csv   next-week contract   (optional, add later)
If <root>/week does not exist, <root> itself is treated as the week folder.

NEW in this version
    * Per-leg premium condition (>=, <=, between) checked on the entry fill price (open of the entry bar).
    * Straddle-level premium filter: net entry premium must be >= mult x average recent move (PremiumFilter/MoveStats).
    * ExpiryCalendar.tdte (trading-day DTE), rule_expiries (Thu->Tue rule), fixed _fallback_expiry edge case.
    * filter_by_dte helper, summarize() helper.
    * open_stores now uses the `root` argument (it was hardcoded before).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import pickle
import re
import warnings
from bisect import bisect_left
import dataclasses
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

FILE_RE = re.compile(r"NIFTY_(\d{4}-\d{2}-\d{2})_1m\.csv$", re.I)
STRIKE_STEP = 50


# =========================================================================== errors
class DataNotFound(Exception):
    """Required option data is missing."""


class NextWeekDataNotFound(DataNotFound):
    """A leg uses next-week options but the next_week folder is missing or empty."""


# =========================================================================== data access
def _scan_files(root: str, skip_dirs=()) -> dict:
    files = {}
    for dp, dirs, fns in os.walk(root):
        dirs[:] = [d for d in dirs if d.lower() not in skip_dirs]
        for fn in fns:
            m = FILE_RE.search(fn)
            if m:
                files[pd.Timestamp(m.group(1)).date()] = os.path.join(dp, fn)
    return files


def resolve_dirs(root: str):
    """-> (week_dir, next_week_dir). next_week_dir may not exist."""
    root = os.path.abspath(os.path.expanduser(root))
    if os.path.basename(root).lower() == "week":
        return root, os.path.join(os.path.dirname(root), "next_week")
    week = os.path.join(root, "week")
    return (week if os.path.isdir(week) else root), os.path.join(root, "next_week")


CACHE_VERSION = 3                                  # 3: packed layout (a few big arrays per day, not ~300 small ones)
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "days")
_COLS = ("t", "o", "h", "l", "c", "v", "iv")


class DayChain:
    """
    One trading day of 1-minute option data, indexed by (option_type, strike_price).

    Reading a CSV costs ~60-90 ms, so the day is cached on disk the first time it is read (keyed by file size +
    modified time). The cache is PACKED: every contract's bars sit back to back in seven long arrays (time, open, high,
    low, close, volume, IV) plus a table of where each contract starts and ends. Loading that is a handful of big
    reads (~0.5 ms) instead of ~2,000 small ones. contract() returns zero-copy views of one contract's slice.
    Results are bit-for-bit identical to reading the CSV. Delete the cache folder any time; it is rebuilt on demand.
    """

    def __init__(self, path: str, cache_dir: str | None = None):
        stat = os.stat(path)
        stamp = (stat.st_size, stat.st_mtime_ns)
        cpath = None
        if cache_dir:
            cpath = os.path.join(cache_dir, hashlib.md5(os.path.abspath(path).encode()).hexdigest()[:20] + ".pkl")
            try:
                with open(cpath, "rb") as fh:
                    blob = pickle.load(fh)
                if blob.get("v") == CACHE_VERSION and blob.get("stamp") == stamp:
                    self._set(blob["spot_t"], blob["spot_v"], blob["keys"], blob["off"], blob["cols"])
                    return
            except Exception:                              # missing, corrupt or from another version: rebuild
                pass

        df = pd.read_csv(path, parse_dates=["datetime"])
        spot = df.groupby("datetime")["spot"].first().sort_index()
        if "iv" not in df:
            df["iv"] = np.nan
        df = df.sort_values(["option_type", "strike_price", "datetime"], kind="stable")
        ot = df["option_type"].astype(str).to_numpy()
        sk = df["strike_price"].to_numpy()
        brk = np.flatnonzero((ot[1:] != ot[:-1]) | (sk[1:] != sk[:-1])) + 1
        off = np.concatenate([[0], brk, [len(df)]]).astype(np.int64)
        keys = [(str(ot[i]), int(sk[i])) for i in off[:-1]]
        cols = dict(t=df["datetime"].to_numpy(dtype="datetime64[ns]"),
                    o=df["open"].to_numpy(float), h=df["high"].to_numpy(float), l=df["low"].to_numpy(float),
                    c=df["close"].to_numpy(float), v=df["volume"].to_numpy(float), iv=df["iv"].to_numpy(float))
        self._set(spot.index.to_numpy(dtype="datetime64[ns]"), spot.to_numpy(float), keys, off, cols)
        if cpath:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                tmp = cpath + f".{os.getpid()}.tmp"
                with open(tmp, "wb") as fh:
                    pickle.dump(dict(v=CACHE_VERSION, stamp=stamp, spot_t=self.spot_t, spot_v=self.spot_v,
                                     keys=keys, off=off, cols=cols), fh, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, cpath)
            except OSError:                                # read-only disk etc.: just run without the cache
                pass

    def _set(self, spot_t, spot_v, keys, off, cols):
        self.spot_t, self.spot_v = spot_t, spot_v
        self._idx = {k: i for i, k in enumerate(keys)}
        self._off, self._cols, self._views = off, cols, {}

    def contract(self, otype: str, strike: int):
        key = (otype, int(strike))
        v = self._views.get(key)
        if v is None:
            i = self._idx.get(key)
            if i is None:
                return None
            a, b = int(self._off[i]), int(self._off[i + 1])
            v = self._views[key] = {c: self._cols[c][a:b] for c in _COLS}
        return v

    def spot_before(self, ts):
        """Last spot strictly before ts (no lookahead)."""
        i = int(np.searchsorted(self.spot_t, np.datetime64(ts), side="left")) - 1
        return None if i < 0 else float(self.spot_v[i])

    def first_cross(self, price, side, start, end):
        """First minute in [start, end) where spot crossed `price` against a LONG/SHORT; else last minute of the window."""
        i0 = int(np.searchsorted(self.spot_t, np.datetime64(start)))
        i1 = int(np.searchsorted(self.spot_t, np.datetime64(end)))
        if i1 <= i0:
            return None
        seg = self.spot_v[i0:i1]
        hit = np.where(seg <= price)[0] if side == "LONG" else np.where(seg >= price)[0]
        j = i0 + (int(hit[0]) if len(hit) else len(seg) - 1)
        return pd.Timestamp(self.spot_t[j])


class ChainStore:
    """Lazy per-day loader with a tiny cache (trades are processed in date order)."""

    def __init__(self, folder: str, label: str, skip_dirs=(), cache_dir: str | None = None):
        self.folder, self.label, self.cache_dir = folder, label, cache_dir
        self.files = _scan_files(folder, skip_dirs) if os.path.isdir(folder) else {}
        self._cache, self._order = {}, []

    def __bool__(self):
        return bool(self.files)

    def dates(self):
        return sorted(self.files)

    def day(self, d):
        d = pd.Timestamp(d).date()
        if d not in self.files:
            return None
        if d not in self._cache:
            self._cache[d] = DayChain(self.files[d], self.cache_dir)
            self._order.append(d)
            if len(self._order) > 3:
                self._cache.pop(self._order.pop(0), None)
        return self._cache[d]


@dataclass
class Stores:
    week: ChainStore
    next_week: ChainStore


def open_stores(root: str, cache_dir: str | None = DEFAULT_CACHE_DIR) -> Stores:
    """cache_dir=None switches the on-disk speed cache off."""
    week_dir, next_dir = resolve_dirs(root)
    week = ChainStore(week_dir, "week", skip_dirs=("next_week",), cache_dir=cache_dir)
    nxt = ChainStore(next_dir, "next_week", cache_dir=cache_dir)
    if not week:
        raise DataNotFound(f"Week data not found. Looked for NIFTY_YYYY-MM-DD_1m.csv files under: {week_dir}")
    return Stores(week, nxt)


# =========================================================================== expiry calendar
def _fallback_expiry(d: dt.date) -> dt.date:
    """Weekday rule used only beyond the last detected expiry: Thursday until 28-Aug-2025, Tuesday after."""
    wd = 3 if d <= dt.date(2025, 8, 28) else 1
    return d + dt.timedelta(days=(wd - d.weekday()) % 7)


class ExpiryCalendar:
    def __init__(self, expiries):
        self.expiries = sorted(set(expiries))

    def expiry_for(self, d):
        d = pd.Timestamp(d).date()
        i = bisect_left(self.expiries, d)
        return self.expiries[i] if i < len(self.expiries) else _fallback_expiry(d)

    def dte(self, d) -> int:
        """Calendar days to expiry (0 = expiry day)."""
        d = pd.Timestamp(d).date()
        return (self.expiry_for(d) - d).days

    def tdte(self, d, trading_dates) -> int:
        """Trading days to expiry. Comparable across the Thursday and Tuesday eras. trading_dates = sorted list."""
        d = pd.Timestamp(d).date()
        return bisect_left(trading_dates, self.expiry_for(d)) - bisect_left(trading_dates, d)


def rule_expiries(trading_dates):
    """
    Deterministic expiry list from the exchange rule: Thursday up to 28-Aug-2025, Tuesday from 2-Sep-2025.
    If the nominal day is a holiday / has no data, the expiry moves to the previous trading day.
    Use it to cross-check infer_expiries().
    """
    td = sorted(trading_dates)
    tset, out = set(td), []
    d = td[0]
    while d <= td[-1]:
        nominal = (d.weekday() == 3 and d <= dt.date(2025, 8, 28)) or \
                  (d.weekday() == 1 and d >= dt.date(2025, 9, 2))
        if nominal:
            e = d
            while e not in tset and e > td[0]:
                e -= dt.timedelta(days=1)
            out.append(e)
        d += dt.timedelta(days=1)
    return sorted(set(out))


SUMMARY_CACHE_DIR = os.path.dirname(DEFAULT_CACHE_DIR)          # <project>/.cache, independent of the working directory


def _files_stamp(files: dict) -> str:
    """Fingerprint of the folder's CSV files (names, sizes, modified times): changes when any file is added/edited."""
    h = hashlib.md5()
    for d in sorted(files):
        try:
            st_ = os.stat(files[d])
            h.update(f"{d}|{st_.st_size}|{st_.st_mtime_ns}".encode())
        except OSError:
            h.update(f"{d}|missing".encode())
    return h.hexdigest()[:10]


def build_day_summary(week_store: ChainStore, cache_dir=SUMMARY_CACHE_DIR, progress=None) -> pd.DataFrame:
    """ATM straddle at the first and last minute of every day (one pass over the week files, cached).
    The cache is keyed on the folder AND on every file's size / modified time, so changed data is re-read."""
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(os.path.abspath(week_store.folder).encode()).hexdigest()[:6] + _files_stamp(week_store.files)[:6]
    path = os.path.join(cache_dir, f"summary_{key}.csv")
    if os.path.exists(path):
        try:
            s = pd.read_csv(path, parse_dates=["date"])
            return s.assign(date=s["date"].dt.date)
        except Exception:                                   # corrupt cache: rebuild
            pass
    rows, failed, n = [], [], len(week_store.files)
    for k, d in enumerate(week_store.dates(), 1):
        try:
            df = pd.read_csv(week_store.files[d], usecols=["datetime", "strike_label", "option_type", "close", "spot"])
            a = df[df["strike_label"] == "ATM"]
            t0, t1 = a["datetime"].min(), a["datetime"].max()
            rows.append(dict(date=d,
                             first_straddle=a.loc[a["datetime"] == t0, "close"].sum(),
                             last_straddle=a.loc[a["datetime"] == t1, "close"].sum(),
                             spot_open=float(df["spot"].iloc[0])))
        except Exception:
            failed.append(d)
            continue
        if progress and k % 25 == 0:
            progress(k / n)
    if failed:
        warnings.warn(f"build_day_summary: {len(failed)} file(s) could not be read and were left out "
                      f"(first: {failed[0]}). Expiry detection may be off around those days.")
    s = pd.DataFrame(rows)
    s.to_csv(path, index=False)
    return s


def build_spot_daily(week_store: ChainStore, cache_dir=SUMMARY_CACHE_DIR) -> pd.DataFrame:
    """NIFTY open (first minute) and close (last minute) of every day, from the spot column of the week files.
    Cached like the day summary (rebuilt when any file changes). Columns: date, open, close."""
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(os.path.abspath(week_store.folder).encode()).hexdigest()[:6] + _files_stamp(week_store.files)[:6]
    path = os.path.join(cache_dir, f"spot_daily_{key}.csv")
    if os.path.exists(path):
        try:
            s = pd.read_csv(path, parse_dates=["date"])
            return s.assign(date=s["date"].dt.date)
        except Exception:
            pass
    rows = []
    for d in week_store.dates():
        try:
            day = week_store.day(d)
        except Exception:
            continue
        if day is not None and len(day.spot_v):
            rows.append(dict(date=d, open=float(day.spot_v[0]), close=float(day.spot_v[-1])))
    s = pd.DataFrame(rows, columns=["date", "open", "close"])
    s.to_csv(path, index=False)
    return s


def infer_expiries(summary: pd.DataFrame, ratio: float = 1.8):
    """
    A weekly contract rolls when the ATM straddle jumps from ~0 (expiry-day close) back to a full week of premium.
    expiry = the trading day BEFORE the jump.
    """
    s = summary.sort_values("date").reset_index(drop=True)
    prev_last = s["last_straddle"].shift(1)
    r = (s["first_straddle"] / prev_last.where(prev_last > 0)).fillna(0.0).to_numpy()
    dates = list(s["date"])

    picked = [i for i in range(1, len(s)) if r[i] > ratio]          # i = first day of the NEW contract
    # drop near-duplicates (< 4 days apart): keep the stronger jump
    cleaned = []
    for i in picked:
        if cleaned and (dates[i] - dates[cleaned[-1]]).days < 4:
            if r[i] > r[cleaned[-1]]:
                cleaned[-1] = i
        else:
            cleaned.append(i)
    # repair gaps > 8 days: take the strongest jump inside the gap (needs a clear jump, > 1.3x)
    repaired = list(cleaned)
    for a, b in zip(cleaned[:-1], cleaned[1:]):
        if (dates[b] - dates[a]).days > 8:
            lo, hi = a + 2, b - 1
            if hi >= lo:
                k = lo + int(np.argmax(r[lo:hi + 1]))
                if r[k] > 1.3:
                    repaired.append(k)
    repaired = sorted(set(repaired))
    return [dates[i - 1] for i in repaired]                          # expiry = trading day before the roll


# =========================================================================== charges
# STT on option premium, SELL side only, in percent. Verify against your contract notes.
STT_SCHEDULE = [(dt.date(2000, 1, 1), 0.05), (dt.date(2023, 4, 1), 0.0625),
                (dt.date(2024, 10, 1), 0.10), (dt.date(2026, 4, 1), 0.15)]


def stt_rate_pct(d: dt.date) -> float:
    rate = STT_SCHEDULE[0][1]
    for start, r in STT_SCHEDULE:
        if d >= start:
            rate = r
    return rate


@dataclass
class CostConfig:
    include_charges: bool = True
    brokerage_per_order: float = 20.0      # Rs, flat per order
    txn_pct: float = 0.03503               # NSE options exchange charge, % of premium turnover
    sebi_per_crore: float = 10.0           # Rs per crore of turnover
    stamp_buy_pct: float = 0.003           # % of buy-side premium
    gst_pct: float = 18.0                  # on brokerage + exchange + SEBI
    stt_override_pct: float | None = None  # None = date-based schedule
    slippage_pct: float = 0.0              # % of premium, applied on EVERY fill (entry and exit)


def leg_charges(buy_px, sell_px, qty, d, cc: CostConfig) -> dict:
    zero = dict(brokerage_entry=0.0, brokerage_exit=0.0, stt=0.0, exchange=0.0, sebi=0.0, gst=0.0, stamp=0.0)
    if not cc.include_charges:
        return zero
    buy_v, sell_v = buy_px * qty, sell_px * qty
    turnover = buy_v + sell_v
    b_in = b_out = cc.brokerage_per_order
    exch = cc.txn_pct / 100 * turnover
    sebi = cc.sebi_per_crore / 1e7 * turnover
    gst = cc.gst_pct / 100 * (b_in + b_out + exch + sebi)
    stt_pct = cc.stt_override_pct if cc.stt_override_pct is not None else stt_rate_pct(d)
    return dict(brokerage_entry=b_in, brokerage_exit=b_out, stt=stt_pct / 100 * sell_v,
                exchange=exch, sebi=sebi, gst=gst, stamp=cc.stamp_buy_pct / 100 * buy_v)


# =========================================================================== legs
STRIKE_CHOICES = [f"ITM{n}" for n in range(10, 0, -1)] + ["ATM"] + [f"OTM{n}" for n in range(1, 11)]
LINK_CHOICES = {
    "off": "Independent",
    "sl": "Exit this leg's hedge group if this leg hits SL / trailing SL (group 0 = all legs)",
    "sl_tp": "Exit this leg's hedge group if this leg hits SL, trailing SL or TP (group 0 = all legs)",
}
EXPIRY_CHOICES = {
    "week": "Current week",
    "next_week": "Next week",
    "roll2": "Current week, shift to next week if 2 days to expiry",
    "roll1": "Current week, shift to next week if 1 day to expiry",
    "roll0": "Current week, shift to next week if expiry is today",
}
# premium condition on the leg (checked against the entry fill price = open of the entry bar)
PREM_OPS = {
    ">=": "Premium greater than or equal to value",
    "<=": "Premium less than or equal to value",
    "between": "Premium between value and value 2 (inclusive)",
}
IV_SRC = {
    "leg": "This option's IV",
    "atm": "ATM IV (same CALL/PUT)",
}
PREM_SKIP = {
    "trade": "Skip the whole trade",
    "leg": "Skip only this leg (careful with hedged structures)",
}


@dataclass
class Leg:
    lots: int = 1
    action: str = "SELL"            # BUY / SELL
    otype: str = "PUT"              # CALL / PUT  (as written for `applies_to`)
    applies_to: str = "LONG"        # LONG / SHORT / BOTH  (signal direction)
    mirror: bool = True             # also trade the opposite signal with CALL<->PUT flipped
    expiry: str = "week"            # key of EXPIRY_CHOICES
    strike: str = "ATM"             # one of STRIKE_CHOICES
    tp_on: bool = False
    tp_unit: str = "%"
    tp_val: float = 60.0
    sl_on: bool = False
    sl_unit: str = "%"
    sl_val: float = 35.0
    trail_on: bool = False
    trail_unit: str = "%"
    trail_x: float = 1.0            # for every X move in favour ...
    trail_y: float = 0.5            # ... move the stop by Y
    link: str = "off"               # key of LINK_CHOICES: square off the other legs when this one stops out
    # ---- premium condition (entry fill price of THIS leg, in Rs per unit)
    prem_on: bool = False
    prem_op: str = ">="             # key of PREM_OPS
    prem_val: float = 50.0
    prem_val2: float = 150.0        # upper bound, only used when prem_op == "between"
    prem_skip: str = "trade"        # key of PREM_SKIP: what to do when the condition fails
    # ---- IV condition (implied volatility in %, from the data's iv column, known just before the entry fill)
    iv_on: bool = False
    iv_src: str = "leg"             # key of IV_SRC: this option's own IV, or the ATM option of the same type
    iv_op: str = ">="               # key of PREM_OPS
    iv_val: float = 12.0
    iv_val2: float = 20.0           # upper bound, only used when iv_op == "between"
    iv_skip: str = "trade"          # key of PREM_SKIP
    # ---- hedge group: a linked exit only squares off legs with the SAME group number.
    # 0 = no group, a linked exit from this leg closes ALL legs (the old behaviour).
    # Example hedged straddle: short CE + long CE wing = group 1, short PE + long PE wing = group 2.
    group: int = 0
    # ---- mirror: 0 = off; N = this leg copies every setting of leg N with CALL <-> PUT swapped (its own hedge group
    # is kept). Lets you build one side of an iron condor / iron fly and mirror it to the other side.
    mirror_of: int = 0

    def uses_next_week(self) -> bool:
        return self.expiry != "week"

    def effective_otype(self, side: str):
        """CALL/PUT to trade for this signal side, or None if the leg does not apply."""
        if side == "NONE":                     # time-based entries have no direction: trade the leg as written
            return self.otype
        if self.applies_to in ("BOTH", side):
            return self.otype
        if self.mirror:
            return "CALL" if self.otype == "PUT" else "PUT"
        return None

    def label(self) -> str:
        base = f"{self.lots}x {self.action} {self.strike} {self.otype} [{EXPIRY_CHOICES[self.expiry].split(',')[0]}]"
        if self.prem_on:
            cond = f"{self.prem_val:g}-{self.prem_val2:g}" if self.prem_op == "between" else f"{self.prem_op} {self.prem_val:g}"
            base += f" (prem {cond})"
        if self.iv_on:
            cond = f"{self.iv_val:g}-{self.iv_val2:g}" if self.iv_op == "between" else f"{self.iv_op} {self.iv_val:g}"
            base += f" (IV {cond}{', ATM' if self.iv_src == 'atm' else ''})"
        return base


def flip_otype(otype: str) -> str:
    return "CALL" if otype == "PUT" else "PUT"


def resolve_mirrors(legs):
    """Legs with mirror_of = N take every setting of leg N (1-based, must be an earlier leg) with CALL/PUT swapped,
    keeping their own hedge group. Earlier legs are resolved first, so a mirror of a mirror works."""
    out = list(legs)
    for i, lg in enumerate(out):
        k = int(lg.mirror_of or 0)
        if 1 <= k <= i:
            src = out[k - 1]
            out[i] = dataclasses.replace(src, otype=flip_otype(src.otype), group=lg.group, mirror_of=k)
    return out


def premium_ok(leg: Leg, prem: float) -> bool:
    """True if the leg's entry premium satisfies its premium condition (or no condition is set)."""
    if not leg.prem_on:
        return True
    if leg.prem_op == ">=":
        return prem >= leg.prem_val
    if leg.prem_op == "<=":
        return prem <= leg.prem_val
    return leg.prem_val <= prem <= leg.prem_val2


def iv_at(arr: dict | None, entry_ts) -> float:
    """Implied volatility (%) known just before the entry fill: the bar before the entry bar, or the entry bar itself
    when entry is on the day's first bar. NaN when the data has no IV there (0, negative or blank)."""
    if arr is None or "iv" not in arr:
        return float("nan")
    t = arr["t"]
    i0 = int(np.searchsorted(t, np.datetime64(entry_ts)))
    i = i0 - 1 if i0 > 0 else i0
    if i >= len(t):
        return float("nan")
    v = float(arr["iv"][i])
    return v if np.isfinite(v) and v > 0 else float("nan")


def iv_ok(leg: Leg, iv: float) -> bool:
    """True if the IV satisfies the leg's IV condition (or no condition is set). Missing IV never passes."""
    if not leg.iv_on:
        return True
    if not np.isfinite(iv):
        return False
    if leg.iv_op == ">=":
        return iv >= leg.iv_val
    if leg.iv_op == "<=":
        return iv <= leg.iv_val
    return leg.iv_val <= iv <= leg.iv_val2


def strike_for(atm: int, otype: str, choice: str) -> int:
    if choice == "ATM":
        return atm
    kind, n = choice[:3], int(choice[3:])
    up = (kind == "OTM") == (otype == "CALL")       # OTM call / ITM put -> above ATM
    return atm + n * STRIKE_STEP if up else atm - n * STRIKE_STEP


def preset_legs(name: str):
    """Ready-made structures. LONG = bullish signal; the mirror flag creates the bearish version."""
    L = Leg
    p = {
        "Buy ITM option (directional)": [L(action="BUY", otype="CALL", strike="ITM3")],
        "Sell ITM option (directional)": [L(action="SELL", otype="PUT", strike="ITM3")],
        "Credit spread (bull put / bear call)": [L(action="SELL", otype="PUT", strike="ATM"),
                                                 L(action="BUY", otype="PUT", strike="OTM5")],
        "Debit spread (bull call / bear put)": [L(action="BUY", otype="CALL", strike="ATM"),
                                                L(action="SELL", otype="CALL", strike="OTM5")],
        "Short straddle": [L(action="SELL", otype="CALL", applies_to="BOTH", mirror=False, strike="ATM"),
                           L(action="SELL", otype="PUT", applies_to="BOTH", mirror=False, strike="ATM")],
        "Short strangle": [L(action="SELL", otype="CALL", applies_to="BOTH", mirror=False, strike="OTM3"),
                           L(action="SELL", otype="PUT", applies_to="BOTH", mirror=False, strike="OTM3")],
        "Long straddle": [L(action="BUY", otype="CALL", applies_to="BOTH", mirror=False, strike="ATM"),
                          L(action="BUY", otype="PUT", applies_to="BOTH", mirror=False, strike="ATM")],
        "Hedged straddle (stop per side)": [
            L(action="SELL", otype="CALL", applies_to="BOTH", mirror=False, strike="ATM", sl_on=True, sl_val=35.0,
              link="sl", group=1),
            L(action="BUY", otype="CALL", applies_to="BOTH", mirror=False, strike="OTM6", group=1),
            L(action="SELL", otype="PUT", applies_to="BOTH", mirror=False, strike="ATM", sl_on=True, sl_val=35.0,
              link="sl", group=2),
            L(action="BUY", otype="PUT", applies_to="BOTH", mirror=False, strike="OTM6", group=2)],
        "Iron condor": [L(action="SELL", otype="CALL", applies_to="BOTH", mirror=False, strike="OTM3"),
                        L(action="BUY", otype="CALL", applies_to="BOTH", mirror=False, strike="OTM6"),
                        L(action="SELL", otype="PUT", applies_to="BOTH", mirror=False, strike="OTM3"),
                        L(action="BUY", otype="PUT", applies_to="BOTH", mirror=False, strike="OTM6")],
    }
    return p[name]


PRESET_NAMES = ["Buy ITM option (directional)", "Sell ITM option (directional)",
                "Credit spread (bull put / bear call)", "Debit spread (bull call / bear put)",
                "Short straddle", "Short strangle", "Long straddle", "Iron condor",
                "Hedged straddle (stop per side)"]


def check_legs(legs):
    """Data-free checks of the legs (also used when saving a strategy). Raises ValueError."""
    if not legs:
        raise ValueError("Add at least one leg.")
    for i, lg in enumerate(legs, 1):
        if lg.link not in LINK_CHOICES:
            raise ValueError(f"Leg {i}: unknown linked-exit option '{lg.link}'.")
        if lg.link == "sl" and not lg.sl_on:
            raise ValueError(f"Leg {i}: a linked exit on SL needs Stop loss switched on for this leg.")
        if lg.link == "sl_tp" and not (lg.sl_on or lg.tp_on):
            raise ValueError(f"Leg {i}: a linked exit needs Stop loss or Take profit switched on for this leg.")
        if lg.trail_on and not lg.sl_on:
            raise ValueError(f"Leg {i}: trailing stop needs a stop loss to trail. Switch Stop Loss on.")
        if not (0 <= int(lg.group) <= 9):
            raise ValueError(f"Leg {i}: hedge group must be between 0 and 9.")
        if lg.prem_on:
            if lg.prem_op not in PREM_OPS:
                raise ValueError(f"Leg {i}: unknown premium condition '{lg.prem_op}'.")
            if lg.prem_skip not in PREM_SKIP:
                raise ValueError(f"Leg {i}: unknown premium skip option '{lg.prem_skip}'.")
            if lg.prem_op == "between" and lg.prem_val2 <= lg.prem_val:
                raise ValueError(f"Leg {i}: premium value 2 (upper bound) must be greater than value (lower bound).")
        if lg.iv_on:
            if lg.iv_op not in PREM_OPS or lg.iv_skip not in PREM_SKIP or lg.iv_src not in IV_SRC:
                raise ValueError(f"Leg {i}: unknown IV condition setting.")
            if lg.iv_op == "between" and lg.iv_val2 <= lg.iv_val:
                raise ValueError(f"Leg {i}: IV value 2 (upper bound) must be greater than value (lower bound).")
        for attr, ok in (("action", ("BUY", "SELL")), ("otype", ("CALL", "PUT")), ("applies_to", ("LONG", "SHORT", "BOTH")),
                         ("expiry", EXPIRY_CHOICES), ("strike", STRIKE_CHOICES), ("tp_unit", ("%", "pts")),
                         ("sl_unit", ("%", "pts")), ("trail_unit", ("%", "pts"))):
            if getattr(lg, attr) not in ok:
                raise ValueError(f"Leg {i}: invalid {attr} '{getattr(lg, attr)}'.")
        if int(lg.lots) < 1:
            raise ValueError(f"Leg {i}: lots must be at least 1.")
        if not (0 <= int(lg.mirror_of) < i):
            raise ValueError(f"Leg {i}: can only mirror an earlier leg (1 to {i - 1}).")


def validate_legs(legs, stores: Stores):
    """Raise before running anything if a leg is invalid or needs data that is not there."""
    check_legs(legs)
    for i, lg in enumerate(legs, 1):
        if lg.uses_next_week() and not stores.next_week:
            raise NextWeekDataNotFound(
                f"Next week data not found (leg {i} uses '{EXPIRY_CHOICES[lg.expiry]}'). "
                f"Expected NIFTY_YYYY-MM-DD_1m.csv files under: {stores.next_week.folder}")


# =========================================================================== straddle-level premium filter
# "Is the premium worth selling?" -> net entry premium must be >= mult x average recent move of the index.
@dataclass
class PremiumFilter:
    enabled: bool = False
    metric: str = "win_range"     # day_range / day_oc / win_range / win_oc
    lookback: int = 8             # previous trading days (or previous expiry days when dte_only is used in MoveStats)
    mult: float = 1.0             # net premium must be >= mult x average move


def build_move_table(week_store: ChainStore, entry_time: dt.time, exit_time: dt.time, progress=None) -> pd.DataFrame:
    """
    Per day: full-day range / open-close move, and the same inside the entry_time..exit_time window,
    all from the 1-minute spot series. Index = date.
    """
    rows, dates = [], week_store.dates()
    n = len(dates)
    for k, d in enumerate(dates, 1):
        day = week_store.day(d)
        if day is None or len(day.spot_v) < 2:
            continue
        t, v = day.spot_t, day.spot_v
        s = np.datetime64(dt.datetime.combine(d, entry_time))
        e = np.datetime64(dt.datetime.combine(d, exit_time))
        i0, i1 = int(np.searchsorted(t, s)), int(np.searchsorted(t, e, side="right"))
        w = v[i0:i1]
        rows.append(dict(date=d,
                         day_range=float(v.max() - v.min()),
                         day_oc=float(abs(v[-1] - v[0])),
                         win_range=float(w.max() - w.min()) if len(w) > 1 else np.nan,
                         win_oc=float(abs(w[-1] - w[0])) if len(w) > 1 else np.nan))
        if progress and k % 25 == 0:
            progress(k / n)
    return pd.DataFrame(rows).set_index("date")


class MoveStats:
    """
    Rolling average move using ONLY days strictly before the queried date (a day never sees its own move).
    dte_only=0 -> average over previous expiry days only (needs `calendar`).
    since      -> ignore history before this date (e.g. dt.date(2025, 9, 2) to restart for the Tuesday era).
    """

    def __init__(self, table: pd.DataFrame, metric: str = "win_range", lookback: int = 8,
                 calendar: ExpiryCalendar | None = None, dte_only: int | None = None, since: dt.date | None = None):
        t = table
        if since is not None:
            t = t[[d >= since for d in t.index]]
        if dte_only is not None:
            if calendar is None:
                raise ValueError("MoveStats: dte_only needs the expiry calendar.")
            t = t[[calendar.dte(d) == dte_only for d in t.index]]
        roll = t[metric].rolling(lookback, min_periods=max(3, lookback // 2)).mean()
        self._dates = list(t.index)            # sorted dates
        self._vals = roll.to_numpy(float)      # average of the window ENDING on that date

    def avg_move(self, d):
        """Average over the window ending at the latest table date strictly BEFORE d (no lookahead)."""
        d = pd.Timestamp(d).date()
        i = bisect_left(self._dates, d) - 1
        if i < 0 or np.isnan(self._vals[i]):
            return None
        return float(self._vals[i])


def filter_by_dte(signals: pd.DataFrame, calendar: ExpiryCalendar, dtes=(0,)) -> pd.DataFrame:
    """Keep only signal rows whose entry date has a calendar DTE in `dtes` (0 = expiry day)."""
    keep = signals["entry_time"].dt.date.map(calendar.dte).isin(set(dtes))
    return signals[keep].reset_index(drop=True)


# =========================================================================== trade log
SAMPLE_LOG = pd.DataFrame({
    "trade_type": ["LONG", "SHORT", "LONG", "SHORT"],
    "entry_time": ["2024-03-04 11:35:00", "2024-03-05 12:10:00", "2024-03-06 11:40:00", "2024-03-07 13:05:00"],
    "exit_time": ["2024-03-04 15:15:00", "2024-03-05 13:20:00", "2024-03-06 15:15:00", "2024-03-07 15:15:00"],
    "exit_reason": ["TIME", "SL", "TIME", "TIME"],
    "sl_price": [np.nan, 22180.5, np.nan, np.nan],
    "entry_price": [22010.0, 22150.0, 22080.0, 22200.0],
    "exit_price": [22095.0, 22180.5, 22140.0, 22155.0],
})

LOG_COLUMNS = [
    ("trade_type", "Yes", "LONG / SHORT", "Direction of the signal. BUY/SELL, 1/-1 are also accepted."),
    ("entry_time", "Yes", "2024-03-04 11:35:00", "When the signal fired (bar CLOSE time). Option fills at the next minute's open."),
    ("exit_time", "Yes", "2024-03-04 15:15:00", "When the signal exited. Must be later than entry_time and on the same day."),
    ("exit_reason", "No", "TIME / SL / EOD / REVERSAL", "Blank = TIME. SL exits use sl_price to find the exact minute the index hit the stop."),
    ("sl_price", "No", "22180.5", "Index stop level. Only needed on rows where exit_reason is SL."),
    ("entry_price", "No", "22010.0", "Index price at entry. Reference only, not used for pricing."),
    ("exit_price", "No", "22095.0", "Index price at exit. Reference only, not used for pricing."),
]


def validate_trade_log(df: pd.DataFrame):
    """-> (clean_df | None, errors, warnings). Errors block the run; warnings do not."""
    errs, warns = [], []
    d = df.copy()
    d.columns = [str(c).strip().lower().replace(" ", "_") for c in d.columns]
    alias = {"side": "trade_type", "type": "trade_type", "direction": "trade_type", "signal": "trade_type",
             "entry_ts": "entry_time", "entry_datetime": "entry_time", "exit_ts": "exit_time",
             "exit_datetime": "exit_time", "reason": "exit_reason"}
    d = d.rename(columns={k: v for k, v in alias.items() if k in d.columns and v not in d.columns})
    for col in ("trade_type", "entry_time", "exit_time"):
        if col not in d.columns:
            errs.append(f"Missing required column '{col}'. See the format guide above.")
    if errs:
        return None, errs, warns

    side_map = {"LONG": "LONG", "BUY": "LONG", "1": "LONG", "1.0": "LONG", "L": "LONG",
                "SHORT": "SHORT", "SELL": "SHORT", "-1": "SHORT", "-1.0": "SHORT", "S": "SHORT"}
    d["trade_type"] = d["trade_type"].astype(str).str.strip().str.upper().map(side_map)
    bad = d.index[d["trade_type"].isna()]
    if len(bad):
        errs.append(f"trade_type must be LONG or SHORT. Problem rows (1-based): {[int(i) + 2 for i in bad[:8]]}")

    for col in ("entry_time", "exit_time"):
        d[col] = pd.to_datetime(d[col], errors="coerce")
        bad = d.index[d[col].isna()]
        if len(bad):
            errs.append(f"{col} could not be read as a date-time. Problem rows (1-based): {[int(i) + 2 for i in bad[:8]]}")
    if errs:
        return None, errs, warns

    bad = d.index[d["exit_time"] <= d["entry_time"]]
    if len(bad):
        errs.append(f"exit_time must be after entry_time. Problem rows: {[int(i) + 2 for i in bad[:8]]}")
    bad = d.index[d["entry_time"].dt.date != d["exit_time"].dt.date]
    if len(bad):
        errs.append(f"Trades must start and end on the same day (intraday). Problem rows: {[int(i) + 2 for i in bad[:8]]}")
    if errs:
        return None, errs, warns

    if "exit_reason" not in d.columns:
        d["exit_reason"] = "TIME"
    d["exit_reason"] = d["exit_reason"].fillna("TIME").astype(str).str.strip().str.upper().replace({"": "TIME"})
    d["sl_price"] = pd.to_numeric(d["sl_price"], errors="coerce") if "sl_price" in d.columns else np.nan
    miss = d[(d["exit_reason"] == "SL") & d["sl_price"].isna()]
    if len(miss):
        warns.append(f"{len(miss)} SL exit(s) have no sl_price; they will exit at the last minute before exit_time.")
    if len(d) < 30:
        warns.append(f"Only {len(d)} trades. Results this thin say very little about an edge.")
    return d.sort_values("entry_time").reset_index(drop=True), errs, warns


# =========================================================================== simulation
def signal_exit_point(day: DayChain, tr, bar_minutes: int, mode: str, fixed_time):
    """(timestamp, 'open'|'close') at which the signal itself exits."""
    exit_ts = pd.Timestamp(tr["exit_time"])
    if mode == "fixed":
        return pd.Timestamp(dt.datetime.combine(exit_ts.date(), fixed_time)), "open"
    reason = str(tr.get("exit_reason", "TIME")).upper()
    if reason == "TIME":
        return exit_ts, "open"
    sl = tr.get("sl_price", np.nan)
    if reason == "SL" and pd.notna(sl):
        m = day.first_cross(float(sl), tr["trade_type"], exit_ts - pd.Timedelta(minutes=bar_minutes), exit_ts)
        if m is not None:
            return m, "close"
    return exit_ts - pd.Timedelta(minutes=1), "close"


def _pts(on, unit, val, ref):
    if not on:
        return None
    return float(val) if unit == "pts" else ref * float(val) / 100.0


def simulate_leg(arr: dict, entry_ts, exit_ts, exit_field: str, leg: Leg):
    """
    Walk the minute bars of one contract.
      * entry fill  = open of the first bar at/after entry_ts
      * premium condition (if set on the leg) is checked on that entry fill; failing returns (None, "premium_filter")
      * per bar: gap through a level fills at the open; otherwise the level; if SL and TP are both inside one
        bar the STOP is assumed first (conservative)
      * trailing: for every X in favour (measured on bar highs/lows up to and including the previous bar's
        update) the stop moves Y. A new stop level applies from the NEXT bar.
      * otherwise exit with the signal at exit_ts
    Returns (result dict, None) or (None, reason).
    """
    t = arr["t"]
    e64 = np.datetime64(entry_ts)
    i0 = int(np.searchsorted(t, e64))
    if i0 >= len(t) or (t[i0] - e64) > np.timedelta64(2, "m"):
        return None, "no_entry_bar"
    x64 = np.datetime64(exit_ts)
    fallback = False
    tol = np.timedelta64(5, "m")
    if exit_field == "open":
        pm = int(np.searchsorted(t, x64))
        field = "open"
        if pm >= len(t) or (t[pm] - x64) > tol:          # no bar near the exit time: use the last bar before it
            pm, field, fallback = min(pm, len(t)) - 1, "close", True
    else:
        pm = int(np.searchsorted(t, x64, side="right")) - 1
        field = "close"
        if pm >= 0 and (x64 - t[pm]) > tol:
            fallback = True
        if pm < i0:
            pm, field, fallback = i0, "open", True
    pm = max(pm, i0)

    o, h, l, c, v = arr["o"], arr["h"], arr["l"], arr["c"], arr["v"]
    entry = float(o[i0])
    if not premium_ok(leg, entry):
        return None, "premium_filter"
    d = 1 if leg.action == "BUY" else -1                 # +1: profit when price rises

    sl_pts = _pts(leg.sl_on, leg.sl_unit, leg.sl_val, entry)
    tp_pts = _pts(leg.tp_on, leg.tp_unit, leg.tp_val, entry)
    sl_init = entry - d * sl_pts if sl_pts is not None else None
    tp_px = entry + d * tp_pts if tp_pts is not None else None
    trail = leg.trail_on and sl_pts is not None
    x_pts = _pts(True, leg.trail_unit, leg.trail_x, entry) if trail else 0.0
    y_pts = _pts(True, leg.trail_unit, leg.trail_y, entry) if trail else 0.0

    # ---- stop / target check, vectorised over the bars i0..last_check (same rules, same order as a bar loop):
    # per bar 1) gap through the stop at the open, 2) gap through the target, 3) stop touched, 4) target touched.
    last_check = pm - 1 if field == "open" else pm
    n = last_check + 1 - i0
    if n > 0 and (sl_init is not None or tp_px is not None):
        oj, hj, lj = o[i0:last_check + 1], h[i0:last_check + 1], l[i0:last_check + 1]
        steps = np.zeros(n, dtype=np.int64)
        if sl_init is None:
            sl = None
        elif trail and x_pts > 0:
            # best price reached BEFORE each bar (the entry price for the first bar): a moved stop applies from the
            # next bar, exactly like updating it at the end of every bar
            run = np.maximum.accumulate(hj) if d == 1 else np.minimum.accumulate(lj)
            prev = np.empty(n)
            prev[0] = entry
            prev[1:] = np.maximum(run[:-1], entry) if d == 1 else np.minimum(run[:-1], entry)
            steps = np.floor(d * (prev - entry) / x_pts).astype(np.int64)
            sl = np.where(steps > 0, sl_init + d * steps * y_pts, sl_init)
        else:
            sl = np.full(n, sl_init)
        false = np.zeros(n, dtype=bool)
        if d == 1:
            c1 = oj <= sl if sl is not None else false
            c2 = oj >= tp_px if tp_px is not None else false
            c3 = lj <= sl if sl is not None else false
            c4 = hj >= tp_px if tp_px is not None else false
        else:
            c1 = oj >= sl if sl is not None else false
            c2 = oj <= tp_px if tp_px is not None else false
            c3 = hj >= sl if sl is not None else false
            c4 = lj <= tp_px if tp_px is not None else false
        hit = c1 | c2 | c3 | c4
        if hit.any():
            k = int(np.argmax(hit))
            if c1[k]:
                reason, fill = "SL", oj[k]
            elif c2[k]:
                reason, fill = "TP", oj[k]
            elif c3[k]:
                reason, fill = "SL", sl[k]
            else:
                reason, fill = "TP", tp_px
            if reason == "SL" and steps[k] > 0:
                reason = "TRAIL"
            j = i0 + k
            return dict(entry_ts=pd.Timestamp(t[i0]), entry_raw=entry, exit_ts=pd.Timestamp(t[j]),
                        exit_raw=float(fill), reason=reason, fallback=fallback,
                        illiquid=bool(v[i0] == 0 or v[j] == 0)), None

    exit_raw = float(o[pm] if field == "open" else c[pm])
    return dict(entry_ts=pd.Timestamp(t[i0]), entry_raw=entry, exit_ts=pd.Timestamp(t[pm]),
                exit_raw=exit_raw, reason="SIGNAL", fallback=fallback,
                illiquid=bool(v[i0] == 0 or v[pm] == 0)), None


# =========================================================================== time-based entries (intraday)
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def generate_time_signals(week_store: ChainStore, entry_time: dt.time, exit_time: dt.time,
                          weekdays=(0, 1, 2, 3, 4), start: dt.date | None = None, end: dt.date | None = None):
    """
    A strategy with no signal file: enter at `entry_time` and exit at `exit_time` on every selected weekday
    that has data. trade_type is "NONE" (no direction), so every leg is traded exactly as written.
    """
    if exit_time <= entry_time:
        raise ValueError("Exit time must be after entry time (intraday only).")
    dates = week_store.dates()
    start, end = start or dates[0], end or dates[-1]
    rows = [(dt.datetime.combine(d, entry_time), dt.datetime.combine(d, exit_time))
            for d in dates if start <= d <= end and d.weekday() in set(weekdays)]
    out = pd.DataFrame(rows, columns=["entry_time", "exit_time"])
    out["trade_type"] = "NONE"
    out["exit_reason"] = "TIME"
    out["sl_price"] = np.nan
    return out[["trade_type", "entry_time", "exit_time", "exit_reason", "sl_price"]].reset_index(drop=True)


# =========================================================================== backtest
def _is_link_trigger(lg: Leg, res: dict) -> bool:
    if lg.link == "off":
        return False
    return res["reason"] in ("SL", "TRAIL") or (lg.link == "sl_tp" and res["reason"] == "TP")


def run_backtest(trades: pd.DataFrame, legs, cc: CostConfig, stores: Stores, calendar: ExpiryCalendar | None,
                 lot_size: int = 65, exit_mode: str = "signal", fixed_exit=dt.time(15, 15),
                 bar_minutes: int = 5, progress=None, reentry: dict | None = None,
                 prem_filter: PremiumFilter | None = None, move_stats: MoveStats | None = None):
    """
    -> (trades_out, legs_out, skips). One trades_out row per trade, one legs_out row per leg.

    reentry (time-based entries only): dict(max_per_day=N, cooldown=minutes, last_entry=datetime.time).
    When every leg has been closed early (stop / target / linked exit) the strategy may enter again after `cooldown`
    minutes, up to N trades that day, no new entry after `last_entry`. The scheduled exit time still applies.

    Premium rules (both are optional and can be combined):
      * Leg premium condition: Leg.prem_on / prem_op / prem_val / prem_val2 / prem_skip (checked in simulate_leg).
      * prem_filter + move_stats: net entry premium (sells minus buys) must be >= prem_filter.mult x average move.
    """
    validate_legs(legs, stores)
    if calendar is None and any(lg.expiry.startswith("roll") for lg in legs):
        raise ValueError("A roll rule needs the expiry calendar (it is built from the week files).")
    if prem_filter is not None and prem_filter.enabled and move_stats is None:
        raise ValueError("The premium filter is switched on but no MoveStats was given (build it with build_move_table).")
    use_pf = prem_filter is not None and prem_filter.enabled
    t_rows, l_rows, skips = [], [], {}
    slip = cc.slippage_pct / 100.0

    def skip(reason):
        skips[reason] = skips.get(reason, 0) + 1

    def price_one(tr, entry_ts, tid, n_of_day):
        """Price one trade starting at entry_ts. Appends rows and returns the last exit time, or None if skipped."""
        d, side = entry_ts.date(), tr["trade_type"]
        day_w = stores.week.day(d)
        if day_w is None:
            skip("no week file for that date")
            return None
        active = [(lg, lg.effective_otype(side)) for lg in legs]
        active = [(lg, ot) for lg, ot in active if ot is not None]
        if not active:
            skip("no leg applies to this signal side")
            return None

        exit_ts, exit_field = signal_exit_point(day_w, tr, bar_minutes, exit_mode, fixed_exit)
        spot_entry = day_w.spot_before(entry_ts)          # index level at entry, kept in the logs for margin analysis
        dte = calendar.dte(d) if calendar is not None else None

        priced = []
        for lg, otype in active:
            use_next = lg.expiry == "next_week" or (lg.expiry.startswith("roll") and dte is not None
                                                    and dte <= int(lg.expiry[-1]))
            src = "next_week" if use_next else "week"
            day = stores.next_week.day(d) if use_next else day_w
            if day is None:
                skip(f"no {src} file for that date")
                return None
            spot = day.spot_before(entry_ts)
            if spot is None:
                skip("no spot before entry")
                return None
            atm = int(round(spot / STRIKE_STEP) * STRIKE_STEP)
            strike = strike_for(atm, otype, lg.strike)
            arr = day.contract(otype, strike)
            if arr is None:
                skip("strike outside the +/-10 strike window")
                return None
            ref = arr if lg.iv_src == "leg" else day.contract(otype, atm)
            iv_v = iv_at(ref, entry_ts)
            if not iv_ok(lg, iv_v):
                what = "IV missing in the data" if not np.isfinite(iv_v) else "IV condition not met"
                if lg.iv_skip == "leg":
                    skip(f"leg skipped: {what}")
                    continue
                skip(f"trade skipped: {what}")
                return None
            res, why = simulate_leg(arr, entry_ts, exit_ts, exit_field, lg)
            if res is None:
                if why == "premium_filter":
                    if lg.prem_skip == "leg":
                        skip("leg skipped by premium condition")
                        continue                                  # drop just this leg
                    skip("trade skipped: leg premium condition not met")
                    return None                                   # drop the whole trade
                skip(why)
                return None
            priced.append([lg, otype, strike, src, res, arr, iv_v])

        if not priced:
            skip("all legs filtered by premium / IV conditions")
            return None

        # ---- straddle-level filter: net premium collected vs average recent move
        net_prem, avg_move = None, None
        net_prem = sum(p[4]["entry_raw"] * (1 if p[0].action == "SELL" else -1) for p in priced)
        if use_pf:
            avg_move = move_stats.avg_move(d)
            if avg_move is None:
                skip("premium filter: not enough history for average move")
                return None
            if prem_filter.mult > 0 and net_prem < prem_filter.mult * avg_move:
                skip(f"premium below {prem_filter.mult:g}x average move")
                return None

        # ---- linked exits, processed in time order. A trigger in hedge group g squares off the legs of group g that
        # are still open (group 0 = every leg). Legs in other groups keep running.
        done = set()
        while True:
            cands = [(p[4]["exit_ts"], k) for k, p in enumerate(priced)
                     if k not in done and _is_link_trigger(p[0], p[4])]
            if not cands:
                break
            t_link, k_trig = min(cands)
            done.add(k_trig)
            g = int(priced[k_trig][0].group)
            for k, p in enumerate(priced):
                if k == k_trig or p[4]["exit_ts"] <= t_link:
                    continue
                if g != 0 and int(p[0].group) != g:
                    continue
                new, why = simulate_leg(p[5], entry_ts, t_link, "close", p[0])
                if new is None:
                    skip(why)
                    return None
                if new["reason"] == "SIGNAL":
                    new["reason"] = "LINKED"
                p[4] = new

        tot = dict(gross=0.0, slippage=0.0, brokerage_entry=0.0, brokerage_exit=0.0,
                   stt=0.0, exchange=0.0, sebi=0.0, gst=0.0, stamp=0.0)
        last_exit, desc, new_l = entry_ts, [], []
        for k, (lg, otype, strike, src, r, _arr, iv_v) in enumerate(priced, 1):
            qty = lg.lots * lot_size
            e, x = r["entry_raw"], r["exit_raw"]
            if lg.action == "BUY":
                buy_px, sell_px = e * (1 + slip), x * (1 - slip)
                gross = (x - e) * qty
                slip_cost = ((buy_px - e) + (x - sell_px)) * qty
            else:
                sell_px, buy_px = e * (1 - slip), x * (1 + slip)
                gross = (e - x) * qty
                slip_cost = ((e - sell_px) + (buy_px - x)) * qty
            ch = leg_charges(buy_px, sell_px, qty, d, cc)
            charges = ch["stt"] + ch["exchange"] + ch["sebi"] + ch["gst"] + ch["stamp"]
            brok = ch["brokerage_entry"] + ch["brokerage_exit"]
            net = gross - slip_cost - brok - charges
            tot["gross"] += gross
            tot["slippage"] += slip_cost
            for key in ("brokerage_entry", "brokerage_exit", "stt", "exchange", "sebi", "gst", "stamp"):
                tot[key] += ch[key]
            last_exit = max(last_exit, r["exit_ts"])
            desc.append(f"{lg.action[0]}{otype[0]}{strike}")
            new_l.append(dict(trade_id=tid, leg=k, date=d, side=side, action=lg.action, option=otype, strike=strike,
                              expiry_src=src, lots=lg.lots, qty=qty, spot_entry=spot_entry, entry_time=r["entry_ts"], exit_time=r["exit_ts"],
                              entry_raw=e, exit_raw=x, entry_iv=iv_v, exit_reason=r["reason"], gross=gross, slippage=slip_cost,
                              brokerage=brok, charges=charges, net=net,
                              illiquid=r["illiquid"], exit_fallback=r["fallback"]))
        l_rows.extend(new_l)
        brokerage = tot["brokerage_entry"] + tot["brokerage_exit"]
        charges = tot["stt"] + tot["exchange"] + tot["sebi"] + tot["gst"] + tot["stamp"]
        t_rows.append(dict(trade_id=tid, date=d, side=side, trade_no=n_of_day, entry_time=entry_ts, exit_time=last_exit,
                           n_legs=len(priced), legs=" ".join(desc), dte=dte, spot_entry=spot_entry,
                           entry_premium=net_prem, avg_move=avg_move,
                           gross=tot["gross"], slippage=tot["slippage"], brokerage=brokerage,
                           brokerage_entry=tot["brokerage_entry"], brokerage_exit=tot["brokerage_exit"],
                           stt=tot["stt"], gst=tot["gst"], exchange=tot["exchange"], sebi=tot["sebi"],
                           stamp=tot["stamp"], charges=charges,
                           net=tot["gross"] - tot["slippage"] - brokerage - charges,
                           any_illiquid=any(p[4]["illiquid"] for p in priced)))
        return last_exit

    next_id, n = 0, len(trades)
    for row_no, (_, tr) in enumerate(trades.iterrows()):
        if progress and row_no % 25 == 0:
            progress(row_no / max(n, 1))
        row_exit = pd.Timestamp(tr["exit_time"])
        entry_ts, n_today = pd.Timestamp(tr["entry_time"]), 0
        while True:
            last_exit = price_one(tr, entry_ts, next_id, n_today + 1)
            if last_exit is None:
                break
            next_id += 1
            n_today += 1
            if not reentry or tr["trade_type"] != "NONE" or n_today >= int(reentry["max_per_day"]):
                break
            if last_exit >= row_exit:                      # ran all the way to the scheduled exit
                break
            nxt = last_exit + pd.Timedelta(minutes=1 + int(reentry["cooldown"]))
            if nxt.time() > reentry["last_entry"] or nxt >= row_exit:
                break
            entry_ts = nxt
    if progress:
        progress(1.0)
    return pd.DataFrame(t_rows), pd.DataFrame(l_rows), skips


# =========================================================================== results summary
def summarize(t: pd.DataFrame) -> dict:
    """Headline statistics from trades_out."""
    if t.empty:
        return {}
    t = t.sort_values("exit_time")                      # same ordering as reporting.kpis, so drawdowns agree
    pnl = t["net"]
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    eq = pnl.cumsum()
    daily = t.groupby("date")["net"].sum()
    return dict(
        trades=len(t),
        net_pnl=float(pnl.sum()),
        win_rate_pct=float(100 * (pnl > 0).mean()),
        profit_factor=float(wins.sum() / -losses.sum()) if len(losses) else float("inf"),
        avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        expectancy=float(pnl.mean()),
        max_drawdown=float((eq - eq.cummax()).min()),
        best_day=float(daily.max()), worst_day=float(daily.min()),
        total_charges=float((t["brokerage"] + t["charges"]).sum()),
        total_slippage=float(t["slippage"].sum()),
    )


# helper used by the UI caption
def brokerage_caption(n_legs: int, cc: CostConfig) -> str:
    if not cc.include_charges:
        return "Charges are switched off."
    per = cc.brokerage_per_order
    return (f"Brokerage per trade with {n_legs} leg(s): Rs {2 * n_legs * per:,.0f} "
            f"(entry Rs {n_legs * per:,.0f} + exit Rs {n_legs * per:,.0f})")