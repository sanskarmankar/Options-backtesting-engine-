"""
shared.py - session-state helpers shared by the builder page and the Saved Strategies page.

Streamlit drops the value of any widget that is not drawn on the current page, so moving between pages would reset
the builder. keep_alive() re-assigns every builder key at the top of each page to prevent that.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

import engine as E
import reporting as R
import storage

# ---- remembered settings (the data folder survives restarts)
SETTINGS_FILE = Path(__file__).resolve().parent / ".option_lab_settings.json"


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text())
    except Exception:
        return {}


def save_setting(key: str, value) -> None:
    cfg = load_settings()
    if cfg.get(key) != value:
        cfg[key] = value
        try:
            SETTINGS_FILE.write_text(json.dumps(cfg, indent=2))
        except OSError:
            pass


DEFAULT_DATA_ROOT = "/Users/user7/Desktop/algos11/index_data/data/index options"

MODE_CSV = "Upload trade log CSV"
MODE_TIME = "Time-based entries (no CSV)"

FIELDS = ["lots", "action", "otype", "applies_to", "mirror", "expiry", "strike",
          "tp_on", "tp_unit", "tp_val", "sl_on", "sl_unit", "sl_val",
          "trail_on", "trail_unit", "trail_x", "trail_y", "link",
          "prem_on", "prem_op", "prem_val", "prem_val2", "prem_skip",
          "iv_on", "iv_src", "iv_op", "iv_val", "iv_val2", "iv_skip", "group", "mirror_of"]
FILTER_KEYS = ["dte_sel", "cal_src", "pf_on", "pf_metric", "pf_lookback", "pf_mult"]
CAL_DETECTED = "Detected from data"

DEFAULTS = {
    "root_input": os.environ.get("OPTION_LAB_DATA") or load_settings().get("data_root", DEFAULT_DATA_ROOT),
    "capital": 500_000, "lot_size": 65,
    "include_charges": True, "brokerage": 20.0, "slippage": 0.10,
    "txn_pct": 0.03503, "stamp_pct": 0.003, "stt_on": False, "stt_val": 0.15,
    "exit_mode_ui": "When the signal exits", "fixed_exit": dt.time(15, 15),
    "src_mode": MODE_CSV,
    "tb_entry": dt.time(9, 20), "tb_exit": dt.time(15, 15), "tb_days": list(E.WEEKDAY_NAMES),
    "tb_max": 1, "tb_cool": 5, "tb_last": dt.time(14, 30),
    "save_margin": 0.0,
    "dte_sel": [], "cal_src": CAL_DETECTED, "pf_on": False, "pf_metric": "win_range", "pf_lookback": 8, "pf_mult": 1.0,
}
OPTIONAL_KEYS = ["tb_start", "tb_end", "save_name", "save_with_results"]


# --------------------------------------------------------------------------- legs
def new_leg_state(leg) -> dict:
    base = {f: getattr(E.Leg(), f) for f in FIELDS}
    src = leg if isinstance(leg, dict) else {f: getattr(leg, f) for f in FIELDS}
    base.update({k: v for k, v in src.items() if k in FIELDS})
    return {"uid": uuid.uuid4().hex[:6], "init": base}


def leg_key(ls: dict, field: str) -> str:
    return f"{ls['uid']}_{field}"


def snapshot(ls: dict) -> dict:
    return {f: st.session_state.get(leg_key(ls, f), ls["init"][f]) for f in FIELDS}


def read_leg(ls: dict) -> E.Leg:
    return E.Leg(**snapshot(ls))


def read_legs(ss=None) -> list:
    """Every leg from the widgets, with mirrored legs resolved (copied from their source with CALL/PUT swapped)."""
    ss = ss or st.session_state
    return E.resolve_mirrors([read_leg(ls) for ls in ss.legs])


def init_state():
    ss = st.session_state
    for k, v in DEFAULTS.items():
        ss.setdefault(k, v)
    if "legs" not in ss:
        ss.legs = [new_leg_state(l) for l in E.preset_legs("Sell ITM option (directional)")]
    ss.setdefault("nonce", 0)


def keep_alive():
    """Call first thing on every page."""
    ss = st.session_state
    init_state()
    keys = list(DEFAULTS) + [k for k in OPTIONAL_KEYS if k in ss]
    for ls in ss.legs:
        keys += [leg_key(ls, f) for f in FIELDS]
    for k in keys:
        if k in ss:
            ss[k] = ss[k]


# --------------------------------------------------------------------------- config <-> session
def _norm(o):
    if isinstance(o, bool):
        return o
    if isinstance(o, (int, float)):
        return float(o)
    if isinstance(o, dict):
        return {k: _norm(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_norm(v) for v in o]
    return str(o) if not isinstance(o, str) and o is not None else o


def hhmm(t: dt.time) -> str:
    return t.strftime("%H:%M")


def parse_hhmm(s: str) -> dt.time:
    h, m = [int(x) for x in s.split(":")[:2]]
    return dt.time(h, m)


def current_config() -> dict:
    ss = st.session_state
    mode = "csv" if ss.src_mode == MODE_CSV else "time"
    return dict(
        mode=mode,
        time=dict(entry=hhmm(ss.tb_entry), exit=hhmm(ss.tb_exit), weekdays=list(ss.tb_days),
                  max_trades=int(ss.tb_max), cooldown=int(ss.tb_cool), last_entry=hhmm(ss.tb_last),
                  start=str(ss.get("tb_start")) if ss.get("tb_start") else None,
                  end=str(ss.get("tb_end")) if ss.get("tb_end") else None),
        legs=[{f: getattr(lg, f) for f in FIELDS} for lg in read_legs(ss)],      # mirrors saved resolved
        costs=dict(include_charges=bool(ss.include_charges), brokerage=float(ss.brokerage),
                   slippage=float(ss.slippage), txn_pct=float(ss.txn_pct), stamp_pct=float(ss.stamp_pct),
                   stt_on=bool(ss.stt_on), stt_val=float(ss.stt_val)),
        filters=dict(dte_sel=[int(x) for x in ss.dte_sel], cal_src=ss.cal_src, pf_on=bool(ss.pf_on),
                     pf_metric=ss.pf_metric, pf_lookback=int(ss.pf_lookback), pf_mult=float(ss.pf_mult)),
        margin=float(ss.get("save_margin") or 0.0),
        lot_size=int(ss.lot_size), capital=int(ss.capital),
        exit_mode_ui=ss.exit_mode_ui, fixed_exit=hhmm(ss.fixed_exit))


def cost_config_from(costs: dict) -> E.CostConfig:
    return E.CostConfig(include_charges=costs["include_charges"], brokerage_per_order=costs["brokerage"],
                        txn_pct=costs["txn_pct"], stamp_buy_pct=costs["stamp_pct"],
                        stt_override_pct=costs["stt_val"] if costs["stt_on"] else None,
                        slippage_pct=costs["slippage"])


def signature(legs, cc, ss=None) -> str:
    """Fingerprint of everything that changes the result (used to flag stale results)."""
    ss = ss or st.session_state
    t = ss.get("trades")
    src = ("time", hhmm(ss.tb_entry), hhmm(ss.tb_exit), sorted(ss.tb_days), str(ss.get("tb_start")), str(ss.get("tb_end")),
           int(ss.tb_max), int(ss.tb_cool), hhmm(ss.tb_last)) \
        if ss.src_mode == MODE_TIME else \
        ("csv", None if t is None else (len(t), pd.Timestamp(t["entry_time"].iloc[0]).isoformat(),
                                        pd.Timestamp(t["exit_time"].iloc[-1]).isoformat()))
    return json.dumps(_norm(dict(legs=[vars(l) for l in legs], cc=vars(cc), lot=ss.lot_size, cap=ss.capital,
                                 exit_mode=ss.exit_mode_ui, fixed=hhmm(ss.fixed_exit), src=src)), sort_keys=True)


def reentry_from_state(ss=None):
    """Re-entry settings for run_backtest, or None when the limiter is 1 trade a day or entries come from a CSV."""
    ss = ss or st.session_state
    if ss.src_mode != MODE_TIME or int(ss.tb_max) <= 1:
        return None
    return dict(max_per_day=int(ss.tb_max), cooldown=int(ss.tb_cool), last_entry=ss.tb_last)


def describe_entries(ss=None) -> str:
    ss = ss or st.session_state
    if ss.src_mode == MODE_TIME:
        d = ss.tb_days
        days = "every weekday" if len(d) == 5 else ", ".join(d)
        more = f", up to {int(ss.tb_max)} trades a day" if int(ss.tb_max) > 1 else ""
        return f"Time-based: {days}, enter {hhmm(ss.tb_entry)}, exit {hhmm(ss.tb_exit)}{more}"
    return f"Signals from {ss.get('trades_name', 'trade log')}"


def apply_saved(name: str) -> str:
    """Load a saved strategy into session_state (legs, settings, CSV, results). Returns a status message."""
    cfg, signals, trades, legs_df = storage.load_strategy(name)
    ss = st.session_state
    c, tm = cfg["costs"], cfg["time"]
    ss.capital, ss.lot_size = int(cfg["capital"]), int(cfg["lot_size"])
    ss.include_charges, ss.brokerage, ss.slippage = bool(c["include_charges"]), float(c["brokerage"]), float(c["slippage"])
    ss.txn_pct, ss.stamp_pct = float(c["txn_pct"]), float(c["stamp_pct"])
    ss.stt_on, ss.stt_val = bool(c["stt_on"]), float(c["stt_val"])
    ss.exit_mode_ui, ss.fixed_exit = cfg["exit_mode_ui"], parse_hhmm(cfg["fixed_exit"])
    ss.src_mode = MODE_CSV if cfg["mode"] == "csv" else MODE_TIME
    ss.tb_entry, ss.tb_exit, ss.tb_days = parse_hhmm(tm["entry"]), parse_hhmm(tm["exit"]), list(tm["weekdays"])
    ss.tb_max, ss.tb_cool = int(tm.get("max_trades", 1)), int(tm.get("cooldown", 5))
    ss.tb_last = parse_hhmm(tm.get("last_entry", "14:30"))
    ss.tb_start = dt.date.fromisoformat(tm["start"]) if tm.get("start") else None
    ss.tb_end = dt.date.fromisoformat(tm["end"]) if tm.get("end") else None
    ss.legs = [new_leg_state(l) for l in cfg["legs"]]
    f = cfg.get("filters") or {}                    # strategies saved before version 2 have no filters block
    for k in FILTER_KEYS:
        ss[k] = type(DEFAULTS[k])(f[k]) if k in f and k != "dte_sel" else (list(f["dte_sel"]) if k == "dte_sel" and k in f else DEFAULTS[k])
    if signals is not None:
        ss.trades, ss.trades_name = signals, f"{cfg['name']} (saved)"
    else:
        ss.pop("trades", None)
    ss.save_margin = float(cfg.get("margin") or 0.0)
    ss.save_name = cfg["name"]
    ss.loaded_name = cfg["name"]

    msg = f"Loaded '{cfg['name']}': {len(cfg['legs'])} leg(s)"
    if signals is not None:
        msg += f", {len(signals)} signals"
    if trades is not None and legs_df is not None:
        summ = cfg.get("summary") or {}
        ss.result = dict(trades=trades, legs=legs_df, skips=summ.get("skips", {}),
                         sig=None, lot_size=int(cfg["lot_size"]), capital=int(cfg["capital"]),
                         cc=cost_config_from(c), exit_mode="signal" if cfg["exit_mode_ui"].startswith("When") else "fixed",
                         n_signals=int(summ.get("n_signals", len(trades))), used_calendar=bool(summ.get("used_calendar", False)),
                         entries=summ.get("entries", ""), filters=summ.get("filters", ""), pf_mult=summ.get("pf_mult"))
        ss.refresh_sig = True                       # re-fingerprint on the next run so it is not flagged stale
        msg += f", {len(trades)} trades of saved results"
    else:
        ss.pop("result", None)
        msg += ". No results were saved with it, press Run backtest"
    msg += "."
    if int(cfg.get("version", 1)) < 2:
        if any(l.get("link", "off") != "off" and "group" not in l for l in cfg["legs"]):
            msg += (" WARNING: saved by an older version that did not store hedge groups, so linked exits now close ALL "
                    "legs. Check the Hedge group numbers on the Legs tab before trusting the results.")
        if "filters" not in cfg:
            msg += " Days-to-expiry and premium filters were not stored by that version, so they are reset."
    return msg


# --------------------------------------------------------------------------- small UI pieces
def kpi(col, label, value, color="#f8fafc"):
    col.markdown(
        f"<div style='padding:10px 14px;border:1px solid #334155;border-radius:10px;background:#0f172a'>"
        f"<div style='color:#94a3b8;font-size:12px'>{label}</div>"
        f"<div style='color:{color};font-size:22px;font-weight:700'>{value}</div></div>",
        unsafe_allow_html=True)


def pn(v):
    return R.GREEN if v > 0 else R.RED if v < 0 else R.MUTED


def rnd(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes("number").columns
    return df.assign(**{c: df[c].round(2) for c in num})


def legs_summary(legs: list[dict]) -> str:
    return " + ".join(f"{l['action']} {l['lots']}x {l['strike']} {l['otype']}" for l in legs)
