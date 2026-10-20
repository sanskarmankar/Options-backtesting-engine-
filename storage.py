"""
storage.py - save / load / list / delete named strategies on disk (no Streamlit imports).

saved_strategies/
    <name>/
        strategy.json        legs, settings, entry builder, summary stats
        signals.csv          the trade log you uploaded (CSV strategies only)
        results_trades.csv   one row per trade   (only if results were saved)
        results_legs.csv     one row per leg     (only if results were saved)
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd

import engine as E

SAVE_DIR = Path(os.environ.get("OPTION_LAB_SAVE_DIR", Path(__file__).resolve().parent / "saved_strategies"))
VERSION = 2          # 2: legs store premium condition + hedge group; filters block added


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\- ]+", "", name).strip().replace(" ", "_")
    return s[:60]


def _dir(name: str) -> Path:
    return SAVE_DIR / slug(name)


def exists(name: str) -> bool:
    return bool(slug(name)) and (_dir(name) / "strategy.json").exists()


REQUIRED_TOP = ("mode", "time", "legs", "costs", "filters", "lot_size", "capital", "exit_mode_ui", "fixed_exit")
REQUIRED_TIME = ("entry", "exit", "weekdays", "max_trades", "cooldown", "last_entry", "start", "end")
REQUIRED_COSTS = ("include_charges", "brokerage", "slippage", "txn_pct", "stamp_pct", "stt_on", "stt_val")
REQUIRED_FILTERS = ("dte_sel", "cal_src", "pf_on", "pf_metric", "pf_lookback", "pf_mult")
REQUIRED_TRADE_COLS = ("trade_id", "date", "entry_time", "exit_time", "gross", "slippage", "brokerage", "charges", "net")
REQUIRED_LEG_COLS = ("trade_id", "leg", "date", "action", "option", "strike", "lots", "entry_time", "exit_time",
                     "entry_raw", "exit_raw", "exit_reason", "net")
REQUIRED_SIGNAL_COLS = ("trade_type", "entry_time", "exit_time")


def validate_config(cfg: dict, signals=None, trades=None, legs=None) -> list[str]:
    """Everything a saved strategy needs to be reloaded and re-run identically. Returns a list of problems."""
    bad = []
    for k in REQUIRED_TOP:
        if k not in cfg:
            bad.append(f"missing setting '{k}'")
    if bad:
        return bad
    for block, keys in (("time", REQUIRED_TIME), ("costs", REQUIRED_COSTS), ("filters", REQUIRED_FILTERS)):
        bad += [f"missing {block}.{k}" for k in keys if k not in cfg[block]]
    if cfg["mode"] not in ("csv", "time"):
        bad.append(f"unknown entry mode '{cfg['mode']}'")
    leg_fields = [f.name for f in dataclasses.fields(E.Leg)]
    leg_objs = []
    for i, l in enumerate(cfg["legs"], 1):
        miss = [f for f in leg_fields if f not in l]
        if miss:
            bad.append(f"leg {i} is missing {', '.join(miss)}")
            continue
        try:
            leg_objs.append(E.Leg(**{f: l[f] for f in leg_fields}))
        except TypeError as ex:
            bad.append(f"leg {i}: {ex}")
    if leg_objs and len(leg_objs) == len(cfg["legs"]):
        try:
            E.check_legs(leg_objs)                  # all together: mirrors refer to earlier legs
        except ValueError as ex:
            bad.append(str(ex))
    if not cfg["legs"]:
        bad.append("no legs")
    if cfg["mode"] == "time":
        t = cfg["time"]
        if not t["weekdays"]:
            bad.append("no weekdays selected")
        if t["entry"] >= t["exit"]:
            bad.append("time-based exit must be after entry")
    elif signals is None or len(signals) == 0:
        bad.append("CSV strategy has no signal rows to save")
    else:
        bad += [f"signals file lacks column '{c}'" for c in REQUIRED_SIGNAL_COLS if c not in signals.columns]
    if (trades is None) != (legs is None):
        bad.append("results need both the trades table and the legs table")
    elif trades is not None:
        if trades.empty:
            bad.append("the trade log is empty")
        bad += [f"trade log lacks column '{c}'" for c in REQUIRED_TRADE_COLS if c not in trades.columns]
        bad += [f"leg log lacks column '{c}'" for c in REQUIRED_LEG_COLS if c not in legs.columns]
    return bad


def save_strategy(name: str, config: dict, signals: pd.DataFrame | None = None,
                  trades: pd.DataFrame | None = None, legs: pd.DataFrame | None = None,
                  summary: dict | None = None) -> Path:
    if not slug(name):
        raise ValueError("Give the strategy a name (letters, numbers, spaces, - and _).")
    d = _dir(name)
    cfg = dict(config)
    cfg.update(version=VERSION, name=name.strip(), saved_at=dt.datetime.now().isoformat(timespec="seconds"),
               signals_file=None, results_saved=False, summary=summary)
    if legs is not None and "qty" not in legs.columns:           # logs from before qty was recorded
        legs = legs.assign(qty=legs["lots"] * int(cfg.get("lot_size", 1)))
    if trades is not None and not trades.empty:
        cfg["logs"] = dict(trades=int(len(trades)), legs=int(len(legs)) if legs is not None else 0,
                           first_date=str(min(trades["date"])), last_date=str(max(trades["date"])),
                           lot_size=cfg.get("lot_size"), capital=cfg.get("capital"))
    try:
        if float(cfg.get("margin") or 0) < 0:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("Not saved: margin must be a number of 0 or more.")
    problems = validate_config(cfg, signals, trades, legs)
    if problems:
        raise ValueError("Not saved, required parameters are missing or invalid: " + "; ".join(problems))
    tmp = SAVE_DIR / f".tmp_{slug(name)}"             # write beside the old copy, swap only when complete and verified
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        if signals is not None:
            signals.to_csv(tmp / "signals.csv", index=False)
            cfg["signals_file"] = "signals.csv"
        if trades is not None and legs is not None:
            trades.to_csv(tmp / "results_trades.csv", index=False)
            legs.to_csv(tmp / "results_legs.csv", index=False)
            cfg["results_saved"] = True
        (tmp / "strategy.json").write_text(json.dumps(cfg, indent=2, default=str))
        back = json.loads((tmp / "strategy.json").read_text())          # verify what is on disk
        if back["legs"] != json.loads(json.dumps(cfg["legs"], default=str)) or validate_config(back, signals, trades, legs):
            raise ValueError("Not saved: the file written to disk did not match the strategy.")
        if signals is not None and len(pd.read_csv(tmp / "signals.csv")) != len(signals):
            raise ValueError("Not saved: signals.csv row count mismatch.")
        if trades is not None and len(pd.read_csv(tmp / "results_trades.csv")) != len(trades):
            raise ValueError("Not saved: results_trades.csv row count mismatch.")
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if d.exists():
        shutil.rmtree(d)
    tmp.rename(d)
    return d


def read_config(name: str) -> dict:
    return json.loads((_dir(name) / "strategy.json").read_text())


def load_strategy(name: str):
    """-> (config, signals_df | None, trades_df | None, legs_df | None)"""
    d = _dir(name)
    if not (d / "strategy.json").exists():
        raise FileNotFoundError(f"No saved strategy called '{name}'.")
    cfg = read_config(name)
    signals = trades = legs = None
    if cfg.get("signals_file") and (d / cfg["signals_file"]).exists():
        signals = pd.read_csv(d / cfg["signals_file"], parse_dates=["entry_time", "exit_time"])
    if cfg.get("results_saved"):
        trades = read_trades(name)
        legs = pd.read_csv(d / "results_legs.csv", parse_dates=["entry_time", "exit_time"])
        legs["date"] = pd.to_datetime(legs["date"]).dt.date
    return cfg, signals, trades, legs


def read_trades(name: str) -> pd.DataFrame | None:
    f = _dir(name) / "results_trades.csv"
    if not f.exists():
        return None
    t = pd.read_csv(f, parse_dates=["entry_time", "exit_time"])
    t["date"] = pd.to_datetime(t["date"]).dt.date
    return t


def read_legs(name: str) -> pd.DataFrame | None:
    f = _dir(name) / "results_legs.csv"
    if not f.exists():
        return None
    g = pd.read_csv(f, parse_dates=["entry_time", "exit_time"])
    g["date"] = pd.to_datetime(g["date"]).dt.date
    return g


def load_logs(names: list[str]) -> dict:
    """Trade and leg logs of several saved strategies, each with a 'strategy' column. For the combine feature.
    -> {name: dict(config=..., trades=df, legs=df)}; raises if a strategy was saved without its logs."""
    out = {}
    for n in names:
        cfg, t, g = read_config(n), read_trades(n), read_legs(n)
        if t is None or g is None:
            raise ValueError(f"'{n}' was saved without its trade logs; open it, run the backtest and save it again.")
        out[n] = dict(config=cfg, trades=t.assign(strategy=cfg["name"]), legs=g.assign(strategy=cfg["name"]))
    return out


def list_strategies() -> list[dict]:
    """Config of every saved strategy, newest first."""
    out = []
    if SAVE_DIR.exists():
        for d in SAVE_DIR.iterdir():
            if d.name.startswith("."):
                continue
            f = d / "strategy.json"
            if f.exists():
                try:
                    cfg = json.loads(f.read_text())
                    cfg["_slug"] = d.name
                    cfg["_mtime"] = f.stat().st_mtime
                    out.append(cfg)
                except Exception:
                    continue
    return sorted(out, key=lambda c: c.get("saved_at", ""), reverse=True)


def set_margin(name: str, margin: float) -> None:
    """Update the margin (Rs) of a saved strategy without touching its logs."""
    margin = float(margin)
    if margin < 0:
        raise ValueError("Margin cannot be negative.")
    f = _dir(name) / "strategy.json"
    cfg = json.loads(f.read_text())
    cfg["margin"] = margin
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2, default=str))
    tmp.replace(f)


def delete_strategy(name: str) -> None:
    d = _dir(name)
    if d.exists() and SAVE_DIR in d.parents:
        shutil.rmtree(d)


# =========================================================================== saved combinations (Strategy Combiner)
# saved_combinations/<name>/combination.json   members, margins, settings, summary at save time
#                          /combined_trades.csv every member's trades in the period, with a 'strategy' column
COMBO_DIR = Path(os.environ.get("OPTION_LAB_COMBO_DIR", SAVE_DIR.with_name("saved_combinations")))


def _combo_dir(name: str) -> Path:
    return COMBO_DIR / slug(name)


def combo_exists(name: str) -> bool:
    return bool(slug(name)) and (_combo_dir(name) / "combination.json").exists()


def save_combo(name: str, cfg: dict, trades: pd.DataFrame) -> Path:
    """cfg: members=[{slug, name, saved_at, margin, lot_size}], total_margin, override, common_period, period, summary."""
    if not slug(name):
        raise ValueError("Give the combination a name (letters, numbers, spaces, - and _).")
    if len(cfg.get("members", [])) < 2:
        raise ValueError("A combination needs at least two strategies.")
    if trades is None or trades.empty:
        raise ValueError("The combination has no trades to save.")
    cfg = dict(cfg, version=1, name=name.strip(), saved_at=dt.datetime.now().isoformat(timespec="seconds"))
    d, tmp = _combo_dir(name), COMBO_DIR / f".tmp_{slug(name)}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        trades.to_csv(tmp / "combined_trades.csv", index=False)
        (tmp / "combination.json").write_text(json.dumps(cfg, indent=2, default=str))
        back = json.loads((tmp / "combination.json").read_text())
        if [m["slug"] for m in back["members"]] != [m["slug"] for m in cfg["members"]] or \
                len(pd.read_csv(tmp / "combined_trades.csv")) != len(trades):
            raise ValueError("Not saved: the files written to disk did not match.")
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if d.exists():
        shutil.rmtree(d)
    tmp.rename(d)
    return d


def list_combos() -> list[dict]:
    """Every saved combination's config, newest first."""
    out = []
    if COMBO_DIR.exists():
        for d in COMBO_DIR.iterdir():
            f = d / "combination.json"
            if d.name.startswith(".") or not f.exists():
                continue
            try:
                cfg = json.loads(f.read_text())
                cfg["_slug"] = d.name
                out.append(cfg)
            except Exception:
                continue
    return sorted(out, key=lambda c: c.get("saved_at", ""), reverse=True)


def read_combo_trades(name: str) -> pd.DataFrame | None:
    f = _combo_dir(name) / "combined_trades.csv"
    if not f.exists():
        return None
    t = pd.read_csv(f, parse_dates=["entry_time", "exit_time"])
    t["date"] = pd.to_datetime(t["date"]).dt.date
    return t


def delete_combo(name: str) -> None:
    d = _combo_dir(name)
    if d.exists() and COMBO_DIR in d.parents:
        shutil.rmtree(d)
