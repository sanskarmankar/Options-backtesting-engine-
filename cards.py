"""cards.py - strategy card body shared by the Saved Strategies and Strategy Combiner pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import reporting as R
import shared
import storage


@st.cache_data(show_spinner=False)
def load_trades(name: str, mtime: float) -> pd.DataFrame | None:
    return storage.read_trades(name)


def entries_label(cfg: dict) -> str:
    if cfg["mode"] == "time":
        t = cfg["time"]
        days = "every weekday" if len(t["weekdays"]) == 5 else ", ".join(t["weekdays"])
        more = f", up to {t['max_trades']}/day" if t.get("max_trades", 1) > 1 else ""
        return f"Time-based, {days}, {t['entry']} to {t['exit']}{more}"
    n = (cfg.get("summary") or {}).get("n_signals")
    return "Signal CSV" + (f" ({n} signals)" if n else "")


def margin_of(cfg: dict) -> float:
    return float(cfg.get("margin") or 0.0)


def header(cfg: dict):
    st.markdown(f"### {cfg['name']}")
    st.caption(f"{entries_label(cfg)}  \n{shared.legs_summary(cfg['legs'])}  \nSaved {cfg['saved_at'].replace('T', ' ')}")


def stats(cfg: dict, key: str) -> bool:
    """KPI tiles + compact equity curve. Returns False when the strategy has no saved trade logs."""
    t = load_trades(cfg["_slug"], cfg["_mtime"]) if cfg.get("results_saved") else None
    if t is None or t.empty:
        st.info("No trade logs were saved with this strategy. Load it, press Run backtest, and save again with "
                "'Include the trade logs' ticked.")
        return False
    k = R.kpis(t, cfg["capital"])
    margin = margin_of(cfg)
    s = st.columns(4)
    shared.kpi(s[0], "Net profit", R.inr(k["net"], sign=True), shared.pn(k["net"]))
    shared.kpi(s[1], "Max drawdown", R.inr(-k["max_dd"]), R.RED)
    shared.kpi(s[2], "Avg profit per trade", R.inr(k["avg_trade"], sign=True), shared.pn(k["avg_trade"]))
    shared.kpi(s[3], "Margin required", f"Rs {R.inr(margin)}" if margin else "not set", "#f8fafc" if margin else R.MUTED)
    st.plotly_chart(R.equity_fig(t, compact=True), width="stretch", key=f"eq_{key}")
    st.caption(f"{k['trades']} trades" + (f"  |  return on margin {100 * k['net'] / margin:.0f}%" if margin else ""))
    return True
