"""
combo_ui.py - the combined-result analysis shared by the Strategy Combiner (live selection) and the Combined
Strategies page (a saved combination, from its own saved trade log).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics as A
import cards
import engine as E
import reporting as R
import results_ui
import shared
import storage

ss = st.session_state


def summary_of(combined: pd.DataFrame, cap: float, total_margin: float, dd_sum: float) -> dict:
    """Headline numbers stored with a saved combination."""
    kc = R.kpis(combined, cap)
    first, last = pd.to_datetime(combined["entry_time"]).min(), pd.to_datetime(combined["exit_time"]).max()
    return dict(net=float(kc["net"]), max_dd=float(kc["max_dd"]),
                pf=float(kc["profit_factor"]) if pd.notna(kc["profit_factor"]) else None,
                win_rate=float(kc["win_rate"]), trades=int(kc["trades"]), avg_trade=float(kc["avg_trade"]),
                capital=cap, cagr_capital=A.cagr(float(kc["net"]), cap, first, last),
                cagr_margin=A.cagr(float(kc["net"]), total_margin, first, last) if total_margin else None,
                dd_if_together=dd_sum)


def member_status(combo: dict, by_slug: dict) -> list[str]:
    """Problems with a saved combination's members today: deleted, or re-saved since the combination was saved."""
    out = []
    for m in combo["members"]:
        cur = by_slug.get(m["slug"])
        if cur is None:
            out.append(f"'{m['name']}' was deleted")
        elif not cur.get("results_saved"):
            out.append(f"'{m['name']}' no longer has trade logs")
        elif cur.get("saved_at") != m.get("saved_at"):
            out.append(f"'{m['name']}' was re-saved since (its results may have changed)")
    return out


def load_into_combiner(combo: dict, by_slug: dict) -> str:
    """Select a saved combination's strategies on the Strategy Combiner with its margins and settings."""
    present = [m for m in combo["members"] if m["slug"] in by_slug and by_slug[m["slug"]].get("results_saved")]
    ss["combine_sel"] = [m["slug"] for m in present]
    for m in present:
        ss[f"cm_{m['slug']}"] = float(m.get("margin") or 0.0)
        ss[f"cm_src_{m['slug']}"] = cards.margin_of(by_slug[m["slug"]])
    ss["cm_common"] = bool(combo.get("common_period", True))
    ss["cm_override"] = bool(combo.get("override", False))
    ss["cm_total"] = float(combo.get("total_margin") or 0.0)
    ss["cmb_save_name"] = combo["name"]
    probs = member_status(combo, by_slug)
    return (f"Loaded '{combo['name']}': {len(present)} of {len(combo['members'])} strategies selected, margins restored."
            + (" Note: " + "; ".join(probs) + "." if probs else ""))


def _week_folder():
    try:
        w = E.resolve_dirs(ss.get("root_input") or shared.DEFAULT_DATA_ROOT)[0]
        return w if os.path.isdir(w) else None
    except Exception:
        return None


def analysis(frames: dict, margins: dict, total_margin: float, cap: float, cal_label: str, key: str,
             after_kpis=None):
    """
    Full analysis of several strategies' trades taken together.
    frames = {strategy name: trades}, margins = {strategy name: Rs}. after_kpis() is drawn right after the headline
    numbers (the Strategy Combiner puts its Save box there).
    """
    combined = pd.concat(frames.values(), ignore_index=True)
    kc = R.kpis(combined, cap)
    rows = []
    for n, f in list(frames.items()) + [("COMBINED", combined)]:
        k = R.kpis(f, cap)
        m = total_margin if n == "COMBINED" else margins.get(n, 0.0)
        rows.append({"Strategy": n, "Trades": k["trades"], "Net profit": k["net"], "Max drawdown": k["max_dd"],
                     "Profit factor": k["profit_factor"], "Win rate %": k["win_rate"], "Avg per trade": k["avg_trade"],
                     "Margin": m, "Return on margin %": 100 * k["net"] / m if m else np.nan,
                     "Drawdown % of margin": 100 * k["max_dd"] / m if m else np.nan})
    tbl = pd.DataFrame(rows)
    dd_sum = float(sum(r["Max drawdown"] for r in rows[:-1]))

    st.markdown("### Combined statistics")
    results_ui.kpi_block(combined, cap, margin=total_margin or None)
    st.caption(f"Combined max drawdown Rs {R.inr(kc['max_dd'])} against Rs {R.inr(dd_sum)} if every strategy's worst "
               f"drawdown happened together"
               + (f" ({100 * kc['max_dd'] / total_margin:.1f}% of margin)." if total_margin else ".")
               + f" Capital = sum of the strategies' capitals (Rs {R.inr(cap)}).")
    if after_kpis is not None:
        after_kpis(combined, kc, dd_sum)

    st.markdown("#### Strategy by strategy")
    show = tbl.copy()
    show["Net profit"] = [R.inr(v, sign=True) for v in tbl["Net profit"]]
    show["Max drawdown"] = [R.inr(-v) for v in tbl["Max drawdown"]]
    show["Profit factor"] = ["-" if pd.isna(v) else f"{v:.2f}" for v in tbl["Profit factor"]]
    show["Win rate %"] = [f"{v:.1f}%" for v in tbl["Win rate %"]]
    show["Avg per trade"] = [R.inr(v, sign=True) for v in tbl["Avg per trade"]]
    show["Margin"] = [R.inr(v) if v else "-" for v in tbl["Margin"]]
    show["Return on margin %"] = ["-" if pd.isna(v) else f"{v:.0f}%" for v in tbl["Return on margin %"]]
    show["Drawdown % of margin"] = ["-" if pd.isna(v) else f"{v:.1f}%" for v in tbl["Drawdown % of margin"]]
    st.dataframe(show, hide_index=True, width="stretch")

    daily = A.daily_pnl(frames)
    fig = go.Figure()
    for n in daily.columns:
        fig.add_scatter(x=daily.index, y=daily[n].cumsum(), name=n, mode="lines", line=dict(width=1.2), opacity=0.7)
    fig.add_scatter(x=daily.index, y=daily.sum(axis=1).cumsum(), name="COMBINED", mode="lines",
                    line=dict(width=3, color=R.GREEN))
    fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      title=dict(text="Cumulative net P&L by strategy (Rs)", font=dict(size=15)),
                      yaxis=dict(gridcolor="rgba(148,163,184,0.15)"), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch", key=f"{key}by_strategy")

    st.markdown("#### How they move together")
    st.dataframe(daily.corr().round(2).style.background_gradient(cmap="RdYlGn_r", vmin=-1, vmax=1), width="stretch")
    st.caption("Correlation of daily P&L. Close to 1 = they win and lose on the same days (little diversification), "
               "near 0 or negative = they cover for each other. Days a strategy did not trade count as 0.")

    results_ui.charts_block(combined, key, base=total_margin or float(cap),
                            base_label="margin" if total_margin else "capital", week_folder=_week_folder())
    results_ui.day_tables(combined, combined["dte"] if "dte" in combined else None, cal_label)

    st.markdown("## Validation")
    results_ui.validation_block(combined, cap, key, show_sweep=False)
    st.caption("Everything above uses closed trades in exit-time order. Intraday dips while trades are open are not "
               "included, and margin is the amount typed in.")
    return combined


def save_combo(name: str, members: list, total_margin: float, override: bool, common: bool, period, combined,
               cap: float, dd_sum: float):
    storage.save_combo(name, dict(members=members, total_margin=float(total_margin), override=bool(override),
                                  common_period=bool(common), period=[str(period[0]), str(period[1])],
                                  summary=summary_of(combined, cap, total_margin, dd_sum)), combined)
