"""
opt_ui.py - the Optimize tab of the Strategy Builder: pick parameters to vary, enter their value lists, run every
combination, then rank and explore the results. The heavy lifting is in optimizer.py.
"""
from __future__ import annotations

import datetime as dt
import os
import time as _time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as E
import optimizer as O
import reporting as R
import shared
from shared import MODE_TIME

ss = st.session_state

TIME_ROWS = [("entry", "tb_entry", "Entry time", "time"), ("exit", "tb_exit", "Exit time", "time"),
             ("max_trades", "tb_max", "Max trades per day", "int"),
             ("cooldown", "tb_cool", "Wait before re-entering (min)", "int"),
             ("last_entry", "tb_last", "No new entries after", "time")]
METRIC_LABELS = {"trades": "Trades", "net": "Net profit (Rs)", "return_pct": "Return %", "cagr_pct": "CAGR %",
                 "max_dd": "Max drawdown (Rs)", "ret_dd": "Return / drawdown", "pf": "Profit factor",
                 "win_rate": "Win rate %", "avg_trade": "Avg profit per trade (Rs)"}


def _cur(v):
    return v.strftime("%H:%M") if isinstance(v, dt.time) else f"{v:g}" if isinstance(v, float) else str(v)


def _list_input(key: str, kind: str, default: str, label: str):
    """Text box for a value list + parsed preview. -> list or None (on error)."""
    ss.setdefault(key, default)
    st.text_input(label, key=key, label_visibility="collapsed",
                  help="Comma-separated values, or a range from-to/step. Examples: 30, 40, 50 or 20-60/10 or "
                       "09:15-10:00/15 (step in minutes).")
    try:
        vals = O.parse_values(ss[key], kind)
    except ValueError as ex:
        st.error(str(ex))
        return None
    if not vals:
        st.error("Enter at least one value.")
        return None
    st.caption(f"{len(vals)} value(s): " + ", ".join(_cur(v) for v in vals[:12]) + (" ..." if len(vals) > 12 else ""))
    return vals


def _vary(key: str, label: str) -> bool:
    ss.setdefault(key, False)
    return st.checkbox(label, key=key)


# =========================================================================== parameter picker
def _time_dims(time_mode: bool):
    dims, ok = [], True
    st.markdown("##### Entries")
    if not time_mode:
        st.info("Entries come from your CSV, so only the leg parameters below can be optimized.")
        return dims, ok
    for kind, skey, label, vtype in TIME_ROWS:
        c = st.columns([1.6, 3.4])
        with c[0]:
            on = _vary(f"opt_v_{kind}", label)
        with c[1]:
            if on:
                vals = _list_input(f"opt_l_{kind}", vtype, _cur(ss[skey]), label)
                if vals is None:
                    ok = False
                else:
                    dims.append(O.Dim(label, kind, vals))
            else:
                st.caption(f"{_cur(ss[skey])} (current)")
    if any(d.kind in ("cooldown", "last_entry") for d in dims) and not any(
            d.kind == "max_trades" and max(d.values) > 1 for d in dims) and int(ss.tb_max) <= 1:
        st.caption("Cooldown and 'no new entries after' only matter when max trades per day is above 1.")
    return dims, ok


def _leg_dims(legs, leg_states):
    dims, follow, ok = [], {}, True
    st.markdown("##### Legs")
    for i, (lg, ls) in enumerate(zip(legs, leg_states)):
        uid = ls["uid"]
        with st.expander(f"Leg {i + 1}: {lg.label()}", expanded=i == 0):
            if lg.mirror_of:                               # mirrored on tab 2: always moves with its source leg
                k = int(lg.mirror_of) - 1
                while k in follow:
                    k = follow[k]
                follow[i] = k
                st.caption(f"⇄ Mirror of Leg {lg.mirror_of} (set on tab 2): every value varied there is applied here "
                           f"too, with CALL/PUT swapped.")
                continue
            if i > 0:
                opts = ["Own values"] + [f"Same as Leg {k + 1}" for k in range(i)]
                ss.setdefault(f"opt_{uid}_follow", "Own values")
                if ss[f"opt_{uid}_follow"] not in opts:
                    ss[f"opt_{uid}_follow"] = "Own values"
                st.selectbox("Values", opts, key=f"opt_{uid}_follow",
                             help="'Same as Leg N' moves this leg's strike, stop loss, target, trailing stop and premium "
                                  "together with Leg N in every combination, e.g. the call and put of a straddle.")
                if ss[f"opt_{uid}_follow"] != "Own values":
                    k = int(ss[f"opt_{uid}_follow"].split()[-1]) - 1
                    while k in follow:                     # follow a follower -> its leader
                        k = follow[k]
                    follow[i] = k
                    st.caption(f"Follows Leg {k + 1}: whatever is varied there is applied here too.")
                    continue

            # strike
            c = st.columns([1.6, 3.4])
            with c[0]:
                on = _vary(f"opt_{uid}_strike_on", "Strike (ITM / ATM / OTM)")
            with c[1]:
                if on:
                    ss.setdefault(f"opt_{uid}_strike", [lg.strike])
                    vals = st.multiselect("Strikes", E.STRIKE_CHOICES, key=f"opt_{uid}_strike",
                                          label_visibility="collapsed")
                    if vals:
                        dims.append(O.Dim(f"L{i + 1} strike", "strike",
                                          [s for s in E.STRIKE_CHOICES if s in vals], i))
                    else:
                        st.error("Pick at least one strike.")
                        ok = False
                else:
                    st.caption(f"{lg.strike} (current)")

            # stop loss / take profit
            for kind, name, on_cur, unit, val in (("sl", "Stop loss", lg.sl_on, lg.sl_unit, lg.sl_val),
                                                  ("tp", "Take profit", lg.tp_on, lg.tp_unit, lg.tp_val)):
                c = st.columns([1.6, 3.4])
                with c[0]:
                    on = _vary(f"opt_{uid}_{kind}_on", f"{name} ({unit})")
                with c[1]:
                    if on:
                        vals = _list_input(f"opt_{uid}_{kind}", "float", f"{val:g}", name)
                        none = st.checkbox(f"Also test with no {name.lower()}", key=f"opt_{uid}_{kind}_none")
                        if vals is None:
                            ok = False
                        else:
                            dims.append(O.Dim(f"L{i + 1} {kind.upper()} {unit}", kind, ([None] if none else []) + vals, i))
                    else:
                        st.caption(f"{val:g} {unit} (current)" if on_cur else "off (current)")

            # trailing stop
            c = st.columns([1.6, 3.4])
            with c[0]:
                on = _vary(f"opt_{uid}_trail_on", f"Trailing stop ({lg.trail_unit})")
            with c[1]:
                if on:
                    a, b = st.columns(2)
                    with a:
                        st.caption("For every X in favour")
                        xs = _list_input(f"opt_{uid}_trail_x", "float", f"{lg.trail_x:g}", "X")
                    with b:
                        st.caption("move the stop by Y")
                        ys = _list_input(f"opt_{uid}_trail_y", "float", f"{lg.trail_y:g}", "Y")
                    none = st.checkbox("Also test with no trailing stop", key=f"opt_{uid}_trail_none")
                    if xs is None or ys is None:
                        ok = False
                    else:
                        pairs = [(x, y) for x in xs for y in ys if x > 0]
                        dims.append(O.Dim(f"L{i + 1} trail {lg.trail_unit}", "trail", ([None] if none else []) + pairs, i))
                    st.caption("Trailing needs a stop loss; combinations without one are skipped.")
                else:
                    st.caption(f"{lg.trail_x:g}/{lg.trail_y:g} {lg.trail_unit} (current)" if lg.trail_on else "off (current)")

            # premium and IV conditions (same shape: one value, or low/high pairs for 'between')
            for kind, name, unit in (("prem", "Premium condition", "Rs"), ("iv", "IV condition", "%")):
                c = st.columns([1.6, 3.4])
                op, on_cur = getattr(lg, f"{kind}_op"), getattr(lg, f"{kind}_on")
                v1, v2 = getattr(lg, f"{kind}_val"), getattr(lg, f"{kind}_val2")
                extra = " of ATM" if kind == "iv" and lg.iv_src == "atm" else ""
                with c[0]:
                    on = _vary(f"opt_{uid}_{kind}_on", f"{name}{extra} ({op})")
                with c[1]:
                    if on:
                        if op == "between":
                            a, b = st.columns(2)
                            with a:
                                st.caption(f"Lower bound ({unit})")
                                los = _list_input(f"opt_{uid}_{kind}", "float", f"{v1:g}", "low")
                            with b:
                                st.caption(f"Upper bound ({unit})")
                                his = _list_input(f"opt_{uid}_{kind}2", "float", f"{v2:g}", "high")
                            vals = None if los is None or his is None else [(lo, hi) for lo in los for hi in his if hi > lo]
                        else:
                            st.caption(f"{name.split()[0]} {op} value ({unit}). Change the condition type on tab 2.")
                            vals = _list_input(f"opt_{uid}_{kind}", "float", f"{v1:g}", "value")
                        none = st.checkbox(f"Also test with no {name.lower()}", key=f"opt_{uid}_{kind}_none")
                        if vals is None:
                            ok = False
                        elif not vals and not none:
                            st.error("No valid low/high pair (upper must be above lower).")
                            ok = False
                        else:
                            label = f"L{i + 1} {'premium' if kind == 'prem' else 'IV'}"
                            dims.append(O.Dim(label, kind, ([None] if none else []) + vals, i))
                    else:
                        st.caption(((f"{v1:g}-{v2:g}" if op == "between" else f"{op} {v1:g}") + " (current)")
                                   if on_cur else "off (current)")
    return dims, follow, ok


# =========================================================================== load a combination into the builder
def _load_into_builder(rank_label: str):
    opt = ss.get("opt")
    row = opt["rows"].loc[opt["pick"][rank_label]]
    if [ls["uid"] for ls in ss.legs] != opt["uids"]:
        ss.flash = "The legs changed since the optimizer ran (added / removed), so the combination could not be loaded."
        return
    legs, time = O.apply(opt["base_legs"], opt["base_time"], opt["follow"], opt["dims"], opt["values"][int(row["_i"])])
    for ls, lg in zip(ss.legs, legs):
        for f in shared.FIELDS:
            ss[shared.leg_key(ls, f)] = getattr(lg, f)
    if opt["mode"] == "time":
        ss.tb_entry, ss.tb_exit, ss.tb_last = time["entry"], time["exit"], time["last_entry"]
        ss.tb_max, ss.tb_cool = int(time["max_trades"]), int(time["cooldown"])
    ss.flash = f"Loaded combination {rank_label} into tabs 1 and 2. Press Run backtest to see its full results."


# =========================================================================== main
def render(stores, legs, cc, lot_size, capital, exit_mode, get_calendar):
    st.markdown("Tick **Vary** on any parameter and type the values to test. Everything you don't tick stays as set "
                "on tabs 1 and 2. Every combination is backtested and ranked below.")
    if stores is None:
        st.error("Load the option data folder first (sidebar).")
        return
    if not legs:
        st.info("Add legs on tab 2 first.")
        return
    time_mode = ss.src_mode == MODE_TIME
    if not time_mode and "trades" not in ss:
        st.info("Upload a trade log on tab 1, or switch to time-based entries.")
        return

    left, right = st.columns([1, 1], gap="large")
    with left:
        tdims, ok1 = _time_dims(time_mode)
    with right:
        ldims, follow, ok2 = _leg_dims(legs, ss.legs)
    dims = tdims + ldims

    base_time = dict(entry=ss.tb_entry, exit=ss.tb_exit, max_trades=int(ss.tb_max), cooldown=int(ss.tb_cool),
                     last_entry=ss.tb_last, weekdays=[E.WEEKDAY_NAMES.index(d) for d in ss.tb_days],
                     start=ss.get("tb_start"), end=ss.get("tb_end"))
    mode = "time" if time_mode else "csv"

    st.divider()
    n = O.count(dims)
    valid, bad, err = [], {}, None
    if not dims:
        st.info("Nothing is varied yet. Tick **Vary** on at least one parameter.")
    elif ok1 and ok2:
        try:
            valid, bad = O.expand(dims, legs, base_time, follow, mode)
        except ValueError as ex:
            err = str(ex)
    formula = " × ".join(f"{len(d.values)} {d.label}" for d in dims)
    if dims:
        st.markdown(f"**{formula} = {n:,} combinations**" + (f", {sum(bad.values())} invalid and skipped" if bad else "")
                    + (f" → **{len(valid):,} backtests**" if valid else ""))
        if bad:
            st.caption("Skipped: " + "; ".join(f"{v} × {k}" for k, v in bad.items()))
        if err:
            st.error(err)
        elif len(valid) > O.WARN_COMBOS:
            st.warning(f"{len(valid):,} backtests is a lot. Results with this many combinations are also easy to "
                       f"overfit: prefer a coarse grid first, then zoom in.")

    c = st.columns([1, 1, 2])
    cpu = os.cpu_count() or 2
    ss.setdefault("opt_workers", max(1, cpu - 1))
    c[0].number_input("Parallel workers", min_value=1, max_value=cpu, step=1, key="opt_workers",
                      help=f"Backtests run side by side on {cpu} CPU cores. More workers = faster, but the computer is busier.")
    spr = ss.get("opt_sec_per_run")
    if valid and spr:
        c[1].metric("Estimated time", f"~{max(1, round(len(valid) * spr / int(ss.opt_workers)))} s")
    elif valid:
        c[1].caption("A time estimate appears after the first run.")
    run = c[2].button("Run optimizer", type="primary", disabled=not valid or bool(err), width="stretch")

    if run:
        calendar = get_calendar(stores)
        settings = dict(mode=mode, cc=cc, lot_size=int(lot_size), capital=float(capital), exit_mode=exit_mode,
                        fixed_exit=ss.fixed_exit, dte_sel=sorted(int(x) for x in ss.dte_sel),
                        pf=dict(metric=ss.pf_metric, lookback=int(ss.pf_lookback), mult=float(ss.pf_mult))
                        if ss.pf_on and time_mode else None)
        csv = ss.trades.reset_index(drop=True) if mode == "csv" else None
        jobs = [(i, lg_, t_) for i, (_v, lg_, t_) in enumerate(valid)]
        bar = st.progress(0.0, text=f"Running {len(jobs):,} backtests on {int(ss.opt_workers)} worker(s)...")
        results, t0 = {}, _time.time()
        ss.opt = None
        try:
            for done, (i, m) in enumerate(O.run_grid(jobs, stores.week.folder, calendar.expiries, csv, settings,
                                                     int(ss.opt_workers)), 1):
                results[i] = m
                el = _time.time() - t0
                bar.progress(done / len(jobs), text=f"{done:,} / {len(jobs):,} done, {el:.0f} s elapsed, "
                                                   f"~{el / done * (len(jobs) - done):.0f} s left")
        finally:
            bar.empty()
            if results:
                ss.opt_sec_per_run = (_time.time() - t0) * int(ss.opt_workers) / len(results)
                rows = []
                for i, m in results.items():
                    r = {d.label: O.fmt_value(d.kind, v) for d, v in zip(dims, valid[i][0])}
                    r.update(m)
                    r["_i"] = i
                    rows.append(r)
                ss.opt = dict(rows=pd.DataFrame(rows), dims=dims, values=[v for v, _l, _t in valid], follow=follow,
                              base_legs=legs, base_time=base_time, mode=mode, uids=[ls["uid"] for ls in ss.legs],
                              labels=[d.label for d in dims], n_total=len(jobs),
                              when=dt.datetime.now().strftime("%d %b %H:%M"), capital=float(capital))

    if ss.get("opt"):
        _results(ss.opt)


# =========================================================================== results
def _fmt_table(df: pd.DataFrame, labels: list, extra: list = ()) -> pd.DataFrame:
    out = pd.DataFrame({"Rank": range(1, len(df) + 1)})
    for c in labels:
        out[c] = df[c].to_numpy()
    for c in extra:
        out[c] = df[c].round(1).to_numpy()
    num = lambda v, f: "-" if pd.isna(v) else ("inf" if np.isinf(v) else f(v))
    out["Trades"] = df["trades"].to_numpy()
    out["Avg profit / trade"] = [num(v, lambda x: R.inr(x, sign=True)) for v in df["avg_trade"]]
    out["Return %"] = [num(v, lambda x: f"{x:.1f}%") for v in df["return_pct"]]
    out["CAGR %"] = [num(v, lambda x: f"{x:.1f}%") for v in df["cagr_pct"]]
    out["Net profit"] = [R.inr(v, sign=True) for v in df["net"]]
    out["Max drawdown"] = [R.inr(-v) for v in df["max_dd"]]
    out["Return / DD"] = [num(v, lambda x: f"{x:.2f}") for v in df["ret_dd"]]
    out["Profit factor"] = [num(v, lambda x: f"{x:.2f}") for v in df["pf"]]
    out["Win rate"] = [num(v, lambda x: f"{x:.1f}%") for v in df["win_rate"]]
    return out


def _results(opt: dict):
    st.divider()
    labels = opt["labels"]
    df = opt["rows"]
    errs = df[df["error"].notna() & (df["error"] != "no entries")]
    st.markdown(f"### Optimizer results  \n{len(df):,} of {opt['n_total']:,} combinations, run {opt['when']}.")
    if [ls["uid"] for ls in ss.legs] != opt["uids"]:
        st.warning("Legs were added or removed since this run. Rankings still show, but 'Load into builder' is off.")
    if len(errs):
        st.warning(f"{len(errs)} combination(s) failed, e.g. {errs['error'].iloc[0]}")

    f1, f2 = st.columns([1, 3])
    ss.setdefault("opt_min_trades", 30)
    f1.number_input("Min trades", min_value=0, step=10, key="opt_min_trades",
                    help="Combinations with fewer trades are hidden from every ranking: thin results are mostly luck.")
    d = df[df["trades"] >= int(ss.opt_min_trades)].copy()
    f2.caption(f"{len(d):,} combinations have at least {int(ss.opt_min_trades)} trades.")
    if d.empty:
        st.info("No combination has that many trades. Lower 'Min trades'.")
        return

    t_rank, t_heat, t_sens = st.tabs(["🏆 Rankings", "🔥 Heatmap", "📈 Sensitivity"])
    with t_rank:
        r_ret, r_avg, r_custom = st.tabs(["By returns", "By avg profit per trade", "Custom score (equal-weighted)"])
        with r_ret:
            ranked = d.sort_values(["return_pct", "trades"], ascending=False)
            st.dataframe(_fmt_table(ranked, labels), hide_index=True, width="stretch")
            _pick_and_load(ranked, labels, "ret")
        with r_avg:
            ranked = d.sort_values(["avg_trade", "trades"], ascending=False)
            st.dataframe(_fmt_table(ranked, labels), hide_index=True, width="stretch")
            _pick_and_load(ranked, labels, "avg")
        with r_custom:
            ss.setdefault("opt_score_metrics", ["Avg profit per trade", "Returns", "No. of trades"])
            st.multiselect("Metrics in the score (each gets the same weight)", list(O.RANK_METRICS),
                           key="opt_score_metrics")
            chosen = ss.opt_score_metrics
            if not chosen:
                st.info("Pick at least one metric.")
            else:
                ranked = O.custom_score(d, chosen)
                extra = ["Score"] + [f"rank: {m}" for m in chosen]
                st.dataframe(_fmt_table(ranked, labels, extra), hide_index=True, width="stretch")
                st.caption("Each metric is turned into its percentile rank among these combinations (0 = worst, "
                           "100 = best; for max drawdown, smaller is better), and the ranks are averaged with equal "
                           "weight. Returns are in Rs and trade counts in units, so ranking first stops any one metric "
                           "dominating just because its numbers are bigger.")
                _pick_and_load(ranked, labels, "custom")
        st.download_button("Download all combinations (CSV)", df.drop(columns=["_i"]).to_csv(index=False),
                           "optimizer_results.csv", "text/csv")

    with t_heat:
        if len(labels) < 2:
            st.info("Vary at least two parameters to draw a heatmap.")
        else:
            h = st.columns(3)
            xl = h[0].selectbox("Across", labels, index=0, key="opt_hx")
            yl = h[1].selectbox("Down", [l for l in labels if l != xl], index=0, key="opt_hy")
            ml = h[2].selectbox("Colour by", list(METRIC_LABELS), format_func=METRIC_LABELS.get,
                                index=list(METRIC_LABELS).index("return_pct"), key="opt_hm")
            dd = d.assign(**{ml: d[ml].replace([np.inf, -np.inf], np.nan)})
            pv = dd.pivot_table(index=yl, columns=xl, values=ml, aggfunc="mean")
            pv = pv.reindex(index=_order(d[yl]), columns=_order(d[xl]))
            fig = go.Figure(go.Heatmap(z=pv.to_numpy(), x=[str(c) for c in pv.columns], y=[str(i) for i in pv.index],
                                       colorscale="RdYlGn", reversescale=ml == "max_dd",
                                       texttemplate="%{z:.3s}", hovertemplate=f"{xl} %{{x}}<br>{yl} %{{y}}<br>"
                                       f"{METRIC_LABELS[ml]}: %{{z:,.2f}}<extra></extra>"))
            fig.update_layout(template="plotly_dark", height=max(320, 40 * len(pv.index) + 120),
                              margin=dict(l=10, r=10, t=30, b=10), xaxis_title=xl, yaxis_title=yl)
            st.plotly_chart(fig, width="stretch")
            st.caption("Each cell = average over every other varied parameter. Look for a broad green area: one "
                       "green cell surrounded by red is usually luck, not an edge.")

    with t_sens:
        ml = st.selectbox("Metric", list(METRIC_LABELS), format_func=METRIC_LABELS.get,
                          index=list(METRIC_LABELS).index("avg_trade"), key="opt_sm")
        cols = st.columns(2)
        for k, lab in enumerate(labels):
            g = d.assign(_m=d[ml].replace([np.inf, -np.inf], np.nan)).groupby(lab)["_m"].mean()
            g = g.reindex(_order(d[lab]))
            fig = go.Figure(go.Bar(x=[str(i) for i in g.index], y=g.to_numpy(),
                                   marker_color=[R.GREEN if v >= 0 else R.RED for v in g.fillna(0)]))
            fig.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=36, b=10),
                              title=dict(text=lab, font=dict(size=13)), yaxis_title=METRIC_LABELS[ml])
            cols[k % 2].plotly_chart(fig, width="stretch", key=f"opt_sens_{k}")
        st.caption("Average of the metric at each value, across all the other varied parameters. A parameter whose "
                   "bars are all similar barely matters; a smooth slope shows a real effect.")


def _order(s: pd.Series) -> list:
    """Natural order of a parameter's values ('off' first, numbers numerically, strikes ITM10..ATM..OTM10)."""
    vals = list(pd.unique(s))

    def key(v):
        if v == "off":
            return (0, 0)
        if v in E.STRIKE_CHOICES:
            return (1, E.STRIKE_CHOICES.index(v))
        try:
            return (2, float(str(v).replace(":", "").replace("/", ".").split("-")[0]))
        except ValueError:
            return (3, str(v))
    return sorted(vals, key=key)


def _pick_and_load(ranked: pd.DataFrame, labels: list, tag: str):
    top = ranked.head(50)
    opts = {f"#{r + 1}  " + ", ".join(f"{l} {row[l]}" for l in labels): idx
            for r, (idx, row) in enumerate(top.iterrows())}
    ss.opt["pick"] = {**ss.opt.get("pick", {}), **opts}
    c = st.columns([4, 1])
    choice = c[0].selectbox("Combination", list(opts), key=f"opt_pick_{tag}", label_visibility="collapsed")
    c[1].button("Load into builder", key=f"opt_load_{tag}", on_click=_load_into_builder, args=(choice,),
                disabled=[ls["uid"] for ls in ss.legs] != ss.opt["uids"], width="stretch",
                help="Sets tabs 1 and 2 to this combination. Then press Run backtest for the full results, "
                     "and save it as a strategy if you like it.")
