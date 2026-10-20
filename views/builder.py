"""Strategy Builder page."""
from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics as A
import engine as E
import reporting as R
import opt_ui
import results_ui
import shared
import storage
from shared import MODE_CSV, MODE_TIME, kpi, pn, rnd

ss = st.session_state

# ---- labels used by the new controls
CAL_DETECTED = shared.CAL_DETECTED
CAL_RULE = "Exchange rule (Thu to Aug 2025, Tue after)"
PF_METRICS = {
    "win_range": "High-low range inside the entry-exit window",
    "win_oc": "Open-to-close move inside the window",
    "day_range": "Full-day high-low range",
    "day_oc": "Full-day open-to-close move",
}
PREM_SHORT = {">=": "Premium >= value", "<=": "Premium <= value", "between": "Between value and value 2"}
SKIP_SHORT = {"trade": "Skip the whole trade", "leg": "Skip only this leg"}
MC_METHODS = {"shuffle": "Shuffle the order of the same trades", "bootstrap": "Resample trades with replacement"}
MC_UNITS = {"trade": "Per trade", "day": "Per day"}

# defaults for the new widgets (a widget with a key and no `value=` needs its state seeded)
for _k, _v in (("dte_sel", []), ("cal_src", CAL_DETECTED), ("pf_on", False), ("pf_metric", "win_range"),
               ("pf_lookback", 8), ("pf_mult", 1.0), ("mc_sims", 5000), ("mc_method", "shuffle"),
               ("mc_unit", "trade"), ("mc_seed", 0)):
    ss.setdefault(_k, _v)

_LEG_DEFAULTS = E.Leg()


@st.cache_resource(show_spinner=False)
def get_stores(root: str, _nonce: int):
    return E.open_stores(root)


def cb_add():
    ss.legs.append(shared.new_leg_state(E.Leg()))


def _set_leg(ls, lg):
    for f in shared.FIELDS:
        ss[shared.leg_key(ls, f)] = getattr(lg, f)


def cb_dup(uid):
    i = next(k for k, ls in enumerate(ss.legs) if ls["uid"] == uid)
    for ls in ss.legs[i + 1:]:                       # legs below shift down by one: keep their mirror pointing right
        k = ss.get(shared.leg_key(ls, "mirror_of"), ls["init"].get("mirror_of", 0))
        if k > i + 1:
            ss[shared.leg_key(ls, "mirror_of")] = k + 1
    ss.legs.insert(i + 1, shared.new_leg_state(shared.snapshot(ss.legs[i])))


def cb_del(uid):
    i = next(k for k, ls in enumerate(ss.legs) if ls["uid"] == uid)
    resolved = shared.read_legs()
    for j, ls in enumerate(ss.legs[i + 1:], start=i + 1):
        key = shared.leg_key(ls, "mirror_of")
        k = ss.get(key, ls["init"].get("mirror_of", 0))
        if k == i + 1:                               # its source is being deleted: freeze its current settings
            _set_leg(ls, dataclasses.replace(resolved[j], mirror_of=0))
        elif k > i + 1:
            ss[key] = k - 1
    ss.legs = [ls for ls in ss.legs if ls["uid"] != uid]


def cb_add_mirrors():
    """Append a mirrored copy (CALL <-> PUT) of every leg that is not itself a mirror. Hedge group n -> n + 1."""
    resolved = shared.read_legs()
    for i, lg in enumerate(resolved):
        if lg.mirror_of:
            continue
        ss.legs.append(shared.new_leg_state(dataclasses.replace(
            lg, otype=E.flip_otype(lg.otype), mirror_of=i + 1, group=min(lg.group + 1, 9) if lg.group else 0)))


def cb_preset():
    ss.legs = [shared.new_leg_state(l) for l in E.preset_legs(ss.preset_name)]


def W(ls, field):
    """Widget key for a leg field; seeds session_state so widgets need no `value=` argument.
    Falls back to the Leg default for fields that older saved strategies do not have."""
    k = shared.leg_key(ls, field)
    ss.setdefault(k, ls["init"].get(field, getattr(_LEG_DEFAULTS, field)))
    return k


NEW_LEG_FIELDS = ("prem_on", "prem_op", "prem_val", "prem_val2", "prem_skip", "group")


def read_legs():
    """Legs from the widgets, with mirrored legs resolved (see engine.resolve_mirrors)."""
    return shared.read_legs()


def extra_sig():
    """Settings that live outside the legs / costs but change the result."""
    return (tuple(sorted(ss.get("dte_sel", []))), ss.get("cal_src"),
            bool(ss.get("pf_on")) and ss.get("src_mode") == MODE_TIME,
            ss.get("pf_metric"), ss.get("pf_lookback"), ss.get("pf_mult"))


def cur_sig():
    return (shared.signature(legs_now, cc), extra_sig())


def describe_filters(sel, pf):
    parts = []
    if sel:
        parts.append("days to expiry in " + ", ".join(str(x) for x in sel))
    if pf is not None:
        parts.append(f"net premium >= {pf.mult:g} x average move ({PF_METRICS[pf.metric].lower()}, {pf.lookback} days)")
    return "; ".join(parts)


def get_calendar(stores):
    """Expiry calendar built with the method picked on tab 1 (detected from data, or exchange rule). Cached per method."""
    ckey = (stores.week.folder, len(stores.week.files), ss.cal_src)
    if ss.get("cal_key") != ckey:
        if ss.cal_src == CAL_RULE:
            ss.cal = (E.ExpiryCalendar(E.rule_expiries(stores.week.dates())), None)
        else:
            bar = st.progress(0.0, text="Reading every week file once to detect weekly expiries (cached after this)...")
            summ = E.build_day_summary(stores.week, progress=lambda f: bar.progress(min(f, 1.0)))
            bar.empty()
            ss.cal = (E.ExpiryCalendar(E.infer_expiries(summ)), summ)
        ss.cal_key = ckey
    return ss.cal[0]


if ss.get("flash"):
    st.success(ss.pop("flash"))

# =========================================================================== sidebar
sb = st.sidebar
sb.subheader("Data")
sb.text_input("Option data folder", key="root_input",
              help="Paste the full path of the folder that contains `week/` (required) and `next_week/` (optional, add "
                   "later). It is remembered for next time.")
root = ss.root_input.strip().strip('"').strip("'")      # tolerate pasted quotes and spaces
shared.save_setting("data_root", root)                  # remembered for the next launch
stores, data_error = None, None
try:
    stores = get_stores(root, ss.nonce)
    wd = stores.week.dates()
    sb.success(f"week: {len(wd)} days ({wd[0]} to {wd[-1]})")
    if stores.next_week:
        nd = stores.next_week.dates()
        sb.success(f"next_week: {len(nd)} days ({nd[0]} to {nd[-1]})")
    else:
        sb.info("next_week folder not found. Fine until a leg uses next week.")
except E.DataNotFound as ex:
    data_error = str(ex)
    sb.error(data_error)
if sb.button("Rescan folders", width="stretch"):
    ss.nonce += 1
    st.rerun()

sb.subheader("Account")
capital = sb.number_input("Capital (Rs)", min_value=10_000, max_value=100_000_000, step=50_000, key="capital")
lot_size = sb.number_input("Lot size", min_value=1, max_value=1000, step=1, key="lot_size",
                           help="One value for all years, so results stay comparable. Rupee P&L scales with it.")

sb.subheader("Costs")
include_charges = sb.toggle("Include charges", key="include_charges")
brokerage = sb.number_input("Brokerage per order (Rs)", min_value=0.0, max_value=500.0, step=1.0,
                            key="brokerage", disabled=not include_charges)
slippage = sb.number_input("Slippage (% of premium, every fill)", min_value=0.0, max_value=10.0, step=0.05,
                           format="%.2f", key="slippage")
with sb.expander("Charges detail"):
    st.number_input("Exchange charge (% of premium turnover)", min_value=0.0, max_value=1.0, format="%.5f", key="txn_pct")
    st.number_input("Stamp duty (% of buy premium)", min_value=0.0, max_value=1.0, format="%.4f", key="stamp_pct")
    st.checkbox("Override STT %", key="stt_on")
    st.number_input("STT % on sell premium", min_value=0.0, max_value=1.0, format="%.4f", key="stt_val",
                    disabled=not ss.stt_on)
    st.dataframe(pd.DataFrame([
        ["Brokerage", f"Rs {brokerage:g} per order (entry + exit, per leg)"],
        ["STT", f"{ss.stt_val if ss.stt_on else E.stt_rate_pct(dt.date.today())}% of premium, sell side only"],
        ["Exchange", f"{ss.txn_pct}% of premium turnover"],
        ["SEBI", "Rs 10 per crore of turnover"],
        ["GST", "18% on brokerage + exchange + SEBI"],
        ["Stamp duty", f"{ss.stamp_pct}% of buy premium"],
        ["Slippage", f"{slippage}% of premium on every fill"]], columns=["Item", "How it is charged"]),
        hide_index=True, width="stretch")
    st.caption("STT by period: " + "; ".join(f"{s:%b %Y} {r}%" for s, r in E.STT_SCHEDULE[1:]) +
               ". Check against your contract note.")
cc = E.CostConfig(include_charges=include_charges, brokerage_per_order=brokerage, txn_pct=ss.txn_pct,
                  stamp_buy_pct=ss.stamp_pct, stt_override_pct=ss.stt_val if ss.stt_on else None,
                  slippage_pct=slippage)

sb.subheader("Exit")
sb.radio("Legs also exit", ["When the signal exits", "At a fixed time"], key="exit_mode_ui",
         label_visibility="collapsed", disabled=ss.src_mode == MODE_TIME,
         help="Time-based entries always exit at their own exit time.")
sb.time_input("Exit time", key="fixed_exit", disabled=ss.exit_mode_ui == "When the signal exits" or ss.src_mode == MODE_TIME)
exit_mode = "fixed" if (ss.exit_mode_ui == "At a fixed time" and ss.src_mode == MODE_CSV) else "signal"

legs_now = read_legs()
sb.caption(E.brokerage_caption(len(legs_now), cc))
run_clicked = sb.button("Run backtest", type="primary", width="stretch")

sb.divider()
sb.subheader("Save strategy")
ss.setdefault("save_with_results", True)
sb.text_input("Name", key="save_name", placeholder="e.g. ORB credit spread, SL on short leg")
sb.checkbox("Include the trade logs (needed to combine strategies)", key="save_with_results")
sb.number_input("Margin required (Rs, optional)", min_value=0.0, step=10_000.0, key="save_margin",
                help="Type the margin your broker blocks for this strategy. It cannot be calculated from the data here. "
                     "It is shown on the Saved Strategies page and used by the Strategy Combiner. 0 = not set.")
save_clicked = sb.button("Save strategy", width="stretch")
sb.caption("Saves the legs, settings, entry setup, the signal CSV and the trade + leg logs (entry/exit, strikes, "
           "quantity, index level, charges). Open them on the Saved Strategies page.")

# =========================================================================== tabs
st.title("Strategy Builder")
if ss.get("loaded_name"):
    st.caption(f"Working on: **{ss.loaded_name}** (loaded). Change anything, run again, and save under the same or a new name.")
tab_up, tab_legs, tab_res, tab_tr, tab_val, tab_opt = st.tabs(
    ["1 · Entries", "2 · Legs", "3 · Results", "4 · Trades", "5 · Validation", "6 · Optimize"])

# ---- tab 1: entries --------------------------------------------------------
tb_valid, tb_count = True, 0
with tab_up:
    st.radio("Where do the entries come from?", [MODE_CSV, MODE_TIME], horizontal=True, key="src_mode")

    if ss.src_mode == MODE_CSV:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            up = st.file_uploader("Drag trade_logs.csv here", type=["csv"], label_visibility="collapsed")
        with c2:
            st.download_button("Download a sample CSV", E.SAMPLE_LOG.to_csv(index=False), "sample_trade_log.csv",
                               "text/csv", width="stretch")
            st.caption("Intraday signals only: one row per trade, entry and exit on the same day.")
        with st.expander("What columns does the CSV need?", expanded=up is None and "trades" not in ss):
            st.dataframe(pd.DataFrame(E.LOG_COLUMNS, columns=["Column", "Required", "Example", "Meaning"]),
                         hide_index=True, width="stretch")
            st.markdown("**Example rows**")
            st.dataframe(E.SAMPLE_LOG, hide_index=True, width="stretch")
            st.caption("Column names are case-insensitive. Your own backtest output (trade_type, entry_time, exit_time, "
                       "exit_reason, sl_price ...) works as it is. Dates must exist in your option data folder.")
        if up is not None:
            try:
                clean, errs, warns = E.validate_trade_log(pd.read_csv(up))
            except Exception as ex:
                clean, errs, warns = None, [f"Could not read the file as CSV: {ex}"], []
            for m in errs:
                st.error(m)
            for m in warns:
                st.warning(m)
            if clean is not None:
                ss.trades, ss.trades_name = clean, up.name
        if "trades" in ss:
            t = ss.trades
            st.success(f"{len(t)} trades loaded from {ss.get('trades_name')}: "
                       f"{t['entry_time'].min():%d %b %Y} to {t['entry_time'].max():%d %b %Y}. "
                       f"LONG {int((t.trade_type == 'LONG').sum())}, SHORT {int((t.trade_type == 'SHORT').sum())}.")
            st.dataframe(t.head(10), hide_index=True, width="stretch")
    else:
        st.markdown("**Enter at a fixed time, exit at a fixed time, same day.** No CSV needed. "
                    "Every leg is traded exactly as written on each entry.")
        a, b = st.columns(2)
        a.time_input("Entry time", key="tb_entry", step=60)
        b.time_input("Exit time", key="tb_exit", step=60)
        tb_valid = ss.tb_exit > ss.tb_entry
        if not tb_valid:
            st.error("Exit time must be after entry time. Positions are always closed the same day.")
        st.multiselect("Trade on", E.WEEKDAY_NAMES, key="tb_days")
        st.number_input("Max trades per day", min_value=1, max_value=10, step=1, key="tb_max",
                        help="1 = enter once, at the entry time. More than 1 = whenever ALL legs have been closed early "
                             "(stop, target or linked exit) the strategy may enter again, up to this many trades a day.")
        if ss.tb_max > 1:
            r1, r2 = st.columns(2)
            r1.number_input("Wait before re-entering (minutes)", min_value=0, max_value=120, step=1, key="tb_cool")
            r2.time_input("No new entries after", key="tb_last", step=60)
            if ss.tb_last <= ss.tb_entry:
                st.warning("'No new entries after' is not later than the entry time, so no re-entry can happen.")
            st.caption("Re-entry happens only when every leg was closed before the exit time. Legs with no stop or target "
                       "(or no linked exit) run to the exit time, so the day ends after that one trade.")
        if stores is not None:
            lo, hi = stores.week.dates()[0], stores.week.dates()[-1]
            for k, default in (("tb_start", lo), ("tb_end", hi)):
                v = ss.get(k) or default
                ss[k] = min(max(v, lo), hi)
            d1, d2 = st.columns(2)
            d1.date_input("From", key="tb_start", min_value=lo, max_value=hi)
            d2.date_input("To", key="tb_end", min_value=lo, max_value=hi)
            if tb_valid and ss.tb_days:
                tb_count = len(E.generate_time_signals(stores.week, ss.tb_entry, ss.tb_exit,
                                                       [E.WEEKDAY_NAMES.index(d) for d in ss.tb_days],
                                                       ss.tb_start, ss.tb_end))
                st.info(f"{tb_count} entry days: {shared.describe_entries()}. "
                        f"The filters below narrow this further when you press Run backtest.")
        st.caption("Overnight holding is not supported: your files cannot tell us the expiry date reliably.")

    # ---- filters (both modes) ----
    st.divider()
    st.markdown("#### Filters")
    f1, f2 = st.columns([1.6, 1.4])
    f1.multiselect("Only enter when calendar days to expiry is", [0, 1, 2, 3, 4, 5, 6], key="dte_sel",
                   help="0 = expiry day. Leave empty to trade every day. Nifty weekly expiry was Thursday until "
                        "28 Aug 2025 and Tuesday after, so use days to expiry instead of a weekday.")
    f2.radio("Expiry calendar", [CAL_DETECTED, CAL_RULE], key="cal_src",
             help="'Detected' infers expiries from the straddle jumps in your files (first run reads every file once, "
                  "then it is cached). 'Exchange rule' uses the Thursday-to-Tuesday rule and moves holidays to the "
                  "previous trading day.")
    with st.container(border=True):
        st.checkbox("Premium filter: skip the entry unless net premium is at least multiplier x average move",
                    key="pf_on", disabled=ss.src_mode != MODE_TIME,
                    help="Net premium = premium of the sold legs minus the bought legs, at the entry fill. The average "
                         "move comes from earlier days only, so there is no look-ahead. With exactly one days-to-expiry "
                         "value selected, the average is taken over earlier days with that same days-to-expiry only.")
        g1, g2, g3 = st.columns([2.2, 1, 1])
        g1.selectbox("Average move is measured as", list(PF_METRICS), format_func=PF_METRICS.get, key="pf_metric",
                     disabled=not ss.pf_on or ss.src_mode != MODE_TIME)
        g2.number_input("Lookback (days)", min_value=2, max_value=60, step=1, key="pf_lookback",
                        disabled=not ss.pf_on or ss.src_mode != MODE_TIME)
        g3.number_input("Multiplier (0 = skip nothing, only record)", min_value=0.0, max_value=10.0, step=0.1, format="%.2f",
                        key="pf_mult",
                        help="0 records the premium-to-move ratio on every trade without skipping any, so the table on "
                             "tab 5 can show every multiplier from one run.",
                        disabled=not ss.pf_on or ss.src_mode != MODE_TIME)
        if ss.src_mode != MODE_TIME:
            st.caption("The premium filter needs fixed entry and exit times, so it works with time-based entries only "
                       "and is ignored for CSV entries.")
    st.caption("A premium condition on a single leg (for example 'only if this option is at least Rs 80') "
               "is set on tab 2, inside each leg.")

# ---- tab 2: legs -----------------------------------------------------------
with tab_legs:
    p1, p2, p3, p4 = st.columns([2, 1, 1, 1.3])
    p1.selectbox("Start from a preset", E.PRESET_NAMES, key="preset_name", label_visibility="collapsed")
    p2.button("Load preset", on_click=cb_preset, width="stretch")
    p3.button("+ Add leg", on_click=cb_add, width="stretch")
    p4.button("⇄ Add mirrored legs", on_click=cb_add_mirrors, width="stretch",
              help="Adds a copy of every leg with CALL and PUT swapped, linked to the original: change the original "
                   "and the mirror follows. Build the call side of an iron condor / iron fly, then press this.")
    st.caption("Presets are starting points. Edit anything. Each entry opens ALL legs that apply to its direction.")
    time_mode = ss.src_mode == MODE_TIME
    if time_mode:
        st.info("Time-based entries have no direction, so 'On signal' and 'Mirror' are not used: every leg is traded as written.")

    res_legs = read_legs()
    for i, ls in enumerate(ss.legs):
        uid = ls["uid"]
        with st.container(border=True):
            if i > 0:
                mkey = W(ls, "mirror_of")
                if ss[mkey] not in range(0, i + 1):
                    ss[mkey] = 0
                mo = st.selectbox("Mirror of", list(range(0, i + 1)), key=mkey,
                                  format_func=lambda k: "Off: set this leg by hand" if k == 0
                                  else f"Leg {k}, with CALL ⇄ PUT swapped",
                                  help="Copies every setting of the chosen leg (action, strike, lots, expiry, stop loss, "
                                       "target, trailing, premium condition, linked exit) with CALL and PUT swapped. "
                                       "Change that leg and this one follows. The hedge group stays this leg's own.")
                if mo:
                    c = st.columns([0.7, 4.6, 1.0, 0.45, 0.45])
                    c[0].markdown(f"**Leg {i + 1}**")
                    c[1].markdown(f"⇄ **{res_legs[i].label()}**  \n"
                                  f"<span style='color:#94a3b8;font-size:13px'>Mirror of Leg {mo}: all settings copied, "
                                  f"CALL/PUT swapped.</span>", unsafe_allow_html=True)
                    c[2].number_input("Hedge group", min_value=0, max_value=9, step=1, key=W(ls, "group"))
                    c[3].button("⧉", key=f"{uid}_dup", on_click=cb_dup, args=(uid,), help="Duplicate this leg")
                    c[4].button("🗑", key=f"{uid}_del", on_click=cb_del, args=(uid,), help="Delete this leg")
                    continue
            h = st.columns([0.7, 0.9, 1.1, 1.1, 1.5, 1.7, 0.45, 0.45])
            h[0].markdown(f"**Leg {i + 1}**")
            h[1].number_input("Lots", min_value=1, max_value=200, key=W(ls, "lots"))
            h[2].radio("Action", ["BUY", "SELL"], horizontal=True, key=W(ls, "action"))
            h[3].radio("Option", ["CALL", "PUT"], horizontal=True, key=W(ls, "otype"))
            applies = h[4].radio("On signal", ["LONG", "SHORT", "BOTH"], horizontal=True, key=W(ls, "applies_to"),
                                 disabled=time_mode)
            h[5].checkbox("Mirror on opposite signal (CALL/PUT flipped)", key=W(ls, "mirror"),
                          disabled=applies == "BOTH" or time_mode)
            h[6].button("⧉", key=f"{uid}_dup", on_click=cb_dup, args=(uid,), help="Duplicate this leg")
            h[7].button("🗑", key=f"{uid}_del", on_click=cb_del, args=(uid,), help="Delete this leg")

            r2 = st.columns([2.0, 1.1, 2.8, 1.0])
            r2[0].selectbox("Expiry", list(E.EXPIRY_CHOICES), format_func=E.EXPIRY_CHOICES.get, key=W(ls, "expiry"))
            r2[1].selectbox("Strike", E.STRIKE_CHOICES, key=W(ls, "strike"))
            r2[2].selectbox("Linked exit", list(E.LINK_CHOICES), format_func=E.LINK_CHOICES.get, key=W(ls, "link"),
                            help="When this leg exits on its own stop (or target), the legs in its hedge group are "
                                 "squared off in that same minute. Group 0 = every leg. Needs Stop loss / Take profit "
                                 "on this leg.")
            r2[3].number_input("Hedge group", min_value=0, max_value=9, step=1, key=W(ls, "group"),
                               help="Legs with the same number (1 to 9) belong together, for example a short call and "
                                    "its long call wing. A linked exit only closes legs of the SAME group, so the other "
                                    "side of a hedged straddle keeps running. 0 = no group: a linked exit closes all legs.")

            r3 = st.columns(3)
            for col, name, pre in ((r3[0], "Take profit", "tp"), (r3[1], "Stop loss", "sl")):
                with col:
                    on = st.checkbox(name, key=W(ls, f"{pre}_on"))
                    x, y = st.columns([1, 1])
                    x.radio("Unit", ["%", "pts"], horizontal=True, key=W(ls, f"{pre}_unit"), disabled=not on,
                            label_visibility="collapsed")
                    y.number_input(name, min_value=0.0, max_value=100000.0, step=1.0, key=W(ls, f"{pre}_val"),
                                   disabled=not on, label_visibility="collapsed")
            with r3[2]:
                on = st.checkbox("Trailing stop", key=W(ls, "trail_on"),
                                 help="For every X in your favour, move the stop loss by Y. Needs Stop loss on.")
                x, y, z = st.columns([1, 1, 1])
                x.radio("Unit", ["%", "pts"], horizontal=True, key=W(ls, "trail_unit"), disabled=not on,
                        label_visibility="collapsed")
                y.number_input("For every", min_value=0.0, max_value=100000.0, step=0.5, key=W(ls, "trail_x"),
                               disabled=not on, help="For every ...")
                z.number_input("Move SL by", min_value=0.0, max_value=100000.0, step=0.5, key=W(ls, "trail_y"),
                               disabled=not on, help="... move the stop by")

            # ---- premium condition on this leg
            r4 = st.columns([1.5, 2.0, 1.1, 1.1, 2.0])
            pon = r4[0].checkbox("Premium condition", key=W(ls, "prem_on"),
                                 help="Only open this leg if its entry price (the open of the entry minute) meets the "
                                      "condition. Premium is the option's own traded price, per unit.")
            pop = r4[1].selectbox("Condition", list(PREM_SHORT), format_func=PREM_SHORT.get, key=W(ls, "prem_op"),
                                  disabled=not pon, label_visibility="collapsed")
            r4[2].number_input("Value (Rs)", min_value=0.0, max_value=100000.0, step=1.0, key=W(ls, "prem_val"),
                               disabled=not pon)
            r4[3].number_input("Value 2 (Rs)", min_value=0.0, max_value=100000.0, step=1.0, key=W(ls, "prem_val2"),
                               disabled=not pon or pop != "between",
                               help="Upper bound, used only for 'Between value and value 2'.")
            r4[4].radio("If not met", list(SKIP_SHORT), format_func=SKIP_SHORT.get, horizontal=True,
                        key=W(ls, "prem_skip"), disabled=not pon,
                        help="'Skip only this leg' can leave a hedged structure unhedged (a naked short). "
                             "'Skip the whole trade' is the safe choice.")

            # ---- IV condition on this leg
            r5 = st.columns([1.5, 2.0, 1.1, 1.1, 2.0])
            ion = r5[0].checkbox("IV condition", key=W(ls, "iv_on"),
                                 help="Only open this leg if implied volatility (%, from the data's iv column) meets the "
                                      "condition. Uses the IV of the minute before the entry fill, so there is no "
                                      "look-ahead. If the data has no IV at that minute, the condition counts as not met.")
            r5[0].radio("IV of", list(E.IV_SRC), format_func=E.IV_SRC.get, key=W(ls, "iv_src"), disabled=not ion,
                        label_visibility="collapsed",
                        help="ATM IV is steadier and almost always present; deep ITM options often have no IV in the data.")
            iop = r5[1].selectbox("IV condition type", list(PREM_SHORT), key=W(ls, "iv_op"), disabled=not ion,
                                  format_func=lambda k: PREM_SHORT[k].replace("Premium", "IV"),
                                  label_visibility="collapsed")
            r5[2].number_input("IV value (%)", min_value=0.0, max_value=500.0, step=0.5, key=W(ls, "iv_val"),
                               disabled=not ion)
            r5[3].number_input("IV value 2 (%)", min_value=0.0, max_value=500.0, step=0.5, key=W(ls, "iv_val2"),
                               disabled=not ion or iop != "between",
                               help="Upper bound, used only for 'Between value and value 2'.")
            r5[4].radio("If IV not met", list(SKIP_SHORT), format_func=SKIP_SHORT.get, horizontal=True,
                        key=W(ls, "iv_skip"), disabled=not ion)

    legs_now = read_legs()
    if legs_now:
        st.markdown("**What an entry will do**")
        for side in (("NONE",) if time_mode else ("LONG", "SHORT")):
            parts = []
            for lg in legs_now:
                ot = lg.effective_otype(side)
                if ot:
                    link = ""
                    if lg.link != "off":
                        link = f" [linked, group {lg.group}]" if lg.group else " [linked, all legs]"
                    prem = ""
                    if lg.iv_on:
                        icond = (f"{lg.iv_val:g} to {lg.iv_val2:g}" if lg.iv_op == "between"
                                 else f"{lg.iv_op} {lg.iv_val:g}")
                        prem += f" [IV {icond}{' (ATM)' if lg.iv_src == 'atm' else ''}]"
                    if lg.prem_on:
                        cond = (f"{lg.prem_val:g} to {lg.prem_val2:g}" if lg.prem_op == "between"
                                else f"{lg.prem_op} {lg.prem_val:g}")
                        prem = f" [premium {cond}]"
                    parts.append(f"{lg.action} {lg.lots}x {lg.strike} {ot} "
                                 f"({E.EXPIRY_CHOICES[lg.expiry].split(',')[0].lower()}){link}{prem}")
            label = "every entry" if side == "NONE" else f"**{side}**"
            st.write(f"On {label}: " + (" + ".join(parts) if parts else "no legs, the signal is skipped"))
        st.markdown("#### Strategy shape")
        shape_sides = ("NONE",) if time_mode else ("LONG", "SHORT")
        shape_tabs = st.tabs([f"On {s_} signal" for s_ in shape_sides]) if len(shape_sides) > 1 else [st.container()]
        for s_, box in zip(shape_sides, shape_tabs):
            with box:
                results_ui.payoff_panel(legs_now, s_, int(lot_size))
        if any(lg.uses_next_week() for lg in legs_now) and stores is not None and not stores.next_week:
            st.warning("A leg uses next week but the next_week data folder was not found. "
                       "Running will stop with 'next week data not found'.")
        if any((lg.prem_on and lg.prem_skip == "leg") or (lg.iv_on and lg.iv_skip == "leg") for lg in legs_now) \
                and len(legs_now) > 1:
            st.warning("A leg is set to 'Skip only this leg' on a multi-leg structure. If a hedge leg is skipped, "
                       "the remaining legs are traded unhedged.")
    else:
        st.info("No legs yet. Load a preset or press + Add leg.")

# =========================================================================== run
if run_clicked:
    ss.pop("run_error", None)
    try:
        if data_error:
            raise E.DataNotFound(data_error)
        E.validate_legs(legs_now, stores)
        if ss.src_mode == MODE_TIME:
            if not tb_valid:
                raise ValueError("Exit time must be after entry time.")
            if not ss.tb_days:
                raise ValueError("Pick at least one weekday to trade on.")
            signals = E.generate_time_signals(stores.week, ss.tb_entry, ss.tb_exit,
                                              [E.WEEKDAY_NAMES.index(d) for d in ss.tb_days], ss.tb_start, ss.tb_end)
            if signals.empty:
                raise ValueError("No trading days match those entry settings.")
        else:
            if "trades" not in ss:
                raise ValueError("Upload a trade log first (tab 1).")
            signals = ss.trades.reset_index(drop=True)

        # ---- expiry calendar (method chosen on tab 1). Always built: the results are broken down by days to expiry.
        sel = sorted(int(x) for x in ss.dte_sel)
        calendar = get_calendar(stores)
        if sel:
            signals = E.filter_by_dte(signals, calendar, sel)
            if signals.empty:
                raise ValueError(f"No entries fall on days to expiry {sel}. Check the expiry calendar setting.")

        # ---- premium vs average move filter (time-based entries only)
        pf, move_stats = None, None
        if ss.pf_on and ss.src_mode == MODE_TIME:
            pf = E.PremiumFilter(True, ss.pf_metric, int(ss.pf_lookback), float(ss.pf_mult))
            mkey = (stores.week.folder, len(stores.week.files), ss.tb_entry, ss.tb_exit)
            if ss.get("move_key") != mkey:
                bar = st.progress(0.0, text="Measuring daily index moves (cached after this)...")
                ss.move_tbl = E.build_move_table(stores.week, ss.tb_entry, ss.tb_exit,
                                                 progress=lambda f: bar.progress(min(f, 1.0)))
                bar.empty()
                ss.move_key = mkey
            move_stats = E.MoveStats(ss.move_tbl, pf.metric, pf.lookback, calendar=calendar,
                                     dte_only=sel[0] if len(sel) == 1 else None)

        bar = st.progress(0.0, text="Pricing trades from minute option data...")
        tr, lg, skips = E.run_backtest(signals, legs_now, cc, stores, calendar, lot_size=lot_size,
                                       exit_mode=exit_mode, fixed_exit=ss.fixed_exit,
                                       progress=lambda f: bar.progress(min(f, 1.0)),
                                       reentry=shared.reentry_from_state(),
                                       prem_filter=pf, move_stats=move_stats)
        bar.empty()
        ss.result = dict(trades=tr, legs=lg, skips=skips, sig=cur_sig(), lot_size=lot_size,
                         capital=capital, cc=cc, exit_mode=exit_mode, n_signals=len(signals),
                         used_calendar=calendar is not None, entries=shared.describe_entries(),
                         filters=describe_filters(sel, pf), pf_mult=pf.mult if pf else None)
        ss.pop("mc", None)                                # an old simulation no longer matches these trades
        if tr.empty:
            ss.run_error = f"No trades could be priced. Reasons: {skips}"
    except (E.DataNotFound, ValueError) as ex:          # includes NextWeekDataNotFound
        ss.run_error = str(ex)
        ss.pop("result", None)

if ss.pop("refresh_sig", False) and ss.get("result"):
    ss.result["sig"] = cur_sig()

# =========================================================================== save
if save_clicked:
    name = (ss.save_name or "").strip()
    res = ss.get("result")
    if not storage.slug(name):
        st.sidebar.error("Give the strategy a name first.")
    elif ss.src_mode == MODE_CSV and "trades" not in ss:
        st.sidebar.error("Upload a trade log first, so it can be saved with the strategy.")
    else:
        fresh = res is not None and res.get("sig") == cur_sig() and not res["trades"].empty
        if ss.save_with_results and not fresh:
            st.sidebar.error("No up-to-date trade logs to save. Press Run backtest first (the logs are needed to combine "
                             "strategies later), or untick 'Include the trade logs' to save the setup only.")
        else:
            with_results = bool(ss.save_with_results)
            summary = None
            if with_results:
                k = R.kpis(res["trades"], capital)
                summary = dict(net=float(k["net"]), max_dd=float(k["max_dd"]), avg_trade=float(k["avg_trade"]),
                               trades=int(k["trades"]), win_rate=float(k["win_rate"]), skips=res["skips"],
                               n_signals=int(res["n_signals"]), used_calendar=bool(res["used_calendar"]),
                               entries=res.get("entries", ""), filters=res.get("filters", ""),
                               pf_mult=res.get("pf_mult"))
            replaced = storage.exists(name)
            try:
                storage.save_strategy(name, shared.current_config(),
                                      signals=ss.trades if ss.src_mode == MODE_CSV else None,
                                      trades=res["trades"] if with_results else None,
                                      legs=res["legs"] if with_results else None, summary=summary)
                ss.loaded_name = name
                st.sidebar.success(f"{'Replaced' if replaced else 'Saved'} '{name}', all parameters verified" +
                                   (" with trade logs." if with_results else " (setup only, cannot be combined)."))
            except ValueError as ex:
                st.sidebar.error(str(ex))

# =========================================================================== tab 3: results
with tab_res:
    if ss.get("run_error"):
        st.error(ss.run_error)
    res = ss.get("result")
    if res is None or res["trades"].empty:
        if not ss.get("run_error"):
            st.info("Set up the entries and legs, then press **Run backtest** in the sidebar.")
    else:
        t, cap = res["trades"], res["capital"]
        if res.get("sig") != cur_sig():
            st.warning("Inputs changed since this result was calculated. Press Run backtest to refresh.")
        if res.get("entries"):
            st.caption(f"Entries: {res['entries']}")
        if res.get("filters"):
            st.caption(f"Filters: {res['filters']}")
        results_ui.kpi_block(t, cap, margin=float(ss.get("save_margin") or 0.0) or None, n_signals=res["n_signals"])
        if res["skips"]:
            st.warning("Skipped entries: " + ", ".join(f"{v} x {k}" for k, v in res["skips"].items()))
        _m = float(ss.get("save_margin") or 0.0)
        results_ui.charts_block(t, base=_m or float(cap), base_label="margin" if _m else "capital",
                                week_folder=stores.week.folder if stores is not None else None)

        dte_key = t["dte"] if "dte" in t and t["dte"].notna().any() else None
        if dte_key is None and stores is not None:       # e.g. results loaded from an older save
            dte_key = t["date"].map(get_calendar(stores).dte)
        results_ui.day_tables(t, dte_key, ss.cal_src)

        with st.expander("Assumptions behind these numbers"):
            ill = float(t["any_illiquid"].mean() * 100) if "any_illiquid" in t else 0.0
            st.markdown(
                f"- Entry fill: **open of the first 1-minute bar** after the entry time. Exit: the exit minute, "
                f"or earlier on the leg's own TP / SL / trailing stop.\n"
                f"- Slippage {res['cc'].slippage_pct}% of premium on every fill. **There is no bid/ask in the files**, "
                f"so real spreads may be wider, especially for deep ITM or far OTM strikes.\n"
                f"- If a stop and a target sit inside the same minute bar, the **stop is assumed to hit first**. A gap "
                f"through a level fills at that bar's open.\n"
                f"- Trailing stops use bar highs/lows, and a moved stop applies from the **next** bar.\n"
                f"- **Linked exits:** when a linked leg stops out, the legs in its hedge group close at the close of that "
                f"same minute (exit reason LINKED). Legs in other groups keep running. The stop fill itself happens "
                f"somewhere inside that minute, so this is approximate.\n"
                f"- **Premium conditions** use the open of the entry minute, the same price as the fill.\n"
                f"- Lot size {res['lot_size']} for every year. STT rate depends on the trade date; check it against your contract note.\n"
                f"- **{ill:.0f}%** of trades had a zero-volume bar at an entry or exit (stale price). Treat those fills with suspicion.")
            if res["used_calendar"] and "cal" in ss:
                exps = ss.cal[0].expiries
                gaps = [(a, b) for a, b in zip(exps[:-1], exps[1:]) if (b - a).days > 8]
                st.markdown(f"- Weekly expiries were **detected from your data** ({len(exps)} found, last: {exps[-1]}). "
                            + (f"{len(gaps)} gap(s) longer than 8 days, e.g. {gaps[0][0]} to {gaps[0][1]}; "
                               f"those weeks may have a wrong days-to-expiry." if gaps else "No gaps."))
        st.download_button("Download trades (CSV)", t.to_csv(index=False), "option_lab_trades.csv", "text/csv")

# =========================================================================== tab 4: trades
with tab_tr:
    res = ss.get("result")
    if res is None or res["trades"].empty:
        st.info("Run a backtest to see the trade-by-trade breakdown.")
    else:
        t = res["trades"].copy()
        sides = ["All"] + sorted(t["side"].unique())
        if len(sides) > 2:
            side = st.radio("Side", sides, horizontal=True)
            if side != "All":
                t = t[t["side"] == side]
        extra = [c for c in ("dte", "entry_premium", "avg_move") if c in t and t[c].notna().any()]
        cols_t = ["date", "side"] + (["trade_no"] if "trade_no" in t and t["trade_no"].max() > 1 else []) + \
                 ["legs", "entry_time", "exit_time"] + extra + \
                 ["gross", "slippage", "brokerage", "charges", "net", "any_illiquid"]
        show = rnd(t[cols_t])
        st.dataframe(show, hide_index=True, width="stretch")
        with st.expander("Leg-by-leg detail (entry/exit premium, exit reason)"):
            lg = res["legs"]
            lg = lg[lg["trade_id"].isin(t["trade_id"])]
            st.dataframe(rnd(lg), hide_index=True, width="stretch")
            st.caption("exit_reason: SIGNAL = held until the exit time, TP / SL = the leg's own target or stop, "
                       "TRAIL = stopped out after the stop had been trailed, LINKED = closed because another leg in its "
                       "hedge group stopped out. entry_premium = net premium collected (sold minus bought legs). "
                       "avg_move = the average move the premium filter compared it with.")

# =========================================================================== tab 5: validation
with tab_val:
    res = ss.get("result")
    if res is None or res["trades"].empty:
        st.info("Run a backtest first. Monte Carlo and rolling returns are calculated from its trades.")
    else:
        t, cap = res["trades"], res["capital"]
        results_ui.validation_block(t, cap, "", res.get("pf_mult"))

# =========================================================================== tab 6: optimize
with tab_opt:
    opt_ui.render(stores, legs_now, cc, lot_size, capital, exit_mode, get_calendar)
