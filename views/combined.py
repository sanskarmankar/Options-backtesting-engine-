"""Combined Strategies page: every saved combination as a card; open one for its members and the full analysis."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import cards
import combo_ui
import reporting as R
import shared
import storage

ss = st.session_state


@st.cache_data(show_spinner=False)
def combo_trades(cslug: str, saved_at: str) -> pd.DataFrame | None:
    return storage.read_combo_trades(cslug)


def open_combo(cslug: str | None):
    ss["open_combo"] = cslug


def to_combiner(combo: dict, by_slug: dict):
    ss["cmb_flash"] = combo_ui.load_into_combiner(combo, by_slug)
    ss["goto_combiner"] = True


combos = storage.list_combos()
by_combo = {c["_slug"]: c for c in combos}
by_slug = {c["_slug"]: c for c in storage.list_strategies()}
if ss.pop("goto_combiner", False):
    st.switch_page("views/combiner.py")
if ss.get("open_combo") not in by_combo:
    ss["open_combo"] = None


def kpi_row(c: dict):
    sm = c.get("summary", {})
    margin = float(c.get("total_margin") or 0.0)
    cm = sm.get("cagr_margin")
    k = st.columns(5)
    shared.kpi(k[0], "Net profit", R.inr(sm.get("net", 0), sign=True), shared.pn(sm.get("net", 0)))
    shared.kpi(k[1], "Max drawdown", R.inr(-sm.get("max_dd", 0)), R.RED)
    shared.kpi(k[2], "Avg profit per trade", R.inr(sm.get("avg_trade", 0), sign=True), shared.pn(sm.get("avg_trade", 0)))
    shared.kpi(k[3], "Margin", f"Rs {R.inr(margin)}" if margin else "not set", "#f8fafc" if margin else R.MUTED)
    shared.kpi(k[4], "CAGR on margin", "-" if cm is None or pd.isna(cm) else f"{100 * cm:.1f}%",
               shared.pn(cm) if cm is not None and pd.notna(cm) else R.MUTED)


# =========================================================================== one combination, opened
if ss["open_combo"]:
    c = by_combo[ss["open_combo"]]
    st.button("← All combined strategies", on_click=open_combo, args=(None,))
    st.title(c["name"])
    st.caption(f"Combined strategy · {len(c['members'])} strategies · {c['period'][0]} to {c['period'][1]} · "
               f"saved {c['saved_at'].replace('T', ' ')}")

    st.markdown("### Strategies in this combination")
    rows = []
    for m in c["members"]:
        cur = by_slug.get(m["slug"])
        if cur is None:
            status = "Deleted since"
        elif cur.get("saved_at") != m.get("saved_at"):
            status = "Re-saved since"
        else:
            status = "Unchanged"
        rows.append({"Strategy": m["name"],
                     "Entries": m.get("entries") or (cards.entries_label(cur) if cur else "-"),
                     "Legs": m.get("legs") or (shared.legs_summary(cur["legs"]) if cur else "-"),
                     "Margin": f"Rs {R.inr(m['margin'])}" if m.get("margin") else "-",
                     "Lot size": m.get("lot_size", "-"), "Strategy saved": str(m.get("saved_at", "-")).replace("T", " "),
                     "Status": status})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    margin = float(c.get("total_margin") or 0.0)
    st.caption(f"Total margin Rs {R.inr(margin)}" + (" (typed for the whole basket)" if c.get("override") else
                                                     " (sum of the strategies)") +
               (" · only the dates every strategy covers" if c.get("common_period", True) else " · each strategy's own dates") +
               ". Everything below is from this combination's own saved trade log, so it does not change if a "
                "strategy is re-saved or deleted later.")
    b = st.columns([1.4, 1, 3])
    b[0].button("Edit in Strategy Combiner", on_click=to_combiner, args=(c, by_slug), width="stretch",
                help="Selects the same strategies with these margins on the Strategy Combiner, using their CURRENT "
                     "saved results.")
    with b[1].popover("Delete", width="stretch"):
        st.write(f"Delete the combination **{c['name']}**? The strategies themselves are kept.")
        if st.button("Yes, delete", key="cdel_open"):
            storage.delete_combo(c["name"])
            ss["open_combo"] = None
            st.rerun()
    probs = combo_ui.member_status(c, by_slug)
    if probs:
        st.warning("Since this combination was saved: " + "; ".join(probs) + ".")

    t = combo_trades(c["_slug"], c["saved_at"])
    if t is None or t.empty:
        st.error("This combination's trade log is missing.")
        st.stop()
    order = [m["name"] for m in c["members"]]
    frames = {n: t[t["strategy"] == n] for n in order if (t["strategy"] == n).any()}
    margins = {m["name"]: float(m.get("margin") or 0.0) for m in c["members"]}
    cap = float(c.get("summary", {}).get("capital") or sum(float(m.get("capital") or 0) for m in c["members"]) or 1)
    cal = sorted({(by_slug.get(m["slug"]) or {}).get("filters", {}).get("cal_src", "as saved") for m in c["members"]})
    st.divider()
    combo_ui.analysis(frames, margins, margin, cap, " / ".join(cal), f"cbd_{c['_slug']}_")
    st.stop()

# =========================================================================== all combinations
st.title("Combined Strategies")
if not combos:
    st.info("No combined strategies yet. On the **Strategy Combiner** page, select two or more strategies and press "
            "**Save combination**.")
    st.stop()
top = st.columns([3, 1])
top[0].caption(f"{len(combos)} saved. Open one to see the strategies in it and its full analysis.")
per_row = top[1].radio("Cards per row", [1, 2], index=1, horizontal=True, label_visibility="collapsed", key="cbd_per_row")

for i in range(0, len(combos), per_row):
    row = st.columns(per_row)
    for col, c in zip(row, combos[i:i + per_row]):
        with col.container(border=True):
            h = st.columns([3, 1, 1])
            h[0].markdown(f"### {c['name']}")
            h[0].caption("🧩 " + " + ".join(m["name"] for m in c["members"]) +
                         f"  \n{c['period'][0]} to {c['period'][1]}  \nSaved {c['saved_at'].replace('T', ' ')}")
            h[1].button("Open", key=f"copen_{c['_slug']}", type="primary", width="stretch", on_click=open_combo,
                        args=(c["_slug"],))
            with h[2].popover("Delete", width="stretch"):
                st.write(f"Delete the combination **{c['name']}**? The strategies themselves are kept.")
                if st.button("Yes, delete", key=f"cdel_{c['_slug']}"):
                    storage.delete_combo(c["name"])
                    st.rerun()
            kpi_row(c)
            t = combo_trades(c["_slug"], c["saved_at"])
            if t is not None and not t.empty:
                st.plotly_chart(R.equity_fig(t, compact=True), width="stretch", key=f"ceq_{c['_slug']}")
            probs = combo_ui.member_status(c, by_slug)
            if probs:
                st.caption("⚠️ Since saving: " + "; ".join(probs) + ".")
