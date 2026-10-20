"""Saved Strategies page: one card per strategy = name, key stats, margin and equity curve."""
from __future__ import annotations

import streamlit as st

import cards
import shared
import storage

st.title("Saved Strategies")


def card(col, cfg: dict):
    name, key = cfg["name"], cfg["_slug"]
    with col.container(border=True):
        head = st.columns([3, 1, 1, 1])
        with head[0]:
            cards.header(cfg)
        if head[1].button("Load", key=f"load_{key}", type="primary", width="stretch",
                          help="Open it in the Strategy Builder with its legs, settings, CSV and results."):
            try:
                st.session_state.flash = shared.apply_saved(name)
                st.switch_page("views/builder.py")
            except FileNotFoundError as ex:
                st.error(str(ex))
        with head[2].popover("Margin", width="stretch"):
            st.caption("Margin you need to run this strategy (Rs). Entered by you, used by the Strategy Combiner.")
            m = st.number_input("Margin required (Rs)", min_value=0.0, step=10_000.0, key=f"mg_{key}",
                                value=cards.margin_of(cfg))
            if st.button("Save margin", key=f"mgs_{key}"):
                storage.set_margin(name, m)
                st.rerun()
        with head[3].popover("Delete", width="stretch"):
            st.write(f"Delete **{name}** and all its files?")
            if st.button("Yes, delete", key=f"del_{key}"):
                storage.delete_strategy(name)
                cards.load_trades.clear()
                st.rerun()
        cards.stats(cfg, key)


items = storage.list_strategies()
if not items:
    st.info("Nothing saved yet. In the Strategy Builder, run a backtest, type a name in the sidebar and press "
            "**Save strategy**. It will appear here with its equity curve.")
    st.stop()

top = st.columns([3, 1])
top[0].caption(f"{len(items)} saved. Load opens a strategy in the builder with its legs, settings, signal CSV and results.")
per_row = top[1].radio("Cards per row", [1, 2], index=1, horizontal=True, label_visibility="collapsed")

for i in range(0, len(items), per_row):
    row = st.columns(per_row)
    for col, cfg in zip(row, items[i:i + per_row]):
        card(col, cfg)
