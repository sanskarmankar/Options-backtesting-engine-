"""Strategy Combiner page: click saved strategies to select / deselect them, see how they perform together, save it."""
from __future__ import annotations

import streamlit as st

import analytics as A
import cards
import combo_ui
import shared
import storage

ss = st.session_state
st.title("Strategy Combiner")
st.caption("Click a card to select it, click again to deselect. Needs strategies saved with their trade logs. "
           "Saved combinations are on the **Combined Strategies** page.")
if ss.get("cmb_flash"):
    st.success(ss.pop("cmb_flash"))

items = storage.list_strategies()
if not items:
    st.info("Nothing saved yet. Save strategies from the Strategy Builder first.")
    st.stop()

by_slug = {c["_slug"]: c for c in items}
ss["combine_sel"] = [s for s in ss.get("combine_sel", []) if s in by_slug]      # forget deleted strategies


def toggle(slug: str):
    sel = ss["combine_sel"]
    if slug in sel:
        sel.remove(slug)
    else:
        sel.append(slug)


# The whole card is the click target: an invisible full-size button sits on top of it.
st.markdown("""<style>
div[class*="st-key-tile_"] { position: relative; cursor: pointer; transition: background .15s, border-color .15s; }
div[class*="st-key-tile_"]:hover { border-color: #64748b !important; }
div[class*="st-key-pick_"] { position: absolute !important; inset: 0; z-index: 20; margin: 0 !important; }
div[class*="st-key-pick_"] > div, div[class*="st-key-pick_"] [data-testid="stButton"] { height: 100%; width: 100%; }
div[class*="st-key-pick_"] button { height: 100%; width: 100%; opacity: 0; cursor: pointer; }
</style>""", unsafe_allow_html=True)
for slug in ss["combine_sel"]:
    st.markdown(f"""<style>
div.st-key-tile_{slug} {{ border: 2px solid #22c55e !important; background: rgba(34,197,94,0.10) !important; }}
</style>""", unsafe_allow_html=True)

top = st.columns([3, 1])
top[0].caption(f"{len(ss['combine_sel'])} selected of {len(items)}.")
per_row = top[1].radio("Cards per row", [1, 2], index=1, horizontal=True, label_visibility="collapsed")

for i in range(0, len(items), per_row):
    row = st.columns(per_row)
    for col, cfg in zip(row, items[i:i + per_row]):
        slug = cfg["_slug"]
        with col.container(border=True, key=f"tile_{slug}"):
            if slug in ss["combine_sel"]:
                st.markdown("**✓ Selected**")
            cards.header(cfg)
            has_logs = cards.stats(cfg, f"cmb_{slug}")
            if has_logs:
                st.button("Select", key=f"pick_{slug}", on_click=toggle, args=(slug,))

chosen = ss["combine_sel"]
st.divider()
if len(chosen) < 2:
    st.info("Select at least two strategies to see them combined.")
    st.stop()

# =========================================================================== combined analysis
try:
    logs = storage.load_logs(chosen)
except (ValueError, FileNotFoundError) as ex:
    st.error(str(ex))
    st.stop()

names = {s: by_slug[s]["name"] for s in chosen}
st.markdown("## Combined result")

c1, c2 = st.columns([1, 2])
ss.setdefault("cm_common", True)
common = c1.checkbox("Only the dates every strategy covers", key="cm_common",
                     help="Strategies tested over different periods are cut to their overlap, so the comparison is fair.")
frames, (lo, hi) = A.combine_period({names[s]: logs[s]["trades"] for s in chosen}, common)
if lo > hi or any(f.empty for f in frames.values()):
    st.error("These strategies have no dates in common. Untick 'Only the dates every strategy covers'.")
    st.stop()
c2.caption(f"Period: {lo} to {hi}")

lots = {by_slug[s]["lot_size"] for s in chosen}
if len(lots) > 1:
    st.warning(f"The strategies were tested with different lot sizes ({sorted(lots)}). Rupee results are added as saved.")

# ---- margin: typed by you per strategy
st.markdown("#### Margin (Rs)")
mc = st.columns(len(chosen))
margins = {}
for col, s in zip(mc, chosen):
    saved_m = cards.margin_of(by_slug[s])
    if ss.get(f"cm_src_{s}") != saved_m:              # first time, or the margin was changed on Saved Strategies
        ss[f"cm_{s}"], ss[f"cm_src_{s}"] = saved_m, saved_m
    margins[names[s]] = col.number_input(names[s], min_value=0.0, step=10_000.0, key=f"cm_{s}")
sum_margin = float(sum(margins.values()))
o1, o2 = st.columns([1.3, 1])
ss.setdefault("cm_override", False)
override = o1.checkbox("Use a different total margin", key="cm_override",
                       help="Hedged combinations often need less than the sum. "
                            "Type the total your broker shows for the whole basket.")
if override:
    if not ss.get("cm_total"):
        ss["cm_total"] = sum_margin
    total_margin = o2.number_input("Total margin (Rs)", min_value=0.0, step=10_000.0, key="cm_total")
else:
    total_margin = sum_margin
if not total_margin:
    st.caption("No margin entered yet, so return and drawdown as % of margin are not shown.")

cap = float(sum(by_slug[s]["capital"] for s in chosen))


def save_box(combined, kc, dd_sum):
    with st.container(border=True):
        st.markdown("**💾 Save this combination**")
        sv = st.columns([3, 1])
        ss.setdefault("cmb_save_name", "")
        sv[0].text_input("Combination name", key="cmb_save_name", placeholder="e.g. Straddle + condor, 2 lakh basket",
                         label_visibility="collapsed")
        if sv[1].button("Save combination", type="primary", width="stretch"):
            cname = (ss.cmb_save_name or "").strip()
            replaced = storage.combo_exists(cname)
            members = [dict(slug=s, name=names[s], saved_at=by_slug[s].get("saved_at"),
                            margin=float(margins[names[s]]), lot_size=by_slug[s].get("lot_size"),
                            capital=by_slug[s].get("capital"), entries=cards.entries_label(by_slug[s]),
                            legs=shared.legs_summary(by_slug[s]["legs"])) for s in chosen]
            try:
                combo_ui.save_combo(cname, members, total_margin, override, common, (lo, hi), combined, cap, dd_sum)
                ss["cmb_flash"] = (f"{'Replaced' if replaced else 'Saved'} combination '{cname}'. Open it on the "
                                   f"Combined Strategies page.")
                st.rerun()
            except ValueError as ex:
                st.error(str(ex))
        st.caption("Saves the strategies in it (with their legs and entries), the margins, the period, a snapshot of "
                   "the results and the combined trade log. It appears on the Combined Strategies page.")


cal_labels = sorted({by_slug[s].get("filters", {}).get("cal_src", "as saved") for s in chosen})
combo_ui.analysis(frames, margins, total_margin, cap, " / ".join(cal_labels), "cmb_", after_kpis=save_box)
