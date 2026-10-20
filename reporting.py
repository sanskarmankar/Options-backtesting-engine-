"""
reporting.py - everything that turns the trade table into numbers, charts and the year x month table.
Colour language (dark theme): profit = green, loss = red, charges = yellow, brokerage = white, slippage = red.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

GREEN, RED, YELLOW, WHITE = "#22c55e", "#ef4444", "#eab308", "#f8fafc"
YELLOW_DARK, YELLOW_BRIGHT, MUTED = "#ca8a04", "#facc15", "#64748b"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------- formatting
def inr(v: float, sign: bool = False, decimals: int = 0) -> str:
    """Indian digit grouping (12,34,567)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    neg = v < 0
    s = f"{abs(v):.{decimals}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    out = whole + (("." + frac) if frac else "")
    if neg:
        return "-" + out
    return ("+" + out) if sign and v > 0 else out


def max_drawdown(net: pd.Series) -> float:
    """Largest peak-to-trough fall of the cumulative P&L (Rs, positive number). Peak starts at 0."""
    if len(net) == 0:
        return 0.0
    cum = np.concatenate([[0.0], np.cumsum(net.to_numpy(float))])
    return float((np.maximum.accumulate(cum) - cum).max())


# --------------------------------------------------------------------------- KPIs
def kpis(t: pd.DataFrame, capital: float) -> dict:
    t = t.sort_values("exit_time")
    net = t["net"]
    wins, losses = net[net > 0].sum(), net[net < 0].sum()
    dd = max_drawdown(net)
    costs = t["charges"].sum() + t["brokerage"].sum() + t["slippage"].sum()
    return dict(net=net.sum(), gross=t["gross"].sum(), costs=costs, trades=len(t),
                avg_win=net[net > 0].mean() if (net > 0).any() else 0.0,
                avg_loss=net[net < 0].mean() if (net < 0).any() else 0.0,
                best=net.max() if len(t) else 0.0, worst=net.min() if len(t) else 0.0,
                win_rate=(net > 0).mean() * 100 if len(t) else 0.0,
                profit_factor=wins / abs(losses) if losses else np.nan,
                avg_trade=net.mean() if len(t) else 0.0, max_dd=dd,
                max_dd_pct=dd / capital * 100 if capital else np.nan,
                costs_pct_gross=costs / abs(t["gross"].sum()) * 100 if t["gross"].sum() else np.nan)


# --------------------------------------------------------------------------- equity curve
def _sign_split(x: pd.Series, y: np.ndarray):
    """Insert the exact zero-crossing points so green/red segments meet at zero."""
    xs, ys = [x.iloc[0].value], [y[0]]
    for i in range(1, len(y)):
        a, b = y[i - 1], y[i]
        if a * b < 0:
            frac = a / (a - b)
            xs.append(int(x.iloc[i - 1].value + frac * (x.iloc[i].value - x.iloc[i - 1].value)))
            ys.append(0.0)
        xs.append(x.iloc[i].value)
        ys.append(b)
    return pd.to_datetime(xs), np.array(ys, float)


BENCH = "#60a5fa"


def equity_fig(t: pd.DataFrame, compact: bool = False, bench: tuple | None = None) -> go.Figure:
    """bench = (x, y, name): optional benchmark P&L line in Rs (e.g. NIFTY buy and hold) drawn on the same axis."""
    t = t.sort_values("exit_time")
    x = pd.concat([pd.Series([t["entry_time"].iloc[0]]), t["exit_time"]], ignore_index=True)
    y = np.concatenate([[0.0], t["net"].cumsum().to_numpy(float)])
    xs, ys = _sign_split(x, y)
    pos = np.where(ys >= 0, ys, np.nan)
    neg = np.where(ys <= 0, ys, np.nan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=pos, mode="lines", name="Profit", line=dict(color=GREEN, width=2),
                             fill="tozeroy", fillcolor="rgba(34,197,94,0.25)", connectgaps=False,
                             hovertemplate="%{x|%d %b %Y %H:%M}<br>Cumulative P&L: %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=neg, mode="lines", name="Loss", line=dict(color=RED, width=2),
                             fill="tozeroy", fillcolor="rgba(239,68,68,0.25)", connectgaps=False,
                             hovertemplate="%{x|%d %b %Y %H:%M}<br>Cumulative P&L: %{y:,.0f}<extra></extra>"))
    if bench is not None:
        bx, by, bname = bench
        fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", name=bname, line=dict(color=BENCH, width=1.8, dash="dot"),
                                 hovertemplate="%{x|%d %b %Y}<br>" + bname + ": %{y:,.0f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(template="plotly_dark", height=280 if compact else 380,
                      margin=dict(l=10, r=10, t=10 if compact else 30, b=10), showlegend=bench is not None,
                      legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
                      title=None if compact else dict(text="Equity curve: cumulative net P&L (Rs)", font=dict(size=15)),
                      xaxis=dict(title="Time", showgrid=False),
                      yaxis=dict(title="Rs", gridcolor="rgba(148,163,184,0.15)", zeroline=False),
                      hovermode="x unified")
    return fig


# --------------------------------------------------------------------------- year x month table
def year_month_html(t: pd.DataFrame) -> str:
    t = t.copy()
    t["_dt"] = pd.to_datetime(t["exit_time"])
    t["year"], t["month"] = t["_dt"].dt.year, t["_dt"].dt.month
    t = t.sort_values("_dt")

    css = f"""
    <style>
    .ym {{ width:100%; border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }}
    .ym th {{ color:#94a3b8; font-weight:600; text-align:right; padding:8px 10px; border-bottom:1px solid #334155; white-space:nowrap; }}
    .ym td {{ text-align:right; padding:8px 10px; border-bottom:1px solid #1e293b; white-space:nowrap; }}
    .ym td.yr, .ym th.yr {{ text-align:left; color:{WHITE}; font-weight:700; }}
    .ym tr.all td {{ border-top:2px solid #475569; font-weight:700; }}
    .ym .sep {{ border-left:1px solid #334155; }}
    .wrap {{ overflow-x:auto; }}
    </style>"""

    def money_cell(v, cls=""):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return f'<td class="{cls}" style="color:{MUTED}">-</td>'
        color = GREEN if v > 0 else RED if v < 0 else MUTED
        return f'<td class="{cls}" style="color:{color}">{inr(v)}</td>'

    head = ("<tr><th class='yr'>Year</th>" + "".join(f"<th>{m}</th>" for m in MONTHS) +
            "<th class='sep'>Total profit</th><th>Max drawdown</th><th>Charges</th><th>Brokerage</th><th>Trades</th></tr>")
    rows = []
    for yr, g in t.groupby("year"):
        cells = [f"<td class='yr'>{yr}</td>"]
        by_m = g.groupby("month")["net"].sum()
        for m in range(1, 13):
            cells.append(money_cell(by_m.get(m)))
        cells.append(money_cell(g["net"].sum(), "sep"))
        cells.append(f"<td style='color:{RED}'>{inr(-max_drawdown(g['net'])) if max_drawdown(g['net']) else '0'}</td>")
        cells.append(f"<td style='color:{YELLOW}'>{inr(g['charges'].sum())}</td>")
        cells.append(f"<td style='color:{WHITE}'>{inr(g['brokerage'].sum())}</td>")
        cells.append(f"<td style='color:{WHITE}'>{len(g)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    allc = ["<td class='yr'>All</td>"]
    by_m = t.groupby("month")["net"].sum()
    for m in range(1, 13):
        allc.append(money_cell(by_m.get(m)))
    allc.append(money_cell(t["net"].sum(), "sep"))
    allc.append(f"<td style='color:{RED}'>{inr(-max_drawdown(t['net']))}</td>")
    allc.append(f"<td style='color:{YELLOW}'>{inr(t['charges'].sum())}</td>")
    allc.append(f"<td style='color:{WHITE}'>{inr(t['brokerage'].sum())}</td>")
    allc.append(f"<td style='color:{WHITE}'>{len(t)}</td>")
    rows.append("<tr class='all'>" + "".join(allc) + "</tr>")
    return css + f"<div class='wrap'><table class='ym'>{head}{''.join(rows)}</table></div>"


# --------------------------------------------------------------------------- cost bars
def cost_fig(t: pd.DataFrame, include_profit: bool = True) -> go.Figure:
    stt, gst, exch = t["stt"].sum(), t["gst"].sum(), t["exchange"].sum()
    sebi, stamp = t["sebi"].sum(), t["stamp"].sum()
    charges = t["charges"].sum()
    labels = ["STT", "GST", "Exchange", "SEBI", "Stamp duty", "Total charges", "Brokerage", "Slippage"]
    vals = [stt, gst, exch, sebi, stamp, charges, t["brokerage"].sum(), t["slippage"].sum()]
    cols = [YELLOW_DARK] * 5 + [YELLOW_BRIGHT, WHITE, RED]
    if include_profit:
        net = t["net"].sum()
        labels.append("Total profit (net)")
        vals.append(net)
        cols.append(GREEN if net >= 0 else RED)
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=cols, text=[inr(v) for v in vals],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}: Rs %{y:,.0f}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=40, b=10),
                      title=dict(text="Where the money went (Rs)", font=dict(size=15)),
                      xaxis=dict(title=None, tickangle=0), yaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
                      showlegend=False, uniformtext_minsize=10)
    return fig
