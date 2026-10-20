"""
payoff.py - illustrative expiry payoff of the legs you have built, using HYPOTHETICAL prices (no market data).

Fixed assumptions so every user sees the same picture: index at 24,000, implied volatility 14%, 5 days to expiry,
Black-Scholes premiums. Stops, targets and linked exits are not drawn: this is the shape at expiry if every leg is held.
No Streamlit imports.
"""
from __future__ import annotations

import math

import numpy as np

import engine as E

SPOT = 24000.0
IV = 0.14
DAYS = 5
MOVE_PCT = 0.03                 # +/- move used to judge direction


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot: float, strike: float, otype: str, iv: float = IV, days: float = DAYS) -> float:
    t = days / 365.0
    if t <= 0:
        return max(spot - strike, 0.0) if otype == "CALL" else max(strike - spot, 0.0)
    v = iv * math.sqrt(t)
    d1 = (math.log(spot / strike) + 0.5 * v * v) / v
    d2 = d1 - v
    if otype == "CALL":
        return spot * _ncdf(d1) - strike * _ncdf(d2)
    return strike * _ncdf(-d2) - spot * _ncdf(-d1)


def _positions(legs, side: str, lot_size: int):
    atm = int(round(SPOT / E.STRIKE_STEP) * E.STRIKE_STEP)
    out = []
    for lg in legs:
        ot = lg.effective_otype(side)
        if ot is None:
            continue
        k = E.strike_for(atm, ot, lg.strike)
        prem = max(round(bs_price(SPOT, k, ot), 1), 0.5)
        out.append(dict(sign=1 if lg.action == "BUY" else -1, otype=ot, strike=k, qty=int(lg.lots) * lot_size,
                        lots=int(lg.lots), premium=prem, label=f"{lg.action} {lg.lots}x {lg.strike} {ot}"))
    return out


def _pnl(pos, x):
    x = np.asarray(x, float)
    y = np.zeros_like(x)
    for p in pos:
        intrinsic = np.maximum(x - p["strike"], 0) if p["otype"] == "CALL" else np.maximum(p["strike"] - x, 0)
        y += p["sign"] * p["qty"] * (intrinsic - p["premium"])
    return y


def structure_name(pos) -> str:
    """Recognise the common textbook structures; anything else is 'Custom structure'."""
    n = len(pos)
    calls = [p for p in pos if p["otype"] == "CALL"]
    puts = [p for p in pos if p["otype"] == "PUT"]
    same_qty = len({p["qty"] for p in pos}) == 1
    if n == 1:
        p = pos[0]
        return f"{'Long' if p['sign'] > 0 else 'Short'} {p['otype'].lower()}"
    if n == 2 and same_qty:
        a, b = pos
        if a["otype"] == b["otype"] and a["sign"] != b["sign"] and a["strike"] != b["strike"]:
            lo, hi = sorted(pos, key=lambda p: p["strike"])
            if a["otype"] == "CALL":
                return "Bull call spread (debit)" if lo["sign"] > 0 else "Bear call spread (credit)"
            return "Bear put spread (debit)" if hi["sign"] > 0 else "Bull put spread (credit)"
        if len(calls) == 1 and len(puts) == 1 and a["sign"] == b["sign"]:
            kind = "straddle" if calls[0]["strike"] == puts[0]["strike"] else "strangle"
            return f"{'Long' if a['sign'] > 0 else 'Short'} {kind}"
    if n == 4 and len(calls) == 2 and len(puts) == 2 and same_qty:
        sc = [p for p in calls if p["sign"] < 0]
        lc = [p for p in calls if p["sign"] > 0]
        sp = [p for p in puts if p["sign"] < 0]
        lp = [p for p in puts if p["sign"] > 0]
        if len(sc) == len(lc) == len(sp) == len(lp) == 1:
            sc, lc, sp, lp = sc[0], lc[0], sp[0], lp[0]
            if lc["strike"] > sc["strike"] and lp["strike"] < sp["strike"]:
                return "Iron butterfly (iron fly)" if sc["strike"] == sp["strike"] else "Iron condor"
            if lc["strike"] < sc["strike"] and lp["strike"] > sp["strike"]:
                return "Reverse iron butterfly" if lc["strike"] == lp["strike"] else "Reverse iron condor"
    return "Custom structure"


def analyse(legs, side: str = "NONE", lot_size: int = 65) -> dict | None:
    """Payoff curve, max profit / loss, breakevens, net premium and a plain-English bias. None if no leg applies."""
    pos = _positions(legs, side, lot_size)
    if not pos:
        return None
    strikes = sorted({p["strike"] for p in pos})
    lo = min(SPOT * 0.9, strikes[0] - 300)
    hi = max(SPOT * 1.1, strikes[-1] + 300)
    x = np.arange(math.floor(lo), math.ceil(hi) + 1, 1.0)
    y = _pnl(pos, x)

    # slope beyond the last strike (calls only) and below the first strike (puts only)
    right = sum(p["sign"] * p["qty"] for p in pos if p["otype"] == "CALL")
    left = -sum(p["sign"] * p["qty"] for p in pos if p["otype"] == "PUT")      # d(pnl)/dx far below the strikes
    kinks = _pnl(pos, np.array(strikes + [x[0], x[-1]], float))
    max_profit = None if (right > 0 or left < 0) else float(kinks.max())
    max_loss = None if (right < 0 or left > 0) else float(kinks.min())
    up_side_unlimited = "profit" if right > 0 else "loss" if right < 0 else None
    down_side_large = "profit" if left < 0 else "loss" if left > 0 else None

    s = np.sign(y)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    breakevens = [float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])) for i in idx]

    m = SPOT * MOVE_PCT
    p0, pu, pd_ = (float(v) for v in _pnl(pos, [SPOT, SPOT + m, SPOT - m]))
    scale = max(abs(p0), abs(pu), abs(pd_), 1.0)
    if abs(pu - pd_) / scale < 0.25:
        if p0 >= (pu + pd_) / 2:
            bias, bias_long = "Neutral", "Neutral, range-bound: makes money if the index stays near the entry level"
        else:
            bias, bias_long = "Neutral", "Neutral, big-move: makes money if the index moves a lot in either direction"
    elif pu > pd_:
        bias, bias_long = "Directional: bullish", "Directional, bullish: makes money when the index rises"
    else:
        bias, bias_long = "Directional: bearish", "Directional, bearish: makes money when the index falls"

    net_premium = float(sum(-p["sign"] * p["qty"] * p["premium"] for p in pos))       # > 0 = credit received
    return dict(x=x, y=y, positions=pos, name=structure_name(pos), bias=bias, bias_long=bias_long,
                max_profit=max_profit, max_loss=max_loss, up=up_side_unlimited, down=down_side_large,
                breakevens=breakevens, net_premium=net_premium, spot=SPOT)
