"""
Option Lab - entry point.   Run:  streamlit run app.py

Pages
    views/builder.py   build and backtest a strategy (signal CSV or time-based entries, multi-leg, costs, results)
    views/saved.py     saved strategies: name, key stats, margin and equity curve, load or delete
    views/combiner.py  pick saved strategies and see their combined profit, drawdown and margin, save the combination
    views/combined.py  saved combinations: open one for its strategies and the full combined analysis
"""
import streamlit as st

import shared

st.set_page_config(page_title="Option Lab", page_icon="📈", layout="wide")
shared.keep_alive()          # stops Streamlit from wiping builder inputs when you visit another page

pages = [
    st.Page("views/builder.py", title="Strategy Builder", icon="🛠️", default=True),
    st.Page("views/saved.py", title="Saved Strategies", icon="📁"),
    st.Page("views/combiner.py", title="Strategy Combiner", icon="🧩"),
    st.Page("views/combined.py", title="Combined Strategies", icon="📊"),
]
st.navigation(pages).run()
