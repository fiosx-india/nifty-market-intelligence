"""
streamlit_app.py
==================
Streamlit Cloud version of the Nifty Market Intelligence dashboard.
Reuses the same market_analysis.py engine — no Flask/HTML needed.

Deploy on share.streamlit.io with:
    Main file path: streamlit_app.py

⚠️ Educational tool. Not financial advice. No prediction is certain.
"""

import streamlit as st
import pandas as pd
import market_analysis as ma

st.set_page_config(
    page_title="Nifty Market Intelligence",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Nifty 50 Market Intelligence")
st.caption("Technical signal + live news + top movers")

st.info(
    "⚠️ This tool gives a **probabilistic** signal based on public data — "
    "not a certain prediction. No tool can guarantee short-term market "
    "direction. Not financial advice."
)

# Cache the report for 5 minutes so it doesn't re-fetch on every interaction
@st.cache_data(ttl=300)
def load_report():
    return ma.get_full_report()

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 Refresh now"):
        load_report.clear()

with st.spinner("Fetching Nifty data, news, and stock movers..."):
    report = load_report()

# --- Signal ---
st.subheader("Signal")
signal = report.get("signal")
if signal:
    direction = signal["direction"]
    arrow = "⬆️" if direction == "UP" else "⬇️"
    color = "green" if direction == "UP" else "red"

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted direction", f"{arrow} {direction}")
    c2.metric("Confidence", f"{signal['confidence']}%")
    c3.metric("Last close", signal["last_close"])

    st.progress(signal["confidence"] / 100)
    st.caption(
        f"As of: {signal['last_time']} · "
        f"Backtest accuracy (recent data): {signal['backtest_accuracy']}%"
    )
else:
    st.error("Technical signal unavailable right now.")

# --- Movers ---
st.subheader("Top Movers (Nifty 50 sample)")
movers = report.get("movers", [])
if movers:
    df = pd.DataFrame(movers)
    df.columns = ["Stock", "% Change"]
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.warning("No movers data available.")

# --- News ---
st.subheader("Latest News")
news = report.get("news", [])
if news:
    for item in news[:20]:
        tag = item.get("tag", "neutral")
        badge = ""
        if tag == "negative":
            badge = " 🔴 *possible negative driver*"
        elif tag == "positive":
            badge = " 🟢 *possible positive driver*"
        st.markdown(f"**{item['title']}**{badge}")
        st.caption(f"{item['source']} · {item['published']}")
        st.divider()
else:
    st.warning("No news available right now.")

# --- Errors (if any partial failures) ---
if report.get("errors"):
    with st.expander("⚠️ Some data sources had issues"):
        for err in report["errors"]:
            st.text(err)
