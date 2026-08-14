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

# --- Bottom Reversal / Breakdown Scanner ---
st.divider()
st.subheader("🔍 Bottom Reversal / Breakdown Scanner")
st.caption(
    "Scans ~200 liquid NSE stocks for two patterns: stocks near their "
    "52-week LOW that have started moving up recently, and stocks near "
    "their 52-week HIGH that have started moving down recently."
)
st.warning(
    "⚠️ 'Target' below is the stock's own recent historical high/low — "
    "a real price level it has touched before, NOT a prediction of where "
    "or when it will go next. No exact price-and-time forecast is possible."
)

@st.cache_data(ttl=1800)  # scanning ~200 stocks is expensive; cache 30 min
def load_scan():
    return ma.scan_bottom_and_breakdown(top_n=10)

with st.spinner("Scanning ~200 stocks for reversal patterns..."):
    scan = load_scan()

tab1, tab2 = st.tabs(["⬆️ Rising from lows", "⬇️ Falling from highs"])

with tab1:
    bottoming = scan.get("bottoming", [])
    if bottoming:
        df_b = pd.DataFrame(bottoming)
        df_b = df_b[[
            "symbol", "current_price", "pct_above_52w_low",
            "recent_5d_change", "rsi_14", "target_recent_high", "high_date",
        ]]
        df_b.columns = [
            "Stock", "Price", "% above 52w low", "5-day change %",
            "RSI", "Reference target (recent high)", "Target hit on",
        ]
        st.dataframe(df_b, use_container_width=True, hide_index=True)
    else:
        st.info("No stocks currently match this pattern.")

with tab2:
    topping = scan.get("topping", [])
    if topping:
        df_t = pd.DataFrame(topping)
        df_t = df_t[[
            "symbol", "current_price", "pct_below_52w_high",
            "recent_5d_change", "rsi_14", "target_recent_low", "low_date",
        ]]
        df_t.columns = [
            "Stock", "Price", "% below 52w high", "5-day change %",
            "RSI", "Reference target (recent low)", "Target hit on",
        ]
        st.dataframe(df_t, use_container_width=True, hide_index=True)
    else:
        st.info("No stocks currently match this pattern.")

# --- Experimental: 1h/2h/3h projection ---
st.divider()
st.subheader("🧪 1h / 2h / 3h Projection (EXPERIMENTAL)")
st.error(
    "⚠️ **Read this before trusting any number below.** Each stock shows "
    "a **backtest accuracy** — the model's actual hit-rate on that stock's "
    "own recent history. An accuracy near 50% means the prediction is no "
    "better than a coin flip. This is normal and expected — short-term "
    "price direction is inherently very hard to predict. Use the accuracy "
    "numbers to judge for yourself whether to trust a given signal; do not "
    "assume it works. Not financial advice."
)

@st.cache_data(ttl=1800)
def load_projection():
    return ma.get_projection_report(top_n=10)

if st.button("▶️ Run 1h/2h/3h projection (takes ~1-2 min)"):
    with st.spinner("Training per-stock models for each stock and horizon..."):
        proj = load_projection()

    ptab1, ptab2 = st.tabs(["⬆️ Rising candidates", "⬇️ Falling candidates"])

    def render_projection_table(items):
        if not items:
            st.info("No data available.")
            return
        rows = []
        for item in items:
            row = {
                "Stock": item["symbol"],
                "Price": item["current_price"],
                "Reference target": item.get("reference_target", "-"),
            }
            for h in [1, 2, 3]:
                hd = item["horizons"].get(f"{h}h")
                if hd:
                    row[f"{h}h direction"] = hd["direction"]
                    row[f"{h}h backtest acc %"] = hd["backtest_accuracy"]
                    row[f"{h}h expected range %"] = hd["expected_move_range_pct"]
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with ptab1:
        render_projection_table(proj.get("bottoming", []))
    with ptab2:
        render_projection_table(proj.get("topping", []))
else:
    st.caption("Click the button above to run (not run automatically — it's slow and experimental).")
