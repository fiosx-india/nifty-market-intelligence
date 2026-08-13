"""
market_analysis.py
====================
Core "engine" for the Nifty Market Intelligence app.
Contains all data-fetching and analysis functions, reused by app.py.

⚠️ Educational tool. Not financial advice. No prediction is certain.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

try:
    import feedparser
except ImportError:
    feedparser = None


# ----------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------
NIFTY_TICKER = "^NSEI"
INTERVAL = "1h"
PERIOD = "60d"
LOOKAHEAD = 1
TEST_SIZE_RATIO = 0.2

NEWS_FEEDS = {
    "Economic Times Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol Markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "Business Standard Markets": "https://www.business-standard.com/rss/markets-106.rss",
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
}

NIFTY_50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "TATAMOTORS.NS", "MARUTI.NS", "TITAN.NS",
    "TATASTEEL.NS", "HINDALCO.NS", "GRASIM.NS", "BAJFINANCE.NS", "M&M.NS",
]

NEGATIVE_KEYWORDS = [
    "crude", "oil price", "inflation", "rate hike", "geopolitical",
    "tension", "war", "conflict", "sell-off", "selloff", "recession",
    "resign", "downgrade", "weak", "fall", "decline", "cut",
]
POSITIVE_KEYWORDS = [
    "rate cut", "record high", "rally", "surge", "upgrade", "strong",
    "growth", "beat estimates", "buyback", "fii inflow", "recovery",
]

FEATURE_COLS = [
    "SMA_9", "SMA_21", "EMA_9", "EMA_21", "RSI_14",
    "MACD", "MACD_signal", "MACD_hist",
    "BB_pct", "Return_1", "Return_3", "Volatility_10",
]


# ----------------------------------------------------------------
# TECHNICAL SIGNAL
# ----------------------------------------------------------------
def fetch_nifty_data():
    df = yf.download(NIFTY_TICKER, period=PERIOD, interval=INTERVAL, progress=False)
    if df.empty:
        raise RuntimeError("No Nifty data returned — check internet connection.")
    return df.dropna()


def add_indicators(df):
    close = df["Close"]
    df["SMA_9"] = close.rolling(9).mean()
    df["SMA_21"] = close.rolling(21).mean()
    df["EMA_9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA_21"] = close.ewm(span=21, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_upper"] = sma20 + 2 * std20
    df["BB_lower"] = sma20 - 2 * std20
    df["BB_pct"] = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-9)

    df["Return_1"] = close.pct_change(1)
    df["Return_3"] = close.pct_change(3)
    df["Volatility_10"] = df["Return_1"].rolling(10).std()
    return df


def add_target(df, lookahead=LOOKAHEAD):
    future_close = df["Close"].shift(-lookahead)
    df["Target"] = (future_close > df["Close"]).astype(int)
    return df


def get_technical_signal():
    """Returns a dict summarizing the latest technical signal + backtest accuracy."""
    df = fetch_nifty_data()
    df = add_indicators(df)
    df = add_target(df)

    data = df.dropna(subset=FEATURE_COLS + ["Target"]).copy()
    X, y = data[FEATURE_COLS], data["Target"]
    split_idx = int(len(data) * (1 - TEST_SIZE_RATIO))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    latest_row = data.iloc[[-1]][FEATURE_COLS]
    proba = model.predict_proba(latest_row)[0]
    pred = model.predict(latest_row)[0]

    return {
        "last_time": str(data.index[-1]),
        "last_close": round(float(data["Close"].iloc[-1]), 2),
        "backtest_accuracy": round(float(acc) * 100, 1),
        "direction": "UP" if pred == 1 else "DOWN",
        "confidence": round(float(max(proba)) * 100, 1),
    }


# ----------------------------------------------------------------
# STOCK MOVERS
# ----------------------------------------------------------------
def get_movers():
    """Returns a list of dicts: [{ticker, pct_change}, ...] sorted descending."""
    data = yf.download(NIFTY_50_TICKERS, period="2d", interval="1d",
                        progress=False, group_by="ticker")
    rows = []
    for ticker in NIFTY_50_TICKERS:
        try:
            close = data[ticker]["Close"].dropna()
            if len(close) >= 2:
                pct = (close.iloc[-1] / close.iloc[-2] - 1) * 100
                rows.append({
                    "ticker": ticker.replace(".NS", ""),
                    "pct_change": round(float(pct), 2),
                })
        except Exception:
            continue
    rows.sort(key=lambda r: r["pct_change"], reverse=True)
    return rows


# ----------------------------------------------------------------
# NEWS
# ----------------------------------------------------------------
def tag_headline(title):
    t = title.lower()
    if any(k in t for k in NEGATIVE_KEYWORDS):
        return "negative"
    if any(k in t for k in POSITIVE_KEYWORDS):
        return "positive"
    return "neutral"


def get_news():
    """Returns a list of dicts: [{source, title, published, tag}, ...]."""
    if feedparser is None:
        return []

    headlines = []
    for source, url in NEWS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:8]:
                title = entry.get("title", "")
                headlines.append({
                    "source": source,
                    "title": title,
                    "published": entry.get("published", ""),
                    "tag": tag_headline(title),
                })
        except Exception:
            continue
    return headlines


# ----------------------------------------------------------------
# FULL REPORT (used by Flask API)
# ----------------------------------------------------------------
def get_full_report():
    """Combines everything into one dict, safe against partial failures."""
    report = {"signal": None, "movers": [], "news": [], "errors": []}

    try:
        report["signal"] = get_technical_signal()
    except Exception as e:
        report["errors"].append(f"Technical signal failed: {e}")

    try:
        report["movers"] = get_movers()
    except Exception as e:
        report["errors"].append(f"Movers fetch failed: {e}")

    try:
        report["news"] = get_news()
    except Exception as e:
        report["errors"].append(f"News fetch failed: {e}")

    return report
