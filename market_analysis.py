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

    # Newer yfinance versions can return MultiIndex columns (ticker, field)
    # even for a single ticker. Flatten to plain column names like "Close".
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

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
    """Returns a list of dicts: [{source, title, published, tag}, ...],
    sorted newest-first and limited to recent items only (last 48 hours).
    Filters out stale/cached entries some RSS feeds occasionally serve.
    """
    if feedparser is None:
        return []

    import time as _time
    from datetime import datetime, timezone

    cutoff = _time.time() - (48 * 3600)  # only last 48 hours
    headlines = []

    for source, url in NEWS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:15]:
                title = entry.get("title", "")
                published_struct = entry.get("published_parsed")

                if published_struct:
                    published_ts = _time.mktime(published_struct)
                    if published_ts < cutoff:
                        continue  # skip stale entries
                else:
                    published_ts = 0  # unknown date -> sort to the bottom

                headlines.append({
                    "source": source,
                    "title": title,
                    "published": entry.get("published", ""),
                    "tag": tag_headline(title),
                    "_ts": published_ts,
                })
        except Exception:
            continue

    headlines.sort(key=lambda h: h["_ts"], reverse=True)
    for h in headlines:
        del h["_ts"]

    return headlines


# ----------------------------------------------------------------
# BOTTOM-REVERSAL / BREAKDOWN SCANNER
# ----------------------------------------------------------------
# Scans a universe of liquid NSE stocks (loaded from universe.csv, built
# from an NSE Bhavcopy) and flags:
#   - Stocks trading near their 52-week LOW but showing recent upward
#     momentum (a possible "bottoming out" pattern)
#   - Stocks trading near their 52-week HIGH but showing recent downward
#     momentum (a possible "topping out" pattern)
#
# ⚠️ IMPORTANT: The "target" shown is the stock's own recent swing
# high/low — a real historical price level, NOT a prediction of where
# the stock WILL go or WHEN. There is no reliable way to predict an
# exact future price or time window. Use this as context, not a promise.
# ----------------------------------------------------------------
UNIVERSE_FILE = "universe.csv"
SCAN_HISTORY_PERIOD = "1y"
NEAR_EXTREME_PCT = 15.0   # "near" the 52w low/high = within this % band
RECENT_MOMENTUM_DAYS = 5  # how many days of recent price change to check


def load_universe():
    try:
        df = pd.read_csv(UNIVERSE_FILE)
        return df["Symbol"].dropna().tolist()
    except Exception:
        return NIFTY_50_TICKERS  # fallback if universe.csv missing


def _compute_stock_metrics(symbol, hist):
    """Given 1y of daily history for one stock, compute scanner metrics."""
    close = hist["Close"].dropna()
    if len(close) < 60:
        return None

    current = float(close.iloc[-1])
    low_52w = float(close.min())
    high_52w = float(close.max())
    low_date = close.idxmin()
    high_date = close.idxmax()

    # Skip near-flat instruments (liquid/cash funds, illiquid stocks with
    # barely any price movement) — they're not real reversal candidates.
    volatility_range_pct = (high_52w / low_52w - 1) * 100
    if volatility_range_pct < 8:
        return None

    pct_above_low = (current / low_52w - 1) * 100
    pct_below_high = (1 - current / high_52w) * 100

    recent_change = (
        (current / float(close.iloc[-RECENT_MOMENTUM_DAYS - 1]) - 1) * 100
        if len(close) > RECENT_MOMENTUM_DAYS else 0.0
    )

    # RSI-14 for extra context
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = float((100 - (100 / (1 + rs))).iloc[-1])

    return {
        "symbol": symbol.replace(".NS", ""),
        "current_price": round(current, 2),
        "pct_above_52w_low": round(pct_above_low, 2),
        "pct_below_52w_high": round(pct_below_high, 2),
        "recent_5d_change": round(recent_change, 2),
        "rsi_14": round(rsi, 1),
        "target_recent_high": round(high_52w, 2),
        "target_recent_low": round(low_52w, 2),
        "low_date": str(low_date.date()) if hasattr(low_date, "date") else str(low_date),
        "high_date": str(high_date.date()) if hasattr(high_date, "date") else str(high_date),
    }


def scan_bottom_and_breakdown(top_n=10):
    """
    Returns a dict:
      {
        "bottoming": [top_n stocks near 52w low, now showing upward momentum],
        "topping":   [top_n stocks near 52w high, now showing downward momentum],
      }
    """
    universe = load_universe()

    data = yf.download(
        universe, period=SCAN_HISTORY_PERIOD, interval="1d",
        progress=False, group_by="ticker",
    )

    bottoming, topping = [], []

    for symbol in universe:
        try:
            hist = data[symbol] if symbol in data.columns.get_level_values(0) else None
            if hist is None or hist.empty:
                continue
            metrics = _compute_stock_metrics(symbol, hist.dropna())
            if metrics is None:
                continue

            # Near 52-week LOW + recent upward momentum -> "bottoming out"
            if (metrics["pct_above_52w_low"] <= NEAR_EXTREME_PCT
                    and metrics["recent_5d_change"] > 0):
                bottoming.append(metrics)

            # Near 52-week HIGH + recent downward momentum -> "topping out"
            if (metrics["pct_below_52w_high"] <= NEAR_EXTREME_PCT
                    and metrics["recent_5d_change"] < 0):
                topping.append(metrics)

        except Exception:
            continue

    # Rank: strongest recent bounce off the low first
    bottoming.sort(key=lambda m: m["recent_5d_change"], reverse=True)
    # Rank: sharpest recent drop from the high first
    topping.sort(key=lambda m: m["recent_5d_change"])

    return {
        "bottoming": bottoming[:top_n],
        "topping": topping[:top_n],
    }


# ----------------------------------------------------------------
# MULTI-HORIZON PROJECTION (1h / 2h / 3h) — EXPERIMENTAL
# ----------------------------------------------------------------
# For each stock in the current bottoming/topping lists, this trains a
# small model per horizon and reports:
#   - direction (UP/DOWN) the model predicts for that horizon
#   - backtest accuracy for that SAME horizon on that SAME stock's
#     recent history (so you can see for yourself how reliable — or
#     unreliable — it actually is, per stock, per horizon)
#   - an expected % move RANGE based on the stock's own historical
#     volatility (a statistical range, not a promised destination)
#
# ⚠️ This is intentionally labeled EXPERIMENTAL. Backtest accuracy
# hovering near 50% (a coin flip) is the expected, normal outcome —
# that is the honest result, not a bug. Judge usefulness using the
# accuracy numbers shown, not by assuming it works.
# ----------------------------------------------------------------
HORIZONS = [1, 2, 3]  # hours ahead
PROJECTION_PERIOD = "60d"
PROJECTION_INTERVAL = "1h"


def _stock_features(close):
    df = pd.DataFrame({"Close": close})
    df["SMA_9"] = close.rolling(9).mean()
    df["EMA_9"] = close.ewm(span=9, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD_hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

    df["Return_1"] = close.pct_change(1)
    df["Return_3"] = close.pct_change(3)
    return df


PROJ_FEATURES = ["SMA_9", "EMA_9", "RSI_14", "MACD_hist", "Return_1", "Return_3"]


def get_hourly_projection(symbol):
    """Trains a quick per-horizon model for one stock. Returns dict or None."""
    try:
        raw = yf.download(symbol, period=PROJECTION_PERIOD,
                           interval=PROJECTION_INTERVAL, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.dropna()
        if len(raw) < 120:
            return None

        close = raw["Close"]
        feats = _stock_features(close)
        hourly_returns = close.pct_change().dropna()
        std_1h = float(hourly_returns.std())

        result = {}
        for h in HORIZONS:
            data = feats.copy()
            data["Target"] = (close.shift(-h) > close).astype(int)
            data = data.dropna(subset=PROJ_FEATURES + ["Target"])
            if len(data) < 60:
                continue

            X, y = data[PROJ_FEATURES], data["Target"]
            split = int(len(data) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            model = RandomForestClassifier(
                n_estimators=150, max_depth=5, min_samples_leaf=8, random_state=42
            )
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test)) if len(X_test) else float("nan")

            latest = data.iloc[[-1]][PROJ_FEATURES]
            proba = model.predict_proba(latest)[0]
            pred = model.predict(latest)[0]

            expected_move_pct = std_1h * (h ** 0.5) * 100  # volatility scaling

            result[f"{h}h"] = {
                "direction": "UP" if pred == 1 else "DOWN",
                "confidence": round(float(max(proba)) * 100, 1),
                "backtest_accuracy": round(float(acc) * 100, 1) if acc == acc else None,
                "expected_move_range_pct": round(expected_move_pct, 2),
            }

        return {
            "symbol": symbol.replace(".NS", ""),
            "current_price": round(float(close.iloc[-1]), 2),
            "horizons": result,
        }
    except Exception:
        return None


def get_projection_report(top_n=10):
    """
    Runs the bottoming/topping scan, then adds 1h/2h/3h projections for
    each of those stocks. Returns:
      { "bottoming": [...], "topping": [...] }
    each item = output of get_hourly_projection(), or skipped if unavailable.
    """
    scan = scan_bottom_and_breakdown(top_n=top_n)

    bottoming_proj = []
    for stock in scan["bottoming"]:
        proj = get_hourly_projection(stock["symbol"] + ".NS")
        if proj:
            proj["reference_target"] = stock["target_recent_high"]
            bottoming_proj.append(proj)

    topping_proj = []
    for stock in scan["topping"]:
        proj = get_hourly_projection(stock["symbol"] + ".NS")
        if proj:
            proj["reference_target"] = stock["target_recent_low"]
            topping_proj.append(proj)

    return {"bottoming": bottoming_proj, "topping": topping_proj}


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
