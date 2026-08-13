"""
app.py
========
Flask web server for the Nifty Market Intelligence dashboard.

Run with:
    python app.py

Then open in browser:
    http://127.0.0.1:5000

⚠️ Educational tool. Not financial advice. No prediction is certain.
"""

import time
from flask import Flask, jsonify, render_template

import market_analysis as ma

app = Flask(__name__)

# Simple in-memory cache so we don't re-download data on every browser
# refresh (yfinance + news fetch takes a few seconds).
_cache = {"data": None, "timestamp": 0}
CACHE_SECONDS = 300  # refresh underlying data at most once every 5 minutes


def get_cached_report():
    now = time.time()
    if _cache["data"] is None or (now - _cache["timestamp"]) > CACHE_SECONDS:
        _cache["data"] = ma.get_full_report()
        _cache["timestamp"] = now
    return _cache["data"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    report = get_cached_report()
    return jsonify(report)


@app.route("/api/refresh")
def api_refresh():
    """Force a fresh data pull, bypassing the cache."""
    _cache["data"] = ma.get_full_report()
    _cache["timestamp"] = time.time()
    return jsonify(_cache["data"])


if __name__ == "__main__":
    app.run(debug=True, port=5000)
