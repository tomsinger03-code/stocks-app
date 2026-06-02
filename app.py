from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import requests
from dotenv import load_dotenv
import os
from datetime import datetime
import time
from database import init_db, save_stock

load_dotenv()

init_db()

app = Flask(__name__)
CORS(app)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "demo")
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# Simple in-memory cache to protect free API tier (60 req/min limit)
_cache = {}
CACHE_TTL = 60  # seconds

watchlist = {}


def cached_get(url, params, ttl=CACHE_TTL):
    """Cached HTTP GET to avoid hammering Finnhub free tier."""
    key = url + str(sorted(params.items()))
    now = time.time()
    if key in _cache and now - _cache[key]["ts"] < ttl:
        return _cache[key]["data"]
    resp = requests.get(url, params=params, timeout=8)
    data = resp.json()
    _cache[key] = {"data": data, "ts": now}
    return data


# ──────────────────────────────────────────────
# Signal Calculators
# ──────────────────────────────────────────────

def calc_rsi(closes, period=14):
    """Calculate RSI from a list of closing prices."""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def calc_macd_signal(closes):
    """
    Returns a simple MACD signal: 1 = bullish crossover zone, -1 = bearish, 0 = neutral.
    Uses EMA-12 vs EMA-26.
    """
    def ema(prices, period):
        k = 2 / (period + 1)
        e = prices[0]
        for p in prices[1:]:
            e = p * k + e * (1 - k)
        return e

    if len(closes) < 30:
        return 0
    ema12 = ema(closes[-26:], 12)
    ema26 = ema(closes[-26:], 26)
    macd = ema12 - ema26
    # Previous bar
    ema12_prev = ema(closes[-27:-1], 12)
    ema26_prev = ema(closes[-27:-1], 26)
    macd_prev = ema12_prev - ema26_prev

    if macd > 0 and macd_prev <= 0:
        return 1   # bullish crossover
    elif macd < 0 and macd_prev >= 0:
        return -1  # bearish crossover
    elif macd > 0:
        return 0.5
    else:
        return -0.5


def calc_bollinger_position(closes, period=20):
    """
    Where is price relative to its Bollinger Bands?
    Returns 0-1: 0 = at lower band (oversold), 1 = at upper band (overbought).
    """
    if len(closes) < period:
        return None
    window = closes[-period:]
    mean = sum(window) / period
    std = (sum((x - mean) ** 2 for x in window) / period) ** 0.5
    if std == 0:
        return 0.5
    upper = mean + 2 * std
    lower = mean - 2 * std
    current = closes[-1]
    pos = (current - lower) / (upper - lower)
    return round(max(0, min(1, pos)), 3)


def calc_volume_surge(volumes):
    """
    Is current volume significantly above average?
    Returns a ratio: >1.5 = notable surge.
    """
    if len(volumes) < 10:
        return None
    avg = sum(volumes[-20:-1]) / len(volumes[-20:-1])
    if avg == 0:
        return None
    return round(volumes[-1] / avg, 2)


def get_candles(ticker, resolution="D", count=60):
    """Fetch OHLCV candles via yfinance (free, no API key needed)."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="3mo", interval="1d", auto_adjust=True)
        if hist.empty:
            return None
        closes  = [float(x) for x in hist["Close"].dropna().tolist()]
        volumes = [float(x) for x in hist["Volume"].dropna().tolist()]
        if not closes:
            return None
        return {"c": closes, "v": volumes}
    except Exception as e:
        print(f"  yfinance error for {ticker}: {e}")
        return None


# ──────────────────────────────────────────────
# ML-style Momentum Scorer
# ──────────────────────────────────────────────

class MomentumScorer:
    """
    Score a stock 0-100 for short-term momentum / quick-win potential.

    Signals used (all transparent):
      1. RSI position         – sweet spot 40-65 (trending up, not overbought)
      2. MACD crossover       – recent bullish crossover = strong signal
      3. Bollinger position   – price emerging from lower band
      4. Volume surge         – above-average volume confirms moves
      5. 5-day price momentum – short-term direction
      6. Fundamental quality  – PE + dividend safety net
    """

    @staticmethod
    def score(ticker, candles, fundamentals):
        signals = {}
        score = 50

        closes = candles.get("c", [])
        volumes = candles.get("v", [])

        # ── Signal 1: RSI ──────────────────────────
        rsi = calc_rsi(closes)
        signals["rsi"] = rsi
        if rsi is not None:
            if 40 <= rsi <= 60:
                score += 12   # trending, not extreme
            elif 60 < rsi <= 70:
                score += 6    # strong momentum
            elif rsi > 75:
                score -= 10   # overbought
            elif rsi < 30:
                score += 8    # oversold bounce candidate

        # ── Signal 2: MACD ────────────────────────
        macd_sig = calc_macd_signal(closes)
        signals["macd"] = macd_sig
        score += macd_sig * 10

        # ── Signal 3: Bollinger Position ──────────
        boll = calc_bollinger_position(closes)
        signals["bollinger"] = boll
        if boll is not None:
            if 0.2 <= boll <= 0.5:
                score += 10   # emerging from lower band
            elif boll > 0.85:
                score -= 8    # near upper band — stretched

        # ── Signal 4: Volume Surge ────────────────
        vol_surge = calc_volume_surge(volumes)
        signals["volumeSurge"] = vol_surge
        if vol_surge is not None:
            if vol_surge >= 2.0:
                score += 12
            elif vol_surge >= 1.5:
                score += 7
            elif vol_surge < 0.7:
                score -= 5    # drying volume = weak conviction

        # ── Signal 5: 5-day momentum ─────────────
        if len(closes) >= 6:
            mom5 = (closes[-1] - closes[-6]) / closes[-6] * 100
            signals["momentum5d"] = round(mom5, 2)
            if 1 <= mom5 <= 8:
                score += 8    # healthy short-term move
            elif mom5 > 8:
                score -= 5    # might be overextended
            elif mom5 < -5:
                score -= 8
        else:
            signals["momentum5d"] = None

        # ── Signal 6: Fundamental quality ─────────
        pe = fundamentals.get("pe")
        if pe and 10 <= pe <= 30:
            score += 5
        elif pe and pe > 60:
            score -= 5

        div = fundamentals.get("dividendYield", 0) or 0
        if div > 0.02:
            score += 3  # dividend support

        signals["score"] = round(max(0, min(100, score)))
        signals["label"] = MomentumScorer._label(signals["score"])
        return signals

    @staticmethod
    def _label(score):
        if score >= 72:
            return "STRONG SETUP"
        elif score >= 60:
            return "GOOD SETUP"
        elif score >= 45:
            return "NEUTRAL"
        else:
            return "WEAK / AVOID"


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route('/api/stock/<ticker>', methods=['GET'])
def get_stock(ticker):
    """Fetch stock data + momentum ML score."""
    try:
        ticker = ticker.upper()

        quote_data = cached_get(f"{FINNHUB_BASE_URL}/quote", {"symbol": ticker, "token": FINNHUB_API_KEY})
        if not quote_data.get('c'):
            return jsonify({"error": "Stock not found"}), 404

        company_data = cached_get(f"{FINNHUB_BASE_URL}/stock/profile2", {"symbol": ticker, "token": FINNHUB_API_KEY})
        financials_data = cached_get(f"{FINNHUB_BASE_URL}/stock/metric", {"symbol": ticker, "metric": "all", "token": FINNHUB_API_KEY})

        metrics = financials_data.get('metric', {})
        fundamentals = {
            "pe": metrics.get('peBasic'),
            "marketCap": metrics.get('marketCapBasic'),
            "dividendYield": metrics.get('dividendYield'),
            "week52High": metrics.get('52WeekHigh'),
            "week52Low": metrics.get('52WeekLow'),
        }

        stock_data = {
            "ticker": ticker,
            "price": quote_data.get('c'),
            "change": quote_data.get('d'),
            "changePercent": quote_data.get('dp'),
            "high": quote_data.get('h'),
            "low": quote_data.get('l'),
            "open": quote_data.get('o'),
            "volume": quote_data.get('v'),
            "company": {
                "name": company_data.get('name'),
                "industry": company_data.get('finnhubIndustry'),
                "logo": company_data.get('logo'),
            },
            "financials": fundamentals,
        }

        # Momentum ML score
        candles = get_candles(ticker)
        if candles:
            momentum = MomentumScorer.score(ticker, candles, fundamentals)
        else:
            momentum = {"score": 50, "label": "NEUTRAL", "error": "Candle data unavailable"}

        stock_data["momentum"] = momentum

        # Legacy recommendation for backwards compat
        s = momentum["score"]
        if s >= 72:
            action = "STRONG BUY"
        elif s >= 60:
            action = "BUY"
        elif s >= 45:
            action = "HOLD"
        else:
            action = "SELL"

        stock_data["recommendation"] = {
            "action": action,
            "score": s
        }

        save_stock(
            ticker=ticker,
            price=stock_data["price"],
            score=s,
            rsi=momentum.get("rsi"),
            macd=momentum.get("macd"),
            bollinger=momentum.get("bollinger"),
            volume_surge=momentum.get("volumeSurge"),
            momentum5d=momentum.get("momentum5d")
        )

        return jsonify(stock_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/movers', methods=['GET'])
def get_movers():
    """
    Scan a curated list of liquid stocks and return ranked by momentum score.
    This is the "quick win" scanner — short-term setup quality.
    """
    scan_list = [
        # Large-cap tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        # Finance
        "JPM", "GS", "BAC",
        # Consumer / healthcare
        "V", "JNJ", "UNH", "PG",
        # Energy / commodities
        "XOM", "CVX",
        # Growth
        "SHOP", "CRWD", "PLTR", "SNOW",
    ]

    results = []
    for ticker in scan_list:
        try:
            print(f"  Scanning {ticker}...")
            quote = cached_get(f"{FINNHUB_BASE_URL}/quote", {"symbol": ticker, "token": FINNHUB_API_KEY})
            if not quote.get('c'):
                print(f"  {ticker}: no quote data")
                continue
            time.sleep(1)
            metrics = cached_get(f"{FINNHUB_BASE_URL}/stock/metric", {"symbol": ticker, "metric": "all", "token": FINNHUB_API_KEY}).get('metric', {})
            time.sleep(1)
            fundamentals = {
                "pe": metrics.get('peBasic'),
                "dividendYield": metrics.get('dividendYield'),
            }
            candles = get_candles(ticker)
            if not candles:
                print(f"  {ticker}: no candle data")
                continue
            time.sleep(1)
            momentum = MomentumScorer.score(ticker, candles, fundamentals)
            print(f"  {ticker}: score={momentum['score']} label={momentum['label']}")
            results.append({
                "ticker": ticker,
                "price": quote.get('c'),
                "changePercent": quote.get('dp'),
                "score": momentum["score"],
                "label": momentum["label"],
                "signals": {
                    "rsi": momentum.get("rsi"),
                    "momentum5d": momentum.get("momentum5d"),
                    "volumeSurge": momentum.get("volumeSurge"),
                    "bollinger": momentum.get("bollinger"),
                }
            })
        except Exception as e:
            print(f"  {ticker}: ERROR - {e}")
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return jsonify({"results": results, "scanned": len(results), "timestamp": datetime.now().isoformat()})


@app.route('/api/screener', methods=['POST'])
def screener():
    """Screen stocks based on criteria (original feature, kept intact)."""
    try:
        criteria = request.json
        popular_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ"]
        results = []

        for ticker in popular_stocks:
            quote = cached_get(f"{FINNHUB_BASE_URL}/quote", {"symbol": ticker, "token": FINNHUB_API_KEY})
            metrics = cached_get(f"{FINNHUB_BASE_URL}/stock/metric", {"symbol": ticker, "metric": "all", "token": FINNHUB_API_KEY}).get('metric', {})
            data = {
                "price": quote.get('c'),
                "pe": metrics.get('peBasic'),
                "dividendYield": metrics.get('dividendYield'),
                "week52High": metrics.get('52WeekHigh'),
                "week52Low": metrics.get('52WeekLow'),
                "marketCap": metrics.get('marketCapBasic'),
            }

            passes = True
            if criteria.get('minPE') and data.get('pe') and data['pe'] < criteria['minPE']:
                passes = False
            if criteria.get('maxPE') and data.get('pe') and data['pe'] > criteria['maxPE']:
                passes = False
            if criteria.get('minDividend') and data.get('dividendYield') and data['dividendYield'] < criteria['minDividend']:
                passes = False

            if passes and data.get('price'):
                candles = get_candles(ticker)
                score = MomentumScorer.score(ticker, candles, data)["score"] if candles else 50
                results.append({"ticker": ticker, "price": data['price'], "pe": data.get('pe'),
                                 "dividendYield": data.get('dividendYield'), "score": score})

        results.sort(key=lambda x: x['score'], reverse=True)
        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/watchlist', methods=['GET', 'POST', 'DELETE'])
def handle_watchlist():
    if request.method == 'GET':
        return jsonify({"watchlist": watchlist})
    elif request.method == 'POST':
        ticker = request.json.get('ticker', '').upper()
        if ticker:
            watchlist[ticker] = datetime.now().isoformat()
            return jsonify({"message": f"{ticker} added", "watchlist": watchlist})
        return jsonify({"error": "Ticker required"}), 400
    elif request.method == 'DELETE':
        ticker = request.json.get('ticker', '').upper()
        if ticker in watchlist:
            del watchlist[ticker]
            return jsonify({"message": f"{ticker} removed", "watchlist": watchlist})
        return jsonify({"error": "Not found"}), 404


@app.route('/api/watchlist/<ticker>', methods=['DELETE'])
def remove_from_watchlist(ticker):
    ticker = ticker.upper()
    if ticker in watchlist:
        del watchlist[ticker]
        return jsonify({"message": f"{ticker} removed"})
    return jsonify({"error": "Not found"}), 404


@app.route('/')
def index():
    return send_file('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(__import__('os').environ.get('PORT', 5000)))
