from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import requests
from dotenv import load_dotenv
import os
from datetime import datetime
import time
from database import init_db, save_stock, init_picks_table, save_pick, update_pick_price, update_pick_outcome, get_all_picks, get_open_picks
from signals import get_all_signals, get_float_data
from sheets_sync import add_pick_to_sheet, update_pick_in_sheet, sync_all_picks_to_sheet, is_connected as sheets_connected
from ml_model import (
    train_model_background, predict, get_status,
    SCAN_UNIVERSE, extract_features, FEATURE_COLS
)

load_dotenv()

init_db()
init_picks_table()
train_model_background()  # start ML training in background

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



def get_news_sentiment(ticker):
    """
    Fetch recent news sentiment from Finnhub.
    Returns: { score: -1 to 1, count: int, headlines: [...] }
    """
    try:
        from datetime import timedelta
        import datetime as dt
        to_date   = dt.date.today().isoformat()
        from_date = (dt.date.today() - timedelta(days=7)).isoformat()
        data = cached_get(
            f"{FINNHUB_BASE_URL}/company-news",
            {"symbol": ticker, "from": from_date, "to": to_date, "token": FINNHUB_API_KEY},
            ttl=1800
        )
        if not data or not isinstance(data, list):
            return {"score": 0, "count": 0, "headlines": []}

        # Finnhub doesn't give sentiment scores on free tier news
        # but we can do keyword scoring on headlines
        positive_words = ["surge","soar","jump","gain","beat","record","launch",
                          "win","award","contract","approve","approved","fda",
                          "partnership","buy","upgrade","growth","profit","revenue"]
        negative_words = ["fall","drop","loss","miss","cut","downgrade","lawsuit",
                          "fail","decline","warning","risk","probe","fraud","delisted"]

        score_sum = 0
        headlines = []
        for article in data[:10]:
            headline = (article.get("headline") or "").lower()
            s = sum(1 for w in positive_words if w in headline)
            s -= sum(1 for w in negative_words if w in headline)
            score_sum += s
            headlines.append({
                "title": article.get("headline", ""),
                "source": article.get("source", ""),
                "datetime": article.get("datetime", 0),
                "sentiment": "positive" if s > 0 else "negative" if s < 0 else "neutral"
            })

        count = len(data)
        norm_score = max(-1, min(1, score_sum / max(count, 1)))
        return {"score": round(norm_score, 3), "count": count, "headlines": headlines[:5]}

    except Exception as e:
        return {"score": 0, "count": 0, "headlines": [], "error": str(e)}


def check_sec_filings(ticker):
    """
    Check for recent SEC 8-K filings (material events = potential catalysts).
    Uses SEC EDGAR free API.
    """
    try:
        url = f"https://data.sec.gov/submissions/CIK{ticker}.json"
        # Use ticker->CIK lookup
        lookup = cached_get(
            "https://efts.sec.gov/LATEST/search-index?q=%22" + ticker + "%22&dateRange=custom&startdt=" +
            (__import__('datetime').date.today() - __import__('datetime').timedelta(days=14)).isoformat() +
            "&enddt=" + __import__('datetime').date.today().isoformat() + "&forms=8-K",
            {},
            ttl=3600
        )
        # Simplified: just check EDGAR full-text search
        hits = lookup.get("hits", {}).get("hits", []) if isinstance(lookup, dict) else []
        recent_8k = len(hits) > 0
        return {"recent_8k": recent_8k, "count": len(hits)}
    except:
        return {"recent_8k": False, "count": 0}



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
    """
    Smart screener — finds stocks ABOUT to spike.
    Uses threading for speed. Excludes leveraged ETFs.
    """
    try:
        import yfinance as yf
        from ml_model import SCAN_UNIVERSE, extract_features
        from concurrent.futures import ThreadPoolExecutor, as_completed

        criteria = request.json or {}
        min_price       = float(criteria.get('minPrice', 1))
        max_price       = float(criteria.get('maxPrice', 15))
        min_vol_surge   = float(criteria.get('minVolSurge', 3.0))
        max_rsi         = float(criteria.get('maxRsi', 65))
        min_consec_down = int(criteria.get('minConsecDown', 2))
        near_low_pct    = float(criteria.get('nearLowPct', 150))

        # Leveraged ETF keywords to exclude
        EXCLUDE_KEYWORDS = ['2x','3x','-2x','-3x','ultra','leverage','leveraged',
                            'proshares','direxion','2xl','3xl']

        results = []
        skipped = []

        def check_ticker(ticker):
            # Skip leveraged ETFs
            tl = ticker.lower()
            if any(k in tl for k in EXCLUDE_KEYWORDS):
                return None
            try:
                # Quick float/reverse split check first
                float_info = get_float_data(ticker)
                if float_info.get("hasReverseSplit"):
                    return None  # Skip reverse split stocks
                
                hist = yf.Ticker(ticker).history(period="3mo", interval="1d", auto_adjust=True)
                if hist is None or len(hist) < 25:
                    return None
                closes  = hist["Close"].tolist()
                volumes = hist["Volume"].tolist()
                opens   = hist["Open"].tolist()
                price   = closes[-1]
                if not (min_price <= price <= max_price):
                    return None
                feats = extract_features(closes, volumes, opens)
                if not feats:
                    return None
                vol_surge    = feats.get("vol_surge_20d", 1)
                rsi          = feats.get("rsi", 50)
                mom3         = feats.get("mom3d", 0) or 0
                pct_above_low = feats.get("price_vs_52low", 999)
                consec       = feats.get("consec_down", 0)
                if vol_surge < min_vol_surge: return None
                if rsi > max_rsi: return None
                if mom3 >= 15: return None
                if pct_above_low > near_low_pct: return None
                if consec < min_consec_down: return None
                change_pct = (closes[-1]-closes[-2])/closes[-2]*100 if len(closes)>=2 else 0
                # Get extended signals
                ext = get_all_signals(ticker)
                
                # Add float size to signals
                float_m = ext["float"].get("floatM")
                
                return {
                    "ticker": ticker,
                    "price": round(price, 2),
                    "changePercent": round(change_pct, 2),
                    "catalystScore": ext["catalystScore"],
                    "signals": {
                        "rsi": round(rsi, 1),
                        "volSurge20": round(vol_surge, 1),
                        "volSurge5": round(feats.get("vol_surge_5d", 1), 1),
                        "mom3d": round(mom3, 2),
                        "mom5d": round(feats.get("mom5d", 0), 2),
                        "consecDown": consec,
                        "pctAbove52wLow": round(pct_above_low, 1),
                        "bollSqueeze": round(feats.get("boll_squeeze", 0), 4),
                        "atr": round(feats.get("atr_pct", 0), 2),
                        "floatM": float_m,
                        "smallFloat": ext["float"].get("smallFloat"),
                        "recent8K": ext["sec"].get("recent8K"),
                        "insiderBuying": ext["insider"].get("insiderBuying"),
                        "shortPct": ext["short"].get("shortPct"),
                        "squeezeCandidate": ext["short"].get("squeezeCandidate"),
                        "redditMentions": ext["reddit"].get("mentions24h"),
                        "redditTrending": ext["reddit"].get("trending"),
                        "earningsBeat": ext["earnings"].get("recentBeat"),
                    }
                }
            except:
                return None

        # Run in parallel — much faster than sequential
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(check_ticker, t): t for t in SCAN_UNIVERSE}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        # Sort by volume surge
        results.sort(key=lambda x: x["signals"]["volSurge20"], reverse=True)

        return jsonify({
            "results": results,
            "passed": len(results),
            "scanned": len(SCAN_UNIVERSE),
            "filters_applied": criteria,
        })

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



@app.route('/api/ml/status', methods=['GET'])
def ml_status():
    """Check if the ML model is trained and ready."""
    return jsonify(get_status())


@app.route('/api/ml/scan', methods=['GET'])
def ml_scan():
    """
    Score all stocks in the universe by ML probability of 15%+ gain in 3 days.
    Returns sorted list. Model must be trained first (/api/ml/status to check).
    """
    status = get_status()
    if status["status"] != "ready":
        return jsonify({
            "error": "Model not ready yet",
            "status": status["status"],
            "log": status["log"]
        }), 202

    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

    # Use live discovery if available, fall back to SCAN_UNIVERSE
    try:
        import urllib.request, json as _json
        from datetime import date
        scan_list = []

        # Yahoo most active
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives&count=50&formatted=false"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as r:
            data = _json.loads(r.read())
        quotes = data.get("finance",{}).get("result",[{}])[0].get("quotes",[])
        for q in quotes:
            t = q.get("symbol","")
            if t and len(t) <= 5 and "." not in t:
                scan_list.append(t)

        # Add known volatile stocks
        scan_list = list(dict.fromkeys(scan_list + SCAN_UNIVERSE))[:75]
        print(f"ML scan: {len(scan_list)} candidates (live + known)")
    except Exception as e:
        print(f"Live discovery failed, using fixed list: {e}")
        scan_list = SCAN_UNIVERSE[:50]
    
    results = []
    errors = []

    def score_ticker(ticker):
        try:
            # Skip leveraged ETFs
            tl = ticker.lower()
            if any(k in tl for k in ['2x','3x','-2x','-3x','ultra','sqqq','tqqq','spxu','uvxy']):
                return None

            # Check for reverse split
            tk = yf.Ticker(ticker)
            try:
                actions = tk.actions
                if actions is not None and not actions.empty and "Stock Splits" in actions.columns:
                    splits = actions["Stock Splits"].tail(10)
                    if any(0 < v < 1 for v in splits if v != 0):
                        return None  # Skip reverse split stocks
            except:
                pass

            hist = tk.history(period="3mo", interval="1d", auto_adjust=True)
            if hist is None or len(hist) < 25:
                return None
            closes  = hist["Close"].tolist()
            volumes = hist["Volume"].tolist()
            opens   = hist["Open"].tolist()
            prob = predict(closes, volumes, opens)
            if prob is None:
                return None
            return (ticker, closes, volumes, opens, prob)
        except:
            return None

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(score_ticker, t): t for t in scan_list}
        for future in as_completed(futures, timeout=90):
            res = future.result()
            if not res:
                continue
            ticker, closes, volumes, opens, prob = res
            # Current price and recent change
            price = closes[-1]
            change_pct = (closes[-1] - closes[-2]) / closes[-2] * 100 if len(closes) >= 2 else 0

            # ── FILTER OUT STOCKS THAT ALREADY SPIKED ──
            # Don't suggest stocks where the move already happened
            from ml_model import extract_features, FEATURE_COLS
            feats = extract_features(closes, volumes, opens)

            if feats:
                # Already ran: up 15%+ in last 3 days
                mom3 = feats.get("mom3d", 0) or 0
                if mom3 >= 15:
                    errors.append(f"{ticker}: already spiked +{mom3:.1f}% (skipped)")
                    continue

                # Already overbought
                rsi = feats.get("rsi", 50) or 50
                if rsi > 78:
                    errors.append(f"{ticker}: overbought RSI {rsi:.0f} (skipped)")
                    continue

                # Volume surge already peaked (yesterday was bigger than today)
                vols = volumes[-3:] if len(volumes) >= 3 else volumes
                if len(vols) >= 2 and vols[-2] > vols[-1] * 2:
                    # Volume was 2x bigger yesterday = surge already passed
                    errors.append(f"{ticker}: volume surge already peaked (skipped)")
                    continue

                # Price already at or near 52w high (no room to run)
                pct_from_high = feats.get("price_vs_52high", 100) or 100
                if pct_from_high < 3:
                    errors.append(f"{ticker}: near 52w high (skipped)")
                    continue

            # News sentiment
            news = get_news_sentiment(ticker)

            # Extended catalyst signals
            ext = get_all_signals(ticker)
            
            # Skip reverse splits
            if ext.get("reverseSplit"):
                continue

            results.append({
                "ticker": ticker,
                "price": round(price, 2),
                "changePercent": round(change_pct, 2),
                "mlProb": prob,
                "mlPct": round(prob * 100, 1),
                "day1": prediction.get("day1", {}),
                "day2": prediction.get("day2", {}),
                "day3": prediction.get("day3", {}),
                "bestDay": prediction.get("bestDay", 3),
                "bestGain": prediction.get("bestGain", 0),
                "catalystScore": ext["catalystScore"],
                "news": {
                    "score": news["score"],
                    "count": news["count"],
                    "headlines": news["headlines"][:2],
                },
                "signals": {
                    "rsi": round(feats["rsi"], 1) if feats else None,
                    "volSurge20": round(feats["vol_surge_20d"], 1) if feats else None,
                    "volSurge5": round(feats["vol_surge_5d"], 1) if feats else None,
                    "mom5": round(feats["mom5d"], 2) if feats else None,
                    "atr": round(feats["atr_pct"], 2) if feats else None,
                    "consecDown": feats["consec_down"] if feats else None,
                    "bollSqueeze": round(feats["boll_squeeze"], 4) if feats else None,
                    "floatM": ext["float"].get("floatM"),
                    "recent8K": ext["sec"].get("recent8K"),
                    "insiderBuying": ext["insider"].get("insiderBuying"),
                    "squeezeCandidate": ext["short"].get("squeezeCandidate"),
                    "redditTrending": ext["reddit"].get("trending"),
                    "shortPct": ext["short"].get("shortPct"),
                } if feats else {}
            })

    results.sort(key=lambda x: x["mlProb"], reverse=True)

    return jsonify({
        "results": results,
        "scanned": len(results),
        "errors": len(errors),
        "error_details": errors[:5],
        "timestamp": datetime.now().isoformat(),
        "model_trained_at": status.get("trained_at"),
        "note": "mlProb = model probability of 15%+ gain within 3 trading days. NOT financial advice."
    })


@app.route('/api/ml/retrain', methods=['POST'])
def ml_retrain():
    """Force retrain the model (deletes cached model file)."""
    import os
    if os.path.exists("momentum_model.pkl"):
        os.remove("momentum_model.pkl")
    train_model_background()
    return jsonify({"message": "Retraining started", "status": "training"})



@app.route('/api/picks', methods=['GET'])
def get_picks():
    """Get all tracked picks."""
    picks = get_all_picks()
    total = len(picks)
    wins = sum(1 for p in picks if p.get('outcome') == 'WIN')
    slow_wins = sum(1 for p in picks if p.get('outcome') == 'SLOW WIN')
    losses = sum(1 for p in picks if p.get('outcome') == 'LOSS')
    return jsonify({
        "picks": picks,
        "sheetsConnected": sheets_connected(),
        "sheetId": "1f4FtUqsXuVlyptRxbsSSiouoVa7IEDM2Y3SNWSYc7ho",
        "stats": {
            "total": total,
            "wins": wins,
            "slowWins": slow_wins,
            "losses": losses,
            "open": total - wins - slow_wins - losses,
            "winRate": round((wins + slow_wins) / max(total, 1) * 100, 1)
        }
    })


@app.route('/api/picks', methods=['POST'])
def add_pick():
    """Manually save a pick from ML scan."""
    data = request.json or {}
    ticker = data.get('ticker', '').upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    entry_price    = data.get('entryPrice', 0)
    ml_prob        = data.get('mlProb', 0)
    predicted_gain = data.get('predictedGain', 15)
    predicted_days = data.get('predictedDays', 3)
    pick_id = save_pick(
        ticker=ticker,
        entry_price=entry_price,
        ml_prob=ml_prob,
        predicted_gain_pct=predicted_gain,
        predicted_days=predicted_days
    )
    # Sync to Google Sheets
    from datetime import datetime as _dt
    add_pick_to_sheet(
        pick_id, ticker, _dt.now().strftime('%Y-%m-%d'),
        entry_price, ml_prob, predicted_gain, predicted_days
    )
    return jsonify({"id": pick_id, "message": f"{ticker} pick saved", "sheets": sheets_connected()})


@app.route('/api/picks/<int:pick_id>/price', methods=['POST'])
def add_pick_price(pick_id):
    """Add a daily price update for a pick."""
    data = request.json or {}
    day = int(data.get('day', 1))
    price = float(data.get('price', 0))
    if not (1 <= day <= 5):
        return jsonify({"error": "day must be 1-5"}), 400
    update_pick_price(pick_id, day, price)
    update_pick_in_sheet(pick_id, day=day, price=price)
    return jsonify({"message": f"Day {day} price updated"})


@app.route('/api/picks/check-outcomes', methods=['POST'])
def check_outcomes():
    """
    Auto-check open picks against current prices.
    Labels WIN / SLOW WIN / LOSS based on your rules:
    - Hits target within predicted_days+1 = WIN
    - Hits target but later = SLOW WIN
    - Predicted days passed, target not hit = LOSS
    """
    import yfinance as yf
    from datetime import datetime, timedelta

    open_picks = get_open_picks()
    updated = []

    for pick in open_picks:
        try:
            ticker = pick['ticker']
            entry  = pick['entry_price'] or 0
            if entry <= 0:
                continue

            target_gain = pick['predicted_gain_pct'] or 15
            pred_days   = pick['predicted_days'] or 3
            pick_date   = pick['pick_date']

            # Get current price + recent history
            hist = yf.Ticker(ticker).history(period="1mo", interval="1d", auto_adjust=True)
            if hist is None or len(hist) < 1:
                continue

            closes = hist["Close"].tolist()
            dates  = [str(d.date()) for d in hist.index.tolist()]

            # Find index of pick date
            try:
                start_idx = next(i for i, d in enumerate(dates) if d >= pick_date)
            except StopIteration:
                start_idx = len(dates) - 1

            target_price = entry * (1 + target_gain / 100)
            days_elapsed = len(dates) - start_idx
            outcome = None
            outcome_day = None
            actual_gain = None

            # Check each day after pick
            for i, (d, c) in enumerate(zip(dates[start_idx:], closes[start_idx:]), 1):
                gain_pct = (c - entry) / entry * 100
                if c >= target_price:
                    if i <= pred_days + 1:
                        outcome = "WIN"
                    else:
                        outcome = "SLOW WIN"
                    outcome_day = i
                    actual_gain = round(gain_pct, 2)
                    break

                # Update daily prices in DB
                if i <= 5:
                    update_pick_price(pick['id'], i, round(c, 2))

            # If predicted window passed and no hit
            if outcome is None and days_elapsed > pred_days + 1:
                latest_gain = (closes[-1] - entry) / entry * 100
                outcome = "LOSS"
                outcome_day = days_elapsed
                actual_gain = round(latest_gain, 2)

            if outcome:
                update_pick_outcome(pick['id'], outcome, outcome_day, actual_gain)
                updated.append({
                    "ticker": ticker,
                    "outcome": outcome,
                    "day": outcome_day,
                    "gain": actual_gain
                })

        except Exception as e:
            print(f"Outcome check error {pick['ticker']}: {e}")
            continue

    # Sync all to sheets after outcome check
    if updated:
        all_picks = get_all_picks()
        sync_all_picks_to_sheet(all_picks)
    return jsonify({"updated": updated, "checked": len(open_picks)})



@app.route('/api/discover', methods=['GET'])
def discover_candidates():
    """
    Discover today's candidates dynamically from live sources.
    Replaces the fixed SCAN_UNIVERSE for daily fresh picks.
    """
    try:
        import urllib.request, urllib.parse, json as _json
        from datetime import date, timedelta

        candidates = set()

        # Source 1: Yahoo Finance most active
        try:
            url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives&count=50&formatted=false"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                data = _json.loads(r.read())
            quotes = data.get("finance",{}).get("result",[{}])[0].get("quotes",[])
            for q in quotes:
                t = q.get("symbol","")
                if t and len(t) <= 5 and "." not in t:
                    candidates.add(t)
            print(f"Yahoo actives: {len(candidates)}")
        except Exception as e:
            print(f"Yahoo error: {e}")

        # Source 2: Yahoo Finance day gainers
        try:
            url2 = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=day_gainers&count=50&formatted=false"
            req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req2, timeout=8) as r:
                data2 = _json.loads(r.read())
            quotes2 = data2.get("finance",{}).get("result",[{}])[0].get("quotes",[])
            for q in quotes2:
                t = q.get("symbol","")
                if t and len(t) <= 5 and "." not in t:
                    candidates.add(t)
            print(f"After gainers: {len(candidates)}")
        except Exception as e:
            print(f"Yahoo gainers error: {e}")

        # Source 3: SEC 8-K filers today
        try:
            today = date.today().isoformat()
            url3 = f"https://efts.sec.gov/LATEST/search-index?forms=8-K&dateRange=custom&startdt={today}&enddt={today}"
            req3 = urllib.request.Request(url3, headers={"User-Agent": "SingerScout research@singer-scout.com"})
            with urllib.request.urlopen(req3, timeout=8) as r:
                data3 = _json.loads(r.read())
            hits = data3.get("hits",{}).get("hits",[]) or []
            for h in hits[:30]:
                src = h.get("_source",{})
                ticker = src.get("ticker","").upper()
                if ticker and len(ticker) <= 5 and ticker.isalpha():
                    candidates.add(ticker)
            print(f"After SEC 8-K: {len(candidates)}")
        except Exception as e:
            print(f"SEC error: {e}")

        # Always include known volatile stocks
        candidates.update(SCAN_UNIVERSE)

        return jsonify({
            "candidates": list(candidates),
            "count": len(candidates),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e), "candidates": SCAN_UNIVERSE}), 500



@app.route('/api/send-report', methods=['POST'])
def send_report():
    """Manually trigger email report."""
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        gmail_password = os.getenv("GMAIL_APP_PASSWORD")
        gmail_from     = os.getenv("GMAIL_FROM", "tomsinger03@gmail.com")
        gmail_to       = os.getenv("GMAIL_TO", "tomsinger03@gmail.com")

        if not gmail_password:
            return jsonify({"error": "GMAIL_APP_PASSWORD not set in environment"}), 400

        # Get latest picks from DB
        picks = get_all_picks()
        open_picks = [p for p in picks if not p.get("outcome")][:5]

        today_str = datetime.now().strftime("%A %d %B %Y")
        subject = f"Singer Scout Report — {today_str}"

        picks_html = ""
        for p in open_picks:
            picks_html += f"""
            <div style="background:#1a2030;border:1px solid #242d3d;border-radius:8px;padding:12px;margin-bottom:10px;">
                <span style="font-size:18px;font-weight:800;color:#00e5a0;font-family:monospace;">{p['ticker']}</span>
                <span style="font-size:14px;margin-left:10px;font-family:monospace;">${p.get('entry_price','–')}</span>
                <span style="font-size:12px;margin-left:8px;color:#ff9f1c;font-family:monospace;">ML: {round((p.get('ml_prob') or 0)*100)}%</span>
                <div style="font-size:11px;color:#64748b;margin-top:6px;font-family:monospace;">
                    Target: +{p.get('predicted_gain_pct',15)}% in {p.get('predicted_days',3)} days · Added {p.get('pick_date','')}
                </div>
                <div style="font-size:12px;color:{'#00e5a0' if p.get('outcome')=='WIN' else '#ff4560' if p.get('outcome')=='LOSS' else '#ff9f1c'};margin-top:4px;font-family:monospace;">
                    {p.get('outcome','OPEN')} {f"(+{p.get('actual_gain_pct')}%)" if p.get('actual_gain_pct') else ''}
                </div>
            </div>"""

        html = f"""
        <div style="background:#0b0e13;color:#e2e8f0;font-family:sans-serif;padding:20px;max-width:600px;">
            <div style="font-size:24px;font-weight:800;color:#00e5a0;margin-bottom:16px;">SINGER SCOUT</div>
            <div style="font-size:12px;color:#64748b;font-family:monospace;margin-bottom:20px;">{today_str}</div>
            <div style="font-size:13px;font-weight:700;color:#64748b;margin-bottom:10px;">OPEN PICKS</div>
            {picks_html if picks_html else '<div style="color:#64748b;font-family:monospace;">No open picks.</div>'}
            <div style="margin-top:20px;">
                <a href="https://stocks-app-ojo2.onrender.com" style="background:#00e5a0;color:#000;padding:10px 20px;border-radius:8px;text-decoration:none;font-weight:700;">
                    Open Singer Scout →
                </a>
            </div>
        </div>"""

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = gmail_from
        msg["To"]      = gmail_to
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_from, gmail_password)
            server.sendmail(gmail_from, gmail_to, msg.as_string())

        return jsonify({"message": f"Report sent to {gmail_to}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
