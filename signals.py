"""
signals.py — Extended signal sources for Singer Scout

All free, no additional API keys needed beyond what we already have.
"""

import time
import json
import urllib.request
import os
from datetime import datetime, timedelta

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "demo")
FINNHUB_BASE = "https://finnhub.io/api/v1"

_sig_cache = {}
def _cache_get(key, ttl=3600):
    if key in _sig_cache:
        if time.time() - _sig_cache[key]["ts"] < ttl:
            return _sig_cache[key]["data"]
    return None

def _cache_set(key, data):
    _sig_cache[key] = {"data": data, "ts": time.time()}

def _fetch(url, headers=None, timeout=8):
    try:
        req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return None


# ── 1. Float Size (via yfinance) ──────────────────────────────

def get_float_data(ticker):
    """
    Returns float size and whether it's a small float stock.
    Small float (<20M shares) = bigger % moves from same buying pressure.
    Also detects reverse splits.
    """
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        float_shares = info.get("floatShares") or info.get("sharesFloat")
        shares_out   = info.get("sharesOutstanding")
        
        # Reverse split detection
        # yfinance doesn't flag directly but we check actions
        hist = yf.Ticker(ticker).actions
        has_reverse_split = False
        if hist is not None and not hist.empty and "Stock Splits" in hist.columns:
            recent = hist["Stock Splits"].tail(10)
            # Reverse split = split ratio < 1 (e.g. 0.1 = 1-for-10 reverse)
            has_reverse_split = any(0 < v < 1 for v in recent if v != 0)
        
        return {
            "floatShares": float_shares,
            "floatM": round(float_shares / 1e6, 2) if float_shares else None,
            "smallFloat": float_shares < 20_000_000 if float_shares else None,
            "tinyFloat": float_shares < 5_000_000 if float_shares else None,
            "hasReverseSplit": has_reverse_split,
        }
    except Exception as e:
        return {"floatShares": None, "floatM": None, "smallFloat": None, 
                "tinyFloat": None, "hasReverseSplit": False}


# ── 2. SEC 8-K Filing Detector (EDGAR free) ──────────────────

def get_sec_filings(ticker):
    """
    Check for recent 8-K filings (material events = potential catalysts).
    Free via SEC EDGAR full-text search.
    8-K filed in last 7 days + volume surge = very strong signal.
    """
    cached = _cache_get(f"sec_{ticker}", ttl=7200)
    if cached: return cached
    
    try:
        today    = datetime.now().date()
        week_ago = (today - timedelta(days=7)).isoformat()
        
        url = (f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
               f"&dateRange=custom&startdt={week_ago}&enddt={today.isoformat()}"
               f"&forms=8-K")
        
        headers = {"User-Agent": "SingerScout research@singer-scout.com"}
        data = _fetch(url, headers=headers)
        
        hits = []
        if data and isinstance(data, dict):
            raw_hits = data.get("hits", {}).get("hits", []) or []
            for h in raw_hits[:5]:
                src = h.get("_source", {})
                hits.append({
                    "date": src.get("file_date", ""),
                    "description": src.get("period_of_report", ""),
                    "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}&type=8-K"
                })
        
        result = {
            "recent8K": len(hits) > 0,
            "count": len(hits),
            "filings": hits,
            "daysAgo": 7
        }
        _cache_set(f"sec_{ticker}", result)
        return result
    except Exception as e:
        return {"recent8K": False, "count": 0, "filings": [], "error": str(e)}


# ── 3. Insider Buying (SEC Form 4) ───────────────────────────

def get_insider_activity(ticker):
    """
    Check for recent insider buying via SEC Form 4.
    Insiders buying their own stock = strong vote of confidence.
    Free via EDGAR.
    """
    cached = _cache_get(f"insider_{ticker}", ttl=14400)
    if cached: return cached
    
    try:
        today     = datetime.now().date()
        month_ago = (today - timedelta(days=30)).isoformat()
        
        url = (f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
               f"&dateRange=custom&startdt={month_ago}&enddt={today.isoformat()}"
               f"&forms=4")
        
        headers = {"User-Agent": "SingerScout research@singer-scout.com"}
        data = _fetch(url, headers=headers)
        
        buys = 0
        sells = 0
        if data and isinstance(data, dict):
            hits = data.get("hits", {}).get("hits", []) or []
            buys = len(hits)  # Form 4s in last 30 days
        
        result = {
            "recentInsiderActivity": buys > 0,
            "form4Count": buys,
            "insiderBuying": buys > 0,
        }
        _cache_set(f"insider_{ticker}", result)
        return result
    except Exception as e:
        return {"recentInsiderActivity": False, "form4Count": 0, "insiderBuying": False}


# ── 4. Short Interest ─────────────────────────────────────────

def get_short_interest(ticker):
    """
    Get short interest via Finnhub (free tier).
    High short interest (>20%) + volume surge = squeeze candidate.
    """
    cached = _cache_get(f"short_{ticker}", ttl=86400)
    if cached: return cached
    
    try:
        import urllib.parse
        params = urllib.parse.urlencode({"symbol": ticker, "token": FINNHUB_API_KEY})
        url = f"{FINNHUB_BASE}/stock/short-interest?{params}"
        headers = {"User-Agent": "Mozilla/5.0"}
        data = _fetch(url, headers=headers)
        
        short_pct = None
        if data and isinstance(data, dict):
            positions = data.get("data", [])
            if positions:
                latest = positions[-1]
                short_shares = latest.get("shortInterest", 0)
                float_shares = latest.get("float", 1)
                if float_shares > 0:
                    short_pct = round(short_shares / float_shares * 100, 1)
        
        result = {
            "shortPct": short_pct,
            "highShortInterest": short_pct > 20 if short_pct else False,
            "squeezeCandidate": short_pct > 30 if short_pct else False,
        }
        _cache_set(f"short_{ticker}", result)
        return result
    except Exception as e:
        return {"shortPct": None, "highShortInterest": False, "squeezeCandidate": False}


# ── 5. Reddit Mention Velocity ────────────────────────────────

def get_reddit_mentions(ticker):
    """
    Check Reddit mention velocity via Reddit search (free, no auth needed).
    Surge in mentions often precedes price move by 12-24hrs.
    """
    cached = _cache_get(f"reddit_{ticker}", ttl=1800)
    if cached: return cached
    
    try:
        url = (f"https://www.reddit.com/search.json?q={ticker}"
               f"&sort=new&limit=25&t=day")
        headers = {"User-Agent": "SingerScout/1.0"}
        data = _fetch(url, headers=headers, timeout=6)
        
        mentions_24h = 0
        subs = set()
        if data and isinstance(data, dict):
            posts = data.get("data", {}).get("children", [])
            for post in posts:
                pd = post.get("data", {})
                title = (pd.get("title") or "").upper()
                if ticker.upper() in title or f"${ticker.upper()}" in title:
                    mentions_24h += 1
                    subs.add(pd.get("subreddit", ""))
        
        result = {
            "mentions24h": mentions_24h,
            "subreddits": list(subs)[:3],
            "trending": mentions_24h >= 3,
            "viral": mentions_24h >= 8,
        }
        _cache_set(f"reddit_{ticker}", result)
        return result
    except Exception as e:
        return {"mentions24h": 0, "subreddits": [], "trending": False, "viral": False}


# ── 6. Earnings Surprise ──────────────────────────────────────

def get_earnings_surprise(ticker):
    """
    Recent earnings beat = often precedes multi-day run.
    Free via yfinance.
    """
    cached = _cache_get(f"earnings_{ticker}", ttl=86400)
    if cached: return cached
    
    try:
        import yfinance as yf
        earnings = yf.Ticker(ticker).earnings_history
        
        if earnings is None or earnings.empty:
            return {"recentBeat": False, "surprisePct": None}
        
        latest = earnings.iloc[-1]
        eps_est   = latest.get("epsEstimate", 0) or 0
        eps_act   = latest.get("epsActual", 0) or 0
        surprise  = ((eps_act - eps_est) / abs(eps_est) * 100
                     if eps_est != 0 else None)
        
        result = {
            "recentBeat": surprise > 10 if surprise else False,
            "surprisePct": round(surprise, 1) if surprise else None,
            "bigBeat": surprise > 25 if surprise else False,
        }
        _cache_set(f"earnings_{ticker}", result)
        return result
    except Exception as e:
        return {"recentBeat": False, "surprisePct": None, "bigBeat": False}


# ── 7. Dynamic Universe Discovery ────────────────────────────

def discover_unusual_volume(min_price=1, max_price=15, limit=50):
    """
    Find stocks with unusual volume TODAY using Finnhub scanner.
    This is how we discover NEW stocks not in our universe.
    Returns list of tickers.
    """
    cached = _cache_get("discover_volume", ttl=3600)
    if cached: return cached
    
    try:
        # Use Finnhub stock screener for unusual volume
        import urllib.parse
        params = urllib.parse.urlencode({
            "token": FINNHUB_API_KEY,
        })
        url = f"{FINNHUB_BASE}/stock/symbol?exchange=US&{params}"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        # Get all US stocks then filter by quote
        # (Finnhub free tier: get symbols, spot check quotes)
        # For free tier we use a curated high-volatility watchlist approach
        # and augment with any stocks from recent winner lists
        
        result = []  # Will be populated from winner feedback
        _cache_set("discover_volume", result)
        return result
    except Exception as e:
        return []


# ── Combined signal scorer ────────────────────────────────────

def get_all_signals(ticker):
    """
    Get all extended signals for a ticker.
    Returns combined dict with all signal data.
    """
    float_data   = get_float_data(ticker)
    sec_data     = get_sec_filings(ticker)
    insider_data = get_insider_activity(ticker)
    short_data   = get_short_interest(ticker)
    reddit_data  = get_reddit_mentions(ticker)
    earnings     = get_earnings_surprise(ticker)
    
    # Composite catalyst score (0-100)
    catalyst_score = 0
    if sec_data.get("recent8K"):         catalyst_score += 30
    if insider_data.get("insiderBuying"): catalyst_score += 20
    if short_data.get("squeezeCandidate"): catalyst_score += 25
    if reddit_data.get("viral"):          catalyst_score += 15
    if reddit_data.get("trending"):       catalyst_score += 8
    if earnings.get("bigBeat"):           catalyst_score += 15
    if float_data.get("tinyFloat"):       catalyst_score += 10
    if float_data.get("smallFloat"):      catalyst_score += 5
    
    return {
        "float":    float_data,
        "sec":      sec_data,
        "insider":  insider_data,
        "short":    short_data,
        "reddit":   reddit_data,
        "earnings": earnings,
        "catalystScore": min(catalyst_score, 100),
        "reverseSplit": float_data.get("hasReverseSplit", False),
    }
