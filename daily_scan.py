"""
daily_scan.py — Singer Scout Daily Morning Scan

Runs independently via GitHub Actions every weekday at 9am ET.
1. Discovers today's unusual volume stocks
2. Filters with your criteria
3. Scores with ML signals
4. Sends email report to tomsinger03@gmail.com
"""

import os
import sys
import time
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, date

# ── Config from environment ──
FINNHUB_API_KEY  = os.getenv("FINNHUB_API_KEY", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
GMAIL_FROM       = os.getenv("GMAIL_FROM", "tomsinger03@gmail.com")
GMAIL_TO         = os.getenv("GMAIL_TO", "tomsinger03@gmail.com")
RENDER_URL       = os.getenv("RENDER_URL", "https://stocks-app-ojo2.onrender.com")

print(f"Singer Scout Daily Scan — {datetime.now().strftime('%Y-%m-%d %H:%M')} ET")

# ── Step 1: Discover today's candidates ──

def get_finnhub_movers():
    """Get unusual volume stocks from Finnhub."""
    import urllib.request, urllib.parse
    try:
        params = urllib.parse.urlencode({"token": FINNHUB_API_KEY})
        url = f"https://finnhub.io/api/v1/stock/symbol?exchange=US&{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "SingerScout/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            symbols = json.loads(r.read())
        # Get only common stocks, not ETFs
        tickers = [s["symbol"] for s in symbols
                   if s.get("type") == "Common Stock"
                   and "." not in s["symbol"]
                   and len(s["symbol"]) <= 5]
        print(f"Found {len(tickers)} US common stocks from Finnhub")
        return tickers[:2000]  # Cap at 2000
    except Exception as e:
        print(f"Finnhub symbols error: {e}")
        return []

def get_yahoo_actives():
    """Scrape Yahoo Finance most active stocks."""
    try:
        import urllib.request
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives&count=50&formatted=false"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        })
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
        tickers = [q["symbol"] for q in quotes if q.get("symbol")]
        print(f"Yahoo actives: {tickers[:10]}...")
        return tickers
    except Exception as e:
        print(f"Yahoo actives error: {e}")
        return []

def get_sec_8k_filers():
    """Get companies that filed 8-K today — catalyst stocks."""
    try:
        import urllib.request
        today = date.today().isoformat()
        url = f"https://efts.sec.gov/LATEST/search-index?forms=8-K&dateRange=custom&startdt={today}&enddt={today}"
        req = urllib.request.Request(url, headers={"User-Agent": "SingerScout research@singer-scout.com"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        hits = data.get("hits", {}).get("hits", []) or []
        tickers = []
        for h in hits[:50]:
            src = h.get("_source", {})
            ticker = src.get("period_of_report", "").upper()
            entity = src.get("entity_name", "")
            # Extract ticker from entity name if possible
            if ticker and len(ticker) <= 5 and ticker.isalpha():
                tickers.append(ticker)
        print(f"SEC 8-K filers today: {len(tickers)}")
        return tickers
    except Exception as e:
        print(f"SEC 8-K error: {e}")
        return []

# ── Step 2: Score candidates ──

def score_candidates(tickers):
    """Filter and score candidates using our signals."""
    import yfinance as yf
    import numpy as np

    # Screener criteria (your preset values)
    MIN_PRICE    = 1.0
    MAX_PRICE    = 15.0
    MIN_VOL_SURGE = 3.0
    MAX_RSI      = 65.0
    MAX_PCT_ABOVE_LOW = 150.0
    MIN_CONSEC_DOWN = 2

    results = []
    checked = 0

    for ticker in tickers:
        try:
            # Skip obvious ETFs and leveraged products
            tl = ticker.lower()
            if any(k in tl for k in ['2x','3x','ul','proshares','direxion']):
                continue

            hist = yf.Ticker(ticker).history(period="3mo", interval="1d", auto_adjust=True)
            if hist is None or len(hist) < 25:
                continue

            closes  = hist["Close"].tolist()
            volumes = hist["Volume"].tolist()
            price   = closes[-1]
            checked += 1

            # Price filter
            if not (MIN_PRICE <= price <= MAX_PRICE):
                continue

            # Basic signal calculation (inline for standalone script)
            # Volume surge
            avg_vol_20 = sum(volumes[-21:-1]) / 20 if len(volumes) >= 21 else sum(volumes)/len(volumes)
            vol_surge = volumes[-1] / avg_vol_20 if avg_vol_20 > 0 else 1

            if vol_surge < MIN_VOL_SURGE:
                continue

            # RSI
            gains = [max(closes[i]-closes[i-1], 0) for i in range(1, len(closes))]
            losses = [max(closes[i-1]-closes[i], 0) for i in range(1, len(closes))]
            ag = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0.01
            al = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0.01
            rsi = 100 - (100 / (1 + ag/al)) if al > 0 else 50

            if rsi > MAX_RSI:
                continue

            # Already spiked check
            mom3 = (closes[-1] - closes[-4]) / closes[-4] * 100 if len(closes) >= 4 else 0
            if mom3 >= 15:
                continue

            # Near 52w low
            low_52 = min(closes[-252:]) if len(closes) >= 252 else min(closes)
            pct_above_low = (price - low_52) / low_52 * 100 if low_52 > 0 else 999
            if pct_above_low > MAX_PCT_ABOVE_LOW:
                continue

            # Consecutive down days
            consec = 0
            for i in range(len(closes)-2, max(len(closes)-8, 0), -1):
                if closes[i] < closes[i-1]:
                    consec += 1
                else:
                    break

            if consec < MIN_CONSEC_DOWN:
                continue

            # Bollinger squeeze
            window = closes[-20:]
            mean20 = sum(window) / 20
            std20 = (sum((x-mean20)**2 for x in window) / 20) ** 0.5
            boll_squeeze = std20 / mean20 if mean20 > 0 else 0

            # Composite score for ranking
            score = 0
            score += min(vol_surge * 5, 30)      # volume is king
            score += max(0, (30 - rsi))           # oversold bonus
            score += min(consec * 5, 20)          # down days = spring
            score += max(0, 20 - pct_above_low/5) # near low bonus
            score += (1 - boll_squeeze) * 10       # squeeze bonus

            change_pct = (closes[-1]-closes[-2])/closes[-2]*100 if len(closes)>=2 else 0

            results.append({
                "ticker": ticker,
                "price": round(price, 2),
                "changePercent": round(change_pct, 2),
                "score": round(score, 1),
                "signals": {
                    "rsi": round(rsi, 1),
                    "volSurge": round(vol_surge, 1),
                    "consecDown": consec,
                    "pctAboveLow": round(pct_above_low, 1),
                    "mom3d": round(mom3, 2),
                }
            })

            time.sleep(0.1)

        except Exception as e:
            continue

        if checked % 50 == 0:
            print(f"  Checked {checked} stocks, {len(results)} passed so far...")

    results.sort(key=lambda x: x["score"], reverse=True)
    print(f"Screener: checked {checked}, {len(results)} passed filters")
    return results[:20]  # Top 20


# ── Step 3: Send email ──

def send_email(picks):
    """Send HTML email report with top picks."""

    today_str = datetime.now().strftime("%A %d %B %Y")
    uk_time   = datetime.now().strftime("%H:%M")

    if not picks:
        subject = f"Singer Scout — No picks today {today_str}"
        body_text = "No stocks passed the filters today. Market may be quiet."
    else:
        subject = f"Singer Scout — {len(picks)} picks for {today_str} 📈"

    # Build HTML email
    picks_html = ""
    for i, p in enumerate(picks[:5], 1):
        sig = p.get("signals", {})
        score = p.get("score", 0)
        score_color = "#00e5a0" if score >= 40 else "#ff9f1c" if score >= 25 else "#64748b"

        picks_html += f"""
        <div style="background:#1a2030;border:1px solid #242d3d;border-radius:8px;padding:16px;margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <div>
                    <span style="font-size:20px;font-weight:800;color:#00e5a0;font-family:monospace;">{p['ticker']}</span>
                    <span style="font-size:16px;margin-left:10px;color:#e2e8f0;font-family:monospace;">${p['price']}</span>
                    <span style="font-size:12px;margin-left:8px;color:{'#00e5a0' if p['changePercent']>=0 else '#ff4560'};">
                        {'+' if p['changePercent']>=0 else ''}{p['changePercent']}% today
                    </span>
                </div>
                <div style="font-family:monospace;font-size:22px;font-weight:800;color:{score_color};">
                    {score}
                </div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px;">
                <div style="background:#0b0e13;border-radius:6px;padding:8px;text-align:center;">
                    <div style="font-size:10px;color:#64748b;font-family:monospace;">RSI</div>
                    <div style="font-size:16px;font-weight:700;color:#e2e8f0;font-family:monospace;">{sig.get('rsi','–')}</div>
                </div>
                <div style="background:#0b0e13;border-radius:6px;padding:8px;text-align:center;">
                    <div style="font-size:10px;color:#64748b;font-family:monospace;">VOL SURGE</div>
                    <div style="font-size:16px;font-weight:700;color:#00e5a0;font-family:monospace;">{sig.get('volSurge','–')}×</div>
                </div>
                <div style="background:#0b0e13;border-radius:6px;padding:8px;text-align:center;">
                    <div style="font-size:10px;color:#64748b;font-family:monospace;">DOWN DAYS</div>
                    <div style="font-size:16px;font-weight:700;color:#ff9f1c;font-family:monospace;">{sig.get('consecDown','–')}d↓</div>
                </div>
            </div>
            <div style="font-size:11px;color:#64748b;font-family:monospace;">
                {sig.get('pctAboveLow','–')}% above 52w low · Mom3d: {sig.get('mom3d','–')}%
            </div>
            <a href="https://finance.yahoo.com/quote/{p['ticker']}" 
               style="display:inline-block;margin-top:8px;background:#00e5a0;color:#000;padding:6px 14px;border-radius:6px;font-size:12px;font-weight:700;text-decoration:none;">
                View {p['ticker']} →
            </a>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <body style="background:#0b0e13;color:#e2e8f0;font-family:sans-serif;padding:20px;max-width:600px;margin:0 auto;">
        
        <div style="text-align:center;margin-bottom:24px;padding-bottom:16px;border-bottom:1px solid #242d3d;">
            <div style="font-size:28px;font-weight:800;color:#00e5a0;letter-spacing:-1px;">SINGER SCOUT</div>
            <div style="font-size:12px;color:#64748b;font-family:monospace;margin-top:4px;">Daily Morning Report · {today_str}</div>
        </div>

        <div style="background:#131720;border:1px solid #242d3d;border-radius:8px;padding:12px;margin-bottom:20px;font-family:monospace;font-size:12px;color:#64748b;">
            ⚠️ Pattern recognition only — not financial advice. Always do your own research. Max £10-15 per pick.
        </div>

        <div style="font-size:14px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#64748b;margin-bottom:12px;">
            Top {len(picks[:5])} picks today
        </div>

        {picks_html if picks else '<div style="color:#64748b;font-family:monospace;">No stocks passed filters today.</div>'}

        <div style="margin-top:24px;padding-top:16px;border-top:1px solid #242d3d;text-align:center;">
            <a href="{RENDER_URL}" style="background:#131720;color:#00e5a0;padding:10px 20px;border-radius:8px;text-decoration:none;font-weight:700;border:1px solid #00e5a0;">
                Open Singer Scout App →
            </a>
        </div>

        <div style="margin-top:16px;font-size:10px;color:#64748b;text-align:center;font-family:monospace;">
            Singer Scout · Auto-scan ran at {uk_time} UTC · NYSE opens 14:30 UK time
        </div>
    </body>
    </html>
    """

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_FROM
        msg["To"]      = GMAIL_TO
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_FROM, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_FROM, GMAIL_TO, msg.as_string())

        print(f"✅ Email sent to {GMAIL_TO}")
        return True
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False


# ── Main ──
if __name__ == "__main__":
    print("\n=== Step 1: Discovering candidates ===")
    
    # Combine sources
    candidates = set()
    
    # Yahoo most active (fast, reliable)
    yahoo = get_yahoo_actives()
    candidates.update(yahoo)
    
    # SEC 8-K filers today (catalyst stocks)
    sec = get_sec_8k_filers()
    candidates.update(sec)
    
    # Always include known volatile stocks as backup
    known = [
        "BJDX","LASE","DXST","STAK","RKTO","SBFM","CXAI","TWAV",
        "HCAT","WXM","STI","VERU","FOXX","EDHL","INDP","SPRC",
        "MARA","RIOT","CLSK","NVAX","OCGN","PLTR","HOOD","SOFI",
        "IONQ","QUBT","RGTI","NIO","XPEV","SMCI","CELH","AFRM",
    ]
    candidates.update(known)
    
    candidates = list(candidates)
    print(f"Total candidates to screen: {len(candidates)}")
    
    print("\n=== Step 2: Screening candidates ===")
    picks = score_candidates(candidates)
    print(f"Top picks: {[p['ticker'] for p in picks[:5]]}")
    
    print("\n=== Step 3: Sending email ===")
    if not GMAIL_APP_PASSWORD:
        print("No GMAIL_APP_PASSWORD set — skipping email")
        print("Top picks would be:", [p['ticker'] for p in picks[:5]])
    else:
        send_email(picks)
    
    print("\n=== Done ===")
    sys.exit(0)
