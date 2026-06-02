"""
ml_model.py

Trains a Random Forest classifier to predict whether a stock will gain
15%+ within the next 3 trading days.

Features used (all derived from price/volume history - no external data needed):
  - RSI 14
  - MACD signal
  - Bollinger Band position
  - Volume surge (vs 20-day avg)
  - 5-day momentum
  - 10-day momentum
  - ATR ratio (volatility normalised to price)
  - Price vs 20-day SMA
  - Price vs 50-day SMA
  - Bollinger Band width (squeeze detector)
  - Volume trend (5d vs 20d avg volume)
  - Day of week (Mon-Fri encoded)
  - Gap (today open vs yesterday close)
"""

import os
import time
import pickle
import threading
import numpy as np

# ── lazy imports to avoid slow startup ──
_sklearn_loaded = False
_RandomForest = None

def _load_sklearn():
    global _sklearn_loaded, _RandomForest
    if not _sklearn_loaded:
        from sklearn.ensemble import RandomForestClassifier
        _RandomForest = RandomForestClassifier
        _sklearn_loaded = True

# ── Liquid stocks to train/scan on ──
# ~300 liquid US stocks across sectors, price >$5, avg volume >500k
SCAN_UNIVERSE = [
    # Mega cap tech
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AVGO","ORCL","CRM",
    "ADBE","AMD","INTC","QCOM","TXN","MU","AMAT","LRCX","KLAC","MRVL",
    "SNOW","PLTR","UBER","LYFT","ABNB","DASH","RBLX","COIN","HOOD","SOFI",
    # Finance
    "JPM","BAC","GS","MS","WFC","C","BLK","SCHW","AXP","V","MA","PYPL",
    "COF","DFS","SYF","ALLY","USB","PNC","TFC","KEY",
    # Healthcare
    "JNJ","UNH","PFE","ABBV","MRK","LLY","BMY","AMGN","GILD","BIIB",
    "REGN","VRTX","MRNA","BNTX","DXCM","ISRG","SYK","MDT","BSX","EW",
    # Consumer
    "AMZN","WMT","TGT","COST","HD","LOW","NKE","SBUX","MCD","YUM",
    "CMG","DPZ","DKNG","PENN","MGM","LVS","WYNN","CZR","NCLH","CCL",
    # Energy
    "XOM","CVX","COP","EOG","PXD","SLB","HAL","BKR","MPC","PSX",
    "VLO","DVN","FANG","OXY","HES","APA","MRO","RRC","AR","EQT",
    # Industrial/EV
    "BA","CAT","DE","HON","GE","MMM","RTX","LMT","NOC","GD",
    "RIVN","LCID","NIO","LI","XPEV","FSR","GOEV","NKLA","WKHS","REE",
    # ETFs (liquid, good for signals)
    "SPY","QQQ","IWM","XLF","XLK","XLE","XLV","XLI","ARKK","SOXS",
    # Growth / high vol
    "SHOP","SQ","TWLO","NET","DDOG","CRWD","ZS","OKTA","PANW","FTNT",
    "ZM","DOCN","GTLB","MDB","ESTC","CFLT","HUBS","BILL","PCTY","WEX",
    "AFRM","UPST","OPEN","OPENDOOR","LMND","ROOT","HIPPO","CLOV","ACMR",
    # Biotech (volatile - good for 15% moves)
    "SGEN","ALNY","BMRN","EXAS","NKTR","FATE","CRSP","EDIT","NTLA","BEAM",
    "PACB","ILMN","VEEV","IDXX","ZBH","HOLX","TECH","NTRA","NEOG","MASI",
    # Media/Comm
    "NFLX","DIS","CMCSA","T","VZ","TMUS","CHTR","PARA","WBD","FOX",
    "SPOT","PINS","SNAP","TWTR","MTCH","IAC","ZG","TRIP","EXPE","BKNG",
    # Commodities/Materials
    "FCX","NEM","GOLD","WPM","AEM","KGC","HL","PAAS","AG","EXK",
    "AA","CENX","STLD","NUE","CLF","X","RS","CMC","SCCO","MP",
    # REITs
    "AMT","PLD","EQIX","CCI","SPG","O","VICI","WPC","NNN","STOR",
    # Small/mid cap high momentum
    "IONQ","QUBT","RGTI","ARQQ","QMCO","BTBT","MARA","RIOT","CLSK","HUT",
    "SMCI","WOLF","LAZR","MVIS","OUST","LIDR","AEVA","INVZ","VLDR","HYLN",
]

# deduplicate
SCAN_UNIVERSE = list(dict.fromkeys(SCAN_UNIVERSE))

# ── Feature extraction ──

def _ema(prices, period):
    if len(prices) < period:
        return None
    k = 2 / (period + 1)
    e = prices[0]
    for p in prices[1:]:
        e = p * k + e * (1 - k)
    return e

def extract_features(closes, volumes, opens=None):
    """
    Extract ML features from price/volume arrays.
    Returns a dict of features, or None if not enough data.
    """
    if len(closes) < 55:
        return None

    c = closes
    v = volumes

    # RSI
    gains, losses = [], []
    for i in range(1, len(c)):
        diff = c[i] - c[i-1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    ag = sum(gains[-14:]) / 14
    al = sum(losses[-14:]) / 14
    rsi = 100 - (100 / (1 + ag/al)) if al != 0 else 100

    # MACD
    ema12 = _ema(c[-26:], 12)
    ema26 = _ema(c[-26:], 26)
    macd = (ema12 - ema26) / c[-1] if ema12 and ema26 else 0

    # Bollinger
    window = c[-20:]
    mean20 = sum(window) / 20
    std20 = (sum((x - mean20)**2 for x in window) / 20) ** 0.5
    boll_pos = (c[-1] - (mean20 - 2*std20)) / (4*std20) if std20 > 0 else 0.5
    boll_width = (4 * std20) / mean20 if mean20 > 0 else 0  # squeeze: low = tight

    # Volume surge
    avg_vol = sum(v[-21:-1]) / 20 if len(v) >= 21 else sum(v) / len(v)
    vol_surge = v[-1] / avg_vol if avg_vol > 0 else 1

    # Volume trend
    vol5 = sum(v[-5:]) / 5
    vol20 = sum(v[-20:]) / 20
    vol_trend = vol5 / vol20 if vol20 > 0 else 1

    # Momentum
    mom5  = (c[-1] - c[-6])  / c[-6]  * 100 if len(c) >= 6  else 0
    mom10 = (c[-1] - c[-11]) / c[-11] * 100 if len(c) >= 11 else 0

    # ATR (average true range normalised)
    trs = [abs(c[i] - c[i-1]) for i in range(max(1, len(c)-14), len(c))]
    atr = (sum(trs) / len(trs)) / c[-1] * 100 if trs and c[-1] > 0 else 0

    # vs Moving averages
    sma20 = sum(c[-20:]) / 20
    sma50 = sum(c[-50:]) / 50
    vs_sma20 = (c[-1] - sma20) / sma20 * 100 if sma20 > 0 else 0
    vs_sma50 = (c[-1] - sma50) / sma50 * 100 if sma50 > 0 else 0

    # Gap (open vs prev close) - use closes as proxy if opens not available
    gap = (c[-1] - c[-2]) / c[-2] * 100 if opens is None else \
          (opens[-1] - c[-2]) / c[-2] * 100 if len(opens) >= 2 and c[-2] > 0 else 0

    return {
        "rsi": rsi,
        "macd": macd,
        "boll_pos": max(0, min(1, boll_pos)),
        "boll_width": boll_width,
        "vol_surge": min(vol_surge, 10),
        "vol_trend": min(vol_trend, 5),
        "mom5": mom5,
        "mom10": mom10,
        "atr": atr,
        "vs_sma20": vs_sma20,
        "vs_sma50": vs_sma50,
        "gap": gap,
    }

FEATURE_COLS = [
    "rsi","macd","boll_pos","boll_width",
    "vol_surge","vol_trend","mom5","mom10",
    "atr","vs_sma20","vs_sma50","gap"
]

# ── Model state ──
_model = None
_model_lock = threading.Lock()
_model_status = "untrained"   # untrained | training | ready | failed
_model_trained_at = None
_training_log = []

MODEL_PATH = "momentum_model.pkl"

def get_status():
    return {
        "status": _model_status,
        "trained_at": _model_trained_at,
        "log": _training_log[-10:],  # last 10 log lines
    }

def _log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    _training_log.append(line)
    print(line)

try:
    from datetime import datetime
except ImportError:
    pass

# ── Training ──

def train_model_background():
    """Kick off training in a background thread."""
    global _model_status
    if _model_status == "training":
        return
    _model_status = "training"
    t = threading.Thread(target=_train_model, daemon=True)
    t.start()

def _train_model():
    global _model, _model_status, _model_trained_at

    try:
        _load_sklearn()
        import yfinance as yf

        # Try loading cached model first
        if os.path.exists(MODEL_PATH):
            age = time.time() - os.path.getmtime(MODEL_PATH)
            if age < 86400:  # less than 24h old
                _log("Loading cached model from disk...")
                with open(MODEL_PATH, "rb") as f:
                    _model = pickle.load(f)
                _model_status = "ready"
                _model_trained_at = datetime.now().isoformat()
                _log("Cached model loaded OK")
                return

        _log(f"Starting training on {len(SCAN_UNIVERSE)} stocks (2yr history)...")
        _log("This takes 5-10 minutes on first run...")

        X, y = [], []
        trained_on = 0
        TARGET_GAIN = 0.15   # 15%
        FORWARD_DAYS = 3

        for i, ticker in enumerate(SCAN_UNIVERSE):
            try:
                hist = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=True)
                if hist is None or len(hist) < 60:
                    continue

                closes  = hist["Close"].tolist()
                volumes = hist["Volume"].tolist()
                opens   = hist["Open"].tolist()
                highs   = hist["High"].tolist()

                # Slide a window across history
                for j in range(55, len(closes) - FORWARD_DAYS):
                    feats = extract_features(
                        closes[:j+1],
                        volumes[:j+1],
                        opens[:j+1]
                    )
                    if feats is None:
                        continue

                    # Label: did price gain 15%+ in next 3 days (using daily highs)?
                    current_price = closes[j]
                    future_high = max(highs[j+1:j+1+FORWARD_DAYS])
                    gain = (future_high - current_price) / current_price
                    label = 1 if gain >= TARGET_GAIN else 0

                    X.append([feats[col] for col in FEATURE_COLS])
                    y.append(label)

                trained_on += 1
                if trained_on % 20 == 0:
                    pos = sum(y)
                    _log(f"  {trained_on}/{len(SCAN_UNIVERSE)} stocks processed — {len(X)} samples, {pos} positives ({100*pos/len(y):.1f}%)")

                time.sleep(0.1)  # be gentle with yfinance

            except Exception as e:
                _log(f"  {ticker}: skipped ({e})")
                continue

        if len(X) < 100:
            _log("Not enough training data — check network/yfinance")
            _model_status = "failed"
            return

        X = np.array(X)
        y = np.array(y)

        pos_rate = sum(y) / len(y)
        _log(f"Training on {len(X)} samples — {100*pos_rate:.1f}% positive (15%+ moves)")
        _log("Fitting Random Forest...")

        # Balance classes — 15% moves are rare so we weight them up
        clf = _RandomForest(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )
        clf.fit(X, y)

        # Save to disk
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(clf, f)

        _model = clf
        _model_status = "ready"
        _model_trained_at = datetime.now().isoformat()

        importances = sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1])
        _log("Top features: " + ", ".join(f"{k}={v:.3f}" for k,v in importances[:5]))
        _log(f"Model ready! Trained on {len(X)} samples from {trained_on} stocks.")

    except Exception as e:
        _log(f"Training failed: {e}")
        _model_status = "failed"
        import traceback
        traceback.print_exc()


# ── Prediction ──

def predict(closes, volumes, opens=None):
    """
    Returns probability (0-1) of 15%+ gain in next 3 days.
    Returns None if model not ready or not enough data.
    """
    if _model is None or _model_status != "ready":
        return None

    feats = extract_features(closes, volumes, opens)
    if feats is None:
        return None

    X = np.array([[feats[col] for col in FEATURE_COLS]])
    prob = _model.predict_proba(X)[0]

    # index of class 1 (positive)
    classes = list(_model.classes_)
    if 1 in classes:
        return round(float(prob[classes.index(1)]), 4)
    return None
