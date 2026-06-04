"""
ml_model.py  v2 — Micro/Small Cap Explosive Move Predictor

Trained on stocks that have historically made 15%+ moves in 1-3 days.
Target universe: $1-$30, liquid enough to trade, small float preferred.

Key insight from real movers (DXST +183%, LASE +132%, BJDX +91% etc):
  - Volume explosion is the #1 signal (10-100x normal volume)
  - Price compressed near lows (coiled spring)
  - Small market cap / float
  - RSI recovering from oversold
  - Bollinger Band squeeze releasing

Features:
  1. vol_surge_5d    — today vol vs 5d avg (immediate)
  2. vol_surge_20d   — today vol vs 20d avg (broader)
  3. vol_acceleration — 5d vol trend accelerating
  4. rsi             — momentum position
  5. boll_squeeze    — band width (tight = coiled)
  6. boll_pos        — position in bands
  7. price_vs_52low  — % above 52-week low (near lows = ready to spring)
  8. price_vs_52high — % below 52-week high
  9. mom3d           — 3-day price momentum
  10. mom5d          — 5-day price momentum
  11. atr_pct        — volatility normalised to price
  12. consec_down    — consecutive down days before today (oversold pressure)
"""

import os
import time
import pickle
import threading
import numpy as np
from datetime import datetime

_sklearn_loaded = False
_RandomForest = None

def _load_sklearn():
    global _sklearn_loaded, _RandomForest
    if not _sklearn_loaded:
        from sklearn.ensemble import RandomForestClassifier
        _RandomForest = RandomForestClassifier
        _sklearn_loaded = True

# ── Universe: small/micro cap liquid stocks where big moves happen ──
# Mix of known volatile names + sectors that produce explosive moves
SCAN_UNIVERSE = [
    # Biotech/Health tech (FDA catalysts, volatile)
    "SIGA","NVAX","OCGN","CLOV","FFIE","IDEANOMICS","MMAT","CENN","MULN",
    "NKLA","WKHS","GOEV","RIDE","SOLO","XPEV","LI","NIO",
    "ATOS","BBAI","GFAI","INPX","IMPP","EDBL","DRUG","BURU",
    "PHAT","SHOT","MAPS","GRPN","COMS","ENVB","AEYE","CNEY",
    "HUDI","JFIN","PFIS","NXPL","CJET","AIXI","ZJYL","BJDX",
    "ABTS","SBFM","RKTO","LASE","FOFO","SWMR","BBGI","PUSA",
    # Small cap tech/commercial
    "SOPA","CODA","MFON","XBRT","GMGI","AWIN","ITRG","BSFC",
    "GSUN","BTOG","LIQT","VERB","TNXP","GREE","MARA","RIOT",
    "CLSK","HUT","BTBT","CIFR","SDIG","IREN","CORZ","WULF",
    # Micro cap industrial/services
    "DXST","STAK","JLHL","WTF","FOFO",
    # ETFs for calibration
    "SOXS","SQQQ","TQQQ","SPXU","UVXY",
    # Mid cap growth (still volatile)
    "PLTR","HOOD","SOFI","AFRM","UPST","OPEN","LMND","ROOT",
    "CLOV","WISH","BARK","OPAD","SPIR","ATIP","BMBL","BIRD",
    "RBLX","DKNG","PENN","ACMR","IONQ","QUBT","RGTI",
    "LAZR","MVIS","OUST","LIDR","VLDR","HYLN","NKLA",
    # Today's winners added
    "CXAI","TWAV","HCAT","WXM","SBFM","AVOS","SAVG",
    # New winners 03/06
    "STI","VERU","FOXX","EDHL","INDP","SPRC","LNZA","LSE","ROLR","MNTS","BNAI","ASTI",
    "BJDX","LASE","DXST","STAK","RKTO","ZJYL","JLHL","SWMR","FOFO","PUSA","ABTS",
    "SMCI","WOLF","CELH","NKTR","FATE","CRSP","EDIT","NTLA",
    "BEAM","PACB","VERV","TELA","RCKT","RETA","IRON","PRLD",
    # More small caps with history of big moves
    "CGEN","DARE","APRE","PDSB","MRKR","TALS","NRXP","SABS",
    "ILUS","NXXT","LPCN","ENTX","BHAT","HUDI","ATHE","CNET",
    "BIMI","SFUN","RETO","TOUR","CIFS","CLPS","AIFU","TAOP",
]
# Remove duplicates and exclude leveraged ETFs
_ETF_EXCLUDE = ['2x','3x','-2x','-3x','ultra','leverage']
SCAN_UNIVERSE = list(dict.fromkeys([
    t for t in SCAN_UNIVERSE
    if not any(k in t.lower() for k in _ETF_EXCLUDE)
]))

FEATURE_COLS = [
    "vol_surge_5d","vol_surge_20d","vol_acceleration",
    "rsi","boll_squeeze","boll_pos",
    "price_vs_52low","price_vs_52high",
    "mom3d","mom5d","atr_pct","consec_down"
]

def extract_features(closes, volumes, highs=None, lows=None):
    if len(closes) < 25 or len(volumes) < 25:
        return None

    c = closes
    v = volumes

    # Volume surges — THE key signal
    avg_vol_5  = sum(v[-6:-1]) / 5  if len(v) >= 6  else sum(v)/len(v)
    avg_vol_20 = sum(v[-21:-1]) / 20 if len(v) >= 21 else sum(v)/len(v)
    vol_surge_5  = v[-1] / avg_vol_5  if avg_vol_5  > 0 else 1
    vol_surge_20 = v[-1] / avg_vol_20 if avg_vol_20 > 0 else 1

    # Volume acceleration (is volume building over last 3 days?)
    if len(v) >= 4:
        recent_avg = sum(v[-3:]) / 3
        prior_avg  = sum(v[-6:-3]) / 3 if len(v) >= 6 else avg_vol_20
        vol_accel  = recent_avg / prior_avg if prior_avg > 0 else 1
    else:
        vol_accel = 1

    # RSI 14
    gains, losses = [], []
    for i in range(1, len(c)):
        diff = c[i] - c[i-1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    ag = sum(gains[-14:]) / 14 if len(gains) >= 14 else sum(gains)/max(len(gains),1)
    al = sum(losses[-14:]) / 14 if len(losses) >= 14 else sum(losses)/max(len(losses),1)
    rsi = 100 - (100 / (1 + ag/al)) if al > 0 else 50

    # Bollinger squeeze (band width = how compressed)
    window = c[-20:]
    mean20 = sum(window) / 20
    std20  = (sum((x - mean20)**2 for x in window) / 20) ** 0.5
    boll_squeeze = std20 / mean20 if mean20 > 0 else 0  # low = tight squeeze
    boll_pos = (c[-1] - (mean20 - 2*std20)) / (4*std20) if std20 > 0 else 0.5
    boll_pos = max(0, min(1, boll_pos))

    # 52-week position (use available history)
    high_52 = max(c[-252:]) if len(c) >= 252 else max(c)
    low_52  = min(c[-252:]) if len(c) >= 252 else min(c)
    rng = high_52 - low_52
    price_vs_52low  = (c[-1] - low_52)  / low_52  * 100 if low_52  > 0 else 0
    price_vs_52high = (high_52 - c[-1]) / high_52 * 100 if high_52 > 0 else 0

    # Momentum
    mom3d = (c[-1] - c[-4]) / c[-4] * 100 if len(c) >= 4 and c[-4] > 0 else 0
    mom5d = (c[-1] - c[-6]) / c[-6] * 100 if len(c) >= 6 and c[-6] > 0 else 0

    # ATR
    trs = [abs(c[i] - c[i-1]) for i in range(max(1, len(c)-14), len(c))]
    atr_pct = (sum(trs)/len(trs)) / c[-1] * 100 if trs and c[-1] > 0 else 0

    # Consecutive down days
    consec_down = 0
    for i in range(len(c)-2, max(len(c)-8, 0), -1):
        if c[i] < c[i-1]:
            consec_down += 1
        else:
            break

    return {
        "vol_surge_5d":    min(vol_surge_5,  50),
        "vol_surge_20d":   min(vol_surge_20, 50),
        "vol_acceleration":min(vol_accel, 10),
        "rsi":             rsi,
        "boll_squeeze":    boll_squeeze,
        "boll_pos":        boll_pos,
        "price_vs_52low":  min(price_vs_52low,  500),
        "price_vs_52high": min(price_vs_52high, 500),
        "mom3d":           mom3d,
        "mom5d":           mom5d,
        "atr_pct":         min(atr_pct, 30),
        "consec_down":     consec_down,
    }

# ── Model state ──
_model      = None
_model_lock = threading.Lock()
_model_status      = "untrained"
_model_trained_at  = None
_training_log      = []
_progress = {
    "current": 0,
    "total": 0,
    "ticker": "",
    "samples": 0,
    "positives": 0,
    "pct": 0.0,
    "phase": "idle",  # idle | fetching | training | done
}
MODEL_PATH = "momentum_model.pkl"

def get_status():
    return {
        "status":     _model_status,
        "trained_at": _model_trained_at,
        "log":        _training_log[-20:],
        "progress":   _progress.copy(),
    }

def _log(msg):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    _training_log.append(line)
    print(line)

def train_model_background():
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

        if os.path.exists(MODEL_PATH):
            age = time.time() - os.path.getmtime(MODEL_PATH)
            if age < 86400:
                _log("Loading cached model...")
                with open(MODEL_PATH, "rb") as f:
                    _model = pickle.load(f)
                _model_status     = "ready"
                _model_trained_at = datetime.now().isoformat()
                _log("Cached model loaded OK")
                return

        total = len(SCAN_UNIVERSE)
        _log(f"Training on {total} small/micro cap stocks...")
        _log("Target: 15%+ gain within 3 trading days")

        _progress.update({"total": total, "current": 0, "phase": "fetching",
                          "samples": 0, "positives": 0, "pct": 0.0, "ticker": ""})

        X, y = [], []
        trained_on  = 0
        TARGET_GAIN = 0.15
        FWD_DAYS    = 3

        for ticker in SCAN_UNIVERSE:
            try:
                _progress["ticker"] = ticker
                _progress["phase"]  = "fetching"

                hist = yf.Ticker(ticker).history(period="2y", interval="1d", auto_adjust=True)
                if hist is None or len(hist) < 30:
                    _progress["current"] += 1
                    continue

                closes  = hist["Close"].tolist()
                volumes = hist["Volume"].tolist()
                highs   = hist["High"].tolist()

                _progress["phase"] = "processing"
                for j in range(25, len(closes) - FWD_DAYS):
                    feats = extract_features(closes[:j+1], volumes[:j+1])
                    if feats is None:
                        continue
                    future_high  = max(highs[j+1:j+1+FWD_DAYS])
                    gain         = (future_high - closes[j]) / closes[j] if closes[j] > 0 else 0
                    label        = 1 if gain >= TARGET_GAIN else 0
                    X.append([feats[col] for col in FEATURE_COLS])
                    y.append(label)

                trained_on += 1
                pos = sum(y)
                pct = 100*pos/len(y) if y else 0
                _progress.update({
                    "current":   trained_on,
                    "samples":   len(X),
                    "positives": pos,
                    "pct":       round(pct, 1),
                    "ticker":    ticker,
                })
                if trained_on % 10 == 0:
                    _log(f"  {trained_on}/{total} — {len(X)} samples, {pos} positives ({pct:.1f}%)")
                time.sleep(0.15)

            except Exception as e:
                _log(f"  {ticker}: skipped ({e})")
                _progress["current"] = _progress.get("current", 0) + 1
                continue

        if len(X) < 50:
            _log("Not enough data to train")
            _model_status = "failed"
            return

        X = np.array(X)
        y = np.array(y)
        pos_rate = sum(y)/len(y)
        _progress["phase"] = "training"
        _log(f"Fitting on {len(X)} samples — {100*pos_rate:.1f}% positive rate")

        clf = _RandomForest(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )
        clf.fit(X, y)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(clf, f)

        _model            = clf
        _model_status     = "ready"
        _model_trained_at = datetime.now().isoformat()

        importances = sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1])
        _log("Feature importance: " + ", ".join(f"{k}={v:.3f}" for k,v in importances[:5]))
        _log(f"Done! Model ready.")

    except Exception as e:
        _log(f"Training failed: {e}")
        _model_status = "failed"
        import traceback; traceback.print_exc()


def predict(closes, volumes, opens=None):
    if _model is None or _model_status != "ready":
        return None
    feats = extract_features(closes, volumes)
    if feats is None:
        return None
    X      = np.array([[feats[col] for col in FEATURE_COLS]])
    prob   = _model.predict_proba(X)[0]
    classes = list(_model.classes_)
    if 1 in classes:
        return round(float(prob[classes.index(1)]), 4)
    return None
