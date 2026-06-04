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

def _save_model_to_drive(model_data):
    """Save trained model to Google Drive for persistence across restarts."""
    try:
        import json
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseUpload
        import io

        creds_json = os.getenv("GOOGLE_CREDS_JSON")
        if not creds_json:
            return False

        creds = Credentials.from_service_account_info(
            json.loads(creds_json),
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        drive = build("drive", "v3", credentials=creds)

        # Pickle the model
        model_bytes = pickle.dumps(model_data)
        media = MediaIoBaseUpload(
            io.BytesIO(model_bytes),
            mimetype="application/octet-stream",
            resumable=False
        )

        # Check if file already exists
        results = drive.files().list(
            q="name='singer_scout_model.pkl' and trashed=false",
            fields="files(id, name)"
        ).execute()
        files = results.get("files", [])

        if files:
            # Update existing file
            drive.files().update(
                fileId=files[0]["id"],
                media_body=media
            ).execute()
        else:
            # Create new file
            drive.files().create(
                body={"name": "singer_scout_model.pkl"},
                media_body=media
            ).execute()

        print("Model saved to Google Drive")
        return True
    except Exception as e:
        print(f"Drive save failed: {e}")
        return False


def _load_model_from_drive():
    """Load trained model from Google Drive."""
    try:
        import json
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        import io

        creds_json = os.getenv("GOOGLE_CREDS_JSON")
        if not creds_json:
            return None

        creds = Credentials.from_service_account_info(
            json.loads(creds_json),
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        drive = build("drive", "v3", credentials=creds)

        # Find model file
        results = drive.files().list(
            q="name='singer_scout_model.pkl' and trashed=false",
            fields="files(id, name, modifiedTime)"
        ).execute()
        files = results.get("files", [])

        if not files:
            print("No saved model found in Drive")
            return None

        # Download
        file_id = files[0]["id"]
        request = drive.files().get_media(fileId=file_id)
        buffer = io.BytesIO()
        from googleapiclient.http import MediaIoBaseDownload
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        buffer.seek(0)
        model_data = pickle.loads(buffer.read())
        print(f"Model loaded from Google Drive (saved {files[0].get('modifiedTime', 'unknown')})")
        return model_data

    except Exception as e:
        print(f"Drive load failed: {e}")
        return None

_sklearn_loaded = False
_RandomForest = None

def _load_sklearn():
    global _sklearn_loaded, _RandomForest
    if not _sklearn_loaded:
        from sklearn.ensemble import RandomForestClassifier
        _RandomForest = RandomForestClassifier
        _sklearn_loaded = True

# ── Universe: curated 50 high-volatility small/micro caps ──
# These are proven movers — updated from real winner lists
SCAN_UNIVERSE = [
    # Proven big movers from real data
    "BJDX","LASE","DXST","STAK","RKTO","ZJYL","SWMR","FOFO","PUSA","ABTS",
    "SBFM","STI","VERU","FOXX","EDHL","TWAV","INDP","WXM","SPRC","LNZA",
    "LSE","ROLR","MNTS","BNAI","ASTI","CXAI","HCAT",
    # Crypto/high vol
    "MARA","RIOT","CLSK","HUT","BTBT","CIFR","IREN","WULF",
    # Biotech volatile
    "NVAX","OCGN","NKTR","FATE","CRSP","SIGA","ATOS",
    # Small cap growth
    "PLTR","HOOD","SOFI","AFRM","UPST","IONQ","QUBT","RGTI",
    # Chinese ADRs (very volatile)
    "NIO","XPEV","LI","JFU","HUDI","JFIN",
    # Recent additions
    "SMCI","WOLF","CELH","ACMR","BBAI","GFAI",
]

# Remove duplicates
SCAN_UNIVERSE = list(dict.fromkeys(SCAN_UNIVERSE))

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

# ── Model state — 3 models, one per timeframe ──
_models = {"day1": None, "day2": None, "day3": None}
_gain_models = {"day1": None, "day2": None, "day3": None}  # predict expected gain
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
# Use persistent disk if available (Render), otherwise local
MODEL_PATH = "/data/momentum_model.pkl" if os.path.exists("/data") else "momentum_model.pkl"

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

        # Try loading from Google Drive first (survives restarts)
        _log("Checking Google Drive for saved model...")
        drive_model = _load_model_from_drive()
        if drive_model and isinstance(drive_model, dict) and "models" in drive_model:
            _models.update(drive_model["models"])
            _gain_models.update(drive_model.get("gain_models", {}))
            _model_status     = "ready"
            _model_trained_at = datetime.now().isoformat()
            _log("Model loaded from Google Drive — no retraining needed!")
            return

        # Fall back to local cache
        if os.path.exists(MODEL_PATH):
            age = time.time() - os.path.getmtime(MODEL_PATH)
            if age < 86400:
                _log("Loading local cached model...")
                with open(MODEL_PATH, "rb") as f:
                    saved = pickle.load(f)
                if isinstance(saved, dict) and "models" in saved:
                    _models.update(saved["models"])
                    _gain_models.update(saved.get("gain_models", {}))
                else:
                    _log("Old model format - retraining...")
                    os.remove(MODEL_PATH)
                _model_status     = "ready"
                _model_trained_at = datetime.now().isoformat()
                _log("Local cached model loaded OK")
                return

        total = len(SCAN_UNIVERSE)
        _log(f"Training on {total} stocks — 3 separate day models...")
        _log("Predicts: most likely % gain for Day 1, Day 2, Day 3 separately")

        _progress.update({"total": total, "current": 0, "phase": "fetching",
                          "samples": 0, "positives": 0, "pct": 0.0, "ticker": ""})

        # Separate training data for each day
        X_all = []
        y_d1 = []  # actual % gain day 1
        y_d2 = []  # actual % gain day 2
        y_d3 = []  # actual % gain day 3
        trained_on = 0

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
                for j in range(25, len(closes) - 3):
                    feats = extract_features(closes[:j+1], volumes[:j+1])
                    if feats is None:
                        continue

                    base = closes[j]
                    if base <= 0:
                        continue

                    # Actual % gain using high of each day
                    g1 = (highs[j+1] - base) / base * 100
                    g2 = (max(highs[j+1:j+3]) - base) / base * 100
                    g3 = (max(highs[j+1:j+4]) - base) / base * 100

                    X_all.append([feats[col] for col in FEATURE_COLS])
                    y_d1.append(round(g1, 3))
                    y_d2.append(round(g2, 3))
                    y_d3.append(round(g3, 3))

                trained_on += 1
                pos = sum(1 for g in y_d3 if g >= 15)
                pct = 100*pos/len(y_d3) if y_d3 else 0
                _progress.update({
                    "current":   trained_on,
                    "samples":   len(X_all),
                    "positives": pos,
                    "pct":       round(pct, 1),
                    "ticker":    ticker,
                })
                if trained_on % 10 == 0:
                    _log(f"  {trained_on}/{total} — {len(X_all)} samples, {pos} with 15%+ by day3 ({pct:.1f}%)")
                time.sleep(0.15)

            except Exception as e:
                _log(f"  {ticker}: skipped ({e})")
                _progress["current"] = _progress.get("current", 0) + 1
                continue

        if len(X_all) < 50:
            _log("Not enough data to train")
            _model_status = "failed"
            return

        X = np.array(X_all)
        _progress["phase"] = "training"
        _log(f"Fitting 3 models on {len(X)} samples...")

        from sklearn.ensemble import GradientBoostingRegressor

        trained_models = {}
        trained_gain_models = {}

        for day_key, y_gains in [("day1", y_d1), ("day2", y_d2), ("day3", y_d3)]:
            # Classifier: will it gain 10%+?
            y_cls = np.array([1 if g >= 10 else 0 for g in y_gains])
            clf = _RandomForest(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            )
            clf.fit(X, y_cls)
            trained_models[day_key] = clf

            # Regressor: expected % gain (on positive examples only)
            pos_idx = [i for i,g in enumerate(y_gains) if g >= 5]
            if len(pos_idx) > 50:
                X_pos = X[pos_idx]
                y_pos = np.array([y_gains[i] for i in pos_idx])
                reg = GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, random_state=42
                )
                reg.fit(X_pos, y_pos)
                trained_gain_models[day_key] = reg
            
            pos_rate = sum(y_cls)/len(y_cls)
            _log(f"  {day_key}: {pos_rate*100:.1f}% hit 10%+")

        model_data = {"models": trained_models, "gain_models": trained_gain_models}

        # Save locally
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model_data, f)

        # Save to Google Drive for persistence across restarts
        _log("Saving model to Google Drive...")
        _save_model_to_drive(model_data)

        _models.update(trained_models)
        _gain_models.update(trained_gain_models)
        _model_status     = "ready"
        _model_trained_at = datetime.now().isoformat()
        _log(f"All 3 models ready — saved to Google Drive!")

    except Exception as e:
        _log(f"Training failed: {e}")
        _model_status = "failed"
        import traceback; traceback.print_exc()


def predict(closes, volumes, opens=None):
    """
    Returns dict with Day 1/2/3 predictions:
    {
        day1: {prob: 0.34, expectedGain: 8.2},
        day2: {prob: 0.58, expectedGain: 15.1},
        day3: {prob: 0.71, expectedGain: 22.4},
        bestDay: 3,
        bestProb: 0.71,
        bestGain: 22.4
    }
    """
    if _model_status != "ready" or not _models.get("day1"):
        return None
    feats = extract_features(closes, volumes)
    if feats is None:
        return None

    X = np.array([[feats[col] for col in FEATURE_COLS]])
    result = {}

    for day_key in ["day1", "day2", "day3"]:
        clf = _models.get(day_key)
        if clf is None:
            result[day_key] = {"prob": 0, "expectedGain": 0}
            continue

        classes = list(clf.classes_)
        proba = clf.predict_proba(X)[0]
        prob = float(proba[classes.index(1)]) if 1 in classes else 0

        # Expected gain from regressor
        reg = _gain_models.get(day_key)
        expected_gain = 0
        if reg and prob > 0.2:
            try:
                expected_gain = float(reg.predict(X)[0])
                expected_gain = max(0, round(expected_gain, 1))
            except:
                expected_gain = 0

        result[day_key] = {
            "prob": round(prob, 4),
            "probPct": round(prob * 100, 1),
            "expectedGain": expected_gain
        }

    # Find best day (highest probability)
    best = max(result.items(), key=lambda x: x[1]["prob"])
    result["bestDay"] = int(best[0].replace("day",""))
    result["bestProb"] = best[1]["prob"]
    result["bestProbPct"] = best[1]["probPct"]
    result["bestGain"] = best[1]["expectedGain"]

    # Legacy field for backwards compat
    result["mlProb"] = best[1]["prob"]
    result["mlPct"] = best[1]["probPct"]

    return result
