from datetime import datetime
import sqlite3

DB_NAME = "momentum.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS stock_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        price REAL,
        score INTEGER,

        rsi REAL,
        macd REAL,
        bollinger REAL,
        volume_surge REAL,
        momentum5d REAL,

        lookup_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def save_stock(
    ticker,
    price,
    score,
    rsi=None,
    macd=None,
    bollinger=None,
    volume_surge=None,
    momentum5d=None
):
    conn = sqlite3.connect(DB_NAME)

    conn.execute(
        """
        INSERT INTO stock_history
        (
            ticker,
            price,
            score,
            rsi,
            macd,
            bollinger,
            volume_surge,
            momentum5d
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ticker,
            price,
            score,
            rsi,
            macd,
            bollinger,
            volume_surge,
            momentum5d
        )
    )

    conn.commit()
    conn.close()

def init_picks_table():
    conn = sqlite3.connect(DB_NAME)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS picks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        pick_date TEXT NOT NULL,
        entry_price REAL,
        ml_prob REAL,
        predicted_gain_pct REAL,
        predicted_days INTEGER,
        day1_price REAL,
        day2_price REAL,
        day3_price REAL,
        day4_price REAL,
        day5_price REAL,
        outcome TEXT,
        outcome_day INTEGER,
        actual_gain_pct REAL,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()


def save_pick(ticker, entry_price, ml_prob, predicted_gain_pct=15, predicted_days=3):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.execute(
        """INSERT INTO picks (ticker, pick_date, entry_price, ml_prob,
           predicted_gain_pct, predicted_days)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (ticker, datetime.now().strftime('%Y-%m-%d'),
         entry_price, ml_prob, predicted_gain_pct, predicted_days)
    )
    pick_id = cur.lastrowid
    conn.commit()
    conn.close()
    return pick_id


def update_pick_price(pick_id, day, price):
    col = f"day{day}_price"
    conn = sqlite3.connect(DB_NAME)
    conn.execute(f"UPDATE picks SET {col}=? WHERE id=?", (price, pick_id))
    conn.commit()
    conn.close()


def update_pick_outcome(pick_id, outcome, outcome_day, actual_gain_pct):
    conn = sqlite3.connect(DB_NAME)
    conn.execute(
        """UPDATE picks SET outcome=?, outcome_day=?, actual_gain_pct=?
           WHERE id=?""",
        (outcome, outcome_day, actual_gain_pct, pick_id)
    )
    conn.commit()
    conn.close()


def get_all_picks():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM picks ORDER BY created_at DESC LIMIT 100"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_open_picks():
    """Picks with no outcome yet — need price checking."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT * FROM picks WHERE outcome IS NULL
           ORDER BY created_at DESC""",
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
