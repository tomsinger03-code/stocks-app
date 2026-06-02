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