"""
sheets_sync.py — Singer Scout Google Sheets integration

Syncs picks to/from Google Sheets automatically.
Sheet structure:
  Col A: ID
  Col B: Ticker
  Col C: Pick Date
  Col D: Entry Price
  Col E: ML Probability %
  Col F: Target Gain %
  Col G: Predicted Days
  Col H: Day 1 Price
  Col I: Day 2 Price
  Col J: Day 3 Price
  Col K: Day 4 Price
  Col L: Day 5 Price
  Col M: Outcome
  Col N: Outcome Day
  Col O: Actual Gain %
  Col P: Notes
"""

import os
import json
import time

SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "1f4FtUqsXuVlyptRxbsSSiouoVa7IEDM2Y3SNWSYc7ho")
CREDS_FILE = os.getenv("GOOGLE_CREDS_FILE", "singer-scout-creds.json")
SHEET_NAME = "Picks"

_sheets_service = None


def _get_service():
    global _sheets_service
    if _sheets_service:
        return _sheets_service
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]

        # Try file first, then environment variable (for Render)
        if os.path.exists(CREDS_FILE):
            creds = Credentials.from_service_account_file(CREDS_FILE, scopes=scopes)
        else:
            creds_json = os.getenv("GOOGLE_CREDS_JSON")
            if not creds_json:
                print("Sheets: no credentials found")
                return None
            creds_info = json.loads(creds_json)
            creds = Credentials.from_service_account_info(creds_info, scopes=scopes)

        _sheets_service = build("sheets", "v4", credentials=creds)
        print("Google Sheets connected OK")
        return _sheets_service

    except Exception as e:
        print(f"Sheets connection failed: {e}")
        return None


def _ensure_header(service):
    """Make sure the header row exists."""
    try:
        header = [[
            "ID", "Ticker", "Pick Date", "Entry Price $",
            "ML Prob %", "Target Gain %", "Predicted Days",
            "Day 1 $", "Day 2 $", "Day 3 $", "Day 4 $", "Day 5 $",
            "Outcome", "Outcome Day", "Actual Gain %", "Notes"
        ]]
        result = service.spreadsheets().values().get(
            spreadsheetId=SHEET_ID,
            range=f"{SHEET_NAME}!A1:P1"
        ).execute()
        existing = result.get("values", [])
        if not existing or existing[0][0] != "ID":
            service.spreadsheets().values().update(
                spreadsheetId=SHEET_ID,
                range=f"{SHEET_NAME}!A1",
                valueInputOption="RAW",
                body={"values": header}
            ).execute()
    except Exception as e:
        print(f"Header setup error: {e}")


def add_pick_to_sheet(pick_id, ticker, pick_date, entry_price,
                      ml_prob, predicted_gain, predicted_days):
    """Add a new pick row to Google Sheets."""
    service = _get_service()
    if not service:
        return False
    try:
        _ensure_header(service)
        row = [[
            pick_id, ticker, pick_date,
            round(entry_price, 2),
            round(ml_prob * 100, 1),
            predicted_gain, predicted_days,
            "", "", "", "", "",  # day prices empty
            "", "", "", ""       # outcome empty
        ]]
        service.spreadsheets().values().append(
            spreadsheetId=SHEET_ID,
            range=f"{SHEET_NAME}!A:P",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": row}
        ).execute()
        print(f"Sheets: added {ticker} pick {pick_id}")
        return True
    except Exception as e:
        print(f"Sheets add_pick error: {e}")
        return False


def update_pick_in_sheet(pick_id, day=None, price=None,
                         outcome=None, outcome_day=None, actual_gain=None):
    """Update a pick row in Google Sheets by ID."""
    service = _get_service()
    if not service:
        return False
    try:
        # Find the row with this pick_id
        result = service.spreadsheets().values().get(
            spreadsheetId=SHEET_ID,
            range=f"{SHEET_NAME}!A:A"
        ).execute()
        ids = result.get("values", [])
        row_num = None
        for i, row in enumerate(ids):
            if row and str(row[0]) == str(pick_id):
                row_num = i + 1  # 1-indexed
                break

        if not row_num:
            print(f"Sheets: pick {pick_id} not found")
            return False

        updates = []

        # Day price columns: H=8, I=9, J=10, K=11, L=12
        if day and price:
            col_letter = chr(ord('H') + day - 1)  # H=Day1, I=Day2...
            updates.append({
                "range": f"{SHEET_NAME}!{col_letter}{row_num}",
                "values": [[round(price, 2)]]
            })

        # Outcome columns: M=outcome, N=outcome_day, O=actual_gain
        if outcome:
            updates.append({
                "range": f"{SHEET_NAME}!M{row_num}:O{row_num}",
                "values": [[outcome, outcome_day or "", actual_gain or ""]]
            })

        if updates:
            service.spreadsheets().values().batchUpdate(
                spreadsheetId=SHEET_ID,
                body={"valueInputOption": "RAW", "data": updates}
            ).execute()
            print(f"Sheets: updated pick {pick_id} row {row_num}")

        return True

    except Exception as e:
        print(f"Sheets update error: {e}")
        return False


def sync_all_picks_to_sheet(picks):
    """Full sync — write all picks to sheet (clears and rewrites)."""
    service = _get_service()
    if not service:
        return False
    try:
        _ensure_header(service)

        rows = []
        for p in picks:
            rows.append([
                p.get("id", ""),
                p.get("ticker", ""),
                p.get("pick_date", ""),
                p.get("entry_price", ""),
                round((p.get("ml_prob") or 0) * 100, 1),
                p.get("predicted_gain_pct", 15),
                p.get("predicted_days", 3),
                p.get("day1_price") or "",
                p.get("day2_price") or "",
                p.get("day3_price") or "",
                p.get("day4_price") or "",
                p.get("day5_price") or "",
                p.get("outcome") or "",
                p.get("outcome_day") or "",
                p.get("actual_gain_pct") or "",
                p.get("notes") or "",
            ])

        if rows:
            service.spreadsheets().values().update(
                spreadsheetId=SHEET_ID,
                range=f"{SHEET_NAME}!A2:P{len(rows)+1}",
                valueInputOption="RAW",
                body={"values": rows}
            ).execute()
            print(f"Sheets: synced {len(rows)} picks")
        return True

    except Exception as e:
        print(f"Sheets full sync error: {e}")
        return False


def is_connected():
    return _get_service() is not None
