import sqlite3
import os
import json
import threading
from datetime import datetime

DB_LOCK = threading.Lock()

class AuditDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        # 10s timeout prevents database lock issues under concurrent access
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        with DB_LOCK:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        tx_id TEXT NOT NULL,
                        amount REAL,
                        calibrated_risk REAL,
                        prescribed_action TEXT,
                        confidence TEXT,
                        sop_id TEXT,
                        rationale TEXT,
                        executed_tool TEXT,
                        execution_status TEXT,
                        is_human_override INTEGER DEFAULT 0,
                        analyst_note TEXT
                    )
                """)
                conn.commit()

    def log_event(self, entry: dict):
        with DB_LOCK:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO audit_logs (
                        timestamp, tx_id, amount, calibrated_risk, prescribed_action,
                        confidence, sop_id, rationale, executed_tool, execution_status,
                        is_human_override, analyst_note
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.get("timestamp", datetime.now().isoformat()),
                    entry.get("tx_id", "anon_tx"),
                    float(entry.get("amount", 0.0)),
                    float(entry.get("calibrated_risk", 0.0)),
                    entry.get("prescribed_action", "UNKNOWN"),
                    entry.get("confidence", "low"),
                    entry.get("sop_id", "N/A"),
                    entry.get("rationale", ""),
                    entry.get("executed_tool", "none"),
                    entry.get("execution_status", "COMPLETED"),
                    1 if entry.get("is_human_override", False) else 0,
                    entry.get("analyst_note", "")
                ))
                conn.commit()

    def fetch_recent(self, limit: int = 50):
        with DB_LOCK:
            with self._get_connection() as conn:
                import pandas as pd
                return pd.read_sql_query(
                    "SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?",
                    conn,
                    params=(limit,)
                )