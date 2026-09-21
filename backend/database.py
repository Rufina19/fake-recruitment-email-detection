import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "analysis_reports.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_id TEXT,
            subject TEXT,
            prediction TEXT,
            risk_level TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def insert_analysis(subject: str, prediction: str, risk_level: str):
    conn = get_connection()
    cur = conn.cursor()
    email_id = f"EML-{int(datetime.utcnow().timestamp())}"
    created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        """
        INSERT INTO analysis_reports (email_id, subject, prediction, risk_level, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (email_id, subject[:255], prediction, risk_level, created_at),
    )
    conn.commit()
    conn.close()


def fetch_stats():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM analysis_reports")
    total = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*) FROM analysis_reports WHERE LOWER(prediction) = 'fake'"
    )
    fraud = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*) FROM analysis_reports WHERE LOWER(prediction) = 'genuine'"
    )
    genuine = cur.fetchone()[0]

    conn.close()
    return {
        "total_emails": total,
        "fraud_detected": fraud,
        "genuine_emails": genuine,
    }


def fetch_recent_activity(limit: int = 10):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT email_id, subject, prediction, risk_level, created_at
        FROM analysis_reports
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def fetch_reports(search_term: str = "", risk_level: str = "", prediction: str = ""):
    conn = get_connection()
    cur = conn.cursor()
    query = """
        SELECT email_id, subject, prediction, risk_level, created_at
        FROM analysis_reports
        WHERE 1=1
    """
    params = []

    if search_term:
        query += " AND (LOWER(subject) LIKE ? OR LOWER(email_id) LIKE ?)"
        like = f"%{search_term.lower()}%"
        params.extend([like, like])

    if risk_level:
        query += " AND LOWER(risk_level) = ?"
        params.append(risk_level.lower())

    if prediction:
        query += " AND LOWER(prediction) = ?"
        params.append(prediction.lower())

    query += " ORDER BY created_at DESC"

    cur.execute(query, params)
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows

