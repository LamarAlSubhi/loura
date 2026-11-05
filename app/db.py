"""SQLite connection and schema for tickets."""

import sqlite3
from pathlib import Path

# No "processing" status: a ticket being worked on is `pending` with a claim, so a
# crash needs no cleanup. The CHECKs back up the Python validation.
SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id               TEXT PRIMARY KEY CHECK (length(trim(id)) > 0),
    subject          TEXT NOT NULL DEFAULT '',
    body             TEXT NOT NULL CHECK (length(trim(body)) > 0),

    status           TEXT NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'classified', 'failed')),

    category         TEXT CHECK (category IN ('billing', 'technical', 'account', 'other')),
    priority         TEXT CHECK (priority IN ('low', 'medium', 'high')),
    summary          TEXT CHECK (length(trim(summary)) > 0),
    failure_reason   TEXT CHECK (length(trim(failure_reason)) > 0),

    attempts         INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    claim_token      TEXT,
    claim_expires_at REAL,

    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,

    CHECK (
        (status = 'classified'
            AND category IS NOT NULL AND priority IS NOT NULL AND summary IS NOT NULL)
        OR (status <> 'classified'
            AND category IS NULL AND priority IS NULL AND summary IS NULL)
    ),
    CHECK ((status = 'failed') = (failure_reason IS NOT NULL)),
    
    CHECK ((claim_token IS NULL) = (claim_expires_at IS NULL)),
    CHECK (status = 'pending' OR claim_token IS NULL)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_tickets_created ON tickets (created_at, id);
CREATE INDEX IF NOT EXISTS idx_tickets_category ON tickets (category, created_at, id);
CREATE INDEX IF NOT EXISTS idx_tickets_priority ON tickets (priority, created_at, id);
CREATE INDEX IF NOT EXISTS idx_tickets_claimable ON tickets (status, claim_expires_at);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=5.0, autocommit=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
