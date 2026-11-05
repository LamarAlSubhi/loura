"""All SQL for reading and writing tickets."""

import sqlite3

from app.models import Classification


class TicketConflict(Exception):
    """The id already exists with a different subject or body."""


def create_ticket(
    conn: sqlite3.Connection, ticket_id: str, subject: str, body: str, now: float
) -> tuple[sqlite3.Row, bool]:
    """Returns the stored row and whether this call created it."""
    cursor = conn.execute(
        """
        INSERT INTO tickets (id, subject, body, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (id) DO NOTHING
        """,
        (ticket_id, subject, body, now, now),
    )
    row = get_ticket(conn, ticket_id)
    created = cursor.rowcount == 1
    if not created and (row["subject"], row["body"]) != (subject, body):
        raise TicketConflict(ticket_id)
    return row, created


def get_ticket(conn: sqlite3.Connection, ticket_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()


def list_tickets(
    conn: sqlite3.Connection,
    category: str | None,
    priority: str | None,
    limit: int,
    offset: int,
) -> tuple[list[sqlite3.Row], int]:
    conditions = []
    params = []
    if category is not None:
        conditions.append("category = ?")
        params.append(category)
    if priority is not None:
        conditions.append("priority = ?")
        params.append(priority)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    total = conn.execute(f"SELECT count(*) FROM tickets {where}", params).fetchone()[0]
    rows = conn.execute(
        f"SELECT * FROM tickets {where} ORDER BY created_at, id LIMIT ? OFFSET ?",
        [*params, limit, offset],
    ).fetchall()
    return rows, total


def claimable_ticket_ids(conn: sqlite3.Connection, now: float, limit: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT id FROM tickets
        WHERE status = 'pending' AND (claim_expires_at IS NULL OR claim_expires_at <= ?)
        ORDER BY created_at, id
        LIMIT ?
        """,
        (now, limit),
    ).fetchall()
    return [row["id"] for row in rows]


# fetchall() on RETURNING matters: until the statement finishes it holds the write lock.
def claim_ticket(
    conn: sqlite3.Connection, ticket_id: str, token: str, now: float, expires_at: float
) -> sqlite3.Row | None:
    rows = conn.execute(
        """
        UPDATE tickets
        SET claim_token = ?, claim_expires_at = ?, updated_at = ?
        WHERE id = ? AND status = 'pending'
          AND (claim_expires_at IS NULL OR claim_expires_at <= ?)
        RETURNING *
        """,
        (token, expires_at, now, ticket_id, now),
    ).fetchall()
    return rows[0] if rows else None


def start_attempt(
    conn: sqlite3.Connection, ticket_id: str, token: str, now: float, expires_at: float
) -> int | None:
    """Returns the attempt number, or None if the claim was lost."""
    rows = conn.execute(
        """
        UPDATE tickets
        SET attempts = attempts + 1, claim_expires_at = ?, updated_at = ?
        WHERE id = ? AND claim_token = ? AND status = 'pending'
        RETURNING attempts
        """,
        (expires_at, now, ticket_id, token),
    ).fetchall()
    return rows[0]["attempts"] if rows else None


def save_classification(
    conn: sqlite3.Connection, ticket_id: str, token: str, result: Classification, now: float
) -> bool:
    cursor = conn.execute(
        """
        UPDATE tickets
        SET status = 'classified', category = ?, priority = ?, summary = ?,
            claim_token = NULL, claim_expires_at = NULL, updated_at = ?
        WHERE id = ? AND claim_token = ? AND status = 'pending'
        """,
        (result.category, result.priority, result.summary, now, ticket_id, token),
    )
    return cursor.rowcount == 1


def mark_failed(
    conn: sqlite3.Connection, ticket_id: str, token: str, reason: str, now: float
) -> bool:
    cursor = conn.execute(
        """
        UPDATE tickets
        SET status = 'failed', failure_reason = ?,
            claim_token = NULL, claim_expires_at = NULL, updated_at = ?
        WHERE id = ? AND claim_token = ? AND status = 'pending'
        """,
        (reason, now, ticket_id, token),
    )
    return cursor.rowcount == 1


def release_claim(
    conn: sqlite3.Connection, ticket_id: str, token: str, refund_attempt: bool, now: float
) -> bool:
    cursor = conn.execute(
        """
        UPDATE tickets
        SET claim_token = NULL, claim_expires_at = NULL, attempts = attempts - ?, updated_at = ?
        WHERE id = ? AND claim_token = ? AND status = 'pending'
        """,
        (1 if refund_attempt else 0, now, ticket_id, token),
    )
    return cursor.rowcount == 1
