import sqlite3

import pytest

from app.db import connect, init_db


@pytest.fixture
def path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def conn(path):
    connection = connect(path)
    init_db(connection)
    yield connection
    connection.close()


def insert(conn, **overrides):
    row = {"id": "t-1", "body": "hello", "created_at": 1.0, "updated_at": 1.0, **overrides}
    columns = ", ".join(row)
    placeholders = ", ".join(f":{name}" for name in row)
    conn.execute(f"INSERT INTO tickets ({columns}) VALUES ({placeholders})", row)


CLASSIFIED = {"status": "classified", "category": "billing", "priority": "low", "summary": "A summary."}


def test_new_ticket_defaults_to_pending_with_nothing_set(conn):
    insert(conn)
    row = conn.execute("SELECT * FROM tickets WHERE id = 't-1'").fetchone()
    assert row["status"] == "pending"
    assert row["subject"] == ""
    assert row["attempts"] == 0
    for column in ("category", "priority", "summary", "failure_reason", "claim_token", "claim_expires_at"):
        assert row[column] is None


def test_init_db_is_safe_to_rerun_and_keeps_data(conn):
    insert(conn)
    init_db(conn)
    assert conn.execute("SELECT count(*) FROM tickets").fetchone()[0] == 1


def test_tickets_survive_reconnecting(conn, path):
    insert(conn)
    conn.close()
    reopened = connect(path)
    assert reopened.execute("SELECT id FROM tickets").fetchone()["id"] == "t-1"
    reopened.close()


def test_duplicate_id_is_rejected(conn):
    insert(conn)
    with pytest.raises(sqlite3.IntegrityError):
        insert(conn)


@pytest.mark.parametrize(
    "valid",
    [
        CLASSIFIED,
        {"status": "failed", "failure_reason": "gave up"},
        {"claim_token": "abc", "claim_expires_at": 2.0},
    ],
    ids=["classified", "failed", "pending-with-claim"],
)
def test_valid_states_are_accepted(conn, valid):
    insert(conn, **valid)


@pytest.mark.parametrize(
    "invalid",
    [
        {"id": "  "},
        {"body": "  "},
        {"status": "processing"},
        {**CLASSIFIED, "category": "sales"},
        {**CLASSIFIED, "priority": "urgent"},
        {**CLASSIFIED, "category": None},
        {**CLASSIFIED, "priority": None},
        {**CLASSIFIED, "summary": None},
        {**CLASSIFIED, "summary": "  "},
        {**CLASSIFIED, "failure_reason": "gave up"},
        {**CLASSIFIED, "claim_token": "abc", "claim_expires_at": 2.0},
        {"category": "billing"},
        {"failure_reason": "gave up"},
        {"status": "failed"},
        {"status": "failed", "failure_reason": "gave up", "category": "billing"},
        {"status": "failed", "failure_reason": "gave up", "claim_token": "abc", "claim_expires_at": 2.0},
        {"claim_token": "abc"},
        {"claim_expires_at": 2.0},
        {"attempts": -1},
    ],
    ids=lambda case: ",".join(f"{k}={v}" for k, v in case.items()),
)
def test_invalid_states_are_rejected(conn, invalid):
    with pytest.raises(sqlite3.IntegrityError):
        insert(conn, **invalid)
