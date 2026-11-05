from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app import tickets
from app.db import connect, init_db
from app.main import create_app

TICKET = {"id": "t-1", "subject": "Charged twice", "body": "Please refund one."}


class RecordingWorker:
    """Records hand-offs without classifying, so these tests only cover the API."""

    def __init__(self, queued):
        self.queued = queued

    async def start(self):
        pass

    async def stop(self):
        pass

    async def submit(self, ticket_id):
        self.queued.append(ticket_id)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def queued():
    """Ticket ids handed to classification."""
    return []


@pytest.fixture
def client(db_path, queued):
    # A clock that ticks once per call makes list order predictable.
    times = iter(range(1_000, 2_000))
    app = create_app(db_path, worker=RecordingWorker(queued), clock=lambda: float(next(times)))
    with TestClient(app) as test_client:
        yield test_client


def classify(db_path, ticket_id, category, priority):
    conn = connect(db_path)
    conn.execute(
        "UPDATE tickets SET status = 'classified', category = ?, priority = ?, summary = 'A summary.'"
        " WHERE id = ?",
        (category, priority, ticket_id),
    )
    conn.close()


def assert_error(response, status, code):
    assert response.status_code == status
    error = response.json()["error"]
    assert error["code"] == code
    assert error["message"]


# Ingest


def test_create_returns_pending_ticket_with_nothing_classified(client):
    response = client.post("/tickets", json=TICKET)
    assert response.status_code == 201
    ticket = response.json()
    assert ticket["id"] == "t-1"
    assert ticket["status"] == "pending"
    for field in ("category", "priority", "summary", "failure_reason"):
        assert ticket[field] is None


def test_new_ticket_is_handed_to_classification(client, queued):
    client.post("/tickets", json=TICKET)
    assert queued == ["t-1"]


def test_empty_or_missing_subject_is_accepted(client):
    assert client.post("/tickets", json={"id": "t-1008", "subject": "", "body": "asdf"}).status_code == 201
    response = client.post("/tickets", json={"id": "t-1009", "body": "asdf"})
    assert response.status_code == 201
    assert response.json()["subject"] == ""


@pytest.mark.parametrize(
    "payload",
    [
        {"subject": "x", "body": "x"},
        {"id": "t-1", "subject": "x"},
        {"id": "", "body": "x"},
        {"id": "a/b", "body": "x"},
        {"id": "x" * 129, "body": "x"},
        {"id": 1, "body": "x"},
        {"id": "t-1", "body": ""},
        {"id": "t-1", "body": " \n\t "},
        {"id": "t-1", "body": "x" * 20_001},
        {"id": "t-1", "subject": "x" * 501, "body": "x"},
    ],
    ids=["no-id", "no-body", "empty-id", "slash-in-id", "long-id", "numeric-id",
         "empty-body", "blank-body", "long-body", "long-subject"],
)
def test_invalid_ticket_is_rejected_and_not_stored(client, queued, payload):
    assert_error(client.post("/tickets", json=payload), 422, "invalid_request")
    assert client.get("/tickets").json()["total"] == 0
    assert queued == []


def test_malformed_json_uses_the_error_format(client):
    response = client.post("/tickets", content="{oops", headers={"content-type": "application/json"})
    assert_error(response, 422, "invalid_request")


def test_validation_error_does_not_echo_ticket_content(client):
    response = client.post("/tickets", json={"id": "bad/id", "body": "secret text"})
    assert "secret text" not in response.text


# Duplicates


def test_same_content_returns_existing_ticket_without_reclassifying(client, queued):
    first = client.post("/tickets", json=TICKET)
    second = client.post("/tickets", json=TICKET)
    assert second.status_code == 200
    assert second.json() == first.json()
    assert queued == ["t-1"]
    assert client.get("/tickets").json()["total"] == 1


@pytest.mark.parametrize("change", [{"subject": "Different"}, {"body": "Different"}])
def test_different_content_conflicts_and_keeps_the_original(client, queued, change):
    client.post("/tickets", json=TICKET)
    response = client.post("/tickets", json={**TICKET, **change})
    assert_error(response, 409, "ticket_conflict")
    stored = client.get("/tickets/t-1").json()
    assert (stored["subject"], stored["body"]) == (TICKET["subject"], TICKET["body"])
    assert queued == ["t-1"]


def test_duplicate_does_not_undo_a_finished_classification(client, db_path, queued):
    client.post("/tickets", json=TICKET)
    classify(db_path, "t-1", "billing", "low")
    response = client.post("/tickets", json=TICKET)
    assert response.status_code == 200
    assert response.json()["status"] == "classified"
    assert queued == ["t-1"]


def test_simultaneous_duplicates_create_exactly_one_ticket(db_path):
    setup = connect(db_path)
    init_db(setup)
    setup.close()

    def submit(_):
        conn = connect(db_path)
        try:
            return tickets.create_ticket(conn, "t-1", "s", "b", now=1.0)[1]
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(submit, range(32)))
    assert results.count(True) == 1


# Fetch


def test_fetch_returns_the_ticket(client):
    created = client.post("/tickets", json=TICKET).json()
    response = client.get("/tickets/t-1")
    assert response.status_code == 200
    assert response.json() == created


def test_fetch_unknown_ticket_is_404(client):
    assert_error(client.get("/tickets/nope"), 404, "not_found")


def test_unknown_route_uses_the_error_format(client):
    assert_error(client.get("/nope"), 404, "not_found")


def test_claim_details_are_not_exposed(client, db_path):
    client.post("/tickets", json=TICKET)
    conn = connect(db_path)
    conn.execute("UPDATE tickets SET claim_token = 'abc', claim_expires_at = 5.0")
    conn.close()
    assert "claim" not in client.get("/tickets/t-1").text
    assert "claim" not in client.get("/tickets").text


def test_tickets_survive_an_app_restart(client, db_path):
    client.post("/tickets", json=TICKET)
    with TestClient(create_app(db_path, worker=RecordingWorker([]))) as restarted:
        assert restarted.get("/tickets/t-1").status_code == 200


def test_unexpected_error_uses_the_error_format_and_hides_details(db_path, monkeypatch):
    def explode(conn, ticket_id):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(tickets, "get_ticket", explode)
    app = create_app(db_path, worker=RecordingWorker([]))
    with TestClient(app, raise_server_exceptions=False) as api:
        response = api.get("/tickets/t-1")
    assert_error(response, 500, "internal_error")
    assert "secret" not in response.text


# List


@pytest.fixture
def seeded(client, db_path):
    labels = {
        "a": ("billing", "low"),
        "b": ("billing", "high"),
        "c": ("technical", "high"),
        "d": None,
    }
    for ticket_id, label in labels.items():
        client.post("/tickets", json={"id": ticket_id, "body": "x"})
        if label:
            classify(db_path, ticket_id, *label)
    return client


def ids(response):
    assert response.status_code == 200
    return [item["id"] for item in response.json()["items"]]


@pytest.mark.parametrize(
    "query, expected",
    [
        ("", ["a", "b", "c", "d"]),
        ("?category=billing", ["a", "b"]),
        ("?priority=high", ["b", "c"]),
        ("?category=billing&priority=high", ["b"]),
        ("?category=account", []),
    ],
)
def test_list_filters(seeded, query, expected):
    assert ids(seeded.get(f"/tickets{query}")) == expected


def test_pagination_walks_every_ticket_once(seeded):
    first = seeded.get("/tickets?limit=3&offset=0").json()
    assert (first["total"], first["limit"], first["offset"]) == (4, 3, 0)
    assert [item["id"] for item in first["items"]] == ["a", "b", "c"]
    assert ids(seeded.get("/tickets?limit=3&offset=3")) == ["d"]
    assert ids(seeded.get("/tickets?limit=3&offset=6")) == []


def test_total_counts_all_matches_not_just_the_page(seeded):
    body = seeded.get("/tickets?category=billing&limit=1").json()
    assert body["total"] == 2
    assert len(body["items"]) == 1


@pytest.mark.parametrize(
    "query",
    ["category=sales", "priority=urgent", "limit=0", "limit=201", "offset=-1", "limit=abc"],
)
def test_invalid_list_parameters_are_rejected(client, query):
    assert_error(client.get(f"/tickets?{query}"), 422, "invalid_request")
