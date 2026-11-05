"""The sample tickets load, and the awkward ones behave as the README describes."""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.classifier import build_prompt
from app.db import connect, init_db
from app.llm import FakeLLMClient
from app.main import create_app
from app.tickets import create_ticket, get_ticket
from app.worker import Worker

SAMPLES = json.loads((Path(__file__).parent.parent / "data" / "sample_tickets.json").read_text())
BY_ID = {ticket["id"]: ticket for ticket in SAMPLES}


class NoWorker:
    async def start(self):
        pass

    async def stop(self):
        pass

    async def submit(self, ticket_id):
        pass


def test_all_ten_samples_are_present():
    assert list(BY_ID) == [f"t-{n}" for n in range(1001, 1011)]


def test_every_sample_loads_through_the_api(tmp_path):
    with TestClient(create_app(tmp_path / "test.db", worker=NoWorker())) as api:
        for ticket in SAMPLES:
            assert api.post("/tickets", json=ticket).status_code == 201, ticket["id"]
        assert api.get("/tickets").json()["total"] == 10
        for ticket in SAMPLES:
            assert api.post("/tickets", json=ticket).status_code == 200, ticket["id"]
        assert api.get("/tickets").json()["total"] == 10


def test_t1008_keeps_its_empty_subject(tmp_path):
    with TestClient(create_app(tmp_path / "test.db", worker=NoWorker())) as api:
        assert api.post("/tickets", json=BY_ID["t-1008"]).json()["subject"] == ""


@pytest.mark.parametrize("ticket_id", ["t-1005", "t-1010"])
def test_awkward_text_stays_inside_the_ticket_data(ticket_id):
    system, user = build_prompt(BY_ID[ticket_id]["subject"], BY_ID[ticket_id]["body"])
    assert "Ignore all previous" not in system
    assert "we have fixed" not in system
    assert json.loads(user.removeprefix("<ticket>\n").removesuffix("\n</ticket>")) == {
        "subject": BY_ID[ticket_id]["subject"],
        "body": BY_ID[ticket_id]["body"],
    }


# These labels come from the fake's keyword rules, so this only guards the full path
# from sample to stored row. It says nothing about how a real model would do.
EXPECTED_FROM_FAKE = {
    "t-1001": ("billing", "medium"),
    "t-1002": ("account", "medium"),
    "t-1003": ("technical", "high"),
    "t-1004": ("account", "medium"),
    "t-1005": ("billing", "medium"),  # the demanded technical/high is not followed
    "t-1006": ("other", "low"),
    "t-1007": ("billing", "medium"),
    "t-1008": ("other", "low"),
    "t-1009": ("technical", "medium"),  # the export, not the aside about overcharging
    "t-1010": ("technical", "high"),  # the quoted "we have fixed" doesn't count
}


def test_samples_run_through_the_worker_with_the_fake_model(tmp_path):
    path = tmp_path / "test.db"
    conn = connect(path)
    init_db(conn)
    for ticket in SAMPLES:
        create_ticket(conn, ticket["id"], ticket["subject"], ticket["body"], now=1.0)

    async def main():
        worker = Worker(path, FakeLLMClient(failure_rate=0.0, seed=1, delay=0))
        await asyncio.gather(*(worker.process(ticket["id"]) for ticket in SAMPLES))

    asyncio.run(main())
    for ticket_id, expected in EXPECTED_FROM_FAKE.items():
        row = get_ticket(conn, ticket_id)
        assert (row["category"], row["priority"]) == expected, ticket_id
        assert row["status"] == "classified"
        assert "approved" not in row["summary"].lower()
    conn.close()
