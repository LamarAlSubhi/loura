import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from app import tickets
from app.db import connect, init_db
from app.llm import BROKEN_RESPONSES
from app.main import create_app
from app.worker import Settings, Worker
from tests.helpers import ScriptedLLMClient

TTL = 60.0


def output(category="billing", priority="low", summary="Customer was charged twice."):
    return json.dumps({"category": category, "priority": priority, "summary": summary})


class Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


class Harness:
    def __init__(self, path):
        self.path = path
        self.clock = Clock()
        self.sleeps = []
        conn = connect(path)
        init_db(conn)
        conn.close()

    def add(self, *ids):
        conn = connect(self.path)
        for ticket_id in ids:
            tickets.create_ticket(conn, ticket_id, "subject", "body", now=self.clock())
        conn.close()

    def read(self, ticket_id):
        conn = connect(self.path)
        row = dict(tickets.get_ticket(conn, ticket_id))
        conn.close()
        return row

    def sql(self, statement, *params):
        conn = connect(self.path)
        conn.execute(statement, params)
        conn.close()

    def worker(self, client, **settings):
        async def sleep(seconds):
            self.sleeps.append(seconds)

        return Worker(self.path, client, Settings(**settings), clock=self.clock, sleep=sleep)

    def crash_while_holding(self, ticket_id, attempts=1):
        """Leave a ticket as a process that died mid-attempt would."""
        self.sql(
            "UPDATE tickets SET claim_token = 'dead', claim_expires_at = ?, attempts = ? WHERE id = ?",
            self.clock() + TTL, attempts, ticket_id,
        )


@pytest.fixture
def h(tmp_path):
    return Harness(tmp_path / "test.db")


# Lifecycle


def test_success_moves_to_classified_and_clears_the_claim(h):
    h.add("t-1")
    client = ScriptedLLMClient([output()])
    asyncio.run(h.worker(client).process("t-1"))
    row = h.read("t-1")
    assert row["status"] == "classified"
    assert (row["category"], row["priority"]) == ("billing", "low")
    assert row["summary"] == "Customer was charged twice."
    assert row["attempts"] == 1
    assert row["failure_reason"] is None
    assert row["claim_token"] is None and row["claim_expires_at"] is None


@pytest.mark.parametrize("status_change", ["classified", "failed"])
def test_finished_tickets_are_never_processed_again(h, status_change):
    h.add("t-1")
    first = ScriptedLLMClient([output()] if status_change == "classified" else ["bad"] * 3)
    asyncio.run(h.worker(first).process("t-1"))
    before = h.read("t-1")
    assert before["status"] == status_change

    async def again():
        worker = h.worker(second)
        await worker.process("t-1")
        await worker.sweep()
        await asyncio.gather(*worker._tasks)

    second = ScriptedLLMClient([output("technical")])
    asyncio.run(again())
    assert second.calls == []
    assert h.read("t-1") == before


# Retries


def test_retries_then_succeeds(h):
    h.add("t-1")
    client = ScriptedLLMClient([TimeoutError(), "not json", output()])
    asyncio.run(h.worker(client).process("t-1"))
    row = h.read("t-1")
    assert row["status"] == "classified"
    assert row["attempts"] == 3
    assert h.sleeps == [1.0, 2.0]


def test_retries_run_out_and_the_ticket_fails_with_the_last_reason(h):
    h.add("t-1")
    client = ScriptedLLMClient([TimeoutError(), TimeoutError(), "not json", output()])
    asyncio.run(h.worker(client).process("t-1"))
    row = h.read("t-1")
    assert row["status"] == "failed"
    assert row["attempts"] == 3
    assert len(client.calls) == 3
    assert row["failure_reason"] == "gave up after 3 attempts: invalid model output: output: json_invalid"
    assert row["claim_token"] is None
    assert h.sleeps == [1.0, 2.0]


text_responses = {k: v for k, v in BROKEN_RESPONSES.items() if isinstance(v, str)}


@pytest.mark.parametrize("broken", text_responses.values(), ids=text_responses.keys())
def test_invalid_output_never_reaches_the_data_store(h, broken):
    h.add("t-1")
    asyncio.run(h.worker(ScriptedLLMClient([broken] * 3)).process("t-1"))
    row = h.read("t-1")
    assert row["status"] == "failed"
    assert row["category"] is None and row["priority"] is None and row["summary"] is None
    assert "charged twice" not in row["failure_reason"].lower()


# Concurrency


def test_concurrency_limit_is_respected(h):
    ids = [f"t-{n}" for n in range(10)]
    h.add(*ids)

    class Tracking:
        running = 0
        peak = 0

        async def complete(self, system, user):
            Tracking.running += 1
            Tracking.peak = max(Tracking.peak, Tracking.running)
            await asyncio.sleep(0.01)
            Tracking.running -= 1
            return output()

    async def main():
        worker = h.worker(Tracking(), concurrency=3)
        await asyncio.gather(*(worker.process(i) for i in ids))

    asyncio.run(main())
    assert Tracking.peak == 3
    assert all(h.read(i)["status"] == "classified" for i in ids)


def test_sweep_starts_no_more_than_the_limit_and_finishes_the_backlog(h):
    ids = [f"t-{n}" for n in range(5)]
    h.add(*ids)
    client = ScriptedLLMClient([output()] * 5)

    async def main():
        worker = h.worker(client, concurrency=2)
        await worker.sweep()
        assert len(worker._tasks) == 2
        await asyncio.gather(*worker._tasks)
        await worker.sweep()
        await asyncio.gather(*worker._tasks)
        await worker.sweep()
        await asyncio.gather(*worker._tasks)

    asyncio.run(main())
    assert len(client.calls) == 5
    assert all(h.read(i)["status"] == "classified" for i in ids)


# No double-processing


def test_same_worker_handed_a_ticket_twice_calls_the_model_once(h):
    h.add("t-1")
    client = ScriptedLLMClient([output(), output()])

    async def main():
        worker = h.worker(client)
        await asyncio.gather(worker.process("t-1"), worker.process("t-1"))

    asyncio.run(main())
    assert len(client.calls) == 1


def test_competing_workers_call_the_model_once(h):
    """Separate workers share only the database, like separate processes would."""
    h.add("t-1")
    client = ScriptedLLMClient([output()] * 8)

    async def main():
        workers = [h.worker(client) for _ in range(8)]
        await asyncio.gather(*(w.process("t-1") for w in workers))

    asyncio.run(main())
    assert len(client.calls) == 1
    assert h.read("t-1")["attempts"] == 1


def test_new_ticket_and_sweeper_racing_call_the_model_once(h):
    h.add("t-1")
    client = ScriptedLLMClient([output(), output()])

    async def main():
        worker = h.worker(client)
        await asyncio.gather(worker.process("t-1"), worker.sweep())
        await asyncio.gather(*worker._tasks)

    asyncio.run(main())
    assert len(client.calls) == 1


class Gated:
    """Answers only when released, to act as a slow model."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, system, user):
        self.started.set()
        await self.release.wait()
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def takeover(h, slow_outcome):
    """A slow worker loses its claim to a second worker, which finishes first."""

    async def main():
        slow = Gated(slow_outcome)
        task = asyncio.create_task(h.worker(slow).process("t-1"))
        await slow.started.wait()
        h.clock.now += TTL
        await h.worker(ScriptedLLMClient([output("technical", "high")])).process("t-1")
        slow.release.set()
        await task

    asyncio.run(main())


def test_result_from_a_worker_that_lost_its_claim_is_discarded(h):
    h.add("t-1")
    takeover(h, output("billing", "low"))
    row = h.read("t-1")
    assert (row["category"], row["priority"]) == ("technical", "high")
    assert row["attempts"] == 2


def test_failure_from_a_worker_that_lost_its_claim_does_not_overwrite_the_result(h):
    h.add("t-1")
    takeover(h, TimeoutError())
    row = h.read("t-1")
    assert row["status"] == "classified"
    assert row["failure_reason"] is None
    assert row["attempts"] == 2


def test_second_worker_stays_away_from_a_live_claim(h):
    h.add("t-1")
    second = ScriptedLLMClient([output("technical")])

    async def main():
        slow = Gated(output())
        task = asyncio.create_task(h.worker(slow).process("t-1"))
        await slow.started.wait()
        await h.worker(second).process("t-1")
        slow.release.set()
        await task

    asyncio.run(main())
    assert second.calls == []
    row = h.read("t-1")
    assert row["category"] == "billing"
    assert row["attempts"] == 1


@pytest.mark.parametrize("old_outcome", [output("billing"), TimeoutError()], ids=["result", "failure"])
def test_old_worker_cannot_touch_a_ticket_the_new_worker_still_holds(h, old_outcome):
    """The token is what protects the ticket here, since its status is still pending."""
    h.add("t-1")

    async def main():
        old = Gated(old_outcome)
        old_task = asyncio.create_task(h.worker(old).process("t-1"))
        await old.started.wait()
        h.clock.now += TTL

        new = Gated(output("technical", "high"))
        new_task = asyncio.create_task(h.worker(new).process("t-1"))
        await new.started.wait()

        old.release.set()
        await old_task
        row = h.read("t-1")
        assert row["status"] == "pending"
        assert row["category"] is None and row["failure_reason"] is None

        new.release.set()
        await new_task

    asyncio.run(main())
    row = h.read("t-1")
    assert (row["status"], row["category"]) == ("classified", "technical")
    assert row["attempts"] == 2


def test_old_worker_on_its_last_attempt_cannot_fail_a_ticket_the_new_worker_holds(h):
    h.add("t-1")

    async def main():
        old = Gated(TimeoutError())
        old_task = asyncio.create_task(h.worker(old, max_attempts=1).process("t-1"))
        await old.started.wait()
        h.clock.now += TTL

        new = Gated(output("technical", "high"))
        new_task = asyncio.create_task(h.worker(new).process("t-1"))
        await new.started.wait()

        old.release.set()
        await old_task
        assert h.read("t-1")["status"] == "pending"

        new.release.set()
        await new_task

    asyncio.run(main())
    row = h.read("t-1")
    assert row["status"] == "classified"
    assert row["failure_reason"] is None


def test_worker_that_lost_its_claim_during_backoff_stops_retrying(h):
    h.add("t-1")

    async def main():
        client = ScriptedLLMClient([TimeoutError(), output()])
        worker = h.worker(client)

        async def lose_claim(seconds):
            h.sql("UPDATE tickets SET claim_token = 'other' WHERE id = 't-1'")

        worker._sleep = lose_claim
        await worker.process("t-1")
        return client

    client = asyncio.run(main())
    assert len(client.calls) == 1
    assert h.read("t-1")["status"] == "pending"


# Crash recovery


def sweep_once(h, client):
    async def main():
        worker = h.worker(client)
        await worker.sweep()
        await asyncio.gather(*worker._tasks)

    asyncio.run(main())


def test_sweep_picks_up_a_ticket_that_was_never_started(h):
    h.add("t-1")
    sweep_once(h, ScriptedLLMClient([output()]))
    assert h.read("t-1")["status"] == "classified"


def test_live_claim_is_left_alone(h):
    h.add("t-1")
    h.crash_while_holding("t-1")
    client = ScriptedLLMClient([output()])
    h.clock.now += TTL - 1
    sweep_once(h, client)
    assert client.calls == []
    assert h.read("t-1")["status"] == "pending"


def test_expired_claim_is_recovered(h):
    h.add("t-1")
    h.crash_while_holding("t-1")
    h.clock.now += TTL
    sweep_once(h, ScriptedLLMClient([output()]))
    row = h.read("t-1")
    assert row["status"] == "classified"
    assert row["attempts"] == 2


def test_attempts_survive_a_crash(h):
    h.add("t-1")
    h.crash_while_holding("t-1", attempts=2)
    h.clock.now += TTL
    client = ScriptedLLMClient(["bad", output()])
    sweep_once(h, client)
    row = h.read("t-1")
    assert row["status"] == "failed"
    assert row["attempts"] == 3
    assert len(client.calls) == 1


def test_ticket_that_keeps_crashing_the_worker_ends_up_failed(h):
    h.add("t-1")
    h.crash_while_holding("t-1", attempts=3)
    h.clock.now += TTL
    client = ScriptedLLMClient([output()])
    sweep_once(h, client)
    row = h.read("t-1")
    assert row["status"] == "failed"
    assert "interrupted" in row["failure_reason"]
    assert client.calls == []


def test_unexpected_error_leaves_the_ticket_recoverable(h):
    h.add("t-1")

    async def main():
        worker = h.worker(ScriptedLLMClient([output()]))
        original = worker._run_attempts

        async def explode(ticket, token):
            raise RuntimeError("bug")

        worker._run_attempts = explode
        await worker.process("t-1")
        assert h.read("t-1")["claim_token"] is not None
        h.clock.now += TTL
        worker._run_attempts = original
        await worker.process("t-1")

    asyncio.run(main())
    assert h.read("t-1")["status"] == "classified"


# Graceful shutdown


def test_submit_returns_before_the_work_finishes(h):
    h.add("t-1")

    async def main():
        slow = Gated(output())
        worker = h.worker(slow)
        await worker.submit("t-1")
        await slow.started.wait()
        assert h.read("t-1")["status"] == "pending"
        slow.release.set()
        await asyncio.gather(*worker._tasks)

    asyncio.run(main())
    assert h.read("t-1")["status"] == "classified"


def test_stop_lets_running_work_finish_within_the_grace_period(h):
    h.add("t-1")

    async def main():
        slow = Gated(output())
        worker = h.worker(slow, shutdown_grace=5.0)
        await worker.submit("t-1")
        await slow.started.wait()
        stopping = asyncio.create_task(worker.stop())
        await asyncio.sleep(0.01)
        assert h.read("t-1")["status"] == "pending"
        slow.release.set()
        await stopping

    asyncio.run(main())
    row = h.read("t-1")
    assert row["status"] == "classified"
    assert row["attempts"] == 1


def test_work_past_the_grace_period_is_released_and_redone_exactly_once(h):
    h.add("t-1")
    never_answers = None

    async def first_run():
        nonlocal never_answers
        never_answers = Gated(output("technical"))
        worker = h.worker(never_answers, shutdown_grace=0.01)
        await worker.submit("t-1")
        await never_answers.started.wait()
        await worker.stop()

    asyncio.run(first_run())
    row = h.read("t-1")
    assert row["status"] == "pending"
    assert row["claim_token"] is None and row["claim_expires_at"] is None
    assert row["attempts"] == 0

    # No waiting for an expiry, and the same clock: the next run can take it right away.
    client = ScriptedLLMClient([output()])
    sweep_once(h, client)
    row = h.read("t-1")
    assert len(client.calls) == 1
    assert (row["status"], row["category"], row["attempts"]) == ("classified", "billing", 1)


def test_shutdown_during_backoff_releases_the_ticket_and_keeps_the_attempt(h):
    h.add("t-1")
    client = ScriptedLLMClient([TimeoutError(), output()])

    async def main():
        worker = h.worker(client)

        async def shutdown_begins(seconds):
            worker._stopping = True

        worker._sleep = shutdown_begins
        await worker.process("t-1")

    asyncio.run(main())
    row = h.read("t-1")
    assert len(client.calls) == 1
    assert row["status"] == "pending"
    assert row["claim_token"] is None
    assert row["attempts"] == 1  # this attempt really did fail


def test_last_failure_during_shutdown_still_marks_the_ticket_failed(h):
    h.add("t-1")

    async def main():
        slow = Gated(TimeoutError())
        worker = h.worker(slow, max_attempts=1, shutdown_grace=5.0)
        await worker.submit("t-1")
        await slow.started.wait()
        stopping = asyncio.create_task(worker.stop())
        await asyncio.sleep(0.01)
        slow.release.set()
        await stopping

    asyncio.run(main())
    assert h.read("t-1")["status"] == "failed"


def test_queued_tickets_are_not_started_during_shutdown(h):
    h.add("running", "queued")

    async def main():
        slow = Gated(output())
        worker = h.worker(slow, concurrency=1, shutdown_grace=5.0)
        await worker.submit("running")
        await slow.started.wait()
        await worker.submit("queued")
        await asyncio.sleep(0.01)
        stopping = asyncio.create_task(worker.stop())
        await asyncio.sleep(0.01)
        slow.release.set()
        await stopping

    asyncio.run(main())
    assert h.read("running")["status"] == "classified"
    queued = h.read("queued")
    assert (queued["status"], queued["attempts"], queued["claim_token"]) == ("pending", 0, None)


def test_nothing_starts_after_stop(h):
    h.add("t-1")
    client = ScriptedLLMClient([output()])

    async def main():
        worker = h.worker(client)
        await worker.stop()
        await worker.submit("t-1")
        await worker.process("t-1")
        await worker.sweep()
        await asyncio.gather(*worker._tasks)

    asyncio.run(main())
    assert client.calls == []
    assert h.read("t-1")["status"] == "pending"


def test_stop_ends_the_sweeper(h):
    async def main():
        worker = h.worker(ScriptedLLMClient([]), sweep_interval=0.01)
        await worker.start()
        sweeper = worker._sweeper
        await worker.stop()
        assert sweeper.cancelled()

    asyncio.run(main())


def test_release_needs_the_matching_token(h):
    h.add("t-1")
    h.crash_while_holding("t-1")
    conn = connect(h.path)
    assert tickets.release_claim(conn, "t-1", "someone-else", True, now=1.0) is False
    conn.close()
    row = h.read("t-1")
    assert (row["claim_token"], row["attempts"]) == ("dead", 1)


# Settings


@pytest.mark.parametrize(
    "bad",
    [
        {"concurrency": 0},
        {"max_attempts": 0},
        {"claim_ttl": 30.0},
        {"model_timeout": 59.0},
    ],
)
def test_unsafe_settings_are_refused(bad):
    with pytest.raises(ValueError):
        Settings(**bad)


def test_concurrency_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_CLASSIFICATIONS", "7")
    assert Settings.from_env().concurrency == 7


# Through the running app


def wait_for(check, seconds=5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.01)
    return False


def test_api_returns_pending_then_shows_the_result(h):
    client = ScriptedLLMClient([output()])
    app = create_app(h.path, worker=h.worker(client))
    with TestClient(app) as api:
        created = api.post("/tickets", json={"id": "t-1", "body": "Please refund."})
        assert created.status_code == 201
        assert created.json()["status"] == "pending"
        assert wait_for(lambda: api.get("/tickets/t-1").json()["status"] == "classified")
        assert api.post("/tickets", json={"id": "t-1", "body": "Please refund."}).status_code == 200
        assert [t["id"] for t in api.get("/tickets?category=billing").json()["items"]] == ["t-1"]
    assert len(client.calls) == 1


def test_create_request_does_not_wait_for_the_model(h):
    """The ticket must be stored and answered before classification finishes."""
    seen = []

    class Checking:
        async def complete(self, system, user):
            seen.append(h.read("t-1")["status"])
            return output()

    with TestClient(create_app(h.path, worker=h.worker(Checking()))) as api:
        body = api.post("/tickets", json={"id": "t-1", "body": "x"}).json()
        assert wait_for(lambda: seen)
    assert body["status"] == "pending"
    assert seen == ["pending"]


def test_app_start_recovers_tickets_left_by_a_previous_run(h):
    h.add("waiting", "crashed", "held")
    h.crash_while_holding("crashed")
    h.sql("UPDATE tickets SET claim_expires_at = ? WHERE id = 'crashed'", h.clock() - 1)
    h.crash_while_holding("held")
    client = ScriptedLLMClient([output()] * 2)
    worker = h.worker(client, sweep_interval=0.01)

    with TestClient(create_app(h.path, worker=worker)) as api:
        def recovered():
            status = {i: api.get(f"/tickets/{i}").json()["status"] for i in ("waiting", "crashed")}
            return set(status.values()) == {"classified"}

        assert wait_for(recovered)
        assert api.get("/tickets/held").json()["status"] == "pending"
    assert len(client.calls) == 2
