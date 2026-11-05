"""Background classification: claiming, retries, concurrency, recovery and shutdown."""

import asyncio
import logging
import os
import sqlite3
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app import tickets
from app.classifier import DEFAULT_TIMEOUT_SECONDS, ClassificationError, classify
from app.db import connect
from app.llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    concurrency: int = 4
    max_attempts: int = 3
    backoff_base: float = 1.0
    model_timeout: float = DEFAULT_TIMEOUT_SECONDS
    claim_ttl: float = 60.0
    sweep_interval: float = 5.0
    shutdown_grace: float = 10.0

    def __post_init__(self):
        if self.concurrency < 1 or self.max_attempts < 1:
            raise ValueError("concurrency and max_attempts must be at least 1")
        # A shorter claim could expire on healthy work and let another worker take it.
        longest_backoff = self.backoff_base * 2 ** max(self.max_attempts - 2, 0)
        if self.claim_ttl <= self.model_timeout + longest_backoff:
            raise ValueError("claim_ttl must exceed model_timeout plus the longest backoff")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(concurrency=int(os.environ.get("MAX_CONCURRENT_CLASSIFICATIONS", "4")))


class Worker:
    def __init__(
        self,
        db_path: str | Path,
        client: LLMClient,
        settings: Settings | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self._db_path = db_path
        self._client = client
        self._settings = settings or Settings()
        self._clock = clock
        self._sleep = sleep
        self._semaphore = asyncio.Semaphore(self._settings.concurrency)
        self._active: set[str] = set()
        self._tasks: set[asyncio.Task] = set()
        self._sweeper: asyncio.Task | None = None
        self._stopping = False

    async def start(self) -> None:
        self._sweeper = asyncio.create_task(self._sweep_forever())

    async def stop(self) -> None:
        """Let running work finish within the grace period, then release what's left."""
        self._stopping = True
        if self._sweeper:
            self._sweeper.cancel()
            await asyncio.gather(self._sweeper, return_exceptions=True)
            self._sweeper = None
        if self._tasks:
            _, unfinished = await asyncio.wait(self._tasks, timeout=self._settings.shutdown_grace)
            for task in unfinished:
                task.cancel()
            await asyncio.gather(*unfinished, return_exceptions=True)

    async def submit(self, ticket_id: str) -> None:
        # A separate task, so stop() can see it. Awaiting here would tie the work to the
        # request, and uvicorn waits for requests before it runs the shutdown hook.
        self._spawn(ticket_id)

    async def process(self, ticket_id: str) -> None:
        if ticket_id in self._active:
            return
        self._active.add(ticket_id)
        try:
            # Slot before claim, so a claim's clock doesn't run while queueing.
            async with self._semaphore:
                token = uuid.uuid4().hex
                ticket = await self._db(
                    tickets.claim_ticket, ticket_id, token, self._clock(), self._expiry()
                )
                if ticket is not None:
                    await self._run_attempts(ticket, token)
        except Exception:
            # The ticket stays claimed, so the sweeper retries it after the claim expires.
            logger.exception("Unexpected error processing ticket %s", ticket_id)
        finally:
            self._active.discard(ticket_id)

    async def sweep(self) -> None:
        free = self._settings.concurrency - len(self._active)
        if free <= 0:
            return
        ids = await self._db(
            tickets.claimable_ticket_ids, self._clock(), free + len(self._active)
        )
        for ticket_id in [i for i in ids if i not in self._active][:free]:
            self._spawn(ticket_id)

    def _spawn(self, ticket_id: str) -> None:
        task = asyncio.create_task(self.process(ticket_id))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _sweep_forever(self) -> None:
        while True:
            try:
                await self.sweep()
            except Exception:
                logger.exception("Sweep failed")
            await asyncio.sleep(self._settings.sweep_interval)

    async def _run_attempts(self, ticket: sqlite3.Row, token: str) -> None:
        ticket_id = ticket["id"]
        max_attempts = self._settings.max_attempts
        if ticket["attempts"] >= max_attempts:
            # Attempts are counted up front, so this is a ticket whose attempts all died
            # mid-call. Without this check it could crash the process forever.
            reason = f"gave up: {ticket['attempts']} attempts were interrupted before finishing"
            await self._db(tickets.mark_failed, ticket_id, token, reason, self._clock())
            return

        attempt_running = False
        try:
            while True:
                if self._stopping:
                    await self._db(tickets.release_claim, ticket_id, token, False, self._clock())
                    return
                attempt = await self._db(
                    tickets.start_attempt, ticket_id, token, self._clock(), self._expiry()
                )
                if attempt is None:
                    logger.warning("Ticket %s: claim lost, stopping", ticket_id)
                    return
                attempt_running = True
                try:
                    result = await classify(
                        self._client,
                        ticket["subject"],
                        ticket["body"],
                        timeout=self._settings.model_timeout,
                    )
                except ClassificationError as exc:
                    attempt_running = False
                    logger.warning(
                        "Ticket %s: attempt %d failed: %s", ticket_id, attempt, exc.reason
                    )
                    if attempt >= max_attempts:
                        reason = f"gave up after {attempt} attempts: {exc.reason}"
                        await self._db(tickets.mark_failed, ticket_id, token, reason, self._clock())
                        return
                    if not self._stopping:
                        await self._sleep(self._settings.backoff_base * 2 ** (attempt - 1))
                else:
                    attempt_running = False
                    saved = await self._db(
                        tickets.save_classification, ticket_id, token, result, self._clock()
                    )
                    if not saved:
                        logger.warning("Ticket %s: claim lost, result discarded", ticket_id)
                    return
        except asyncio.CancelledError:
            # Being shut down isn't the ticket's fault, so the attempt doesn't count.
            await self._db(tickets.release_claim, ticket_id, token, attempt_running, self._clock())
            raise

    def _expiry(self) -> float:
        return self._clock() + self._settings.claim_ttl

    async def _db(self, operation: Callable[..., Any], *args: Any) -> Any:
        # In a thread, because waiting on a locked database would otherwise block the event loop.
        def run() -> Any:
            conn = connect(self._db_path)
            try:
                return operation(conn, *args)
            finally:
                conn.close()

        return await asyncio.to_thread(run)
