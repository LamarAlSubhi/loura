# Ticket triage service

An HTTP service that ingests support tickets, classifies each one in the background with an LLM, and serves the results.

Classification uses a fake LLM client, so no API key is needed. It returns plausible answers based on keywords, and broken ones about 20% of the time, because the point of this project is how the service handles an unreliable model.

## Run it

Requires Python 3.12 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app
```

In a second terminal, load the sample tickets and see how they ended up:

```bash
python3 scripts/load_samples.py
```

Run the tests with the environment active:

```bash
pytest
```

If port 8000 is taken, use `uvicorn app.main:app --port 8001` and `python3 scripts/load_samples.py --url http://127.0.0.1:8001`.

Optional environment variables: `TICKETS_DB_PATH` (default `tickets.db`) and `MAX_CONCURRENT_CLASSIFICATIONS` (default 4).

## API

| Request | Response |
|---|---|
| `POST /tickets` with `id`, `body`, optional `subject` | `201` new, `200` same content already stored, `409` same id with different content, `422` invalid |
| `GET /tickets/{id}` | `200` or `404` |
| `GET /tickets?category=&priority=&limit=&offset=` | `200` with `items`, `total`, `limit`, `offset` |

Errors look like `{"error": {"code": "...", "message": "..."}}`. Interactive docs are at `/docs` while the service runs.

## Layout

- `app/main.py`: routes and startup
- `app/models.py`: request, response and classification shapes
- `app/tickets.py`: all ticket SQL
- `app/db.py`: connection and schema
- `app/classifier.py`: prompt and output validation
- `app/llm.py`: model interface and fake client
- `app/worker.py`: claiming, retries, concurrency, recovery and shutdown

## Decisions

**Storage: SQLite.** No setup, tickets survive restarts, and the primary key makes duplicates impossible even when two requests race. The database also has CHECK constraints, so a bug in the Python validation still can't store an invalid category or a half-finished ticket. The limit is one writer at a time on one machine.

**Async work: FastAPI BackgroundTasks plus a sweeper.** A new ticket is committed, then handed to the worker after the response. If that hand-off is lost, the sweeper finds the ticket, since the database row is the only queue. BackgroundTasks only starts a task the worker owns. If it ran the work itself, shutdown would wait on it, and the worker couldn't interrupt it.

**Concurrency: 4 at once, configurable.** This protects the provider from a burst of tickets. A slot is taken before the claim, so a claim can't expire while waiting in the queue, and is held during backoff, which eases off a struggling provider.

**Restart: expiring claims.** There's no `processing` status. A ticket being worked on stays `pending` with a claim token and an expiry, set by one atomic update. If the process dies, the claim expires after 60 seconds and the sweeper picks the ticket up. A worker that lost its claim can't write, because every write checks the token.

**Retries: 3 attempts with backoff of 1s then 2s.** Timeouts, provider errors and invalid output all count, since a bad response may succeed on a second try. After the third, the ticket is `failed` with the last reason. Attempts are counted before the model call, so a ticket that crashes the process every time still ends up `failed`.

**Invalid output is rejected, not repaired.** The response must be JSON with exactly `category`, `priority` and `summary`, using allowed values and a single sentence. Digging JSON out of surrounding text could pick up JSON a ticket tricked the model into repeating. Failure reasons contain only the field and error type, never model output.

**Prompt injection: containment.** Ticket text goes only in the user message, JSON-encoded inside tags it can't close. The model has no tools, and output is strictly validated. Suspicious tickets aren't rejected, because that also blocks real customers, and t-1005 contains a real question.

**Duplicates: compare content.** The same id and content returns the stored ticket, so clients can retry safely. Different content returns `409` rather than silently dropping the change.

**Shutdown: finish or release.** On SIGTERM nothing new starts, running work gets 10 seconds to finish, and whatever remains releases its claim so the next run can take it immediately. An interrupted attempt isn't counted against the ticket. A hard kill falls back to claim expiry.

**List filters: category and priority only.** Pending and failed tickets have neither, so they appear only in unfiltered lists.

## Sample tickets

`data/sample_tickets.json` holds t-1001 to t-1010, all of which load. With the fake client:

- **t-1005 (injection)** is stored and classified as `billing`/`medium`. The fake uses keywords and ignores instructions, so this says nothing about a real model.
- **t-1008 (empty subject)** loads and is classified `other`/`low`.
- **t-1009 (two issues)** is `technical`, for the export problem rather than the overcharge aside.
- **t-1010 (quoted reply)** is `technical`/`high`, and the quoted "we have fixed the issue" doesn't change that.

Because of the fake's failure rate, about 1 ticket in 125 uses all 3 attempts and ends up `failed`, which the loader shows.

## Weaknesses

- **Injection isn't prevented.** Validation checks shape, not truth. If a real model obeyed t-1005 and returned `technical`, `high` and "Approved for immediate refund.", it would be stored. A test records this. Anything reading `summary` must treat it as untrusted.
- **No real model has been used.** Nothing here measures accuracy or resistance to injection.
- **Classification can run twice.** If a claim expires during a model call that's still going, a second call can start. Only one result is saved, but both cost money. Settings refuse a claim shorter than the model timeout plus backoff, which makes this unlikely.
- **Failed tickets stay failed.** During a long outage, each ticket uses its 3 attempts and fails, and nothing retries it. There's also no circuit breaker.
- **Failed tickets are hard to find,** since finding them means paging through the unfiltered list.
- **The one-sentence check is a heuristic.** "Mr. Smith" looks like two sentences, so that answer is rejected and retried.
- **Backlogs clear slowly after a restart.** The sweeper starts at most 4 tickets every 5 seconds, however quickly they finish, so 1,000 waiting tickets take about 20 minutes.
- **Bursts aren't bounded in memory.** Every new ticket gets its own waiting task, so a large burst means a large number of tasks.
- **Backoff has no jitter,** so tickets that fail together retry together.
- **Offset pagination** can return a shifting `total`, and gets slower on deep pages.
- **One machine only,** because of SQLite.

## With more time

- A real provider behind `LLMClient`, plus an evaluation script with labelled tickets
- Re-classifying failed tickets, or ones classified before a prompt change
- A `status` filter on the list endpoint
- A circuit breaker and jitter for provider outages
- Metrics on attempts, failure reasons and queue depth
- Postgres, if this ever needed more than one machine
