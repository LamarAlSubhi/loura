#!/usr/bin/env python3
"""Post the sample tickets to a running service, then show how each one ended up.

Safe to run repeatedly, because the service ignores duplicates. Uses only the
standard library, so it runs inside or outside the virtual environment.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "sample_tickets.json"


def request(url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"content-type": "application/json"}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data, headers)) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.load(error)
        except ValueError:
            return error.code, {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--wait", type=float, default=30.0, help="seconds to wait for results")
    args = parser.parse_args()

    samples = json.loads(SAMPLES.read_text())
    try:
        posted = [request(f"{args.url}/tickets", ticket) for ticket in samples]
    except urllib.error.URLError as error:
        print(f"Could not reach {args.url}: {error.reason}. Is the service running?", file=sys.stderr)
        return 1

    rejected = [
        (ticket["id"], code, body) for ticket, (code, body) in zip(samples, posted) if code >= 400
    ]
    for ticket_id, code, body in rejected:
        if "error" not in body:
            print(f"{args.url} isn't the ticket service (got {code}). Try --url.", file=sys.stderr)
            return 1
        print(f"{ticket_id}: rejected with {code}: {body['error']['message']}", file=sys.stderr)
    print(f"Submitted {len(samples) - len(rejected)} of {len(samples)} tickets.\n")

    ids = [ticket["id"] for ticket, (code, _) in zip(samples, posted) if code < 400]
    deadline = time.monotonic() + args.wait
    while True:
        tickets = [request(f"{args.url}/tickets/{ticket_id}")[1] for ticket_id in ids]
        if time.monotonic() >= deadline or all(t["status"] != "pending" for t in tickets):
            break
        time.sleep(0.2)

    for t in tickets:
        detail = t["summary"] or t["failure_reason"] or ""
        print(
            f"{t['id']}  {t['status']:<10}  {t['category'] or '-':<9}  {t['priority'] or '-':<6}  "
            f"attempts={t['attempts']}  {detail}"
        )
    return 1 if rejected else 0


if __name__ == "__main__":
    sys.exit(main())
