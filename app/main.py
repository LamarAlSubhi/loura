import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app import tickets
from app.db import connect, init_db
from app.llm import FakeLLMClient
from app.models import Category, Priority, TicketIn, TicketOut, TicketPage
from app.worker import Settings, Worker

HTTP_ERROR_CODES = {404: "not_found", 405: "method_not_allowed"}


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def get_conn(request: Request) -> Iterator[sqlite3.Connection]:
    # FastAPI may run this and the route in different threads, hence check_same_thread=False.
    conn = connect(request.app.state.db_path)
    try:
        yield conn
    finally:
        conn.close()


Conn = Annotated[sqlite3.Connection, Depends(get_conn)]
router = APIRouter()


@router.post("/tickets", status_code=201)
def create_ticket(
    ticket: TicketIn,
    conn: Conn,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
) -> TicketOut:
    state = request.app.state
    try:
        row, created = tickets.create_ticket(
            conn, ticket.id, ticket.subject, ticket.body, now=state.clock()
        )
    except tickets.TicketConflict:
        return error_response(
            409, "ticket_conflict", f"Ticket {ticket.id} already exists with different content."
        )
    if created:
        background_tasks.add_task(state.worker.submit, ticket.id)
    else:
        response.status_code = 200
    return TicketOut.model_validate(dict(row))


@router.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str, conn: Conn) -> TicketOut:
    row = tickets.get_ticket(conn, ticket_id)
    if row is None:
        raise HTTPException(404, f"Ticket {ticket_id} not found.")
    return TicketOut.model_validate(dict(row))


@router.get("/tickets")
def list_tickets(
    conn: Conn,
    category: Category | None = None,
    priority: Priority | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> TicketPage:
    rows, total = tickets.list_tickets(conn, category, priority, limit, offset)
    items = [TicketOut.model_validate(dict(row)) for row in rows]
    return TicketPage(items=items, total=total, limit=limit, offset=offset)


def create_app(
    db_path: str | Path | None = None,
    worker: Worker | None = None,
    clock: Callable[[], float] = time.time,
) -> FastAPI:
    db_path = db_path or os.environ.get("TICKETS_DB_PATH", "tickets.db")
    worker = worker or Worker(db_path, FakeLLMClient(), Settings.from_env(), clock=clock)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        conn = connect(app.state.db_path)
        try:
            init_db(conn)
        finally:
            conn.close()
        await worker.start()
        yield
        await worker.stop()

    app = FastAPI(title="Ticket triage", lifespan=lifespan)
    app.state.db_path = db_path
    app.state.worker = worker
    app.state.clock = clock
    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def handle_invalid_request(request: Request, exc: RequestValidationError):
        # Location and message only, so ticket content is never echoed back.
        problems = [
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        ]
        return error_response(422, "invalid_request", "; ".join(problems))

    @app.exception_handler(HTTPException)
    async def handle_http_error(request: Request, exc: HTTPException):
        code = HTTP_ERROR_CODES.get(exc.status_code, "http_error")
        return error_response(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception):
        return error_response(500, "internal_error", "Something went wrong.")

    return app


app = create_app()
