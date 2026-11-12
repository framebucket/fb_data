from __future__ import annotations

from datetime import datetime
from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb

from .db import Database

PROCESS_MEDIA = "process_media"
SCAN_FACES = "scan_faces"
BUILD_ALBUM = "build_album"
DESIGN_ALBUM = "design_album"
RENDER_INVITATION = "render_invitation"

OUTBOX_FACE_MATCH = "face_match"
OUTBOX_ALBUM_READY = "album_ready"
OUTBOX_ALBUM_FAILED = "album_failed"


def enqueue(
    connection: Connection,
    kind: str,
    payload: dict[str, Any],
    max_attempts: int = 5,
    delay_seconds: int = 0,
) -> int:
    """Queue a job inside the caller's transaction so it becomes visible only on commit."""
    row = connection.execute(
        "INSERT INTO fb_data.job (kind, payload, max_attempts, run_after) "
        "VALUES (%s, %s, %s, now() + make_interval(secs => %s)) RETURNING id",
        (kind, Jsonb(payload), max_attempts, delay_seconds),
    ).fetchone()
    return int(row["id"])


def claim(database: Database, worker_id: str, lease_seconds: int) -> dict | None:
    """Atomically take the next due job. Jobs whose worker died are reclaimed after the lease."""
    with database.transaction() as connection:
        return connection.execute(
            """
            UPDATE fb_data.job
            SET status = 'RUNNING', locked_by = %s, locked_at = now(),
                attempts = attempts + 1, updated_at = now()
            WHERE id = (
                SELECT id FROM fb_data.job
                WHERE (status = 'QUEUED' AND run_after <= now())
                   OR (status = 'RUNNING' AND locked_at < now() - make_interval(secs => %s))
                ORDER BY run_after, id
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING id, kind, payload, attempts, max_attempts
            """,
            (worker_id, lease_seconds),
        ).fetchone()


def complete(database: Database, job_id: int) -> None:
    with database.transaction() as connection:
        connection.execute(
            "UPDATE fb_data.job SET status = 'DONE', last_error = NULL, updated_at = now() WHERE id = %s",
            (job_id,),
        )


def fail(database: Database, job: dict, error: str, permanent: bool = False) -> bool:
    """Record a failure. Returns True when the job will not be retried."""
    final = permanent or int(job["attempts"]) >= int(job["max_attempts"])
    backoff = min(1800, 30 * 2 ** max(0, int(job["attempts"]) - 1))
    with database.transaction() as connection:
        connection.execute(
            """
            UPDATE fb_data.job
            SET status = %s, last_error = %s, locked_by = NULL, locked_at = NULL,
                run_after = now() + make_interval(secs => %s), updated_at = now()
            WHERE id = %s
            """,
            ("FAILED" if final else "QUEUED", error[:2000], 0 if final else backoff, job["id"]),
        )
    return final


def publish(connection: Connection, kind: str, event_id: str, payload: dict[str, Any]) -> None:
    connection.execute(
        "INSERT INTO fb_data.outbox (kind, event_id, payload) VALUES (%s, %s, %s)",
        (kind, event_id, Jsonb(payload)),
    )


def claim_outbox(database: Database, limit: int, lease_seconds: int) -> list[dict]:
    with database.transaction() as connection:
        rows = connection.execute(
            """
            UPDATE fb_data.outbox
            SET claimed_until = now() + make_interval(secs => %s)
            WHERE id IN (
                SELECT id FROM fb_data.outbox
                WHERE delivered_at IS NULL AND (claimed_until IS NULL OR claimed_until < now())
                ORDER BY id
                FOR UPDATE SKIP LOCKED
                LIMIT %s
            )
            RETURNING id, kind, event_id, payload, created_at
            """,
            (lease_seconds, limit),
        ).fetchall()
    rows.sort(key=lambda row: row["id"])
    return [
        {
            "id": row["id"],
            "kind": row["kind"],
            "event_id": row["event_id"],
            "payload": row["payload"],
            "created_at": _iso(row["created_at"]),
        }
        for row in rows
    ]


def ack_outbox(database: Database, ids: list[int]) -> int:
    if not ids:
        return 0
    with database.transaction() as connection:
        return connection.execute(
            "UPDATE fb_data.outbox SET delivered_at = now() WHERE id = ANY(%s) AND delivered_at IS NULL",
            (ids,),
        ).rowcount


def purge_finished(database: Database, retention_days: int = 14) -> None:
    with database.transaction() as connection:
        connection.execute(
            "DELETE FROM fb_data.job WHERE status IN ('DONE', 'FAILED') "
            "AND updated_at < now() - make_interval(days => %s)",
            (retention_days,),
        )
        connection.execute(
            "DELETE FROM fb_data.outbox WHERE delivered_at < now() - make_interval(days => %s)",
            (retention_days,),
        )


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None
