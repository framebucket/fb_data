from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

SCHEMA = "fb_data"
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
_MIGRATION_LOCK_KEY = 4_651_284_301  # arbitrary constant shared by every fb_data instance


class Database:
    """Pooled PostgreSQL access shared by the API and worker processes."""

    def __init__(self, url: str | None, max_size: int = 10) -> None:
        if not url:
            raise ValueError(
                "Set FB_DATABASE_URL (or DATABASE_URL) to the PostgreSQL database used by FrameBucket."
            )
        self.pool = ConnectionPool(
            _normalize_url(url),
            min_size=1,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
            open=True,
            name="fb-data",
        )

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        """Yield a connection inside one transaction: committed on success, rolled back on error."""
        with self.pool.connection() as connection:
            yield connection

    def ping(self) -> None:
        with self.transaction() as connection:
            connection.execute("SELECT 1")

    def close(self) -> None:
        self.pool.close()

    def migrate(self) -> list[str]:
        """Apply pending SQL files from app/migrations exactly once, guarded by an advisory lock."""
        applied_now: list[str] = []
        with self.pool.connection() as connection:
            connection.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK_KEY,))
            connection.commit()
            try:
                connection.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
                connection.execute(
                    f"CREATE TABLE IF NOT EXISTS {SCHEMA}.schema_migrations ("
                    "version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
                )
                connection.commit()
                applied = {
                    row["version"]
                    for row in connection.execute(f"SELECT version FROM {SCHEMA}.schema_migrations")
                }
                connection.commit()
                for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                    if path.name in applied:
                        continue
                    logger.info("Applying data-service migration %s", path.name)
                    # Without parameters psycopg uses the simple-query protocol, so one file may hold
                    # several statements; the whole file and its bookkeeping row share a transaction.
                    connection.execute(path.read_text(encoding="utf-8"))
                    connection.execute(
                        f"INSERT INTO {SCHEMA}.schema_migrations (version) VALUES (%s)", (path.name,)
                    )
                    connection.commit()
                    applied_now.append(path.name)
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK_KEY,))
                connection.commit()
        return applied_now


def _normalize_url(url: str) -> str:
    # Spring-style JDBC URLs are a common copy/paste mistake; libpq needs postgresql://.
    if url.startswith("jdbc:"):
        url = url[len("jdbc:"):]
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url
