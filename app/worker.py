"""Background worker: `python -m app.worker`.

Run one or more replicas next to the API. Each replica runs FB_WORKER_CONCURRENCY threads that
claim jobs from fb_data.job with FOR UPDATE SKIP LOCKED, so any number of replicas can share the
queue. A job whose worker dies is picked up again after FB_JOB_LEASE_SECONDS.
"""
from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time

from . import jobs
from .config import Settings
from .service import PermanentJobError, PhotoService

logger = logging.getLogger("app.worker")
MAINTENANCE_INTERVAL_SECONDS = 600


class WorkerPool:
    def __init__(self, service: PhotoService, settings: Settings) -> None:
        self.service = service
        self.settings = settings
        self.stopping = threading.Event()
        self.threads: list[threading.Thread] = []
        self.identity = f"{socket.gethostname()}-{os.getpid()}"

    def start(self) -> None:
        for index in range(self.settings.worker_concurrency):
            thread = threading.Thread(
                target=self._run, args=(f"{self.identity}-{index}", index == 0),
                name=f"fb-worker-{index}", daemon=True,
            )
            thread.start()
            self.threads.append(thread)
        logger.info("Started %s worker thread(s) as %s", len(self.threads), self.identity)

    def stop(self, timeout: float | None = None) -> None:
        self.stopping.set()
        for thread in self.threads:
            thread.join(timeout)

    def wait(self) -> None:
        while any(thread.is_alive() for thread in self.threads):
            for thread in self.threads:
                thread.join(1.0)

    def _run(self, worker_id: str, runs_maintenance: bool) -> None:
        next_maintenance = time.monotonic() if runs_maintenance else float("inf")
        while not self.stopping.is_set():
            if time.monotonic() >= next_maintenance:
                self._maintenance()
                next_maintenance = time.monotonic() + MAINTENANCE_INTERVAL_SECONDS
            try:
                job = jobs.claim(self.service.db, worker_id, self.settings.job_lease_seconds)
            except Exception:
                logger.exception("Could not claim a job; retrying shortly.")
                self.stopping.wait(5)
                continue
            if not job:
                self.stopping.wait(self.settings.worker_poll_seconds)
                continue
            self._execute(job)

    def _execute(self, job: dict) -> None:
        started = time.monotonic()
        try:
            self.service.run_job(job["kind"], job["payload"])
        except Exception as exc:
            permanent = isinstance(exc, PermanentJobError)
            message = f"{type(exc).__name__}: {exc}"
            logger.exception("Job %s (%s) failed on attempt %s.", job["id"], job["kind"], job["attempts"])
            try:
                final = jobs.fail(self.service.db, job, message, permanent)
                if final:
                    self.service.job_failed(job["kind"], job["payload"], message)
            except Exception:
                logger.exception("Could not record the failure of job %s.", job["id"])
            return
        try:
            jobs.complete(self.service.db, job["id"])
        except Exception:
            logger.exception("Job %s finished but could not be marked done; it may run again.", job["id"])
        logger.info("Job %s (%s) done in %.1fs.", job["id"], job["kind"], time.monotonic() - started)

    def _maintenance(self) -> None:
        try:
            removed = self.service.cleanup_abandoned_uploads()
            if removed:
                logger.info("Removed %s abandoned upload(s).", removed)
            jobs.purge_finished(self.service.db)
        except Exception:
            logger.exception("Worker maintenance failed.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()
    service = PhotoService(settings)
    service.db.migrate()
    pool = WorkerPool(service, settings)

    def shutdown(signum, _frame) -> None:
        logger.info("Received signal %s; finishing current jobs before exit.", signum)
        pool.stopping.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    pool.start()
    pool.wait()
    service.db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
