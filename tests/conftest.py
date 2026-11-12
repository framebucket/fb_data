"""Shared fixtures for the data-service test suite.

The tests run the real service code against a real PostgreSQL database and local-disk storage.
Point TEST_DATABASE_URL (or FB_DATABASE_URL) at an empty, disposable database, e.g. the Postgres
service container in CI:  postgresql://postgres:postgres@localhost:5432/fb_test
"""
from __future__ import annotations

import datetime as dt
import io
import os
import random
import uuid
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFilter


def _configure_environment(tmp_root: Path) -> None:
    url = os.getenv("TEST_DATABASE_URL") or os.getenv("FB_DATABASE_URL")
    if not url:
        pytest.exit("Set TEST_DATABASE_URL to a disposable PostgreSQL database to run the tests.", returncode=2)
    os.environ.update(
        FB_DATABASE_URL=url,
        FB_STORAGE_BACKEND="local",
        FB_DATA_ROOT=str(tmp_root / "storage"),
        FB_TEMP_ROOT=str(tmp_root / "tmp"),
        FB_FACE_MATCHING_ENABLED="false",      # no model download in CI; face-aware code falls back gracefully
        FB_EMBEDDED_WORKER="false",
        FB_SERVICE_TOKEN="test-service-token",
        FB_PUBLIC_BASE_URL="http://testserver",
        FB_DB_POOL_SIZE="4",
    )


@pytest.fixture(scope="session")
def service(tmp_path_factory):
    _configure_environment(tmp_path_factory.mktemp("fbdata"))
    from app import db as db_module
    from app.config import Settings
    from app.service import PhotoService

    settings = Settings.from_env()
    svc = PhotoService(settings, database=db_module.Database(settings.database_url, settings.db_pool_size))
    svc.db.migrate()
    yield svc
    svc.db.close()


@pytest.fixture
def ids():
    """Fresh photographer / customer / event ids for one test."""
    return {"pid": str(uuid.uuid4()), "cid": str(uuid.uuid4()), "eid": str(uuid.uuid4())}


@pytest.fixture
def event(service, ids):
    service.create_event(ids["eid"], ids["pid"], ids["cid"], "TEST" + ids["eid"][:4])
    return ids


def make_photo(path: Path, seed: int, size=(1800, 1200), taken: dt.datetime | None = None) -> Path:
    """A synthetic 'photo': graded background, a few shapes, an EXIF capture time."""
    rng = random.Random(seed)
    width, height = size
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    top = tuple(rng.randint(40, 220) for _ in range(3))
    bottom = tuple(rng.randint(20, 200) for _ in range(3))
    for y in range(height):
        t = y / height
        draw.line([(0, y), (width, y)], fill=tuple(int(a * (1 - t) + b * t) for a, b in zip(top, bottom)))
    for _ in range(6):
        x, y = rng.randint(0, width), rng.randint(0, height)
        r = rng.randint(60, 260)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=tuple(rng.randint(0, 255) for _ in range(3)))
    image = image.filter(ImageFilter.GaussianBlur(1.2))
    exif = Image.Exif()
    when = taken or dt.datetime(2026, 1, 25, 9, 0) + dt.timedelta(minutes=seed * 3)
    exif.get_ifd(0x8769)[0x9003] = when.strftime("%Y:%m:%d %H:%M:%S")   # Exif IFD -> DateTimeOriginal
    image.save(path, "JPEG", quality=88, exif=exif.tobytes())
    return path


@pytest.fixture
def photos(tmp_path):
    """Factory: photos(n, portrait_every=3) -> list of JPEG paths (mixed landscape / portrait)."""
    def build(count: int, portrait_every: int = 3) -> list[Path]:
        out = []
        for index in range(count):
            size = (1200, 1800) if portrait_every and index % portrait_every == 1 else (1800, 1200)
            out.append(make_photo(tmp_path / f"IMG_{index:04d}.jpg", index, size))
        return out
    return build


def jpeg_bytes(seed: int, size=(1400, 1000)) -> bytes:
    buffer = io.BytesIO()
    path = Path(f"/tmp/fb-photo-{uuid.uuid4().hex}.jpg")
    make_photo(path, seed, size)
    data = path.read_bytes()
    path.unlink(missing_ok=True)
    buffer.write(data)
    return buffer.getvalue()


def drain(service, kinds: set[str] | None = None, limit: int = 500) -> list[dict]:
    """Run queued jobs inline (what the worker does), returning what ran."""
    from app import jobs

    ran = []
    for _ in range(limit):
        job = jobs.claim(service.db, "pytest", service.settings.job_lease_seconds)
        if not job:
            break
        if kinds and job["kind"] not in kinds:
            jobs.complete(service.db, job["id"])
            continue
        try:
            service.run_job(job["kind"], job["payload"])
            jobs.complete(service.db, job["id"])
            ran.append({"kind": job["kind"], "ok": True})
        except Exception as exc:  # mirror the worker: record, give up after the last attempt
            if jobs.fail(service.db, job, str(exc), permanent=True):
                service.job_failed(job["kind"], job["payload"], str(exc))
            ran.append({"kind": job["kind"], "ok": False, "error": str(exc)})
    return ran
