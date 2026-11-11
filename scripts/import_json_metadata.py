"""One-time import of gallery metadata written by fb_data 0.1 (JSON files in storage) into PostgreSQL.

Run from the repository root with the same FB_STORAGE_BACKEND / FB_DATA_ROOT / S3 settings the
service used before, plus FB_DATABASE_URL:

    python scripts/import_json_metadata.py            # import everything, skip rows that exist
    python scripts/import_json_metadata.py --dry-run  # report only

The JSON files are left untouched. Re-running is safe: existing rows are not changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from app import jobs
from app.config import Settings
from app.db import Database
from app.storage import create_storage

STATUS_MAP = {
    "READY": "READY",
    "SUBMITTED_FOR_ALBUM": "SUBMITTED_FOR_ALBUM",
    "PROCESSING_FAILED": "PROCESSING_FAILED",
    "PROCESSING": "PROCESSING_FAILED",  # interrupted under the old synchronous pipeline
}


def read_json(storage, key: str, default=None):
    try:
        return json.loads(storage.read_text(key))
    except FileNotFoundError:
        return default
    except (OSError, ValueError):
        logging.exception("Could not read %s", key)
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = Settings.from_env()
    storage = create_storage(settings)
    database = Database(settings.database_url, 4)
    database.migrate()

    keys = storage.list_keys("metadata/events/")
    event_keys = sorted(key for key in keys if key.endswith("/event.json"))
    submission_keys = sorted(
        key for key in storage.list_keys("album_design/") if key.endswith("/submission.json")
    )
    selection_keys = sorted(
        key for key in storage.list_keys("tmp/album-selections/") if key.endswith(".json")
    )
    counts = dict(events=0, folders=0, media=0, guests=0, submissions=0, selections=0, scans=0)

    for event_key in event_keys:
        event = read_json(storage, event_key)
        if not event:
            continue
        event_id = event["event_id"]
        base = f"metadata/events/{event_id}"
        folders = read_json(storage, f"{base}/folders.json", {"folders": []}).get("folders", [])
        photos = read_json(storage, f"{base}/photos.json", {"photos": []}).get("photos", [])
        face_index = read_json(storage, f"{base}/face-index.json", {}) or {}
        photo_embeddings = face_index.get("photos", {})
        if args.dry_run:
            print(f"{event_id}: {len(folders)} folders, {len(photos)} media, {len(face_index.get('guests', {}))} guests")
            continue

        with database.transaction() as connection:
            max_number = max((int(photo.get("photo_number") or 0) for photo in photos), default=0)
            inserted = connection.execute(
                "INSERT INTO fb_data.event (event_id, event_code, photographer_id, customer_id, next_photo_number, created_at) "
                "VALUES (%s, %s, %s, %s, %s, COALESCE(%s::timestamptz, now())) ON CONFLICT (event_id) DO NOTHING",
                (event_id, event.get("event_code") or event_id, event["photographer_id"], event["customer_id"],
                 max_number + 1, event.get("created_at")),
            ).rowcount
            counts["events"] += inserted
            connection.execute(
                "UPDATE fb_data.event SET next_photo_number = GREATEST(next_photo_number, %s) WHERE event_id = %s",
                (max_number + 1, event_id),
            )
            for folder in [{"folder_id": "root", "name": "Gallery"}, *folders]:
                counts["folders"] += connection.execute(
                    "INSERT INTO fb_data.folder (event_id, folder_id, name) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    (event_id, folder["folder_id"], folder["name"]),
                ).rowcount
            for photo in photos:
                status = STATUS_MAP.get(photo.get("status"), "PROCESSING_FAILED")
                embeddings = photo_embeddings.get(photo["photo_id"])
                counts["media"] += connection.execute(
                    """
                    INSERT INTO fb_data.media (photo_id, event_id, photo_number, folder_id, media_type,
                        original_filename, original_key, processed_key, status, width, height, processed_bytes,
                        watermarked, ai_enhanced, error, album_submission_id, face_embeddings, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        COALESCE(%s::timestamptz, now()))
                    ON CONFLICT (photo_id) DO NOTHING
                    """,
                    (photo["photo_id"], event_id, photo.get("photo_number"), photo.get("folder_id", "root"),
                     photo.get("media_type", "image"), photo["original_filename"], photo["original_key"],
                     photo["processed_key"], status, photo.get("width"), photo.get("height"),
                     photo.get("processed_bytes"), bool(photo.get("watermarked")), bool(photo.get("ai_enhanced")),
                     photo.get("error"), photo.get("album_submission_id"),
                     Jsonb(embeddings) if isinstance(embeddings, list) else None, photo.get("created_at")),
                ).rowcount
                if (status in ("READY", "SUBMITTED_FOR_ALBUM") and photo.get("media_type", "image") == "image"
                        and not isinstance(embeddings, list) and settings.face_matching_enabled):
                    jobs.enqueue(connection, jobs.SCAN_FACES, {"photo_id": photo["photo_id"]})
                    counts["scans"] += 1
            for guest_id, embedding in face_index.get("guests", {}).items():
                counts["guests"] += connection.execute(
                    "INSERT INTO fb_data.face_guest (event_id, guest_id, embedding) VALUES (%s, %s, %s) "
                    "ON CONFLICT DO NOTHING",
                    (event_id, guest_id, Jsonb(embedding)),
                ).rowcount

    if not args.dry_run:
        for key in submission_keys:
            submission = read_json(storage, key)
            if not submission or not submission.get("share_token"):
                continue
            with database.transaction() as connection:
                if not connection.execute(
                    "SELECT 1 FROM fb_data.event WHERE event_id = %s", (submission["event_id"],)
                ).fetchone():
                    continue
                created = submission.get("created_at") or datetime.now(timezone.utc).isoformat()
                expires = submission.get("expires_at") or (
                    datetime.fromisoformat(created) + timedelta(days=30)
                ).isoformat()
                counts["submissions"] += connection.execute(
                    """
                    INSERT INTO fb_data.album_submission (submission_id, event_id, photographer_id, customer_id,
                        token_hash, archive_key, status, photo_ids, photo_numbers, created_at, ready_at, expires_at)
                    VALUES (%s, %s, %s, %s, %s, %s, 'READY', %s, %s, %s, %s, %s)
                    ON CONFLICT (submission_id) DO NOTHING
                    """,
                    (submission["submission_id"], submission["event_id"], submission["photographer_id"],
                     submission["customer_id"], hashlib.sha256(submission["share_token"].encode()).hexdigest(),
                     submission["archive_key"], Jsonb(submission.get("photo_ids", [])),
                     Jsonb(submission.get("photo_numbers", [])), created, created, expires),
                ).rowcount
        for key in selection_keys:
            selection = read_json(storage, key)
            if not selection or not selection.get("photo_ids"):
                continue
            with database.transaction() as connection:
                counts["selections"] += connection.execute(
                    "INSERT INTO fb_data.album_selection (event_id, customer_id, photo_ids) "
                    "SELECT %s, %s, %s WHERE EXISTS (SELECT 1 FROM fb_data.event WHERE event_id = %s) "
                    "ON CONFLICT DO NOTHING",
                    (selection["event_id"], selection["customer_id"], Jsonb(selection["photo_ids"]),
                     selection["event_id"]),
                ).rowcount

    print(("Dry run: " if args.dry_run else "Imported: ") + ", ".join(f"{v} {k}" for k, v in counts.items()))
    database.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
