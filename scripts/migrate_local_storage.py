from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.config import Settings
from app.storage import create_storage, normalize_key

MIGRATED_ROOTS = ("metadata", "photographer", "events", "album_design")
ALBUM_SELECTION_PREFIX = "tmp/album-selections/"


def should_migrate(key: str) -> bool:
    return key.split("/", 1)[0] in MIGRATED_ROOTS or key.startswith(ALBUM_SELECTION_PREFIX)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy FrameBucket local data into the configured remote storage backend."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace remote objects that already exist.",
    )
    args = parser.parse_args()
    settings = Settings.from_env()
    if settings.storage_backend == "local":
        parser.error("Set FB_STORAGE_BACKEND to 's3' or 'google_drive' for the destination.")
    source_root = settings.data_root
    if not source_root.is_dir():
        parser.error(f"FB_DATA_ROOT is not a directory: {source_root}")

    destination = create_storage(settings)
    copied = skipped = failed = 0
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        key = normalize_key(source.relative_to(source_root).as_posix())
        if not should_migrate(key):
            continue
        try:
            if not args.overwrite and destination.exists(key):
                skipped += 1
                continue
            destination.upload_file(key, source)
            copied += 1
            print(f"Copied {key}")
        except Exception:
            logging.exception("Could not migrate storage key %s", key)
            failed += 1

    print(f"Migration finished: {copied} copied, {skipped} already present, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
