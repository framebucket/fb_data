from __future__ import annotations

import json
import re
import shutil
import threading
from pathlib import Path, PurePosixPath

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def validate_id(value: str, label: str) -> str:
    if not ID_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must contain only letters, digits, '_' or '-'.")
    return value


def safe_filename(value: str) -> str:
    normalized = value.replace("\\", "/")
    name = PurePosixPath(normalized).name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not name:
        raise ValueError("Filename is empty or invalid.")
    return name[:180]


class LocalStorage:
    """Local disk adapter; production storage is designed to use private object storage."""

    def __init__(self, root: Path, metadata_root: Path, private_root: Path, event_root: Path, album_root: Path):
        self.root = root
        self.metadata_root = metadata_root
        self.private_root = private_root
        self.event_root = event_root
        self.album_root = album_root
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    def lock_for_event(self, event_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(event_id, threading.RLock())

    def prepare_event(self, photographer_id: str, event_id: str) -> None:
        validate_id(photographer_id, "photographer_id")
        validate_id(event_id, "event_id")
        (self.private_root / photographer_id / event_id / "originals" / "root").mkdir(parents=True, exist_ok=True)
        (self.event_root / event_id / "photos" / "root").mkdir(parents=True, exist_ok=True)
        (self.event_root / event_id / "videos" / "root").mkdir(parents=True, exist_ok=True)
        (self.metadata_root / event_id).mkdir(parents=True, exist_ok=True)
        folders_path = self.folders_path(event_id)
        if not folders_path.exists():
            folders_path.write_text(json.dumps({"folders": [{"folder_id": "root", "name": "Gallery"}]}, indent=2), encoding="utf-8")

    def original_path(self, photographer_id: str, event_id: str, photo_id: str, filename: str, folder_id: str = "root") -> Path:
        validate_id(folder_id, "folder_id")
        return self.private_root / photographer_id / event_id / "originals" / folder_id / f"{photo_id}_{safe_filename(filename)}"

    def processed_path(self, event_id: str, photo_id: str, folder_id: str = "root", filename: str = "") -> Path:
        validate_id(folder_id, "folder_id")
        suffix = Path(filename).suffix.lower()
        if suffix in {".mp4", ".mov", ".m4v", ".webm", ".avi"}:
            return self.event_root / event_id / "videos" / folder_id / f"{photo_id}{suffix}"
        return self.event_root / event_id / "photos" / folder_id / f"{photo_id}.jpg"

    def manifest_path(self, event_id: str) -> Path:
        return self.metadata_root / event_id / "photos.json"

    def folders_path(self, event_id: str) -> Path:
        return self.metadata_root / event_id / "folders.json"

    def read_folders(self, event_id: str) -> list[dict]:
        path = self.folders_path(event_id)
        if not path.exists():
            return [{"folder_id": "root", "name": "Gallery"}]
        return json.loads(path.read_text(encoding="utf-8")).get("folders", [])

    def write_folders(self, event_id: str, folders: list[dict]) -> None:
        path = self.folders_path(event_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps({"folders": folders}, indent=2), encoding="utf-8")
        temporary.replace(path)

    def read_manifest(self, event_id: str) -> dict:
        path = self.manifest_path(event_id)
        if not path.exists():
            return {"event_id": event_id, "photos": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def write_manifest(self, event_id: str, manifest: dict) -> None:
        path = self.manifest_path(event_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(path)

    def create_album_directory(self, event_id: str, submission_id: str) -> Path:
        directory = self.album_root / event_id / submission_id
        directory.mkdir(parents=True, exist_ok=False)
        return directory

    def move_original_to_album(self, source: Path, album_directory: Path, photo_id: str, photo_number: int) -> Path:
        destination = album_directory / f"{photo_number:05d}_{source.name.split('_', 1)[-1]}"
        shutil.move(str(source), str(destination))
        return destination
