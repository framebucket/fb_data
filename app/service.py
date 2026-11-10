from __future__ import annotations

import json
import logging
import mimetypes
import os
import secrets
import shutil
import tempfile
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import Settings
from .imaging import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, ImageProcessor
from .storage import LocalStorage, safe_filename, validate_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhotoResult:
    photo_id: str
    photo_number: int
    event_id: str
    photographer_id: str
    customer_id: str
    original_filename: str
    original_key: str
    processed_key: str
    status: str
    media_type: str
    folder_id: str
    width: int | None = None
    height: int | None = None
    processed_bytes: int | None = None
    watermarked: bool = False
    ai_enhanced: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


class PhotoService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.storage = LocalStorage(
            settings.data_root,
            settings.metadata_root,
            settings.private_root,
            settings.event_root,
            settings.album_root,
        )
        self.processor = ImageProcessor(
            logo_path=settings.logo_path,
            enhancer_model=settings.enhancer_model,
            max_processed_bytes=settings.max_processed_bytes,
        )

    def create_event(self, event_id: str, photographer_id: str, customer_id: str, event_code: str | None = None) -> dict:
        event_id = validate_id(event_id, "event_id")
        photographer_id = validate_id(photographer_id, "photographer_id")
        customer_id = validate_id(customer_id, "customer_id")
        self.storage.prepare_event(photographer_id, event_id)

        event_file = self.settings.metadata_root / event_id / "event.json"
        with self.storage.lock_for_event(event_id):
            if event_file.exists():
                existing = json.loads(event_file.read_text(encoding="utf-8"))
                if (
                    existing.get("photographer_id") != photographer_id
                    or existing.get("customer_id") != customer_id
                ):
                    raise ValueError("This event ID is already assigned to other users.")
                return existing

            event = {
                "event_id": event_id,
                "event_code": event_code or event_id,
                "photographer_id": photographer_id,
                "customer_id": customer_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "customer_gallery_key": f"events/{event_id}",
                "photographer_originals_key": f"photographer/{photographer_id}/{event_id}/originals",
            }
            temporary = event_file.with_name(event_file.name + ".tmp")
            temporary.write_text(json.dumps(event, indent=2), encoding="utf-8")
            temporary.replace(event_file)
            self.storage.write_manifest(event_id, {"event_id": event_id, "photos": []})
            return event

    def list_folders(self, event_id: str, photographer_id: str, customer_id: str) -> list[dict]:
        event = self._owned_event(event_id, photographer_id, customer_id)
        return self.storage.read_folders(event["event_id"])

    def create_folder(self, event_id: str, photographer_id: str, customer_id: str, name: str) -> dict:
        event = self._owned_event(event_id, photographer_id, customer_id)
        cleaned_name = " ".join(name.split()).strip()
        if not cleaned_name or len(cleaned_name) > 80:
            raise ValueError("Folder name must contain between 1 and 80 characters.")
        with self.storage.lock_for_event(event["event_id"]):
            folders = self.storage.read_folders(event["event_id"])
            if any(folder["name"].casefold() == cleaned_name.casefold() for folder in folders):
                raise ValueError("A folder with this name already exists.")
            folder = {"folder_id": uuid.uuid4().hex, "name": cleaned_name}
            folders.append(folder)
            self.storage.write_folders(event["event_id"], folders)
            for kind in ("photos", "videos"):
                (self.settings.event_root / event_id / kind / folder["folder_id"]).mkdir(parents=True, exist_ok=True)
            (self.settings.private_root / photographer_id / event_id / "originals" / folder["folder_id"]).mkdir(parents=True, exist_ok=True)
            return folder

    def _album_selection_path(self, event_id: str, customer_id: str) -> Path:
        validate_id(event_id, "event_id")
        validate_id(customer_id, "customer_id")
        return self.settings.temp_root / "album-selections" / event_id / f"{customer_id}.json"

    def album_selection(self, event_id: str, photographer_id: str, customer_id: str) -> dict:
        self._owned_event(event_id, photographer_id, customer_id)
        selection_path = self._album_selection_path(event_id, customer_id)
        if not selection_path.is_file():
            return {"photo_ids": []}
        try:
            saved = json.loads(selection_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"photo_ids": []}
        available = {
            photo["photo_id"]
            for photo in self.list_gallery(event_id, customer_id)
            if photo.get("media_type", "image") == "image"
        }
        return {"photo_ids": [photo_id for photo_id in saved.get("photo_ids", []) if photo_id in available]}

    def save_album_selection(
        self,
        event_id: str,
        photographer_id: str,
        customer_id: str,
        photo_ids: list[str],
    ) -> dict:
        self._owned_event(event_id, photographer_id, customer_id)
        unique_photo_ids = list(dict.fromkeys(photo_ids))
        if len(unique_photo_ids) > 5000:
            raise ValueError("A selection can contain at most 5000 photos.")

        with self.storage.lock_for_event(event_id):
            manifest = self.storage.read_manifest(event_id)
            photos_by_id = {photo["photo_id"]: photo for photo in manifest["photos"]}
            for photo_id in unique_photo_ids:
                photo = photos_by_id.get(photo_id)
                if not photo or photo.get("status") != "READY":
                    raise ValueError(f"Photo {photo_id} is not available for selection.")
                if photo.get("media_type", "image") != "image":
                    raise ValueError("Videos cannot be selected for album design.")
                if photo.get("customer_id") != customer_id or photo.get("photographer_id") != photographer_id:
                    raise ValueError(f"Photo {photo_id} belongs to another customer.")

            selection_path = self._album_selection_path(event_id, customer_id)
            if not unique_photo_ids:
                selection_path.unlink(missing_ok=True)
                return {"photo_ids": [], "photo_count": 0}

            selection_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = selection_path.with_name(selection_path.name + ".tmp")
            temporary.write_text(
                json.dumps({
                    "event_id": event_id,
                    "customer_id": customer_id,
                    "photo_ids": unique_photo_ids,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }, indent=2),
                encoding="utf-8",
            )
            temporary.replace(selection_path)
            return {"photo_ids": unique_photo_ids, "photo_count": len(unique_photo_ids)}

    def download_gallery(
        self,
        event_id: str,
        customer_id: str,
        photo_ids: list[str] | None = None,
    ) -> tuple[Path, str, str, bool]:
        available = self.list_gallery(event_id, customer_id)
        if photo_ids:
            requested_ids = list(dict.fromkeys(photo_ids))
            available_by_id = {photo["photo_id"]: photo for photo in available}
            missing = [photo_id for photo_id in requested_ids if photo_id not in available_by_id]
            if missing:
                raise ValueError("One or more selected photos are not available for download.")
            selected = [available_by_id[photo_id] for photo_id in requested_ids]
        else:
            selected = available
        if not selected:
            raise ValueError("There are no gallery photos available to download.")

        event_file = self.settings.metadata_root / event_id / "event.json"
        event = json.loads(event_file.read_text(encoding="utf-8"))
        manifest_photos = {
            photo["photo_id"]: photo
            for photo in self.storage.read_manifest(event_id)["photos"]
        }
        folder_names = {
            folder["folder_id"]: folder["name"]
            for folder in self.storage.read_folders(event_id)
        }
        event_root = self.settings.event_root.resolve()

        downloadable: list[tuple[dict, Path]] = []
        for photo in selected:
            manifest_photo = manifest_photos.get(photo["photo_id"])
            if (
                not manifest_photo
                or manifest_photo.get("customer_id") != customer_id
                or manifest_photo.get("status") not in {"READY", "SUBMITTED_FOR_ALBUM"}
            ):
                raise ValueError("One or more selected photos are not available for download.")
            path = (self.settings.data_root / manifest_photo["processed_key"]).resolve()
            if not path.is_relative_to(event_root) or not path.is_file():
                raise ValueError("A processed gallery photo is unavailable for download.")
            downloadable.append((photo, path))

        if len(downloadable) == 1:
            photo, path = downloadable[0]
            original_name = Path(photo["original_filename"])
            filename = safe_filename(
                f"{photo['photo_number']:05d}_{original_name.stem}{path.suffix or '.jpg'}"
            )
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            return path, filename, media_type, False

        self.settings.temp_root.mkdir(parents=True, exist_ok=True)
        file_descriptor, archive_name = tempfile.mkstemp(
            prefix="customer-gallery-", suffix=".zip", dir=self.settings.temp_root
        )
        archive_path = Path(archive_name)
        try:
            with os.fdopen(file_descriptor, "wb") as archive_file:
                with zipfile.ZipFile(
                    archive_file, "w", compression=zipfile.ZIP_STORED, allowZip64=True
                ) as archive:
                    for photo, path in downloadable:
                        folder_name = safe_filename(folder_names.get(photo.get("folder_id", "root"), "Gallery"))
                        original_name = Path(photo["original_filename"])
                        suffix = path.suffix or ".jpg"
                        entry_name = safe_filename(
                            f"{photo['photo_number']:05d}_{original_name.stem}{suffix}"
                        )
                        archive.write(path, f"{folder_name}/{entry_name}")
            event_code = safe_filename(event.get("event_code", event_id))
            return archive_path, f"FrameBucket-{event_code}-gallery.zip", "application/zip", True
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise

    def _owned_event(self, event_id: str, photographer_id: str, customer_id: str) -> dict:
        event_id = validate_id(event_id, "event_id")
        photographer_id = validate_id(photographer_id, "photographer_id")
        customer_id = validate_id(customer_id, "customer_id")
        event_file = self.settings.metadata_root / event_id / "event.json"
        if not event_file.is_file():
            raise ValueError("Event not found.")
        event = json.loads(event_file.read_text(encoding="utf-8"))
        if event.get("photographer_id") != photographer_id or event.get("customer_id") != customer_id:
            raise ValueError("The event does not belong to the supplied photographer/customer.")
        return event

    def upload(
        self,
        source: Path,
        original_filename: str,
        event_id: str,
        photographer_id: str,
        customer_id: str,
        folder_id: str = "root",
    ) -> PhotoResult:
        event_id = validate_id(event_id, "event_id")
        photographer_id = validate_id(photographer_id, "photographer_id")
        customer_id = validate_id(customer_id, "customer_id")
        original_filename = safe_filename(original_filename)
        extension = Path(original_filename).suffix.lower()
        if extension not in IMAGE_EXTENSIONS and extension not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported image or video type: {extension}")
        media_type = "video" if extension in VIDEO_EXTENSIONS else "image"
        folder_id = validate_id(folder_id, "folder_id")

        event_file = self.settings.metadata_root / event_id / "event.json"
        if not event_file.exists():
            self.create_event(event_id, photographer_id, customer_id)
        else:
            event = json.loads(event_file.read_text(encoding="utf-8"))
            if (
                event.get("photographer_id") != photographer_id
                or event.get("customer_id") != customer_id
            ):
                raise ValueError("The event does not belong to the supplied photographer/customer.")

        photo_id = uuid.uuid4().hex
        if not any(folder["folder_id"] == folder_id for folder in self.storage.read_folders(event_id)):
            raise ValueError("Folder not found for this event.")
        original_path = self.storage.original_path(
            photographer_id, event_id, photo_id, original_filename, folder_id
        )
        processed_path = self.storage.processed_path(event_id, photo_id, folder_id, original_filename)
        relative_original = original_path.relative_to(self.settings.data_root).as_posix()
        relative_processed = processed_path.relative_to(self.settings.data_root).as_posix()

        with self.storage.lock_for_event(event_id):
            manifest = self.storage.read_manifest(event_id)
            photo_number = max(
                (int(photo.get("photo_number", 0)) for photo in manifest["photos"]),
                default=0,
            ) + 1
            record = {
                "photo_id": photo_id,
                "photo_number": photo_number,
                "event_id": event_id,
                "photographer_id": photographer_id,
                "customer_id": customer_id,
                "original_filename": original_filename,
                "media_type": media_type,
                "folder_id": folder_id,
                "original_key": relative_original,
                "processed_key": relative_processed,
                "status": "PROCESSING",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest["photos"].append(record)
            self.storage.write_manifest(event_id, manifest)

        def copy_original() -> None:
            original_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = original_path.with_name(original_path.name + ".tmp")
            try:
                shutil.copy2(source, temporary)
                temporary.replace(original_path)
            finally:
                temporary.unlink(missing_ok=True)

        try:
            # Preserve the photographer's exact source while processing a
            # separate read of that source at the same time.
            def copy_gallery_video() -> dict:
                processed_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = processed_path.with_name(processed_path.name + ".tmp")
                try:
                    shutil.copy2(source, temporary)
                    temporary.replace(processed_path)
                finally:
                    temporary.unlink(missing_ok=True)
                return {"width": None, "height": None, "bytes": processed_path.stat().st_size, "watermarked": False, "ai_enhanced": False}

            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="fb-photo") as pool:
                original_future = pool.submit(copy_original)
                process_future = (
                    pool.submit(self.processor.process, source, processed_path)
                    if media_type == "image"
                    else pool.submit(copy_gallery_video)
                )
                original_future.result()
                processing = process_future.result()

            with self.storage.lock_for_event(event_id):
                manifest = self.storage.read_manifest(event_id)
                saved = next(p for p in manifest["photos"] if p["photo_id"] == photo_id)
                saved.update(
                    status="READY",
                    width=processing["width"],
                    height=processing["height"],
                    processed_bytes=processing["bytes"],
                    watermarked=processing["watermarked"],
                    ai_enhanced=processing["ai_enhanced"],
                )
                self.storage.write_manifest(event_id, manifest)

            return PhotoResult(
                photo_id=photo_id,
                photo_number=photo_number,
                event_id=event_id,
                photographer_id=photographer_id,
                customer_id=customer_id,
                original_filename=original_filename,
                original_key=relative_original,
                processed_key=relative_processed,
                status="READY",
                media_type=media_type,
                folder_id=folder_id,
                width=processing["width"],
                height=processing["height"],
                processed_bytes=processing["bytes"],
                watermarked=bool(processing["watermarked"]),
                ai_enhanced=bool(processing["ai_enhanced"]),
            )
        except Exception as exc:
            logger.exception("Photo processing failed for %s", photo_id)
            with self.storage.lock_for_event(event_id):
                manifest = self.storage.read_manifest(event_id)
                saved = next(p for p in manifest["photos"] if p["photo_id"] == photo_id)
                saved.update(status="PROCESSING_FAILED", error=str(exc)[:500])
                self.storage.write_manifest(event_id, manifest)
            raise

    def process_directory(
        self,
        input_directory: Path,
        event_id: str,
        photographer_id: str,
        customer_id: str,
        workers: int = 2,
    ) -> list[dict]:
        if not input_directory.is_dir():
            raise FileNotFoundError(f"Input folder does not exist: {input_directory}")
        self.create_event(event_id, photographer_id, customer_id)
        sources = sorted(
            (
                path for path in input_directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name.lower(),
        )
        if not sources:
            return []

        def process_one(path: Path) -> dict:
            try:
                return self.upload(
                    path,
                    path.name,
                    event_id,
                    photographer_id,
                    customer_id,
                ).as_dict()
            except Exception as exc:
                return {
                    "original_filename": path.name,
                    "status": "PROCESSING_FAILED",
                    "error": str(exc),
                }

        with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="fb-batch") as pool:
            return list(pool.map(process_one, sources))

    def submit_album(
        self,
        event_id: str,
        photographer_id: str,
        customer_id: str,
        photo_ids: list[str],
    ) -> dict:
        event_id = validate_id(event_id, "event_id")
        photographer_id = validate_id(photographer_id, "photographer_id")
        customer_id = validate_id(customer_id, "customer_id")
        unique_photo_ids = list(dict.fromkeys(photo_ids))
        if not unique_photo_ids:
            raise ValueError("Select at least one photo for album design.")

        event_file = self.settings.metadata_root / event_id / "event.json"
        if not event_file.exists():
            raise ValueError("Event not found.")
        event = json.loads(event_file.read_text(encoding="utf-8"))
        if (
            event.get("photographer_id") != photographer_id
            or event.get("customer_id") != customer_id
        ):
            raise ValueError("The event does not belong to the supplied photographer/customer.")

        with self.storage.lock_for_event(event_id):
            manifest = self.storage.read_manifest(event_id)
            photos_by_id = {photo["photo_id"]: photo for photo in manifest["photos"]}
            selected = []
            for photo_id in unique_photo_ids:
                photo = photos_by_id.get(photo_id)
                if not photo:
                    raise ValueError(f"Photo {photo_id} does not belong to this event.")
                if photo.get("status") != "READY":
                    raise ValueError(f"Photo {photo_id} is not ready for album design.")
                if photo.get("media_type", "image") != "image":
                    raise ValueError("Videos cannot be submitted for album design.")
                if (
                    photo.get("photographer_id") != photographer_id
                    or photo.get("customer_id") != customer_id
                ):
                    raise ValueError(f"Photo {photo_id} belongs to another customer.")
                selected.append(photo)

            submission_id = uuid.uuid4().hex
            token = secrets.token_urlsafe(32)
            album_directory = self.storage.create_album_directory(event_id, submission_id)
            moved: list[tuple[dict, Path, Path]] = []
            try:
                for photo in selected:
                    source = (self.settings.data_root / photo["original_key"]).resolve()
                    expected_root = (
                        self.settings.private_root / photographer_id / event_id
                    ).resolve()
                    if not source.is_relative_to(expected_root) or not source.is_file():
                        raise ValueError("A selected photographer original is unavailable.")
                    destination = self.storage.move_original_to_album(
                        source, album_directory, photo["photo_id"], photo["photo_number"]
                    )
                    moved.append((photo, source, destination))

                archive_path = album_directory / "album-design.zip"
                with zipfile.ZipFile(
                    archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True
                ) as archive:
                    for photo, _, destination in moved:
                        archive.write(destination, arcname=destination.name)

                expires_at = datetime.now(timezone.utc) + timedelta(days=30)
                submission = {
                    "submission_id": submission_id,
                    "event_id": event_id,
                    "event_code": event.get("event_code", event_id),
                    "photographer_id": photographer_id,
                    "customer_id": customer_id,
                    "photo_ids": unique_photo_ids,
                    "photo_numbers": [
                        {
                            "photo_number": photo["photo_number"],
                            "original_filename": photo["original_filename"],
                        }
                        for photo in selected
                    ],
                    "share_token": token,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "expires_at": expires_at.isoformat(),
                    "archive_key": archive_path.relative_to(self.settings.data_root).as_posix(),
                }
                (album_directory / "submission.json").write_text(
                    json.dumps(submission, indent=2), encoding="utf-8"
                )

                for photo, _, destination in moved:
                    photo["original_key"] = destination.relative_to(
                        self.settings.data_root
                    ).as_posix()
                    photo["album_submission_id"] = submission_id
                    photo["status"] = "SUBMITTED_FOR_ALBUM"
                self.storage.write_manifest(event_id, manifest)
                self._album_selection_path(event_id, customer_id).unlink(missing_ok=True)
            except Exception:
                for _, source, destination in reversed(moved):
                    if destination.exists():
                        source.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(destination), str(source))
                shutil.rmtree(album_directory, ignore_errors=True)
                raise

        return {
            "submission_id": submission_id,
            "event_id": event_id,
            "photo_count": len(selected),
            "photo_numbers": submission["photo_numbers"],
            "album_design_link": f"{self.settings.public_base_url}/api/v1/album-design/{token}",
            "expires_at": submission["expires_at"],
            "status": "SUBMITTED",
        }

    def resolve_album_link(self, token: str) -> tuple[Path, str]:
        if not token or len(token) > 200:
            raise FileNotFoundError("Album design link not found.")
        for submission_file in self.settings.album_root.glob("*/*/submission.json"):
            try:
                submission = json.loads(submission_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not secrets.compare_digest(submission.get("share_token", ""), token):
                continue
            expires = datetime.fromisoformat(submission["expires_at"])
            if expires <= datetime.now(timezone.utc):
                raise FileNotFoundError("Album design link has expired.")
            archive = (self.settings.data_root / submission["archive_key"]).resolve()
            if not archive.is_file() or not archive.is_relative_to(self.settings.album_root.resolve()):
                raise FileNotFoundError("Album design archive not found.")
            return archive, f"FrameBucket-{submission.get('event_code', submission['event_id'])}-album-design.zip"
        raise FileNotFoundError("Album design link not found.")

    def list_gallery(self, event_id: str, customer_id: str, folder_id: str | None = None) -> list[dict]:
        event_id = validate_id(event_id, "event_id")
        customer_id = validate_id(customer_id, "customer_id")
        event_file = self.settings.metadata_root / event_id / "event.json"
        if not event_file.is_file():
            raise ValueError("Event not found.")
        event = json.loads(event_file.read_text(encoding="utf-8"))
        if event.get("customer_id") != customer_id:
            raise ValueError("Customer does not have access to this event.")
        manifest = self.storage.read_manifest(event_id)
        return [
            {
                "photo_id": photo["photo_id"],
                "photo_number": photo["photo_number"],
                "original_filename": photo["original_filename"],
                "media_type": photo.get("media_type", "image"),
                "folder_id": photo.get("folder_id", "root"),
                "width": photo.get("width"),
                "height": photo.get("height"),
                "status": photo["status"],
                "ai_enhanced": photo.get("ai_enhanced", False),
                "watermarked": photo.get("watermarked", False),
            }
            for photo in manifest["photos"]
            if photo.get("status") in {"READY", "SUBMITTED_FOR_ALBUM"}
            and (folder_id is None or photo.get("folder_id", "root") == folder_id)
        ]

    def media_path(self, event_id: str, photo_id: str, customer_id: str) -> Path:
        event_id = validate_id(event_id, "event_id")
        customer_id = validate_id(customer_id, "customer_id")
        event_file = self.settings.metadata_root / event_id / "event.json"
        if not event_file.is_file():
            raise ValueError("Event not found.")
        event = json.loads(event_file.read_text(encoding="utf-8"))
        if event.get("customer_id") != customer_id:
            raise ValueError("Customer does not have access to this event.")
        manifest = self.storage.read_manifest(event_id)
        photo = next((p for p in manifest["photos"] if p["photo_id"] == photo_id), None)
        if not photo or photo.get("status") not in {"READY", "SUBMITTED_FOR_ALBUM"}:
            raise ValueError("Processed photo not found.")
        path = (self.settings.data_root / photo["processed_key"]).resolve()
        if not path.is_relative_to(self.settings.event_root.resolve()) or not path.is_file():
            raise ValueError("Processed photo not found.")
        return path

    def preview_path(self, event_id: str, photo_id: str, customer_id: str) -> Path:
        return self.media_path(event_id, photo_id, customer_id)

    def original_path_for_photographer(self, event_id: str, photo_id: str, photographer_id: str, customer_id: str) -> Path:
        event = self._owned_event(event_id, photographer_id, customer_id)
        manifest = self.storage.read_manifest(event["event_id"])
        photo = next((item for item in manifest["photos"] if item["photo_id"] == photo_id), None)
        if not photo or photo.get("photographer_id") != photographer_id:
            raise ValueError("Photographer original not found.")
        path = (self.settings.data_root / photo["original_key"]).resolve()
        allowed_roots = [
            (self.settings.private_root / photographer_id / event_id).resolve(),
            (self.settings.album_root / event_id).resolve(),
        ]
        if not any(path.is_relative_to(root) for root in allowed_roots) or not path.is_file():
            raise ValueError("Photographer original not found.")
        return path
