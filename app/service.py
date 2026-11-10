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
from .storage import StorageBackend, create_storage, normalize_key, safe_filename, validate_id

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
        self.storage: StorageBackend = create_storage(settings)
        settings.temp_root.mkdir(parents=True, exist_ok=True)
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

        event_key = self.storage.event_key(event_id)
        with self.storage.lock_for_event(event_id):
            if self.storage.exists(event_key):
                existing = self.storage.read_json(event_key)
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
            self.storage.write_json(event_key, event)
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
            return folder

    def _album_selection_key(self, event_id: str, customer_id: str) -> str:
        return self.storage.album_selection_key(event_id, customer_id)

    def album_selection(self, event_id: str, photographer_id: str, customer_id: str) -> dict:
        self._owned_event(event_id, photographer_id, customer_id)
        selection_key = self._album_selection_key(event_id, customer_id)
        if not self.storage.exists(selection_key):
            return {"photo_ids": []}
        try:
            saved = self.storage.read_json(selection_key)
        except (OSError, ValueError, json.JSONDecodeError):
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

            selection_key = self._album_selection_key(event_id, customer_id)
            if not unique_photo_ids:
                self.storage.delete(selection_key)
                return {"photo_ids": [], "photo_count": 0}

            self.storage.write_json(selection_key, {
                    "event_id": event_id,
                    "customer_id": customer_id,
                    "photo_ids": unique_photo_ids,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
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

        event = self._read_event(event_id)
        manifest_photos = {
            photo["photo_id"]: photo
            for photo in self.storage.read_manifest(event_id)["photos"]
        }
        folder_names = {
            folder["folder_id"]: folder["name"]
            for folder in self.storage.read_folders(event_id)
        }
        downloadable: list[tuple[dict, Path, bool]] = []
        for photo in selected:
            manifest_photo = manifest_photos.get(photo["photo_id"])
            if (
                not manifest_photo
                or manifest_photo.get("customer_id") != customer_id
                or manifest_photo.get("status") not in {"READY", "SUBMITTED_FOR_ALBUM"}
            ):
                raise ValueError("One or more selected photos are not available for download.")
            try:
                key = self._validated_processed_key(event_id, manifest_photo)
                path, cleanup = self.storage.materialize(key)
            except Exception as exc:
                for _, staged, remove in downloadable:
                    if remove:
                        staged.unlink(missing_ok=True)
                if isinstance(exc, (FileNotFoundError, OSError, ValueError)):
                    raise ValueError("A processed gallery photo is unavailable for download.") from exc
                raise
            downloadable.append((photo, path, cleanup))

        if len(downloadable) == 1:
            photo, path, cleanup = downloadable[0]
            original_name = Path(photo["original_filename"])
            filename = safe_filename(
                f"{photo['photo_number']:05d}_{original_name.stem}{path.suffix or '.jpg'}"
            )
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            return path, filename, media_type, cleanup

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
                    for photo, path, _ in downloadable:
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
        finally:
            for _, path, cleanup in downloadable:
                if cleanup:
                    path.unlink(missing_ok=True)

    def _owned_event(self, event_id: str, photographer_id: str, customer_id: str) -> dict:
        event_id = validate_id(event_id, "event_id")
        photographer_id = validate_id(photographer_id, "photographer_id")
        customer_id = validate_id(customer_id, "customer_id")
        event = self._read_event(event_id)
        if event.get("photographer_id") != photographer_id or event.get("customer_id") != customer_id:
            raise ValueError("The event does not belong to the supplied photographer/customer.")
        return event

    def _read_event(self, event_id: str) -> dict:
        event_id = validate_id(event_id, "event_id")
        try:
            return self.storage.read_json(self.storage.event_key(event_id))
        except FileNotFoundError as exc:
            raise ValueError("Event not found.") from exc

    @staticmethod
    def _validated_processed_key(event_id: str, photo: dict) -> str:
        key = normalize_key(photo.get("processed_key", ""))
        if not key.startswith((f"events/{event_id}/photos/", f"events/{event_id}/videos/")):
            raise ValueError("Processed photo not found.")
        return key

    @staticmethod
    def _validated_original_key(event_id: str, photographer_id: str, photo: dict) -> str:
        key = normalize_key(photo.get("original_key", ""))
        private_prefix = f"photographer/{photographer_id}/{event_id}/originals/"
        album_prefix = f"album_design/{event_id}/"
        if not key.startswith((private_prefix, album_prefix)):
            raise ValueError("Photographer original not found.")
        return key

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

        try:
            event = self._read_event(event_id)
        except ValueError:
            self.create_event(event_id, photographer_id, customer_id)
        else:
            if event.get("photographer_id") != photographer_id or event.get("customer_id") != customer_id:
                raise ValueError("The event does not belong to the supplied photographer/customer.")

        photo_id = uuid.uuid4().hex
        if not any(folder["folder_id"] == folder_id for folder in self.storage.read_folders(event_id)):
            raise ValueError("Folder not found for this event.")
        original_key = self.storage.original_key(
            photographer_id, event_id, photo_id, original_filename, folder_id
        )
        processed_key = self.storage.processed_key(event_id, photo_id, folder_id, original_filename)

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
                "original_key": original_key,
                "processed_key": processed_key,
                "status": "PROCESSING",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest["photos"].append(record)
            self.storage.write_manifest(event_id, manifest)

        self.settings.temp_root.mkdir(parents=True, exist_ok=True)
        processed_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="processed-", suffix=Path(processed_key).suffix, dir=self.settings.temp_root, delete=False
            ) as temporary:
                processed_path = Path(temporary.name)
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="fb-photo") as pool:
                original_future = pool.submit(self.storage.upload_file, original_key, source)
                if media_type == "image":
                    process_future = pool.submit(self.processor.process, source, processed_path)
                else:
                    process_future = None
                    shutil.copy2(source, processed_path)
                processing = process_future.result() if process_future else {
                    "width": None,
                    "height": None,
                    "bytes": processed_path.stat().st_size,
                    "watermarked": False,
                    "ai_enhanced": False,
                }
                self.storage.upload_file(processed_key, processed_path)
                original_future.result()

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
                original_key=original_key,
                processed_key=processed_key,
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
            try:
                self.storage.delete(processed_key)
            except Exception:
                logger.exception("Failed to clean up processed storage key for %s", photo_id)
            with self.storage.lock_for_event(event_id):
                manifest = self.storage.read_manifest(event_id)
                saved = next(p for p in manifest["photos"] if p["photo_id"] == photo_id)
                saved.update(status="PROCESSING_FAILED", error=str(exc)[:500])
                self.storage.write_manifest(event_id, manifest)
            raise
        finally:
            if processed_path:
                processed_path.unlink(missing_ok=True)

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

        event = self._read_event(event_id)
        if event.get("photographer_id") != photographer_id or event.get("customer_id") != customer_id:
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
                if photo.get("photographer_id") != photographer_id or photo.get("customer_id") != customer_id:
                    raise ValueError(f"Photo {photo_id} belongs to another customer.")
                selected.append(photo)

            submission_id = uuid.uuid4().hex
            token = secrets.token_urlsafe(32)
            submission_prefix = f"album_design/{event_id}/{submission_id}"
            archive_key = f"{submission_prefix}/album-design.zip"
            submission_key = f"{submission_prefix}/submission.json"
            created_keys: list[str] = []
            staged_files: list[tuple[dict, Path, bool]] = []
            archive_path: Path | None = None
            try:
                for photo in selected:
                    source_key = self._validated_original_key(event_id, photographer_id, photo)
                    try:
                        source, cleanup = self.storage.materialize(
                            source_key, suffix=Path(photo["original_filename"]).suffix
                        )
                    except (FileNotFoundError, OSError) as exc:
                        raise ValueError("A selected photographer original is unavailable.") from exc
                    staged_files.append((photo, source, cleanup))
                    album_original_key = self.storage.album_submission_key(
                        event_id,
                        submission_id,
                        f"{photo['photo_number']:05d}_{photo['original_filename']}",
                    )
                    self.storage.upload_file(album_original_key, source)
                    created_keys.append(album_original_key)

                with tempfile.NamedTemporaryFile(
                    prefix="album-design-", suffix=".zip", dir=self.settings.temp_root, delete=False
                ) as temporary:
                    archive_path = Path(temporary.name)
                with zipfile.ZipFile(
                    archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True
                ) as archive:
                    for photo, source, _ in staged_files:
                        entry_name = safe_filename(
                            f"{photo['photo_number']:05d}_{photo['original_filename']}"
                        )
                        archive.write(source, arcname=entry_name)

                self.storage.upload_file(archive_key, archive_path)
                created_keys.append(archive_key)
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
                            "folder_id": photo.get("folder_id", "root"),
                        }
                        for photo in selected
                    ],
                    "share_token": token,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "expires_at": expires_at.isoformat(),
                    "archive_key": archive_key,
                }
                self.storage.write_json(submission_key, submission)
                created_keys.append(submission_key)

                for photo in selected:
                    photo["album_design_key"] = self.storage.album_submission_key(
                        event_id,
                        submission_id,
                        f"{photo['photo_number']:05d}_{photo['original_filename']}",
                    )
                    photo["album_submission_id"] = submission_id
                    photo["status"] = "SUBMITTED_FOR_ALBUM"
                self.storage.write_manifest(event_id, manifest)
                try:
                    self.storage.delete(self._album_selection_key(event_id, customer_id))
                except Exception:
                    logger.exception("Could not remove album selection draft for event %s", event_id)
            except Exception:
                for key in reversed(created_keys):
                    try:
                        self.storage.delete(key)
                    except Exception:
                        logger.exception("Failed to clean up album storage key %s", key)
                raise
            finally:
                for _, staged, cleanup in staged_files:
                    if cleanup:
                        staged.unlink(missing_ok=True)
                if archive_path:
                    archive_path.unlink(missing_ok=True)

        return {
            "submission_id": submission_id,
            "event_id": event_id,
            "photo_count": len(selected),
            "photo_numbers": submission["photo_numbers"],
            "album_design_link": f"{self.settings.public_base_url}/api/v1/album-design/{token}",
            "expires_at": submission["expires_at"],
            "status": "SUBMITTED",
        }

    def resolve_album_link(self, token: str) -> tuple[Path, str, bool]:
        if not token or len(token) > 200:
            raise FileNotFoundError("Album design link not found.")
        for submission_key in self.storage.list_keys("album_design/"):
            if not submission_key.endswith("/submission.json"):
                continue
            try:
                submission = self.storage.read_json(submission_key)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if not secrets.compare_digest(submission.get("share_token", ""), token):
                continue
            try:
                expires = datetime.fromisoformat(submission["expires_at"])
                archive_key = normalize_key(submission["archive_key"])
                event_id = validate_id(submission["event_id"], "event_id")
                submission_id = validate_id(submission["submission_id"], "submission_id")
            except (KeyError, TypeError, ValueError) as exc:
                raise FileNotFoundError("Album design archive not found.") from exc
            if expires <= datetime.now(timezone.utc):
                raise FileNotFoundError("Album design link has expired.")
            if not archive_key.startswith(f"album_design/{event_id}/{submission_id}/"):
                raise FileNotFoundError("Album design archive not found.")
            try:
                archive, cleanup = self.storage.materialize(
                    archive_key, suffix=".zip", prefix="fb-album-"
                )
            except (FileNotFoundError, OSError) as exc:
                raise FileNotFoundError("Album design archive not found.") from exc
            event_code = safe_filename(submission.get("event_code", event_id))
            return archive, f"FrameBucket-{event_code}-album-design.zip", cleanup
        raise FileNotFoundError("Album design link not found.")

    def list_gallery(self, event_id: str, customer_id: str, folder_id: str | None = None) -> list[dict]:
        event_id = validate_id(event_id, "event_id")
        customer_id = validate_id(customer_id, "customer_id")
        event = self._read_event(event_id)
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

    def media_path(self, event_id: str, photo_id: str, customer_id: str) -> tuple[Path, bool]:
        event_id = validate_id(event_id, "event_id")
        photo_id = validate_id(photo_id, "photo_id")
        customer_id = validate_id(customer_id, "customer_id")
        event = self._read_event(event_id)
        if event.get("customer_id") != customer_id:
            raise ValueError("Customer does not have access to this event.")
        manifest = self.storage.read_manifest(event_id)
        photo = next((item for item in manifest["photos"] if item["photo_id"] == photo_id), None)
        if not photo or photo.get("status") not in {"READY", "SUBMITTED_FOR_ALBUM"}:
            raise ValueError("Processed photo not found.")
        key = self._validated_processed_key(event_id, photo)
        try:
            return self.storage.materialize(key)
        except (FileNotFoundError, OSError) as exc:
            raise ValueError("Processed photo not found.") from exc

    def preview_path(self, event_id: str, photo_id: str, customer_id: str) -> tuple[Path, bool]:
        return self.media_path(event_id, photo_id, customer_id)

    def original_path_for_photographer(
        self, event_id: str, photo_id: str, photographer_id: str, customer_id: str
    ) -> tuple[Path, bool, str]:
        event = self._owned_event(event_id, photographer_id, customer_id)
        photo_id = validate_id(photo_id, "photo_id")
        manifest = self.storage.read_manifest(event["event_id"])
        photo = next((item for item in manifest["photos"] if item["photo_id"] == photo_id), None)
        if not photo or photo.get("photographer_id") != photographer_id:
            raise ValueError("Photographer original not found.")
        key = self._validated_original_key(event_id, photographer_id, photo)
        try:
            path, cleanup = self.storage.materialize(
                key, suffix=Path(photo["original_filename"]).suffix
            )
            return path, cleanup, safe_filename(photo["original_filename"])
        except (FileNotFoundError, OSError) as exc:
            raise ValueError("Photographer original not found.") from exc
