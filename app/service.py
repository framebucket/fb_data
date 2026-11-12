from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import secrets
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps
from psycopg import Connection
from psycopg.errors import UniqueViolation
from psycopg.types.json import Jsonb

from . import jobs
from . import style as style_engine
from .album_service import AlbumDesignMixin, DesignNotReady  # noqa: F401  (re-exported for the API)
from .invitation_service import InvitationMixin
from .config import Settings
from .db import Database
from .face_matching import MAX_IMAGE_PIXELS, EventFaceMatcher
from .imaging import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, ImageProcessor, StudioLook
from .storage import StorageBackend, create_storage, normalize_key, safe_filename, validate_id

logger = logging.getLogger(__name__)

VISIBLE_STATUSES = ("READY", "SUBMITTED_FOR_ALBUM")
PENDING_STATUSES = ("QUEUED", "PROCESSING", "PROCESSING_FAILED")
ROOT_FOLDER = {"folder_id": "root", "name": "Gallery"}
ALBUM_LINK_DAYS = 30


class AlbumNotReady(Exception):
    """The album-design archive is still being built by a worker."""


class PermanentJobError(Exception):
    """A job failure that retrying cannot fix."""


@dataclass(frozen=True)
class DirectDownload:
    url: str


class PhotoService(AlbumDesignMixin, InvitationMixin):
    def __init__(self, settings: Settings, database: Database | None = None) -> None:
        self.settings = settings
        self.storage: StorageBackend = create_storage(settings)
        self.db = database or Database(settings.database_url, settings.db_pool_size)
        settings.temp_root.mkdir(parents=True, exist_ok=True)
        self.face_matcher = EventFaceMatcher(settings)
        self.processor = ImageProcessor(
            logo_path=settings.logo_path,
            enhancer_model=settings.enhancer_model,
            max_processed_bytes=settings.max_processed_bytes,
            face_finder=self.face_matcher.face_boxes if settings.face_matching_enabled else None,
        )
        self._looks: dict[str, tuple[float, StudioLook]] = {}
        self._looks_lock = threading.Lock()

    # ------------------------------------------------------------------ events and folders

    def create_event(self, event_id: str, photographer_id: str, customer_id: str, event_code: str | None = None) -> dict:
        event_id = validate_id(event_id, "event_id")
        photographer_id = validate_id(photographer_id, "photographer_id")
        customer_id = validate_id(customer_id, "customer_id")
        with self.db.transaction() as connection:
            connection.execute(
                "INSERT INTO fb_data.event (event_id, event_code, photographer_id, customer_id) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT (event_id) DO NOTHING",
                (event_id, event_code or event_id, photographer_id, customer_id),
            )
            event = self._event(connection, event_id)
            if event["photographer_id"] != photographer_id or event["customer_id"] != customer_id:
                raise ValueError("This event ID is already assigned to other users.")
            if event_code and event["event_code"] != event_code:
                connection.execute(
                    "UPDATE fb_data.event SET event_code = %s WHERE event_id = %s", (event_code, event_id)
                )
                event["event_code"] = event_code
            connection.execute(
                "INSERT INTO fb_data.folder (event_id, folder_id, name) VALUES (%s, 'root', 'Gallery') "
                "ON CONFLICT DO NOTHING",
                (event_id,),
            )
        return self._event_dict(event)

    def list_folders(self, event_id: str, photographer_id: str, customer_id: str) -> list[dict]:
        with self.db.transaction() as connection:
            event = self._owned_event(connection, event_id, photographer_id, customer_id)
            return self._folders(connection, event["event_id"])

    def create_folder(self, event_id: str, photographer_id: str, customer_id: str, name: str) -> dict:
        cleaned_name = " ".join(name.split()).strip()
        if not cleaned_name or len(cleaned_name) > 80:
            raise ValueError("Folder name must contain between 1 and 80 characters.")
        folder = {"folder_id": uuid.uuid4().hex, "name": cleaned_name}
        try:
            with self.db.transaction() as connection:
                event = self._owned_event(connection, event_id, photographer_id, customer_id)
                if cleaned_name.casefold() == ROOT_FOLDER["name"].casefold():
                    raise ValueError("A folder with this name already exists.")
                connection.execute(
                    "INSERT INTO fb_data.folder (event_id, folder_id, name) VALUES (%s, %s, %s)",
                    (event["event_id"], folder["folder_id"], cleaned_name),
                )
        except UniqueViolation as exc:
            raise ValueError("A folder with this name already exists.") from exc
        return folder

    # ------------------------------------------------------------------ uploads

    def init_upload(
        self,
        event_id: str,
        photographer_id: str,
        customer_id: str,
        folder_id: str,
        filename: str,
        size_bytes: int,
        content_type: str | None = None,
    ) -> dict:
        """Register an upload and return where the client should send the file bytes."""
        folder_id = validate_id(folder_id or "root", "folder_id")
        original_filename = safe_filename(filename)
        extension = Path(original_filename).suffix.lower()
        if extension not in IMAGE_EXTENSIONS and extension not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported image or video type: {extension or 'unknown'}")
        if size_bytes <= 0:
            raise ValueError("Uploaded file is empty.")
        if size_bytes > self.settings.max_upload_bytes:
            raise ValueError("File exceeds the upload limit.")
        media_type = "video" if extension in VIDEO_EXTENSIONS else "image"
        content_type = _clean_content_type(content_type) or (
            mimetypes.guess_type(original_filename)[0] or "application/octet-stream"
        )
        photo_id = uuid.uuid4().hex

        with self.db.transaction() as connection:
            event = self._owned_event(connection, event_id, photographer_id, customer_id)
            if not connection.execute(
                "SELECT 1 FROM fb_data.folder WHERE event_id = %s AND folder_id = %s",
                (event["event_id"], folder_id),
            ).fetchone():
                raise ValueError("Folder not found for this event.")
            original_key = self.storage.original_key(
                event["photographer_id"], event["event_id"], photo_id, original_filename, folder_id
            )
            processed_key = self.storage.processed_key(event["event_id"], photo_id, folder_id, original_filename)
            connection.execute(
                """
                INSERT INTO fb_data.media (photo_id, event_id, folder_id, media_type, original_filename,
                    content_type, original_key, processed_key, status, size_bytes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'UPLOADING', %s)
                """,
                (photo_id, event["event_id"], folder_id, media_type, original_filename, content_type,
                 original_key, processed_key, size_bytes),
            )

        upload_url = None
        if self.settings.direct_uploads:
            upload_url = self.storage.presign_put(
                original_key, content_type, self.settings.upload_url_ttl_seconds
            )
        return {
            "photo_id": photo_id,
            "media_type": media_type,
            "original_filename": original_filename,
            "upload_method": "PUT",
            "upload_url": upload_url,
            "upload_headers": {"Content-Type": content_type} if upload_url else {},
            "expires_in": self.settings.upload_url_ttl_seconds if upload_url else None,
        }

    def receive_upload_content(
        self, event_id: str, photo_id: str, photographer_id: str, customer_id: str, source: Path
    ) -> None:
        """Store bytes sent through the service when the backend cannot issue direct upload URLs."""
        with self.db.transaction() as connection:
            self._owned_event(connection, event_id, photographer_id, customer_id)
            media = self._media(connection, event_id, photo_id)
            if media["status"] != "UPLOADING":
                raise ValueError("This upload has already been completed.")
        self.storage.upload_file(media["original_key"], source)

    def complete_upload(self, event_id: str, photo_id: str, photographer_id: str, customer_id: str) -> dict:
        """Verify the original arrived, number it, and queue processing. Safe to call twice."""
        with self.db.transaction() as connection:
            self._owned_event(connection, event_id, photographer_id, customer_id)
            media = self._media(connection, event_id, photo_id)
        if media["status"] != "UPLOADING":
            return self._media_dict(media)

        actual_size = self.storage.object_size(media["original_key"])
        if not actual_size:
            raise ValueError("The uploaded file was not received. Upload it again.")
        if actual_size > self.settings.max_upload_bytes:
            self.storage.delete(media["original_key"])
            with self.db.transaction() as connection:
                connection.execute("DELETE FROM fb_data.media WHERE photo_id = %s", (photo_id,))
            raise ValueError("File exceeds the upload limit.")

        with self.db.transaction() as connection:
            locked = connection.execute(
                "SELECT status FROM fb_data.media WHERE photo_id = %s FOR UPDATE", (photo_id,)
            ).fetchone()
            if locked and locked["status"] == "UPLOADING":
                number = connection.execute(
                    "UPDATE fb_data.event SET next_photo_number = next_photo_number + 1 "
                    "WHERE event_id = %s RETURNING next_photo_number - 1 AS photo_number",
                    (event_id,),
                ).fetchone()["photo_number"]
                connection.execute(
                    "UPDATE fb_data.media SET status = 'QUEUED', photo_number = %s, size_bytes = %s, "
                    "updated_at = now() WHERE photo_id = %s",
                    (number, actual_size, photo_id),
                )
                jobs.enqueue(connection, jobs.PROCESS_MEDIA, {"photo_id": photo_id},
                             self.settings.job_max_attempts)
            media = self._media(connection, event_id, photo_id)
        return self._media_dict(media)

    def ingest_file(
        self,
        source: Path,
        original_filename: str,
        event_id: str,
        photographer_id: str,
        customer_id: str,
        folder_id: str = "root",
    ) -> dict:
        """Server-side upload (legacy multipart endpoint and CLI): store the original, then queue it."""
        grant = self.init_upload(
            event_id, photographer_id, customer_id, folder_id, original_filename, source.stat().st_size
        )
        with self.db.transaction() as connection:
            media = self._media(connection, event_id, grant["photo_id"])
        self.storage.upload_file(media["original_key"], source)
        return self.complete_upload(event_id, grant["photo_id"], photographer_id, customer_id)

    # ------------------------------------------------------------------ worker jobs

    def run_job(self, kind: str, payload: dict) -> None:
        if kind == jobs.PROCESS_MEDIA:
            self.process_media(payload["photo_id"])
        elif kind == jobs.SCAN_FACES:
            self.scan_faces(payload["photo_id"])
        elif kind == jobs.BUILD_ALBUM:
            self.build_album(payload["submission_id"])
        elif kind == jobs.DESIGN_ALBUM:
            self.design_album(payload["design_id"])
        elif kind == jobs.RENDER_INVITATION:
            self.render_invitation(payload["event_id"])
        else:
            raise PermanentJobError(f"Unknown job kind: {kind}")

    def job_failed(self, kind: str, payload: dict, error: str) -> None:
        """Called once a job has exhausted its retries."""
        with self.db.transaction() as connection:
            if kind == jobs.PROCESS_MEDIA:
                connection.execute(
                    "UPDATE fb_data.media SET status = 'PROCESSING_FAILED', error = %s, updated_at = now() "
                    "WHERE photo_id = %s AND status IN ('QUEUED', 'PROCESSING')",
                    (error[:500], payload.get("photo_id")),
                )
            elif kind == jobs.BUILD_ALBUM:
                row = connection.execute(
                    "UPDATE fb_data.album_submission SET status = 'FAILED', error = %s "
                    "WHERE submission_id = %s RETURNING event_id",
                    (error[:500], payload.get("submission_id")),
                ).fetchone()
                if row:
                    jobs.publish(connection, jobs.OUTBOX_ALBUM_FAILED, row["event_id"],
                                 {"submission_id": payload.get("submission_id"), "error": error[:500]})
            elif kind == jobs.DESIGN_ALBUM:
                connection.execute(
                    "UPDATE fb_data.album_design SET status = 'FAILED', error = %s WHERE design_id = %s",
                    (error[:500], payload.get("design_id")),
                )
            elif kind == jobs.RENDER_INVITATION:
                connection.execute(
                    "UPDATE fb_data.invitation SET status = 'FAILED', error = %s WHERE event_id = %s AND status = 'RENDERING'",
                    (error[:500], payload.get("event_id")),
                )

    def process_media(self, photo_id: str) -> None:
        photo_id = validate_id(photo_id, "photo_id")
        with self.db.transaction() as connection:
            media = connection.execute(
                "UPDATE fb_data.media SET status = 'PROCESSING', updated_at = now() "
                "WHERE photo_id = %s AND status IN ('QUEUED', 'PROCESSING') RETURNING *",
                (photo_id,),
            ).fetchone()
        if not media:
            logger.info("Skipping media %s; it is no longer waiting for processing.", photo_id)
            return

        if media["media_type"] == "video":
            self.storage.copy(media["original_key"], media["processed_key"])
            result = {"width": None, "height": None, "bytes": media["size_bytes"],
                      "watermarked": False, "ai_enhanced": False}
            embeddings = None
        else:
            source, source_cleanup = self.storage.materialize(
                media["original_key"], suffix=Path(media["original_filename"]).suffix
            )
            with tempfile.NamedTemporaryFile(
                prefix="processed-", suffix=".jpg", dir=self.settings.temp_root, delete=False
            ) as temporary:
                processed_path = Path(temporary.name)
            photographer_id = self._event_owner(media["event_id"])
            try:
                result = self.processor.process(source, processed_path, self.studio_look(photographer_id))
                self._record_baseline(photographer_id, result.get("analysis"))
                self.storage.upload_file(media["processed_key"], processed_path)
                embeddings = self._embeddings_or_none(processed_path)
            finally:
                processed_path.unlink(missing_ok=True)
                if source_cleanup:
                    source.unlink(missing_ok=True)

        with self.db.transaction() as connection:
            connection.execute(
                """
                UPDATE fb_data.media
                SET status = 'READY', width = %s, height = %s, processed_bytes = %s,
                    watermarked = %s, ai_enhanced = %s, error = NULL, updated_at = now()
                WHERE photo_id = %s
                """,
                (result["width"], result["height"], result["bytes"], bool(result["watermarked"]),
                 bool(result["ai_enhanced"]), photo_id),
            )
            if embeddings is not None:
                self._store_embeddings_and_match(connection, media, embeddings)

    # ------------------------------------------------------------------ studio look (editing style + logo)

    LOOK_CACHE_SECONDS = 60
    MAX_REFERENCE_BYTES = 40 * 1024 * 1024
    PREVIEW_SIDE = 1800

    def _event_owner(self, event_id: str) -> str | None:
        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT photographer_id FROM fb_data.event WHERE event_id = %s", (event_id,)
            ).fetchone()
        return row["photographer_id"] if row else None

    def studio_look(self, photographer_id: str | None) -> StudioLook:
        """The studio's look and logo, cached briefly so a large upload doesn't hit the database per photo."""
        if not photographer_id:
            return StudioLook()
        now = time.monotonic()
        with self._looks_lock:
            cached = self._looks.get(photographer_id)
            if cached and cached[0] > now:
                return cached[1]
        row = self._style_row(photographer_id)
        look = StudioLook(
            profile=row["profile"] if row and row["apply_style"] else None,
            baseline=row["baseline"] if row else None,
            strength=(row["strength"] if row else 80) / 100.0,
            logo=self._studio_logo(photographer_id),
            stamp_logo=row["stamp_logo"] if row else True,
        )
        with self._looks_lock:
            self._looks[photographer_id] = (now + self.LOOK_CACHE_SECONDS, look)
        return look

    def _forget_look(self, photographer_id: str) -> None:
        with self._looks_lock:
            self._looks.pop(photographer_id, None)

    def _style_row(self, photographer_id: str) -> dict | None:
        with self.db.transaction() as connection:
            return connection.execute(
                "SELECT * FROM fb_data.studio_style WHERE photographer_id = %s", (photographer_id,)
            ).fetchone()

    def _studio_logo(self, photographer_id: str) -> bytes | None:
        """The logo the studio uploaded in FrameBucket's branding settings (owned by the Java app)."""
        try:
            with self.db.transaction() as connection:
                row = connection.execute(
                    "SELECT logo FROM public.fb_studio_brand WHERE photographer_id::text = %s",
                    (photographer_id,),
                ).fetchone()
        except Exception:  # the branding table is absent when the service runs on its own
            logger.debug("Studio branding is unavailable; using the default logo.", exc_info=True)
            return None
        return bytes(row["logo"]) if row and row["logo"] else None

    def _record_baseline(self, photographer_id: str | None, analysis: dict | None) -> None:
        if not photographer_id or not analysis:
            return
        try:
            with self.db.transaction() as connection:
                row = connection.execute(
                    "SELECT baseline FROM fb_data.studio_style WHERE photographer_id = %s FOR UPDATE",
                    (photographer_id,),
                ).fetchone()
                baseline = style_engine.update_baseline(row["baseline"] if row else None, analysis)
                connection.execute(
                    """
                    INSERT INTO fb_data.studio_style (photographer_id, baseline) VALUES (%s, %s)
                    ON CONFLICT (photographer_id) DO UPDATE SET baseline = EXCLUDED.baseline
                    """,
                    (photographer_id, Jsonb(baseline)),
                )
        except Exception:  # statistics must never fail a photo
            logger.warning("Could not update the editing baseline for %s.", photographer_id, exc_info=True)

    def studio_style(self, photographer_id: str) -> dict:
        photographer_id = validate_id(photographer_id, "photographer_id")
        row = self._style_row(photographer_id)
        profile = row["profile"] if row else None
        return {
            "has_style": bool(profile),
            "reference_count": row["reference_count"] if row else 0,
            "strength": row["strength"] if row else 80,
            "apply_style": row["apply_style"] if row else True,
            "stamp_logo": row["stamp_logo"] if row else True,
            "summary": profile.get("summary") if profile else None,
            "baseline_samples": int((row["baseline"] or {}).get("samples", 0)) if row else 0,
            "has_logo": self._studio_logo(photographer_id) is not None or self.processor.logo_path is not None,
            "updated_at": row["updated_at"].isoformat() if row else None,
        }

    def save_style_references(self, photographer_id: str, references: list[bytes]) -> dict:
        photographer_id = validate_id(photographer_id, "photographer_id")
        images = [style_engine.load_reference(data) for data in references]
        profile = style_engine.build_profile(images)
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO fb_data.studio_style (photographer_id, profile, reference_count, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (photographer_id) DO UPDATE
                SET profile = EXCLUDED.profile, reference_count = EXCLUDED.reference_count,
                    apply_style = TRUE, updated_at = now()
                """,
                (photographer_id, Jsonb(profile), len(images)),
            )
        self._forget_look(photographer_id)
        return self.studio_style(photographer_id)

    def update_style_settings(
        self, photographer_id: str, strength: int | None, apply_style: bool | None, stamp_logo: bool | None
    ) -> dict:
        photographer_id = validate_id(photographer_id, "photographer_id")
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO fb_data.studio_style (photographer_id, strength, apply_style, stamp_logo, updated_at)
                VALUES (%s, COALESCE(%s, 80), COALESCE(%s, TRUE), COALESCE(%s, TRUE), now())
                ON CONFLICT (photographer_id) DO UPDATE
                SET strength = COALESCE(%s, fb_data.studio_style.strength),
                    apply_style = COALESCE(%s, fb_data.studio_style.apply_style),
                    stamp_logo = COALESCE(%s, fb_data.studio_style.stamp_logo),
                    updated_at = now()
                """,
                (photographer_id, strength, apply_style, stamp_logo, strength, apply_style, stamp_logo),
            )
        self._forget_look(photographer_id)
        return self.studio_style(photographer_id)

    def clear_style(self, photographer_id: str) -> dict:
        photographer_id = validate_id(photographer_id, "photographer_id")
        with self.db.transaction() as connection:
            connection.execute(
                "UPDATE fb_data.studio_style SET profile = NULL, reference_count = 0, updated_at = now() "
                "WHERE photographer_id = %s",
                (photographer_id,),
            )
        self._forget_look(photographer_id)
        return self.studio_style(photographer_id)

    def style_preview(self, photographer_id: str, filename: str, data: bytes, strength: int | None = None) -> bytes:
        """Render one sample photo exactly as processing would (look + logo), at preview size."""
        photographer_id = validate_id(photographer_id, "photographer_id")
        suffix = Path(filename or "sample.jpg").suffix.lower() or ".jpg"
        if suffix not in IMAGE_EXTENSIONS:
            raise ValueError("Preview needs a JPG, PNG, WebP, TIFF or RAW photo.")
        row = self._style_row(photographer_id)
        if not row or not row["profile"]:
            raise ValueError("Upload reference photos first to create the studio look.")
        with tempfile.NamedTemporaryFile(prefix="preview-", suffix=suffix, dir=self.settings.temp_root,
                                         delete=False) as temporary:
            temporary.write(data)
            source = Path(temporary.name)
        try:
            image = self.processor._load(source)
        except Exception as exc:
            raise ValueError("That photo could not be opened.") from exc
        finally:
            source.unlink(missing_ok=True)
        image = style_engine.downscale(image, self.PREVIEW_SIDE)
        look = StudioLook(
            profile=row["profile"],
            baseline=row["baseline"],
            strength=(row["strength"] if strength is None else max(0, min(100, strength))) / 100.0,
            logo=self._studio_logo(photographer_id),
            stamp_logo=row["stamp_logo"],
        )
        rendered, _ = self.processor.render(image, look)
        stream = BytesIO()
        rendered.save(stream, format="JPEG", quality=88, optimize=True, progressive=True)
        return stream.getvalue()

    def scan_faces(self, photo_id: str) -> None:
        """Compute face embeddings for an already processed photo (used for photos imported from JSON)."""
        with self.db.transaction() as connection:
            media = connection.execute(
                "SELECT * FROM fb_data.media WHERE photo_id = %s", (photo_id,)
            ).fetchone()
        if (
            not media or media["media_type"] != "image" or media["status"] not in VISIBLE_STATUSES
            or media["face_embeddings"] is not None
        ):
            return
        path, cleanup = self.storage.materialize(self._processed_key(media), suffix=".jpg")
        try:
            embeddings = self._embeddings_or_none(path)
        finally:
            if cleanup:
                path.unlink(missing_ok=True)
        if embeddings is None:
            raise RuntimeError("Face models are unavailable; the scan will be retried.")
        with self.db.transaction() as connection:
            self._store_embeddings_and_match(connection, media, embeddings)

    def _embeddings_or_none(self, processed_path: Path) -> list[list[float]] | None:
        if not self.settings.face_matching_enabled:
            return None
        try:
            return self.face_matcher.face_embeddings(processed_path.read_bytes())
        except Exception:
            logger.exception("Face scan failed for %s; the photo stays available in the gallery.", processed_path.name)
            return None

    def _store_embeddings_and_match(self, connection: Connection, media: dict, embeddings: list) -> None:
        # Serialise with guest registration for the same event so a guest registering while this
        # photo is processed is matched either here or by the registration, never by neither.
        self._lock_event_faces(connection, media["event_id"])
        connection.execute(
            "UPDATE fb_data.media SET face_embeddings = %s WHERE photo_id = %s",
            (Jsonb(embeddings), media["photo_id"]),
        )
        guests = {
            row["guest_id"]: row["embedding"]
            for row in connection.execute(
                "SELECT guest_id, embedding FROM fb_data.face_guest WHERE event_id = %s", (media["event_id"],)
            )
        }
        number = connection.execute(
            "SELECT photo_number FROM fb_data.media WHERE photo_id = %s", (media["photo_id"],)
        ).fetchone()["photo_number"]
        for guest_id in self.face_matcher.matching_guest_embeddings(embeddings, guests):
            jobs.publish(connection, jobs.OUTBOX_FACE_MATCH, media["event_id"], {
                "guest_id": guest_id,
                "photo_id": media["photo_id"],
                "photo_number": number,
            })

    def cleanup_abandoned_uploads(self) -> int:
        with self.db.transaction() as connection:
            rows = connection.execute(
                "DELETE FROM fb_data.media WHERE status = 'UPLOADING' "
                "AND created_at < now() - make_interval(hours => %s) RETURNING original_key",
                (self.settings.abandoned_upload_hours,),
            ).fetchall()
        for row in rows:
            try:
                self.storage.delete(row["original_key"])
            except Exception:
                logger.exception("Could not delete abandoned upload %s", row["original_key"])
        return len(rows)

    # ------------------------------------------------------------------ guest face matching

    def register_guest_face(self, event_id: str, guest_id: str, customer_id: str, selfie_bytes: bytes) -> dict:
        event_id = validate_id(event_id, "event_id")
        guest_id = validate_id(guest_id, "guest_id")
        customer_id = validate_id(customer_id, "customer_id")
        if not self.settings.face_matching_enabled:
            raise ValueError("Event face matching is disabled on the Python data service.")
        with self.db.transaction() as connection:
            event = self._event(connection, event_id)
            if event["customer_id"] != customer_id:
                raise ValueError("The event does not belong to the supplied customer.")

        embedding = self.face_matcher.selfie_embedding(selfie_bytes)
        with self.db.transaction() as connection:
            self._lock_event_faces(connection, event_id)
            connection.execute(
                "INSERT INTO fb_data.face_guest (event_id, guest_id, embedding) VALUES (%s, %s, %s) "
                "ON CONFLICT (event_id, guest_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                (event_id, guest_id, Jsonb(embedding)),
            )
            matched: list[str] = []
            unscanned: list[str] = []
            for row in connection.execute(
                "SELECT photo_id, face_embeddings FROM fb_data.media WHERE event_id = %s "
                "AND media_type = 'image' AND status = ANY(%s) ORDER BY photo_number",
                (event_id, list(VISIBLE_STATUSES)),
            ):
                if row["face_embeddings"] is None:
                    unscanned.append(row["photo_id"])
                elif self.face_matcher.matching_guest_embeddings(row["face_embeddings"], {guest_id: embedding}):
                    matched.append(row["photo_id"])
            # Photos without stored embeddings are scanned by workers; matches arrive via the outbox.
            for photo_id in unscanned:
                jobs.enqueue(connection, jobs.SCAN_FACES, {"photo_id": photo_id}, self.settings.job_max_attempts)
        return {"guest_id": guest_id, "matched_photo_ids": matched, "pending_scan_count": len(unscanned)}

    def delete_guest_face(self, event_id: str, guest_id: str, customer_id: str) -> None:
        event_id = validate_id(event_id, "event_id")
        guest_id = validate_id(guest_id, "guest_id")
        customer_id = validate_id(customer_id, "customer_id")
        with self.db.transaction() as connection:
            event = self._event(connection, event_id)
            if event["customer_id"] != customer_id:
                raise ValueError("The event does not belong to the supplied customer.")
            connection.execute(
                "DELETE FROM fb_data.face_guest WHERE event_id = %s AND guest_id = %s", (event_id, guest_id)
            )
            connection.execute(
                "DELETE FROM fb_data.outbox WHERE event_id = %s AND kind = %s AND delivered_at IS NULL "
                "AND payload->>'guest_id' = %s",
                (event_id, jobs.OUTBOX_FACE_MATCH, guest_id),
            )

    @staticmethod
    def _lock_event_faces(connection: Connection, event_id: str) -> None:
        connection.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("fb_data.faces:" + event_id,))

    # ------------------------------------------------------------------ album selection and submission

    def album_selection(self, event_id: str, photographer_id: str, customer_id: str) -> dict:
        with self.db.transaction() as connection:
            event = self._owned_event(connection, event_id, photographer_id, customer_id)
            row = connection.execute(
                "SELECT photo_ids FROM fb_data.album_selection WHERE event_id = %s AND customer_id = %s",
                (event["event_id"], customer_id),
            ).fetchone()
            if not row:
                return {"photo_ids": []}
            available = {
                item["photo_id"]
                for item in connection.execute(
                    "SELECT photo_id FROM fb_data.media WHERE event_id = %s AND media_type = 'image' "
                    "AND status = ANY(%s)",
                    (event["event_id"], list(VISIBLE_STATUSES)),
                )
            }
        return {"photo_ids": [photo_id for photo_id in row["photo_ids"] if photo_id in available]}

    def save_album_selection(self, event_id: str, photographer_id: str, customer_id: str, photo_ids: list[str]) -> dict:
        unique_photo_ids = list(dict.fromkeys(photo_ids))
        if len(unique_photo_ids) > 50000:
            raise ValueError("A selection can contain at most 50000 photos.")
        with self.db.transaction() as connection:
            event = self._owned_event(connection, event_id, photographer_id, customer_id)
            if not unique_photo_ids:
                connection.execute(
                    "DELETE FROM fb_data.album_selection WHERE event_id = %s AND customer_id = %s",
                    (event["event_id"], customer_id),
                )
                return {"photo_ids": [], "photo_count": 0}
            rows = {
                row["photo_id"]: row
                for row in connection.execute(
                    "SELECT photo_id, status, media_type FROM fb_data.media "
                    "WHERE event_id = %s AND photo_id = ANY(%s)",
                    (event["event_id"], unique_photo_ids),
                )
            }
            for photo_id in unique_photo_ids:
                row = rows.get(photo_id)
                # Photos sent in an earlier album submission can be chosen again.
                if not row or row["status"] not in VISIBLE_STATUSES:
                    raise ValueError(f"Photo {photo_id} is not available for selection.")
                if row["media_type"] != "image":
                    raise ValueError("Videos cannot be selected for album design.")
            connection.execute(
                """
                INSERT INTO fb_data.album_selection (event_id, customer_id, photo_ids, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (event_id, customer_id)
                DO UPDATE SET photo_ids = EXCLUDED.photo_ids, updated_at = now()
                """,
                (event["event_id"], customer_id, Jsonb(unique_photo_ids)),
            )
        return {"photo_ids": unique_photo_ids, "photo_count": len(unique_photo_ids)}

    def submit_album(self, event_id: str, photographer_id: str, customer_id: str, photo_ids: list[str]) -> dict:
        """Reserve the selection and return the album link now; a worker builds the ZIP."""
        unique_photo_ids = list(dict.fromkeys(photo_ids))
        if not unique_photo_ids:
            raise ValueError("Select at least one photo for album design.")
        submission_id = uuid.uuid4().hex
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(days=ALBUM_LINK_DAYS)
        with self.db.transaction() as connection:
            event = self._owned_event(connection, event_id, photographer_id, customer_id)
            rows = {
                row["photo_id"]: row
                for row in connection.execute(
                    "SELECT * FROM fb_data.media WHERE event_id = %s AND photo_id = ANY(%s) FOR UPDATE",
                    (event["event_id"], unique_photo_ids),
                )
            }
            selected = []
            for photo_id in unique_photo_ids:
                row = rows.get(photo_id)
                if not row:
                    raise ValueError(f"Photo {photo_id} does not belong to this event.")
                if row["status"] not in VISIBLE_STATUSES:
                    raise ValueError(f"Photo {photo_id} is not ready for album design.")
                if row["media_type"] != "image":
                    raise ValueError("Videos cannot be submitted for album design.")
                selected.append(row)
            photo_numbers = [
                {"photo_number": row["photo_number"], "original_filename": row["original_filename"],
                 "folder_id": row["folder_id"]}
                for row in selected
            ]
            connection.execute(
                """
                INSERT INTO fb_data.album_submission (submission_id, event_id, photographer_id, customer_id,
                    token_hash, archive_key, status, photo_ids, photo_numbers, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, 'PREPARING', %s, %s, %s)
                """,
                (submission_id, event["event_id"], photographer_id, customer_id, _token_hash(token),
                 f"album_design/{event['event_id']}/{submission_id}/album-design.zip",
                 Jsonb(unique_photo_ids), Jsonb(photo_numbers), expires_at),
            )
            connection.execute(
                "UPDATE fb_data.media SET status = 'SUBMITTED_FOR_ALBUM', album_submission_id = %s, "
                "updated_at = now() WHERE photo_id = ANY(%s)",
                (submission_id, unique_photo_ids),
            )
            connection.execute(
                "DELETE FROM fb_data.album_selection WHERE event_id = %s AND customer_id = %s",
                (event["event_id"], customer_id),
            )
            jobs.enqueue(connection, jobs.BUILD_ALBUM, {"submission_id": submission_id},
                         self.settings.job_max_attempts)
        return {
            "submission_id": submission_id,
            "event_id": event_id,
            "photo_count": len(selected),
            "photo_numbers": photo_numbers,
            "album_design_link": f"{self.settings.public_base_url}/api/v1/album-design/{token}",
            "expires_at": expires_at.isoformat(),
            "archive_status": "PREPARING",
            "status": "SUBMITTED",
        }

    def build_album(self, submission_id: str) -> None:
        with self.db.transaction() as connection:
            submission = connection.execute(
                "SELECT * FROM fb_data.album_submission WHERE submission_id = %s", (submission_id,)
            ).fetchone()
            if not submission or submission["status"] != "PREPARING":
                return
            media = {
                row["photo_id"]: row
                for row in connection.execute(
                    "SELECT * FROM fb_data.media WHERE photo_id = ANY(%s)", (list(submission["photo_ids"]),)
                )
            }
        missing = [photo_id for photo_id in submission["photo_ids"] if photo_id not in media]
        if missing:
            raise PermanentJobError("Some selected photos no longer exist.")

        self.settings.temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="album-design-", suffix=".zip", dir=self.settings.temp_root, delete=False
        ) as temporary:
            archive_path = Path(temporary.name)
        try:
            # Stage one original at a time so peak temporary disk is the ZIP plus one source file.
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for photo_id in submission["photo_ids"]:
                    row = media[photo_id]
                    source_key = self._original_key(row, submission["photographer_id"])
                    try:
                        source, cleanup = self.storage.materialize(
                            source_key, suffix=Path(row["original_filename"]).suffix
                        )
                    except FileNotFoundError as exc:
                        raise PermanentJobError(f"Original for photo {row['photo_number']} is missing.") from exc
                    try:
                        archive.write(
                            source, arcname=safe_filename(f"{row['photo_number']:05d}_{row['original_filename']}")
                        )
                    finally:
                        if cleanup:
                            source.unlink(missing_ok=True)
            self.storage.upload_file(submission["archive_key"], archive_path)
        finally:
            archive_path.unlink(missing_ok=True)

        with self.db.transaction() as connection:
            connection.execute(
                "UPDATE fb_data.album_submission SET status = 'READY', ready_at = now(), error = NULL "
                "WHERE submission_id = %s",
                (submission_id,),
            )
            jobs.publish(connection, jobs.OUTBOX_ALBUM_READY, submission["event_id"], {
                "submission_id": submission_id,
                "photo_count": len(submission["photo_ids"]),
                "expires_at": submission["expires_at"].isoformat(),
            })

    def resolve_album_link(self, token: str) -> tuple[Path | DirectDownload, str, bool]:
        if not token or len(token) > 200:
            raise FileNotFoundError("Album design link not found.")
        with self.db.transaction() as connection:
            submission = connection.execute(
                "SELECT s.*, e.event_code FROM fb_data.album_submission s "
                "JOIN fb_data.event e ON e.event_id = s.event_id WHERE s.token_hash = %s",
                (_token_hash(token),),
            ).fetchone()
        if not submission:
            raise FileNotFoundError("Album design link not found.")
        if submission["expires_at"] <= datetime.now(timezone.utc):
            raise FileNotFoundError("Album design link has expired.")
        if submission["status"] == "PREPARING":
            raise AlbumNotReady("The album photos are still being packaged. Try this link again in a few minutes.")
        if submission["status"] != "READY":
            raise FileNotFoundError("The album design archive could not be prepared. Ask the customer to contact you.")
        archive_key = normalize_key(submission["archive_key"])
        filename = f"FrameBucket-{safe_filename(submission['event_code'])}-album-design.zip"
        url = self.storage.presign_get(
            archive_key, self.settings.media_url_ttl_seconds, filename=filename, attachment=True,
            content_type="application/zip",
        )
        if url:
            return DirectDownload(url), filename, False
        try:
            archive, cleanup = self.storage.materialize(archive_key, suffix=".zip", prefix="fb-album-")
        except (FileNotFoundError, OSError) as exc:
            raise FileNotFoundError("Album design archive not found.") from exc
        return archive, filename, cleanup

    # ------------------------------------------------------------------ gallery reads

    def list_gallery(
        self, event_id: str, customer_id: str, folder_id: str | None = None, include_pending: bool = False
    ) -> list[dict]:
        statuses = list(VISIBLE_STATUSES) + (list(PENDING_STATUSES) if include_pending else [])
        with self.db.transaction() as connection:
            event = self._event(connection, validate_id(event_id, "event_id"))
            if event["customer_id"] != validate_id(customer_id, "customer_id"):
                raise ValueError("Customer does not have access to this event.")
            query = (
                "SELECT * FROM fb_data.media WHERE event_id = %s AND status = ANY(%s)"
                + (" AND folder_id = %s" if folder_id is not None else "")
                + " ORDER BY photo_number"
            )
            params = (event["event_id"], statuses) + ((folder_id,) if folder_id is not None else ())
            return [self._media_dict(row) for row in connection.execute(query, params)]

    def media_url(self, event_id: str, photo_id: str, customer_id: str) -> str | None:
        media = self._visible_media(event_id, photo_id, customer_id)
        return self.storage.presign_get(
            self._processed_key(media),
            self.settings.media_url_ttl_seconds,
            content_type=mimetypes.guess_type(media["processed_key"])[0],
        )

    def media_path(self, event_id: str, photo_id: str, customer_id: str) -> tuple[Path, bool]:
        media = self._visible_media(event_id, photo_id, customer_id)
        try:
            return self.storage.materialize(self._processed_key(media))
        except (FileNotFoundError, OSError) as exc:
            raise ValueError("Processed photo not found.") from exc

    def preview_path(self, event_id: str, photo_id: str, customer_id: str) -> tuple[Path, bool]:
        return self.media_path(event_id, photo_id, customer_id)

    def original_url(self, event_id: str, photo_id: str, photographer_id: str, customer_id: str) -> str | None:
        media = self._photographer_media(event_id, photo_id, photographer_id, customer_id)
        return self.storage.presign_get(
            self._original_key(media, photographer_id),
            self.settings.media_url_ttl_seconds,
            filename=media["original_filename"],
            attachment=True,
        )

    def original_path_for_photographer(
        self, event_id: str, photo_id: str, photographer_id: str, customer_id: str
    ) -> tuple[Path, bool, str]:
        media = self._photographer_media(event_id, photo_id, photographer_id, customer_id)
        try:
            path, cleanup = self.storage.materialize(
                self._original_key(media, photographer_id), suffix=Path(media["original_filename"]).suffix
            )
        except (FileNotFoundError, OSError) as exc:
            raise ValueError("Photographer original not found.") from exc
        return path, cleanup, safe_filename(media["original_filename"])

    def whatsapp_preview_bytes(self, event_id: str, photo_id: str, customer_id: str) -> bytes:
        media = self._visible_media(event_id, photo_id, customer_id)
        if media["media_type"] == "video":
            raise ValueError("The requested event photo is not available.")
        path, cleanup = self.storage.materialize(self._processed_key(media), suffix=".jpg")
        try:
            max_bytes = 4_800_000
            if path.stat().st_size <= max_bytes:
                return path.read_bytes()
            with Image.open(path) as source:
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ValueError("The processed photo is too large to prepare for WhatsApp.")
                image = ImageOps.exif_transpose(source).convert("RGB")
            for max_side in (None, 4000, 3200, 2560, 2048):
                candidate = image
                if max_side and max(image.size) > max_side:
                    ratio = max_side / max(image.size)
                    candidate = image.resize(
                        (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
                        Image.Resampling.LANCZOS,
                    )
                for quality in (94, 91, 88, 85, 82, 78):
                    output = BytesIO()
                    candidate.save(output, format="JPEG", quality=quality, optimize=True, progressive=True)
                    result = output.getvalue()
                    if len(result) <= max_bytes:
                        return result
            raise ValueError("The photo could not be reduced to WhatsApp's image-size limit.")
        finally:
            if cleanup:
                path.unlink(missing_ok=True)

    def download_gallery(
        self, event_id: str, customer_id: str, photo_ids: list[str] | None = None
    ) -> tuple[Path, str, str, bool]:
        available = self.list_gallery(event_id, customer_id)
        if photo_ids:
            requested_ids = list(dict.fromkeys(photo_ids))
            available_by_id = {photo["photo_id"]: photo for photo in available}
            if any(photo_id not in available_by_id for photo_id in requested_ids):
                raise ValueError("One or more selected photos are not available for download.")
            selected = [available_by_id[photo_id] for photo_id in requested_ids]
        else:
            selected = available
        if not selected:
            raise ValueError("There are no gallery photos available to download.")

        with self.db.transaction() as connection:
            event = self._event(connection, event_id)
            folder_names = {folder["folder_id"]: folder["name"] for folder in self._folders(connection, event_id)}
            keys = {
                row["photo_id"]: row
                for row in connection.execute(
                    "SELECT photo_id, event_id, processed_key FROM fb_data.media WHERE photo_id = ANY(%s)",
                    ([photo["photo_id"] for photo in selected],),
                )
            }

        def stage(photo: dict) -> tuple[Path, bool]:
            try:
                return self.storage.materialize(self._processed_key(keys[photo["photo_id"]]))
            except (FileNotFoundError, OSError, KeyError) as exc:
                raise ValueError("A processed gallery photo is unavailable for download.") from exc

        if len(selected) == 1:
            photo = selected[0]
            path, cleanup = stage(photo)
            filename = safe_filename(
                f"{photo['photo_number']:05d}_{Path(photo['original_filename']).stem}{path.suffix or '.jpg'}"
            )
            return path, filename, mimetypes.guess_type(path.name)[0] or "application/octet-stream", cleanup

        self.settings.temp_root.mkdir(parents=True, exist_ok=True)
        file_descriptor, archive_name = tempfile.mkstemp(
            prefix="customer-gallery-", suffix=".zip", dir=self.settings.temp_root
        )
        archive_path = Path(archive_name)
        try:
            with os.fdopen(file_descriptor, "wb") as archive_file:
                with zipfile.ZipFile(archive_file, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                    for photo in selected:
                        path, cleanup = stage(photo)
                        try:
                            folder_name = safe_filename(folder_names.get(photo["folder_id"], "Gallery"))
                            entry_name = safe_filename(
                                f"{photo['photo_number']:05d}_{Path(photo['original_filename']).stem}"
                                f"{path.suffix or '.jpg'}"
                            )
                            archive.write(path, f"{folder_name}/{entry_name}")
                        finally:
                            if cleanup:
                                path.unlink(missing_ok=True)
            event_code = safe_filename(event["event_code"])
            return archive_path, f"FrameBucket-{event_code}-gallery.zip", "application/zip", True
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------ CLI batch helper

    def process_directory(
        self, input_directory: Path, event_id: str, photographer_id: str, customer_id: str, workers: int = 2
    ) -> list[dict]:
        """Ingest every image in a folder and process it in this process (no worker needed)."""
        if not input_directory.is_dir():
            raise FileNotFoundError(f"Input folder does not exist: {input_directory}")
        self.create_event(event_id, photographer_id, customer_id)
        sources = sorted(
            (path for path in input_directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: path.name.lower(),
        )
        results = []
        for path in sources:
            try:
                media = self.ingest_file(path, path.name, event_id, photographer_id, customer_id)
                self.process_media(media["photo_id"])
                results.append(self._media_dict(self._media_by_id(media["photo_id"])))
            except Exception as exc:
                results.append({"original_filename": path.name, "status": "PROCESSING_FAILED", "error": str(exc)})
        return results

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _event(connection: Connection, event_id: str) -> dict:
        event = connection.execute(
            "SELECT * FROM fb_data.event WHERE event_id = %s", (validate_id(event_id, "event_id"),)
        ).fetchone()
        if not event:
            raise ValueError("Event not found.")
        return event

    def _owned_event(self, connection: Connection, event_id: str, photographer_id: str, customer_id: str) -> dict:
        photographer_id = validate_id(photographer_id, "photographer_id")
        customer_id = validate_id(customer_id, "customer_id")
        event = self._event(connection, event_id)
        if event["photographer_id"] != photographer_id or event["customer_id"] != customer_id:
            raise ValueError("The event does not belong to the supplied photographer/customer.")
        return event

    @staticmethod
    def _media(connection: Connection, event_id: str, photo_id: str) -> dict:
        media = connection.execute(
            "SELECT * FROM fb_data.media WHERE event_id = %s AND photo_id = %s",
            (validate_id(event_id, "event_id"), validate_id(photo_id, "photo_id")),
        ).fetchone()
        if not media:
            raise ValueError("Media not found for this event.")
        return media

    def _media_by_id(self, photo_id: str) -> dict:
        with self.db.transaction() as connection:
            return connection.execute("SELECT * FROM fb_data.media WHERE photo_id = %s", (photo_id,)).fetchone()

    def _visible_media(self, event_id: str, photo_id: str, customer_id: str) -> dict:
        with self.db.transaction() as connection:
            event = self._event(connection, event_id)
            if event["customer_id"] != validate_id(customer_id, "customer_id"):
                raise ValueError("Customer does not have access to this event.")
            media = self._media(connection, event_id, photo_id)
        if media["status"] not in VISIBLE_STATUSES:
            raise ValueError("Processed photo not found.")
        return media

    def _photographer_media(self, event_id: str, photo_id: str, photographer_id: str, customer_id: str) -> dict:
        with self.db.transaction() as connection:
            self._owned_event(connection, event_id, photographer_id, customer_id)
            media = self._media(connection, event_id, photo_id)
        if media["status"] == "UPLOADING":
            raise ValueError("Photographer original not found.")
        return media

    @staticmethod
    def _folders(connection: Connection, event_id: str) -> list[dict]:
        rows = connection.execute(
            "SELECT folder_id, name FROM fb_data.folder WHERE event_id = %s "
            "ORDER BY (folder_id <> 'root'), created_at, name",
            (event_id,),
        ).fetchall()
        return [dict(row) for row in rows] or [dict(ROOT_FOLDER)]

    @staticmethod
    def _processed_key(media: dict) -> str:
        key = normalize_key(media["processed_key"])
        event_id = media["event_id"]
        if not key.startswith((f"events/{event_id}/photos/", f"events/{event_id}/videos/")):
            raise ValueError("Processed photo not found.")
        return key

    @staticmethod
    def _original_key(media: dict, photographer_id: str) -> str:
        key = normalize_key(media["original_key"])
        if not key.startswith(f"photographer/{photographer_id}/{media['event_id']}/originals/"):
            raise ValueError("Photographer original not found.")
        return key

    @staticmethod
    def _event_dict(event: dict) -> dict:
        return {
            "event_id": event["event_id"],
            "event_code": event["event_code"],
            "photographer_id": event["photographer_id"],
            "customer_id": event["customer_id"],
            "created_at": event["created_at"].isoformat(),
            "customer_gallery_key": f"events/{event['event_id']}",
            "photographer_originals_key": f"photographer/{event['photographer_id']}/{event['event_id']}/originals",
        }

    @staticmethod
    def _media_dict(media: dict) -> dict:
        return {
            "photo_id": media["photo_id"],
            "photo_number": media["photo_number"],
            "original_filename": media["original_filename"],
            "media_type": media["media_type"],
            "folder_id": media["folder_id"],
            "width": media["width"],
            "height": media["height"],
            "size_bytes": media["size_bytes"],
            "status": media["status"],
            "error": media["error"],
            "ai_enhanced": media["ai_enhanced"],
            "watermarked": media["watermarked"],
        }


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _clean_content_type(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().lower()
    if len(value) > 100 or "/" not in value or any(character in value for character in "\r\n;,"):
        return None
    return value
