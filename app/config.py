from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

load_dotenv()


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _google_folder_id(value: str) -> str | None:
    raw = value.strip()
    if not raw:
        return None
    if "://" not in raw:
        return raw
    parsed = urlsplit(raw)
    if parsed.hostname not in {"drive.google.com", "www.drive.google.com"}:
        raise ValueError("FB_GOOGLE_DRIVE_FOLDER_ID must be a Drive folder ID or Drive folder URL.")
    parts = [part for part in parsed.path.split("/") if part]
    try:
        folder_index = parts.index("folders")
        folder_id = parts[folder_index + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("FB_GOOGLE_DRIVE_FOLDER_ID URL must contain /folders/<folder-id>.") from exc
    if not folder_id:
        raise ValueError("FB_GOOGLE_DRIVE_FOLDER_ID URL does not contain a folder ID.")
    return folder_id


@dataclass(frozen=True)
class Settings:
    storage_backend: str
    data_root: Path
    temp_root: Path
    logo_path: Path | None
    enhancer_model: Path | None
    service_token: str | None
    public_base_url: str
    max_upload_bytes: int
    max_processed_bytes: int | None
    s3_endpoint_url: str | None
    s3_bucket: str | None
    s3_region: str | None
    s3_access_key_id: str | None
    s3_secret_access_key: str | None
    s3_prefix: str
    google_drive_folder_id: str | None
    google_client_id: str | None
    google_client_secret: str | None
    google_refresh_token: str | None
    face_matching_enabled: bool
    face_model_dir: Path
    face_detector_model_url: str
    face_recognizer_model_url: str
    face_match_threshold: float
    database_url: str | None
    db_pool_size: int
    direct_uploads: bool
    upload_url_ttl_seconds: int
    media_url_ttl_seconds: int
    worker_concurrency: int
    worker_poll_seconds: float
    job_lease_seconds: int
    job_max_attempts: int
    embedded_worker: bool
    abandoned_upload_hours: int

    @classmethod
    def from_env(cls) -> "Settings":
        default_root = (
            Path.home()
            / "OneDrive"
            / "Documents"
            / "FrameBucket"
            / "data_output"
        )
        logo_raw = os.getenv("FB_LOGO_PATH", "").strip()
        model_raw = os.getenv("FB_ENHANCER_MODEL", "").strip()
        max_output_raw = os.getenv("FB_MAX_PROCESSED_BYTES", "").strip()
        storage_backend = os.getenv("FB_STORAGE_BACKEND", "local").strip().lower()
        data_root = _path_from_env("FB_DATA_ROOT", default_root)
        temp_default = (
            data_root / "tmp"
            if storage_backend == "local"
            else Path(tempfile.gettempdir()) / "framebucket-data"
        )
        temp_root = _path_from_env("FB_TEMP_ROOT", temp_default)
        return cls(
            storage_backend=storage_backend,
            data_root=data_root,
            temp_root=temp_root,
            logo_path=Path(logo_raw).expanduser().resolve() if logo_raw else None,
            enhancer_model=Path(model_raw).expanduser().resolve() if model_raw else None,
            service_token=os.getenv("FB_SERVICE_TOKEN") or None,
            public_base_url=os.getenv("FB_PUBLIC_BASE_URL", "http://127.0.0.1:8001").rstrip("/"),
            max_upload_bytes=int(os.getenv("FB_MAX_UPLOAD_BYTES", str(1024 * 1024 * 1024))),
            max_processed_bytes=int(max_output_raw) if max_output_raw else None,
            s3_endpoint_url=(os.getenv("FB_S3_ENDPOINT_URL") or os.getenv("ENDPOINT") or "").strip() or None,
            s3_bucket=(os.getenv("FB_S3_BUCKET") or os.getenv("BUCKET") or "").strip() or None,
            s3_region=(os.getenv("FB_S3_REGION") or os.getenv("REGION") or "").strip() or None,
            s3_access_key_id=(os.getenv("FB_S3_ACCESS_KEY_ID") or os.getenv("ACCESS_KEY_ID") or "").strip() or None,
            s3_secret_access_key=(
                os.getenv("FB_S3_SECRET_ACCESS_KEY") or os.getenv("SECRET_ACCESS_KEY") or ""
            ).strip() or None,
            s3_prefix=os.getenv("FB_S3_PREFIX", "").strip().strip("/"),
            google_drive_folder_id=_google_folder_id(os.getenv("FB_GOOGLE_DRIVE_FOLDER_ID", "")),
            google_client_id=os.getenv("FB_GOOGLE_CLIENT_ID", "").strip() or None,
            google_client_secret=os.getenv("FB_GOOGLE_CLIENT_SECRET", "").strip() or None,
            google_refresh_token=os.getenv("FB_GOOGLE_REFRESH_TOKEN", "").strip() or None,
            face_matching_enabled=os.getenv("FB_FACE_MATCHING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
            face_model_dir=_path_from_env("FB_FACE_MODEL_DIR", temp_root / "face-models"),
            face_detector_model_url=os.getenv(
                "FB_FACE_DETECTOR_MODEL_URL",
                "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            ).strip(),
            face_recognizer_model_url=os.getenv(
                "FB_FACE_RECOGNIZER_MODEL_URL",
                "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
            ).strip(),
            face_match_threshold=float(os.getenv("FB_FACE_MATCH_THRESHOLD", "0.42")),
            database_url=(os.getenv("FB_DATABASE_URL") or os.getenv("DATABASE_URL") or "").strip() or None,
            db_pool_size=max(2, int(os.getenv("FB_DB_POOL_SIZE", "10"))),
            direct_uploads=_flag("FB_DIRECT_UPLOADS", True),
            upload_url_ttl_seconds=int(os.getenv("FB_UPLOAD_URL_TTL_SECONDS", "3600")),
            media_url_ttl_seconds=int(os.getenv("FB_MEDIA_URL_TTL_SECONDS", "600")),
            worker_concurrency=max(1, int(os.getenv("FB_WORKER_CONCURRENCY", "2"))),
            worker_poll_seconds=float(os.getenv("FB_WORKER_POLL_SECONDS", "2")),
            job_lease_seconds=int(os.getenv("FB_JOB_LEASE_SECONDS", "1800")),
            job_max_attempts=max(1, int(os.getenv("FB_JOB_MAX_ATTEMPTS", "5"))),
            embedded_worker=_flag("FB_EMBEDDED_WORKER", False),
            abandoned_upload_hours=int(os.getenv("FB_ABANDONED_UPLOAD_HOURS", "24")),
        )

    @property
    def input_dir(self) -> Path:
        return self.data_root / "input"

    @property
    def private_root(self) -> Path:
        return self.data_root / "photographer"

    @property
    def event_root(self) -> Path:
        return self.data_root / "events"

    @property
    def album_root(self) -> Path:
        return self.data_root / "album_design"

    @property
    def metadata_root(self) -> Path:
        return self.data_root / "metadata" / "events"
