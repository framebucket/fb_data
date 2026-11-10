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
