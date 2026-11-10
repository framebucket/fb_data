from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    data_root: Path
    logo_path: Path | None
    enhancer_model: Path | None
    service_token: str | None
    public_base_url: str
    max_upload_bytes: int
    max_processed_bytes: int | None

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
        return cls(
            data_root=_path_from_env("FB_DATA_ROOT", default_root),
            logo_path=Path(logo_raw).expanduser().resolve() if logo_raw else None,
            enhancer_model=Path(model_raw).expanduser().resolve() if model_raw else None,
            service_token=os.getenv("FB_SERVICE_TOKEN") or None,
            public_base_url=os.getenv("FB_PUBLIC_BASE_URL", "http://127.0.0.1:8001").rstrip("/"),
            max_upload_bytes=int(os.getenv("FB_MAX_UPLOAD_BYTES", str(1024 * 1024 * 1024))),
            max_processed_bytes=int(max_output_raw) if max_output_raw else None,
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

    @property
    def temp_root(self) -> Path:
        return self.data_root / "tmp"
