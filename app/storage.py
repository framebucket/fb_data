from __future__ import annotations

import io
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path, PurePosixPath

from .config import Settings

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".avi"}


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


def normalize_key(value: str) -> str:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if not path.parts or path == PurePosixPath(".") or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Storage key is invalid.")
    return path.as_posix()


class StorageBackend:
    """Object operations shared by local disk, S3-compatible, and Drive storage."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    def lock_for_event(self, event_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(event_id, threading.RLock())

    def event_key(self, event_id: str) -> str:
        return f"metadata/events/{validate_id(event_id, 'event_id')}/event.json"

    def manifest_key(self, event_id: str) -> str:
        return f"metadata/events/{validate_id(event_id, 'event_id')}/photos.json"

    def folders_key(self, event_id: str) -> str:
        return f"metadata/events/{validate_id(event_id, 'event_id')}/folders.json"

    def face_index_key(self, event_id: str) -> str:
        return f"metadata/events/{validate_id(event_id, 'event_id')}/face-index.json"

    def original_key(
        self, photographer_id: str, event_id: str, photo_id: str,
        filename: str, folder_id: str = "root",
    ) -> str:
        validate_id(photographer_id, "photographer_id")
        validate_id(event_id, "event_id")
        validate_id(photo_id, "photo_id")
        validate_id(folder_id, "folder_id")
        return f"photographer/{photographer_id}/{event_id}/originals/{folder_id}/{photo_id}_{safe_filename(filename)}"

    def processed_key(self, event_id: str, photo_id: str, folder_id: str = "root", filename: str = "") -> str:
        validate_id(event_id, "event_id")
        validate_id(photo_id, "photo_id")
        validate_id(folder_id, "folder_id")
        suffix = Path(filename).suffix.lower()
        if suffix in VIDEO_SUFFIXES:
            return f"events/{event_id}/videos/{folder_id}/{photo_id}{suffix}"
        return f"events/{event_id}/photos/{folder_id}/{photo_id}.jpg"

    def album_selection_key(self, event_id: str, customer_id: str) -> str:
        validate_id(event_id, "event_id")
        validate_id(customer_id, "customer_id")
        return f"tmp/album-selections/{event_id}/{customer_id}.json"

    def album_submission_key(self, event_id: str, submission_id: str, filename: str) -> str:
        validate_id(event_id, "event_id")
        validate_id(submission_id, "submission_id")
        return f"album_design/{event_id}/{submission_id}/{safe_filename(filename)}"

    def prepare_event(self, photographer_id: str, event_id: str) -> None:
        validate_id(photographer_id, "photographer_id")
        validate_id(event_id, "event_id")
        folders_key = self.folders_key(event_id)
        if not self.exists(folders_key):
            self.write_json(folders_key, {"folders": [{"folder_id": "root", "name": "Gallery"}]})

    def read_folders(self, event_id: str) -> list[dict]:
        result = self.read_json(self.folders_key(event_id), {"folders": [{"folder_id": "root", "name": "Gallery"}]})
        return result.get("folders", [])

    def write_folders(self, event_id: str, folders: list[dict]) -> None:
        self.write_json(self.folders_key(event_id), {"folders": folders})

    def read_manifest(self, event_id: str) -> dict:
        return self.read_json(self.manifest_key(event_id), {"event_id": event_id, "photos": []})

    def write_manifest(self, event_id: str, manifest: dict) -> None:
        self.write_json(self.manifest_key(event_id), manifest)

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def read_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    def write_bytes(self, key: str, content: bytes, content_type: str = "application/octet-stream") -> None:
        raise NotImplementedError

    def upload_file(self, key: str, source: Path) -> None:
        raise NotImplementedError

    def download_file(self, key: str, destination: Path) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def list_keys(self, prefix: str) -> list[str]:
        raise NotImplementedError

    def object_size(self, key: str) -> int | None:
        """Size in bytes, or None when the object does not exist."""
        raise NotImplementedError

    def copy(self, source_key: str, destination_key: str) -> None:
        path, cleanup = self.materialize(source_key)
        try:
            self.upload_file(destination_key, path)
        finally:
            if cleanup:
                path.unlink(missing_ok=True)

    def presign_put(self, key: str, content_type: str, expires_seconds: int) -> str | None:
        """Direct browser upload URL, or None when the backend cannot issue one."""
        return None

    def presign_get(
        self,
        key: str,
        expires_seconds: int,
        filename: str | None = None,
        attachment: bool = False,
        content_type: str | None = None,
    ) -> str | None:
        """Short-lived direct download URL, or None when media must be streamed by the service."""
        return None

    def read_text(self, key: str) -> str:
        return self.read_bytes(key).decode("utf-8")

    def write_text(self, key: str, value: str) -> None:
        self.write_bytes(key, value.encode("utf-8"), "application/json; charset=utf-8")

    def read_json(self, key: str, default: dict | None = None) -> dict:
        if not self.exists(key):
            if default is not None:
                return default
            raise FileNotFoundError(key)
        return json.loads(self.read_text(key))

    def write_json(self, key: str, value: dict) -> None:
        self.write_text(key, json.dumps(value, indent=2))

    def materialize(self, key: str, suffix: str = "", prefix: str = "fb-media-") -> tuple[Path, bool]:
        self.settings.temp_root.mkdir(parents=True, exist_ok=True)
        extension = suffix or Path(key).suffix
        with tempfile.NamedTemporaryFile(
            prefix=prefix, suffix=extension, dir=self.settings.temp_root, delete=False
        ) as temporary:
            path = Path(temporary.name)
        try:
            self.download_file(key, path)
            return path, True
        except Exception:
            path.unlink(missing_ok=True)
            raise


class LocalStorage(StorageBackend):
    """Filesystem storage adapter for local development and Railway Volumes."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.root = settings.data_root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        normalized = normalize_key(key)
        path = (self.root / Path(*PurePosixPath(normalized).parts)).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Storage key is outside the configured data root.")
        return path

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def write_bytes(self, key: str, content: bytes, content_type: str = "application/octet-stream") -> None:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            temporary.write_bytes(content)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def upload_file(self, key: str, source: Path) -> None:
        destination = self._path(key)
        if source.resolve() == destination.resolve():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            shutil.copy2(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def download_file(self, key: str, destination: Path) -> None:
        source = self._path(key)
        if source.resolve() == destination.resolve():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def object_size(self, key: str) -> int | None:
        path = self._path(key)
        return path.stat().st_size if path.is_file() else None

    def copy(self, source_key: str, destination_key: str) -> None:
        self.upload_file(destination_key, self._path(source_key))

    def list_keys(self, prefix: str) -> list[str]:
        root = self._path(prefix) if prefix else self.root
        if root.is_file():
            return [root.relative_to(self.root).as_posix()]
        if not root.exists():
            return []
        return [path.relative_to(self.root).as_posix() for path in root.rglob("*") if path.is_file()]

    def materialize(self, key: str, suffix: str = "", prefix: str = "fb-media-") -> tuple[Path, bool]:
        return self._path(key), False


class S3Storage(StorageBackend):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        # FB_S3_ENDPOINT_URL is only needed for S3-compatible stores (Railway Buckets, MinIO).
        # On AWS, leave the endpoint and access keys unset: boto3 then uses the default credential
        # chain (EC2 instance role or EKS Pod Identity), so no long-lived keys are stored.
        missing = [
            name for name, value in (
                ("FB_S3_BUCKET", settings.s3_bucket),
                ("FB_S3_REGION", settings.s3_region),
            ) if not value
        ]
        if missing:
            raise ValueError("S3 storage is missing required settings: " + ", ".join(missing))
        if bool(settings.s3_access_key_id) != bool(settings.s3_secret_access_key):
            raise ValueError(
                "Set both FB_S3_ACCESS_KEY_ID and FB_S3_SECRET_ACCESS_KEY, or neither to use AWS role credentials."
            )

        import boto3
        from botocore.config import Config
        from boto3.s3.transfer import TransferConfig

        session = boto3.Session(
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            region_name=settings.s3_region,
        )
        self.bucket = settings.s3_bucket
        raw_prefix = settings.s3_prefix.strip("/")
        self.prefix = normalize_key(raw_prefix) if raw_prefix else ""
        self.client = session.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            region_name=settings.s3_region,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": os.getenv("FB_S3_ADDRESSING_STYLE", "virtual")},
                retries={"total_max_attempts": 4, "mode": "standard"},
                connect_timeout=10,
                read_timeout=120,
                max_pool_connections=16,
            ),
        )
        self.transfer_config = TransferConfig(
            multipart_threshold=16 * 1024 * 1024,
            multipart_chunksize=8 * 1024 * 1024,
            max_concurrency=4,
            use_threads=True,
        )

    def _remote_key(self, key: str) -> str:
        normalized = normalize_key(key)
        return f"{self.prefix}/{normalized}" if self.prefix else normalized

    def _local_key(self, remote_key: str) -> str:
        return remote_key[len(self.prefix) + 1:] if self.prefix else remote_key

    def exists(self, key: str) -> bool:
        remote_key = self._remote_key(key)
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=remote_key, MaxKeys=1)
        return any(item["Key"] == remote_key for item in response.get("Contents", []))

    def read_bytes(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=self._remote_key(key))
        with response["Body"] as body:
            return body.read()

    def write_bytes(self, key: str, content: bytes, content_type: str = "application/octet-stream") -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._remote_key(key),
            Body=content,
            ContentType=content_type,
        )

    def upload_file(self, key: str, source: Path) -> None:
        self.client.upload_file(
            str(source), self.bucket, self._remote_key(key), Config=self.transfer_config
        )

    def download_file(self, key: str, destination: Path) -> None:
        from botocore.exceptions import ClientError

        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download_file(
                self.bucket, self._remote_key(key), str(destination), Config=self.transfer_config
            )
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                raise FileNotFoundError(key) from exc
            raise

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._remote_key(key))

    def object_size(self, key: str) -> int | None:
        from botocore.exceptions import ClientError

        try:
            response = self.client.head_object(Bucket=self.bucket, Key=self._remote_key(key))
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        return int(response["ContentLength"])

    def copy(self, source_key: str, destination_key: str) -> None:
        self.client.copy(
            {"Bucket": self.bucket, "Key": self._remote_key(source_key)},
            self.bucket,
            self._remote_key(destination_key),
            Config=self.transfer_config,
        )

    def ensure_cors(self, origins: list[str]) -> None:
        """Allow the web app's origins to PUT (direct uploads) and GET presigned URLs from the browser.

        Without this, browsers block presigned uploads to Railway Buckets / S3 ("upload was interrupted").
        """
        self.client.put_bucket_cors(
            Bucket=self.bucket,
            CORSConfiguration={"CORSRules": [{
                "AllowedOrigins": origins,
                "AllowedMethods": ["GET", "PUT", "HEAD"],
                "AllowedHeaders": ["*"],
                "ExposeHeaders": ["ETag"],
                "MaxAgeSeconds": 3600,
            }]},
        )

    def presign_put(self, key: str, content_type: str, expires_seconds: int) -> str | None:
        return self.client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": self._remote_key(key), "ContentType": content_type},
            ExpiresIn=expires_seconds,
            HttpMethod="PUT",
        )

    def presign_get(
        self,
        key: str,
        expires_seconds: int,
        filename: str | None = None,
        attachment: bool = False,
        content_type: str | None = None,
    ) -> str | None:
        params = {"Bucket": self.bucket, "Key": self._remote_key(key)}
        if filename or attachment:
            disposition = "attachment" if attachment else "inline"
            if filename:
                disposition += f'; filename="{safe_filename(filename)}"'
            params["ResponseContentDisposition"] = disposition
        if content_type:
            params["ResponseContentType"] = content_type
        params["ResponseCacheControl"] = "private, max-age=300"
        return self.client.generate_presigned_url("get_object", Params=params, ExpiresIn=expires_seconds)

    def list_keys(self, prefix: str) -> list[str]:
        remote_prefix = self._remote_key(prefix) if prefix else self.prefix
        paginator = self.client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=remote_prefix):
            for item in page.get("Contents", []):
                key = self._local_key(item["Key"])
                if not prefix or key.startswith(prefix):
                    keys.append(key)
        return keys


class GoogleDriveStorage(StorageBackend):
    DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        missing = [
            name for name, value in (
                ("FB_GOOGLE_DRIVE_FOLDER_ID", settings.google_drive_folder_id),
                ("FB_GOOGLE_CLIENT_ID", settings.google_client_id),
                ("FB_GOOGLE_CLIENT_SECRET", settings.google_client_secret),
                ("FB_GOOGLE_REFRESH_TOKEN", settings.google_refresh_token),
            ) if not value
        ]
        if missing:
            raise ValueError("Google Drive storage is missing required settings: " + ", ".join(missing))

        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        self.folder_id = settings.google_drive_folder_id
        self.credentials = Credentials(
            token=None,
            refresh_token=settings.google_refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
            scopes=[self.DRIVE_SCOPE],
        )
        self.credentials.refresh(Request())
        self.drive = build("drive", "v3", credentials=self.credentials, cache_discovery=False)
        self._file_ids: dict[str, str] = {}
        self._drive_lock = threading.RLock()

    @staticmethod
    def _drive_name(key: str) -> str:
        normalized = normalize_key(key)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        return f"FrameBucket-{digest}-{safe_filename(PurePosixPath(normalized).name)}"

    @staticmethod
    def _drive_description(key: str) -> str:
        return f"framebucket-key:{normalize_key(key)}"

    @staticmethod
    def _from_drive_description(description: str | None) -> str | None:
        if not description or not description.startswith("framebucket-key:"):
            return None
        try:
            return normalize_key(description.removeprefix("framebucket-key:"))
        except ValueError:
            return None

    def _find_file_id(self, key: str) -> str | None:
        normalized = normalize_key(key)
        with self._drive_lock:
            cached = self._file_ids.get(normalized)
            if cached:
                return cached
            name = self._drive_name(normalized)
            parent_id = self.folder_id.replace("'", "\\'")
            response = self.drive.files().list(
                q=f"'{parent_id}' in parents and name = '{name}' and trashed = false",
                pageSize=10,
                orderBy="modifiedTime desc",
                fields="files(id,name,description)",
            ).execute()
            files = response.get("files", [])
            exact = next(
                (item for item in files if self._from_drive_description(item.get("description")) == normalized),
                None,
            )
            if not exact:
                return None
            file_id = exact["id"]
            self._file_ids[normalized] = file_id
            return file_id

    def exists(self, key: str) -> bool:
        return self._find_file_id(key) is not None

    def read_bytes(self, key: str) -> bytes:
        from googleapiclient.http import MediaIoBaseDownload

        with self._drive_lock:
            file_id = self._find_file_id(key)
            if not file_id:
                raise FileNotFoundError(key)
            output = io.BytesIO()
            request = self.drive.files().get(fileId=file_id, alt="media")
            downloader = MediaIoBaseDownload(output, request, chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return output.getvalue()

    def write_bytes(self, key: str, content: bytes, content_type: str = "application/octet-stream") -> None:
        from googleapiclient.http import MediaIoBaseUpload

        normalized = normalize_key(key)
        with self._drive_lock:
            file_id = self._find_file_id(normalized)
            media = MediaIoBaseUpload(io.BytesIO(content), mimetype=content_type, resumable=False)
            if file_id:
                self.drive.files().update(fileId=file_id, media_body=media, fields="id").execute()
            else:
                result = self.drive.files().create(
                    body={
                        "name": self._drive_name(normalized),
                        "description": self._drive_description(normalized),
                        "parents": [self.folder_id],
                    },
                    media_body=media,
                    fields="id",
                ).execute()
                self._file_ids[normalized] = result["id"]

    def upload_file(self, key: str, source: Path) -> None:
        from googleapiclient.http import MediaFileUpload

        normalized = normalize_key(key)
        with self._drive_lock:
            file_id = self._find_file_id(normalized)
            media = MediaFileUpload(
                str(source),
                mimetype="application/octet-stream",
                chunksize=8 * 1024 * 1024,
                resumable=True,
            )
            if file_id:
                self.drive.files().update(fileId=file_id, media_body=media, fields="id").execute()
            else:
                result = self.drive.files().create(
                    body={
                        "name": self._drive_name(normalized),
                        "description": self._drive_description(normalized),
                        "parents": [self.folder_id],
                    },
                    media_body=media,
                    fields="id",
                ).execute()
                self._file_ids[normalized] = result["id"]

    def download_file(self, key: str, destination: Path) -> None:
        from googleapiclient.http import MediaIoBaseDownload

        with self._drive_lock:
            file_id = self._find_file_id(key)
            if not file_id:
                raise FileNotFoundError(key)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as output:
                request = self.drive.files().get(fileId=file_id, alt="media")
                downloader = MediaIoBaseDownload(output, request, chunksize=8 * 1024 * 1024)
                done = False
                while not done:
                    _, done = downloader.next_chunk()

    def delete(self, key: str) -> None:
        normalized = normalize_key(key)
        with self._drive_lock:
            file_id = self._find_file_id(normalized)
            if not file_id:
                return
            self.drive.files().delete(fileId=file_id).execute()
            self._file_ids.pop(normalized, None)

    def object_size(self, key: str) -> int | None:
        with self._drive_lock:
            file_id = self._find_file_id(normalize_key(key))
            if not file_id:
                return None
            metadata = self.drive.files().get(fileId=file_id, fields="size").execute()
        return int(metadata.get("size", 0))

    def list_keys(self, prefix: str) -> list[str]:
        with self._drive_lock:
            page_token = None
            keys: list[str] = []
            parent_id = self.folder_id.replace("'", "\\'")
            while True:
                response = self.drive.files().list(
                    q=f"'{parent_id}' in parents and trashed = false",
                    pageSize=1000,
                    pageToken=page_token,
                    fields="nextPageToken,files(id,name,description)",
                ).execute()
                for item in response.get("files", []):
                    key = self._from_drive_description(item.get("description"))
                    if key is None:
                        continue
                    if key.startswith(prefix):
                        self._file_ids[key] = item["id"]
                        keys.append(key)
                page_token = response.get("nextPageToken")
                if not page_token:
                    return keys


def create_storage(settings: Settings) -> StorageBackend:
    backend = settings.storage_backend
    if backend == "local":
        return LocalStorage(settings)
    if backend == "s3":
        return S3Storage(settings)
    if backend == "google_drive":
        return GoogleDriveStorage(settings)
    raise ValueError("FB_STORAGE_BACKEND must be 'local', 's3', or 'google_drive'.")
