from __future__ import annotations

import hmac
import logging
import mimetypes
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from . import jobs
from .config import Settings
from .imaging import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from .service import AlbumNotReady, DesignNotReady, DirectDownload, PhotoService
from .style import MAX_REFERENCES, StyleError
from .invitation import InvitationError
from .storage import safe_filename

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
settings = Settings.from_env()
service = PhotoService(settings)
MAX_GUEST_SELFIE_BYTES = 8 * 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.db.migrate()
    _configure_bucket_cors()
    embedded = None
    if settings.embedded_worker:
        from .worker import WorkerPool

        embedded = WorkerPool(service, settings)
        embedded.start()
    try:
        yield
    finally:
        if embedded:
            embedded.stop()
        service.db.close()


def _configure_bucket_cors() -> None:
    """FB_S3_CORS_ORIGINS=https://app.example.com,https://other -> bucket CORS for browser uploads/downloads."""
    import os

    origins = [origin.strip().rstrip("/") for origin in os.getenv("FB_S3_CORS_ORIGINS", "").split(",") if origin.strip()]
    if not origins or not hasattr(service.storage, "ensure_cors"):
        return
    try:
        service.storage.ensure_cors(origins)
        logging.getLogger(__name__).info("Bucket CORS allows %s", ", ".join(origins))
    except Exception:
        logging.getLogger(__name__).exception("Could not set the bucket CORS rules; browser uploads may be blocked.")


app = FastAPI(
    title="FrameBucket Data Service",
    version="0.2.0",
    description="Gallery and photo-processing service with local, S3-compatible, and Google Drive storage.",
    lifespan=lifespan,
)


class EventCreate(BaseModel):
    event_id: str = Field(min_length=1, max_length=128)
    event_code: str | None = Field(default=None, max_length=200)
    photographer_id: str = Field(min_length=1, max_length=128)
    customer_id: str = Field(min_length=1, max_length=128)


class FolderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class AlbumSubmissionCreate(BaseModel):
    photographer_id: str = Field(min_length=1, max_length=128)
    customer_id: str = Field(min_length=1, max_length=128)
    photo_ids: list[str] = Field(min_length=1, max_length=50000)


class AlbumSelectionUpdate(BaseModel):
    photographer_id: str = Field(min_length=1, max_length=128)
    customer_id: str = Field(min_length=1, max_length=128)
    photo_ids: list[str] = Field(default_factory=list, max_length=50000)


class AlbumDesignCreate(BaseModel):
    photographer_id: str = Field(min_length=1, max_length=128)
    customer_id: str = Field(min_length=1, max_length=128)
    submission_id: str = Field(min_length=1, max_length=128)
    height_in: float = Field(ge=6, le=24)
    width_in: float = Field(ge=12, le=48)
    sheets: int = Field(ge=1, le=120)
    dpi: int = Field(default=300)
    size_label: str | None = Field(default=None, max_length=20)


class OwnerRef(BaseModel):
    photographer_id: str = Field(min_length=1, max_length=128)
    customer_id: str = Field(min_length=1, max_length=128)


class GalleryDownloadRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=128)
    photo_ids: list[str] = Field(default_factory=list, max_length=50000)


class UploadInit(BaseModel):
    photographer_id: str = Field(min_length=1, max_length=128)
    customer_id: str = Field(min_length=1, max_length=128)
    folder_id: str = Field(default="root", min_length=1, max_length=128)
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)
    content_type: str | None = Field(default=None, max_length=255)


class StyleSettings(BaseModel):
    strength: int | None = Field(default=None, ge=0, le=100)
    apply_style: bool | None = None
    stamp_logo: bool | None = None


class OutboxClaim(BaseModel):
    limit: int = Field(default=50, ge=1, le=500)
    lease_seconds: int = Field(default=120, ge=10, le=3600)


class OutboxAck(BaseModel):
    ids: list[int] = Field(default_factory=list, max_length=500)


def require_service_key(
    supplied: Annotated[str | None, Header(alias="X-FrameBucket-Service-Key")] = None,
) -> None:
    expected = settings.service_token
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Set FB_SERVICE_TOKEN before using the data-service API.",
        )
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid service credentials.")


@app.get("/health")
def health() -> dict:
    try:
        service.db.ping()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database unavailable.") from exc
    return {"status": "ok", "service": "fb-data"}


@app.post("/api/v1/events", dependencies=[Depends(require_service_key)])
def create_event(payload: EventCreate) -> dict:
    try:
        return service.create_event(
            payload.event_id, payload.photographer_id, payload.customer_id, payload.event_code
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/v1/events/{event_id}/folders", dependencies=[Depends(require_service_key)])
def list_folders(
    event_id: str,
    photographer_id: str = Query(min_length=1, max_length=128),
    customer_id: str = Query(min_length=1, max_length=128),
) -> dict:
    try:
        return {"folders": service.list_folders(event_id, photographer_id, customer_id)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/events/{event_id}/album-selection", dependencies=[Depends(require_service_key)])
def get_album_selection(
    event_id: str,
    photographer_id: str = Query(min_length=1, max_length=128),
    customer_id: str = Query(min_length=1, max_length=128),
) -> dict:
    try:
        return service.album_selection(event_id, photographer_id, customer_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/v1/events/{event_id}/album-selection", dependencies=[Depends(require_service_key)])
def save_album_selection(event_id: str, payload: AlbumSelectionUpdate) -> dict:
    try:
        return service.save_album_selection(
            event_id, payload.photographer_id, payload.customer_id, payload.photo_ids
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/events/{event_id}/folders", dependencies=[Depends(require_service_key)])
def create_folder(
    event_id: str,
    payload: FolderCreate,
    photographer_id: str = Query(min_length=1, max_length=128),
    customer_id: str = Query(min_length=1, max_length=128),
) -> dict:
    try:
        return service.create_folder(event_id, photographer_id, customer_id, payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/events/{event_id}/uploads", dependencies=[Depends(require_service_key)])
def init_upload(event_id: str, payload: UploadInit) -> dict:
    """Register an upload. Returns a direct (presigned) PUT URL when the storage backend supports it;
    otherwise upload_url is null and the caller sends the bytes to the /content endpoint."""
    try:
        return service.init_upload(
            event_id, payload.photographer_id, payload.customer_id, payload.folder_id,
            payload.filename, payload.size_bytes, payload.content_type,
        )
    except ValueError as exc:
        status = 413 if "upload limit" in str(exc) else 422
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@app.put("/api/v1/events/{event_id}/uploads/{photo_id}/content", dependencies=[Depends(require_service_key)])
async def upload_content(
    event_id: str,
    photo_id: str,
    request: Request,
    photographer_id: str = Query(min_length=1, max_length=128),
    customer_id: str = Query(min_length=1, max_length=128),
) -> dict:
    temporary_path = await _receive_body(request)
    try:
        await run_in_threadpool(
            service.receive_upload_content, event_id, photo_id, photographer_id, customer_id, temporary_path
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return {"status": "RECEIVED"}


@app.post("/api/v1/events/{event_id}/uploads/{photo_id}/complete", dependencies=[Depends(require_service_key)])
def complete_upload(
    event_id: str,
    photo_id: str,
    photographer_id: str = Query(min_length=1, max_length=128),
    customer_id: str = Query(min_length=1, max_length=128),
) -> dict:
    try:
        return service.complete_upload(event_id, photo_id, photographer_id, customer_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/events/{event_id}/photos", dependencies=[Depends(require_service_key)])
async def upload_photo(
    event_id: str,
    file: UploadFile = File(...),
    photographer_id: str = Query(min_length=1, max_length=128),
    customer_id: str = Query(min_length=1, max_length=128),
    folder_id: str = Query(default="root", min_length=1, max_length=128),
) -> dict:
    """Single-request multipart upload for scripts. The file is queued; a worker processes it."""
    try:
        original_filename = safe_filename(file.filename or "upload")
        suffix = Path(original_filename).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS and suffix not in VIDEO_EXTENSIONS:
            raise HTTPException(status_code=415, detail="Unsupported image or video type.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    settings.temp_root.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    total = 0
    try:
        with tempfile.NamedTemporaryFile(
            prefix="upload-", suffix=suffix, dir=settings.temp_root, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="File exceeds the upload limit.")
                temporary.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        try:
            return await run_in_threadpool(
                service.ingest_file, temporary_path, original_filename,
                event_id, photographer_id, customer_id, folder_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
        await file.close()


async def _receive_body(request: Request) -> Path:
    settings.temp_root.mkdir(parents=True, exist_ok=True)
    total = 0
    with tempfile.NamedTemporaryFile(prefix="upload-", dir=settings.temp_root, delete=False) as temporary:
        path = Path(temporary.name)
        try:
            async for chunk in request.stream():
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="File exceeds the upload limit.")
                temporary.write(chunk)
        except BaseException:
            temporary.close()
            path.unlink(missing_ok=True)
            raise
    if total == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return path


# ------------------------------------------------------------------ studio look (editing style + logo)

@app.get("/api/v1/studios/{photographer_id}/style", dependencies=[Depends(require_service_key)])
def get_studio_style(photographer_id: str) -> dict:
    try:
        return service.studio_style(photographer_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/studios/{photographer_id}/style/references", dependencies=[Depends(require_service_key)])
async def upload_style_references(photographer_id: str, files: list[UploadFile] = File(...)) -> dict:
    try:
        if len(files) > MAX_REFERENCES:
            raise HTTPException(status_code=422, detail=f"Upload at most {MAX_REFERENCES} reference photos.")
        contents: list[bytes] = []
        for upload in files:
            data = await upload.read(service.MAX_REFERENCE_BYTES + 1)
            if len(data) > service.MAX_REFERENCE_BYTES:
                raise HTTPException(status_code=413, detail="Each reference photo must be 40 MB or smaller.")
            if data:
                contents.append(data)
        try:
            return await run_in_threadpool(service.save_style_references, photographer_id, contents)
        except (StyleError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        for upload in files:
            await upload.close()


@app.put("/api/v1/studios/{photographer_id}/style", dependencies=[Depends(require_service_key)])
def update_studio_style(photographer_id: str, payload: StyleSettings) -> dict:
    try:
        return service.update_style_settings(photographer_id, payload.strength, payload.apply_style, payload.stamp_logo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/v1/studios/{photographer_id}/style", dependencies=[Depends(require_service_key)])
def delete_studio_style(photographer_id: str) -> dict:
    try:
        return service.clear_style(photographer_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/studios/{photographer_id}/style/preview", dependencies=[Depends(require_service_key)])
async def preview_studio_style(
    photographer_id: str,
    file: UploadFile = File(...),
    strength: int | None = Query(default=None, ge=0, le=100),
) -> Response:
    try:
        data = await file.read(service.MAX_REFERENCE_BYTES + 1)
        if not data:
            raise HTTPException(status_code=400, detail="Choose a sample photo.")
        if len(data) > service.MAX_REFERENCE_BYTES:
            raise HTTPException(status_code=413, detail="The sample photo must be 40 MB or smaller.")
        try:
            jpeg = await run_in_threadpool(service.style_preview, photographer_id, file.filename or "", data, strength)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    finally:
        await file.close()


@app.post("/api/v1/outbox/claim", dependencies=[Depends(require_service_key)])
def claim_outbox(payload: OutboxClaim) -> dict:
    """Lease pending notifications (face matches, album archives) for the Java application."""
    return {"items": jobs.claim_outbox(service.db, payload.limit, payload.lease_seconds)}


@app.post("/api/v1/outbox/ack", dependencies=[Depends(require_service_key)])
def ack_outbox(payload: OutboxAck) -> dict:
    return {"acknowledged": jobs.ack_outbox(service.db, payload.ids)}


@app.post("/api/v1/events/{event_id}/guests/{guest_id}/face", dependencies=[Depends(require_service_key)])
async def register_guest_face(
    event_id: str,
    guest_id: str,
    file: UploadFile = File(...),
    customer_id: str = Query(min_length=1, max_length=128),
) -> dict:
    try:
        content = await file.read(MAX_GUEST_SELFIE_BYTES + 1)
        if not content:
            raise HTTPException(status_code=400, detail="Choose a selfie image.")
        if len(content) > MAX_GUEST_SELFIE_BYTES:
            raise HTTPException(status_code=413, detail="Selfies must be 8 MB or smaller.")
        try:
            return await run_in_threadpool(
                service.register_guest_face,
                event_id,
                guest_id,
                customer_id,
                content,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logging.getLogger(__name__).exception("Guest face registration failed for event %s.", event_id)
            raise HTTPException(status_code=503, detail=f"Face matching is temporarily unavailable: {exc}") from exc
    finally:
        await file.close()


@app.delete("/api/v1/events/{event_id}/guests/{guest_id}/face", dependencies=[Depends(require_service_key)])
def delete_guest_face(
    event_id: str,
    guest_id: str,
    customer_id: str = Query(min_length=1, max_length=128),
) -> dict:
    try:
        service.delete_guest_face(event_id, guest_id, customer_id)
        return {"status": "deleted"}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get(
    "/api/v1/events/{event_id}/photos",
    dependencies=[Depends(require_service_key)],
)
def list_event_photos(
    event_id: str,
    customer_id: str = Query(min_length=1, max_length=128),
    folder_id: str | None = Query(default=None, max_length=128),
    include_pending: bool = Query(default=False),
) -> dict:
    try:
        return {"photos": service.list_gallery(event_id, customer_id, folder_id, include_pending)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/events/{event_id}/downloads", dependencies=[Depends(require_service_key)])
def download_customer_gallery(event_id: str, payload: GalleryDownloadRequest) -> FileResponse:
    try:
        download_path, filename, media_type, remove_after_send = service.download_gallery(
            event_id, payload.customer_id, payload.photo_ids
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return FileResponse(
        download_path,
        media_type=media_type,
        filename=filename,
        headers={"Cache-Control": "private, no-store"},
        background=(
            BackgroundTask(lambda: download_path.unlink(missing_ok=True))
            if remove_after_send else None
        ),
    )


@app.get(
    "/api/v1/events/{event_id}/photos/{photo_id}/whatsapp-preview",
    dependencies=[Depends(require_service_key)],
)
def get_whatsapp_preview(
    event_id: str,
    photo_id: str,
    customer_id: str = Query(min_length=1, max_length=128),
) -> Response:
    try:
        image = service.whatsapp_preview_bytes(event_id, photo_id, customer_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(
        content=image,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, no-store"},
    )


@app.get(
    "/api/v1/events/{event_id}/photos/{photo_id}/preview-url",
    dependencies=[Depends(require_service_key)],
)
def get_preview_url(
    event_id: str,
    photo_id: str,
    customer_id: str = Query(min_length=1, max_length=128),
) -> dict:
    """Short-lived direct URL for the processed media, or null when it must be streamed."""
    try:
        return {"url": service.media_url(event_id, photo_id, customer_id)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get(
    "/api/v1/events/{event_id}/photos/{photo_id}/original-url",
    dependencies=[Depends(require_service_key)],
)
def get_original_url(
    event_id: str,
    photo_id: str,
    photographer_id: str = Query(min_length=1, max_length=128),
    customer_id: str = Query(min_length=1, max_length=128),
) -> dict:
    try:
        return {"url": service.original_url(event_id, photo_id, photographer_id, customer_id)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get(
    "/api/v1/events/{event_id}/photos/{photo_id}/preview",
    dependencies=[Depends(require_service_key)],
)
def get_preview(
    event_id: str,
    photo_id: str,
    request: Request,
    customer_id: str = Query(min_length=1, max_length=128),
) -> Response:
    try:
        path, cleanup = service.preview_path(event_id, photo_id, customer_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    size = path.stat().st_size
    headers = {"Cache-Control": "private, no-store", "Accept-Ranges": "bytes"}
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(
            path,
            media_type=media_type,
            headers=headers,
            background=BackgroundTask(lambda: path.unlink(missing_ok=True)) if cleanup else None,
        )

    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match or size == 0 or (not match.group(1) and not match.group(2)):
        if cleanup:
            path.unlink(missing_ok=True)
        return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{size}"})
    if match.group(1):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else size - 1
    else:
        suffix_length = int(match.group(2))
        start = max(0, size - suffix_length)
        end = size - 1
    if start >= size or end < start:
        if cleanup:
            path.unlink(missing_ok=True)
        return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{size}"})
    end = min(end, size - 1)
    length = end - start + 1

    def chunks():
        with path.open("rb") as media_file:
            media_file.seek(start)
            remaining = length
            while remaining:
                chunk = media_file.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        chunks(),
        status_code=206,
        media_type=media_type,
        headers={
            **headers,
            "Content-Length": str(length),
            "Content-Range": f"bytes {start}-{end}/{size}",
        },
        background=BackgroundTask(lambda: path.unlink(missing_ok=True)) if cleanup else None,
    )


@app.get(
    "/api/v1/events/{event_id}/photos/{photo_id}/original",
    dependencies=[Depends(require_service_key)],
)
def get_original(
    event_id: str,
    photo_id: str,
    photographer_id: str = Query(min_length=1, max_length=128),
    customer_id: str = Query(min_length=1, max_length=128),
) -> FileResponse:
    try:
        path, cleanup, original_filename = service.original_path_for_photographer(
            event_id, photo_id, photographer_id, customer_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(
        path,
        media_type=media_type,
        filename=original_filename,
        headers={"Cache-Control": "private, no-store"},
        background=BackgroundTask(lambda: path.unlink(missing_ok=True)) if cleanup else None,
    )


@app.post(
    "/api/v1/events/{event_id}/album-submissions",
    dependencies=[Depends(require_service_key)],
)
def submit_album(event_id: str, payload: AlbumSubmissionCreate) -> dict:
    try:
        return service.submit_album(
            event_id,
            payload.photographer_id,
            payload.customer_id,
            payload.photo_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/v1/album-design/{token}", response_model=None)
def download_album_design(token: str) -> Response:
    try:
        path, filename, cleanup = service.resolve_album_link(token)
    except AlbumNotReady as exc:
        return JSONResponse(
            status_code=202,
            content={"status": "PREPARING", "detail": str(exc)},
            headers={"Retry-After": "60", "Cache-Control": "no-store"},
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(path, DirectDownload):
        return RedirectResponse(path.url, status_code=302, headers={"Cache-Control": "no-store"})
    return FileResponse(
        path,
        media_type="application/zip",
        filename=filename,
        headers={"Cache-Control": "private, no-store"},
        background=BackgroundTask(lambda: path.unlink(missing_ok=True)) if cleanup else None,
    )


# ------------------------------------------------------------------ album auto-design


@app.get("/api/v1/events/{event_id}/album-designs", dependencies=[Depends(require_service_key)])
def list_album_designs(
    event_id: str,
    photographer_id: str = Query(min_length=1, max_length=128),
    customer_id: str = Query(min_length=1, max_length=128),
) -> dict:
    try:
        return {"submissions": service.album_submissions(event_id, photographer_id, customer_id)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/events/{event_id}/album-designs", dependencies=[Depends(require_service_key)])
def create_album_design(event_id: str, payload: AlbumDesignCreate) -> dict:
    try:
        return service.create_album_design(
            event_id, payload.photographer_id, payload.customer_id, payload.submission_id,
            payload.height_in, payload.width_in, payload.sheets, payload.dpi, payload.size_label,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/v1/events/{event_id}/album-designs/{design_id}", dependencies=[Depends(require_service_key)])
def get_album_design(
    event_id: str,
    design_id: str,
    photographer_id: str = Query(min_length=1, max_length=128),
    customer_id: str = Query(min_length=1, max_length=128),
) -> dict:
    try:
        return service.album_design_status(event_id, design_id, photographer_id, customer_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get(
    "/api/v1/events/{event_id}/album-designs/{design_id}/spreads/{index}",
    dependencies=[Depends(require_service_key)],
)
def get_album_design_spread(
    event_id: str,
    design_id: str,
    index: int,
    photographer_id: str = Query(min_length=1, max_length=128),
    customer_id: str = Query(min_length=1, max_length=128),
) -> FileResponse:
    try:
        path, cleanup = service.album_design_preview(event_id, design_id, index, photographer_id, customer_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
        background=BackgroundTask(lambda: path.unlink(missing_ok=True)) if cleanup else None,
    )


@app.get(
    "/api/v1/events/{event_id}/album-designs/{design_id}/download/{kind}",
    dependencies=[Depends(require_service_key)],
    response_model=None,
)
def download_album_design_archive(
    event_id: str,
    design_id: str,
    kind: str,
    photographer_id: str = Query(min_length=1, max_length=128),
    customer_id: str = Query(min_length=1, max_length=128),
) -> Response:
    if kind not in {"psd", "jpeg"}:
        raise HTTPException(status_code=404, detail="Unknown archive.")
    try:
        path, filename, cleanup = service.album_design_download(
            event_id, design_id, kind, photographer_id, customer_id
        )
    except DesignNotReady as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(path, DirectDownload):
        return RedirectResponse(path.url, status_code=302, headers={"Cache-Control": "no-store"})
    return FileResponse(
        path,
        media_type="application/zip",
        filename=filename,
        headers={"Cache-Control": "private, no-store"},
        background=BackgroundTask(lambda: path.unlink(missing_ok=True)) if cleanup else None,
    )


@app.post(
    "/api/v1/events/{event_id}/album-designs/{design_id}/publish",
    dependencies=[Depends(require_service_key)],
)
def publish_album_design(event_id: str, design_id: str, payload: OwnerRef) -> dict:
    try:
        return service.publish_album_design(event_id, design_id, payload.photographer_id, payload.customer_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ------------------------------------------------------------------ customer wedding invitations


def _invitation_call(call):
    try:
        return call()
    except InvitationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class InvitationDetails(BaseModel):
    photographer_id: str = Field(min_length=1, max_length=128)
    customer_id: str = Field(min_length=1, max_length=128)
    details: dict


class InvitationFinalize(BaseModel):
    photographer_id: str = Field(min_length=1, max_length=128)
    customer_id: str = Field(min_length=1, max_length=128)
    index: int = Field(ge=0, le=20)


@app.get("/api/v1/events/{event_id}/invitation", dependencies=[Depends(require_service_key)])
def get_invitation(event_id: str, photographer_id: str = Query(min_length=1, max_length=128),
                   customer_id: str = Query(min_length=1, max_length=128)) -> dict:
    return _invitation_call(lambda: service.invitation(event_id, photographer_id, customer_id))


@app.put("/api/v1/events/{event_id}/invitation", dependencies=[Depends(require_service_key)])
def put_invitation(event_id: str, payload: InvitationDetails) -> dict:
    return _invitation_call(lambda: service.save_invitation_details(event_id, payload.photographer_id, payload.customer_id, payload.details))


@app.post("/api/v1/events/{event_id}/invitation/photo", dependencies=[Depends(require_service_key)])
async def post_invitation_photo(event_id: str, photographer_id: str = Query(min_length=1, max_length=128),
                                customer_id: str = Query(min_length=1, max_length=128),
                                file: UploadFile = File(...)) -> dict:
    data = await file.read(25 * 1024 * 1024 + 1)
    return await run_in_threadpool(_invitation_call, lambda: service.save_invitation_photo(event_id, photographer_id, customer_id, data))


@app.delete("/api/v1/events/{event_id}/invitation/photo", dependencies=[Depends(require_service_key)])
def delete_invitation_photo(event_id: str, photographer_id: str = Query(min_length=1, max_length=128),
                            customer_id: str = Query(min_length=1, max_length=128)) -> dict:
    return _invitation_call(lambda: service.delete_invitation_photo(event_id, photographer_id, customer_id))


@app.post("/api/v1/events/{event_id}/invitation/designs", dependencies=[Depends(require_service_key)])
def post_invitation_designs(event_id: str, payload: OwnerRef) -> dict:
    return _invitation_call(lambda: service.generate_invitation_designs(event_id, payload.photographer_id, payload.customer_id))


@app.get("/api/v1/events/{event_id}/invitation/designs/{index}", dependencies=[Depends(require_service_key)])
def get_invitation_design(event_id: str, index: int, customer_id: str = Query(min_length=1, max_length=128)) -> Response:
    image = _invitation_call(lambda: service.invitation_preview(event_id, customer_id, index))
    return Response(content=image, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=600"})


@app.post("/api/v1/events/{event_id}/invitation/finalize", dependencies=[Depends(require_service_key)])
def post_invitation_finalize(event_id: str, payload: InvitationFinalize) -> dict:
    return _invitation_call(lambda: service.finalize_invitation(event_id, payload.photographer_id, payload.customer_id, payload.index))


@app.get("/api/v1/events/{event_id}/invitation/{kind}", dependencies=[Depends(require_service_key)])
def get_invitation_file(event_id: str, kind: str, customer_id: str = Query(min_length=1, max_length=128)) -> FileResponse:
    if kind not in {"video", "poster"}:
        raise HTTPException(status_code=404, detail="Unknown file.")
    path, cleanup, media_type = _invitation_call(lambda: service.invitation_file(event_id, customer_id, kind))
    return FileResponse(
        path,
        media_type=media_type,
        filename=f"wedding-invitation.{'mp4' if kind == 'video' else 'jpg'}",
        headers={"Cache-Control": "private, no-store"},
        background=BackgroundTask(lambda: path.unlink(missing_ok=True)) if cleanup else None,
    )
