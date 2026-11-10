from __future__ import annotations

import hmac
import logging
import mimetypes
import re
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .imaging import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from .service import PhotoService
from .storage import safe_filename

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
settings = Settings.from_env()
service = PhotoService(settings)

app = FastAPI(
    title="FrameBucket Data Service",
    version="0.1.0",
    description="Local-first gallery and photo-processing service.",
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
    photo_ids: list[str] = Field(min_length=1, max_length=5000)


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


@app.post("/api/v1/events/{event_id}/photos", dependencies=[Depends(require_service_key)])
async def upload_photo(
    event_id: str,
    file: UploadFile = File(...),
    photographer_id: str = Query(min_length=1, max_length=128),
    customer_id: str = Query(min_length=1, max_length=128),
    folder_id: str = Query(default="root", min_length=1, max_length=128),
) -> dict:
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
            result = await run_in_threadpool(
                service.upload,
                temporary_path,
                original_filename,
                event_id,
                photographer_id,
                customer_id,
                folder_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logging.getLogger(__name__).exception("Upload processing failed.")
            raise HTTPException(status_code=422, detail=f"Could not process media: {exc}") from exc
        return result.as_dict()
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
        await file.close()


@app.get(
    "/api/v1/events/{event_id}/photos",
    dependencies=[Depends(require_service_key)],
)
def list_event_photos(
    event_id: str,
    customer_id: str = Query(min_length=1, max_length=128),
    folder_id: str | None = Query(default=None, max_length=128),
) -> dict:
    try:
        return {"photos": service.list_gallery(event_id, customer_id, folder_id)}
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
        path = service.preview_path(event_id, photo_id, customer_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    size = path.stat().st_size
    headers = {"Cache-Control": "private, no-store", "Accept-Ranges": "bytes"}
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(path, media_type=media_type, headers=headers)

    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match or size == 0 or (not match.group(1) and not match.group(2)):
        return Response(status_code=416, headers={**headers, "Content-Range": f"bytes */{size}"})
    if match.group(1):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else size - 1
    else:
        suffix_length = int(match.group(2))
        start = max(0, size - suffix_length)
        end = size - 1
    if start >= size or end < start:
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
        path = service.original_path_for_photographer(event_id, photo_id, photographer_id, customer_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name, headers={"Cache-Control": "private, no-store"})


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


@app.get("/api/v1/album-design/{token}")
def download_album_design(token: str) -> FileResponse:
    try:
        path, filename = service.resolve_album_link(token)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        path,
        media_type="application/zip",
        filename=filename,
        headers={"Cache-Control": "private, no-store"},
    )
