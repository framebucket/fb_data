"""Album auto-design: data-service workflow around ``album_design`` (jobs, storage, API-facing methods).

Mixed into PhotoService; uses its database, storage, image processor, face matcher and studio look.
"""
from __future__ import annotations

import logging
import random
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path

from PIL import Image
from psycopg.types.json import Jsonb

from . import album_design as design
from . import jobs
from . import psd
from .imaging import StudioLook
from .storage import normalize_key, safe_filename, validate_id

logger = logging.getLogger(__name__)

WORK_SIDE = 6000            # corrected working copy of each photo (covers a full-bleed hero on large spreads)


class DesignNotReady(Exception):
    """The album design is still being generated."""


class AlbumDesignMixin:
    # ------------------------------------------------------------------ API-facing

    def album_submissions(self, event_id: str, photographer_id: str, customer_id: str) -> list[dict]:
        with self.db.transaction() as connection:
            known = connection.execute("SELECT 1 FROM fb_data.event WHERE event_id = %s",
                                       (validate_id(event_id, "event_id"),)).fetchone()
            if not known:          # no gallery for this booking yet, so nothing has been submitted
                return []
            event = self._owned_event(connection, event_id, photographer_id, customer_id)
            submissions = connection.execute(
                "SELECT submission_id, status, photo_ids, created_at FROM fb_data.album_submission "
                "WHERE event_id = %s ORDER BY created_at DESC",
                (event["event_id"],),
            ).fetchall()
            designs = connection.execute(
                "SELECT * FROM fb_data.album_design WHERE event_id = %s ORDER BY created_at DESC",
                (event["event_id"],),
            ).fetchall()
        by_submission: dict[str, list[dict]] = {}
        for row in designs:
            by_submission.setdefault(row["submission_id"], []).append(self._design_dict(row))
        return [
            {
                "submission_id": row["submission_id"],
                "status": row["status"],
                "photo_count": len(row["photo_ids"]),
                "created_at": row["created_at"].isoformat(),
                "designs": by_submission.get(row["submission_id"], []),
            }
            for row in submissions
        ]

    def create_album_design(
        self, event_id: str, photographer_id: str, customer_id: str, submission_id: str,
        height_in: float, width_in: float, sheets: int, dpi: int = 300, size_label: str | None = None,
    ) -> dict:
        submission_id = validate_id(submission_id, "submission_id")
        if not (6 <= height_in <= 24 and 12 <= width_in <= 48 and width_in > height_in):
            raise ValueError("Album size must be height 6–24 in and open-spread width 12–48 in (wider than tall).")
        if not 1 <= sheets <= 120:
            raise ValueError("Choose between 1 and 120 sheets.")
        if dpi not in (150, 200, 240, 300):
            raise ValueError("Resolution must be 150, 200, 240 or 300 ppi.")
        if round(width_in * dpi) > 30000:
            raise ValueError("That size is too large for Photoshop at this resolution; choose a lower ppi.")
        label = (size_label or f"{height_in:g}x{width_in:g}")[:20]
        design_id = uuid.uuid4().hex
        with self.db.transaction() as connection:
            event = self._owned_event(connection, event_id, photographer_id, customer_id)
            submission = connection.execute(
                "SELECT * FROM fb_data.album_submission WHERE submission_id = %s AND event_id = %s",
                (submission_id, event["event_id"]),
            ).fetchone()
            if not submission:
                raise ValueError("That album selection was not found for this event.")
            busy = connection.execute(
                "SELECT 1 FROM fb_data.album_design WHERE event_id = %s AND status IN ('QUEUED', 'PROCESSING')",
                (event["event_id"],),
            ).fetchone()
            if busy:
                raise ValueError("An album design is already being generated for this event. Please wait for it.")
            version = connection.execute(
                "SELECT COALESCE(max(version), 0) + 1 AS v FROM fb_data.album_design WHERE event_id = %s",
                (event["event_id"],),
            ).fetchone()["v"]
            connection.execute(
                """
                INSERT INTO fb_data.album_design (design_id, event_id, submission_id, photographer_id, version,
                    size_label, height_in, width_in, sheets, dpi, seed, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'QUEUED')
                """,
                (design_id, event["event_id"], submission_id, photographer_id, version, label, height_in,
                 width_in, sheets, dpi, random.randrange(1, 2_000_000_000)),
            )
            jobs.enqueue(connection, jobs.DESIGN_ALBUM, {"design_id": design_id}, 2)
        return self.album_design_status(event_id, design_id, photographer_id, customer_id)

    def album_design_status(self, event_id: str, design_id: str, photographer_id: str, customer_id: str) -> dict:
        return self._design_dict(self._owned_design(event_id, design_id, photographer_id, customer_id))

    def album_design_preview(self, event_id: str, design_id: str, index: int, photographer_id: str,
                             customer_id: str) -> tuple[Path, bool]:
        row = self._owned_design(event_id, design_id, photographer_id, customer_id)
        spreads = row["spreads"] or []
        if row["status"] != "READY" or not 0 <= index < len(spreads):
            raise FileNotFoundError("Spread preview not found.")
        return self.storage.materialize(normalize_key(spreads[index]["preview_key"]), suffix=".jpg")

    def album_design_download(self, event_id: str, design_id: str, kind: str, photographer_id: str,
                              customer_id: str):
        """Returns (DirectDownload | Path, filename, cleanup) for the print-JPEG or PSD archive."""
        from .service import DirectDownload

        row = self._owned_design(event_id, design_id, photographer_id, customer_id)
        if row["status"] != "READY":
            raise DesignNotReady("The album design is still being generated.")
        key = row["psd_zip_key"] if kind == "psd" else row["jpeg_zip_key"]
        if not key:
            raise FileNotFoundError("Archive not found.")
        with self.db.transaction() as connection:
            code = connection.execute("SELECT event_code FROM fb_data.event WHERE event_id = %s",
                                      (row["event_id"],)).fetchone()["event_code"]
        filename = safe_filename(f"{code}-album-v{row['version']}-{row['size_label']}-{'photoshop' if kind == 'psd' else 'print-jpeg'}.zip")
        url = self.storage.presign_get(key, self.settings.media_url_ttl_seconds, filename=filename,
                                       attachment=True, content_type="application/zip")
        if url:
            return DirectDownload(url), filename, False
        archive, cleanup = self.storage.materialize(normalize_key(key), suffix=".zip", prefix="fb-album-")
        return archive, filename, cleanup

    def publish_album_design(self, event_id: str, design_id: str, photographer_id: str, customer_id: str) -> dict:
        """Copy the spread previews into a gallery folder so the existing proofing flow can send them."""
        row = self._owned_design(event_id, design_id, photographer_id, customer_id)
        if row["status"] != "READY":
            raise ValueError("The album design is not ready yet.")
        if row["gallery_folder_id"]:
            return {"folder_id": row["gallery_folder_id"], "spreads": len(row["spreads"] or [])}
        folder_id = uuid.uuid4().hex
        name = f"Album design v{row['version']} ({row['size_label']})"
        with self.db.transaction() as connection:
            existing = connection.execute(
                "SELECT folder_id FROM fb_data.folder WHERE event_id = %s AND lower(name) = lower(%s)",
                (row["event_id"], name)).fetchone()
            if existing:
                name = f"{name} {design_id[:6]}"
            connection.execute("INSERT INTO fb_data.folder (event_id, folder_id, name) VALUES (%s, %s, %s)",
                               (row["event_id"], folder_id, name))
        for spread in row["spreads"] or []:
            photo_id = uuid.uuid4().hex
            filename = f"spread-{spread['index'] + 1:02d}.jpg"
            processed_key = self.storage.processed_key(row["event_id"], photo_id, folder_id, filename)
            original_key = self.storage.original_key(photographer_id, row["event_id"], photo_id, filename, folder_id)
            self.storage.copy(normalize_key(spread["preview_key"]), processed_key)
            self.storage.copy(normalize_key(spread["preview_key"]), original_key)
            with self.db.transaction() as connection:
                number = connection.execute(
                    "UPDATE fb_data.event SET next_photo_number = next_photo_number + 1 WHERE event_id = %s "
                    "RETURNING next_photo_number - 1 AS n", (row["event_id"],)).fetchone()["n"]
                connection.execute(
                    """
                    INSERT INTO fb_data.media (photo_id, event_id, photo_number, folder_id, media_type,
                        original_filename, content_type, original_key, processed_key, status, width, height,
                        size_bytes, processed_bytes)
                    VALUES (%s, %s, %s, %s, 'image', %s, 'image/jpeg', %s, %s, 'READY', %s, %s, %s, %s)
                    """,
                    (photo_id, row["event_id"], number, folder_id, filename, original_key, processed_key,
                     spread.get("preview_width"), spread.get("preview_height"), spread.get("preview_bytes"),
                     spread.get("preview_bytes")),
                )
        with self.db.transaction() as connection:
            connection.execute("UPDATE fb_data.album_design SET gallery_folder_id = %s WHERE design_id = %s",
                               (folder_id, design_id))
        return {"folder_id": folder_id, "spreads": len(row["spreads"] or [])}

    # ------------------------------------------------------------------ worker job

    def design_album(self, design_id: str) -> None:
        design_id = validate_id(design_id, "design_id")
        with self.db.transaction() as connection:
            row = connection.execute(
                "UPDATE fb_data.album_design SET status = 'PROCESSING', progress = 1, error = NULL "
                "WHERE design_id = %s AND status IN ('QUEUED', 'PROCESSING') RETURNING *", (design_id,)
            ).fetchone()
            if not row:
                return
            submission = connection.execute("SELECT * FROM fb_data.album_submission WHERE submission_id = %s",
                                            (row["submission_id"],)).fetchone()
            media = {
                m["photo_id"]: m for m in connection.execute(
                    "SELECT * FROM fb_data.media WHERE photo_id = ANY(%s)", (list(submission["photo_ids"]),))
            }
        photo_rows = [media[p] for p in submission["photo_ids"] if p in media]
        if not photo_rows:
            raise ValueError("None of the selected photos are available any more.")

        spec = design.AlbumSpec(float(row["height_in"]), float(row["width_in"]), int(row["sheets"]),
                                int(row["dpi"]), int(row["seed"]))
        base_look = self.studio_look(row["photographer_id"])
        look = StudioLook(profile=base_look.profile, baseline=base_look.baseline, strength=base_look.strength,
                          logo=None, stamp_logo=False, presets=base_look.presets, preset_target=base_look.preset_target)
        work_dir = Path(tempfile.mkdtemp(prefix=f"album-{design_id[:8]}-", dir=self.settings.temp_root))
        try:
            infos: list[design.PhotoInfo] = []
            work_files: dict[str, Path] = {}
            for position, media_row in enumerate(photo_rows, start=1):
                source, cleanup = self.storage.materialize(
                    self._original_key(media_row, row["photographer_id"]),
                    suffix=Path(media_row["original_filename"]).suffix,
                )
                try:
                    original = self.processor._load(source)
                    exif_source = Image.open(source) if source.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff"} else None
                    original.thumbnail((WORK_SIDE, WORK_SIDE), Image.Resampling.LANCZOS)
                    corrected, _ = self.processor.render(original, look)
                    work = work_dir / f"{media_row['photo_id']}.jpg"
                    corrected.save(work, "JPEG", quality=93, subsampling=0)
                    work_files[media_row["photo_id"]] = work
                    infos.append(design.analyse_photo(
                        media_row["photo_id"], media_row["photo_number"] or position, media_row["original_filename"],
                        corrected, exif_source, self.face_matcher.face_boxes if self.settings.face_matching_enabled else None,
                    ))
                    if exif_source is not None:
                        exif_source.close()
                finally:
                    if cleanup:
                        source.unlink(missing_ok=True)
                self._design_progress(design_id, 1 + int(49 * position / len(photo_rows)))

            plans = design.plan_album(infos, spec, self.reference_layouts(row["photographer_id"]))

            def load(info: design.PhotoInfo) -> Image.Image:
                with Image.open(work_files[info.photo_id]) as opened:
                    return opened.convert("RGB")

            prefix = f"album_design/{row['event_id']}/{design_id}"
            jpeg_zip = work_dir / "print-jpeg.zip"
            psd_zip = work_dir / "photoshop.zip"
            spreads_meta = []
            with zipfile.ZipFile(jpeg_zip, "w", zipfile.ZIP_STORED, allowZip64=True) as jpegs, \
                    zipfile.ZipFile(psd_zip, "w", zipfile.ZIP_STORED, allowZip64=True) as psds:
                for plan in plans:
                    rendered = design.render_spread(plan, spec, load)
                    number = plan.index + 1
                    print_jpeg = design.jpeg_bytes(rendered.composite, 95, spec.dpi)
                    jpegs.writestr(f"spread-{number:02d}.jpg", print_jpeg)
                    psd_path = work_dir / f"spread-{number:02d}.psd"
                    psd.save(rendered.document, psd_path, rendered.composite)
                    psds.write(psd_path, f"spread-{number:02d}.psd")
                    psd_path.unlink(missing_ok=True)
                    small = design.preview(rendered.composite)
                    preview_bytes = design.jpeg_bytes(small, 86)
                    preview_key = f"{prefix}/preview-{number:02d}.jpg"
                    self.storage.write_bytes(preview_key, preview_bytes, "image/jpeg")
                    spreads_meta.append({
                        "index": plan.index, "recipe": plan.recipe, "reference": plan.reference, "preview_key": preview_key,
                        "preview_width": small.width, "preview_height": small.height,
                        "preview_bytes": len(preview_bytes),
                        "photos": [p.photo.number for p in plan.placements],
                    })
                    rendered.composite.close()
                    self._design_progress(design_id, 50 + int(48 * number / len(plans)))
                readme = _readme(spec, row, len(plans), len(infos))
                jpegs.writestr("README.txt", readme)
                psds.writestr("README.txt", readme)
            self.storage.upload_file(f"{prefix}/print-jpeg.zip", jpeg_zip)
            self.storage.upload_file(f"{prefix}/photoshop.zip", psd_zip)
            with self.db.transaction() as connection:
                connection.execute(
                    """
                    UPDATE fb_data.album_design SET status = 'READY', progress = 100, spreads = %s,
                        jpeg_zip_key = %s, psd_zip_key = %s, ready_at = now(), error = NULL
                    WHERE design_id = %s
                    """,
                    (Jsonb(spreads_meta), f"{prefix}/print-jpeg.zip", f"{prefix}/photoshop.zip", design_id),
                )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    # ------------------------------------------------------------------ helpers

    def _design_progress(self, design_id: str, percent: int) -> None:
        """Record progress and keep the job's lease alive (large albums take longer than one lease)."""
        with self.db.transaction() as connection:
            connection.execute("UPDATE fb_data.album_design SET progress = %s WHERE design_id = %s",
                               (min(99, percent), design_id))
            connection.execute(
                "UPDATE fb_data.job SET locked_at = now() WHERE kind = %s AND status = 'RUNNING' "
                "AND payload ->> 'design_id' = %s", (jobs.DESIGN_ALBUM, design_id))

    def _owned_design(self, event_id: str, design_id: str, photographer_id: str, customer_id: str) -> dict:
        design_id = validate_id(design_id, "design_id")
        with self.db.transaction() as connection:
            event = self._owned_event(connection, event_id, photographer_id, customer_id)
            row = connection.execute(
                "SELECT * FROM fb_data.album_design WHERE design_id = %s AND event_id = %s",
                (design_id, event["event_id"]),
            ).fetchone()
        if not row:
            raise FileNotFoundError("Album design not found.")
        return row

    @staticmethod
    def _design_dict(row: dict) -> dict:
        spreads = row["spreads"] or []
        return {
            "design_id": row["design_id"],
            "submission_id": row["submission_id"],
            "version": row["version"],
            "size_label": row["size_label"],
            "height_in": float(row["height_in"]),
            "width_in": float(row["width_in"]),
            "sheets": row["sheets"],
            "dpi": row["dpi"],
            "status": row["status"],
            "progress": row["progress"],
            "error": row["error"],
            "spreads": [{"index": s["index"], "recipe": s["recipe"], "photos": s.get("photos", [])} for s in spreads],
            "published_folder_id": row["gallery_folder_id"],
            "created_at": row["created_at"].isoformat(),
            "ready_at": row["ready_at"].isoformat() if row["ready_at"] else None,
        }


def _readme(spec: design.AlbumSpec, row: dict, spreads: int, photos: int) -> str:
    width, height = spec.size_px
    return (
        f"FrameBucket album design v{row['version']}\n"
        f"Size: {row['size_label']} in (height x open spread), {spreads} spreads, {photos} photos\n"
        f"Each spread: {width} x {height} px at {spec.dpi} ppi, sRGB.\n\n"
        "print-jpeg.zip  - flattened spreads ready for the lab.\n"
        "photoshop.zip   - one layered PSD per spread: Background, panels, white frames and one layer per\n"
        "                  photo (named with the photo number and file name). Guides mark the centre fold\n"
        "                  and a 0.25 in safe area. Swap or re-crop photos using the originals from the\n"
        "                  customer's album selection download.\n"
    )
