"""Customer wedding invitations: details, optional couple photo, five designs, regenerate, finalize to MP4.

Mixed into PhotoService. The Java backend authorises the customer for the event; this layer only checks that
the event/customer pair matches the invitation it created.
"""
from __future__ import annotations

import io
import random
import shutil
import tempfile
from pathlib import Path

from PIL import Image
from psycopg.types.json import Jsonb

from . import invitation as engine
from . import jobs
from .storage import normalize_key, validate_id

MAX_BATCHES = 40                 # regenerate limit per invitation
MAX_PHOTO_BYTES = 25 * 1024 * 1024
PREVIEW_WIDTH = 720


class InvitationMixin:
    # ------------------------------------------------------------------ API-facing

    def invitation(self, event_id: str, photographer_id: str, customer_id: str) -> dict:
        row = self._invitation_row(event_id, customer_id)
        if not row:
            return {"exists": False, "details": None, "has_photo": False, "designs": [], "status": "DRAFT",
                    "regenerations_left": MAX_BATCHES, "video_ready": False, "ffmpeg": engine.ffmpeg_available()}
        return self._invitation_dict(row)

    def save_invitation_details(self, event_id: str, photographer_id: str, customer_id: str, raw: dict) -> dict:
        details = engine.normalise_details(raw)
        event_id, photographer_id, customer_id = (validate_id(event_id, "event_id"), validate_id(photographer_id, "photographer_id"),
                                                  validate_id(customer_id, "customer_id"))
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO fb_data.invitation (event_id, photographer_id, customer_id, details)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (event_id) DO UPDATE SET details = EXCLUDED.details, updated_at = now()
                WHERE fb_data.invitation.customer_id = EXCLUDED.customer_id
                """,
                (event_id, photographer_id, customer_id, Jsonb(details)),
            )
        return self.invitation(event_id, photographer_id, customer_id)

    def save_invitation_photo(self, event_id: str, photographer_id: str, customer_id: str, data: bytes) -> dict:
        row = self._require_invitation(event_id, customer_id)
        if len(data) > MAX_PHOTO_BYTES:
            raise engine.InvitationError("The photo must be 25 MB or smaller.")
        image, faces = engine.prepare_photo(data, self.face_matcher.face_boxes if self.settings.face_matching_enabled else None)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=92)
        key = f"invitations/{row['event_id']}/couple-{random.randrange(1 << 40):x}.jpg"
        self.storage.write_bytes(key, buffer.getvalue(), "image/jpeg")
        old = row["photo_key"]
        with self.db.transaction() as connection:
            connection.execute("UPDATE fb_data.invitation SET photo_key = %s, photo_faces = %s, updated_at = now() "
                               "WHERE event_id = %s", (key, Jsonb(faces), row["event_id"]))
        if old:
            self._quiet_delete(old)
        return self.invitation(event_id, photographer_id, customer_id) | {"faces_found": len(faces)}

    def delete_invitation_photo(self, event_id: str, photographer_id: str, customer_id: str) -> dict:
        row = self._require_invitation(event_id, customer_id)
        with self.db.transaction() as connection:
            connection.execute("UPDATE fb_data.invitation SET photo_key = NULL, photo_faces = NULL, updated_at = now() "
                               "WHERE event_id = %s", (row["event_id"],))
        if row["photo_key"]:
            self._quiet_delete(row["photo_key"])
        return self.invitation(event_id, photographer_id, customer_id)

    def generate_invitation_designs(self, event_id: str, photographer_id: str, customer_id: str) -> dict:
        """Five fresh designs (themes not shown in the previous batch, where possible)."""
        row = self._require_invitation(event_id, customer_id)
        if not row["details"]:
            raise engine.InvitationError("Fill in the invitation details first.")
        if row["batches"] >= MAX_BATCHES:
            raise engine.InvitationError("You've reached the limit of new designs for this invitation.")
        if row["status"] == "RENDERING":
            raise engine.InvitationError("Your video is being made right now. Please wait for it to finish.")
        photo, faces = self._invitation_photo(row)
        previous = {d["theme"] for d in (row["designs"] or [])}
        seed = random.randrange(1, 2_000_000_000)
        choices = engine.choose_designs(seed, photo is not None, avoid=previous)
        batch = row["batches"] + 1
        designs = []
        for index, choice in enumerate(choices):
            poster = engine.render_poster(engine.build_scene(row["details"], choice, photo, faces))
            preview = poster.resize((PREVIEW_WIDTH, int(PREVIEW_WIDTH * poster.height / poster.width)), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            preview.save(buffer, "JPEG", quality=86, optimize=True)
            key = f"invitations/{row['event_id']}/batch-{batch}-{index}.jpg"
            self.storage.write_bytes(key, buffer.getvalue(), "image/jpeg")
            designs.append(choice.as_dict() | {"preview_key": key})
        old_keys = [d["preview_key"] for d in (row["designs"] or [])]
        with self.db.transaction() as connection:
            connection.execute(
                "UPDATE fb_data.invitation SET designs = %s, batches = %s, updated_at = now() WHERE event_id = %s",
                (Jsonb(designs), batch, row["event_id"]),
            )
        for key in old_keys:
            if key != (row["chosen"] or {}).get("preview_key"):
                self._quiet_delete(key)
        return self.invitation(event_id, photographer_id, customer_id)

    def invitation_preview(self, event_id: str, customer_id: str, index: int) -> bytes:
        row = self._require_invitation(event_id, customer_id)
        designs = row["designs"] or []
        if not 0 <= index < len(designs):
            raise FileNotFoundError("Design not found.")
        return self.storage.read_bytes(normalize_key(designs[index]["preview_key"]))

    def finalize_invitation(self, event_id: str, photographer_id: str, customer_id: str, index: int) -> dict:
        row = self._require_invitation(event_id, customer_id)
        designs = row["designs"] or []
        if not 0 <= index < len(designs):
            raise engine.InvitationError("Choose one of the designs.")
        if row["status"] == "RENDERING":
            raise engine.InvitationError("Your video is already being made.")
        chosen = designs[index]
        with self.db.transaction() as connection:
            connection.execute(
                "UPDATE fb_data.invitation SET chosen = %s, status = 'RENDERING', progress = 0, error = NULL, "
                "updated_at = now() WHERE event_id = %s",
                (Jsonb(chosen | {"details": row["details"], "photo_key": row["photo_key"],
                                 "photo_faces": row["photo_faces"]}), row["event_id"]),
            )
            jobs.enqueue(connection, jobs.RENDER_INVITATION, {"event_id": row["event_id"]}, 2)
        return self.invitation(event_id, photographer_id, customer_id)

    def invitation_file(self, event_id: str, customer_id: str, kind: str) -> tuple[Path, bool, str]:
        """(path, cleanup, content type) for the finished video or poster."""
        row = self._require_invitation(event_id, customer_id)
        key = row["video_key"] if kind == "video" else row["poster_key"]
        if row["status"] != "READY" or not key:
            raise FileNotFoundError("The invitation isn't ready yet.")
        suffix = ".mp4" if kind == "video" else ".jpg"
        path, cleanup = self.storage.materialize(normalize_key(key), suffix=suffix, prefix="fb-invite-")
        return path, cleanup, "video/mp4" if kind == "video" else "image/jpeg"

    # ------------------------------------------------------------------ worker job

    def render_invitation(self, event_id: str) -> None:
        with self.db.transaction() as connection:
            row = connection.execute("SELECT * FROM fb_data.invitation WHERE event_id = %s", (validate_id(event_id, "event_id"),)).fetchone()
        if not row or row["status"] != "RENDERING" or not row["chosen"]:
            return
        chosen = row["chosen"]
        photo, faces = None, chosen.get("photo_faces") or []
        if chosen.get("photo_key"):
            try:
                with Image.open(io.BytesIO(self.storage.read_bytes(normalize_key(chosen["photo_key"])))) as opened:
                    photo = opened.convert("RGB")
            except Exception:
                photo = None
        choice = engine.DesignChoice(chosen["theme"], chosen["layout"], int(chosen["seed"]))
        scene = engine.build_scene(chosen["details"], choice, photo, faces)
        work = Path(tempfile.mkdtemp(prefix="invite-", dir=self.settings.temp_root))
        try:
            poster = engine.render_poster(scene)
            poster_path = work / "invitation.jpg"
            poster.save(poster_path, "JPEG", quality=93, optimize=True)
            video_path = work / "invitation.mp4"

            def progress(fraction: float) -> None:
                with self.db.transaction() as connection:
                    connection.execute("UPDATE fb_data.invitation SET progress = %s WHERE event_id = %s",
                                       (int(5 + 90 * fraction), row["event_id"]))

            engine.render_video(scene, video_path, progress)
            stamp = random.randrange(1 << 40)
            poster_key = f"invitations/{row['event_id']}/final-{stamp:x}.jpg"
            video_key = f"invitations/{row['event_id']}/final-{stamp:x}.mp4"
            self.storage.upload_file(poster_key, poster_path)
            self.storage.upload_file(video_key, video_path)
            with self.db.transaction() as connection:
                connection.execute(
                    "UPDATE fb_data.invitation SET status = 'READY', progress = 100, poster_key = %s, video_key = %s, "
                    "error = NULL, ready_at = now(), updated_at = now() WHERE event_id = %s",
                    (poster_key, video_key, row["event_id"]),
                )
            for key in (row["poster_key"], row["video_key"]):
                if key:
                    self._quiet_delete(key)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # ------------------------------------------------------------------ helpers

    def _invitation_row(self, event_id: str, customer_id: str) -> dict | None:
        with self.db.transaction() as connection:
            row = connection.execute("SELECT * FROM fb_data.invitation WHERE event_id = %s",
                                     (validate_id(event_id, "event_id"),)).fetchone()
        if row and row["customer_id"] != validate_id(customer_id, "customer_id"):
            raise ValueError("This invitation belongs to another customer.")
        return row

    def _require_invitation(self, event_id: str, customer_id: str) -> dict:
        row = self._invitation_row(event_id, customer_id)
        if not row:
            raise engine.InvitationError("Fill in the invitation details first.")
        return row

    def _invitation_photo(self, row: dict) -> tuple[Image.Image | None, list]:
        if not row["photo_key"]:
            return None, []
        try:
            with Image.open(io.BytesIO(self.storage.read_bytes(normalize_key(row["photo_key"])))) as opened:
                return opened.convert("RGB"), row["photo_faces"] or []
        except Exception:
            return None, []

    def _quiet_delete(self, key: str) -> None:
        try:
            self.storage.delete(normalize_key(key))
        except Exception:
            pass

    @staticmethod
    def _invitation_dict(row: dict) -> dict:
        chosen = row["chosen"] or {}
        return {
            "exists": True,
            "details": row["details"],
            "has_photo": bool(row["photo_key"]),
            "designs": [{"index": i, "theme": d["theme"], "label": d["label"], "layout": d["layout"]}
                        for i, d in enumerate(row["designs"] or [])],
            "chosen": {"theme": chosen.get("theme"), "label": chosen.get("label"), "layout": chosen.get("layout")} if chosen else None,
            "status": row["status"],
            "progress": row["progress"],
            "error": row["error"],
            "regenerations_left": max(0, MAX_BATCHES - row["batches"]),
            "video_ready": row["status"] == "READY" and bool(row["video_key"]),
            "ffmpeg": engine.ffmpeg_available(),
            "updated_at": row["updated_at"].isoformat(),
        }
