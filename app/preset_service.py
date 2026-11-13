"""Studio editing presets: upload once, the service picks the right one for each photo."""
from __future__ import annotations

import logging
import tempfile
import uuid
from io import BytesIO
from pathlib import Path

from psycopg.types.json import Jsonb

from . import presets as preset_engine
from . import style as style_engine
from .imaging import IMAGE_EXTENSIONS, StudioLook
from .storage import validate_id

logger = logging.getLogger(__name__)


class PresetLimitReached(ValueError):
    """The studio already has as many presets as its plan allows."""


class PresetMixin:
    MAX_PRESET_BYTES = preset_engine.MAX_FILE_BYTES

    # ---- reading ---------------------------------------------------------------------------------

    def _preset_rows(self, photographer_id: str, with_lut: bool = False) -> list[dict]:
        columns = "preset_id, name, filename, kind, scene, summary, created_at" + (", lut" if with_lut else "")
        with self.db.transaction() as connection:
            return list(connection.execute(
                f"SELECT {columns} FROM fb_data.studio_preset WHERE photographer_id = %s ORDER BY created_at, preset_id",
                (photographer_id,),
            ).fetchall())

    def _compiled_presets(self, photographer_id: str) -> list[preset_engine.Preset]:
        compiled = []
        for row in self._preset_rows(photographer_id, with_lut=True):
            try:
                compiled.append(preset_engine.Preset(row["preset_id"], row["name"], row["scene"],
                                                     preset_engine.unpack(bytes(row["lut"])), row["summary"] or {}))
            except Exception:  # a damaged row must never stop processing
                logger.warning("Skipping unreadable preset %s.", row["preset_id"], exc_info=True)
        return compiled

    @staticmethod
    def _preset_json(row: dict) -> dict:
        summary = row["summary"] or {}
        return {
            "id": row["preset_id"],
            "name": row["name"],
            "filename": row["filename"],
            "kind": row["kind"],
            "scene": row["scene"],
            "words": summary.get("words", []),
            "swatches": summary.get("swatches", []),
            "skipped": summary.get("skipped", []),
            "created_at": row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else row["created_at"],
        }

    def studio_presets(self, photographer_id: str) -> dict:
        photographer_id = validate_id(photographer_id, "photographer_id")
        rows = self._preset_rows(photographer_id)
        style_row = self._style_row(photographer_id)
        return {
            "presets": [self._preset_json(row) for row in rows],
            "count": len(rows),
            "use_presets": style_row["use_presets"] if style_row and "use_presets" in style_row else True,
        }

    # ---- changing --------------------------------------------------------------------------------

    def add_preset(self, photographer_id: str, filename: str, data: bytes, name: str | None = None,
                   scene: str = "any", limit: int | None = None) -> dict:
        photographer_id = validate_id(photographer_id, "photographer_id")
        if scene not in preset_engine.SCENES:
            raise ValueError("Scene must be one of: " + ", ".join(preset_engine.SCENES) + ".")
        if not data:
            raise ValueError("The preset file is empty.")
        parsed_name, lut, summary = preset_engine.parse(filename or "", data)
        clean_name = (name or "").strip()[:80] or parsed_name
        preset_id = uuid.uuid4().hex
        kind = summary.pop("kind")
        with self.db.transaction() as connection:
            # One upload at a time per studio, so the plan limit can't be overshot by parallel uploads.
            connection.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("preset:" + photographer_id,))
            count = connection.execute(
                "SELECT count(*) AS n FROM fb_data.studio_preset WHERE photographer_id = %s", (photographer_id,)
            ).fetchone()["n"]
            if limit is not None and int(count) >= limit:
                raise PresetLimitReached(
                    f"Your plan allows {limit} editing presets. Delete one or upgrade your plan to add more.")
            connection.execute(
                """
                INSERT INTO fb_data.studio_preset (preset_id, photographer_id, name, filename, kind, scene, lut, summary)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (preset_id, photographer_id, clean_name, (filename or "preset")[-200:], kind, scene,
                 preset_engine.pack(lut), Jsonb(summary)),
            )
        self._forget_look(photographer_id)
        return self.studio_presets(photographer_id)

    def update_preset(self, photographer_id: str, preset_id: str, name: str | None, scene: str | None) -> dict:
        photographer_id = validate_id(photographer_id, "photographer_id")
        if scene is not None and scene not in preset_engine.SCENES:
            raise ValueError("Scene must be one of: " + ", ".join(preset_engine.SCENES) + ".")
        clean_name = name.strip()[:80] if name is not None else None
        if clean_name == "":
            raise ValueError("Give the preset a name.")
        with self.db.transaction() as connection:
            row = connection.execute(
                """
                UPDATE fb_data.studio_preset SET name = COALESCE(%s, name), scene = COALESCE(%s, scene)
                WHERE photographer_id = %s AND preset_id = %s RETURNING preset_id
                """,
                (clean_name, scene, photographer_id, preset_id),
            ).fetchone()
        if not row:
            raise LookupError("Preset not found.")
        self._forget_look(photographer_id)
        return self.studio_presets(photographer_id)

    def delete_preset(self, photographer_id: str, preset_id: str) -> dict:
        photographer_id = validate_id(photographer_id, "photographer_id")
        with self.db.transaction() as connection:
            row = connection.execute(
                "DELETE FROM fb_data.studio_preset WHERE photographer_id = %s AND preset_id = %s RETURNING preset_id",
                (photographer_id, preset_id),
            ).fetchone()
        if not row:
            raise LookupError("Preset not found.")
        self._forget_look(photographer_id)
        return self.studio_presets(photographer_id)

    def set_use_presets(self, photographer_id: str, enabled: bool) -> dict:
        photographer_id = validate_id(photographer_id, "photographer_id")
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO fb_data.studio_style (photographer_id, use_presets, updated_at) VALUES (%s, %s, now())
                ON CONFLICT (photographer_id) DO UPDATE SET use_presets = EXCLUDED.use_presets, updated_at = now()
                """,
                (photographer_id, bool(enabled)),
            )
        self._forget_look(photographer_id)
        return self.studio_presets(photographer_id)

    # ---- trying it on a photo -------------------------------------------------------------------

    def preset_preview(self, photographer_id: str, filename: str, data: bytes,
                       preset_id: str | None = None) -> tuple[bytes, dict]:
        """Render a sample photo as processing would; the AI picks the preset unless one is forced."""
        photographer_id = validate_id(photographer_id, "photographer_id")
        suffix = Path(filename or "sample.jpg").suffix.lower() or ".jpg"
        if suffix not in IMAGE_EXTENSIONS:
            raise ValueError("Preview needs a JPG, PNG, WebP, TIFF or RAW photo.")
        compiled = self._compiled_presets(photographer_id)
        if preset_id:
            compiled = [preset for preset in compiled if preset.preset_id == preset_id]
        if not compiled:
            raise ValueError("Upload an editing preset first.")
        with tempfile.NamedTemporaryFile(prefix="preset-", suffix=suffix, dir=self.settings.temp_root,
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
        row = self._style_row(photographer_id)
        look = StudioLook(
            profile=row["profile"] if row else None,
            baseline=row["baseline"] if row else None,
            strength=(row["strength"] if row else 80) / 100.0,
            logo=self._studio_logo(photographer_id),
            stamp_logo=row["stamp_logo"] if row else True,
            presets=tuple(compiled),
        )
        rendered, info = self.processor.render(image, look)
        stream = BytesIO()
        rendered.save(stream, format="JPEG", quality=88, optimize=True, progressive=True)
        return stream.getvalue(), {"preset": info.get("preset"), "scene": info.get("scene")}
