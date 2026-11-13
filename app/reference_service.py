"""Reference album designs (Studio and Business plans): upload finished spreads once, the album designer
follows the best-fitting one for every sheet."""
from __future__ import annotations

import uuid

from psycopg.types.json import Jsonb

from . import album_reference
from .storage import validate_id


class ReferenceLimitReached(ValueError):
    """The studio already has as many reference designs as its plan allows."""


class AlbumReferenceMixin:
    MAX_ALBUM_REFERENCE_BYTES = album_reference.MAX_REFERENCE_BYTES

    def _reference_rows(self, photographer_id: str) -> list[dict]:
        with self.db.transaction() as connection:
            return list(connection.execute(
                "SELECT reference_id, name, filename, layout, created_at FROM fb_data.album_reference "
                "WHERE photographer_id = %s ORDER BY created_at, reference_id",
                (photographer_id,),
            ).fetchall())

    def reference_layouts(self, photographer_id: str | None) -> list[dict]:
        """Layouts used by the album designer (each tagged with its reference id)."""
        if not photographer_id:
            return []
        layouts = []
        for row in self._reference_rows(photographer_id):
            layout = dict(row["layout"] or {})
            if layout.get("boxes"):
                layout["reference_id"] = row["reference_id"]
                layout["name"] = row["name"]
                layouts.append(layout)
        return layouts

    def album_references(self, photographer_id: str) -> dict:
        photographer_id = validate_id(photographer_id, "photographer_id")
        rows = self._reference_rows(photographer_id)
        return {
            "references": [{
                "id": row["reference_id"],
                "name": row["name"],
                "filename": row["filename"],
                "count": (row["layout"] or {}).get("count", 0),
                "kind": (row["layout"] or {}).get("kind", ""),
                "background": (row["layout"] or {}).get("background"),
                "boxes": (row["layout"] or {}).get("boxes", []),
                "created_at": row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else row["created_at"],
            } for row in rows],
            "count": len(rows),
        }

    def add_album_references(self, photographer_id: str, files: list[tuple[str, bytes]], limit: int | None) -> dict:
        """Analyse and store spreads. All-or-nothing against the limit; unreadable files are reported."""
        photographer_id = validate_id(photographer_id, "photographer_id")
        if not files:
            raise ValueError("Choose the album spreads to upload.")
        analysed, problems = [], []
        for filename, data in files:
            try:
                image = album_reference.load(data)
                analysed.append((filename, album_reference.analyse(image), album_reference.thumbnail(image)))
                image.close()
            except album_reference.ReferenceError as exc:
                problems.append(f"{filename}: {exc}")
        if not analysed:
            raise ValueError(" ".join(problems) or "None of those files could be used.")
        with self.db.transaction() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("albumref:" + photographer_id,))
            count = int(connection.execute(
                "SELECT count(*) AS n FROM fb_data.album_reference WHERE photographer_id = %s", (photographer_id,)
            ).fetchone()["n"])
            if limit is not None and count + len(analysed) > limit:
                left = max(0, limit - count)
                raise ReferenceLimitReached(
                    f"Your plan allows {limit} reference designs and you have {count}. "
                    + (f"You can add {left} more; " if left else "")
                    + "delete some or upgrade your plan.")
            for filename, layout, thumb in analysed:
                name = (filename or "Spread").rsplit("/", 1)[-1].rsplit(".", 1)[0][:80] or "Spread"
                connection.execute(
                    """
                    INSERT INTO fb_data.album_reference (reference_id, photographer_id, name, filename, thumb, layout)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (uuid.uuid4().hex, photographer_id, name, (filename or "spread.jpg")[-200:], thumb, Jsonb(layout)),
                )
        result = self.album_references(photographer_id)
        result["problems"] = problems
        return result

    def delete_album_reference(self, photographer_id: str, reference_id: str) -> dict:
        photographer_id = validate_id(photographer_id, "photographer_id")
        with self.db.transaction() as connection:
            row = connection.execute(
                "DELETE FROM fb_data.album_reference WHERE photographer_id = %s AND reference_id = %s RETURNING reference_id",
                (photographer_id, reference_id),
            ).fetchone()
        if not row:
            raise LookupError("Reference design not found.")
        return self.album_references(photographer_id)

    def album_reference_thumb(self, photographer_id: str, reference_id: str) -> bytes:
        photographer_id = validate_id(photographer_id, "photographer_id")
        with self.db.transaction() as connection:
            row = connection.execute(
                "SELECT thumb FROM fb_data.album_reference WHERE photographer_id = %s AND reference_id = %s",
                (photographer_id, reference_id),
            ).fetchone()
        if not row:
            raise LookupError("Reference design not found.")
        return bytes(row["thumb"])
