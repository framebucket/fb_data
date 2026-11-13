"""Reference album designs: layout detection from a finished spread and per-sheet matching."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app import album_design, album_reference


def spread(boxes, size=(2400, 1200), background=(246, 238, 220), frame=8, seed=0):
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", size, background)
    for x, y, w, h in boxes:
        px = [int(v) for v in (x * size[0], y * size[1], w * size[0], h * size[1])]
        if frame:
            ImageDraw.Draw(image).rectangle([px[0] - frame, px[1] - frame, px[0] + px[2] + frame, px[1] + px[3] + frame],
                                            fill=(255, 255, 255))
        noise = rng.integers(20, 200, (px[3], px[2], 3), dtype=np.uint8)
        noise[:, :, int(rng.integers(0, 3))] //= 3
        image.paste(Image.fromarray(noise), (px[0], px[1]))
    stream = io.BytesIO()
    image.save(stream, "JPEG", quality=90)
    return stream.getvalue()


HERO_GRID = [(0.04, 0.06, 0.42, 0.88), (0.54, 0.06, 0.20, 0.42), (0.76, 0.06, 0.20, 0.42), (0.54, 0.52, 0.42, 0.42)]


def test_layout_is_recovered():
    layout = album_reference.analyse(album_reference.load(spread(HERO_GRID)))
    assert layout["count"] == 4 and layout["pages"] == [1, 3]
    assert all(layout["framed"])
    for expected in HERO_GRID:
        assert any(all(abs(a - b) < 0.025 for a, b in zip(found, expected)) for found in layout["boxes"]), layout["boxes"]
    assert layout["background"].startswith("#f")


def test_touching_photos_split_by_white_gutter():
    boxes = [(0.03, 0.05, 0.3, 0.9), (0.335, 0.05, 0.3, 0.9), (0.64, 0.05, 0.33, 0.9)]
    layout = album_reference.analyse(album_reference.load(spread(boxes, frame=5)))
    assert layout["count"] == 3


def test_full_bleed_panorama_and_bad_uploads():
    layout = album_reference.analyse(album_reference.load(spread([(0, 0, 1, 1)], frame=0)))
    assert layout["count"] == 1 and layout["kind"] == "panorama"
    with pytest.raises(album_reference.ReferenceError):
        album_reference.load(b"not an image")
    portrait = io.BytesIO(); Image.new("RGB", (800, 1200), "white").save(portrait, "PNG")
    with pytest.raises(album_reference.ReferenceError):
        album_reference.load(portrait.getvalue())


def _photo(i, aspect):
    return album_design.PhotoInfo(f"p{i}", i, f"IMG_{i}.jpg", int(1000 * aspect), 1000, None, [], 50.0,
                                  np.full(51, i % 3, dtype=np.float32))


def test_sheets_follow_matching_reference():
    layout = album_reference.analyse(album_reference.load(spread(HERO_GRID)))
    layout["reference_id"] = "ref1"
    spread_aspect = 2.0
    aspects = [0.66, 1.0, 1.0, 2.0]
    index, assignment = album_reference.choose([layout], aspects, spread_aspect, {}, None)
    assert index == 0 and sorted(assignment) == [0, 1, 2, 3]
    assert album_reference.choose([layout], [1.0, 1.0], spread_aspect, {}, None)[0] is None   # wrong count

    photos = [_photo(i, a) for i, a in enumerate(aspects * 2)]
    spec = album_design.AlbumSpec(12, 24, 2, dpi=50)
    plans = album_design.plan_album(photos, spec, [layout])
    assert [p.recipe for p in plans] == ["reference", "reference"] and plans[0].reference == "ref1"
    width, height = spec.size_px
    for placement in plans[0].placements:
        x, y, w, h = placement.box
        assert 0 <= x and x + w <= width + 1 and 0 <= y and y + h <= height + 1
    # Without references the built-in recipes are used.
    assert all(p.recipe != "reference" for p in album_design.plan_album(photos, spec))
