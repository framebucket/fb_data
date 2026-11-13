"""Reference album designs: the studio uploads finished spreads it likes (JPG/PNG exports from its album
software) once. Each spread is reduced to a layout: where the photos sit, whether they are framed, the
background colour and any full-bleed hero. When an album is designed, every sheet is matched to the
reference whose layout suits that sheet's photos best (count and shapes), and laid out the same way.

Detection (OpenCV, on a ~1200 px copy):
1. Background colour = median of the outer border ring.
2. Photo pixels = pixels far from the background in Lab, or with strong local texture; closed, opened and
   filled, so each photo becomes one solid blob.
3. Blobs become rectangles; tiny ones (ornaments, text) are dropped and overlapping ones merged.
4. A white ring just outside a rectangle means the photo is framed.
"""
from __future__ import annotations

import io
import math
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

ANALYSIS_WIDTH = 1200
MIN_BOX_AREA = 0.012          # of the spread; smaller blobs are ornaments or text
MAX_BOXES = 16
MAX_REFERENCE_BYTES = 40 * 1024 * 1024
THUMB_WIDTH = 480


class ReferenceError(ValueError):
    """The upload isn't a usable album spread."""


def load(data: bytes) -> Image.Image:
    if len(data) > MAX_REFERENCE_BYTES:
        raise ReferenceError("Each reference spread must be 40 MB or smaller.")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            if opened.width * opened.height > 200_000_000:
                raise ReferenceError("That spread is larger than 200 megapixels.")
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ReferenceError("Reference spreads must be JPG, PNG or WebP images.") from exc
    aspect = image.width / max(1, image.height)
    if aspect < 1.2 or aspect > 4.0:
        raise ReferenceError("Upload a full open spread (both pages side by side), wider than it is tall.")
    return image


def thumbnail(image: Image.Image) -> bytes:
    small = image.copy()
    small.thumbnail((THUMB_WIDTH, THUMB_WIDTH), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    small.save(stream, "JPEG", quality=82, optimize=True)
    return stream.getvalue()


def analyse(image: Image.Image) -> dict:
    """Return the layout of a spread: normalised boxes (x, y, w, h), frames, background, kind."""
    small = image.convert("RGB")
    if small.width > ANALYSIS_WIDTH:
        small = small.resize((ANALYSIS_WIDTH, max(1, round(small.height * ANALYSIS_WIDTH / small.width))),
                             Image.Resampling.BILINEAR)
    rgb = np.asarray(small)
    height, width = rgb.shape[:2]
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    ring = max(3, round(min(width, height) * 0.012))
    border = np.concatenate([lab[:ring].reshape(-1, 3), lab[-ring:].reshape(-1, 3),
                             lab[:, :ring].reshape(-1, 3), lab[:, -ring:].reshape(-1, 3)])
    background = np.median(border, axis=0)
    border_spread = float(np.median(np.linalg.norm(border - background, axis=1)))

    if border_spread > 14:
        # The edges aren't a plain page colour: one photo bleeds over the whole spread (a panorama).
        bg_rgb = cv2.cvtColor(background.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_Lab2RGB)[0, 0]
        return {"version": 1, "aspect": round(width / height, 4), "boxes": [[0.0, 0.0, 1.0, 1.0]], "framed": [False],
                "full_bleed": [True], "count": 1, "pages": [1, 0],
                "background": "#%02x%02x%02x" % tuple(int(v) for v in np.clip(bg_rgb * 255, 0, 255)), "kind": "panorama"}
    distance = np.linalg.norm(lab - background, axis=2)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    texture = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (0, 0), 3)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    # White frames and gutters between photos: bright, colourless and flat. They separate photos, so they
    # are never part of a photo blob even when the background is a similar cream.
    # Only thin, long, straight runs count, so a white dress or sky inside a photo isn't mistaken for one.
    bright = ((lab[..., 0] > 86) & (chroma < 10)).astype(np.uint8)
    long_h = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((1, max(15, width // 12)), np.uint8))
    long_v = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((max(15, height // 12), 1), np.uint8))
    thick = max(9, round(min(width, height) * 0.03))
    wide_v = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((thick, 1), np.uint8))
    wide_h = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((1, thick), np.uint8))
    gutter = ((long_h > 0) & (wide_v == 0)) | ((long_v > 0) & (wide_h == 0))
    gutter = cv2.dilate(gutter.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    photo = ((distance > max(10.0, border_spread * 3)) | (texture > 6.0)).astype(np.uint8) * 255
    small_kernel = np.ones((3, 3), np.uint8)
    photo = cv2.morphologyEx(photo, cv2.MORPH_CLOSE, small_kernel, iterations=2)  # fill speckle inside photos
    photo[gutter] = 0                                                              # then cut along the gutters
    mask = cv2.morphologyEx(photo, cv2.MORPH_OPEN, small_kernel, iterations=2)    # and drop thin bridges

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[list[float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < MIN_BOX_AREA * width * height:
            continue
        fill = cv2.contourArea(contour) / max(1.0, w * h)
        if fill < 0.55:            # ragged shapes (flourishes, text blocks) aren't photo frames
            continue
        boxes.append([x, y, w, h])
    boxes = _merge(boxes)
    # A layout that fills (almost) the whole spread is one full-bleed panorama.
    if not boxes or (len(boxes) == 1 and boxes[0][2] * boxes[0][3] > 0.9 * width * height):
        boxes = [[0, 0, width, height]]
    boxes = sorted(boxes, key=lambda b: (round(b[0] / (width * 0.05)), b[1]))[:MAX_BOXES]

    framed = [_framed(rgb, box) for box in boxes]
    normalised = [[round(x / width, 4), round(y / height, 4), round(w / width, 4), round(h / height, 4)]
                  for x, y, w, h in boxes]
    fold = 0.5
    left = sum(1 for b in normalised if b[0] + b[2] / 2 < fold)
    full_bleed = [b[2] * b[3] > 0.38 and (b[0] < 0.02 or b[0] + b[2] > 0.98) for b in normalised]
    bg_rgb = cv2.cvtColor(background.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_Lab2RGB)[0, 0]
    colour = "#%02x%02x%02x" % tuple(int(v) for v in np.clip(bg_rgb * 255, 0, 255))
    return {
        "version": 1,
        "aspect": round(width / height, 4),
        "boxes": normalised,
        "framed": framed,
        "full_bleed": full_bleed,
        "count": len(normalised),
        "pages": [left, len(normalised) - left],
        "background": colour,
        "kind": _kind(normalised, full_bleed),
    }


def _merge(boxes: list[list[float]]) -> list[list[float]]:
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ix = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                iy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                if ix > 0 and iy > 0 and ix * iy > 0.3 * min(a[2] * a[3], b[2] * b[3]):
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


def _framed(rgb: np.ndarray, box: Sequence[float]) -> bool:
    """A white line running along each edge of the photo (just inside or outside the box) means a frame."""
    x, y, w, h = (int(v) for v in box)
    height, width = rgb.shape[:2]
    reach = max(4, round(min(w, h) * 0.025))
    white = np.all(rgb > 228, axis=2)

    def best(lines) -> float:
        fractions = [float(np.mean(line)) for line in lines if line.size]
        return max(fractions) if fractions else 0.0

    x0, x1 = max(0, x), min(width, x + w)
    y0, y1 = max(0, y), min(height, y + h)
    sides = [
        best(white[r, x0:x1] for r in range(max(0, y - reach), min(height, y + reach))),
        best(white[r, x0:x1] for r in range(max(0, y + h - reach), min(height, y + h + reach))),
        best(white[y0:y1, c] for c in range(max(0, x - reach), min(width, x + reach))),
        best(white[y0:y1, c] for c in range(max(0, x + w - reach), min(width, x + w + reach))),
    ]
    return float(np.median(sides)) > 0.6


def _kind(boxes: Sequence[Sequence[float]], full_bleed: Sequence[bool]) -> str:
    if len(boxes) == 1:
        return "panorama" if boxes[0][2] > 0.8 else "single hero"
    if any(full_bleed):
        return "hero page + grid"
    return "grid"


# --------------------------------------------------------------------------- matching sheets to references

def match_cost(layout: dict, aspects: Sequence[float], spread_aspect: float) -> tuple[float, list[int]]:
    """How well a sheet's photos fit a reference layout. Returns (cost, box index for each photo)."""
    boxes = layout["boxes"]
    if len(boxes) != len(aspects):
        return math.inf, []
    box_aspects = [(b[2] * spread_aspect) / max(1e-6, b[3]) for b in boxes]
    # Pair the most landscape photo with the most landscape box, and so on (minimises total crop).
    photo_order = sorted(range(len(aspects)), key=lambda i: aspects[i])
    box_order = sorted(range(len(boxes)), key=lambda i: box_aspects[i])
    assignment = [0] * len(aspects)
    cost = 0.0
    for p, b in zip(photo_order, box_order):
        assignment[p] = b
        cost += abs(math.log(aspects[p] / box_aspects[b]))
    cost /= len(aspects)
    cost += 0.25 * abs(math.log(layout.get("aspect", spread_aspect) / spread_aspect))
    return cost, assignment


def choose(layouts: Sequence[dict], aspects: Sequence[float], spread_aspect: float,
           used: dict[int, int], previous: int | None) -> tuple[int | None, list[int]]:
    """Index of the best reference for a sheet (None if none fits), with the photo->box assignment."""
    best: tuple[float, int | None, list[int]] = (math.inf, None, [])
    for index, layout in enumerate(layouts):
        cost, assignment = match_cost(layout, aspects, spread_aspect)
        if not math.isfinite(cost):
            continue
        cost += 0.15 * used.get(index, 0) + (0.4 if index == previous else 0.0)   # keep the album varied
        if cost < best[0]:
            best = (cost, index, assignment)
    # Too much cropping: the built-in designer does better than forcing the reference.
    if best[1] is None or best[0] > 0.75:
        return None, []
    return best[1], best[2]
