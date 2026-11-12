"""Automatic album design: from a customer's selected photos to print-ready spreads and layered PSDs.

Pipeline
1. Analyse every photo on a small proxy: capture time (EXIF), orientation, faces, sharpness and a
   "hero" score (big, sharp faces), plus a colour signature used to detect scene changes.
2. Story: order photos by capture time (photo number as fallback), cut them into one contiguous chunk
   per spread (sheet), preferring cuts at scene changes and keeping spread sizes even.
3. Spread recipes, in the style of Indian event albums: a full-page hero next to a grid, grids on
   both pages, a full-bleed panorama, white-bordered photos on soft tinted backgrounds, offset colour
   panels and faded "ghost" photo backgrounds.
4. Page layouts are justified grids (rows or columns) chosen to minimise cropping; every crop is
   face-aware and faces are kept out of the centre fold.
5. Render each spread at print resolution (JPEG for the lab), a screen preview, and a layered PSD
   (background, panels, frames and one layer per photo) with fold and safe-area guides.

Everything runs locally; no external AI service.
"""
from __future__ import annotations

import hashlib
import io
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from . import psd

logger = logging.getLogger(__name__)

# Spread sizes in inches as (height, open spread width), named the way Indian labs quote them.
ALBUM_SIZES: dict[str, tuple[float, float]] = {
    "12x24": (12, 24), "12x30": (12, 30), "12x36": (12, 36), "10x24": (10, 24),
    "15x24": (15, 24), "15x30": (15, 30), "16x24": (16, 24), "17x24": (17, 24), "18x24": (18, 24),
}
MAX_PHOTOS_PER_PAGE = 8
PREVIEW_WIDTH = 2400
ANALYSIS_SIDE = 768

Box = tuple[float, float, float, float]          # x, y, w, h


# --------------------------------------------------------------------------- data

@dataclass
class PhotoInfo:
    photo_id: str
    number: int
    filename: str
    width: int
    height: int
    taken_at: datetime | None
    faces: list[Box]                              # normalised 0..1 (x, y, w, h)
    sharpness: float
    signature: np.ndarray
    hero: float = 0.0

    @property
    def aspect(self) -> float:
        return self.width / max(1, self.height)


@dataclass
class Placement:
    photo: PhotoInfo
    box: Box                                      # pixels on the spread canvas
    border: int


@dataclass
class Panel:
    box: Box
    colour: tuple[int, int, int]
    opacity: int = 255
    name: str = "Panel"


@dataclass
class SpreadPlan:
    index: int
    recipe: str
    background: tuple[int, int, int]
    ghost: PhotoInfo | None = None
    ghost_page: int | None = None                 # 0 left, 1 right, None whole spread
    panels: list[Panel] = field(default_factory=list)
    placements: list[Placement] = field(default_factory=list)


@dataclass(frozen=True)
class AlbumSpec:
    height_in: float
    width_in: float
    spreads: int
    dpi: int = 300
    seed: int = 0

    @property
    def size_px(self) -> tuple[int, int]:
        return round(self.width_in * self.dpi), round(self.height_in * self.dpi)


# Background families seen in the studio samples: creams, butter yellow, sage, stone, blush, white.
THEMES = [
    {"name": "Butter", "bg": [(246, 238, 204), (243, 233, 191), (250, 245, 226)], "panel": (229, 214, 150)},
    {"name": "Sage", "bg": [(222, 226, 219), (205, 211, 203), (236, 238, 233)], "panel": (190, 198, 189)},
    {"name": "Ivory", "bg": [(250, 247, 240), (244, 239, 228), (255, 255, 255)], "panel": (226, 216, 196)},
    {"name": "Blush", "bg": [(248, 238, 236), (242, 228, 226), (252, 246, 245)], "panel": (226, 200, 196)},
]


# --------------------------------------------------------------------------- analysis

def analyse_photo(
    photo_id: str, number: int, filename: str, image: Image.Image, exif_source: Image.Image | None,
    face_finder: Callable[[Image.Image], list[tuple[int, int, int, int]]] | None,
) -> PhotoInfo:
    """``image`` is the oriented photo (any size); a small proxy is analysed."""
    width, height = image.size
    proxy = image.convert("RGB")
    proxy.thumbnail((ANALYSIS_SIDE, ANALYSIS_SIDE), Image.Resampling.BILINEAR)
    faces: list[Box] = []
    if face_finder is not None:
        try:
            for x, y, w, h in face_finder(proxy):
                faces.append((x / proxy.width, y / proxy.height, w / proxy.width, h / proxy.height))
        except Exception:  # face model unavailable: layouts simply fall back to centre-weighted crops
            logger.debug("Face detection failed during album analysis.", exc_info=True)
    faces = _dedupe_faces(faces)
    gray = np.asarray(proxy.convert("L"), dtype=np.float32)
    laplacian = gray[1:-1, 1:-1] * 4 - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    sharpness = float(laplacian.var())
    small = np.asarray(proxy.resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    signature = np.concatenate([small.mean(axis=(0, 1)), small.reshape(-1, 3)[::8].reshape(-1)])
    return PhotoInfo(photo_id, number, filename, width, height, _taken_at(exif_source or image), faces,
                     sharpness, signature)


def _dedupe_faces(faces: list[Box]) -> list[Box]:
    kept: list[Box] = []
    for face in sorted(faces, key=lambda f: -f[2] * f[3]):
        if all(_iou(face, other) < 0.4 for other in kept):
            kept.append(face)
    return kept


def _iou(a: Box, b: Box) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def _taken_at(image: Image.Image) -> datetime | None:
    try:
        exif = image.getexif()
        raw = exif.get_ifd(0x8769).get(36867) or exif.get(306)
        return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S") if raw else None
    except Exception:
        return None


def score_heroes(photos: Sequence[PhotoInfo]) -> None:
    """Hero score 0..1: big, well-framed faces in a sharp photo (relative to this album)."""
    if not photos:
        return
    sharp = np.array([p.sharpness for p in photos])
    low, high = np.percentile(sharp, 10), np.percentile(sharp, 90)
    for photo in photos:
        biggest = max((f[2] * f[3] for f in photo.faces), default=0.0)
        face_term = min(1.0, math.sqrt(biggest) / 0.22)          # a face ~22% of the frame side = full marks
        crowd = len(photo.faces)
        people_term = 1.0 if 1 <= crowd <= 4 else 0.6 if crowd > 4 else 0.25
        sharp_term = float(np.clip((photo.sharpness - low) / max(1e-6, high - low), 0, 1))
        photo.hero = 0.5 * face_term + 0.2 * people_term + 0.3 * sharp_term


# --------------------------------------------------------------------------- story

def order_photos(photos: Sequence[PhotoInfo]) -> list[PhotoInfo]:
    dated = sum(1 for p in photos if p.taken_at)
    if dated >= max(3, len(photos) * 0.6):
        fallback = datetime.max
        return sorted(photos, key=lambda p: (p.taken_at or fallback, p.number))
    return sorted(photos, key=lambda p: p.number)


def scene_breaks(photos: Sequence[PhotoInfo]) -> list[float]:
    """Strength 0..1 of a scene change *before* each photo (index 0 is always 1)."""
    breaks = [1.0]
    for previous, current in zip(photos, photos[1:]):
        colour = float(np.abs(previous.signature - current.signature).mean()) / 0.25
        gap = 0.0
        if previous.taken_at and current.taken_at:
            minutes = abs((current.taken_at - previous.taken_at).total_seconds()) / 60
            gap = min(1.0, minutes / 20)
        breaks.append(float(min(1.0, max(gap, colour))))
    return breaks


def split_into_spreads(photos: Sequence[PhotoInfo], spreads: int) -> list[list[PhotoInfo]]:
    """Contiguous chunks, one per spread: even sizes, cuts at scene changes (dynamic programming)."""
    count = len(photos)
    spreads = max(1, min(spreads, count))
    target = count / spreads
    breaks = scene_breaks(photos)
    inf = float("inf")
    # cost[k][i] = best cost to place the first i photos on k spreads
    cost = [[inf] * (count + 1) for _ in range(spreads + 1)]
    back = [[0] * (count + 1) for _ in range(spreads + 1)]
    cost[0][0] = 0.0
    max_chunk = max(2 * MAX_PHOTOS_PER_PAGE, math.ceil(target * 2.2))
    for k in range(1, spreads + 1):
        for i in range(k, count - (spreads - k) + 1):
            best, arg = inf, 0
            for j in range(max(k - 1, i - max_chunk), i):
                if cost[k - 1][j] == inf:
                    continue
                size = i - j
                size_cost = ((size - target) / max(1.0, target)) ** 2
                cut_cost = 0.0 if i == count else (1.0 - breaks[i]) * 0.6
                total = cost[k - 1][j] + size_cost + cut_cost
                if total < best:
                    best, arg = total, j
            cost[k][i], back[k][i] = best, arg
    chunks, i = [], count
    for k in range(spreads, 0, -1):
        j = back[k][i]
        chunks.append(list(photos[j:i]))
        i = j
    return list(reversed(chunks))


# --------------------------------------------------------------------------- page layouts

def _compositions(n: int, parts: int) -> Iterable[tuple[int, ...]]:
    for cuts in combinations(range(1, n), parts - 1):
        bounds = (0, *cuts, n)
        yield tuple(bounds[i + 1] - bounds[i] for i in range(parts))


def justified_layout(aspects: Sequence[float], box: Box, gap: float) -> tuple[list[Box], float]:
    """Fill ``box`` with the photos in order, as rows or as columns, with the least cropping.

    Returns the cell boxes and a cost (mean |log aspect change| plus penalties for tiny cells).
    """
    x, y, width, height = box
    n = len(aspects)
    best: tuple[float, list[Box]] = (float("inf"), [])
    for by_rows in (True, False):
        for lines in range(1, min(n, 4) + 1):
            for sizes in _compositions(n, lines):
                if max(sizes) - min(sizes) > 2:
                    continue
                cells: list[Box] = []
                start = 0
                line_weights = []
                for size in sizes:
                    group = aspects[start:start + size]
                    # natural thickness of the line when it spans the full width (rows) or height (columns)
                    span = (width if by_rows else height) - gap * (size - 1)
                    weight = span / sum(group) if by_rows else span / sum(1 / a for a in group)
                    line_weights.append(weight)
                    start += size
                total_span = (height if by_rows else width) - gap * (lines - 1)
                scale = total_span / sum(line_weights)
                offset = 0.0
                start = 0
                distortion = 0.0
                smallest = float("inf")
                for size, weight in zip(sizes, line_weights):
                    thickness = weight * scale
                    group = aspects[start:start + size]
                    shares = group if by_rows else [1 / a for a in group]
                    span = (width if by_rows else height) - gap * (size - 1)
                    along = 0.0
                    for aspect, share in zip(group, shares):
                        length = span * share / sum(shares)
                        if by_rows:
                            cell = (x + along, y + offset, length, thickness)
                        else:
                            cell = (x + offset, y + along, thickness, length)
                        cells.append(cell)
                        distortion += abs(math.log((cell[2] / cell[3]) / aspect))
                        smallest = min(smallest, min(cell[2], cell[3]))
                        along += length + gap
                    offset += thickness + gap
                    start += size
                cost = distortion / n
                if smallest < 0.16 * min(width, height):
                    cost += 1.5
                if lines == 1 and n >= 4:
                    cost += 0.6                       # long thin strips look poor in albums
                if cost < best[0]:
                    best = (cost, cells)
    return best[1], best[0]


# --------------------------------------------------------------------------- spread planning

def plan_album(photos: Sequence[PhotoInfo], spec: AlbumSpec) -> list[SpreadPlan]:
    rng = random.Random(spec.seed)
    score_heroes(photos)
    ordered = order_photos(photos)
    chunks = split_into_spreads(ordered, spec.spreads)
    theme = THEMES[rng.randrange(len(THEMES))]
    width, height = spec.size_px
    plans = []
    hero_side = rng.randrange(2)
    for index, chunk in enumerate(chunks):
        plan = _plan_spread(index, chunk, theme, width, height, rng, hero_side)
        hero_side ^= 1
        plans.append(plan)
    return plans


def _plan_spread(index: int, chunk: list[PhotoInfo], theme: dict, width: int, height: int,
                 rng: random.Random, hero_side: int) -> SpreadPlan:
    page_w = width / 2
    border = max(4, round(height * 0.0045))
    gap = height * 0.012
    margin = height * 0.045
    fold_margin = height * 0.04
    background = theme["bg"][index % len(theme["bg"])]
    plan = SpreadPlan(index=index, recipe="grid", background=background)

    def page_box(page: int, inset: float = 1.0) -> Box:
        left = page * page_w + (margin if page == 0 else fold_margin) * inset
        right = (page + 1) * page_w - (fold_margin if page == 0 else margin) * inset
        return (left, margin * inset, right - left, height - 2 * margin * inset)

    def place(page_photos: list[PhotoInfo], box: Box, framed: bool = True) -> float:
        if len(page_photos) == 1:
            # A single photo keeps its own shape when the box is very different (no portrait->landscape crops).
            x, y, w, h = box
            aspect = page_photos[0].aspect
            if abs(math.log(aspect / (w / h))) > 0.25:
                if aspect < w / h:
                    new_w = h * aspect
                    box = (x + (w - new_w) / 2, y, new_w, h)
                else:
                    new_h = w / aspect
                    box = (x, y + (h - new_h) / 2, w, new_h)
        cells, cost = justified_layout([p.aspect for p in page_photos], box, gap)
        for photo, cell in zip(page_photos, cells):
            plan.placements.append(Placement(photo, cell, border if framed else 0))
        return cost

    count = len(chunk)
    if count == 1:
        photo = chunk[0]
        center_faces = any(abs((f[0] + f[2] / 2) - 0.5) < 0.12 for f in photo.faces)
        if photo.aspect >= 1.6 and not center_faces:
            plan.recipe = "panorama"
            plan.placements.append(Placement(photo, (0, 0, width, height), 0))
        else:
            plan.recipe = "hero-ghost"
            if abs(math.log(photo.aspect / (page_w / height))) <= 0.3:
                plan.placements.append(Placement(photo, (hero_side * page_w, 0, page_w, height), 0))
            else:
                place([photo], page_box(hero_side, 0.8))
            plan.ghost, plan.ghost_page = photo, 1 - hero_side
        return plan

    with_people = [p for p in chunk if p.faces]
    best = max(with_people or chunk, key=lambda p: p.hero)
    hero_page = (count >= 3 and bool(best.faces) and best.hero >= 0.5
                 and count - 1 <= MAX_PHOTOS_PER_PAGE)
    if count == 2:
        plan.recipe = "two-heroes"
        for page, photo in enumerate(chunk):
            place([photo], page_box(page, 1.2))
        plan.panels.extend(_offset_panels(page_box(0, 1.2), theme, rng, height))
        plan.panels.extend(_offset_panels(page_box(1, 1.2), theme, rng, height))
        return plan

    if hero_page:
        plan.recipe = "hero-page"
        rest = [p for p in chunk if p is not best]
        page_aspect = page_w / height
        fits_page = abs(math.log(best.aspect / page_aspect)) <= 0.3
        hero_full_bleed = fits_page and rng.random() < 0.7
        if hero_full_bleed:
            plan.placements.append(Placement(best, (hero_side * page_w, 0, page_w, height), 0))
        else:
            place([best], page_box(hero_side, 0.8))
        grid_box = page_box(1 - hero_side)
        if rng.random() < 0.45:
            plan.panels.append(_frame_panel(grid_box, height))
        else:
            plan.panels.extend(_offset_panels(grid_box, theme, rng, height))
        if not fits_page:                       # framed hero narrower than its page: faded backdrop behind it
            plan.ghost, plan.ghost_page = best, hero_side
        elif rng.random() < 0.4:
            plan.ghost, plan.ghost_page = best, 1 - hero_side
        place(rest, grid_box)
        return plan

    # Two grids: split the chunk where the two pages lay out best.
    plan.recipe = "two-grids"
    best_split, best_cost = 1, float("inf")
    for split in range(1, count):
        left, right = chunk[:split], chunk[split:]
        if max(len(left), len(right)) > MAX_PHOTOS_PER_PAGE:
            continue
        _, left_cost = justified_layout([p.aspect for p in left], page_box(0), gap)
        _, right_cost = justified_layout([p.aspect for p in right], page_box(1), gap)
        cost = left_cost + right_cost + 0.08 * abs(len(left) - len(right))
        page_aspect = page_box(0)[2] / page_box(0)[3]
        for side in (left, right):
            if len(side) == 1:                      # a lone photo of the wrong shape leaves the page half empty
                aspect = side[0].aspect
                cost += 2.0 * (1 - min(aspect, page_aspect) / max(aspect, page_aspect))
        if cost < best_cost:
            best_split, best_cost = split, cost
    style = rng.random()
    for page, page_photos in enumerate((chunk[:best_split], chunk[best_split:])):
        box = page_box(page)
        if style < 0.35:
            plan.panels.append(_frame_panel(box, height))
        elif style < 0.75:
            plan.panels.extend(_offset_panels(box, theme, rng, height))
        place(page_photos, box)
        if len(page_photos) == 1 and plan.ghost is None:
            aspect, page_aspect = page_photos[0].aspect, box[2] / box[3]
            if abs(math.log(aspect / page_aspect)) > 0.25:   # fill the rest of the page with a faded backdrop
                plan.ghost, plan.ghost_page = page_photos[0], page
    if style >= 0.75 and plan.ghost is None:
        plan.ghost, plan.ghost_page = max(chunk, key=lambda p: p.hero), None
    return plan


def _frame_panel(box: Box, height: int) -> Panel:
    pad = height * 0.018
    x, y, w, h = box
    return Panel((x - pad, y - pad, w + 2 * pad, h + 2 * pad), (255, 255, 255), 255, "White frame")


def _offset_panels(box: Box, theme: dict, rng: random.Random, height: int) -> list[Panel]:
    x, y, w, h = box
    shift = height * 0.035
    colour = theme["panel"]
    panels = []
    if rng.random() < 0.5:
        panels.append(Panel((x - shift, y - shift, w * 0.55, h * 0.45), colour, 255, "Accent panel"))
        panels.append(Panel((x + w * 0.45 + shift, y + h * 0.55 + shift, w * 0.55, h * 0.45 - shift),
                            colour, 255, "Accent panel"))
    else:
        panels.append(Panel((x - shift * 0.6, y + h * 0.18, w + shift * 1.2, h * 0.64), colour, 140, "Band"))
    return panels


# --------------------------------------------------------------------------- crops

def face_aware_crop(photo: PhotoInfo, image_size: tuple[int, int], target_aspect: float,
                    fold_x: float | None = None) -> tuple[int, int, int, int]:
    """Largest crop of the given aspect that keeps faces in frame (and, if possible, off the fold).

    ``fold_x`` is where the album's centre fold falls inside the target cell, as a 0..1 fraction.
    """
    width, height = image_size
    if width / height > target_aspect:
        crop_w, crop_h = height * target_aspect, float(height)
    else:
        crop_w, crop_h = float(width), width / target_aspect
    faces = [(fx * width, fy * height, fw * width, fh * height) for fx, fy, fw, fh in photo.faces]

    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    if crop_w < width - 1:                                    # slide horizontally
        if faces:
            left = min(f[0] - f[2] * 0.4 for f in faces)
            right = max(f[0] + f[2] * 1.4 for f in faces)
            centre = (left + right) / 2 if right - left <= crop_w else max(faces, key=lambda f: f[2] * f[3])[0]
        else:
            centre = width / 2
        x0 = clamp(centre - crop_w / 2, 0, width - crop_w)
        if fold_x is not None and faces:
            x0 = _avoid_fold(x0, crop_w, width, faces, fold_x)
        y0 = 0.0
    else:                                                     # slide vertically, favour the top (heads)
        if faces:
            top = min(f[1] - f[3] * 0.6 for f in faces)
            bottom = max(f[1] + f[3] * 1.6 for f in faces)
            centre_y = (top + bottom) / 2 if bottom - top <= crop_h else top + crop_h / 2
        else:
            centre_y = height * 0.42
        y0 = clamp(centre_y - crop_h / 2, 0, height - crop_h)
        x0 = 0.0
        if fold_x is not None and faces:
            x0 = 0.0
    return int(round(x0)), int(round(y0)), int(round(x0 + crop_w)), int(round(y0 + crop_h))


def _avoid_fold(x0: float, crop_w: float, width: int, faces, fold_x: float) -> float:
    band = crop_w * 0.05
    candidates = [x0] + [x0 + d * crop_w * 0.04 for d in range(-6, 7) if d]
    best, best_hit = x0, float("inf")
    for candidate in candidates:
        candidate = max(0.0, min(width - crop_w, candidate))
        fold = candidate + fold_x * crop_w
        hit = sum(max(0.0, min(f[0] + f[2], fold + band) - max(f[0], fold - band)) for f in faces)
        if hit < best_hit - 1e-6 or (abs(hit - best_hit) < 1e-6 and abs(candidate - x0) < abs(best - x0)):
            best, best_hit = candidate, hit
    return best


# --------------------------------------------------------------------------- rendering

@dataclass
class RenderedSpread:
    composite: Image.Image
    document: psd.Document


def render_spread(plan: SpreadPlan, spec: AlbumSpec, load: Callable[[PhotoInfo], Image.Image],
                  with_layers: bool = True) -> RenderedSpread:
    """``load`` returns the corrected, oriented full-size photo (studio look applied, no logo)."""
    width, height = spec.size_px
    page_w = width // 2
    canvas = Image.new("RGB", (width, height), plan.background)
    document = psd.Document(width, height, spec.dpi)
    cache: dict[str, Image.Image] = {}

    def photo(info: PhotoInfo) -> Image.Image:
        if info.photo_id not in cache:
            cache[info.photo_id] = load(info)
        return cache[info.photo_id]

    # Background (+ faded ghost photo)
    background = Image.new("RGB", (width, height), plan.background)
    if plan.ghost is not None:
        if plan.ghost_page is None:
            area = (0, 0, width, height)
        else:
            area = (plan.ghost_page * page_w, 0, page_w, height)
        source = photo(plan.ghost)
        crop = face_aware_crop(plan.ghost, source.size, area[2] / area[3])
        ghost = source.crop(crop).resize((area[2], area[3]), Image.Resampling.LANCZOS)
        ghost = ghost.filter(ImageFilter.GaussianBlur(max(2, height // 400)))
        tinted = Image.blend(Image.new("RGB", ghost.size, plan.background), ghost, 0.2)
        background.paste(tinted, (area[0], area[1]))
    canvas.paste(background)
    if with_layers:
        document.layers.append(psd.layer_from(background, "Background", 0, 0))

    for number, panel in enumerate(plan.panels, start=1):
        x, y, w, h = (int(round(v)) for v in panel.box)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        block = Image.new("RGBA", (x1 - x0, y1 - y0), panel.colour + (panel.opacity,))
        canvas.paste(block.convert("RGB"), (x0, y0), block)
        if with_layers:
            document.layers.append(psd.Layer(f"{panel.name} {number}", block, x0, y0))

    for slot, placement in enumerate(plan.placements, start=1):
        x, y, w, h = placement.box
        left, top = int(round(x)), int(round(y))
        cell_w, cell_h = max(1, int(round(x + w)) - left), max(1, int(round(y + h)) - top)
        fold = None
        if left < page_w < left + cell_w:
            fold = (page_w - left) / cell_w
        source = photo(placement.photo)
        crop = face_aware_crop(placement.photo, source.size, cell_w / cell_h, fold)
        fitted = source.crop(crop).resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        if placement.border:
            b = placement.border
            frame = Image.new("RGB", (cell_w + 2 * b, cell_h + 2 * b), (255, 255, 255))
            canvas.paste(frame, (left - b, top - b))
            if with_layers:
                document.layers.append(psd.layer_from(frame, f"Frame {slot:02d}", left - b, top - b))
        canvas.paste(fitted, (left, top))
        if with_layers:
            name = f"Photo {placement.photo.number:05d} - {placement.photo.filename}"
            document.layers.append(psd.layer_from(fitted, name[:120], left, top))

    safe = round(spec.dpi * 0.25)
    document.guides = [psd.Guide(page_w, True), psd.Guide(safe, True), psd.Guide(width - safe, True),
                       psd.Guide(safe, False), psd.Guide(height - safe, False)]
    return RenderedSpread(canvas, document)


def preview(image: Image.Image, width: int = PREVIEW_WIDTH) -> Image.Image:
    if image.width <= width:
        return image.copy()
    ratio = width / image.width
    return image.resize((width, round(image.height * ratio)), Image.Resampling.LANCZOS)


def jpeg_bytes(image: Image.Image, quality: int = 92, dpi: int | None = None) -> bytes:
    stream = io.BytesIO()
    options = {"quality": quality, "optimize": True, "subsampling": 0}
    if dpi:
        options["dpi"] = (dpi, dpi)
    image.save(stream, "JPEG", **options)
    return stream.getvalue()


def seed_for(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def summary(plans: Sequence[SpreadPlan]) -> list[dict]:
    return [{"index": p.index, "recipe": p.recipe, "photos": [pl.photo.number for pl in p.placements]}
            for p in plans]
