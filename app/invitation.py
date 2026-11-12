"""Wedding invitation designer: five distinct designs from the couple's details, and an animated MP4.

Everything is drawn here (no external AI service): procedural ornaments (mandalas, temple arches, florals,
marigold torans, leaf washes), bundled fonts, and an optional couple photo that is face-aware cropped into an
arch, a circle or a full-bleed header. The same scene renders as a still poster and, frame by frame, as a
vertical 1080x1920 video (staggered reveals, slow zoom toward the couple's faces, drifting particles) piped
to ffmpeg.
"""
from __future__ import annotations

import math
import random
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

W, H = 1080, 1920
FPS = 24
DURATION = 13.0
FONT_DIR = Path(__file__).parent / "assets" / "fonts"

KINDS = {
    "WEDDING": "Wedding Invitation",
    "ENGAGEMENT": "Engagement Ceremony",
    "RECEPTION": "Wedding Reception",
    "SAVE_THE_DATE": "Save the Date",
}
DEFAULT_MESSAGE = "Together with their families, request the pleasure of your company to celebrate their union"

Box = tuple[float, float, float, float]
FaceFinder = Callable[[Image.Image], list[tuple[int, int, int, int]]]


class InvitationError(ValueError):
    """Bad invitation details or photo."""


# --------------------------------------------------------------------------- details

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _clean(value, limit: int) -> str:
    text = _CONTROL.sub(" ", unicodedata.normalize("NFC", str(value or ""))).strip()
    return re.sub(r"\s{2,}", " ", text)[:limit]


def normalise_details(raw: dict) -> dict:
    """Validated, trimmed invitation details (what the customer typed)."""
    if not isinstance(raw, dict):
        raise InvitationError("Invitation details are missing.")
    kind = str(raw.get("kind") or "WEDDING").upper()
    if kind not in KINDS:
        kind = "WEDDING"
    details = {
        "kind": kind,
        "bride": _clean(raw.get("bride"), 40),
        "groom": _clean(raw.get("groom"), 40),
        "bride_family": _clean(raw.get("bride_family"), 90),
        "groom_family": _clean(raw.get("groom_family"), 90),
        "invocation": _clean(raw.get("invocation"), 60),
        "message": _clean(raw.get("message"), 160) or DEFAULT_MESSAGE,
        "hosts": _clean(raw.get("hosts"), 120),
        "rsvp": _clean(raw.get("rsvp"), 60),
        "events": [],
    }
    if not details["bride"] or not details["groom"]:
        raise InvitationError("Enter the bride's and groom's names.")
    for item in (raw.get("events") or [])[:3]:
        if not isinstance(item, dict):
            continue
        line = {
            "title": _clean(item.get("title"), 40),
            "date": _clean(item.get("date"), 40),
            "time": _clean(item.get("time"), 30),
            "venue": _clean(item.get("venue"), 70),
            "address": _clean(item.get("address"), 110),
        }
        if line["date"] or line["venue"]:
            details["events"].append(line)
    if not details["events"]:
        raise InvitationError("Add the date and venue of at least one function.")
    return details


# --------------------------------------------------------------------------- fonts

FONT_FILES = {
    "script": "texgyrechorus-mediumitalic.otf",
    "serif": "texgyrepagella-regular.otf",
    "serif_bold": "texgyrepagella-bold.otf",
    "serif_italic": "texgyrepagella-italic.otf",
    "lora": "Lora-Variable.ttf",
    "lora_italic": "Lora-Italic-Variable.ttf",
    "sans": "Poppins-Regular.ttf",
    "sans_medium": "Poppins-Medium.ttf",
    "sans_light": "Poppins-Light.ttf",
    "indic": "FreeSerif.ttf",
    "indic_bold": "FreeSerifBold.ttf",
}


@lru_cache(maxsize=256)
def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / FONT_FILES[name]), max(8, int(size)))


def _needs_indic(text: str) -> bool:
    return any(ord(ch) > 0x24F and not 0x2000 <= ord(ch) <= 0x206F for ch in text)


def _font_for(text: str, name: str, size: int) -> ImageFont.FreeTypeFont:
    if _needs_indic(text):              # names typed in Kannada, Tamil, Hindi...: a font that has the glyphs
        return font("indic_bold" if name in {"serif_bold", "sans_medium"} else "indic", size * 0.9)
    return font(name, size)


# --------------------------------------------------------------------------- themes

@dataclass(frozen=True)
class Theme:
    key: str
    label: str
    bg_top: tuple
    bg_bottom: tuple
    ink: tuple           # body text
    accent: tuple        # headings, ornaments
    soft: tuple          # secondary ornament colour
    name_ink: tuple
    names_font: str = "script"
    head_font: str = "serif"
    body_font: str = "lora"
    ornament: str = "mandala"        # mandala | floral | temple | toran | leaves | minimal | stars
    particles: str = "gold"          # gold | petals | bokeh | none
    dark: bool = True


THEMES: dict[str, Theme] = {t.key: t for t in [
    Theme("royal_maroon", "Royal maroon & gold", (92, 14, 28), (48, 6, 14), (246, 226, 196), (222, 178, 92), (166, 112, 52),
          (246, 214, 140), ornament="mandala", particles="gold"),
    Theme("floral_blush", "Blush florals", (252, 238, 234), (246, 222, 216), (96, 52, 60), (184, 92, 108), (122, 150, 108),
          (150, 56, 78), body_font="lora", ornament="floral", particles="petals", dark=False),
    Theme("temple_ivory", "Temple ivory", (252, 246, 232), (240, 228, 204), (92, 36, 24), (176, 120, 40), (140, 36, 30),
          (128, 30, 26), head_font="serif_bold", ornament="temple", particles="gold", dark=False),
    Theme("emerald_night", "Emerald night", (14, 60, 50), (6, 30, 26), (232, 236, 220), (216, 182, 104), (120, 170, 140),
          (240, 210, 140), ornament="stars", particles="bokeh"),
    Theme("minimal_ivory", "Minimal ivory", (250, 248, 243), (244, 240, 232), (58, 56, 54), (176, 146, 96), (200, 190, 172),
          (60, 56, 52), names_font="lora_italic", head_font="sans_light", body_font="lora", ornament="minimal",
          particles="none", dark=False),
    Theme("marigold", "Marigold festive", (255, 236, 186), (250, 212, 140), (110, 30, 20), (214, 98, 20), (60, 120, 40),
          (150, 28, 22), head_font="serif_bold", ornament="toran", particles="petals", dark=False),
    Theme("royal_blue", "Royal blue & gold", (22, 36, 92), (10, 16, 48), (232, 230, 246), (226, 190, 110), (110, 140, 220),
          (246, 220, 150), ornament="mandala", particles="gold"),
    Theme("sage_leaves", "Sage leaves", (236, 240, 230), (220, 228, 214), (58, 74, 60), (124, 150, 112), (176, 160, 120),
          (70, 92, 70), body_font="lora", ornament="leaves", particles="none", dark=False),
]}

LAYOUTS_WITH_PHOTO = ["arch", "round", "bleed"]


@dataclass(frozen=True)
class DesignChoice:
    theme: str
    layout: str
    seed: int

    def as_dict(self) -> dict:
        return {"theme": self.theme, "layout": self.layout, "seed": self.seed, "label": THEMES[self.theme].label}


def choose_designs(seed: int, has_photo: bool, count: int = 5, avoid: set[str] | None = None) -> list[DesignChoice]:
    """``count`` designs with different themes (and varied layouts); ``avoid`` steers regenerations to new themes."""
    rng = random.Random(seed)
    keys = list(THEMES)
    rng.shuffle(keys)
    fresh = [k for k in keys if k not in (avoid or set())]
    ordered = (fresh + [k for k in keys if k not in fresh])[:count]
    layouts = LAYOUTS_WITH_PHOTO[:] if has_photo else ["medallion"]
    rng.shuffle(layouts)
    choices = []
    for index, key in enumerate(ordered):
        layout = layouts[index % len(layouts)]
        if has_photo and key == "minimal_ivory" and layout == "bleed":
            layout = "arch"
        choices.append(DesignChoice(key, layout, rng.randrange(1, 2_000_000_000)))
    return choices


# --------------------------------------------------------------------------- scene model

@dataclass
class Layer:
    name: str
    image: Image.Image          # RGBA
    x: int
    y: int
    start: float = 0.0          # seconds when the reveal starts
    length: float = 0.8
    motion: str = "fade"        # fade | rise | grow | zoom
    rise: int = 28


@dataclass
class PhotoLayer:
    source: Image.Image         # RGB, larger than the box for the slow zoom
    mask: Image.Image           # L, box-sized
    box: tuple[int, int, int, int]
    focus: tuple[float, float]  # zoom target inside source (0..1)
    start: float = 0.8
    length: float = 1.4


@dataclass
class Scene:
    theme: Theme
    background: Image.Image     # RGB
    layers: list[Layer] = field(default_factory=list)          # drawn in order
    photo: PhotoLayer | None = None
    photo_frame: list[Layer] = field(default_factory=list)     # drawn above the photo
    shimmer: Box | None = None  # names area for the light sweep
    seed: int = 0


# --------------------------------------------------------------------------- drawing helpers

SS = 3   # supersampling for ornaments


def _rgba(colour: tuple, alpha: int = 255) -> tuple:
    return tuple(colour[:3]) + (alpha,)


def _canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", (width * SS, height * SS), (0, 0, 0, 0))
    return image, ImageDraw.Draw(image)


def _down(image: Image.Image) -> Image.Image:
    return image.resize((image.width // SS, image.height // SS), Image.Resampling.LANCZOS)


def _gradient(top: tuple, bottom: tuple, rng: random.Random, dark: bool) -> Image.Image:
    t = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    colours = np.array(top, np.float32) * (1 - t) + np.array(bottom, np.float32) * t
    image = np.repeat(colours, W, axis=1)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vignette = 1 - 0.28 * (((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H * 0.45) / (H * 0.75)) ** 2)
    image *= np.clip(vignette, 0.72 if dark else 0.9, 1)[..., None]
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 2.2 if dark else 1.6, (H, W, 1)).astype(np.float32)
    return Image.fromarray(np.clip(image + noise, 0, 255).astype(np.uint8), "RGB")


def _mandala(radius: int, colour: tuple, soft: tuple, rng: random.Random, rings: int = 4) -> Image.Image:
    size = radius * 2 + 4
    image, draw = _canvas(size, size)
    c = size * SS / 2
    r = radius * SS
    width = max(SS, int(r * 0.012))
    petals = rng.choice([12, 16, 20, 24])
    for ring in range(rings):
        outer = r * (1 - ring * 0.22)
        inner = outer * 0.72
        count = petals if ring % 2 == 0 else petals // 2 * 2 + 4
        tone = colour if ring % 2 == 0 else soft
        for k in range(count):
            a = 2 * math.pi * k / count + (math.pi / count if ring % 2 else 0)
            mid = (inner + outer) / 2
            half = (outer - inner) / 2
            spread = math.pi / count * 0.9
            pts = [(c + inner * math.cos(a), c + inner * math.sin(a)),
                   (c + mid * math.cos(a - spread), c + mid * math.sin(a - spread)),
                   (c + (outer + half * 0.2) * math.cos(a), c + (outer + half * 0.2) * math.sin(a)),
                   (c + mid * math.cos(a + spread), c + mid * math.sin(a + spread))]
            draw.polygon(pts, outline=_rgba(tone, 235), width=width)
            dot = width * 1.6
            px, py = c + (outer + half * 0.55) * math.cos(a), c + (outer + half * 0.55) * math.sin(a)
            draw.ellipse((px - dot, py - dot, px + dot, py + dot), fill=_rgba(colour, 230))
        draw.ellipse((c - inner, c - inner, c + inner, c + inner), outline=_rgba(tone, 160), width=max(1, width // 2))
    core = r * 0.14
    draw.ellipse((c - core, c - core, c + core, c + core), outline=_rgba(colour, 255), width=width * 2)
    return _down(image)


def _corner_flourish(size: int, colour: tuple, soft: tuple, rng: random.Random) -> Image.Image:
    image, draw = _canvas(size, size)
    s = size * SS
    width = max(SS, s // 140)
    for k in range(3):
        rad = s * (0.95 - k * 0.22)
        draw.arc((-rad, -rad, rad, rad), 0, 90, fill=_rgba(colour if k != 1 else soft, 230), width=width)
    for k in range(9):
        a = math.radians(5 + k * 10)
        rad = s * 0.62
        x, y = rad * math.cos(a), rad * math.sin(a)
        dot = width * (2.4 if k % 2 == 0 else 1.4)
        draw.ellipse((x - dot, y - dot, x + dot, y + dot), fill=_rgba(colour, 240))
    quarter = _mandala(int(size * 0.42), colour, soft, rng, rings=3)
    image = _down(image)
    image.alpha_composite(quarter, (-quarter.width // 2, -quarter.height // 2))
    return image


def _border(colour: tuple, inset: int, double: bool = True) -> Image.Image:
    image, draw = _canvas(W, H)
    width = max(SS, 2 * SS)
    for k in range(2 if double else 1):
        d = (inset + k * 14) * SS
        draw.rectangle((d, d, W * SS - d, H * SS - d), outline=_rgba(colour, 210 if k == 0 else 140), width=width)
    for x, y in [(W / 2, inset), (W / 2, H - inset), (inset, H / 2), (W - inset, H / 2)]:
        r = 12 * SS
        cx, cy = x * SS, y * SS
        draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=_rgba(colour, 230))
    return _down(image)


def _arch_points(box: Box, steps: int = 80) -> list[tuple[float, float]]:
    """Mughal-style pointed arch over a rectangle (clockwise polygon)."""
    x, y, w, h = box
    spring = y + w * 0.62                # where the arch meets the straight sides
    pts = [(x, y + h), (x, spring)]
    for k in range(steps + 1):          # left curve up to the apex
        t = k / steps
        px = x + w / 2 * t
        py = spring - (spring - y) * math.sin(t * math.pi / 2) ** 0.85
        pts.append((px, py))
    for k in range(steps + 1):
        t = 1 - k / steps
        px = x + w - w / 2 * t
        py = spring - (spring - y) * math.sin(t * math.pi / 2) ** 0.85
        pts.append((px, py))
    pts += [(x + w, spring), (x + w, y + h)]
    return pts


def _arch_mask(size: tuple[int, int]) -> Image.Image:
    w, h = size
    image = Image.new("L", (w * SS, h * SS), 0)
    ImageDraw.Draw(image).polygon(_arch_points((0, 0, w * SS, h * SS)), fill=255)
    return image.resize(size, Image.Resampling.LANCZOS)


def _arch_frame(box: Box, colour: tuple, soft: tuple, pad: int = 18) -> Image.Image:
    x, y, w, h = box
    image, draw = _canvas(W, H)
    for k, (offset, tone, alpha) in enumerate([(pad, colour, 240), (pad + 14, soft, 170)]):
        pts = _arch_points((x - offset, y - offset, w + 2 * offset, h + 2 * offset))
        draw.line([(px * SS, py * SS) for px, py in pts] + [(pts[0][0] * SS, pts[0][1] * SS)],
                  fill=_rgba(tone, alpha), width=(4 if k == 0 else 2) * SS, joint="curve")
    apex = (x + w / 2) * SS, (y - pad - 34) * SS
    r = 14 * SS
    draw.polygon([(apex[0], apex[1] - r * 1.8), (apex[0] + r, apex[1]), (apex[0], apex[1] + r), (apex[0] - r, apex[1])],
                 fill=_rgba(colour, 240))
    return _down(image)


def _flower(radius: int, petal: tuple, centre: tuple, rng: random.Random) -> Image.Image:
    size = radius * 2 + 6
    image, draw = _canvas(size, size)
    c = size * SS / 2
    count = rng.choice([5, 6, 8])
    for layer, scale in enumerate([1.0, 0.68]):
        tone = petal if layer == 0 else tuple(min(255, int(v * 1.08 + 12)) for v in petal)
        for k in range(count):
            a = 2 * math.pi * (k + layer * 0.5) / count + rng.random() * 0.2
            length = radius * SS * scale
            px, py = c + length * 0.52 * math.cos(a), c + length * 0.52 * math.sin(a)
            petal_img = Image.new("RGBA", (int(length * 1.1), int(length * 0.72)), (0, 0, 0, 0))
            ImageDraw.Draw(petal_img).ellipse((0, 0, petal_img.width - 1, petal_img.height - 1), fill=_rgba(tone, 235))
            rotated = petal_img.rotate(-math.degrees(a), expand=True, resample=Image.Resampling.BICUBIC)
            image.alpha_composite(rotated, (int(px - rotated.width / 2), int(py - rotated.height / 2)))
    core = radius * SS * 0.2
    draw.ellipse((c - core, c - core, c + core, c + core), fill=_rgba(centre, 255))
    return _down(image).filter(ImageFilter.GaussianBlur(0.6))


def _leaf(length: int, colour: tuple, angle: float) -> Image.Image:
    image = Image.new("RGBA", (length * SS, int(length * 0.42) * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(image)
    d.ellipse((0, 0, image.width - 1, image.height - 1), fill=_rgba(colour, 225))
    d.line((image.width * 0.05, image.height / 2, image.width * 0.95, image.height / 2),
           fill=_rgba(tuple(int(v * 0.8) for v in colour), 200), width=SS)
    image = _down(image)
    return image.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)


def _floral_cluster(width: int, height: int, theme: Theme, rng: random.Random, flip: bool) -> Image.Image:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    petal_tones = [theme.accent, (236, 168, 168), (246, 196, 176), (226, 140, 150)]
    for _ in range(14):
        leaf = _leaf(rng.randint(70, 130), theme.soft, rng.uniform(0, 360))
        x = int(rng.uniform(0, width * 0.85))
        y = int(rng.uniform(0, height * 0.6) if rng.random() < 0.7 else rng.uniform(0, height * 0.9))
        image.alpha_composite(leaf, (max(0, x - leaf.width // 2), max(0, y - leaf.height // 2)))
    for _ in range(7):
        flower = _flower(rng.randint(36, 78), rng.choice(petal_tones), (236, 196, 120), rng)
        x = int(rng.uniform(width * 0.05, width * 0.75))
        y = int(rng.uniform(height * 0.05, height * 0.55))
        image.alpha_composite(flower, (x, y))
    # keep the cluster hugging the corner: fade it out towards the centre of the card
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    fade = np.clip(1.25 - np.hypot(xx / width, yy / height) * 1.05, 0, 1)
    alpha = np.asarray(image.getchannel("A"), np.float32) * fade
    image.putalpha(Image.fromarray(alpha.astype(np.uint8)))
    return ImageOps.mirror(image) if flip else image


def _leaf_wash(theme: Theme, rng: random.Random) -> Image.Image:
    image = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for corner in [(0, 0), (W, 0), (0, H), (W, H)]:
        for _ in range(9):
            base_angle = math.degrees(math.atan2(H / 2 - corner[1], W / 2 - corner[0]))
            sprig = Image.new("RGBA", (420, 420), (0, 0, 0, 0))
            d = ImageDraw.Draw(sprig)
            d.line((210, 210, 400, 210), fill=_rgba(theme.accent, 120), width=3)
            for k in range(6):
                leaf = _leaf(rng.randint(50, 86), rng.choice([theme.accent, theme.soft]), rng.choice([35, -35]) + rng.uniform(-10, 10))
                lx = 220 + k * 30
                sprig.alpha_composite(leaf, (lx - leaf.width // 2, 210 - leaf.height // 2))
            sprig = sprig.rotate(-(base_angle + rng.uniform(-50, 50)), resample=Image.Resampling.BICUBIC)
            a = sprig.getchannel("A").point(lambda v: int(v * 0.55))
            sprig.putalpha(a)
            ox = int(corner[0] - 210 + rng.uniform(-80, 80))
            oy = int(corner[1] - 210 + rng.uniform(-80, 80))
            _paste_clipped(image, sprig, ox, oy)
    return image.filter(ImageFilter.GaussianBlur(1.2))


def _paste_clipped(target: Image.Image, layer: Image.Image, x: int, y: int) -> None:
    left, top = max(0, x), max(0, y)
    right, bottom = min(target.width, x + layer.width), min(target.height, y + layer.height)
    if right <= left or bottom <= top:
        return
    piece = layer.crop((left - x, top - y, right - x, bottom - y))
    target.alpha_composite(piece, (left, top))


def _toran(theme: Theme, rng: random.Random) -> Image.Image:
    """Marigold garland swags with mango leaves along the top of the card."""
    height = 330
    image = Image.new("RGBA", (W, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(image)
    swags = 4
    span = W / swags
    for s in range(swags):
        x0 = s * span
        for k in range(30):
            t = k / 29
            x = x0 + span * t
            y = 40 + math.sin(math.pi * t) * 120
            r = 15 + 3 * math.sin(k)
            tone = (250, 160, 30) if k % 3 else (238, 110, 20)
            d.ellipse((x - r, y - r, x + r, y + r), fill=_rgba(tone, 255), outline=_rgba((200, 90, 10), 255))
        cx = x0 + span / 2
        for k in range(3):                     # hanging strand
            y = 175 + k * 34
            d.ellipse((cx - 13, y - 13, cx + 13, y + 13), fill=_rgba((250, 150, 30) if k % 2 else (236, 90, 40)))
        bell = [(cx - 16, 285), (cx + 16, 285), (cx + 10, 262), (cx - 10, 262)]
        d.polygon(bell, fill=_rgba(theme.accent))
    for s in range(swags + 1):
        x = s * span
        for k in range(5):
            leaf = _leaf(78, theme.soft, -90 + (k - 2) * 22)
            _paste_clipped(image, leaf, int(x - leaf.width / 2 + (k - 2) * 8), 28)
        d.ellipse((x - 22, 18, x + 22, 62), fill=_rgba((236, 100, 30)))
    return image


def _temple_top(theme: Theme) -> Image.Image:
    """Layered gopuram-like crown with bells along the top edge."""
    height = 300
    image, draw = _canvas(W, height)
    s = SS
    cx = W / 2
    for k, (half, top) in enumerate([(380, 150), (300, 110), (220, 70), (140, 34)]):
        tone = theme.accent if k % 2 == 0 else theme.soft
        draw.line([((cx - half) * s, (top + 60) * s), ((cx - half + 30) * s, top * s), ((cx + half - 30) * s, top * s),
                   ((cx + half) * s, (top + 60) * s)], fill=_rgba(tone, 235), width=5 * s)
    draw.ellipse(((cx - 18) * s, 0, (cx + 18) * s, 36 * s), fill=_rgba(theme.accent))
    for k in range(13):
        x = 60 + k * (W - 120) / 12
        draw.line((x * s, 0, x * s, (60 + (k % 2) * 20) * s), fill=_rgba(theme.accent, 200), width=2 * s)
        y = (60 + (k % 2) * 20) * s
        draw.pieslice((x * s - 12 * s, y - 6 * s, x * s + 12 * s, y + 24 * s), 180, 360, fill=_rgba(theme.accent))
    return _down(image)


def _stars(theme: Theme, rng: random.Random) -> Image.Image:
    image = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(image)
    for _ in range(140):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        if 200 < y < H - 200 and 140 < x < W - 140 and rng.random() < 0.8:
            continue
        r = rng.choice([1, 1, 1.5, 2, 3])
        d.ellipse((x - r, y - r, x + r, y + r), fill=_rgba(theme.accent, rng.randint(120, 230)))
    # fairy-light strings across the top
    for row in range(3):
        y0 = 50 + row * 46
        for k in range(28):
            t = k / 27
            x = t * W
            y = y0 + math.sin(t * math.pi) * 60
            glow = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
            ImageDraw.Draw(glow).ellipse((12, 12, 28, 28), fill=_rgba((255, 224, 150), 230))
            glow = glow.filter(ImageFilter.GaussianBlur(4))
            _paste_clipped(image, glow, int(x - 20), int(y - 20))
    return image


# --------------------------------------------------------------------------- text

def _text_line(text: str, font_name: str, size: int, colour: tuple, max_width: int, tracking: float = 0.0,
               shadow: tuple | None = None, upper: bool = False) -> Image.Image:
    text = text.upper() if upper else text
    while True:
        f = _font_for(text, font_name, size)
        spacing = size * tracking
        width = (f.getlength(text) + spacing * max(0, len(text) - 1)) if tracking else f.getlength(text)
        if width <= max_width or size <= 14:
            break
        size = int(size * 0.94)
    ascent, descent = f.getmetrics()
    pad = int(size * 0.35)
    image = Image.new("RGBA", (int(width) + pad * 2, ascent + descent + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    if tracking and not _needs_indic(text):
        x = pad
        for ch in text:
            draw.text((x, pad), ch, font=f, fill=_rgba(colour))
            x += f.getlength(ch) + spacing
    else:
        draw.text((pad, pad), text, font=f, fill=_rgba(colour))
    if shadow:
        glow = Image.new("RGBA", image.size, _rgba(shadow, 0))
        glow.putalpha(image.getchannel("A").filter(ImageFilter.GaussianBlur(size * 0.12)).point(lambda v: int(v * 0.75)))
        glow.alpha_composite(image)
        image = glow
    bbox = image.getchannel("A").getbbox()
    return image.crop((0, bbox[1] - 2, image.width, bbox[3] + 2)) if bbox else image


def _wrap(text: str, font_name: str, size: int, max_width: int) -> list[str]:
    f = _font_for(text, font_name, size)
    words, lines, current = text.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if f.getlength(trial) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _divider(width: int, colour: tuple) -> Image.Image:
    image, draw = _canvas(width, 30)
    s = SS
    mid = width / 2 * s
    draw.line((0, 15 * s, mid - 26 * s, 15 * s), fill=_rgba(colour, 200), width=2 * s)
    draw.line((mid + 26 * s, 15 * s, width * s, 15 * s), fill=_rgba(colour, 200), width=2 * s)
    r = 9 * s
    draw.polygon([(mid, 15 * s - r), (mid + r, 15 * s), (mid, 15 * s + r), (mid - r, 15 * s)], fill=_rgba(colour))
    for dx in (-18, 18):
        draw.ellipse((mid + dx * s - 3 * s, 12 * s, mid + dx * s + 3 * s, 18 * s), fill=_rgba(colour))
    return _down(image)


@dataclass
class Block:
    image: Image.Image
    gap_after: int
    names: bool = False


def _text_blocks(details: dict, theme: Theme, scale: float, max_width: int, on_photo: bool) -> list[Block]:
    ink, accent, names_ink = theme.ink, theme.accent, theme.name_ink
    shadow = (0, 0, 0) if theme.dark else None
    s = lambda v: int(v * scale)
    blocks: list[Block] = []
    blocks.append(Block(_text_line(KINDS[details["kind"]], theme.head_font, s(34), accent, max_width, 0.28, shadow, upper=True), s(26)))
    for line in _wrap(details["message"], theme.body_font, s(32), max_width):
        blocks.append(Block(_text_line(line, theme.body_font, s(32), ink, max_width, 0, shadow), s(8)))
    blocks[-1].gap_after = s(26)
    blocks.append(Block(_text_line(details["bride"], theme.names_font, s(118), names_ink, max_width, 0, shadow), s(4), names=True))
    if details["bride_family"]:
        blocks.append(Block(_text_line(details["bride_family"], theme.body_font, s(26), ink, max_width, 0, shadow), s(8)))
    blocks.append(Block(_text_line("&", theme.names_font, s(76), accent, max_width, 0, shadow), s(4), names=True))
    blocks.append(Block(_text_line(details["groom"], theme.names_font, s(118), names_ink, max_width, 0, shadow), s(4), names=True))
    if details["groom_family"]:
        blocks.append(Block(_text_line(details["groom_family"], theme.body_font, s(26), ink, max_width, 0, shadow), s(8)))
    blocks[-1].gap_after = s(28)
    blocks.append(Block(_divider(min(max_width, s(520)), accent), s(26)))
    for index, event in enumerate(details["events"]):
        if event["title"]:
            blocks.append(Block(_text_line(event["title"], theme.head_font, s(34), accent, max_width, 0.18, shadow, upper=True), s(10)))
        when = "  •  ".join(v for v in (event["date"], event["time"]) if v)
        if when:
            blocks.append(Block(_text_line(when, "serif_bold" if theme.head_font != "sans_light" else "sans_medium", s(38), ink, max_width, 0, shadow), s(8)))
        if event["venue"]:
            blocks.append(Block(_text_line(event["venue"], theme.body_font, s(32), ink, max_width, 0, shadow), s(4)))
        if event["address"]:
            for line in _wrap(event["address"], theme.body_font, s(25), max_width)[:2]:
                blocks.append(Block(_text_line(line, theme.body_font, s(25), ink, max_width, 0, shadow), s(2)))
        blocks[-1].gap_after = s(30) if index < len(details["events"]) - 1 else s(30)
    if details["hosts"]:
        for line in _wrap(details["hosts"], theme.body_font, s(26), max_width)[:2]:
            blocks.append(Block(_text_line(line, "serif_italic" if theme.body_font == "lora" else theme.body_font, s(26), ink, max_width, 0, shadow), s(6)))
    if details["rsvp"]:
        blocks.append(Block(_text_line(details["rsvp"], "sans", s(24), accent, max_width, 0.06, shadow), 0))
    if blocks:
        blocks[-1].gap_after = 0
    return blocks


def _stack_height(blocks: list[Block]) -> int:
    return sum(b.image.height + b.gap_after for b in blocks)


# --------------------------------------------------------------------------- photo

def prepare_photo(data: bytes, face_finder: FaceFinder | None) -> tuple[Image.Image, list[tuple[float, float, float, float]]]:
    """Decode, orient and size the couple photo; returns (RGB image, normalised face boxes)."""
    import io
    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.draft("RGB", (2400, 2400))
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except Exception as exc:
        raise InvitationError("That photo could not be read. Upload a JPG or PNG.") from exc
    if min(image.size) < 400:
        raise InvitationError("The photo is too small; use one at least 800 pixels wide.")
    image.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
    image = ImageOps.autocontrast(image, cutoff=0.5, preserve_tone=True)
    faces = []
    if face_finder is not None:
        try:
            probe = image.copy()
            probe.thumbnail((900, 900))
            for x, y, w, h in face_finder(probe):
                faces.append((x / probe.width, y / probe.height, w / probe.width, h / probe.height))
        except Exception:
            faces = []
    return image, faces


def _focus(faces: list) -> tuple[float, float]:
    if not faces:
        return 0.5, 0.4
    xs = [f[0] + f[2] / 2 for f in faces]
    ys = [f[1] + f[3] / 2 for f in faces]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _cover_crop(image: Image.Image, size: tuple[int, int], focus: tuple[float, float], zoom: float = 1.0) -> Image.Image:
    """Crop ``image`` to the aspect of ``size`` around ``focus`` (faces), ``zoom`` >= 1 tightens."""
    tw, th = size
    aspect = tw / th
    width, height = image.size
    if width / height > aspect:
        ch = height / zoom
        cw = ch * aspect
    else:
        cw = width / zoom
        ch = cw / aspect
    fx, fy = focus[0] * width, focus[1] * height
    left = min(max(0, fx - cw / 2), width - cw)
    top = min(max(0, fy - ch * 0.42), height - ch)
    return image.crop((int(left), int(top), int(left + cw), int(top + ch))).resize(size, Image.Resampling.LANCZOS)


# --------------------------------------------------------------------------- scene building

def build_scene(details: dict, choice: DesignChoice, photo: Image.Image | None = None, faces: list | None = None) -> Scene:
    theme = THEMES[choice.theme]
    rng = random.Random(choice.seed)
    layout = choice.layout if photo is not None else "medallion"
    scene = Scene(theme=theme, background=_gradient(theme.bg_top, theme.bg_bottom, rng, theme.dark), seed=choice.seed)
    add = scene.layers.append
    margin = 110
    max_width = W - 2 * margin

    # ---- ornaments
    if theme.ornament in ("mandala", "stars"):
        if theme.ornament == "stars":
            add(Layer("Stars", _stars(theme, rng), 0, 0, 0.0, 1.6, "fade"))
        else:
            ghost = _mandala(620, theme.accent, theme.soft, rng, rings=5)
            ghost.putalpha(ghost.getchannel("A").point(lambda v: int(v * 0.12)))
            add(Layer("Background mandala", ghost, W // 2 - ghost.width // 2, H // 2 - ghost.height // 2 + 120, 0.0, 2.0, "grow"))
        add(Layer("Border", _border(theme.accent, 46), 0, 0, 0.2, 1.0, "fade"))
        corner = _corner_flourish(300, theme.accent, theme.soft, rng)
        for flip_x, flip_y in [(False, False), (True, False), (False, True), (True, True)]:
            img = corner
            if flip_x:
                img = ImageOps.mirror(img)
            if flip_y:
                img = ImageOps.flip(img)
            add(Layer("Corner", img, W - img.width if flip_x else 0, H - img.height if flip_y else 0, 0.3, 1.0, "grow"))
    elif theme.ornament == "floral":
        top = _floral_cluster(620, 560, theme, rng, False)
        bottom = ImageOps.flip(_floral_cluster(440, 380, theme, random.Random(choice.seed + 1), True))
        add(Layer("Florals top", top, -20, -20, 0.2, 1.2, "grow"))
        add(Layer("Florals bottom", bottom, W - bottom.width + 20, H - bottom.height + 20, 0.4, 1.2, "grow"))
    elif theme.ornament == "temple":
        add(Layer("Border", _border(theme.accent, 40), 0, 0, 0.2, 1.0, "fade"))
        add(Layer("Temple crown", _temple_top(theme), 0, 44, 0.3, 1.0, "rise"))
        base = ImageOps.flip(_temple_top(theme))
        add(Layer("Temple base", base, 0, H - base.height - 44, 0.4, 1.0, "fade"))
    elif theme.ornament == "toran":
        add(Layer("Marigold toran", _toran(theme, rng), 0, 0, 0.2, 1.0, "rise"))
        border = _border(theme.accent, 34, double=False)
        add(Layer("Border", border, 0, 0, 0.3, 1.0, "fade"))
    elif theme.ornament == "leaves":
        add(Layer("Leaves", _leaf_wash(theme, rng), 0, 0, 0.2, 1.4, "fade"))
    elif theme.ornament == "minimal":
        add(Layer("Border", _border(theme.accent, 60), 0, 0, 0.2, 1.0, "fade"))

    # ---- invocation at the very top
    top_y = {"toran": 330, "temple": 340, "floral": 150}.get(theme.ornament, 150)
    if details["invocation"]:
        inv = _text_line(details["invocation"], "serif_italic", 30, theme.accent, max_width, 0, None)
        add(Layer("Invocation", inv, W // 2 - inv.width // 2, top_y - 70 if theme.ornament not in ("toran", "temple") else top_y - 10,
                  0.5, 0.8, "fade"))
        top_y += 30 if theme.ornament not in ("toran", "temple") else 50

    # ---- photo / medallion: sized so the text below can stay at a comfortable size
    bottom_limit = H - {"floral": 250, "temple": 300, "leaves": 150}.get(theme.ornament, 140)
    comfortable = _stack_height(_text_blocks(details, theme, 0.92, max_width, layout == "bleed"))
    room = bottom_limit - top_y - 40 - 60 - comfortable
    text_top = top_y
    if layout == "arch":
        ph = int(min(700, max(420, room)))
        pw = int(ph * 600 / 700)
        box = (W // 2 - pw // 2, top_y + 40, pw, ph)
        _add_photo(scene, photo, faces, box, _arch_mask((box[2], box[3])))
        scene.photo_frame.append(Layer("Arch frame", _arch_frame(box, theme.accent, theme.soft), 0, 0, 0.9, 1.0, "fade"))
        text_top = box[1] + box[3] + 60
    elif layout == "round":
        d = int(min(560, max(400, room)))
        box = (W // 2 - d // 2, top_y + 40, d, d)
        mask = Image.new("L", (d * SS, d * SS), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, d * SS - 1, d * SS - 1), fill=255)
        _add_photo(scene, photo, faces, box, mask.resize((d, d), Image.Resampling.LANCZOS))
        ring, draw = _canvas(d + 80, d + 80)
        for pad, tone, width in [(14, theme.accent, 4), (30, theme.soft, 2)]:
            r0 = (40 - pad) * SS
            draw.ellipse((r0, r0, (d + 80) * SS - r0, (d + 80) * SS - r0), outline=_rgba(tone, 230), width=width * SS)
        scene.photo_frame.append(Layer("Ring", _down(ring), box[0] - 40, box[1] - 40, 0.9, 1.0, "grow"))
        text_top = box[1] + d + 70
    elif layout == "bleed":
        height = int(min(980, max(640, bottom_limit - comfortable - 20)))
        box = (0, 0, W, height)
        mask = np.ones((height, W), np.float32)
        fade_from = int(height * 0.6)
        mask[fade_from:] = np.linspace(1, 0, height - fade_from)[:, None] ** 1.4
        _add_photo(scene, photo, faces, box, Image.fromarray((mask * 255).astype(np.uint8), "L"))
        text_top = height - 30
        # ornaments that would sit on the photo are dropped (in place: ``add`` appends to this list)
        scene.layers[:] = [l for l in scene.layers if l.name not in ("Florals top", "Marigold toran", "Temple crown", "Invocation",
                                                                      "Background mandala")]
    else:  # medallion with monogram
        d = int(min(420, max(300, room)))
        med = _mandala(d // 2, theme.accent, theme.soft, rng, rings=3)
        disc_r = int(d * 0.27)
        disc, ddraw = _canvas(med.width, med.height)
        c = med.width * SS / 2
        ddraw.ellipse((c - disc_r * SS, c - disc_r * SS, c + disc_r * SS, c + disc_r * SS), fill=_rgba(theme.bg_top, 255),
                      outline=_rgba(theme.accent, 255), width=3 * SS)
        med.alpha_composite(_down(disc))
        add(Layer("Medallion", med, W // 2 - med.width // 2, top_y + 30, 0.6, 1.2, "grow"))
        initials = f"{details['bride'][:1]}{details['groom'][:1]}".upper()
        mono = _text_line(f"{initials[0]}{initials[1]}" if len(initials) == 2 else initials, theme.names_font, int(d * 0.22),
                          theme.name_ink, int(disc_r * 1.6), 0, None)
        add(Layer("Monogram", mono, W // 2 - mono.width // 2, top_y + 30 + med.height // 2 - mono.height // 2, 1.0, 1.0, "grow"))
        text_top = top_y + 30 + med.height + 50

    # ---- text stack, scaled to fit the space left
    available = bottom_limit - text_top
    scale = 1.0
    blocks = _text_blocks(details, theme, scale, max_width, layout == "bleed")
    while _stack_height(blocks) > available and scale > 0.5:
        scale *= 0.93
        blocks = _text_blocks(details, theme, scale, max_width, layout == "bleed")
    y = text_top + max(0, (available - _stack_height(blocks)) // 2)
    start = 1.6 if layout != "medallion" else 1.4
    name_top, name_bottom = None, None
    for block in blocks:
        x = W // 2 - block.image.width // 2
        add(Layer("Names" if block.names else "Text", block.image, x, y, start, 0.8, "rise"))
        if block.names:
            name_top = y if name_top is None else name_top
            name_bottom = y + block.image.height
        start += 0.28 if not block.names else 0.45
        y += block.image.height + block.gap_after
    if name_top is not None:
        scene.shimmer = (margin, name_top, max_width, name_bottom - name_top)
    return scene


def _add_photo(scene: Scene, photo: Image.Image, faces: list | None, box: tuple, mask: Image.Image) -> None:
    x, y, w, h = box
    focus = _focus(faces or [])
    zoom_room = 1.12
    source = _cover_crop(photo, (int(w * zoom_room), int(h * zoom_room)), focus)
    # where the faces land inside the zoomable source (so the slow push-in heads for them)
    scene.photo = PhotoLayer(source, mask, (int(x), int(y), int(w), int(h)), (0.5, 0.42))


# --------------------------------------------------------------------------- rendering

def _ease(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 1 - (1 - t) ** 3


def _with_alpha(image: Image.Image, factor: float) -> Image.Image:
    if factor >= 0.999:
        return image
    copy = image.copy()
    copy.putalpha(image.getchannel("A").point(lambda v: int(v * factor)))
    return copy


def _draw_layer(frame: Image.Image, layer: Layer, t: float | None) -> None:
    if t is None:
        _paste_clipped(frame, layer.image, layer.x, layer.y)
        return
    p = _ease((t - layer.start) / layer.length)
    if p <= 0:
        return
    image, x, y = layer.image, layer.x, layer.y
    if layer.motion == "rise":
        y += int(layer.rise * (1 - p))
    elif layer.motion == "grow" and p < 1:
        scale = 0.9 + 0.1 * p
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.BILINEAR)
        x += (layer.image.width - image.width) // 2
        y += (layer.image.height - image.height) // 2
    _paste_clipped(frame, _with_alpha(image, p), x, y)


def _photo_frame(photo: PhotoLayer, t: float | None) -> tuple[Image.Image, tuple[int, int]]:
    x, y, w, h = photo.box
    sw, sh = photo.source.size
    progress = 1.0 if t is None else min(1.0, max(0.0, t / DURATION))
    zoom = 1.0 + 0.1 * progress                      # slow push-in towards the couple
    cw, ch = sw / zoom / 1.12, sh / zoom / 1.12
    fx, fy = photo.focus
    left = min(max(0, fx * sw - cw / 2), sw - cw)
    top = min(max(0, fy * sh - ch / 2), sh - ch)
    crop = photo.source.resize((w, h), Image.Resampling.BILINEAR, box=(left, top, left + cw, top + ch))
    alpha = photo.mask
    if t is not None:
        reveal = _ease((t - photo.start) / photo.length)
        if reveal <= 0:
            return Image.new("RGBA", (w, h), (0, 0, 0, 0)), (x, y)
        if reveal < 1:
            alpha = alpha.point(lambda v: int(v * reveal))
    layer = crop.convert("RGBA")
    layer.putalpha(alpha)
    return layer, (x, y)


class Particles:
    def __init__(self, theme: Theme, seed: int):
        rng = random.Random(seed ^ 0x5EED)
        self.kind = theme.particles
        self.items = []
        count = {"gold": 70, "petals": 26, "bokeh": 30}.get(self.kind, 0)
        for _ in range(count):
            self.items.append({
                "x": rng.uniform(0, W), "y": rng.uniform(0, H), "speed": rng.uniform(18, 60),
                "size": rng.uniform(2, 5) if self.kind == "gold" else rng.uniform(12, 26) if self.kind == "petals" else rng.uniform(18, 46),
                "phase": rng.uniform(0, 6.28), "spin": rng.uniform(-1.5, 1.5),
            })
        self.colour = theme.accent
        self.sprites = {}

    def _sprite(self, size: int, angle: int) -> Image.Image:
        key = (size, angle)
        if key not in self.sprites:
            if self.kind == "petals":
                img = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
                ImageDraw.Draw(img).ellipse((size * 0.5, size * 0.7, size * 1.5, size * 1.3), fill=(236, 150, 150, 220))
                img = img.rotate(angle, resample=Image.Resampling.BICUBIC)
            else:
                img = Image.new("RGBA", (size * 4, size * 4), (0, 0, 0, 0))
                ImageDraw.Draw(img).ellipse((size * 1.5, size * 1.5, size * 2.5, size * 2.5),
                                            fill=_rgba(self.colour if self.kind == "gold" else (255, 226, 160), 255))
                img = img.filter(ImageFilter.GaussianBlur(size * (0.35 if self.kind == "gold" else 0.6)))
            self.sprites[key] = img
        return self.sprites[key]

    def draw(self, frame: Image.Image, t: float) -> None:
        if not self.items:
            return
        fade_in = _ease(t / 1.5)
        for item in self.items:
            if self.kind == "petals":
                y = (item["y"] + item["speed"] * t) % (H + 80) - 40
                x = item["x"] + 30 * math.sin(t * 0.8 + item["phase"])
                angle = int((item["phase"] * 57 + item["spin"] * 60 * t) % 360) // 15 * 15
                sprite = self._sprite(int(item["size"]), angle)
                alpha = 0.85
            else:
                y = (item["y"] - item["speed"] * 0.5 * t) % (H + 80) - 40
                x = item["x"] + 12 * math.sin(t * 0.6 + item["phase"])
                sprite = self._sprite(int(item["size"]) if self.kind == "gold" else int(item["size"] / 4) * 4, 0)
                twinkle = 0.5 + 0.5 * math.sin(t * 2.4 + item["phase"])
                alpha = (0.35 + 0.65 * twinkle) * (0.9 if self.kind == "gold" else 0.35)
            _paste_clipped(frame, _with_alpha(sprite, alpha * fade_in), int(x - sprite.width / 2), int(y - sprite.height / 2))


def render_frame(scene: Scene, t: float | None, particles: Particles | None = None) -> Image.Image:
    """One frame at time ``t`` (seconds); ``t=None`` renders the finished poster."""
    frame = scene.background.convert("RGBA")
    ornament_layers = [l for l in scene.layers if l.name not in ("Text", "Names")]
    text_layers = [l for l in scene.layers if l.name in ("Text", "Names")]
    for layer in ornament_layers:
        _draw_layer(frame, layer, t)
    if scene.photo is not None:
        image, position = _photo_frame(scene.photo, t)
        frame.alpha_composite(image, position)
    for layer in scene.photo_frame:
        _draw_layer(frame, layer, t)
    if particles is not None and t is not None:
        particles.draw(frame, t)
    names = Image.new("RGBA", frame.size, (0, 0, 0, 0)) if scene.shimmer and t is not None else None
    for layer in text_layers:
        _draw_layer(frame, layer, t)
        if names is not None and layer.name == "Names":
            _draw_layer(names, layer, t)
    if names is not None:
        sweep = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        _shimmer_on(sweep, names, scene.shimmer, t)
        frame.alpha_composite(sweep)
    return frame.convert("RGB")


def _shimmer_on(target: Image.Image, names: Image.Image, box: Box, t: float, start: float = 6.4, length: float = 1.8) -> None:
    p = (t - start) / length
    if not 0 <= p <= 1:
        return
    x, y, w, h = (int(v) for v in box)
    band = Image.new("L", (w, h), 0)
    centre = -0.25 * w + p * 1.5 * w
    ImageDraw.Draw(band).polygon([(centre - 50, h), (centre + 40, 0), (centre + 110, 0), (centre + 20, h)], fill=170)
    band = band.filter(ImageFilter.GaussianBlur(18))
    glyphs = names.crop((x, y, x + w, y + h)).getchannel("A")
    light = Image.new("RGBA", (w, h), (255, 248, 226, 0))
    light.putalpha(ImageChops.multiply(band, glyphs))
    target.alpha_composite(light, (x, y))


def render_poster(scene: Scene) -> Image.Image:
    poster = render_frame(scene, None)
    particles = Particles(scene.theme, scene.seed)
    if particles.items and scene.theme.particles != "petals":   # a few static sparkles on the poster too
        rgba = poster.convert("RGBA")
        particles.draw(rgba, 4.0)
        poster = rgba.convert("RGB")
    return poster


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def render_video(scene: Scene, output: Path, progress: Callable[[float], None] | None = None,
                 fps: int = FPS, duration: float = DURATION) -> None:
    """Vertical H.264 MP4 (1080x1920) suitable for WhatsApp status and Instagram reels."""
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is not installed on the data service, so the video can't be rendered.")
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
        "-i", "-", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(output),
    ]
    particles = Particles(scene.theme, scene.seed)
    frames = int(fps * duration)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for index in range(frames):
            t = index / fps
            frame = render_frame(scene, t, particles)
            if t > duration - 0.6:                       # fade out the last moment for a clean loop
                k = (duration - t) / 0.6
                frame = Image.blend(scene.background, frame, max(0.0, k)) if k < 1 else frame
            process.stdin.write(frame.tobytes())
            if progress and index % fps == 0:
                progress(index / frames)
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", "replace")
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {error[:300]}")
    except BrokenPipeError as exc:
        error = process.stderr.read().decode("utf-8", "replace")
        raise RuntimeError(f"ffmpeg stopped: {error[:300]}") from exc
    finally:
        if process.poll() is None:
            process.kill()
