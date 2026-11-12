"""Studio "look" matching (built-in, no external AI service, nothing leaves the server).

A studio uploads 5-40 *finished* photos whose look it wants. Finished photos alone don't say what
the edit was, so the look is learned as the difference between two populations, measured in CIE Lab:

    references  - the studio's finished look
    baseline    - how the studio's photos look after FrameBucket's basic corrections, before styling
                  (a rolling average of its own uploads, starting from a built-in default)

From that difference we derive a fixed preset, applied identically to every photo like a Lightroom
preset or LUT:

* tone curve   - luminance quantile mapping baseline -> references (contrast, matte or crushed blacks)
* colour grade - per-band shift of shadows, mid-tones and highlights (white balance, split toning)
* saturation   - overall chroma gain (muted vs vivid)

Because the same preset is applied to every photo, each photo keeps its own exposure and colours;
only the grade changes. Skin near faces gets a gentler colour change so skin tones stay natural.
"""
from __future__ import annotations

import io
import math
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

PROFILE_VERSION = 2
MIN_REFERENCES = 5
MAX_REFERENCES = 40
ANALYSIS_SIDE = 768
PERCENTILES = [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97, 99]
BANDS = ("shadow", "mid", "high")
MAX_TINT_SHIFT = 14.0          # Lab units per band; keeps the grade believable
CHROMA_GAIN_LIMITS = (0.6, 1.5)
BASELINE_WEIGHT = 0.03         # rolling baseline ~ the studio's last 30-60 photos
STRIP_ROWS = 512               # full-resolution work is done in strips to bound memory

# Typical wedding photos straight out of camera after FrameBucket's basic corrections. Used until a
# studio has processed enough photos for its own rolling baseline.
DEFAULT_BASELINE = {
    "l_pct": [0.6, 4.1, 7.5, 13.4, 22.4, 29.1, 35.9, 44.5, 52.7, 60.1, 68.4, 83.6, 91.1, 94.7, 99.3],
    "tints": {"shadow": [2.0, 3.5], "mid": [3.0, 8.0], "high": [1.0, 6.0]},
    "chroma_pct": [9.5, 16.5, 26.0, 38.5],
    "samples": 0,
}


class StyleError(ValueError):
    """The references cannot produce a usable profile."""


# --------------------------------------------------------------------------- analysis

def load_reference(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as opened:
            if opened.width * opened.height > 80_000_000:
                raise StyleError("A reference photo is larger than 80 megapixels.")
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise StyleError("Reference photos must be JPG, PNG or WebP images.") from exc
    return downscale(image, ANALYSIS_SIDE)


def downscale(image: Image.Image, side: int) -> Image.Image:
    if max(image.size) <= side:
        return image
    ratio = side / max(image.size)
    return image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
                        Image.Resampling.LANCZOS)


def _lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float32 Lab (L 0..100, a/b roughly -128..127)."""
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def _band_weights(luminance: np.ndarray) -> dict[str, np.ndarray]:
    shadow = np.clip((50.0 - luminance) / 30.0, 0.0, 1.0)
    high = np.clip((luminance - 50.0) / 30.0, 0.0, 1.0)
    return {"shadow": shadow, "mid": 1.0 - shadow - high, "high": high}


def analyze(image: Image.Image) -> dict:
    rgb = np.asarray(downscale(image.convert("RGB"), ANALYSIS_SIDE))
    lab = _lab(rgb)
    luminance, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    usable = (luminance > 3) & (luminance < 97)
    if usable.sum() < 500:
        usable = np.ones_like(luminance, dtype=bool)

    tints: dict[str, list[float] | None] = {}
    for band, weight in _band_weights(luminance).items():
        weight = weight * usable
        total = float(weight.sum())
        if total < 0.02 * weight.size:
            tints[band] = None
            continue
        tints[band] = [float((a * weight).sum() / total), float((b * weight).sum() / total)]

    chroma = np.hypot(a, b)
    chroma_area = usable & (luminance > 15) & (luminance < 92)
    chroma_values = chroma[chroma_area] if chroma_area.sum() > 500 else chroma.reshape(-1)
    return {
        "l_pct": [float(v) for v in np.percentile(luminance, PERCENTILES)],
        "tints": tints,
        "chroma_pct": [float(v) for v in np.percentile(chroma_values, [25, 50, 75, 90])],
    }


def aggregate(stats: Sequence[dict]) -> dict:
    l_pct = np.maximum.accumulate(np.mean([s["l_pct"] for s in stats], axis=0))
    tints: dict[str, list[float] | None] = {}
    for band in BANDS:
        values = [s["tints"][band] for s in stats if s["tints"].get(band) is not None]
        tints[band] = [round(float(v), 3) for v in np.mean(values, axis=0)] if values else None
    return {
        "l_pct": [round(float(v), 3) for v in l_pct],
        "tints": tints,
        "chroma_pct": [round(float(v), 3) for v in np.mean([s["chroma_pct"] for s in stats], axis=0)],
    }


def build_profile(references: Sequence[Image.Image]) -> dict:
    if len(references) < MIN_REFERENCES:
        raise StyleError(f"Upload at least {MIN_REFERENCES} reference photos so the look is reliable.")
    if len(references) > MAX_REFERENCES:
        raise StyleError(f"Upload at most {MAX_REFERENCES} reference photos.")
    reference = aggregate([analyze(image) for image in references])
    return {
        "version": PROFILE_VERSION,
        "reference_count": len(references),
        "reference": reference,
        "summary": describe(reference),
    }


def update_baseline(baseline: dict | None, photo_stats: dict) -> dict:
    """Fold one basic-corrected photo into the studio's rolling 'before styling' baseline."""
    current = baseline if baseline and baseline.get("l_pct") else DEFAULT_BASELINE
    samples = int(current.get("samples", 0))
    # Move quickly while the studio has few samples, then settle into a slow rolling average.
    weight = max(BASELINE_WEIGHT, 1.0 / (samples + 2)) if samples else 0.25
    mix = lambda old, new: [round(o + (n - o) * weight, 3) for o, n in zip(old, new)]
    tints = {}
    for band in BANDS:
        old, new = current["tints"].get(band), photo_stats["tints"].get(band)
        tints[band] = mix(old, new) if old and new else (old or new)
    return {
        "l_pct": [round(v, 3) for v in np.maximum.accumulate(mix(current["l_pct"], photo_stats["l_pct"]))],
        "tints": tints,
        "chroma_pct": mix(current["chroma_pct"], photo_stats["chroma_pct"]),
        "samples": samples + 1,
    }


def describe(reference: dict) -> dict:
    """Plain-language summary for the studio's settings page (relative to typical camera output)."""
    l_pct = reference["l_pct"]
    base = DEFAULT_BASELINE
    median = l_pct[PERCENTILES.index(50)]
    contrast = l_pct[PERCENTILES.index(90)] - l_pct[PERCENTILES.index(10)]
    base_contrast = base["l_pct"][PERCENTILES.index(90)] - base["l_pct"][PERCENTILES.index(10)]
    black_point = l_pct[PERCENTILES.index(1)]
    mid = reference["tints"].get("mid") or base["tints"]["mid"]
    shadow = reference["tints"].get("shadow") or mid
    high = reference["tints"].get("high") or mid
    warmth = mid[1] - base["tints"]["mid"][1]
    saturation = reference["chroma_pct"][1] / base["chroma_pct"][1]

    words = [
        "bright and airy" if median >= 55 else "dark and moody" if median <= 36 else "balanced exposure",
        "punchy contrast" if contrast >= base_contrast + 8 else "soft contrast" if contrast <= base_contrast - 8
        else "natural contrast",
    ]
    if black_point >= 6:
        words.append("matte (lifted) blacks")
    words.append("warm tones" if warmth >= 3 else "cool tones" if warmth <= -3 else "neutral white balance")
    if (high[1] - shadow[1]) - (base["tints"]["high"][1] - base["tints"]["shadow"][1]) >= 4:
        words.append("warm highlights, cooler shadows")
    words.append("vivid colours" if saturation >= 1.15 else "muted colours" if saturation <= 0.87 else "natural saturation")
    return {
        "brightness": round(median, 1),
        "contrast": round(contrast, 1),
        "warmth": round(warmth, 1),
        "saturation": round(saturation, 2),
        "words": words,
        "swatches": [_swatch(l, t) for l, t in ((22, shadow), (52, mid), (82, high))],
    }


def _swatch(luminance: float, tint: Sequence[float]) -> str:
    lab = np.array([[[luminance, tint[0] * 2.0, tint[1] * 2.0]]], dtype=np.float32)
    rgb = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)[0, 0] * 255, 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


# --------------------------------------------------------------------------- application

def apply_style(
    image: Image.Image,
    profile: dict,
    baseline: dict | None = None,
    strength: float = 0.8,
    faces: Iterable[tuple[int, int, int, int]] = (),
) -> Image.Image:
    """Return a new image with the studio look applied. ``strength`` is 0..1."""
    strength = float(min(1.0, max(0.0, strength)))
    if strength <= 0.01 or not profile or profile.get("version") != PROFILE_VERSION:
        return image
    base = baseline if baseline and baseline.get("l_pct") else DEFAULT_BASELINE
    lut_l, lut_da, lut_db, chroma_gain = _preset_tables(base, profile["reference"], strength)

    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    protect = _skin_protection(image, faces, (width, height))
    output = np.empty_like(rgb)
    soft_gain = np.float32(1.0 + (chroma_gain - 1.0) * 0.35)
    for top in range(0, height, STRIP_ROWS):
        bottom = min(height, top + STRIP_ROWS)
        lab = _lab(rgb[top:bottom])
        index = np.clip((lab[..., 0] * 10).astype(np.int32), 0, 1000)
        a, b = lab[..., 1], lab[..., 2]
        new_a = (a + lut_da[index]) * chroma_gain
        new_b = (b + lut_db[index]) * chroma_gain
        if protect is not None:
            keep = protect[top:bottom]
            # On skin near faces, follow only about a third of the colour grade.
            new_a = new_a * (1 - keep) + (a + lut_da[index] * 0.35) * soft_gain * keep
            new_b = new_b * (1 - keep) + (b + lut_db[index] * 0.35) * soft_gain * keep
        lab[..., 0] = lut_l[index]
        lab[..., 1] = new_a
        lab[..., 2] = new_b
        back = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
        output[top:bottom] = np.clip(back * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(output, mode="RGB")


def _preset_tables(base: dict, reference: dict, strength: float):
    grid = np.linspace(0.0, 100.0, 1001)

    # Tone curve: quantile mapping from the studio's unstyled photos to its finished look.
    xp = np.concatenate([[0.0], np.maximum.accumulate(base["l_pct"]), [100.0]])
    xp = xp + np.arange(xp.size) * 1e-4                      # strictly increasing for np.interp
    fp = np.maximum.accumulate(np.clip(np.concatenate([[0.0], reference["l_pct"], [100.0]]), 0.0, 100.0))
    delta = _smooth(np.interp(grid, xp, fp) - grid, 15)
    lut_l = np.maximum.accumulate(np.clip(grid + delta * strength, 0.0, 100.0)).astype(np.float32)

    # Colour grade per tonal band, blended smoothly across luminance.
    weights = _band_weights(grid)
    lut_da = np.zeros_like(grid)
    lut_db = np.zeros_like(grid)
    for band in BANDS:
        target, current = reference["tints"].get(band), base["tints"].get(band)
        if target is None or current is None:
            continue
        shift = np.array(target, dtype=np.float64) - np.array(current, dtype=np.float64)
        norm = float(np.hypot(*shift))
        if norm > MAX_TINT_SHIFT:
            shift *= MAX_TINT_SHIFT / norm
        lut_da += weights[band] * shift[0] * strength
        lut_db += weights[band] * shift[1] * strength

    ratio = reference["chroma_pct"][1] / max(1.0, base["chroma_pct"][1])
    ratio = min(CHROMA_GAIN_LIMITS[1], max(CHROMA_GAIN_LIMITS[0], ratio))
    chroma_gain = np.float32(1.0 + (ratio - 1.0) * strength)
    return lut_l, lut_da.astype(np.float32), lut_db.astype(np.float32), chroma_gain


def _smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    radius = int(math.ceil(sigma * 3))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(values, radius, mode="edge"), kernel, mode="valid")


def _skin_protection(image: Image.Image, faces, size: tuple[int, int]):
    """0..1 float32 mask at full resolution: skin near faces (strong) and elsewhere (light)."""
    faces = list(faces)
    small = downscale(image.convert("RGB"), 1024)
    scale = small.width / size[0]
    ycrcb = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2YCrCb)
    skin = cv2.inRange(ycrcb, (40, 135, 80), (255, 175, 130)).astype(np.float32) / 255.0
    near_face = np.zeros_like(skin)
    for x, y, w, h in faces:
        x0, y0 = int((x - w * 0.6) * scale), int((y - h * 0.4) * scale)
        x1, y1 = int((x + w * 1.6) * scale), int((y + h * 2.2) * scale)
        near_face[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 1.0
    mask = skin * (0.25 + 0.75 * near_face)
    if float(mask.max()) <= 0:
        return None
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2.0, small.width / 300))
    return cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
