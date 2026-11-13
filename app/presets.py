"""Editing presets: the studio uploads its own Lightroom / Camera Raw presets or LUTs once, and for each photo
FrameBucket picks the preset that suits it best and finishes the photo on top of it.

Supported files
* ``.xmp``        Lightroom Classic / Lightroom / Camera Raw preset (Develop settings)
* ``.lrtemplate`` older Lightroom Classic preset
* ``.cube``       3D LUT (Resolve, Premiere, VSCO-style LUT packs)

Everything global in a preset is a per-pixel colour mapping, so every preset is compiled once, when it is
uploaded, into a 33x33x33 3D LUT and then applied with trilinear interpolation. Lightroom's processing is
proprietary, so the conversion is a close approximation of: exposure, contrast, highlights, shadows, whites,
blacks, white balance (temperature/tint), vibrance, saturation, the point tone curve (master and RGB), the
parametric curve, HSL (hue/saturation/luminance per colour), split toning and colour grading. Local effects
(clarity, texture, dehaze, sharpening, noise reduction, grain, vignette, masks, lens corrections) can't be
expressed as a colour mapping and are skipped; they are listed back to the photographer.

Choosing a preset per photo (nothing leaves the server): every candidate is rendered on a small copy of the
photo and scored on exposure, clipping, skin tones, colour intensity, how close it lands to the studio's
learned look (if any), and the scene the photographer tagged the preset for (day outdoor, indoor, night,
portrait). The lowest score wins.
"""
from __future__ import annotations

import io
import math
import re
import zlib
from dataclasses import dataclass, field
from typing import Iterable, Sequence
from xml.etree import ElementTree

import cv2
import numpy as np
from PIL import Image

from . import style

LUT_SIZE = 33
MAX_FILE_BYTES = 4 * 1024 * 1024
SCORE_SIDE = 320
SCENES = ("any", "outdoor", "indoor", "night", "portrait")
EXTENSIONS = (".xmp", ".lrtemplate", ".cube")

# Settings that are local or optical: recognised, but not representable as a colour mapping.
SKIPPED = {
    "Clarity2012": "clarity", "Clarity": "clarity", "Texture": "texture", "Dehaze": "dehaze",
    "Sharpness": "sharpening", "LuminanceSmoothing": "noise reduction", "ColorNoiseReduction": "noise reduction",
    "GrainAmount": "grain", "PostCropVignetteAmount": "vignette", "VignetteAmount": "vignette",
    "LensProfileEnable": "lens corrections", "AutoLateralCA": "lens corrections",
}
HSL_COLOURS = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
HSL_CENTRES = (0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0)


class PresetError(ValueError):
    """The file isn't a preset we can read."""


@dataclass
class Preset:
    """A compiled preset, ready to apply."""
    preset_id: str
    name: str
    scene: str
    lut: np.ndarray                          # float32 (N, N, N, 3), indexed [r, g, b], values 0..1
    summary: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- parsing

def parse(filename: str, data: bytes) -> tuple[str, np.ndarray, dict]:
    """Return (suggested name, LUT, summary) for an uploaded preset file."""
    if len(data) > MAX_FILE_BYTES:
        raise PresetError("Preset files must be 4 MB or smaller.")
    lower = (filename or "").lower()
    text = data.decode("utf-8", errors="replace").lstrip("﻿")
    stem = re.sub(r"\.(xmp|lrtemplate|cube)$", "", (filename or "Preset").rsplit("/", 1)[-1], flags=re.I).strip()
    if lower.endswith(".cube") or re.search(r"^\s*LUT_3D_SIZE\s+\d+", text, re.M):
        name, lut = _parse_cube(text)
        settings: dict = {}
        skipped: list[str] = []
        kind = "cube"
    elif lower.endswith(".lrtemplate") or re.search(r"^\s*s\s*=\s*\{", text):
        name, settings = _parse_lrtemplate(text)
        kind = "lrtemplate"
        lut, skipped = build_lut(settings)
    elif lower.endswith(".xmp") or "crs:" in text:
        name, settings = _parse_xmp(text)
        kind = "xmp"
        lut, skipped = build_lut(settings)
    else:
        raise PresetError("Upload a Lightroom preset (.xmp or .lrtemplate) or a 3D LUT (.cube).")
    if kind != "cube" and not _has_effect(settings):
        raise PresetError("This preset has no colour or tone settings FrameBucket can use.")
    summary = describe(lut)
    summary.update({"kind": kind, "skipped": sorted(set(skipped)), "settings": len(settings)})
    return (name or stem or "Preset")[:80], lut, summary


def _number(value) -> float | None:
    try:
        return float(str(value).strip().replace("+", ""))
    except (TypeError, ValueError):
        return None


def _parse_xmp(text: str) -> tuple[str, dict]:
    settings: dict = {}
    # Attribute form: crs:Exposure2012="+0.35"
    for key, value in re.findall(r'crs:([A-Za-z0-9]+)\s*=\s*"([^"]*)"', text):
        number = _number(value)
        if number is not None:
            settings[key] = number
        elif key == "Name":
            settings["__name"] = value
    # Element form: <crs:Exposure2012>0.35</crs:Exposure2012>
    for key, value in re.findall(r"<crs:([A-Za-z0-9]+)>([^<]*)</crs:\1>", text):
        number = _number(value)
        if number is not None:
            settings.setdefault(key, number)
    # Tone curves: <crs:ToneCurvePV2012><rdf:Seq><rdf:li>0, 0</rdf:li>...
    for key in ("ToneCurvePV2012", "ToneCurvePV2012Red", "ToneCurvePV2012Green", "ToneCurvePV2012Blue", "ToneCurve"):
        match = re.search(rf"<crs:{key}>(.*?)</crs:{key}>", text, re.S)
        if match:
            points = [tuple(float(v) for v in item.split(","))
                      for item in re.findall(r"<rdf:li>\s*([-0-9.]+\s*,\s*[-0-9.]+)\s*</rdf:li>", match.group(1))]
            if len(points) >= 2:
                settings[key] = points
    name = settings.pop("__name", "")
    if not name:
        title = re.search(r"<crs:Name>.*?<rdf:li[^>]*>([^<]+)</rdf:li>", text, re.S)
        name = title.group(1).strip() if title else ""
    try:  # well-formedness check, so binary or HTML junk is refused
        ElementTree.fromstring(text[text.find("<"):].encode("utf-8")) if "<" in text else None
    except ElementTree.ParseError as exc:
        if not settings:
            raise PresetError("That .xmp file couldn't be read.") from exc
    return name, settings


def _parse_lrtemplate(text: str) -> tuple[str, dict]:
    settings: dict = {}
    body = text
    block = re.search(r"settings\s*=\s*\{(.*)\}\s*,?\s*\}\s*,?\s*(?:version|\}|$)", text, re.S)
    if block:
        body = block.group(1)
    for key, value in re.findall(r"\b([A-Za-z][A-Za-z0-9]*)\s*=\s*(-?[0-9.]+)\s*,", body):
        settings[key] = float(value)
    for key in ("ToneCurvePV2012", "ToneCurvePV2012Red", "ToneCurvePV2012Green", "ToneCurvePV2012Blue", "ToneCurve"):
        match = re.search(rf"\b{key}\s*=\s*\{{([^}}]*)\}}", body)
        if match:
            values = [float(v) for v in re.findall(r"-?[0-9.]+", match.group(1))]
            if len(values) >= 4:
                settings[key] = list(zip(values[0::2], values[1::2]))
    title = re.search(r'\btitle\s*=\s*"([^"]+)"', text) or re.search(r'internalName\s*=\s*"([^"]+)"', text)
    return (title.group(1).strip() if title else ""), settings


def _parse_cube(text: str) -> tuple[str, np.ndarray]:
    size = None
    domain_min, domain_max = np.zeros(3), np.ones(3)
    title = ""
    rows: list[list[float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper.startswith("TITLE"):
            title = line[5:].strip().strip('"')
        elif upper.startswith("LUT_3D_SIZE"):
            size = int(line.split()[1])
        elif upper.startswith("LUT_1D_SIZE"):
            raise PresetError("1D LUTs aren't supported; export a 3D .cube LUT.")
        elif upper.startswith("DOMAIN_MIN"):
            domain_min = np.array([float(v) for v in line.split()[1:4]])
        elif upper.startswith("DOMAIN_MAX"):
            domain_max = np.array([float(v) for v in line.split()[1:4]])
        elif upper[0].isalpha():
            continue
        else:
            parts = line.split()
            if len(parts) >= 3:
                rows.append([float(v) for v in parts[:3]])
    if not size or size < 2 or size > 129:
        raise PresetError("That .cube file has no valid LUT_3D_SIZE.")
    if len(rows) != size ** 3:
        raise PresetError(f"That .cube file should have {size ** 3} colour rows but has {len(rows)}.")
    table = np.asarray(rows, dtype=np.float32)
    table = (table - domain_min) / np.maximum(domain_max - domain_min, 1e-6)
    # .cube order: red changes fastest, then green, then blue -> reshape to [b, g, r] and transpose to [r, g, b].
    lut = table.reshape(size, size, size, 3).transpose(2, 1, 0, 3)
    if size != LUT_SIZE:
        lut = _resample(lut, LUT_SIZE)
    return title, np.clip(lut, 0.0, 1.0).astype(np.float32)


def _resample(lut: np.ndarray, size: int) -> np.ndarray:
    grid = _grid(size).reshape(-1, 3)
    return apply_lut_array(grid, lut).reshape(size, size, size, 3)


def _has_effect(settings: dict) -> bool:
    return any(key not in SKIPPED and (isinstance(value, list) or abs(float(value)) > 1e-6)
               for key, value in settings.items()
               if key not in ("Temperature", "Tint", "Version", "ProcessVersion"))


# --------------------------------------------------------------------------- compiling settings into a LUT

def _grid(size: int = LUT_SIZE) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([r, g, b], axis=-1)


def _to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _set_luma(rgb: np.ndarray, old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Change brightness while keeping the colour's hue and roughly its saturation."""
    ratio = np.where(old > 1e-4, new / np.maximum(old, 1e-4), 1.0)[..., None]
    scaled = rgb * ratio
    # Where scaling would clip, move towards white instead (like a highlight roll-off).
    over = np.max(scaled, axis=-1, keepdims=True)
    blend = np.clip((over - 1.0) / np.maximum(over, 1e-4), 0.0, 1.0)
    return scaled * (1 - blend) + new[..., None] * blend


def _monotone_curve(points: Sequence[tuple[float, float]], x: np.ndarray, scale: float = 255.0) -> np.ndarray:
    """Monotone cubic (Fritsch-Carlson) curve through Lightroom's 0..255 points."""
    pts = sorted({(float(a) / scale, float(b) / scale) for a, b in points})
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    if xs.size < 2:
        return x
    h = np.diff(xs)
    h[h == 0] = 1e-6
    delta = np.diff(ys) / h
    m = np.zeros_like(xs)
    m[0], m[-1] = delta[0], delta[-1]
    for i in range(1, xs.size - 1):
        if delta[i - 1] * delta[i] <= 0:
            m[i] = 0.0
        else:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    xc = np.clip(x, xs[0], xs[-1])
    index = np.clip(np.searchsorted(xs, xc) - 1, 0, xs.size - 2)
    t = (xc - xs[index]) / h[index]
    h00, h10 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t
    h01, h11 = -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    y = h00 * ys[index] + h10 * h[index] * m[index] + h01 * ys[index + 1] + h11 * h[index] * m[index + 1]
    # Outside the first/last point Lightroom holds the end values.
    y = np.where(x < xs[0], ys[0], np.where(x > xs[-1], ys[-1], y))
    return np.clip(y, 0.0, 1.0)


def _hue_colour(hue: float) -> np.ndarray:
    rgb = cv2.cvtColor(np.array([[[hue / 2.0, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
    return rgb.astype(np.float32) / 255.0


def build_lut(settings: dict) -> tuple[np.ndarray, list[str]]:
    """Compile Lightroom/ACR develop settings into a 3D LUT. Returns (lut, skipped setting names)."""
    s = lambda key, default=0.0: float(settings.get(key, default)) if not isinstance(settings.get(key), list) else default
    skipped = [label for key, label in SKIPPED.items() if key in settings and abs(s(key)) > 1e-6
               and not (key == "LensProfileEnable")]
    rgb = _grid()

    # White balance and exposure, in linear light.
    linear = _to_linear(rgb)
    if "IncrementalTemperature" in settings or "IncrementalTint" in settings:
        warmth, tint = s("IncrementalTemperature") / 100.0, s("IncrementalTint") / 100.0
    elif "Temperature" in settings:
        # Raw presets store a Kelvin value; compare with daylight in mireds.
        warmth = float(np.clip((1e6 / 5500 - 1e6 / max(2000.0, s("Temperature"))) / 60.0, -1.0, 1.0))
        tint = float(np.clip(s("Tint") / 100.0, -1.0, 1.0))
    else:
        warmth = tint = 0.0
    gains = np.array([1 + 0.35 * warmth, 1 - 0.15 * tint, 1 - 0.45 * warmth], dtype=np.float32)
    linear = linear * gains / max(1.0, float(_luma(gains[None])[0]))
    exposure = s("Exposure2012", s("Exposure"))
    linear = linear * (2.0 ** exposure)
    rgb = _to_srgb(linear)

    # Tone: contrast, then highlights/shadows/whites/blacks, as a luminance curve.
    luma = np.clip(_luma(rgb), 0.0, None)
    x = np.clip(luma, 0.0, 1.0)
    y = x.copy()
    contrast = s("Contrast2012", s("Contrast") - 25.0 if "Contrast" in settings else 0.0) / 100.0
    if abs(contrast) > 1e-4:
        sigmoid = lambda v, k: 1 / (1 + np.exp(-k * (v - 0.5)))
        k = 1.0 + 7.0 * abs(contrast)
        s_curve = (sigmoid(y, k) - sigmoid(0, k)) / (sigmoid(1, k) - sigmoid(0, k))
        y = y + (s_curve - y) * (1.0 if contrast > 0 else -0.6) * min(1.0, abs(contrast) * 1.4)
    highlights, shadows = s("Highlights2012", -s("HighlightRecovery")) / 100.0, s("Shadows2012", s("FillLight")) / 100.0
    whites, blacks = s("Whites2012") / 100.0, s("Blacks2012") / 100.0
    y = y + highlights * 0.22 * np.clip((y - 0.45) / 0.4, 0, 1) ** 1.5 * (1 - y) * (2.2 if highlights > 0 else 3.5)
    y = y + shadows * 0.25 * np.clip((0.6 - y) / 0.5, 0, 1) ** 1.5 * (y + 0.06) * 2.0
    y = y + whites * 0.12 * np.clip((y - 0.6) / 0.4, 0, 1) ** 2
    y = y + blacks * 0.10 * np.clip((0.35 - y) / 0.35, 0, 1) ** 2
    # Parametric curve (regions split at 25/50/75%).
    for key, centre in (("ParametricShadows", 0.15), ("ParametricDarks", 0.38), ("ParametricLights", 0.62),
                        ("ParametricHighlights", 0.85)):
        amount = s(key) / 100.0
        if amount:
            y = y + amount * 0.12 * np.exp(-((y - centre) / 0.16) ** 2)
    y = np.clip(y, 0.0, 1.0)
    rgb = np.clip(_set_luma(rgb, np.maximum(luma, 1e-4), y), 0.0, 1.0)

    # Point curves: master, then per channel.
    master = settings.get("ToneCurvePV2012") or settings.get("ToneCurve")
    if isinstance(master, list):
        rgb = _monotone_curve(master, rgb)
    for channel, key in enumerate(("ToneCurvePV2012Red", "ToneCurvePV2012Green", "ToneCurvePV2012Blue")):
        curve = settings.get(key)
        if isinstance(curve, list):
            rgb[..., channel] = _monotone_curve(curve, rgb[..., channel])

    # Colour: HSL, vibrance and saturation in HSV space.
    hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_RGB2HSV).reshape(rgb.shape)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if any(f"{kind}Adjustment{c}" in settings for kind in ("Hue", "Saturation", "Luminance") for c in HSL_COLOURS):
        new_hue, sat_gain, lum_shift = hue.copy(), np.ones_like(sat), np.zeros_like(val)
        for colour, centre in zip(HSL_COLOURS, HSL_CENTRES):
            distance = np.abs((hue - centre + 180.0) % 360.0 - 180.0)
            width = 45.0 if colour in ("Green", "Aqua", "Blue") else 30.0
            weight = np.clip(1 - distance / width, 0, 1) * np.clip(sat * 3.0, 0, 1)
            new_hue = new_hue + weight * s(f"HueAdjustment{colour}") * 0.3
            sat_gain = sat_gain + weight * s(f"SaturationAdjustment{colour}") / 100.0
            lum_shift = lum_shift + weight * s(f"LuminanceAdjustment{colour}") / 100.0 * 0.25
        hue = new_hue % 360.0
        sat = sat * np.clip(sat_gain, 0.0, 2.0)
        val = np.clip(val * (1 + lum_shift), 0.0, 1.0)
    vibrance, saturation = s("Vibrance") / 100.0, s("Saturation") / 100.0
    if vibrance:
        sat = sat * (1 + vibrance * (1 - sat) * (0.6 if vibrance > 0 else 1.0))
    if saturation:
        sat = sat * (1 + saturation)
    hsv = np.stack([hue, np.clip(sat, 0.0, 1.0), val], axis=-1).astype(np.float32)
    rgb = cv2.cvtColor(hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2RGB).reshape(rgb.shape)

    # Split toning / colour grading: tint shadows, mid-tones and highlights.
    luma = _luma(rgb)
    balance = s("SplitToningBalance", s("ColorGradeBlending", 50.0) - 50.0) / 100.0
    pivot = 0.5 + balance * 0.25
    shadow_w = np.clip((pivot - luma) / pivot, 0, 1) ** 1.2
    high_w = np.clip((luma - pivot) / (1 - pivot), 0, 1) ** 1.2
    mid_w = np.clip(1 - np.abs(luma - 0.5) / 0.35, 0, 1)
    for hue_key, sat_key, alt_hue, alt_sat, weight in (
        ("SplitToningShadowHue", "SplitToningShadowSaturation", "ColorGradeShadowHue", "ColorGradeShadowSat", shadow_w),
        ("SplitToningHighlightHue", "SplitToningHighlightSaturation", "ColorGradeHighlightHue", "ColorGradeHighlightSat", high_w),
        (None, None, "ColorGradeMidtoneHue", "ColorGradeMidtoneSat", mid_w),
    ):
        amount = (s(sat_key) if sat_key else 0.0) or s(alt_sat)
        if amount <= 0:
            continue
        hue_value = (s(hue_key) if hue_key and sat_key and s(sat_key) else s(alt_hue))
        colour = _hue_colour(hue_value)
        offset = (colour - float(_luma(colour[None])[0])) * (amount / 100.0) * 0.28
        rgb = rgb + weight[..., None] * offset
    global_sat = s("ColorGradeGlobalSat")
    if global_sat:
        colour = _hue_colour(s("ColorGradeGlobalHue"))
        rgb = rgb + (colour - float(_luma(colour[None])[0])) * (global_sat / 100.0) * 0.15

    return np.clip(rgb, 0.0, 1.0).astype(np.float32), skipped


# --------------------------------------------------------------------------- applying

def apply_lut_array(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Trilinear lookup for float RGB values 0..1 of any shape (..., 3)."""
    size = lut.shape[0]
    scaled = np.clip(rgb, 0.0, 1.0) * (size - 1)
    base = np.minimum(np.floor(scaled).astype(np.int32), size - 2)
    frac = scaled - base
    r0, g0, b0 = base[..., 0], base[..., 1], base[..., 2]
    fr, fg, fb = frac[..., 0:1], frac[..., 1:2], frac[..., 2:3]
    c000 = lut[r0, g0, b0]
    c100 = lut[r0 + 1, g0, b0]
    c010 = lut[r0, g0 + 1, b0]
    c110 = lut[r0 + 1, g0 + 1, b0]
    c001 = lut[r0, g0, b0 + 1]
    c101 = lut[r0 + 1, g0, b0 + 1]
    c011 = lut[r0, g0 + 1, b0 + 1]
    c111 = lut[r0 + 1, g0 + 1, b0 + 1]
    c00 = c000 + (c100 - c000) * fr
    c10 = c010 + (c110 - c010) * fr
    c01 = c001 + (c101 - c001) * fr
    c11 = c011 + (c111 - c011) * fr
    c0 = c00 + (c10 - c00) * fg
    c1 = c01 + (c11 - c01) * fg
    return c0 + (c1 - c0) * fb


def apply(image: Image.Image, lut: np.ndarray, strength: float = 1.0,
          protect: np.ndarray | None = None, rows: int = 384) -> Image.Image:
    """Apply a LUT to a PIL image, in strips. ``protect`` (0..1 per pixel) softens it on skin."""
    strength = float(min(1.0, max(0.0, strength)))
    source = np.asarray(image.convert("RGB"))
    if strength <= 0.01:
        return image.convert("RGB")
    output = np.empty_like(source)
    height = source.shape[0]
    for top in range(0, height, rows):
        bottom = min(height, top + rows)
        strip = source[top:bottom].astype(np.float32) / 255.0
        mapped = apply_lut_array(strip, lut)
        amount = strength if protect is None else strength * (1 - 0.45 * protect[top:bottom, :, None])
        mixed = strip + (mapped - strip) * amount
        output[top:bottom] = np.clip(mixed * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(output, mode="RGB")


# --------------------------------------------------------------------------- storing

def pack(lut: np.ndarray) -> bytes:
    return zlib.compress(np.round(np.clip(lut, 0, 1) * 65535).astype("<u2").tobytes(), 6)


def unpack(blob: bytes) -> np.ndarray:
    values = np.frombuffer(zlib.decompress(blob), dtype="<u2").astype(np.float32) / 65535.0
    return values.reshape(LUT_SIZE, LUT_SIZE, LUT_SIZE, 3)


# --------------------------------------------------------------------------- describing

def describe(lut: np.ndarray) -> dict:
    """Plain words and swatches for the settings page."""
    greys = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    mapped = apply_lut_array(np.stack([greys] * 3, axis=-1), lut)
    luma = _luma(mapped)
    lab = cv2.cvtColor(mapped.reshape(1, -1, 3).astype(np.float32), cv2.COLOR_RGB2Lab)[0]
    warmth = float(np.mean(lab[3:9, 2]))
    tint = float(np.mean(lab[3:9, 1]))
    brightness = float(luma[5] - greys[5])
    contrast = float((luma[8] - luma[2]) - (greys[8] - greys[2]))
    chips = np.array([[0.80, 0.25, 0.20], [0.25, 0.55, 0.30], [0.25, 0.40, 0.75], [0.85, 0.62, 0.50]], dtype=np.float32)
    chip_in = cv2.cvtColor(chips.reshape(1, -1, 3), cv2.COLOR_RGB2HSV)[0, :, 1]
    chip_out = cv2.cvtColor(apply_lut_array(chips, lut).reshape(1, -1, 3).astype(np.float32), cv2.COLOR_RGB2HSV)[0, :, 1]
    saturation = float(np.mean(chip_out) / max(1e-3, np.mean(chip_in)))
    words = [
        "brighter" if brightness > 0.04 else "darker" if brightness < -0.04 else "even exposure",
        "punchy contrast" if contrast > 0.06 else "soft, low contrast" if contrast < -0.06 else "natural contrast",
        "warm" if warmth > 3 else "cool" if warmth < -3 else "neutral",
        "vivid colour" if saturation > 1.12 else "muted colour" if saturation < 0.85 else "natural colour",
    ]
    if mapped[0].max() > 0.06:
        words.append("matte blacks")
    if abs(tint) > 3 and abs(tint) > abs(warmth) * 0.6:
        words.append("magenta tint" if tint > 0 else "green tint")
    swatches = ["#%02x%02x%02x" % tuple(int(v) for v in np.clip(mapped[i] * 255, 0, 255)) for i in (2, 5, 8)]
    return {"words": words, "swatches": swatches, "warmth": round(warmth, 1), "brightness": round(brightness, 3),
            "contrast": round(contrast, 3), "saturation": round(saturation, 2)}


# --------------------------------------------------------------------------- choosing per photo

def classify(image: Image.Image, faces: Iterable[tuple[int, int, int, int]] = ()) -> dict:
    """Rough scene of a photo: night, indoor, outdoor; portrait when a face fills the frame."""
    small = np.asarray(style.downscale(image.convert("RGB"), SCORE_SIDE))
    lab = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    luminance = lab[..., 0]
    median = float(np.median(luminance))
    top = lab[: max(1, lab.shape[0] // 3)]
    sky = float(np.mean((top[..., 0] > 55) & (top[..., 2] < -4)))
    greenery = float(np.mean((lab[..., 1] < -8) & (lab[..., 0] > 20)))
    warm_cast = float(np.median(lab[..., 2]))
    face_area = 0.0
    width, height = image.size
    for _, _, w, h in faces:
        face_area = max(face_area, (w * h) / max(1, width * height))
    if median < 22:
        scene = "night"
    elif face_area > 0.04:
        scene = "portrait"
    elif sky > 0.12 or greenery > 0.15 or median > 58:
        scene = "outdoor"
    else:
        scene = "indoor" if warm_cast > 12 or median < 45 else "outdoor"
    return {"scene": scene, "median": median, "face_area": face_area}


def _stats(rgb: np.ndarray) -> dict:
    lab = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2Lab)
    luminance = lab[..., 0]
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    return {
        "median": float(np.median(luminance)),
        "high_clip": float(np.mean(luminance > 98.0)),
        "low_clip": float(np.mean(luminance < 1.5)),
        "chroma": float(np.median(chroma[(luminance > 15) & (luminance < 92)])) if np.any((luminance > 15) & (luminance < 92)) else 0.0,
        "lab": lab,
    }


def _skin_penalty(lab: np.ndarray, mask: np.ndarray | None) -> float:
    if mask is None or mask.sum() < 30:
        return 0.0
    a, b = lab[..., 1][mask], lab[..., 2][mask]
    hue = math.degrees(math.atan2(float(np.median(b)), float(np.median(a))))
    chroma = float(np.hypot(np.median(a), np.median(b)))
    penalty = 0.0
    # Natural skin sits around 40-65 degrees in Lab hue (orange-red); outside reads green, grey or magenta.
    if hue < 35:
        penalty += (35 - hue) / 6.0
    elif hue > 70:
        penalty += (hue - 70) / 6.0
    if chroma > 38:
        penalty += (chroma - 38) / 5.0
    elif chroma < 10:
        penalty += (10 - chroma) / 4.0
    return penalty


def choose(image: Image.Image, presets: Sequence[Preset], faces: Iterable[tuple[int, int, int, int]] = (),
           profile: dict | None = None, original: Image.Image | None = None) -> tuple[Preset | None, dict]:
    """Pick the preset that suits this photo. Returns (preset, explanation).

    ``original`` is the photo before automatic exposure correction; the scene (night, indoor...) is judged on it,
    because the correction brightens night shots.
    """
    if not presets:
        return None, {}
    faces = list(faces or [])
    if len(presets) == 1:
        return presets[0], {"scene": classify(original or image, faces)["scene"], "scores": {presets[0].preset_id: 0.0}}
    small_image = style.downscale(image.convert("RGB"), SCORE_SIDE)
    source = np.asarray(small_image).astype(np.float32) / 255.0
    ratio = small_image.width / max(1, image.width)
    scene = classify(original or image, faces)
    skin = None
    if faces:
        ycrcb = cv2.cvtColor(np.asarray(small_image), cv2.COLOR_RGB2YCrCb)
        candidate = cv2.inRange(ycrcb, (40, 135, 80), (255, 175, 130)) > 0
        near = np.zeros(candidate.shape, dtype=bool)
        for x, y, w, h in faces:
            x0, y0, x1, y1 = (int(v * ratio) for v in (x, y, x + w, y + h))
            near[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
        skin = candidate & near
    before = _stats(source)
    target_median = 54.0
    target_chroma = None
    if profile and profile.get("reference"):
        reference = profile["reference"]
        target_median = float(reference["l_pct"][style.PERCENTILES.index(50)])
        target_chroma = float(reference["chroma_pct"][1])
    elif scene["scene"] == "night":
        target_median = 38.0

    scores: dict[str, float] = {}
    for preset in presets:
        after = _stats(apply_lut_array(source, preset.lut))
        # Aim half-way between the photo's own exposure and the target, so presets aren't used to "fix" exposure.
        wanted = before["median"] + (target_median - before["median"]) * 0.5
        score = abs(after["median"] - wanted) / 8.0
        score += max(0.0, after["high_clip"] - before["high_clip"]) * 60.0
        score += max(0.0, after["low_clip"] - before["low_clip"]) * 45.0
        score += _skin_penalty(after["lab"], skin) * (1.6 if scene["scene"] == "portrait" else 1.0)
        if target_chroma:
            score += abs(after["chroma"] - target_chroma) / 6.0
        elif after["chroma"] > 42:
            score += (after["chroma"] - 42) / 5.0
        if preset.scene == scene["scene"]:
            score -= 2.0
        elif preset.scene != "any":
            score += 1.5
        scores[preset.preset_id] = round(score, 3)
    best = min(presets, key=lambda p: (scores[p.preset_id], p.name))
    return best, {"scene": scene["scene"], "scores": scores}
