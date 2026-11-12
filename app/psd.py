"""Minimal layered Photoshop (PSD) writer: 8-bit RGB document, RGBA pixel layers, guides and print DPI.

Only what album design needs, written from the Adobe PSD file-format specification. Channel data is
PackBits (RLE) compressed, one scanline at a time; the scanlines are produced by libtiff through Pillow
(TIFF PackBits with one row per strip is byte-for-byte the PSD scanline encoding), so large spreads
encode quickly without a pure-Python compressor.
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from typing import BinaryIO, Sequence

import numpy as np
from PIL import Image


@dataclass
class Layer:
    name: str
    image: Image.Image                  # RGBA (RGB is promoted to opaque RGBA)
    left: int = 0
    top: int = 0
    opacity: int = 255
    hidden: bool = False


@dataclass
class Guide:
    position: int                       # pixels
    vertical: bool                      # True = vertical line at x=position


@dataclass
class Document:
    width: int
    height: int
    dpi: int = 300
    layers: list[Layer] = field(default_factory=list)
    guides: list[Guide] = field(default_factory=list)


def _packbits_rows(channel: np.ndarray) -> tuple[list[int], bytes]:
    """PackBits-encode each row of a 2-D uint8 array. Returns (row byte counts, concatenated rows)."""
    height = channel.shape[0]
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(channel), "L").save(
        buffer, "TIFF", compression="packbits", tiffinfo={278: 1}
    )
    raw = buffer.getvalue()
    with Image.open(io.BytesIO(raw)) as tiff:
        offsets = list(tiff.tag_v2[273])
        counts = list(tiff.tag_v2[279])
    if len(offsets) != height:
        raise RuntimeError("Unexpected TIFF strip layout while packing PSD channel data.")
    data = b"".join(raw[offset:offset + count] for offset, count in zip(offsets, counts))
    return counts, data


def _channel_block(channel: np.ndarray) -> bytes:
    """One layer channel: compression flag + scanline byte counts + packed scanlines."""
    if channel.size == 0:
        return struct.pack(">H", 0)
    counts, data = _packbits_rows(channel)
    if max(counts) > 0xFFFF:
        raise RuntimeError("Scanline too long for PSD RLE.")
    return struct.pack(">H", 1) + struct.pack(f">{len(counts)}H", *counts) + data


def _pascal(name: str, pad_to: int) -> bytes:
    encoded = name.encode("ascii", "replace")[:255]
    raw = bytes([len(encoded)]) + encoded
    while len(raw) % pad_to:
        raw += b"\0"
    return raw


def _unicode_name_block(name: str) -> bytes:
    """'luni' additional layer info so Photoshop shows the full (non-ASCII) layer name."""
    text = name[:255]
    payload = struct.pack(">I", len(text)) + text.encode("utf-16-be")
    if len(payload) % 2:
        payload += b"\0"
    return b"8BIMluni" + struct.pack(">I", len(payload)) + payload


def _resources(document: Document) -> bytes:
    blocks = []

    def resource(resource_id: int, data: bytes) -> bytes:
        if len(data) % 2:
            data += b"\0"
        return b"8BIM" + struct.pack(">H", resource_id) + b"\0\0" + struct.pack(">I", len(data)) + data

    dpi_fixed = int(document.dpi) << 16
    # ResolutionInfo: hRes, hResUnit(1=ppi), widthUnit(1=inch), vRes, vResUnit, heightUnit
    blocks.append(resource(0x03ED, struct.pack(">IHHIHH", dpi_fixed, 1, 1, dpi_fixed, 1, 1)))
    if document.guides:
        data = struct.pack(">I", 1) + struct.pack(">II", 576, 576) + struct.pack(">I", len(document.guides))
        for guide in document.guides:
            data += struct.pack(">IB", int(guide.position) * 32, 0 if guide.vertical else 1)
        blocks.append(resource(0x0408, data))
    body = b"".join(blocks)
    return struct.pack(">I", len(body)) + body


def _layer_section(document: Document) -> bytes:
    records = io.BytesIO()
    pixel_data = io.BytesIO()
    for layer in document.layers:
        image = layer.image if layer.image.mode == "RGBA" else layer.image.convert("RGBA")
        width, height = image.size
        pixels = np.asarray(image)
        channels = [(-1, pixels[..., 3]), (0, pixels[..., 0]), (1, pixels[..., 1]), (2, pixels[..., 2])]
        blocks = [(channel_id, _channel_block(data)) for channel_id, data in channels]

        records.write(struct.pack(">iiii", layer.top, layer.left, layer.top + height, layer.left + width))
        records.write(struct.pack(">H", len(blocks)))
        for channel_id, block in blocks:
            records.write(struct.pack(">hI", channel_id, len(block)))
        flags = 0x02 if layer.hidden else 0x00
        records.write(b"8BIMnorm" + struct.pack(">BBBB", max(0, min(255, layer.opacity)), 0, flags, 0))
        extra = struct.pack(">I", 0) + struct.pack(">I", 0) + _pascal(layer.name, 4) + _unicode_name_block(layer.name)
        records.write(struct.pack(">I", len(extra)) + extra)
        for _, block in blocks:
            pixel_data.write(block)

    info = struct.pack(">h", len(document.layers)) + records.getvalue() + pixel_data.getvalue()
    if len(info) % 2:
        info += b"\0"
    layer_info = struct.pack(">I", len(info)) + info
    global_mask = struct.pack(">I", 0)
    section = layer_info + global_mask
    return struct.pack(">I", len(section)) + section


def _composite(document: Document) -> Image.Image:
    canvas = Image.new("RGBA", (document.width, document.height), (255, 255, 255, 255))
    for layer in document.layers:
        if layer.hidden:
            continue
        image = layer.image if layer.image.mode == "RGBA" else layer.image.convert("RGBA")
        if layer.opacity < 255:
            alpha = image.getchannel("A").point(lambda value: value * layer.opacity // 255)
            image = image.copy()
            image.putalpha(alpha)
        canvas.alpha_composite(image, (layer.left, layer.top))
    return canvas.convert("RGB")


def write(document: Document, output: BinaryIO, composite: Image.Image | None = None) -> None:
    """Write the document. ``composite`` (the flattened RGB render) is computed if not supplied."""
    if not 1 <= document.width <= 30000 or not 1 <= document.height <= 30000:
        raise ValueError("PSD documents are limited to 30,000 pixels per side.")
    output.write(b"8BPS" + struct.pack(">H", 1) + b"\0" * 6)
    output.write(struct.pack(">HIIHH", 3, document.height, document.width, 8, 3))
    output.write(struct.pack(">I", 0))                       # colour mode data
    output.write(_resources(document))
    output.write(_layer_section(document))

    merged = composite if composite is not None else _composite(document)
    if merged.size != (document.width, document.height):
        raise ValueError("Composite size does not match the document.")
    pixels = np.asarray(merged.convert("RGB"))
    all_counts: list[int] = []
    all_data: list[bytes] = []
    for index in range(3):
        counts, data = _packbits_rows(pixels[..., index])
        all_counts.extend(counts)
        all_data.append(data)
    output.write(struct.pack(">H", 1))
    output.write(struct.pack(f">{len(all_counts)}H", *all_counts))
    for data in all_data:
        output.write(data)


def save(document: Document, path, composite: Image.Image | None = None) -> None:
    with open(path, "wb") as handle:
        write(document, handle, composite)


def layer_from(image: Image.Image, name: str, left: int, top: int, opacity: int = 255) -> Layer:
    return Layer(name=name, image=image.convert("RGBA"), left=left, top=top, opacity=opacity)


__all__: Sequence[str] = ("Document", "Guide", "Layer", "layer_from", "save", "write")
