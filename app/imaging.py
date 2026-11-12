from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat

from . import style

logger = logging.getLogger(__name__)

RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".dng", ".nef", ".orf", ".raf", ".rw2"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".avi"}

IMAGE_EXTENSIONS = RAW_EXTENSIONS | {
    ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"
}


FaceBoxes = list[tuple[int, int, int, int]]


@dataclass(frozen=True)
class StudioLook:
    """Per-studio processing options, loaded by the service for each photo."""

    profile: dict | None = None          # learned look (style.build_profile); None = basic corrections only
    baseline: dict | None = None         # studio's rolling "before styling" statistics
    strength: float = 0.8                # 0..1
    logo: bytes | None = None            # studio logo; None falls back to the operator-wide FB_LOGO_PATH
    stamp_logo: bool = True


class OnnxImageEnhancer:
    """Optional model adapter.

    The configured ONNX model must accept RGB float32 NCHW in [0, 1] and return
    RGB float32 NCHW in [0, 1]. Spatial dimensions must be dynamic or match each image.
    """

    def __init__(self, model_path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "FB_ENHANCER_MODEL is set, but onnxruntime is not installed. "
                "Install the 'ai' extra."
            ) from exc

        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("The local enhancer model must have exactly one input.")
        self._input = inputs[0]
        shape = self._input.shape
        if len(shape) != 4 or shape[1] not in (3, "3", None):
            raise ValueError("The local enhancer model input must use NCHW RGB layout.")

    def enhance(self, image: Image.Image) -> Image.Image:
        import numpy as np

        input_shape = self._input.shape
        height = input_shape[2] if isinstance(input_shape[2], int) else image.height
        width = input_shape[3] if isinstance(input_shape[3], int) else image.width
        model_input = image
        if (width, height) != image.size:
            raise ValueError("The ONNX model must support dynamic image dimensions or match this image size.")

        tensor = np.asarray(model_input, dtype=np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        output = self._session.run(None, {self._input.name: tensor})[0]
        if output.ndim != 4 or output.shape[0] != 1 or output.shape[1] != 3:
            raise ValueError("The local enhancer model must return one NCHW RGB image.")

        pixels = np.clip(output[0].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        enhanced = Image.fromarray(pixels, mode="RGB")
        if enhanced.width < image.width or enhanced.height < image.height:
            enhanced = enhanced.resize(image.size, Image.Resampling.LANCZOS)
        return enhanced


class FaceDetector:
    """Conservative local face detector used to keep the logo away from faces."""

    def __init__(self) -> None:
        self._cv2 = None
        self._cascade = None
        self._numpy = None
        try:
            import cv2
            import numpy

            cascade_path = (
                Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            )
            cascade = cv2.CascadeClassifier(str(cascade_path))
            if not cascade.empty():
                self._cv2 = cv2
                self._cascade = cascade
                self._numpy = numpy
        except ImportError:
            logger.info("OpenCV unavailable; logo placement will be skipped.")

    @property
    def ready(self) -> bool:
        return self._cv2 is not None and self._cascade is not None and self._numpy is not None

    def boxes(self, image: Image.Image) -> list[tuple[int, int, int, int]]:
        if not self.ready:
            return []
        cv2 = self._cv2
        rgb = self._numpy.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        boxes = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(max(24, image.width // 35), max(24, image.height // 35)),
        )
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in boxes]


class ImageProcessor:
    def __init__(
        self,
        *,
        logo_path: Path | None = None,
        enhancer_model: Path | None = None,
        max_processed_bytes: int | None = None,
        face_finder: Callable[[Image.Image], FaceBoxes] | None = None,
    ) -> None:
        self.logo_path = logo_path if logo_path and logo_path.is_file() else None
        self.max_processed_bytes = max_processed_bytes
        self.enhancer = OnnxImageEnhancer(enhancer_model) if enhancer_model else None
        self.face_detector = FaceDetector()
        self.face_finder = face_finder

    def process(self, source: Path, destination: Path, look: StudioLook | None = None) -> dict:
        image = self._load(source)
        source_size = image.size
        image, info = self.render(image, look or StudioLook())

        destination.parent.mkdir(parents=True, exist_ok=True)
        target_bytes = max(1, source.stat().st_size - 1)
        if self.max_processed_bytes is not None:
            target_bytes = min(target_bytes, self.max_processed_bytes)
        encoded = self._encode_jpeg(image, target_bytes)
        destination_tmp = destination.with_name(destination.name + ".tmp")
        try:
            destination_tmp.write_bytes(encoded)
            os.replace(destination_tmp, destination)
        finally:
            destination_tmp.unlink(missing_ok=True)

        return {
            "width": image.width,
            "height": image.height,
            "original_width": source_size[0],
            "original_height": source_size[1],
            "bytes": len(encoded),
            "watermarked": info["watermarked"],
            "ai_enhanced": self.enhancer is not None or info["styled"],
            "styled": info["styled"],
            "analysis": info["analysis"],
        }

    def render(self, image: Image.Image, look: StudioLook) -> tuple[Image.Image, dict]:
        """Basic corrections, then the studio look, then sharpening and the face-safe logo."""
        if self.enhancer:
            image = self.enhancer.enhance(image)
        image = self._balance(image)
        analysis = style.analyze(image)
        faces: FaceBoxes | None = None

        styled = False
        if look.profile and look.strength > 0:
            faces = self._faces(image)
            image = style.apply_style(image, look.profile, look.baseline, look.strength, faces or [])
            styled = True
        image = self._sharpen(image)

        watermarked = False
        logo = self._logo_image(look)
        if logo is not None:
            if faces is None:
                faces = self._faces(image)
            watermarked = self._apply_logo(image, logo, faces)
        return image, {"styled": styled, "watermarked": watermarked, "analysis": analysis}

    def _faces(self, image: Image.Image) -> FaceBoxes | None:
        """Faces from the YuNet model plus the Haar cascade (union, for recall); None if neither works."""
        boxes: FaceBoxes = []
        found_any_detector = False
        if self.face_finder is not None:
            try:
                boxes.extend(self.face_finder(image))
                found_any_detector = True
            except Exception:  # model download or runtime failure: fall back to the cascade
                logger.warning("Face model unavailable for logo placement; using the fallback detector.",
                               exc_info=True)
        if self.face_detector.ready:
            small = style.downscale(image, 1600)
            ratio = image.width / small.width
            boxes.extend((int(x * ratio), int(y * ratio), int(w * ratio), int(h * ratio))
                         for x, y, w, h in self.face_detector.boxes(small))
            found_any_detector = True
        return boxes if found_any_detector else None

    def _logo_image(self, look: StudioLook) -> Image.Image | None:
        if not look.stamp_logo:
            return None
        try:
            if look.logo:
                with Image.open(io.BytesIO(look.logo)) as opened:
                    return opened.convert("RGBA")
            if self.logo_path:
                with Image.open(self.logo_path) as opened:
                    return opened.convert("RGBA")
        except (OSError, Image.DecompressionBombError):
            logger.warning("The studio logo could not be read; skipping it.")
        return None

    @staticmethod
    def _load(source: Path) -> Image.Image:
        if source.suffix.lower() in RAW_EXTENSIONS:
            try:
                import rawpy
                import numpy as np
            except ImportError as exc:
                raise RuntimeError(
                    f"RAW support for {source.suffix} requires the 'raw' extra."
                ) from exc
            with rawpy.imread(str(source)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=False,
                    output_bps=8,
                )
            return Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")

        with Image.open(source) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")

    @staticmethod
    def _balance(image: Image.Image) -> Image.Image:
        # Gentle exposure/color/contrast corrections keep the original look.
        image = ImageOps.autocontrast(image, cutoff=0.35)
        luminance = ImageOps.grayscale(image).resize((1, 1)).getpixel((0, 0))
        brightness_factor = 1.08 if luminance < 92 else 0.98 if luminance > 185 else 1.02
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        image = ImageEnhance.Color(image).enhance(1.045)
        return ImageEnhance.Contrast(image).enhance(1.035)

    @staticmethod
    def _sharpen(image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=4))

    @classmethod
    def _balance_and_detail(cls, image: Image.Image) -> Image.Image:
        return cls._sharpen(cls._balance(image))

    # Preferred corners first; the centre-bottom/top spots are used only when every corner has a face.
    _LOGO_SPOTS = ("bottom-right", "bottom-left", "top-right", "top-left", "bottom-center", "top-center")

    def _apply_logo(self, image: Image.Image, logo: Image.Image, faces: FaceBoxes | None) -> bool:
        """Stamp the logo where it cannot cover a face. If faces can't be detected, skip it."""
        if faces is None:
            logger.warning("Skipping logo: face detection is unavailable.")
            return False

        # Keep-out zones: each face plus hair/headpiece above, chin and neck below.
        zones = [(fx - fw * 0.45, fy - fh * 0.6, fx + fw * 1.45, fy + fh * 1.5) for fx, fy, fw, fh in faces]
        margin = max(12, int(min(image.size) * 0.025))
        gray = ImageOps.grayscale(image)
        logo_level = self._logo_level(logo)

        for scale in (1.0, 0.8, 0.65):
            mark = logo.copy()
            mark.thumbnail((max(1, int(image.width * 0.14 * scale)), max(1, int(image.height * 0.06 * scale))),
                           Image.Resampling.LANCZOS)
            spots = {
                "bottom-right": (image.width - mark.width - margin, image.height - mark.height - margin),
                "bottom-left": (margin, image.height - mark.height - margin),
                "top-right": (image.width - mark.width - margin, margin),
                "top-left": (margin, margin),
                "bottom-center": ((image.width - mark.width) // 2, image.height - mark.height - margin),
                "top-center": ((image.width - mark.width) // 2, margin),
            }
            candidates: list[tuple[float, tuple[int, int]]] = []
            for preference, name in enumerate(self._LOGO_SPOTS):
                x, y = spots[name]
                pad = margin // 2
                rect = (x - pad, y - pad, x + mark.width + pad, y + mark.height + pad)
                if any(min(rect[2], z[2]) > max(rect[0], z[0]) and min(rect[3], z[3]) > max(rect[1], z[1])
                       for z in zones):
                    continue
                patch = gray.crop(rect)
                busyness = image_stat_variance(patch.resize((16, 16))) ** 0.5
                # Prefer calm areas where the logo stands out, then the preferred corner.
                contrast = abs(logo_level - ImageStat.Stat(patch).mean[0])
                legibility_penalty = max(0.0, 90.0 - contrast) * 0.5
                candidates.append((busyness + legibility_penalty + preference * 6.0, (x, y)))
            if candidates:
                _, position = min(candidates, key=lambda candidate: candidate[0])
                self._paste_logo(image, mark, position)
                return True

        logger.info("Skipping logo because every candidate area is too close to a face.")
        return False

    @staticmethod
    def _logo_level(mark: Image.Image) -> float:
        solid = mark.getchannel("A").point(lambda value: 255 if value > 128 else 0)
        gray = ImageOps.grayscale(mark.convert("RGB"))
        return (ImageStat.Stat(gray, solid) if solid.getbbox() else ImageStat.Stat(gray)).mean[0]

    @staticmethod
    def _paste_logo(image: Image.Image, mark: Image.Image, position: tuple[int, int]) -> None:
        alpha = mark.getchannel("A")
        box = (position[0], position[1], position[0] + mark.width, position[1] + mark.height)
        backdrop_level = ImageStat.Stat(ImageOps.grayscale(image.crop(box))).mean[0]
        logo_level = ImageProcessor._logo_level(mark)
        if abs(logo_level - backdrop_level) < 90:
            # Low contrast: a soft halo keeps the logo readable without drawing a hard box.
            halo_colour = (0, 0, 0) if logo_level > 127 else (255, 255, 255)
            radius = max(2, mark.height // 10)
            halo_alpha = alpha.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(radius))
            halo = Image.new("RGBA", mark.size, halo_colour + (0,))
            halo.putalpha(halo_alpha.point(lambda value: round(value * 0.55)))
            image.paste(halo, position, halo)
        faded = mark.copy()
        faded.putalpha(alpha.point(lambda value: round(value * 0.72)))
        image.paste(faded, position, faded)

    def _encode_jpeg(self, image: Image.Image, target_bytes: int) -> bytes:
        quality = 94
        encoded = self._jpeg_bytes(image, quality)
        if len(encoded) <= target_bytes:
            return encoded

        lower, upper = 78, 93
        best: bytes | None = None
        while lower <= upper:
            quality = (lower + upper) // 2
            candidate = self._jpeg_bytes(image, quality)
            if len(candidate) <= target_bytes:
                best = candidate
                lower = quality + 1
            else:
                upper = quality - 1

        # If the requested byte target cannot be met, preserve the dimensions
        # and return the highest available fallback quality.
        return best if best is not None else self._jpeg_bytes(image, 78)

    @staticmethod
    def _jpeg_bytes(image: Image.Image, quality: int) -> bytes:
        stream = io.BytesIO()
        image.save(
            stream,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
            subsampling=0,
        )
        return stream.getvalue()


def image_stat_variance(image: Image.Image) -> float:
    histogram = image.histogram()
    count = max(1, sum(histogram))
    mean = sum(index * value for index, value in enumerate(histogram)) / count
    return sum(((index - mean) ** 2) * value for index, value in enumerate(histogram)) / count
