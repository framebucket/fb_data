from __future__ import annotations

import io
import logging
import os
from pathlib import Path


from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

RAW_EXTENSIONS = {".arw", ".cr2", ".cr3", ".dng", ".nef", ".orf", ".raf", ".rw2"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".avi"}

IMAGE_EXTENSIONS = RAW_EXTENSIONS | {
    ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"
}


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
    ) -> None:
        self.logo_path = logo_path if logo_path and logo_path.is_file() else None
        self.max_processed_bytes = max_processed_bytes
        self.enhancer = OnnxImageEnhancer(enhancer_model) if enhancer_model else None
        self.face_detector = FaceDetector()

    def process(self, source: Path, destination: Path) -> dict[str, int | bool]:
        image = self._load(source)
        source_size = image.size

        if self.enhancer:
            image = self.enhancer.enhance(image)

        image = self._balance_and_detail(image)
        watermarked = self._apply_logo(image) if self.logo_path else False

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
            "watermarked": watermarked,
            "ai_enhanced": self.enhancer is not None,
        }

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
    def _balance_and_detail(image: Image.Image) -> Image.Image:
        # Gentle exposure/color/contrast corrections keep the original look.
        image = ImageOps.autocontrast(image, cutoff=0.35)
        luminance = ImageOps.grayscale(image).resize((1, 1)).getpixel((0, 0))
        brightness_factor = 1.08 if luminance < 92 else 0.98 if luminance > 185 else 1.02
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        image = ImageEnhance.Color(image).enhance(1.045)
        image = ImageEnhance.Contrast(image).enhance(1.035)
        return image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=4))

    def _apply_logo(self, image: Image.Image) -> bool:
        # If face detection is unavailable, omit the logo instead of guessing a
        # position that could cover a face.
        if not self.face_detector.ready or not self.logo_path:
            logger.warning("Skipping logo: face detection is unavailable.")
            return False

        with Image.open(self.logo_path) as opened_logo:
            logo = opened_logo.convert("RGBA")

        max_width = max(1, int(image.width * 0.14))
        max_height = max(1, int(image.height * 0.06))
        logo.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        alpha = logo.getchannel("A").point(lambda value: round(value * 0.62))
        logo.putalpha(alpha)

        margin = max(12, int(min(image.size) * 0.025))
        positions = [
            (margin, margin),
            (image.width - logo.width - margin, margin),
            (margin, image.height - logo.height - margin),
            (image.width - logo.width - margin, image.height - logo.height - margin),
            ((image.width - logo.width) // 2, image.height - logo.height - margin),
        ]
        faces = self.face_detector.boxes(image)
        gray = ImageOps.grayscale(image)
        candidates: list[tuple[float, tuple[int, int]]] = []

        for x, y in positions:
            rect = (x, y, x + logo.width, y + logo.height)
            overlaps_face = False
            for fx, fy, fw, fh in faces:
                pad_x, pad_y = int(fw * 0.18), int(fh * 0.18)
                face = (fx - pad_x, fy - pad_y, fx + fw + pad_x, fy + fh + pad_y)
                intersection = max(0, min(rect[2], face[2]) - max(rect[0], face[0])) * max(
                    0, min(rect[3], face[3]) - max(rect[1], face[1])
                )
                if intersection:
                    overlaps_face = True
                    break
            if overlaps_face:
                continue
            patch = gray.crop(rect).resize((8, 8))
            candidates.append((image_stat_variance(patch), (x, y)))

        if not candidates:
            logger.info("Skipping logo because every candidate area overlaps a face.")
            return False

        _, position = min(candidates, key=lambda candidate: candidate[0])
        image.paste(logo, position, logo)
        return True

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
