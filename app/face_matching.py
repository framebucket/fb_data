from __future__ import annotations

import logging
import os
import threading
import urllib.request
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import Settings

logger = logging.getLogger(__name__)

DETECTOR_FILENAME = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILENAME = "face_recognition_sface_2021dec.onnx"
MAX_IMAGE_SIDE = 2400
MAX_IMAGE_PIXELS = 60_000_000


class EventFaceMatcher:
    """Self-hosted face embeddings using the OpenCV Zoo YuNet and SFace models."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._detector = None
        self._recognizer = None

    def selfie_embedding(self, image_bytes: bytes) -> list[float]:
        embeddings = self.face_embeddings(image_bytes)
        if len(embeddings) != 1:
            if not embeddings:
                raise ValueError("We could not find a face in that selfie. Try a clear, well-lit photo facing the camera.")
            raise ValueError("The selfie should contain only one face.")
        return embeddings[0]

    def face_embeddings(self, image_bytes: bytes) -> list[list[float]]:
        image = self._decode_image(image_bytes)
        return [self._embedding(image, face) for face in self._detect(image)]

    def matching_guest_ids(self, image_bytes: bytes, guests: dict[str, list[float]]) -> list[str]:
        return self.matching_guest_embeddings(self.face_embeddings(image_bytes), guests)

    def matching_guest_embeddings(
        self,
        face_embeddings: list[list[float]],
        guests: dict[str, list[float]],
    ) -> list[str]:
        if not guests or not face_embeddings:
            return []
        targets: dict[str, np.ndarray] = {}
        for guest_id, vector in guests.items():
            target = np.asarray(vector, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(target))
            if target.size and norm > 0:
                targets[guest_id] = target / norm

        matched: set[str] = set()
        threshold = self.settings.face_match_threshold
        for vector in face_embeddings:
            candidate = np.asarray(vector, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(candidate))
            if candidate.size == 0 or norm == 0:
                continue
            candidate /= norm
            for guest_id, target in targets.items():
                if candidate.shape == target.shape and float(np.dot(candidate, target)) >= threshold:
                    matched.add(guest_id)
        return sorted(matched)

    def face_boxes(self, image: Image.Image) -> list[tuple[int, int, int, int]]:
        """Face rectangles (x, y, w, h) in ``image`` pixels, tuned for recall rather than precision.

        Used to keep the studio logo and colour grading away from faces, so a missed face is worse than
        a false alarm: the threshold is lower than the one used for guest matching and small faces count.
        """
        detector = self._logo_detector()
        rgb = image.convert("RGB")
        scale = 1.0
        if max(rgb.size) > 1600:
            scale = 1600 / max(rgb.size)
            rgb = rgb.resize((max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
                             Image.Resampling.BILINEAR)
        frame = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        with self._lock:
            detector.setInputSize((frame.shape[1], frame.shape[0]))
            _, detections = detector.detect(frame)
        if detections is None:
            return []
        boxes = []
        for face in detections:
            x, y, w, h = (float(v) / scale for v in face[:4])
            if w >= 10 / scale and h >= 10 / scale:
                boxes.append((int(x), int(y), int(w), int(h)))
        return boxes

    def _logo_detector(self):
        with self._lock:
            if getattr(self, "_recall_detector", None) is not None:
                return self._recall_detector
            self.settings.face_model_dir.mkdir(parents=True, exist_ok=True)
            detector_path = self.settings.face_model_dir / DETECTOR_FILENAME
            self._ensure_model(detector_path, self.settings.face_detector_model_url, 5 * 1024 * 1024)
            self._recall_detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), 0.6, 0.3, 5000)
            return self._recall_detector

    def _models(self):
        with self._lock:
            if self._detector is not None and self._recognizer is not None:
                return self._detector, self._recognizer

            self.settings.face_model_dir.mkdir(parents=True, exist_ok=True)
            detector_path = self.settings.face_model_dir / DETECTOR_FILENAME
            recognizer_path = self.settings.face_model_dir / RECOGNIZER_FILENAME
            self._ensure_model(detector_path, self.settings.face_detector_model_url, 5 * 1024 * 1024)
            self._ensure_model(recognizer_path, self.settings.face_recognizer_model_url, 75 * 1024 * 1024)

            detector = cv2.FaceDetectorYN.create(
                str(detector_path), "", (320, 320), 0.82, 0.3, 5000
            )
            recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
            self._detector = detector
            self._recognizer = recognizer
            logger.info("Self-hosted event face matching models are ready.")
            return detector, recognizer

    @staticmethod
    def _ensure_model(destination: Path, url: str, maximum_bytes: int) -> None:
        if destination.is_file() and destination.stat().st_size > 100_000:
            return
        if not url:
            raise RuntimeError("Set the face-matching model URL or mount the model file in FB_FACE_MODEL_DIR.")
        temporary = destination.with_suffix(destination.suffix + ".download")
        request = urllib.request.Request(url, headers={"User-Agent": "FrameBucket-face-matcher/1.0"})
        total = 0
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > maximum_bytes:
                        raise RuntimeError("The configured face model exceeds the allowed download size.")
                    output.write(chunk)
            if total < 100_000:
                raise RuntimeError("The face model download was incomplete. Check the configured model URL.")
            with temporary.open("rb") as model_file:
                prefix = model_file.read(200)
            if b"git-lfs.github.com/spec" in prefix or b"version https://git-lfs" in prefix:
                raise RuntimeError("The model host returned a Git LFS pointer instead of the ONNX model.")
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise ValueError("The uploaded selfie is empty.")
        try:
            with Image.open(BytesIO(image_bytes)) as source:
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ValueError("The image is too large to process. Use a photo under 60 megapixels.")
                image = ImageOps.exif_transpose(source).convert("RGB")
                if max(image.size) > MAX_IMAGE_SIDE:
                    ratio = MAX_IMAGE_SIDE / max(image.size)
                    image = image.resize(
                        (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
                        Image.Resampling.LANCZOS,
                    )
                return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Upload a valid JPG or PNG image.") from exc

    def _detect(self, image: np.ndarray) -> list[np.ndarray]:
        detector, _ = self._models()
        with self._lock:
            detector.setInputSize((image.shape[1], image.shape[0]))
            _, detections = detector.detect(image)
        if detections is None:
            return []
        return [face for face in detections if face[2] >= 32 and face[3] >= 32]

    def _embedding(self, image: np.ndarray, face: np.ndarray) -> list[float]:
        _, recognizer = self._models()
        with self._lock:
            aligned = recognizer.alignCrop(image, face)
            feature = recognizer.feature(aligned).reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(feature))
        if not norm:
            raise ValueError("The selfie could not be analyzed. Try another clear, front-facing photo.")
        return (feature / norm).astype(float).tolist()
