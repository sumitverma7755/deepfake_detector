"""Preprocessing utilities for image and video deepfake analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
_FACE_DETECTOR_UNSET = object()
_FACE_DETECTOR: cv2.CascadeClassifier | None | object = _FACE_DETECTOR_UNSET


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image as an RGB numpy array."""
    image_path = str(image_path)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Unable to load image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _get_opencv_face_detector() -> cv2.CascadeClassifier | None:
    """Return Haar-cascade detector if available."""
    global _FACE_DETECTOR
    if _FACE_DETECTOR is not _FACE_DETECTOR_UNSET:
        return _FACE_DETECTOR  # type: ignore[return-value]

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        LOGGER.warning("OpenCV Haar cascade not found at %s", cascade_path)
        _FACE_DETECTOR = None
        return None

    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        LOGGER.warning("OpenCV face detector could not be initialized.")
        _FACE_DETECTOR = None
        return None
    _FACE_DETECTOR = detector
    return detector


def extract_faces(
    image: np.ndarray,
    expand_ratio: float = 0.15,
    detection_max_side: int = 960,
) -> List[np.ndarray]:
    """Extract face crops from an RGB image using OpenCV Haar cascades."""
    detector = _get_opencv_face_detector()
    if detector is None:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detect_gray = gray
    scale = 1.0
    image_h, image_w = gray.shape[:2]
    max_side = max(image_h, image_w)
    if max_side > detection_max_side > 0:
        scale = detection_max_side / float(max_side)
        detect_gray = cv2.resize(
            gray,
            (max(1, int(image_w * scale)), max(1, int(image_h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    detections = detector.detectMultiScale(detect_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    faces: List[np.ndarray] = []
    for (x, y, w, h) in detections:
        if scale != 1.0:
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)

        dx = int(w * expand_ratio)
        dy = int(h * expand_ratio)
        x1 = max(0, x - dx)
        y1 = max(0, y - dy)
        x2 = min(image_w, x + w + dx)
        y2 = min(image_h, y + h + dy)

        face_crop = image[y1:y2, x1:x2]
        if face_crop.size > 0:
            faces.append(face_crop)

    return faces


def build_uniform_sample_indices(total_frames: int, max_frames: int) -> np.ndarray:
    """Build evenly spaced frame indices without duplicates."""
    if total_frames <= 0 or max_frames <= 0:
        return np.empty((0,), dtype=np.int32)

    sample_count = min(total_frames, max_frames)
    if sample_count == 1:
        return np.asarray([0], dtype=np.int32)

    max_index = total_frames - 1
    denominator = sample_count - 1
    indices = [(i * max_index) // denominator for i in range(sample_count)]
    return np.asarray(indices, dtype=np.int32)


def extract_frames(video_path: str | Path, max_frames: int = 100) -> List[np.ndarray]:
    """Extract up to `max_frames` RGB frames from a video file."""
    video_path = str(video_path)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: List[np.ndarray] = []

    if frame_count > 0:
        indices = build_uniform_sample_indices(frame_count, max_frames)
        for idx in indices:
            current = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            skip = max(0, int(idx) - current)
            for _ in range(skip):
                if not capture.grab():
                    break
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                continue
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    else:
        while len(frames) < max_frames:
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    capture.release()
    return frames


def preprocess_frames(frames: List[np.ndarray], target_size=(224, 224)) -> List[np.ndarray]:
    """Resize and normalize frames for model inference."""
    processed_frames: List[np.ndarray] = []
    for frame in frames:
        resized = cv2.resize(frame, target_size)
        normalized = resized.astype(np.float32) / 255.0
        processed_frames.append(normalized)
    return processed_frames


def extract_frequency_features(image: np.ndarray) -> np.ndarray:
    """Extract a normalized 3-channel FFT magnitude representation."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1.0)

    magnitude_norm = cv2.normalize(magnitude, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    return np.stack([magnitude_norm, magnitude_norm, magnitude_norm], axis=-1).astype(np.float32)
