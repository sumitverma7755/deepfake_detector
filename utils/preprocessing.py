"""Preprocessing utilities for image and video deepfake analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image as an RGB numpy array."""
    image_path = str(image_path)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Unable to load image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _get_opencv_face_detector() -> cv2.CascadeClassifier | None:
    """Return Haar-cascade detector if available."""
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        LOGGER.warning("OpenCV Haar cascade not found at %s", cascade_path)
        return None

    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        LOGGER.warning("OpenCV face detector could not be initialized.")
        return None
    return detector


def extract_faces(image: np.ndarray, expand_ratio: float = 0.15) -> List[np.ndarray]:
    """Extract face crops from an RGB image using OpenCV Haar cascades."""
    detector = _get_opencv_face_detector()
    if detector is None:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detections = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    faces: List[np.ndarray] = []
    image_h, image_w = image.shape[:2]
    for (x, y, w, h) in detections:
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


def extract_frames(video_path: str | Path, max_frames: int = 100) -> List[np.ndarray]:
    """Extract up to `max_frames` RGB frames from a video file."""
    video_path = str(video_path)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: List[np.ndarray] = []

    if frame_count > 0:
        sample_count = min(max_frames, frame_count)
        indices = np.linspace(0, frame_count - 1, sample_count, dtype=int)
        for idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
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
