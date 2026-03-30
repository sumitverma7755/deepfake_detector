"""Inference service with TensorFlow fallback and DirectML GPU acceleration on Windows."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf

from config.settings import ENABLE_DIRECTML, MODELS_DIR
from core.types import DetectionResult
from utils.preprocessing import extract_faces, extract_frequency_features

try:  # Optional GPU backend on native Windows
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

try:  # Optional conversion bridge from Keras -> ONNX
    import tf2onnx
except Exception:  # pragma: no cover - optional dependency
    tf2onnx = None

LOGGER = logging.getLogger(__name__)


class InferenceService:
    """Model runtime that loads once and serves image/video predictions."""

    def __init__(self, model_dir: Path | str = MODELS_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.model: tf.keras.Model | None = None
        self.model_source_path: Path | None = None

        self.expected_height = 224
        self.expected_width = 224
        self.expected_channels = 3
        self.expected_flat_dim: int | None = None

        self.onnx_session: Any | None = None
        self.onnx_input_name: str | None = None

        self.runtime_backend = "tensorflow"
        self.runtime_provider = "CPUExecutionProvider"

        self._load_model_once()
        self._try_enable_directml_runtime()

    @property
    def runtime_description(self) -> str:
        return f"{self.runtime_backend} ({self.runtime_provider})"

    def _load_model_once(self) -> None:
        candidates = [
            self.model_dir / "efficientnet_deepfake_detector.h5",
            self.model_dir / "resnet_face_detector.h5",
            self.model_dir / "frequency_detector.h5",
            self.model_dir / "model_checkpoint.h5",
            self.model_dir / "gpu_trained_model.h5",
        ]

        for path in candidates:
            if not path.exists():
                continue
            try:
                self.model = tf.keras.models.load_model(path, compile=False)
                self.model_source_path = path
                self._configure_expected_input_shape(self.model)
                LOGGER.info("Loaded inference model: %s", path)
                break
            except Exception as exc:  # pragma: no cover - runtime backend dependent
                LOGGER.warning("Failed to load model %s: %s", path, exc)

        if self.model is None:
            LOGGER.warning("No compatible model found. Building fallback binary model.")
            self.model = self._build_fallback_model()
            self.model_source_path = None
            self._configure_expected_input_shape(self.model)

        if tf.config.list_physical_devices("GPU"):
            self.runtime_provider = "CUDAExecutionProvider"

    def _configure_expected_input_shape(self, model: tf.keras.Model) -> None:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]

        if not isinstance(shape, tuple):
            self.expected_height = 224
            self.expected_width = 224
            self.expected_channels = 3
            self.expected_flat_dim = None
            return

        if len(shape) == 4:
            _, h, w, c = shape
            self.expected_height = int(h or 224)
            self.expected_width = int(w or 224)
            self.expected_channels = int(c or 3)
            self.expected_flat_dim = None
            return

        if len(shape) == 2:
            _, flat_dim = shape
            self.expected_flat_dim = int(flat_dim or (224 * 224 * 3))
            self.expected_height = 224
            self.expected_width = 224
            self.expected_channels = 3
            return

        self.expected_height = 224
        self.expected_width = 224
        self.expected_channels = 3
        self.expected_flat_dim = None

    @staticmethod
    def _build_fallback_model() -> tf.keras.Model:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.Conv2D(16, 3, activation="relu"),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        return model

    def _keras_input_signature(self) -> tuple[tf.TensorSpec, ...]:
        if self.expected_flat_dim is not None:
            return (tf.TensorSpec([None, self.expected_flat_dim], tf.float32, name="input"),)

        return (
            tf.TensorSpec(
                [None, self.expected_height, self.expected_width, self.expected_channels],
                tf.float32,
                name="input",
            ),
        )

    def _ensure_onnx_model(self) -> Path | None:
        if self.model is None or tf2onnx is None:
            return None

        runtime_dir = self.model_dir / ".runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)

        source_stem = self.model_source_path.stem if self.model_source_path else "fallback_model"
        onnx_path = runtime_dir / f"{source_stem}.onnx"

        needs_export = not onnx_path.exists()
        if self.model_source_path and onnx_path.exists():
            needs_export = onnx_path.stat().st_mtime < self.model_source_path.stat().st_mtime

        if not needs_export:
            return onnx_path

        try:
            tf2onnx.convert.from_keras(
                self.model,
                input_signature=self._keras_input_signature(),
                opset=17,
                output_path=str(onnx_path),
            )
            LOGGER.info("Exported ONNX model to %s", onnx_path)
            return onnx_path
        except Exception as exc:
            LOGGER.warning("Keras->ONNX direct export failed: %s", exc)

        try:
            input_signature = self._keras_input_signature()

            @tf.function(input_signature=input_signature)
            def serving_fn(x):
                return self.model(x)

            tf2onnx.convert.from_function(
                serving_fn,
                input_signature=input_signature,
                opset=17,
                output_path=str(onnx_path),
            )
            LOGGER.info("Exported ONNX model via tf.function to %s", onnx_path)
            return onnx_path
        except Exception as exc:
            LOGGER.warning("Failed to export ONNX model for DirectML runtime: %s", exc)
            return None

    def _try_enable_directml_runtime(self) -> None:
        if not ENABLE_DIRECTML:
            LOGGER.info("DirectML runtime disabled by config; using TensorFlow runtime")
            return

        if ort is None:
            LOGGER.info("onnxruntime is not installed; using TensorFlow runtime")
            return

        available = ort.get_available_providers()
        if "DmlExecutionProvider" not in available:
            LOGGER.info("DirectML provider unavailable (%s); using TensorFlow runtime", available)
            return

        onnx_model_path = self._ensure_onnx_model()
        if onnx_model_path is None:
            return

        try:
            session = ort.InferenceSession(
                str(onnx_model_path),
                providers=["DmlExecutionProvider", "CPUExecutionProvider"],
            )
            self.onnx_session = session
            self.onnx_input_name = session.get_inputs()[0].name
            providers = session.get_providers()
            self.runtime_backend = "onnxruntime"
            self.runtime_provider = providers[0] if providers else "CPUExecutionProvider"
            LOGGER.info("Using runtime backend: %s (%s)", self.runtime_backend, self.runtime_provider)
        except Exception as exc:
            LOGGER.warning("Failed to initialize DirectML session: %s", exc)

    def _prepare_tensor(self, image_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image_rgb, (self.expected_width, self.expected_height))

        if resized.ndim == 2:
            resized = np.stack([resized, resized, resized], axis=-1)

        if self.expected_channels == 1 and resized.ndim == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)[..., np.newaxis]

        if self.expected_channels == 3 and resized.shape[-1] == 1:
            resized = np.repeat(resized, 3, axis=-1)

        normalized = resized.astype(np.float32) / 255.0

        if self.expected_flat_dim is not None:
            flat = normalized.reshape(-1)
            if flat.size < self.expected_flat_dim:
                padded = np.zeros((self.expected_flat_dim,), dtype=np.float32)
                padded[: flat.size] = flat
                flat = padded
            elif flat.size > self.expected_flat_dim:
                flat = flat[: self.expected_flat_dim]
            return flat.astype(np.float32)

        return normalized.astype(np.float32)

    def _probability_from_prediction(self, prediction: np.ndarray) -> float:
        raw = np.asarray(prediction, dtype=np.float32).reshape(-1)
        if raw.size == 0:
            return 0.5

        if raw.size == 1:
            value = float(raw[0])
            if value < 0.0 or value > 1.0:
                value = float(1.0 / (1.0 + np.exp(-value)))
            return float(np.clip(value, 0.0, 1.0))

        if np.any(raw < 0) or not np.isclose(float(raw.sum()), 1.0, atol=1e-2):
            logits = raw - np.max(raw)
            probs = np.exp(logits)
            probs = probs / np.sum(probs)
        else:
            probs = raw

        if probs.size >= 2:
            return float(np.clip(probs[1], 0.0, 1.0))
        return float(np.clip(np.max(probs), 0.0, 1.0))

    def _predict_probabilities(self, batch: np.ndarray) -> np.ndarray:
        batch = np.asarray(batch, dtype=np.float32)

        if self.onnx_session is not None and self.onnx_input_name is not None:
            outputs = self.onnx_session.run(None, {self.onnx_input_name: batch})
            predictions = np.asarray(outputs[0])
        else:
            if self.model is None:
                raise RuntimeError("Model is not initialized")
            predictions = np.asarray(self.model.predict(batch, verbose=0))

        if predictions.ndim == 1:
            predictions = predictions[:, np.newaxis]

        probabilities = [self._probability_from_prediction(row) for row in predictions]
        return np.asarray(probabilities, dtype=np.float32)

    def _method_frame_budget(self, method: str) -> int:
        if method == "fast":
            return 20
        if method == "robust":
            return 72
        return 40

    def _sample_video_frames(self, video_path: Path, max_frames: int) -> list[np.ndarray]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sampled: list[np.ndarray] = []

        if total_frames > 0:
            indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=int)
            for idx in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    continue
                sampled.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        else:
            while len(sampled) < max_frames:
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    break
                sampled.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        capture.release()
        return sampled

    @staticmethod
    def _largest_face_or_none(frame_rgb: np.ndarray) -> np.ndarray | None:
        faces = extract_faces(frame_rgb)
        if not faces:
            return None
        return max(faces, key=lambda face: face.shape[0] * face.shape[1])

    def _prepare_frame_by_method(self, frame_rgb: np.ndarray, method: str) -> tuple[np.ndarray, list[str]]:
        notes: list[str] = []

        if method == "face-focus":
            face = self._largest_face_or_none(frame_rgb)
            if face is not None:
                notes.append("Face-focused crop used")
                return face, notes
            notes.append("No face found; fallback to full frame")
            return frame_rgb, notes

        if method == "frequency":
            notes.append("Frequency-domain transform used")
            return extract_frequency_features(frame_rgb), notes

        return frame_rgb, notes

    def predict_image(self, image_path: str | Path, threshold: float = 0.5, method: str = "balanced") -> DetectionResult:
        start = time.perf_counter()
        image_path = Path(image_path)

        frame_bgr = cv2.imread(str(image_path))
        if frame_bgr is None:
            raise ValueError(f"Unable to read image file: {image_path}")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        prepared, notes = self._prepare_frame_by_method(frame_rgb, method)

        tensor = self._prepare_tensor(prepared)
        batch = np.expand_dims(tensor, axis=0)

        probability = float(self._predict_probabilities(batch)[0])
        confidence = float(max(probability, 1.0 - probability))

        return DetectionResult(
            media_path=str(image_path),
            media_type="image",
            method=method,
            threshold=float(threshold),
            fake_probability=probability,
            confidence=confidence,
            is_fake=probability >= threshold,
            duration_seconds=time.perf_counter() - start,
            frames_analyzed=1,
            notes=notes,
            metadata={
                "runtime_backend": self.runtime_backend,
                "runtime_provider": self.runtime_provider,
            },
        )

    def predict_video(self, video_path: str | Path, threshold: float = 0.5, method: str = "balanced") -> DetectionResult:
        start = time.perf_counter()
        video_path = Path(video_path)

        max_frames = self._method_frame_budget(method)
        frames = self._sample_video_frames(video_path, max_frames=max_frames)
        if not frames:
            raise ValueError("No readable frames were extracted from the video")

        tensors: list[np.ndarray] = []
        notes: list[str] = []

        for frame in frames:
            prepared, frame_notes = self._prepare_frame_by_method(frame, method)
            notes.extend(frame_notes)
            tensors.append(self._prepare_tensor(prepared))

        batch = np.asarray(tensors, dtype=np.float32)

        probabilities: list[float] = []
        chunk_size = 16
        for start_idx in range(0, len(batch), chunk_size):
            chunk = batch[start_idx : start_idx + chunk_size]
            probabilities.extend(self._predict_probabilities(chunk).tolist())

        probabilities_np = np.asarray(probabilities, dtype=np.float32)
        probability = float(np.mean(probabilities_np))
        confidence = float(max(probability, 1.0 - probability))

        return DetectionResult(
            media_path=str(video_path),
            media_type="video",
            method=method,
            threshold=float(threshold),
            fake_probability=probability,
            confidence=confidence,
            is_fake=probability >= threshold,
            duration_seconds=time.perf_counter() - start,
            frames_analyzed=len(frames),
            notes=notes,
            metadata={
                "runtime_backend": self.runtime_backend,
                "runtime_provider": self.runtime_provider,
                "frame_probabilities": [float(value) for value in probabilities_np[:50]],
                "total_frame_scores": int(len(probabilities_np)),
            },
        )
