"""Model loading and lightweight architecture helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import tensorflow as tf

LOGGER = logging.getLogger(__name__)


MODEL_EXPLANATIONS: Dict[str, str] = {
    "efficientnet": "EfficientNet-based binary classifier for whole-image deepfake cues.",
    "resnet_face": "Face-focused classifier intended for localized forgery artifacts.",
    "frequency": "Frequency-domain classifier for GAN-related spectral artifacts.",
}


def _build_fallback_binary_model(input_shape=(224, 224, 3)) -> tf.keras.Model:
    """Create a small binary classifier as a safe fallback model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_model_from_checkpoint(model_type: str, checkpoint_path: str | Path) -> tf.keras.Model:
    """Load a model checkpoint; return a fallback model when unavailable or invalid."""
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.exists():
        try:
            LOGGER.info("Loading model checkpoint: %s", checkpoint_path)
            return tf.keras.models.load_model(checkpoint_path, compile=False)
        except Exception as exc:  # pragma: no cover - TensorFlow/backend dependent
            LOGGER.warning(
                "Failed to load checkpoint for '%s' from %s: %s. Using fallback model.",
                model_type,
                checkpoint_path,
                exc,
            )
    else:
        LOGGER.warning("Checkpoint not found for '%s': %s", model_type, checkpoint_path)

    return _build_fallback_binary_model()


def get_model_explanation(model_type: str) -> str:
    """Return a human-readable explanation of a model role."""
    return MODEL_EXPLANATIONS.get(
        model_type,
        "Binary classifier used for deepfake likelihood estimation.",
    )
