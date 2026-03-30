"""Visualization helpers for deepfake detection outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _label_from_prediction(prediction: float, threshold: float) -> str:
    return "FAKE" if prediction >= threshold else "REAL"


def plot_detection_result(image: np.ndarray, prediction: float, threshold: float = 0.5, ax=None):
    """Plot an image with a prediction label."""
    created_ax = ax is None
    if created_ax:
        _, ax = plt.subplots(figsize=(8, 6))

    label = _label_from_prediction(prediction, threshold)
    color = "#d32f2f" if label == "FAKE" else "#2e7d32"

    ax.imshow(image)
    ax.axis("off")
    ax.set_title(f"{label} ({prediction:.2%})", color=color, fontsize=14, fontweight="bold")

    if created_ax:
        plt.tight_layout()
        return ax.figure
    return ax


def create_detection_report(
    image: np.ndarray,
    prediction: float,
    face_locations: Sequence[tuple],
    threshold: float,
    output_path: str | Path,
) -> None:
    """Create and save a report image with prediction and optional face boxes."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, ax = plt.subplots(figsize=(10, 7))
    plot_detection_result(image=image, prediction=prediction, threshold=threshold, ax=ax)

    for location in face_locations:
        top, right, bottom, left = location
        width = max(0, right - left)
        height = max(0, bottom - top)
        rect = plt.Rectangle((left, top), width, height, fill=False, edgecolor="#ffeb3b", linewidth=2)
        ax.add_patch(rect)

    summary = [
        f"Prediction: {prediction:.2%}",
        f"Threshold: {threshold:.2f}",
        f"Faces detected: {len(face_locations)}",
    ]
    figure.text(0.02, 0.02, " | ".join(summary), fontsize=10)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def visualize_video_analysis(
    frames: Sequence[np.ndarray],
    predictions: Sequence[dict],
    interval: int,
    output_path: str | Path,
) -> None:
    """Create a montage of sampled video frames with prediction scores."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        figure, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No frames available", ha="center", va="center")
        ax.axis("off")
        figure.savefig(output_path, dpi=180)
        plt.close(figure)
        return

    step = max(1, interval)
    sampled_indices = list(range(0, len(frames), step))[:16]
    sampled_frames = [frames[index] for index in sampled_indices]

    cols = 4
    rows = int(np.ceil(len(sampled_frames) / cols))
    figure, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    prediction_map = {
        p.get("frame_index", idx): p.get("probability", 0.0)
        for idx, p in enumerate(predictions)
    }

    for slot, (ax, frame_idx) in enumerate(zip(axes.flat, sampled_indices)):
        frame = sampled_frames[slot]
        probability = float(prediction_map.get(frame_idx, 0.0))
        label = _label_from_prediction(probability, 0.5)
        color = "#d32f2f" if label == "FAKE" else "#2e7d32"

        ax.imshow(frame)
        ax.set_title(f"Frame {frame_idx} | {probability:.2%}", color=color, fontsize=10)
        ax.axis("off")

    for ax in axes.flat[len(sampled_frames) :]:
        ax.axis("off")

    figure.suptitle("Video Deepfake Analysis", fontsize=14, fontweight="bold")
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
