"""Result exporting helpers for desktop workflows."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from core.types import DetectionResult
from config.settings import OUTPUT_DIR, ensure_runtime_directories

ensure_runtime_directories()


def export_detection_report(result: DetectionResult, destination: str | Path | None = None) -> Path:
    """Export a human-readable report file and return its path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if destination is None:
        media_name = Path(result.media_path).stem
        destination = OUTPUT_DIR / f"{media_name}_report_{timestamp}.txt"
    else:
        destination = Path(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = [
        "DeepFake Detector Report",
        "======================",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Media Path: {result.media_path}",
        f"Media Type: {result.media_type}",
        f"Method: {result.method}",
        f"Threshold: {result.threshold:.2f}",
        f"Prediction: {'FAKE' if result.is_fake else 'REAL'}",
        f"Fake Probability: {result.fake_probability:.4f}",
        f"Confidence: {result.confidence:.4f}",
        f"Duration (s): {result.duration_seconds:.3f}",
        f"Frames Analyzed: {result.frames_analyzed}",
    ]

    if result.notes:
        payload.append("\nNotes:")
        payload.extend([f"- {note}" for note in result.notes])

    if result.metadata:
        payload.append("\nMetadata:")
        for key, value in result.metadata.items():
            payload.append(f"- {key}: {value}")

    destination.write_text("\n".join(payload), encoding="utf-8")
    return destination
