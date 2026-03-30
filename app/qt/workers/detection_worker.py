"""QThread worker for single-media detection."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from core.types import DetectionResult
from services.inference_service import InferenceService


class DetectionWorker(QObject):
    """Runs prediction in a worker thread and emits progress signals."""

    finished = Signal(object)
    error = Signal(str)
    log = Signal(str)

    def __init__(
        self,
        inference_service: InferenceService,
        media_path: str,
        media_type: str,
        threshold: float,
        method: str,
    ) -> None:
        super().__init__()
        self.inference_service = inference_service
        self.media_path = media_path
        self.media_type = media_type
        self.threshold = threshold
        self.method = method

    @Slot()
    def run(self) -> None:
        try:
            self.log.emit(f"Starting scan: {Path(self.media_path).name}")

            if self.media_type == "image":
                result = self.inference_service.predict_image(
                    image_path=self.media_path,
                    threshold=self.threshold,
                    method=self.method,
                )
            elif self.media_type == "video":
                result = self.inference_service.predict_video(
                    video_path=self.media_path,
                    threshold=self.threshold,
                    method=self.method,
                )
            else:
                raise ValueError(f"Unsupported media type: {self.media_type}")

            self.log.emit(
                f"Scan complete in {result.duration_seconds:.2f}s | "
                f"Prediction: {'FAKE' if result.is_fake else 'REAL'} | "
                f"p(fake)={result.fake_probability:.3f}"
            )
            self.finished.emit(result)

        except Exception as exc:
            self.error.emit(str(exc))
