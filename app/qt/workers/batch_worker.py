"""QThread worker for batch directory scans."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from config.settings import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_VIDEO_EXTENSIONS
from core.types import BatchItemResult
from services.inference_service import InferenceService


class BatchWorker(QObject):
    """Runs batch scanning without blocking the main thread."""

    item_complete = Signal(object)
    finished = Signal(object)
    error = Signal(str)
    log = Signal(str)

    def __init__(
        self,
        inference_service: InferenceService,
        directory_path: str,
        threshold: float,
        method: str,
    ) -> None:
        super().__init__()
        self.inference_service = inference_service
        self.directory_path = Path(directory_path)
        self.threshold = threshold
        self.method = method

    def _discover_media_files(self) -> list[Path]:
        files: list[Path] = []
        for path in self.directory_path.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in SUPPORTED_IMAGE_EXTENSIONS or suffix in SUPPORTED_VIDEO_EXTENSIONS:
                files.append(path)
        return sorted(files)

    @Slot()
    def run(self) -> None:
        try:
            if not self.directory_path.exists():
                raise ValueError(f"Directory does not exist: {self.directory_path}")

            files = self._discover_media_files()
            if not files:
                raise ValueError("No supported media files found in selected directory")

            self.log.emit(f"Batch scan started with {len(files)} file(s)")
            results: list[BatchItemResult] = []

            for index, media_path in enumerate(files, start=1):
                try:
                    suffix = media_path.suffix.lower()
                    media_type = "image" if suffix in SUPPORTED_IMAGE_EXTENSIONS else "video"
                    self.log.emit(f"[{index}/{len(files)}] Processing {media_path.name}")

                    if media_type == "image":
                        prediction = self.inference_service.predict_image(
                            image_path=media_path,
                            threshold=self.threshold,
                            method=self.method,
                        )
                    else:
                        prediction = self.inference_service.predict_video(
                            video_path=media_path,
                            threshold=self.threshold,
                            method=self.method,
                        )

                    item = BatchItemResult(
                        media_path=str(media_path),
                        status="FAKE" if prediction.is_fake else "REAL",
                        fake_probability=prediction.fake_probability,
                        confidence=prediction.confidence,
                    )
                    results.append(item)
                    self.item_complete.emit(item)

                except Exception as item_exc:
                    item = BatchItemResult(
                        media_path=str(media_path),
                        status="ERROR",
                        error=str(item_exc),
                    )
                    results.append(item)
                    self.item_complete.emit(item)

            self.log.emit("Batch scan completed")
            self.finished.emit(results)

        except Exception as exc:
            self.error.emit(str(exc))
