"""Media preview widget for images and videos."""

from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class PreviewWidget(QFrame):
    """Preview pane that displays an image or first video frame."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setMinimumHeight(280)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        self.label = QLabel("Preview will appear here")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #94a3b8; font-size: 14px; font-weight: 600;")
        self.label.setWordWrap(True)

        layout.addWidget(self.label)

        self._pixmap: QPixmap | None = None

    def clear(self) -> None:
        self._pixmap = None
        self.label.setPixmap(QPixmap())
        self.label.setText("Preview will appear here")
        self.label.setStyleSheet("color: #94a3b8; font-size: 14px; font-weight: 600;")

    def set_image(self, image_path: str | Path) -> None:
        frame_bgr = cv2.imread(str(image_path))
        if frame_bgr is None:
            self.clear()
            self.label.setText("Could not render image preview")
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._set_frame(frame_rgb)

    def set_video(self, video_path: str | Path) -> None:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            self.clear()
            self.label.setText("Could not open video preview")
            return

        ok, frame_bgr = capture.read()
        capture.release()

        if not ok or frame_bgr is None:
            self.clear()
            self.label.setText("Could not extract first frame")
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._set_frame(frame_rgb)

    def _set_frame(self, frame_rgb) -> None:
        h, w, _ = frame_rgb.shape
        bytes_per_line = 3 * w
        image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(image)
        self._apply_scaled_pixmap()

    def _apply_scaled_pixmap(self) -> None:
        if self._pixmap is None:
            return

        scaled = self._pixmap.scaled(
            self.label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.label.setPixmap(scaled)
        self.label.setText("")

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_scaled_pixmap()
