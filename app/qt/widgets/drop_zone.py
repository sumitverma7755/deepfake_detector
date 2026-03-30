"""Drag and drop upload widget."""

from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class DropZoneWidget(QFrame):
    """Interactive drop area used for image/video uploads."""

    file_dropped = Signal(str)
    clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("Card")
        self.setMinimumHeight(180)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(6)

        self.title = QLabel("Drag & Drop Media Here")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet("font-size: 16px; font-weight: 700; color: #e2e8f0;")

        self.subtitle = QLabel("Supports images and videos. Click to browse.")
        self.subtitle.setObjectName("MutedLabel")
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addStretch(1)
        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        layout.addStretch(1)

        self._set_inactive_style()

    def _set_inactive_style(self) -> None:
        self.setStyleSheet(
            """
            QFrame {
                background-color: #0b1220;
                border: 2px dashed #334155;
                border-radius: 16px;
            }
            """
        )

    def _set_active_style(self) -> None:
        self.setStyleSheet(
            """
            QFrame {
                background-color: #0f2a2a;
                border: 2px dashed #14b8a6;
                border-radius: 16px;
            }
            """
        )

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_active_style()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._set_inactive_style()
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        self._set_inactive_style()
        if not event.mimeData().hasUrls():
            event.ignore()
            return

        local_path = event.mimeData().urls()[0].toLocalFile()
        if local_path:
            self.file_dropped.emit(local_path)
            event.acceptProposedAction()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)
