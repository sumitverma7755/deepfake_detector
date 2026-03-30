"""Animated loading spinner widget."""

from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class SpinnerWidget(QWidget):
    """Simple circular spinner built with QPainter and QTimer."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.setFixedSize(28, 28)
        self.hide()

    def _tick(self) -> None:
        self._angle = (self._angle + 24) % 360
        self.update()

    def start(self) -> None:
        self.show()
        self._timer.start(40)

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    def paintEvent(self, event) -> None:  # noqa: N802
        if not self.isVisible():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self._angle)

        pen = QPen(QColor("#14b8a6"))
        pen.setWidth(3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        radius = min(self.width(), self.height()) // 2 - 4
        rect = (-radius, -radius, radius * 2, radius * 2)
        painter.drawArc(*rect, 0, 220 * 16)
