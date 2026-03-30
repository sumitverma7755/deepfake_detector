"""PySide6 application bootstrap."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from config.settings import ensure_runtime_directories
from .main_window import MainWindow


def launch() -> int:
    """Create and run the Qt application."""
    ensure_runtime_directories()

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName("DeepFake Detector Pro")
    app.setOrganizationName("DeepFake Labs")

    window = MainWindow()
    window.show()

    return app.exec()


def main() -> None:
    raise SystemExit(launch())


if __name__ == "__main__":
    main()
