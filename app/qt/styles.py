"""Qt stylesheet for premium dark desktop visuals."""

from __future__ import annotations


def build_stylesheet() -> str:
    return """
    QMainWindow {
        background-color: #0f172a;
        color: #e2e8f0;
        font-family: 'Segoe UI', 'Inter', sans-serif;
        font-size: 13px;
    }

    QFrame#Sidebar {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    QLabel#BrandTitle {
        font-size: 20px;
        font-weight: 700;
        color: #f8fafc;
    }

    QLabel#MutedLabel {
        color: #94a3b8;
        font-size: 12px;
    }

    QPushButton#NavButton {
        background-color: transparent;
        color: #cbd5e1;
        border: none;
        border-radius: 10px;
        padding: 10px 12px;
        text-align: left;
        font-weight: 600;
    }

    QPushButton#NavButton:hover {
        background-color: #1f2937;
        color: #f8fafc;
    }

    QPushButton#NavButton:checked {
        background-color: #14b8a6;
        color: #042f2e;
    }

    QFrame#TopBar {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
    }

    QFrame#Card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 16px;
    }

    QPushButton#PrimaryButton {
        background-color: #14b8a6;
        color: #042f2e;
        border: none;
        border-radius: 10px;
        padding: 10px 16px;
        font-weight: 700;
    }

    QPushButton#PrimaryButton:hover {
        background-color: #2dd4bf;
    }

    QPushButton#SecondaryButton {
        background-color: #1f2937;
        color: #e2e8f0;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 9px 14px;
        font-weight: 600;
    }

    QPushButton#SecondaryButton:hover {
        background-color: #334155;
        border-color: #475569;
    }

    QComboBox,
    QLineEdit,
    QTextEdit,
    QPlainTextEdit,
    QTableWidget,
    QListWidget {
        background-color: #0b1220;
        color: #e2e8f0;
        border: 1px solid #334155;
        border-radius: 10px;
        selection-background-color: #14b8a6;
        selection-color: #042f2e;
    }

    QComboBox {
        padding: 6px 10px;
    }

    QSlider::groove:horizontal {
        height: 6px;
        background: #1f2937;
        border-radius: 3px;
    }

    QSlider::handle:horizontal {
        width: 16px;
        margin: -5px 0;
        background: #14b8a6;
        border-radius: 8px;
    }

    QLabel#ResultBadge {
        border-radius: 16px;
        padding: 6px 12px;
        font-size: 13px;
        font-weight: 700;
        color: #f8fafc;
        background-color: #334155;
    }

    QProgressBar#BusyBar {
        border: none;
        border-radius: 6px;
        background-color: #1f2937;
        min-height: 8px;
        max-height: 8px;
    }

    QProgressBar#BusyBar::chunk {
        background-color: #14b8a6;
        border-radius: 6px;
    }

    QTableWidget {
        gridline-color: #1f2937;
    }

    QHeaderView::section {
        background-color: #111827;
        color: #94a3b8;
        border: none;
        border-bottom: 1px solid #1f2937;
        padding: 8px;
        font-weight: 700;
    }
    """
