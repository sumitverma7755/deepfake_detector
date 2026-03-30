#!/usr/bin/env python3
"""Legacy compatibility shim that redirects GUI launch to PySide6 desktop app."""

from __future__ import annotations

from app.qt.app import launch


def main() -> None:
    raise SystemExit(launch())


if __name__ == "__main__":
    main()
