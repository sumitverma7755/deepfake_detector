#!/usr/bin/env python3
"""Project entrypoint for desktop (PySide6) and CLI modes."""

from __future__ import annotations

import argparse

from config.settings import ensure_runtime_directories


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepFake Detector launcher")
    parser.add_argument("--cli", action="store_true", help="Run command-line interface instead of desktop UI")
    args, remaining = parser.parse_known_args()

    ensure_runtime_directories()

    if args.cli:
        from app.main import main as cli_main

        cli_main(remaining)
        return

    from app.qt.app import launch

    raise SystemExit(launch())


if __name__ == "__main__":
    main()
