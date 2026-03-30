"""Basic project structure smoke tests."""

from pathlib import Path

from config import settings


def test_runtime_directories_exist_after_init():
    settings.ensure_runtime_directories()
    assert settings.MODELS_DIR.exists()
    assert settings.LOGS_DIR.exists()
    assert settings.OUTPUT_DIR.exists()


def test_project_root_contains_core_folders():
    root = settings.PROJECT_ROOT
    for name in ("app", "models", "utils", "scripts", "config", "logs", "tests", "docs"):
        assert (root / name).exists(), f"Missing expected directory: {name}"
