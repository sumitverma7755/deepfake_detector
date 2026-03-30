"""Project-wide configuration and path constants."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

APP_DIR = PROJECT_ROOT / "app"
MODELS_DIR = PROJECT_ROOT / "models"
UTILS_DIR = PROJECT_ROOT / "utils"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"
OUTPUT_DIR = PROJECT_ROOT / "output"

DEFAULT_LOG_FILE = LOGS_DIR / "deepfake_detector.log"
MODEL_INFO_FILE = MODELS_DIR / "model_info.json"

MODEL_FILES = {
    "efficientnet": MODELS_DIR / "efficientnet_deepfake_detector.h5",
    "resnet_face": MODELS_DIR / "resnet_face_detector.h5",
    "frequency": MODELS_DIR / "frequency_detector.h5",
}

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# DirectML can be unstable on some native Windows driver stacks.
# Keep it opt-in; enable explicitly with DEEPFAKE_ENABLE_DIRECTML=1.
ENABLE_DIRECTML = os.getenv("DEEPFAKE_ENABLE_DIRECTML", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def ensure_runtime_directories() -> None:
    """Create runtime directories required by the application."""
    for directory in (MODELS_DIR, LOGS_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
