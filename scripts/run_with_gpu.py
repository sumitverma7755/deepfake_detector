#!/usr/bin/env python3
"""Run the deepfake detector with GPU-aware checks and CPU fallback."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf

from config.settings import MODELS_DIR, ensure_runtime_directories
from utils.gpu_config import configure_gpu

ensure_runtime_directories()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def detect_directml() -> tuple[bool, list[str]]:
    """Check whether ONNX Runtime DirectML provider is available."""
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        return "DmlExecutionProvider" in providers, providers
    except Exception:
        return False, []


def verify_environment() -> bool:
    """Check whether required runtime dependencies are available."""
    logger.info("Python version: %s", sys.version.split()[0])
    logger.info("TensorFlow version: %s", tf.__version__)

    gpu_available = configure_gpu()
    if gpu_available:
        gpus = tf.config.list_physical_devices("GPU")
        logger.info("GPU is configured successfully (%s device(s)).", len(gpus))
    else:
        logger.warning("TensorFlow CUDA GPU not detected (expected on native Windows TF>=2.11).")

    dml_available, providers = detect_directml()
    if dml_available:
        logger.info("DirectML provider is available: %s", providers)
    else:
        logger.warning("DirectML provider unavailable. Providers detected: %s", providers or "none")

    try:
        import cv2  # noqa: F401
        logger.info("OpenCV is available")
    except ImportError:
        logger.error("OpenCV is not installed. Run: pip install opencv-python")
        return False

    try:
        import numpy  # noqa: F401
        logger.info("NumPy is available")
    except ImportError:
        logger.error("NumPy is not installed. Run: pip install numpy")
        return False

    if not MODELS_DIR.exists():
        logger.warning("Models directory not found: %s", MODELS_DIR)
        logger.info("Run: python scripts/download_models.py")

    try:
        from services.inference_service import InferenceService

        runtime = InferenceService(model_dir=MODELS_DIR).runtime_description
        logger.info("Detector runtime backend: %s", runtime)
    except Exception as exc:
        logger.warning("Could not initialize inference runtime during verification: %s", exc)

    return True


def run_gui() -> None:
    """Run the desktop GUI."""
    logger.info("Starting deepfake detector GUI...")
    configure_gpu()

    try:
        from app.gui import main as gui_main

        gui_main()
    except Exception as exc:
        logger.error("Error running GUI: %s", exc)
        raise SystemExit(1) from exc


def run_cli(cli_args: list[str]) -> None:
    """Run the CLI entrypoint."""
    configure_gpu()
    try:
        from app.main import main as cli_main

        cli_main(cli_args)
    except Exception as exc:
        logger.error("Error running CLI: %s", exc)
        raise SystemExit(1) from exc


def main() -> None:
    """Script entrypoint."""
    parser = argparse.ArgumentParser(description="Run deepfake detector with CPU fallback if GPU is unavailable")
    parser.add_argument("--gui", action="store_true", help="Start the graphical user interface")
    parser.add_argument("--verify", action="store_true", help="Verify environment and exit")
    parser.add_argument("--image", help="Path to the image file to analyze")
    parser.add_argument("--video", help="Path to the video file to analyze")
    parser.add_argument("--batch", help="Directory containing media files to analyze in batch")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (0.0-1.0)")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--frames", type=int, help="Number of frames to analyze for video")
    parser.add_argument("--format", choices=["json", "txt", "html"], default="json", help="Output format")

    args, remaining = parser.parse_known_args()

    if args.verify:
        if verify_environment():
            logger.info("Environment verification completed successfully")
            return
        logger.error("Environment verification failed")
        raise SystemExit(1)

    if not verify_environment():
        logger.error("Environment verification failed. Fix issues before continuing.")
        raise SystemExit(1)

    if args.gui:
        run_gui()
        return

    cli_args: list[str] = []
    if args.image:
        cli_args.extend(["--image", args.image])
    if args.video:
        cli_args.extend(["--video", args.video])
    if args.batch:
        cli_args.extend(["--batch", args.batch])
    if args.threshold is not None:
        cli_args.extend(["--threshold", str(args.threshold)])
    if args.output:
        cli_args.extend(["--output", args.output])
    if args.frames is not None:
        cli_args.extend(["--frames", str(args.frames)])
    if args.format:
        cli_args.extend(["--format", args.format])

    run_cli(cli_args if cli_args else remaining)


if __name__ == "__main__":
    main()
