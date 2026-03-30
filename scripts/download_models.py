#!/usr/bin/env python3
"""Download pre-trained models for deepfake detection."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests
import tensorflow as tf
import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import LOGS_DIR, MODELS_DIR, ensure_runtime_directories

ensure_runtime_directories()

MODEL_DOWNLOAD_LOG = LOGS_DIR / "model_download.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(MODEL_DOWNLOAD_LOG),
    ],
)
logger = logging.getLogger(__name__)

MODEL_URLS = {
    "efficientnet": "https://github.com/yourusername/deepfake-detector/releases/download/v1.0/efficientnet_deepfake_detector.h5",
    "resnet_face": "https://github.com/yourusername/deepfake-detector/releases/download/v1.0/resnet_face_detector.h5",
    "frequency": "https://github.com/yourusername/deepfake-detector/releases/download/v1.0/frequency_detector.h5",
}

CHECKSUMS = {
    "efficientnet_deepfake_detector.h5": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "resnet_face_detector.h5": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "frequency_detector.h5": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
}


def verify_checksum(file_path: Path, expected_checksum: str | None = None) -> bool:
    """Verify SHA-256 checksum for a file."""
    if expected_checksum is None:
        logger.warning("No checksum available for %s; skipping verification.", file_path.name)
        return True

    hasher = hashlib.sha256()
    try:
        with file_path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(8192), b""):
                hasher.update(chunk)
    except OSError as exc:
        logger.error("Failed to read %s for checksum: %s", file_path, exc)
        return False

    calculated = hasher.hexdigest().lower()
    expected = expected_checksum.lower()
    if calculated == expected:
        return True

    logger.error("Checksum mismatch for %s", file_path)
    logger.error("Expected: %s", expected)
    logger.error("Got:      %s", calculated)
    return False


def download_file(url: str, destination_dir: Path, filename: str) -> Path | None:
    """Download a file with progress display."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / filename
    temp_path = output_path.with_suffix(output_path.suffix + ".download")

    try:
        if output_path.exists() and verify_checksum(output_path, CHECKSUMS.get(filename)):
            logger.info("File already exists and is valid: %s", output_path)
            return output_path

        logger.info("Downloading %s", filename)
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with temp_path.open("wb") as file_obj, tqdm.tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            for chunk in response.iter_content(8192):
                if not chunk:
                    continue
                file_obj.write(chunk)
                progress.update(len(chunk))

        temp_path.replace(output_path)
        if verify_checksum(output_path, CHECKSUMS.get(filename)):
            logger.info("Downloaded and verified: %s", output_path)
            return output_path

        logger.error("Downloaded file failed checksum: %s", output_path)
        return None

    except Exception as exc:
        logger.error("Failed to download %s: %s", url, exc)
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        return None


def _summarize_model(model_path: Path) -> dict:
    """Try to load and summarize a Keras model."""
    summary: dict = {}
    try:
        model = tf.keras.models.load_model(model_path)
        summary = {
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "layers": len(model.layers),
            "trainable_params": int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)),
            "non_trainable_params": int(sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)),
        }
    except Exception as exc:  # pragma: no cover - model content dependent
        logger.warning("Unable to inspect %s: %s", model_path.name, exc)
    return summary


def download_all_models(model_dir: Path, force: bool = False) -> dict[str, str]:
    """Download all configured model artifacts."""
    model_dir.mkdir(parents=True, exist_ok=True)
    downloaded_models: dict[str, str] = {}
    download_status: dict[str, str] = {}
    start_time = time.time()

    for model_name, url in MODEL_URLS.items():
        filename = os.path.basename(url)
        destination = model_dir / filename

        if destination.exists() and not force and verify_checksum(destination, CHECKSUMS.get(filename)):
            logger.info("Model already exists and verified: %s", destination)
            downloaded_models[model_name] = str(destination)
            download_status[model_name] = "already_exists"
            continue

        downloaded_path = download_file(url=url, destination_dir=model_dir, filename=filename)
        if downloaded_path is None:
            download_status[model_name] = "failed"
            continue

        downloaded_models[model_name] = str(downloaded_path)
        download_status[model_name] = "downloaded"

    elapsed = time.time() - start_time

    info_payload = {
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "download_time_seconds": elapsed,
        "models": {},
    }

    for model_name, path in downloaded_models.items():
        model_path = Path(path)
        if not model_path.exists():
            continue
        info_payload["models"][model_name] = {
            "filename": model_path.name,
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "status": download_status.get(model_name, "unknown"),
            "summary": _summarize_model(model_path),
        }

    info_file = model_dir / "model_info.json"
    info_file.write_text(json.dumps(info_payload, indent=2), encoding="utf-8")

    logger.info("Model info saved to %s", info_file)
    logger.info("Model download complete in %.2f seconds", elapsed)
    return downloaded_models


def main() -> int:
    """CLI entrypoint for model downloads."""
    parser = argparse.ArgumentParser(description="Download pre-trained deepfake detection models")
    parser.add_argument("--force", action="store_true", help="Force re-download even if models already exist")
    parser.add_argument("--dir", help="Custom destination directory (default: ./models)")
    parser.add_argument("--list", action="store_true", help="List available models without downloading")
    parser.add_argument("--model", help="Download a single model by key")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name, url in MODEL_URLS.items():
            print(f"- {name}: {Path(url).name}")
        return 0

    target_dir = Path(args.dir).resolve() if args.dir else MODELS_DIR

    if args.model:
        if args.model not in MODEL_URLS:
            print(f"Model '{args.model}' not found. Available: {', '.join(MODEL_URLS)}")
            return 1

        url = MODEL_URLS[args.model]
        filename = os.path.basename(url)
        result = download_file(url=url, destination_dir=target_dir, filename=filename)
        if result is None:
            return 1
        print(f"Downloaded {args.model}: {result}")
        return 0

    downloaded = download_all_models(model_dir=target_dir, force=args.force)
    if not downloaded:
        logger.error("No models were downloaded successfully")
        return 1

    print("Successfully downloaded models:")
    for model_name, model_path in downloaded.items():
        print(f"- {model_name}: {model_path}")

    if len(downloaded) < len(MODEL_URLS):
        logger.warning("Some models failed to download")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
