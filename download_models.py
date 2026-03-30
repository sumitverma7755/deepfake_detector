#!/usr/bin/env python3
"""
Download pre-trained models for deepfake detection
"""

import os
import sys
import requests
import tqdm
import hashlib
import json
import tensorflow as tf
import time
import logging
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_download.log')
    ]
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "pretrained")

# Model URLs (example placeholders - replace with actual URLs when available)
MODEL_URLS = {
    'efficientnet': 'https://github.com/yourusername/deepfake-detector/releases/download/v1.0/efficientnet_deepfake_detector.h5',
    'resnet_face': 'https://github.com/yourusername/deepfake-detector/releases/download/v1.0/resnet_face_detector.h5',
    'frequency': 'https://github.com/yourusername/deepfake-detector/releases/download/v1.0/frequency_detector.h5',
    # Add more models as needed
}

# SHA-256 checksums for model files (for verification)
CHECKSUMS = {
    'efficientnet_deepfake_detector.h5': '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
    'resnet_face_detector.h5': '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
    'frequency_detector.h5': '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
    # Add checksums for additional models
}

def download_file(url, destination, filename):
    """
    Download a file with progress indicator.
    
    Args:
        url: URL to download from
        destination: Directory to save the file
        filename: Name to save the file as
        
    Returns:
        Path to the downloaded file if successful, None otherwise
    """
    output_path = os.path.join(destination, filename)
    temp_path = output_path + '.download'

    try:
        # Check if file already exists and is valid
        if os.path.exists(output_path):
            logger.info(f"File already exists: {output_path}")
            if verify_checksum(output_path, CHECKSUMS.get(filename)):
                logger.info("Checksum verification passed, skipping download.")
                return output_path
            else:
                logger.warning("Existing file failed checksum verification. Re-downloading...")
        
        # Create destination directory if it doesn't exist
        os.makedirs(destination, exist_ok=True)
        
        # Download with progress bar
        logger.info(f"Downloading {filename} from {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192  # 8 KB
        
        with open(temp_path, 'wb') as f, tqdm.tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                f.write(data)
                progress_bar.update(len(data))
        
        # Rename temp file to final filename
        os.replace(temp_path, output_path)
        
        # Verify downloaded file
        if verify_checksum(output_path, CHECKSUMS.get(filename)):
            logger.info(f"Successfully downloaded and verified: {filename}")
            return output_path
        else:
            logger.error(f"Checksum verification failed for: {filename}")
            return None
            
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None

def verify_checksum(file_path, expected_checksum=None):
    """
    Verify the SHA-256 checksum of a file.
    
    Args:
        file_path: Path to the file to verify
        expected_checksum: Expected SHA-256 checksum
        
    Returns:
        True if checksum matches or if expected_checksum is None, False otherwise
    """
    # If no expected checksum is provided, skip verification
    if expected_checksum is None:
        logger.warning(f"No checksum available for {os.path.basename(file_path)}. Skipping verification.")
        return True
    
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256()
            while chunk := f.read(8192):
                file_hash.update(chunk)
        
        calculated_checksum = file_hash.hexdigest()
        
        if calculated_checksum.lower() == expected_checksum.lower():
            return True
        else:
            logger.error(f"Checksum mismatch for {file_path}")
            logger.error(f"Expected: {expected_checksum}")
            logger.error(f"Got: {calculated_checksum}")
            return False
            
    except Exception as e:
        logger.error(f"Error verifying checksum for {file_path}: {e}")
        return False

def download_all_models(model_dir=MODEL_DIR, force=False):
    """
    Download all pre-trained models.
    
    Args:
        model_dir: Directory to save models
        force: Force re-download even if files exist
        
    Returns:
        List of successfully downloaded model paths
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    downloaded_models = {}
    download_status = {}
    
    # Track start time for the entire process
    start_time = time.time()
    
    # Download each model
    for model_name, url in MODEL_URLS.items():
        filename = os.path.basename(url)
        
        # Skip if file exists and force is False
        file_path = os.path.join(model_dir, filename)
        if os.path.exists(file_path) and not force:
            if verify_checksum(file_path, CHECKSUMS.get(filename)):
                logger.info(f"Model {model_name} already exists and is valid. Skipping download.")
                downloaded_models[model_name] = file_path
                download_status[model_name] = "already_exists"
                continue
        
        logger.info(f"Downloading model: {model_name}")
        model_path = download_file(url, model_dir, filename)
        
        if model_path:
            downloaded_models[model_name] = model_path
            download_status[model_name] = "downloaded"
        else:
            download_status[model_name] = "failed"
    
    # Calculate download time
    download_time = time.time() - start_time
    
    # Create model info file
    model_info = {
        'download_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'download_time_seconds': download_time,
        'models': {}
    }
    
    # Add info for each model
    for model_name, path in downloaded_models.items():
        if path and os.path.exists(path):
            # Get file size
            file_size = os.path.getsize(path)
            
            # Load model to get summary information if possible
            model_summary = {}
            try:
                model = tf.keras.models.load_model(path)
                model_summary = {
                    'input_shape': str(model.input_shape),
                    'output_shape': str(model.output_shape),
                    'layers': len(model.layers),
                    'trainable_params': int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)),
                    'non_trainable_params': int(sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights))
                }
            except Exception as e:
                logger.warning(f"Could not load model {model_name} for summary: {e}")
            
            model_info['models'][model_name] = {
                'filename': os.path.basename(path),
                'path': path,
                'size_bytes': file_size,
                'status': download_status.get(model_name, 'unknown'),
                'summary': model_summary
            }
    
    # Save model info to JSON file
    info_path = os.path.join(model_dir, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"Model information saved to: {info_path}")
    logger.info(f"Downloaded {len(downloaded_models)} models in {download_time:.2f} seconds")
    
    return downloaded_models

def main():
    """Main function to download models from command line"""
    parser = argparse.ArgumentParser(description='Download pre-trained deepfake detection models')
    parser.add_argument('--force', action='store_true', help='Force re-download even if models exist')
    parser.add_argument('--dir', help='Custom directory to save models (default: ./models/pretrained)')
    parser.add_argument('--list', action='store_true', help='List available models without downloading')
    parser.add_argument('--model', help='Download a specific model by name')
    args = parser.parse_args()
    
    # List available models if requested
    if args.list:
        print("Available models:")
        for name, url in MODEL_URLS.items():
            filename = os.path.basename(url)
            size = "Unknown size"
            print(f"- {name}: {filename} ({size})")
        return 0
    
    # Set model directory
    model_dir = args.dir if args.dir else MODEL_DIR
    
    # Download a specific model if requested
    if args.model:
        if args.model not in MODEL_URLS:
            logger.error(f"Unknown model: {args.model}")
            print(f"Available models: {', '.join(MODEL_URLS.keys())}")
            return 1
        
        url = MODEL_URLS[args.model]
        filename = os.path.basename(url)
        logger.info(f"Downloading model: {args.model}")
        
        model_path = download_file(url, model_dir, filename)
        if model_path:
            logger.info(f"Successfully downloaded: {model_path}")
            return 0
        else:
            logger.error(f"Failed to download model: {args.model}")
            return 1
    
    # Download all models
    downloaded_models = download_all_models(model_dir, args.force)
    
    # Print summary
    print("\nDownload Summary:")
    for model_name, path in downloaded_models.items():
        status = "✅ Success" if path and os.path.exists(path) else "❌ Failed"
        print(f"- {model_name}: {status}")
    
    # Check if any downloads failed
    if len(downloaded_models) < len(MODEL_URLS):
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nDownload cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1) 