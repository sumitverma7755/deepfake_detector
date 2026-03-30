#!/usr/bin/env python3
"""
Run the deepfake detector with fallback to CPU when GPU is not available
"""

import os
import sys
import argparse
import tensorflow as tf
import logging
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import GPU configuration
from gpu_config import configure_gpu, get_device_strategy

def verify_environment():
    """Check if all required components are available"""
    # Check Python version
    python_version = sys.version.split()[0]
    logger.info(f"Python version: {python_version}")
    
    # Check TensorFlow version
    tf_version = tf.__version__
    logger.info(f"TensorFlow version: {tf_version}")
    
    # Check for GPU
    gpu_available = configure_gpu()
    if not gpu_available:
        logger.warning("No GPU detected. Running on CPU instead.")
        logger.warning("Processing will be significantly slower.")
    else:
        gpus = tf.config.list_physical_devices('GPU')
        logger.info(f"Found {len(gpus)} GPU device(s).")
        logger.info("GPU is configured successfully.")
    
    # Check for required libraries
    try:
        import cv2
        logger.info(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        logger.error("OpenCV is not installed. Try 'pip install opencv-python'")
        return False
    
    try:
        import numpy
        logger.info(f"NumPy version: {numpy.__version__}")
    except ImportError:
        logger.error("NumPy is not installed. Try 'pip install numpy'")
        return False
    
    # Try to import dlib (optional)
    try:
        import dlib
        logger.info(f"dlib is installed. Face detection will use dlib.")
        has_dlib = True
    except ImportError:
        logger.warning("dlib is not installed. Face detection will use OpenCV instead.")
        logger.info("For better face detection, install dlib with: pip install dlib")
        has_dlib = False
    
    # Check if model files exist
    models_dir = os.path.join(script_dir, "models", "pretrained")
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory not found: {models_dir}")
        logger.info("Run 'python download_models.py' to download pre-trained models")
    
    return True

def run_gui():
    """Run the GUI application with CPU fallback"""
    logger.info("Starting deepfake detector GUI...")
    
    # Check if GPU is available
    gpu_available = configure_gpu()
    if gpu_available:
        logger.info("Starting with GPU support")
    else:
        logger.info("Starting with CPU (GPU not available)")
    
    # Import here to avoid loading GUI libraries if not needed
    try:
        from deepfake_detector_gui import main
        main()
    except Exception as e:
        logger.error(f"Error running GUI: {e}")
        sys.exit(1)

def run_cli(args):
    """Run the command line interface with CPU fallback"""
    # Check if GPU is available
    gpu_available = configure_gpu()
    if gpu_available:
        logger.info("Running deepfake detector CLI with GPU support...")
    else:
        logger.info("Running deepfake detector CLI with CPU (GPU not available)...")
    
    # Import here to avoid loading everything if not needed
    try:
        from detect import main
        # Set sys.argv for the main function to pick up
        sys.argv = ['detect.py'] + args
        main()
    except Exception as e:
        logger.error(f"Error running CLI: {e}")
        sys.exit(1)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run deepfake detector (with CPU fallback if GPU unavailable)")
    parser.add_argument("--gui", action="store_true", help="Start the graphical user interface")
    parser.add_argument("--verify", action="store_true", help="Verify environment and exit")
    parser.add_argument("--image", help="Path to the image file to analyze")
    parser.add_argument("--video", help="Path to the video file to analyze")
    parser.add_argument("--batch", help="Directory containing media files to analyze in batch")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (0.0-1.0)")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--format", choices=["json", "txt", "html"], default="json", help="Output format")
    
    # Parse arguments
    args, remaining = parser.parse_known_args()
    
    # Verify environment if requested
    if args.verify:
        if verify_environment():
            logger.info("Environment verification completed successfully")
        else:
            logger.error("Environment verification failed")
        return
    
    # Make sure the environment is valid
    if not verify_environment():
        logger.error("Environment verification failed. Fix issues before continuing.")
        return
    
    # Run GUI if requested
    if args.gui:
        run_gui()
        return
    
    # If image, video, or batch is specified, handle them
    if args.image or args.video or args.batch:
        # Create CLI args from known args
        cli_args = []
        if args.image:
            cli_args.extend(["--image", args.image])
        if args.video:
            cli_args.extend(["--video", args.video])
        if args.batch:
            cli_args.extend(["--batch", args.batch])
        if args.threshold:
            cli_args.extend(["--threshold", str(args.threshold)])
        if args.output:
            cli_args.extend(["--output", args.output])
        if args.format:
            cli_args.extend(["--format", args.format])
        
        # Run CLI with these args
        run_cli(cli_args)
        return
    
    # Otherwise, pass remaining args to CLI
    run_cli(remaining)

if __name__ == "__main__":
    main() 