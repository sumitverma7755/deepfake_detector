#!/usr/bin/env python3
"""
Script to help set up GPU support for TensorFlow
"""
import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import shutil

def download_file(url, filename):
    """Download a file from URL"""
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")

def extract_zip(zip_path, extract_path):
    """Extract a zip file"""
    print(f"Extracting {zip_path} to {extract_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extracted to {extract_path}")

def main():
    """Main function"""
    print("=" * 70)
    print("TensorFlow GPU Setup Helper".center(70))
    print("=" * 70)

    # Check Python version
    print(f"\nPython version: {platform.python_version()}")
    
    # Create directories for CUDA and cuDNN
    cuda_dir = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"
    cudnn_dir = "C:\\Program Files\\NVIDIA\\CUDNN"
    
    print("\nChecking CUDA installation...")
    if not os.path.exists(cuda_dir):
        print("CUDA directory not found. Please install CUDA Toolkit 11.2")
        print("Download from: https://developer.nvidia.com/cuda-11.2.0-download-archive")
        return
    
    print("\nChecking cuDNN installation...")
    if not os.path.exists(cudnn_dir):
        print("cuDNN directory not found. Please install cuDNN 8.1")
        print("Download from: https://developer.nvidia.com/cudnn")
        return
    
    # Set environment variables
    print("\nSetting up environment variables...")
    os.environ["CUDA_PATH"] = os.path.join(cuda_dir, "v11.2")
    os.environ["CUDA_HOME"] = os.path.join(cuda_dir, "v11.2")
    os.environ["PATH"] = os.path.join(cuda_dir, "v11.2", "bin") + os.pathsep + os.environ["PATH"]
    
    # Install TensorFlow with GPU support
    print("\nInstalling TensorFlow with GPU support...")
    subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow==2.12.0"])
    
    # Verify installation
    print("\nVerifying installation...")
    verify_cmd = """
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
try:
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
except:
    print("Could not determine if built with CUDA")
"""
    subprocess.run([sys.executable, "-c", verify_cmd])
    
    print("\n" + "=" * 70)
    print("Setup Complete".center(70))
    print("=" * 70)
    
    print("\nPlease ensure you have:")
    print("1. NVIDIA GPU drivers installed")
    print("2. CUDA Toolkit 11.2 installed")
    print("3. cuDNN 8.1 for CUDA 11.2 installed")
    print("4. Added CUDA bin directory to your PATH")
    
    print("\nRestart your computer after installing all components.")
    print("Then run 'python verify_gpu.py' to check your GPU setup.")

if __name__ == "__main__":
    main() 