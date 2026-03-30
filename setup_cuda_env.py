#!/usr/bin/env python3
"""
Script to set up CUDA environment variables and verify the setup
"""
import os
import sys
import subprocess
import platform
import shutil

def check_cuda_installation():
    """Check if CUDA is properly installed"""
    cuda_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2"
    if not os.path.exists(cuda_path):
        print(f"CUDA 11.2 not found at {cuda_path}")
        print("Please install CUDA Toolkit 11.2 from:")
        print("https://developer.nvidia.com/cuda-11.2.0-download-archive")
        return False
    return True

def check_cudnn_installation():
    """Check if cuDNN is properly installed"""
    cuda_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2"
    cudnn_files = [
        os.path.join(cuda_path, "bin", "cudnn64_8.dll"),
        os.path.join(cuda_path, "include", "cudnn.h"),
        os.path.join(cuda_path, "lib", "x64", "cudnn.lib")
    ]
    
    missing_files = [f for f in cudnn_files if not os.path.exists(f)]
    if missing_files:
        print("Missing cuDNN files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease install cuDNN 8.1 from:")
        print("https://developer.nvidia.com/cudnn")
        print("And copy the files to the corresponding CUDA directories")
        return False
    return True

def setup_environment_variables():
    """Set up CUDA environment variables"""
    cuda_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2"
    
    # Set CUDA environment variables
    os.environ["CUDA_PATH"] = cuda_path
    os.environ["CUDA_HOME"] = cuda_path
    
    # Add CUDA directories to PATH
    cuda_bin = os.path.join(cuda_path, "bin")
    cuda_libnvvp = os.path.join(cuda_path, "libnvvp")
    
    if cuda_bin not in os.environ["PATH"]:
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ["PATH"]
    if cuda_libnvvp not in os.environ["PATH"]:
        os.environ["PATH"] = cuda_libnvvp + os.pathsep + os.environ["PATH"]
    
    print("Environment variables set:")
    print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    print(f"PATH includes CUDA: {'CUDA' in os.environ['PATH']}")

def main():
    """Main function"""
    print("=" * 70)
    print("CUDA Environment Setup".center(70))
    print("=" * 70)

    print("\nChecking CUDA installation...")
    if not check_cuda_installation():
        return
    
    print("\nChecking cuDNN installation...")
    if not check_cudnn_installation():
        return
    
    print("\nSetting up environment variables...")
    setup_environment_variables()
    
    print("\nVerifying NVIDIA GPU...")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("NVIDIA GPU detected:")
            print(result.stdout.split("\n")[0])
        else:
            print("Could not detect NVIDIA GPU")
    except:
        print("Could not run nvidia-smi")
    
    print("\n" + "=" * 70)
    print("Setup Complete".center(70))
    print("=" * 70)
    
    print("\nNext steps:")
    print("1. Restart your computer to apply environment variable changes")
    print("2. After restart, run 'python verify_gpu.py' to check GPU setup")
    print("3. If GPU is still not detected, try reinstalling TensorFlow:")
    print("   pip uninstall tensorflow tensorflow-intel")
    print("   pip install tensorflow==2.12.0")

if __name__ == "__main__":
    main() 