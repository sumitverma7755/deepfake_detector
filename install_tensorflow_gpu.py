#!/usr/bin/env python3
"""
Script to install TensorFlow with GPU support
"""
import os
import sys
import subprocess
import platform

def run_command(command):
    """Run a command and print output"""
    print(f"\nRunning: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(e.stderr)
        return False

def main():
    """Main function"""
    print("=" * 70)
    print("TensorFlow GPU Installation Helper".center(70))
    print("=" * 70)

    # Check Python version
    print(f"\nPython version: {platform.python_version()}")
    
    # Uninstall existing TensorFlow
    print("\nUninstalling existing TensorFlow installations...")
    run_command(f"{sys.executable} -m pip uninstall -y tensorflow tensorflow-gpu")
    
    # Install compatible TensorFlow version
    # Use the latest version (2.19.0) for Python 3.12 compatibility
    print("\nInstalling TensorFlow 2.19.0 (latest version)...")
    run_command(f"{sys.executable} -m pip install tensorflow==2.19.0")
    
    # Verify installation
    print("\nVerifying TensorFlow installation...")
    verify_cmd = """
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
try:
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
except:
    print("Could not determine if built with CUDA")
"""
    run_command(f"{sys.executable} -c \"{verify_cmd}\"")
    
    print("\n" + "=" * 70)
    print("Installation Complete".center(70))
    print("=" * 70)
    
    print("\nIf no GPUs were found, please ensure you have:")
    print("1. NVIDIA GPU drivers installed")
    print("2. CUDA Toolkit 11.2 installed")
    print("3. cuDNN 8.1 for CUDA 11.2 installed")
    print("4. Added CUDA bin directory to your PATH")
    
    print("\nRestart your computer after installing all components.")
    print("Then run 'python verify_gpu.py' to check your GPU setup.")

if __name__ == "__main__":
    main() 