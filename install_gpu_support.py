#!/usr/bin/env python3
"""
NVIDIA GPU Support Installation Script
This script helps set up TensorFlow with NVIDIA GPU support.
"""

import subprocess
import sys
import os
import platform

def run_command(command):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error message: {e.stderr}")
        return None

def check_nvidia_drivers():
    """Check if NVIDIA drivers are installed"""
    print("\nChecking for NVIDIA drivers...")
    
    if platform.system() == "Windows":
        result = run_command("nvidia-smi")
        if result is None:
            print("❌ NVIDIA drivers not found or not working properly")
            print("   Please download and install drivers from: https://www.nvidia.com/Download/index.aspx")
            return False
        print("✅ NVIDIA drivers found:")
        print(result.split("\n")[0])
        return True
    else:
        # Linux/Mac check
        result = run_command("which nvidia-smi")
        if result is None or result.strip() == "":
            print("❌ NVIDIA drivers not found")
            return False
        result = run_command("nvidia-smi")
        print("✅ NVIDIA drivers found:")
        print(result.split("\n")[0])
        return True

def check_cuda():
    """Check if CUDA is installed"""
    print("\nChecking for CUDA installation...")
    
    if platform.system() == "Windows":
        # Check CUDA_PATH environment variable
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path and os.path.exists(cuda_path):
            print(f"✅ CUDA found at: {cuda_path}")
            return True
        
        # Check common installation paths
        common_paths = [
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
            "C:\\CUDA",
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                print(f"✅ CUDA found at: {path}")
                print("   NOTE: CUDA_PATH environment variable not set!")
                return True
        
        print("❌ CUDA not found")
        print("   Please download and install CUDA from: https://developer.nvidia.com/cuda-downloads")
        return False
    else:
        # Linux/Mac check
        result = run_command("nvcc --version")
        if result is None:
            print("❌ CUDA not found")
            return False
        print("✅ CUDA found:")
        print(result.split("\n")[0])
        return True

def check_cudnn():
    """Check if cuDNN is installed"""
    print("\nChecking for cuDNN installation...")
    
    if platform.system() == "Windows":
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            cudnn_path = os.path.join(cuda_path, "include", "cudnn.h")
            if os.path.exists(cudnn_path):
                print(f"✅ cuDNN found at: {cudnn_path}")
                return True
        
        print("❌ cuDNN not found or not properly installed")
        print("   Please download and install cuDNN from: https://developer.nvidia.com/cudnn")
        print("   Make sure to place it in your CUDA directory")
        return False
    else:
        # Linux check - simplified
        result = run_command("find /usr -name cudnn.h")
        if result is None or result.strip() == "":
            print("❌ cuDNN not found")
            return False
        print(f"✅ cuDNN found at: {result.strip()}")
        return True

def check_tensorflow():
    """Check TensorFlow and GPU support"""
    print("\nChecking TensorFlow installation...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} installed")
        
        print("\nChecking TensorFlow GPU support...")
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print(f"✅ TensorFlow can see {len(gpu_devices)} GPU(s):")
            for i, device in enumerate(gpu_devices):
                print(f"   GPU {i+1}: {device}")
            return True
        else:
            print("❌ TensorFlow cannot detect GPUs")
            
            # Check if built with CUDA
            try:
                if tf.test.is_built_with_cuda():
                    print("   TensorFlow is built with CUDA but cannot find GPUs")
                else:
                    print("   TensorFlow is NOT built with CUDA support")
            except:
                print("   Could not determine if TensorFlow is built with CUDA")
            
            return False
    except ImportError:
        print("❌ TensorFlow not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking TensorFlow: {e}")
        return False

def install_tensorflow_gpu():
    """Install TensorFlow with GPU support"""
    print("\nInstalling TensorFlow with GPU support...")
    
    # First uninstall existing TensorFlow
    print("Uninstalling existing TensorFlow installation...")
    run_command(f"{sys.executable} -m pip uninstall -y tensorflow tensorflow-gpu")
    
    # Install TensorFlow with GPU support
    print("Installing TensorFlow with GPU support...")
    result = run_command(f"{sys.executable} -m pip install tensorflow")
    
    if result:
        print("✅ TensorFlow with GPU support installed")
        return True
    else:
        print("❌ Failed to install TensorFlow with GPU support")
        return False

def main():
    """Main function"""
    print("=" * 70)
    print("NVIDIA GPU Support Setup for TensorFlow".center(70))
    print("=" * 70)
    
    # Check system configuration
    drivers_ok = check_nvidia_drivers()
    cuda_ok = check_cuda()
    cudnn_ok = check_cudnn()
    tf_ok = check_tensorflow()
    
    print("\n" + "=" * 70)
    print("System Configuration Summary".center(70))
    print("=" * 70)
    print(f"NVIDIA Drivers: {'✅ Installed' if drivers_ok else '❌ Not found'}")
    print(f"CUDA Toolkit: {'✅ Installed' if cuda_ok else '❌ Not found'}")
    print(f"cuDNN Library: {'✅ Installed' if cudnn_ok else '❌ Not found or not detected'}")
    print(f"TensorFlow GPU: {'✅ Working' if tf_ok else '❌ Not working'}")
    
    if not (drivers_ok and cuda_ok and cudnn_ok and tf_ok):
        print("\n" + "=" * 70)
        print("Recommendations".center(70))
        print("=" * 70)
        
        if not drivers_ok:
            print("1. Install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx")
        
        if not cuda_ok:
            print("2. Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads")
            print("   Make sure to add CUDA to your PATH environment variable")
        
        if not cudnn_ok:
            print("3. Install cuDNN from: https://developer.nvidia.com/cudnn")
            print("   (Requires NVIDIA Developer account)")
        
        if not tf_ok and drivers_ok and cuda_ok and cudnn_ok:
            print("\nWould you like to reinstall TensorFlow with GPU support? (y/n)")
            choice = input("> ").strip().lower()
            if choice == 'y':
                install_tensorflow_gpu()
                print("\nPlease restart your application to use GPU acceleration")
            else:
                print("Skipping TensorFlow reinstallation")
    else:
        print("\n✅ Your system is properly configured for TensorFlow with GPU support!")
        
    print("\nFor detailed TensorFlow GPU setup instructions, visit:")
    print("https://www.tensorflow.org/install/gpu")

if __name__ == "__main__":
    main() 