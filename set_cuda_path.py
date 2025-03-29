#!/usr/bin/env python3
"""
Script to set up CUDA environment variables
"""
import os
import sys
import subprocess
import winreg

def set_cuda_path():
    """Set CUDA environment variables in Windows registry"""
    cuda_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2"
    
    # Set CUDA_PATH in system environment variables
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment", 0, winreg.KEY_ALL_ACCESS)
        winreg.SetValueEx(key, "CUDA_PATH", 0, winreg.REG_EXPAND_SZ, cuda_path)
        winreg.SetValueEx(key, "CUDA_HOME", 0, winreg.REG_EXPAND_SZ, cuda_path)
        
        # Get current PATH
        path_value = winreg.QueryValueEx(key, "Path")[0]
        
        # Add CUDA paths if not already present
        cuda_bin = os.path.join(cuda_path, "bin")
        cuda_libnvvp = os.path.join(cuda_path, "libnvvp")
        
        if cuda_bin not in path_value:
            path_value = cuda_bin + ";" + path_value
        if cuda_libnvvp not in path_value:
            path_value = cuda_libnvvp + ";" + path_value
            
        winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, path_value)
        winreg.CloseKey(key)
        
        print("Successfully set CUDA environment variables:")
        print(f"CUDA_PATH: {cuda_path}")
        print(f"CUDA_HOME: {cuda_path}")
        print(f"Added to PATH: {cuda_bin}")
        print(f"Added to PATH: {cuda_libnvvp}")
        
        print("\nPlease restart your computer for the changes to take effect.")
        
    except Exception as e:
        print(f"Error setting environment variables: {e}")
        print("Please run this script as administrator.")

def verify_cuda_path():
    """Verify CUDA installation and paths"""
    cuda_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2"
    
    print("\nVerifying CUDA installation...")
    
    # Check if CUDA directory exists
    if not os.path.exists(cuda_path):
        print(f"Error: CUDA directory not found at {cuda_path}")
        return False
    
    # Check for essential CUDA files
    required_files = [
        os.path.join(cuda_path, "bin", "nvcc.exe"),
        os.path.join(cuda_path, "bin", "cudart64_11.dll"),
        os.path.join(cuda_path, "include", "cuda_runtime.h")
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Missing required CUDA files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nWARNING: Some required files are missing, but we'll set the path anyway.")
        return True  # Return True anyway to proceed with setting the path
    
    print("CUDA installation verified successfully!")
    return True

def main():
    """Main function"""
    print("=" * 70)
    print("CUDA Path Setup".center(70))
    print("=" * 70)
    
    # Always set the path regardless of verification
    if os.path.exists("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2"):
        verify_cuda_path()  # Just for information
        set_cuda_path()
    else:
        print("\nError: CUDA 11.2 directory not found.")
        print("Please ensure CUDA 11.2 is installed before setting the path.")
        print("You can download it from:")
        print("https://developer.nvidia.com/cuda-11.2.0-download-archive")

if __name__ == "__main__":
    main() 