#!/usr/bin/env python3
"""
Script to help install CUDA Toolkit 11.2 and cuDNN 8.1
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
    print("CUDA and cuDNN Installation Helper".center(70))
    print("=" * 70)

    # CUDA Toolkit 11.2 download URL
    cuda_url = "https://developer.nvidia.com/downloads/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.89_win10.exe"
    cuda_installer = "cuda_11.2.0_460.89_win10.exe"

    # cuDNN 8.1 download URL (requires NVIDIA account)
    cudnn_url = "https://developer.nvidia.com/downloads/compute/cudnn/secure/8.1.0.77/local_installers/11.2/cudnn-windows-x86_64-8.1.0.77_cuda11-archive.zip"
    cudnn_zip = "cudnn-windows-x86_64-8.1.0.77_cuda11-archive.zip"

    print("\nPlease follow these steps to install CUDA and cuDNN:")
    print("\n1. Download CUDA Toolkit 11.2:")
    print(f"   - Visit: {cuda_url}")
    print("   - Download and run the installer")
    print("   - Choose 'Express' installation")
    print("\n2. Download cuDNN 8.1:")
    print(f"   - Visit: {cudnn_url}")
    print("   - You'll need to log in with your NVIDIA account")
    print("   - Download the Windows x86_64 version for CUDA 11.x")
    print("\n3. After downloading:")
    print("   - Run the CUDA installer first")
    print("   - Extract the cuDNN zip file")
    print("   - Copy the contents from the cuDNN bin, include, and lib folders")
    print("   - Paste them into the corresponding folders in your CUDA installation")
    print("   - Default CUDA installation path: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2")
    
    print("\n4. Add CUDA to your PATH:")
    print("   - Open System Properties > Advanced > Environment Variables")
    print("   - Under System Variables, find and select 'Path'")
    print("   - Click 'Edit' and add these paths:")
    print("     C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin")
    print("     C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\libnvvp")
    
    print("\n5. After installation:")
    print("   - Restart your computer")
    print("   - Run 'python verify_gpu.py' to check your GPU setup")

if __name__ == "__main__":
    main() 