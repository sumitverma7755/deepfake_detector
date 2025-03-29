#!/usr/bin/env python3
"""
GPU Verification Tool

A simple script to verify GPU setup and display diagnostic information for TensorFlow.
"""

import os
import sys
import platform
import subprocess
import time

def print_heading(text, char='=', width=80):
    """Print a heading with decorative characters"""
    print("\n" + char * width)
    print(text.center(width))
    print(char * width + "\n")

def print_success(text):
    """Print a success message with green color if supported"""
    try:
        # Check if terminal supports colors
        if sys.stdout.isatty():
            if platform.system() == 'Windows':
                # Windows
                os.system('color')
                print("\033[92m✓ " + text + "\033[0m")
            else:
                # Unix/Mac
                print("\033[92m✓ " + text + "\033[0m")
        else:
            print("✓ " + text)
    except:
        print("✓ " + text)

def print_error(text):
    """Print an error message with red color if supported"""
    try:
        # Check if terminal supports colors
        if sys.stdout.isatty():
            if platform.system() == 'Windows':
                # Windows
                os.system('color')
                print("\033[91m✗ " + text + "\033[0m")
            else:
                # Unix/Mac
                print("\033[91m✗ " + text + "\033[0m")
        else:
            print("✗ " + text)
    except:
        print("✗ " + text)

def print_warning(text):
    """Print a warning message with yellow color if supported"""
    try:
        # Check if terminal supports colors
        if sys.stdout.isatty():
            if platform.system() == 'Windows':
                # Windows
                os.system('color')
                print("\033[93m! " + text + "\033[0m")
            else:
                # Unix/Mac
                print("\033[93m! " + text + "\033[0m")
        else:
            print("! " + text)
    except:
        print("! " + text)

def check_nvidia_driver():
    """Check if NVIDIA driver is installed"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except FileNotFoundError:
        return False, "NVIDIA driver not found or 'nvidia-smi' not in PATH"
    except Exception as e:
        return False, str(e)

def check_tensorflow_gpu():
    """Check if TensorFlow can detect and use the GPU"""
    try:
        import tensorflow as tf
        
        # Print TensorFlow version
        print(f"TensorFlow version: {tf.__version__}")
        
        # List physical devices
        physical_devices = tf.config.list_physical_devices()
        gpu_devices = tf.config.list_physical_devices('GPU')
        
        print(f"Physical devices detected: {len(physical_devices)}")
        for device in physical_devices:
            print(f"  - {device.device_type}: {device.name}")
        
        # If GPUs detected, try a simple operation
        if gpu_devices:
            print(f"\nGPU devices detected: {len(gpu_devices)}")
            for device in gpu_devices:
                print(f"  - {device.name}")
            
            # Try to enable memory growth to avoid allocating all memory
            for device in gpu_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except:
                    print_warning(f"Could not enable memory growth for {device.name}")
            
            # Try a simple operation
            print("\nTesting GPU with matrix multiplication...")
            with tf.device('/GPU:0'):
                # Create two random matrices
                a = tf.random.normal([5000, 5000])
                b = tf.random.normal([5000, 5000])
                
                # Warm-up run
                tf.matmul(a, b)
                
                # Timed run
                start_time = time.time()
                c = tf.matmul(a, b)
                # Force execution
                c_result = c.numpy()
                end_time = time.time()
            
            print(f"Matrix multiplication completed in {end_time - start_time:.4f} seconds")
            return True, gpu_devices
        else:
            print("\nNo GPU devices detected by TensorFlow")
            return False, []
            
    except ImportError:
        return False, "TensorFlow not installed. Install with: pip install tensorflow"
    except Exception as e:
        return False, str(e)

def check_tensorflow_cpu():
    """Run CPU test for comparison if GPU test was successful"""
    try:
        import tensorflow as tf
        import time
        
        print("\nTesting CPU with matrix multiplication for comparison...")
        with tf.device('/CPU:0'):
            # Create two random matrices (smaller because CPU is slower)
            a = tf.random.normal([5000, 5000])
            b = tf.random.normal([5000, 5000])
            
            # Warm-up run
            tf.matmul(a, b)
            
            # Timed run
            start_time = time.time()
            c = tf.matmul(a, b)
            # Force execution
            c_result = c.numpy()
            end_time = time.time()
        
        print(f"CPU matrix multiplication completed in {end_time - start_time:.4f} seconds")
        return True
    except Exception as e:
        print(f"CPU test failed: {e}")
        return False

def print_summary(nvidia_ok, nvidia_output, tf_gpu_ok, tf_gpu_devices):
    """Print a summary of the GPU check results"""
    print_heading("SUMMARY", "=")
    
    if nvidia_ok:
        print_success("NVIDIA driver is installed and working")
    else:
        print_error(f"NVIDIA driver issue: {nvidia_output}")
    
    if tf_gpu_ok:
        print_success(f"TensorFlow detected {len(tf_gpu_devices)} GPU(s)")
    else:
        if isinstance(tf_gpu_devices, str):
            print_error(f"TensorFlow GPU issue: {tf_gpu_devices}")
        else:
            print_error("TensorFlow could not detect any GPUs")
    
    # Overall result
    if nvidia_ok and tf_gpu_ok:
        print_heading("GPU SETUP SUCCESSFUL", "=")
        print("Your system is correctly configured for TensorFlow with GPU support.\n")
        print("Run the deepfake detector with:")
        print("  python run_with_gpu.py --gui")
    else:
        print_heading("GPU SETUP NEEDS ATTENTION", "=")
        print("Your GPU configuration needs attention. Please run the FixCuda.py script:")
        print("  python FixCuda.py")
        print("\nOr follow the manual setup instructions in NVIDIA_GPU_SETUP.md")

def main():
    """Main function"""
    print_heading("GPU VERIFICATION TOOL", "=")
    
    # Print system information
    print(f"System: {platform.system()} {platform.release()} {platform.version()}")
    print(f"Python: {platform.python_version()}")
    print(f"Processor: {platform.processor()}")
    
    # Check NVIDIA driver
    print_heading("CHECKING NVIDIA DRIVER", "-")
    nvidia_ok, nvidia_output = check_nvidia_driver()
    
    if nvidia_ok:
        print_success("NVIDIA driver is installed and functioning")
        print("\nDriver information:")
        # Print selective information from nvidia-smi
        lines = nvidia_output.split('\n')
        for line in lines[:15]:  # Print the first few lines
            if any(keyword in line for keyword in ['NVIDIA-SMI', 'Driver Version', 'CUDA Version', '|==========', '| GPU', '| Processes']):
                print(line)
    else:
        print_error(f"NVIDIA driver issue: {nvidia_output}")
        print("\nPossible solutions:")
        print("1. Install NVIDIA GPU drivers from https://www.nvidia.com/Download/index.aspx")
        print("2. Ensure the GPU is properly connected and recognized by the system")
        print("3. Check if 'nvidia-smi' is in your PATH")
    
    # Check TensorFlow GPU support
    print_heading("CHECKING TENSORFLOW GPU SUPPORT", "-")
    tf_gpu_ok, tf_gpu_devices = check_tensorflow_gpu()
    
    # If GPU works, run a CPU test for comparison
    if tf_gpu_ok:
        check_tensorflow_cpu()
    
    # Print summary
    print_summary(nvidia_ok, nvidia_output, tf_gpu_ok, tf_gpu_devices)

if __name__ == "__main__":
    main() 