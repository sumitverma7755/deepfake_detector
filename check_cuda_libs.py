import os
import sys
import glob
import tensorflow as tf

print("=" * 70)
print("TensorFlow CUDA Dependency Check".center(70))
print("=" * 70)

# Print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Check TensorFlow build info
build_info = tf.sysconfig.get_build_info()
print("\nTensorFlow Build Info:")
for key, value in build_info.items():
    if "cuda" in key.lower() or "cudnn" in key.lower():
        print(f"  {key}: {value}")

# Check CUDA environment variables
print("\nCUDA Environment Variables:")
print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")

# Check for cudart64_11.dll in PATH
paths = os.environ.get('PATH', '').split(os.pathsep)
cudart_found = False
for path in paths:
    cudart_path = os.path.join(path, 'cudart64_11.dll')
    if os.path.exists(cudart_path):
        cudart_found = True
        print(f"\nFound cudart64_11.dll at: {cudart_path}")
        break

if not cudart_found:
    print("\nCould not find cudart64_11.dll in PATH")

# Check for TensorFlow CUDA libraries
tf_lib_dir = os.path.dirname(tf.__file__)
print(f"\nTensorFlow library directory: {tf_lib_dir}")

# Look for CUDA libs in TensorFlow directory
cuda_libs = glob.glob(os.path.join(tf_lib_dir, "**", "*cuda*"), recursive=True)
print("\nCUDA-related files in TensorFlow directory:")
for lib in cuda_libs[:10]:  # Limit to 10 to avoid too much output
    print(f"  {lib}")

if len(cuda_libs) > 10:
    print(f"  ... and {len(cuda_libs) - 10} more")

# Check available TensorFlow devices
print("\nTensorFlow Devices:")
try:
    devices = tf.config.list_physical_devices()
    for device in devices:
        print(f"  {device.device_type}: {device.name}")
except Exception as e:
    print(f"  Error listing devices: {e}")

print("\n" + "=" * 70)
print("Check Complete".center(70))
print("=" * 70)

# Print additional help information
print("\nTo fix CUDA issues:")
print("1. Make sure CUDA 11.2 and cuDNN 8.1 are properly installed")
print("2. Ensure cudart64_11.dll is present in C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin")
print("3. If the file is missing, you may need to reinstall CUDA Toolkit 11.2")
print("4. Restart your computer after fixing these issues") 