import os
import tensorflow as tf
import numpy as np

print("=" * 70)
print("TensorFlow GPU Verification".center(70))
print("=" * 70)

# Print TensorFlow version
print(f"\nTensorFlow version: {tf.__version__}")

# Print CUDA environment variables
print("\nCUDA Environment Variables:")
print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")

# Check for GPU devices
physical_devices = tf.config.list_physical_devices('GPU')
print(f"\nAvailable GPUs: {len(physical_devices)}")
for i, device in enumerate(physical_devices):
    print(f"  GPU {i}: {device.name}")

# If no GPUs are found
if not physical_devices:
    print("\nNo GPUs found. Possible issues:")
    print("1. CUDA/cuDNN not properly installed")
    print("2. TensorFlow not built with CUDA support")
    print("3. GPU drivers not properly installed")
    print("4. Environment variables not set correctly")
else:
    # Run a simple GPU test
    print("\nRunning a simple GPU test...")
    
    # Create some random data
    x = tf.random.normal([1000, 1000])
    y = tf.random.normal([1000, 1000])
    
    # Time the matrix multiplication
    import time
    start_time = time.time()
    
    # Run operations on GPU
    z = tf.matmul(x, y)
    
    # Force execution and measure time
    _ = z.numpy()
    end_time = time.time()
    
    print(f"Matrix multiplication completed in {end_time - start_time:.4f} seconds")
    print("GPU test successful!")

print("\n" + "=" * 70)
print("Verification Complete".center(70))
print("=" * 70) 