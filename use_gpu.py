import os
import tensorflow as tf

# Print important environment variables
print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Set explicitly
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Print TensorFlow version
print(f"\nTensorFlow version: {tf.__version__}")

# Check for GPU
print("\nGPU information:")
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Found {len(physical_devices)} GPU(s)")
    for device in physical_devices:
        print(f"  Device: {device}")
        # Enable memory growth
        try:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"  Memory growth enabled")
        except Exception as e:
            print(f"  Error setting memory growth: {e}")
else:
    print("No GPUs found, using CPU")

# Create a simple TensorFlow operation to force device placement
print("\nRunning a simple operation:")
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print(f"  Matrix multiplication result: {c}")
    print("  Operation completed on GPU")
except Exception as e:
    print(f"  Error running on GPU: {e}")
    print("  Falling back to CPU")
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    print(f"  Matrix multiplication result: {c}")

# Print CUDA build information
print("\nCUDA information:")
try:
    print(f"  Built with CUDA: {tf.test.is_built_with_cuda()}")
except Exception as e:
    print(f"  Error checking CUDA build: {e}")

try:
    print(f"  GPU available: {tf.test.is_gpu_available()}")
except Exception as e:
    print(f"  Error checking GPU availability: {e}")

print("\nTo use your NVIDIA GPU with TensorFlow, you need to:")
print("1. Install NVIDIA CUDA Toolkit (11.2 for TensorFlow 2.10)")
print("2. Install cuDNN (8.1 for TensorFlow 2.10)")
print("3. Add CUDA bin directory to PATH")
print("4. Install TensorFlow with GPU support (pip install tensorflow==2.10)")
print("5. Restart your computer after installation") 