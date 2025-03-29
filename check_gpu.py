import tensorflow as tf
import os
import sys

# Redirect output to a file
output_file = "gpu_check_results.txt"
original_stdout = sys.stdout
with open(output_file, 'w') as f:
    sys.stdout = f
    
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i+1}: {gpu}")
        
        # Enable memory growth to avoid taking all GPU memory
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  Enabled memory growth for {gpu}")
            except Exception as e:
                print(f"  Error setting memory growth: {e}")
    else:
        print("No GPUs found. Using CPU.")
    
    # Check if TensorFlow was built with CUDA
    try:
        print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    except Exception as e:
        print(f"Error checking CUDA build: {e}")
    
    # Check CUDA environment variables
    print("\nCUDA Environment Variables:")
    for var in ['CUDA_PATH', 'CUDA_HOME', 'PATH']:
        value = os.environ.get(var, 'Not set')
        if var == 'PATH':
            path_entries = value.split(os.pathsep)
            cuda_entries = [p for p in path_entries if 'cuda' in p.lower()]
            if cuda_entries:
                print(f"  {var}: Contains {len(cuda_entries)} CUDA entries")
                for entry in cuda_entries:
                    print(f"    - {entry}")
            else:
                print(f"  {var}: No CUDA entries found")
        else:
            print(f"  {var}: {value}")
    
    print("\nTo use GPU, you need:")
    print("1. NVIDIA GPU drivers installed")
    print("2. CUDA Toolkit installed (compatible with TensorFlow)")
    print("3. cuDNN installed")
    print("4. TensorFlow built with CUDA support")

# Reset stdout
sys.stdout = original_stdout
print(f"Results written to {output_file}")
print("Run 'type gpu_check_results.txt' to view the results") 