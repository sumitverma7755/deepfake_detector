import tensorflow as tf
import numpy as np
import time

print("=" * 70)
print("GPU Performance Test".center(70))
print("=" * 70)

# Get GPU information
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("No GPU found!")
    exit(1)

print(f"\nGPU Information:")
for gpu in gpus:
    print(f"  {gpu.name}")

# Test 1: Matrix Multiplication
print("\nTest 1: Matrix Multiplication")
sizes = [1000, 2000, 4000]
for size in sizes:
    print(f"\nMatrix size: {size}x{size}")
    
    # Create random matrices
    a = tf.random.normal([size, size])
    b = tf.random.normal([size, size])
    
    # Warm-up run
    _ = tf.matmul(a, b)
    
    # Actual test
    start_time = time.time()
    c = tf.matmul(a, b)
    _ = c.numpy()  # Force execution
    end_time = time.time()
    
    print(f"  Time: {end_time - start_time:.4f} seconds")
    print(f"  Operations per second: {2 * size**3 / (end_time - start_time):.2f}")

# Test 2: Neural Network Forward Pass
print("\nTest 2: Neural Network Forward Pass")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu', input_shape=(1000,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Create random input data
batch_sizes = [32, 64, 128]
x = tf.random.normal([max(batch_sizes), 1000])
y = tf.random.normal([max(batch_sizes), 10])

print("\nNeural Network Forward Pass:")
for batch_size in batch_sizes:
    print(f"\nBatch size: {batch_size}")
    
    # Warm-up run
    _ = model(x[:batch_size])
    
    # Actual test
    start_time = time.time()
    _ = model(x[:batch_size])
    end_time = time.time()
    
    print(f"  Time: {end_time - start_time:.4f} seconds")
    print(f"  Samples per second: {batch_size / (end_time - start_time):.2f}")

# Test 3: Convolution Operation
print("\nTest 3: Convolution Operation")
input_shape = (1, 224, 224, 3)
kernel_size = (3, 3)
filters = 64

# Create random input and kernel
x = tf.random.normal(input_shape)
kernel = tf.random.normal([*kernel_size, input_shape[-1], filters])

# Warm-up run
_ = tf.nn.conv2d(x, kernel, strides=1, padding='SAME')

# Actual test
start_time = time.time()
_ = tf.nn.conv2d(x, kernel, strides=1, padding='SAME')
end_time = time.time()

print(f"\nConvolution Operation:")
print(f"  Input shape: {input_shape}")
print(f"  Kernel size: {kernel_size}")
print(f"  Number of filters: {filters}")
print(f"  Time: {end_time - start_time:.4f} seconds")

print("\n" + "=" * 70)
print("Test Complete".center(70))
print("=" * 70) 