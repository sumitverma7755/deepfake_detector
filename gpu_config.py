#!/usr/bin/env python3
"""
GPU Configuration for TensorFlow
"""

import os
import sys
import logging
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def configure_gpu():
    """
    Configure GPU for TensorFlow to avoid memory errors and enable GPU usage.
    
    Returns:
        bool: True if GPU is available and configured, False otherwise
    """
    try:
        # Clear any existing GPU memory settings
        tf.keras.backend.clear_session()
        
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            logger.warning("No GPU detected. Using CPU for computation.")
            # Set CUDA settings to disable GPU so TF doesn't show errors
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            return False
        
        logger.info(f"Found {len(gpus)} GPU device(s):")
        for gpu in gpus:
            logger.info(f"  - {gpu.name}")
        
        # Configure TensorFlow to use GPU memory efficiently
        for gpu in gpus:
            try:
                # Enable memory growth (allocate only what's needed)
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Memory growth enabled for {gpu.name}")
            except:
                # If memory growth can't be enabled, limit memory
                logger.warning(f"Memory growth could not be enabled for {gpu.name}")
                # Limit to 80% of GPU memory
                virtual_gpus = [
                    tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 14)  # 14GB limit (or 80% of available)
                ]
                try:
                    tf.config.set_logical_device_configuration(gpu, virtual_gpus)
                    logger.info(f"Memory limited to 14GB for {gpu.name}")
                except:
                    # If we can't limit memory, log the error
                    logger.warning(f"Failed to set memory limit for {gpu.name}")
                    pass

        # Set environment variables for optimal performance
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # Optimize thread usage
        
        # Set mixed precision policy
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info(f"Mixed precision policy set to {policy.name}")
        
        # Verify GPU is working
        if is_gpu_working():
            logger.info("GPU configured successfully.")
            return True
        else:
            logger.warning("GPU configuration failed or incorrect versions.")
            return False
        
    except Exception as e:
        logger.error(f"Error configuring GPU: {e}")
        # Fallback to CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return False

def is_gpu_working():
    """
    Test if GPU computation actually works.
    
    Returns:
        bool: True if GPU computation works, False otherwise
    """
    try:
        # Create a simple test computation
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            result = c.numpy()
        
        # If we get here, the computation worked
        return True
    except:
        # If any error occurred, the GPU is not working properly
        return False

def get_device_strategy():
    """
    Get the appropriate distribution strategy based on available devices.
    
    Returns:
        tf.distribute.Strategy: The distribution strategy to use
    """
    try:
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            # For single or multiple GPUs
            if len(gpus) > 1:
                logger.info(f"Using MirroredStrategy for {len(gpus)} GPUs")
                return tf.distribute.MirroredStrategy()
            else:
                logger.info("Using OneDeviceStrategy with GPU")
                return tf.distribute.OneDeviceStrategy(device="/GPU:0")
        else:
            # Fallback to CPU
            logger.info("Using default strategy with CPU")
            return tf.distribute.get_strategy()
            
    except Exception as e:
        logger.error(f"Error creating distribution strategy: {e}")
        # Fallback to default strategy
        return tf.distribute.get_strategy()

# For direct testing
if __name__ == "__main__":
    # Configure GPU
    gpu_available = configure_gpu()
    
    # Print if GPU is available
    if gpu_available:
        print("GPU is available and configured.")
    else:
        print("No GPU available. Using CPU.")
    
    # Get distribution strategy
    strategy = get_device_strategy()
    print(f"Using distribution strategy: {strategy.__class__.__name__}") 