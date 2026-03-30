#!/usr/bin/env python3
"""
CUDA Configuration Fix Script

This script diagnoses and helps fix common CUDA setup issues for TensorFlow.
"""

import os
import sys
import ctypes
import subprocess
import platform
import re
import winreg
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cuda_fix.log')
    ]
)
logger = logging.getLogger(__name__)

def is_admin():
    """Check if the script is running with administrator privileges"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def test_import_tensorflow():
    """Test if TensorFlow can be imported and if it can see GPU devices"""
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow version: {tf.__version__}")
        
        # Check for GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"TensorFlow detected {len(gpus)} GPU(s):")
            for gpu in gpus:
                logger.info(f"  - {gpu.name}")
            return True
        else:
            logger.warning("TensorFlow did not detect any GPUs")
            return False
    except ImportError:
        logger.error("TensorFlow is not installed. Install it with: pip install tensorflow")
        return False
    except Exception as e:
        logger.error(f"Error testing TensorFlow: {e}")
        return False

def get_environment_path():
    """Get the current system PATH"""
    try:
        # Get the PATH environment variable
        path = os.environ.get('PATH', '')
        return path.split(os.pathsep)
    except Exception as e:
        logger.error(f"Error getting PATH: {e}")
        return []

def check_cuda_in_path(paths):
    """Check if CUDA is in the PATH"""
    cuda_paths = []
    for path in paths:
        if 'cuda' in path.lower() and os.path.exists(path):
            cuda_paths.append(path)
    
    if cuda_paths:
        logger.info("Found CUDA paths in system PATH:")
        for path in cuda_paths:
            logger.info(f"  - {path}")
        return cuda_paths
    else:
        logger.warning("No CUDA paths found in system PATH")
        return []

def find_installed_cuda_versions():
    """Find installed CUDA versions"""
    cuda_versions = []
    
    # Check common installation paths
    program_files = os.environ.get('ProgramFiles', 'C:\\Program Files')
    program_files_x86 = os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)')
    
    cuda_base_paths = [
        os.path.join(program_files, 'NVIDIA GPU Computing Toolkit', 'CUDA'),
        os.path.join(program_files_x86, 'NVIDIA GPU Computing Toolkit', 'CUDA')
    ]
    
    for base_path in cuda_base_paths:
        if os.path.exists(base_path):
            for item in os.listdir(base_path):
                version_path = os.path.join(base_path, item)
                if os.path.isdir(version_path) and item.startswith('v'):
                    cuda_versions.append((item.lstrip('v'), version_path))
    
    if cuda_versions:
        logger.info("Found installed CUDA versions:")
        for version, path in cuda_versions:
            logger.info(f"  - CUDA {version} at {path}")
        return cuda_versions
    else:
        logger.warning("No CUDA installations found in standard locations")
        return []

def check_nvidia_smi():
    """Check if nvidia-smi is available and what it reports"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            # Extract driver version from output
            match = re.search(r'Driver Version: (\d+\.\d+\.\d+)', result.stdout)
            driver_version = match.group(1) if match else "Unknown"
            
            # Extract CUDA version from output
            match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
            cuda_version = match.group(1) if match else "Unknown"
            
            logger.info(f"NVIDIA Driver Version: {driver_version}")
            logger.info(f"NVIDIA Driver CUDA Version: {cuda_version}")
            
            # Extract GPU information
            gpu_info = []
            lines = result.stdout.split('\n')
            recording = False
            for line in lines:
                if '| GPU' in line and 'Name' in line:
                    recording = True
                    continue
                if recording and '+' in line:
                    recording = False
                    break
                if recording and '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        gpu_info.append(parts[1].strip() + ' - ' + parts[2].strip())
            
            if gpu_info:
                logger.info("Detected GPUs:")
                for gpu in gpu_info:
                    logger.info(f"  - {gpu}")
            
            return True
        else:
            logger.warning("nvidia-smi command failed with error:")
            logger.warning(result.stderr)
            return False
    except FileNotFoundError:
        logger.warning("nvidia-smi not found. NVIDIA driver may not be installed.")
        return False
    except Exception as e:
        logger.error(f"Error running nvidia-smi: {e}")
        return False

def check_tensorflow_cuda_compatibility(tf_version):
    """Check if TensorFlow version is compatible with installed CUDA"""
    compatibility_map = {
        '2.10': {'cuda': '11.2', 'cudnn': '8.1'},
        '2.11': {'cuda': '11.2', 'cudnn': '8.1'},
        '2.12': {'cuda': '11.8', 'cudnn': '8.6'},
        '2.13': {'cuda': '11.8', 'cudnn': '8.6'},
        '2.14': {'cuda': '11.8', 'cudnn': '8.7'},
        '2.15': {'cuda': '12.1', 'cudnn': '8.9'},
        '2.16': {'cuda': '12.3', 'cudnn': '8.9'},
    }
    
    # Get the major.minor version
    major_minor = '.'.join(tf_version.split('.')[:2])
    
    if major_minor in compatibility_map:
        recommended = compatibility_map[major_minor]
        logger.info(f"TensorFlow {major_minor} is compatible with:")
        logger.info(f"  - CUDA {recommended['cuda']}")
        logger.info(f"  - cuDNN {recommended['cudnn']}")
        return recommended
    else:
        logger.warning(f"Unknown TensorFlow version: {tf_version}")
        logger.info("For latest compatibility information, visit: https://www.tensorflow.org/install/source#gpu")
        return None

def get_cudnn_version(cuda_path):
    """Try to detect cuDNN version in CUDA installation"""
    try:
        include_path = os.path.join(cuda_path, 'include', 'cudnn.h')
        if not os.path.exists(include_path):
            logger.warning(f"cuDNN header not found at {include_path}")
            return None
        
        with open(include_path, 'r') as f:
            content = f.read()
        
        # Look for version information
        major = re.search(r'#define CUDNN_MAJOR (\d+)', content)
        minor = re.search(r'#define CUDNN_MINOR (\d+)', content)
        patch = re.search(r'#define CUDNN_PATCHLEVEL (\d+)', content)
        
        if major and minor and patch:
            version = f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}"
            logger.info(f"Found cuDNN version {version} at {cuda_path}")
            return version
        else:
            logger.warning(f"Could not parse cuDNN version from {include_path}")
            return None
    except Exception as e:
        logger.error(f"Error detecting cuDNN version: {e}")
        return None

def add_cuda_to_path(cuda_path):
    """Add CUDA paths to system PATH (requires admin)"""
    if not is_admin():
        logger.error("Administrator privileges required to modify PATH")
        return False
    
    try:
        bin_path = os.path.join(cuda_path, 'bin')
        libnvvp_path = os.path.join(cuda_path, 'libnvvp')
        cupti_path = os.path.join(cuda_path, 'extras', 'CUPTI', 'lib64')
        
        paths_to_add = []
        
        if os.path.exists(bin_path):
            paths_to_add.append(bin_path)
        
        if os.path.exists(libnvvp_path):
            paths_to_add.append(libnvvp_path)
        
        if os.path.exists(cupti_path):
            paths_to_add.append(cupti_path)
        
        if not paths_to_add:
            logger.warning(f"No valid CUDA paths found in {cuda_path}")
            return False
        
        # Get current PATH
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 'SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment', 0, winreg.KEY_ALL_ACCESS)
        current_path, _ = winreg.QueryValueEx(key, 'Path')
        
        # Make a list of paths
        path_list = current_path.split(';')
        path_list = [p for p in path_list if p and p not in paths_to_add]
        
        # Add new paths
        path_list.extend(paths_to_add)
        
        # Write back to registry
        new_path = ';'.join(path_list)
        winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, new_path)
        winreg.CloseKey(key)
        
        # Update environment for current process
        os.environ['PATH'] = new_path
        
        logger.info("CUDA paths added to system PATH:")
        for path in paths_to_add:
            logger.info(f"  - {path}")
        
        logger.warning("Please restart your computer for the PATH changes to take effect")
        return True
    except Exception as e:
        logger.error(f"Error modifying PATH: {e}")
        return False

def create_batch_script(cuda_path):
    """Create a batch script to add CUDA to path temporarily"""
    try:
        bin_path = os.path.join(cuda_path, 'bin')
        libnvvp_path = os.path.join(cuda_path, 'libnvvp')
        cupti_path = os.path.join(cuda_path, 'extras', 'CUPTI', 'lib64')
        
        with open('add_cuda_to_path.bat', 'w') as f:
            f.write("@echo off\n")
            f.write("echo Adding CUDA to PATH temporarily...\n")
            f.write(f'set PATH={bin_path};{libnvvp_path};{cupti_path};%PATH%\n')
            f.write("echo CUDA paths added to PATH for this session.\n")
            f.write("echo You can now use TensorFlow with GPU support in this command window.\n")
            f.write("cmd\n")
        
        logger.info("Created batch script 'add_cuda_to_path.bat'")
        logger.info("Run this script to open a command prompt with CUDA in the PATH")
        return True
    except Exception as e:
        logger.error(f"Error creating batch script: {e}")
        return False

def prompt_user_for_action(cuda_versions):
    """Prompt the user for action based on findings"""
    print("\n" + "="*80)
    print(" CUDA SETUP ASSISTANT ".center(80, "="))
    print("="*80 + "\n")
    
    if not cuda_versions:
        print("No CUDA installations were found on your system.")
        print("Options:")
        print("1. Install CUDA Toolkit 11.2 (recommended for TensorFlow 2.10.0)")
        print("2. Exit")
        
        choice = input("Enter your choice (1-2): ")
        if choice == '1':
            print("\nDownload CUDA Toolkit 11.2 from:")
            print("https://developer.nvidia.com/cuda-11.2.0-download-archive")
            print("\nRerun this script after installation.")
        return
    
    print("The following CUDA versions were found on your system:")
    for i, (version, path) in enumerate(cuda_versions, 1):
        print(f"{i}. CUDA {version} at {path}")
    
    print("\nOptions:")
    print(f"1-{len(cuda_versions)}. Add CUDA version to system PATH")
    print(f"{len(cuda_versions)+1}. Create temporary PATH script")
    print(f"{len(cuda_versions)+2}. Exit")
    
    choice = input(f"Enter your choice (1-{len(cuda_versions)+2}): ")
    try:
        choice_num = int(choice)
        if 1 <= choice_num <= len(cuda_versions):
            selected_version, selected_path = cuda_versions[choice_num-1]
            if is_admin():
                print(f"\nAdding CUDA {selected_version} to system PATH...")
                add_cuda_to_path(selected_path)
            else:
                print("\nAdministrator privileges required to modify system PATH.")
                print("Please run this script as Administrator and try again.")
        elif choice_num == len(cuda_versions)+1:
            selected_version, selected_path = cuda_versions[0]  # Use first version
            print(f"\nCreating temporary PATH script for CUDA {selected_version}...")
            create_batch_script(selected_path)
    except (ValueError, IndexError):
        print("Invalid choice. Exiting.")

def main():
    """Main function to diagnose and fix CUDA issues"""
    logger.info("="*50)
    logger.info("CUDA SETUP DIAGNOSTIC TOOL".center(50))
    logger.info("="*50)
    
    # Check system information
    logger.info(f"System: {platform.system()} {platform.version()}")
    logger.info(f"Python: {platform.python_version()}")
    
    # Check for NVIDIA GPU using nvidia-smi
    logger.info("\nChecking for NVIDIA GPU...")
    nvidia_smi_available = check_nvidia_smi()
    
    if not nvidia_smi_available:
        logger.warning("Could not detect NVIDIA GPU or driver. Check if:")
        logger.warning("1. Your system has an NVIDIA GPU")
        logger.warning("2. The NVIDIA driver is properly installed")
        logger.warning("3. You may need to restart your computer after driver installation")
    
    # Find installed CUDA versions
    logger.info("\nChecking for installed CUDA versions...")
    cuda_versions = find_installed_cuda_versions()
    
    # Check PATH environment variable
    logger.info("\nChecking if CUDA is in PATH...")
    paths = get_environment_path()
    cuda_paths = check_cuda_in_path(paths)
    
    # Test TensorFlow GPU detection
    logger.info("\nChecking TensorFlow GPU detection...")
    tf_gpu_available = test_import_tensorflow()
    
    # If TensorFlow is installed, check version compatibility
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        logger.info("\nChecking TensorFlow-CUDA compatibility...")
        recommended = check_tensorflow_cuda_compatibility(tf_version)
    except ImportError:
        logger.warning("TensorFlow is not installed")
    
    # Check for cuDNN in CUDA installations
    if cuda_versions:
        logger.info("\nChecking for cuDNN installations...")
        for version, path in cuda_versions:
            cudnn_version = get_cudnn_version(path)
            if cudnn_version:
                logger.info(f"Found cuDNN {cudnn_version} installed with CUDA {version}")
            else:
                logger.warning(f"cuDNN not found for CUDA {version} at {path}")
    
    # Summary
    logger.info("\n" + "="*30 + " SUMMARY " + "="*30)
    
    if nvidia_smi_available:
        logger.info("✓ NVIDIA GPU detected")
    else:
        logger.error("✗ NVIDIA GPU not detected or driver issue")
    
    if cuda_versions:
        cuda_versions_str = ", ".join([f"{version}" for version, _ in cuda_versions])
        logger.info(f"✓ CUDA installed: {cuda_versions_str}")
    else:
        logger.error("✗ No CUDA installations found")
    
    if cuda_paths:
        logger.info("✓ CUDA is in system PATH")
    else:
        logger.error("✗ CUDA is not in system PATH")
    
    if tf_gpu_available:
        logger.info("✓ TensorFlow can detect GPU")
    else:
        logger.error("✗ TensorFlow cannot detect GPU")
    
    # Recommendations
    logger.info("\n" + "="*25 + " RECOMMENDATIONS " + "="*25)
    
    if not tf_gpu_available:
        if not nvidia_smi_available:
            logger.info("1. Install NVIDIA GPU drivers")
            logger.info("   Visit: https://www.nvidia.com/Download/index.aspx")
        
        if not cuda_versions:
            try:
                import tensorflow as tf
                tf_version = tf.__version__
                major_minor = '.'.join(tf_version.split('.')[:2])
                if major_minor in ['2.10', '2.11']:
                    logger.info(f"2. Install CUDA 11.2 for TensorFlow {tf_version}")
                    logger.info("   Visit: https://developer.nvidia.com/cuda-11.2.0-download-archive")
                elif major_minor in ['2.12', '2.13', '2.14']:
                    logger.info(f"2. Install CUDA 11.8 for TensorFlow {tf_version}")
                    logger.info("   Visit: https://developer.nvidia.com/cuda-11.8.0-download-archive")
                else:
                    logger.info(f"2. Install CUDA 12.1+ for TensorFlow {tf_version}")
                    logger.info("   Visit: https://developer.nvidia.com/cuda-downloads")
            except ImportError:
                logger.info("2. Install CUDA 11.2 (recommended for TensorFlow 2.10.0)")
                logger.info("   Visit: https://developer.nvidia.com/cuda-11.2.0-download-archive")
        
        if cuda_versions and not cuda_paths:
            logger.info("3. Add CUDA to system PATH")
            version, path = cuda_versions[0]  # Use first version found
            logger.info(f"   Run this script as Administrator to add CUDA {version} to PATH")
    
    # Prompt for action
    if not tf_gpu_available:
        prompt_user_for_action(cuda_versions)
    else:
        logger.info("\nYour system is correctly configured for TensorFlow with GPU support.")
        logger.info("Run 'python run_with_gpu.py --gui' to start the application with GPU support.")

if __name__ == "__main__":
    # Check if running as admin
    if not is_admin() and platform.system() == 'Windows':
        logger.warning("This script is not running with administrator privileges.")
        logger.warning("Some actions like modifying system PATH will not be available.")
    
    main() 