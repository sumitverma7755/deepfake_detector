# NVIDIA GPU Setup Guide for TensorFlow

This guide will help you set up your NVIDIA GPU for use with TensorFlow in the DeepFake Detector application.

## System Requirements

- An NVIDIA GPU with Compute Capability 3.5 or higher
  - Recommended: RTX 2000 series or newer
- Windows 10 or 11 (64-bit)
- Python 3.8 or newer
- Administrator privileges

## Installation Steps

### Step 1: Verify Your GPU

Before proceeding, verify that you have a compatible NVIDIA GPU:

1. Press `Win + X` and select "Device Manager"
2. Expand "Display adapters"
3. Confirm that an NVIDIA GPU is listed

### Step 2: Install NVIDIA GPU Drivers

1. Download the latest drivers for your GPU from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
2. Install the drivers following the on-screen instructions
3. Restart your computer when prompted

### Step 3: Install CUDA Toolkit 11.2

TensorFlow 2.10.0 works best with CUDA 11.2:

1. Download CUDA Toolkit 11.2 from [NVIDIA CUDA Archive](https://developer.nvidia.com/cuda-11.2.0-download-archive)
2. Select the appropriate version for your operating system
3. Run the installer and follow the installation wizard
   - Choose "Custom Installation" and ensure that both the CUDA Toolkit and CUDA Runtime are selected
   - The default installation path should be `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`

### Step 4: Install cuDNN 8.1 for CUDA 11.2

1. Go to [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
2. Create an NVIDIA account or log in if you already have one
3. Download cuDNN 8.1 for CUDA 11.2
4. Extract the contents of the downloaded zip file
5. Copy the following files to your CUDA installation:
   - Copy `<extracted_folder>\cuda\bin\*.dll` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
   - Copy `<extracted_folder>\cuda\include\*.h` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include`
   - Copy `<extracted_folder>\cuda\lib\x64\*.lib` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64`

### Step 5: Add CUDA to System PATH

1. Press `Win + X` and select "System"
2. Click on "Advanced system settings"
3. Click on "Environment Variables"
4. Under "System variables", find and select the "Path" variable
5. Click "Edit"
6. Click "New" and add the following paths:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\extras\CUPTI\lib64
   ```
7. Click "OK" to close all dialogs

Alternatively, you can use the included scripts:
```
# Run as Administrator
.\add_cuda_to_path.ps1
```

### Step 6: Install TensorFlow with GPU Support

1. Open a new Command Prompt or PowerShell window (to ensure the PATH changes are loaded)
2. Install TensorFlow 2.10.0 with:
   ```
   pip uninstall -y tensorflow tensorflow-gpu
   pip install tensorflow==2.10.0
   ```

### Step 7: Verify Installation

To verify that TensorFlow can access your GPU:

1. Open a Python interpreter:
   ```
   python
   ```

2. Run the following code:
   ```python
   import tensorflow as tf
   print("TensorFlow version:", tf.__version__)
   print("GPU devices:", tf.config.list_physical_devices('GPU'))
   ```

3. If successful, you should see your GPU listed in the output.

Alternatively, use our verification script:
```
python run_with_gpu.py --verify
```

## Troubleshooting

### GPU Not Detected by TensorFlow

1. **CUDA Version Mismatch**:
   - Make sure you have CUDA 11.2 installed for TensorFlow 2.10.0
   - Newer versions of TensorFlow may require different CUDA versions

2. **PATH Issues**:
   - Verify that CUDA paths are correctly added to your system PATH
   - Try running the `add_cuda_to_path.ps1` script as Administrator

3. **Driver Problems**:
   - Update to the latest NVIDIA drivers
   - Verify that the driver is working by running `nvidia-smi` in Command Prompt

4. **TensorFlow Version**:
   - The CUDA version must match the TensorFlow version
   - Try installing TensorFlow 2.10.0 which is compatible with CUDA 11.2

### Common Error Messages

1. **"Could not load dynamic library 'cudart64_110.dll'"**:
   - CUDA is not properly installed or not in PATH
   - Reinstall CUDA 11.2 and ensure its bin directory is in your PATH

2. **"Could not load dynamic library 'nvcuda.dll'"**:
   - NVIDIA driver is not properly installed
   - Reinstall the latest NVIDIA driver for your GPU

3. **"No CUDA-capable device is detected"**:
   - Your GPU may not be CUDA-capable
   - The driver may not be properly installed

### Running the FixCuda.py Script

We've included a diagnostic script that can help identify and fix CUDA issues:

```
python FixCuda.py
```

This script will:
- Check if CUDA paths are in your system PATH
- Verify CUDA and cuDNN installations
- Test if TensorFlow can detect your GPU
- Add missing paths to your PATH variable (with permission)
- Provide detailed error messages and suggestions

## Running with CPU Fallback

If you're unable to configure GPU support, the application will automatically fall back to CPU:

```
python run_with_gpu.py --gui
```

While using CPU will be significantly slower, all functionality will remain available.

## Additional Resources

- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
- [NVIDIA cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

## Contact

If you continue to experience issues, please file an issue on our GitHub repository with:
- Your system specifications (GPU model, OS version)
- Exact error messages
- Steps you've already attempted 