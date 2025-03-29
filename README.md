# DeepFake Detector

A comprehensive deep learning-based tool for detecting deepfake images and videos.

## Overview

This project provides an AI-powered solution for detecting manipulated media (deepfakes). It uses multiple detection strategies including:

- General image manipulation detection using EfficientNet
- Face-specific forgery detection using ResNet50
- Frequency domain analysis for GAN artifact detection
- Temporal inconsistency detection for videos

The tool provides both a command-line interface and a graphical user interface for ease of use.

## Features

- **Multi-model approach**: Combines several detection strategies for more robust results
- **Image and video analysis**: Works with both still images and video files
- **Face detection**: Specialized analysis of facial regions where manipulations are common
- **Frequency analysis**: Detects GAN artifacts in the frequency domain
- **Detailed reports**: Generates visual reports with confidence scores and analysis details
- **Batch processing**: Process multiple files in one operation
- **User-friendly interface**: Both GUI and CLI options available

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for video analysis)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/deepfake-detector.git
   cd deepfake-detector
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download pre-trained models:
   ```
   python download_models.py
   ```

## Usage

### Command Line Interface

Analyze a single image:
```
python detect.py -i path/to/image.jpg
```

Analyze a video:
```
python detect.py -v path/to/video.mp4
```

Process a directory of media files:
```
python detect.py -d path/to/directory
```

Additional options:
```
python detect.py --help
```

### Graphical User Interface

Launch the GUI application:
```
python deepfake_detector_gui.py
```

In the GUI, you can:
- Open individual images or videos
- View detection results with visualizations
- Adjust detection sensitivity
- Process batches of files
- Export detailed reports

## How It Works

The detector uses a multi-layered approach:

1. **Image Analysis**: The entire image is analyzed using an EfficientNet model trained to detect manipulation patterns.

2. **Face Detection and Analysis**: Faces are extracted and analyzed using a specialized ResNet model trained on facial forgery detection.

3. **Frequency Domain Analysis**: The image is transformed to the frequency domain to detect GAN artifacts that may not be visible in the spatial domain.

4. **Temporal Analysis (for videos)**: Frame sequences are analyzed for temporal inconsistencies that often appear in manipulated videos.

5. **Ensemble Decision**: Results from the various models are combined to produce a final classification with confidence score.

## Model Architecture

The project utilizes several model architectures:

- **EfficientNetB3**: For general image manipulation detection
- **ResNet50**: Specialized for facial forgery detection
- **Custom CNN**: For frequency domain analysis
- **LSTM Network**: For temporal inconsistency detection in videos
- **Ensemble Model**: For combining the outputs of individual models

## Development

### Project Structure

```
deepfake-detector/
├── detect.py             # Command-line interface
├── deepfake_detector_gui.py  # Graphical user interface
├── download_models.py    # Script to download pre-trained models
├── requirements.txt      # Python dependencies
├── models/               # Model definitions and weights
│   ├── model_architecture.py
│   ├── pretrained/       # Pre-trained model weights
│   └── __init__.py
├── utils/                # Utility functions
│   ├── preprocessing.py  # Image and video preprocessing
│   ├── visualization.py  # Result visualization
│   └── __init__.py
└── output/               # Default directory for results
```

### Extending the Project

To train your own models, you would need to:

1. Collect a dataset of real and fake images/videos
2. Preprocess the data using the utilities in `utils/preprocessing.py`
3. Modify the model architectures in `models/model_architecture.py`
4. Implement a training script with appropriate data augmentation
5. Save the trained models to the `models/pretrained/` directory

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citations

If you use this project in your research, please cite:

```
@software{deepfake_detector,
  author = {Your Name},
  title = {DeepFake Detector: Multi-model Approach for Manipulated Media Detection},
  year = {2023},
  url = {https://github.com/yourusername/deepfake-detector}
}
```

## Acknowledgments

- This project uses models inspired by research in deepfake detection
- Thanks to the open-source community for libraries and tools that made this project possible

## Disclaimer

This tool is not 100% accurate and should be used as an aid rather than the sole determinant for identifying deepfakes. The technology for creating deepfakes is constantly evolving, and detection methods may need to be updated accordingly.

# Adding CUDA to PATH for GPU Acceleration

This repository contains tools to help you add CUDA to your system PATH, which is essential for TensorFlow to detect and use your NVIDIA GPU for deep learning acceleration.

## Included Tools

1. **add_cuda_to_path.ps1** - PowerShell script to automatically add CUDA to your PATH
2. **add_cuda_to_path.bat** - Batch script for Windows Command Prompt
3. **verify_gpu.py** - Python script to verify your GPU setup
4. **install_gpu_support.py** - Complete diagnostic and installation assistant

## Quick Start Guide

### Step 1: Run the GPU Detection Script

First, check if your GPU is already properly configured:

```
python verify_gpu.py
```

### Step 2: Add CUDA to PATH (if needed)

If the verification shows that CUDA is not in your PATH, use one of these scripts:

**Using PowerShell (Recommended):**
1. Right-click on `add_cuda_to_path.ps1`
2. Select "Run with PowerShell" (as Administrator)

**Using Command Prompt:**
1. Right-click on `add_cuda_to_path.bat`
2. Select "Run as administrator"

### Step 3: Verify the Changes

After running the scripts and restarting your command prompt:

```
python verify_gpu.py
```

### Step 4: Run the Application

Once everything is set up correctly:

```
python deepfake_detector_gui.py
```

## Complete GPU Setup Guide

For a complete guide on setting up NVIDIA GPU for TensorFlow, refer to the [NVIDIA_GPU_SETUP.md](NVIDIA_GPU_SETUP.md) file.

## Troubleshooting

If you encounter any issues:

1. Make sure you've installed CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
2. Install cuDNN from [NVIDIA's developer site](https://developer.nvidia.com/cudnn)
3. Run `python install_gpu_support.py` for diagnostics
4. Try reinstalling TensorFlow with `pip install tensorflow`
5. Restart your computer after making changes to environment variables

## System Requirements

- NVIDIA GPU (Compute Capability 3.5+)
- Windows 10 or 11
- Python 3.7+
- TensorFlow 2.x 

# GPU Support

This project includes full GPU acceleration support to speed up deepfake detection processing. The architecture has been updated to leverage TensorFlow with GPU support for all models.

## GPU Requirements

To use GPU acceleration, you'll need:

1. An NVIDIA GPU with CUDA support
2. CUDA 11.2 and cuDNN 8.1 installed (recommended versions for TensorFlow 2.10.0)
3. TensorFlow 2.10.0 or compatible version

## CPU Fallback Mode

The application is designed to automatically fall back to CPU processing when a GPU is not available or properly configured. This ensures that the application will run on any system, although processing will be significantly slower without GPU acceleration.

To run the application with CPU fallback:

```
python run_with_gpu.py --gui
```

This will launch the application and automatically detect whether GPU is available. If not, it will display appropriate warnings and continue running in CPU mode.

### Verifying Your Environment

You can verify your system's configuration using:

```
python run_with_gpu.py --verify
```

This will check for:
- Python version
- TensorFlow version
- GPU availability
- Required libraries (OpenCV, NumPy, dlib)
- Model availability

### Troubleshooting GPU Detection

If TensorFlow cannot detect your GPU:

1. Ensure you have the correct CUDA version installed (11.2 for TensorFlow 2.10.0)
2. Make sure cuDNN is installed and in your PATH
3. Check that your GPU drivers are up to date
4. Verify TensorFlow was installed with GPU support
5. Try running the `FixCuda.py` script included in this repository

For a complete guide on setting up NVIDIA GPU for TensorFlow, refer to the [NVIDIA_GPU_SETUP.md](NVIDIA_GPU_SETUP.md) file.

## Face Detection Options

The application supports two methods for face detection:

1. **dlib (recommended)**: More accurate face detection, installed with `pip install dlib`
2. **OpenCV**: Used as fallback when dlib is not available

If dlib is not installed, the application will automatically use OpenCV for face detection.

## GPU Configuration

The application uses a dedicated `gpu_config.py` module that:

1. Automatically detects available GPUs
2. Configures TensorFlow to use GPU efficiently when available
3. Enables memory growth to avoid OOM errors
4. Sets up the proper distribution strategy
5. Gracefully falls back to CPU when GPU is not available

Models are now created within the appropriate strategy scope for optimal GPU performance when available.

## Performance Benefits

Using GPU acceleration can significantly improve performance:

- Image processing is typically 5-10x faster
- Video analysis can be 10-20x faster depending on the GPU
- Batch processing of multiple files sees the greatest improvement

For best results, we recommend an NVIDIA GPU with at least 4GB of VRAM, though the application will work with smaller GPUs by automatically managing memory usage. 