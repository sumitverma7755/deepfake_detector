#!/usr/bin/env python3
"""
DeepFake Detector - Command Line Interface
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import cv2
import time
import json
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deepfake_detector.log')
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import our GPU configuration module
from gpu_config import configure_gpu, get_device_strategy

# Configure GPU
gpu_available = configure_gpu()

# Get distribution strategy
strategy = get_device_strategy()

# Import project modules
from utils.preprocessing import (
    load_image,
    extract_faces,
    extract_frames,
    preprocess_frames,
    extract_frequency_features
)
from utils.visualization import (
    plot_detection_result,
    create_detection_report,
    visualize_video_analysis
)
from models.model_architecture import (
    load_model_from_checkpoint,
    get_model_explanation
)

# Paths
MODELS_DIR = os.path.join(script_dir, "models", "pretrained")
OUTPUT_DIR = os.path.join(script_dir, "output")

class DeepfakeDetector:
    """Deepfake detection using multiple models"""
    
    def __init__(self, model_dir=MODELS_DIR, verbose=True):
        """
        Initialize the detector with pre-trained models.
        
        Args:
            model_dir: Directory containing model files
            verbose: Whether to print detailed information
        """
        self.model_dir = model_dir
        self.verbose = verbose
        self.image_model = None
        self.face_model = None
        self.freq_model = None
        self.temporal_model = None
        self.ensemble_model = None
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models"""
        try:
            # Check if models directory exists
            if not os.path.exists(self.model_dir):
                logger.warning(f"Model directory not found: {self.model_dir}")
                logger.info("Please run download_models.py to download pre-trained models.")
                return
            
            # Check for GPU availability
            if gpu_available:
                logger.info("GPU is available and configured for inference.")
            else:
                logger.info("No GPUs found. Using CPU for inference.")
            
            # Load model info
            model_info_path = os.path.join(self.model_dir, "model_info.json")
            if os.path.exists(model_info_path):
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
            else:
                model_info = {}
            
            # Use our strategy for loading models
            with strategy.scope():
                # Load image model (EfficientNet)
                image_model_path = os.path.join(self.model_dir, "efficientnet_deepfake_detector.h5")
                if os.path.exists(image_model_path):
                    logger.info("Loading EfficientNet model...")
                    self.image_model = load_model_from_checkpoint(
                        'efficientnet', 
                        image_model_path
                    )
                    logger.info("EfficientNet model loaded successfully")
                
                # Load face specific model (ResNet)
                face_model_path = os.path.join(self.model_dir, "resnet_face_detector.h5")
                if os.path.exists(face_model_path):
                    logger.info("Loading ResNet face model...")
                    self.face_model = load_model_from_checkpoint(
                        'resnet_face', 
                        face_model_path
                    )
                    logger.info("ResNet face model loaded successfully")
                
                # Load frequency analysis model
                freq_model_path = os.path.join(self.model_dir, "frequency_detector.h5")
                if os.path.exists(freq_model_path):
                    logger.info("Loading frequency analysis model...")
                    self.freq_model = load_model_from_checkpoint(
                        'frequency', 
                        freq_model_path
                    )
                    logger.info("Frequency analysis model loaded successfully")
            
            # Check if any models were loaded
            if not any([self.image_model, self.face_model, self.freq_model]):
                logger.warning("No models were loaded. Detection will not be accurate.")
                # Create a basic model for demo purposes
                logger.info("Creating a basic model for demonstration purposes...")
                with strategy.scope():
                    self.image_model = tf.keras.Sequential([
                        tf.keras.layers.Input(shape=(224, 224, 3)),
                        tf.keras.layers.GlobalAveragePooling2D(),
                        tf.keras.layers.Dense(1, activation='sigmoid')
                    ])
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def detect_image(self, image_path, threshold=0.5, output_dir=None):
        """
        Detect deepfakes in an image.
        
        Args:
            image_path: Path to the image file
            threshold: Threshold for fake/real classification
            output_dir: Directory to save results (if None, use default)
            
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        try:
            # Set output directory
            if output_dir is None:
                output_dir = OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)
            
            # Load image
            logger.info(f"Loading image: {image_path}")
            image = load_image(image_path)
            
            # Extract faces
            logger.info("Extracting faces from image...")
            faces = extract_faces(image)
            
            # Define result structure
            result = {
                'image_path': image_path,
                'faces_detected': len(faces),
                'face_predictions': [],
                'overall_prediction': 0.0,
                'is_fake': False,
                'detection_time': 0.0,
            }
            
            # If no models are loaded, return default result
            if not any([self.image_model, self.face_model, self.freq_model]):
                logger.warning("No models loaded. Cannot perform detection.")
                result['error'] = "No detection models loaded"
                return result
            
            # Analyze whole image
            image_prediction = 0.0
            if self.image_model:
                # Preprocess image for the model
                img_resized = cv2.resize(image, (224, 224))
                img_normalized = img_resized.astype(np.float32) / 255.0
                img_batch = np.expand_dims(img_normalized, axis=0)
                
                # Get prediction
                image_prediction = float(self.image_model.predict(img_batch, verbose=0)[0][0])
                logger.info(f"Whole image prediction: {image_prediction:.4f}")
            
            # Analyze frequency domain
            freq_prediction = 0.0
            if self.freq_model:
                # Extract frequency features
                freq_features = extract_frequency_features(image)
                freq_features = cv2.resize(freq_features, (224, 224))
                freq_batch = np.expand_dims(freq_features, axis=0)
                
                # Get prediction
                freq_prediction = float(self.freq_model.predict(freq_batch, verbose=0)[0][0])
                logger.info(f"Frequency domain prediction: {freq_prediction:.4f}")
            
            # Analyze faces
            face_predictions = []
            if faces and self.face_model:
                for i, face in enumerate(faces):
                    # Preprocess face for the model
                    face_resized = cv2.resize(face, (224, 224))
                    face_normalized = face_resized.astype(np.float32) / 255.0
                    face_batch = np.expand_dims(face_normalized, axis=0)
                    
                    # Get prediction
                    face_pred = float(self.face_model.predict(face_batch, verbose=0)[0][0])
                    
                    # Store result
                    face_predictions.append({
                        'face_index': i,
                        'probability': float(face_pred),
                        'is_fake': face_pred >= threshold
                    })
                    
                    logger.info(f"Face {i+1} prediction: {face_pred:.4f}")
            
            # Combine predictions
            if face_predictions:
                face_avg = np.mean([p['probability'] for p in face_predictions])
                # Weight: 40% image, 40% faces, 20% frequency
                overall_prediction = 0.4 * image_prediction + 0.4 * face_avg + 0.2 * freq_prediction
            else:
                # Weight: 70% image, 30% frequency
                overall_prediction = 0.7 * image_prediction + 0.3 * freq_prediction
            
            # Determine if the image is fake
            is_fake = overall_prediction >= threshold
            
            # Construct face locations for visualization
            face_locations = []
            if faces:
                # For this demo, we'll create approximate face locations
                # In a real implementation, these would be returned by the face detector
                h, w = image.shape[:2]
                for i, face in enumerate(faces):
                    face_h, face_w = face.shape[:2]
                    # This is just an approximation
                    face_ratio = min(face_h / h, face_w / w)
                    top = int(h * 0.2 * i)
                    left = int(w * 0.1)
                    bottom = int(top + face_h)
                    right = int(left + face_w)
                    face_locations.append((top, right, bottom, left))
            
            # Create report image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            report_path = os.path.join(output_dir, f"{base_name}_report_{timestamp}.png")
            
            # Generate report with visualization
            logger.info("Generating detection report...")
            create_detection_report(
                image=image,
                prediction=overall_prediction,
                face_locations=face_locations,
                threshold=threshold,
                output_path=report_path
            )
            
            # Update result
            result.update({
                'overall_prediction': float(overall_prediction),
                'is_fake': bool(is_fake),
                'image_prediction': float(image_prediction),
                'frequency_prediction': float(freq_prediction),
                'face_predictions': face_predictions,
                'report_path': report_path,
                'detection_time': time.time() - start_time
            })
            
            logger.info(f"Detection complete - {'FAKE' if is_fake else 'REAL'} "
                      f"(confidence: {overall_prediction:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting deepfakes in image: {e}")
            return {
                'error': str(e),
                'image_path': image_path,
                'detection_time': time.time() - start_time
            }
    
    def detect_video(self, video_path, threshold=0.5, max_frames=100, output_dir=None):
        """
        Detect deepfakes in a video.
        
        Args:
            video_path: Path to the video file
            threshold: Threshold for fake/real classification
            max_frames: Maximum number of frames to analyze
            output_dir: Directory to save results (if None, use default)
            
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        try:
            # Set output directory
            if output_dir is None:
                output_dir = OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract frames from video
            logger.info(f"Extracting frames from video: {video_path}")
            frames = extract_frames(video_path, max_frames=max_frames)
            
            if len(frames) == 0:
                logger.error(f"Could not extract frames from video: {video_path}")
                return {
                    'error': "Could not extract frames from video",
                    'video_path': video_path,
                    'detection_time': time.time() - start_time
                }
            
            logger.info(f"Extracted {len(frames)} frames")
            
            # Preprocess frames
            processed_frames = preprocess_frames(frames)
            
            # Define result structure
            result = {
                'video_path': video_path,
                'frames_analyzed': len(frames),
                'overall_prediction': 0.0,
                'is_fake': False,
                'frame_predictions': [],
                'detection_time': 0.0,
            }
            
            # Analyze frames
            frame_predictions = []
            for i, frame in enumerate(processed_frames):
                # Create batch for prediction
                frame_batch = np.expand_dims(frame, axis=0)
                
                # Get image-based prediction
                image_prediction = 0.0
                if self.image_model:
                    image_prediction = float(self.image_model.predict(frame_batch, verbose=0)[0][0])
                
                # Get frequency-based prediction
                freq_prediction = 0.0
                if self.freq_model:
                    freq_features = extract_frequency_features(frames[i])
                    freq_features = cv2.resize(freq_features, (224, 224))
                    freq_batch = np.expand_dims(freq_features, axis=0)
                    freq_prediction = float(self.freq_model.predict(freq_batch, verbose=0)[0][0])
                
                # Combine predictions
                frame_pred = 0.7 * image_prediction + 0.3 * freq_prediction
                is_fake = frame_pred >= threshold
                
                # Store frame result
                frame_predictions.append({
                    'frame_index': i,
                    'probability': float(frame_pred),
                    'is_fake': bool(is_fake),
                    'image_prediction': float(image_prediction),
                    'frequency_prediction': float(freq_prediction)
                })
                
                if i % 10 == 0:
                    logger.info(f"Processed frame {i+1}/{len(frames)}: {frame_pred:.4f}")
            
            # Calculate overall prediction
            if frame_predictions:
                overall_prediction = np.mean([p['probability'] for p in frame_predictions])
                is_fake = overall_prediction >= threshold
            else:
                overall_prediction = 0.0
                is_fake = False
            
            # Create visualization
            logger.info("Generating video analysis visualization...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(video_path)
            base_name = os.path.splitext(filename)[0]
            viz_path = os.path.join(output_dir, f"{base_name}_analysis_{timestamp}.png")
            
            visualize_video_analysis(
                frames=frames,
                predictions=frame_predictions,
                interval=max(1, len(frames) // 16),  # At most 16 frames in the visualization
                output_path=viz_path
            )
            
            # Update result
            result.update({
                'overall_prediction': float(overall_prediction),
                'is_fake': bool(is_fake),
                'frame_predictions': frame_predictions,
                'visualization_path': viz_path,
                'detection_time': time.time() - start_time
            })
            
            logger.info(f"Video detection complete - {'FAKE' if is_fake else 'REAL'} "
                      f"(confidence: {overall_prediction:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting deepfakes in video: {e}")
            return {
                'error': str(e),
                'video_path': video_path,
                'detection_time': time.time() - start_time
            }
    
    def export_results(self, result, output_path=None, format='json'):
        """
        Export detection results to a file.
        
        Args:
            result: Detection result dictionary
            output_path: Path to save the result file (if None, auto-generate)
            format: Export format ('json' or 'txt')
            
        Returns:
            Path to the saved file
        """
        try:
            # Set output path if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if 'image_path' in result:
                    filename = os.path.basename(result['image_path'])
                    base_name = os.path.splitext(filename)[0]
                    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_result_{timestamp}.{format}")
                elif 'video_path' in result:
                    filename = os.path.basename(result['video_path'])
                    base_name = os.path.splitext(filename)[0]
                    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_result_{timestamp}.{format}")
                else:
                    output_path = os.path.join(OUTPUT_DIR, f"result_{timestamp}.{format}")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export based on format
            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
            else:
                # Text format
                with open(output_path, 'w') as f:
                    f.write(f"DeepFake Detection Result\n")
                    f.write(f"=======================\n\n")
                    
                    # Write timestamp
                    f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    # Write source info
                    if 'image_path' in result:
                        f.write(f"Image: {result['image_path']}\n")
                    elif 'video_path' in result:
                        f.write(f"Video: {result['video_path']}\n")
                    
                    # Write overall result
                    f.write(f"\nRESULT: {'FAKE' if result.get('is_fake', False) else 'REAL'}\n")
                    f.write(f"Confidence: {result.get('overall_prediction', 0.0):.2%}\n")
                    f.write(f"Detection time: {result.get('detection_time', 0.0):.2f} seconds\n\n")
                    
                    # Write additional details
                    if 'faces_detected' in result:
                        f.write(f"Faces detected: {result['faces_detected']}\n")
                        
                        if result.get('face_predictions'):
                            f.write("\nFace analysis:\n")
                            for face in result['face_predictions']:
                                f.write(f"  Face {face['face_index']+1}: ")
                                f.write(f"{'FAKE' if face['is_fake'] else 'REAL'} ")
                                f.write(f"({face['probability']:.2%})\n")
                    
                    if 'frames_analyzed' in result:
                        f.write(f"\nFrames analyzed: {result['frames_analyzed']}\n")
                        
                        if result.get('frame_predictions'):
                            fake_frames = sum(1 for p in result['frame_predictions'] if p['is_fake'])
                            f.write(f"Frames detected as fake: {fake_frames}\n")
                            f.write(f"Frames detected as real: {result['frames_analyzed'] - fake_frames}\n")
                    
                    # Write paths to additional files
                    if 'report_path' in result:
                        f.write(f"\nDetailed report: {result['report_path']}\n")
                    
                    if 'visualization_path' in result:
                        f.write(f"Video analysis visualization: {result['visualization_path']}\n")
            
            logger.info(f"Results exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return None

def main():
    """Run the command-line interface"""
    parser = argparse.ArgumentParser(description="DeepFake Detector CLI")
    
    # Main options
    parser.add_argument("--image", help="Path to the image file to analyze")
    parser.add_argument("--video", help="Path to the video file to analyze")
    parser.add_argument("--batch", help="Directory containing media files to analyze in batch")
    
    # Detection options
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (0.0-1.0)")
    parser.add_argument("--output", help="Output directory for results", default=OUTPUT_DIR)
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to analyze for video")
    parser.add_argument("--format", choices=["json", "txt", "html"], default="json", help="Output format")
    
    args = parser.parse_args()
    
    # Check if at least one input is provided
    if not any([args.image, args.video, args.batch]):
        parser.print_help()
        sys.exit(1)
    
    # Initialize detector
    try:
        detector = DeepfakeDetector()
    except Exception as e:
        logger.error(f"Error initializing detector: {e}")
        sys.exit(1)
    
    # Check for GPU and log info
    if gpu_available:
        logger.info("Using GPU for inference")
    else:
        logger.info("Using CPU for inference")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process based on input type
    if args.image:
        # Process a single image
        try:
            start_time = time.time()
            result = detector.detect_image(args.image, threshold=args.threshold, output_dir=args.output)
            end_time = time.time()
            
            # Save results
            detector.export_results(result, 
                                    output_path=os.path.join(args.output, f"result_{os.path.basename(args.image)}.{args.format}"),
                                    format=args.format)
            
            logger.info(f"Detection completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Result: {'FAKE' if result.get('is_fake', False) else 'REAL'} with confidence {result.get('confidence', 0):.4f}")
            logger.info(f"Results saved to {args.output}")
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            sys.exit(1)
    
    elif args.video:
        # Process a single video
        try:
            start_time = time.time()
            result = detector.detect_video(args.video, threshold=args.threshold, max_frames=args.frames, output_dir=args.output)
            end_time = time.time()
            
            # Save results
            detector.export_results(result, 
                                    output_path=os.path.join(args.output, f"result_{os.path.basename(args.video)}.{args.format}"),
                                    format=args.format)
            
            logger.info(f"Detection completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Result: {'FAKE' if result.get('is_fake', False) else 'REAL'} with confidence {result.get('confidence', 0):.4f}")
            logger.info(f"Results saved to {args.output}")
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            sys.exit(1)
    
    elif args.batch:
        # Process a batch of files
        try:
            # Check if directory exists
            if not os.path.isdir(args.batch):
                logger.error(f"Batch directory not found: {args.batch}")
                sys.exit(1)
            
            # Get all media files in the directory
            image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
            video_exts = ['.mp4', '.avi', '.mov', '.mkv']
            media_files = []
            
            for root, _, files in os.walk(args.batch):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in image_exts or ext in video_exts:
                        media_files.append(os.path.join(root, file))
            
            logger.info(f"Found {len(media_files)} media files to process")
            
            # Process each file
            results = []
            for i, file_path in enumerate(media_files):
                try:
                    logger.info(f"Processing {i+1}/{len(media_files)}: {file_path}")
                    ext = os.path.splitext(file_path)[1].lower()
                    
                    start_time = time.time()
                    if ext in image_exts:
                        result = detector.detect_image(file_path, threshold=args.threshold, output_dir=args.output)
                    else:
                        result = detector.detect_video(file_path, threshold=args.threshold, max_frames=args.frames, output_dir=args.output)
                    end_time = time.time()
                    
                    # Save individual result
                    detector.export_results(result, 
                                          output_path=os.path.join(args.output, f"result_{os.path.basename(file_path)}.{args.format}"),
                                          format=args.format)
                    
                    # Add to batch results
                    results.append({
                        'file': file_path,
                        'is_fake': result.get('is_fake', False),
                        'confidence': result.get('confidence', 0),
                        'processing_time': end_time - start_time
                    })
                    
                    logger.info(f"Result: {'FAKE' if result.get('is_fake', False) else 'REAL'} with confidence {result.get('confidence', 0):.4f}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    results.append({
                        'file': file_path,
                        'error': str(e)
                    })
            
            # Save batch results
            batch_report_path = os.path.join(args.output, f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.format}")
            with open(batch_report_path, 'w') as f:
                if args.format == 'json':
                    json.dump({'results': results}, f, indent=4)
                elif args.format == 'txt':
                    f.write(f"Batch Processing Results\n")
                    f.write(f"DateTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total Files: {len(media_files)}\n\n")
                    
                    for result in results:
                        if 'error' in result:
                            f.write(f"{result['file']}: ERROR - {result['error']}\n")
                        else:
                            f.write(f"{result['file']}: {'FAKE' if result['is_fake'] else 'REAL'} ({result['confidence']:.4f}) in {result['processing_time']:.2f}s\n")
                
                elif args.format == 'html':
                    # Simple HTML report
                    f.write(f"<html><head><title>Batch Results</title></head><body>")
                    f.write(f"<h1>Batch Processing Results</h1>")
                    f.write(f"<p>DateTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
                    f.write(f"<p>Total Files: {len(media_files)}</p>")
                    
                    f.write(f"<table border='1'><tr><th>File</th><th>Result</th><th>Confidence</th><th>Time</th></tr>")
                    for result in results:
                        if 'error' in result:
                            f.write(f"<tr><td>{result['file']}</td><td colspan='3'>ERROR: {result['error']}</td></tr>")
                        else:
                            f.write(f"<tr><td>{result['file']}</td><td>{'FAKE' if result['is_fake'] else 'REAL'}</td>")
                            f.write(f"<td>{result['confidence']:.4f}</td><td>{result['processing_time']:.2f}s</td></tr>")
                    
                    f.write(f"</table></body></html>")
            
            logger.info(f"Batch processing completed. Processed {len(media_files)} files.")
            logger.info(f"Batch report saved to {batch_report_path}")
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main() 