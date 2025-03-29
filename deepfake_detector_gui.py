#!/usr/bin/env python3
# Add GPU configuration at the top of the file
import os
import sys
import tensorflow as tf

# Add project root to path to import our GPU configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import our GPU configuration module
from gpu_config import configure_gpu, get_device_strategy

# Suppress TensorFlow oneDNN messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure GPU using our module
gpu_available = configure_gpu()

# Create a global strategy for model creation
strategy = get_device_strategy()

"""
DeepFake Detector - GUI Application
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import time
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import project modules
from utils.preprocessing import (
    load_image,
    extract_faces,
    extract_frames,
    preprocess_frames
)
from utils.visualization import (
    plot_detection_result,
    create_detection_report,
    visualize_video_analysis
)

# Import the detector class
from detect import DeepfakeDetector

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
DARK_BG = "#212121"  # Dark background
DARK_FG = "#FFFFFF"  # White text for maximum contrast
ACCENT_COLOR = "#3D5AFE"  # Bright blue accent
FAKE_COLOR = "#F44336"  # Bright red for fake results
REAL_COLOR = "#4CAF50"  # Bright green for real results
WARNING_COLOR = "#FFA000"  # Bright amber for warnings
ENTRY_BG = "#333333"  # Slightly lighter background for input fields
HOVER_COLOR = "#444444"  # Color for button/control hover states

# Global font scaling (increase for better visibility)
FONT_SCALE = 1.2  # Adjust as needed for better visibility
BASE_FONT_SIZE = int(10 * FONT_SCALE)  # Base font size
TITLE_FONT_SIZE = int(16 * FONT_SCALE)  # Large titles
SUBTITLE_FONT_SIZE = int(14 * FONT_SCALE)  # Section headers
BUTTON_FONT_SIZE = int(11 * FONT_SCALE)  # Buttons
TEXT_FONT_SIZE = int(11 * FONT_SCALE)  # Regular text

class DeepfakeDetectorGUI:
    """GUI application for deepfake detection"""
    
    def __init__(self, root):
        """Initialize the GUI application"""
        self.root = root
        self.root.title("DeepFake Detector")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.minsize(900, 600)
        
        # Set background color for the root window
        self.root.configure(bg=DARK_BG)
        
        # Configure styles
        self.configure_styles()
        
        # Initialize detector with error handling
        try:
            # Create models directory if it doesn't exist
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pretrained")
            os.makedirs(models_dir, exist_ok=True)
            
            # Initialize detector
            self.detector = DeepfakeDetector()
            
            # If no models are loaded, create a basic model for testing
            if not hasattr(self.detector, 'image_model') or self.detector.image_model is None:
                print("Creating basic model for testing...")
                
                # Create GPU model using our strategy
                with strategy.scope():
                    print("Creating optimized GPU model...")
                    self.detector.image_model = tf.keras.Sequential([
                        tf.keras.layers.Input(shape=(224, 224, 3)),
                        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),  # Smaller model for GPU
                        tf.keras.layers.MaxPooling2D((2, 2)),
                        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                        tf.keras.layers.GlobalAveragePooling2D(),
                        tf.keras.layers.Dense(1, activation='sigmoid')
                    ])
                
                print("Basic GPU model created successfully")
                
                # Add a message to be shown to the user
                if gpu_available:
                    gpu_message = "A GPU-optimized model has been created for testing purposes. The application is configured to use GPU for processing."
                else:
                    gpu_message = "A model has been created for testing purposes. No GPU was detected, so CPU will be used for processing."
                    
                self.after_idle(lambda: messagebox.showinfo(
                    "Model Created", 
                    gpu_message
                ))
        except Exception as e:
            # Still initialize the GUI even if detector fails
            self.detector = None
            print(f"Warning: Could not initialize detector: {e}")
            # We'll check for self.detector before using it later
        
        # Track current media
        self.current_image_path = None
        self.current_video_path = None
        self.is_processing = False
        self.last_result = None
        
        # Create the main layout
        self.create_main_layout()
        
        # Configure the grid layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Display welcome message
        if self.detector is None:
            self.update_status("Warning: Detector initialization failed. Some features may be unavailable.")
        else:
            self.update_status("Ready. Start by selecting an image or video to analyze.")
        
        # Schedule high contrast application after a short delay to ensure all widgets are created
        self.root.after(200, lambda: apply_high_contrast(self.root))
    
    def configure_styles(self):
        """Configure ttk styles for the application"""
        self.style = ttk.Style()
        
        # Configure theme
        try:
            self.style.theme_use("clam")  # Use a theme that supports customization
        except:
            pass  # If theme not available, use default
        
        # Configure colors with enhanced visibility
        self.style.configure("TFrame", background=DARK_BG)
        self.style.configure("TLabel", background=DARK_BG, foreground=DARK_FG, font=("Segoe UI", TEXT_FONT_SIZE))
        self.style.configure("TButton", background=ACCENT_COLOR, foreground=DARK_FG, font=("Segoe UI", BUTTON_FONT_SIZE))
        self.style.configure("TNotebook", background=DARK_BG, foreground=DARK_FG)
        self.style.configure("TNotebook.Tab", background=DARK_BG, foreground=DARK_FG, padding=[10, 5])
        
        # Enhanced contrast for combobox
        self.style.configure("TCombobox", 
                            selectbackground=ACCENT_COLOR,
                            selectforeground=DARK_FG,
                            fieldbackground=ENTRY_BG,
                            background=ENTRY_BG,
                            foreground=DARK_FG,
                            font=("Segoe UI", TEXT_FONT_SIZE))
        
        # Configure special styles with increased font sizes
        self.style.configure("Title.TLabel", font=("Segoe UI", TITLE_FONT_SIZE, "bold"), foreground="#FFFFFF")
        self.style.configure("Status.TLabel", font=("Segoe UI", BASE_FONT_SIZE), foreground="#FFFFFF")
        self.style.configure("Result.TLabel", font=("Segoe UI", SUBTITLE_FONT_SIZE, "bold"), foreground="#FFFFFF")
        self.style.configure("Fake.TLabel", foreground=FAKE_COLOR, font=("Segoe UI", SUBTITLE_FONT_SIZE, "bold"))
        self.style.configure("Real.TLabel", foreground=REAL_COLOR, font=("Segoe UI", SUBTITLE_FONT_SIZE, "bold"))
        
        # Configure button styles with enhanced visibility
        self.style.configure("Accent.TButton", background=ACCENT_COLOR, foreground="#FFFFFF", font=("Segoe UI", BUTTON_FONT_SIZE, "bold"))
        
        # Apply style maps for hover effects
        self.style.map("TButton",
                     background=[('active', HOVER_COLOR)],
                     foreground=[('active', "#FFFFFF")])
        self.style.map("Accent.TButton",
                     background=[('active', "#304FFE")],
                     foreground=[('active', "#FFFFFF")])
        
        # Apply settings to ensure all text widgets are visible
        self.root.option_add("*TCombobox*Listbox.background", ENTRY_BG)
        self.root.option_add("*TCombobox*Listbox.foreground", DARK_FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT_COLOR)
        self.root.option_add("*TCombobox*Listbox.selectForeground", DARK_FG)
        self.root.option_add("*TCombobox*Listbox.font", f"Segoe UI {TEXT_FONT_SIZE}")
        
        # Ensure entry widgets have high contrast
        self.style.configure("TEntry", 
                           foreground=DARK_FG,
                           fieldbackground=ENTRY_BG,
                           insertcolor=DARK_FG,  # Cursor color
                           font=("Segoe UI", TEXT_FONT_SIZE))
                           
        # Configure Treeview
        self.style.configure("Treeview", 
                           background=ENTRY_BG,
                           foreground=DARK_FG,
                           fieldbackground=ENTRY_BG,
                           font=("Segoe UI", BASE_FONT_SIZE))
        self.style.configure("Treeview.Heading", 
                           background=DARK_BG, 
                           foreground=DARK_FG,
                           font=("Segoe UI", BASE_FONT_SIZE, "bold"))
    
    def create_main_layout(self):
        """Create the main application layout"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Create title frame
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        # App title
        app_title = ttk.Label(title_frame, text="DeepFake Detector", style="Title.TLabel")
        app_title.pack(side=tk.LEFT, padx=10)
        
        # Add controls to title frame
        self.create_control_buttons(title_frame)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        
        # Create tabs
        self.create_detection_tab()
        self.create_batch_tab()
        self.create_help_tab()
        
        # Status bar at the bottom
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        
        self.status_var = tk.StringVar()
        status_label = ttk.Label(status_frame, textvariable=self.status_var, style="Status.TLabel")
        status_label.pack(side=tk.LEFT, padx=10)
        
        # Version info
        version_label = ttk.Label(status_frame, text="v1.0", style="Status.TLabel")
        version_label.pack(side=tk.RIGHT, padx=10)
    
    def create_control_buttons(self, parent):
        """Create control buttons in the title frame"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(side=tk.RIGHT, padx=10)
        
        # Open image button
        open_image_btn = ttk.Button(
            button_frame, 
            text="Open Image", 
            command=self.open_image,
            style="Accent.TButton"
        )
        open_image_btn.pack(side=tk.LEFT, padx=5)
        
        # Open video button
        open_video_btn = ttk.Button(
            button_frame, 
            text="Open Video", 
            command=self.open_video,
            style="Accent.TButton"
        )
        open_video_btn.pack(side=tk.LEFT, padx=5)
        
        # Settings button (placeholder)
        settings_btn = ttk.Button(
            button_frame, 
            text="Settings", 
            command=self.show_settings
        )
        settings_btn.pack(side=tk.LEFT, padx=5)
    
    def create_detection_tab(self):
        """Create the main detection tab"""
        detection_frame = ttk.Frame(self.notebook)
        self.notebook.add(detection_frame, text="Detection")
        
        # Configure the layout
        detection_frame.columnconfigure(0, weight=3)  # Left side (preview)
        detection_frame.columnconfigure(1, weight=2)  # Right side (controls and results)
        detection_frame.rowconfigure(0, weight=1)
        
        # Left side: Preview area with default message
        self.preview_frame = ttk.Frame(detection_frame)
        self.preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Placeholder for when no image is loaded - enhanced visibility
        self.placeholder_label = ttk.Label(
            self.preview_frame, 
            text="No media loaded.\nUse 'Open Image' or 'Open Video' to begin detection.",
            justify="center",
            font=("Segoe UI", 12, "bold"),
            foreground="#FFFFFF"
        )
        self.placeholder_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Right side: Controls and results
        controls_frame = ttk.Frame(detection_frame)
        controls_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        controls_frame.columnconfigure(0, weight=1)
        controls_frame.rowconfigure(1, weight=1)
        
        # Detection controls
        self.create_detection_controls(controls_frame)
        
        # Results display
        self.create_results_display(controls_frame)
    
    def create_detection_controls(self, parent):
        """Create detection control panel"""
        controls_panel = ttk.LabelFrame(parent, text="Detection Controls")
        controls_panel.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        # Grid configuration for controls
        controls_panel.columnconfigure(0, weight=1)
        controls_panel.columnconfigure(1, weight=1)
        
        # Detection threshold
        threshold_label = ttk.Label(controls_panel, text="Detection Threshold:")
        threshold_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        
        self.threshold_var = tk.DoubleVar(value=0.5)
        threshold_scale = ttk.Scale(
            controls_panel,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self.threshold_var,
            length=150
        )
        threshold_scale.grid(row=0, column=1, sticky="ew", padx=10, pady=5)
        
        # Threshold value display
        self.threshold_value_label = ttk.Label(controls_panel, text="0.50")
        self.threshold_value_label.grid(row=0, column=2, padx=(0, 10))
        
        # Update threshold value label when slider moves
        def update_threshold_label(*args):
            self.threshold_value_label.config(text=f"{self.threshold_var.get():.2f}")
        self.threshold_var.trace_add("write", update_threshold_label)
        
        # Analysis method (simplified for this demo)
        method_label = ttk.Label(controls_panel, text="Analysis Method:")
        method_label.grid(row=1, column=0, sticky="w", padx=10, pady=5)
        
        self.method_var = tk.StringVar(value="full")
        method_combo = ttk.Combobox(
            controls_panel,
            textvariable=self.method_var,
            values=["full", "face_only", "frequency"],
            state="readonly",
            width=15
        )
        method_combo.grid(row=1, column=1, sticky="ew", padx=10, pady=5)
        
        # Detect button
        detect_button = ttk.Button(
            controls_panel,
            text="Detect Deepfake",
            command=self.start_detection,
            style="Accent.TButton"
        )
        detect_button.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
    
    def create_results_display(self, parent):
        """Create results display panel"""
        results_panel = ttk.LabelFrame(parent, text="Detection Results")
        results_panel.grid(row=1, column=0, sticky="nsew")
        
        # Configure grid
        results_panel.columnconfigure(0, weight=1)
        results_panel.rowconfigure(2, weight=1)
        
        # Result heading
        result_frame = ttk.Frame(results_panel)
        result_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        
        result_label = ttk.Label(result_frame, text="Result:", font=("Segoe UI", 12))
        result_label.pack(side=tk.LEFT)
        
        self.result_value = ttk.Label(result_frame, text="N/A", style="Result.TLabel")
        self.result_value.pack(side=tk.LEFT, padx=(5, 0))
        
        # Confidence
        confidence_frame = ttk.Frame(results_panel)
        confidence_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        
        confidence_label = ttk.Label(confidence_frame, text="Confidence:", font=("Segoe UI", 12))
        confidence_label.pack(side=tk.LEFT)
        
        self.confidence_value = ttk.Label(confidence_frame, text="N/A", font=("Segoe UI", 12, "bold"))
        self.confidence_value.pack(side=tk.LEFT, padx=(5, 0))
        
        # Detailed results (scrollable text with enhanced visibility)
        details_frame = ttk.Frame(results_panel)
        details_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        
        self.details_text = scrolledtext.ScrolledText(
            details_frame,
            wrap=tk.WORD,
            width=40,
            height=10,
            bg="#333333",  # Darker background for contrast
            fg="#FFFFFF",  # White text for maximum visibility
            font=("Consolas", 11),  # Monospaced font for better readability
            insertbackground="#FFFFFF",  # White cursor
            selectbackground=ACCENT_COLOR,  # Selection background
            selectforeground="#FFFFFF",  # Selection text
            relief=tk.SOLID,  # Add border for better visibility
            borderwidth=1,
            padx=5,
            pady=5
        )
        self.details_text.pack(fill=tk.BOTH, expand=True)
        self.details_text.insert(tk.END, "No analysis has been performed yet.")
        self.details_text.config(state=tk.DISABLED)
        
        # Export button
        export_button = ttk.Button(
            results_panel,
            text="Export Report",
            command=self.export_report
        )
        export_button.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
    
    def create_batch_tab(self):
        """Create the batch processing tab"""
        batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(batch_frame, text="Batch Processing")
        
        # Configure the layout
        batch_frame.columnconfigure(0, weight=1)
        batch_frame.rowconfigure(1, weight=1)
        
        # Controls at the top
        controls_frame = ttk.Frame(batch_frame)
        controls_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        # Select directory button
        select_dir_btn = ttk.Button(
            controls_frame,
            text="Select Directory",
            command=self.select_directory
        )
        select_dir_btn.pack(side=tk.LEFT, padx=5)
        
        # Start batch button
        batch_btn = ttk.Button(
            controls_frame,
            text="Start Batch Processing",
            command=self.start_batch_processing,
            style="Accent.TButton"
        )
        batch_btn.pack(side=tk.LEFT, padx=5)
        
        # Directory label
        self.directory_var = tk.StringVar(value="No directory selected")
        dir_label = ttk.Label(controls_frame, textvariable=self.directory_var, font=("Segoe UI", 11))
        dir_label.pack(side=tk.LEFT, padx=20)
        
        # Results list
        results_frame = ttk.LabelFrame(batch_frame, text="Batch Results")
        results_frame.grid(row=1, column=0, sticky="nsew")
        
        # Configure TreeView styling for better visibility
        self.style.configure("Treeview", 
                           background=ENTRY_BG,
                           foreground=DARK_FG,
                           fieldbackground=ENTRY_BG,
                           font=("Segoe UI", BASE_FONT_SIZE))
        self.style.configure("Treeview.Heading", 
                           background=DARK_BG, 
                           foreground=DARK_FG,
                           font=("Segoe UI", BASE_FONT_SIZE, "bold"))
        self.style.map("Treeview", 
                     background=[('selected', ACCENT_COLOR)],
                     foreground=[('selected', DARK_FG)])
        
        # Treeview for results
        columns = ("file", "result", "confidence", "time")
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show="headings")
        
        # Define headings
        self.results_tree.heading("file", text="Filename")
        self.results_tree.heading("result", text="Result")
        self.results_tree.heading("confidence", text="Confidence")
        self.results_tree.heading("time", text="Processing Time")
        
        # Define columns
        self.results_tree.column("file", width=300)
        self.results_tree.column("result", width=100)
        self.results_tree.column("confidence", width=100)
        self.results_tree.column("time", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscroll=scrollbar.set)
        
        # Grid layout
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure tag colors with enhanced visibility
        self.results_tree.tag_configure("fake", foreground=FAKE_COLOR, font=("Segoe UI", BASE_FONT_SIZE, "bold"))
        self.results_tree.tag_configure("real", foreground=REAL_COLOR, font=("Segoe UI", BASE_FONT_SIZE, "bold"))
        self.results_tree.tag_configure("error", foreground=WARNING_COLOR, font=("Segoe UI", BASE_FONT_SIZE, "bold"))
    
    def create_help_tab(self):
        """Create the help and information tab"""
        help_frame = ttk.Frame(self.notebook)
        self.notebook.add(help_frame, text="Help")
        
        # Configure the layout
        help_frame.columnconfigure(0, weight=1)
        help_frame.rowconfigure(0, weight=1)
        
        # Create a scrolled text for help content with enhanced visibility
        help_text = scrolledtext.ScrolledText(
            help_frame,
            wrap=tk.WORD,
            width=80,
            height=25,
            bg="#333333",  # Darker background for contrast
            fg="#FFFFFF",  # White text for maximum visibility
            font=("Segoe UI", 11),  # Larger font size
            insertbackground="#FFFFFF",  # White cursor
            selectbackground=ACCENT_COLOR,  # Selection background
            selectforeground="#FFFFFF",  # Selection text
            relief=tk.SOLID,  # Add border for better visibility
            borderwidth=1,
            padx=10,
            pady=10
        )
        help_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Add help content
        help_content = """
DeepFake Detector Help

This application helps you detect if an image or video has been manipulated using deepfake technology.

Quick Start:
1. Click "Open Image" or "Open Video" to select a file to analyze
2. Adjust the detection threshold if needed (default: 0.5)
3. Click "Detect Deepfake" to start the analysis
4. View the results and export a detailed report if needed

Understanding Results:
- The detector produces a probability score between 0 and 1
- Higher score = More likely to be a deepfake
- Scores above the threshold are classified as fake
- Confidence reflects how certain the model is of its prediction

Detection Methods:
- Full Analysis: Analyzes the entire image and any faces detected
- Face Only: Focuses only on facial features for detection
- Frequency: Analyzes frequency domain artifacts common in generated images

Batch Processing:
1. Select a directory containing images or videos
2. Click "Start Batch Processing"
3. Results will be displayed in the table
4. Export results when complete

Notes:
- The detector may not catch all deepfakes, especially sophisticated ones
- Very compressed media may be harder to accurately analyze
- Processing videos takes longer than images
- GPU acceleration is used when available for faster processing

For more information, visit: https://github.com/yourusername/deepfake-detector
        """
        
        help_text.insert(tk.END, help_content)
        help_text.config(state=tk.DISABLED)  # Make read-only
    
    def open_image(self):
        """Open an image file for analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return  # User cancelled
        
        self.current_image_path = file_path
        self.current_video_path = None
        
        self.update_status(f"Loaded image: {os.path.basename(file_path)}")
        
        # Load and display the image preview
        try:
            image = Image.open(file_path)
            self.display_image_preview(image)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")
    
    def open_video(self):
        """Open a video file for analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return  # User cancelled
        
        self.current_video_path = file_path
        self.current_image_path = None
        
        self.update_status(f"Loaded video: {os.path.basename(file_path)}")
        
        # Extract and display a preview frame from the video
        try:
            # Try to extract first frame using the utility function
            frames = extract_frames(file_path, max_frames=1)
            if frames:
                first_frame = frames[0]
                # Convert to PIL image
                preview_image = Image.fromarray(first_frame)
                self.display_image_preview(preview_image)
                
                # Update details
                self.details_text.config(state=tk.NORMAL)
                self.details_text.delete(1.0, tk.END)
                self.details_text.insert(tk.END, f"Video: {os.path.basename(file_path)}\n")
                self.details_text.insert(tk.END, f"Use 'Detect Deepfake' to analyze this video.")
                self.details_text.config(state=tk.DISABLED)
            else:
                raise ValueError("Could not extract frames from video")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load video: {e}")
    
    def display_image_preview(self, image):
        """Display an image in the preview area"""
        # Clear the preview area
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        
        # Resize image to fit preview area
        preview_width = self.preview_frame.winfo_width() or 600
        preview_height = self.preview_frame.winfo_height() or 400
        
        # Ensure the image is not resized larger than its original size
        original_width, original_height = image.size
        scale = min(
            preview_width / original_width, 
            preview_height / original_height,
            1.0  # Don't enlarge
        )
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize with high quality
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage for display
        photo = ImageTk.PhotoImage(resized_image)
        
        # Create canvas for display
        canvas = tk.Canvas(
            self.preview_frame, 
            width=preview_width, 
            height=preview_height,
            bg=DARK_BG,
            highlightthickness=0
        )
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Calculate center position
        x = (preview_width - new_width) // 2
        y = (preview_height - new_height) // 2
        
        # Display the image
        canvas.create_image(x, y, anchor=tk.NW, image=photo)
        
        # Store reference to prevent garbage collection
        canvas.image = photo
    
    def start_detection(self):
        """Start the deepfake detection process"""
        if self.is_processing:
            messagebox.showinfo("Processing", "Detection is already in progress")
            return
        
        if not self.current_image_path and not self.current_video_path:
            messagebox.showinfo("No Media", "Please open an image or video first")
            return
            
        if self.detector is None:
            messagebox.showerror("Detector Error", "The deepfake detector could not be initialized. Please check that all required models are available.")
            return
        
        # Start detection in a separate thread to keep UI responsive
        self.is_processing = True
        detection_thread = threading.Thread(
            target=self.run_detection,
            daemon=True
        )
        detection_thread.start()
        
        # Update UI
        self.update_status("Detection in progress...")
    
    def run_detection(self):
        """Run detection in a separate thread"""
        try:
            # Get detection parameters
            threshold = self.threshold_var.get()
            method = self.method_var.get()
            
            # Show processing indicator
            self.show_processing_indicator()
            
            # Run detection based on media type
            if self.current_image_path:
                # Process image
                result = self.detector.detect_image(
                    self.current_image_path,
                    threshold=threshold
                )
            elif self.current_video_path:
                # Process video
                result = self.detector.detect_video(
                    self.current_video_path,
                    threshold=threshold,
                    max_frames=30  # Limit frames for performance
                )
            else:
                # Should not happen
                raise ValueError("No media selected")
            
            # Store result
            self.last_result = result
            
            # Update UI with results
            self.root.after(0, lambda: self.display_results(result))
            
        except Exception as e:
            # Show error in UI thread
            self.root.after(0, lambda: self.show_error(str(e)))
        finally:
            # Reset processing flag
            self.is_processing = False
    
    def show_processing_indicator(self):
        """Show a visual indicator that processing is happening"""
        # Create a progress dialog
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Processing")
        self.progress_window.geometry("300x150")
        self.progress_window.transient(self.root)
        self.progress_window.grab_set()
        self.progress_window.configure(bg=DARK_BG)
        self.progress_window.resizable(False, False)
        
        # Center frame
        frame = ttk.Frame(self.progress_window, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Processing message
        message = ttk.Label(
            frame,
            text="Analyzing media...\nThis may take a moment.",
            justify="center",
            font=("Segoe UI", 12),
            foreground=DARK_FG
        )
        message.pack(pady=(0, 15))
        
        # Progress bar
        progress = ttk.Progressbar(
            frame,
            mode="indeterminate",
            length=250
        )
        progress.pack(pady=5)
        progress.start(15)  # Start animation
        
        # Apply high contrast styling
        apply_high_contrast(self.progress_window)
        
        # Update UI
        self.root.update_idletasks()
    
    def display_results(self, result):
        """Display detection results in the UI"""
        if 'error' in result:
            self.show_error(result['error'])
            return
        
        # Get the prediction value
        if 'overall_prediction' in result:
            prediction = result['overall_prediction']
        else:
            prediction = 0.0
        
        # Determine if fake or real based on threshold
        threshold = self.threshold_var.get()
        is_fake = prediction >= threshold
        
        # Update result label
        result_text = "FAKE" if is_fake else "REAL"
        result_style = "Fake.TLabel" if is_fake else "Real.TLabel"
        self.result_value.config(text=result_text, style=result_style)
        
        # Update confidence
        confidence_text = f"{prediction:.2%}"
        self.confidence_value.config(text=confidence_text)
        
        # Update details text
        details = self.format_result_details(result)
        self.update_details_text(details)
        
        # Update preview with detection visualization
        if result.get('report_path') and os.path.exists(result['report_path']):
            try:
                report_image = Image.open(result['report_path'])
                self.display_image_preview(report_image)
            except Exception as e:
                print(f"Error displaying report image: {e}")
        
        # Update status
        self.update_status(f"Detection complete. Result: {result_text} ({confidence_text})")
    
    def format_result_details(self, result):
        """Format detection results for display in details text"""
        details = ""
        
        # Media type and path
        if 'image_path' in result:
            details += f"Image: {os.path.basename(result['image_path'])}\n"
        elif 'video_path' in result:
            details += f"Video: {os.path.basename(result['video_path'])}\n"
        
        details += f"\nDetection completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # Add overall prediction
        if 'overall_prediction' in result:
            details += f"\nOverall prediction: {result['overall_prediction']:.2%}\n"
        
        # Add processing time
        if 'detection_time' in result:
            details += f"Processing time: {result['detection_time']:.2f} seconds\n"
        
        # Add face detection results
        if 'faces_detected' in result:
            details += f"\nFaces detected: {result['faces_detected']}\n"
            
            if result['face_predictions']:
                details += "\nFace analysis results:\n"
                for i, face in enumerate(result['face_predictions']):
                    details += f"  Face {i+1}: "
                    details += f"{'FAKE' if face['is_fake'] else 'REAL'} "
                    details += f"({face['probability']:.2%})\n"
        
        # Add video specific details
        if 'frames_analyzed' in result:
            details += f"\nFrames analyzed: {result['frames_analyzed']}\n"
            
            # Add summary of frame predictions
            if 'frame_predictions' in result:
                fake_frames = sum(1 for p in result['frame_predictions'] if p['is_fake'])
                details += f"Frames detected as fake: {fake_frames}\n"
                details += f"Frames detected as real: {result['frames_analyzed'] - fake_frames}\n"
        
        # Add report path
        if 'report_path' in result:
            details += f"\nDetailed report saved to:\n{result['report_path']}\n"
        
        return details
    
    def update_details_text(self, text):
        """Update the details text widget"""
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, text)
        self.details_text.config(state=tk.DISABLED)
    
    def export_report(self):
        """Export detection results as a detailed report"""
        if not self.last_result:
            messagebox.showinfo("No Results", "No detection results to export")
            return
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".pdf",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("PNG Image", "*.png"),
                ("Text files", "*.txt"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            # For this demo, we'll just copy the existing report if it exists
            if self.last_result.get('report_path') and os.path.exists(self.last_result['report_path']):
                import shutil
                shutil.copy2(self.last_result['report_path'], file_path)
                messagebox.showinfo("Export Successful", f"Report exported to {file_path}")
            else:
                # If no report exists, create a simple text report
                with open(file_path, 'w') as f:
                    f.write(self.format_result_details(self.last_result))
                messagebox.showinfo("Export Successful", f"Text report exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting report: {e}")
    
    def select_directory(self):
        """Select a directory for batch processing"""
        directory = filedialog.askdirectory(title="Select Directory for Batch Processing")
        if directory:
            self.directory_var.set(directory)
    
    def start_batch_processing(self):
        """Start batch processing of files in the selected directory"""
        directory = self.directory_var.get()
        if directory == "No directory selected" or not os.path.isdir(directory):
            messagebox.showinfo("No Directory", "Please select a valid directory first")
            return
        
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Start batch processing in a separate thread
        threading.Thread(target=self.run_batch_processing, args=(directory,), daemon=True).start()
    
    def run_batch_processing(self, directory):
        """Run batch processing in a background thread"""
        # Update status
        self.update_status(f"Batch processing started on {directory}...")
        
        # Get threshold
        threshold = self.threshold_var.get()
        
        # Get all compatible files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        
        files = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in image_extensions or ext in video_extensions:
                    files.append(file_path)
        
        # Update status
        self.update_status(f"Found {len(files)} files to process...")
        
        # Process each file
        for i, file_path in enumerate(files):
            try:
                # Update status
                self.update_status(f"Processing file {i+1}/{len(files)}: {os.path.basename(file_path)}")
                
                # Determine if image or video
                ext = os.path.splitext(file_path)[1].lower()
                
                # Process file
                if ext in image_extensions:
                    result = self.detector.detect_image(file_path, threshold=threshold)
                else:
                    result = self.detector.detect_video(file_path, threshold=threshold, max_frames=20)
                
                # Add to results
                self.add_batch_result(file_path, result)
                
            except Exception as e:
                # Add error to results
                self.add_batch_error(file_path, str(e))
        
        # Update status
        self.update_status(f"Batch processing complete. Processed {len(files)} files.")
    
    def add_batch_result(self, file_path, result):
        """Add a result to the batch results tree"""
        # Extract filename
        filename = os.path.basename(file_path)
        
        # Get overall prediction
        prediction = result.get('overall_prediction', 0.0)
        
        # Determine result text
        threshold = self.threshold_var.get()
        result_text = "FAKE" if prediction >= threshold else "REAL"
        
        # Format confidence
        confidence = f"{prediction:.2%}"
        
        # Get processing time
        proc_time = f"{result.get('detection_time', 0.0):.2f}s"
        
        # Add to treeview
        item_id = self.results_tree.insert(
            "",
            tk.END,
            values=(filename, result_text, confidence, proc_time)
        )
        
        # Color based on result
        if prediction >= threshold:
            self.results_tree.item(item_id, tags=("fake",))
        else:
            self.results_tree.item(item_id, tags=("real",))
        
        # Configure tag colors
        self.results_tree.tag_configure("fake", foreground=FAKE_COLOR)
        self.results_tree.tag_configure("real", foreground=REAL_COLOR)
    
    def add_batch_error(self, file_path, error_message):
        """Add an error to the batch results tree"""
        # Extract filename
        filename = os.path.basename(file_path)
        
        # Add to treeview
        item_id = self.results_tree.insert(
            "",
            tk.END,
            values=(filename, "ERROR", "-", "-")
        )
        
        # Color based on error
        self.results_tree.item(item_id, tags=("error",))
        
        # Configure tag color
        self.results_tree.tag_configure("error", foreground=WARNING_COLOR)
    
    def show_settings(self):
        """Show settings dialog"""
        settings_dialog = tk.Toplevel(self.root)
        settings_dialog.title("Settings")
        settings_dialog.geometry("400x300")
        settings_dialog.transient(self.root)
        settings_dialog.grab_set()
        settings_dialog.configure(bg=DARK_BG)
        
        # Settings content
        frame = ttk.Frame(settings_dialog, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Add GPU info button
        gpu_info_btn = ttk.Button(
            frame,
            text="GPU Information",
            command=lambda: show_gpu_info(settings_dialog),
            style="Accent.TButton"
        )
        gpu_info_btn.pack(pady=10, fill=tk.X)
        
        # Add other settings options
        label = ttk.Label(
            frame, 
            text="Additional settings will be implemented in a future version.", 
            justify="center",
            font=("Segoe UI", 12),
            foreground=DARK_FG,
            wraplength=300
        )
        label.pack(pady=20)
        
        close_button = ttk.Button(
            frame, 
            text="Close", 
            command=settings_dialog.destroy,
            style="Accent.TButton"
        )
        close_button.pack(pady=10)
        
        # Apply high contrast to the dialog
        apply_high_contrast(settings_dialog)
    
    def show_error(self, error_message):
        """Show an error message in the UI"""
        # Update result label
        self.result_value.config(text="ERROR", style="Fake.TLabel")
        
        # Update confidence
        self.confidence_value.config(text="")
        
        # Update details text
        self.update_details_text(f"Error during detection:\n\n{error_message}")
        
        # Update status
        self.update_status(f"Error: {error_message}")
        
        # Show error dialog
        messagebox.showerror("Detection Error", error_message)
    
    def update_status(self, message):
        """Update the status bar message"""
        self.status_var.set(message)
        self.root.update_idletasks()

def main():
    """Main function to start the GUI application"""
    # Set global exception handler for Tkinter errors
    tk.Tk.report_callback_exception = show_exception_dialog
    
    root = tk.Tk()
    root.title("DeepFake Detector")
    
    # Set app icon (if available)
    try:
        root.iconbitmap("icon.ico")
    except:
        pass
    
    # Configure dark theme
    root.configure(bg=DARK_BG)
    
    # Start application
    app = DeepfakeDetectorGUI(root)
    
    # Run main loop
    root.mainloop()

def show_exception_dialog(exc_type, exc_value, exc_traceback):
    """Show a dialog with exception information"""
    # Format the exception
    import traceback
    error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    
    # Create dialog
    error_window = tk.Toplevel()
    error_window.title("Error")
    error_window.geometry("500x400")
    error_window.configure(bg=DARK_BG)
    
    # Error message frame
    frame = ttk.Frame(error_window, padding=20)
    frame.pack(fill=tk.BOTH, expand=True)
    
    # Error header
    header = ttk.Label(
        frame,
        text="An error occurred:",
        font=("Segoe UI", 12, "bold"),
        foreground=FAKE_COLOR
    )
    header.pack(pady=(0, 10), anchor="w")
    
    # Scrollable error details
    error_text = scrolledtext.ScrolledText(
        frame,
        wrap=tk.WORD,
        width=60,
        height=15,
        bg=ENTRY_BG,
        fg=DARK_FG,
        font=("Consolas", 10)
    )
    error_text.pack(fill=tk.BOTH, expand=True)
    error_text.insert(tk.END, error_msg)
    error_text.config(state=tk.DISABLED)
    
    # Close button
    close_btn = ttk.Button(
        frame,
        text="Close",
        command=error_window.destroy,
        style="Accent.TButton"
    )
    close_btn.pack(pady=10)
    
    # Apply high contrast
    try:
        apply_high_contrast(error_window)
    except:
        pass

def apply_high_contrast(widget):
    """Apply high contrast settings to all widgets recursively"""
    try:
        # Fix comboboxes, which often have visibility issues
        if widget.winfo_class() == 'TCombobox':
            widget.configure(foreground=DARK_FG)
            try:
                widget.tk.eval(f'[ttk::combobox::PopdownWindow {widget}].f.l configure -background {ENTRY_BG} -foreground {DARK_FG}')
            except:
                pass  # Ignore if combobox popdown window doesn't exist yet
        
        # Fix text widgets
        elif widget.winfo_class() == 'Text' or widget.winfo_class() == 'ScrolledText':
            widget.configure(
                background=ENTRY_BG,
                foreground=DARK_FG,
                insertbackground=DARK_FG,  # Cursor color
                selectbackground=ACCENT_COLOR,
                selectforeground=DARK_FG
            )
        
        # Fix labels
        elif widget.winfo_class() == 'TLabel':
            if not any(style in str(widget) for style in ["Fake.TLabel", "Real.TLabel", "Title.TLabel"]):
                widget.configure(foreground=DARK_FG)
        
        # Fix entry widgets
        elif widget.winfo_class() == 'TEntry':
            widget.configure(
                foreground=DARK_FG,
                background=ENTRY_BG
            )
        
        # Fix buttons
        elif widget.winfo_class() == 'TButton':
            if not 'Accent.TButton' in str(widget):
                widget.configure(foreground=DARK_FG)
        
        # Fix TreeView elements to ensure consistent styling
        elif widget.winfo_class() == 'Treeview':
            widget.configure(
                background=ENTRY_BG,
                foreground=DARK_FG,
                fieldbackground=ENTRY_BG
            )
            # Also style headings if possible
            try:
                for col in widget['columns']:
                    widget.heading(col, foreground=DARK_FG)
            except:
                pass
        
        # Process all children widgets
        for child in widget.winfo_children():
            apply_high_contrast(child)
    except Exception as e:
        # Just ignore any errors in styling application
        pass

def verify_gpu_support():
    """Verify if TensorFlow can see and use GPUs"""
    devices = tf.config.list_physical_devices()
    gpu_devices = tf.config.list_physical_devices('GPU')
    
    info = f"TensorFlow version: {tf.__version__}\n\n"
    
    if gpu_devices:
        info += f"GPU support is available!\nFound {len(gpu_devices)} GPU device(s):\n"
        for i, device in enumerate(gpu_devices):
            info += f"  {i+1}. {device.name}\n"
        
        # Add GPU memory info if available
        try:
            gpu_memory = []
            for i, device in enumerate(gpu_devices):
                memory_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                total_memory = memory_info['current'] / (1024 ** 3)  # Convert to GB
                gpu_memory.append(total_memory)
                info += f"     Memory: {total_memory:.2f} GB\n"
        except:
            info += "     (GPU memory information not available)\n"
    else:
        info += "No GPU devices detected. Using CPU for computation.\n"
        cpu_devices = tf.config.list_physical_devices('CPU')
        info += f"Available CPU devices: {len(cpu_devices)}\n"
    
    return info

def show_gpu_info(parent):
    """Show GPU information in a dialog"""
    gpu_info = verify_gpu_support()
    
    # Create a dialog
    info_dialog = tk.Toplevel(parent)
    info_dialog.title("GPU Information")
    info_dialog.geometry("600x400")
    info_dialog.transient(parent)
    info_dialog.grab_set()
    info_dialog.configure(bg=DARK_BG)
    
    # Create frame
    frame = ttk.Frame(info_dialog, padding=20)
    frame.pack(fill=tk.BOTH, expand=True)
    
    # Create text widget
    info_text = scrolledtext.ScrolledText(
        frame,
        wrap=tk.WORD,
        bg=ENTRY_BG,
        fg=DARK_FG,
        font=("Consolas", 11)
    )
    info_text.pack(fill=tk.BOTH, expand=True, pady=10)
    
    # Add info
    info_text.insert(tk.END, "GPU INFORMATION\n")
    info_text.insert(tk.END, "---------------\n\n")
    
    info_text.insert(tk.END, gpu_info)
    
    # Make read-only
    info_text.config(state=tk.DISABLED)
    
    # Close button
    close_btn = ttk.Button(
        frame,
        text="Close",
        command=info_dialog.destroy,
        style="Accent.TButton"
    )
    close_btn.pack(pady=10)
    
    # Apply high contrast
    apply_high_contrast(info_dialog)

if __name__ == "__main__":
    # Set exception handler
    sys.excepthook = show_exception_dialog
    
    # Run GUI
    main() 