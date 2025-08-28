import gradio as gr
import pandas as pd
import os
import shutil
from pathlib import Path
import tempfile
import json
from typing import List, Tuple, Optional, Dict, Any
import subprocess
import sys
from datetime import datetime
import zipfile
import numpy as np
from PIL import Image
import time

# Import local modules
from utils import ImageProcessor, ScaleBarStyle, ProfileStyle, InventoryPosition
from ultralytics import YOLO
import yaml
import torch

class CeramaticGradioApp:
    def __init__(self):
        self.processor = None
        self.temp_dirs = []
        self.training_process = None
        
    def cleanup_temp_dirs(self):
        """Clean up temporary directories"""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        self.temp_dirs = []
    
    def create_temp_dir(self) -> str:
        """Create and track a temporary directory"""
        temp_dir = tempfile.mkdtemp(prefix="ceramatic_")
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def process_images_workflow(
        self,
        # Step 1 inputs
        model_file,
        # Step 2 inputs
        image_files,
        metadata_file,
        metadata_table,
        # Step 3 inputs
        pixel_cm_ratio,
        confidence_threshold,
        median_filter,
        # Step 4 inputs
        profile_style,
        inv_position,
        scale_bar_style,
        scale_cm,
        num_segments,
        # Step 5 inputs
        features_dict,
        # Progress
        progress=gr.Progress(track_tqdm=True)
    ):
        """Process images with step-by-step workflow"""
        try:
            progress(0, desc="Initializing...")
            
            # Create temporary directories
            temp_dir = self.create_temp_dir()
            imgs_dir = os.path.join(temp_dir, "images")
            os.makedirs(imgs_dir, exist_ok=True)
            
            # Validation
            if not model_file:
                return None, "❌ Model file required", None
            if not image_files:
                return None, "❌ No images uploaded", None
            if metadata_file is None and (metadata_table is None or len(metadata_table) == 0):
                return None, "❌ Metadata required (upload file or fill table)", None
            
            # Save uploaded images
            progress(0.2, desc="Loading images...")
            for i, img_file in enumerate(image_files):
                if hasattr(img_file, 'name'):
                    shutil.copy(img_file.name, imgs_dir)
                else:
                    shutil.copy(img_file, imgs_dir)
            
            # Handle metadata
            progress(0.3, desc="Processing metadata...")
            metadata_path = os.path.join(temp_dir, "metadata.xlsx")
            if metadata_file is not None:
                if hasattr(metadata_file, 'name'):
                    shutil.copy(metadata_file.name, metadata_path)
                else:
                    shutil.copy(metadata_file, metadata_path)
            else:
                # Use table data
                df = pd.DataFrame(metadata_table, columns=["TAV", "INV", "DIAM", "FLIP"])
                df = df.dropna(subset=['TAV', 'INV'])
                df['DIAM'] = pd.to_numeric(df['DIAM'], errors='coerce')
                df['FLIP'] = pd.to_numeric(df['FLIP'], errors='coerce').fillna(0).astype(int)
                df.to_excel(metadata_path, index=False)
            
            # Save model file
            progress(0.4, desc="Loading YOLO model...")
            model_path = os.path.join(temp_dir, "model.pt")
            if hasattr(model_file, 'name'):
                shutil.copy(model_file.name, model_path)
            else:
                shutil.copy(model_file, model_path)
            
            # Initialize processor
            progress(0.5, desc="Initializing processor...")
            self.processor = ImageProcessor(
                model_path=model_path,
                pixel_cm_ratio=pixel_cm_ratio
            )
            
            # Process images
            progress(0.6, desc="Processing images...")
            results = self.processor.process_images(
                imgs_dir=imgs_dir,
                tabular_file=metadata_path,
                confidence_threshold=confidence_threshold,
                median_filter=int(median_filter),
                inventory_position=InventoryPosition[inv_position.upper().replace(" ", "_")],
                profile_style=ProfileStyle[profile_style.upper()],
                scale_bar_style=ScaleBarStyle[scale_bar_style.upper()],
                scale_cm=scale_cm,
                num_segments=int(num_segments),
                add_bar=features_dict.get('scale_bar', True),
                add_diameter=features_dict.get('diameter', False),
                add_scale_cm=features_dict.get('scale_text', False),
                diagnostic=features_dict.get('diagnostic', False),
                diagnostic_plots=features_dict.get('diagnostic_plots', False),
                add_continuation_lines=features_dict.get('continuation_lines', False)
            )
            
            # Collect results
            progress(0.8, desc="Collecting results...")
            processed_dir = "processed_imgs"
            output_images = []
            
            if os.path.exists(processed_dir):
                for img_file in os.listdir(processed_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(processed_dir, img_file)
                        output_images.append(img_path)
            
            # PCA analysis if requested
            if features_dict.get('pca_analysis', False) and len(results) >= 3:
                progress(0.9, desc="Running PCA analysis...")
                self.processor.perform_pca_analysis(results)
            
            # Create zip
            progress(0.95, desc="Creating archive...")
            zip_path = os.path.join(temp_dir, "ceramatic_results.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for img_path in output_images:
                    zipf.write(img_path, os.path.basename(img_path))
                
                if features_dict.get('diagnostic_plots', False) and os.path.exists("prediction_diagnostic"):
                    for root, dirs, files in os.walk("prediction_diagnostic"):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, ".")
                            zipf.write(file_path, arcname)
                
                if features_dict.get('pca_analysis', False) and os.path.exists("statistical_analysis"):
                    for root, dirs, files in os.walk("statistical_analysis"):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, ".")
                            zipf.write(file_path, arcname)
            
            progress(1.0, desc="Complete!")
            
            # Gallery for preview
            gallery_images = output_images[:8]  # Show first 8
            
            return gallery_images, f"✅ Successfully processed {len(output_images)} images", zip_path
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}", None

    def create_interface(self):
        """Create the Gradio interface with sequential workflow"""
        with gr.Blocks(
            title="Ceramatic 2.0", 
            theme=gr.themes.Default(
                primary_hue="orange",
                secondary_hue="blue",
                neutral_hue="gray",
                radius_size="md",
                spacing_size="md",
                font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"]
            ),
            css="""
            /* Modern Professional Layout */
            .gradio-container {
                max-width: 1200px !important;
                margin: auto !important;
            }
            
            /* Progress bar at bottom */
            .progress-bar-container {
                position: fixed !important;
                bottom: 0 !important;
                left: 0 !important;
                right: 0 !important;
                height: 60px !important;
                background: rgba(255, 255, 255, 0.95) !important;
                backdrop-filter: blur(10px) !important;
                border-top: 2px solid #e5e7eb !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                z-index: 1000 !important;
                box-shadow: 0 -4px 6px rgba(0, 0, 0, 0.1) !important;
            }
            
            .progress-bar {
                width: 80% !important;
                max-width: 800px !important;
                height: 30px !important;
                background: #f3f4f6 !important;
                border-radius: 15px !important;
                overflow: hidden !important;
                position: relative !important;
            }
            
            .progress-fill {
                height: 100% !important;
                background: linear-gradient(90deg, #ff6b6b 0%, #ee5a6f 50%, #ff6b6b 100%) !important;
                background-size: 200% 100% !important;
                animation: shimmer 2s linear infinite !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                color: white !important;
                font-weight: 600 !important;
                font-size: 14px !important;
            }
            
            @keyframes shimmer {
                0% { background-position: 200% 0; }
                100% { background-position: -200% 0; }
            }
            
            /* Step cards */
            .step-card {
                border: 2px solid #e5e7eb !important;
                border-radius: 12px !important;
                padding: 20px !important;
                margin-bottom: 20px !important;
                background: white !important;
                transition: all 0.3s ease !important;
            }
            
            .step-card:hover {
                border-color: #ff6b6b !important;
                box-shadow: 0 4px 12px rgba(255, 107, 107, 0.15) !important;
            }
            
            .step-header {
                display: flex !important;
                align-items: center !important;
                gap: 12px !important;
                margin-bottom: 16px !important;
            }
            
            .step-number {
                width: 36px !important;
                height: 36px !important;
                background: #ff6b6b !important;
                color: white !important;
                border-radius: 50% !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                font-weight: 700 !important;
                font-size: 18px !important;
            }
            
            .step-title {
                font-size: 20px !important;
                font-weight: 600 !important;
                color: #1f2937 !important;
            }
            
            /* Improved inputs */
            .gr-button-primary {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%) !important;
                border: none !important;
                height: 48px !important;
                font-size: 16px !important;
                font-weight: 600 !important;
                letter-spacing: 0.5px !important;
                transition: all 0.3s ease !important;
            }
            
            .gr-button-primary:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 24px rgba(238, 90, 111, 0.3) !important;
            }
            
            /* Metadata table improvements */
            .dataframe {
                width: 100% !important;
                border-radius: 8px !important;
                overflow: hidden !important;
            }
            
            .dataframe input {
                border: 1px solid #e5e7eb !important;
                border-radius: 4px !important;
                padding: 6px 10px !important;
            }
            
            /* Toggle switches */
            .toggle-group {
                display: flex !important;
                gap: 16px !important;
                flex-wrap: wrap !important;
            }
            
            .toggle-item {
                display: flex !important;
                align-items: center !important;
                gap: 8px !important;
            }
            
            /* Download button special style */
            .download-button {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
                color: white !important;
                padding: 12px 24px !important;
                border-radius: 8px !important;
                font-weight: 600 !important;
                display: inline-flex !important;
                align-items: center !important;
                gap: 8px !important;
                transition: all 0.3s ease !important;
            }
            
            .download-button:hover {
                transform: scale(1.05) !important;
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
            }
            
            /* Console output for training */
            .console-output {
                background: #1e1e1e !important;
                color: #d4d4d4 !important;
                font-family: 'Consolas', 'Monaco', monospace !important;
                padding: 16px !important;
                border-radius: 8px !important;
                height: 400px !important;
                overflow-y: auto !important;
                font-size: 13px !important;
                line-height: 1.5 !important;
            }
            
            /* Hide default footer */
            footer { display: none !important; }
            """
        ) as app:
            # Header
            gr.Markdown(
                """
                # 🏺 **Ceramatic 2.0** - Archaeological Pottery Analysis System
                ### AI-powered documentation and statistical analysis for ceramic artifacts
                """,
                elem_id="header"
            )
            
            # Progress bar container (hidden by default)
            progress_bar = gr.HTML(
                """
                <div class="progress-bar-container" style="display: none;" id="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill" style="width: 0%;">
                            <span id="progress-text">Initializing...</span>
                        </div>
                    </div>
                </div>
                """,
                visible=False
            )
            
            # Main tabs
            with gr.Tabs():
                # PROCESSING TAB with sequential workflow
                with gr.TabItem("🎯 Image Processing", elem_id="process"):
                    # Step 1: Model
                    with gr.Group(elem_classes="step-card"):
                        with gr.Row(elem_classes="step-header"):
                            gr.HTML('<div class="step-number">1</div>')
                            gr.Markdown("### Load YOLO Model", elem_classes="step-title")
                        
                        model_file = gr.File(
                            label="Upload YOLO Model (.pt file)",
                            file_types=[".pt"],
                            elem_id="model-upload"
                        )
                        gr.Markdown(
                            "📥 [Download pre-trained model](https://drive.google.com/file/d/1b23yWPZ0LKerIM8CWz2DcbhapThCnT7A/view)",
                            elem_id="model-link"
                        )
                    
                    # Step 2: Images & Metadata
                    with gr.Group(elem_classes="step-card"):
                        with gr.Row(elem_classes="step-header"):
                            gr.HTML('<div class="step-number">2</div>')
                            gr.Markdown("### Upload Images & Metadata", elem_classes="step-title")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                image_files = gr.File(
                                    label="Pottery Images",
                                    file_count="multiple",
                                    file_types=["image"],
                                    elem_id="images-upload"
                                )
                            
                            with gr.Column(scale=2):
                                with gr.Tab("Upload Excel"):
                                    metadata_file = gr.File(
                                        label="Metadata Excel/CSV",
                                        file_types=[".xlsx", ".xls", ".csv"],
                                        elem_id="metadata-file"
                                    )
                                
                                with gr.Tab("Manual Entry"):
                                    metadata_table = gr.Dataframe(
                                        headers=["TAV", "INV", "DIAM", "FLIP"],
                                        datatype=["str", "str", "number", "number"],
                                        row_count=10,
                                        col_count=(4, "fixed"),
                                        label="Metadata Table",
                                        interactive=True,
                                        elem_id="metadata-table"
                                    )
                                    with gr.Row():
                                        add_row_btn = gr.Button("➕ Add Row", size="sm")
                                        clear_table_btn = gr.Button("🗑️ Clear Table", size="sm")
                    
                    # Step 3: Processing Parameters
                    with gr.Group(elem_classes="step-card"):
                        with gr.Row(elem_classes="step-header"):
                            gr.HTML('<div class="step-number">3</div>')
                            gr.Markdown("### Processing Parameters", elem_classes="step-title")
                        
                        with gr.Row():
                            pixel_cm_ratio = gr.Number(
                                value=118.11,
                                label="Pixel/CM Ratio",
                                info="Scanner calibration"
                            )
                            confidence_threshold = gr.Slider(
                                0.5, 1.0, value=0.8, step=0.05,
                                label="Detection Confidence",
                                info="YOLO confidence threshold"
                            )
                            median_filter = gr.Number(
                                value=41,
                                label="Median Filter Size",
                                info="Noise reduction"
                            )
                    
                    # Step 4: Visual Style
                    with gr.Group(elem_classes="step-card"):
                        with gr.Row(elem_classes="step-header"):
                            gr.HTML('<div class="step-number">4</div>')
                            gr.Markdown("### Visual Style Options", elem_classes="step-title")
                        
                        with gr.Row():
                            with gr.Column():
                                profile_style = gr.Radio(
                                    ["filled", "outline", "filled_with_outline", "dotted_outline"],
                                    value="filled",
                                    label="Profile Style",
                                    info="How to render pottery profiles"
                                )
                            
                            with gr.Column():
                                inv_position = gr.Dropdown(
                                    ["bottom center", "bottom left", "bottom right",
                                     "top center", "top left", "top right"],
                                    value="bottom center",
                                    label="Inventory Number Position"
                                )
                                scale_bar_style = gr.Dropdown(
                                    ["striped", "solid", "simple"],
                                    value="striped",
                                    label="Scale Bar Style"
                                )
                            
                            with gr.Column():
                                scale_cm = gr.Number(
                                    value=3.0,
                                    label="Scale Length (cm)"
                                )
                                num_segments = gr.Number(
                                    value=3,
                                    label="Scale Bar Segments",
                                    precision=0
                                )
                    
                    # Step 5: Advanced Features (as toggles)
                    with gr.Group(elem_classes="step-card"):
                        with gr.Row(elem_classes="step-header"):
                            gr.HTML('<div class="step-number">5</div>')
                            gr.Markdown("### Advanced Features", elem_classes="step-title")
                        
                        with gr.Row():
                            features_checkboxes = gr.CheckboxGroup(
                                ["scale_bar", "diameter", "scale_text", "continuation_lines", 
                                 "diagnostic", "diagnostic_plots", "pca_analysis"],
                                value=["scale_bar"],
                                label="Enable Features",
                                elem_classes="toggle-group"
                            )
                        
                        gr.Markdown(
                            """
                            - **scale_bar**: Add measurement scale to images
                            - **diameter**: Show diameter measurement line
                            - **scale_text**: Display scale value on bar
                            - **continuation_lines**: Connect pottery fragments
                            - **diagnostic**: Process only first 5 images (testing)
                            - **diagnostic_plots**: Save debug visualizations
                            - **pca_analysis**: Run statistical shape analysis
                            """
                        )
                    
                    # Process button
                    process_btn = gr.Button(
                        "🚀 START PROCESSING",
                        variant="primary",
                        elem_id="process-btn"
                    )
                    
                    # Results section
                    gr.Markdown("## 📊 Results")
                    with gr.Row():
                        with gr.Column(scale=3):
                            gallery = gr.Gallery(
                                label="Processed Images",
                                columns=4,
                                rows=2,
                                object_fit="contain",
                                height="auto",
                                elem_id="results-gallery"
                            )
                        
                        with gr.Column(scale=1):
                            status_output = gr.Markdown(
                                "⏳ *Waiting to process...*",
                                elem_id="status"
                            )
                            download_file = gr.File(
                                label="Download Results",
                                visible=False,
                                elem_id="download-file"
                            )
                            download_btn = gr.Button(
                                "💾 Download ZIP",
                                visible=False,
                                elem_classes="download-button"
                            )
                
                # PCA ANALYSIS TAB
                with gr.TabItem("📊 PCA Analysis", elem_id="pca"):
                    gr.Markdown(
                        """
                        ## Principal Component Analysis
                        Analyze morphological features of processed pottery
                        """
                    )
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            pca_folder = gr.Textbox(
                                value="processed_imgs",
                                label="Processed Images Folder"
                            )
                            pca_min_samples = gr.Number(
                                value=3,
                                label="Minimum Samples",
                                precision=0
                            )
                            pca_features = gr.CheckboxGroup(
                                ["area", "perimeter", "circularity", "aspect_ratio",
                                 "solidity", "eccentricity", "orientation"],
                                value=["area", "circularity", "aspect_ratio"],
                                label="Features to Analyze"
                            )
                            run_pca_btn = gr.Button(
                                "📈 Run PCA Analysis",
                                variant="primary"
                            )
                        
                        with gr.Column(scale=2):
                            with gr.Tab("PCA Plot"):
                                pca_plot = gr.Image(label="PCA Visualization", elem_id="pca-vis")
                                pca_info = gr.Markdown("", elem_id="pca-info")
                            with gr.Tab("Feature Table"):
                                pca_table = gr.Dataframe(label="Extracted Features", elem_id="pca-table")
                            with gr.Tab("Analysis Report"):
                                pca_report = gr.Textbox(
                                    label="Statistical Report",
                                    lines=20,
                                    max_lines=30,
                                    elem_id="pca-report-text"
                                )
                
                # TRAINING TAB
                with gr.TabItem("🤖 Model Training", elem_id="train"):
                    gr.Markdown(
                        """
                        ## Train Custom YOLO Model
                        Create your own pottery detection model
                        """
                    )
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Dataset Setup")
                            with gr.Group():
                                train_images = gr.File(
                                    label="Training Images",
                                    file_count="multiple",
                                    file_types=["image"]
                                )
                                train_labels = gr.File(
                                    label="Training Labels (.txt)",
                                    file_count="multiple",
                                    file_types=[".txt"]
                                )
                                val_images = gr.File(
                                    label="Validation Images",
                                    file_count="multiple",
                                    file_types=["image"]
                                )
                                val_labels = gr.File(
                                    label="Validation Labels (.txt)",
                                    file_count="multiple",
                                    file_types=[".txt"]
                                )
                            
                            gr.Markdown("### Training Parameters")
                            epochs = gr.Slider(10, 300, value=100, step=10, label="Epochs")
                            batch_size = gr.Slider(1, 32, value=8, step=1, label="Batch Size")
                            learning_rate = gr.Number(value=0.01, label="Learning Rate")
                            
                            train_btn = gr.Button("🚀 Start Training", variant="primary")
                        
                        with gr.Column(scale=2):
                            gr.Markdown("### Training Progress")
                            training_console = gr.Textbox(
                                label="Training Output",
                                lines=20,
                                max_lines=30,
                                elem_classes="console-output",
                                interactive=False
                            )
                            
                            with gr.Row():
                                training_plot = gr.Image(label="Training Metrics")
                                model_download = gr.File(label="Download Trained Model")
                
                # HELP TAB
                with gr.TabItem("❓ Help", elem_id="help"):
                    gr.Markdown(
                        """
                        ## Quick Start Guide
                        
                        ### 🎯 Processing Workflow
                        1. **Upload Model**: Load the pre-trained YOLO model
                        2. **Add Images & Metadata**: Upload pottery photos and metadata
                        3. **Set Parameters**: Configure processing settings
                        4. **Choose Style**: Select visual output preferences
                        5. **Enable Features**: Toggle advanced options
                        6. **Process**: Click START PROCESSING
                        
                        ### 📊 Metadata Format
                        | Column | Description | Example |
                        |--------|-------------|---------|
                        | TAV | Filename (no extension) | 001, 002 |
                        | INV | Inventory number | C.123 |
                        | DIAM | Diameter in cm (optional) | 15.5 or empty |
                        | FLIP | Flip profile (0=no, 1=yes) | 0 or 1 |
                        
                        ### 🎨 Profile Styles
                        - **filled**: Solid traditional fill
                        - **outline**: External contour only
                        - **filled_with_outline**: Combined with highlighted edge
                        - **dotted_outline**: Dotted contour for reconstructions
                        
                        ### ✨ Advanced Features
                        - **Continuation Lines**: Auto-connect pottery fragments
                        - **PCA Analysis**: Extract 15+ morphological features
                        - **Interactive Plots**: Zoom, pan, and explore data
                        
                        ### 🤖 Training Custom Models
                        
                        #### Step 1: Prepare Dataset
                        Your dataset needs:
                        - **Images**: JPG/PNG photos of pottery
                        - **Labels**: One .txt file per image with YOLO format annotations
                        
                        #### Step 2: Label Format
                        Each label file should contain one line per object:
                        ```
                        class_id x_center y_center width height
                        ```
                        - All values normalized 0-1
                        - class_id: 0 for pottery profiles
                        - Example: `0 0.5 0.5 0.3 0.6`
                        
                        #### Step 3: Directory Structure
                        ```
                        dataset/
                        ├── images/
                        │   ├── train/
                        │   │   ├── img_001.jpg
                        │   │   ├── img_002.jpg
                        │   │   └── ...
                        │   └── val/
                        │       ├── img_100.jpg
                        │       ├── img_101.jpg
                        │       └── ...
                        └── labels/
                            ├── train/
                            │   ├── img_001.txt
                            │   ├── img_002.txt
                            │   └── ...
                            └── val/
                                ├── img_100.txt
                                ├── img_101.txt
                                └── ...
                        ```
                        
                        #### Step 4: Training Tips
                        - **Epochs**: Start with 100, increase if needed
                        - **Batch Size**: 8 for most GPUs, reduce if OOM
                        - **Learning Rate**: 0.01 is good default
                        - **Dataset Split**: 80% train, 20% validation
                        
                        #### Example Label File Content
                        For a pottery fragment centered in image:
                        ```
                        0 0.512 0.487 0.245 0.532
                        ```
                        This means:
                        - Class: 0 (pottery)
                        - Center X: 51.2% from left
                        - Center Y: 48.7% from top  
                        - Width: 24.5% of image width
                        - Height: 53.2% of image height
                        
                        #### Labeling Tools
                        - [LabelImg](https://github.com/tzutalin/labelImg) - Simple and free
                        - [Roboflow](https://roboflow.com/) - Web-based with YOLO export
                        - [CVAT](https://cvat.org/) - Advanced annotation tool
                        
                        ### 🔗 Resources
                        - [Scientific Paper](https://doi.org/10.1016/j.daach.2025.e00435)
                        - [GitHub Repository](https://github.com/lrncrd/Ceramatic2.0)
                        - [YOLO Documentation](https://docs.ultralytics.com/)
                        """
                    )
            
            # Event handlers
            def update_metadata_table(current_data):
                """Add a new row to metadata table"""
                if current_data is None:
                    current_data = []
                current_data.append(["", "", "", 0])
                return current_data
            
            def clear_metadata_table():
                """Clear the metadata table"""
                return []
            
            def toggle_download_button(file_path):
                """Show/hide download button based on file availability"""
                if file_path:
                    return gr.update(visible=True), gr.update(visible=True)
                return gr.update(visible=False), gr.update(visible=False)
            
            def download_results(file_path):
                """Handle download button click"""
                return file_path
            
            # Convert checkbox values to dict for easier handling
            def checkboxes_to_dict(checkbox_values):
                if checkbox_values is None:
                    return {}
                return {key: key in checkbox_values for key in [
                    "scale_bar", "diameter", "scale_text", "continuation_lines",
                    "diagnostic", "diagnostic_plots", "pca_analysis"
                ]}
            
            # Wire up event handlers
            add_row_btn.click(
                fn=update_metadata_table,
                inputs=[metadata_table],
                outputs=[metadata_table]
            )
            
            clear_table_btn.click(
                fn=clear_metadata_table,
                outputs=[metadata_table]
            )
            
            # Process button with all inputs
            process_btn.click(
                fn=lambda *args: self.process_images_workflow(*args[:-1], checkboxes_to_dict(args[-1])),
                inputs=[
                    model_file, image_files, metadata_file, metadata_table,
                    pixel_cm_ratio, confidence_threshold, median_filter,
                    profile_style, inv_position, scale_bar_style, scale_cm, num_segments,
                    features_checkboxes
                ],
                outputs=[gallery, status_output, download_file]
            ).then(
                fn=toggle_download_button,
                inputs=[download_file],
                outputs=[download_btn, download_file]
            )
            
            download_btn.click(
                fn=download_results,
                inputs=[download_file],
                outputs=[download_file]
            )
        
            # PCA Analysis handler
            def run_pca_analysis(input_folder, min_samples, features_to_show, progress=gr.Progress()):
                """Run PCA analysis on processed images"""
                try:
                    progress(0, desc="Starting PCA analysis...")
                    
                    # Check if folder exists
                    if not os.path.exists(input_folder):
                        return None, None, f"❌ Folder '{input_folder}' not found. Process images first!", None
                    
                    # Load processed images
                    processed_images = []
                    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
                    
                    if len(image_files) == 0:
                        return None, None, f"❌ No images found in '{input_folder}'", None
                    
                    progress(0.3, desc=f"Loading {len(image_files)} images...")
                    
                    for img_file in image_files:
                        img_path = os.path.join(input_folder, img_file)
                        inv_num = os.path.splitext(img_file)[0]
                        # Load image and convert to mask
                        img = Image.open(img_path).convert('L')
                        mask = np.array(img) < 128  # Binary mask
                        processed_images.append((img_file, inv_num, mask))
                    
                    if len(processed_images) < min_samples:
                        return None, None, f"❌ Found only {len(processed_images)} samples, minimum required: {min_samples}", None
                    
                    # Initialize processor if needed
                    if self.processor is None:
                        # Try to find a model file
                        model_paths = ["Ceramatic_model_V1.pt", "models/Ceramatic_model_V1.pt"]
                        model_found = None
                        for path in model_paths:
                            if os.path.exists(path):
                                model_found = path
                                break
                        
                        if model_found:
                            self.processor = ImageProcessor(model_path=model_found)
                        else:
                            return None, None, "❌ No YOLO model found. Please process images first.", None
                    
                    progress(0.5, desc="Extracting shape features...")
                    
                    # Run PCA analysis
                    X_pca, features_df = self.processor.perform_pca_analysis(processed_images)
                    
                    if X_pca is None:
                        return None, None, "❌ PCA analysis failed", None
                    
                    progress(0.8, desc="Loading results...")
                    
                    # Load plot - try PNG first for Gradio compatibility
                    plot_result = None
                    png_path = "statistical_analysis/pca_analysis.png"
                    html_path = "statistical_analysis/pca_analysis_interactive.html"
                    
                    if os.path.exists(png_path):
                        plot_result = png_path  # Gradio can display PNG directly
                    
                    # Load report
                    report_text = ""
                    report_path = "statistical_analysis/pca_summary.txt"
                    if os.path.exists(report_path):
                        with open(report_path, 'r') as f:
                            report_text = f.read()
                    
                    # Filter features to show
                    if features_to_show and len(features_to_show) > 0:
                        display_cols = ['inv_number'] + features_to_show
                        display_df = features_df[display_cols] if all(col in features_df.columns for col in display_cols) else features_df
                    else:
                        display_df = features_df
                    
                    progress(1.0, desc="Complete!")
                    
                    # Create download link for interactive HTML
                    info_msg = f"✅ Analysis completed on {len(processed_images)} samples"
                    if os.path.exists(html_path):
                        info_msg += f"\n📊 [Download Interactive Plot]({html_path})"
                    
                    return plot_result, display_df, report_text, info_msg
                    
                except Exception as e:
                    return None, None, f"❌ Error: {str(e)}", None
            
            # Wire up PCA button
            run_pca_btn.click(
                fn=run_pca_analysis,
                inputs=[pca_folder, pca_min_samples, pca_features],
                outputs=[pca_plot, pca_table, pca_report, pca_info]
            )
            
            # Training handler
            def train_model_live(
                train_imgs, train_lbls, val_imgs, val_lbls,
                epochs, batch_size, lr, 
                progress=gr.Progress()
            ):
                """Train YOLO model with live console output"""
                try:
                    import subprocess
                    import threading
                    import queue
                    
                    progress(0, desc="Preparing dataset...")
                    
                    # Create temp directory
                    temp_dir = self.create_temp_dir()
                    dataset_dir = os.path.join(temp_dir, "dataset")
                    
                    # Create directory structure
                    for split in ['train', 'val']:
                        os.makedirs(os.path.join(dataset_dir, 'images', split), exist_ok=True)
                        os.makedirs(os.path.join(dataset_dir, 'labels', split), exist_ok=True)
                    
                    # Copy files
                    console_output = "=== CERAMATIC 2.0 TRAINING ===\n\n"
                    
                    # Training images
                    if train_imgs:
                        console_output += f"📁 Loading {len(train_imgs)} training images...\n"
                        for i, img in enumerate(train_imgs):
                            src = img.name if hasattr(img, 'name') else img
                            dst = os.path.join(dataset_dir, 'images', 'train', os.path.basename(src))
                            shutil.copy(src, dst)
                    
                    # Training labels
                    if train_lbls:
                        console_output += f"📋 Loading {len(train_lbls)} training labels...\n"
                        for i, lbl in enumerate(train_lbls):
                            src = lbl.name if hasattr(lbl, 'name') else lbl
                            dst = os.path.join(dataset_dir, 'labels', 'train', os.path.basename(src))
                            shutil.copy(src, dst)
                    
                    # Validation images
                    if val_imgs:
                        console_output += f"📁 Loading {len(val_imgs)} validation images...\n"
                        for i, img in enumerate(val_imgs):
                            src = img.name if hasattr(img, 'name') else img
                            dst = os.path.join(dataset_dir, 'images', 'val', os.path.basename(src))
                            shutil.copy(src, dst)
                    
                    # Validation labels
                    if val_lbls:
                        console_output += f"📋 Loading {len(val_lbls)} validation labels...\n"
                        for i, lbl in enumerate(val_lbls):
                            src = lbl.name if hasattr(lbl, 'name') else lbl
                            dst = os.path.join(dataset_dir, 'labels', 'val', os.path.basename(src))
                            shutil.copy(src, dst)
                    
                    # Create dataset YAML
                    yaml_content = {
                        'train': os.path.join(dataset_dir, 'images', 'train'),
                        'val': os.path.join(dataset_dir, 'images', 'val'),
                        'nc': 1,
                        'names': ['pottery']
                    }
                    yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
                    with open(yaml_path, 'w') as f:
                        yaml.dump(yaml_content, f)
                    
                    console_output += f"\n✅ Dataset prepared!\n"
                    console_output += f"📊 Training: {len(train_imgs) if train_imgs else 0} images\n"
                    console_output += f"📊 Validation: {len(val_imgs) if val_imgs else 0} images\n"
                    console_output += f"\n⚙️ Configuration:\n"
                    console_output += f"   - Epochs: {epochs}\n"
                    console_output += f"   - Batch Size: {batch_size}\n"
                    console_output += f"   - Learning Rate: {lr}\n"
                    console_output += f"\n🚀 Starting training...\n\n"
                    
                    yield console_output, None, None
                    
                    # Training script
                    training_script = f'''
import torch
from ultralytics import YOLO
import os

# Detect device
if torch.cuda.is_available():
    device = 'cuda'
    print(f"🖥️  Using GPU: {{torch.cuda.get_device_name(0)}}")
elif torch.backends.mps.is_available():
    device = 'mps'
    print("🖥️  Using Apple Silicon GPU")
else:
    device = 'cpu'
    print("🖥️  Using CPU (training will be slower)")

print("\\n" + "="*50)
print("CERAMATIC 2.0 - POTTERY DETECTION TRAINING")
print("="*50 + "\\n")

# Initialize model
model = YOLO('yolov8n-seg.pt')

# Train
results = model.train(
    data='{yaml_path}',
    epochs={epochs},
    batch={batch_size},
    imgsz=640,
    lr0={lr},
    device=device,
    project='{temp_dir}',
    name='pottery_model',
    exist_ok=True,
    verbose=True
)

print("\\n✅ Training complete!")
                    '''
                    
                    # Save training script
                    script_path = os.path.join(temp_dir, 'train.py')
                    with open(script_path, 'w') as f:
                        f.write(training_script)
                    
                    # Run training with live output
                    process = subprocess.Popen(
                        [sys.executable, script_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                    
                    # Stream output
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            break
                        console_output += line
                        
                        # Check for metrics plot
                        plot_path = os.path.join(temp_dir, 'pottery_model', 'results.png')
                        if os.path.exists(plot_path):
                            yield console_output, plot_path, None
                        else:
                            yield console_output, None, None
                    
                    process.wait()
                    
                    # Get final results
                    best_model = os.path.join(temp_dir, 'pottery_model', 'weights', 'best.pt')
                    if os.path.exists(best_model):
                        console_output += "\n\n✅ Training completed successfully!"
                        console_output += f"\n📦 Model saved: best.pt"
                        yield console_output, plot_path if os.path.exists(plot_path) else None, best_model
                    else:
                        console_output += "\n\n❌ Training failed - no model produced"
                        yield console_output, None, None
                    
                except Exception as e:
                    error_msg = f"❌ Training Error: {str(e)}"
                    yield error_msg, None, None
            
            # Wire up training button
            train_btn.click(
                fn=train_model_live,
                inputs=[train_images, train_labels, val_images, val_labels,
                        epochs, batch_size, learning_rate],
                outputs=[training_console, training_plot, model_download]
            )
        
        return app

def main():
    """Main entry point"""
    app_instance = CeramaticGradioApp()
    interface = app_instance.create_interface()
    
    print("\n🌐 Starting Ceramatic 2.0...")
    print("🔗 Open in browser: http://127.0.0.1:7860\n")
    
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )

if __name__ == "__main__":
    main()