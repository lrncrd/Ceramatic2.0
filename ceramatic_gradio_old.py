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

    def process_images(
        self,
        model_file,
        image_files,
        metadata_file,
        metadata_table,
        pixel_cm_ratio,
        confidence_threshold,
        median_filter,
        inv_position,
        profile_style,
        scale_bar_style,
        scale_cm,
        num_segments,
        add_bar,
        add_diameter,
        add_scale_cm,
        diagnostic,
        diagnostic_plots,
        add_continuation_lines,
        perform_pca,
        progress=gr.Progress(track_tqdm=True)
    ):
        """Process images with Ceramatic"""
        try:
            progress(0, desc="🔄 Inizializzazione...")
            time.sleep(0.5)  # Small delay for UI update
            
            # Create temporary directories
            temp_dir = self.create_temp_dir()
            imgs_dir = os.path.join(temp_dir, "images")
            os.makedirs(imgs_dir, exist_ok=True)
            
            # Validation
            if not image_files:
                return None, None, "❌ Nessuna immagine caricata", [], gr.update(visible=False)
            if metadata_file is None and (metadata_table is None or len(metadata_table) == 0):
                return None, None, "❌ Metadata richiesti (carica file Excel o compila tabella)", [], gr.update(visible=False)
            if model_file is None:
                return None, None, "❌ File modello richiesto", [], gr.update(visible=False)
            
            # Save uploaded images
            progress(0.15, desc="📂 Caricamento immagini...")
            time.sleep(0.3)
            
            for i, img_file in enumerate(image_files):
                progress(0.15 + (0.1 * i / len(image_files)), desc=f"📂 Caricamento immagine {i+1}/{len(image_files)}...")
                if hasattr(img_file, 'name'):
                    shutil.copy(img_file.name, imgs_dir)
                else:
                    shutil.copy(img_file, imgs_dir)
            
            # Handle metadata - either from file or table
            progress(0.25, desc="📊 Elaborazione metadata...")
            time.sleep(0.3)
            
            metadata_path = os.path.join(temp_dir, "metadata.xlsx")
            if metadata_file is not None:
                # Use uploaded file
                if hasattr(metadata_file, 'name'):
                    shutil.copy(metadata_file.name, metadata_path)
                else:
                    shutil.copy(metadata_file, metadata_path)
            else:
                # Use table data
                df = pd.DataFrame(metadata_table, columns=["TAV", "INV", "DIAM", "FLIP"])
                # Clean empty rows
                df = df.dropna(subset=['TAV', 'INV'])
                # Convert DIAM to numeric, replacing empty with NaN
                df['DIAM'] = pd.to_numeric(df['DIAM'], errors='coerce')
                # Convert FLIP to numeric 
                df['FLIP'] = pd.to_numeric(df['FLIP'], errors='coerce').fillna(0).astype(int)
                # Save to Excel
                df.to_excel(metadata_path, index=False)
            
            # Save model file
            progress(0.35, desc="🤖 Caricamento modello YOLO...")
            time.sleep(0.3)
            
            model_path = os.path.join(temp_dir, "model.pt")
            if hasattr(model_file, 'name'):
                shutil.copy(model_file.name, model_path)
            else:
                shutil.copy(model_file, model_path)
            
            # Initialize processor - Fixed: only pass model_path and pixel_cm_ratio
            progress(0.45, desc="⚙️ Inizializzazione processore...")
            time.sleep(0.3)
            
            self.processor = ImageProcessor(
                model_path=model_path,
                pixel_cm_ratio=pixel_cm_ratio
            )
            
            # Process images with progress updates
            progress(0.55, desc="🎨 Elaborazione immagini in corso...")
            
            # Fixed: pass all parameters to process_images, not to __init__
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
                add_bar=add_bar,
                add_diameter=add_diameter,
                add_scale_cm=add_scale_cm,
                diagnostic=diagnostic,
                diagnostic_plots=diagnostic_plots,
                add_continuation_lines=add_continuation_lines
            )
            
            # Collect processed images
            progress(0.80, desc="📦 Raccolta risultati...")
            time.sleep(0.3)
            
            processed_dir = "processed_imgs"
            output_images = []
            
            if os.path.exists(processed_dir):
                for img_file in os.listdir(processed_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(processed_dir, img_file)
                        output_images.append(img_path)
            
            # Create zip archive
            progress(0.90, desc="🗜️ Creazione archivio ZIP...")
            time.sleep(0.3)
            
            zip_path = os.path.join(temp_dir, "results.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for img_path in output_images:
                    zipf.write(img_path, os.path.basename(img_path))
                
                if diagnostic_plots and os.path.exists("prediction_diagnostic"):
                    for root, dirs, files in os.walk("prediction_diagnostic"):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, ".")
                            zipf.write(file_path, arcname)
            
            # Perform PCA analysis if requested and enough samples
            if perform_pca and len(results) >= 3:
                progress(0.95, desc="📊 Analisi PCA...")
                time.sleep(0.3)
                try:
                    self.processor.perform_pca_analysis(results)
                    # Add PCA results to zip if they exist
                    if os.path.exists("statistical_analysis"):
                        with zipfile.ZipFile(zip_path, 'a') as zipf:
                            for root, dirs, files in os.walk("statistical_analysis"):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arcname = os.path.relpath(file_path, ".")
                                    zipf.write(file_path, arcname)
                except Exception as e:
                    print(f"Warning: PCA analysis failed: {e}")
            
            # Create results summary
            summary = f"✅ **Completato!** Elaborate {len(output_images)} immagini con successo"
            if perform_pca and len(results) >= 3:
                summary += "\n📊 Analisi PCA completata (vedi statistical_analysis nel ZIP)"
            
            # Display first processed image as preview
            preview_image = output_images[0] if output_images else None
            
            progress(1.0, desc="✅ Completato!")
            time.sleep(0.5)
            
            return preview_image, zip_path, summary, output_images, gr.update(visible=False)
            
        except Exception as e:
            return None, None, f"❌ **Errore**: {str(e)}", [], gr.update(visible=False)

    def train_model(
        self,
        dataset_yaml,
        train_images,
        val_images,
        train_labels,
        val_labels,
        epochs,
        batch_size,
        image_size,
        learning_rate,
        device,
        pretrained_model,
        progress=gr.Progress(track_tqdm=True)
    ):
        """Train a new YOLO model"""
        try:
            progress(0.1, desc="🔄 Preparazione training...")
            
            # Create temporary directory for dataset
            temp_dir = self.create_temp_dir()
            dataset_dir = os.path.join(temp_dir, "dataset")
            
            # Create directory structure
            os.makedirs(os.path.join(dataset_dir, "images", "train"), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, "images", "val"), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, "labels", "train"), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, "labels", "val"), exist_ok=True)
            
            progress(0.2, desc="📂 Caricamento dataset...")
            
            # Copy training images
            if train_images:
                for i, img in enumerate(train_images):
                    progress(0.2 + (0.1 * i / len(train_images)), desc=f"📂 Caricamento train image {i+1}/{len(train_images)}...")
                    if hasattr(img, 'name'):
                        shutil.copy(img.name, os.path.join(dataset_dir, "images", "train"))
                    else:
                        shutil.copy(img, os.path.join(dataset_dir, "images", "train"))
            
            # Copy validation images
            if val_images:
                for i, img in enumerate(val_images):
                    progress(0.3 + (0.05 * i / len(val_images)), desc=f"📂 Caricamento val image {i+1}/{len(val_images)}...")
                    if hasattr(img, 'name'):
                        shutil.copy(img.name, os.path.join(dataset_dir, "images", "val"))
                    else:
                        shutil.copy(img, os.path.join(dataset_dir, "images", "val"))
            
            # Copy labels
            if train_labels:
                for label in train_labels:
                    if hasattr(label, 'name'):
                        shutil.copy(label.name, os.path.join(dataset_dir, "labels", "train"))
                    else:
                        shutil.copy(label, os.path.join(dataset_dir, "labels", "train"))
            
            if val_labels:
                for label in val_labels:
                    if hasattr(label, 'name'):
                        shutil.copy(label.name, os.path.join(dataset_dir, "labels", "val"))
                    else:
                        shutil.copy(label, os.path.join(dataset_dir, "labels", "val"))
            
            # Create or copy dataset YAML
            progress(0.35, desc="⚙️ Configurazione dataset...")
            yaml_path = os.path.join(dataset_dir, "dataset.yaml")
            
            if dataset_yaml:
                if hasattr(dataset_yaml, 'name'):
                    shutil.copy(dataset_yaml.name, yaml_path)
                else:
                    shutil.copy(dataset_yaml, yaml_path)
            else:
                # Create default YAML
                yaml_content = {
                    'train': os.path.join(dataset_dir, "images", "train"),
                    'val': os.path.join(dataset_dir, "images", "val"),
                    'nc': 1,
                    'names': ['pottery']
                }
                with open(yaml_path, 'w') as f:
                    yaml.dump(yaml_content, f)
            
            progress(0.4, desc="🤖 Inizializzazione modello...")
            
            # Initialize YOLO model
            if pretrained_model and hasattr(pretrained_model, 'name'):
                model = YOLO(pretrained_model.name)
            else:
                model = YOLO('yolov8n-seg.pt')  # Default small segmentation model
            
            # Set device - Auto detection for CUDA, MPS (Apple Silicon), or CPU
            if device == "Auto":
                if torch.cuda.is_available():
                    device = 'cuda'
                elif torch.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'
            else:
                device = device.lower()
            
            progress(0.5, desc=f"🎯 Avvio training su {device.upper()}...")
            
            # Train model
            results = model.train(
                data=yaml_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=image_size,
                lr0=learning_rate,
                device=device,
                project=temp_dir,
                name='train',
                exist_ok=True,
                verbose=True
            )
            
            progress(0.9, desc="💾 Salvataggio modello...")
            
            # Get best model path
            best_model_path = os.path.join(temp_dir, "train", "weights", "best.pt")
            last_model_path = os.path.join(temp_dir, "train", "weights", "last.pt")
            
            # Create results zip
            zip_path = os.path.join(temp_dir, "training_results.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # Add models
                if os.path.exists(best_model_path):
                    zipf.write(best_model_path, "weights/best.pt")
                if os.path.exists(last_model_path):
                    zipf.write(last_model_path, "weights/last.pt")
                
                # Add training plots
                train_dir = os.path.join(temp_dir, "train")
                for file in os.listdir(train_dir):
                    if file.endswith(('.png', '.jpg', '.csv', '.yaml')):
                        file_path = os.path.join(train_dir, file)
                        zipf.write(file_path, f"results/{file}")
            
            # Create summary
            summary = f"✅ **Training completato!** {epochs} epoche su {device.upper()}"
            
            # Get confusion matrix or results plot if available
            preview_image = None
            for plot_name in ['confusion_matrix.png', 'results.png', 'labels.jpg']:
                plot_path = os.path.join(temp_dir, "train", plot_name)
                if os.path.exists(plot_path):
                    preview_image = plot_path
                    break
            
            progress(1.0, desc="✅ Completato!")
            return preview_image, zip_path, summary, best_model_path, gr.update(visible=False)
            
        except Exception as e:
            return None, None, f"❌ **Errore**: {str(e)}", None, gr.update(visible=False)
    
    def run_pca_analysis(self, input_folder, min_samples, features_to_show):
        """Run PCA analysis on processed images"""
        try:
            # Check if folder exists
            if not os.path.exists(input_folder):
                return None, None, None, f"❌ Cartella '{input_folder}' non trovata"
            
            # Load processed images
            processed_images = []
            for img_file in os.listdir(input_folder):
                if img_file.endswith(('.jpg', '.png')):
                    img_path = os.path.join(input_folder, img_file)
                    # Extract inv number from filename
                    inv_num = os.path.splitext(img_file)[0]
                    # Load image as mask
                    img = Image.open(img_path).convert('L')
                    mask = np.array(img) < 128  # Convert to binary mask
                    processed_images.append((img_file, inv_num, mask))
            
            if len(processed_images) < min_samples:
                return None, None, None, f"❌ Trovati solo {len(processed_images)} campioni, minimo richiesto: {min_samples}"
            
            # Initialize processor if needed
            if self.processor is None:
                self.processor = ImageProcessor(model_path="Ceramatic_model_V1.pt")
            
            # Run PCA analysis
            X_pca, features_df = self.processor.perform_pca_analysis(processed_images)
            
            if X_pca is None:
                return None, None, None, "❌ Analisi PCA fallita"
            
            # Get plot, table, and report
            plot_path = "statistical_analysis/pca_analysis.png"
            if not os.path.exists(plot_path):
                plot_path = None
            
            # Create report
            report_path = "statistical_analysis/pca_summary.txt"
            report_text = ""
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    report_text = f.read()
            
            # Filter features to show
            display_df = features_df[['inv_number'] + features_to_show] if features_to_show else features_df
            
            # Create zip with results
            zip_path = "pca_results.zip"
            with zipfile.ZipFile(zip_path, 'w') as zf:
                if os.path.exists("statistical_analysis"):
                    for root, dirs, files in os.walk("statistical_analysis"):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zf.write(file_path, os.path.relpath(file_path))
            
            info = f"✅ Analisi completata su {len(processed_images)} campioni"
            
            return plot_path, display_df, report_text, info, zip_path
            
        except Exception as e:
            return None, None, None, f"❌ Errore: {str(e)}", None

    def create_interface(self):
        """Create the Gradio interface with colorful friendly layout"""
        with gr.Blocks(
            title="Ceramatic 2.0", 
            theme=gr.themes.Soft(
                primary_hue="orange",
                secondary_hue="blue",
                spacing_size="md",
                radius_size="md",
                font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui", "sans-serif"]
            ),
            css="""
            /* Colorful Friendly Layout */
            .gradio-container {
                padding: 20px !important; 
                max-width: 1600px !important; 
                margin: auto !important;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
            }
            
            /* Groups with colorful borders */
            .gr-group {
                padding: 15px !important; 
                gap: 12px !important;
                border: 2px solid #e0e7ff !important;
                border-radius: 12px !important;
                background: white !important;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
                margin-bottom: 15px !important;
            }
            
            .gr-box {
                padding: 12px !important;
                border-radius: 8px !important;
                background: #fafbfc !important;
            }
            
            .gr-form {gap: 12px !important;}
            
            /* Labels with better spacing */
            .gr-input-label {
                margin-bottom: 6px !important; 
                font-size: 0.95em !important; 
                font-weight: 600 !important;
                color: #374151 !important;
                letter-spacing: 0.5px !important;
            }
            
            /* Colorful buttons */
            .gr-button {
                min-height: 45px !important; 
                font-weight: 600 !important;
                border-radius: 8px !important;
                transition: all 0.3s ease !important;
                text-transform: uppercase !important;
                letter-spacing: 1px !important;
            }
            
            .gr-button-primary {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%) !important;
                border: none !important;
                color: white !important;
            }
            
            .gr-button-primary:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(238,90,111,0.4) !important;
            }
            
            /* Spacing */
            .gr-padded {padding: 12px !important;}
            .block {margin-bottom: 12px !important;}
            
            /* Headers */
            h1 {
                font-size: 2.0em !important; 
                margin: 20px 0 !important;
                color: #1f2937 !important;
                text-align: center !important;
                font-weight: 700 !important;
            }
            
            h3 {
                font-size: 1.2em !important; 
                margin: 12px 0 !important; 
                font-weight: 600 !important;
                color: #4b5563 !important;
            }
            
            /* Tab Styling */
            .tabs {
                border-radius: 12px !important;
                background: white !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
                padding: 10px !important;
            }
            
            .tabitem {
                padding: 20px !important;
                background: #f9fafb !important;
                border-radius: 8px !important;
            }
            
            .tab-nav {
                background: #f3f4f6 !important;
                border-radius: 8px !important;
                padding: 8px !important;
            }
            
            /* Input/Output Styling */
            .gr-file {
                min-height: 80px !important;
                border: 2px dashed #cbd5e1 !important;
                border-radius: 8px !important;
                background: #f8fafc !important;
                transition: all 0.3s ease !important;
            }
            
            .gr-file:hover {
                border-color: #93c5fd !important;
                background: #eff6ff !important;
            }
            
            .gr-image {
                border-radius: 12px !important;
                border: 3px solid #e5e7eb !important;
                overflow: hidden !important;
            }
            
            .gr-gallery {
                border-radius: 12px !important;
                background: white !important;
                padding: 10px !important;
                border: 2px solid #e5e7eb !important;
            }
            
            /* Progress Bar Styling */
            .progress-bar {
                background: linear-gradient(90deg, #f59e0b 0%, #ef4444 100%);
                border-radius: 12px;
                height: 6px !important;
                animation: pulse 2s ease-in-out infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }
            
            /* Checkboxes with better styling */
            .gr-check-radio {
                margin: 6px 0 !important;
                accent-color: #f59e0b !important;
            }
            
            /* Dropdowns */
            .gr-dropdown {
                border: 2px solid #e5e7eb !important;
                border-radius: 8px !important;
                background: white !important;
            }
            
            /* Numbers and sliders */
            .gr-number input, .gr-text input {
                border: 2px solid #e5e7eb !important;
                border-radius: 8px !important;
                padding: 8px 12px !important;
                font-size: 0.95em !important;
            }
            
            .gr-number input:focus, .gr-text input:focus {
                border-color: #f59e0b !important;
                outline: none !important;
                box-shadow: 0 0 0 3px rgba(245,158,11,0.1) !important;
            }
            
            /* Table styling */
            .gr-dataframe {
                border-radius: 8px !important;
                overflow: hidden !important;
                border: 2px solid #e5e7eb !important;
            }
            
            .gr-dataframe table {
                font-size: 0.9em !important;
            }
            
            .gr-dataframe th {
                background: #f3f4f6 !important;
                font-weight: 600 !important;
                padding: 10px !important;
                color: #374151 !important;
            }
            
            .gr-dataframe td {
                padding: 8px !important;
                border-bottom: 1px solid #e5e7eb !important;
            }
            
            /* Status messages */
            .gr-markdown {
                line-height: 1.6 !important;
            }
            
            /* Footer removal */
            footer {display: none !important;}
            
            /* Responsive Design */
            @media (max-width: 768px) {
                .gr-row {flex-direction: column !important;}
                h1 {font-size: 1.5em !important;}
            }
            """
        ) as app:
            gr.Markdown("# 🏺 **Ceramatic 2.0** - Sistema Avanzato di Analisi Ceramiche Archeologiche")
            gr.Markdown("### 🎯 Documentazione automatica e analisi statistica per reperti ceramici", elem_id="subtitle")
            
            # Main Interface with Tabs
            with gr.Tabs() as tabs:
                
                # PROCESSING TAB - Redesigned with better layout
                with gr.TabItem("🎨 Elaborazione Immagini", elem_id="process"):
                    # Progress bar placeholder
                    process_progress = gr.Markdown("", visible=False)
                    
                    with gr.Row():
                        # Left Column: Input Files
                        with gr.Column(scale=2):
                            gr.Markdown("### 📂 Input Files")
                            
                            with gr.Group():
                                model_file = gr.File(
                                    label="🤖 Modello YOLO (.pt)", 
                                    file_types=[".pt"], 
                                    elem_id="model-upload"
                                )
                                gr.Markdown("*Scarica il modello da [Google Drive](https://drive.google.com/file/d/1b23yWPZ0LKerIM8CWz2DcbhapThCnT7A/view)*", elem_id="model-help")
                            
                            with gr.Group():
                                image_files = gr.File(
                                    label="🖼️ Immagini Ceramiche", 
                                    file_count="multiple", 
                                    file_types=["image"],
                                    elem_id="image-upload"
                                )
                                gr.Markdown("*Formati supportati: JPG, PNG, TIFF*")
                            
                            with gr.Group():
                                gr.Markdown("### 📊 Metadata")
                                with gr.Tab("Carica Excel"):
                                    metadata_file = gr.File(
                                        label="📋 File Excel/CSV", 
                                        file_types=[".xlsx", ".xls", ".csv"],
                                        elem_id="metadata-upload"
                                    )
                                with gr.Tab("Inserisci Manualmente"):
                                    metadata_table = gr.Dataframe(
                                        headers=["TAV", "INV", "DIAM", "FLIP"],
                                        datatype=["str", "str", "number", "number"],
                                        row_count=5,
                                        col_count=(4, "fixed"),
                                        label="Tabella Metadata",
                                        interactive=True,
                                        elem_id="metadata-table"
                                    )
                                    gr.Examples(
                                        examples=[
                                            [["001", "C.123", 15.5, 0], ["002", "C.124", None, 1]],
                                        ],
                                        inputs=metadata_table,
                                        label="Esempio formato tabella"
                                    )
                        
                        # Middle Column: Parameters
                        with gr.Column(scale=2):
                            gr.Markdown("### ⚙️ Parametri Elaborazione")
                            
                            with gr.Group():
                                gr.Markdown("**📐 Calibrazione**")
                                pixel_cm_ratio = gr.Number(
                                    value=118.11, 
                                    label="Rapporto Pixel/CM", 
                                    info="Calibrazione scanner"
                                )
                                with gr.Row():
                                    confidence_threshold = gr.Slider(
                                        0.5, 1.0, value=0.8, step=0.05, 
                                        label="Soglia Confidenza",
                                        info="Sensibilità rilevamento"
                                    )
                                    median_filter = gr.Number(
                                        value=41, 
                                        label="Filtro Mediano",
                                        info="Riduzione rumore"
                                    )
                            
                            with gr.Group():
                                gr.Markdown("**🎨 Stile Visualizzazione**")
                                profile_style = gr.Radio(
                                    ["filled", "outline", "filled_with_outline", "dotted_outline"], 
                                    value="filled", 
                                    label="Stile Profilo",
                                    info="Scegli come visualizzare il profilo ceramico"
                                )
                                
                                with gr.Row():
                                    inv_position = gr.Dropdown(
                                        ["bottom center", "bottom left", "bottom right", 
                                         "top center", "top left", "top right"],
                                        value="bottom center", 
                                        label="Posizione Inventario"
                                    )
                                    scale_bar_style = gr.Dropdown(
                                        ["striped", "solid", "simple"], 
                                        value="striped", 
                                        label="Stile Scala"
                                    )
                                
                                with gr.Row():
                                    scale_cm = gr.Number(
                                        value=3.0, 
                                        label="Lunghezza Scala (cm)"
                                    )
                                    num_segments = gr.Number(
                                        value=3, 
                                        label="Segmenti Scala",
                                        precision=0
                                    )
                            
                            with gr.Group():
                                gr.Markdown("**✨ Opzioni Avanzate**")
                                with gr.Row():
                                    with gr.Column():
                                        add_bar = gr.Checkbox(label="Aggiungi Scala", value=True)
                                        add_diameter = gr.Checkbox(label="Mostra Diametro", value=False)
                                        add_scale_cm = gr.Checkbox(label="Testo Scala", value=False)
                                        add_continuation_lines = gr.Checkbox(label="Linee Continuazione", value=False)
                                    with gr.Column():
                                        diagnostic = gr.Checkbox(label="Modo Test (5 img)", value=False)
                                        diagnostic_plots = gr.Checkbox(label="Salva Plot Diagnostici", value=False)
                                        perform_pca = gr.Checkbox(label="Analisi PCA", value=False)
                            
                            process_btn = gr.Button(
                                "🚀 AVVIA ELABORAZIONE", 
                                variant="primary", 
                                elem_id="process-btn"
                            )
                        
                        # Right Column: Results  
                        with gr.Column(scale=3):
                            gr.Markdown("### 📸 Risultati Elaborazione")
                            
                            output_status = gr.Markdown(
                                "*🕐 In attesa di elaborazione...*", 
                                elem_id="status"
                            )
                            
                            output_preview = gr.Image(
                                label="Anteprima Ultima Immagine", 
                                elem_id="preview"
                            )
                            
                            with gr.Row():
                                output_zip = gr.File(
                                    label="📦 Scarica Risultati (ZIP)", 
                                    elem_id="download"
                                )
                            
                            output_gallery = gr.Gallery(
                                label="Galleria Immagini Elaborate", 
                                columns=3, 
                                rows=2,
                                object_fit="contain",
                                show_label=True,
                                elem_id="gallery"
                            )
                
                # PCA STATISTICS TAB  
                with gr.TabItem("📊 Analisi Statistica PCA", elem_id="pca"):
                    gr.Markdown("### 📈 Analisi delle Componenti Principali (PCA)")
                    gr.Markdown("*Analizza le caratteristiche morfologiche dei reperti ceramici*")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 🔍 Configurazione Analisi")
                            
                            with gr.Group():
                                pca_input_folder = gr.Textbox(
                                    value="processed_imgs",
                                    label="📁 Cartella Immagini Elaborate",
                                    info="Percorso delle immagini processate"
                                )
                                pca_min_samples = gr.Number(
                                    value=3,
                                    label="Campioni Minimi",
                                    info="Minimo 3 campioni per PCA",
                                    precision=0
                                )
                                pca_features_to_show = gr.CheckboxGroup(
                                    ["area", "perimeter", "circularity", "aspect_ratio", 
                                     "solidity", "eccentricity", "orientation"],
                                    value=["area", "circularity", "aspect_ratio"],
                                    label="Features da Analizzare"
                                )
                                run_pca_btn = gr.Button(
                                    "📊 ESEGUI ANALISI PCA",
                                    variant="primary"
                                )
                            
                            gr.Markdown("### 📋 Info Analisi")
                            pca_info = gr.Markdown("*Clicca 'Esegui Analisi' per iniziare*")
                        
                        with gr.Column(scale=3):
                            gr.Markdown("### 📊 Risultati Analisi")
                            
                            with gr.Tab("Grafico PCA"):
                                pca_plot = gr.Image(
                                    label="Visualizzazione PCA",
                                    elem_id="pca-plot"
                                )
                            
                            with gr.Tab("Tabella Features"):
                                pca_features_table = gr.Dataframe(
                                    label="Features Estratte",
                                    elem_id="features-table"
                                )
                            
                            with gr.Tab("Report Testuale"):
                                pca_report = gr.Textbox(
                                    label="Report Analisi",
                                    lines=15,
                                    max_lines=30,
                                    elem_id="pca-report"
                                )
                            
                            pca_download = gr.File(
                                label="📥 Scarica Risultati PCA",
                                elem_id="pca-download"
                            )
                
                # TRAINING TAB
                with gr.TabItem("🤖 Training Modello", elem_id="train"):
                    train_progress = gr.Markdown("", visible=False)
                    
                    with gr.Row():
                        # Dataset column
                        with gr.Column(scale=2):
                            gr.Markdown("### 📁 Dataset")
                            dataset_yaml = gr.File(label="Config YAML (opz.)", file_types=[".yaml"], height=60)
                            pretrained_model = gr.File(label="Pre-trained (opz.)", file_types=[".pt"], height=60)
                            with gr.Row():
                                train_images = gr.File(label="Train Imgs", file_count="multiple", file_types=["image"], height=60)
                                val_images = gr.File(label="Val Imgs", file_count="multiple", file_types=["image"], height=60)
                            with gr.Row():
                                train_labels = gr.File(label="Train Labels", file_count="multiple", file_types=[".txt"], height=60)
                                val_labels = gr.File(label="Val Labels", file_count="multiple", file_types=[".txt"], height=60)
                        
                        # Parameters column
                        with gr.Column(scale=2):
                            gr.Markdown("### ⚙️ Parametri")
                            with gr.Row():
                                epochs = gr.Slider(10, 500, value=100, step=10, label="Epoche")
                                batch_size = gr.Slider(1, 32, value=8, step=1, label="Batch")
                            with gr.Row():
                                image_size = gr.Slider(320, 1280, value=640, step=32, label="Img Size")
                                learning_rate = gr.Number(value=0.01, label="Learn Rate")
                            device = gr.Dropdown(["Auto", "CPU", "CUDA", "MPS"], value="Auto", label="Device")
                        
                        # Results column
                        with gr.Column(scale=3):
                            gr.Markdown("### 📊 Risultati Training")
                            train_status = gr.Markdown("*In attesa di training...*")
                            train_preview = gr.Image(label="Metriche", height=200)
                            with gr.Row():
                                train_model_output = gr.File(label="📦 Best Model")
                                train_zip = gr.File(label="📦 All Results")
                    
                    with gr.Row():
                        gr.Column(scale=1)
                        train_btn = gr.Button("🎯 **AVVIA TRAINING**", variant="primary", scale=2)
                        gr.Column(scale=1)
                
                # HELP TAB - Updated
                with gr.TabItem("❓ Guida & Info", elem_id="help"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("""
### 🚀 Guida Rapida

#### 1️⃣ **Preparazione**
- [📥 Scarica Modello YOLO](https://drive.google.com/file/d/1b23yWPZ0LKerIM8CWz2DcbhapThCnT7A/view)
- Prepara immagini ceramiche (JPG/PNG/TIFF)
- Crea metadata Excel O usa tabella integrata

#### 2️⃣ **Elaborazione**
1. Carica modello e immagini
2. Inserisci metadata (file o tabella)
3. Configura parametri e stile
4. Clicca **AVVIA ELABORAZIONE**

#### 3️⃣ **Analisi PCA** 
1. Elabora prima le immagini
2. Vai alla tab "Analisi Statistica"
3. Seleziona features da analizzare
4. Clicca **ESEGUI ANALISI PCA**

### 📊 Formato Metadata

| Colonna | Descrizione | Esempio |
|---------|-------------|---------|
| **TAV** | Nome file (senza estensione) | 001, 002 |
| **INV** | Numero inventario | C.123 |
| **DIAM** | Diametro in cm (opzionale) | 15.5 o NaN |
| **FLIP** | Inverti profilo (0=no, 1=sì) | 0 o 1 |

### 🎨 Stili Profilo
- **Filled**: Riempimento solido tradizionale
- **Outline**: Solo contorno esterno
- **Filled+Outline**: Combinazione con bordo evidenziato  
- **Dotted**: Contorno punteggiato per ricostruzioni

### ✨ Features Avanzate
- **Linee Continuazione**: Collegamento automatico frammenti
- **Analisi PCA**: 15+ features morfologiche estratte
- **Tabella Metadata**: Inserimento diretto senza Excel
- **Export Multiplo**: ZIP con immagini + statistiche

### 📈 Features PCA Analizzate
- Area, Perimetro, Circolarità
- Aspect Ratio, Solidità, Eccentricità
- Orientamento, Diametro Equivalente
- Altezza/Larghezza e rapporti

### 🔗 Collegamenti Utili
- [📄 Paper Scientifico](https://doi.org/10.1016/j.daach.2025.e00435)
- [💻 Repository GitHub](https://github.com/lrncrd/Ceramatic2.0)
- [📧 Contatto: lrncrd](https://github.com/lrncrd)
                            """)
            
            # Event handlers with progress
            process_btn.click(
                fn=self.process_images,
                inputs=[
                    model_file, image_files, metadata_file, metadata_table,
                    pixel_cm_ratio, confidence_threshold, median_filter,
                    inv_position, profile_style, scale_bar_style,
                    scale_cm, num_segments,
                    add_bar, add_diameter, add_scale_cm,
                    diagnostic, diagnostic_plots, add_continuation_lines, perform_pca
                ],
                outputs=[output_preview, output_zip, output_status, output_gallery, process_progress]
            )
            
            run_pca_btn.click(
                fn=self.run_pca_analysis,
                inputs=[pca_input_folder, pca_min_samples, pca_features_to_show],
                outputs=[pca_plot, pca_features_table, pca_report, pca_info, pca_download]
            )
            
            train_btn.click(
                fn=self.train_model,
                inputs=[
                    dataset_yaml, train_images, val_images,
                    train_labels, val_labels,
                    epochs, batch_size, image_size, learning_rate,
                    device, pretrained_model
                ],
                outputs=[train_preview, train_zip, train_status, train_model_output, train_progress]
            )
        
        return app

def main():
    """Main entry point"""
    import socket
    
    # Check for model in various locations
    model_locations = [
        "Ceramatic_model_V1.pt",
        "models/Ceramatic_model_V1.pt",
        "model/Ceramatic_model_V1.pt"
    ]
    
    model_found = False
    for path in model_locations:
        if os.path.exists(path):
            print(f"✅ Modello trovato: {path}")
            model_found = True
            break
    
    if not model_found:
        print("⚠️  Modello non trovato. Scaricalo da:")
        print("   https://drive.google.com/file/d/1b23yWPZ0LKerIM8CWz2DcbhapThCnT7A/view")
        print("   e posizionalo nella cartella 'models/' o nella cartella principale")
    
    app_instance = CeramaticGradioApp()
    interface = app_instance.create_interface()
    
    # Find available port - try different ranges if 7860-7870 are busy
    def find_free_port(start_port=7860, max_attempts=20):
        import random
        # Try sequential ports first
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(('127.0.0.1', port))
                    return port
            except OSError:
                continue
        
        # If all sequential ports busy, try random ports
        for _ in range(10):
            port = random.randint(8000, 9000)
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(('127.0.0.1', port))
                    return port
            except OSError:
                continue
        return None
    
    # Launch the interface
    port = find_free_port()
    if port:
        print(f"\n🌐 Avvio server su porta {port}...")
        print(f"🔗 Apri nel browser: http://127.0.0.1:{port}\n")
        try:
            interface.launch(
                server_name="127.0.0.1",
                server_port=port,
                share=False,
                inbrowser=True,
                quiet=True,
                prevent_thread_lock=False
            )
        except OSError as e:
            print(f"❌ Errore: {e}")
            print("\n💡 Prova a eseguire: ./kill_ports.sh")
            print("   per liberare le porte, poi riavvia l'app")
    else:
        print("❌ Nessuna porta disponibile.")
        print("\n💡 Esegui: ./kill_ports.sh")
        print("   per liberare le porte, poi riavvia l'app")

if __name__ == "__main__":
    main()