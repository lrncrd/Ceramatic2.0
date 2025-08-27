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
            if metadata_file is None:
                return None, None, "❌ File metadata richiesto", [], gr.update(visible=False)
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
            
            # Save metadata file
            progress(0.25, desc="📊 Caricamento metadata...")
            time.sleep(0.3)
            
            metadata_path = os.path.join(temp_dir, "metadata.xlsx")
            if hasattr(metadata_file, 'name'):
                shutil.copy(metadata_file.name, metadata_path)
            else:
                shutil.copy(metadata_file, metadata_path)
            
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
                diagnostic_plots=diagnostic_plots
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
            
            # Create results summary
            summary = f"✅ **Completato!** Elaborate {len(output_images)} immagini con successo"
            
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

    def create_interface(self):
        """Create the Gradio interface with compact professional layout"""
        with gr.Blocks(
            title="Ceramatic 2.0", 
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="gray",
                spacing_size="sm",
                radius_size="sm"
            ),
            css="""
            /* Compact Professional Layout */
            .gradio-container {padding: 10px !important;}
            .gr-group {padding: 8px !important; gap: 8px !important;}
            .gr-box {padding: 8px !important;}
            .gr-form {gap: 8px !important;}
            .gr-input-label {margin-bottom: 2px !important; font-size: 0.9em !important;}
            .gr-button {min-height: 38px !important;}
            .gr-padded {padding: 8px !important;}
            .block {margin-bottom: 8px !important;}
            h1 {font-size: 1.5em !important; margin: 10px 0 !important;}
            h3 {font-size: 1.1em !important; margin: 8px 0 !important;}
            footer {display: none !important;}
            
            /* Progress Bar Styling */
            .progress-bar {
                background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
                border-radius: 8px;
                animation: pulse 2s ease-in-out infinite;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.8; }
            }
            """
        ) as app:
            gr.Markdown("# 🏺 **Ceramatic 2.0** - Archaeological Pottery Analysis")
            
            # Main Interface with Tabs
            with gr.Tabs() as tabs:
                
                # PROCESSING TAB - Redesigned for compactness
                with gr.TabItem("📊 Elaborazione", elem_id="process"):
                    # Progress bar placeholder
                    process_progress = gr.Markdown("", visible=False)
                    
                    with gr.Row():
                        # Column 1: Files
                        with gr.Column(scale=2):
                            gr.Markdown("### 📁 File Input")
                            model_file = gr.File(label="Modello YOLO (.pt)", file_types=[".pt"], height=80)
                            image_files = gr.File(label="Immagini", file_count="multiple", file_types=["image"], height=80)
                            metadata_file = gr.File(label="Excel Metadata", file_types=[".xlsx", ".xls", ".csv"], height=80)
                        
                        # Column 2: Core Parameters
                        with gr.Column(scale=2):
                            gr.Markdown("### ⚙️ Parametri")
                            with gr.Row():
                                pixel_cm_ratio = gr.Number(value=118.11, label="Pixel/CM")
                                confidence_threshold = gr.Slider(0.5, 1.0, value=0.8, step=0.05, label="Confidence")
                            with gr.Row():
                                median_filter = gr.Number(value=41, label="Filtro Med.")
                                scale_cm = gr.Number(value=3.0, label="Scala CM")
                            with gr.Row():
                                inv_position = gr.Dropdown(
                                    ["bottom center", "bottom left", "bottom right", "top center", "top left", "top right"],
                                    value="bottom center", label="Pos. Inventario"
                                )
                                num_segments = gr.Number(value=3, label="Segmenti", precision=0)
                        
                        # Column 3: Options
                        with gr.Column(scale=2):
                            gr.Markdown("### 🎨 Opzioni")
                            with gr.Row():
                                with gr.Column():
                                    add_bar = gr.Checkbox(label="Barra Scala", value=True)
                                    add_diameter = gr.Checkbox(label="Linea Diametro", value=False)
                                    add_scale_cm = gr.Checkbox(label="Mostra CM", value=False)
                                with gr.Column():
                                    diagnostic = gr.Checkbox(label="Diagnostica (5 img)", value=False)
                                    diagnostic_plots = gr.Checkbox(label="Salva Plot", value=False)
                            with gr.Row():
                                profile_style = gr.Dropdown(["filled"], value="filled", label="Profilo")
                                scale_bar_style = gr.Dropdown(["striped", "solid", "simple"], value="striped", label="Stile Scala")
                        
                        # Column 4: Results
                        with gr.Column(scale=3):
                            gr.Markdown("### 📸 Risultati")
                            output_status = gr.Markdown("*In attesa di elaborazione...*")
                            with gr.Row():
                                output_preview = gr.Image(label="Anteprima", height=200, elem_id="preview")
                            with gr.Row():
                                output_zip = gr.File(label="📦 Download ZIP", elem_id="download")
                            output_gallery = gr.Gallery(
                                label="Immagini Processate", 
                                columns=3, 
                                rows=1,
                                height=150,
                                object_fit="contain"
                            )
                    
                    # Process button centered
                    with gr.Row():
                        gr.Column(scale=1)
                        process_btn = gr.Button("🚀 **AVVIA ELABORAZIONE**", variant="primary", scale=2)
                        gr.Column(scale=1)
                
                # TRAINING TAB
                with gr.TabItem("🎯 Training", elem_id="train"):
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
                
                # HELP TAB - Compact
                with gr.TabItem("❓ Aiuto"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("""
### 🚀 Quick Start
1. [📥 Download Modello](https://drive.google.com/file/d/1b23yWPZ0LKerIM8CWz2DcbhapThCnT7A/view)
2. Prepara immagini JPG/PNG
3. Crea Excel: TAV|INV|DIAM|FLIP
4. Carica → Configura → Elabora

### 📊 Parametri
- **Confidence**: 0.8 (soglia rilevamento)
- **Pixel/CM**: 118.11 (calibrazione)
- **Filtro**: 41 (riduzione rumore)

### 🔗 Links
- [Paper](https://doi.org/10.1016/j.daach.2025.e00435)
- [GitHub](https://github.com/lrncrd/Ceramatic2.0)
                            """)
                        
                        with gr.Column(scale=2):
                            gr.Markdown("""
### 📋 File Excel Richiesto

| TAV | INV | DIAM | FLIP |
|-----|-----|------|------|
| img_001 | C.123 | 15.5 | 0 |
| img_002 | C.124 | NaN | 1 |

- **TAV**: Nome file senza estensione
- **INV**: Numero inventario
- **DIAM**: Diametro in cm (opzionale, usa NaN)
- **FLIP**: 0=normale, 1=inverti profilo

### ⚠️ Troubleshooting
- **Nessuna ceramica**: Riduci confidence
- **Errore metadata**: Controlla colonne Excel
- **Out of memory**: Riduci batch size
                            """)
            
            # Event handlers with progress
            process_btn.click(
                fn=self.process_images,
                inputs=[
                    model_file, image_files, metadata_file,
                    pixel_cm_ratio, confidence_threshold, median_filter,
                    inv_position, profile_style, scale_bar_style,
                    scale_cm, num_segments,
                    add_bar, add_diameter, add_scale_cm,
                    diagnostic, diagnostic_plots
                ],
                outputs=[output_preview, output_zip, output_status, output_gallery, process_progress]
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