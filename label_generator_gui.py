#!/usr/bin/env python3
"""
Gradio GUI for YOLO Label Generation
Interactive tool to prepare training datasets for Ceramatic
"""

import gradio as gr
import os
import shutil
import json
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# Import the generation functions
from generate_yolo_labels import (
    generate_labels_from_model, 
    generate_labels_manual,
    convert_labelme_to_yolo
)

def process_auto_generation(
    images_folder, 
    output_folder, 
    model_path, 
    confidence,
    train_split,
    progress=gr.Progress()
):
    """Auto-generate labels using existing model"""
    try:
        progress(0, desc="Starting auto-generation...")
        
        if not os.path.exists(images_folder):
            return None, "❌ Images folder not found"
        
        # Try multiple model locations
        model_locations = [
            model_path,
            os.path.join("models", model_path),
            os.path.join("..", model_path),
            "models/Ceramatic_model_V1.pt",
            "../Ceramatic_model_V1.pt",
            "yolov8m-seg.pt"  # Fallback to generic YOLO model
        ]
        
        model_found = None
        for path in model_locations:
            if os.path.exists(path):
                model_found = path
                break
        
        if not model_found:
            return None, """❌ Model file not found!
            
Please download the model from:
https://drive.google.com/file/d/1b23yWPZ0LKerIM8CWz2DcbhapThCnT7A/view

Save it as 'Ceramatic_model_V1.pt' in the project folder."""
        
        model_path = model_found
        
        # Count images
        image_files = list(Path(images_folder).glob("*.jpg")) + \
                     list(Path(images_folder).glob("*.png")) + \
                     list(Path(images_folder).glob("*.tiff"))
        
        if len(image_files) == 0:
            return None, "❌ No images found in folder"
        
        progress(0.1, desc=f"Found {len(image_files)} images")
        
        # Create output directories
        output_path = Path(output_folder)
        if output_path.exists():
            # Ask for confirmation before overwriting
            shutil.rmtree(output_path)
        
        train_images = output_path / "train" / "images"
        train_labels = output_path / "train" / "labels"
        val_images = output_path / "val" / "images"
        val_labels = output_path / "val" / "labels"
        
        for dir in [train_images, train_labels, val_images, val_labels]:
            dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        progress(0.2, desc="Loading model...")
        model = YOLO(model_path)
        
        # Split data
        split_idx = int(len(image_files) * (train_split / 100))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        total_detections = 0
        
        # Process training images
        for idx, img_path in enumerate(train_files):
            progress(
                0.2 + (idx / len(image_files)) * 0.7,
                desc=f"Processing training image {idx+1}/{len(train_files)}"
            )
            
            # Copy image
            shutil.copy(img_path, train_images / img_path.name)
            
            # Generate label
            results = model(str(img_path), conf=confidence)
            
            if results[0].masks is not None:
                img = cv2.imread(str(img_path))
                h, w = img.shape[:2]
                
                label_path = train_labels / f"{img_path.stem}.txt"
                with open(label_path, 'w') as f:
                    for mask in results[0].masks.xy:
                        normalized_points = []
                        for point in mask:
                            x_norm = point[0] / w
                            y_norm = point[1] / h
                            normalized_points.extend([x_norm, y_norm])
                        
                        f.write("0 ")
                        f.write(" ".join([f"{coord:.6f}" for coord in normalized_points]))
                        f.write("\n")
                
                total_detections += len(results[0].masks.xy)
        
        # Process validation images
        for idx, img_path in enumerate(val_files):
            progress(
                0.2 + ((len(train_files) + idx) / len(image_files)) * 0.7,
                desc=f"Processing validation image {idx+1}/{len(val_files)}"
            )
            
            # Copy image
            shutil.copy(img_path, val_images / img_path.name)
            
            # Generate label
            results = model(str(img_path), conf=confidence)
            
            if results[0].masks is not None:
                img = cv2.imread(str(img_path))
                h, w = img.shape[:2]
                
                label_path = val_labels / f"{img_path.stem}.txt"
                with open(label_path, 'w') as f:
                    for mask in results[0].masks.xy:
                        normalized_points = []
                        for point in mask:
                            x_norm = point[0] / w
                            y_norm = point[1] / h
                            normalized_points.extend([x_norm, y_norm])
                        
                        f.write("0 ")
                        f.write(" ".join([f"{coord:.6f}" for coord in normalized_points]))
                        f.write("\n")
                
                total_detections += len(results[0].masks.xy)
        
        # Create dataset.yaml
        progress(0.95, desc="Creating configuration file...")
        yaml_content = f"""path: {os.path.abspath(output_folder)}
train: train/images
val: val/images

names:
  0: Profile
"""
        
        yaml_path = output_path / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        progress(1.0, desc="Complete!")
        
        summary = f"""✅ Dataset successfully created!

📁 Location: {output_folder}
📊 Statistics:
   - Total images: {len(image_files)}
   - Training images: {len(train_files)}
   - Validation images: {len(val_files)}
   - Total detections: {total_detections}
   - Config file: {yaml_path}

🎯 Next steps:
1. Review the generated labels (optional)
2. Use this dataset for training in Ceramatic
3. Dataset path for training: {yaml_path}
"""
        
        return str(yaml_path), summary
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def create_manual_structure(images_folder, output_folder, train_split):
    """Create empty dataset structure for manual annotation"""
    try:
        if not os.path.exists(images_folder):
            return None, "❌ Images folder not found"
        
        # Count images
        image_files = list(Path(images_folder).glob("*.jpg")) + \
                     list(Path(images_folder).glob("*.png"))
        
        if len(image_files) == 0:
            return None, "❌ No images found in folder"
        
        # Create structure
        output_path = Path(output_folder)
        train_images = output_path / "train" / "images"
        train_labels = output_path / "train" / "labels"
        val_images = output_path / "val" / "images"
        val_labels = output_path / "val" / "labels"
        
        for dir in [train_images, train_labels, val_images, val_labels]:
            dir.mkdir(parents=True, exist_ok=True)
        
        # Split and copy images
        split_idx = int(len(image_files) * (train_split / 100))
        
        for idx, img_path in enumerate(image_files):
            is_train = idx < split_idx
            dest_images = train_images if is_train else val_images
            dest_labels = train_labels if is_train else val_labels
            
            shutil.copy(img_path, dest_images / img_path.name)
            (dest_labels / f"{img_path.stem}.txt").touch()
        
        # Create dataset.yaml
        yaml_content = f"""path: {os.path.abspath(output_folder)}
train: train/images
val: val/images

names:
  0: Profile
"""
        
        yaml_path = output_path / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        summary = f"""✅ Empty dataset structure created!

📁 Location: {output_folder}
📊 Structure:
   - Training images: {split_idx}
   - Validation images: {len(image_files) - split_idx}
   
🎯 Next steps:
1. Use LabelMe or CVAT to annotate the images
2. Export annotations in YOLO format to the labels folders
3. Or use the 'Convert LabelMe' tab if using LabelMe

📝 Annotation tools:
- LabelMe: pip install labelme && labelme {train_images}
- CVAT: https://www.cvat.ai/
- Roboflow: https://roboflow.com/
"""
        
        return str(yaml_path), summary
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def launch_labelme(folder_path=None):
    """Launch LabelMe annotation tool"""
    try:
        import subprocess
        import sys
        
        # Check if labelme is installed
        try:
            import labelme
        except ImportError:
            # Try to install labelme
            subprocess.check_call([sys.executable, "-m", "pip", "install", "labelme"])
        
        # Launch labelme
        cmd = [sys.executable, "-m", "labelme"]
        if folder_path and os.path.exists(folder_path):
            cmd.append(folder_path)
        
        subprocess.Popen(cmd)
        return f"✅ LabelMe launched! {'Opened folder: ' + folder_path if folder_path else 'Select a folder from the File menu'}"
        
    except Exception as e:
        return f"❌ Error launching LabelMe: {str(e)}\n\nTry installing manually: pip install labelme"


def convert_labelme_annotations(labelme_folder, output_folder):
    """Convert LabelMe JSON to YOLO format"""
    try:
        json_files = list(Path(labelme_folder).glob("*.json"))
        
        if len(json_files) == 0:
            return "❌ No JSON files found in folder", None
        
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        converted = 0
        conversion_details = []
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if 'imageWidth' not in data or 'imageHeight' not in data:
                continue
                
            img_width = data['imageWidth']
            img_height = data['imageHeight']
            
            label_file = output_path / f"{json_file.stem}.txt"
            
            polygon_count = 0
            with open(label_file, 'w') as f:
                for shape in data.get('shapes', []):
                    if shape['shape_type'] == 'polygon':
                        points = shape['points']
                        
                        normalized = []
                        for x, y in points:
                            normalized.extend([x/img_width, y/img_height])
                        
                        f.write("0 ")
                        f.write(" ".join([f"{coord:.6f}" for coord in normalized]))
                        f.write("\n")
                        polygon_count += 1
            
            converted += 1
            conversion_details.append(f"{json_file.stem}: {polygon_count} polygons")
        
        # Create a summary file
        summary_path = output_path / "conversion_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("LabelMe to YOLO Conversion Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Total files converted: {converted}\n\n")
            for detail in conversion_details:
                f.write(f"  {detail}\n")
        
        result = f"""✅ Conversion complete!

📊 Results:
   - JSON files found: {len(json_files)}
   - Successfully converted: {converted}
   - Output folder: {output_folder}
   - Summary saved: {summary_path}

🎯 Next step:
   Copy these label files to your dataset's labels folders
"""
        
        return result, str(output_folder)
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None


# Create Gradio interface
with gr.Blocks(title="YOLO Label Generator", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 🏺 YOLO Label Generator for Ceramatic
        
        Generate training labels for pottery segmentation
        """
    )
    
    with gr.Tabs():
        # Auto-generation tab
        with gr.TabItem("🤖 Auto-Generate Labels"):
            gr.Markdown(
                """
                Use existing Ceramatic model to automatically generate labels.
                These can be refined manually later if needed.
                """
            )
            
            with gr.Row():
                with gr.Column():
                    auto_images = gr.Textbox(
                        label="Images Folder",
                        placeholder="/path/to/pottery/images",
                        value="demo/example_imgs"
                    )
                    auto_output = gr.Textbox(
                        label="Output Dataset Folder",
                        placeholder="/path/to/output/dataset",
                        value="generated_dataset"
                    )
                    auto_model = gr.Textbox(
                        label="Model Path",
                        value="Ceramatic_model_V1.pt",
                        info="Download from: https://drive.google.com/file/d/1b23yWPZ0LKerIM8CWz2DcbhapThCnT7A/view or use 'yolov8m-seg.pt' for testing"
                    )
                    
                    with gr.Row():
                        auto_confidence = gr.Slider(
                            0.1, 1.0, 
                            value=0.8,
                            label="Detection Confidence",
                            info="Lower = more detections"
                        )
                        auto_split = gr.Slider(
                            50, 90,
                            value=80,
                            label="Train Split %",
                            info="Percentage for training"
                        )
                    
                    auto_generate_btn = gr.Button(
                        "🚀 Generate Labels",
                        variant="primary"
                    )
                
                with gr.Column():
                    auto_yaml_path = gr.Textbox(
                        label="Generated YAML Path",
                        interactive=False
                    )
                    auto_result = gr.Textbox(
                        label="Results",
                        lines=15,
                        interactive=False
                    )
        
        # Manual structure tab
        with gr.TabItem("📝 Manual Annotation Setup"):
            gr.Markdown(
                """
                Create dataset structure for manual annotation tools like LabelMe.
                """
            )
            
            with gr.Row():
                with gr.Column():
                    manual_images = gr.Textbox(
                        label="Images Folder",
                        placeholder="/path/to/pottery/images"
                    )
                    manual_output = gr.Textbox(
                        label="Output Dataset Folder",
                        placeholder="/path/to/output/dataset"
                    )
                    manual_split = gr.Slider(
                        50, 90,
                        value=80,
                        label="Train Split %"
                    )
                    with gr.Row():
                        manual_create_btn = gr.Button(
                            "📁 Create Structure",
                            variant="primary"
                        )
                        launch_labelme_btn = gr.Button(
                            "🎨 Open LabelMe",
                            variant="secondary"
                        )
                
                with gr.Column():
                    manual_yaml_path = gr.Textbox(
                        label="Generated YAML Path",
                        interactive=False
                    )
                    manual_result = gr.Textbox(
                        label="Results",
                        lines=10,
                        interactive=False
                    )
        
        # Convert LabelMe tab
        with gr.TabItem("🔄 Convert LabelMe"):
            gr.Markdown(
                """
                Convert LabelMe JSON annotations to YOLO format.
                """
            )
            
            with gr.Row():
                with gr.Column():
                    convert_input = gr.Textbox(
                        label="LabelMe JSON Folder",
                        placeholder="/path/to/labelme/annotations"
                    )
                    convert_output = gr.Textbox(
                        label="Output Labels Folder",
                        placeholder="/path/to/output/labels"
                    )
                    with gr.Row():
                        convert_btn = gr.Button(
                            "🔄 Convert to YOLO",
                            variant="primary"
                        )
                        open_labelme_convert_btn = gr.Button(
                            "🎨 Open LabelMe",
                            variant="secondary"
                        )
                
                with gr.Column():
                    convert_result = gr.Textbox(
                        label="Conversion Results",
                        lines=8,
                        interactive=False
                    )
                    convert_output_path = gr.Textbox(
                        label="Output Path",
                        interactive=False
                    )
        
        # Help tab
        with gr.TabItem("❓ Help"):
            gr.Markdown(
                """
                ## Workflow Guide
                
                ### Option 1: Automatic Generation (Recommended)
                1. Go to "Auto-Generate Labels" tab
                2. Select folder with pottery images
                3. Specify where to save the dataset
                4. Adjust confidence (0.8 is good default)
                5. Click Generate
                6. Use the generated dataset.yaml path for training
                
                ### Option 2: Manual Annotation
                1. Go to "Manual Annotation Setup" tab
                2. Create empty dataset structure with "Create Structure"
                3. Click "Open LabelMe" to start annotating
                4. In LabelMe:
                   - File → Open Dir → select train/images
                   - Edit → Create Polygons (or press 'P')
                   - Click around pottery profile
                   - Double-click to close polygon
                   - Enter label: "Profile"
                   - Ctrl+S to save
                   - 'D' for next image
                5. After annotating all images:
                   - Go to "Convert LabelMe" tab
                   - Input folder: dataset/train/images
                   - Convert to YOLO format
                
                ### Label Format
                YOLO segmentation format (per line):
                ```
                class_id x1 y1 x2 y2 x3 y3 ... xn yn
                ```
                - class_id: Always 0 for single class "Profile"
                - x,y: Normalized coordinates (0-1) of polygon vertices
                
                ### Multi-class for Typology (Future)
                Modify dataset.yaml for pottery types:
                ```yaml
                names:
                  0: amphora
                  1: olla
                  2: plate
                  3: bowl
                ```
                """
            )
    
    # Wire up buttons
    auto_generate_btn.click(
        fn=process_auto_generation,
        inputs=[
            auto_images, auto_output, auto_model, 
            auto_confidence, auto_split
        ],
        outputs=[auto_yaml_path, auto_result]
    )
    
    manual_create_btn.click(
        fn=create_manual_structure,
        inputs=[manual_images, manual_output, manual_split],
        outputs=[manual_yaml_path, manual_result]
    )
    
    # Launch LabelMe for manual annotation
    def launch_labelme_manual():
        # Try to get the created dataset path
        if manual_output.value and os.path.exists(manual_output.value):
            train_path = os.path.join(manual_output.value, "train", "images")
            if os.path.exists(train_path):
                return launch_labelme(train_path)
        return launch_labelme()
    
    launch_labelme_btn.click(
        fn=launch_labelme_manual,
        outputs=[manual_result]
    )
    
    convert_btn.click(
        fn=convert_labelme_annotations,
        inputs=[convert_input, convert_output],
        outputs=[convert_result, convert_output_path]
    )
    
    # Launch LabelMe for conversion
    def launch_labelme_convert():
        if convert_input.value and os.path.exists(convert_input.value):
            return launch_labelme(convert_input.value)
        return launch_labelme()
    
    open_labelme_convert_btn.click(
        fn=launch_labelme_convert,
        outputs=[convert_result]
    )


if __name__ == "__main__":
    # Use different port to avoid conflict with main Ceramatic app
    app.launch(share=False, server_port=7861)