#!/usr/bin/env python3
"""
Generate YOLO format labels from images using existing Ceramatic model
This creates initial annotations that can be refined manually
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
from ultralytics import YOLO

def generate_labels_from_model(
    images_dir,
    output_dir,
    model_path="Ceramatic_model_V1.pt",
    confidence_threshold=0.8
):
    """
    Generate YOLO polygon labels using existing model predictions
    """
    # Create output directories
    train_images = Path(output_dir) / "train" / "images"
    train_labels = Path(output_dir) / "train" / "labels"
    val_images = Path(output_dir) / "val" / "images"
    val_labels = Path(output_dir) / "val" / "labels"
    
    for dir in [train_images, train_labels, val_images, val_labels]:
        dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Process images
    image_files = list(Path(images_dir).glob("*.jpg")) + \
                  list(Path(images_dir).glob("*.png")) + \
                  list(Path(images_dir).glob("*.tiff"))
    
    print(f"Found {len(image_files)} images to process")
    
    # Split 80/20 for train/val
    split_idx = int(len(image_files) * 0.8)
    
    for idx, img_path in enumerate(image_files):
        print(f"Processing {idx+1}/{len(image_files)}: {img_path.name}")
        
        # Decide train or val
        is_train = idx < split_idx
        dest_images = train_images if is_train else val_images
        dest_labels = train_labels if is_train else val_labels
        
        # Copy image
        dest_img_path = dest_images / img_path.name
        Image.open(img_path).save(dest_img_path)
        
        # Run inference
        results = model(str(img_path), conf=confidence_threshold)
        
        if results[0].masks is not None:
            # Get image dimensions
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            
            # Create label file
            label_path = dest_labels / f"{img_path.stem}.txt"
            
            with open(label_path, 'w') as f:
                for mask in results[0].masks.xy:
                    # Normalize coordinates
                    normalized_points = []
                    for point in mask:
                        x_norm = point[0] / w
                        y_norm = point[1] / h
                        normalized_points.extend([x_norm, y_norm])
                    
                    # Write as: class_id x1 y1 x2 y2 ... xn yn
                    f.write("0 ")  # Class 0 for Profile
                    f.write(" ".join([f"{coord:.6f}" for coord in normalized_points]))
                    f.write("\n")
            
            print(f"  ✓ Generated label with {len(results[0].masks.xy)} polygons")
        else:
            print(f"  ⚠ No detections found")
    
    # Create dataset.yaml
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: train/images
val: val/images

names:
  0: Profile
"""
    
    yaml_path = Path(output_dir) / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Dataset created at: {output_dir}")
    print(f"   - Training images: {len(list(train_images.glob('*')))} ")
    print(f"   - Validation images: {len(list(val_images.glob('*')))}")
    print(f"   - YAML config: {yaml_path}")
    
    return yaml_path


def generate_labels_manual(images_dir, output_dir):
    """
    Create empty label structure for manual annotation
    """
    # Create directories
    train_images = Path(output_dir) / "train" / "images"
    train_labels = Path(output_dir) / "train" / "labels"
    val_images = Path(output_dir) / "val" / "images"
    val_labels = Path(output_dir) / "val" / "labels"
    
    for dir in [train_images, train_labels, val_images, val_labels]:
        dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images and create empty labels
    image_files = list(Path(images_dir).glob("*.jpg")) + \
                  list(Path(images_dir).glob("*.png"))
    
    split_idx = int(len(image_files) * 0.8)
    
    for idx, img_path in enumerate(image_files):
        is_train = idx < split_idx
        dest_images = train_images if is_train else val_images
        dest_labels = train_labels if is_train else val_labels
        
        # Copy image
        Image.open(img_path).save(dest_images / img_path.name)
        
        # Create empty label
        (dest_labels / f"{img_path.stem}.txt").touch()
    
    print(f"\n✅ Empty dataset structure created at: {output_dir}")
    print("   Now use a labeling tool like LabelMe to annotate the images")
    print("   Export annotations in YOLO polygon format")


def convert_labelme_to_yolo(labelme_dir, output_dir):
    """
    Convert LabelMe JSON annotations to YOLO format
    """
    import json
    
    for json_file in Path(labelme_dir).glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        img_width = data['imageWidth']
        img_height = data['imageHeight']
        
        # Create corresponding label file
        label_file = Path(output_dir) / f"{json_file.stem}.txt"
        
        with open(label_file, 'w') as f:
            for shape in data['shapes']:
                if shape['shape_type'] == 'polygon':
                    points = shape['points']
                    
                    # Normalize and flatten
                    normalized = []
                    for x, y in points:
                        normalized.extend([x/img_width, y/img_height])
                    
                    # Write: class_id x1 y1 x2 y2 ...
                    f.write("0 ")
                    f.write(" ".join([f"{coord:.6f}" for coord in normalized]))
                    f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YOLO labels for pottery training")
    parser.add_argument("images_dir", help="Directory containing pottery images")
    parser.add_argument("output_dir", help="Output directory for YOLO dataset")
    parser.add_argument("--mode", choices=["auto", "manual", "convert_labelme"], 
                        default="auto", help="Label generation mode")
    parser.add_argument("--model_path", default="Ceramatic_model_V1.pt",
                        help="Path to existing model (for auto mode)")
    parser.add_argument("--confidence", type=float, default=0.8,
                        help="Confidence threshold for auto labeling")
    
    args = parser.parse_args()
    
    if args.mode == "auto":
        generate_labels_from_model(
            args.images_dir,
            args.output_dir,
            args.model_path,
            args.confidence
        )
    elif args.mode == "manual":
        generate_labels_manual(args.images_dir, args.output_dir)
    elif args.mode == "convert_labelme":
        convert_labelme_to_yolo(args.images_dir, args.output_dir)