# YOLO Label Generator Guide

## Quick Start

### 1. Launch the Label Generator
```bash
python label_generator_gui.py
```

Or click "🏷️ Open Label Generator" in the Ceramatic Training tab.

## Usage Options

### Option 1: Auto-Generate Labels (Recommended)
1. **Prepare your pottery images** in a folder
2. Go to **"Auto-Generate Labels"** tab
3. Set:
   - Images Folder: `path/to/your/images`
   - Output Dataset: `my_dataset` (will be created)
   - Model Path: `Ceramatic_model_V1.pt`
   - Confidence: 0.8 (lower = more detections)
   - Train Split: 80% (80% train, 20% validation)
4. Click **"Generate Labels"**
5. Use the generated `dataset.yaml` path for training

### Option 2: Manual Annotation
1. Go to **"Manual Annotation Setup"** tab
2. Create empty dataset structure
3. Use **LabelMe** to annotate:
   ```bash
   pip install labelme
   labelme my_dataset/train/images
   ```
4. Draw polygons around pottery profiles
5. Save annotations
6. Use **"Convert LabelMe"** tab to convert to YOLO format

## Dataset Structure
```
my_dataset/
├── train/
│   ├── images/     # .jpg/.png files
│   └── labels/     # .txt polygon labels
├── val/
│   ├── images/
│   └── labels/
└── dataset.yaml    # configuration
```

## Label Format (YOLO Polygons)
Each `.txt` file contains:
```
0 x1 y1 x2 y2 x3 y3 ... xn yn
```
- `0` = class ID (always 0 for "Profile")
- `x,y` = normalized coordinates (0-1)

## Command Line Usage
```bash
# Auto-generate with existing model
python generate_yolo_labels.py images/ dataset/ --mode auto

# Create structure for manual annotation
python generate_yolo_labels.py images/ dataset/ --mode manual

# Convert LabelMe JSON to YOLO
python generate_yolo_labels.py json_folder/ labels/ --mode convert_labelme
```

## Tips
- **Review auto-generated labels** - they may need refinement
- **Use high confidence (0.8+)** for cleaner initial labels
- **For typology classification** - modify dataset.yaml:
  ```yaml
  names:
    0: amphora
    1: olla
    2: plate
    3: bowl
  ```

## Troubleshooting
- **No detections?** Lower confidence threshold
- **Too many detections?** Increase confidence
- **Wrong shapes?** Use manual annotation tools
- **Import errors?** Ensure all dependencies installed:
  ```bash
  pip install ultralytics opencv-python pillow gradio
  ```