# %%
import os
import json
import numpy as np
from pathlib import Path
import shutil
# %%
CLASS_ID_MAP = {
    "Profile":0
}

IMG_EXTS = [".png", ".jpg", ".jpeg"]

def json2yolo(input_path, output_path=None, class_id_maps=None):
    """Converts a labelAything annotation json file to YOLO txt format
    """
    with open(input_path, "r") as f:
        image_label_json = json.load(f)
    h_w = [image_label_json['imageWidth'], image_label_json['imageHeight']]
    res = []
    for poly_shape in image_label_json['shapes']:
        norm_poly = np.array(poly_shape['points']) / h_w
        str_poly = " ".join(str(x) for x in norm_poly.flatten())
        class_id = class_id_maps[poly_shape['label']] if class_id_maps else poly_shape['label']
        res.append(f"{class_id} {str_poly}")
    res = "\n".join(res)
    filename = Path(input_path).stem + ".txt"
    write_path = output_path if output_path else Path(input_path).parent / filename
    with open(write_path, "w") as f:
        f.write(res)
    return image_label_json['imagePath']

def convert_json_labels_to_yolo(data_dir, out_dir, class_id_maps=CLASS_ID_MAP, val_split=0.8):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    os.makedirs(out_dir / "train", exist_ok=True)
    os.makedirs(out_dir / "val", exist_ok=True)
    os.makedirs(out_dir / "train" / "images", exist_ok=True)
    os.makedirs(out_dir / "train" / "labels", exist_ok=True)
    os.makedirs(out_dir / "val" / "labels", exist_ok=True)
    os.makedirs(out_dir / "val" / "images", exist_ok=True)
    json_annotation_files = list(data_dir.glob("*.json"))
    val_split_idx = int(len(json_annotation_files) * val_split)
    for i, json_label_file in enumerate(json_annotation_files):
        data_split = "train" if i <= val_split_idx else "val"
        out_label_path = out_dir / data_split/ "labels" / (json_label_file.stem + ".txt")
        img_file = json2yolo(json_label_file, out_label_path, class_id_maps=class_id_maps)
        shutil.copy(data_dir/img_file, out_dir/data_split/"images"/img_file)
        
        

if __name__ == "__main__":
    # json2yolo("./narde/data/page24.json", class_id_maps=CLASS_ID_MAP)
    convert_json_labels_to_yolo("./Scansioni", "./datasets/ollae", class_id_maps=CLASS_ID_MAP)
# %%