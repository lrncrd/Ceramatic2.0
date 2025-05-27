# Ceramatic 2.0 

<p align="center">
<img src="https://github.com/lrncrd/Ceramatic2.0/blob/ceramatic-main/imgs/logo_ceramatic.png" width="300"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.7+-blue.svg" alt="Python Version"
> <img src="https://img.shields.io/badge/YOLO-v8-green.svg" alt="YOLO">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg" alt="Platform">
  <img src="https://img.shields.io/badge/status-active-brightgreen.svg" alt="Status">
</p>
<p align="center">
  <a href="https://doi.org/10.1016/j.daach.2025.e00435">
    <img src="https://img.shields.io/badge/DOI-10.1016%2Fj.daach.2025.e00435-blue.svg" alt="DOI">
  </a>
  <img src="https://img.shields.io/badge/Journal-Digital%20Applications%20in%20Archaeology-darkgreen.svg" alt="Journal">
  <img src="https://img.shields.io/badge/Year-2025-red.svg" alt="Publication Year">
</p>
<p align="center">
  <strong>AI-powered archaeological pottery documentation tool</strong>
</p>

---

## 📖 About

Supplementary materials for the paper "*From pencil to pixel: assessing ceramatic 2.0 against manual and laser-aided techniques in archaeological pottery documentation*"

Ceramatic 2.0 is a computer vision tool designed to automate the documentation and analysis of archaeological pottery. 

## 🚀 Quick Start

Both a command line interface (CLI) and a Jupyter Notebook are available for easy usage. 

### Prerequisites
- Python 3.7 or higher
- CUDA-compatible GPU (recommended for faster processing)

### Installation & Usage

1. **Download the model weights** from [Google Drive](https://drive.google.com/file/d/1b23yWPZ0LKerIM8CWz2DcbhapThCnT7A/view?usp=drive_link)

2. **Run with example data:**
```bash
python ceramatic2.py --model_path "Ceramatic_model_V1.pt" --imgs_dir "demo/example_imgs" --tabular_file "demo/metadata_example.xlsx" --diagnostic_plots --add_bar --install_requirements
```

## 📋 Command Line Arguments

### Required Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | `str` | `"Ceramatic_model_V1.pt"` | Path to the YOLO model file |
| `--imgs_dir` | `str` | `"demo/example_imgs"` | Directory containing ceramic images |
| `--tabular_file` | `str` | `"demo/metadata_example.xlsx"` | Path to the metadata Excel file |

### Processing Parameters
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--PIXEL_CM_RATIO` | `float` | `118.11` | Pixel to centimeter conversion ratio |
| `--confidence_threshold` | `float` | `0.8` | YOLO model confidence threshold (0.0-1.0) |
| `--median_filter` | `int` | `41` | Size of median blur filter for noise reduction |

### Visual Style Options
| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--inv_position` | `str` | `"bottom_center"` | `top_left`, `top_center`, `top_right`, `bottom_left`, `bottom_center`, `bottom_right` | Position of inventory number |
| `--profile_style` | `str` | `"filled"` | `filled` | Pottery profile rendering style |
| `--scale_bar_style` | `str` | `"striped"` | `striped`, `solid` | Scale bar appearance |
| `--scale_cm` | `float` | `3.0` | - | Length of scale bar in centimeters |
| `--num_segments` | `int` | `3` | - | Number of segments in striped scale bar |

### Feature Flags
| Argument | Description |
|----------|-------------|
| `--diagnostic` | Process only first 5 images for testing |
| `--diagnostic_plots` | Save intermediate processing plots |
| `--add_bar` | Include scale bar in output images |
| `--add_diameter` | Add diameter measurement line |
| `--add_scale_cm` | Display scale measurements on scale bar |
| `--install_requirements` | Auto-install Python dependencies |

## 💡 Usage Examples

### Basic Processing
```bash
python ceramatic2.py \
  --model_path "Ceramatic_model_V1.pt" \
  --imgs_dir "pottery_photos/" \
  --tabular_file "inventory.xlsx" \
  --add_bar \
  --add_diameter \
  --add_scale_cm \
  --profile_style "filled" \
  --inv_position "bottom_right"
```

### Quick Diagnostic Check
```bash
python ceramatic2.py \
  --diagnostic \
  --diagnostic_plots \
  --confidence_threshold 0.7
```


## 📁 Input File Requirements

### Image Directory Structure
```
imgs_dir/
├── pottery_001.jpg
├── pottery_002.png
├── pottery_003.tiff
└── ...
```
(Supported formats: `.jpg`, `.png`, `.tiff`)

### Metadata File Format
The tabular file should be an spreadsheet (`.xlsx`; `.xls`; `.csv`) file structured with the following columns:

| Column Name | Description |
|-------------|-------------|
|TAV | File's filename (without path or file extension) |
| INV | Inventory ID (unique identifier for each pottery piece - *see below*) |
| DIAM | Diameter measurement (optional - *NaN* for no diameter) |
| FLIP | Indicate if the profile need to be flipped (1) or not (0) |

> ⚠️ **Important**: As you can see in the `demo/metadata_example.xlsx`, INV placement needs to follow the order **left to right** in the scanned table. 

#### Example:

See `demo/metadata_example.xlsx` for a sample metadata file.

## 🎯 Output

Ceramatic 2.0 generates:

- **Diagnostic plots** (if enabled) showing processing steps
- **Technical drawings** ready for archaeological publication

## 🛠️ Development Roadmap

### Completed ✅
- [x] Add inventory placement options
- [x] Cross-platform compatibility (Windows, macOS, Linux)
- [x] Graphic scale styles enhancement

### In Progress 🚧

- [ ] Profile style improvements (filled/outline variants)
- [ ] Basic statistical plotting (PCA analysis)
- [ ] Continuation lines for fragmented pottery
- [ ] Training using a newer YOLO version

## 📦 Version History

| Version | Date | Changes |
|---------|------|---------|
| **1.1.0** | 27/05/2025 | 🎉 Major update<br>✅ Total code refactoring <br>✅ Enhanced scale bar styles and options <br> |
| **1.0.1** | 17/10/2024 | ✅ Added inventory placement options<br>✅ Unix compatibility (Ubuntu 24.04 WSL/native, macOS 15.0) |
| **1.0.0** | 14/10/2024 | 🎉 Initial release |

## 🔧 Technical Requirements

- **Python**: 3.7+
- **Operating Systems**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 5GB free space for model and dependencies
- **GPU**: CUDA-compatible (optional but recommended)

## 📚 Dependencies

Key libraries automatically installed with `--install_requirements`:

- PyTorch / torchvision
- ultralytics (YOLO)
- OpenCV
- NumPy / Pandas
- Matplotlib / Pillow
- openpyxl (Excel support)

## 🤝 Citation

If you use Ceramatic 2.0 in your research, please cite:

```bibtex
@article{alessandri_pencil_2025,
	title = {From pencil to pixel: assessing ceramatic 2.0 against manual and laser-aided techniques in archaeological pottery documentation},
	volume = {37},
	issn = {22120548},
	url = {https://linkinghub.elsevier.com/retrieve/pii/S2212054825000372},
	doi = {10.1016/j.daach.2025.e00435},
	shorttitle = {From pencil to pixel},
	pages = {e00435},
	journaltitle = {Digital Applications in Archaeology and Cultural Heritage},
	author = {Alessandri, L. and Cardarelli, L. and Cesaretti, A. and Dan, R. and Fiorillo, A. and Sotgia, A. and Cusimano, L. and Della Sala, G.A. and Gianni, V. and Rossi, C.},
	date = {2025-06},
}
```

## 📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 🆘 Support & Issues

- **Documentation**: Check this README and inline help (`python ceramatic2.py --help`)
- **Issues**: Report bugs and feature requests on GitHub Issues
- **Contributions**: Contributions are welcome! Please submit a pull request or open an issue.


## 👥 Contributors


<a href="https://github.com/lrncrd/Ceramatic2.0/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=lrncrd/Ceramatic2.0" />
</a>

---

<p align="center">
<img src="https://github.com/lrncrd/Ceramatic2.0/blob/ceramatic-main/imgs/Salt_Project_logo.jpg" width="200"/>
</p>

<p align="center">
<strong>Supported by the Netherlands Organisation for Scientific Research (NWO)</strong><br>
Grant 406.20.HW.013
</p>

<p align="center">
Developed with ❤️ by [Lorenzo Cardarelli](https://github.com/lrncrd)
</p>
