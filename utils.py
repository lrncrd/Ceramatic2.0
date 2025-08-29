from pathlib import Path
import os
from typing import List, Tuple, Optional, Union, Literal, Dict
from enum import Enum
import numpy as np
from skimage import morphology
from skimage.filters import threshold_otsu
import cv2
from scipy.optimize import minimize_scalar
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from skimage.morphology import remove_small_objects
from skimage import measure
from scipy.ndimage import binary_dilation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class ScaleBarStyle(Enum):
    """Enumeration for different scale bar styles."""
    SIMPLE = "simple"
    STRIPED = "striped"
    SOLID = "solid"

        
    


class ProfileStyle(Enum):
    """Enumeration for different profile rendering styles."""
    FILLED = "filled"
    OUTLINE = "outline"
    FILLED_WITH_OUTLINE = "filled_with_outline"
    DOTTED_OUTLINE = "dotted_outline"



class InventoryPosition(Enum):
    """Enumeration for inventory number placement options."""
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    TOP_CENTER = "top_center"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    BOTTOM_CENTER = "bottom_center"



class ImageProcessor:
    """Main class for processing archaeological pottery images using YOLO segmentation."""
    
    def __init__(self, model_path: str, pixel_cm_ratio: float = 118.11):
        """
        Initialize the ImageProcessor.
        
        Args:
            model_path: Path to the YOLO model
            pixel_cm_ratio: Conversion ratio from pixels to centimeters
        """
        self.model_path = model_path
        self.pixel_cm_ratio = pixel_cm_ratio
        self.model = None
        self._setup_directories()

    
    def _setup_directories(self) -> None:
        """Create necessary output directories."""
        self.root_dir = Path(".")
        os.makedirs(self.root_dir / "prediction_diagnostic", exist_ok=True)
        os.makedirs(self.root_dir / "processed_imgs", exist_ok=True)
    
    def _load_model(self) -> None:
        """Load the YOLO model if not already loaded."""
        if self.model is None:
            print(f"Loading YOLO model from: {self.model_path}")
            import torch
            
            # Detect available device
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
                print("Using Apple Silicon GPU (MPS)")
            else:
                device = 'cpu'
                print("Using CPU (this will be slower)")
            
            self.model = YOLO(self.model_path, task='segment')
            self.device = device
    
    def _load_tabular_data(self, tabular_file: Path) -> pd.DataFrame:
        """
        Load tabular data from CSV or Excel file.
        
        Args:
            tabular_file: Path to the tabular file
            
        Returns:
            DataFrame with the tabular data
            
        Raises:
            ValueError: If file format is not supported
        """
        file_ext = tabular_file.suffix.lower()
        if file_ext == ".csv":
            return pd.read_csv(tabular_file)
        elif file_ext in [".xlsx", ".xls"]:
            return pd.read_excel(tabular_file)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _preprocess_image(self, img_path: Path) -> Image.Image:
        """
        Preprocess image by padding if height > width.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Preprocessed PIL Image
        """
        img = Image.open(img_path)
        if img.size[1] >= img.size[0]:
            padding = int(img.size[1] - img.size[0])
            img = ImageOps.expand(img, border=(padding, 0), fill='white')
        return img
    
    def _get_yolo_predictions(self, img: Image.Image, confidence_threshold: float) -> tuple:
        """
        Get YOLO model predictions for an image.
        
        Args:
            img: PIL Image
            confidence_threshold: Confidence threshold for predictions
            
        Returns:
            Tuple of (results, masks_array, result_array)
        """
        print(f"Starting YOLO prediction with confidence={confidence_threshold}, image size={img.size}")
        import time
        start_time = time.time()
        
        try:
            results = self.model.predict(
                img, 
                save_crop=False, 
                conf=confidence_threshold, 
                retina_masks=True, 
                verbose=True,  # Changed to True for debugging
                imgsz=1024,
                device=self.device if hasattr(self, 'device') else None
            )
            
            elapsed = time.time() - start_time
            print(f"YOLO prediction completed in {elapsed:.2f} seconds")
            
            if results and len(results) > 0 and results[0].masks:
                print(f"Found {len(results[0].masks.data)} pottery pieces")
            else:
                print("No pottery pieces detected in image")
                
        except Exception as e:
            print(f"ERROR in YOLO prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        result_array = results[0].plot(masks=True)
        extracted_masks = results[0].masks.data
        masks_array = extracted_masks.cpu().numpy()
        
        return results, masks_array, result_array
    
    def _sort_masks_by_position(self, masks_array: np.ndarray) -> List[Tuple[int, int]]:
        """
        Sort masks by x-coordinate of their bounding boxes.
        
        Args:
            masks_array: Array of masks
            
        Returns:
            List of tuples (mask_index, x_coordinate) sorted by x_coordinate
        """
        order = []
        for i in range(len(masks_array)):
            bbox = find_bounding_box(masks_array[i])
            x_min = bbox[2]  # cmin
            order.append((i, x_min))
        
        return sorted(order, key=lambda x: x[1])
    
    def _process_mask(self, mask: np.ndarray, median_filter_size: int) -> np.ndarray:
        """
        Apply morphological operations to clean up a mask.
        
        Args:
            mask: Input binary mask
            median_filter_size: Size of median blur filter
            
        Returns:
            Processed mask
        """
        kernel = np.ones((9, 9), np.uint8)
        
        # Convert to uint8 format
        processed = (mask.astype(np.uint8) * 255)
        
        # Apply filtering operations
        processed = cv2.GaussianBlur(processed, (9, 9), 0)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        processed = cv2.medianBlur(processed, median_filter_size)
        
        return processed
    
    def _extract_mask_region(self, processed_mask: np.ndarray, flip: bool) -> np.ndarray:
        """
        Extract the region of interest from a processed mask.
        
        Args:
            processed_mask: Processed mask array
            flip: Whether to flip the mask horizontally
            
        Returns:
            Extracted and cleaned mask region
        """
        # Find bounding box and crop
        rmin, rmax, cmin, cmax = find_bounding_box(processed_mask)
        mask = processed_mask[rmin:rmax, cmin:cmax]
        
        # Normalize and clean
        mask = mask / 255
        mask = remove_small_objects(
            mask.astype(bool), 
            min_size=mask.sum() // 10, 
            connectivity=1
        ).astype(int)
        
        # Apply flip if needed
        if flip:
            mask = np.flip(mask, axis=1)
            
        return mask
    
    def _create_profile_analysis(self, mask: np.ndarray) -> Tuple[List[Tuple[int, int]], int]:
        """
        Analyze the mask profile to find first pixels and gradient information.
        
        Args:
            mask: Binary mask array
            
        Returns:
            Tuple of (first_pixel_list, last_zero_gradient_index)
        """
        # Find first pixel in each row
        first_pixels = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] == 1:
                    first_pixels.append((i, j))
                    break
        
        # Calculate gradients and find last zero gradient
        if len(first_pixels) > 1:
            y_coords = np.array([pixel[1] for pixel in first_pixels])
            gradients = np.gradient(y_coords, axis=0)
            zero_gradient_mask = gradients == 0
            zero_indices = np.where(zero_gradient_mask)[0]
            last_zero_gradient = zero_indices[-1] if len(zero_indices) > 0 else 0
        else:
            last_zero_gradient = 0
            
        return first_pixels, last_zero_gradient
    
    def _create_contour_analysis(self, mask: np.ndarray, first_pixels: List[Tuple[int, int]]) -> np.ndarray:
        """
        Create contour analysis for the mask.
        
        Args:
            mask: Binary mask
            first_pixels: List of first pixel coordinates
            
        Returns:
            Contour image array
        """
        padded_mask = np.pad(mask, (1, 1))
        contours = measure.find_contours(padded_mask)
        contour_image = np.zeros_like(padded_mask)
        contours_list = []
        
        # Create contour image
        for contour in contours:
            for coord in contour:
                x, y = int(coord[0]), int(coord[1])
                contour_image[x, y] = 1
                contours_list.append((x, y))
        
        # Find starting point based on first pixel
        if first_pixels:
            first_pixel = first_pixels[0]
            possible_x = [first_pixel[1] + 1, first_pixel[1], first_pixel[1] - 1]
            possible_y = [first_pixel[0] + 1, first_pixel[0], first_pixel[0] - 1]
            
            values = []
            for px in possible_x:
                for py in possible_y:
                    if (py + 1 < contour_image.shape[0] and 
                        px + 1 < contour_image.shape[1] and
                        contour_image[py + 1, px + 1] == 1):
                        values.append((py + 1, px + 1))
            
            if values and values[0] in contours_list:
                index_point = contours_list.index(values[0])
                contour_image_2 = np.zeros_like(contour_image)
                for x, y in contours_list[:index_point]:
                    contour_image_2[x, y] = 1
                return contour_image_2
        
        return contour_image
    
    def _create_reconstructed_pot(self, mask: np.ndarray, diameter_cm: float, 
                                 first_pixels: List[Tuple[int, int]], 
                                 last_zero_gradient: int, 
                                 contour_analysis: np.ndarray) -> np.ndarray:
        """
        Create a reconstructed pot profile with symmetry.
        
        Args:
            mask: Original mask
            diameter_cm: Diameter in centimeters
            first_pixels: List of first pixel coordinates
            last_zero_gradient: Index of last zero gradient
            contour_analysis: Contour analysis result
            
        Returns:
            Reconstructed pot image
        """
        diam_pixels = diameter_cm * self.pixel_cm_ratio
        
        if diam_pixels <= 0:
            return mask
        
        # Create first pixel mask
        first_pixel_mask = np.zeros_like(mask)
        for i, j in first_pixels:
            if i < last_zero_gradient:
                first_pixel_mask[i, j] = 1
            elif i > last_zero_gradient and i > len(first_pixels) // 2:
                first_pixel_mask[i, j] = 0
        
        # Apply morphological operations
        max_values = np.max(np.where(first_pixel_mask), axis=1)
        contour_analysis[max_values[0]:, :] = 0
        first_pixel_mask = binary_dilation(contour_analysis, iterations=5)
        first_pixel_mask = first_pixel_mask[1:-1, 1:-1]
        
        # Create reconstructed pot
        pot_width = int(diam_pixels + (first_pixels[0][1] * 2)) if first_pixels else int(diam_pixels)
        empty_mask = np.zeros((mask.shape[0], pot_width))
        
        # Apply original mask
        empty_mask[:, :mask.shape[1]] = mask
        
        # Create symmetry
        empty_mask_flipped = np.flip(empty_mask, axis=1)
        empty_mask_flipped[:, :mask.shape[1]] = first_pixel_mask
        empty_mask = np.flip(empty_mask_flipped, axis=1)
        
        # Add rim
        empty_mask[0:5, :] = 1
        
        # Clean up based on first pixels
        empty_mask = np.flip(empty_mask, axis=1)
        for i, j in first_pixels:
            empty_mask[i, :j] = 0
        
        empty_mask = np.flip(empty_mask, axis=1)
        for i, j in first_pixels:
            empty_mask[i, :j] = 0
        
        # Add symmetry line
        center = empty_mask.shape[1] // 2
        empty_mask[:, center:center + 5] = 1
        
        return empty_mask
    
    def _apply_profile_style(self, mask: np.ndarray, style: ProfileStyle) -> np.ndarray:
        """
        Apply different profile styles to the mask.
        
        Args:
            mask: Input binary mask
            style: Profile style to apply
            
        Returns:
            Styled mask
        """
        if style == ProfileStyle.FILLED:
            return mask
        
        elif style == ProfileStyle.OUTLINE:
            # Create outline by finding edges
            from scipy import ndimage
            # Dilate and subtract original to get outline
            dilated = ndimage.binary_dilation(mask, iterations=2)
            outline = dilated.astype(int) - mask.astype(int)
            return outline.astype(bool)
        
        elif style == ProfileStyle.FILLED_WITH_OUTLINE:
            # Combine filled and outline
            from scipy import ndimage
            # Create thicker outline
            dilated = ndimage.binary_dilation(mask, iterations=3)
            outline = dilated.astype(int) - mask.astype(int)
            # Combine with original filled mask
            combined = np.logical_or(mask, outline)
            # Make outline darker by creating a weighted mask
            result = mask.astype(float)
            result[outline.astype(bool)] = 0.5  # Gray outline
            return result
        
        elif style == ProfileStyle.DOTTED_OUTLINE:
            # Create dotted outline effect
            from scipy import ndimage
            # Get outline
            dilated = ndimage.binary_dilation(mask, iterations=2)
            outline = dilated.astype(int) - mask.astype(int)
            
            # Create dotted pattern
            y_coords, x_coords = np.where(outline)
            dotted = np.zeros_like(outline)
            # Keep every nth pixel for dotted effect
            for i in range(0, len(y_coords), 3):
                dotted[y_coords[i], x_coords[i]] = 1
            return dotted.astype(bool)
        
        return mask
    
    def _add_scale_bar(self, img: Image.Image, style: ScaleBarStyle, 
                      scale_cm: float = 1.0, add_scale_cm: bool = True, num_segments:int =3) -> Image.Image:
        """
        Add scale bar to image with different styles.
        
        Args:
            img: PIL Image
            style: Scale bar style
            scale_cm: Scale length in centimeters
            
        Returns:
            Image with scale bar
        """
        np_img = np.array(img)
        height, width = np_img.shape[:2]
        
        # Calculate scale bar dimensions
        bar_length_pixels = int(scale_cm * self.pixel_cm_ratio)
        bar_start_x = int(width * 0.05)
        bar_end_x = bar_start_x + bar_length_pixels
        bar_y = int(height * 0.9)
        
        if style == ScaleBarStyle.SIMPLE:
            # Simple black bar
            bar_thickness = 8
            np_img[bar_y:bar_y + bar_thickness, bar_start_x:bar_end_x] = 0
            
        elif style == ScaleBarStyle.SOLID:
            # Solid bar with border
            bar_thickness = 10
            # Draw border
            np_img[bar_y-1:bar_y+bar_thickness+1, bar_start_x-1:bar_end_x+1] = 0
            # Fill with white
            np_img[bar_y:bar_y+bar_thickness, bar_start_x:bar_end_x] = 255
            # Add black edges
            np_img[bar_y:bar_y+bar_thickness, bar_start_x:bar_start_x+2] = 0
            np_img[bar_y:bar_y+bar_thickness, bar_end_x-2:bar_end_x] = 0

            
        elif style == ScaleBarStyle.STRIPED:
            # STRIPED scale bar with even segments and black outline
            bar_thickness = 8
            outline_thickness = 1
            # Allow num_segments as a parameter (default to 3 if not provided)
            
            num_segments = num_segments

            # Draw black outline around the scale bar
            np_img[bar_y - outline_thickness:bar_y, bar_start_x - outline_thickness:bar_end_x + outline_thickness] = 0  # Top
            np_img[bar_y + bar_thickness:bar_y + bar_thickness + outline_thickness, bar_start_x - outline_thickness:bar_end_x + outline_thickness] = 0  # Bottom
            np_img[bar_y - outline_thickness:bar_y + bar_thickness + outline_thickness, bar_start_x - outline_thickness:bar_start_x] = 0  # Left
            np_img[bar_y - outline_thickness:bar_y + bar_thickness + outline_thickness, bar_end_x:bar_end_x + outline_thickness] = 0  # Right

            # Use float to avoid rounding errors
            total_width = bar_end_x - bar_start_x
            segment_width = total_width / num_segments

            for i in range(num_segments):
                seg_start = int(round(bar_start_x + i * segment_width))
                seg_end = int(round(bar_start_x + (i + 1) * segment_width))

                colour = 0 if i % 2 == 0 else 255
                np_img[bar_y:bar_y + bar_thickness, seg_start:seg_end] = colour

        img_with_bar = Image.fromarray(np_img)

        if add_scale_cm:
        
            # Add scale text
            
            draw = ImageDraw.Draw(img_with_bar)
            
            try:
                font = ImageFont.truetype("arial.ttf", 30)
            except OSError:
                font = ImageFont.load_default()
            
            scale_text = f"{scale_cm} cm"
            text_x = bar_start_x
            text_y = bar_y + 20
            draw.text((text_x, text_y), scale_text, fill=0, font=font)
            
        return img_with_bar
    
    def _add_diameter_line(self, img: Image.Image, diameter_cm: float, rim_y: int = None, add_diameter: bool = True) -> Image.Image:
        """
        Add diameter line and label slightly above the rim to the image.
        
        Args:
            img: PIL Image
            diameter_cm: Diameter in centimeters
            rim_y: Y coordinate of the rim (top of the pot)
            add_diameter: Whether to add the diameter line and label
        
        Returns:
            Image with diameter label and optional line
        """
        np_img = np.array(img)
        height, width = np_img.shape[:2]

        # Calculate diameter line position
        center_x = width // 2
        line_y = rim_y if rim_y is not None else int(height * 0.10)  # Fallback if rim_y not provided

        draw = ImageDraw.Draw(img)

        if add_diameter:
            # Prepare font
            try:
                font = ImageFont.truetype("arial.ttf", 25)
            except OSError:
                font = ImageFont.load_default()

            # Create the diameter label
            diameter_text = f"Ø {diameter_cm} cm"
            text_bbox = draw.textbbox((0, 0), diameter_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Centre horizontally and position slightly above the rim line
            text_x = center_x - text_width // 2
            text_y = max(line_y - text_height - int(height * 0.01), 0)  # Ensure it doesn’t go above image
            
            draw.text((text_x, text_y), diameter_text, fill=0, font=font)

        return img

    
    def _add_inventory_number(self, img: Image.Image, inv_number: int, 
                             position: InventoryPosition, font_size: int = 100) -> Image.Image:
        """
        Add inventory number to image at specified position.
        
        Args:
            img: PIL Image
            inv_number: Inventory number
            position: Position for inventory number
            font_size: Font size for inventory number
            
        Returns:
            Image with inventory number
        """
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
        
        inv_text = str(inv_number)
        text_bbox = draw.textbbox((0, 0), inv_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        width, height = img.size
        margin = 25
        
        # Calculate position based on enum
        if position == InventoryPosition.TOP_LEFT:
            x, y = margin, margin
        elif position == InventoryPosition.TOP_RIGHT:
            x, y = width - text_width - margin, margin
        elif position == InventoryPosition.TOP_CENTER:
            x, y = (width - text_width) // 2, margin
        elif position == InventoryPosition.BOTTOM_LEFT:
            x, y = margin, height - text_height - margin
        elif position == InventoryPosition.BOTTOM_RIGHT:
            x, y = width - text_width - margin, height - text_height - margin
        elif position == InventoryPosition.BOTTOM_CENTER:
            x, y = (width - text_width) // 2, height - text_height - margin
        else:
            x, y = margin, margin  # Default to top-left
        
        draw.text((x, y), inv_text, fill=0, font=font)
        
        return img
    
    def _create_final_image(self, mask: np.ndarray, inv_number: int, diameter_cm: float,
                           profile_style: ProfileStyle = ProfileStyle.FILLED,
                           scale_bar_style: ScaleBarStyle = ScaleBarStyle.SIMPLE,
                           inventory_position: InventoryPosition = InventoryPosition.TOP_LEFT,
                           add_bar: bool = False,
                           add_scale_cm: bool = True,
                           scale_cm: float = 1.0,
                           num_segments: int = 3,
                           add_diameter: bool = True) -> Image.Image:
        """
        Create the final processed image with all styling options.
        
        Args:
            mask: Processed mask
            inv_number: Inventory number
            diameter_cm: Diameter in centimeters
            profile_style: Style for the pottery profile
            scale_bar_style: Style for the scale bar
            diameter_line_style: Style for diameter line
            inventory_position: Position for inventory number
            add_bar: Whether to add scale bar
            scale_cm: Scale bar length in cm
            
        Returns:
            Final PIL Image
        """
        # Apply profile style
        styled_mask = self._apply_profile_style(mask, profile_style)
        
        # Convert to PIL and handle different mask types
        if styled_mask.dtype == np.float64 or styled_mask.dtype == np.float32:
            # Handle masks with grayscale values (e.g., filled_with_outline)
            img = Image.fromarray((styled_mask * 255).astype(np.uint8))
        else:
            # Handle boolean masks
            img = Image.fromarray((styled_mask * 255).astype(np.uint8))
        
        img = img.convert("L")
        img = ImageOps.invert(img)
        img = ImageOps.expand(img, border=200, fill='white')
        
        # Add scale bar if requested
        if add_bar:
            img = self._add_scale_bar(img, scale_bar_style, scale_cm, add_scale_cm=add_scale_cm, num_segments=num_segments)
        
        # Add diameter line
        if diameter_cm > 0 and add_diameter:
            img = self._add_diameter_line(img, diameter_cm, add_diameter=add_diameter)
        
        # Add inventory number
        img = self._add_inventory_number(img, inv_number, inventory_position)
        
        return img
    
    def _save_diagnostic_plot(self, result_array: np.ndarray, img_name: str) -> None:
        """Save diagnostic plot if enabled."""
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(result_array)
        fig.savefig(
            self.root_dir / "prediction_diagnostic" / f"diagnostic_{img_name}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)
    
    def _validate_diameter(self, diameter: Union[str, int, float], img_name: str) -> bool:
        """
        Validate diameter value.
        
        Args:
            diameter: Diameter value to validate
            img_name: Image name for error reporting
            
        Returns:
            True if valid, False otherwise
        """
        import numpy as np
        import pandas as pd
        
        # Check for NaN or None
        if pd.isna(diameter) or diameter is None:
            print(f"Error in image {img_name}: diameter is NaN or None. The image will be skipped.")
            return False
            
        # Check for empty strings
        if isinstance(diameter, str) and diameter.isspace():
            print(f"Error in image {img_name}: diameter is a space. The image will be skipped.")
            return False
        
        # Accept Python int/float and numpy numeric types
        if not isinstance(diameter, (int, float, np.integer, np.floating)):
            print(f"Error in image {img_name}: diameter is not a number (type: {type(diameter)}). The image will be skipped.")
            return False
        
        return True
    
    def process_images(self, imgs_dir: str, tabular_file: str, 
                    confidence_threshold: float = 0.8,
                    diagnostic: bool = False, 
                    diagnostic_plots: bool = False,
                    profile_style: ProfileStyle = ProfileStyle.FILLED,
                    scale_bar_style: ScaleBarStyle = ScaleBarStyle.SIMPLE,
                    inventory_position: InventoryPosition = InventoryPosition.TOP_LEFT,
                    add_bar: bool = False,
                    add_scale_cm: bool = True,
                    num_segments: int = 3,
                    scale_cm: float = 1.0,
                    add_diameter: bool = True,
                    median_filter: int = 41,
                    add_continuation_lines: bool = False,
                    ) -> List[Tuple[str, int, np.ndarray]]:
        """
        Process images using YOLO segmentation and create reconstructed pottery profiles.
        
        Args:
            imgs_dir: Directory containing images
            tabular_file: Path to tabular data file
            confidence_threshold: YOLO confidence threshold
            diagnostic: Process only first 5 images for testing
            diagnostic_plots: Save diagnostic plots
            profile_style: Style for pottery profiles
            scale_bar_style: Style for scale bars
            diameter_line_style: Style for diameter lines
            inventory_position: Position for inventory numbers
            add_bar: Add scale bar to images
            scale_cm: Scale bar length in centimeters
            median_filter: Size of median blur filter
            
        Returns:
            List of tuples (image_name, inventory_number, processed_mask)
        """
        self._load_model()
        
        imgs_path = self.root_dir / imgs_dir
        tabular_path = self.root_dir / tabular_file
        
        # Validate paths exist
        if not imgs_path.exists():
            print(f"Error: Images directory '{imgs_path}' does not exist")
            return []
        
        if not tabular_path.exists():
            print(f"Error: Tabular file '{tabular_path}' does not exist")
            return []
        
        # Load tabular data
        try:
            tabular_data = self._load_tabular_data(tabular_path)
            print(f"Loaded tabular data with {len(tabular_data)} rows")
            print(f"Columns: {list(tabular_data.columns)}")
        except Exception as e:
            print(f"Error loading tabular file: {e}")
            return []
        
        # Validate required columns exist
        required_columns = ["TAV", "DIAM", "INV", "FLIP"]
        missing_columns = [col for col in required_columns if col not in tabular_data.columns]
        if missing_columns:
            print(f"Error: Missing required columns in tabular data: {missing_columns}")
            return []
        
        processed_imgs = []
        
        # Get image list - filter for image files only
        try:
            img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            img_list = [f for f in os.listdir(imgs_path) 
                    if Path(f).suffix.lower() in img_extensions]
            
            if not img_list:
                print(f"No image files found in {imgs_path}")
                return []
                
            print(f"Found {len(img_list)} image files")
            
            if diagnostic:
                img_list = img_list[:5]
                print(f"Diagnostic mode: processing first {len(img_list)} images")
                
        except Exception as e:
            print(f"Error reading images directory: {e}")
            return []
        
        # Process each image
        for img_filename in tqdm(img_list, desc="Processing images"):
            try:
                img_name = Path(img_filename).stem  # Use stem to get filename without extension
                img_path = imgs_path / img_filename
                
                print(f"\nProcessing image: {img_filename} (name: {img_name})")
                
                # Check if image file exists and is readable
                if not img_path.exists():
                    print(f"Warning: Image file {img_path} does not exist")
                    continue
                    
                # Preprocess image
                try:
                    img = self._preprocess_image(img_path)
                    print(f"Image loaded successfully: {img.size}")
                except Exception as e:
                    print(f"Error loading image {img_filename}: {e}")
                    continue
                
                # Get YOLO predictions
                try:
                    results, masks_array, result_array = self._get_yolo_predictions(
                        img, confidence_threshold
                    )
                    print(f"YOLO detected {len(masks_array)} masks")
                    
                    # Check if any masks were detected
                    if len(masks_array) == 0:
                        print(f"No masks detected in {img_filename}")
                        continue
                        
                except Exception as e:
                    print(f"Error in YOLO prediction for {img_filename}: {e}")
                    continue
                
                # Save diagnostic plot if requested
                if diagnostic_plots:
                    try:
                        self._save_diagnostic_plot(result_array, img_name)
                    except Exception as e:
                        print(f"Error saving diagnostic plot for {img_filename}: {e}")
                
                # Sort masks by position
                try:
                    sorted_masks = self._sort_masks_by_position(masks_array)
                    print(f"Masks sorted by position")
                except Exception as e:
                    print(f"Error sorting masks for {img_filename}: {e}")
                    continue
                
                # Get corresponding tabular data - try both string and integer matching
                try:
                    # Try matching as integer first
                    try:
                        img_name_int = int(img_name)
                        df_info = tabular_data.loc[tabular_data["TAV"] == img_name_int]
                    except ValueError:
                        # If conversion fails, try string matching
                        df_info = tabular_data.loc[tabular_data["TAV"].astype(str) == img_name]
                    
                    if df_info.empty:
                        print(f"No tabular data found for image {img_name}")
                        continue
                        
                    df_info = df_info.copy()
                    
                    # Handle FLIP column safely
                    if "FLIP" in df_info.columns:
                        df_info["FLIP"] = df_info["FLIP"].astype(bool)
                    else:
                        df_info["FLIP"] = False  # Default value
                        
                    print(f"Found {len(df_info)} rows in tabular data")
                    
                except Exception as e:
                    print(f"Error matching tabular data for {img_filename}: {e}")
                    continue
                
                # Check if number of masks matches tabular data
                if len(sorted_masks) == len(df_info):
                    num_to_process = len(sorted_masks)
                
                    # Process each mask
                    for i in range(num_to_process):
                        try:
                            mask_idx, _ = sorted_masks[i]
                            row_data = df_info.iloc[i]
                            
                            print(f"  Processing mask {i+1}/{num_to_process}")
                            
                            # Validate diameter
                            if not self._validate_diameter(row_data["DIAM"], img_name):
                                continue
                            
                            # Process the mask
                            processed_mask = self._process_mask(masks_array[mask_idx], median_filter)
                            
                            # Extract mask region
                            mask_region = self._extract_mask_region(processed_mask, row_data["FLIP"])
                            
                            # Analyze profile
                            first_pixels, last_zero_gradient = self._create_profile_analysis(mask_region)
                            
                            # Create contour analysis
                            contour_analysis = self._create_contour_analysis(mask_region, first_pixels)
                            
                            # Create reconstructed pot or use original mask
                            diameter_cm = float(row_data["DIAM"])
                            if diameter_cm > 0:
                                final_mask = self._create_reconstructed_pot(
                                    mask_region, diameter_cm, first_pixels, 
                                    last_zero_gradient, contour_analysis
                                )
                            else:
                                final_mask = mask_region
                            
                            # Add continuation lines if requested and fragments detected
                            if add_continuation_lines:
                                fragments = self.detect_fragments(final_mask)
                                if len(fragments) > 1:
                                    print(f"  Detected {len(fragments)} fragments, adding continuation lines")
                                    final_mask = self.add_continuation_lines(final_mask)
                            
                            # Create final image with all styling parameters
                            inv_number = int(row_data['INV'])
                            final_img = self._create_final_image(
                                final_mask, 
                                inv_number, 
                                diameter_cm,
                                profile_style=profile_style,
                                scale_bar_style=scale_bar_style,
                                inventory_position=inventory_position,
                                add_bar=add_bar,
                                add_scale_cm = add_scale_cm,
                                scale_cm=scale_cm,
                                num_segments=num_segments,
                                add_diameter=add_diameter
                            )
                            
                            # Save image
                            output_path = self.root_dir / "processed_imgs" / f"{inv_number}.jpg"
                            final_img.save(output_path)
                            print(f"  Saved: {output_path}")
                            
                            # Add to results
                            processed_imgs.append((img_name, inv_number, final_mask))
                            
                        except Exception as e:
                            print(f"Error processing mask {i} in image {img_filename}: {e}")
                            continue
                else:
                    print(f"Warning: Number of masks ({len(sorted_masks)}) does not match tabular data rows ({len(df_info)}) for {img_filename}. Skipping this image.")
                    continue
                        
            except Exception as e:
                print(f"Error processing image {img_filename}: {e}")
                continue
        
        print(f"\nProcessing complete. Successfully processed {len(processed_imgs)} items.")
        return processed_imgs
    
    def extract_shape_features(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Extract shape features from a pottery mask for statistical analysis.
        
        Args:
            mask: Binary mask of pottery profile
            
        Returns:
            Dictionary of shape features
        """
        from skimage import measure
        import cv2
        
        # Ensure mask is binary
        mask = mask.astype(bool)
        
        # Get basic properties
        props = measure.regionprops(mask.astype(int))[0]
        
        # Find contour for more advanced features
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            
            # Fit ellipse if possible
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
            else:
                major_axis = props.major_axis_length
                minor_axis = props.minor_axis_length
        else:
            major_axis = props.major_axis_length
            minor_axis = props.minor_axis_length
        
        # Calculate features
        features = {
            'area': props.area,
            'perimeter': props.perimeter,
            'major_axis_length': major_axis,
            'minor_axis_length': minor_axis,
            'eccentricity': props.eccentricity,
            'solidity': props.solidity,
            'extent': props.extent,
            'orientation': props.orientation,
            'circularity': 4 * np.pi * props.area / (props.perimeter ** 2) if props.perimeter > 0 else 0,
            'aspect_ratio': major_axis / minor_axis if minor_axis > 0 else 1,
            'convex_area': props.convex_area,
            'equivalent_diameter': props.equivalent_diameter,
        }
        
        # Add height and width
        bbox = props.bbox
        features['height'] = bbox[2] - bbox[0]
        features['width'] = bbox[3] - bbox[1]
        features['height_width_ratio'] = features['height'] / features['width'] if features['width'] > 0 else 1
        
        return features
    
    def perform_pca_analysis(self, processed_images: List[Tuple[str, int, np.ndarray]], 
                           save_plot: bool = True, output_dir: str = "statistical_analysis") -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Perform PCA analysis on processed pottery images.
        
        Args:
            processed_images: List of tuples (image_name, inv_number, mask)
            save_plot: Whether to save PCA plot
            output_dir: Directory to save analysis results
            
        Returns:
            Tuple of (transformed_data, features_df)
        """
        if len(processed_images) < 3:
            print("Warning: PCA requires at least 3 samples. Skipping analysis.")
            return None, None
        
        # Extract features for all images
        feature_list = []
        inv_numbers = []
        
        for img_name, inv_num, mask in processed_images:
            try:
                features = self.extract_shape_features(mask)
                feature_list.append(list(features.values()))
                inv_numbers.append(inv_num)
            except Exception as e:
                print(f"Error extracting features for {img_name}: {e}")
                continue
        
        if len(feature_list) < 3:
            print("Not enough valid samples for PCA.")
            return None, None
        
        # Create feature matrix
        feature_names = list(self.extract_shape_features(processed_images[0][2]).keys())
        X = np.array(feature_list)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=min(2, len(feature_list)))
        X_pca = pca.fit_transform(X_scaled)
        
        # Create DataFrame with results
        features_df = pd.DataFrame(X, columns=feature_names)
        features_df['inv_number'] = inv_numbers
        features_df['PC1'] = X_pca[:, 0]
        if X_pca.shape[1] > 1:
            features_df['PC2'] = X_pca[:, 1]
        
        # Save plot if requested
        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create interactive Plotly figure with subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['PCA - Pottery Shape Analysis', 
                               'Top 10 Feature Contributions to PC1',
                               'Distribution of Key Shape Features',
                               'Feature Correlation Matrix'],
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                       [{"type": "box"}, {"type": "heatmap"}]],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            # PCA scatter plot
            if X_pca.shape[1] > 1:
                fig.add_trace(
                    go.Scatter(
                        x=X_pca[:, 0], 
                        y=X_pca[:, 1],
                        mode='markers+text',
                        text=[str(inv) for inv in inv_numbers],
                        textposition='top center',
                        marker=dict(size=12, color='#ff6b6b', line=dict(width=2, color='darkred')),
                        name='Pottery samples'
                    ),
                    row=1, col=1
                )
                fig.update_xaxes(title_text=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', row=1, col=1)
                fig.update_yaxes(title_text=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', row=1, col=1)
            else:
                fig.add_trace(
                    go.Scatter(
                        x=X_pca[:, 0], 
                        y=np.zeros_like(X_pca[:, 0]),
                        mode='markers+text',
                        text=[str(inv) for inv in inv_numbers],
                        textposition='top center',
                        marker=dict(size=12, color='#ff6b6b'),
                        name='Pottery samples'
                    ),
                    row=1, col=1
                )
                fig.update_xaxes(title_text=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', row=1, col=1)
            
            # Feature importance bar chart
            feature_importance = np.abs(pca.components_[0])
            sorted_idx = np.argsort(feature_importance)[::-1][:10]
            fig.add_trace(
                go.Bar(
                    x=feature_importance[sorted_idx],
                    y=[feature_names[i] for i in sorted_idx],
                    orientation='h',
                    marker_color='#4ecdc4',
                    name='Feature Importance'
                ),
                row=1, col=2
            )
            fig.update_xaxes(title_text='Absolute Loading on PC1', row=1, col=2)
            
            # Box plots for key features
            for i, feature in enumerate(['area', 'aspect_ratio', 'circularity']):
                fig.add_trace(
                    go.Box(
                        y=features_df[feature],
                        name=feature,
                        marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1'][i]
                    ),
                    row=2, col=1
                )
            
            # Correlation heatmap
            top_features = ['area', 'perimeter', 'aspect_ratio', 'circularity', 'solidity']
            corr_matrix = features_df[top_features].corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=top_features,
                    y=top_features,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    name='Correlation'
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=False,
                title_text="<b>PCA Analysis Results - Interactive Visualization</b>",
                title_font=dict(size=20)
            )
            
            # Save as interactive HTML
            fig.write_html(os.path.join(output_dir, 'pca_analysis_interactive.html'))
            # Save static image
            fig.write_image(os.path.join(output_dir, 'pca_analysis.png'), width=1200, height=800)
            
            # Save feature data
            features_df.to_csv(os.path.join(output_dir, 'shape_features.csv'), index=False)
            
            # Save PCA summary
            with open(os.path.join(output_dir, 'pca_summary.txt'), 'w') as f:
                f.write("PCA Analysis Summary\n")
                f.write("===================\n\n")
                f.write(f"Number of samples: {len(feature_list)}\n")
                f.write(f"Number of features: {len(feature_names)}\n")
                f.write(f"Explained variance ratio: {pca.explained_variance_ratio_}\n")
                f.write(f"Total variance explained: {sum(pca.explained_variance_ratio_):.1%}\n\n")
                f.write("Feature statistics:\n")
                f.write(features_df.describe().to_string())
        
        return X_pca, features_df
    
    def add_continuation_lines(self, mask: np.ndarray, gap_threshold: int = 20) -> np.ndarray:
        """
        Add continuation lines to fragmented pottery profiles.
        
        Args:
            mask: Binary mask of pottery profile
            gap_threshold: Minimum gap size in pixels to add continuation lines
            
        Returns:
            Mask with continuation lines added
        """
        import cv2
        from scipy import ndimage
        
        # Create a copy to work with
        result_mask = mask.copy()
        
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        
        # Get the main contour (largest)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Extract boundary points from left side (profile edge)
        boundary_points = []
        for point in main_contour:
            boundary_points.append((point[0][1], point[0][0]))  # (y, x) format
        
        # Sort by y-coordinate
        boundary_points.sort(key=lambda p: p[0])
        
        # Find gaps in the profile
        gaps = []
        for i in range(1, len(boundary_points)):
            y_gap = boundary_points[i][0] - boundary_points[i-1][0]
            if y_gap > gap_threshold:
                # Found a gap
                gap_start = boundary_points[i-1]
                gap_end = boundary_points[i]
                gaps.append((gap_start, gap_end))
        
        # Draw continuation lines for gaps
        for gap_start, gap_end in gaps:
            # Create a dashed line pattern
            y1, x1 = gap_start
            y2, x2 = gap_end
            
            # Calculate line parameters
            num_dashes = max(3, (y2 - y1) // 10)
            dash_length = (y2 - y1) / (2 * num_dashes)
            
            # Draw dashed line
            for i in range(num_dashes):
                dash_start_y = int(y1 + i * 2 * dash_length)
                dash_end_y = int(y1 + (i * 2 + 1) * dash_length)
                
                # Interpolate x coordinates
                t_start = (dash_start_y - y1) / (y2 - y1)
                t_end = (dash_end_y - y1) / (y2 - y1)
                dash_start_x = int(x1 + t_start * (x2 - x1))
                dash_end_x = int(x1 + t_end * (x2 - x1))
                
                # Draw the dash segment
                cv2.line(result_mask, (dash_start_x, dash_start_y), 
                        (dash_end_x, dash_end_y), 1, thickness=2)
        
        return result_mask
    
    def detect_fragments(self, mask: np.ndarray, min_fragment_size: int = 100) -> List[np.ndarray]:
        """
        Detect separate fragments in a pottery mask.
        
        Args:
            mask: Binary mask
            min_fragment_size: Minimum size for a valid fragment
            
        Returns:
            List of fragment masks
        """
        from skimage import measure
        
        # Label connected components
        labeled_mask = measure.label(mask, connectivity=1)
        regions = measure.regionprops(labeled_mask)
        
        # Filter fragments by size
        fragments = []
        for region in regions:
            if region.area >= min_fragment_size:
                # Create individual fragment mask
                fragment_mask = np.zeros_like(mask)
                fragment_mask[labeled_mask == region.label] = 1
                fragments.append(fragment_mask)
        
        return fragments
    
def find_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find the bounding box of a binary mask.
    
    Args:
        mask: Binary mask array
        
    Returns:
        Tuple of (rmin, rmax, cmin, cmax) - bounding box coordinates
    """
    # Find all non-zero coordinates
    coords = np.where(mask > 0)
    
    if len(coords[0]) == 0:
        # Empty mask
        return 0, 0, 0, 0
    
    # Calculate bounding box
    rmin = int(np.min(coords[0]))
    rmax = int(np.max(coords[0])) + 1
    cmin = int(np.min(coords[1]))
    cmax = int(np.max(coords[1])) + 1
    
    return rmin, rmax, cmin, cmax

if __name__ == "__main__":
    # Example usage
    processor = ImageProcessor(model_path="Ceramatic_model_V1.pt")
    results = processor.process_images(
        imgs_dir="eccoli/imgs",
        tabular_file="eccoli/metadata_example.xlsx",
        confidence_threshold=0.8,
        diagnostic=True,
        diagnostic_plots=True,
        profile_style=ProfileStyle.FILLED,
        scale_bar_style=ScaleBarStyle.STRIPED,
        inventory_position=InventoryPosition.BOTTOM_CENTER,
        add_bar=True,
        add_scale_cm = True,
        scale_cm=3.0,
        median_filter=41,
        add_diameter=True,
        num_segments=3
    )
