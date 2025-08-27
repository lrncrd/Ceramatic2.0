from pathlib import Path
import os
from typing import List, Tuple, Optional, Union, Literal
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
from skimage.morphology import remove_small_objects
from skimage import measure
from scipy.ndimage import binary_dilation
import warnings

warnings.filterwarnings("ignore")


class ScaleBarStyle(Enum):
    """Enumeration for different scale bar styles."""
    SIMPLE = "simple"
    STRIPED = "striped"

        
    


class ProfileStyle(Enum):
    """Enumeration for different profile rendering styles."""
    FILLED = "filled"



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
        
        ### to add other styles in the future
        
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
        
        # Convert to PIL and invert
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
