import argparse
import subprocess
import sys
from utils import ImageProcessor, ProfileStyle, ScaleBarStyle, InventoryPosition


def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def main():
    parser = argparse.ArgumentParser(description="Apply image processing model to ceramic images")
    parser.add_argument("--model_path", type=str, default="Ceramatic_model_V1.pt", help="Path to the YOLO model file")
    parser.add_argument("--imgs_dir", type=str, default="demo/example_imgs", help="Directory containing test images")
    parser.add_argument("--tabular_file", type=str, default="demo/metadata_example.xlsx", help="Path to the metadata file")
    parser.add_argument("--PIXEL_CM_RATIO", type=float, default=118.11, help="Pixel to cm ratio")
    parser.add_argument("--confidence_threshold", type=float, default=0.8, help="YOLO confidence threshold")
    parser.add_argument("--median_filter", type=int, default=41, help="Size of the median blur filter")
    parser.add_argument("--inv_position", type=str, default="bottom_center", choices=[pos.value for pos in InventoryPosition], help="Position of the inventory number")
    parser.add_argument("--profile_style", type=str, default="filled", choices=[style.value for style in ProfileStyle], help="Pottery profile rendering style")
    parser.add_argument("--scale_bar_style", type=str, default="striped", choices=[style.value for style in ScaleBarStyle], help="Scale bar style")
    parser.add_argument("--scale_cm", type=float, default=3.0, help="Length of the scale bar in cm")
    parser.add_argument("--num_segments", type=int, default=3, help="Number of segments in the striped scale bar")
    parser.add_argument("--diagnostic", action="store_true", help="Enable diagnostic mode (process only 5 images)")
    parser.add_argument("--diagnostic_plots", action="store_true", help="Enable saving of diagnostic plots")
    parser.add_argument("--add_bar", action="store_true", help="Add scale bar to output images")
    parser.add_argument("--add_diameter", action="store_true", help="Add diameter line to images")
    parser.add_argument("--add_scale_cm", action="store_true", help="Display scale cm text on the scale bar")
    parser.add_argument("--install_requirements", action="store_true", help="Install requirements before running")

    args = parser.parse_args()

    if args.install_requirements:
        install_requirements()

    # Initialise the processor
    processor = ImageProcessor(model_path=args.model_path, pixel_cm_ratio=args.PIXEL_CM_RATIO)

    # Apply model
    processed_images = processor.process_images(
        imgs_dir=args.imgs_dir,
        tabular_file=args.tabular_file,
        confidence_threshold=args.confidence_threshold,
        diagnostic=args.diagnostic,
        diagnostic_plots=args.diagnostic_plots,
        profile_style=ProfileStyle(args.profile_style),
        scale_bar_style=ScaleBarStyle(args.scale_bar_style),
        inventory_position=InventoryPosition(args.inv_position),
        add_bar=args.add_bar,
        add_scale_cm=args.add_scale_cm,
        scale_cm=args.scale_cm,
        num_segments=args.num_segments,
        add_diameter=args.add_diameter,
        median_filter=args.median_filter,
    )

    print(f"✅ Processed {len(processed_images)} image(s).")


if __name__ == "__main__":
    main()
