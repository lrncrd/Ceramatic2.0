import argparse
import subprocess
import sys

from utils import apply_model

def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def main():
    parser = argparse.ArgumentParser(description="Apply image processing model")
    parser.add_argument("--model_path", type=str, default="Ceramatic_model_V1.pt", help="Path to the model file")
    parser.add_argument("--imgs_dir", type=str, default="inference/imgs", help="Directory containing test images")
    parser.add_argument("--tabular_file", type=str, default="inference/metadata_example.xlsx", help="Path to the metadata file")
    parser.add_argument("--diagnostic", action="store_true", help="Enable diagnostic mode")
    parser.add_argument("--diagnostic_plots", action="store_true", help="Enable diagnostic plots")
    parser.add_argument("--add_bar", action="store_true", help="Add bar to the output")
    parser.add_argument("--install_requirements", action="store_true", help="Install required packages before running")
    

    args = parser.parse_args()

    if args.install_requirements:
        install_requirements()

    processed_images = apply_model(
        model_path=args.model_path,
        imgs_dir=args.imgs_dir,
        tabular_file=args.tabular_file,
        diagnostic=args.diagnostic,
        diagnostic_plots=args.diagnostic_plots,
        add_bar=args.add_bar
    )

    print(f"Processed {len(processed_images)} images.")

if __name__ == "__main__":
    main()