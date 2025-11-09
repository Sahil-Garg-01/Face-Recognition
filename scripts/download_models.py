"""
Download YOLOv8 Face Detection Model
This script downloads the pretrained YOLOv8-face model for face detection.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)

    print(f"✅ Downloaded: {destination}")

def main():
    """Download YOLO face detection model"""
    print("="*60)
    print("Downloading YOLOv8 Face Detection Model")
    print("="*60)

    # Create models directory
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    # YOLOv8-face model URL (from official repo)
    # Using YOLOv8n (nano) for faster CPU inference
    model_url = "https://github.com/derronqi/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"
    model_path = models_dir / "yolov8n-face.pt"

    if model_path.exists():
        print(f"Model already exists: {model_path}")
        print("Skipping download.")
    else:
        print(f"Downloading to: {model_path}")
        download_file(model_url, model_path)

    print("\n" + "="*60)
    print("✅ Model ready!")
    print("="*60)
    print(f"\nModel location: {model_path.absolute()}")
    print("\nNext step: Open notebooks/1_data_prep.ipynb and run all cells")

if __name__ == "__main__":
    main()