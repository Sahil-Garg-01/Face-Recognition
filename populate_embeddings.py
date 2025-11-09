"""
Quick script to populate embeddings database from gallery images
"""

import glob
import os
from pathlib import Path
from src.detection import FaceDetector
from src.embedding import FaceEmbedder, EmbeddingDatabase
import cv2

def populate_database():
    """Populate embeddings database from gallery_aligned folder"""
    
    print("="*60)
    print("Populating Embeddings Database")
    print("="*60)
    
    # Initialize components
    print("\nInitializing components...")
    detector = FaceDetector()
    embedder = FaceEmbedder()
    db = EmbeddingDatabase("embeddings.db")
    
    # Clear existing data
    print("Clearing existing data...")
    db.clear_database()
    
    # Get all images from gallery folder
    gallery_path = Path("data/gallery")
    if not gallery_path.exists():
        print(f"Error: {gallery_path} not found!")
        return
    
    # Find all identity folders
    identity_folders = sorted([d for d in gallery_path.iterdir() if d.is_dir()])
    print(f"Found {len(identity_folders)} identities")
    
    total_embeddings = 0
    
    # Process each identity
    for identity_folder in identity_folders:
        identity_name = identity_folder.name
        
        # Get all images in this folder (PNG and JPG)
        image_files = sorted(glob.glob(str(identity_folder / "*.png")))
        image_files += sorted(glob.glob(str(identity_folder / "*.jpg")))
        
        if not image_files:
            print(f"  {identity_name}: No images found")
            continue
        
        print(f"\n  Processing {identity_name}... ({len(image_files)} images)")
        
        # Process each image
        for image_path in image_files:
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Detect face
                detections = detector.detect(image)
                if not detections:
                    continue
                
                # Get first detected face
                det = detections[0]
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                face_img = image[y1:y2, x1:x2]
                
                if face_img.size == 0:
                    continue
                
                # Extract embedding
                embedding = embedder.extract_embedding(face_img)
                
                # Add to database
                db.add_embedding(identity_name, image_path, embedding)
                total_embeddings += 1
                print(f"    ✓ {os.path.basename(image_path)}")
                
            except Exception as e:
                print(f"    ✗ Error processing {image_path}: {e}")
        
        print(f"    Total: {len([f for f in image_files if f])} processed")
    
    # Print statistics
    stats = db.get_db_stats()
    print(f"\n{'='*60}")
    print(f"Database Statistics:")
    print(f"  Total Identities: {stats['total_identities']}")
    print(f"  Total Embeddings: {stats['total_embeddings']}")
    print(f"  Database Size: {stats['db_size_mb']:.2f} MB")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    populate_database()
