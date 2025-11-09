"""
Face Recognition Visual Demo
Shows actual face recognition with bounding boxes and identity labels on CCTV sample frames
"""

import cv2
import numpy as np
import glob
import os
from src.detection import FaceDetector
from src.embedding import FaceEmbedder, EmbeddingDatabase
from src.matching import FaceRecognitionMatcher

def draw_results(image, detections, recognitions):
    """
    Draw bounding boxes and identity labels on image
    
    Args:
        image: Input image (BGR)
        detections: List of detected faces with bbox and confidence
        recognitions: List of recognition results with identity and confidence
    
    Returns:
        Image with drawn results
    """
    result = image.copy()
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        det_conf = det['confidence']
        
        # Draw green bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Get recognition result
        if i < len(recognitions) and recognitions[i]['matched']:
            identity = recognitions[i]['identity']
            rec_conf = recognitions[i]['confidence']
            label = f"{identity} ({rec_conf:.3f})"
            label_color = (0, 255, 0)  # Green for matched
        else:
            label = "Unknown"
            label_color = (0, 0, 255)  # Red for unknown
        
        # Draw label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1
        text_y = y1 - 10
        
        cv2.rectangle(result, 
                     (text_x, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     label_color, -1)
        
        # Draw label text
        cv2.putText(result, label, (text_x + 2, text_y - 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return result

def process_single_image(image_path, detector, embedder, matcher, display=True):
    """
    Process a single image: detect faces, recognize, draw results
    
    Args:
        image_path: Path to input image
        detector: FaceDetector instance
        embedder: FaceEmbedder instance
        matcher: FaceRecognitionMatcher instance
        display: Whether to display the result
    
    Returns:
        Result image with drawn annotations
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Detect faces
    detections = detector.detect(image)
    print(f"Detected {len(detections)} face(s)")
    
    # Extract embeddings and recognize
    recognitions = []
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        face_img = image[y1:y2, x1:x2]
        
        if face_img.size > 0:
            # Extract embedding
            embedding = embedder.extract_embedding(face_img)
            
            # Recognize
            match = matcher.match(embedding)
            recognitions.append(match)
            
            # Print result
            if match['matched']:
                print(f"  Face {idx+1}: {match['identity']} (confidence: {match['confidence']:.3f})")
            else:
                print(f"  Face {idx+1}: Unknown")
    
    # Draw results
    result_image = draw_results(image, detections, recognitions)
    
    # Display result
    if display:
        cv2.imshow("Face Recognition Demo", result_image)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result_image

def main():
    """Main demo function"""
    print("\n" + "="*60)
    print("Face Recognition Visual Demo")
    print("="*60)
    
    # Initialize components
    print("\nInitializing detector, embedder, and matcher...")
    detector = FaceDetector()
    embedder = FaceEmbedder()
    db = EmbeddingDatabase("embeddings.db")
    
    # Get all embeddings from database
    embeddings_dict = db.get_all_embeddings()
    
    if not embeddings_dict:
        print("Error: No embeddings found in database!")
        print("Please run notebooks/3_feature_extractor.ipynb first to generate embeddings.")
        return
    
    matcher = FaceRecognitionMatcher(embeddings_dict)
    print(f"✓ All components initialized")
    print(f"✓ Loaded {len(embeddings_dict)} identities from database")
    
    # Get sample images from gallery
    print("\nFinding CCTV sample images...")
    sample_images = sorted(glob.glob("data/gallery/person*/Screenshot*.png"))
    
    if not sample_images:
        sample_images = sorted(glob.glob("data/gallery/person*/*.jpg"))
    
    if not sample_images:
        print("Error: No sample images found in data/gallery/")
        return
    
    # Limit to 5 samples for demo
    sample_images = sample_images[:5]
    print(f"Found {len(sample_images)} sample images for demo")
    
    # Process images
    results = []
    for img_path in sample_images:
        result_img = process_single_image(img_path, detector, embedder, matcher, display=True)
        if result_img is not None:
            results.append(result_img)
    
    # Save demo results
    print(f"\n{'='*60}")
    print("Saving demo results...")
    
    output_dir = "demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (img_path, result_img) in enumerate(zip(sample_images[:len(results)], results)):
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"result_{idx+1}_{filename}")
        cv2.imwrite(output_path, result_img)
        print(f"✓ Saved: {output_path}")
    
    # Create a composite image showing all results
    if results:
        print("\nCreating composite demo image...")
        composite = create_composite(results, results_per_row=3)
        composite_path = os.path.join(output_dir, "demo_composite.jpg")
        cv2.imwrite(composite_path, composite)
        print(f"✓ Saved composite: {composite_path}")
    
    print(f"\n{'='*60}")
    print("Demo completed!")
    print(f"Results saved in '{output_dir}/' directory")
    print("Use these images to create a GIF/video demo")
    print("="*60 + "\n")

def create_composite(images, results_per_row=3):
    """
    Create a composite image from multiple results
    
    Args:
        images: List of result images
        results_per_row: Number of images per row
    
    Returns:
        Composite image
    """
    if not images:
        return None
    
    # Resize images to consistent size
    h, w = 300, 300
    resized = [cv2.resize(img, (w, h)) for img in images]
    
    # Calculate grid
    num_rows = (len(resized) + results_per_row - 1) // results_per_row
    
    # Pad with black images if needed
    total_needed = num_rows * results_per_row
    while len(resized) < total_needed:
        resized.append(np.zeros((h, w, 3), dtype=np.uint8))
    
    # Create rows
    rows = []
    for i in range(num_rows):
        row_start = i * results_per_row
        row_end = min((i + 1) * results_per_row, len(resized))
        row_images = resized[row_start:row_end]
        
        # Pad row if needed
        while len(row_images) < results_per_row:
            row_images.append(np.zeros((h, w, 3), dtype=np.uint8))
        
        row = cv2.hconcat(row_images)
        rows.append(row)
    
    # Concatenate rows
    composite = cv2.vconcat(rows)
    
    # Add title
    cv2.putText(composite, "Face Recognition Demo - CCTV Samples",
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    return composite

if __name__ == "__main__":
    main()
