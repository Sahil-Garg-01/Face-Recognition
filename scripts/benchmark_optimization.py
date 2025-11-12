"""
Benchmark and Optimization Script
Measures CPU latency (ms) and throughput (FPS) for face recognition pipeline
Tests post-processing effectiveness (NMS, quality filtering)
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import FaceDetector
from src.embedding import FaceEmbedder
from src.matching import FaceRecognitionMatcher
from src.embedding import EmbeddingDatabase
from src.optimization import PerformanceBenchmark

# ============================================================================
# Configuration
# ============================================================================

BASE_PATH = Path(__file__).parent.parent
MODEL_PATH = BASE_PATH / 'models' / 'yolov8n.pt'
DB_PATH = BASE_PATH / 'data' / 'embeddings.db'
VALIDATION_PATH = BASE_PATH / 'data' / 'validation'
OUTPUT_PATH = BASE_PATH / 'models' / 'onnx'
REPORT_PATH = BASE_PATH / 'optimization_report.json'

# ============================================================================
# Benchmarking Functions
# ============================================================================

def load_test_images(validation_path: Path, max_images: int = 20):
    """Load validation images for benchmarking"""
    images = []
    image_paths = []
    
    for person_dir in sorted(validation_path.glob('*')):
        if person_dir.is_dir():
            # Support both .jpg and .png files
            for img_path in list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png')):
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append(img)
                    image_paths.append(img_path)
                    
                    if len(images) >= max_images:
                        break
        
        if len(images) >= max_images:
            break
    
    return images, image_paths


def benchmark_detection(detector, images, iterations=10):
    """Benchmark face detection performance"""
    benchmark = PerformanceBenchmark(warmup_iterations=5, test_iterations=iterations)
    results = benchmark.benchmark_detection(detector, images)
    return results


def benchmark_embedding(embedder, images, iterations=10):
    """Benchmark face embedding extraction performance"""
    face_images = [cv2.resize(img, (160, 160)) for img in images]
    benchmark = PerformanceBenchmark(warmup_iterations=5, test_iterations=iterations)
    results = benchmark.benchmark_embedding(embedder, face_images)
    return results


def benchmark_end_to_end(detector, embedder, matcher, images, iterations=10):
    """Benchmark complete face recognition pipeline performance"""
    benchmark = PerformanceBenchmark(warmup_iterations=5, test_iterations=iterations)
    
    # Preprocess images: resize validation images to standard size for embedding
    processed_images = []
    for img in images:
        # Resize to at least 160x160 if smaller (FaceNet requirement)
        if img.shape[0] < 160 or img.shape[1] < 160:
            img_resized = cv2.resize(img, (160, 160), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img
        processed_images.append(img_resized)
    
    results = benchmark.benchmark_end_to_end(detector, embedder, matcher, processed_images)
    return results


def test_postprocessing_effectiveness(detector, images):
    """Test effectiveness of post-processing (NMS, quality filtering)"""
    detector.apply_postprocessing = False
    detections_without = [len(detector.detect(img)) for img in images]
    
    detector.apply_postprocessing = True
    detections_with = [len(detector.detect(img)) for img in images]
    
    avg_without = np.mean(detections_without)
    avg_with = np.mean(detections_with)
    reduction = ((avg_without - avg_with) / avg_without * 100) if avg_without > 0 else 0
    
    return {
        'avg_detections_without_postprocessing': float(avg_without),
        'avg_detections_with_postprocessing': float(avg_with),
        'false_positive_reduction_percent': float(reduction)
    }


def convert_models_to_onnx(embedder, output_path):
    """Convert FaceNet model to ONNX format for optimized inference"""
    try:
        onnx_path = embedder.convert_to_onnx(str(output_path), "facenet_embedder")
        return {
            'facenet_onnx': onnx_path,
            'status': 'success'
        }
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e)
        }


# ============================================================================
# Main Benchmarking Script
# ============================================================================

def main():
    """Main benchmarking and optimization workflow"""
    print("=" * 80)
    print("Face Recognition System - Optimization & Benchmarking")
    print("=" * 80)
    
    images, image_paths = load_test_images(VALIDATION_PATH, max_images=20)
    
    if not images:
        print("Error: No test images found!")
        return
    
    print(f"\nLoaded {len(images)} test images")
    
    # Initialize components
    detector_with_pp = FaceDetector(
        model_path=str(MODEL_PATH),
        confidence=0.5,
        nms_threshold=0.5,
        min_face_area=400,
        apply_postprocessing=True
    )
    
    detector_without_pp = FaceDetector(
        model_path=str(MODEL_PATH),
        confidence=0.5,
        apply_postprocessing=False
    )
    
    embedder = FaceEmbedder(model_name='vggface2')
    
    embedding_db = EmbeddingDatabase(str(DB_PATH))
    all_embeddings = embedding_db.get_all_embeddings()
    
    matcher = None
    if all_embeddings:
        matcher = FaceRecognitionMatcher(
            embeddings_dict=all_embeddings,
            threshold=0.6,
            top_k=5
        )
        print(f"Matcher initialized with {len(all_embeddings)} identities")
    
    print("\nRunning benchmarks...")
    
    results = {
        'test_images': len(images),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results['detection_with_postprocessing'] = benchmark_detection(
        detector_with_pp, images, iterations=len(images)
    )
    
    results['detection_without_postprocessing'] = benchmark_detection(
        detector_without_pp, images, iterations=len(images)
    )
    
    results['embedding'] = benchmark_embedding(embedder, images, iterations=len(images))
    
    if matcher:
        results['end_to_end'] = benchmark_end_to_end(
            detector_with_pp, embedder, matcher, images, iterations=len(images)
        )
    
    results['postprocessing_effectiveness'] = test_postprocessing_effectiveness(
        detector_with_pp, images
    )
    
    results['onnx_conversion'] = convert_models_to_onnx(embedder, OUTPUT_PATH)
    
    with open(REPORT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark complete. Results saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
