"""
Evaluation & Robustness Assessment Script
Measures accuracy (precision/recall), identification rates (top-1, top-5), 
discusses failure modes and mitigations
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import FaceDetector
from src.embedding import FaceEmbedder, EmbeddingDatabase
from src.matching import FaceRecognitionMatcher, MatchingEvaluator

# ============================================================================
# Configuration
# ============================================================================

BASE_PATH = Path(__file__).parent.parent
MODEL_PATH = BASE_PATH / 'models' / 'yolov8n.pt'
DB_PATH = BASE_PATH / 'data' / 'embeddings.db'
VALIDATION_PATH = BASE_PATH / 'data' / 'validation'
REPORT_PATH = BASE_PATH / 'evaluation_report.json'

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_validation_set(validation_path: Path) -> Tuple[List[np.ndarray], List[str], List[Path]]:
    """Load validation images and ground truth labels"""
    images = []
    labels = []
    paths = []
    
    for person_dir in sorted(validation_path.glob('*')):
        if person_dir.is_dir():
            person_name = person_dir.name
            
            for img_path in sorted(person_dir.glob('*.png')) + sorted(person_dir.glob('*.jpg')):
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append(img)
                    labels.append(person_name)
                    paths.append(img_path)
    
    return images, labels, paths


def create_corrupted_variants(image: np.ndarray, 
                            corruption_type: str = 'blur',
                            severity: float = 0.5) -> np.ndarray:
    """
    Create corrupted versions of images to test robustness
    
    Args:
        image: Input image
        corruption_type: 'blur', 'noise', 'occlusion', 'brightness', 'contrast'
        severity: 0-1 severity level
        
    Returns:
        Corrupted image
    """
    img_corrupted = image.copy().astype(np.float32)
    
    if corruption_type == 'blur':
        # Gaussian blur - simulates out-of-focus/low resolution
        kernel_size = int(1 + severity * 10) * 2 + 1
        img_corrupted = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    elif corruption_type == 'noise':
        # Gaussian noise - simulates low-light compression
        noise = np.random.randn(*image.shape) * (severity * 30)
        img_corrupted = image.astype(np.float32) + noise
        img_corrupted = np.clip(img_corrupted, 0, 255)
    
    elif corruption_type == 'occlusion':
        # Random occlusion (simulates partial face/mask/sunglasses)
        img_corrupted = image.copy().astype(np.float32)
        h, w = image.shape[:2]
        occlusion_height = int(h * severity * 0.6)  # Occlude top 60% of face
        img_corrupted[:occlusion_height, :] = 0  # Black out top portion
    
    elif corruption_type == 'brightness':
        # Brightness change - simulates different lighting
        factor = 1.0 + (severity * 0.8 - 0.4)  # Range: 0.6-1.8
        img_corrupted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    
    elif corruption_type == 'contrast':
        # Contrast adjustment - simulates poor lighting/shadows
        factor = 1.0 + (severity * 1.0 - 0.5)  # Range: 0.5-1.5
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        l_channel = cv2.convertScaleAbs(l_channel, alpha=factor, beta=0)
        lab[:, :, 0] = np.clip(l_channel, 0, 255).astype(np.uint8)
        img_corrupted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return np.clip(img_corrupted, 0, 255).astype(np.uint8)


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_accuracy(detector: FaceDetector,
                     embedder: FaceEmbedder,
                     matcher: FaceRecognitionMatcher,
                     images: List[np.ndarray],
                     labels: List[str]) -> Dict:
    """
    Evaluate identification accuracy on clean validation set
    
    Returns:
        - Top-1 accuracy
        - Top-5 accuracy
        - Per-identity accuracy
        - Precision, Recall, F1-score
    """
    print("\nEvaluating accuracy on clean validation set...")
    
    predictions = []
    confidences = []
    detected_faces = 0
    failed_detections = 0
    
    for img, label in zip(tqdm(images, desc="Processing"), labels):
        # Detect face
        detections = detector.detect(img)
        
        if not detections:
            failed_detections += 1
            predictions.append('Unknown')
            confidences.append(0.0)
            continue
        
        detected_faces += 1
        
        # Extract face region
        x1, y1, x2, y2 = detections[0]['bbox']
        face_img = img[max(0, y1):min(img.shape[0], y2), 
                      max(0, x1):min(img.shape[1], x2)]
        
        # Resize for embedding
        if face_img.shape[0] < 160 or face_img.shape[1] < 160:
            face_img = cv2.resize(face_img, (160, 160))
        
        # Extract embedding
        embedding = embedder.extract_embedding(face_img)
        
        # Match
        match_result = matcher.match(embedding)
        predictions.append(match_result['identity'])
        confidences.append(match_result['confidence'])
    
    # Compute accuracy metrics
    accuracy = matcher.get_identification_accuracy(
        embedder.extract_embeddings_batch([
            cv2.resize(img, (160, 160)) if img.shape[0] < 160 or img.shape[1] < 160 else img
            for img in images
        ]),
        labels
    )
    
    # Compute confusion matrix and per-class metrics
    cm, identities = MatchingEvaluator.compute_confusion_matrix(predictions, labels)
    per_class = MatchingEvaluator.compute_per_class_metrics(cm)
    
    # Average precision and recall
    precisions = [per_class[i]['precision'] for i in range(len(identities))]
    recalls = [per_class[i]['recall'] for i in range(len(identities))]
    f1_scores = [per_class[i]['f1_score'] for i in range(len(identities))]
    
    return {
        'top_1_accuracy': accuracy['top_1_accuracy'],
        'top_5_accuracy': accuracy['top_5_accuracy'],
        'top_1_correct': accuracy['top_1_correct'],
        'total_samples': accuracy['total_samples'],
        'detected_faces': detected_faces,
        'failed_detections': failed_detections,
        'detection_rate': detected_faces / len(images),
        'avg_precision': float(np.mean(precisions)),
        'avg_recall': float(np.mean(recalls)),
        'avg_f1_score': float(np.mean(f1_scores)),
        'macro_precision': float(np.mean(precisions)),
        'macro_recall': float(np.mean(recalls)),
        'macro_f1': float(np.mean(f1_scores))
    }


def evaluate_robustness(detector: FaceDetector,
                       embedder: FaceEmbedder,
                       matcher: FaceRecognitionMatcher,
                       images: List[np.ndarray],
                       labels: List[str],
                       corruption_types: List[str] = None) -> Dict:
    """
    Evaluate robustness to various corruptions
    """
    if corruption_types is None:
        corruption_types = ['blur', 'noise', 'occlusion', 'brightness', 'contrast']
    
    print("\nEvaluating robustness to image corruptions...")
    
    robustness_results = {}
    
    for corruption in corruption_types:
        print(f"\n  Testing {corruption}...")
        
        predictions = []
        confidences = []
        
        for img, label in zip(tqdm(images, desc=f"  {corruption}", leave=False), labels):
            # Create corrupted variant
            img_corrupted = create_corrupted_variants(img, corruption, severity=0.7)
            
            # Detect face
            detections = detector.detect(img_corrupted)
            
            if not detections:
                predictions.append('Unknown')
                confidences.append(0.0)
                continue
            
            # Extract face region
            x1, y1, x2, y2 = detections[0]['bbox']
            face_img = img_corrupted[max(0, y1):min(img_corrupted.shape[0], y2), 
                                   max(0, x1):min(img_corrupted.shape[1], x2)]
            
            # Resize for embedding
            if face_img.shape[0] < 160 or face_img.shape[1] < 160:
                face_img = cv2.resize(face_img, (160, 160))
            
            # Extract embedding
            embedding = embedder.extract_embedding(face_img)
            
            # Match
            match_result = matcher.match(embedding)
            predictions.append(match_result['identity'])
            confidences.append(match_result['confidence'])
        
        # Compute accuracy
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        accuracy = correct / len(labels)
        
        robustness_results[corruption] = {
            'accuracy': accuracy,
            'correct_predictions': correct,
            'total_samples': len(labels)
        }
    
    return robustness_results


def analyze_failure_modes(detector: FaceDetector,
                         embedder: FaceEmbedder,
                         matcher: FaceRecognitionMatcher,
                         images: List[np.ndarray],
                         labels: List[str]) -> Dict:
    """
    Analyze failure modes and provide recommendations
    """
    print("\nAnalyzing failure modes...")
    
    failures = {
        'detection_failures': [],
        'matching_failures': [],
        'low_confidence_matches': []
    }
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        # Detection
        detections = detector.detect(img)
        
        if not detections:
            failures['detection_failures'].append({
                'image_index': idx,
                'label': label,
                'reason': 'No face detected'
            })
            continue
        
        # Extract and match
        x1, y1, x2, y2 = detections[0]['bbox']
        face_img = img[max(0, y1):min(img.shape[0], y2), 
                      max(0, x1):min(img.shape[1], x2)]
        
        if face_img.shape[0] < 160 or face_img.shape[1] < 160:
            face_img = cv2.resize(face_img, (160, 160))
        
        embedding = embedder.extract_embedding(face_img)
        match_result = matcher.match(embedding)
        
        # Matching failure
        if match_result['identity'] != label:
            failures['matching_failures'].append({
                'image_index': idx,
                'label': label,
                'predicted': match_result['identity'],
                'confidence': float(match_result['confidence']),
                'threshold': matcher.threshold
            })
        
        # Low confidence matches
        if match_result['confidence'] < 0.7 and match_result['matched']:
            failures['low_confidence_matches'].append({
                'image_index': idx,
                'label': label,
                'predicted': match_result['identity'],
                'confidence': float(match_result['confidence'])
            })
    
    return failures


def generate_mitigations() -> Dict:
    """
    Generate mitigation strategies for common failure modes
    """
    mitigations = {
        'occlusions': {
            'description': 'Faces with masks, glasses, or partial coverage',
            'mitigations': [
                'Use landmark-based face alignment (already implemented)',
                'Train on augmented data with occlusions',
                'Use attention mechanisms to focus on visible regions',
                'Deploy ensemble methods for robustness'
            ]
        },
        'low_light': {
            'description': 'Poor lighting conditions or shadows',
            'mitigations': [
                'Apply histogram equalization preprocessing',
                'Use CLAHE (Contrast Limited Adaptive Histogram Equalization)',
                'Train on low-light augmented data',
                'Increase detection confidence threshold in low-light'
            ]
        },
        'pose_variation': {
            'description': 'Non-frontal face angles',
            'mitigations': [
                'Use 3D face alignment for pose normalization',
                'Train on multi-pose datasets',
                'Use pose-robust embeddings (e.g., ArcFace with margin)',
                'Apply synthetic pose augmentation during training'
            ]
        },
        'image_quality': {
            'description': 'Blurry, low-resolution images',
            'mitigations': [
                'Apply super-resolution preprocessing',
                'Use image quality assessment filters',
                'Train on quality-augmented data',
                'Reject very low-quality detections'
            ]
        },
        'small_faces': {
            'description': 'Faces too small in image',
            'mitigations': [
                'Use multi-scale detection (already in YOLO)',
                'Increase min_face_area threshold tuning',
                'Apply upsampling for small detections',
                'Use contextual information for small face detection'
            ]
        },
        'false_positives': {
            'description': 'Detecting non-face objects as faces',
            'mitigations': [
                'Increase NMS threshold (already tuned to 0.5)',
                'Use face landmark validation',
                'Apply face quality filters (already implemented)',
                'Use multiple detection models for consensus'
            ]
        }
    }
    
    return mitigations


# ============================================================================
# Main Evaluation Script
# ============================================================================

def main():
    """Main evaluation workflow"""
    print("=" * 80)
    print("Face Recognition System - Evaluation & Robustness Assessment")
    print("=" * 80)
    
    # Load data
    images, labels, paths = load_validation_set(VALIDATION_PATH)
    print(f"\nLoaded {len(images)} validation images from {len(set(labels))} identities")
    
    # Initialize components
    detector = FaceDetector(
        model_path=str(MODEL_PATH),
        confidence=0.5,
        nms_threshold=0.5,
        min_face_area=400,
        apply_postprocessing=True
    )
    
    embedder = FaceEmbedder(model_name='vggface2')
    
    embedding_db = EmbeddingDatabase(str(DB_PATH))
    all_embeddings = embedding_db.get_all_embeddings()
    
    if not all_embeddings:
        print("Error: No embeddings in database!")
        return
    
    matcher = FaceRecognitionMatcher(
        embeddings_dict=all_embeddings,
        threshold=0.6,
        top_k=5
    )
    
    # Run evaluations
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': {
            'total_images': len(images),
            'total_identities': len(set(labels)),
            'samples_per_identity': len(images) // len(set(labels))
        }
    }
    
    # Clean accuracy
    results['clean_accuracy'] = evaluate_accuracy(detector, embedder, matcher, images, labels)
    
    # Robustness
    results['robustness'] = evaluate_robustness(
        detector, embedder, matcher, images, labels,
        corruption_types=['blur', 'noise', 'occlusion', 'brightness', 'contrast']
    )
    
    # Failure modes
    results['failure_analysis'] = analyze_failure_modes(detector, embedder, matcher, images, labels)
    
    # Mitigations
    results['mitigations'] = generate_mitigations()
    
    # Save report
    with open(REPORT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Evaluation complete! Report saved to: {REPORT_PATH}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\n📊 Clean Accuracy:")
    print(f"  Top-1: {results['clean_accuracy']['top_1_accuracy']:.2%}")
    print(f"  Top-5: {results['clean_accuracy']['top_5_accuracy']:.2%}")
    print(f"  Detection Rate: {results['clean_accuracy']['detection_rate']:.2%}")
    print(f"  Avg Precision: {results['clean_accuracy']['avg_precision']:.2%}")
    print(f"  Avg Recall: {results['clean_accuracy']['avg_recall']:.2%}")
    
    print(f"\n🔧 Robustness (at 0.7 severity):")
    for corruption, stats in results['robustness'].items():
        print(f"  {corruption.capitalize()}: {stats['accuracy']:.2%}")


if __name__ == "__main__":
    main()
