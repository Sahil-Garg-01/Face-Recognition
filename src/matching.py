"""
Face Matching Pipeline Module
Implements cosine similarity matching with configurable threshold and top-K
"""

import numpy as np
from typing import List, Tuple, Dict
import logging
import time

logger = logging.getLogger(__name__)


class FaceRecognitionMatcher:
    """Face recognition using cosine similarity matching"""
    
    def __init__(self, embeddings_dict: Dict[str, np.ndarray], 
                 threshold: float = 0.6, top_k: int = 5):
        """
        Initialize matcher with gallery embeddings
        
        Args:
            embeddings_dict: Dictionary {identity_name: embeddings_array}
            threshold: Similarity threshold for match (0-1)
            top_k: Return top K matches
        """
        self.embeddings_dict = embeddings_dict
        self.threshold = threshold
        self.top_k = top_k
        
        # Build gallery
        self.gallery_embeddings = []
        self.gallery_identities = []
        self.gallery_indices = []  # Map embedding index to identity
        
        idx = 0
        for identity_name, embeddings in embeddings_dict.items():
            for embedding in embeddings:
                self.gallery_embeddings.append(embedding)
                self.gallery_identities.append(identity_name)
                self.gallery_indices.append(idx)
                idx += 1
        
        self.gallery_embeddings = np.array(self.gallery_embeddings)
        self.num_gallery = len(self.gallery_embeddings)
        
        logger.info(f"Matcher initialized with {len(embeddings_dict)} identities and {self.num_gallery} embeddings")
    
    def compute_cosine_similarity(self, query_embedding: np.ndarray, 
                                 gallery_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and gallery embeddings
        
        Args:
            query_embedding: Single embedding (512,)
            gallery_embeddings: Gallery embeddings (N, 512)
            
        Returns:
            Similarity scores (N,)
        """
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Normalize gallery
        gallery_norms = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity
        similarities = np.dot(gallery_norms, query_norm)
        
        return similarities
    
    def match(self, query_embedding: np.ndarray) -> Dict:
        """
        Find matches for query embedding
        
        Args:
            query_embedding: Query embedding (512,)
            
        Returns:
            Dictionary with:
                - 'identity': Top-1 matched identity
                - 'confidence': Top-1 confidence score
                - 'matched': Boolean (confidence >= threshold)
                - 'top_k_matches': List of top-K matches
        """
        start_time = time.time()
        
        # Compute similarities
        similarities = self.compute_cosine_similarity(query_embedding, self.gallery_embeddings)
        
        # Get top-K
        top_k_indices = np.argsort(similarities)[::-1][:self.top_k]
        top_k_similarities = similarities[top_k_indices]
        
        # Top-1 match
        top_1_idx = top_k_indices[0]
        top_1_identity = self.gallery_identities[top_1_idx]
        top_1_confidence = float(top_k_similarities[0])
        
        # Check if matched (above threshold)
        is_matched = top_1_confidence >= self.threshold
        
        # Build top-K results
        top_k_matches = []
        for i, idx in enumerate(top_k_indices):
            top_k_matches.append({
                'rank': i + 1,
                'identity': self.gallery_identities[idx],
                'confidence': float(top_k_similarities[i])
            })
        
        elapsed_time = time.time() - start_time
        
        result = {
            'identity': top_1_identity if is_matched else 'Unknown',
            'confidence': top_1_confidence,
            'matched': is_matched,
            'threshold': self.threshold,
            'top_k_matches': top_k_matches,
            'inference_time_ms': elapsed_time * 1000
        }
        
        return result
    
    def match_batch(self, query_embeddings: np.ndarray) -> List[Dict]:
        """
        Match multiple query embeddings
        
        Args:
            query_embeddings: Query embeddings (N, 512)
            
        Returns:
            List of match results
        """
        results = []
        for embedding in query_embeddings:
            result = self.match(embedding)
            results.append(result)
        
        return results
    
    def get_identification_accuracy(self, query_embeddings: np.ndarray, 
                                    ground_truth_identities: List[str]) -> Dict:
        """
        Compute identification accuracy on test set
        
        Args:
            query_embeddings: Query embeddings (N, 512)
            ground_truth_identities: Ground truth labels
            
        Returns:
            Dictionary with top-1 and top-5 accuracy
        """
        matches = self.match_batch(query_embeddings)
        
        top_1_correct = 0
        top_5_correct = 0
        
        for match, gt_identity in zip(matches, ground_truth_identities):
            # Top-1 accuracy
            if match['identity'] == gt_identity:
                top_1_correct += 1
            
            # Top-5 accuracy
            top_5_identities = [m['identity'] for m in match['top_k_matches']]
            if gt_identity in top_5_identities:
                top_5_correct += 1
        
        n = len(query_embeddings)
        
        return {
            'top_1_accuracy': top_1_correct / n if n > 0 else 0,
            'top_5_accuracy': top_5_correct / n if n > 0 else 0,
            'top_1_correct': top_1_correct,
            'top_5_correct': top_5_correct,
            'total_samples': n
        }
    
    def get_per_identity_accuracy(self, query_embeddings: np.ndarray,
                                 ground_truth_identities: List[str]) -> Dict:
        """
        Get per-identity accuracy breakdown
        
        Args:
            query_embeddings: Query embeddings
            ground_truth_identities: Ground truth labels
            
        Returns:
            Dictionary with per-identity accuracy
        """
        matches = self.match_batch(query_embeddings)
        
        identity_stats = {}
        
        for match, gt_identity in zip(matches, ground_truth_identities):
            if gt_identity not in identity_stats:
                identity_stats[gt_identity] = {
                    'total': 0,
                    'top_1_correct': 0,
                    'top_5_correct': 0
                }
            
            identity_stats[gt_identity]['total'] += 1
            
            # Top-1
            if match['identity'] == gt_identity:
                identity_stats[gt_identity]['top_1_correct'] += 1
            
            # Top-5
            top_5_identities = [m['identity'] for m in match['top_k_matches']]
            if gt_identity in top_5_identities:
                identity_stats[gt_identity]['top_5_correct'] += 1
        
        # Calculate accuracies
        for identity in identity_stats:
            total = identity_stats[identity]['total']
            identity_stats[identity]['top_1_accuracy'] = identity_stats[identity]['top_1_correct'] / total
            identity_stats[identity]['top_5_accuracy'] = identity_stats[identity]['top_5_correct'] / total
        
        return identity_stats
    
    def set_threshold(self, threshold: float):
        """Update matching threshold"""
        self.threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Threshold updated to {self.threshold}")
    
    def set_top_k(self, top_k: int):
        """Update top-K value"""
        self.top_k = max(1, min(self.num_gallery, top_k))
        logger.info(f"Top-K updated to {self.top_k}")


class MatchingEvaluator:
    """Evaluate matching performance"""
    
    @staticmethod
    def compute_confusion_matrix(predictions: List[str], 
                                ground_truth: List[str]) -> np.ndarray:
        """
        Compute confusion matrix
        
        Args:
            predictions: Predicted identities
            ground_truth: Ground truth identities
            
        Returns:
            Confusion matrix
        """
        unique_identities = sorted(set(ground_truth + predictions))
        identity_to_idx = {name: idx for idx, name in enumerate(unique_identities)}
        
        n_classes = len(unique_identities)
        cm = np.zeros((n_classes, n_classes))
        
        for pred, gt in zip(predictions, ground_truth):
            pred_idx = identity_to_idx[pred]
            gt_idx = identity_to_idx[gt]
            cm[gt_idx, pred_idx] += 1
        
        return cm, unique_identities
    
    @staticmethod
    def compute_per_class_metrics(cm: np.ndarray) -> Dict:
        """
        Compute per-class precision, recall, F1
        
        Args:
            cm: Confusion matrix
            
        Returns:
            Dictionary with per-class metrics
        """
        metrics = {}
        
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[i] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        return metrics
    
    @staticmethod
    def threshold_analysis(confidences: List[float], 
                          ground_truth: List[str],
                          predictions: List[str],
                          thresholds: List[float] = None) -> Dict:
        """
        Analyze matching performance at different thresholds
        
        Args:
            confidences: Top-1 confidence scores
            ground_truth: Ground truth identities
            predictions: Predicted identities
            thresholds: List of thresholds to test (0-1)
            
        Returns:
            Dictionary with metrics at each threshold
        """
        if thresholds is None:
            thresholds = np.linspace(0.3, 1.0, 15)
        
        results = []
        
        for threshold in thresholds:
            accepted = 0
            correct = 0
            
            for conf, gt, pred in zip(confidences, ground_truth, predictions):
                if conf >= threshold:
                    accepted += 1
                    if pred == gt:
                        correct += 1
            
            accuracy = correct / accepted if accepted > 0 else 0
            coverage = accepted / len(confidences)
            
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'coverage': coverage,
                'accepted': accepted,
                'correct': correct
            })
        
        return results
