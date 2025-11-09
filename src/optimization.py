"""
Optimization Module
ONNX conversion, benchmarking, and post-processing optimizations
"""

import numpy as np
import torch
import time
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Optimize models for CPU inference"""
    
    @staticmethod
    def convert_to_onnx(model: torch.nn.Module, 
                       input_shape: Tuple, 
                       output_path: str,
                       model_name: str = "model"):
        """
        Convert PyTorch model to ONNX format
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape (e.g., (1, 3, 112, 112))
            output_path: Path to save ONNX model
            model_name: Name for ONNX model
            
        Returns:
            Path to saved ONNX model
        """
        try:
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(input_shape)
            
            # Export to ONNX
            onnx_path = Path(output_path) / f"{model_name}.onnx"
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                opset_version=12,
                do_constant_folding=True,
                verbose=False
            )
            
            logger.info(f"Model converted to ONNX: {onnx_path}")
            return str(onnx_path)
        
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
            raise
    
    @staticmethod
    def quantize_model(model: torch.nn.Module) -> torch.nn.Module:
        """
        Quantize model for faster inference (int8)
        
        Args:
            model: PyTorch model
            
        Returns:
            Quantized model
        """
        try:
            model.eval()
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            logger.info("Model quantized to int8")
            return quantized_model
        
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model


class PerformanceBenchmark:
    """Benchmark inference performance"""
    
    def __init__(self, warmup_iterations: int = 10, 
                 test_iterations: int = 100):
        """
        Initialize benchmark
        
        Args:
            warmup_iterations: Number of warmup runs
            test_iterations: Number of test runs for measurement
        """
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations
    
    def benchmark_detection(self, detector, images: List[np.ndarray]) -> Dict:
        """
        Benchmark face detection
        
        Args:
            detector: FaceDetector instance
            images: List of test images
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Benchmarking detection on {len(images)} images...")
        
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = detector.detect(images[0])
        
        # Benchmark
        times = []
        for img in images:
            start = time.time()
            _ = detector.detect(img)
            times.append(time.time() - start)
        
        times = np.array(times)
        
        return {
            'num_images': len(images),
            'mean_latency_ms': np.mean(times) * 1000,
            'std_latency_ms': np.std(times) * 1000,
            'min_latency_ms': np.min(times) * 1000,
            'max_latency_ms': np.max(times) * 1000,
            'fps': 1.0 / np.mean(times)
        }
    
    def benchmark_embedding(self, embedder, images: List[np.ndarray]) -> Dict:
        """
        Benchmark face embedding extraction
        
        Args:
            embedder: FaceEmbedder instance
            images: List of test images
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Benchmarking embedding on {len(images)} images...")
        
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = embedder.extract_embedding(images[0])
        
        # Benchmark
        times = []
        for img in images:
            start = time.time()
            _ = embedder.extract_embedding(img)
            times.append(time.time() - start)
        
        times = np.array(times)
        
        return {
            'num_images': len(images),
            'mean_latency_ms': np.mean(times) * 1000,
            'std_latency_ms': np.std(times) * 1000,
            'min_latency_ms': np.min(times) * 1000,
            'max_latency_ms': np.max(times) * 1000,
            'fps': 1.0 / np.mean(times)
        }
    
    def benchmark_matching(self, matcher, embeddings: np.ndarray) -> Dict:
        """
        Benchmark face matching
        
        Args:
            matcher: FaceRecognitionMatcher instance
            embeddings: Query embeddings (N, 512)
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Benchmarking matching on {len(embeddings)} embeddings...")
        
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = matcher.match(embeddings[0])
        
        # Benchmark
        times = []
        for embedding in embeddings:
            start = time.time()
            _ = matcher.match(embedding)
            times.append(time.time() - start)
        
        times = np.array(times)
        
        return {
            'num_queries': len(embeddings),
            'gallery_size': matcher.num_gallery,
            'mean_latency_ms': np.mean(times) * 1000,
            'std_latency_ms': np.std(times) * 1000,
            'min_latency_ms': np.min(times) * 1000,
            'max_latency_ms': np.max(times) * 1000,
            'fps': 1.0 / np.mean(times)
        }
    
    def benchmark_end_to_end(self, detector, embedder, matcher, 
                            images: List[np.ndarray]) -> Dict:
        """
        Benchmark complete pipeline
        
        Args:
            detector: FaceDetector instance
            embedder: FaceEmbedder instance
            matcher: FaceRecognitionMatcher instance
            images: List of test images
            
        Returns:
            Dictionary with end-to-end performance metrics
        """
        logger.info(f"Benchmarking end-to-end on {len(images)} images...")
        
        detection_times = []
        embedding_times = []
        matching_times = []
        total_times = []
        
        for img in images:
            # Full pipeline
            start_total = time.time()
            
            # Detection
            start_det = time.time()
            detections = detector.detect(img)
            detection_times.append(time.time() - start_det)
            
            # Embedding & Matching
            if detections:
                start_emb = time.time()
                x1, y1, x2, y2 = detections[0]['bbox']
                face_img = img[max(0, y1):min(img.shape[0], y2), 
                              max(0, x1):min(img.shape[1], x2)]
                embedding = embedder.extract_embedding(face_img)
                embedding_times.append(time.time() - start_emb)
                
                start_match = time.time()
                _ = matcher.match(embedding)
                matching_times.append(time.time() - start_match)
            
            total_times.append(time.time() - start_total)
        
        detection_times = np.array(detection_times)
        embedding_times = np.array(embedding_times)
        matching_times = np.array(matching_times)
        total_times = np.array(total_times)
        
        return {
            'num_images': len(images),
            'detection': {
                'mean_ms': np.mean(detection_times) * 1000,
                'std_ms': np.std(detection_times) * 1000
            },
            'embedding': {
                'mean_ms': np.mean(embedding_times) * 1000,
                'std_ms': np.std(embedding_times) * 1000
            },
            'matching': {
                'mean_ms': np.mean(matching_times) * 1000,
                'std_ms': np.std(matching_times) * 1000
            },
            'total': {
                'mean_ms': np.mean(total_times) * 1000,
                'std_ms': np.std(total_times) * 1000,
                'fps': 1.0 / np.mean(total_times)
            }
        }


class PostProcessing:
    """Post-processing techniques to improve results"""
    
    @staticmethod
    def apply_nms(detections: List[Dict], nms_threshold: float = 0.5) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        
        Args:
            detections: List of detection dictionaries
            nms_threshold: IoU threshold for NMS
            
        Returns:
            Filtered detections
        """
        if not detections:
            return []
        
        # Get boxes and scores
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # Apply NMS
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # Calculate IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / union
            
            # Keep detections with IoU < threshold
            order = order[np.where(iou < nms_threshold)[0] + 1]
        
        return [detections[i] for i in keep]
    
    @staticmethod
    def filter_by_face_quality(detections: List[Dict], 
                              min_confidence: float = 0.7,
                              min_area: int = 20*20) -> List[Dict]:
        """
        Filter detections by quality metrics
        
        Args:
            detections: List of detection dictionaries
            min_confidence: Minimum confidence threshold
            min_area: Minimum face area (pixels)
            
        Returns:
            Filtered detections
        """
        filtered = []
        
        for det in detections:
            # Check confidence
            if det['confidence'] < min_confidence:
                continue
            
            # Check area
            if det['area'] < min_area:
                continue
            
            filtered.append(det)
        
        return filtered
    
    @staticmethod
    def smooth_detections(detections_list: List[List[Dict]], 
                         temporal_threshold: float = 0.3) -> List[List[Dict]]:
        """
        Smooth detections across frames (for video)
        
        Args:
            detections_list: List of detection lists per frame
            temporal_threshold: Max distance to link detections
            
        Returns:
            Smoothed detections
        """
        # This is a simple implementation for temporal smoothing
        # Could be enhanced with more sophisticated tracking
        smoothed = []
        
        for detections in detections_list:
            filtered = PostProcessing.filter_by_face_quality(detections)
            filtered = PostProcessing.apply_nms(filtered)
            smoothed.append(filtered)
        
        return smoothed


class CPUOptimization:
    """CPU-specific optimizations"""
    
    @staticmethod
    def get_device_info() -> Dict:
        """Get CPU and system information"""
        import psutil
        import platform
        
        info = {
            'os': platform.system(),
            'python': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        }
        
        return info
    
    @staticmethod
    def optimize_torch_for_cpu():
        """Optimize PyTorch for CPU inference"""
        # Use MKL when available
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        
        logger.info("PyTorch optimized for CPU")
    
    @staticmethod
    def profile_memory_usage(model: torch.nn.Module, 
                           input_shape: Tuple) -> Dict:
        """Profile memory usage during inference"""
        import tracemalloc
        
        tracemalloc.start()
        
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'current_mb': current / (1024**2),
            'peak_mb': peak / (1024**2)
        }
