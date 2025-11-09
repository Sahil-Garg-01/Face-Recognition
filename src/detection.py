"""
Face Detection Module
Implements YOLO v8 face detector
"""

import numpy as np
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """YOLO v8 based face detector"""
    
    def __init__(self, model_path: str = None, confidence: float = 0.5):
        """
        Initialize face detector
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence: Detection confidence threshold
        """
        if model_path is None:
            self.model = YOLO('yolov8n.pt')
        else:
            self.model = YOLO(model_path)
        
        self.confidence = confidence
    
    def detect(self, image: np.ndarray) -> list:
        """
        Detect faces in image
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of detections with bbox and confidence
        """
        results = self.model(image, conf=self.confidence, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'area': (x2 - x1) * (y2 - y1)
                    }
                    
                    detections.append(detection)
        
        return detections
