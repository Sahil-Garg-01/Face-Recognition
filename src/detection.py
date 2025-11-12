"""
Face Detection Module
Implements YOLO v8 face detector with post-processing optimizations
"""

import numpy as np
from ultralytics import YOLO
import logging
import cv2

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available - landmark detection disabled")


class FaceDetector:
    """YOLO v8 based face detector with post-processing"""
    
    def __init__(self, model_path: str = None, confidence: float = 0.5,
                 nms_threshold: float = 0.5, min_face_area: int = 400,
                 apply_postprocessing: bool = True):
        """
        Initialize face detector
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence: Detection confidence threshold
            nms_threshold: IoU threshold for NMS post-processing
            min_face_area: Minimum face area in pixels (default: 20x20=400)
            apply_postprocessing: Apply NMS and quality filtering
        """
        if model_path is None:
            self.model = YOLO('yolov8n.pt')
        else:
            self.model = YOLO(model_path)
        
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.min_face_area = min_face_area
        self.apply_postprocessing = apply_postprocessing
        
        # Initialize landmark detector if available
        if MEDIAPIPE_AVAILABLE:
            self.landmark_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=confidence
            )
        else:
            self.landmark_detector = None
        
        logger.info(f"FaceDetector initialized: confidence={confidence}, "
                   f"NMS={nms_threshold}, min_area={min_face_area}, "
                   f"postprocessing={apply_postprocessing}, "
                   f"landmarks={'enabled' if MEDIAPIPE_AVAILABLE else 'disabled'}")
    
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
                        'area': (x2 - x1) * (y2 - y1),
                        'landmarks': None
                    }
                    
                    detections.append(detection)
        
        # Extract landmarks if available
        if self.landmark_detector is not None and detections:
            detections = self._extract_landmarks(image, detections)
        
        # Apply post-processing if enabled
        if self.apply_postprocessing:
            detections = self._apply_postprocessing(detections)
        
        return detections
    
    def _apply_postprocessing(self, detections: list) -> list:
        """
        Apply post-processing to reduce false positives
        
        Args:
            detections: Raw detections from model
            
        Returns:
            Filtered detections after quality filtering and NMS
        """
        if not detections:
            return []
        
        # Step 1: Quality filtering
        detections = self._filter_by_quality(detections)
        
        # Step 2: NMS
        detections = self._apply_nms(detections)
        
        return detections
    
    def _extract_landmarks(self, image: np.ndarray, detections: list) -> list:
        """
        Extract 5-point landmarks for detected faces
        
        Args:
            image: Input image (BGR)
            detections: List of face detections
            
        Returns:
            Detections with landmarks added
        """
        try:
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.landmark_detector.process(image_rgb)
            
            if not results.detections:
                return detections
            
            h, w = image.shape[:2]
            
            # Match detections with landmarks
            for idx, detection in enumerate(detections):
                if idx < len(results.detections):
                    mp_detection = results.detections[idx]
                    
                    # Extract 5-point landmarks from MediaPipe keypoints
                    # MediaPipe provides: right_eye, left_eye, nose_tip, mouth_center, right_ear, left_ear
                    landmarks_5pt = []
                    
                    # Use available keypoints for 5-point template
                    if mp_detection.keypoints:
                        keypoints = mp_detection.keypoints
                        # Left eye (keypoint 0)
                        if len(keypoints) > 0:
                            landmarks_5pt.append([keypoints[0].x * w, keypoints[0].y * h])
                        # Right eye (keypoint 1)
                        if len(keypoints) > 1:
                            landmarks_5pt.append([keypoints[1].x * w, keypoints[1].y * h])
                        # Nose (keypoint 2)
                        if len(keypoints) > 2:
                            landmarks_5pt.append([keypoints[2].x * w, keypoints[2].y * h])
                        
                        # If we have at least 5 keypoints, use them; otherwise estimate
                        if len(keypoints) >= 5:
                            # Left mouth corner (approximate from available points)
                            landmarks_5pt.append([keypoints[3].x * w, keypoints[3].y * h])
                            # Right mouth corner
                            landmarks_5pt.append([keypoints[4].x * w, keypoints[4].y * h])
                        elif len(landmarks_5pt) >= 3:
                            # Estimate mouth corners from nose and face bbox
                            x1, y1, x2, y2 = detection['bbox']
                            mouth_y = y1 + (y2 - y1) * 0.75
                            landmarks_5pt.append([x1 + (x2 - x1) * 0.25, mouth_y])
                            landmarks_5pt.append([x1 + (x2 - x1) * 0.75, mouth_y])
                    
                    if len(landmarks_5pt) == 5:
                        detection['landmarks'] = np.array(landmarks_5pt, dtype=np.float32)
        
        except Exception as e:
            logger.debug(f"Landmark extraction error: {e}")
        
        return detections
    
    def _filter_by_quality(self, detections: list) -> list:
        """
        Filter detections by quality metrics
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Filtered detections
        """
        filtered = []
        
        for det in detections:
            # Check confidence (already filtered by model)
            if det['confidence'] < self.confidence:
                continue
            
            # Check minimum face area
            if det['area'] < self.min_face_area:
                continue
            
            filtered.append(det)
        
        return filtered
    
    def _apply_nms(self, detections: list) -> list:
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Filtered detections after NMS
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
            order = order[np.where(iou < self.nms_threshold)[0] + 1]
        
        return [detections[i] for i in keep]
