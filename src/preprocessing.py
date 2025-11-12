"""
Face Preprocessing Module
Implements face alignment (5-point landmarks) and normalization
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def align_face(img: np.ndarray, landmarks: np.ndarray, 
               output_size: tuple = (112, 112)) -> np.ndarray:
    """
    Align face using 5-point landmarks (eyes, nose, mouth corners)
    
    Args:
        img: Input image (BGR)
        landmarks: 5x2 array of facial landmarks [x, y] format
        output_size: Desired output size (default: 112x112)
    
    Returns:
        Aligned and cropped face image
    """
    # Standard 5-point template for 112x112 face
    template = np.array([
        [38.2946, 51.6963],   # Left eye
        [73.5318, 51.5014],   # Right eye
        [56.0252, 71.7366],   # Nose tip
        [41.5493, 92.3655],   # Left mouth corner
        [70.7299, 92.2041]    # Right mouth corner
    ], dtype=np.float32)
    
    # Ensure landmarks are float32
    landmarks = landmarks.astype(np.float32)
    
    # Estimate affine transformation
    tform = cv2.estimateAffinePartial2D(landmarks, template)[0]
    
    if tform is None:
        logger.warning("Failed to estimate affine transformation, returning resized image")
        return cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
    
    # Apply transformation
    aligned = cv2.warpAffine(img, tform, output_size, flags=cv2.INTER_LINEAR)
    
    return aligned


def normalize_face(img: np.ndarray) -> np.ndarray:
    """
    Normalize face image for embedding extraction
    Converts to RGB and scales to [-1, 1] range
    
    Args:
        img: Input image (BGR or RGB, uint8)
    
    Returns:
        Normalized image (RGB, float32, range [-1, 1])
    """
    # Convert BGR to RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    
    # Convert to float and normalize to [-1, 1]
    img_normalized = img_rgb.astype(np.float32) / 127.5 - 1.0
    
    return img_normalized
