"""
FastAPI Microservice for Face Recognition
REST endpoints for face detection and recognition
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List
import logging
import time

from fastapi import FastAPI, File, UploadFile, HTTPException, Query

# Import modules
from src.detection import FaceDetector
from src.embedding import FaceEmbedder, EmbeddingDatabase
from src.matching import FaceRecognitionMatcher
from src.optimization import PerformanceBenchmark

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Initialize Components
# ============================================================================

# Get base path
BASE_PATH = Path(__file__).parent.parent

# Paths
MODEL_PATH = BASE_PATH / 'models' / 'yolov8n.pt'
DB_PATH = BASE_PATH / 'data' / 'embeddings.db'

# Initialize components
try:
    detector = FaceDetector(
        model_path=str(MODEL_PATH), 
        confidence=0.5,
        nms_threshold=0.5,
        min_face_area=400,
        apply_postprocessing=True
    )
    logger.info("✓ Face detector initialized with post-processing")
except Exception as e:
    logger.error(f"Error initializing detector: {e}")
    detector = None

try:
    embedder = FaceEmbedder(model_name='vggface2')
    logger.info("✓ Face embedder initialized")
except Exception as e:
    logger.error(f"Error initializing embedder: {e}")
    embedder = None

try:
    embedding_db = EmbeddingDatabase(str(DB_PATH))
    logger.info("✓ Embedding database initialized")
except Exception as e:
    logger.error(f"Error initializing embedding database: {e}")
    embedding_db = None

# Initialize matcher
try:
    all_embeddings = embedding_db.get_all_embeddings()
    if all_embeddings:
        matcher = FaceRecognitionMatcher(
            embeddings_dict=all_embeddings,
            threshold=0.6,
            top_k=5
        )
        logger.info(f"✓ Face matcher initialized with {len(all_embeddings)} identities")
    else:
        matcher = None
        logger.warning("No embeddings found in database")
except Exception as e:
    logger.error(f"Error initializing matcher: {e}")
    matcher = None

# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="Face Recognition Service",
    description="API for face detection and recognition",
    version="1.0.0"
)

# ============================================================================
# Face Detection Endpoint - REQUIRED
# ============================================================================

@app.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    """
    Detect faces in uploaded image
    
    Args:
        file: Image file (JPG, PNG)
        
    Returns:
        List of detected faces with bounding boxes and confidence scores
    """
    if not detector:
        raise HTTPException(status_code=503, detail="Face detector not available")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect faces
        detections = detector.detect(image)
        
        # Format results
        results = {
            "num_detections": len(detections),
            "image_size": list(image.shape[:2]),
            "detections": [
                {
                    "bbox": det['bbox'],
                    "confidence": det['confidence'],
                    "area": det['area']
                }
                for det in detections
            ]
        }
        
        return results
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Face Recognition Endpoint
# ============================================================================

@app.post("/recognize", tags=["Recognition"])
async def recognize_face(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(0.6, ge=0.0, le=1.0)
):
    """
    Recognize identity from uploaded image
    
    Args:
        file: Image file containing face
        threshold: Confidence threshold for matching (0-1)
        
    Returns:
        Recognized identity with confidence and top-K matches
    """
    if not detector or not embedder or not matcher:
        raise HTTPException(status_code=503, detail="Recognition pipeline not available")
    
    try:
        # Set threshold
        matcher.set_threshold(threshold)
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect faces
        detections = detector.detect(image)
        
        if not detections:
            return {
                "status": "no_faces_detected",
                "num_faces": 0,
                "recognitions": []
            }
        
        # Extract embeddings for all detected faces
        recognitions = []
        
        for det_idx, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            
            # Extract face region
            face_image = image[max(0, y1):min(image.shape[0], y2), 
                              max(0, x1):min(image.shape[1], x2)]
            
            if face_image.size == 0:
                continue
            
            try:
                # Extract embedding with optional landmarks
                landmarks = detection.get('landmarks')
                embedding = embedder.extract_embedding(face_image, landmarks)
                
                # Match against gallery
                match_result = matcher.match(embedding)
                
                recognitions.append({
                    "face_index": det_idx,
                    "bbox": detection['bbox'],
                    "confidence": float(detection['confidence']),
                    "identity": match_result['identity'],
                    "recognition_confidence": match_result['confidence'],
                    "matched": match_result['matched'],
                    "top_k_matches": match_result['top_k_matches']
                })
            
            except Exception as e:
                logger.warning(f"Error recognizing face {det_idx}: {e}")
                continue
        
        return {
            "status": "success",
            "num_faces": len(detections),
            "recognized_faces": len(recognitions),
            "recognitions": recognitions
        }
    
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Identity Management Endpoints
# ============================================================================

@app.post("/add_identity", tags=["Identity Management"])
async def add_identity(
    identity_name: str = Query(..., min_length=1),
    file: UploadFile = File(...)
):
    """
    Add new identity to gallery
    
    Args:
        identity_name: Name of person
        file: Face image for embedding extraction
        
    Returns:
        Confirmation with embedding ID
    """
    if not embedder or not embedding_db:
        raise HTTPException(status_code=503, detail="Service not available")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Extract embedding
        embedding = embedder.extract_embedding(image)
        
        # Create unique image path with timestamp to avoid duplicates
        import time
        timestamp = int(time.time() * 1000)
        unique_image_path = f"{file.filename or 'uploaded'}_{timestamp}"
        
        # Add to database
        embedding_id = embedding_db.add_embedding(
            identity_name=identity_name,
            image_path=unique_image_path,
            embedding=embedding
        )
        
        return {
            "status": "success",
            "identity_name": identity_name,
            "embedding_id": embedding_id
        }
    
    except Exception as e:
        logger.error(f"Add identity error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_identities", tags=["Identity Management"])
async def list_identities(limit: Optional[int] = Query(None, ge=1)):
    """
    List all registered identities
    
    Args:
        limit: Maximum number of identities to return
        
    Returns:
        List of identities with metadata
    """
    if not embedding_db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        identities = embedding_db.get_identities()
        if limit:
            identities = identities[:limit]
        return {
            "status": "success",
            "count": len(identities),
            "identities": identities
        }
    
    except Exception as e:
        logger.error(f"List identities error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
