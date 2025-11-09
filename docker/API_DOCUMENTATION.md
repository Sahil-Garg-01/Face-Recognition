# Face Recognition Service - API Documentation

## Overview
The Face Recognition Service provides REST API endpoints for face detection, embedding extraction, and identity recognition. The service uses FastAPI with automatic Swagger documentation.

---

## Base URL
```
http://localhost:8000
```

## Automatic Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Endpoints

### 1. Detect Faces
Detect all faces in an image.

**Endpoint:**
```
POST /detect
```

**Parameters:**
- `file` (multipart/form-data): Image file (JPG, PNG)

**Response:**
```json
{
  "num_detections": 2,
  "image_size": [1080, 720],
  "detections": [
    {
      "bbox": [100, 150, 300, 400],
      "confidence": 0.95,
      "area": 45000
    },
    {
      "bbox": [400, 200, 600, 450],
      "confidence": 0.92,
      "area": 40000
    }
  ]
}
```

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -F "file=@image.jpg"
```

**Example (Python):**
```python
import requests

with open('image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/detect', files=files)
    print(response.json())
```

---

### 2. Recognize Identity
Recognize identity of person in image.

**Endpoint:**
```
POST /recognize
```

**Parameters:**
- `file` (multipart/form-data): Image file (JPG, PNG)
- `threshold` (query, optional): Confidence threshold (0.0-1.0, default: 0.6)

**Response:**
```json
{
  "status": "success",
  "num_faces": 1,
  "recognized_faces": 1,
  "recognitions": [
    {
      "face_index": 0,
      "bbox": [100, 150, 300, 400],
      "confidence": 0.95,
      "identity": "person1",
      "recognition_confidence": 0.8523,
      "matched": true,
      "top_k_matches": [
        {
          "rank": 1,
          "identity": "person1",
          "confidence": 0.8523
        },
        {
          "rank": 2,
          "identity": "person5",
          "confidence": 0.6234
        }
      ]
    }
  ]
}
```

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/recognize?threshold=0.6" \
  -H "accept: application/json" \
  -F "file=@face.jpg"
```

**Example (Python):**
```python
import requests

with open('face.jpg', 'rb') as f:
    files = {'file': f}
    params = {'threshold': 0.6}
    response = requests.post(
        'http://localhost:8000/recognize',
        files=files,
        params=params
    )
    result = response.json()
    for rec in result['recognitions']:
        print(f"Identity: {rec['identity']}, Confidence: {rec['recognition_confidence']:.4f}")
```

---

### 3. Add Identity
Add new person to gallery with their face image.

**Endpoint:**
```
POST /add_identity
```

**Parameters:**
- `identity_name` (query): Name of person
- `file` (multipart/form-data): Face image (JPG, PNG)

**Response:**
```json
{
  "status": "success",
  "identity_name": "john_doe",
  "embedding_id": 79,
  "timestamp": "2025-11-09T12:34:56.789123"
}
```

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/add_identity?identity_name=john_doe" \
  -H "accept: application/json" \
  -F "file=@john_face.jpg"
```

**Example (Python):**
```python
import requests

with open('john_face.jpg', 'rb') as f:
    files = {'file': f}
    params = {'identity_name': 'john_doe'}
    response = requests.post(
        'http://localhost:8000/add_identity',
        files=files,
        params=params
    )
    print(response.json())
```

---

### 4. List Identities
Get list of all registered identities.

**Endpoint:**
```
GET /list_identities
```

**Parameters:**
- `limit` (query, optional): Maximum number of identities to return

**Response:**
```json
{
  "status": "success",
  "count": 20,
  "identities": [
    {
      "id": 1,
      "name": "person1",
      "num_images": 4,
      "created_at": "2025-11-09T10:00:00"
    },
    {
      "id": 2,
      "name": "person2",
      "num_images": 4,
      "created_at": "2025-11-09T10:05:00"
    }
  ]
}
```

**Example (curl):**
```bash
curl -X GET "http://localhost:8000/list_identities?limit=10"
```

**Example (Python):**
```python
import requests

response = requests.get('http://localhost:8000/list_identities?limit=10')
identities = response.json()['identities']
for identity in identities:
    print(f"{identity['name']}: {identity['num_images']} images")
```

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid image file"
}
```

### 404 Not Found
```json
{
  "detail": "Identity 'unknown' not found"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Face detector not available"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Error message"
}
```

---

## Running the Service

### Using Python
```bash
# Activate virtual environment
.venv\Scripts\activate

# Run API server
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### Using Docker
```bash
# Build image
docker build -t face-recognition-service .

# Run container
docker run -p 8000:8000 face-recognition-service
```

### Using Docker Compose
```bash
docker-compose up -d
```

---

## Configuration

### Matching Threshold
Control sensitivity of face recognition:
- `threshold = 0.3`: Very lenient (many false positives)
- `threshold = 0.6`: Balanced (default)
- `threshold = 0.8`: Strict (fewer false positives)

Use the `threshold` parameter in `/recognize` endpoint to adjust.

### Models
The service uses:
- **Face Detection**: YOLO v8 (nano) - `models/yolov8n.pt`
- **Face Embedding**: FaceNet (VGGFace2) - Auto-downloaded on first use
- **Embedding Dimension**: 512-D vectors
- **Similarity Metric**: Cosine similarity

---

## Performance Notes
- **First request**: May take longer due to model loading (~10-15 seconds)
- **Subsequent requests**: Fast inference (~100-500ms per face)
- **Batch operations**: Process multiple images for better efficiency
- **GPU**: Automatically uses GPU if available, falls back to CPU

---

## Example Workflow

```python
import requests

# 1. List all identities
response = requests.get('http://localhost:8000/list_identities')
print("Total people:", response.json()['count'])

# 2. Add new person
with open('john.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/add_identity?identity_name=john',
        files={'file': f}
    )
    print("Added:", response.json())

# 3. Recognize person
with open('unknown_person.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/recognize',
        files={'file': f}
    )
    result = response.json()
    for rec in result['recognitions']:
        print(f"Identified as: {rec['identity']} ({rec['recognition_confidence']:.2%})")

# 4. Detect faces
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect',
        files={'file': f}
    )
    print("Detections:", response.json())
```

---

## Support
For issues or feature requests, refer to the project documentation or README.md.
