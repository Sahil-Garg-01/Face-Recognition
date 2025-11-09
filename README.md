# Face Recognition Service

## 📋 Assignment Tasks (7) - ALL COMPLETE 

| Task | Name | Status | Deliverables |
|------|------|--------|--------------|
| 1 | Data Preparation |  COMPLETE | `notebooks/1_data_prep.ipynb` |
| 2 | Face Detection |  COMPLETE | `src/detection.py`, `notebooks/2_face_detection.ipynb` |
| 3 | Feature Extraction |  COMPLETE | `src/embedding.py`, `notebooks/3_feature_extractor.ipynb` |
| 4 | Matching Pipeline |  COMPLETE | `src/matching.py`, `notebooks/4_matching_pipeline.ipynb` |
| 5 | Microservice |  COMPLETE | `src/api.py`, `Dockerfile`, `docker-compose.yml` |
| 6 | Optimization |  COMPLETE | `src/optimization.py`, benchmarks |
| 7 | Evaluation |  COMPLETE | `report.md`, `face_recognition_demo.gif` |

---

## 📁 Project Structure

```
assignment/
├── data/                        # Dataset folder (gallery/validation)
├── models/                      # Model weights
│   ├── yolov8n.pt              # YOLO v8 (6.2 MB)
│   └── yolov8n-face.pt         # YOLO face detection
├── src/                         # Source code (6 modules)
│   ├── detection.py            # Face detector
│   ├── embedding.py            # Feature extractor + DB
│   ├── matching.py             # Cosine similarity matcher
│   ├── database.py             # Database operations
│   ├── api.py                  # FastAPI microservice
│   └── optimization.py         # Model optimization
├── notebooks/                   # Jupyter notebooks (4 tasks)
│   ├── 1_data_prep.ipynb
│   ├── 2_face_detection.ipynb
│   ├── 3_feature_extractor.ipynb
│   └── 4_matching_pipeline.ipynb
├── docker/                      # Docker setup
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── API_DOCUMENTATION.md
├── scripts/                     # Utilities
├── requirements.txt
├── .gitignore
├── README.md
├── report.md                    # Technical report
└── face_recognition_demo.gif    # Demo video
```

---

## 🚀 Quick Start

### Setup
```powershell
# Clone and setup
git clone https://github.com/Sahil-Garg-01/Face-Recognition.git
cd Face-Recognition

# Virtual environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run API

**Option 1: Direct (Development)**
```powershell
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

**Option 2: Docker**
```powershell
docker-compose up -d
```

Access Swagger UI: `http://localhost:8000/docs`

---

## 🎯 API Endpoints

- `POST /detect` - Face detection with bounding boxes
- `POST /recognize` - Identity recognition against gallery
- `POST /add_identity` - Add new person to database
- `GET /list_identities` - List all registered identities

---

## 📊 Results

**Accuracy:**
- Detection: 100% precision, 100% recall
- Recognition: 100% Top-1 accuracy, 100% Top-5 accuracy

**Performance (CPU - Intel i7):**
- Detection: 65ms (15.4 FPS)
- Embedding: 120ms (8.3 FPS)
- Matching: 0.8ms (1250 FPS)
- **End-to-End: 186ms (5.4 FPS)**

**Resource:**
- Models: 150MB
- Database: 0.41MB (94 embeddings)
- Memory: 400MB runtime

---

## 📝 Documentation

- **`report.md`** - Technical report with methodology, benchmarks, limitations
- **`docker/API_DOCUMENTATION.md`** - API reference
- **Notebooks** - Step-by-step implementation

---

## 🎬 Demo

Visual demo: `face_recognition_demo.gif` (showing recognition on CCTV samples)

---

## Author
Sahil - AI Engineer

**Status**: Production Ready 
