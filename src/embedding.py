"""
Face Embedding Extraction Module
Implements FaceNet feature extraction with database storage
"""

import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging

import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

logger = logging.getLogger(__name__)


class FaceEmbedder:
    """FaceNet based face embedding extractor"""
    
    def __init__(self, model_name: str = 'vggface2', device: str = None):
        """
        Initialize face embedder
        
        Args:
            model_name: Model weights ('vggface2' or 'casia-webface')
            device: 'cpu' or 'cuda' (auto-detect if None)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        logger.info(f"Using device: {device}")
        
        # Load pretrained model
        self.model = InceptionResnetV1(pretrained=model_name)
        self.model = self.model.to(device)
        self.model.eval()
        
        logger.info(f"FaceNet model loaded (pretrained on {model_name})")
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from face image
        
        Args:
            face_image: Face image (BGR or RGB, should be 112x112 or larger)
            
        Returns:
            512-D embedding vector (L2 normalized)
        """
        # Preprocess image
        if face_image.ndim == 3 and face_image.shape[2] == 3:
            # Assume BGR, convert to RGB
            if face_image.mean() > 100:  # Likely BGR
                face_image = face_image[..., ::-1]
        
        # Normalize to 0-1 range
        if face_image.max() > 1.0:
            face_image = face_image.astype(np.float32) / 255.0
        else:
            face_image = face_image.astype(np.float32)
        
        # Convert to tensor (C, H, W)
        face_tensor = torch.from_numpy(face_image)
        if face_tensor.dim() == 3 and face_tensor.shape[2] == 3:
            face_tensor = face_tensor.permute(2, 0, 1)
        
        # Add batch dimension
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(face_tensor)
        
        # Normalize to L2 unit sphere
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy()[0]
    
    def extract_embeddings_batch(self, face_images: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings from multiple face images
        
        Args:
            face_images: List of face images
            
        Returns:
            Array of embeddings (N, 512)
        """
        embeddings = []
        
        for face_image in face_images:
            embedding = self.extract_embedding(face_image)
            embeddings.append(embedding)
        
        return np.array(embeddings)


class EmbeddingDatabase:
    """SQLite database for storing embeddings"""
    
    def __init__(self, db_path: str):
        """
        Initialize embedding database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity_name TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(identity_name, image_path)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS identities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    num_images INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indices for fast queries
            conn.execute('CREATE INDEX IF NOT EXISTS idx_identity_name ON embeddings(identity_name)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_identities_name ON identities(name)')
            
            conn.commit()
        
        logger.info(f"Database initialized: {self.db_path}")
    
    def add_embedding(self, identity_name: str, image_path: str, embedding: np.ndarray) -> int:
        """
        Add embedding to database
        
        Args:
            identity_name: Person's name
            image_path: Path to source image
            embedding: 512-D embedding vector
            
        Returns:
            ID of inserted record
        """
        embedding_bytes = embedding.astype(np.float32).tobytes()
        
        with sqlite3.connect(self.db_path) as conn:
            # Insert or update identity count
            conn.execute('''
                INSERT OR IGNORE INTO identities (name, num_images)
                VALUES (?, 0)
            ''', (identity_name,))
            
            # Insert embedding
            cursor = conn.execute('''
                INSERT INTO embeddings (identity_name, image_path, embedding)
                VALUES (?, ?, ?)
            ''', (identity_name, str(image_path), embedding_bytes))
            
            # Update identity image count
            conn.execute('''
                UPDATE identities
                SET num_images = (
                    SELECT COUNT(*) FROM embeddings WHERE identity_name = ?
                )
                WHERE name = ?
            ''', (identity_name, identity_name))
            
            conn.commit()
        
        return cursor.lastrowid
    
    def get_embedding(self, identity_name: str, image_path: str) -> np.ndarray:
        """Get embedding for specific image"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT embedding FROM embeddings
                WHERE identity_name = ? AND image_path = ?
            ''', (identity_name, str(image_path)))
            
            row = cursor.fetchone()
            if row:
                return np.frombuffer(row[0], dtype=np.float32)
        
        return None
    
    def get_all_embeddings(self, identity_name: str = None) -> Dict[str, np.ndarray]:
        """
        Get all embeddings (optionally filtered by identity)
        
        Args:
            identity_name: Filter by identity (None = all)
            
        Returns:
            Dictionary {identity_name: embeddings_array}
        """
        embeddings_dict = {}
        
        with sqlite3.connect(self.db_path) as conn:
            if identity_name:
                query = 'SELECT identity_name, embedding FROM embeddings WHERE identity_name = ?'
                params = (identity_name,)
            else:
                query = 'SELECT identity_name, embedding FROM embeddings'
                params = ()
            
            cursor = conn.execute(query, params)
            
            current_identity = None
            identity_embeddings = []
            
            for row in cursor.fetchall():
                name, emb_bytes = row
                embedding = np.frombuffer(emb_bytes, dtype=np.float32)
                
                if current_identity != name:
                    if current_identity is not None:
                        embeddings_dict[current_identity] = np.array(identity_embeddings)
                    current_identity = name
                    identity_embeddings = []
                
                identity_embeddings.append(embedding)
            
            # Add last identity
            if current_identity is not None:
                embeddings_dict[current_identity] = np.array(identity_embeddings)
        
        return embeddings_dict
    
    def get_identities(self) -> List[Dict]:
        """Get list of all identities with metadata"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT name, num_images, created_at FROM identities ORDER BY name')
            
            identities = []
            for row in cursor.fetchall():
                identities.append({
                    'name': row[0],
                    'num_images': row[1],
                    'created_at': row[2]
                })
        
        return identities
    
    def get_embedding_count(self) -> int:
        """Get total number of embeddings"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM embeddings')
            return cursor.fetchone()[0]
    
    def get_identity_count(self) -> int:
        """Get total number of identities"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM identities')
            return cursor.fetchone()[0]
    
    def clear_database(self):
        """Clear all data from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM embeddings')
            conn.execute('DELETE FROM identities')
            conn.commit()
        
        logger.warning("Database cleared!")
    
    def get_db_stats(self) -> Dict:
        """Get database statistics"""
        return {
            'total_identities': self.get_identity_count(),
            'total_embeddings': self.get_embedding_count(),
            'db_path': str(self.db_path),
            'db_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
        }
