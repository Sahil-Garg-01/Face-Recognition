"""
Database CRUD Operations Module
Handles all database operations for the microservice
"""

import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class DatabaseHandler:
    """Handle all database operations"""
    
    def __init__(self, db_path: str):
        """
        Initialize database handler
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self._ensure_connection()
    
    def _ensure_connection(self):
        """Ensure database exists and is accessible"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.commit()
    
    def add_identity(self, identity_name: str, embedding: np.ndarray, 
                    image_path: str = None) -> int:
        """
        Add new identity/embedding to database
        
        Args:
            identity_name: Name of person
            embedding: 512-D embedding vector
            image_path: Path to source image
            
        Returns:
            ID of inserted record
        """
        embedding_bytes = embedding.astype(np.float32).tobytes()
        
        with sqlite3.connect(self.db_path) as conn:
            # Insert identity if not exists
            conn.execute('''
                INSERT OR IGNORE INTO identities (name, num_images)
                VALUES (?, 0)
            ''', (identity_name,))
            
            # Insert embedding
            cursor = conn.execute('''
                INSERT INTO embeddings (identity_name, image_path, embedding)
                VALUES (?, ?, ?)
            ''', (identity_name, image_path or '', embedding_bytes))
            
            # Update count
            conn.execute('''
                UPDATE identities
                SET num_images = (
                    SELECT COUNT(*) FROM embeddings WHERE identity_name = ?
                )
                WHERE name = ?
            ''', (identity_name, identity_name))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_identity(self, identity_name: str) -> Dict:
        """Get identity info"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT id, name, num_images, created_at
                FROM identities WHERE name = ?
            ''', (identity_name,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'num_images': row[2],
                    'created_at': row[3]
                }
        
        return None
    
    def list_identities(self, limit: int = None) -> List[Dict]:
        """
        List all identities
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of identity dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            if limit:
                query = '''
                    SELECT id, name, num_images, created_at
                    FROM identities ORDER BY name LIMIT ?
                '''
                cursor = conn.execute(query, (limit,))
            else:
                query = '''
                    SELECT id, name, num_images, created_at
                    FROM identities ORDER BY name
                '''
                cursor = conn.execute(query)
            
            identities = []
            for row in cursor.fetchall():
                identities.append({
                    'id': row[0],
                    'name': row[1],
                    'num_images': row[2],
                    'created_at': row[3]
                })
        
        return identities
    
    def delete_identity(self, identity_name: str) -> bool:
        """Delete identity and all its embeddings"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM embeddings WHERE identity_name = ?', (identity_name,))
            conn.execute('DELETE FROM identities WHERE name = ?', (identity_name,))
            conn.commit()
        
        return True
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Total identities
            cursor = conn.execute('SELECT COUNT(*) FROM identities')
            num_identities = cursor.fetchone()[0]
            
            # Total embeddings
            cursor = conn.execute('SELECT COUNT(*) FROM embeddings')
            num_embeddings = cursor.fetchone()[0]
            
            # DB size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        
        return {
            'num_identities': num_identities,
            'num_embeddings': num_embeddings,
            'db_size_mb': db_size / (1024 * 1024),
            'db_path': str(self.db_path)
        }
    
    def export_embeddings(self) -> Dict[str, np.ndarray]:
        """Export all embeddings from database"""
        embeddings_dict = {}
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT identity_name, embedding FROM embeddings')
            
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
            
            if current_identity is not None:
                embeddings_dict[current_identity] = np.array(identity_embeddings)
        
        return embeddings_dict
