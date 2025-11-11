# 🔥 VECTOR SERVICE - USES embedding_1 COLUMN
# Updated to use the new embedding_1 column

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Any
import asyncio
import json

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)

class CorrectedVectorService:
    """
    Vector service using embedding_1 column (new embeddings)
    Handles embeddings stored as arrays/lists/strings
    """
    
    def __init__(self):
        self.openai_client = None
        self.embedding_cache = {}
        self.embedding_key = "embedding_1"  # ← USE embedding_1 COLUMN
        self._init_openai()
    
    def _init_openai(self):
        """Initialize OpenAI for embeddings"""
        try:
            if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
                self.openai_client = openai.AsyncOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY")
                )
                logger.info("[VECTOR] ✅ OpenAI client initialized (using embedding_1)")
            else:
                logger.warning("[VECTOR] ⚠️  No OpenAI API key")
        except Exception as e:
            logger.error(f"[VECTOR] OpenAI init failed: {e}")
    
    async def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text"""
        try:
            if text in self.embedding_cache:
                return self.embedding_cache[text]
            
            if self.openai_client:
                response = await self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text,
                    dimensions=1536
                )
                embedding = response.data[0].embedding
                self.embedding_cache[text] = embedding
                logger.info(f"[VECTOR] Created embedding_1 (dim={len(embedding)})")
                return embedding
            else:
                mock_embedding = np.random.rand(1536).tolist()
                return mock_embedding
                
        except Exception as e:
            logger.error(f"[VECTOR] Embedding creation failed: {e}")
            return np.zeros(1536).tolist()
    
    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts"""
        try:
            tasks = [self.create_embedding(text) for text in texts]
            embeddings = await asyncio.gather(*tasks)
            logger.info(f"[VECTOR] Created {len(embeddings)} embeddings_1 in batch")
            return embeddings
        except Exception as e:
            logger.error(f"[VECTOR] Batch embedding failed: {e}")
            return [np.zeros(1536).tolist() for _ in texts]
    
    def cosine_similarity(self, vec1, vec2) -> float:
        """
        Calculate cosine similarity - handles arrays/lists/strings
        """
        try:
            # Convert strings to lists
            if isinstance(vec1, str):
                try:
                    vec1 = json.loads(vec1) if vec1.startswith('[') else eval(vec1)
                except Exception as e:
                    logger.error(f"[VECTOR] Failed to parse vec1: {e}")
                    return 0.0
            
            if isinstance(vec2, str):
                try:
                    vec2 = json.loads(vec2) if vec2.startswith('[') else eval(vec2)
                except Exception as e:
                    logger.error(f"[VECTOR] Failed to parse vec2: {e}")
                    return 0.0
            
            # Handle NULL
            if vec1 is None or vec2 is None:
                return 0.0
            
            # Convert to numpy
            v1 = np.array(vec1, dtype=np.float32)
            v2 = np.array(vec2, dtype=np.float32)
            
            # Calculate
            dot = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot / (norm1 * norm2))
        except Exception as e:
            logger.error(f"[VECTOR] Similarity failed: {e}")
            return 0.0
    
    async def search_similar_tables(
        self, 
        user_prompt: str, 
        all_tables: List[Dict[str, Any]],
        similarity_threshold: float = 0.7,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar tables using embedding_1"""
        try:
            logger.info(f"[VECTOR] Searching similar tables for: '{user_prompt}'")
            
            query_embedding = await self.create_embedding(user_prompt)
            
            similar = await self.find_most_similar(
                query_embedding,
                all_tables,
                self.embedding_key,  # ← USE embedding_1
                max_results,
                similarity_threshold
            )
            
            logger.info(f"[VECTOR] Found {len(similar)} similar tables")
            return similar
            
        except Exception as e:
            logger.error(f"[VECTOR] Table search failed: {e}")
            return []
    
    async def search_similar_columns(
        self,
        user_prompt: str,
        all_columns: List[Dict[str, Any]],
        selected_table_names: Optional[List[str]] = None,
        similarity_threshold: float = 0.7,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for similar columns using embedding_1"""
        try:
            logger.info(f"[VECTOR] Searching similar columns for: '{user_prompt}'")
            
            if selected_table_names:
                all_columns = [
                    col for col in all_columns 
                    if col.get('table_name') in selected_table_names
                ]
                logger.info(f"[VECTOR] Filtered to {len(all_columns)} columns")
            
            query_embedding = await self.create_embedding(user_prompt)
            
            similar = await self.find_most_similar(
                query_embedding,
                all_columns,
                self.embedding_key,  # ← USE embedding_1
                max_results,
                similarity_threshold
            )
            
            logger.info(f"[VECTOR] Found {len(similar)} similar columns")
            return similar
            
        except Exception as e:
            logger.error(f"[VECTOR] Column search failed: {e}")
            return []
    
    async def search_similar_sql_patterns(
        self,
        user_prompt: str,
        all_patterns: List[Dict[str, Any]],
        similarity_threshold: float = 0.7,
        max_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Search for similar SQL patterns using embedding_1"""
        try:
            logger.info(f"[VECTOR] Searching similar SQL patterns for: '{user_prompt}'")
            
            query_embedding = await self.create_embedding(user_prompt)
            
            similar = await self.find_most_similar(
                query_embedding,
                all_patterns,
                self.embedding_key,  # ← USE embedding_1
                max_results,
                similarity_threshold
            )
            
            logger.info(f"[VECTOR] Found {len(similar)} similar SQL patterns")
            return similar
            
        except Exception as e:
            logger.error(f"[VECTOR] SQL pattern search failed: {e}")
            return []
    
    async def find_most_similar(
        self,
        query_embedding: List[float],
        candidates: List[Dict[str, Any]],
        embedding_key: str = "embedding_1",  # ← DEFAULT embedding_1
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find most similar items using embedding_1"""
        try:
            similarities = []
            
            for candidate in candidates:
                if embedding_key not in candidate:
                    continue
                
                candidate_embedding = candidate[embedding_key]
                similarity = self.cosine_similarity(query_embedding, candidate_embedding)
                
                if similarity >= threshold:
                    similarities.append({
                        **candidate,
                        'similarity_score': similarity
                    })
            
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            logger.info(f"[VECTOR] Found {len(similarities)} items above threshold {threshold}")
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"[VECTOR] Find similar failed: {e}")
            return []
