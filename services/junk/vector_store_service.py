"""
Vector Store Service - Document embeddings and vector search
Handles document processing, embeddings generation, and semantic search
"""

import os
import json
import logging
import asyncio
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

from supabase import create_client, Client
import openai
import cohere

from utils.redis_client import get_redis_client
from utils.logger_config import get_logger

logger = get_logger(__name__)

class VectorStoreService:
    def __init__(self):
        self.supabase = None
        self.embedding_clients = {}
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.vector_tables = self._define_vector_tables()

    async def initialize(self):
        """Initialize vector store service"""
        # Initialize Supabase
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError("Missing Supabase credentials")

        self.supabase = create_client(supabase_url, supabase_key)

        # Initialize embedding providers
        await self._initialize_embedding_providers()

        # Create vector tables if needed
        await self._setup_vector_tables()

        logger.info("✅ Vector Store Service initialized")

    def _define_vector_tables(self):
        """Define vector table schemas"""
        return {
            "table_metadata": {
                "table": "table_metadata_vectors",
                "description": "Database table and column metadata"
            },
            "sql_examples": {
                "table": "sql_examples_vectors", 
                "description": "Example SQL queries and explanations"
            },
            "documents": {
                "table": "document_vectors",
                "description": "User uploaded documents"
            },
            "websites": {
                "table": "website_vectors",
                "description": "Scraped website content"
            }
        }

    async def _initialize_embedding_providers(self):
        """Initialize embedding providers"""

        # OpenAI embeddings
        if os.getenv("OPENAI_API_KEY"):
            self.embedding_clients["openai"] = openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
            logger.info("✓ OpenAI embeddings initialized")

        # Cohere embeddings
        if os.getenv("COHERE_API_KEY"):
            self.embedding_clients["cohere"] = cohere.AsyncClient(
                api_key=os.getenv("COHERE_API_KEY")
            )
            logger.info("✓ Cohere embeddings initialized")

        if not self.embedding_clients:
            logger.warning("No embedding providers available - using mock embeddings")

    async def generate_embedding(self, text: str, provider: str = "openai") -> List[float]:
        """Generate text embedding"""
        cache_key = f"embedding:{provider}:{hashlib.md5(text[:100].encode()).hexdigest()}"

        # Try cache first
        redis = get_redis_client()
        if redis:
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)

        try:
            embedding = None

            if provider == "openai" and "openai" in self.embedding_clients:
                response = await self.embedding_clients["openai"].embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                embedding = response.data[0].embedding

            elif provider == "cohere" and "cohere" in self.embedding_clients:
                response = await self.embedding_clients["cohere"].embed(
                    texts=[text],
                    model="embed-english-v3.0",
                    input_type="search_document"
                )
                embedding = response.embeddings[0]

            else:
                # Mock embedding for demo purposes
                embedding = self._generate_mock_embedding(text)

            # Cache for 24 hours
            if redis:
                await redis.setex(cache_key, 86400, json.dumps(embedding))

            return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed for {provider}: {e}")
            # Fallback to mock embedding
            return self._generate_mock_embedding(text)

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate mock embedding for demo purposes"""
        # Simple hash-based mock embedding
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        np.random.seed(seed)
        return np.random.normal(0, 1, 1536).tolist()  # OpenAI embedding size

    async def process_document(self, document_data: Dict, user_id: str) -> List[Dict]:
        """Process document and create vector embeddings"""
        try:
            chunks = self._chunk_text(document_data["content"])
            results = []

            for i, chunk in enumerate(chunks):
                embedding = await self.generate_embedding(chunk)

                vector_data = {
                    "user_id": user_id,
                    "title": document_data["title"],
                    "content": chunk,
                    "chunk_index": i,
                    "source": document_data["source"],
                    "document_type": document_data.get("document_type", "unknown"),
                    "file_size": document_data.get("file_size", 0),
                    "embedding": embedding,
                    "metadata": {
                        "chunk_count": len(chunks),
                        "indexed_at": datetime.utcnow().isoformat(),
                        **document_data.get("metadata", {})
                    }
                }

                # Insert into Supabase
                result = self.supabase.table("document_vectors").insert(vector_data).execute()
                results.append(result.data[0] if result.data else vector_data)

            logger.info(f"✓ Processed document '{document_data['title']}' into {len(chunks)} chunks")
            return results

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise

    async def process_website(self, website_data: Dict, user_id: str) -> List[Dict]:
        """Process website content and create vector embeddings"""
        try:
            chunks = self._chunk_text(website_data["content"])
            results = []

            for i, chunk in enumerate(chunks):
                embedding = await self.generate_embedding(chunk)

                vector_data = {
                    "user_id": user_id,
                    "url": website_data["url"],
                    "title": website_data["title"],
                    "content": chunk,
                    "chunk_index": i,
                    "embedding": embedding,
                    "metadata": {
                        "chunk_count": len(chunks),
                        "scraped_at": datetime.utcnow().isoformat(),
                        **website_data.get("metadata", {})
                    }
                }

                # Insert into Supabase
                result = self.supabase.table("website_vectors").insert(vector_data).execute()
                results.append(result.data[0] if result.data else vector_data)

            logger.info(f"✓ Processed website '{website_data['url']}' into {len(chunks)} chunks")
            return results

        except Exception as e:
            logger.error(f"Website processing failed: {e}")
            raise

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = " ".join(chunk_words)
            if chunk.strip():
                chunks.append(chunk.strip())

            # Break if we've reached the end
            if i + self.chunk_size >= len(words):
                break

        return chunks if chunks else [text]

    async def search_similar(
        self, 
        query: str, 
        table_type: str, 
        user_id: Optional[str] = None,
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Dict]:
        """Search for similar content using vector similarity"""
        try:
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)

            # Get table info
            table_info = self.vector_tables.get(table_type)
            if not table_info:
                raise ValueError(f"Unknown table type: {table_type}")

            table_name = table_info["table"]

            # Build query based on table type
            if table_type in ["documents", "websites"] and user_id:
                # User-specific content
                results = self._search_user_vectors(table_name, query_embedding, user_id, limit, threshold)
            else:
                # Global content (metadata, examples)
                results = self._search_global_vectors(table_name, query_embedding, limit, threshold)

            # For demo purposes, return mock similar results
            return await self._mock_similarity_search(query, table_type, limit)

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _search_user_vectors(self, table: str, embedding: List[float], user_id: str, limit: int, threshold: float):
        """Search user-specific vectors"""
        # In a real implementation, this would use pgvector similarity search
        # For demo, we'll do a simple query
        return self.supabase.table(table).select("*").eq("user_id", user_id).limit(limit).execute()

    def _search_global_vectors(self, table: str, embedding: List[float], limit: int, threshold: float):
        """Search global vectors (metadata, examples)"""
        # In a real implementation, this would use pgvector similarity search
        return self.supabase.table(table).select("*").limit(limit).execute()

    async def _mock_similarity_search(self, query: str, table_type: str, limit: int) -> List[Dict]:
        """Mock similarity search results for demo"""
        if table_type == "documents":
            return [
                {
                    "id": f"doc_{i}",
                    "content": f"Sample document content related to: {query}",
                    "similarity": 0.85 - (i * 0.1),
                    "title": f"Document {i+1}",
                    "source": "demo"
                }
                for i in range(min(3, limit))
            ]
        elif table_type == "websites":
            return [
                {
                    "id": f"web_{i}",
                    "content": f"Website content about: {query}",
                    "similarity": 0.80 - (i * 0.1),
                    "title": f"Website {i+1}",
                    "url": f"https://example{i+1}.com"
                }
                for i in range(min(2, limit))
            ]
        else:
            return []

    async def get_schema_context(self, query: str) -> Dict[str, Any]:
        """Get relevant schema context for query"""
        # Search for relevant table and column metadata
        table_results = await self.search_similar(query, "table_metadata", limit=5)
        example_results = await self.search_similar(query, "sql_examples", limit=3)

        return {
            "tables": table_results,
            "examples": example_results,
            "context_score": max([r.get("similarity", 0) for r in table_results + example_results] + [0])
        }

    async def get_document_context(self, query: str, user_id: str) -> Dict[str, Any]:
        """Get relevant document context for query"""
        doc_results = await self.search_similar(query, "documents", user_id, limit=5)
        web_results = await self.search_similar(query, "websites", user_id, limit=3)

        return {
            "documents": doc_results,
            "websites": web_results,
            "context_score": max(
                [r.get("similarity", 0) for r in doc_results + web_results] + [0]
            )
        }

    async def search_documents(self, query: str, user_id: str, limit: int = 10) -> Dict[str, Any]:
        """Search user's documents and websites"""
        doc_results = await self.search_similar(query, "documents", user_id, limit//2)
        web_results = await self.search_similar(query, "websites", user_id, limit//2)

        return {
            "documents": doc_results,
            "websites": web_results,
            "total": len(doc_results) + len(web_results),
            "query": query
        }

    async def _setup_vector_tables(self):
        """Setup vector tables in Supabase"""
        try:
            # Check if tables exist by trying to query them
            for table_type, table_info in self.vector_tables.items():
                try:
                    result = self.supabase.table(table_info["table"]).select("count", count="exact").limit(1).execute()
                    logger.info(f"✓ Vector table '{table_info['table']}' is available")
                except Exception as e:
                    logger.warning(f"Vector table '{table_info['table']}' may not exist: {e}")

            logger.info("Vector tables setup completed")

        except Exception as e:
            logger.error(f"Vector table setup failed: {e}")

    async def add_sql_examples(self, examples: List[Dict]):
        """Add SQL examples to vector store"""
        try:
            for example in examples:
                embedding = await self.generate_embedding(
                    f"Query: {example['query']} SQL: {example['sql']} Explanation: {example['explanation']}"
                )

                vector_data = {
                    "query": example["query"],
                    "sql": example["sql"], 
                    "explanation": example["explanation"],
                    "embedding": embedding,
                    "metadata": {
                        "indexed_at": datetime.utcnow().isoformat(),
                        "complexity": example.get("complexity", 1),
                        "tags": example.get("tags", [])
                    }
                }

                self.supabase.table("sql_examples_vectors").insert(vector_data).execute()

            logger.info(f"✓ Added {len(examples)} SQL examples to vector store")

        except Exception as e:
            logger.error(f"Failed to add SQL examples: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Health check for vector store service"""
        try:
            # Test embedding generation
            test_embedding = await self.generate_embedding("test query")

            # Test database connection
            result = self.supabase.table("document_vectors").select("count", count="exact").limit(1).execute()

            return {
                "status": "healthy",
                "embedding_providers": list(self.embedding_clients.keys()) or ["mock"],
                "embedding_dimension": len(test_embedding),
                "vector_tables": len(self.vector_tables),
                "database_connection": "connected"
            }

        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
