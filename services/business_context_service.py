# 🏢 BUSINESS CONTEXT SERVICE - Non-Data Question Handler

import logging
from typing import Dict, Any, List, Optional
from database_supabase_client import CorrectedSupabaseClient
from services.vector_service import CorrectedVectorService
from services.llm_service import LLMService
from utils.config import config

logger = logging.getLogger(__name__)

class BusinessContextService:
    """
    Handles non-data questions using business context
    Part of your architecture: Non-Data Question → Business Context Search → LLM Answer
    """
    
    def __init__(self, supabase_client: CorrectedSupabaseClient, vector_service: CorrectedVectorService, llm_service: LLMService):
        self.supabase_client = supabase_client
        self.vector_service = vector_service
        self.llm_service = llm_service
        logger.info("✅ Business Context Service initialized")
    
    async def answer_business_question(self, user_prompt: str, user_id: str = None) -> Dict[str, Any]:
        """
        Answer business question following your exact architecture:
        1. Convert prompt to vector embedding
        2. Search business context (70%+ similarity)  
        3. Use exact prompt template for LLM
        4. Return formatted business answer
        
        Args:
            user_prompt: User's business question
            user_id: Optional user identifier
            
        Returns:
            Dict with business answer and metadata
        """
        try:
            logger.info(f"🏢 Processing business question: {user_prompt[:50]}...")
            
            # Step 1: Convert user prompt to vector embedding
            query_embedding = await self.vector_service.create_embedding(user_prompt)
            
            # Step 2: Search business context with 70%+ cosine similarity threshold
            context_results = await self.supabase_client.search_business_context(
                query_embedding,
                similarity_threshold=config.COSINE_SIMILARITY_THRESHOLD,
                limit=config.MAX_VECTOR_RESULTS
            )
            
            logger.info(f"📊 Found {len(context_results)} relevant business context pieces")
            
            # Step 3: Generate answer using exact prompt template from your PDF
            business_answer = await self.llm_service.answer_business_question(
                user_prompt=user_prompt,
                business_context=context_results
            )
            
            # Step 4: Format and return response
            return {
                "success": True,
                "answer": business_answer,
                "context_used": len(context_results),
                "context_pieces": [
                    {
                        "content": ctx.get('document_piece', ''),
                        "similarity": ctx.get('similarity', 0.0),
                        "source": ctx.get('source_document', 'Unknown')
                    }
                    for ctx in context_results
                ],
                "metadata": {
                    "embedding_model": config.EMBEDDING_MODEL,
                    "similarity_threshold": config.COSINE_SIMILARITY_THRESHOLD,
                    "llm_model": config.PRIMARY_LLM,
                    "total_context_pieces": len(context_results)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Business question processing failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "answer": "I encountered an error while processing your business question.",
                "context_used": 0,
                "context_pieces": []
            }
    
    async def add_business_context(
        self, 
        document_content: str, 
        source_document: str = None,
        chunk_size: int = None
    ) -> Dict[str, Any]:
        """
        Add business context document, chunking into 300-character pieces as per architecture
        
        Args:
            document_content: Full document content
            source_document: Source document name/path
            chunk_size: Override default chunk size (300 chars from architecture)
            
        Returns:
            Dict with processing results
        """
        try:
            chunk_size = chunk_size or config.CONTEXT_CHUNK_SIZE  # 300 chars from architecture
            
            logger.info(f"📄 Processing business document: {len(document_content)} chars")
            
            # Split document into chunks of 300 characters (as specified in your PDF)
            chunks = self._split_document_into_chunks(document_content, chunk_size)
            
            logger.info(f"✂️ Split document into {len(chunks)} chunks of ~{chunk_size} chars each")
            
            # Create embeddings for each chunk
            embeddings = await self.vector_service.create_embeddings_batch([chunk for chunk in chunks])
            
            # Insert chunks with embeddings into business_context table
            inserted_count = 0
            for chunk, embedding in zip(chunks, embeddings):
                try:
                    await self.supabase_client.insert_business_context(
                        document_piece=chunk,
                        embedding=embedding,
                        source_document=source_document
                    )
                    inserted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to insert chunk: {e}")
            
            logger.info(f"✅ Inserted {inserted_count}/{len(chunks)} business context chunks")
            
            return {
                "success": True,
                "total_chunks": len(chunks),
                "inserted_chunks": inserted_count,
                "failed_chunks": len(chunks) - inserted_count,
                "chunk_size": chunk_size,
                "source_document": source_document
            }
            
        except Exception as e:
            logger.error(f"❌ Business context addition failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "total_chunks": 0,
                "inserted_chunks": 0
            }
    
    def _split_document_into_chunks(self, content: str, chunk_size: int) -> List[str]:
        """
        Split document into chunks of specified size (300 chars from architecture)
        Tries to break at sentence boundaries when possible
        
        Args:
            content: Document content to split
            chunk_size: Size of each chunk in characters
            
        Returns:
            List of document chunks
        """
        if not content.strip():
            return []
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences first
        sentences = content.replace('\n', ' ').split('. ')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add period back if it's not the last sentence
            if not sentence.endswith('.') and sentence != sentences[-1]:
                sentence += '.'
            
            # If adding this sentence would exceed chunk size
            if len(current_chunk + ' ' + sentence) > chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Sentence itself is too long, need to split it
                    while len(sentence) > chunk_size:
                        chunks.append(sentence[:chunk_size].strip())
                        sentence = sentence[chunk_size:]
                    current_chunk = sentence
            else:
                # Add sentence to current chunk
                current_chunk += ' ' + sentence if current_chunk else sentence
        
        # Add final chunk if exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Validate chunk sizes (must be <= 300 chars as per database constraint)
        validated_chunks = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                validated_chunks.append(chunk)
            else:
                # Split oversized chunk
                while len(chunk) > chunk_size:
                    validated_chunks.append(chunk[:chunk_size])
                    chunk = chunk[chunk_size:]
                if chunk.strip():
                    validated_chunks.append(chunk.strip())
        
        return validated_chunks
    
    async def search_business_context(
        self, 
        query: str, 
        similarity_threshold: float = None,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search business context for relevant information
        
        Args:
            query: Search query
            similarity_threshold: Override default similarity threshold
            limit: Maximum results to return
            
        Returns:
            List of relevant business context pieces
        """
        try:
            threshold = similarity_threshold or config.COSINE_SIMILARITY_THRESHOLD
            max_results = limit or config.MAX_VECTOR_RESULTS
            
            # Create query embedding
            query_embedding = await self.vector_service.create_embedding(query)
            
            # Search business context
            results = await self.supabase_client.search_business_context(
                query_embedding,
                similarity_threshold=threshold,
                limit=max_results
            )
            
            logger.info(f"🔍 Found {len(results)} business context matches for: {query[:50]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Business context search failed: {e}")
            return []
    
    async def get_context_statistics(self) -> Dict[str, Any]:
        """Get statistics about business context data"""
        try:
            # This would require custom queries to get statistics
            # For now, return placeholder data
            return {
                "total_context_pieces": 0,
                "average_piece_length": 0,
                "total_sources": 0,
                "embedding_model": config.EMBEDDING_MODEL,
                "chunk_size": config.CONTEXT_CHUNK_SIZE
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get context statistics: {e}")
            return {
                "error": str(e)
            }