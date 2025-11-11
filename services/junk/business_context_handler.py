# 📚 BUSINESS CONTEXT HANDLER - POINT 4
"""
Handles non-data queries using business_context table from Supabase
Provides intelligent answers for policy, procedures, and business information
"""

import logging
from typing import Dict, List, Any, Optional
import re

from supabase import Client
from utils.logger_config import get_logger

logger = get_logger(__name__)

class BusinessContextHandler:
    """
    BUSINESS CONTEXT HANDLER - POINT 4
    Uses business_context table to answer non-data queries
    """
    
    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.business_contexts = {}
        self.initialized = False
    
    async def initialize(self):
        """Load all business contexts from Supabase"""
        try:
            logger.info("📚 Loading business contexts from Supabase...")
            
            # Load all business contexts
            result = self.supabase.table("business_context").select("*").execute()
            contexts = result.data or []
            
            # Organize by context_type for faster lookup
            for context in contexts:
                context_type = context.get('context_type', 'general')
                if context_type not in self.business_contexts:
                    self.business_contexts[context_type] = []
                self.business_contexts[context_type].append(context)
            
            self.initialized = True
            logger.info(f"✅ Loaded {len(contexts)} business contexts from {len(self.business_contexts)} categories")
            
        except Exception as e:
            logger.error(f"❌ Business context initialization failed: {e}")
            self.initialized = True  # Continue with empty contexts
    
    async def handle_business_query(self, query: str) -> Dict[str, Any]:
        """
        MAIN BUSINESS CONTEXT METHOD - POINT 4
        Find and return relevant business context information
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            logger.info(f"📚 Handling business query: '{query}'")
            
            # Find best matching context
            matching_context = await self._find_matching_context(query)
            
            if not matching_context:
                return {
                    "success": False,
                    "query": query,
                    "error": "No relevant business information found",
                    "suggestion": "Try rephrasing your question or ask about policies, procedures, or company information"
                }
            
            # Format response
            response = await self._format_business_response(matching_context, query)
            
            return {
                "success": True,
                "query": query,
                "context_type": matching_context.get('context_type', 'general'),
                "title": matching_context.get('title', 'Business Information'),
                "answer": response["answer"],
                "full_context": response["full_context"],
                "source": matching_context.get('source', 'Business Context Database'),
                "last_updated": matching_context.get('updated_at', matching_context.get('created_at', 'Unknown')),
                "related_topics": response["related_topics"],
                "confidence": response["confidence"]
            }
            
        except Exception as e:
            logger.error(f"❌ Business query handling error: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e)
            }
    
    async def _find_matching_context(self, query: str) -> Optional[Dict[str, Any]]:
        """Find the best matching business context"""
        try:
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            best_match = None
            best_score = 0.0
            
            # Search through all contexts
            for context_type, contexts in self.business_contexts.items():
                for context in contexts:
                    score = self._calculate_context_score(query, query_words, context)
                    
                    if score > best_score:
                        best_score = score
                        best_match = context
            
            # Only return if score is above threshold
            if best_score > 0.3:
                logger.info(f"📚 Found matching context: '{best_match.get('title', 'Unknown')}' (score: {best_score:.3f})")
                return best_match
            
            logger.info(f"📚 No good context match found (best score: {best_score:.3f})")
            return None
            
        except Exception as e:
            logger.error(f"❌ Context matching error: {e}")
            return None
    
    def _calculate_context_score(self, query: str, query_words: set, context: Dict[str, Any]) -> float:
        """Calculate similarity score between query and context"""
        try:
            query_lower = query.lower()
            score = 0.0
            
            # Check title match (high weight)
            title = context.get('title', '').lower()
            if title:
                title_words = set(title.split())
                title_overlap = len(query_words.intersection(title_words))
                if title_overlap > 0:
                    score += (title_overlap / max(len(query_words), len(title_words))) * 3.0
                
                # Exact phrase match in title
                if any(word in title for word in query_lower.split() if len(word) > 2):
                    score += 1.0
            
            # Check description match (medium weight)
            description = context.get('description', '').lower()
            if description:
                desc_words = set(description.split())
                desc_overlap = len(query_words.intersection(desc_words))
                if desc_overlap > 0:
                    score += (desc_overlap / max(len(query_words), len(desc_words))) * 2.0
                
                # Partial phrase matches
                for query_word in query_lower.split():
                    if len(query_word) > 3 and query_word in description:
                        score += 0.5
            
            # Check keywords match (high weight)
            keywords = context.get('keywords', [])
            if keywords:
                keyword_matches = sum(1 for keyword in keywords if keyword.lower() in query_lower)
                if keyword_matches > 0:
                    score += keyword_matches * 2.0
            
            # Check context_type match (medium weight)
            context_type = context.get('context_type', '').lower()
            if context_type:
                if context_type in query_lower:
                    score += 1.5
                
                # Specific context type matching
                type_keywords = {
                    'policy': ['policy', 'rule', 'regulation', 'guideline'],
                    'procedure': ['procedure', 'process', 'how to', 'steps'],
                    'contact': ['contact', 'phone', 'email', 'address', 'support'],
                    'hr': ['employee', 'staff', 'hr', 'human resources', 'benefit'],
                    'it': ['it', 'technical', 'computer', 'software', 'system'],
                    'finance': ['finance', 'accounting', 'budget', 'expense', 'reimbursement']
                }
                
                if context_type in type_keywords:
                    for keyword in type_keywords[context_type]:
                        if keyword in query_lower:
                            score += 1.0
            
            # Content match (low weight)
            content = context.get('content', '').lower()
            if content:
                content_words = set(content.split())
                content_overlap = len(query_words.intersection(content_words))
                if content_overlap > 0:
                    score += (content_overlap / max(len(query_words), len(content_words))) * 1.0
            
            return score
            
        except Exception as e:
            logger.error(f"❌ Score calculation error: {e}")
            return 0.0
    
    async def _format_business_response(self, context: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Format business context response"""
        try:
            title = context.get('title', 'Business Information')
            description = context.get('description', '')
            content = context.get('content', '')
            context_type = context.get('context_type', 'general')
            
            # Create main answer
            answer = ""
            if description:
                answer = description
            
            if content and content != description:
                if answer:
                    answer += "\n\n"
                answer += content
            
            # If no content, create a basic response
            if not answer:
                answer = f"Information about {title}"
            
            # Full context for detailed view
            full_context = {
                "title": title,
                "description": description,
                "content": content,
                "type": context_type,
                "metadata": {
                    "source": context.get('source', ''),
                    "tags": context.get('tags', []),
                    "priority": context.get('priority', 'normal'),
                    "last_updated": context.get('updated_at', context.get('created_at'))
                }
            }
            
            # Find related topics
            related_topics = self._find_related_topics(context)
            
            # Calculate confidence based on match quality
            confidence = self._calculate_response_confidence(context, query)
            
            return {
                "answer": answer,
                "full_context": full_context,
                "related_topics": related_topics,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"❌ Response formatting error: {e}")
            return {
                "answer": "Error formatting business context response",
                "full_context": {},
                "related_topics": [],
                "confidence": 0.5
            }
    
    def _find_related_topics(self, context: Dict[str, Any]) -> List[str]:
        """Find related topics from other contexts"""
        try:
            context_type = context.get('context_type', '')
            current_title = context.get('title', '').lower()
            related = []
            
            # Find contexts of the same type
            if context_type in self.business_contexts:
                for other_context in self.business_contexts[context_type]:
                    other_title = other_context.get('title', '')
                    if other_title.lower() != current_title and other_title:
                        related.append(other_title)
            
            # Limit to top 5 related topics
            return related[:5]
            
        except Exception as e:
            logger.error(f"❌ Related topics error: {e}")
            return []
    
    def _calculate_response_confidence(self, context: Dict[str, Any], query: str) -> float:
        """Calculate confidence in the response"""
        try:
            # Base confidence
            confidence = 0.7
            
            # Boost for exact matches
            title = context.get('title', '').lower()
            query_lower = query.lower()
            
            if any(word in title for word in query_lower.split() if len(word) > 2):
                confidence += 0.2
            
            # Boost for keyword matches
            keywords = context.get('keywords', [])
            keyword_matches = sum(1 for keyword in keywords if keyword.lower() in query_lower)
            if keyword_matches > 0:
                confidence += min(0.2, keyword_matches * 0.1)
            
            # Reduce if content is sparse
            content_length = len(context.get('content', '') + context.get('description', ''))
            if content_length < 50:
                confidence -= 0.1
            
            return min(1.0, max(0.3, confidence))
            
        except Exception as e:
            logger.error(f"❌ Confidence calculation error: {e}")
            return 0.6

# Factory function
async def create_business_context_handler(supabase: Client) -> BusinessContextHandler:
    """Create and initialize business context handler"""
    handler = BusinessContextHandler(supabase)
    await handler.initialize()
    return handler