# 🎯 VECTORIZED QUERY CLASSIFIER - POINT 1
"""
Vectorized classification: Data Query vs Non-Data Query using RAG
Uses embeddings to classify user intent with high accuracy
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re

# Try importing advanced libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from utils.logger_config import get_logger

logger = get_logger(__name__)

# VECTORIZED CLASSIFICATION DATABASE - COMPREHENSIVE TRAINING DATA
CLASSIFICATION_TRAINING_DATA = [
    # DATA QUERIES - Intent to retrieve/manipulate data
    {"text": "show me all customers", "label": "data_query", "confidence": 1.0},
    {"text": "count total orders", "label": "data_query", "confidence": 1.0},
    {"text": "list products by category", "label": "data_query", "confidence": 1.0},
    {"text": "top 10 sales performers", "label": "data_query", "confidence": 1.0},
    {"text": "average revenue per month", "label": "data_query", "confidence": 1.0},
    {"text": "customers from last year", "label": "data_query", "confidence": 1.0},
    {"text": "products with low inventory", "label": "data_query", "confidence": 1.0},
    {"text": "revenue by region", "label": "data_query", "confidence": 1.0},
    {"text": "most popular items", "label": "data_query", "confidence": 1.0},
    {"text": "sales trend analysis", "label": "data_query", "confidence": 1.0},
    {"text": "customer segmentation report", "label": "data_query", "confidence": 1.0},
    {"text": "inventory levels", "label": "data_query", "confidence": 1.0},
    {"text": "profit margins by product", "label": "data_query", "confidence": 1.0},
    {"text": "employee performance metrics", "label": "data_query", "confidence": 1.0},
    {"text": "monthly recurring revenue", "label": "data_query", "confidence": 1.0},
    {"text": "churn rate calculation", "label": "data_query", "confidence": 1.0},
    {"text": "conversion funnel data", "label": "data_query", "confidence": 1.0},
    {"text": "user engagement statistics", "label": "data_query", "confidence": 1.0},
    {"text": "financial summary report", "label": "data_query", "confidence": 1.0},
    {"text": "operational KPIs dashboard", "label": "data_query", "confidence": 1.0},
    {"text": "get customer details", "label": "data_query", "confidence": 1.0},
    {"text": "fetch order history", "label": "data_query", "confidence": 1.0},
    {"text": "retrieve product information", "label": "data_query", "confidence": 1.0},
    {"text": "display sales data", "label": "data_query", "confidence": 1.0},
    {"text": "extract user records", "label": "data_query", "confidence": 1.0},
    {"text": "find transactions", "label": "data_query", "confidence": 1.0},
    {"text": "search inventory", "label": "data_query", "confidence": 1.0},
    {"text": "query database", "label": "data_query", "confidence": 1.0},
    {"text": "analyze performance", "label": "data_query", "confidence": 1.0},
    {"text": "calculate totals", "label": "data_query", "confidence": 1.0},
    {"text": "sum revenues", "label": "data_query", "confidence": 1.0},
    {"text": "aggregate data", "label": "data_query", "confidence": 1.0},
    {"text": "group by category", "label": "data_query", "confidence": 1.0},
    {"text": "order by date", "label": "data_query", "confidence": 1.0},
    {"text": "filter results", "label": "data_query", "confidence": 1.0},
    {"text": "sort customers", "label": "data_query", "confidence": 1.0},
    {"text": "rank products", "label": "data_query", "confidence": 1.0},
    {"text": "percentage breakdown", "label": "data_query", "confidence": 1.0},
    {"text": "growth rate metrics", "label": "data_query", "confidence": 1.0},
    {"text": "comparison analysis", "label": "data_query", "confidence": 1.0},
    
    # NON-DATA QUERIES - Business context, policies, procedures
    {"text": "what is our refund policy", "label": "non_data_query", "confidence": 1.0},
    {"text": "company mission statement", "label": "non_data_query", "confidence": 1.0},
    {"text": "how to contact support", "label": "non_data_query", "confidence": 1.0},
    {"text": "office hours information", "label": "non_data_query", "confidence": 1.0},
    {"text": "employee handbook guidelines", "label": "non_data_query", "confidence": 1.0},
    {"text": "privacy policy details", "label": "non_data_query", "confidence": 1.0},
    {"text": "terms of service", "label": "non_data_query", "confidence": 1.0},
    {"text": "business process workflow", "label": "non_data_query", "confidence": 1.0},
    {"text": "organizational structure", "label": "non_data_query", "confidence": 1.0},
    {"text": "company values", "label": "non_data_query", "confidence": 1.0},
    {"text": "code of conduct", "label": "non_data_query", "confidence": 1.0},
    {"text": "safety procedures", "label": "non_data_query", "confidence": 1.0},
    {"text": "emergency protocols", "label": "non_data_query", "confidence": 1.0},
    {"text": "training requirements", "label": "non_data_query", "confidence": 1.0},
    {"text": "compliance standards", "label": "non_data_query", "confidence": 1.0},
    {"text": "quality assurance process", "label": "non_data_query", "confidence": 1.0},
    {"text": "customer service guidelines", "label": "non_data_query", "confidence": 1.0},
    {"text": "product documentation", "label": "non_data_query", "confidence": 1.0},
    {"text": "user manual instructions", "label": "non_data_query", "confidence": 1.0},
    {"text": "installation guide", "label": "non_data_query", "confidence": 1.0},
    {"text": "troubleshooting steps", "label": "non_data_query", "confidence": 1.0},
    {"text": "frequently asked questions", "label": "non_data_query", "confidence": 1.0},
    {"text": "help documentation", "label": "non_data_query", "confidence": 1.0},
    {"text": "company history", "label": "non_data_query", "confidence": 1.0},
    {"text": "about our team", "label": "non_data_query", "confidence": 1.0},
    {"text": "career opportunities", "label": "non_data_query", "confidence": 1.0},
    {"text": "benefits package", "label": "non_data_query", "confidence": 1.0},
    {"text": "vacation policy", "label": "non_data_query", "confidence": 1.0},
    {"text": "remote work guidelines", "label": "non_data_query", "confidence": 1.0},
    {"text": "dress code policy", "label": "non_data_query", "confidence": 1.0},
    {"text": "meeting room booking", "label": "non_data_query", "confidence": 1.0},
    {"text": "parking regulations", "label": "non_data_query", "confidence": 1.0},
    {"text": "IT support procedures", "label": "non_data_query", "confidence": 1.0},
    {"text": "expense reporting process", "label": "non_data_query", "confidence": 1.0},
    {"text": "travel reimbursement", "label": "non_data_query", "confidence": 1.0},
    {"text": "performance review cycle", "label": "non_data_query", "confidence": 1.0},
    {"text": "promotion criteria", "label": "non_data_query", "confidence": 1.0},
    {"text": "disciplinary procedures", "label": "non_data_query", "confidence": 1.0},
    {"text": "whistleblower protection", "label": "non_data_query", "confidence": 1.0},
    {"text": "intellectual property rights", "label": "non_data_query", "confidence": 1.0}
]

class VectorizedQueryClassifier:
    """
    VECTORIZED QUERY CLASSIFIER - RAG BASED CLASSIFICATION
    Uses advanced embeddings and vector similarity for accurate classification
    """
    
    def __init__(self):
        self.embeddings_model = None
        self.faiss_index = None
        self.training_embeddings = None
        self.training_labels = []
        self.training_texts = []
        self.initialized = False
    
    async def initialize(self):
        """Initialize the vectorized classifier"""
        try:
            logger.info("🧠 Initializing Vectorized Query Classifier...")
            
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning("⚠️ Sentence transformers not available, using keyword fallback")
                self.initialized = True
                return
            
            # Initialize embeddings model
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embeddings model loaded")
            
            # Prepare training data
            self.training_texts = [item["text"] for item in CLASSIFICATION_TRAINING_DATA]
            self.training_labels = [item["label"] for item in CLASSIFICATION_TRAINING_DATA]
            
            # Generate embeddings for training data
            self.training_embeddings = self.embeddings_model.encode(
                self.training_texts, convert_to_numpy=True
            )
            logger.info(f"✅ Generated embeddings for {len(self.training_texts)} training examples")
            
            # Initialize FAISS index if available
            if FAISS_AVAILABLE:
                dimension = self.training_embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)
                
                # Normalize embeddings
                norms = np.linalg.norm(self.training_embeddings, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                normalized_embeddings = self.training_embeddings / norms
                
                self.faiss_index.add(normalized_embeddings.astype('float32'))
                logger.info(f"⚡ FAISS index initialized with {self.faiss_index.ntotal} examples")
            
            self.initialized = True
            logger.info("🎯 Vectorized Query Classifier ready!")
            
        except Exception as e:
            logger.error(f"❌ Classifier initialization failed: {e}")
            self.initialized = True  # Continue with fallback
    
    async def classify_query(self, query: str) -> Dict[str, Any]:
        """
        MAIN CLASSIFICATION METHOD
        Returns: {
            "query_type": "data_query" | "non_data_query",
            "confidence": float,
            "reasoning": str,
            "similar_examples": List[str]
        }
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            query = query.strip()
            logger.info(f"🔍 Classifying query: '{query}'")
            
            if self.embeddings_model and self.faiss_index:
                return await self._vectorized_classification(query)
            else:
                return self._keyword_classification(query)
                
        except Exception as e:
            logger.error(f"❌ Classification error: {e}")
            return {
                "query_type": "data_query",  # Default to data query
                "confidence": 0.5,
                "reasoning": f"Error in classification: {str(e)}",
                "similar_examples": []
            }
    
    async def _vectorized_classification(self, query: str) -> Dict[str, Any]:
        """Advanced vectorized classification using embeddings and FAISS"""
        try:
            # Generate query embedding
            query_embedding = self.embeddings_model.encode([query], convert_to_numpy=True)
            query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            query_norm = np.maximum(query_norm, 1e-8)
            query_normalized = query_embedding / query_norm
            
            # Find most similar training examples
            k = min(10, len(self.training_texts))  # Top 10 similar examples
            similarities, indices = self.faiss_index.search(query_normalized.astype('float32'), k)
            
            # Analyze results
            data_query_scores = []
            non_data_query_scores = []
            similar_examples = []
            
            for similarity, idx in zip(similarities[0], indices[0]):
                if 0 <= idx < len(self.training_labels):
                    label = self.training_labels[idx]
                    text = self.training_texts[idx]
                    
                    if label == "data_query":
                        data_query_scores.append(similarity)
                    else:
                        non_data_query_scores.append(similarity)
                    
                    similar_examples.append({
                        "text": text,
                        "label": label,
                        "similarity": float(similarity)
                    })
            
            # Calculate confidence scores
            avg_data_score = np.mean(data_query_scores) if data_query_scores else 0.0
            avg_non_data_score = np.mean(non_data_query_scores) if non_data_query_scores else 0.0
            
            # Make classification decision
            if avg_data_score > avg_non_data_score:
                query_type = "data_query"
                confidence = float(avg_data_score)
                reasoning = f"Query matches data retrieval patterns (score: {avg_data_score:.3f})"
            else:
                query_type = "non_data_query" 
                confidence = float(avg_non_data_score)
                reasoning = f"Query matches business context patterns (score: {avg_non_data_score:.3f})"
            
            # Boost confidence based on keyword patterns
            data_keywords = ['show', 'get', 'list', 'count', 'sum', 'average', 'total', 'top', 'bottom', 'analyze', 'report', 'data', 'records', 'customers', 'orders', 'products', 'sales', 'revenue']
            non_data_keywords = ['policy', 'procedure', 'guidelines', 'help', 'support', 'contact', 'about', 'how to', 'what is', 'company', 'mission', 'values', 'terms', 'privacy']
            
            query_lower = query.lower()
            data_keyword_matches = sum(1 for keyword in data_keywords if keyword in query_lower)
            non_data_keyword_matches = sum(1 for keyword in non_data_keywords if keyword in query_lower)
            
            if data_keyword_matches > non_data_keyword_matches and query_type == "data_query":
                confidence = min(1.0, confidence + 0.1)
            elif non_data_keyword_matches > data_keyword_matches and query_type == "non_data_query":
                confidence = min(1.0, confidence + 0.1)
            
            logger.info(f"🎯 Classification: {query_type} (confidence: {confidence:.3f})")
            
            return {
                "query_type": query_type,
                "confidence": confidence,
                "reasoning": reasoning,
                "similar_examples": [ex["text"] for ex in similar_examples[:5]]
            }
            
        except Exception as e:
            logger.error(f"❌ Vectorized classification error: {e}")
            return self._keyword_classification(query)
    
    def _keyword_classification(self, query: str) -> Dict[str, Any]:
        """Fallback keyword-based classification"""
        try:
            query_lower = query.lower()
            
            # Strong data query indicators
            strong_data_keywords = ['show', 'list', 'get', 'fetch', 'retrieve', 'display', 'count', 'sum', 'average', 'total', 'top', 'bottom', 'analyze', 'report', 'calculate', 'find', 'search', 'query']
            
            # Strong non-data query indicators  
            strong_non_data_keywords = ['policy', 'procedure', 'guidelines', 'help', 'support', 'contact', 'about', 'mission', 'values', 'terms', 'privacy', 'how to', 'what is', 'explain', 'define']
            
            # Data-related nouns
            data_nouns = ['customers', 'orders', 'products', 'sales', 'revenue', 'users', 'transactions', 'inventory', 'data', 'records', 'metrics', 'KPIs', 'statistics']
            
            # Business context nouns
            business_nouns = ['company', 'organization', 'team', 'office', 'employee', 'handbook', 'documentation', 'manual', 'guide', 'process', 'workflow']
            
            # Count matches
            strong_data_matches = sum(1 for keyword in strong_data_keywords if keyword in query_lower)
            strong_non_data_matches = sum(1 for keyword in strong_non_data_keywords if keyword in query_lower)
            data_noun_matches = sum(1 for noun in data_nouns if noun in query_lower)
            business_noun_matches = sum(1 for noun in business_nouns if noun in query_lower)
            
            # Calculate scores
            data_score = strong_data_matches * 2 + data_noun_matches
            non_data_score = strong_non_data_matches * 2 + business_noun_matches
            
            # Make decision
            if data_score > non_data_score:
                query_type = "data_query"
                confidence = min(1.0, 0.6 + (data_score * 0.1))
                reasoning = f"Keyword analysis indicates data retrieval intent (data_score: {data_score}, non_data_score: {non_data_score})"
            elif non_data_score > data_score:
                query_type = "non_data_query"
                confidence = min(1.0, 0.6 + (non_data_score * 0.1))
                reasoning = f"Keyword analysis indicates business context intent (data_score: {data_score}, non_data_score: {non_data_score})"
            else:
                # Default to data query if uncertain
                query_type = "data_query"
                confidence = 0.5
                reasoning = "Ambiguous query, defaulting to data query"
            
            logger.info(f"🔎 Keyword classification: {query_type} (confidence: {confidence:.3f})")
            
            return {
                "query_type": query_type,
                "confidence": confidence,
                "reasoning": reasoning,
                "similar_examples": []
            }
            
        except Exception as e:
            logger.error(f"❌ Keyword classification error: {e}")
            return {
                "query_type": "data_query",
                "confidence": 0.5,
                "reasoning": f"Classification error: {str(e)}",
                "similar_examples": []
            }

# Factory function
async def create_vectorized_classifier() -> VectorizedQueryClassifier:
    """Create and initialize the vectorized classifier"""
    classifier = VectorizedQueryClassifier()
    await classifier.initialize()
    return classifier