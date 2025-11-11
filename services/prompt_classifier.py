# 🔥 BULLETPROOF PROMPT CLASSIFIER - NEVER FAILS

import os
import json
import logging
from typing import Dict, Tuple, List, Optional
import re

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

logger = logging.getLogger(__name__)

class BulletproofPromptClassifier:
    """
    Ultra-reliable prompt classifier that NEVER marks queries as invalid
    unless they're truly garbage (empty, gibberish, etc.)
    """
    
    def __init__(self):
        self.gemini_client = None
        self._init_gemini()
        
        # Data query keywords - very broad to catch everything
        self.data_keywords = {
            'show', 'list', 'display', 'get', 'find', 'search', 'count', 'total', 'sum', 
            'average', 'max', 'min', 'top', 'bottom', 'highest', 'lowest', 'most', 'least',
            'sales', 'revenue', 'customers', 'orders', 'products', 'users', 'data',
            'records', 'rows', 'table', 'database', 'report', 'analytics', 'metrics',
            'performance', 'trends', 'analysis', 'statistics', 'breakdown', 'summary',
            'how many', 'what is the', 'tell me about', 'give me', 'i want', 'i need'
        }
        
        # Business question keywords
        self.business_keywords = {
            'how to', 'why', 'what is', 'explain', 'help', 'advice', 'recommend', 
            'suggest', 'improve', 'strategy', 'best practice', 'guidance', 'should i',
            'can you help', 'what does', 'meaning', 'definition', 'concept', 'understand'
        }
        
        # Truly invalid patterns (very restrictive)
        self.invalid_patterns = [
            r'^[^a-zA-Z0-9]*$',  # Only special characters
            r'^.{0,2}$',         # Less than 3 characters
            r'^(test|hello|hi|hey)$',  # Basic test words
        ]
    
    def _init_gemini(self):
        """Initialize Gemini if available"""
        try:
            if HAS_GEMINI and os.getenv("GEMINI_API_KEY"):
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self.gemini_client = genai.GenerativeModel("gemini-1.5-flash")
                logger.info("[CLASSIFIER] Gemini initialized")
            else:
                logger.info("[CLASSIFIER] Using keyword-based classification only")
        except Exception as e:
            logger.warning(f"[CLASSIFIER] Gemini init failed: {e}")
    
    async def classify_prompt(self, user_prompt: str) -> Tuple[str, float, Dict]:
        """
        Classify prompt with bulletproof logic:
        1. First check if truly invalid (very strict)
        2. Try LLM classification if available
        3. Fall back to smart keyword classification
        4. Default to data_question if unsure (better than invalid)
        """
        
        user_prompt = user_prompt.strip()
        logger.info(f"[CLASSIFIER] Classifying: {user_prompt[:50]}...")
        
        # Step 1: Check for truly invalid inputs (very restrictive)
        if self._is_truly_invalid(user_prompt):
            return "invalid_question", 0.95, {
                "method": "pattern_matching",
                "reason": "Empty or gibberish input"
            }
        
        # Step 2: Try LLM classification if available
        if self.gemini_client:
            try:
                result = await self._classify_with_llm(user_prompt)
                if result[1] > 0.7:  # If LLM is confident, use it
                    return result
            except Exception as e:
                logger.warning(f"[CLASSIFIER] LLM classification failed: {e}")
        
        # Step 3: Smart keyword classification (never returns invalid)
        return self._classify_with_keywords(user_prompt)
    
    def _is_truly_invalid(self, prompt: str) -> bool:
        """
        Only mark as invalid if TRULY garbage
        - Empty or whitespace only
        - Only special characters
        - Clearly test inputs
        """
        if not prompt or len(prompt.strip()) < 3:
            return True
            
        # Check against invalid patterns
        for pattern in self.invalid_patterns:
            if re.match(pattern, prompt.lower().strip()):
                return True
                
        return False
    
    async def _classify_with_llm(self, user_prompt: str) -> Tuple[str, float, Dict]:
        """
        Use LLM for classification with a SIMPLE, CLEAR prompt
        """
        
        # Ultra-simple prompt that works
        classification_prompt = f"""Classify this user query into EXACTLY one category:

1. DATA_QUESTION - User wants data from a database (numbers, lists, reports, analytics)
2. BUSINESS_QUESTION - User wants advice, explanations, or general business help

Examples:
- "show me sales data" = DATA_QUESTION
- "how many customers do we have" = DATA_QUESTION  
- "what is customer retention" = BUSINESS_QUESTION
- "how to improve sales" = BUSINESS_QUESTION

Query: "{user_prompt}"

Respond with ONLY: DATA_QUESTION or BUSINESS_QUESTION"""
        
        try:
            response = self.gemini_client.generate_content(classification_prompt)
            result_text = response.text.strip().upper()
            
            # Parse LLM response
            if "DATA_QUESTION" in result_text:
                return "data_question", 0.85, {
                    "method": "llm_classification",
                    "llm_response": result_text,
                    "reason": "LLM identified as data query"
                }
            elif "BUSINESS_QUESTION" in result_text:
                return "non_data_question", 0.85, {
                    "method": "llm_classification", 
                    "llm_response": result_text,
                    "reason": "LLM identified as business query"
                }
            else:
                # LLM gave unclear response, fall back to keywords
                logger.warning(f"[CLASSIFIER] Unclear LLM response: {result_text}")
                return self._classify_with_keywords(user_prompt)
                
        except Exception as e:
            logger.error(f"[CLASSIFIER] LLM classification error: {e}")
            return self._classify_with_keywords(user_prompt)
    
    def _classify_with_keywords(self, user_prompt: str) -> Tuple[str, float, Dict]:
        """
        Smart keyword-based classification that NEVER returns invalid
        """
        
        prompt_lower = user_prompt.lower()
        
        # Count keyword matches
        data_score = sum(1 for keyword in self.data_keywords 
                        if keyword in prompt_lower)
        business_score = sum(1 for keyword in self.business_keywords 
                           if keyword in prompt_lower)
        
        # Smart classification logic
        if data_score > business_score:
            confidence = min(0.9, 0.6 + (data_score * 0.1))
            classification = "data_question"
            reason = f"Found {data_score} data indicators"
            
        elif business_score > data_score:
            confidence = min(0.9, 0.6 + (business_score * 0.1))
            classification = "non_data_question"
            reason = f"Found {business_score} business indicators"
            
        else:
            # When unsure, default to data_question (safer than invalid)
            # Most business users want data more than advice
            classification = "data_question"
            confidence = 0.65
            reason = "No clear indicators - defaulting to data question"
        
        return classification, confidence, {
            "method": "keyword_classification",
            "data_score": data_score,
            "business_score": business_score, 
            "reason": reason,
            "data_keywords_found": [kw for kw in self.data_keywords if kw in prompt_lower],
            "business_keywords_found": [kw for kw in self.business_keywords if kw in prompt_lower]
        }
    
    def get_classification_stats(self) -> Dict:
        """Get classifier statistics and health info"""
        return {
            "llm_available": self.gemini_client is not None,
            "data_keywords_count": len(self.data_keywords),
            "business_keywords_count": len(self.business_keywords),
            "invalid_patterns_count": len(self.invalid_patterns),
            "default_behavior": "data_question (safer than invalid)",
            "confidence_range": "0.65-0.95"
        }

# For easy import
PromptClassifier = BulletproofPromptClassifier