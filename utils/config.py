# 🔧 FIXED CONFIG.PY - Correct Supabase Table Names + Visualization + ALLOWS EXTRA ENV VARS

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class Config(BaseSettings):
    """Configuration for Text-to-SQL System with CORRECT table names"""
    
    # API Keys - You have all three!
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str = Field(..., env="ANTHROPIC_API_KEY") 
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    
    # Supabase Configuration
    SUPABASE_URL: str = Field(..., env="SUPABASE_URL")
    SUPABASE_SERVICE_KEY: str = Field(..., env="SUPABASE_SERVICE_KEY")
    
    # LLM Configuration (Your Architecture Specs)
    PRIMARY_LLM: str = "claude-3-5-sonnet-20241022"  # Claude 3.5 Sonnet as primary
    FALLBACK_LLM_1: str = "gpt-4"  # OpenAI as fallback
    FALLBACK_LLM_2: str = "gemini-pro"  # Gemini as final fallback
    
    # Embedding Configuration (Exact from your PDF)
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # OpenAI text-embedding-3-small
    EMBEDDING_DIMENSIONS: int = 512  # 512 dimensions as specified
    
    # Vector Search Configuration (From your architecture)
    COSINE_SIMILARITY_THRESHOLD: float = 0.7  # 70% threshold as specified
    MAX_VECTOR_RESULTS: int = 10  # Reasonable limit
    
    # CORRECTED Supabase Table Names (Your clarification)
    TABLE_DESCRIPTIONS_TABLE: str = "table_descriptions"  # ✅ CORRECTED
    COLUMN_METADATA_TABLE: str = "column_metadata"       # ✅ CORRECT
    COMMON_PROMPTS_TABLE: str = "common_prompt_sqls"     # ✅ KEEPING AS IS
    BUSINESS_CONTEXT_TABLE: str = "business_context"     # ✅ CORRECT
    
    # Prompt Classification
    DATA_QUESTION_THRESHOLD: float = 0.8  # Confidence threshold for data questions
    
    # Business Context Configuration
    CONTEXT_CHUNK_SIZE: int = 300  # 300 characters as specified in architecture
    
    # Visualization Configuration (Added)
    ENABLE_VISUALIZATION: bool = True  # Enable charts after SQL results
    CHARTS_DIRECTORY: str = "static/charts"
    
    # FastAPI Configuration
    API_TITLE: str = "Text-to-SQL System"
    API_DESCRIPTION: str = "Exact implementation of your PDF architecture with visualization"
    API_VERSION: str = "1.0.0"
    API_PORT: int = 8000
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # ✅ FIXES THE PYDANTIC VALIDATION ERROR - Allows extra env variables

# Global config instance
config = Config()

# LLM Provider Configuration
LLM_CONFIGS = {
    "claude-3-5-sonnet-20241022": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4000,
        "temperature": 0.1,
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    "gpt-4": {
        "provider": "openai", 
        "model": "gpt-4",
        "max_tokens": 4000,
        "temperature": 0.1,
        "api_key_env": "OPENAI_API_KEY"
    },
    "gemini-pro": {
        "provider": "gemini",
        "model": "gemini-pro",
        "max_tokens": 4000,
        "temperature": 0.1,
        "api_key_env": "GEMINI_API_KEY"
    }
}

# Prompt Templates (Exact from your PDF architecture)
PROMPT_TEMPLATES = {
    "data_question": """Write a supabase SQL query to answer <User Prompt>. Use the relevant tables and columns from this list only
Tables: <Table Metadata>
Columns: <Column Metadata>
Here are some example prompt and SQL query that might be matching to user's prompt. Learn from this Prompt SQL combinations
<Common Prompt SQLs>""",
    
    "non_data_question": """Answer the user's question <User Prompt> using this <Business Context>""",
    
    "classification": """Classify the following user prompt into one of these categories:
1. Data Question - Asking for data from database (queries, analytics, reports)
2. Non-Data Question - General business questions, explanations, definitions
3. Invalid Question - Unclear, inappropriate, or off-topic questions

User Prompt: {user_prompt}

Respond with only the number (1, 2, or 3) and confidence score (0.0-1.0):"""
}

# CORRECTED Database Schema (For the 4 required tables)
DATABASE_SCHEMA = {
    "table_descriptions": {  # ✅ CORRECTED NAME
        "table_name": "TEXT PRIMARY KEY",
        "table_description": "TEXT NOT NULL",
        "vector_embedding": "VECTOR(512)"
    },
    "column_metadata": {  # ✅ CORRECT
        "id": "SERIAL PRIMARY KEY",
        "table_name": "TEXT NOT NULL",
        "column_name": "TEXT NOT NULL", 
        "column_description": "TEXT NOT NULL",
        "vector_embedding": "VECTOR(512)"
    },
    "common_prompt_sqls": {  # ✅ KEEPING AS IS
        "id": "SERIAL PRIMARY KEY",
        "prompt": "TEXT NOT NULL",
        "sql_query": "TEXT NOT NULL",
        "vector_embedding": "VECTOR(512)"
    },
    "business_context": {  # ✅ CORRECT
        "id": "SERIAL PRIMARY KEY",
        "document_piece": "TEXT NOT NULL CHECK (length(document_piece) <= 300)",
        "vector_embedding": "VECTOR(512)"
    }
}