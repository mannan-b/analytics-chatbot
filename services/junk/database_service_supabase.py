# 🎯 FIXED DATABASE SERVICE - SUPABASE ONLY!

import os
import logging
from typing import Dict, List, Any, Optional
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self):
        self.supabase: Optional[Client] = None
        self.initialized = False
    
    def initialize(self):
        """Initialize Supabase client"""
        try:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
            
            if not url or not key:
                raise Exception("Missing Supabase credentials")
            
            self.supabase = create_client(url, key)
            self.initialized = True
            logger.info("✅ Supabase database initialized")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def get_table_descriptions(self) -> List[Dict[str, Any]]:
        """Get all table descriptions from table_descriptions table"""
        try:
            result = self.supabase.table("table_descriptions").select("*").execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get table descriptions: {e}")
            return []
    
    def get_column_metadata(self, table_name: str = None) -> List[Dict[str, Any]]:
        """Get column metadata from column_metadata table"""
        try:
            query = self.supabase.table("column_metadata").select("*")
            if table_name:
                query = query.eq("table_name", table_name)
            
            result = query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get column metadata: {e}")
            return []
    
    def get_business_context(self, query: str = None) -> List[Dict[str, Any]]:
        """Get business context from business_context table"""
        try:
            query_builder = self.supabase.table("business_context").select("*")
            
            # If query provided, search in relevant fields
            if query:
                query_lower = query.lower()
                # Use ilike for case-insensitive search
                query_builder = query_builder.or_(
                    f"context.ilike.%{query_lower}%,"
                    f"domain.ilike.%{query_lower}%,"
                    f"keywords.ilike.%{query_lower}%"
                )
            
            result = query_builder.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get business context: {e}")
            return []
    
    def search_sql_queries(self, natural_language_query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar SQL queries in the vector database"""
        try:
            # For now, let's assume there's a sql_queries table with embeddings
            # This would use vector similarity search in production
            query_lower = natural_language_query.lower()
            
            # Try to find matching queries using text search
            # In production, this would use vector similarity
            result = self.supabase.table("sql_queries").select("*").ilike("natural_language", f"%{query_lower}%").limit(limit).execute()
            
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to search SQL queries: {e}")
            return []
    
    def execute_sql_query(self, table_name: str, filters: Dict = None, limit: int = 50) -> Dict[str, Any]:
        """Execute SQL query on specified table"""
        try:
            query = self.supabase.table(table_name).select("*")
            
            # Apply filters if provided
            if filters:
                for key, value in filters.items():
                    query = query.eq(key, value)
            
            # Apply limit
            query = query.limit(limit)
            
            result = query.execute()
            
            return {
                "success": True,
                "data": result.data or [],
                "count": len(result.data or [])
            }
            
        except Exception as e:
            logger.error(f"Failed to execute query on {table_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    def get_available_tables(self) -> List[str]:
        """Get list of available tables from table_descriptions"""
        try:
            descriptions = self.get_table_descriptions()
            return [desc["table_name"] for desc in descriptions if desc.get("table_name")]
        except Exception as e:
            logger.error(f"Failed to get available tables: {e}")
            return []

# Global instance
_db_service = None

def get_database_service() -> DatabaseService:
    """Get the global database service instance"""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
        _db_service.initialize()
    return _db_service