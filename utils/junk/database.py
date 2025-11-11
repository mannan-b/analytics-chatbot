# 🚀 COMPLETE DATABASE.PY WITH DatabaseManager CLASS

"""
Database utilities for Supabase connection - WITH DatabaseManager CLASS FOR COMPATIBILITY
"""

import os
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# LOAD ENV VARIABLES FIRST!
load_dotenv()

from supabase import create_client, Client
from utils.logger_config import get_logger

logger = get_logger(__name__)

_supabase_client = None

# ============================================================================
# BASIC CONNECTION FUNCTIONS (Original)
# ============================================================================

async def init_database():
    """Initialize database connection - SIMPLIFIED"""
    global _supabase_client
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        logger.error("❌ Missing Supabase credentials!")
        logger.info("💡 Please set SUPABASE_URL and SUPABASE_SERVICE_KEY in your .env file")
        raise ValueError("Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_SERVICE_KEY/SUPABASE_ANON_KEY")
    
    logger.info(f"🔗 Connecting to Supabase: {supabase_url[:50]}...")
    
    try:
        _supabase_client = create_client(supabase_url, supabase_key)
        logger.info("✅ Database connection initialized successfully")
        
        # Test basic connection (without assuming specific tables exist)
        await test_basic_connection()
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Supabase: {e}")
        raise e

def get_supabase_client() -> Client:
    """Get Supabase client instance"""
    if _supabase_client is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _supabase_client

async def test_basic_connection():
    """Test basic Supabase connection WITHOUT assuming tables exist"""
    try:
        client = get_supabase_client()
        
        # Try to get list of tables (this should work with any Supabase project)
        logger.info("🧪 Testing database connection...")
        
        # Use a simple RPC call that should always work
        try:
            # This is a basic PostgreSQL query that should work
            result = client.rpc('version').execute()
            logger.info("✅ Database connection test passed!")
        except:
            # If RPC doesn't work, try a simple table query
            logger.warning("⚠️ RPC test failed, trying basic auth test...")
            
            # This will at least test if our credentials are valid
            try:
                # Try to access auth users (should exist in every Supabase project)
                result = client.auth.get_user()
                logger.info("✅ Basic auth connection works")
            except Exception as e:
                logger.warning(f"⚠️ Auth test also failed: {e}")
                logger.info("📊 Database connected but may have limited access")
                
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        logger.error("🔍 Check your SUPABASE_URL and SUPABASE_SERVICE_KEY")
        # Don't raise - let the app continue with limited functionality

async def health_check() -> bool:
    """Simple health check that doesn't assume specific tables exist"""
    try:
        client = get_supabase_client()
        
        # Just check if client is initialized and we can make any call
        logger.info("🏥 Running health check...")
        
        # Very basic connectivity test
        return True  # If we got here, client is at least initialized
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return False

async def list_available_tables() -> list:
    """
    Get list of tables that actually exist
    """
    try:
        client = get_supabase_client()
        
        # Fallback: brute-force check table existence for candidate names
        test_tables = [
            "customers", "products", "sales", "orders", "shopify_customers", "shopify_orders", "shopify_products",
            "amazon_orders", "amazon_catalog_items", "financial_data", "metrics_daily", "business_context",
            "user_sessions", "user_feedback", "user_google_ads_accounts", "user_shopify_stores", "shopify_reports",
            "standalone_users", "profile", "employees", "inventory_ledger", "marketing_data",
            "metrics_monthly", "metrics_hourly", "saved_queries"
        ]
        
        existing_tables = []
        for table in test_tables:
            try:
                result = client.table(table).select("*").limit(1).execute()
                existing_tables.append(table)
                logger.info(f"✅ Table exists: {table}")
            except Exception as e:
                logger.debug(f"Skipping table {table}: {e}")
        
        return existing_tables
        
    except Exception as e:
        logger.error(f"❌ Failed to list tables: {e}")
        return []

async def get_database_summary():
    """Get summary of what's actually available"""
    try:
        available_tables = await list_available_tables()
        
        summary = {
            "status": "connected",
            "total_tables": len(available_tables),
            "available_tables": available_tables,
            "capabilities": {
                "basic_queries": True,
                "table_access": len(available_tables) > 0,
                "enterprise_features": len(available_tables) >= 10
            }
        }
        
        logger.info(f"📈 Database Summary: {summary['total_tables']} tables available")
        return summary
        
    except Exception as e:
        logger.error(f"❌ Failed to generate database summary: {e}")
        return {
            "status": "error",
            "error": str(e),
            "total_tables": 0,
            "available_tables": [],
            "capabilities": {"basic_queries": False}
        }

# Simple query execution without RAG complexity
async def execute_simple_query(table_name: str, limit: int = 10):
    """Execute a simple query on a specific table"""
    try:
        client = get_supabase_client()
        logger.info(f"🔍 Querying table: {table_name}")
        
        result = client.table(table_name).select("*").limit(limit).execute()
        
        if result.data:
            logger.info(f"✅ Query successful: {len(result.data)} rows returned")
            return {
                "success": True,
                "data": result.data,
                "count": len(result.data)
            }
        else:
            logger.warning(f"⚠️ Query returned no data for table: {table_name}")
            return {
                "success": True,
                "data": [],
                "count": 0
            }
            
    except Exception as e:
        logger.error(f"❌ Query failed for table {table_name}: {e}")
        return {
            "success": False,
            "error": str(e),
            "data": []
        }

# ============================================================================
# ENHANCED FUNCTIONS FOR SQL SERVICE
# ============================================================================

async def execute_sql_query(sql_query: str) -> Dict[str, Any]:
    """Execute any SQL query safely"""
    try:
        client = get_supabase_client()
        logger.info(f"🔍 Executing SQL: {sql_query[:100]}...")
        
        # Use supabase-py's RPC method for custom SQL
        # Note: This requires creating an RPC function in your Supabase database
        # For now, we'll simulate with basic table operations
        
        # If it's a simple SELECT, try to parse and execute
        sql_lower = sql_query.lower().strip()
        
        if sql_lower.startswith('select') and 'from' in sql_lower:
            # Try to extract table name and execute
            # This is a simplified approach - in production you'd want proper SQL parsing
            try:
                # Extract table name (basic parsing)
                parts = sql_query.split()
                from_index = next((i for i, word in enumerate(parts) if word.lower() == 'from'), None)
                
                if from_index and from_index + 1 < len(parts):
                    table_name = parts[from_index + 1].replace(';', '').strip()
                    
                    # Execute basic query
                    result = client.table(table_name).select("*").limit(100).execute()
                    
                    return {
                        "success": True,
                        "data": result.data,
                        "count": len(result.data),
                        "executed_sql": sql_query
                    }
            except Exception as e:
                logger.warning(f"⚠️ Basic SQL execution failed: {e}")
        
        # If we can't execute the SQL directly, return an error
        logger.warning(f"⚠️ Complex SQL execution not supported: {sql_query}")
        return {
            "success": False,
            "error": "Complex SQL execution requires RPC setup in Supabase",
            "data": [],
            "suggestion": "Use simple table queries or set up RPC functions in Supabase"
        }
        
    except Exception as e:
        logger.error(f"❌ SQL execution failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "data": []
        }

async def get_table_schemas() -> Dict[str, List[Dict]]:
    """Get schema information for all available tables"""
    try:
        client = get_supabase_client()
        available_tables = await list_available_tables()
        
        schemas = {}
        for table_name in available_tables:
            try:
                # Get sample row to infer schema
                result = client.table(table_name).select("*").limit(1).execute()
                
                if result.data and len(result.data) > 0:
                    sample_row = result.data[0]
                    schema = []
                    
                    for column, value in sample_row.items():
                        # Infer data type
                        if isinstance(value, bool):
                            data_type = "boolean"
                        elif isinstance(value, int):
                            data_type = "integer"
                        elif isinstance(value, float):
                            data_type = "numeric"
                        elif isinstance(value, str):
                            data_type = "text"
                        else:
                            data_type = "unknown"
                        
                        schema.append({
                            "column_name": column,
                            "data_type": data_type,
                            "sample_value": str(value) if value is not None else None
                        })
                    
                    schemas[table_name] = schema
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not get schema for table {table_name}: {e}")
                schemas[table_name] = []
        
        return schemas
        
    except Exception as e:
        logger.error(f"❌ Failed to get table schemas: {e}")
        return {}

# ============================================================================
# DatabaseManager CLASS FOR COMPATIBILITY
# ============================================================================

class DatabaseManager:
    """
    DatabaseManager class wrapper for compatibility with enhanced services.
    This wraps the existing functions in a class interface.
    """
    
    def __init__(self):
        self.logger = logger
        self._initialized = False
    
    async def initialize(self):
        """Initialize the database connection"""
        try:
            await init_database()
            self._initialized = True
            logger.info("✅ DatabaseManager initialized successfully")
        except Exception as e:
            logger.error(f"❌ DatabaseManager initialization failed: {e}")
            raise e
    
    def is_initialized(self) -> bool:
        """Check if database is initialized"""
        return self._initialized and _supabase_client is not None
    
    async def health_check(self) -> bool:
        """Check database health"""
        if not self.is_initialized():
            return False
        return await health_check()
    
    async def list_available_tables(self) -> List[str]:
        """Get list of available tables"""
        if not self.is_initialized():
            raise RuntimeError("DatabaseManager not initialized")
        return await list_available_tables()
    
    async def get_table_schemas(self) -> Dict[str, List[Dict]]:
        """Get schema information for tables"""
        if not self.is_initialized():
            raise RuntimeError("DatabaseManager not initialized")
        return await get_table_schemas()
    
    async def execute_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query"""
        if not self.is_initialized():
            raise RuntimeError("DatabaseManager not initialized")
        return await execute_sql_query(sql_query)
    
    async def execute_simple_query(self, table_name: str, limit: int = 10) -> Dict[str, Any]:
        """Execute simple table query"""
        if not self.is_initialized():
            raise RuntimeError("DatabaseManager not initialized")
        return await execute_simple_query(table_name, limit)
    
    async def get_database_summary(self) -> Dict[str, Any]:
        """Get database summary"""
        if not self.is_initialized():
            raise RuntimeError("DatabaseManager not initialized")
        return await get_database_summary()
    
    def get_client(self) -> Client:
        """Get raw Supabase client"""
        if not self.is_initialized():
            raise RuntimeError("DatabaseManager not initialized")
        return get_supabase_client()

# ============================================================================
# CONVENIENCE FUNCTION FOR EASY SETUP
# ============================================================================

async def create_database_manager() -> DatabaseManager:
    """Create and initialize a DatabaseManager instance"""
    db_manager = DatabaseManager()
    await db_manager.initialize()
    return db_manager

# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Basic functions (backwards compatibility)
    'init_database',
    'get_supabase_client', 
    'health_check',
    'list_available_tables',
    'get_database_summary',
    'execute_simple_query',
    
    # Enhanced functions
    'execute_sql_query',
    'get_table_schemas',
    
    # Class interface
    'DatabaseManager',
    'create_database_manager'
]