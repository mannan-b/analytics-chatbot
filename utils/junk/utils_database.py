# 🎯 FIXED UTILS - DATABASE.PY (Synchronous functions for compatibility)

"""
Database utilities for Supabase connection
"""
import os
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables first!
load_dotenv()

from supabase import create_client, Client

logger = logging.getLogger(__name__)

_supabase_client = None

def init_database():
    """Initialize database connection (synchronous)"""
    global _supabase_client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        logger.error("❌ Missing Supabase credentials!")
        logger.info("💡 Please set SUPABASE_URL and SUPABASE_SERVICE_KEY in your .env file")
        # Don't raise error, just log it for better compatibility
        return False
    
    logger.info(f"🔗 Connecting to Supabase: {supabase_url[:50]}...")
    
    try:
        _supabase_client = create_client(supabase_url, supabase_key)
        logger.info("✅ Database connection initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize Supabase: {e}")
        return False

def get_supabase_client() -> Client:
    """Get Supabase client instance"""
    global _supabase_client
    if _supabase_client is None:
        # Try to initialize if not already done
        if not init_database():
            # Return None if can't initialize - let calling code handle it
            return None
    return _supabase_client

def list_available_tables() -> List[str]:
    """Get list of tables that actually exist (synchronous)"""
    try:
        client = get_supabase_client()
        if not client:
            logger.warning("⚠️ Database not initialized, using fallback tables")
            return ["customers", "orders", "products", "sales"]  # Fallback list
        
        # Common table names to test
        test_tables = [
            "customers", "products", "sales", "orders", "shopify_customers", "shopify_orders", 
            "shopify_products", "amazon_orders", "amazon_catalog_items", "financial_data", 
            "metrics_daily", "business_context", "user_sessions", "user_feedback", 
            "user_google_ads_accounts", "user_shopify_stores", "shopify_reports",
            "standalone_users", "profile", "employees", "inventory_ledger", "marketing_data"
        ]
        
        existing_tables = []
        for table in test_tables:
            try:
                result = client.table(table).select("*").limit(1).execute()
                existing_tables.append(table)
                logger.debug(f"✅ Table exists: {table}")
            except Exception as e:
                logger.debug(f"Skipping table {table}: {e}")
        
        # If no tables found, return fallback
        if not existing_tables:
            logger.warning("⚠️ No accessible tables found, using fallback list")
            return ["customers", "orders", "products", "sales"]
            
        return existing_tables
    except Exception as e:
        logger.error(f"❌ Failed to list tables: {e}")
        # Return fallback tables so system still works
        return ["customers", "orders", "products", "sales"]

def execute_simple_query(table_name: str, limit: int = 10) -> Dict[str, Any]:
    """Execute a simple query on a specific table (synchronous)"""
    try:
        client = get_supabase_client()
        if not client:
            logger.error("❌ Database client not available")
            return {
                "success": False,
                "error": "Database not initialized",
                "data": []
            }
            
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

def execute_sql_query(sql_query: str, params: Dict = None) -> Dict[str, Any]:
    """Execute a raw SQL query (if supported)"""
    try:
        client = get_supabase_client()
        if not client:
            return {
                "success": False,
                "error": "Database not initialized",
                "data": []
            }
        
        # For now, we'll parse simple SQL and convert to Supabase operations
        # This is a simplified implementation
        sql_lower = sql_query.lower().strip()
        
        if sql_lower.startswith('select'):
            # Extract table name from SQL
            import re
            table_match = re.search(r'from\s+(\w+)', sql_lower)
            if table_match:
                table_name = table_match.group(1)
                
                # Check if it's a COUNT query
                if 'count(' in sql_lower:
                    result = client.table(table_name).select("*", count="exact").execute()
                    return {
                        "success": True,
                        "data": [{"total": result.count}],
                        "count": 1
                    }
                else:
                    # Regular select
                    result = client.table(table_name).select("*").limit(50).execute()
                    return {
                        "success": True,
                        "data": result.data or [],
                        "count": len(result.data or [])
                    }
            else:
                raise Exception("Could not parse table name from SQL")
        else:
            raise Exception("Only SELECT queries are supported")
            
    except Exception as e:
        logger.error(f"❌ SQL query failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "data": []
        }

# Initialize on import if possible
try:
    init_database()
except:
    logger.info("Database will be initialized on first use")
    pass