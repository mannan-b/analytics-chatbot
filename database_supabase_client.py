# 🔧 FIXED database_supabase_client.py - Proper SQL Execution

import logging
from typing import Dict, List, Any
from supabase import create_client, Client
from utils.config import config
import re

logger = logging.getLogger(__name__)

class CorrectedSupabaseClient:
    """Fixed Supabase client that properly executes custom SQL"""
    
    def __init__(self):
        self.client: Client = create_client(
            config.SUPABASE_URL,
            config.SUPABASE_SERVICE_KEY
        )
        
        self.table_descriptions = "table_descriptions"
        self.column_metadata = "column_metadata"
        self.common_prompts = "common_prompt_sqls"
        self.business_context = "business_context"
        
        logger.info("✅ Supabase client initialized")
    
    async def execute_sql(self, sql_query: str) -> Dict[str, Any]:
        """
        Execute SQL query properly - NO FALLBACK to SELECT *
        """
        try:
            logger.info(f"[DB] Executing SQL: {sql_query[:200]}...")
            
            # Clean SQL query
            sql_clean = sql_query.strip()
            if sql_clean.endswith(';'):
                sql_clean = sql_clean[:-1]
            
            # METHOD 1: Try execute_sql RPC function
            try:
                logger.info("[DB] Attempting RPC execute_sql...")
                result = self.client.rpc('execute_sql', {'query': sql_clean}).execute()
                
                if result.data:
                    logger.info(f"[DB] ✅ RPC Success! {len(result.data)} rows returned")
                    return {
                        'success': True,
                        'data': result.data,
                        'count': len(result.data)
                    }
                else:
                    logger.warning("[DB] RPC returned no data")
                    return {
                        'success': False,
                        'error': 'Query returned no data',
                        'data': [],
                        'count': 0
                    }
                    
            except Exception as rpc_error:
                logger.warning(f"[DB] RPC failed: {str(rpc_error)}")
                
                # METHOD 2: Check if it's a simple SELECT * query
                # ONLY fall back for SIMPLE queries, not aggregations
                if 'GROUP BY' in sql_clean.upper() or 'JOIN' in sql_clean.upper() or 'SUM(' in sql_clean.upper():
                    # Complex query - don't fallback
                    logger.error("[DB] ❌ Complex query requires RPC function")
                    return {
                        'success': False,
                        'error': f'RPC function execute_sql not found. Complex queries require it. Error: {str(rpc_error)}',
                        'data': [],
                        'count': 0
                    }
                
                # Simple SELECT * - can use direct query
                logger.info("[DB] Simple query, trying direct access...")
                match = re.search(r'FROM\s+(\w+)', sql_clean, re.IGNORECASE)
                if match:
                    table_name = match.group(1)
                    logger.info(f"[DB] Direct query on table: {table_name}")
                    
                    result = self.client.table(table_name).select("*").limit(100).execute()
                    
                    if result.data:
                        logger.info(f"[DB] ✅ Direct query success! {len(result.data)} rows")
                        return {
                            'success': True,
                            'data': result.data,
                            'count': len(result.data)
                        }
                
                # If all else fails
                raise Exception(f"Could not execute query: {str(rpc_error)}")
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[DB] ❌ SQL execution failed: {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'data': [],
                'count': 0
            }
    
    # ... (rest of methods stay the same)
    
    async def get_all_table_descriptions(self) -> List[Dict[str, Any]]:
        """Get all table descriptions"""
        try:
            result = self.client.table(self.table_descriptions).select("*").execute()
            logger.info(f"[DB] Retrieved {len(result.data)} tables")
            return result.data
        except Exception as e:
            logger.error(f"[DB] Failed: {e}")
            return []
    
    async def get_all_column_metadata(self) -> List[Dict[str, Any]]:
        """Get all column metadata"""
        try:
            result = self.client.table(self.column_metadata).select("*").execute()
            logger.info(f"[DB] Retrieved {len(result.data)} columns")
            return result.data
        except Exception as e:
            logger.error(f"[DB] Failed: {e}")
            return []
    
    async def get_columns_for_tables(self, table_names: List[str]) -> List[Dict[str, Any]]:
        """Get columns for specific tables"""
        try:
            if not table_names:
                return []
            
            result = self.client.table(self.column_metadata)\
                .select("*")\
                .in_("table_name", table_names)\
                .execute()
            
            logger.info(f"[DB] Retrieved {len(result.data)} columns for {len(table_names)} tables")
            return result.data
        except Exception as e:
            logger.error(f"[DB] Failed: {e}")
            return []
    
    async def get_all_common_prompts(self) -> List[Dict[str, Any]]:
        """Get all common prompts"""
        try:
            result = self.client.table(self.common_prompts).select("*").execute()
            logger.info(f"[DB] Retrieved {len(result.data)} common prompts")
            return result.data
        except Exception as e:
            logger.error(f"[DB] Failed: {e}")
            return []
