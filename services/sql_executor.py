# ⚡ SQL EXECUTOR - Executes SQL in Supabase (FIXED IMPORT)

import logging
from typing import Dict, Any, Optional
from database_supabase_client import CorrectedSupabaseClient as SupabaseClient

logger = logging.getLogger(__name__)

class SQLExecutor:
    """
    SQL Executor that runs SQL queries in Supabase
    Part of your architecture: "Run the SQL in Supabase → Display the output data"
    """

    def __init__(self, supabase_client: SupabaseClient):
        self.supabase_client = supabase_client
        logger.info("✅ SQL Executor initialized")

    async def execute_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Execute SQL query in Supabase and return formatted results

        Args:
            sql_query: SQL query to execute

        Returns:
            Dict with success, data, and metadata
        """
        try:
            logger.info(f"⚡ Executing SQL: {sql_query[:100]}...")

            # Execute the query using Supabase client
            result = await self.supabase_client.execute_sql(sql_query)

            if result['success']:
                logger.info(f"✅ Query executed successfully - {result['count']} rows returned")
                return {
                    "success": True,
                    "data": result['data'],
                    "row_count": result['count'],
                    "sql_query": sql_query,
                    "columns": list(result['data'][0].keys()) if result['data'] else []
                }
            else:
                logger.error(f"❌ Query failed: {result.get('error')}")
                return {
                    "success": False,
                    "error": result.get('error'),
                    "sql_query": sql_query
                }

        except Exception as e:
            logger.error(f"❌ SQL execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "sql_query": sql_query
            }

    async def validate_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Validate SQL query without executing it

        Args:
            sql_query: SQL query to validate

        Returns:
            Dict with validation result
        """
        try:
            # Basic SQL validation
            dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE']
            upper_query = sql_query.upper()

            for keyword in dangerous_keywords:
                if keyword in upper_query:
                    return {
                        "valid": False,
                        "error": f"Query contains dangerous keyword: {keyword}"
                    }

            # Check for SELECT statement
            if not upper_query.strip().startswith('SELECT'):
                return {
                    "valid": False,
                    "error": "Only SELECT queries are allowed"
                }

            return {"valid": True}

        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }
