import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from supabase import Client

logger = logging.getLogger(__name__)

class QueryHistoryService:
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
        self.table_name = "query_history"
    
    async def save_query(
        self,
        user_id: str,
        query_text: str,
        sql_generated: Optional[str] = None,
        result_count: Optional[int] = None,
        classification: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        try:
            result = self.client.table(self.table_name).insert({
                'user_id': user_id,
                'query_text': query_text,
                'sql_generated': sql_generated,
                'result_count': result_count,
                'classification': classification,
                'confidence': confidence
            }).execute()
            return {'success': True}
        except Exception as e:
            logger.error(f"Failed to save query: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_recent_queries(
        self,
        user_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            result = self.client.table(self.table_name)\
                .select("*")\
                .eq("user_id", user_id)\
                .order("created_at", desc=True)\
                .limit(limit)\
                .execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []
    
    async def get_context_string(self, user_id: str, limit: int = 5) -> str:
        queries = await self.get_recent_queries(user_id, limit=limit)
        if not queries:
            return ""
        
        context_parts = ["Previous queries:"]
        for q in reversed(queries):
            context_parts.append(f"- {q['query_text']}")
        return "\n".join(context_parts)
