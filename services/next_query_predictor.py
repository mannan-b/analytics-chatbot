import logging
from typing import List, Dict, Any
from services.llm_service import LLMService

logger = logging.getLogger(__name__)

class NextQueryPredictor:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def predict_next_queries(
        self,
        user_query: str,
        sql_query: str,
        result_data: List[Dict[str, Any]],
        limit: int = 3
    ) -> List[str]:
        try:
            if not result_data:
                return [
                    "What are the trends over time?",
                    "Show me the distribution by category",
                    "Compare with last period"
                ]
            
            columns = list(result_data[0].keys()) if result_data else []
            row_count = len(result_data)
            
            prompt = f"""Based on this query result, suggest {limit} follow-up questions.

USER QUERY: "{user_query}"
RESULT: {row_count} rows with columns: {', '.join(columns)}

Suggest {limit} natural follow-up questions (each 5-15 words):"""

            response = await self.llm_service.get_completion(prompt, max_tokens=150, temperature=0.7)
            
            suggestions = [
                line.strip().lstrip('123456789.-) ')
                for line in response.strip().split('\n')
                if line.strip() and len(line.strip()) > 10
            ][:limit]
            
            return suggestions if suggestions else [
                f"Show more details about {user_query}",
                "Break this down by time period",
                "Compare with previous results"
            ]
            
        except Exception as e:
            logger.error(f"Next query prediction failed: {e}")
            return [
                "Show me the top entries",
                "What's the breakdown by category?",
                "Compare this with other metrics"
            ]
