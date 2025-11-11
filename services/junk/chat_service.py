# FIXED CHAT SERVICE - Streamlined flow

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from services.sql_service import AdvancedSQLService
from services.visualization_service import SmartVisualizationService
from services.llm_service import LLMService

from utils.logger_config import get_logger

logger = get_logger(__name__)

class EnhancedChatService:
    def __init__(self, llm_service: LLMService = None, doc_service=None, vector_service=None):
        self.llm_service = llm_service or LLMService()
        self.sql_service = AdvancedSQLService(self.llm_service)
        self.viz_service = SmartVisualizationService()
        
        self.conversation_contexts = {}
        self.logger = logger
    
    async def process_message(self, query: str, user_id: str, session_id: str, 
                             conversation_id: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Main message processing - streamlined flow"""
        try:
            # Store context
            self._update_conversation_context(conversation_id, query, context)
            
            # Check for feedback first
            if self._is_feedback(query):
                return await self._handle_feedback(query, user_id, conversation_id)
            
            # Process query through SQL service (handles classification internally)
            result = await self.sql_service.classify_and_convert_query(query, user_id)
            
            if not result['success']:
                return {
                    'type': 'error',
                    'success': False,
                    'message': result.get('error', 'Query processing failed'),
                    'suggestions': result.get('suggestions', ['Try rephrasing your question'])
                }
            
            # Handle different query types
            query_type = result.get('query_type')
            
            if query_type == 'simple_answer':
                return {
                    'type': 'simple_response',
                    'success': True,
                    'message': result['message'],
                    'answer': result['answer']
                }
            
            elif query_type == 'business_query':
                return {
                    'type': 'business_response',
                    'success': True,
                    'message': result['message'],
                    'answer': result['answer'],
                    'source': result.get('source', 'business_context')
                }
            
            elif query_type == 'data_query':
                # Store for context WITH THE ORIGINAL QUERY
                result['user_query'] = query  # Add this!
                self._store_query_result(conversation_id, result)
                
                response = {
                    'type': 'data_response',
                    'success': True,
                    'message': f"Found {len(result['data'])} records",
                    'data': result['data'][:50],  # Limit display
                    'total_records': len(result['data']),
                    'sql_query': {
                        'sql': result['sql_query'],
                        'table': result['table_used']
                    },
                    'analysis': result['analysis'],
                    'feedback_id': result['feedback_id']
                }
                
                # Create visualization if recommended
                if result.get('visualization') and result['visualization']['type'] != 'none':
                    try:
                        viz_result = await self.viz_service.create_smart_visualization(
                            result['data'][:50],
                            query_context=query,
                            chart_title=f"Data from {result['table_used']}"
                        )
                        
                        if viz_result.get('success'):
                            response['visualization'] = {
                                'chart_type': viz_result['chart_type'],
                                'chart_url': viz_result.get('chart_url'),
                                'chart_base64': viz_result.get('chart_base64'),
                                'insights': viz_result.get('insights', []),
                                'recommendation': result['visualization']['reason']
                            }
                    except Exception as e:
                        logger.warning(f"Visualization failed: {e}")
                
                return response
            
            else:
                return {
                    'type': 'general_response',
                    'success': True,
                    'message': 'Query processed',
                    'result': result
                }
                
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            return {
                'type': 'error',
                'success': False,
                'message': 'Sorry, I encountered an error processing your request.',
                'error': str(e)
            }
    
    def _is_feedback(self, query: str) -> bool:
        """Check if query is feedback"""
        query_lower = query.lower()
        feedback_indicators = ['correct', 'wrong', 'incorrect', 'that was', 'feedback']
        return any(indicator in query_lower for indicator in feedback_indicators)
    
    async def _handle_feedback(self, query: str, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """Handle feedback on previous response"""
        context = self.conversation_contexts.get(conversation_id, {})
        last_response = context.get('last_sql_response')
        
        if not last_response:
            return {
                'type': 'feedback_response',
                'success': False,
                'message': 'No previous query to provide feedback on'
            }
        
        # Determine if positive or negative
        query_lower = query.lower()
        is_correct = any(word in query_lower for word in ['correct', 'right', 'good', 'yes'])
        
        # Extract corrected SQL if provided
        correct_sql = None
        if not is_correct and 'select' in query_lower:
            # Try to extract SQL from feedback
            import re
            sql_match = re.search(r'(SELECT.*?)(?:\.|$)', query, re.IGNORECASE | re.DOTALL)
            if sql_match:
                correct_sql = sql_match.group(1).strip()
        
        # Submit feedback
        feedback_result = await self.sql_service.submit_feedback(
            last_response['feedback_id'],
            last_response.get('user_query', ''),
            last_response.get('sql_query', ''),
            is_correct,
            correct_sql,
            user_id
        )
        
        if feedback_result['success']:
            stats = self.sql_service.get_feedback_stats()
            
            return {
                'type': 'feedback_response',
                'success': True,
                'message': '✅ Thank you for your feedback! I\'m learning from it.' if is_correct else '📝 Thank you! I\'ve noted this and will improve.',
                'learning_impact': 'Your feedback helps me get better at understanding queries',
                'stats': {
                    'accuracy_rate': stats['accuracy_rate'] * 100,
                    'total_feedback': stats['total_feedback'],
                    'learned_patterns': stats['learned_patterns']
                }
            }
        
        return {
            'type': 'feedback_response',
            'success': False,
            'message': 'Failed to record feedback'
        }
    
    def _update_conversation_context(self, conversation_id: str, query: str, context: Optional[Dict]):
        """Update conversation context"""
        if conversation_id not in self.conversation_contexts:
            self.conversation_contexts[conversation_id] = {
                'messages': [],
                'last_sql_response': None
            }
        
        self.conversation_contexts[conversation_id]['messages'].append({
            'query': query,
            'timestamp': datetime.now().isoformat()
        })
    
    def _store_query_result(self, conversation_id: str, result: Dict):
        """Store query result for feedback"""
        if conversation_id in self.conversation_contexts:
            self.conversation_contexts[conversation_id]['last_sql_response'] = {
                'feedback_id': result.get('feedback_id'),
                'user_query': result.get('user_query'),  # This might be missing!
                'sql_query': result.get('sql_query')
            }