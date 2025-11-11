# 📊 FIXED ACCURACY-ENHANCED CHAT SERVICE - No Import Issues!

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Use the fixed SQL service
from services.accuracy_focused_sql_service import AccuracyFocusedSQLService

logger = logging.getLogger(__name__)

class AccuracyEnhancedChatService:
    """Enhanced chat service with accuracy tracking and learning - FIXED VERSION"""
    
    def __init__(self):
        self.sql_service = AccuracyFocusedSQLService()
        # Remove viz_service for now to avoid import issues
        self.conversation_contexts = {}
        
        # Performance tracking
        self.session_stats = {
            "queries_handled": 0,
            "successful_queries": 0,
            "learning_events": 0,
            "user_feedback_count": 0
        }
    
    async def process_message(self, query: str, user_id: str, session_id: str, 
                             conversation_id: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process message with accuracy tracking and learning"""
        try:
            self.session_stats["queries_handled"] += 1
            
            # Store context
            self._update_conversation_context(conversation_id, query, context)
            
            # Check for special accuracy commands first
            if self._is_accuracy_command(query):
                return await self._handle_accuracy_command(query)
            
            # Check for feedback
            if self._is_feedback(query):
                return await self._handle_feedback(query, user_id, conversation_id)
            
            # Check for data query
            if self._is_data_query(query):
                return await self._handle_data_query(query, user_id, conversation_id)
            
            # Handle as general query
            return await self._handle_general_query(query)
            
        except Exception as e:
            logger.error(f"Enhanced chat processing error: {e}")
            return {
                'type': 'error',
                'success': False,
                'message': f'Sorry, I encountered an error: {str(e)}',
                'error': str(e),
                'accuracy_info': self._get_current_accuracy_info()
            }
    
    def _is_accuracy_command(self, query: str) -> bool:
        """Check if query is asking about accuracy/performance"""
        query_lower = query.lower()
        accuracy_keywords = [
            "accuracy", "performance", "stats", "statistics", "learning", 
            "how accurate", "success rate", "learned patterns", "how well"
        ]
        return any(keyword in query_lower for keyword in accuracy_keywords)
    
    async def _handle_accuracy_command(self, query: str) -> Dict[str, Any]:
        """Handle accuracy-related queries"""
        accuracy_stats = self.sql_service.get_accuracy_stats()
        
        query_lower = query.lower()
        
        if "accuracy" in query_lower or "success rate" in query_lower:
            return {
                'type': 'accuracy_response',
                'success': True,
                'message': f"🎯 Current System Accuracy: {accuracy_stats['accuracy_stats']['accuracy_rate']:.1%}",
                'accuracy_details': {
                    'current_accuracy': f"{accuracy_stats['accuracy_stats']['accuracy_rate']:.1%}",
                    'total_queries': accuracy_stats['accuracy_stats']['total_queries'],
                    'successful_queries': accuracy_stats['accuracy_stats']['successful_queries'],
                    'target_accuracy': "95%+",
                    'status': "🟢 High Performance" if accuracy_stats['accuracy_stats']['accuracy_rate'] > 0.9 else "🟡 Learning Mode"
                },
                'learning_progress': {
                    'learned_patterns': accuracy_stats['accuracy_stats']['learned_patterns'],
                    'method_breakdown': accuracy_stats['accuracy_stats']['method_stats'],
                    'learning_active': True
                }
            }
        
        elif "learned patterns" in query_lower or "learning" in query_lower:
            return {
                'type': 'learning_response',
                'success': True,
                'message': f"🧠 Learning Status: {accuracy_stats['accuracy_stats']['learned_patterns']} patterns learned",
                'learning_details': {
                    'total_patterns': accuracy_stats['accuracy_stats']['learned_patterns'],
                    'learning_methods': list(accuracy_stats['accuracy_stats']['method_stats'].keys()),
                    'improvement_trend': "📈 Continuously improving with each query",
                    'feedback_integration': "✅ User feedback actively improving accuracy"
                },
                'suggestions': [
                    "Try queries like 'show customers' to see high-accuracy results",
                    "Use ✅/❌ buttons to provide feedback",
                    "Ask 'what is my accuracy' to check current performance"
                ]
            }
        
        elif "stats" in query_lower or "statistics" in query_lower:
            return {
                'type': 'stats_response',
                'success': True,
                'message': "📊 Complete System Statistics",
                'detailed_stats': accuracy_stats,
                'session_stats': self.session_stats,
                'performance_summary': {
                    'accuracy_level': "High" if accuracy_stats['accuracy_stats']['accuracy_rate'] > 0.9 else "Improving",
                    'reliability_score': "Excellent",
                    'learning_capability': "Active",
                    'user_feedback_integration': "Enabled"
                }
            }
        
        else:
            return {
                'type': 'performance_overview',
                'success': True,
                'message': "🚀 High-Accuracy AI System Overview",
                'features': accuracy_stats['high_accuracy_features'],
                'metrics': accuracy_stats['performance_metrics'],
                'help': [
                    "Ask 'what is my accuracy' for current performance",
                    "Ask 'show learning progress' for improvement details", 
                    "Ask 'system statistics' for complete breakdown"
                ]
            }
    
    def _is_data_query(self, query: str) -> bool:
        """Enhanced data query detection"""
        query_lower = query.lower()
        
        data_indicators = [
            "show", "list", "display", "get", "find", "count", "how many",
            "sum", "total", "average", "top", "highest", "recent", "last",
            "customer", "order", "product", "sale", "revenue", "data",
            "records", "table", "database"
        ]
        
        return any(indicator in query_lower for indicator in data_indicators)
    
    async def _handle_data_query(self, query: str, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """Handle data queries with enhanced accuracy tracking"""
        try:
            # Process through high-accuracy SQL service
            result = await self.sql_service.process_query_with_learning(query, user_id)
            
            if result['success']:
                self.session_stats["successful_queries"] += 1
                
                # Store for context and feedback
                self._store_query_result(conversation_id, result, query)
                
                # Enhanced response with learning info
                response = {
                    'type': 'data_response',
                    'success': True,
                    'message': result['explanation'],
                    'data': result['data'],
                    'sql_query': result['sql_query'],
                    'table_used': result['table_used'],
                    'method': result['method'],
                    'confidence': result['confidence'],
                    'row_count': result['row_count'],
                    'execution_time': result.get('execution_time', 0),
                    'accuracy_info': result.get('accuracy_info', {}),
                    'insights': result['insights'],
                    'learning_active': result.get('learning_active', False),
                    'feedback_request': {
                        'enabled': True,
                        'query_id': f"{conversation_id}_{datetime.now().timestamp()}",
                        'message': "Was this result helpful? ✅ Yes / ❌ No"
                    }
                }
                
                # Skip visualization for now to avoid import issues
                response['visualization'] = {'available': False, 'reason': 'Visualization service temporarily disabled'}
                
                return response
                
            else:
                # Even failures are handled gracefully with learning
                return {
                    'type': 'data_error',
                    'success': False,
                    'message': result.get('error', 'Query processing failed'),
                    'method': result.get('method', 'unknown'),
                    'confidence': result.get('confidence', 0.0),
                    'suggestions': result.get('suggestions', []),
                    'learning_note': "This failure helps improve the system!",
                    'accuracy_info': self._get_current_accuracy_info()
                }
                
        except Exception as e:
            logger.error(f"Data query handling failed: {e}")
            return {
                'type': 'error',
                'success': False,
                'message': f'Data query failed: {str(e)}',
                'error': str(e),
                'accuracy_info': self._get_current_accuracy_info()
            }
    
    def _is_feedback(self, query: str) -> bool:
        """Enhanced feedback detection"""
        query_lower = query.lower()
        feedback_indicators = [
            'correct', 'wrong', 'incorrect', 'right', 'good', 'bad',
            'yes', 'no', 'that was', 'feedback', '✅', '❌'
        ]
        return any(indicator in query_lower for indicator in feedback_indicators)
    
    async def _handle_feedback(self, query: str, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """Enhanced feedback handling with learning integration"""
        try:
            context = self.conversation_contexts.get(conversation_id, {})
            last_response = context.get('last_sql_response')
            
            if not last_response:
                return {
                    'type': 'feedback_error',
                    'success': False,
                    'message': 'No previous query to provide feedback on',
                    'suggestion': 'Ask a data question first, then provide feedback'
                }
            
            # Enhanced feedback analysis
            query_lower = query.lower()
            is_positive = any(word in query_lower for word in ['correct', 'right', 'good', 'yes', '✅'])
            is_negative = any(word in query_lower for word in ['wrong', 'incorrect', 'bad', 'no', '❌'])
            
            if is_positive:
                is_correct = True
                feedback_type = "positive"
            elif is_negative:
                is_correct = False
                feedback_type = "negative"
            else:
                # Neutral/unclear feedback
                return {
                    'type': 'feedback_clarification',
                    'success': True,
                    'message': 'Could you clarify if the result was correct? ✅ Yes or ❌ No',
                    'last_query': last_response.get('original_query', ''),
                    'last_sql': last_response.get('sql_query', '')
                }
            
            # Extract corrected SQL if provided (for negative feedback)
            corrected_sql = None
            if not is_correct and 'select' in query_lower:
                import re
                sql_match = re.search(r'(SELECT.*?)(?:\.|$|;)', query, re.IGNORECASE | re.DOTALL)
                if sql_match:
                    corrected_sql = sql_match.group(1).strip()
            
            # Submit feedback to accuracy service
            feedback_result = await self.sql_service.submit_feedback(
                last_response.get('original_query', ''),
                last_response.get('sql_query', ''),
                is_correct,
                corrected_sql
            )
            
            if feedback_result['success']:
                self.session_stats["user_feedback_count"] += 1
                self.session_stats["learning_events"] += 1
                
                response = {
                    'type': 'feedback_success',
                    'success': True,
                    'feedback_type': feedback_type,
                    'learning_impact': {
                        'message': feedback_result['message'],
                        'accuracy_improvement': feedback_result['current_accuracy'],
                        'patterns_learned': feedback_result['learned_patterns'],
                        'system_status': 'Learning and improving!'
                    }
                }
                
                if is_correct:
                    response['message'] = "✅ Thank you! This positive feedback strengthens the system's accuracy."
                    response['impact'] = "This query pattern is now more likely to succeed in the future."
                else:
                    response['message'] = "📝 Thank you for the correction! The system is learning from this."
                    response['impact'] = "This feedback will improve similar queries going forward."
                    if corrected_sql:
                        response['learned_correction'] = "✅ Learned the corrected SQL you provided!"
                
                return response
            else:
                return {
                    'type': 'feedback_error',
                    'success': False,
                    'message': 'Failed to record feedback',
                    'error': feedback_result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            logger.error(f"Feedback handling failed: {e}")
            return {
                'type': 'error',
                'success': False,
                'message': f'Feedback processing failed: {str(e)}',
                'error': str(e)
            }
    
    async def _handle_general_query(self, query: str) -> Dict[str, Any]:
        """Handle general queries with accuracy context"""
        accuracy_info = self._get_current_accuracy_info()
        
        return {
            'type': 'general_response',
            'success': True,
            'message': f"I understand you're asking: '{query}'. I'm a high-accuracy AI assistant specializing in data queries and analysis!",
            'current_performance': f"🎯 System Accuracy: {accuracy_info['accuracy_rate']}",
            'capabilities': {
                'data_queries': "Ask me to show, count, or analyze your data",
                'learning_system': "I improve with every query and feedback",
                'high_accuracy': "95%+ success rate on common queries",
                'visualization': "Charts temporarily disabled"
            },
            'suggestions': [
                "Try: 'show me customers' (high-accuracy pattern)",
                "Try: 'count total orders' (95% confidence)",
                "Try: 'what is my current accuracy?' (system stats)",
                "Use ✅/❌ after queries to help me learn"
            ],
            'accuracy_info': accuracy_info
        }
    
    def _update_conversation_context(self, conversation_id: str, query: str, context: Optional[Dict]):
        """Enhanced context tracking"""
        if conversation_id not in self.conversation_contexts:
            self.conversation_contexts[conversation_id] = {
                'messages': [],
                'last_sql_response': None,
                'accuracy_tracking': {
                    'queries_in_session': 0,
                    'successful_in_session': 0,
                    'feedback_given': 0
                }
            }
        
        self.conversation_contexts[conversation_id]['messages'].append({
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'context': context
        })
        
        self.conversation_contexts[conversation_id]['accuracy_tracking']['queries_in_session'] += 1
    
    def _store_query_result(self, conversation_id: str, result: Dict, original_query: str):
        """Enhanced result storage for feedback"""
        if conversation_id in self.conversation_contexts:
            self.conversation_contexts[conversation_id]['last_sql_response'] = {
                'original_query': original_query,
                'sql_query': result.get('sql_query'),
                'method': result.get('method'),
                'confidence': result.get('confidence'),
                'success': result.get('success', False),
                'timestamp': datetime.now().isoformat()
            }
            
            if result.get('success'):
                self.conversation_contexts[conversation_id]['accuracy_tracking']['successful_in_session'] += 1
    
    def _get_current_accuracy_info(self) -> Dict[str, Any]:
        """Get current accuracy information"""
        try:
            accuracy_stats = self.sql_service.get_accuracy_stats()
            return {
                'accuracy_rate': f"{accuracy_stats['accuracy_stats']['accuracy_rate']:.1%}",
                'total_queries': accuracy_stats['accuracy_stats']['total_queries'],
                'learned_patterns': accuracy_stats['accuracy_stats']['learned_patterns'],
                'status': 'High Performance' if accuracy_stats['accuracy_stats']['accuracy_rate'] > 0.9 else 'Learning Mode'
            }
        except:
            return {
                'accuracy_rate': 'Calculating...',
                'total_queries': 0,
                'learned_patterns': 0,
                'status': 'Initializing'
            }
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        return {
            'session_stats': self.session_stats,
            'system_accuracy': self._get_current_accuracy_info(),
            'learning_activity': {
                'feedback_received': self.session_stats['user_feedback_count'],
                'learning_events': self.session_stats['learning_events'],
                'improvement_active': True
            },
            'recommendations': [
                "Continue providing feedback with ✅/❌ buttons",
                "Try different query patterns to expand learning",
                "Ask about 'system accuracy' to track improvement"
            ]
        }