# LOCAL FEEDBACK DATABASE - SQLite

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
import os

class FeedbackDatabase:
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with feedback table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_id TEXT UNIQUE NOT NULL,
                user_query TEXT NOT NULL,
                generated_sql TEXT NOT NULL,
                is_correct INTEGER NOT NULL,
                correct_sql TEXT,
                user_id TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create learned patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_pattern TEXT UNIQUE NOT NULL,
                sql_query TEXT NOT NULL,
                times_used INTEGER DEFAULT 1,
                last_used TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_feedback(self, feedback_id: str, user_query: str, generated_sql: str,
                    is_correct: bool, correct_sql: Optional[str] = None, 
                    user_id: Optional[str] = None) -> bool:
        """Add feedback to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO feedback 
                (feedback_id, user_query, generated_sql, is_correct, correct_sql, user_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback_id,
                user_query,
                generated_sql,
                1 if is_correct else 0,
                correct_sql,
                user_id,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"Error adding feedback: {e}")
            return False
    
    def add_learned_pattern(self, query_pattern: str, sql_query: str):
        """Add or update learned pattern"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if pattern exists
            cursor.execute('SELECT id, times_used FROM learned_patterns WHERE query_pattern = ?', 
                          (query_pattern,))
            result = cursor.fetchone()
            
            if result:
                # Update existing pattern
                cursor.execute('''
                    UPDATE learned_patterns 
                    SET sql_query = ?, times_used = ?, last_used = ?
                    WHERE query_pattern = ?
                ''', (sql_query, result[1] + 1, datetime.now().isoformat(), query_pattern))
            else:
                # Insert new pattern
                cursor.execute('''
                    INSERT INTO learned_patterns (query_pattern, sql_query, last_used)
                    VALUES (?, ?, ?)
                ''', (query_pattern, sql_query, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"Error adding learned pattern: {e}")
            return False
    
    def get_learned_pattern(self, query_pattern: str) -> Optional[str]:
        """Get learned SQL for a query pattern"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT sql_query FROM learned_patterns WHERE query_pattern = ?',
                          (query_pattern,))
            result = cursor.fetchone()
            
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            print(f"Error getting learned pattern: {e}")
            return None
    
    def get_all_feedback(self) -> List[Dict]:
        """Get all feedback records"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT feedback_id, user_query, generated_sql, is_correct, 
                       correct_sql, user_id, timestamp
                FROM feedback
                ORDER BY created_at DESC
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            feedback_list = []
            for row in results:
                feedback_list.append({
                    'feedback_id': row[0],
                    'user_query': row[1],
                    'generated_sql': row[2],
                    'is_correct': bool(row[3]),
                    'correct_sql': row[4],
                    'user_id': row[5],
                    'timestamp': row[6]
                })
            
            return feedback_list
            
        except Exception as e:
            print(f"Error getting feedback: {e}")
            return []
    
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total feedback
            cursor.execute('SELECT COUNT(*) FROM feedback')
            total = cursor.fetchone()[0]
            
            # Correct feedback
            cursor.execute('SELECT COUNT(*) FROM feedback WHERE is_correct = 1')
            correct = cursor.fetchone()[0]
            
            # Learned patterns
            cursor.execute('SELECT COUNT(*) FROM learned_patterns')
            patterns = cursor.fetchone()[0]
            
            # Recent feedback (last 10)
            cursor.execute('''
                SELECT user_query, is_correct, timestamp
                FROM feedback
                ORDER BY created_at DESC
                LIMIT 10
            ''')
            recent = cursor.fetchall()
            
            conn.close()
            
            return {
                'total_feedback': total,
                'correct_responses': correct,
                'accuracy_rate': correct / total if total > 0 else 0.0,
                'learned_patterns': patterns,
                'recent_feedback': [
                    {
                        'query': r[0],
                        'correct': bool(r[1]),
                        'time': r[2]
                    }
                    for r in recent
                ]
            }
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {
                'total_feedback': 0,
                'accuracy_rate': 0.0,
                'learned_patterns': 0
            }
    
    def clear_all_feedback(self):
        """Clear all feedback (for testing)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM feedback')
            cursor.execute('DELETE FROM learned_patterns')
            
            conn.commit()
            conn.close()
            
            return True
        except Exception as e:
            print(f"Error clearing feedback: {e}")
            return False
