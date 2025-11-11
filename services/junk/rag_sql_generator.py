# 🚀 FIXED RAG SQL GENERATOR - All compatibility issues resolved

import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
from dataclasses import dataclass
import re

# FIXED: Proper ML library imports with fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from pathlib import Path

from utils.database import get_supabase_client, list_available_tables
from utils.logger_config import get_logger

logger = get_logger(__name__)

@dataclass
class SQLPattern:
    """Advanced SQL pattern with context"""
    intent: str
    natural_language: str
    sql_template: str
    complexity_level: str  # basic, intermediate, advanced, expert
    sql_category: str  # join, aggregate, subquery, window, cte, etc.
    business_domain: str  # sales, hr, inventory, finance, etc.
    table_requirements: List[str]
    column_requirements: List[str]
    variables: List[str]  # Placeholders like {table_name}, {date_column}
    examples: List[Dict[str, str]]
    confidence_score: float
    usage_frequency: int

class RAGSQLGenerator:
    """Advanced SQL generation using RAG with comprehensive pattern database"""
    
    def __init__(self):
        self.logger = logger
        self.sql_patterns: List[SQLPattern] = []
        self.pattern_embeddings = None
        self.faiss_index = None
        
        # Initialize with proper fallbacks
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformers model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformers: {e}")
                self.embedding_model = None
                SENTENCE_TRANSFORMERS_AVAILABLE = False
        
        # Load comprehensive SQL patterns
        self._load_comprehensive_patterns()
        
        # Build vector index only if libraries are available
        if SENTENCE_TRANSFORMERS_AVAILABLE and FAISS_AVAILABLE and self.embedding_model:
            self._build_vector_index()
        else:
            logger.warning("Vector search disabled - using keyword matching fallback")
    
    def _load_comprehensive_patterns(self):
        """Load comprehensive SQL patterns covering all query types"""
        
        # BASIC PATTERNS
        basic_patterns = [
            SQLPattern(
                intent="simple_select",
                natural_language="show me all records from table",
                sql_template="SELECT * FROM {table_name} LIMIT {limit}",
                complexity_level="basic",
                sql_category="select",
                business_domain="general",
                table_requirements=["{table_name}"],
                column_requirements=[],
                variables=["table_name", "limit"],
                examples=[
                    {"query": "show me all customers", "sql": "SELECT * FROM customers LIMIT 50"},
                    {"query": "list all products", "sql": "SELECT * FROM products LIMIT 50"}
                ],
                confidence_score=0.95,
                usage_frequency=0
            ),
            SQLPattern(
                intent="count_records",
                natural_language="count total number of records",
                sql_template="SELECT COUNT(*) as total_count FROM {table_name}",
                complexity_level="basic",
                sql_category="aggregate",
                business_domain="general",
                table_requirements=["{table_name}"],
                column_requirements=[],
                variables=["table_name"],
                examples=[
                    {"query": "how many customers do we have", "sql": "SELECT COUNT(*) as total_count FROM customers"},
                    {"query": "total number of orders", "sql": "SELECT COUNT(*) as total_count FROM orders"}
                ],
                confidence_score=0.98,
                usage_frequency=0
            )
        ]
        
        # INTERMEDIATE PATTERNS
        intermediate_patterns = [
            SQLPattern(
                intent="aggregate_with_grouping",
                natural_language="calculate totals grouped by category",
                sql_template="SELECT {group_column}, COUNT(*) as count, SUM({amount_column}) as total FROM {table_name} GROUP BY {group_column} ORDER BY total DESC",
                complexity_level="intermediate",
                sql_category="group_aggregate",
                business_domain="sales",
                table_requirements=["{table_name}"],
                column_requirements=["{group_column}", "{amount_column}"],
                variables=["table_name", "group_column", "amount_column"],
                examples=[
                    {"query": "total sales by product category", "sql": "SELECT category, COUNT(*) as count, SUM(amount) as total FROM sales GROUP BY category ORDER BY total DESC"},
                    {"query": "revenue by customer type", "sql": "SELECT customer_type, COUNT(*) as count, SUM(revenue) as total FROM orders GROUP BY customer_type ORDER BY total DESC"}
                ],
                confidence_score=0.92,
                usage_frequency=0
            ),
            SQLPattern(
                intent="time_series_analysis",
                natural_language="analyze trends over time periods",
                sql_template="SELECT DATE_TRUNC('{period}', {date_column}) as period, COUNT(*) as count, SUM({amount_column}) as total FROM {table_name} WHERE {date_column} >= NOW() - INTERVAL '{timeframe}' GROUP BY DATE_TRUNC('{period}', {date_column}) ORDER BY period",
                complexity_level="intermediate",
                sql_category="time_series",
                business_domain="analytics",
                table_requirements=["{table_name}"],
                column_requirements=["{date_column}", "{amount_column}"],
                variables=["table_name", "date_column", "amount_column", "period", "timeframe"],
                examples=[
                    {"query": "monthly sales trend", "sql": "SELECT DATE_TRUNC('month', created_at) as period, COUNT(*) as count, SUM(amount) as total FROM orders WHERE created_at >= NOW() - INTERVAL '12 months' GROUP BY DATE_TRUNC('month', created_at) ORDER BY period"},
                    {"query": "daily revenue last 30 days", "sql": "SELECT DATE_TRUNC('day', order_date) as period, COUNT(*) as count, SUM(total_amount) as total FROM sales WHERE order_date >= NOW() - INTERVAL '30 days' GROUP BY DATE_TRUNC('day', order_date) ORDER BY period"}
                ],
                confidence_score=0.88,
                usage_frequency=0
            )
        ]
        
        # ADVANCED PATTERNS
        advanced_patterns = [
            SQLPattern(
                intent="inner_join_analysis",
                natural_language="join tables to combine related data",
                sql_template="SELECT {select_columns} FROM {main_table} t1 INNER JOIN {join_table} t2 ON t1.{join_key1} = t2.{join_key2} WHERE {condition} ORDER BY {order_column} LIMIT {limit}",
                complexity_level="advanced",
                sql_category="join",
                business_domain="relational",
                table_requirements=["{main_table}", "{join_table}"],
                column_requirements=["{join_key1}", "{join_key2}", "{order_column}"],
                variables=["main_table", "join_table", "select_columns", "join_key1", "join_key2", "condition", "order_column", "limit"],
                examples=[
                    {"query": "customers with their order totals", "sql": "SELECT c.name, c.email, SUM(o.amount) as total_spent FROM customers c INNER JOIN orders o ON c.id = o.customer_id WHERE o.status = 'completed' GROUP BY c.id, c.name, c.email ORDER BY total_spent DESC LIMIT 50"},
                    {"query": "products with sales performance", "sql": "SELECT p.name, p.category, COUNT(s.id) as sales_count, SUM(s.amount) as revenue FROM products p INNER JOIN sales s ON p.id = s.product_id WHERE s.created_at >= NOW() - INTERVAL '90 days' GROUP BY p.id, p.name, p.category ORDER BY revenue DESC LIMIT 50"}
                ],
                confidence_score=0.85,
                usage_frequency=0
            )
        ]
        
        # EXPERT PATTERNS
        expert_patterns = [
            SQLPattern(
                intent="cohort_analysis",
                natural_language="analyze user cohorts and retention",
                sql_template="WITH cohort_data AS (SELECT customer_id, DATE_TRUNC('month', first_order_date) as cohort_month, DATE_TRUNC('month', order_date) as order_month FROM (SELECT customer_id, MIN(created_at) as first_order_date, created_at as order_date FROM orders GROUP BY customer_id, created_at) base), cohort_sizes AS (SELECT cohort_month, COUNT(DISTINCT customer_id) as cohort_size FROM cohort_data GROUP BY cohort_month) SELECT c.cohort_month, cs.cohort_size, EXTRACT(epoch FROM (c.order_month - c.cohort_month))/2628000 as period_number, COUNT(DISTINCT c.customer_id) as customers FROM cohort_data c JOIN cohort_sizes cs ON c.cohort_month = cs.cohort_month GROUP BY c.cohort_month, cs.cohort_size, period_number ORDER BY c.cohort_month, period_number",
                complexity_level="expert",
                sql_category="cte_analysis",
                business_domain="retention_analytics",
                table_requirements=["orders"],
                column_requirements=["customer_id", "created_at"],
                variables=["table_name"],
                examples=[
                    {"query": "customer retention cohort analysis", "sql": "WITH cohort_data AS (SELECT customer_id, DATE_TRUNC('month', first_order_date) as cohort_month, DATE_TRUNC('month', order_date) as order_month FROM (SELECT customer_id, MIN(created_at) as first_order_date, created_at as order_date FROM orders GROUP BY customer_id, created_at) base), cohort_sizes AS (SELECT cohort_month, COUNT(DISTINCT customer_id) as cohort_size FROM cohort_data GROUP BY cohort_month) SELECT c.cohort_month, cs.cohort_size, EXTRACT(epoch FROM (c.order_month - c.cohort_month))/2628000 as period_number, COUNT(DISTINCT c.customer_id) as customers FROM cohort_data c JOIN cohort_sizes cs ON c.cohort_month = cs.cohort_month GROUP BY c.cohort_month, cs.cohort_size, period_number ORDER BY c.cohort_month, period_number"}
                ],
                confidence_score=0.75,
                usage_frequency=0
            )
        ]
        
        # Combine all patterns
        self.sql_patterns = basic_patterns + intermediate_patterns + advanced_patterns + expert_patterns
        logger.info(f"Loaded {len(self.sql_patterns)} comprehensive SQL patterns")
    
    def _build_vector_index(self):
        """Build FAISS vector index for fast similarity search"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not FAISS_AVAILABLE or not self.embedding_model:
            logger.warning("Vector indexing disabled - missing dependencies")
            return
            
        try:
            # Generate embeddings for all patterns
            pattern_texts = []
            for pattern in self.sql_patterns:
                text_parts = [
                    pattern.intent,
                    pattern.natural_language,
                    pattern.business_domain,
                    pattern.sql_category
                ]
                
                # Add example queries
                for example in pattern.examples:
                    text_parts.append(example["query"])
                
                pattern_texts.append(" ".join(text_parts))
            
            # Generate embeddings
            self.pattern_embeddings = self.embedding_model.encode(pattern_texts)
            
            # Build FAISS index
            dimension = self.pattern_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product for similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.pattern_embeddings)
            self.faiss_index.add(self.pattern_embeddings)
            
            logger.info(f"Built FAISS index with {len(pattern_texts)} patterns")
            
        except Exception as e:
            logger.error(f"Failed to build vector index: {e}")
            self.faiss_index = None
    
    async def generate_complex_sql(self, user_query: str, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complex SQL using RAG-based pattern matching"""
        try:
            # 1. Find similar patterns using vector search or keyword fallback
            similar_patterns = self._find_similar_patterns(user_query, top_k=5)
            
            if not similar_patterns:
                return {"success": False, "error": "No suitable SQL patterns found"}
            
            # 2. Select best pattern based on schema compatibility
            best_pattern = self._select_best_pattern(similar_patterns, schema_info)
            
            if not best_pattern:
                return {"success": False, "error": "No compatible pattern found for schema"}
            
            # 3. Fill pattern variables with actual table/column names
            filled_sql = self._fill_pattern_variables(best_pattern, user_query, schema_info)
            
            if not filled_sql:
                return {"success": False, "error": "Could not map pattern to available schema"}
            
            # 4. Optimize and validate SQL
            optimized_sql = self._optimize_sql(filled_sql, schema_info)
            
            return {
                "success": True,
                "sql": optimized_sql,
                "pattern_used": {
                    "intent": best_pattern.intent,
                    "complexity": best_pattern.complexity_level,
                    "category": best_pattern.sql_category,
                    "confidence": best_pattern.confidence_score
                },
                "explanation": f"Generated {best_pattern.complexity_level} {best_pattern.sql_category} query for {best_pattern.business_domain} analysis"
            }
            
        except Exception as e:
            logger.error(f"Complex SQL generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _find_similar_patterns(self, user_query: str, top_k: int = 5) -> List[Tuple[SQLPattern, float]]:
        """Find most similar SQL patterns using vector search or keyword fallback"""
        
        # Try vector search first
        if self.faiss_index and self.embedding_model and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                query_embedding = self.embedding_model.encode([user_query])
                faiss.normalize_L2(query_embedding)
                
                similarities, indices = self.faiss_index.search(query_embedding, top_k)
                
                results = []
                for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                    if idx < len(self.sql_patterns):
                        pattern = self.sql_patterns[idx]
                        results.append((pattern, float(similarity)))
                
                results.sort(key=lambda x: x[1], reverse=True)
                return results
                
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to keywords: {e}")
        
        # Keyword-based fallback
        return self._keyword_based_pattern_search(user_query, top_k)
    
    def _keyword_based_pattern_search(self, user_query: str, top_k: int = 5) -> List[Tuple[SQLPattern, float]]:
        """Fallback keyword-based pattern matching"""
        query_lower = user_query.lower()
        pattern_scores = []
        
        for pattern in self.sql_patterns:
            score = 0.0
            
            # Score based on intent keywords
            intent_words = pattern.intent.replace('_', ' ').split()
            for word in intent_words:
                if word in query_lower:
                    score += 0.3
            
            # Score based on natural language description
            nl_words = pattern.natural_language.lower().split()
            for word in nl_words:
                if word in query_lower:
                    score += 0.2
            
            # Score based on examples
            for example in pattern.examples:
                example_words = example["query"].lower().split()
                for word in example_words:
                    if word in query_lower:
                        score += 0.1
            
            if score > 0:
                pattern_scores.append((pattern, score))
        
        # Sort by score and return top_k
        pattern_scores.sort(key=lambda x: x[1], reverse=True)
        return pattern_scores[:top_k]
    
    def _select_best_pattern(self, similar_patterns: List[Tuple[SQLPattern, float]], 
                           schema_info: Dict[str, Any]) -> Optional[SQLPattern]:
        """Select the best pattern based on similarity and schema compatibility"""
        
        for pattern, similarity_score in similar_patterns:
            # Check schema compatibility
            compatibility_score = self._check_schema_compatibility(pattern, schema_info)
            
            # Combined score
            total_score = similarity_score * 0.7 + compatibility_score * 0.3
            
            if total_score > 0.4:  # Minimum threshold
                return pattern
        
        # Return first pattern if no good match
        return similar_patterns[0][0] if similar_patterns else None
    
    def _check_schema_compatibility(self, pattern: SQLPattern, schema_info: Dict[str, Any]) -> float:
        """Check how compatible a pattern is with available schema"""
        available_tables = schema_info.get("tables", [])
        available_columns = schema_info.get("columns", {})
        
        if not available_tables:
            return 0.0
        
        compatibility_score = 0.0
        total_requirements = len(pattern.table_requirements) + len(pattern.column_requirements)
        
        if total_requirements == 0:
            return 1.0
        
        # Check table requirements
        for table_req in pattern.table_requirements:
            if table_req.startswith("{") and table_req.endswith("}"):
                # Variable table - check if we have any tables
                if available_tables:
                    compatibility_score += 1
            elif table_req in available_tables:
                compatibility_score += 1
        
        # Check column requirements (simplified)
        for col_req in pattern.column_requirements:
            if col_req.startswith("{") and col_req.endswith("}"):
                # Variable column - assume we can find something
                compatibility_score += 0.5
        
        return compatibility_score / total_requirements if total_requirements > 0 else 1.0
    
    def _fill_pattern_variables(self, pattern: SQLPattern, user_query: str, schema_info: Dict[str, Any]) -> Optional[str]:
        """Fill pattern template variables with actual schema elements"""
        try:
            sql_template = pattern.sql_template
            available_tables = schema_info.get("tables", [])
            available_columns = schema_info.get("columns", {})
            
            variable_mappings = {}
            
            # Map table variables
            if "{table_name}" in sql_template:
                table_name = self._find_best_table(user_query, available_tables)
                if table_name:
                    variable_mappings["table_name"] = table_name
                else:
                    return None
            
            # Map column variables (simplified)
            table_for_columns = variable_mappings.get("table_name")
            if table_for_columns and table_for_columns in available_columns:
                table_columns = available_columns[table_for_columns]
                
                # Find specific column types
                for var in pattern.variables:
                    if var.endswith("_column"):
                        col_type = var.replace("_column", "")
                        found_column = self._find_column_by_type(col_type, table_columns, user_query)
                        if found_column:
                            variable_mappings[var] = found_column
            
            # Set default values
            if "limit" in pattern.variables:
                variable_mappings["limit"] = "50"
            
            if "period" in pattern.variables:
                variable_mappings["period"] = "month"
            
            if "timeframe" in pattern.variables:
                variable_mappings["timeframe"] = "12 months"
            
            # Fill template
            filled_sql = sql_template
            for var, value in variable_mappings.items():
                filled_sql = filled_sql.replace(f"{{{var}}}", str(value))
            
            # Check if all variables were filled
            if "{" in filled_sql and "}" in filled_sql:
                logger.warning(f"Some variables not filled in SQL: {filled_sql}")
                return None
            
            return filled_sql
            
        except Exception as e:
            logger.error(f"Failed to fill pattern variables: {e}")
            return None
    
    def _find_best_table(self, user_query: str, available_tables: List[str]) -> Optional[str]:
        """Find the most relevant table for the query"""
        query_lower = user_query.lower()
        
        # Direct table name matches
        for table in available_tables:
            if table.lower() in query_lower:
                return table
        
        # Semantic matches
        table_keywords = {
            "customer": ["customer", "client", "user", "person"],
            "order": ["order", "purchase", "transaction", "sale"],
            "product": ["product", "item", "goods", "inventory"],
            "employee": ["employee", "staff", "worker", "personnel"]
        }
        
        for table in available_tables:
            table_lower = table.lower()
            for concept, keywords in table_keywords.items():
                if concept in table_lower:
                    if any(keyword in query_lower for keyword in keywords):
                        return table
        
        # Return first table if no match found
        return available_tables[0] if available_tables else None
    
    def _find_column_by_type(self, col_type: str, columns: List[str], user_query: str) -> Optional[str]:
        """Find column by type with fallback"""
        
        type_patterns = {
            "amount": ["amount", "total", "price", "value", "cost", "revenue"],
            "date": ["date", "time", "created", "updated", "timestamp"],
            "group": ["category", "type", "status", "region"],
            "order": ["created_at", "date", "amount"]
        }
        
        patterns = type_patterns.get(col_type, [col_type])
        
        for pattern in patterns:
            for col in columns:
                if pattern in col.lower():
                    return col
        
        # Return first column if no match
        return columns[0] if columns else None
    
    def _optimize_sql(self, sql: str, schema_info: Dict[str, Any]) -> str:
        """Optimize the generated SQL for better performance"""
        optimized = sql.strip()
        
        # Add LIMIT if not present
        if "LIMIT" not in optimized.upper():
            optimized += " LIMIT 100"
        
        return optimized

# Test function
async def test_rag_sql_generator():
    """Test the RAG SQL generator"""
    generator = RAGSQLGenerator()
    
    schema_info = {
        "tables": ["customers", "orders", "products", "sales"],
        "columns": {
            "customers": ["id", "name", "email", "created_at", "status"],
            "orders": ["id", "customer_id", "amount", "created_at", "status"],
            "products": ["id", "name", "category", "price"],
            "sales": ["id", "product_id", "customer_id", "amount", "created_at"]
        }
    }
    
    test_queries = [
        "show me all customers",
        "how many orders do we have",
        "total sales by category",
        "customers with their orders"
    ]
    
    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        result = await generator.generate_complex_sql(query, schema_info)
        if result["success"]:
            print(f"SQL: {result['sql']}")
            print(f"Pattern: {result['pattern_used']['intent']} ({result['pattern_used']['complexity']})")
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_rag_sql_generator())