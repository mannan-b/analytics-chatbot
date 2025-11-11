# 🎯 FINAL FIXED RAG SYSTEM - USES YOUR ACTUAL SUPABASE TABLES
"""
FINAL version that uses:
- table_descriptions (for tables)
- column_metadata (for columns) 
- business_context (for non-data queries)
NO MORE sql_patterns table dependency!
"""

import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
import re
from dataclasses import dataclass

# Try importing advanced libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("⚠️ sentence-transformers not available, using fallback matching")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("⚠️ FAISS not available, using simple similarity")

from supabase import Client
from utils.database import init_database, get_supabase_client
from utils.logger_config import get_logger

# Static SQL patterns - no database dependency!
COMPREHENSIVE_SQL_PATTERNS = [
    # BASIC SELECT PATTERNS
    {
        "natural_language": "show all {table}",
        "sql_template": "SELECT * FROM {table} LIMIT 10",
        "category": "basic_select",
        "complexity": "basic",
        "description": "Show all records from table",
        "keywords": ["show", "all", "list", "display"]
    },
    {
        "natural_language": "get {table}",
        "sql_template": "SELECT * FROM {table} LIMIT 10",
        "category": "basic_select", 
        "complexity": "basic",
        "description": "Get records from table",
        "keywords": ["get", "fetch", "retrieve"]
    },
    {
        "natural_language": "list {table}",
        "sql_template": "SELECT * FROM {table} LIMIT 20",
        "category": "basic_select",
        "complexity": "basic", 
        "description": "List records from table",
        "keywords": ["list", "display", "show"]
    },
    
    # COUNT PATTERNS
    {
        "natural_language": "count {table}",
        "sql_template": "SELECT COUNT(*) as total_count FROM {table}",
        "category": "count",
        "complexity": "basic",
        "description": "Count total records",
        "keywords": ["count", "total", "how many"]
    },
    {
        "natural_language": "how many {table}",
        "sql_template": "SELECT COUNT(*) as total_count FROM {table}",
        "category": "count",
        "complexity": "basic",
        "description": "Count total records",
        "keywords": ["how", "many", "total"]
    },
    
    # TOP/LIMIT PATTERNS
    {
        "natural_language": "top 10 {table}",
        "sql_template": "SELECT * FROM {table} ORDER BY {column} DESC LIMIT 10",
        "category": "top_n",
        "complexity": "intermediate",
        "description": "Get top 10 records",
        "keywords": ["top", "best", "highest"]
    },
    {
        "natural_language": "first 5 {table}",
        "sql_template": "SELECT * FROM {table} LIMIT 5",
        "category": "limited_select",
        "complexity": "basic",
        "description": "Get first 5 records",
        "keywords": ["first", "initial"]
    },
    
    # AGGREGATION PATTERNS
    {
        "natural_language": "sum of {column} in {table}",
        "sql_template": "SELECT SUM({column}) as total FROM {table}",
        "category": "aggregation",
        "complexity": "intermediate",
        "description": "Sum column values",
        "keywords": ["sum", "total", "add"]
    },
    {
        "natural_language": "average {column} in {table}",
        "sql_template": "SELECT AVG({column}) as average FROM {table}",
        "category": "aggregation",
        "complexity": "intermediate", 
        "description": "Average column values",
        "keywords": ["average", "avg", "mean"]
    },
    {
        "natural_language": "maximum {column} in {table}",
        "sql_template": "SELECT MAX({column}) as maximum FROM {table}",
        "category": "aggregation",
        "complexity": "intermediate",
        "description": "Maximum column value",
        "keywords": ["maximum", "max", "highest"]
    },
    {
        "natural_language": "minimum {column} in {table}",
        "sql_template": "SELECT MIN({column}) as minimum FROM {table}",
        "category": "aggregation", 
        "complexity": "intermediate",
        "description": "Minimum column value",
        "keywords": ["minimum", "min", "lowest"]
    },
    
    # FILTERING PATTERNS
    {
        "natural_language": "{table} where {column} equals {value}",
        "sql_template": "SELECT * FROM {table} WHERE {column} = '{value}'",
        "category": "filtered_select",
        "complexity": "intermediate",
        "description": "Filter by exact match",
        "keywords": ["where", "equals", "is"]
    },
    {
        "natural_language": "{table} with {column} containing {value}",
        "sql_template": "SELECT * FROM {table} WHERE {column} LIKE '%{value}%'",
        "category": "text_search",
        "complexity": "intermediate", 
        "description": "Search text in column",
        "keywords": ["containing", "with", "having"]
    },
    
    # GROUPING PATTERNS
    {
        "natural_language": "{table} grouped by {column}",
        "sql_template": "SELECT {column}, COUNT(*) as count FROM {table} GROUP BY {column}",
        "category": "grouped_select",
        "complexity": "advanced",
        "description": "Group records by column",
        "keywords": ["grouped", "group by"]
    },
    {
        "natural_language": "count {table} by {column}",
        "sql_template": "SELECT {column}, COUNT(*) as count FROM {table} GROUP BY {column}",
        "category": "grouped_count",
        "complexity": "advanced",
        "description": "Count records grouped by column", 
        "keywords": ["count", "by", "grouped"]
    },
    
    # SORTING PATTERNS
    {
        "natural_language": "{table} sorted by {column}",
        "sql_template": "SELECT * FROM {table} ORDER BY {column}",
        "category": "sorted_select",
        "complexity": "intermediate",
        "description": "Sort records by column",
        "keywords": ["sorted", "ordered", "order by"]
    },
    {
        "natural_language": "{table} ordered by {column} descending",
        "sql_template": "SELECT * FROM {table} ORDER BY {column} DESC",
        "category": "sorted_select",
        "complexity": "intermediate",
        "description": "Sort records descending",
        "keywords": ["descending", "desc", "highest first"]
    },
    
    # JOIN PATTERNS  
    {
        "natural_language": "{table1} with {table2}",
        "sql_template": "SELECT * FROM {table1} t1 JOIN {table2} t2 ON t1.id = t2.{table1}_id",
        "category": "join",
        "complexity": "advanced",
        "description": "Join two tables",
        "keywords": ["with", "join", "including"]
    },
    
    # BUSINESS PATTERNS
    {
        "natural_language": "customers and their orders",
        "sql_template": "SELECT c.*, o.* FROM customers c LEFT JOIN orders o ON c.id = o.customer_id",
        "category": "business_join",
        "complexity": "advanced", 
        "description": "Customers with orders",
        "keywords": ["customers", "orders", "their"]
    },
    {
        "natural_language": "products by category",
        "sql_template": "SELECT category, COUNT(*) as product_count FROM products GROUP BY category",
        "category": "business_grouping",
        "complexity": "advanced",
        "description": "Products grouped by category", 
        "keywords": ["products", "category", "by"]
    }
]

logger = get_logger(__name__)

@dataclass
class SQLMatch:
    """SQL pattern match result"""
    natural_language: str
    sql_template: str
    similarity_score: float
    category: str
    complexity: str
    description: str
    matched_keywords: List[str]
    confidence: str

class SuperbaseMetadataHandler:
    """Handles queries using your actual Supabase tables"""
    
    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.tables_cache = {}
        self.columns_cache = {}
        self.business_context_cache = {}
        self.initialized = False
    
    def initialize(self):
        """Load metadata from your Supabase tables"""
        try:
            # Load table_descriptions
            try:
                tables_result = self.supabase.table("table_descriptions").select("*").execute()
                tables = tables_result.data or []
                
                for table in tables:
                    table_name = table['table_name']
                    self.tables_cache[table_name] = table
                    # Create search mappings
                    self.tables_cache[table_name.lower()] = table
                    if table_name.endswith('s'):
                        self.tables_cache[table_name[:-1].lower()] = table  # singular
                
                logger.info(f"✅ Loaded {len(tables)} tables from table_descriptions")
            except Exception as e:
                logger.warning(f"⚠️ Could not load table_descriptions: {e}")
            
            # Load column_metadata
            try:
                columns_result = self.supabase.table("column_metadata").select("*").execute()
                columns = columns_result.data or []
                
                for column in columns:
                    table_name = column['table_name']
                    if table_name not in self.columns_cache:
                        self.columns_cache[table_name] = []
                    self.columns_cache[table_name].append(column)
                
                logger.info(f"✅ Loaded {len(columns)} columns from column_metadata")
            except Exception as e:
                logger.warning(f"⚠️ Could not load column_metadata: {e}")
            
            # Load business_context
            try:
                context_result = self.supabase.table("business_context").select("*").execute()
                contexts = context_result.data or []
                
                for context in contexts:
                    context_type = context.get('context_type', 'general')
                    if context_type not in self.business_context_cache:
                        self.business_context_cache[context_type] = []
                    self.business_context_cache[context_type].append(context)
                
                logger.info(f"✅ Loaded {len(contexts)} business contexts")
            except Exception as e:
                logger.warning(f"⚠️ Could not load business_context: {e}")
            
            self.initialized = True
            logger.info("✅ Supabase metadata handler initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Metadata handler initialization issues: {e}")
            self.initialized = True
    
    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get table info from table_descriptions"""
        return self.tables_cache.get(table_name) or self.tables_cache.get(table_name.lower())
    
    def get_table_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """Get columns from column_metadata"""
        return self.columns_cache.get(table_name, [])
    
    def get_available_tables(self) -> List[str]:
        """Get list of available tables"""
        return [table for table in self.tables_cache.keys() if not table.islower()]
    
    def find_business_context(self, query: str) -> Optional[Dict[str, Any]]:
        """Find relevant business context"""
        query_lower = query.lower()
        
        # Search through all business contexts
        for context_type, contexts in self.business_context_cache.items():
            for context in contexts:
                title = context.get('title', '').lower()
                description = context.get('description', '').lower()
                keywords = context.get('keywords', [])
                
                # Check if query matches
                if any(keyword.lower() in query_lower for keyword in keywords):
                    return context
                elif any(word in title or word in description for word in query_lower.split()):
                    return context
        
        return None
    
    def map_table_name(self, natural_name: str) -> str:
        """Map natural language to actual table name"""
        # Direct lookup
        table_info = self.get_table_info(natural_name)
        if table_info:
            return table_info['table_name']
        
        # Fuzzy matching
        natural_lower = natural_name.lower()
        for table_name in self.get_available_tables():
            if natural_lower in table_name.lower() or table_name.lower() in natural_lower:
                return table_name
        
        # Common mappings
        common_mappings = {
            'customer': 'customers',
            'order': 'orders', 
            'product': 'products',
            'user': 'users'
        }
        
        return common_mappings.get(natural_lower, natural_name)
    
    def map_column_name(self, table_name: str, natural_name: str) -> str:
        """Map natural language to actual column name"""
        columns = self.get_table_columns(table_name)
        natural_lower = natural_name.lower()
        
        # Direct match
        for column in columns:
            if column['column_name'].lower() == natural_lower:
                return column['column_name']
        
        # Fuzzy match
        for column in columns:
            col_name = column['column_name'].lower()
            if natural_lower in col_name or col_name in natural_lower:
                return column['column_name']
        
        # Common column patterns
        if 'name' in natural_lower:
            return 'name'
        elif 'price' in natural_lower or 'amount' in natural_lower:
            return 'price'
        elif 'date' in natural_lower:
            return 'created_at'
        
        return natural_name

class FinalRAGSQLSystem:
    """
    FINAL RAG SYSTEM - Uses your actual Supabase tables!
    """
    
    def __init__(self):
        self.supabase: Client = None
        self.metadata_handler: SuperbaseMetadataHandler = None
        self.embeddings_model = None
        self.faiss_index = None
        self.pattern_embeddings = None
        self.patterns_list = COMPREHENSIVE_SQL_PATTERNS
        self.initialized = False
    
    async def initialize(self):
        """Initialize the final RAG system"""
        try:
            logger.info("🚀 Initializing Final RAG SQL System...")
            
            # Initialize database
            await init_database()
            logger.info("✅ Database initialized")
            
            # Get Supabase client
            self.supabase = get_supabase_client()
            logger.info("✅ Supabase client ready")
            
            # Initialize metadata handler
            self.metadata_handler = SuperbaseMetadataHandler(self.supabase)
            self.metadata_handler.initialize()
            logger.info("✅ Metadata handler initialized")
            
            # Initialize embeddings if available
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    await self._initialize_embeddings()
                    logger.info("✅ Embeddings initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Embeddings failed: {e}")
            
            # Initialize FAISS if available
            if FAISS_AVAILABLE and self.pattern_embeddings is not None:
                try:
                    await self._initialize_faiss()
                    logger.info("✅ FAISS index ready")
                except Exception as e:
                    logger.warning(f"⚠️ FAISS failed: {e}")
            
            self.initialized = True
            logger.info("🎉 Final RAG SQL System fully initialized!")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            self.initialized = True  # Continue with basic functionality
            logger.info("🚨 Emergency mode: Basic functionality only")
    
    async def _initialize_embeddings(self):
        """Initialize embeddings"""
        try:
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            pattern_texts = [p['natural_language'] for p in self.patterns_list]
            self.pattern_embeddings = self.embeddings_model.encode(pattern_texts, convert_to_numpy=True)
            
        except Exception as e:
            logger.error(f"❌ Embeddings error: {e}")
            self.embeddings_model = None
            self.pattern_embeddings = None
    
    async def _initialize_faiss(self):
        """Initialize FAISS - FIXED"""
        try:
            if self.pattern_embeddings is None or len(self.pattern_embeddings) == 0:
                return
            
            dimension = self.pattern_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings
            norms = np.linalg.norm(self.pattern_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # Prevent division by zero
            normalized_embeddings = self.pattern_embeddings / norms
            
            self.faiss_index.add(normalized_embeddings.astype('float32'))
            
        except Exception as e:
            logger.error(f"❌ FAISS error: {e}")
            self.faiss_index = None
    
    async def generate_sql(self, natural_query: str) -> Dict[str, Any]:
        """MAIN METHOD - Generate SQL from natural language"""
        if not self.initialized:
            return {"success": False, "error": "System not initialized"}
        
        try:
            logger.info(f"🔍 Processing query: {natural_query}")
            
            # STEP 1: Check if it's a business context query
            business_context = self.metadata_handler.find_business_context(natural_query)
            if business_context:
                return {
                    "success": True,
                    "query_type": "business_context",
                    "natural_query": natural_query,
                    "response": business_context.get('description', 'No description available'),
                    "context": business_context,
                    "sql_query": None
                }
            
            # STEP 2: Find matching SQL patterns
            matches = await self._find_best_matches(natural_query)
            if not matches:
                return {
                    "success": False,
                    "error": "No matching SQL patterns found",
                    "query": natural_query
                }
            
            # STEP 3: Generate SQL
            best_match = matches[0]
            entities = self._extract_entities(natural_query)
            final_sql = self._generate_final_sql(best_match, entities, natural_query)
            
            return {
                "success": True,
                "query_type": "data_query",
                "natural_query": natural_query,
                "sql_query": final_sql,
                "pattern_matched": best_match.natural_language,
                "similarity_score": best_match.similarity_score,
                "confidence": best_match.confidence,
                "category": best_match.category,
                "complexity": best_match.complexity,
                "entities_found": entities,
                "alternative_matches": [
                    {
                        "pattern": m.natural_language,
                        "score": m.similarity_score,
                        "category": m.category
                    } for m in matches[1:4]
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating SQL: {e}")
            return {"success": False, "error": str(e), "query": natural_query}
    
    async def _find_best_matches(self, query: str) -> List[SQLMatch]:
        """Find best matching patterns"""
        if self.embeddings_model and self.faiss_index:
            return await self._semantic_search(query)
        else:
            return self._keyword_search(query)
    
    async def _semantic_search(self, query: str) -> List[SQLMatch]:
        """Semantic search with FAISS"""
        try:
            query_embedding = self.embeddings_model.encode([query], convert_to_numpy=True)
            query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            query_norm = np.maximum(query_norm, 1e-8)
            query_normalized = query_embedding / query_norm
            
            k = min(10, len(self.patterns_list))
            similarities, indices = self.faiss_index.search(query_normalized.astype('float32'), k)
            
            matches = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if 0 <= idx < len(self.patterns_list):
                    pattern = self.patterns_list[idx]
                    confidence = "high" if similarity > 0.8 else "medium" if similarity > 0.6 else "low"
                    
                    match = SQLMatch(
                        natural_language=pattern['natural_language'],
                        sql_template=pattern['sql_template'],
                        similarity_score=float(similarity),
                        category=pattern['category'],
                        complexity=pattern['complexity'],
                        description=pattern['description'],
                        matched_keywords=self._extract_keywords(query),
                        confidence=confidence
                    )
                    matches.append(match)
            
            return [m for m in matches if m.similarity_score > 0.3]
            
        except Exception as e:
            logger.error(f"❌ Semantic search error: {e}")
            return self._keyword_search(query)
    
    def _keyword_search(self, query: str) -> List[SQLMatch]:
        """Fallback keyword search"""
        try:
            query_words = set(self._extract_keywords(query))
            scored_patterns = []
            
            for pattern in self.patterns_list:
                pattern_keywords = set(pattern.get('keywords', []))
                pattern_text_words = set(self._extract_keywords(pattern['natural_language']))
                all_pattern_words = pattern_keywords.union(pattern_text_words)
                
                common_words = query_words.intersection(all_pattern_words)
                if common_words:
                    score = len(common_words) / max(len(query_words), len(all_pattern_words))
                    confidence = "high" if score > 0.7 else "medium" if score > 0.4 else "low"
                    
                    match = SQLMatch(
                        natural_language=pattern['natural_language'],
                        sql_template=pattern['sql_template'],
                        similarity_score=score,
                        category=pattern['category'],
                        complexity=pattern['complexity'],
                        description=pattern['description'],
                        matched_keywords=list(common_words),
                        confidence=confidence
                    )
                    scored_patterns.append(match)
            
            scored_patterns.sort(key=lambda x: x.similarity_score, reverse=True)
            return [m for m in scored_patterns[:10] if m.similarity_score > 0.2]
            
        except Exception as e:
            logger.error(f"❌ Keyword search error: {e}")
            return []
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities using metadata handler"""
        entities = {"tables": [], "columns": [], "values": [], "numbers": []}
        
        words = query.lower().split()
        
        # Extract tables using metadata
        for word in words:
            mapped_table = self.metadata_handler.map_table_name(word)
            if mapped_table and mapped_table not in entities["tables"]:
                # Verify table exists
                if self.metadata_handler.get_table_info(mapped_table):
                    entities["tables"].append(mapped_table)
        
        # Extract numbers and values
        entities["numbers"] = [int(n) for n in re.findall(r'\d+', query)]
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", query)
        entities["values"] = [match[0] or match[1] for match in quoted if match[0] or match[1]]
        
        return entities
    
    def _generate_final_sql(self, match: SQLMatch, entities: Dict[str, Any], query: str) -> str:
        """Generate final SQL using metadata"""
        sql_template = match.sql_template
        
        # Replace table placeholders
        if entities["tables"]:
            primary_table = entities["tables"][0]
            sql_template = sql_template.replace("{table}", primary_table)
            sql_template = sql_template.replace("{table1}", primary_table)
            
            if len(entities["tables"]) > 1:
                sql_template = sql_template.replace("{table2}", entities["tables"][1])
        else:
            # Use available tables from metadata
            available_tables = self.metadata_handler.get_available_tables()
            fallback_table = available_tables[0] if available_tables else "your_table"
            sql_template = sql_template.replace("{table}", fallback_table)
            sql_template = sql_template.replace("{table1}", fallback_table)
        
        # Replace column placeholders
        if "{column}" in sql_template:
            table_name = entities["tables"][0] if entities["tables"] else None
            if table_name:
                guessed_column = self.metadata_handler.map_column_name(table_name, self._guess_column_from_query(query))
            else:
                guessed_column = self._guess_column_from_query(query)
            sql_template = sql_template.replace("{column}", guessed_column)
        
        # Replace value placeholders
        if "{value}" in sql_template:
            if entities["values"]:
                sql_template = sql_template.replace("{value}", entities["values"][0])
            elif entities["numbers"]:
                sql_template = sql_template.replace("{value}", str(entities["numbers"][0]))
            else:
                sql_template = sql_template.replace("{value}", "1")
        
        # Clean up remaining placeholders
        cleanup_map = {
            "{date_column}": "created_at",
            "{text_column}": "name", 
            "{join_column}": "id",
            "{group_column}": "category"
        }
        for placeholder, replacement in cleanup_map.items():
            sql_template = sql_template.replace(placeholder, replacement)
        
        return sql_template
    
    def _guess_column_from_query(self, query: str) -> str:
        """Guess column name from query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['name', 'title', 'label']):
            return 'name'
        elif any(word in query_lower for word in ['price', 'amount', 'cost', 'total']):
            return 'price'
        elif any(word in query_lower for word in ['date', 'time', 'created']):
            return 'created_at'
        elif any(word in query_lower for word in ['email', 'mail']):
            return 'email'
        elif any(word in query_lower for word in ['status', 'state']):
            return 'status'
        
        return 'id'
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords"""
        clean_text = re.sub(r'\{[^}]*\}', '', text.lower())
        words = re.findall(r'\w+', clean_text)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'}
        return [word for word in words if len(word) > 2 and word not in stop_words]

# Factory function
async def create_sophisticated_rag_system() -> FinalRAGSQLSystem:
    """Create and initialize the final RAG system"""
    system = FinalRAGSQLSystem()
    await system.initialize()
    return system

# Aliases for compatibility
create_bulletproof_rag_system = create_sophisticated_rag_system
SophisticatedRAGSQLSystem = FinalRAGSQLSystem
BulletproofRAGSQLSystem = FinalRAGSQLSystem