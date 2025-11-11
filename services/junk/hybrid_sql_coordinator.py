# 🤖 FIXED HYBRID SQL COORDINATOR - All compatibility issues resolved

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json

from utils.logger_config import get_logger

logger = get_logger(__name__)

class GenerationMethod(Enum):
    RAG = "rag"
    AST = "ast"
    RULE_BASED = "rule_based"
    LLM = "llm"
    HYBRID = "hybrid"

@dataclass
class SQLCandidate:
    """SQL query candidate from different generation methods"""
    sql: str
    method: GenerationMethod
    confidence: float
    complexity: str
    explanation: str
    metadata: Dict[str, Any]
    execution_time: Optional[float] = None
    estimated_cost: Optional[float] = None

@dataclass
class QueryRequirements:
    """Requirements extracted from user query"""
    tables_needed: List[str]
    columns_needed: List[str]
    operations_needed: List[str]
    complexity_level: str
    performance_priority: str  # speed, accuracy, complexity
    user_expertise_level: str  # beginner, intermediate, advanced

class HybridSQLCoordinator:
    """Coordinates multiple SQL generation methods for optimal results"""
    
    def __init__(self):
        self.logger = logger
        
        # Initialize generators with proper imports and fallbacks
        self.rag_generator = None
        self.ast_generator = None
        self.rule_builder = None
        
        # Try to initialize each generator with fallbacks
        self._initialize_generators()
        
        # Method selection strategy
        self.method_strategies = {
            "simple_queries": [GenerationMethod.RULE_BASED, GenerationMethod.AST],
            "complex_analytics": [GenerationMethod.RAG, GenerationMethod.LLM],
            "joins_required": [GenerationMethod.RAG, GenerationMethod.AST],
            "aggregations": [GenerationMethod.RULE_BASED, GenerationMethod.AST],
            "time_series": [GenerationMethod.RAG, GenerationMethod.LLM],
            "expert_analysis": [GenerationMethod.RAG, GenerationMethod.LLM]
        }
    
    def _initialize_generators(self):
        """Initialize SQL generators with proper error handling"""
        try:
            # Try importing RAG generator
            try:
                from services.rag_sql_generator import RAGSQLGenerator
                self.rag_generator = RAGSQLGenerator()
                logger.info("RAG SQL Generator initialized successfully")
            except ImportError as e:
                logger.warning(f"RAG SQL Generator not available: {e}")
                # Fallback to basic implementation
                self.rag_generator = BasicSQLGenerator("rag")
            
            # Try importing AST generator
            try:
                from services.ast_sql_generator import ASTSQLGenerator
                self.ast_generator = ASTSQLGenerator()
                logger.info("AST SQL Generator initialized successfully")
            except ImportError as e:
                logger.warning(f"AST SQL Generator not available: {e}")
                self.ast_generator = BasicSQLGenerator("ast")
            
            # Try importing Rule-based builder
            try:
                from services.rule_based_sql_builder import RuleBasedSQLBuilder
                self.rule_builder = RuleBasedSQLBuilder()
                logger.info("Rule-based SQL Builder initialized successfully")
            except ImportError as e:
                logger.warning(f"Rule-based SQL Builder not available: {e}")
                self.rule_builder = BasicSQLGenerator("rule_based")
        
        except Exception as e:
            logger.error(f"Failed to initialize SQL generators: {e}")
            # Use fallback generators
            self.rag_generator = BasicSQLGenerator("rag")
            self.ast_generator = BasicSQLGenerator("ast")
            self.rule_builder = BasicSQLGenerator("rule_based")
    
    async def generate_optimal_sql(self, user_query: str, schema_info: Dict[str, Any], 
                                 user_preferences: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate optimal SQL using multiple methods and intelligent selection"""
        try:
            # Step 1: Analyze query requirements
            requirements = self._analyze_query_requirements(user_query, schema_info)
            
            # Step 2: Select appropriate generation methods
            selected_methods = self._select_generation_methods(requirements, user_preferences)
            
            # Step 3: Generate SQL candidates using multiple methods
            candidates = await self._generate_sql_candidates(
                user_query, schema_info, selected_methods
            )
            
            if not candidates:
                return {"success": False, "error": "No SQL candidates generated"}
            
            # Step 4: Evaluate and rank candidates
            ranked_candidates = self._evaluate_and_rank_candidates(candidates, requirements)
            
            # Step 5: Select best candidate
            best_solution = self._select_best_solution(ranked_candidates, requirements)
            
            # Step 6: Validate and optimize final SQL
            final_sql = self._validate_and_optimize(best_solution, schema_info)
            
            return {
                "success": True,
                "sql": final_sql["sql"],
                "method": best_solution.method.value,
                "confidence": best_solution.confidence,
                "complexity": best_solution.complexity,
                "explanation": best_solution.explanation,
                "alternatives": [
                    {
                        "sql": candidate.sql,
                        "method": candidate.method.value,
                        "confidence": candidate.confidence,
                        "explanation": candidate.explanation
                    }
                    for candidate in ranked_candidates[1:3]  # Show top alternatives
                ],
                "generation_stats": {
                    "candidates_generated": len(candidates),
                    "methods_used": [method.value for method in selected_methods],
                    "selection_criteria": requirements.__dict__,
                    "optimization_applied": final_sql.get("optimizations", [])
                }
            }
            
        except Exception as e:
            logger.error(f"Hybrid SQL generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _analyze_query_requirements(self, user_query: str, schema_info: Dict[str, Any]) -> QueryRequirements:
        """Analyze user query to determine requirements and optimal approach"""
        query_lower = user_query.lower()
        
        # Determine tables needed
        available_tables = schema_info.get("tables", [])
        tables_needed = []
        
        for table in available_tables:
            if table.lower() in query_lower:
                tables_needed.append(table)
        
        # If no direct matches, use semantic matching
        if not tables_needed:
            table_keywords = {
                "customers": ["customer", "client", "user", "buyer"],
                "orders": ["order", "purchase", "transaction", "sale"],
                "products": ["product", "item", "goods", "inventory"],
                "sales": ["sales", "revenue", "earnings"]
            }
            
            for table in available_tables:
                table_lower = table.lower()
                for concept, keywords in table_keywords.items():
                    if concept in table_lower:
                        if any(keyword in query_lower for keyword in keywords):
                            tables_needed.append(table)
                            break
        
        # Determine operations needed
        operations_needed = []
        
        operation_indicators = {
            "join": ["with", "join", "combine", "along with", "together with"],
            "aggregate": ["total", "sum", "count", "average", "max", "min"],
            "group_by": ["by", "group by", "breakdown", "per", "each"],
            "filter": ["where", "filter", "only", "active", "completed", "recent"],
            "order": ["top", "best", "highest", "lowest", "rank", "sort"],
            "subquery": ["who have", "that have", "with more than", "exceeding"],
            "window": ["rank", "row_number", "running total", "moving average"],
            "time_analysis": ["trend", "over time", "monthly", "daily", "growth"]
        }
        
        for operation, indicators in operation_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                operations_needed.append(operation)
        
        # Determine complexity level
        complexity_score = 0
        
        if len(tables_needed) > 1:
            complexity_score += 2
        if "join" in operations_needed:
            complexity_score += 2
        if "subquery" in operations_needed:
            complexity_score += 3
        if "window" in operations_needed:
            complexity_score += 3
        if "time_analysis" in operations_needed:
            complexity_score += 2
        if len(operations_needed) > 3:
            complexity_score += 1
        
        if complexity_score >= 7:
            complexity_level = "expert"
        elif complexity_score >= 4:
            complexity_level = "advanced"
        elif complexity_score >= 2:
            complexity_level = "intermediate"
        else:
            complexity_level = "basic"
        
        # Determine columns needed (simplified)
        common_columns = ["name", "id", "date", "amount", "status", "type", "category"]
        columns_needed = [col for col in common_columns if col in query_lower]
        
        return QueryRequirements(
            tables_needed=tables_needed or available_tables[:1],
            columns_needed=columns_needed,
            operations_needed=operations_needed,
            complexity_level=complexity_level,
            performance_priority="accuracy",  # Default
            user_expertise_level="intermediate"  # Default
        )
    
    def _select_generation_methods(self, requirements: QueryRequirements, 
                                 user_preferences: Optional[Dict[str, str]] = None) -> List[GenerationMethod]:
        """Select optimal generation methods based on requirements"""
        
        selected_methods = []
        
        # Rule-based selection based on requirements
        if requirements.complexity_level == "basic":
            selected_methods.extend([GenerationMethod.RULE_BASED, GenerationMethod.AST])
            
        elif requirements.complexity_level == "intermediate":
            if "join" in requirements.operations_needed:
                selected_methods.extend([GenerationMethod.RAG, GenerationMethod.AST])
            else:
                selected_methods.extend([GenerationMethod.RULE_BASED, GenerationMethod.RAG])
                
        elif requirements.complexity_level == "advanced":
            selected_methods.extend([GenerationMethod.RAG, GenerationMethod.AST])
            
        else:  # expert
            selected_methods.extend([GenerationMethod.RAG, GenerationMethod.LLM])
        
        # Ensure we have at least 2 methods for comparison
        if len(selected_methods) < 2:
            all_methods = [GenerationMethod.RAG, GenerationMethod.AST, GenerationMethod.RULE_BASED]
            for method in all_methods:
                if method not in selected_methods:
                    selected_methods.append(method)
                    if len(selected_methods) >= 2:
                        break
        
        # Limit to maximum 3 methods for performance
        return selected_methods[:3]
    
    async def _generate_sql_candidates(self, user_query: str, schema_info: Dict[str, Any], 
                                     methods: List[GenerationMethod]) -> List[SQLCandidate]:
        """Generate SQL candidates using selected methods"""
        candidates = []
        
        # Generate candidates from each method
        for method in methods:
            try:
                if method == GenerationMethod.RAG and self.rag_generator:
                    result = await self._generate_rag_candidate(user_query, schema_info)
                elif method == GenerationMethod.AST and self.ast_generator:
                    result = await self._generate_ast_candidate(user_query, schema_info)
                elif method == GenerationMethod.RULE_BASED and self.rule_builder:
                    result = await self._generate_rule_based_candidate(user_query, schema_info)
                else:
                    continue
                
                if result:
                    candidates.append(result)
                    
            except Exception as e:
                logger.warning(f"SQL generation method {method.value} failed: {e}")
                continue
        
        return candidates
    
    async def _generate_rag_candidate(self, user_query: str, schema_info: Dict[str, Any]) -> Optional[SQLCandidate]:
        """Generate candidate using RAG method"""
        try:
            if hasattr(self.rag_generator, 'generate_complex_sql'):
                start_time = datetime.now()
                result = await self.rag_generator.generate_complex_sql(user_query, schema_info)
                generation_time = (datetime.now() - start_time).total_seconds()
                
                if result.get("success"):
                    return SQLCandidate(
                        sql=result["sql"],
                        method=GenerationMethod.RAG,
                        confidence=result.get("pattern_used", {}).get("confidence", 0.8),
                        complexity=result.get("pattern_used", {}).get("complexity", "intermediate"),
                        explanation=result.get("explanation", "Generated using RAG method"),
                        metadata=result.get("pattern_used", {}),
                        execution_time=generation_time
                    )
            else:
                # Fallback for basic generator
                result = await self.rag_generator.generate_sql(user_query, schema_info)
                if result.get("success"):
                    return SQLCandidate(
                        sql=result["sql"],
                        method=GenerationMethod.RAG,
                        confidence=0.7,
                        complexity="basic",
                        explanation="Generated using basic RAG fallback",
                        metadata={},
                        execution_time=0.1
                    )
        except Exception as e:
            logger.error(f"RAG candidate generation failed: {e}")
        
        return None
    
    async def _generate_ast_candidate(self, user_query: str, schema_info: Dict[str, Any]) -> Optional[SQLCandidate]:
        """Generate candidate using AST method"""
        try:
            if hasattr(self.ast_generator, 'generate_ast_sql'):
                start_time = datetime.now()
                result = await self.ast_generator.generate_ast_sql(user_query, schema_info)
                generation_time = (datetime.now() - start_time).total_seconds()
                
                if result.get("success"):
                    return SQLCandidate(
                        sql=result["sql"],
                        method=GenerationMethod.AST,
                        confidence=0.8,  # Default confidence for AST
                        complexity=result.get("complexity", "intermediate"),
                        explanation=result.get("explanation", "Generated using AST method"),
                        metadata={"ast": result.get("ast"), "parsed_intent": result.get("parsed_intent")},
                        execution_time=generation_time
                    )
            else:
                # Fallback
                result = await self.ast_generator.generate_sql(user_query, schema_info)
                if result.get("success"):
                    return SQLCandidate(
                        sql=result["sql"],
                        method=GenerationMethod.AST,
                        confidence=0.75,
                        complexity="basic",
                        explanation="Generated using basic AST fallback",
                        metadata={},
                        execution_time=0.1
                    )
        except Exception as e:
            logger.error(f"AST candidate generation failed: {e}")
        
        return None
    
    async def _generate_rule_based_candidate(self, user_query: str, schema_info: Dict[str, Any]) -> Optional[SQLCandidate]:
        """Generate candidate using rule-based method"""
        try:
            if hasattr(self.rule_builder, 'build_complex_sql'):
                start_time = datetime.now()
                result = await self.rule_builder.build_complex_sql(user_query, schema_info)
                generation_time = (datetime.now() - start_time).total_seconds()
                
                if result.get("success"):
                    return SQLCandidate(
                        sql=result["sql"],
                        method=GenerationMethod.RULE_BASED,
                        confidence=0.85,  # Default confidence for rule-based
                        complexity=result.get("complexity", "intermediate"),
                        explanation=result.get("explanation", "Generated using rule-based method"),
                        metadata={"query_plan": result.get("query_plan"), "rules_applied": result.get("rules_applied")},
                        execution_time=generation_time
                    )
            else:
                # Fallback
                result = await self.rule_builder.generate_sql(user_query, schema_info)
                if result.get("success"):
                    return SQLCandidate(
                        sql=result["sql"],
                        method=GenerationMethod.RULE_BASED,
                        confidence=0.8,
                        complexity="basic",
                        explanation="Generated using basic rule-based fallback",
                        metadata={},
                        execution_time=0.1
                    )
        except Exception as e:
            logger.error(f"Rule-based candidate generation failed: {e}")
        
        return None
    
    def _evaluate_and_rank_candidates(self, candidates: List[SQLCandidate], 
                                    requirements: QueryRequirements) -> List[SQLCandidate]:
        """Evaluate and rank SQL candidates"""
        
        for candidate in candidates:
            score = 0.0
            
            # Base confidence score (40% weight)
            score += candidate.confidence * 0.4
            
            # Complexity match score (20% weight)
            complexity_match = self._calculate_complexity_match(candidate.complexity, requirements.complexity_level)
            score += complexity_match * 0.2
            
            # Performance score (20% weight)
            performance_score = self._calculate_performance_score(candidate)
            score += performance_score * 0.2
            
            # Method reliability score (20% weight)
            method_reliability = self._get_method_reliability_score(candidate.method, requirements)
            score += method_reliability * 0.2
            
            # Store calculated score
            candidate.metadata["evaluation_score"] = score
        
        # Sort by evaluation score
        return sorted(candidates, key=lambda x: x.metadata.get("evaluation_score", 0), reverse=True)
    
    def _calculate_complexity_match(self, candidate_complexity: str, required_complexity: str) -> float:
        """Calculate how well candidate complexity matches requirements"""
        complexity_levels = ["basic", "intermediate", "advanced", "expert"]
        
        try:
            candidate_idx = complexity_levels.index(candidate_complexity)
            required_idx = complexity_levels.index(required_complexity)
            
            # Perfect match
            if candidate_idx == required_idx:
                return 1.0
            # One level difference
            elif abs(candidate_idx - required_idx) == 1:
                return 0.7
            # Two levels difference
            elif abs(candidate_idx - required_idx) == 2:
                return 0.4
            # Three levels difference
            else:
                return 0.1
                
        except ValueError:
            return 0.5  # Default if complexity level not found
    
    def _calculate_performance_score(self, candidate: SQLCandidate) -> float:
        """Calculate performance score based on execution time and estimated cost"""
        score = 1.0
        
        # Execution time penalty (faster is better)
        if candidate.execution_time:
            if candidate.execution_time < 0.1:
                score *= 1.0  # Very fast
            elif candidate.execution_time < 0.5:
                score *= 0.9  # Fast
            elif candidate.execution_time < 1.0:
                score *= 0.8  # Moderate
            else:
                score *= 0.6  # Slow
        
        return score
    
    def _get_method_reliability_score(self, method: GenerationMethod, requirements: QueryRequirements) -> float:
        """Get reliability score for method based on requirements"""
        
        # Method reliability for different complexity levels
        reliability_matrix = {
            GenerationMethod.RULE_BASED: {
                "basic": 0.9,
                "intermediate": 0.8,
                "advanced": 0.6,
                "expert": 0.4
            },
            GenerationMethod.AST: {
                "basic": 0.8,
                "intermediate": 0.8,
                "advanced": 0.7,
                "expert": 0.6
            },
            GenerationMethod.RAG: {
                "basic": 0.7,
                "intermediate": 0.9,
                "advanced": 0.9,
                "expert": 0.8
            },
            GenerationMethod.LLM: {
                "basic": 0.6,
                "intermediate": 0.7,
                "advanced": 0.8,
                "expert": 0.9
            }
        }
        
        return reliability_matrix.get(method, {}).get(requirements.complexity_level, 0.5)
    
    def _select_best_solution(self, ranked_candidates: List[SQLCandidate], 
                            requirements: QueryRequirements) -> SQLCandidate:
        """Select the best solution"""
        
        if not ranked_candidates:
            raise Exception("No candidates available for selection")
        
        best_candidate = ranked_candidates[0]
        
        # For this simplified version, just return the best candidate
        # In a full implementation, you might create hybrid solutions
        
        return best_candidate
    
    def _validate_and_optimize(self, solution: SQLCandidate, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and optimize the final SQL solution"""
        try:
            optimized_sql = solution.sql.strip()
            optimizations = []
            
            # Basic SQL validation
            if not optimized_sql.strip().upper().startswith("SELECT"):
                raise Exception("Generated SQL must start with SELECT")
            
            # Add LIMIT if not present for safety
            if "LIMIT" not in optimized_sql.upper():
                if solution.complexity in ["basic", "intermediate"]:
                    optimized_sql += " LIMIT 100"
                    optimizations.append("Added safety LIMIT")
                else:
                    optimized_sql += " LIMIT 50"
                    optimizations.append("Added safety LIMIT for complex query")
            
            # Format SQL for readability
            formatted_sql = self._format_sql(optimized_sql)
            if formatted_sql != optimized_sql:
                optimizations.append("Improved SQL formatting")
            
            return {
                "sql": formatted_sql,
                "optimizations": optimizations,
                "validation_passed": True
            }
            
        except Exception as e:
            logger.error(f"SQL validation failed: {e}")
            return {
                "sql": solution.sql,
                "optimizations": [],
                "validation_passed": False,
                "validation_error": str(e)
            }
    
    def _format_sql(self, sql: str) -> str:
        """Format SQL for better readability"""
        # Basic SQL formatting
        formatted = sql.strip()
        
        # Add line breaks for major clauses
        major_clauses = ["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING", "LIMIT"]
        
        for clause in major_clauses:
            formatted = formatted.replace(f" {clause} ", f"\n{clause} ")
        
        # Clean up extra whitespace
        lines = [line.strip() for line in formatted.split('\n') if line.strip()]
        
        return '\n'.join(lines)

class BasicSQLGenerator:
    """Basic fallback SQL generator for when advanced generators aren't available"""
    
    def __init__(self, method_name: str):
        self.method_name = method_name
        self.logger = logger
    
    async def generate_sql(self, user_query: str, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic SQL as fallback"""
        try:
            available_tables = schema_info.get("tables", [])
            
            if not available_tables:
                return {"success": False, "error": "No tables available"}
            
            query_lower = user_query.lower()
            
            # Very basic logic
            if any(word in query_lower for word in ["count", "how many"]):
                sql = f"SELECT COUNT(*) as count FROM {available_tables[0]}"
            else:
                sql = f"SELECT * FROM {available_tables[0]} LIMIT 50"
            
            return {
                "success": True,
                "sql": sql,
                "method": f"basic_{self.method_name}",
                "explanation": f"Basic {self.method_name} fallback implementation"
            }
            
        except Exception as e:
            logger.error(f"Basic SQL generation failed: {e}")
            return {"success": False, "error": str(e)}

# Test function
async def test_hybrid_sql_coordinator():
    """Test the hybrid SQL coordinator"""
    
    coordinator = HybridSQLCoordinator()
    
    schema_info = {
        "tables": ["customers", "orders", "products", "sales"],
        "columns": {
            "customers": ["id", "name", "email", "created_at", "status"],
            "orders": ["id", "customer_id", "product_id", "amount", "created_at", "status"],
            "products": ["id", "name", "category", "price"],
            "sales": ["id", "product_id", "customer_id", "amount", "created_at", "region"]
        }
    }
    
    test_queries = [
        "show me all customers",  # Basic
        "total revenue by product category",  # Intermediate
        "top 10 customers by total spend",  # Advanced
        "customer retention analysis"  # Expert
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        
        result = await coordinator.generate_optimal_sql(query, schema_info)
        
        if result["success"]:
            print(f"Method: {result['method']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Complexity: {result['complexity']}")
            print(f"Explanation: {result['explanation']}")
            print(f"\nSQL:\n{result['sql']}")
            
            if result.get("alternatives"):
                print(f"\nAlternatives generated: {len(result['alternatives'])}")
                for i, alt in enumerate(result["alternatives"]):
                    print(f"  {i+1}. {alt['method']} (confidence: {alt['confidence']:.2f})")
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    asyncio.run(test_hybrid_sql_coordinator())