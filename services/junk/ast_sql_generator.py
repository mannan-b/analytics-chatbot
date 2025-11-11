# 🎯 FIXED AST SQL GENERATOR - All compatibility issues resolved

import logging
import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from utils.logger_config import get_logger

logger = get_logger(__name__)

class QueryType(Enum):
    SELECT = "select"
    AGGREGATE = "aggregate"
    JOIN = "join"
    SUBQUERY = "subquery"
    WINDOW = "window"
    CTE = "cte"
    UNION = "union"

class ColumnType(Enum):
    IDENTIFIER = "id"
    NUMERIC = "numeric"
    TEXT = "text"
    DATE = "date"
    BOOLEAN = "boolean"

@dataclass
class QueryComponent:
    """Represents a component of the SQL query"""
    component_type: str
    tables: List[str]
    columns: List[str]
    conditions: List[str]
    aggregations: List[str]
    groupings: List[str]
    orderings: List[str]
    limits: Optional[int]

@dataclass
class ParsedIntent:
    """Parsed user intent with semantic understanding"""
    action: str  # SELECT, COUNT, SUM, etc.
    entities: List[str]  # customers, orders, products
    attributes: List[str]  # name, amount, date
    relationships: List[str]  # with, by, from
    conditions: List[str]  # where, having
    temporal: List[str]  # last month, daily, etc.
    aggregation: List[str]  # total, average, count
    ordering: List[str]  # top, bottom, highest

class ASTSQLGenerator:
    """Generate complex SQL using Abstract Syntax Tree approach"""
    
    def __init__(self):
        self.logger = logger
        
        # Semantic patterns for intent parsing
        self.action_patterns = {
            "select": ["show", "list", "display", "get", "find", "retrieve", "view"],
            "count": ["how many", "count", "number of", "total number"],
            "sum": ["total", "sum", "add up", "combine"],
            "average": ["average", "avg", "mean"],
            "max": ["highest", "maximum", "max", "greatest", "top"],
            "min": ["lowest", "minimum", "min", "smallest", "bottom"],
            "group": ["by", "group by", "breakdown", "categorize"],
            "join": ["with", "join", "combine", "merge", "relate"],
            "rank": ["rank", "ranking", "top", "best", "worst"]
        }
        
        self.entity_patterns = {
            "customer": ["customer", "client", "user", "buyer", "person"],
            "order": ["order", "purchase", "transaction", "sale", "booking"],
            "product": ["product", "item", "goods", "service", "merchandise"],
            "employee": ["employee", "staff", "worker", "personnel", "team member"],
            "category": ["category", "type", "group", "classification", "segment"],
            "region": ["region", "area", "location", "territory", "zone"],
            "time": ["time", "date", "period", "duration", "interval"]
        }
        
        self.condition_patterns = {
            "equals": ["is", "equals", "=", "exactly"],
            "greater": ["more than", "greater than", "above", "over", ">"],
            "less": ["less than", "below", "under", "fewer than", "<"],
            "between": ["between", "from", "to", "range"],
            "like": ["contains", "includes", "like", "similar to"],
            "in": ["in", "among", "one of", "within"],
            "recent": ["recent", "last", "past", "latest"],
            "active": ["active", "current", "ongoing", "live"],
            "completed": ["completed", "finished", "done", "closed"]
        }
        
        self.temporal_patterns = {
            "daily": ["daily", "per day", "each day", "day by day"],
            "weekly": ["weekly", "per week", "each week", "week by week"],
            "monthly": ["monthly", "per month", "each month", "month by month"],
            "yearly": ["yearly", "annual", "per year", "each year"],
            "last_30_days": ["last 30 days", "past month", "recent month"],
            "last_week": ["last week", "past week", "previous week"],
            "this_year": ["this year", "current year", "ytd"],
            "quarter": ["quarter", "quarterly", "q1", "q2", "q3", "q4"]
        }
    
    async def generate_ast_sql(self, user_query: str, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SQL using AST approach"""
        try:
            # Step 1: Parse user intent into semantic components
            parsed_intent = self._parse_user_intent(user_query)
            
            # Step 2: Build Abstract Syntax Tree
            ast = self._build_query_ast(parsed_intent, schema_info)
            
            if not ast:
                return {"success": False, "error": "Could not parse query intent"}
            
            # Step 3: Generate SQL from AST
            sql = self._generate_sql_from_ast(ast, schema_info)
            
            if not sql:
                return {"success": False, "error": "Could not generate SQL from AST"}
            
            # Step 4: Optimize and validate
            optimized_sql = self._optimize_generated_sql(sql, schema_info)
            
            return {
                "success": True,
                "sql": optimized_sql,
                "ast": self._ast_to_dict(ast),
                "parsed_intent": self._intent_to_dict(parsed_intent),
                "complexity": self._calculate_complexity(ast),
                "explanation": self._explain_query_logic(ast, parsed_intent)
            }
            
        except Exception as e:
            logger.error(f"AST SQL generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _parse_user_intent(self, user_query: str) -> ParsedIntent:
        """Parse user query into semantic intent components"""
        query_lower = user_query.lower()
        
        # Extract actions
        actions = []
        for action, patterns in self.action_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                actions.append(action)
        
        # Extract entities
        entities = []
        for entity, patterns in self.entity_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                entities.append(entity)
        
        # Extract conditions
        conditions = []
        for condition, patterns in self.condition_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                conditions.append(condition)
        
        # Extract temporal elements
        temporal = []
        for time_pattern, patterns in self.temporal_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                temporal.append(time_pattern)
        
        # Extract attributes (look for column-like words)
        attribute_keywords = ["name", "email", "amount", "price", "total", "count", "date", "status", "type"]
        attributes = [attr for attr in attribute_keywords if attr in query_lower]
        
        # Extract relationships
        relationship_keywords = ["with", "by", "from", "in", "on", "for"]
        relationships = [rel for rel in relationship_keywords if rel in query_lower]
        
        # Extract aggregation hints
        aggregation = []
        if any(word in query_lower for word in ["total", "sum", "add"]):
            aggregation.append("sum")
        if any(word in query_lower for word in ["average", "avg", "mean"]):
            aggregation.append("avg")
        if any(word in query_lower for word in ["count", "number", "how many"]):
            aggregation.append("count")
        
        # Extract ordering hints
        ordering = []
        if any(word in query_lower for word in ["top", "highest", "best", "descending"]):
            ordering.append("desc")
        if any(word in query_lower for word in ["bottom", "lowest", "worst", "ascending"]):
            ordering.append("asc")
        
        return ParsedIntent(
            action=actions[0] if actions else "select",
            entities=entities,
            attributes=attributes,
            relationships=relationships,
            conditions=conditions,
            temporal=temporal,
            aggregation=aggregation,
            ordering=ordering
        )
    
    def _build_query_ast(self, intent: ParsedIntent, schema_info: Dict[str, Any]) -> Optional[QueryComponent]:
        """Build Abstract Syntax Tree from parsed intent"""
        try:
            available_tables = schema_info.get("tables", [])
            available_columns = schema_info.get("columns", {})
            
            # Determine primary table
            primary_table = self._resolve_primary_table(intent.entities, available_tables)
            if not primary_table:
                return None
            
            # Build query components
            component = QueryComponent(
                component_type=self._determine_query_type(intent),
                tables=[primary_table],
                columns=[],
                conditions=[],
                aggregations=[],
                groupings=[],
                orderings=[],
                limits=None
            )
            
            # Add columns based on intent
            if intent.action in ["select", "show", "list"]:
                if intent.attributes:
                    # Specific attributes requested
                    component.columns = self._resolve_columns(intent.attributes, primary_table, available_columns)
                else:
                    # All columns or smart selection
                    component.columns = ["*"]
            
            # Add aggregations
            if intent.aggregation:
                for agg in intent.aggregation:
                    if agg == "count":
                        component.aggregations.append("COUNT(*) as count")
                    elif agg == "sum":
                        amount_col = self._find_amount_column(primary_table, available_columns)
                        if amount_col:
                            component.aggregations.append(f"SUM({amount_col}) as total")
                    elif agg == "avg":
                        amount_col = self._find_amount_column(primary_table, available_columns)
                        if amount_col:
                            component.aggregations.append(f"AVG({amount_col}) as average")
            
            # Add grouping
            if "by" in intent.relationships or intent.action == "group":
                group_column = self._find_grouping_column(intent, primary_table, available_columns)
                if group_column:
                    component.groupings.append(group_column)
                    if group_column not in component.columns and "*" not in component.columns:
                        component.columns.insert(0, group_column)
            
            # Add conditions
            component.conditions = self._build_conditions(intent, primary_table, available_columns)
            
            # Add ordering
            if intent.ordering:
                order_column = self._find_ordering_column(intent, primary_table, available_columns)
                if order_column:
                    order_direction = "DESC" if "desc" in intent.ordering else "ASC"
                    component.orderings.append(f"{order_column} {order_direction}")
            
            # Add temporal conditions
            if intent.temporal:
                temporal_conditions = self._build_temporal_conditions(intent.temporal, primary_table, available_columns)
                component.conditions.extend(temporal_conditions)
            
            # Set limits
            component.limits = self._determine_limit(intent)
            
            # Check if we need joins (simplified)
            if len(intent.entities) > 1 or "with" in intent.relationships:
                # For this fixed version, we'll keep it simple and not implement full joins
                pass
            
            return component
            
        except Exception as e:
            logger.error(f"Failed to build query AST: {e}")
            return None
    
    def _generate_sql_from_ast(self, ast: QueryComponent, schema_info: Dict[str, Any]) -> Optional[str]:
        """Generate SQL query from AST"""
        try:
            # Build SELECT clause
            if ast.aggregations and not ast.columns:
                select_clause = ", ".join(ast.aggregations)
            elif ast.aggregations and ast.groupings:
                group_cols = ast.groupings
                agg_cols = ast.aggregations
                select_clause = ", ".join(group_cols + agg_cols)
            elif ast.columns:
                select_clause = ", ".join(ast.columns)
            else:
                select_clause = "*"
            
            # Build FROM clause
            from_clause = ast.tables[0] if ast.tables else "unknown_table"
            
            # Build WHERE clause
            where_clauses = ast.conditions
            
            # Build GROUP BY clause
            group_by_clause = ast.groupings
            
            # Build ORDER BY clause
            order_by_clause = ast.orderings
            
            # Assemble final SQL
            sql_parts = [f"SELECT {select_clause}", f"FROM {from_clause}"]
            
            if where_clauses:
                sql_parts.append(f"WHERE {' AND '.join(where_clauses)}")
            
            if group_by_clause:
                sql_parts.append(f"GROUP BY {', '.join(group_by_clause)}")
            
            if order_by_clause:
                sql_parts.append(f"ORDER BY {', '.join(order_by_clause)}")
            
            if ast.limits:
                sql_parts.append(f"LIMIT {ast.limits}")
            
            return "\n".join(sql_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate SQL from AST: {e}")
            return None
    
    def _resolve_primary_table(self, entities: List[str], available_tables: List[str]) -> Optional[str]:
        """Resolve the primary table from entities"""
        if not entities or not available_tables:
            return available_tables[0] if available_tables else None
        
        # Direct matches
        for entity in entities:
            entity_plural = entity + "s"
            for table in available_tables:
                if entity.lower() in table.lower() or entity_plural.lower() in table.lower():
                    return table
        
        # Semantic matches
        entity_table_map = {
            "customer": ["customers", "clients", "users"],
            "order": ["orders", "purchases", "transactions", "sales"],
            "product": ["products", "items", "inventory"],
            "employee": ["employees", "staff", "workers"]
        }
        
        for entity in entities:
            if entity in entity_table_map:
                for candidate in entity_table_map[entity]:
                    if candidate in available_tables:
                        return candidate
        
        return available_tables[0] if available_tables else None
    
    def _resolve_columns(self, attributes: List[str], table: str, available_columns: Dict[str, List[str]]) -> List[str]:
        """Resolve column names from attributes"""
        if table not in available_columns:
            return ["*"]
        
        table_columns = available_columns[table]
        resolved = []
        
        for attr in attributes:
            # Direct match
            if attr in table_columns:
                resolved.append(attr)
                continue
            
            # Fuzzy match
            for col in table_columns:
                if attr.lower() in col.lower() or col.lower() in attr.lower():
                    resolved.append(col)
                    break
        
        return resolved if resolved else ["*"]
    
    def _build_conditions(self, intent: ParsedIntent, table: str, available_columns: Dict[str, List[str]]) -> List[str]:
        """Build WHERE conditions from intent"""
        conditions = []
        
        if table not in available_columns:
            return conditions
        
        table_columns = available_columns[table]
        
        # Status conditions
        if "active" in intent.conditions:
            status_col = self._find_column_by_pattern(table_columns, ["status", "state"])
            if status_col:
                conditions.append(f"{status_col} = 'active'")
        
        if "completed" in intent.conditions:
            status_col = self._find_column_by_pattern(table_columns, ["status", "state"])
            if status_col:
                conditions.append(f"{status_col} = 'completed'")
        
        return conditions
    
    def _build_temporal_conditions(self, temporal: List[str], table: str, available_columns: Dict[str, List[str]]) -> List[str]:
        """Build temporal WHERE conditions"""
        conditions = []
        
        if table not in available_columns:
            return conditions
        
        table_columns = available_columns[table]
        date_col = self._find_column_by_pattern(table_columns, ["date", "created", "updated", "time"])
        
        if not date_col:
            return conditions
        
        for temp in temporal:
            if temp == "last_30_days":
                conditions.append(f"{date_col} >= NOW() - INTERVAL '30 days'")
            elif temp == "this_year":
                conditions.append(f"EXTRACT(YEAR FROM {date_col}) = EXTRACT(YEAR FROM NOW())")
            elif temp == "last_week":
                conditions.append(f"{date_col} >= NOW() - INTERVAL '7 days'")
        
        return conditions
    
    def _find_column_by_pattern(self, columns: List[str], patterns: List[str]) -> Optional[str]:
        """Find column matching patterns"""
        for pattern in patterns:
            for col in columns:
                if pattern.lower() in col.lower():
                    return col
        return None
    
    def _find_amount_column(self, table: str, available_columns: Dict[str, List[str]]) -> Optional[str]:
        """Find column representing monetary amounts"""
        if table not in available_columns:
            return None
        
        amount_patterns = ["amount", "total", "price", "value", "cost", "revenue"]
        return self._find_column_by_pattern(available_columns[table], amount_patterns)
    
    def _find_grouping_column(self, intent: ParsedIntent, table: str, available_columns: Dict[str, List[str]]) -> Optional[str]:
        """Find appropriate column for grouping"""
        if table not in available_columns:
            return None
        
        grouping_patterns = ["category", "type", "status", "region", "department"]
        
        # Check if specific attribute mentioned
        for attr in intent.attributes:
            if attr in available_columns[table]:
                return attr
        
        return self._find_column_by_pattern(available_columns[table], grouping_patterns)
    
    def _find_ordering_column(self, intent: ParsedIntent, table: str, available_columns: Dict[str, List[str]]) -> Optional[str]:
        """Find appropriate column for ordering"""
        if table not in available_columns:
            return None
        
        # If aggregation is used, order by aggregated column
        if intent.aggregation:
            if "sum" in intent.aggregation:
                return self._find_amount_column(table, available_columns)
            elif "count" in intent.aggregation:
                return "count"
        
        # Default to date column
        return self._find_column_by_pattern(available_columns[table], ["date", "created", "updated"])
    
    def _determine_query_type(self, intent: ParsedIntent) -> str:
        """Determine the type of query from intent"""
        if intent.aggregation:
            return QueryType.AGGREGATE.value
        elif len(intent.entities) > 1 or "with" in intent.relationships:
            return QueryType.JOIN.value
        else:
            return QueryType.SELECT.value
    
    def _determine_limit(self, intent: ParsedIntent) -> int:
        """Determine appropriate LIMIT from intent"""
        if intent.action in ["count", "sum", "average"]:
            return 1  # Aggregations return single row
        elif "top" in intent.ordering:
            return 10  # Top N queries
        else:
            return 50  # Default limit
    
    def _optimize_generated_sql(self, sql: str, schema_info: Dict[str, Any]) -> str:
        """Optimize the generated SQL"""
        optimized = sql.strip()
        
        # Ensure proper formatting
        lines = optimized.split('\n')
        formatted_lines = []
        for line in lines:
            formatted_lines.append(line.strip())
        
        return '\n'.join(formatted_lines)
    
    def _calculate_complexity(self, ast: QueryComponent) -> str:
        """Calculate query complexity level"""
        complexity_score = 0
        
        if len(ast.tables) > 1:
            complexity_score += 2  # JOINs
        if ast.aggregations:
            complexity_score += 1  # Aggregations
        if ast.groupings:
            complexity_score += 1  # GROUP BY
        if len(ast.conditions) > 2:
            complexity_score += 1  # Multiple conditions
        
        if complexity_score >= 4:
            return "expert"
        elif complexity_score >= 2:
            return "advanced"
        elif complexity_score >= 1:
            return "intermediate"
        else:
            return "basic"
    
    def _explain_query_logic(self, ast: QueryComponent, intent: ParsedIntent) -> str:
        """Generate human-readable explanation of the query logic"""
        explanations = []
        
        if ast.aggregations:
            explanations.append(f"Calculating {', '.join(intent.aggregation)} from {ast.tables[0]}")
        else:
            explanations.append(f"Retrieving data from {ast.tables[0]}")
        
        if len(ast.tables) > 1:
            explanations.append(f"Joining with {', '.join(ast.tables[1:])}")
        
        if ast.conditions:
            explanations.append(f"Filtering by {len(ast.conditions)} conditions")
        
        if ast.groupings:
            explanations.append(f"Grouping by {', '.join(ast.groupings)}")
        
        if ast.orderings:
            explanations.append(f"Ordering by {', '.join(ast.orderings)}")
        
        return ". ".join(explanations)
    
    def _ast_to_dict(self, ast: QueryComponent) -> Dict[str, Any]:
        """Convert AST to dictionary for JSON serialization"""
        return {
            "component_type": ast.component_type,
            "tables": ast.tables,
            "columns": ast.columns,
            "conditions": ast.conditions,
            "aggregations": ast.aggregations,
            "groupings": ast.groupings,
            "orderings": ast.orderings,
            "limits": ast.limits
        }
    
    def _intent_to_dict(self, intent: ParsedIntent) -> Dict[str, Any]:
        """Convert parsed intent to dictionary"""
        return {
            "action": intent.action,
            "entities": intent.entities,
            "attributes": intent.attributes,
            "relationships": intent.relationships,
            "conditions": intent.conditions,
            "temporal": intent.temporal,
            "aggregation": intent.aggregation,
            "ordering": intent.ordering
        }

# Test function
async def test_ast_sql_generator():
    """Test the AST SQL generator"""
    generator = ASTSQLGenerator()
    
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
        "count total orders",
        "total sales by category",
        "top customers by amount"
    ]
    
    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        result = await generator.generate_ast_sql(query, schema_info)
        if result["success"]:
            print(f"SQL: {result['sql']}")
            print(f"Complexity: {result['complexity']}")
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_ast_sql_generator())