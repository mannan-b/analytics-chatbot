# 🏗️ FIXED RULE-BASED SQL BUILDER - All compatibility issues resolved

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from utils.logger_config import get_logger

logger = get_logger(__name__)

class SQLOperation(Enum):
    SELECT = "SELECT"
    JOIN = "JOIN"
    WHERE = "WHERE"
    GROUP_BY = "GROUP BY"
    HAVING = "HAVING"
    ORDER_BY = "ORDER BY"
    LIMIT = "LIMIT"
    SUBQUERY = "SUBQUERY"
    CTE = "CTE"
    WINDOW = "WINDOW"
    UNION = "UNION"

@dataclass
class QueryRule:
    """Rule for query construction"""
    rule_id: str
    priority: int
    conditions: List[str]  # Conditions to match in user query
    sql_pattern: str
    variables: Dict[str, str]
    dependencies: List[str] = field(default_factory=list)
    complexity_score: int = 1

@dataclass
class QueryPlan:
    """Query execution plan with multiple steps"""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    joins: List[Dict[str, str]] = field(default_factory=list)
    aggregations: List[str] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    orderings: List[str] = field(default_factory=list)
    complexity_level: str = "basic"

class RuleBasedSQLBuilder:
    """Advanced SQL builder using rule-based query planning"""
    
    def __init__(self):
        self.logger = logger
        self.rules: List[QueryRule] = []
        self.query_templates = {}
        
        # Load comprehensive rule set
        self._load_sql_rules()
        self._load_query_templates()
    
    def _load_sql_rules(self):
        """Load comprehensive SQL construction rules"""
        
        # BASIC SELECTION RULES
        basic_rules = [
            QueryRule(
                rule_id="simple_select_all",
                priority=1,
                conditions=["show", "list", "display", "all"],
                sql_pattern="SELECT * FROM {table}",
                variables={"table": "{detected_table}"},
                complexity_score=1
            ),
            QueryRule(
                rule_id="count_records",
                priority=2,
                conditions=["how many", "count", "total number"],
                sql_pattern="SELECT COUNT(*) as count FROM {table}",
                variables={"table": "{detected_table}"},
                complexity_score=1
            ),
            QueryRule(
                rule_id="specific_columns",
                priority=3,
                conditions=["show", "get", "find"],
                sql_pattern="SELECT {columns} FROM {table}",
                variables={"columns": "{detected_columns}", "table": "{detected_table}"},
                complexity_score=2
            )
        ]
        
        # AGGREGATION RULES
        aggregation_rules = [
            QueryRule(
                rule_id="sum_aggregation",
                priority=10,
                conditions=["total", "sum", "add up"],
                sql_pattern="SELECT SUM({amount_column}) as total FROM {table}",
                variables={"amount_column": "{detected_amount_column}", "table": "{detected_table}"},
                complexity_score=3
            ),
            QueryRule(
                rule_id="average_calculation",
                priority=11,
                conditions=["average", "avg", "mean"],
                sql_pattern="SELECT AVG({amount_column}) as average FROM {table}",
                variables={"amount_column": "{detected_amount_column}", "table": "{detected_table}"},
                complexity_score=3
            ),
            QueryRule(
                rule_id="group_by_aggregation",
                priority=15,
                conditions=["by", "group by", "breakdown", "per"],
                sql_pattern="SELECT {group_column}, COUNT(*) as count, SUM({amount_column}) as total FROM {table} GROUP BY {group_column}",
                variables={
                    "group_column": "{detected_group_column}",
                    "amount_column": "{detected_amount_column}",
                    "table": "{detected_table}"
                },
                dependencies=["sum_aggregation", "count_records"],
                complexity_score=5
            )
        ]
        
        # FILTERING RULES
        filtering_rules = [
            QueryRule(
                rule_id="status_filter",
                priority=20,
                conditions=["active", "inactive", "completed", "pending"],
                sql_pattern="WHERE {status_column} = '{status_value}'",
                variables={
                    "status_column": "{detected_status_column}",
                    "status_value": "{detected_status_value}"
                },
                complexity_score=2
            ),
            QueryRule(
                rule_id="date_range_filter",
                priority=21,
                conditions=["last", "past", "recent", "since"],
                sql_pattern="WHERE {date_column} >= {date_expression}",
                variables={
                    "date_column": "{detected_date_column}",
                    "date_expression": "{calculated_date_expression}"
                },
                complexity_score=3
            )
        ]
        
        # JOIN RULES
        join_rules = [
            QueryRule(
                rule_id="customer_orders_join",
                priority=30,
                conditions=["customers with orders", "customer orders", "orders by customer"],
                sql_pattern="SELECT {select_columns} FROM customers c INNER JOIN orders o ON c.id = o.customer_id",
                variables={
                    "select_columns": "c.name, c.email, COUNT(o.id) as order_count, SUM(o.amount) as total_spent"
                },
                complexity_score=6
            ),
            QueryRule(
                rule_id="product_sales_join",
                priority=31,
                conditions=["products with sales", "product sales", "sales by product"],
                sql_pattern="SELECT {select_columns} FROM products p INNER JOIN sales s ON p.id = s.product_id",
                variables={
                    "select_columns": "p.name, p.category, COUNT(s.id) as sales_count, SUM(s.amount) as revenue"
                },
                complexity_score=6
            )
        ]
        
        # ADVANCED RULES
        advanced_rules = [
            QueryRule(
                rule_id="ranking_query",
                priority=40,
                conditions=["top", "best", "highest", "rank", "ranking"],
                sql_pattern="SELECT *, ROW_NUMBER() OVER (ORDER BY {rank_column} DESC) as rank FROM ({base_query}) ranked",
                variables={
                    "rank_column": "{detected_rank_column}",
                    "base_query": "{generated_base_query}"
                },
                dependencies=["group_by_aggregation"],
                complexity_score=8
            ),
            QueryRule(
                rule_id="time_series_analysis",
                priority=42,
                conditions=["trend", "over time", "monthly", "daily", "weekly"],
                sql_pattern="SELECT DATE_TRUNC('{period}', {date_column}) as period, COUNT(*) as count, SUM({amount_column}) as total FROM {table} WHERE {date_column} >= NOW() - INTERVAL '{timeframe}' GROUP BY DATE_TRUNC('{period}', {date_column}) ORDER BY period",
                variables={
                    "period": "{detected_period}",
                    "date_column": "{detected_date_column}",
                    "amount_column": "{detected_amount_column}",
                    "table": "{detected_table}",
                    "timeframe": "{detected_timeframe}"
                },
                dependencies=["group_by_aggregation", "date_range_filter"],
                complexity_score=10
            )
        ]
        
        # Combine all rules
        self.rules = basic_rules + aggregation_rules + filtering_rules + join_rules + advanced_rules
        
        # Sort by priority
        self.rules.sort(key=lambda x: x.priority)
        
        logger.info(f"Loaded {len(self.rules)} SQL construction rules")
    
    def _load_query_templates(self):
        """Load advanced query templates for complex scenarios"""
        
        self.query_templates = {
            "customer_analytics": {
                "customer_lifetime_value": """
                SELECT 
                    customer_id,
                    COUNT(DISTINCT order_id) as total_orders,
                    SUM(order_amount) as total_spent,
                    AVG(order_amount) as avg_order_value,
                    DATEDIFF(day, MIN(order_date), MAX(order_date)) as customer_lifespan_days
                FROM orders 
                WHERE status = 'completed'
                GROUP BY customer_id 
                HAVING COUNT(DISTINCT order_id) > 1
                ORDER BY total_spent DESC
                """,
                
                "customer_segmentation": """
                SELECT 
                    CASE 
                        WHEN total_spent >= 10000 THEN 'VIP'
                        WHEN total_spent >= 5000 THEN 'High Value'
                        WHEN total_spent >= 1000 THEN 'Medium Value'
                        ELSE 'Low Value'
                    END as customer_segment,
                    COUNT(*) as customer_count,
                    AVG(total_spent) as avg_segment_value
                FROM (
                    SELECT customer_id, SUM(amount) as total_spent
                    FROM orders 
                    WHERE status = 'completed'
                    GROUP BY customer_id
                ) customer_totals
                GROUP BY customer_segment
                ORDER BY avg_segment_value DESC
                """
            },
            
            "sales_analytics": {
                "monthly_growth": """
                WITH monthly_sales AS (
                    SELECT 
                        DATE_TRUNC('month', order_date) as month,
                        SUM(amount) as monthly_revenue
                    FROM orders 
                    WHERE status = 'completed'
                    GROUP BY DATE_TRUNC('month', order_date)
                )
                SELECT 
                    month,
                    monthly_revenue,
                    LAG(monthly_revenue) OVER (ORDER BY month) as prev_month_revenue,
                    CASE 
                        WHEN LAG(monthly_revenue) OVER (ORDER BY month) IS NOT NULL
                        THEN ((monthly_revenue - LAG(monthly_revenue) OVER (ORDER BY month)) / LAG(monthly_revenue) OVER (ORDER BY month)) * 100
                        ELSE NULL
                    END as growth_percentage
                FROM monthly_sales
                ORDER BY month
                """
            }
        }
    
    async def build_complex_sql(self, user_query: str, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build complex SQL using rule-based approach"""
        try:
            # Step 1: Create query plan
            query_plan = self._create_query_plan(user_query, schema_info)
            
            if not query_plan.steps:
                return {"success": False, "error": "Could not create query plan"}
            
            # Step 2: Apply matching rules
            applicable_rules = self._find_applicable_rules(user_query, schema_info)
            
            if not applicable_rules:
                return {"success": False, "error": "No applicable rules found"}
            
            # Step 3: Check for template matches
            template_sql = self._check_template_match(user_query, schema_info)
            
            if template_sql:
                return {
                    "success": True,
                    "sql": template_sql,
                    "method": "template_match",
                    "complexity": "expert",
                    "explanation": "Used pre-built expert template for complex analysis"
                }
            
            # Step 4: Build SQL from rules and plan
            sql_query = self._build_sql_from_rules(applicable_rules, query_plan, schema_info)
            
            if not sql_query:
                return {"success": False, "error": "Could not construct SQL from rules"}
            
            # Step 5: Optimize and validate
            optimized_sql = self._optimize_rule_based_sql(sql_query, query_plan)
            
            return {
                "success": True,
                "sql": optimized_sql,
                "method": "rule_based",
                "complexity": query_plan.complexity_level,
                "query_plan": self._plan_to_dict(query_plan),
                "rules_applied": [rule.rule_id for rule in applicable_rules],
                "explanation": self._explain_rule_application(applicable_rules, query_plan)
            }
            
        except Exception as e:
            logger.error(f"Rule-based SQL building failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_query_plan(self, user_query: str, schema_info: Dict[str, Any]) -> QueryPlan:
        """Create comprehensive query execution plan"""
        plan = QueryPlan()
        query_lower = user_query.lower()
        
        # Analyze query complexity
        complexity_indicators = {
            "basic": ["show", "list", "get", "count"],
            "intermediate": ["group by", "sum", "average", "order by"],
            "advanced": ["join", "with", "rank", "top"],
            "expert": ["cohort", "retention", "trend analysis", "cte", "window"]
        }
        
        for level, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                plan.complexity_level = level
        
        # Detect required operations
        operations = []
        
        if any(word in query_lower for word in ["join", "with", "combined"]):
            operations.append(SQLOperation.JOIN)
        
        if any(word in query_lower for word in ["group by", "by", "breakdown"]):
            operations.append(SQLOperation.GROUP_BY)
        
        if any(word in query_lower for word in ["where", "filter", "active", "completed"]):
            operations.append(SQLOperation.WHERE)
        
        if any(word in query_lower for word in ["order", "top", "highest", "sort"]):
            operations.append(SQLOperation.ORDER_BY)
        
        # Add steps to plan
        for op in operations:
            plan.steps.append({
                "operation": op.value,
                "priority": self._get_operation_priority(op),
                "required": True
            })
        
        # Sort steps by priority
        plan.steps.sort(key=lambda x: x["priority"])
        
        return plan
    
    def _find_applicable_rules(self, user_query: str, schema_info: Dict[str, Any]) -> List[QueryRule]:
        """Find all rules applicable to the user query"""
        applicable_rules = []
        query_lower = user_query.lower()
        
        for rule in self.rules:
            # Check if rule conditions match
            condition_matches = 0
            for condition in rule.conditions:
                if condition in query_lower:
                    condition_matches += 1
            
            # Rule applies if at least one condition matches
            if condition_matches > 0:
                # Calculate match score
                match_score = condition_matches / len(rule.conditions)
                
                # Check schema compatibility
                schema_compatible = self._check_rule_schema_compatibility(rule, schema_info)
                
                if schema_compatible:
                    # Add match score as attribute
                    rule.match_score = match_score
                    applicable_rules.append(rule)
        
        # Sort by priority and match score
        applicable_rules.sort(key=lambda x: (x.priority, -getattr(x, 'match_score', 0)))
        
        return applicable_rules
    
    def _check_template_match(self, user_query: str, schema_info: Dict[str, Any]) -> Optional[str]:
        """Check if query matches a pre-built template"""
        query_lower = user_query.lower()
        
        # Customer analytics templates
        if any(phrase in query_lower for phrase in ["customer lifetime value", "clv", "customer value"]):
            return self.query_templates["customer_analytics"]["customer_lifetime_value"]
        
        if any(phrase in query_lower for phrase in ["customer segment", "customer classification", "customer groups"]):
            return self.query_templates["customer_analytics"]["customer_segmentation"]
        
        # Sales analytics templates
        if any(phrase in query_lower for phrase in ["monthly growth", "growth rate", "month over month"]):
            return self.query_templates["sales_analytics"]["monthly_growth"]
        
        return None
    
    def _build_sql_from_rules(self, rules: List[QueryRule], plan: QueryPlan, schema_info: Dict[str, Any]) -> Optional[str]:
        """Build SQL query from applicable rules and plan"""
        try:
            if not rules:
                return None
            
            # Start with the highest priority rule as base
            base_rule = rules[0]
            
            # Fill variables in base rule
            filled_pattern = self._fill_rule_variables(base_rule, schema_info)
            if not filled_pattern:
                return None
            
            # For this simplified version, return the filled pattern
            # In a full implementation, you would merge multiple rules
            
            return filled_pattern
            
        except Exception as e:
            logger.error(f"Failed to build SQL from rules: {e}")
            return None
    
    def _fill_rule_variables(self, rule: QueryRule, schema_info: Dict[str, Any]) -> Optional[str]:
        """Fill variables in rule SQL pattern"""
        try:
            pattern = rule.sql_pattern
            available_tables = schema_info.get("tables", [])
            available_columns = schema_info.get("columns", {})
            
            # Replace variables
            for var_name, var_value in rule.variables.items():
                if var_value.startswith("{") and var_value.endswith("}"):
                    # Dynamic variable - need to resolve
                    resolved_value = self._resolve_dynamic_variable(var_value, schema_info)
                    if resolved_value:
                        pattern = pattern.replace(f"{{{var_name}}}", resolved_value)
                else:
                    # Static variable
                    pattern = pattern.replace(f"{{{var_name}}}", var_value)
            
            return pattern
            
        except Exception as e:
            logger.error(f"Failed to fill rule variables: {e}")
            return None
    
    def _resolve_dynamic_variable(self, var_placeholder: str, schema_info: Dict[str, Any]) -> Optional[str]:
        """Resolve dynamic variables like {detected_table}, {detected_amount_column}"""
        var_name = var_placeholder.strip("{}")
        available_tables = schema_info.get("tables", [])
        available_columns = schema_info.get("columns", {})
        
        if var_name == "detected_table":
            return available_tables[0] if available_tables else None
        
        elif var_name == "detected_amount_column":
            for table, columns in available_columns.items():
                for col in columns:
                    if any(keyword in col.lower() for keyword in ["amount", "total", "price", "value", "revenue"]):
                        return col
            return None
        
        elif var_name == "detected_date_column":
            for table, columns in available_columns.items():
                for col in columns:
                    if any(keyword in col.lower() for keyword in ["date", "created", "updated", "time"]):
                        return col
            return None
        
        elif var_name == "detected_status_column":
            for table, columns in available_columns.items():
                for col in columns:
                    if "status" in col.lower():
                        return col
            return None
        
        elif var_name == "detected_group_column":
            for table, columns in available_columns.items():
                for col in columns:
                    if any(keyword in col.lower() for keyword in ["category", "type", "group", "region"]):
                        return col
            return None
        
        elif var_name == "detected_columns":
            # Return first few columns
            for table, columns in available_columns.items():
                if columns:
                    return ", ".join(columns[:3])
            return "*"
        
        return None
    
    def _optimize_rule_based_sql(self, sql: str, plan: QueryPlan) -> str:
        """Optimize SQL generated from rules"""
        # Add performance optimizations
        optimized = sql.strip()
        
        # Add appropriate limits based on complexity
        if "LIMIT" not in optimized.upper():
            if plan.complexity_level in ["basic", "intermediate"]:
                optimized += " LIMIT 100"
            else:
                optimized += " LIMIT 50"
        
        return optimized
    
    def _get_operation_priority(self, operation: SQLOperation) -> int:
        """Get priority order for SQL operations"""
        priorities = {
            SQLOperation.SELECT: 1,
            SQLOperation.JOIN: 2,
            SQLOperation.WHERE: 3,
            SQLOperation.GROUP_BY: 4,
            SQLOperation.HAVING: 5,
            SQLOperation.ORDER_BY: 6,
            SQLOperation.LIMIT: 7
        }
        return priorities.get(operation, 10)
    
    def _check_rule_schema_compatibility(self, rule: QueryRule, schema_info: Dict[str, Any]) -> bool:
        """Check if rule is compatible with available schema"""
        # Basic compatibility check
        return len(schema_info.get("tables", [])) > 0
    
    def _plan_to_dict(self, plan: QueryPlan) -> Dict[str, Any]:
        """Convert query plan to dictionary"""
        return {
            "steps": plan.steps,
            "joins": plan.joins,
            "aggregations": plan.aggregations,
            "filters": plan.filters,
            "orderings": plan.orderings,
            "complexity_level": plan.complexity_level
        }
    
    def _explain_rule_application(self, rules: List[QueryRule], plan: QueryPlan) -> str:
        """Explain how rules were applied"""
        explanations = []
        
        explanations.append(f"Applied {len(rules)} rules with {plan.complexity_level} complexity")
        
        for rule in rules[:3]:  # Show top 3 rules
            explanations.append(f"- {rule.rule_id}: {rule.sql_pattern[:50]}...")
        
        if len(plan.steps) > 0:
            operations = [step["operation"] for step in plan.steps]
            explanations.append(f"Query operations: {', '.join(operations)}")
        
        return ". ".join(explanations)

# Test function
async def test_rule_based_sql_builder():
    """Test the rule-based SQL builder"""
    builder = RuleBasedSQLBuilder()
    
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
        "customer lifetime value"
    ]
    
    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        result = await builder.build_complex_sql(query, schema_info)
        if result["success"]:
            print(f"SQL: {result['sql']}")
            print(f"Method: {result['method']}")
            print(f"Complexity: {result['complexity']}")
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_rule_based_sql_builder())