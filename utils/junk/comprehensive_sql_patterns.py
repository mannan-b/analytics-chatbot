# 🚀 COMPREHENSIVE SQL PATTERNS DATABASE - EVERY POSSIBLE QUERY
"""
Complete systematic SQL patterns covering:
- 1 table, 2 tables, 3+ tables
- SELECT, INSERT, UPDATE, DELETE
- COUNT, SUM, AVG, MIN, MAX, GROUP BY
- JOIN (INNER, LEFT, RIGHT, FULL)
- Subqueries, CTEs, Window functions
- ALIAS, DISTINCT, ORDER BY, LIMIT
- Date functions, string functions
- Complex analytics queries
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ComprehensiveSQLPatterns:
    """Complete SQL patterns database - systematically organized"""
    
    def __init__(self):
        self.patterns = self._build_all_patterns()
        logger.info(f"🎯 Built {len(self.patterns)} comprehensive SQL patterns")
    
    def _build_all_patterns(self) -> List[Dict[str, Any]]:
        """Build every possible SQL pattern systematically"""
        patterns = []
        
        # 1. SINGLE TABLE QUERIES
        patterns.extend(self._single_table_patterns())
        
        # 2. TWO TABLE QUERIES  
        patterns.extend(self._two_table_patterns())
        
        # 3. THREE+ TABLE QUERIES
        patterns.extend(self._multi_table_patterns())
        
        # 4. AGGREGATION QUERIES
        patterns.extend(self._aggregation_patterns())
        
        # 5. ANALYTICS QUERIES
        patterns.extend(self._analytics_patterns())
        
        # 6. TIME-BASED QUERIES
        patterns.extend(self._temporal_patterns())
        
        # 7. STRING/TEXT QUERIES
        patterns.extend(self._text_patterns())
        
        # 8. COMPARISON QUERIES
        patterns.extend(self._comparison_patterns())
        
        # 9. RANKING/TOP-N QUERIES
        patterns.extend(self._ranking_patterns())
        
        # 10. BUSINESS INTELLIGENCE QUERIES
        patterns.extend(self._business_intelligence_patterns())
        
        return patterns
    
    def _single_table_patterns(self) -> List[Dict[str, Any]]:
        """All possible single table query patterns"""
        return [
            # BASIC SELECT
            {
                "natural_language": "show all {table}",
                "sql_template": "SELECT * FROM {table}",
                "variations": ["display all {table}", "get all {table}", "list all {table}", "fetch all {table}"],
                "category": "basic_select",
                "complexity": "basic",
                "table_count": 1,
                "description": "Select all records from a table"
            },
            {
                "natural_language": "show {table} details",
                "sql_template": "SELECT * FROM {table}",
                "variations": ["get {table} information", "{table} data", "view {table}"],
                "category": "basic_select",
                "complexity": "basic",
                "table_count": 1,
                "description": "View table details"
            },
            {
                "natural_language": "show first 10 {table}",
                "sql_template": "SELECT * FROM {table} LIMIT 10",
                "variations": ["first 10 {table}", "top 10 {table}", "10 {table} records"],
                "category": "limited_select",
                "complexity": "basic",
                "table_count": 1,
                "description": "Get first N records"
            },
            {
                "natural_language": "show last 5 {table}",
                "sql_template": "SELECT * FROM {table} ORDER BY id DESC LIMIT 5",
                "variations": ["latest 5 {table}", "recent 5 {table}", "newest 5 {table}"],
                "category": "limited_select",
                "complexity": "basic",
                "table_count": 1,
                "description": "Get latest N records"
            },
            
            # COLUMN SELECTION
            {
                "natural_language": "show {column} from {table}",
                "sql_template": "SELECT {column} FROM {table}",
                "variations": ["get {column} from {table}", "list {column} from {table}"],
                "category": "column_select",
                "complexity": "basic",
                "table_count": 1,
                "description": "Select specific column"
            },
            {
                "natural_language": "show {column1} and {column2} from {table}",
                "sql_template": "SELECT {column1}, {column2} FROM {table}",
                "variations": ["get {column1} and {column2} from {table}", "list {column1}, {column2} from {table}"],
                "category": "multi_column_select",
                "complexity": "basic",
                "table_count": 1,
                "description": "Select multiple columns"
            },
            
            # FILTERING
            {
                "natural_language": "show {table} where {column} equals {value}",
                "sql_template": "SELECT * FROM {table} WHERE {column} = '{value}'",
                "variations": ["{table} where {column} is {value}", "{table} with {column} {value}"],
                "category": "filtered_select",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Filter by exact match"
            },
            {
                "natural_language": "show {table} where {column} contains {value}",
                "sql_template": "SELECT * FROM {table} WHERE {column} LIKE '%{value}%'",
                "variations": ["{table} with {column} containing {value}", "{table} having {column} {value}"],
                "category": "text_search",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Text search in column"
            },
            {
                "natural_language": "show {table} where {column} starts with {value}",
                "sql_template": "SELECT * FROM {table} WHERE {column} LIKE '{value}%'",
                "variations": ["{table} beginning with {value}", "{table} starting with {value}"],
                "category": "text_search",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Prefix text search"
            },
            {
                "natural_language": "show {table} where {column} ends with {value}",
                "sql_template": "SELECT * FROM {table} WHERE {column} LIKE '%{value}'",
                "variations": ["{table} ending with {value}", "{table} finishing with {value}"],
                "category": "text_search",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Suffix text search"
            },
            
            # NUMERIC FILTERING
            {
                "natural_language": "show {table} where {column} greater than {value}",
                "sql_template": "SELECT * FROM {table} WHERE {column} > {value}",
                "variations": ["{table} with {column} > {value}", "{table} where {column} more than {value}"],
                "category": "numeric_filter",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Greater than filter"
            },
            {
                "natural_language": "show {table} where {column} less than {value}",
                "sql_template": "SELECT * FROM {table} WHERE {column} < {value}",
                "variations": ["{table} with {column} < {value}", "{table} where {column} below {value}"],
                "category": "numeric_filter",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Less than filter"
            },
            {
                "natural_language": "show {table} where {column} between {value1} and {value2}",
                "sql_template": "SELECT * FROM {table} WHERE {column} BETWEEN {value1} AND {value2}",
                "variations": ["{table} with {column} from {value1} to {value2}"],
                "category": "range_filter",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Range filter"
            },
            
            # SORTING
            {
                "natural_language": "show {table} sorted by {column}",
                "sql_template": "SELECT * FROM {table} ORDER BY {column}",
                "variations": ["{table} ordered by {column}", "{table} arranged by {column}"],
                "category": "sorted_select",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Sort ascending"
            },
            {
                "natural_language": "show {table} sorted by {column} descending",
                "sql_template": "SELECT * FROM {table} ORDER BY {column} DESC",
                "variations": ["{table} ordered by {column} desc", "{table} by {column} highest first"],
                "category": "sorted_select",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Sort descending"
            },
            
            # DISTINCT
            {
                "natural_language": "show unique {column} from {table}",
                "sql_template": "SELECT DISTINCT {column} FROM {table}",
                "variations": ["distinct {column} from {table}", "different {column} from {table}"],
                "category": "distinct_select",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Get unique values"
            },
            
            # COUNTING
            {
                "natural_language": "count {table}",
                "sql_template": "SELECT COUNT(*) as total_count FROM {table}",
                "variations": ["how many {table}", "number of {table}", "total {table}"],
                "category": "count",
                "complexity": "basic",
                "table_count": 1,
                "description": "Count all records"
            },
            {
                "natural_language": "count {table} where {column} equals {value}",
                "sql_template": "SELECT COUNT(*) as count FROM {table} WHERE {column} = '{value}'",
                "variations": ["how many {table} where {column} is {value}"],
                "category": "conditional_count",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Count with condition"
            },
        ]
    
    def _two_table_patterns(self) -> List[Dict[str, Any]]:
        """All possible two table query patterns"""
        return [
            # INNER JOIN
            {
                "natural_language": "show {table1} with {table2}",
                "sql_template": "SELECT * FROM {table1} t1 INNER JOIN {table2} t2 ON t1.{join_column} = t2.{join_column}",
                "variations": ["get {table1} and {table2}", "join {table1} with {table2}"],
                "category": "inner_join",
                "complexity": "intermediate",
                "table_count": 2,
                "description": "Inner join two tables"
            },
            {
                "natural_language": "show {table1} and their {table2}",
                "sql_template": "SELECT t1.*, t2.* FROM {table1} t1 LEFT JOIN {table2} t2 ON t1.id = t2.{table1}_id",
                "variations": ["get {table1} with their {table2}", "{table1} and related {table2}"],
                "category": "left_join",
                "complexity": "intermediate",
                "table_count": 2,
                "description": "Left join to get related records"
            },
            
            # SPECIFIC JOIN QUERIES
            {
                "natural_language": "show {table1} with {table2} details",
                "sql_template": "SELECT t1.*, t2.* FROM {table1} t1 INNER JOIN {table2} t2 ON t1.{join_key} = t2.id",
                "variations": ["{table1} including {table2} information"],
                "category": "detailed_join",
                "complexity": "intermediate",
                "table_count": 2,
                "description": "Detailed join with all columns"
            },
            
            # AGGREGATION WITH JOIN
            {
                "natural_language": "count {table1} by {table2}",
                "sql_template": "SELECT t2.name, COUNT(t1.id) as count FROM {table1} t1 INNER JOIN {table2} t2 ON t1.{table2}_id = t2.id GROUP BY t2.id, t2.name",
                "variations": ["how many {table1} per {table2}", "{table1} count by {table2}"],
                "category": "join_aggregation",
                "complexity": "advanced",
                "table_count": 2,
                "description": "Count with grouping across tables"
            },
            
            # SUM WITH JOIN
            {
                "natural_language": "total {column} by {table2}",
                "sql_template": "SELECT t2.name, SUM(t1.{column}) as total FROM {table1} t1 INNER JOIN {table2} t2 ON t1.{table2}_id = t2.id GROUP BY t2.id, t2.name",
                "variations": ["sum of {column} by {table2}", "{column} total per {table2}"],
                "category": "join_sum",
                "complexity": "advanced",
                "table_count": 2,
                "description": "Sum with grouping across tables"
            }
        ]
    
    def _multi_table_patterns(self) -> List[Dict[str, Any]]:
        """Three or more table query patterns"""
        return [
            # THREE TABLE JOIN
            {
                "natural_language": "show {table1} with {table2} and {table3}",
                "sql_template": """SELECT t1.*, t2.*, t3.* 
                                  FROM {table1} t1 
                                  INNER JOIN {table2} t2 ON t1.{table2}_id = t2.id 
                                  INNER JOIN {table3} t3 ON t2.{table3}_id = t3.id""",
                "variations": ["{table1} including {table2} and {table3}"],
                "category": "triple_join",
                "complexity": "advanced",
                "table_count": 3,
                "description": "Three table join"
            }
        ]
    
    def _aggregation_patterns(self) -> List[Dict[str, Any]]:
        """All aggregation function patterns"""
        return [
            # COUNT
            {
                "natural_language": "count total {table}",
                "sql_template": "SELECT COUNT(*) as total FROM {table}",
                "variations": ["total number of {table}", "how many {table}"],
                "category": "count_all",
                "complexity": "basic",
                "table_count": 1,
                "description": "Count all records"
            },
            {
                "natural_language": "count distinct {column} in {table}",
                "sql_template": "SELECT COUNT(DISTINCT {column}) as unique_count FROM {table}",
                "variations": ["how many different {column}", "unique {column} count"],
                "category": "count_distinct",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Count unique values"
            },
            
            # SUM
            {
                "natural_language": "sum of {column} in {table}",
                "sql_template": "SELECT SUM({column}) as total FROM {table}",
                "variations": ["total {column}", "{column} sum", "add up {column}"],
                "category": "sum",
                "complexity": "basic",
                "table_count": 1,
                "description": "Sum numeric column"
            },
            {
                "natural_language": "sum {column} by {group_column}",
                "sql_template": "SELECT {group_column}, SUM({column}) as total FROM {table} GROUP BY {group_column}",
                "variations": ["{column} total by {group_column}", "{column} sum grouped by {group_column}"],
                "category": "grouped_sum",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Sum with grouping"
            },
            
            # AVERAGE
            {
                "natural_language": "average {column} in {table}",
                "sql_template": "SELECT AVG({column}) as average FROM {table}",
                "variations": ["mean {column}", "{column} average", "avg {column}"],
                "category": "average",
                "complexity": "basic",
                "table_count": 1,
                "description": "Calculate average"
            },
            {
                "natural_language": "average {column} by {group_column}",
                "sql_template": "SELECT {group_column}, AVG({column}) as average FROM {table} GROUP BY {group_column}",
                "variations": ["{column} average by {group_column}", "mean {column} grouped by {group_column}"],
                "category": "grouped_average",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Average with grouping"
            },
            
            # MIN/MAX
            {
                "natural_language": "maximum {column} in {table}",
                "sql_template": "SELECT MAX({column}) as maximum FROM {table}",
                "variations": ["highest {column}", "max {column}", "largest {column}"],
                "category": "max",
                "complexity": "basic",
                "table_count": 1,
                "description": "Find maximum value"
            },
            {
                "natural_language": "minimum {column} in {table}",
                "sql_template": "SELECT MIN({column}) as minimum FROM {table}",
                "variations": ["lowest {column}", "min {column}", "smallest {column}"],
                "category": "min",
                "complexity": "basic",
                "table_count": 1,
                "description": "Find minimum value"
            }
        ]
    
    def _analytics_patterns(self) -> List[Dict[str, Any]]:
        """Advanced analytics query patterns"""
        return [
            # WINDOW FUNCTIONS
            {
                "natural_language": "rank {table} by {column}",
                "sql_template": "SELECT *, RANK() OVER (ORDER BY {column} DESC) as rank FROM {table}",
                "variations": ["ranking of {table} by {column}", "{table} ranked by {column}"],
                "category": "ranking",
                "complexity": "advanced",
                "table_count": 1,
                "description": "Rank records by column"
            },
            {
                "natural_language": "running total of {column} in {table}",
                "sql_template": "SELECT *, SUM({column}) OVER (ORDER BY id) as running_total FROM {table}",
                "variations": ["cumulative sum of {column}", "{column} running sum"],
                "category": "running_total",
                "complexity": "advanced",
                "table_count": 1,
                "description": "Calculate running total"
            },
            
            # PERCENTILES
            {
                "natural_language": "percentile of {column} in {table}",
                "sql_template": "SELECT {column}, PERCENT_RANK() OVER (ORDER BY {column}) as percentile FROM {table}",
                "variations": ["{column} percentile", "percentage rank of {column}"],
                "category": "percentile",
                "complexity": "expert",
                "table_count": 1,
                "description": "Calculate percentile ranks"
            }
        ]
    
    def _temporal_patterns(self) -> List[Dict[str, Any]]:
        """Time-based query patterns"""
        return [
            # DATE FILTERING
            {
                "natural_language": "show {table} from today",
                "sql_template": "SELECT * FROM {table} WHERE DATE({date_column}) = CURRENT_DATE",
                "variations": ["today's {table}", "{table} for today"],
                "category": "date_filter",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Filter by today's date"
            },
            {
                "natural_language": "show {table} from last week",
                "sql_template": "SELECT * FROM {table} WHERE {date_column} >= CURRENT_DATE - INTERVAL '7 days'",
                "variations": ["past week {table}", "{table} from past 7 days"],
                "category": "date_range",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Filter by last week"
            },
            {
                "natural_language": "show {table} from last month",
                "sql_template": "SELECT * FROM {table} WHERE {date_column} >= CURRENT_DATE - INTERVAL '30 days'",
                "variations": ["past month {table}", "{table} from past 30 days"],
                "category": "date_range",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Filter by last month"
            },
            
            # DATE GROUPING
            {
                "natural_language": "count {table} by date",
                "sql_template": "SELECT DATE({date_column}) as date, COUNT(*) as count FROM {table} GROUP BY DATE({date_column}) ORDER BY date",
                "variations": ["{table} count by day", "daily {table} count"],
                "category": "date_grouping",
                "complexity": "advanced",
                "table_count": 1,
                "description": "Group by date"
            },
            {
                "natural_language": "monthly {column} total",
                "sql_template": "SELECT DATE_TRUNC('month', {date_column}) as month, SUM({column}) as total FROM {table} GROUP BY DATE_TRUNC('month', {date_column}) ORDER BY month",
                "variations": ["{column} total by month", "monthly {column} sum"],
                "category": "monthly_aggregation",
                "complexity": "advanced",
                "table_count": 1,
                "description": "Monthly aggregation"
            }
        ]
    
    def _text_patterns(self) -> List[Dict[str, Any]]:
        """Text/string query patterns"""
        return [
            {
                "natural_language": "search {table} for {text}",
                "sql_template": "SELECT * FROM {table} WHERE {text_column} ILIKE '%{text}%'",
                "variations": ["find {text} in {table}", "{table} containing {text}"],
                "category": "text_search",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Case-insensitive text search"
            }
        ]
    
    def _comparison_patterns(self) -> List[Dict[str, Any]]:
        """Comparison query patterns"""
        return [
            {
                "natural_language": "compare {column} between {table1} and {table2}",
                "sql_template": """SELECT 
                                    '{table1}' as table_name, AVG({column}) as avg_value FROM {table1}
                                  UNION ALL
                                  SELECT 
                                    '{table2}' as table_name, AVG({column}) as avg_value FROM {table2}""",
                "variations": ["{column} comparison between {table1} and {table2}"],
                "category": "table_comparison",
                "complexity": "advanced",
                "table_count": 2,
                "description": "Compare metrics between tables"
            }
        ]
    
    def _ranking_patterns(self) -> List[Dict[str, Any]]:
        """Ranking and top-N query patterns"""
        return [
            {
                "natural_language": "top 10 {table} by {column}",
                "sql_template": "SELECT * FROM {table} ORDER BY {column} DESC LIMIT 10",
                "variations": ["highest 10 {table} by {column}", "best 10 {table} by {column}"],
                "category": "top_n",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Get top N records"
            },
            {
                "natural_language": "bottom 5 {table} by {column}",
                "sql_template": "SELECT * FROM {table} ORDER BY {column} ASC LIMIT 5",
                "variations": ["lowest 5 {table} by {column}", "worst 5 {table} by {column}"],
                "category": "bottom_n",
                "complexity": "intermediate",
                "table_count": 1,
                "description": "Get bottom N records"
            }
        ]
    
    def _business_intelligence_patterns(self) -> List[Dict[str, Any]]:
        """Complex business intelligence query patterns"""
        return [
            # COHORT ANALYSIS
            {
                "natural_language": "{table1} cohort analysis by {table2}",
                "sql_template": """WITH cohorts AS (
                                    SELECT 
                                      t1.id,
                                      t1.{date_column},
                                      DATE_TRUNC('month', t1.{date_column}) as cohort_month,
                                      t2.name as segment
                                    FROM {table1} t1
                                    INNER JOIN {table2} t2 ON t1.{table2}_id = t2.id
                                  )
                                  SELECT 
                                    cohort_month,
                                    segment,
                                    COUNT(*) as count
                                  FROM cohorts
                                  GROUP BY cohort_month, segment
                                  ORDER BY cohort_month, segment""",
                "variations": ["cohort analysis for {table1}", "{table1} segmented by {table2}"],
                "category": "cohort_analysis",
                "complexity": "expert",
                "table_count": 2,
                "description": "Cohort analysis with segmentation"
            },
            
            # FUNNEL ANALYSIS
            {
                "natural_language": "{table} conversion funnel",
                "sql_template": """SELECT 
                                    stage,
                                    COUNT(*) as count,
                                    COUNT(*) * 100.0 / LAG(COUNT(*)) OVER (ORDER BY stage) as conversion_rate
                                  FROM {table}
                                  GROUP BY stage
                                  ORDER BY stage""",
                "variations": ["{table} funnel analysis", "conversion rates for {table}"],
                "category": "funnel_analysis",
                "complexity": "expert",
                "table_count": 1,
                "description": "Conversion funnel analysis"
            }
        ]
    
    def get_all_patterns(self) -> List[Dict[str, Any]]:
        """Return all patterns"""
        return self.patterns
    
    def get_patterns_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get patterns by category"""
        return [p for p in self.patterns if p.get('category') == category]
    
    def get_patterns_by_complexity(self, complexity: str) -> List[Dict[str, Any]]:
        """Get patterns by complexity level"""
        return [p for p in self.patterns if p.get('complexity') == complexity]
    
    def get_patterns_by_table_count(self, table_count: int) -> List[Dict[str, Any]]:
        """Get patterns by number of tables"""
        return [p for p in self.patterns if p.get('table_count') == table_count]
    
    def export_for_supabase(self) -> List[Dict[str, Any]]:
        """Export patterns in Supabase-ready format"""
        supabase_patterns = []
        
        for pattern in self.patterns:
            # Main pattern
            main_pattern = {
                "natural_language": pattern["natural_language"],
                "sql_template": pattern["sql_template"],
                "category": pattern.get("category", "unknown"),
                "complexity": pattern.get("complexity", "basic"),
                "table_count": pattern.get("table_count", 1),
                "description": pattern.get("description", ""),
                "keywords": self._extract_keywords(pattern["natural_language"]),
                "created_at": datetime.now().isoformat()
            }
            supabase_patterns.append(main_pattern)
            
            # Add variations
            for variation in pattern.get("variations", []):
                variation_pattern = {
                    **main_pattern,
                    "natural_language": variation,
                    "is_variation": True,
                    "parent_pattern": pattern["natural_language"]
                }
                supabase_patterns.append(variation_pattern)
        
        return supabase_patterns
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from natural language text"""
        # Simple keyword extraction
        words = text.lower().replace('{', '').replace('}', '').split()
        keywords = [word for word in words if len(word) > 2 and word not in ['the', 'and', 'from', 'where', 'with']]
        return keywords

# Create comprehensive patterns instance
comprehensive_patterns = ComprehensiveSQLPatterns()

if __name__ == "__main__":
    patterns = comprehensive_patterns.get_all_patterns()
    print(f"🎯 Generated {len(patterns)} comprehensive SQL patterns")
    
    # Show pattern breakdown
    categories = {}
    complexities = {}
    
    for pattern in patterns:
        cat = pattern.get('category', 'unknown')
        comp = pattern.get('complexity', 'unknown')
        
        categories[cat] = categories.get(cat, 0) + 1
        complexities[comp] = complexities.get(comp, 0) + 1
    
    print("\n📊 Pattern Categories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    print("\n🎚️ Complexity Levels:")
    for comp, count in sorted(complexities.items()):
        print(f"  {comp}: {count}")
    
    # Export for Supabase
    supabase_data = comprehensive_patterns.export_for_supabase()
    print(f"\n💾 Supabase export ready: {len(supabase_data)} records")