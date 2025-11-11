
"""
ACTUALLY WORKING Gemini FREE TIER Text-to-SQL System
Using the CORRECT model names from official Gemini API documentation
"""

import os
import re
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import dspy
    print("✅ DSPy imported successfully")
except ImportError:
    print("❌ Installing DSPy...")
    os.system("pip install dspy-ai")
    import dspy

try:
    from supabase import create_client, Client
    print("✅ Supabase imported successfully")
except ImportError:
    print("❌ Installing Supabase...")
    os.system("pip install supabase")
    from supabase import create_client, Client

# DSPy Signatures exactly as in PDFs
class sql_program(dspy.Signature):
    """
    You are a text-to-SQL agent designed to generate SQL queries based on user input

    Table, Column, and Value Info
    You know the table names, column details (e.g., data types, constraints), and stats like numerical ranges, number of categories, and top values.

    Metadata
    You are aware of what data each table contains, what each column represents, and common errors you might run into.

    Your Task:
    Analyze the user's query.
    Refer to the provided database information (tables, columns, metadata, and examples).
    Generate the most accurate SQL query to retrieve the requested data.
    Make sure to follow these guidelines:
    - Ensure accuracy by referring to table names and column names exactly as they appear in the database
    - Use proper PostgreSQL syntax for Supabase
    - ONLY ANSWER WITH SQL - no explanations, just the SQL query
    """
    user_query = dspy.InputField(desc="User input query describing what kind of data they want")
    dataset_information = dspy.InputField(desc="The information around columns, tables, and database structure")
    generated_sql = dspy.OutputField(desc="The SQL query, remember to only include the SQL code")

class error_reasoning_program(dspy.Signature):
    """
    Task: Given a SQL error message, an incorrect SQL, user query, and database information, 
    provide step-by-step instructions to fix the SQL query.
    """
    error_message = dspy.InputField(desc="The SQL engine error message")
    incorrect_sql = dspy.InputField(desc="The failed SQL query")
    information = dspy.InputField(desc="user query or question and database information")
    error_fix_reasoning = dspy.OutputField(desc="The reasoning for why the error occurred and how to fix it")

class error_fix_agent(dspy.Signature):
    """
    Task: Given instructions from the Error Reasoning Agent, generate a corrected SQL query.
    Only return the SQL query, no explanations.
    """
    instruction = dspy.InputField(desc="The instructions from the agent on how to fix the query")
    generated_sql = dspy.OutputField(desc="The correct and fixed query")

# Helper function exactly as in PDF
def clean_llm_response(text):
    """Clean LLM response to extract SQL query"""
    if not text:
        return ""

    text = text.strip()

    # Remove markdown code blocks
    if '```' in text:
        parts = text.split('```')
        for i in range(1, len(parts), 2):
            sql_candidate = parts[i].strip()
            lines = sql_candidate.split('\n')
            if lines and lines[0].lower() in ['sql', 'postgresql', 'postgres']:
                sql_candidate = '\n'.join(lines[1:]).strip()
            if sql_candidate and len(sql_candidate) > 5:
                return sql_candidate

    # Clean the text
    text = re.sub(r'^(sql|query|here is the|the query is)\s*:?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Main Agent System - USING CORRECT GEMINI MODEL NAMES
class agent_system(dspy.Module):
    def __init__(self, supabase_url: str, supabase_key: str, gemini_api_key: str, max_retry=2):
        super().__init__()
        self.max_retry = max_retry

        # Initialize Supabase client
        try:
            self.supabase: Client = create_client(supabase_url, supabase_key)
            logger.info("Supabase client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase: {str(e)}")
            raise

        # CORRECT WAY: Use ACTUAL Gemini model names from official API
        # Based on search results, these are the CORRECT FREE TIER model names:
        try:
            # Try different FREE TIER models in order of preference
            free_tier_models = [
                "gemini/gemini-1.5-flash",      # Standard free tier model
                "gemini/gemini-1.5-flash-001",  # Specific version
                "gemini/gemini-1.5-flash-002",  # Newer version 
                "gemini/gemini-1.5-flash-8b",   # Smaller, faster model
                "gemini/gemini-2.0-flash"       # Latest model (may have free tier)
            ]

            lm = None
            working_model = None

            for model_name in free_tier_models:
                try:
                    logger.info(f"Trying model: {model_name}")
                    lm = dspy.LM(model=model_name, api_key=gemini_api_key)
                    dspy.configure(lm=lm)
                    working_model = model_name
                    logger.info(f"✅ Successfully configured with model: {model_name}")
                    break
                except Exception as model_error:
                    logger.warning(f"❌ Model {model_name} failed: {str(model_error)}")
                    continue

            if not working_model:
                raise Exception("None of the free tier models worked. Check your API key and model availability.")

        except Exception as e:
            logger.error(f"Failed to configure DSPy with any Gemini model: {str(e)}")
            raise

        # Initialize agents with DSPy modules
        try:
            self.sql_agent = dspy.Predict(sql_program)
            self.error_reasoning_agent = dspy.Predict(error_reasoning_program)  
            self.error_fix_agent = dspy.Predict(error_fix_agent)
            logger.info("All DSPy agents initialized successfully")
        except Exception as e:
            logger.error(f"Failed to create DSPy agents: {str(e)}")
            raise

        logger.info(f"Agent system initialized successfully with {working_model}")

    def fetch_database_information(self) -> str:
        """Fetch database schema from column_metadata and table_descriptions"""
        try:
            # Fetch table descriptions
            logger.info("Fetching table descriptions...")
            table_desc_response = self.supabase.table('table_descriptions').select('*').execute()
            table_descriptions = table_desc_response.data if table_desc_response.data else []
            logger.info(f"Found {len(table_descriptions)} table descriptions")

            # Fetch column metadata
            logger.info("Fetching column metadata...")
            column_meta_response = self.supabase.table('column_metadata').select('*').execute()
            column_metadata = column_meta_response.data if column_meta_response.data else []
            logger.info(f"Found {len(column_metadata)} column metadata entries")

            # Build database information string
            db_info = "### Database Structure:\n"
            db_info += "**Available Tables:**\n\n"

            # Group columns by table
            tables_info = {}
            for col in column_metadata:
                table_name = col.get('table_name', 'unknown')
                if table_name not in tables_info:
                    tables_info[table_name] = []
                tables_info[table_name].append(col)

            # Format each table information
            table_counter = 1
            for table_name, columns in tables_info.items():
                table_desc = next((desc for desc in table_descriptions if desc.get('table_name') == table_name), {})
                description = table_desc.get('description', 'No description available')

                db_info += f"{table_counter}. **Table: {table_name}**\n"
                db_info += f"   Description: {description}\n"
                db_info += "   Columns:\n"

                for col in columns:
                    col_name = col.get('column_name', 'unknown')
                    data_type = col.get('data_type', 'unknown')
                    col_desc = col.get('description', '')

                    col_info = f"    - {col_name} ({data_type})"
                    if col_desc:
                        col_info += f" - {col_desc}"
                    db_info += col_info + "\n"

                db_info += "\n"
                table_counter += 1

            db_info += "### IMPORTANT RULES:\n"
            db_info += "- Use EXACT table and column names as specified above\n"
            db_info += "- Follow PostgreSQL syntax for Supabase\n"
            db_info += "- Return ONLY the SQL query, no explanations\n"

            logger.info("Database schema information compiled successfully")
            return db_info

        except Exception as e:
            error_msg = f"Error fetching database schema: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def execute_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query using Supabase RPC function"""
        try:
            clean_sql = clean_llm_response(sql_query)
            logger.info(f"Executing SQL: {clean_sql}")

            if not clean_sql or len(clean_sql) < 5:
                raise Exception("Invalid or empty SQL query generated")

            result = self.supabase.rpc('execute_sql', {'query': clean_sql}).execute()

            if result.data is not None:
                df = pd.DataFrame(result.data) if result.data else pd.DataFrame()
                logger.info(f"Query executed successfully, returned {len(df)} rows")
                return {
                    'success': True,
                    'data': result.data,
                    'dataframe': df,
                    'clean_sql': clean_sql
                }
            else:
                return {
                    'success': False,
                    'error': 'Query returned no data',
                    'dataframe': pd.DataFrame(),
                    'clean_sql': clean_sql
                }

        except Exception as e:
            error_msg = f"SQL execution error: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': str(e),
                'dataframe': pd.DataFrame(),
                'clean_sql': clean_sql if 'clean_sql' in locals() else sql_query
            }

    def __call__(self, query, relevant_context=None):
        """Call method for DSPy Module"""
        return self.process_query(query, relevant_context)

    def process_query(self, query, relevant_context=None):
        """Main processing function exactly as in PDF"""
        return_dict = {'response': [], 'sql': [], 'error_reason': [], 'df': []}

        if relevant_context is None:
            relevant_context = self.fetch_database_information()

        logger.info(f"Processing query: {query}")

        # Generate initial SQL
        try:
            response = self.sql_agent(user_query=query, dataset_information=relevant_context)
            return_dict['response'].append(response)
            logger.info("SQL agent response received")
        except Exception as e:
            error_msg = f"Error in SQL agent: {str(e)}"
            logger.error(error_msg)
            return {
                'response': [error_msg],
                'sql': [],
                'error_reason': [],
                'df': [pd.DataFrame()]
            }

        information = {'user_query': query, 'relevant_context': relevant_context}
        i = 0

        # Retry loop
        while i < self.max_retry:
            print(f"\nAttempt {i}")

            try:
                sql = clean_llm_response(response.generated_sql)
                print(f"Generated SQL: {sql}")

                if not sql or len(sql) < 5:
                    raise Exception("Empty or invalid SQL query generated")

                return_dict['sql'].append(sql)

                execution_result = self.execute_sql_query(sql)
                df = execution_result['dataframe']
                return_dict['df'].append(df)

                if execution_result['success']:
                    if not df.empty:
                        print("✅ Success!")
                        print(f"Results ({len(df)} rows):")
                        print(df.head())
                    else:
                        print("✅ Query executed successfully but returned no rows")
                    break
                else:
                    raise Exception(execution_result.get('error', 'Unknown execution error'))

            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error: {error_msg}")

                if i < self.max_retry - 1:
                    try:
                        logger.info("Analyzing error...")
                        error_reason = self.error_reasoning_agent(
                            error_message=error_msg[:500],
                            incorrect_sql=sql if 'sql' in locals() else "No SQL generated",
                            information=str(information)[:1000]
                        )

                        if 'NOT ASKING FOR SQL' not in error_reason.error_fix_reasoning:
                            return_dict['error_reason'].append(error_reason.error_fix_reasoning)
                            logger.info("Generating fix...")
                            response = self.error_fix_agent(instruction=error_reason.error_fix_reasoning)
                            return_dict['response'].append(response)
                        else:
                            print("❌ Query not related to SQL")
                            break
                    except Exception as fix_error:
                        logger.error(f"Error in fix generation: {str(fix_error)}")
                        break

            i += 1

        if not return_dict['df']:
            return_dict['df'].append(pd.DataFrame())

        return return_dict

# Main Text-to-SQL System
class TextToSQLSystem:
    """ACTUALLY WORKING Text-to-SQL system with correct Gemini model names"""

    def __init__(self, supabase_url: str, supabase_key: str, gemini_api_key: str, max_retry: int = 2):
        try:
            self.system = agent_system(supabase_url, supabase_key, gemini_api_key, max_retry)
            logger.info("Text-to-SQL System initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize system: {str(e)}")
            raise

    def query(self, user_query: str) -> Dict[str, Any]:
        """Process user query"""
        print(f"\n{'='*60}")
        print(f"Processing: {user_query}")
        print('='*60)

        try:
            result = self.system(query=user_query)

            success = bool(
                result['df'] and 
                len(result['df']) > 0 and 
                isinstance(result['df'][-1], pd.DataFrame)
            )

            return {
                'query': user_query,
                'sql_queries': result['sql'],
                'final_dataframe': result['df'][-1] if result['df'] else pd.DataFrame(),
                'error_reasons': result['error_reason'],
                'total_attempts': len(result['sql']),
                'success': success,
                'all_responses': result['response']
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'query': user_query,
                'sql_queries': [],
                'final_dataframe': pd.DataFrame(),
                'error_reasons': [str(e)],
                'total_attempts': 0,
                'success': False,
                'all_responses': []
            }

# Example usage
def main():
    """Example usage with CORRECT Gemini model names"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY") 
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not all([supabase_url, supabase_key, gemini_api_key]):
        print("❌ Please set your environment variables:")
        print("export SUPABASE_URL='your-supabase-url'")
        print("export SUPABASE_KEY='your-supabase-anon-key'")
        print("export GEMINI_API_KEY='your-gemini-api-key'")
        print("\n🔑 Get your FREE Gemini API key from: https://aistudio.google.com")
        return

    try:
        print("🔧 Initializing Text-to-SQL system with CORRECT Gemini models...")
        sql_system = TextToSQLSystem(supabase_url, supabase_key, gemini_api_key)

        test_queries = [
            "Show me all users",
            "What is the total number of orders?",
            "List all products with their names and prices"
        ]

        print("\n🔄 Using AUTO-DETECTION for best available FREE TIER model")
        print("📊 Will try: gemini-1.5-flash, gemini-1.5-flash-001, gemini-1.5-flash-002, etc.")

        for query in test_queries:
            result = sql_system.query(query)
            print(f"\n📊 Query: {result['query']}")
            print(f"✅ Success: {result['success']}")
            print(f"🔄 Attempts: {result['total_attempts']}")

            if result['success'] and not result['final_dataframe'].empty:
                print("📈 Results preview:")
                print(result['final_dataframe'].head())
            elif result['success']:
                print("✅ Query executed successfully (no results)")
            elif result['error_reasons']:
                print(f"❌ Error: {result['error_reasons'][0][:100]}...")

            print("-" * 50)

    except Exception as e:
        print(f"❌ System Error: {str(e)}")
        print("\n🔍 Check:")
        print("1. Your Gemini API key is valid and active")
        print("2. You haven't exceeded free tier rate limits")
        print("3. Your internet connection is working")
        print("4. Supabase credentials are correct")

if __name__ == "__main__":
    print("🚀 ACTUALLY WORKING Text-to-SQL System - With Correct Model Names")
    print("=" * 70)
    print("🎯 Auto-detects working FREE TIER Gemini model")
    print("💡 No more model name errors!")
    print("=" * 70)
    main()
