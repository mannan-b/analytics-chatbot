# 🔍 DIAGNOSTIC SCRIPT - Find the Real Problem
# Run this: python diagnose.py

import os
import sys
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def diagnose():
    """Run comprehensive diagnostics"""

    print("="*80)
    print("🔍 TEXT-TO-SQL SYSTEM DIAGNOSTICS")
    print("="*80)
    print()

    issues = []

    # 1. Check environment variables
    print("1. Checking Environment Variables...")
    required_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", 
                     "SUPABASE_URL", "SUPABASE_SERVICE_KEY"]

    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"   ✅ {var}: {'*' * 10} (set)")
        else:
            print(f"   ❌ {var}: MISSING")
            issues.append(f"Missing environment variable: {var}")
    print()

    # 2. Check imports
    print("2. Checking Python Imports...")
    try:
        from utils.config import config
        print("   ✅ utils.config - OK")
    except Exception as e:
        print(f"   ❌ utils.config - FAILED: {e}")
        issues.append("Cannot import utils.config")

    try:
        from utils.logger import setup_logging
        print("   ✅ utils.logger - OK")
    except Exception as e:
        print(f"   ❌ utils.logger - FAILED: {e}")
        issues.append("Cannot import utils.logger")

    try:
        from database_supabase_client import CorrectedSupabaseClient
        print("   ✅ database_supabase_client - OK")
    except Exception as e:
        print(f"   ❌ database_supabase_client - FAILED: {e}")
        issues.append("Cannot import database_supabase_client")

    try:
        from services.vector_service import CorrectedVectorService
        print("   ✅ services.vector_service - OK")
    except Exception as e:
        print(f"   ❌ services.vector_service - FAILED: {e}")
        issues.append("Cannot import services.vector_service")

    print()

    # 3. Check if vector_service has required methods
    print("3. Checking Vector Service Methods...")
    try:
        from services.vector_service import CorrectedVectorService
        service = CorrectedVectorService()

        required_methods = [
            "create_embedding",
            "create_embeddings_batch",
            "search_similar_tables",
            "search_similar_columns",
            "search_similar_sql_patterns",
            "find_most_similar"
        ]

        for method in required_methods:
            if hasattr(service, method):
                print(f"   ✅ {method}() - EXISTS")
            else:
                print(f"   ❌ {method}() - MISSING")
                issues.append(f"Missing method: {method}")
    except Exception as e:
        print(f"   ❌ Cannot check methods: {e}")
        issues.append(f"Cannot instantiate vector_service: {e}")

    print()

    # 4. Check database client methods
    print("4. Checking Database Client Methods...")
    try:
        from database_supabase_client import CorrectedSupabaseClient

        required_methods = [
            "get_all_table_descriptions",
            "get_all_column_metadata",
            "get_columns_for_tables",
            "get_all_common_prompts",
            "search_business_context",
            "insert_business_context",
            "execute_sql"
        ]

        for method in required_methods:
            if hasattr(CorrectedSupabaseClient, method):
                print(f"   ✅ {method}() - EXISTS")
            else:
                print(f"   ❌ {method}() - MISSING")
                issues.append(f"Missing method: {method}")
    except Exception as e:
        print(f"   ❌ Cannot check methods: {e}")

    print()

    # 5. Check database connection and data
    print("5. Checking Supabase Connection & Data...")
    try:
        from supabase import create_client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

        if not supabase_url or not supabase_key:
            print("   ❌ Missing Supabase credentials")
            issues.append("Missing Supabase credentials")
        else:
            client = create_client(supabase_url, supabase_key)
            print("   ✅ Supabase client created")

            # Check tables
            tables = ["table_descriptions", "column_metadata", "common_prompt_sqls", "business_context"]
            for table in tables:
                try:
                    result = client.table(table).select("*").limit(1).execute()
                    count = len(result.data) if result.data else 0
                    if count > 0:
                        print(f"   ✅ {table}: Has data")
                    else:
                        print(f"   ⚠️  {table}: EMPTY (no data)")
                        issues.append(f"Table '{table}' is empty - needs data!")
                except Exception as e:
                    print(f"   ❌ {table}: Error - {e}")
                    issues.append(f"Cannot access table '{table}'")
    except Exception as e:
        print(f"   ❌ Supabase connection failed: {e}")
        issues.append(f"Supabase connection error: {e}")

    print()

    # 6. Summary
    print("="*80)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*80)

    if not issues:
        print("✅ All checks passed! Your system should work.")
    else:
        print(f"❌ Found {len(issues)} issue(s):\n")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")

    print()
    print("="*80)

    return issues

if __name__ == "__main__":
    issues = asyncio.run(diagnose())

    if issues:
        print("\n🔧 RECOMMENDED FIXES:\n")

        if any("Missing environment variable" in i for i in issues):
            print("1. Create/update .env file with all required keys")

        if any("Cannot import" in i for i in issues):
            print("2. Organize files: services/__init__.py and utils/__init__.py")

        if any("Missing method" in i for i in issues):
            print("3. Replace files with FIXED versions provided")

        if any("empty" in i.lower() for i in issues):
            print("4. CRITICAL: Populate your Supabase tables with data!")
            print("   Your tables are EMPTY - that's why queries fail")

        if any("Cannot access table" in i for i in issues):
            print("5. Run supabase_rpc_functions.sql in Supabase")
