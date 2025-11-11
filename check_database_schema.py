#!/usr/bin/env python3
# 🔍 CHECK YOUR DATABASE SCHEMA
# This shows you EXACTLY what tables/columns exist

import os
import asyncio
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

async def check_schema():
    """Check what's actually in your Supabase database"""

    print("="*80)
    print("🔍 CHECKING YOUR DATABASE SCHEMA")
    print("="*80)
    print()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

    if not supabase_url or not supabase_key:
        print("❌ Missing Supabase credentials in .env")
        return

    client = create_client(supabase_url, supabase_key)

    # Check what's in table_descriptions
    print("1️⃣  TABLE DESCRIPTIONS")
    print("-"*80)
    try:
        result = client.table("table_descriptions").select("table_name, table_description").execute()
        if result.data:
            for table in result.data:
                print(f"   📊 {table['table_name']}")
                print(f"      Description: {table['table_description']}")
                print()
        else:
            print("   ❌ NO TABLES FOUND")
            print("   Your table_descriptions table is EMPTY!")
            print()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print()

    # Check what's in column_metadata
    print("2️⃣  COLUMN METADATA")
    print("-"*80)
    try:
        result = client.table("column_metadata").select("table_name, column_name, column_description").execute()
        if result.data:
            # Group by table
            tables = {}
            for col in result.data:
                table_name = col['table_name']
                if table_name not in tables:
                    tables[table_name] = []
                tables[table_name].append(col)

            for table_name, columns in tables.items():
                print(f"   📊 {table_name}:")
                for col in columns:
                    print(f"      - {col['column_name']}: {col['column_description']}")
                print()
        else:
            print("   ❌ NO COLUMNS FOUND")
            print("   Your column_metadata table is EMPTY!")
            print()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print()

    # Check common patterns
    print("3️⃣  SQL PATTERNS")
    print("-"*80)
    try:
        result = client.table("common_prompt_sqls").select("prompt, sql_query").limit(5).execute()
        if result.data:
            for pattern in result.data:
                print(f"   💡 {pattern['prompt']}")
                print(f"      SQL: {pattern['sql_query'][:100]}...")
                print()
        else:
            print("   ❌ NO PATTERNS FOUND")
            print()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print()

    print("="*80)
    print("🎯 DIAGNOSIS")
    print("="*80)
    print()
    print("If you see empty tables above, that's your problem!")
    print("The LLM has no schema information, so it can't generate SQL.")
    print()
    print("Solution:")
    print("  1. Populate your Supabase tables with actual schema")
    print("  2. Or run: python setup_real_database.py")

if __name__ == "__main__":
    asyncio.run(check_schema())
