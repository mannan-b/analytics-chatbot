#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE DATABASE CHECK
Tells you EXACTLY what data exists and what's missing
"""

import asyncio
import sys
from database_supabase_client import CorrectedSupabaseClient

async def check_database():
    print("🔍 COMPREHENSIVE DATABASE CHECK")
    print("="*80)
    print()
    
    client = CorrectedSupabaseClient()
    
    # Check tables
    print("📊 CHECKING: table_descriptions")
    print("-"*80)
    tables = await client.get_all_table_descriptions()
    print(f"✅ Total rows: {len(tables)}")
    
    if tables:
        print(f"\n📋 Sample data from first row:")
        sample = tables[0]
        print(f"   - table_name: {sample.get('table_name', 'MISSING')}")
        print(f"   - table_description: {sample.get('table_description', 'MISSING')[:80]}...")
        print(f"   - embedding key exists: {('embedding' in sample)}")
        print(f"   - embedding is not None: {sample.get('embedding') is not None}")
        
        if sample.get('embedding'):
            emb = sample.get('embedding')
            print(f"   - embedding type: {type(emb)}")
            print(f"   - embedding length: {len(emb) if isinstance(emb, (list, tuple)) else 'N/A'}")
            print(f"   - embedding sample: {emb[:3] if isinstance(emb, (list, tuple)) else emb}")
        else:
            print(f"   ❌ embedding is NULL or missing!")
        
        # Count embeddings
        with_embeddings = sum(1 for t in tables if t.get('embedding') is not None)
        print(f"\n📊 Tables WITH embeddings: {with_embeddings}/{len(tables)}")
        
        if with_embeddings == 0:
            print("   ❌ NO EMBEDDINGS FOUND!")
            print("   🔧 You need to generate embeddings for your tables")
        
        print(f"\n📋 All tables in database:")
        for i, t in enumerate(tables, 1):
            has_emb = "✅" if t.get('embedding') else "❌"
            print(f"   {i}. {has_emb} {t.get('table_name', 'unknown')}")
    else:
        print("   ❌ NO TABLES FOUND IN DATABASE!")
        print("   🔧 Your table_descriptions table is empty")
    
    # Check columns
    print("\n" + "="*80)
    print("📋 CHECKING: column_metadata")
    print("-"*80)
    columns = await client.get_all_column_metadata()
    print(f"✅ Total rows: {len(columns)}")
    
    if columns:
        print(f"\n📋 Sample data from first row:")
        sample = columns[0]
        print(f"   - table_name: {sample.get('table_name', 'MISSING')}")
        print(f"   - column_name: {sample.get('column_name', 'MISSING')}")
        print(f"   - column_description: {sample.get('column_description', 'MISSING')[:60]}...")
        print(f"   - embedding key exists: {('embedding' in sample)}")
        print(f"   - embedding is not None: {sample.get('embedding') is not None}")
        
        if sample.get('embedding'):
            emb = sample.get('embedding')
            print(f"   - embedding type: {type(emb)}")
            print(f"   - embedding length: {len(emb) if isinstance(emb, (list, tuple)) else 'N/A'}")
        else:
            print(f"   ❌ embedding is NULL or missing!")
        
        # Count embeddings
        with_embeddings = sum(1 for c in columns if c.get('embedding') is not None)
        print(f"\n📊 Columns WITH embeddings: {with_embeddings}/{len(columns)}")
        
        if with_embeddings == 0:
            print("   ❌ NO EMBEDDINGS FOUND!")
            print("   🔧 You need to generate embeddings for your columns")
    else:
        print("   ❌ NO COLUMNS FOUND IN DATABASE!")
        print("   🔧 Your column_metadata table is empty")
    
    # Check business context
    print("\n" + "="*80)
    print("💼 CHECKING: business_context")
    print("-"*80)
    contexts = await client.get_all_business_context()
    print(f"✅ Total rows: {len(contexts)}")
    
    if contexts:
        print(f"\n📋 Sample data from first row:")
        sample = contexts[0]
        print(f"   - document_piece: {sample.get('document_piece', 'MISSING')[:60]}...")
        print(f"   - embedding key exists: {('embedding' in sample)}")
        print(f"   - embedding is not None: {sample.get('embedding') is not None}")
        
        if sample.get('embedding'):
            emb = sample.get('embedding')
            print(f"   - embedding type: {type(emb)}")
            print(f"   - embedding length: {len(emb) if isinstance(emb, (list, tuple)) else 'N/A'}")
        else:
            print(f"   ❌ embedding is NULL or missing!")
        
        with_embeddings = sum(1 for c in contexts if c.get('embedding') is not None)
        print(f"\n📊 Contexts WITH embeddings: {with_embeddings}/{len(contexts)}")
    else:
        print("   ⚠️ NO BUSINESS CONTEXT FOUND (optional)")
    
    # SUMMARY
    print("\n" + "="*80)
    print("📊 SUMMARY & DIAGNOSIS")
    print("="*80)
    
    tables_with_emb = sum(1 for t in tables if t.get('embedding') is not None) if tables else 0
    cols_with_emb = sum(1 for c in columns if c.get('embedding') is not None) if columns else 0
    
    print(f"\n✅ Tables: {len(tables)} rows, {tables_with_emb} with embeddings")
    print(f"✅ Columns: {len(columns)} rows, {cols_with_emb} with embeddings")
    print(f"✅ Business Contexts: {len(contexts)} rows")
    
    # Diagnosis
    print("\n🔍 DIAGNOSIS:")
    
    if len(tables) == 0:
        print("❌ CRITICAL: table_descriptions is EMPTY")
        print("   → Add data to table_descriptions table in Supabase")
    elif tables_with_emb == 0:
        print("❌ CRITICAL: table_descriptions has NO EMBEDDINGS")
        print("   → Generate embeddings for your table descriptions")
        print("   → Check if column is named 'embedding' or 'vector_embedding'")
    else:
        print("✅ table_descriptions looks good!")
    
    if len(columns) == 0:
        print("❌ CRITICAL: column_metadata is EMPTY")
        print("   → Add data to column_metadata table in Supabase")
    elif cols_with_emb == 0:
        print("❌ CRITICAL: column_metadata has NO EMBEDDINGS")
        print("   → Generate embeddings for your column metadata")
    else:
        print("✅ column_metadata looks good!")
    
    # Final verdict
    print("\n" + "="*80)
    if tables_with_emb > 0 and cols_with_emb > 0:
        print("🎉 DATABASE IS READY! You can start querying.")
    else:
        print("⚠️ DATABASE NEEDS SETUP!")
        print("\nNEXT STEPS:")
        print("1. Make sure tables have data")
        print("2. Generate embeddings for all rows")
        print("3. Re-run this check")
    print("="*80)

if __name__ == "__main__":
    try:
        asyncio.run(check_database())
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
