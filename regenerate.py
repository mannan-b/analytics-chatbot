# 🔄 REGENERATE EMBEDDINGS - FIXED VERSION WITH .env LOADING
# Creates embedding_1 column using combined text for semantic richness

import asyncio
import os
from dotenv import load_dotenv
from supabase import create_client
from services.vector_service import CorrectedVectorService

# LOAD .env file
load_dotenv()

# Get credentials from .env
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate credentials
if not SUPABASE_URL:
    print("❌ ERROR: SUPABASE_URL not found in .env")
    print("Add this to your .env file:")
    print('SUPABASE_URL="https://your-project.supabase.co"')
    exit(1)

if not SUPABASE_KEY:
    print("❌ ERROR: SUPABASE_SERVICE_KEY not found in .env")
    print("Add this to your .env file:")
    print('SUPABASE_SERVICE_KEY="your-service-role-key"')
    exit(1)

if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY not found in .env")
    print("Add this to your .env file:")
    print('OPENAI_API_KEY="sk-your-key"')
    exit(1)

print("✅ Credentials loaded from .env")
print(f"   SUPABASE_URL: {SUPABASE_URL[:50]}...")
print(f"   SUPABASE_KEY: {SUPABASE_KEY[:20]}...")
print(f"   OPENAI_API_KEY: {OPENAI_API_KEY[:20]}...")
print()

async def regenerate_embeddings_v2():
    """Regenerate embeddings with proper text combination"""
    
    print("="*80)
    print("🔄 REGENERATING EMBEDDINGS (embedding_1) - VERSION 2")
    print("="*80)
    
    # Initialize clients
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    vector_service = CorrectedVectorService()
    
    # 1. Regenerate table_descriptions embeddings
    # Combine: table_name + " : " + description
    print("\n1️⃣  Processing table_descriptions...")
    tables_response = supabase.table("table_descriptions").select("*").execute()
    tables = tables_response.data
    
    print(f"   Found {len(tables)} tables")
    
    for i, table in enumerate(tables, 1):
        table_name = table['table_name']
        description = table.get('description', '')
        
        # COMBINE: table_name : description
        combined_text = f"{table_name} : {description}"
        
        if not combined_text.strip():
            print(f"   ⚠️  [{i}/{len(tables)}] {table_name} - Empty text, skipping")
            continue
        
        # Create new embedding from combined text
        embedding = await vector_service.create_embedding(combined_text)
        
        # Update embedding_1 column (NOT embedding)
        supabase.table("table_descriptions")\
            .update({"embedding_1": embedding})\
            .eq("table_name", table_name)\
            .execute()
        
        print(f"   ✅ [{i}/{len(tables)}] {table_name} - embedding_1 created")
    
    # 2. Regenerate column_metadata embeddings
    # Combine: table_name.column_name : description
    print("\n2️⃣  Processing column_metadata...")
    columns_response = supabase.table("column_metadata").select("*").execute()
    columns = columns_response.data
    
    print(f"   Found {len(columns)} columns")
    
    for i, column in enumerate(columns, 1):
        table_name = column['table_name']
        column_name = column['column_name']
        description = column.get('description', '')
        
        # COMBINE: table_name.column_name : description
        combined_text = f"{table_name}.{column_name} : {description}"
        
        if not combined_text.strip():
            print(f"   ⚠️  [{i}/{len(columns)}] {table_name}.{column_name} - Empty text, skipping")
            continue
        
        # Create new embedding from combined text
        embedding = await vector_service.create_embedding(combined_text)
        
        # Update embedding_1 column (NOT embedding)
        supabase.table("column_metadata")\
            .update({"embedding_1": embedding})\
            .eq("table_name", table_name)\
            .eq("column_name", column_name)\
            .execute()
        
        if i % 10 == 0 or i == len(columns):
            print(f"   ✅ [{i}/{len(columns)}] {table_name}.{column_name} - embedding_1 created")
    
    print(f"   ✅ All {len(columns)} columns processed")
    
    # 3. Regenerate common_prompt_sqls embeddings
    print("\n3️⃣  Processing common_prompt_sqls...")
    patterns_response = supabase.table("common_prompt_sqls").select("*").execute()
    patterns = patterns_response.data
    
    print(f"   Found {len(patterns)} SQL patterns")
    
    for i, pattern in enumerate(patterns, 1):
        pattern_id = pattern.get('id')
        prompt = pattern.get('prompt', '')
        
        if not prompt.strip():
            print(f"   ⚠️  [{i}/{len(patterns)}] ID {pattern_id} - Empty prompt, skipping")
            continue
        
        # Just use prompt as is (already semantically rich)
        combined_text = prompt
        
        # Create new embedding
        embedding = await vector_service.create_embedding(combined_text)
        
        # Update embedding_1 column (NOT embedding)
        supabase.table("common_prompt_sqls")\
            .update({"embedding_1": embedding})\
            .eq("id", pattern_id)\
            .execute()
        
        print(f"   ✅ [{i}/{len(patterns)}] {prompt[:50]}... - embedding_1 created")
    
    # 4. Regenerate business_context embeddings
    # Document pieces are already ~300 chars, just embed directly
    print("\n4️⃣  Processing business_context...")
    context_response = supabase.table("business_context").select("*").execute()
    contexts = context_response.data
    
    print(f"   Found {len(contexts)} business context pieces")
    
    for i, context in enumerate(contexts, 1):
        context_id = context.get('id')
        document_piece = context.get('document_piece', '')
        
        if not document_piece.strip():
            print(f"   ⚠️  [{i}/{len(contexts)}] ID {context_id} - Empty document, skipping")
            continue
        
        # Document piece is already ~300 chars, use as is
        combined_text = document_piece
        
        # Create new embedding
        embedding = await vector_service.create_embedding(combined_text)
        
        # Update embedding_1 column (NOT embedding)
        supabase.table("business_context")\
            .update({"embedding_1": embedding})\
            .eq("id", context_id)\
            .execute()
        
        if i % 5 == 0 or i == len(contexts):
            print(f"   ✅ [{i}/{len(contexts)}] ID {context_id} - embedding_1 created")
    
    print(f"   ✅ All {len(contexts)} business context pieces processed")
    
    print("\n" + "="*80)
    print("🎉 ALL embedding_1 COLUMNS GENERATED SUCCESSFULLY!")
    print("="*80)
    print("\n✅ New embedding_1 column created (old embedding column preserved)")
    print("✅ Text properly combined for semantic richness:")
    print("   - Tables: table_name : description")
    print("   - Columns: table_name.column_name : description")
    print("   - SQL Patterns: prompt (as is)")
    print("   - Business Context: document_piece (as is, ~300 chars)")
    print("\n📝 Next steps:")
    print("   1. cp vector_service_USE_EMBEDDING_1.py services/vector_service.py")
    print("   2. cp database_supabase_client_USE_EMBEDDING_1.py database_supabase_client.py")
    print("   3. uvicorn mainn:app --reload")
    print("   4. Test queries!")

if __name__ == "__main__":
    try:
        asyncio.run(regenerate_embeddings_v2())
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)