 # ULTIMATE FIXED TEXT-TO-SQL SYSTEM WITH ALL FEATURES

import sys
import os
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import traceback
import numpy as np

# Import services
from database_supabase_client import CorrectedSupabaseClient as SupabaseClient
from services.vector_service import CorrectedVectorService as VectorService
from services.prompt_classifier import PromptClassifier
from services.llm_service import LLMService
from services.visualization_service import SmartVisualizationService
from services.query_history_service import QueryHistoryService
from services.next_query_predictor import NextQueryPredictor
from services.pdf_processor import PDFProcessor
from utils.config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Text-to-SQL System - ULTIMATE FIXED WITH FEATURES")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
logger.info("🔧 Initializing services...")
try:
    supabase_client = SupabaseClient()
    vector_service = VectorService()
    prompt_classifier = PromptClassifier()
    llm_service = LLMService()
    viz_service = SmartVisualizationService()
    query_history = QueryHistoryService(supabase_client.client)
    next_predictor = NextQueryPredictor(llm_service)
    pdf_processor = PDFProcessor()
    logger.info("✅ All services initialized successfully (including NEW features)")
except Exception as e:
    logger.error(f"❌ Service initialization failed: {e}")
    raise

# Models
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default"

@app.on_event("startup")
async def startup():
    logger.info("🚀 Text-to-SQL System Started (ULTIMATE FIXED VERSION WITH FEATURES)")
    try:
        tables = await supabase_client.get_all_table_descriptions()
        logger.info(f"✅ Connected to database - {len(tables)} tables found")
    except Exception as e:
        logger.error(f"⚠️  Database connection issue: {e}")

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process user query - COMPLETE with execution and visualization"""
    user_prompt = request.query.strip()
    logger.info(f"\n{'='*60}\n📝 Query: {user_prompt}\n{'='*60}")
    
    try:
        # Classify
        classification, confidence, reasoning = await prompt_classifier.classify_prompt(user_prompt)
        logger.info(f"📊 Classification: {classification} ({confidence:.2f})")
        
        if classification == "invalid_question":
            return JSONResponse({
                "success": False,
                "error": "Please rephrase your question",
                "classification": classification,
                "confidence": confidence
            })
        
        # Handle based on type
        if classification == "data_question":
            result = await handle_data_query(user_prompt)
        else:
            result = await handle_business_query(user_prompt)
        
        # NEW: Save to history (SYNC - no await)
        try:
            await query_history.save_query(
                user_id=request.user_id,
                query_text=user_prompt,
                sql_generated=result.get('sql_query'),
                result_count=result.get('row_count'),
                classification=classification,
                confidence=confidence
            )
        except Exception as e:
            logger.warning(f"⚠️  Could not save query history: {e}")
        
        # Build response
        response = {
            "success": result.get("error") is None,
            "classification": classification,
            "confidence": confidence,
            **result
        }
        
        return JSONResponse(response)
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Error: {error_msg}")
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": error_msg,
            "classification": "error",
            "confidence": 0.0
        })

async def handle_data_query(user_prompt: str) -> Dict[str, Any]:
    """Handle data query with FIXED vector search"""
    try:
        # Get metadata
        logger.info("📊 Fetching database metadata...")
        all_tables = await supabase_client.get_all_table_descriptions()
        all_columns = await supabase_client.get_all_column_metadata()
        all_patterns = await supabase_client.get_all_common_prompts()
        logger.info(f"✅ Found: {len(all_tables)} tables, {len(all_columns)} columns")
        
        if not all_tables:
            return {"error": "No tables in database", "tables_found": 0, "columns_found": 0, "sql_patterns": 0}
        
        if not all_columns:
            return {"error": "No columns in database", "tables_found": len(all_tables), "columns_found": 0}
        
        # Vector search for tables
        logger.info("🔍 Searching for relevant tables...")
        relevant_tables = await vector_service.search_similar_tables(
            user_prompt, all_tables, similarity_threshold=0.3, max_results=5
        )
        
        if not relevant_tables:
            logger.warning("⚠️  No similar tables found, using all")
            relevant_tables = all_tables[:5]
        
        table_names = [t['table_name'] for t in relevant_tables]
        logger.info(f"✅ Using {len(relevant_tables)} tables: {table_names}")
        
        # Get columns for those tables
        table_columns = await supabase_client.get_columns_for_tables(table_names)
        logger.info("🔍 Searching for relevant columns...")
        
        relevant_columns = await vector_service.search_similar_columns(
            user_prompt, table_columns, selected_table_names=table_names,
            similarity_threshold=0.3, max_results=15
        )
        
        if not relevant_columns:
            relevant_columns = table_columns[:15]
        
        logger.info(f"✅ Using {len(relevant_columns)} columns")
        
        # Get similar SQL patterns
        similar_patterns = await vector_service.search_similar_sql_patterns(
            user_prompt, all_patterns, similarity_threshold=0.5, max_results=3
        )
        
        # Generate SQL
        logger.info("🤖 Generating SQL...")
        sql_query = await generate_sql(user_prompt, relevant_tables, relevant_columns, similar_patterns)
        logger.info(f"✅ Generated SQL:\n{sql_query}")
        
        # Execute SQL
        logger.info("⚡ Executing query...")
        result = await supabase_client.execute_sql(sql_query)
        
        if result.get('success'):
            row_count = result.get('count', 0)
            data = result.get('data', [])
            logger.info(f"✅ Success! {row_count} rows returned")
            
            # Try to generate visualization
            chart = None
            try:
                if data and len(data) > 0:
                    logger.info("📊 Generating visualization...")
                    viz_result = await viz_service.create_smart_visualization(
                        data=data,
                        query_context=user_prompt,
                        chart_title=f"Results: {user_prompt}"
                    )
                    if viz_result.get('success'):
                        chart = viz_result
                        logger.info(f"✅ Visualization: {viz_result.get('chart_type')}")
            except Exception as e:
                logger.warning(f"⚠️  Visualization failed: {e}")
            
            return {
                "sql_query": sql_query,
                "data": data,
                "row_count": row_count,
                "tables_found": len(relevant_tables),
                "columns_found": len(relevant_columns),
                "sql_patterns": len(similar_patterns),
                "visualization": chart,
                "tables_used": table_names
            }
        else:
            logger.error(f"❌ SQL execution failed: {result.get('error')}")
            return {
                "error": result.get('error', 'SQL execution failed'),
                "sql_query": sql_query,
                "tables_found": len(relevant_tables),
                "columns_found": len(relevant_columns),
                "sql_patterns": len(similar_patterns)
            }
    
    except Exception as e:
        logger.error(f"❌ Data query failed: {e}")
        traceback.print_exc()
        return {"error": str(e), "tables_found": 0, "columns_found": 0, "sql_patterns": 0}

async def handle_business_query(user_prompt: str) -> Dict[str, Any]:
    """Handle business/context query"""
    try:
        logger.info("💼 Processing business question...")
        return {
            "answer": f"Business question: '{user_prompt}' - context feature available",
            "type": "business_question"
        }
    except Exception as e:
        logger.error(f"❌ Business query failed: {e}")
        return {"error": str(e)}

async def generate_sql(prompt: str, tables: List[Dict], columns: List[Dict], patterns: List[Dict]) -> str:
    """Generate SQL with improved prompt"""
    tables_str = "\n".join([
        f"- {t['table_name']}: {t.get('table_description') or t.get('description', 'No description')}"
        for t in tables
    ])
    
    columns_str = "\n".join([
        f"- {c['table_name']}.{c['column_name']}: {c.get('column_description') or c.get('description', 'No description')}"
        for c in columns[:20]
    ])
    
    patterns_str = ""
    if patterns:
        patterns_str = "\n\nEXAMPLE QUERIES:\n" + "\n\n".join([
            f"Q: {p.get('prompt', '')}\nSQL: {p.get('sql_query', '')}"
            for p in patterns[:3]
        ])
    
    llm_prompt = f"""You are a PostgreSQL expert. Generate a query for: "{prompt}"

AVAILABLE TABLES:
{tables_str}

AVAILABLE COLUMNS:
{columns_str}

{patterns_str}

REQUIREMENTS:
1. Use ONLY tables and columns listed above
2. For multi-table queries, use proper JOINs
3. For aggregations: include GROUP BY with meaningful aliases
4. For "top N": use ORDER BY + LIMIT
5. Always add LIMIT 100 for safety
6. Return ONLY the SQL query

SQL Query:"""
    
    sql = await llm_service.get_completion(llm_prompt, max_tokens=1000, temperature=0.1)
    sql = sql.strip().replace("```sql", "").replace("```", "").strip()
    
    # Ensure LIMIT
    if "LIMIT" not in sql.upper() and "SELECT" in sql.upper():
        sql = sql.rstrip(";") + "\nLIMIT 100;"
    
    return sql

# NEW ENDPOINTS

@app.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 10):
    """Get query history - NEW FEATURE"""
    try:
        queries = query_history.get_recent_queries(user_id, limit)
        return {"success": True, "history": queries}
    except Exception as e:
        logger.error(f"❌ History fetch failed: {e}")
        return {"success": False, "error": str(e)}

@app.post("/suggest-next")
async def suggest_next(request: dict):
    """Suggest next queries - NEW FEATURE"""
    try:
        suggestions = await next_predictor.predict_next_queries(
            request.get('user_query', ''),
            request.get('sql_query', ''),
            request.get('result_data', [])
        )
        return {"success": True, "suggestions": suggestions}
    except Exception as e:
        logger.error(f"❌ Next query prediction failed: {e}")
        return {"success": False, "error": str(e)}

@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF for business context - NEW FEATURE"""
    try:
        pdf_bytes = await file.read()
        result = await pdf_processor.process_pdf_bytes(pdf_bytes, file.filename)
        
        if not result['success']:
            return {"success": False, "error": result['error']}
        
        stored = 0
        for chunk in result['chunks']:
            embedding = await vector_service.create_embedding(chunk)
            # NO AWAIT - Supabase client is synchronous
            supabase_client.client.table('business_context').insert({
                'document_piece': chunk,
                'embedding_1': embedding,
                'source_document': file.filename
            }).execute()
            stored += 1
        
        return {
            "success": True,
            "filename": file.filename,
            "pages": result['total_pages'],
            "chunks_stored": stored
        }
    except Exception as e:
        logger.error(f"❌ PDF upload failed: {e}")
        return {"success": False, "error": str(e)}

@app.get("/")
async def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Text-to-SQL API (ULTIMATE FIXED) - /docs for API"}

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "3.0.0-ultimate-fixed"}

@app.get("/debug/db")
async def debug_db():
    """Debug database - shows counts"""
    try:
        tables = await supabase_client.get_all_table_descriptions()
        columns = await supabase_client.get_all_column_metadata()
        patterns = await supabase_client.get_all_common_prompts()
        return {
            "status": "ok",
            "tables_count": len(tables),
            "columns_count": len(columns),
            "patterns_count": len(patterns),
            "sample_table": tables[0] if tables else None,
            "sample_column": columns[0] if columns else None
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Static files
try:
    if os.path.exists("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Could not mount static: {e}")

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting ULTIMATE FIXED version with ALL FEATURES...")
    uvicorn.run(app, host="0.0.0.0", port=8000)