# 🚀 ULTIMATE TEXT-TO-SQL SYSTEM WITH NEW FEATURES
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
app = FastAPI(title="Text-to-SQL System WITH NEW FEATURES")

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
    logger.info("✅ All services initialized (including NEW features)")
except Exception as e:
    logger.error(f"❌ Service initialization failed: {e}")
    raise

# Models
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default"

@app.on_event("startup")
async def startup():
    logger.info("🚀 Text-to-SQL System Started WITH NEW FEATURES")
    try:
        tables = await supabase_client.get_all_table_descriptions()
        logger.info(f"✅ Connected - {len(tables)} tables found")
    except Exception as e:
        logger.error(f"⚠️  Database issue: {e}")

@app.post("/query")
async def process_query(request: QueryRequest):
    user_prompt = request.query.strip()
    logger.info(f"\n{'='*60}\n📝 Query: {user_prompt}\n{'='*60}")

    try:
        classification, confidence, reasoning = await prompt_classifier.classify_prompt(user_prompt)
        logger.info(f"📊 Classification: {classification} ({confidence:.2f})")

        if classification == "invalid_question":
            return JSONResponse({
                "success": False,
                "error": "Please rephrase",
                "classification": classification,
                "confidence": confidence
            })

        if classification == "data_question":
            result = await handle_data_query(user_prompt)
        else:
            result = await handle_business_query(user_prompt)

        # NEW: Save to history
        try:
            await query_history.save_query(
                user_id=request.user_id,
                query_text=user_prompt,
                sql_generated=result.get('sql_query'),
                result_count=result.get('row_count'),
                classification=classification,
                confidence=confidence
            )
        except:
            pass

        response = {
            "success": result.get("error") is None,
            "classification": classification,
            "confidence": confidence,
            **result
        }
        return JSONResponse(response)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "classification": "error",
            "confidence": 0.0
        })

async def handle_data_query(user_prompt: str) -> Dict[str, Any]:
    try:
        logger.info("📊 Fetching metadata...")
        all_tables = await supabase_client.get_all_table_descriptions()
        all_columns = await supabase_client.get_all_column_metadata()
        all_patterns = await supabase_client.get_all_common_prompts()

        if not all_tables or not all_columns:
            return {"error": "No data", "tables_found": 0, "columns_found": 0}

        relevant_tables = await vector_service.search_similar_tables(
            user_prompt, all_tables, similarity_threshold=0.3, max_results=5
        )

        if not relevant_tables:
            relevant_tables = all_tables[:5]

        table_names = [t['table_name'] for t in relevant_tables]
        table_columns = await supabase_client.get_columns_for_tables(table_names)

        relevant_columns = await vector_service.search_similar_columns(
            user_prompt, table_columns, selected_table_names=table_names,
            similarity_threshold=0.3, max_results=15
        )

        if not relevant_columns:
            relevant_columns = table_columns[:15]

        similar_patterns = await vector_service.search_similar_sql_patterns(
            user_prompt, all_patterns, similarity_threshold=0.5, max_results=3
        )

        sql_query = await generate_sql(user_prompt, relevant_tables, relevant_columns, similar_patterns)
        logger.info(f"✅ SQL:\n{sql_query}")

        result = await supabase_client.execute_sql(sql_query)

        if result.get('success'):
            data = result.get('data', [])
            chart = None

            try:
                if data:
                    viz_result = await viz_service.create_smart_visualization(
                        data=data, query_context=user_prompt,
                        chart_title=f"Results: {user_prompt}"
                    )
                    if viz_result.get('success'):
                        chart = viz_result
            except:
                pass

            return {
                "sql_query": sql_query,
                "data": data,
                "row_count": len(data),
                "tables_found": len(relevant_tables),
                "columns_found": len(relevant_columns),
                "sql_patterns": len(similar_patterns),
                "visualization": chart,
                "tables_used": table_names
            }
        else:
            return {
                "error": result.get('error'),
                "sql_query": sql_query,
                "tables_found": len(relevant_tables),
                "columns_found": len(relevant_columns)
            }
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        return {"error": str(e)}

async def handle_business_query(user_prompt: str) -> Dict[str, Any]:
    return {"answer": f"Business question: '{user_prompt}'"}

async def generate_sql(prompt: str, tables: List[Dict], columns: List[Dict], patterns: List[Dict]) -> str:
    tables_str = "\n".join([
        f"- {t['table_name']}: {t.get('table_description') or t.get('description', 'N/A')}"
        for t in tables
    ])
    columns_str = "\n".join([
        f"- {c['table_name']}.{c['column_name']}: {c.get('column_description') or c.get('description', 'N/A')}"
        for c in columns[:20]
    ])
    patterns_str = ""
    if patterns:
        patterns_str = "\n\nEXAMPLES:\n" + "\n".join([
            f"Q: {p.get('prompt', '')}\nSQL: {p.get('sql_query', '')}"
            for p in patterns[:2]
        ])

    llm_prompt = f"""Generate PostgreSQL for: "{prompt}"

TABLES:
{tables_str}

COLUMNS:
{columns_str}
{patterns_str}

RULES:
1. Use ONLY listed tables/columns
2. Use JOINs for multi-table
3. Add GROUP BY for aggregations
4. Add LIMIT 100
5. Return ONLY SQL

SQL:"""

    sql = await llm_service.get_completion(llm_prompt, max_tokens=1000, temperature=0.1)
    sql = sql.strip().replace("```sql", "").replace("```", "").strip()

    if "LIMIT" not in sql.upper():
        sql = sql.rstrip(";") + "\nLIMIT 100;"

    return sql

# NEW ENDPOINTS
@app.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 10):
    queries = await query_history.get_recent_queries(user_id, limit)
    return {"success": True, "history": queries}

@app.post("/suggest-next")
async def suggest_next(request: dict):
    suggestions = await next_predictor.predict_next_queries(
        request.get('user_query', ''),
        request.get('sql_query', ''),
        request.get('result_data', [])
    )
    return {"success": True, "suggestions": suggestions}

@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        result = await pdf_processor.process_pdf_bytes(pdf_bytes, file.filename)

        if not result['success']:
            return {"success": False, "error": result['error']}

        stored = 0
        for chunk in result['chunks']:
            embedding = await vector_service.create_embedding(chunk)
            supabase_client.client.table('business_context').insert({
                'content': chunk,
                'embedding_1': embedding
            }).execute()
            stored += 1

        return {
            "success": True,
            "filename": file.filename,
            "pages": result['total_pages'],
            "chunks_stored": stored
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/")
async def root():
    if os.path.exists("indexx.html"):
        return FileResponse("indexx.html")
    return {"message": "Text-to-SQL API"}

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "3.0.0"}

try:
    if os.path.exists("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
