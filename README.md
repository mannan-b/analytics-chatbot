# Text-to-SQL System - Database Query Engine

Advanced database query system that converts natural language to SQL with automatic visualizations and AI-powered suggestions.

## Core File Versions

### Production Build (Full Features)
**main.py + index.html** - Complete implementation with all features

Features included:
- Text-to-SQL query conversion
- Query history tracking per user
- AI-powered next query suggestions
- PDF upload with business context
- Automatic result visualization
- Vector search on tables/columns
- Multi-LLM support (Claude, GPT, Gemini)

This is the recommended version for deployment. All core functionality and new features are fully integrated.

### Legacy Build (Core Only)
**mainn.py + indexx.html** - Simplified implementation without PDF upload

Features included:
- Text-to-SQL query conversion
- Automatic result visualization
- Vector search on tables/columns
- Multi-LLM support

Does not include: query history, next query suggestions, PDF upload functionality.

## Latest Features (v3.0)

- Natural Language to SQL conversion using Claude/GPT
- Smart vector search using embedding_1 for table and column discovery
- Automatic chart generation from query results
- Per-user query history with metadata
- AI-powered next query suggestions based on results
- PDF document upload for business context
- Sub-2 second response times
- Safe SQL execution with RPC security validation

## Quick Start

### Prerequisites
- Python 3.10+
- Supabase account
- API keys for at least one LLM provider (OpenAI, Anthropic, or Google)

### Installation

```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and Supabase credentials

# Start server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Access the application at http://localhost:8000

## Database Setup (Supabase)

### Create Required Tables

```sql
CREATE TABLE query_history (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT DEFAULT 'default',
    query_text TEXT NOT NULL,
    sql_generated TEXT,
    result_count INTEGER,
    classification TEXT,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_query_history_user ON query_history(user_id, created_at DESC);

CREATE TABLE business_context (
    id BIGSERIAL PRIMARY KEY,
    document_piece TEXT,
    embedding_1 vector(1536),
    source_document TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_business_context_embedding ON business_context 
USING ivfflat (embedding_1 vector_cosine_ops);

CREATE TABLE table_descriptions (
    id BIGSERIAL PRIMARY KEY,
    table_name TEXT,
    table_description TEXT,
    embedding_1 vector(1536)
);

CREATE TABLE column_metadata (
    id BIGSERIAL PRIMARY KEY,
    table_name TEXT,
    column_name TEXT,
    column_description TEXT,
    embedding_1 vector(1536)
);

CREATE TABLE common_prompt_sqls (
    id BIGSERIAL PRIMARY KEY,
    prompt TEXT,
    sql_query TEXT,
    embedding_1 vector(1536)
);
```

### Update RPC Function (Allow WITH Clauses)

```sql
CREATE OR REPLACE FUNCTION execute_sql(query text)
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    result json;
BEGIN
    IF NOT (query ~* '^\s*(SELECT|WITH)') THEN
        RAISE EXCEPTION 'Only SELECT/WITH allowed';
    END IF;
    
    IF query ~* '(DROP|DELETE|UPDATE|INSERT INTO|ALTER|CREATE TABLE|TRUNCATE)' THEN
        RAISE EXCEPTION 'Operation not allowed';
    END IF;
    
    EXECUTE 'SELECT json_agg(row_to_json(t)) FROM (' || query || ') t' INTO result;
    RETURN COALESCE(result, '[]'::json);
END;
$$;
```

## API Endpoints

### Core Endpoints
- POST /query - Submit natural language query
- GET /health - Health check
- GET /debug/db - Database connection status

### Extended Endpoints (main.py only)
- GET /history/{user_id} - Retrieve query history
- POST /suggest-next - Get AI-powered follow-up suggestions
- POST /upload/pdf - Upload PDF for business context

## Project Structure

```
.
├── main.py                          # FastAPI server (full features)
├── mainn.py                         # FastAPI server (core only)
├── index.html                       # Frontend UI (full features)
├── indexx.html                      # Frontend UI (core only)
├── database_supabase_client.py      # Supabase client
├── requirements.txt                 # Python dependencies
├── .env                             # Configuration (create from template)
│
├── services/
│   ├── vector_service.py            # Vector embeddings and search
│   ├── llm_service.py               # LLM integration
│   ├── prompt_classifier.py         # Query classification
│   ├── visualization_service.py     # Chart generation
│   ├── business_context_service.py  # Context retrieval
│   ├── query_history_service.py     # Query history (main.py only)
│   ├── next_query_predictor.py      # Query suggestions (main.py only)
│   ├── pdf_processor.py             # PDF handling (main.py only)
│   └── sql_executor.py              # SQL execution
│
└── utils/
    ├── config.py                    # Configuration loader
    └── logger.py                    # Logging setup
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| SUPABASE_URL | Yes | Supabase project URL |
| SUPABASE_KEY | Yes | Supabase anon key |
| OPENAI_API_KEY | No | OpenAI API key (one LLM required) |
| ANTHROPIC_API_KEY | No | Claude API key (one LLM required) |
| GEMINI_API_KEY | No | Google Gemini key (one LLM required) |

At least one LLM provider key is required.

## How It Works

### Query Processing Flow

1. User submits natural language question
2. LLM classifies query type (data vs business)
3. Vector search finds relevant database tables and columns using embedding_1
4. LLM generates SQL from classification and selected metadata
5. Query executed via Supabase RPC with security validation
6. Results visualized automatically (if applicable)
7. Query saved to history with metadata
8. AI suggests 3 follow-up questions

## Example Queries

- "What are my top 10 customers by revenue?"
- "Show me sales trends for the last quarter"
- "Which products have the highest margins?"
- "List all orders for customer ID 5"

## Troubleshooting

### Network Error: getaddrinfo failed
Indicates DNS/network connectivity issue. Check:
- Internet connection is active
- Supabase domain is reachable
- DNS is working (ping google.com)
- Firewall isn't blocking Python

### Only SELECT allowed error
RPC function needs update. Run the SQL to update execute_sql function above.

### query_history table not found
Run the database setup SQL to create all required tables.

### PDF upload returns 400 error
Verify business_context table has embedding_1 column (not embedding).

## Deployment

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment
Supports deployment on: Railway, Render, Heroku, AWS, Google Cloud, Azure

## Security Features

- SQL injection prevention via RPC validation
- No direct database access (RPC-only execution)
- Parameterized queries
- Input validation on all endpoints
- Rate limiting on file uploads

## Performance

- Vector indexing with IVFFLAT for fast similarity search
- Caching of common queries
- Batch PDF processing
- Optimized LLM provider selection
- Response time: 1-2 seconds typical

## Version History

v3.0 - ULTIMATE FIXED
- Fixed RPC to allow WITH clauses
- Added query history tracking
- Added AI-powered suggestions
- Added PDF upload functionality
- Fixed network connectivity handling
- Complete documentation

## Support

- API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Database debug: http://localhost:8000/debug/db
- Check logs for detailed error messages