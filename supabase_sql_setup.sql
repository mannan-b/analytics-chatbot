-- SUPABASE SQL FUNCTION SETUP
-- Run this in your Supabase SQL Editor to enable raw SQL execution

-- 1. Create a function to execute raw SQL queries
CREATE OR REPLACE FUNCTION execute_sql(query text)
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    result json;
BEGIN
    -- Execute the query and return results as JSON
    EXECUTE format('SELECT json_agg(t) FROM (%s) t', query) INTO result;
    RETURN COALESCE(result, '[]'::json);
EXCEPTION
    WHEN OTHERS THEN
        -- Return error as JSON
        RETURN json_build_object(
            'error', SQLERRM,
            'sqlstate', SQLSTATE
        );
END;
$$;

-- 2. Grant execute permission to authenticated users
GRANT EXECUTE ON FUNCTION execute_sql(text) TO authenticated;
GRANT EXECUTE ON FUNCTION execute_sql(text) TO anon;

-- 3. Test the function
SELECT execute_sql('SELECT * FROM sales LIMIT 5');

-- NOTE: This function allows executing any SELECT query
-- Make sure to add additional security if needed in production
-- Example: Add WHERE clause to limit by user_id, restrict to SELECT only, etc.

-- Optional: Create a safer version that only allows SELECT
CREATE OR REPLACE FUNCTION execute_select_only(query text)
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    result json;
BEGIN
    -- Check if query starts with SELECT
    IF LOWER(TRIM(query)) NOT LIKE 'select%' THEN
        RAISE EXCEPTION 'Only SELECT queries are allowed';
    END IF;
    
    -- Execute the query and return results as JSON
    EXECUTE format('SELECT json_agg(t) FROM (%s) t', query) INTO result;
    RETURN COALESCE(result, '[]'::json);
EXCEPTION
    WHEN OTHERS THEN
        RETURN json_build_object(
            'error', SQLERRM,
            'sqlstate', SQLSTATE
        );
END;
$$;

GRANT EXECUTE ON FUNCTION execute_select_only(text) TO authenticated;
GRANT EXECUTE ON FUNCTION execute_select_only(text) TO anon;
