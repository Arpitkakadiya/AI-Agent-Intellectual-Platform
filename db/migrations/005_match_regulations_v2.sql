-- match_regulations_v2:
-- A safer/cheaper vector search RPC for Regulation Explorer when:
-- - embeddings are high-dimension (vector(3072))
-- - pgvector index types may be restricted in Supabase
--
-- Improvements:
-- - Allows passing `category_filter` so we reduce the number of rows scanned/sorted.
-- - Raises `statement_timeout` locally for this function call to avoid cancellations.
-- - Uses separate branches for NULL vs non-NULL `filter_jurisdiction` to keep plans simple.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE OR REPLACE FUNCTION match_regulations_v2(
  query_embedding vector(3072),
  match_count int,
  filter_jurisdiction int DEFAULT NULL,
  category_filter text DEFAULT NULL
) RETURNS TABLE(id int, chunk_text text, similarity float, metadata jsonb)
LANGUAGE plpgsql AS $$
BEGIN
  -- Allow longer scans when Supabase's global statement_timeout is low.
  -- This is local to this function call.
  PERFORM set_config('statement_timeout', '60000', true); -- 60s

  IF filter_jurisdiction IS NULL THEN
    RETURN QUERY
    SELECT
      e.id,
      e.chunk_text,
      1 - (e.embedding <=> query_embedding) AS similarity,
      row_to_json(r)::jsonb AS metadata
    FROM regulation_embeddings e
    JOIN regulations r ON r.id = e.regulation_id
    WHERE r.is_current = true
      AND (category_filter IS NULL OR r.category = category_filter)
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
  END IF;

  RETURN QUERY
  SELECT
    e.id,
    e.chunk_text,
    1 - (e.embedding <=> query_embedding) AS similarity,
    row_to_json(r)::jsonb AS metadata
  FROM regulation_embeddings e
  JOIN regulations r ON r.id = e.regulation_id
  WHERE r.is_current = true
    AND (
      r.jurisdiction_id = filter_jurisdiction
      OR r.jurisdiction_id IN (
        SELECT j.id FROM jurisdictions j WHERE j.type = 'federal'
      )
      OR EXISTS (
        SELECT 1
        FROM jurisdictions sel
        WHERE sel.id = filter_jurisdiction
          AND sel.type = 'city'
          AND r.jurisdiction_id = sel.parent_id
      )
    )
    AND (category_filter IS NULL OR r.category = category_filter)
  ORDER BY e.embedding <=> query_embedding
  LIMIT match_count;
END; $$;

