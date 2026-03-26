-- Migration 006: Hybrid retrieval support
--
-- Adds:
-- 1. tsvector column + GIN index on regulation_embeddings for full-text search
-- 2. Trigger to auto-populate tsvector on insert/update
-- 3. match_regulations_lexical() RPC for keyword search
-- 4. match_regulations_v3() RPC with explicit jurisdiction ID array
--
-- Run this in the Supabase SQL editor after deploying the code changes.

-- =========================================================================
-- 1. Full-text search column + index
-- =========================================================================

ALTER TABLE regulation_embeddings
  ADD COLUMN IF NOT EXISTS chunk_tsv tsvector;

UPDATE regulation_embeddings
  SET chunk_tsv = to_tsvector('english', coalesce(chunk_text, ''))
  WHERE chunk_tsv IS NULL;

CREATE INDEX IF NOT EXISTS regulation_embeddings_chunk_tsv_gin_idx
  ON regulation_embeddings USING gin(chunk_tsv);

-- Auto-populate on insert/update
CREATE OR REPLACE FUNCTION regulation_embeddings_tsv_trigger()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  NEW.chunk_tsv := to_tsvector('english', coalesce(NEW.chunk_text, ''));
  RETURN NEW;
END; $$;

DROP TRIGGER IF EXISTS trg_regulation_embeddings_tsv ON regulation_embeddings;
CREATE TRIGGER trg_regulation_embeddings_tsv
  BEFORE INSERT OR UPDATE OF chunk_text ON regulation_embeddings
  FOR EACH ROW EXECUTE FUNCTION regulation_embeddings_tsv_trigger();

-- =========================================================================
-- 2. Lexical search RPC
-- =========================================================================

CREATE OR REPLACE FUNCTION match_regulations_lexical(
  search_query text,
  match_count int DEFAULT 10,
  filter_jurisdictions int[] DEFAULT NULL,
  category_filter text DEFAULT NULL
) RETURNS TABLE(id int, chunk_text text, rank float, metadata jsonb)
LANGUAGE plpgsql AS $$
DECLARE
  tsq tsquery;
BEGIN
  PERFORM set_config('statement_timeout', '30000', true);

  tsq := websearch_to_tsquery('english', search_query);

  RETURN QUERY
  SELECT
    e.id,
    e.chunk_text,
    ts_rank_cd(e.chunk_tsv, tsq)::float AS rank,
    row_to_json(r)::jsonb AS metadata
  FROM regulation_embeddings e
  JOIN regulations r ON r.id = e.regulation_id
  WHERE r.is_current = true
    AND e.chunk_tsv @@ tsq
    AND (filter_jurisdictions IS NULL
         OR r.jurisdiction_id = ANY(filter_jurisdictions))
    AND (category_filter IS NULL OR r.category = category_filter)
  ORDER BY ts_rank_cd(e.chunk_tsv, tsq) DESC
  LIMIT match_count;
END; $$;

-- =========================================================================
-- 3. Vector search v3 with explicit jurisdiction array
-- =========================================================================

CREATE OR REPLACE FUNCTION match_regulations_v3(
  query_embedding vector(3072),
  match_count int,
  filter_jurisdictions int[] DEFAULT NULL,
  category_filter text DEFAULT NULL
) RETURNS TABLE(id int, chunk_text text, similarity float, metadata jsonb)
LANGUAGE plpgsql AS $$
BEGIN
  PERFORM set_config('statement_timeout', '60000', true);

  IF filter_jurisdictions IS NULL THEN
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
    RETURN;
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
    AND r.jurisdiction_id = ANY(filter_jurisdictions)
    AND (category_filter IS NULL OR r.category = category_filter)
  ORDER BY e.embedding <=> query_embedding
  LIMIT match_count;
END; $$;
