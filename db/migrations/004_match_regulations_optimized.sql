-- Optimize vector search used by the Regulation Explorer.
-- Fixes `canceling statement due to statement timeout` by:
-- 1) Adding an HNSW index for fast approximate nearest-neighbor search.
--    (ivfflat is limited to <=2000 dimensions; HNSW works with any dimension.)
-- 2) Rewriting `match_regulations` to use separate branches for NULL vs
--    non-NULL `filter_jurisdiction`, helping the query planner.

CREATE EXTENSION IF NOT EXISTS vector;

-- HNSW index for cosine distance on 3072-dim embeddings.
-- m=16 and ef_construction=64 are conservative defaults that work well
-- for tables with a few thousand rows.
CREATE INDEX IF NOT EXISTS regulation_embeddings_embedding_hnsw_idx
ON public.regulation_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Re-written function with simpler branches to help the planner.
CREATE OR REPLACE FUNCTION match_regulations(
  query_embedding vector(3072),
  match_count int,
  filter_jurisdiction int DEFAULT NULL
) RETURNS TABLE(id int, chunk_text text, similarity float, metadata jsonb)
LANGUAGE plpgsql AS $$
BEGIN
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
  ORDER BY e.embedding <=> query_embedding
  LIMIT match_count;
END; $$;
