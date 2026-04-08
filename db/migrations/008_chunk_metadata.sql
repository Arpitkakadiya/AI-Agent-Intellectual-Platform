-- Migration 008: Add chunk_metadata JSONB column to regulation_embeddings
--
-- Stores legal-aware chunking metadata (section title, citation hints,
-- effective date flags, chunk index) alongside each embedding row.
-- This is optional — the application falls back gracefully if the
-- column doesn't exist.
--
-- Run this in the Supabase SQL editor.

ALTER TABLE regulation_embeddings
  ADD COLUMN IF NOT EXISTS chunk_metadata JSONB DEFAULT NULL;

COMMENT ON COLUMN regulation_embeddings.chunk_metadata IS
  'Legal-aware chunking metadata: section_title, chunk_index, total_chunks, citation_hint, has_definitions, has_effective_date';
