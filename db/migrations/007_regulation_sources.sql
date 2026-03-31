-- Migration 007: regulation_sources table + app_settings table
--
-- regulation_sources separates "what to scrape" from "what was scraped" (regulations).
-- app_settings stores persistent feature flags like use_db_source_registry.

-- Persistent key-value settings (feature flags, app config)
CREATE TABLE IF NOT EXISTS app_settings (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed the toggle that controls CSV vs DB source registry.
INSERT INTO app_settings (key, value)
VALUES ('use_db_source_registry', 'false')
ON CONFLICT (key) DO NOTHING;

-- Source registry: one row per regulation source URL.
CREATE TABLE IF NOT EXISTS regulation_sources (
  id              SERIAL PRIMARY KEY,
  jurisdiction_id INT  NOT NULL REFERENCES jurisdictions(id) ON DELETE CASCADE,
  source_name     TEXT NOT NULL,
  url             TEXT NOT NULL,
  domain          TEXT NOT NULL DEFAULT 'housing',
  category        TEXT NOT NULL DEFAULT 'General',
  state_code      CHAR(2) NULL,
  is_active       BOOL NOT NULL DEFAULT TRUE,
  last_scraped_at TIMESTAMPTZ NULL,
  last_error      TEXT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS regulation_sources_url_uidx
  ON regulation_sources(url);
CREATE INDEX IF NOT EXISTS regulation_sources_jurisdiction_id_idx
  ON regulation_sources(jurisdiction_id);
CREATE INDEX IF NOT EXISTS regulation_sources_is_active_idx
  ON regulation_sources(is_active);



-- Grant table access
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.regulation_sources TO anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.app_settings TO anon, authenticated;

-- Grant sequence access (needed for inserts with SERIAL columns)
GRANT USAGE, SELECT ON SEQUENCE public.regulation_sources_id_seq TO anon, authenticated;

-- RLS policies (permissive for dev)
ALTER TABLE public.regulation_sources ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "anon_rw_regulation_sources" ON public.regulation_sources;
CREATE POLICY "anon_rw_regulation_sources"
ON public.regulation_sources FOR ALL TO anon
USING (true) WITH CHECK (true);

ALTER TABLE public.app_settings ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "anon_rw_app_settings" ON public.app_settings;
CREATE POLICY "anon_rw_app_settings"
ON public.app_settings FOR ALL TO anon
USING (true) WITH CHECK (true);