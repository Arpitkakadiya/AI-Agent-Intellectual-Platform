# Local Development (Setup + Feature Guide)

This guide explains how to run the Housing Regulation Compliance Agent locally, seed the database, and use the main features.

## 1) Prerequisites

- Python 3.11 (or at least Python 3.9+ for running scripts)
- A Supabase project (Postgres + pgvector)
- A Supabase database schema loaded (see step 3 below)
- Windows: `cmd.exe` available

## 2) Install dependencies

Open a terminal in the project folder (`c:\Arpit\Projects\A Agent\AI Agent For Real Estate`) and run:

```bat
py -3.9 -m pip install -r requirements.txt
```

## 3) Configure Supabase (schema)

In the Supabase Dashboard:

1. Go to **SQL Editor**
2. Run the migration files **in order**:
   - `db/migrations/001_initial_schema.sql`
   - `db/migrations/002_gemini_embedding_3072.sql`
   - `db/migrations/003_match_regulations_include_federal.sql`
   - `db/migrations/004_match_regulations_optimized.sql`
   - `db/migrations/005_match_regulations_v2.sql`
   - `db/migrations/006_hybrid_retrieval.sql`
   - `db/migrations/007_regulation_sources.sql` — adds `regulation_sources` + `app_settings` tables
   - `db/migrations/008_chunk_metadata.sql`
   - `db/migrations/009_email_subscriptions_rls.sql` — RLS for `email_subscriptions`
   - `db/migrations/010_pet_insurance_rls.sql` — RLS for `pet_policies` + `insurance_requirements`
3. Ensure the `vector` extension is enabled (it is created by the migration script).

## 4) Create your environment file

This project reads environment variables from:

- `.env` (project root)

Copy the template:

```bat
copy .env.example .env
```

Edit `.env` and set:

- `SUPABASE_URL`
- `SUPABASE_KEY`
- Optional LLM keys (priority order — first key found is used):
  - `ANTHROPIC_API_KEY` (primary)
  - `OPENAI_API_KEY` (fallback)
  - `GOOGLE_API_KEY` (Google Gemini fallback)
- Optional SMTP keys (for email alerts):
  - `SMTP_EMAIL`, `SMTP_PASSWORD`, `SMTP_HOST`, `SMTP_PORT`

### Which `SUPABASE_KEY` should you use?

For local development, you have two common choices:

1. **Easiest (recommended for seeding):** use the Supabase **service_role** key in `.env` while running the seed/index scripts.
2. **Keep `anon` (requires SQL policies):** keep `SUPABASE_KEY` as the **anon/public key** and grant the required table permissions + RLS policies (see step 6 for a dev-friendly SQL snippet).

If you use option 1, you can switch back to `anon` later for app runtime if you prefer.

## 5) Seed jurisdictions + regulations + build vector index

Run these scripts in order:

```bat
py -3.9 scripts\seed_jurisdictions.py
py -3.9 scripts\seed_db.py
py -3.9 scripts\index_regulations.py
```

Expected behavior:
- `seed_jurisdictions.py` inserts federal/state/city rows in `jurisdictions`
- `seed_db.py` loads `data/seeds/sources.csv` into `regulations` (with versioning)
- `index_regulations.py` populates vector embeddings into `regulation_embeddings`

If you get permission errors while seeding, use service_role for scripts or apply the permission/policy SQL in step 6.

## 6) (If needed) Dev SQL: allow `anon` to seed/index

If your `.env` uses `SUPABASE_KEY` = `anon` and your seed scripts fail with permission/RLS errors, run this SQL in Supabase (SQL Editor). This is intentionally permissive for local/dev.

```sql
GRANT USAGE ON SCHEMA public TO anon, authenticated;

-- Used by core/rag/vector_store.py RPC
GRANT EXECUTE ON FUNCTION public.match_regulations(vector, integer, integer)
TO anon, authenticated;

-- Seed tables
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.jurisdictions TO anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.regulations TO anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.regulation_embeddings TO anon, authenticated;

-- Update log + email alerts
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.regulation_updates TO anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.email_subscriptions TO anon, authenticated;

-- Source registry + app settings (migration 007)
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.regulation_sources TO anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.app_settings TO anon, authenticated;

-- Structured policy tables (pet_policies + insurance_requirements from 001, RLS from 010)
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.pet_policies TO anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.insurance_requirements TO anon, authenticated;
GRANT USAGE, SELECT ON SEQUENCE public.pet_policies_id_seq TO anon, authenticated;
GRANT USAGE, SELECT ON SEQUENCE public.insurance_requirements_id_seq TO anon, authenticated;

-- Policy helpers (if RLS is enabled). These are permissive for dev.
DROP POLICY IF EXISTS "anon_rw_jurisdictions" ON public.jurisdictions;
CREATE POLICY "anon_rw_jurisdictions"
ON public.jurisdictions
FOR ALL
TO anon
USING (true)
WITH CHECK (true);

DROP POLICY IF EXISTS "anon_rw_regulations" ON public.regulations;
CREATE POLICY "anon_rw_regulations"
ON public.regulations
FOR ALL
TO anon
USING (true)
WITH CHECK (true);

DROP POLICY IF EXISTS "anon_rw_regulation_embeddings" ON public.regulation_embeddings;
CREATE POLICY "anon_rw_regulation_embeddings"
ON public.regulation_embeddings
FOR ALL
TO anon
USING (true)
WITH CHECK (true);

DROP POLICY IF EXISTS "anon_rw_regulation_updates" ON public.regulation_updates;
CREATE POLICY "anon_rw_regulation_updates"
ON public.regulation_updates
FOR ALL
TO anon
USING (true)
WITH CHECK (true);

DROP POLICY IF EXISTS "anon_rw_email_subscriptions" ON public.email_subscriptions;
CREATE POLICY "anon_rw_email_subscriptions"
ON public.email_subscriptions
FOR ALL
TO anon
USING (true)
WITH CHECK (true);

DROP POLICY IF EXISTS "anon_rw_regulation_sources" ON public.regulation_sources;
CREATE POLICY "anon_rw_regulation_sources"
ON public.regulation_sources
FOR ALL
TO anon
USING (true)
WITH CHECK (true);

DROP POLICY IF EXISTS "anon_rw_app_settings" ON public.app_settings;
CREATE POLICY "anon_rw_app_settings"
ON public.app_settings
FOR ALL
TO anon
USING (true)
WITH CHECK (true);

ALTER TABLE public.pet_policies ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "anon_rw_pet_policies" ON public.pet_policies;
CREATE POLICY "anon_rw_pet_policies"
ON public.pet_policies
FOR ALL
TO anon
USING (true)
WITH CHECK (true);

ALTER TABLE public.insurance_requirements ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "anon_rw_insurance_requirements" ON public.insurance_requirements;
CREATE POLICY "anon_rw_insurance_requirements"
ON public.insurance_requirements
FOR ALL
TO anon
USING (true)
WITH CHECK (true);
```

After updating permissions/policies, rerun:

```bat
py -3.9 scripts\seed_jurisdictions.py
py -3.9 scripts\seed_db.py
py -3.9 scripts\index_regulations.py
```

## 7) Start the app

Run Streamlit:

```bat
py -3.9 -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.headless=true
```

Open:
- `http://127.0.0.1:8501`

## 8) Use the features

### A) Chat Q&A Agent + Lease Compliance Checker (`pages/1_agent.py`)

In the left sidebar:
- Choose a **state**
- Choose a **city** (or keep statewide selection if available)

Then use:

1. **Chat Q&A**
   - Ask about housing regulation topics (the agent uses vector search over the regulation corpus).
2. **Lease Compliance Checker**
   - Upload a `pdf` or `docx` file
   - Ask a compliance-related question (the UI routes to the compliance checker)

Notes:
- If no LLM API key is configured (Anthropic, OpenAI, or Google Gemini), the app falls back to a rule-based compliance engine.
- Each answer includes a **Sources** section (deduplicated).

### B) Regulation Explorer (`pages/2_explorer.py`)

- Search regulations by keyword/topic
- Filter by **category** and **state**
- View results in a table with source URLs

### C) Update Log (`pages/3_update_log.py`)

- Click **Check for updates now**
- Filter by state
- Review updates (source + summary + affected jurisdictions)

### D) Email Alerts (`pages/4_email_alerts.py`)

If SMTP is configured:
- Subscribe: choose jurisdiction + enter email
- Unsubscribe: disable alerts for an email/jurisdiction
- View active subscriptions

### E) Settings (`pages/5_settings.py`)

- **Load regulations from CSV**: runs `seed_db.py` logic
- **Initialize vector index**: runs vector embedding for unindexed rows
- **Indexing status**: shows per-state indexing counts
- **Source Registry** (requires migration 007):
  - **Global toggle**: switch between CSV and DB as the source provider
  - **Import CSV → DB**: one-click idempotent backfill of `sources.csv` into `regulation_sources`
  - **Source list**: view all registered sources with per-source Activate/Deactivate, Test, and Delete actions
  - **Add source form**: register new URLs directly from the UI
  - **Test Source**: probe a URL for reachability without scraping
- **Scraper**:
  - Manual trigger starts scraping + re-indexing (reads from CSV or DB depending on toggle)

## 9) Troubleshooting

- `permission denied for table ...` or `RLS` errors while running scripts:
  - Use `service_role` temporarily for seed/index scripts, or apply the dev SQL in step 6.
- `Unknown state_code 'CA'`:
  - Your `jurisdictions` table is missing state rows. Rerun `seed_jurisdictions.py`.

