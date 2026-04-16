-- Migration 011: Allow anon/authenticated to read/write regulation_updates (Update Log)
-- Run in Supabase SQL Editor if you see: permission denied for table regulation_updates

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.regulation_updates TO anon, authenticated;
GRANT USAGE, SELECT ON SEQUENCE public.regulation_updates_id_seq TO anon, authenticated;

ALTER TABLE public.regulation_updates ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "anon_rw_regulation_updates" ON public.regulation_updates;
CREATE POLICY "anon_rw_regulation_updates"
ON public.regulation_updates
FOR ALL
TO anon
USING (true)
WITH CHECK (true);

DROP POLICY IF EXISTS "authenticated_rw_regulation_updates" ON public.regulation_updates;
CREATE POLICY "authenticated_rw_regulation_updates"
ON public.regulation_updates
FOR ALL
TO authenticated
USING (true)
WITH CHECK (true);
