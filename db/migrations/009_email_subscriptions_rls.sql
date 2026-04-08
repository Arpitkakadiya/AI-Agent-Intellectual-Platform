-- Migration 009: Grant anon access to email_subscriptions and add RLS policy
-- Run this in the Supabase SQL Editor if you get "permission denied for table email_subscriptions"

-- Grant table-level permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.email_subscriptions TO anon, authenticated;

-- Ensure RLS is enabled (Supabase enables it by default on new tables)
ALTER TABLE public.email_subscriptions ENABLE ROW LEVEL SECURITY;

-- Drop existing policy if present, then create a permissive one
DROP POLICY IF EXISTS "anon_rw_email_subscriptions" ON public.email_subscriptions;
CREATE POLICY "anon_rw_email_subscriptions"
ON public.email_subscriptions
FOR ALL
TO anon
USING (true)
WITH CHECK (true);

DROP POLICY IF EXISTS "authenticated_rw_email_subscriptions" ON public.email_subscriptions;
CREATE POLICY "authenticated_rw_email_subscriptions"
ON public.email_subscriptions
FOR ALL
TO authenticated
USING (true)
WITH CHECK (true);
