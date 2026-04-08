-- Migration 010: Grant anon/authenticated access to pet_policies and insurance_requirements
-- Run this in the Supabase SQL Editor if you get "permission denied for table pet_policies"
-- or "permission denied for table insurance_requirements"

-- Grant table-level permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.pet_policies TO anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.insurance_requirements TO anon, authenticated;

-- Grant sequence access (needed for inserts with SERIAL columns)
GRANT USAGE, SELECT ON SEQUENCE public.pet_policies_id_seq TO anon, authenticated;
GRANT USAGE, SELECT ON SEQUENCE public.insurance_requirements_id_seq TO anon, authenticated;

-- Ensure RLS is enabled
ALTER TABLE public.pet_policies ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.insurance_requirements ENABLE ROW LEVEL SECURITY;

-- pet_policies: permissive policies for dev
DROP POLICY IF EXISTS "anon_rw_pet_policies" ON public.pet_policies;
CREATE POLICY "anon_rw_pet_policies"
ON public.pet_policies
FOR ALL
TO anon
USING (true)
WITH CHECK (true);

DROP POLICY IF EXISTS "authenticated_rw_pet_policies" ON public.pet_policies;
CREATE POLICY "authenticated_rw_pet_policies"
ON public.pet_policies
FOR ALL
TO authenticated
USING (true)
WITH CHECK (true);

-- insurance_requirements: permissive policies for dev
DROP POLICY IF EXISTS "anon_rw_insurance_requirements" ON public.insurance_requirements;
CREATE POLICY "anon_rw_insurance_requirements"
ON public.insurance_requirements
FOR ALL
TO anon
USING (true)
WITH CHECK (true);

DROP POLICY IF EXISTS "authenticated_rw_insurance_requirements" ON public.insurance_requirements;
CREATE POLICY "authenticated_rw_insurance_requirements"
ON public.insurance_requirements
FOR ALL
TO authenticated
USING (true)
WITH CHECK (true);
