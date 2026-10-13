\set ON_ERROR_STOP on
\echo '--- ALIGN V1/V2 base schema checks ---'
DO $$
BEGIN
 IF (SELECT count(*) FROM information_schema.tables
     WHERE table_schema='public' AND table_type='BASE TABLE'
     AND table_name <> 'flyway_schema_history') <> 9 THEN
     RAISE EXCEPTION 'Expected 9 domain tables';
 END IF;
 IF to_regclass('public.v_run_summary') IS NULL OR
    to_regclass('public.v_task_comparison') IS NULL THEN
    RAISE EXCEPTION 'Missing base view';
 END IF;
 IF to_regclass('public.idx_events_run_event_type') IS NOT NULL THEN
    RAISE EXCEPTION 'Expected pre-V3 schema but monitoring index is already installed';
 END IF;
 IF NOT EXISTS (SELECT 1 FROM flyway_schema_history
                WHERE version='2' AND success) THEN
    RAISE EXCEPTION 'V2 not applied';
 END IF;
 RAISE NOTICE 'PASS: V1 and V2 created 9 domain tables and 2 views';
END $$;
