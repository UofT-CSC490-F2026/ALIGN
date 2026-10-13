\set ON_ERROR_STOP on
\echo '--- ALIGN schema verification (data-independent) ---'
DO $$
DECLARE
  domain_tables text[] := ARRAY[
    'tasks','task_goals','runs','events','decision_log','conflict_episodes',
    'evaluations','goal_results','decision_labels'];
  t text;
  n int;
BEGIN
  FOREACH t IN ARRAY domain_tables LOOP
    IF to_regclass('public.' || t) IS NULL THEN
      RAISE EXCEPTION 'Missing domain table %', t;
    END IF;
  END LOOP;
  IF to_regclass('public.v_run_summary') IS NULL OR
     to_regclass('public.v_task_comparison') IS NULL THEN
     RAISE EXCEPTION 'Missing analysis view';
  END IF;
  -- Flyway adds its own bookkeeping table, not a tenth domain entity.
  IF to_regclass('public.flyway_schema_history') IS NULL THEN
    RAISE EXCEPTION 'Missing Flyway schema history';
  END IF;
  SELECT count(*) INTO n FROM information_schema.tables
   WHERE table_schema='public' AND table_type='BASE TABLE'
     AND table_name=ANY(domain_tables);
  IF n<>9 THEN RAISE EXCEPTION 'Expected nine ALIGN domain tables; found %',n; END IF;
  IF to_regclass('public.idx_events_run_event_type') IS NULL THEN
    RAISE EXCEPTION 'V3 migration missing: event monitoring index';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_trigger
     WHERE tgname='trg_decision_event_type' AND NOT tgisinternal) THEN
    RAISE EXCEPTION 'Missing arbiter event constraint trigger';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM flyway_schema_history
     WHERE version='3' AND success) THEN
    RAISE EXCEPTION 'Flyway V3 migration not recorded as successful';
  END IF;
  RAISE NOTICE 'PASS: 9 domain tables, 2 views, Flyway history, V3 index and triggers';
END $$;
\echo '--- Flyway applied migrations ---'
SELECT installed_rank,version,description,type,success
FROM flyway_schema_history ORDER BY installed_rank;
