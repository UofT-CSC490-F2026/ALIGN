\set ON_ERROR_STOP on
\echo '--- Preflight for adopting the EXISTING pre-Flyway ALIGN v3 schema ---'
DO $$
DECLARE
  actual_table_count int;
  t text;
BEGIN
  -- Do not baseline a Flyway-managed database or a partially initialized one.
  IF to_regclass('public.flyway_schema_history') IS NOT NULL THEN
    RAISE EXCEPTION 'Flyway history already exists. Run Flyway info/validate, not baseline.';
  END IF;
  FOREACH t IN ARRAY ARRAY[
    'tasks','task_goals','runs','events','decision_log','conflict_episodes',
    'evaluations','goal_results','decision_labels'
  ] LOOP
    IF to_regclass('public.'||t) IS NULL THEN
      RAISE EXCEPTION 'Expected v3 table % is missing; do not baseline', t;
    END IF;
  END LOOP;
  IF to_regclass('public.v_run_summary') IS NULL OR
     to_regclass('public.v_task_comparison') IS NULL THEN
    RAISE EXCEPTION 'Expected v3 views missing; do not baseline';
  END IF;
  SELECT count(*) INTO actual_table_count FROM information_schema.tables
    WHERE table_schema='public' AND table_type='BASE TABLE';
  IF actual_table_count <> 9 THEN
    RAISE EXCEPTION 'Expected exactly 9 existing ALIGN tables, found %', actual_table_count;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_trigger
      WHERE tgname='trg_decision_event_type' AND NOT tgisinternal) THEN
    RAISE EXCEPTION 'Expected v3 decision event integrity trigger missing';
  END IF;
  RAISE NOTICE 'PASS: v3 legacy schema present. You may now baseline at version 2.';
END $$;
