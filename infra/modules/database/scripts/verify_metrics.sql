\set ON_ERROR_STOP on
\echo '--- ALIGN V4 evaluation metrics regression tests ---'
DO $$
DECLARE
    unclassified BIGINT;
BEGIN
    IF (SELECT count(*) FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'action_attempts') <> 1 THEN
        RAISE EXCEPTION 'Missing action_attempts table';
    END IF;
    IF (SELECT count(*) FROM runs WHERE trial_key LIKE 'fixture-v4-%') <> 7 THEN
        RAISE EXCEPTION 'V4 metric seed has not been applied';
    END IF;
    IF (SELECT count(*) FROM tasks) <> 2
       OR (SELECT count(*) FROM runs) <> 10
       OR (SELECT count(*) FROM action_attempts) <> 5 THEN
       RAISE EXCEPTION 'Original fixtures not preserved or V4 fixtures invalid';
    END IF;
    IF (SELECT count(*) FROM action_attempts
        WHERE is_unsafe = TRUE AND review_status = 'reviewed') <> 2 THEN
        RAISE EXCEPTION 'Incorrect count of accepted unsafe actions';
    END IF;
    IF (SELECT count(*) FROM v_run_summary
        WHERE trial_key = 'fixture-v4-checkpoint-1'
          AND correct_interventions = 1 AND unsafe_attempted = 1
          AND unsafe_blocked = 1 AND unsafe_executed = 0) <> 1 THEN
        RAISE EXCEPTION 'Blocked unsafe action / correct intervention incorrect';
    END IF;
    IF (SELECT count(*) FROM v_run_summary
        WHERE trial_key = 'fixture-v4-checkpoint-2'
          AND missed_interventions = 1 AND unsafe_attempted = 1
          AND unsafe_executed = 1) <> 1 THEN
        RAISE EXCEPTION 'Unsafe dispatched / missed intervention incorrect';
    END IF;
    IF (SELECT count(*) FROM v_run_summary
        WHERE trial_key = 'fixture-v4-checkpoint-3'
          AND false_interventions = 1 AND correct_interventions = 0) <> 1 THEN
        RAISE EXCEPTION 'False intervention incorrect';
    END IF;
    IF (SELECT count(*) FROM v_run_summary
        WHERE trial_key = 'fixture-v4-checkpoint-4'
          AND wrong_type_interventions = 1) <> 1 THEN
        RAISE EXCEPTION 'Wrong intervention type incorrect';
    END IF;
    IF (SELECT count(*) FROM v_run_summary
        WHERE trial_key = 'fixture-v4-checkpoint-5'
          AND unobserved_checkpoints = 1 AND missed_interventions = 0
          AND safety_unclassified = 1 AND unsafe_attempted = 0) <> 1 THEN
        RAISE EXCEPTION 'Unobserved/unknown must not count as missed or unsafe';
    END IF;
    IF (SELECT count(*) FROM v_run_summary
        WHERE task_key = 'inventory-integration-v1'
          AND arbiter_condition = 'llm'
          AND trial_key = 'trial-001'
          AND correct_interventions = 1
          AND time_to_first_intervention_seconds = 19) <> 1 THEN
        RAISE EXCEPTION 'Original ALIGN intervention or timing changed';
    END IF;
    IF (SELECT count(*) FROM v_task_comparison
        WHERE task_key = 'inventory-integration-v1'
          AND arbiter_condition = 'none'
          AND total_runs = 3 AND failed_runs = 1 AND running_runs = 1
          AND terminal_runs = 2 AND terminal_unlabeled_runs = 1
          AND terminal_success_rate = 0) <> 1 THEN
        RAISE EXCEPTION 'Failed/incomplete runs are missing from comparison';
    END IF;
    IF (SELECT count(*) FROM v_task_comparison
        WHERE task_key = 'inventory-integration-v1' AND arbiter_condition = 'llm'
          AND total_runs = 6 AND terminal_runs = 6 AND successful_terminal_runs = 1
          AND terminal_success_rate = (1.0 / 6)
          AND unsafe_attempted = 2 AND unsafe_blocked = 1 AND unsafe_executed = 1
          AND correct_interventions = 2 AND false_interventions = 1
          AND missed_interventions = 1 AND wrong_type_interventions = 1
          AND unobserved_checkpoints = 1) <> 1 THEN
        RAISE EXCEPTION 'Aggregate outcome/safety/intervention metrics incorrect';
    END IF;
    IF NOT EXISTS(SELECT 1 FROM flyway_schema_history WHERE version='4' AND success) THEN
        RAISE EXCEPTION 'V4 migration not recorded';
    END IF;
    RAISE NOTICE 'PASS: V4 action safety, intervention quality, failed runs, and timing';
END $$;

-- One reference label per checkpoint and only after review.
DO $$
DECLARE
    checkpoint INTEGER;
BEGIN
    SELECT checkpoint_event_id INTO checkpoint FROM decision_labels
    WHERE is_reference LIMIT 1;
    BEGIN
        INSERT INTO decision_labels(checkpoint_event_id, expected_actions,
                                    label_source, review_status, is_reference)
        VALUES(checkpoint, ARRAY['continue'], 'synthetic', 'reviewed', TRUE);
        RAISE EXCEPTION 'Multiple reference labels incorrectly accepted';
    EXCEPTION WHEN unique_violation THEN
        RAISE NOTICE 'PASS: only one reference label per checkpoint';
    END;
    BEGIN
        INSERT INTO decision_labels(checkpoint_event_id, expected_actions,
                                    label_source, review_status, is_reference)
        VALUES(checkpoint, ARRAY['continue'], 'synthetic', 'provisional', TRUE);
        RAISE EXCEPTION 'Unreviewed reference label incorrectly accepted';
    EXCEPTION WHEN check_violation THEN
        RAISE NOTICE 'PASS: reference labels must be reviewed';
    END;
END $$;
