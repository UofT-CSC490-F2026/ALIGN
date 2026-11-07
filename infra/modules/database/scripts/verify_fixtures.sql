\set ON_ERROR_STOP on
\echo '--- ALIGN synthetic fixtures: ten tables and two reporting views ---'
SELECT table_name FROM information_schema.tables
 WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
 AND table_name <> 'flyway_schema_history' ORDER BY table_name;
SELECT table_name FROM information_schema.views
 WHERE table_schema = 'public' ORDER BY table_name;
SELECT run_id, task_key, arbiter_condition, outcome_correct,
       goals_passed, goals_evaluated, goal_completion_rate,
       num_conflicts, interventions
FROM v_run_summary ORDER BY run_id;
SELECT task_key, arbiter_condition, evaluated_runs, outcome_success_rate,
       avg_goal_completion_rate
FROM v_task_comparison ORDER BY task_key, arbiter_condition;

DO $$
BEGIN
 IF (SELECT count(*) FROM information_schema.tables
     WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
       AND table_name <> 'flyway_schema_history') <> 10 THEN
     RAISE EXCEPTION 'Expected 10 base tables';
 END IF;
 IF (SELECT count(*) FROM information_schema.views WHERE table_schema = 'public') <> 2 THEN
     RAISE EXCEPTION 'Expected 2 views';
 END IF;
 IF (SELECT count(*) FROM tasks) <> 2 OR (SELECT count(*) FROM runs) <> 10 THEN
     RAISE EXCEPTION 'Fixture counts incorrect';
 END IF;
 IF (SELECT count(*) FROM events) <> 21 THEN
     RAISE EXCEPTION 'Expected 12 timeline events';
 END IF;
 IF (SELECT count(*) FROM decision_log) <> 6 THEN
     RAISE EXCEPTION 'Expected 2 arbiter decisions';
 END IF;
 IF EXISTS (SELECT 1 FROM decision_log d JOIN events e ON e.event_id=d.event_id
            WHERE e.actor_type <> 'arbiter' OR e.event_type <> 'arbiter_decision'
               OR d.run_id<>e.run_id) THEN
     RAISE EXCEPTION 'Invalid arbiter event reference';
 END IF;
 IF (SELECT count(*) FROM v_run_summary WHERE arbiter_condition='none' AND trial_key='trial-001' AND interventions=0) <> 1 THEN
     RAISE EXCEPTION 'No-arbiter baseline should have zero interventions';
 END IF;
 IF (SELECT count(*) FROM v_run_summary WHERE arbiter_condition='llm'
        AND outcome_correct AND task_key='inventory-integration-v1'
        AND goal_completion_rate=1 AND interventions=1) <> 1 THEN
     RAISE EXCEPTION 'ALIGN illustrative successful run summary incorrect';
 END IF;
 IF (SELECT count(*) FROM v_run_summary WHERE task_key='contradictory-stock-v1'
        AND outcome_correct AND goal_completion_rate IS NULL) <> 1 THEN
     RAISE EXCEPTION 'Escalation should be successful with undefined goal completion rate';
 END IF;
 RAISE NOTICE 'PASS: counts, relationships, baseline, goal metrics, escalation';
END $$;

-- Negative tests: incorrect cross-task and cross-run relationships must be rejected.
DO $$
DECLARE
    base_run INTEGER;
    base_task INTEGER;
    other_goal INTEGER;
    other_conflict INTEGER;
    align_decision INTEGER;
BEGIN
    SELECT r.run_id, r.task_id INTO base_run,base_task FROM runs r
    JOIN tasks t ON r.task_id=t.task_id
    WHERE t.task_key='inventory-integration-v1' AND r.arbiter_condition='none';
    SELECT g.goal_id INTO other_goal FROM task_goals g
    JOIN tasks t ON t.task_id=g.task_id WHERE t.task_key='contradictory-stock-v1' LIMIT 1;
    BEGIN
        INSERT INTO goal_results (run_id, task_id, goal_id, result)
        VALUES (base_run, base_task, other_goal, 'passed');
        RAISE EXCEPTION 'FAIL: cross-task goal link was accepted';
    EXCEPTION WHEN foreign_key_violation THEN
        RAISE NOTICE 'PASS: cross-task goal link rejected by foreign key';
    END;
    SELECT c.conflict_id INTO other_conflict FROM conflict_episodes c
    JOIN runs r ON r.run_id=c.run_id
    JOIN tasks t ON t.task_id=r.task_id WHERE t.task_key='contradictory-stock-v1';
    SELECT d.event_id INTO align_decision FROM decision_log d
    JOIN runs r ON r.run_id=d.run_id
    JOIN tasks t ON t.task_id=r.task_id WHERE t.task_key='inventory-integration-v1';
    BEGIN
        UPDATE decision_log SET conflict_id=other_conflict WHERE event_id=align_decision;
        RAISE EXCEPTION 'FAIL: cross-run conflict link was accepted';
    EXCEPTION WHEN foreign_key_violation THEN
        RAISE NOTICE 'PASS: cross-run conflict link rejected by foreign key';
    END;
END $$;

\echo '--- ALIGN evaluation metrics regression tests ---'
DO $$
DECLARE
    unclassified BIGINT;
BEGIN
    IF (SELECT count(*) FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'action_attempts') <> 1 THEN
        RAISE EXCEPTION 'Missing action_attempts table';
    END IF;
    IF (SELECT count(*) FROM runs WHERE trial_key LIKE 'fixture-v4-%') <> 7 THEN
        RAISE EXCEPTION 'Metric seed has not been applied';
    END IF;
    IF (SELECT count(*) FROM tasks) <> 2
       OR (SELECT count(*) FROM runs) <> 10
       OR (SELECT count(*) FROM action_attempts) <> 5 THEN
       RAISE EXCEPTION 'Synthetic fixture counts invalid';
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
    IF NOT EXISTS(SELECT 1 FROM flyway_schema_history WHERE version='2' AND success) THEN
        RAISE EXCEPTION 'Reporting views migration not recorded';
    END IF;
    RAISE NOTICE 'PASS: action safety, intervention quality, failed runs, and timing';
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

