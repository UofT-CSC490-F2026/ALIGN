\set ON_ERROR_STOP on
\echo '--- Original ALIGN fixture test: 9 original domain tables, 2 views, 2 tasks, 3 runs ---'
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
       AND table_name NOT IN ('flyway_schema_history', 'action_attempts')) <> 9 THEN
     RAISE EXCEPTION 'Expected 9 base tables';
 END IF;
 IF (SELECT count(*) FROM information_schema.views WHERE table_schema = 'public') <> 2 THEN
     RAISE EXCEPTION 'Expected 2 views';
 END IF;
 IF (SELECT count(*) FROM tasks) <> 2 OR (SELECT count(*) FROM runs) <> 3 THEN
     RAISE EXCEPTION 'Fixture counts incorrect';
 END IF;
 IF (SELECT count(*) FROM events) <> 12 THEN
     RAISE EXCEPTION 'Expected 12 timeline events';
 END IF;
 IF (SELECT count(*) FROM decision_log) <> 2 THEN
     RAISE EXCEPTION 'Expected 2 arbiter decisions';
 END IF;
 IF EXISTS (SELECT 1 FROM decision_log d JOIN events e ON e.event_id=d.event_id
            WHERE e.actor_type <> 'arbiter' OR e.event_type <> 'arbiter_decision'
               OR d.run_id<>e.run_id) THEN
     RAISE EXCEPTION 'Invalid arbiter event reference';
 END IF;
 IF (SELECT count(*) FROM v_run_summary WHERE arbiter_condition='none' AND interventions=0) <> 1 THEN
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
