-- SYNTHETIC fixtures for schema verification only.
-- These are not actual agent experiments, independent gold labels, or runnable repositories.
BEGIN;
DO $$
DECLARE
    task_inventory INTEGER;
    task_design INTEGER;
    goal_reserve INTEGER;
    goal_cancel INTEGER;
    goal_consistency INTEGER;
    goal_reject INTEGER;
    goal_accept INTEGER;
    run_baseline INTEGER;
    run_align INTEGER;
    run_design INTEGER;
    base_edit_a INTEGER;
    base_edit_b INTEGER;
    base_failure INTEGER;
    align_edit_a INTEGER;
    align_edit_b INTEGER;
    align_failure INTEGER;
    align_decision INTEGER;
    align_fix INTEGER;
    align_pass INTEGER;
    design_msg_a INTEGER;
    design_msg_b INTEGER;
    design_decision INTEGER;
    conflict_baseline INTEGER;
    conflict_align INTEGER;
    conflict_design INTEGER;
BEGIN
    -- Repeatable local setup: if the fixture is fully present, do nothing.
    -- Do not silently complete a partially loaded fixture (that warrants inspection).
    IF EXISTS (SELECT 1 FROM tasks WHERE task_key IN
               ('inventory-integration-v1','contradictory-stock-v1')) THEN
        IF (SELECT count(*) FROM tasks WHERE task_key IN
            ('inventory-integration-v1','contradictory-stock-v1')) <> 2 THEN
            RAISE EXCEPTION 'Partial ALIGN fixture detected; refusing to seed';
        END IF;
        RAISE NOTICE 'ALIGN synthetic fixtures already present; seed skipped';
        RETURN;
    END IF;
    INSERT INTO tasks (task_key, dataset_source, version, repo_url, base_commit, shared_goal,
                       conflict_type, expected_outcome, metadata)
    VALUES ('inventory-integration-v1', 'synthetic_fixture', 1, 'fixture://orders-backend',
            'synthetic-not-a-git-commit', 'Implement compatible order and inventory features',
            'execution_conflict', 'complete_all_goals',
            '{"fixture_only": true}'::jsonb)
    RETURNING task_id INTO task_inventory;

    INSERT INTO tasks (task_key, dataset_source, version, repo_url, base_commit, shared_goal,
                       conflict_type, expected_outcome, metadata)
    VALUES ('contradictory-stock-v1', 'synthetic_fixture', 1, 'fixture://orders-backend',
            'synthetic-not-a-git-commit', 'Identify contradictory requirements and escalate',
            'design_conflict', 'escalate', '{"fixture_only": true}'::jsonb)
    RETURNING task_id INTO task_design;

    INSERT INTO task_goals (task_id, goal_key, owner_type, owner_agent_id, description, success_criteria)
    VALUES (task_inventory, 'reservation', 'agent', 'agent_a', 'Support inventory reservations',
            '{"check": "reservation_test"}'::jsonb)
    RETURNING goal_id INTO goal_reserve;
    INSERT INTO task_goals (task_id, goal_key, owner_type, owner_agent_id, description, success_criteria)
    VALUES (task_inventory, 'cancellation', 'agent', 'agent_b', 'Release stock on cancellation',
            '{"check": "cancellation_test"}'::jsonb)
    RETURNING goal_id INTO goal_cancel;
    INSERT INTO task_goals (task_id, goal_key, owner_type, description, success_criteria, is_critical)
    VALUES (task_inventory, 'consistency', 'shared', 'Preserve inventory consistency',
            '{"check": "inventory_invariant"}'::jsonb, true)
    RETURNING goal_id INTO goal_consistency;

    INSERT INTO task_goals (task_id, goal_key, owner_type, owner_agent_id, description)
    VALUES (task_design, 'reject', 'agent', 'agent_a', 'Always reject out-of-stock orders')
    RETURNING goal_id INTO goal_reject;
    INSERT INTO task_goals (task_id, goal_key, owner_type, owner_agent_id, description)
    VALUES (task_design, 'accept', 'agent', 'agent_b', 'Always accept out-of-stock orders')
    RETURNING goal_id INTO goal_accept;

    INSERT INTO runs (task_id, trial_key, arbiter_condition, agent_config, seed, status,
                      started_at, finished_at)
    VALUES (task_inventory, 'trial-001', 'none', '{"agent_a":"fixture-model","agent_b":"fixture-model"}'::jsonb,
            42, 'completed', '2026-10-12T10:00:00Z', '2026-10-12T10:02:00Z')
    RETURNING run_id INTO run_baseline;
    INSERT INTO runs (task_id, trial_key, arbiter_condition, agent_config, arbiter_config, seed, status,
                      started_at, finished_at)
    VALUES (task_inventory, 'trial-001', 'llm', '{"agent_a":"fixture-model","agent_b":"fixture-model"}'::jsonb,
            '{"arbiter":"fixture-model"}'::jsonb, 42, 'completed',
            '2026-10-12T10:10:00Z', '2026-10-12T10:12:20Z')
    RETURNING run_id INTO run_align;
    INSERT INTO runs (task_id, trial_key, arbiter_condition, agent_config, arbiter_config, seed, status,
                      started_at, finished_at)
    VALUES (task_design, 'trial-002', 'llm', '{"agent_a":"fixture-model","agent_b":"fixture-model"}'::jsonb,
            '{"arbiter":"fixture-model"}'::jsonb, 43, 'completed',
            '2026-10-12T11:00:00Z', '2026-10-12T11:01:10Z')
    RETURNING run_id INTO run_design;

    -- Baseline: conflict occurred, but no arbiter decisions are recorded.
    INSERT INTO events (run_id, sequence_number, actor_type, actor_id, event_type, occurred_at, payload)
    VALUES (run_baseline, 1, 'agent', 'agent_a', 'file_edit', '2026-10-12T10:00:10Z', '{"file":"inventory.py"}')
    RETURNING event_id INTO base_edit_a;
    INSERT INTO events (run_id, sequence_number, actor_type, actor_id, event_type, occurred_at, payload)
    VALUES (run_baseline, 2, 'agent', 'agent_b', 'file_edit', '2026-10-12T10:00:20Z', '{"file":"orders.py"}')
    RETURNING event_id INTO base_edit_b;
    INSERT INTO events (run_id, sequence_number, actor_type, event_type, occurred_at, payload)
    VALUES (run_baseline, 3, 'system', 'test_failure', '2026-10-12T10:00:35Z', '{"check":"inventory_invariant"}')
    RETURNING event_id INTO base_failure;
    INSERT INTO conflict_episodes (run_id, conflict_type, onset_event_id, status,
                                   label_source, description)
    VALUES (run_baseline, 'execution_conflict', base_edit_b, 'unresolved', 'synthetic',
            'Incompatible reservation and cancellation implementations (fixture)')
    RETURNING conflict_id INTO conflict_baseline;
    INSERT INTO decision_labels (checkpoint_event_id, expected_actions, label_source, rationale)
    VALUES (base_failure, ARRAY['query','mediate'], 'synthetic',
            'Provisional fixture label for failed invariant on baseline trace');

    -- ALIGN run: related conflict, mediated, subsequently resolved.
    INSERT INTO events (run_id, sequence_number, actor_type, actor_id, event_type, occurred_at, observed_at, payload)
    VALUES (run_align, 1, 'agent', 'agent_a', 'file_edit', '2026-10-12T10:10:10Z',
            '2026-10-12T10:10:11Z', '{"file":"inventory.py"}')
    RETURNING event_id INTO align_edit_a;
    INSERT INTO events (run_id, sequence_number, actor_type, actor_id, event_type, occurred_at, observed_at, payload)
    VALUES (run_align, 2, 'agent', 'agent_b', 'file_edit', '2026-10-12T10:10:20Z',
            '2026-10-12T10:10:21Z', '{"file":"orders.py"}')
    RETURNING event_id INTO align_edit_b;
    INSERT INTO events (run_id, sequence_number, actor_type, event_type, occurred_at, observed_at, payload)
    VALUES (run_align, 3, 'system', 'test_failure', '2026-10-12T10:10:34Z',
            '2026-10-12T10:10:36Z', '{"check":"inventory_invariant"}')
    RETURNING event_id INTO align_failure;
    INSERT INTO events (run_id, sequence_number, actor_type, actor_id, event_type, occurred_at, payload)
    VALUES (run_align, 4, 'arbiter', 'align', 'arbiter_decision', '2026-10-12T10:10:39Z',
            '{"checkpoint":"integration_test_failure"}')
    RETURNING event_id INTO align_decision;
    INSERT INTO events (run_id, sequence_number, actor_type, actor_id, event_type, occurred_at, payload)
    VALUES (run_align, 5, 'agent', 'agent_b', 'file_edit', '2026-10-12T10:11:00Z',
            '{"file":"orders.py","change":"coordinate_reservations"}')
    RETURNING event_id INTO align_fix;
    INSERT INTO events (run_id, sequence_number, actor_type, event_type, occurred_at, payload)
    VALUES (run_align, 6, 'system', 'test_pass', '2026-10-12T10:11:30Z',
            '{"check":"inventory_invariant"}')
    RETURNING event_id INTO align_pass;
    INSERT INTO conflict_episodes (run_id, conflict_type, onset_event_id, detected_event_id,
                                   resolution_event_id, status, label_source, description)
    VALUES (run_align, 'execution_conflict', align_edit_b, align_failure, align_pass,
            'resolved', 'synthetic', 'Coordinated schema/logic conflict (fixture)')
    RETURNING conflict_id INTO conflict_align;
    INSERT INTO decision_log (event_id, run_id, trigger_event_id, conflict_id,
                              arbiter_action, reason, affected_agents, decision_latency_ms)
    VALUES (align_decision, run_align, align_failure, conflict_align, 'mediate',
            'Coordinate the reservation contract before continuing', ARRAY['agent_a','agent_b'], 3000);
    INSERT INTO decision_labels (checkpoint_event_id, expected_actions, label_source, rationale)
    VALUES (align_failure, ARRAY['query','mediate'], 'synthetic', 'Provisional fixture label');

    -- Design conflict: escalation is correct even though goals remain unimplemented.
    INSERT INTO events (run_id, sequence_number, actor_type, actor_id, event_type, occurred_at, payload)
    VALUES (run_design, 1, 'agent', 'agent_a', 'message', '2026-10-12T11:00:07Z',
            '{"requirement":"reject all out-of-stock orders"}')
    RETURNING event_id INTO design_msg_a;
    INSERT INTO events (run_id, sequence_number, actor_type, actor_id, event_type, occurred_at, payload)
    VALUES (run_design, 2, 'agent', 'agent_b', 'message', '2026-10-12T11:00:12Z',
            '{"requirement":"accept all out-of-stock orders"}')
    RETURNING event_id INTO design_msg_b;
    INSERT INTO events (run_id, sequence_number, actor_type, actor_id, event_type, occurred_at, payload)
    VALUES (run_design, 3, 'arbiter', 'align', 'arbiter_decision', '2026-10-12T11:00:22Z',
            '{"checkpoint":"incompatible_requirements"}')
    RETURNING event_id INTO design_decision;
    INSERT INTO conflict_episodes (run_id, conflict_type, onset_event_id, detected_event_id,
                                   resolution_event_id, status, label_source, description)
    VALUES (run_design, 'design_conflict', design_msg_b, design_msg_b,
            design_decision, 'escalated', 'synthetic', 'Contradictory stock-acceptance requirements (fixture)')
    RETURNING conflict_id INTO conflict_design;
    INSERT INTO decision_log (event_id, run_id, trigger_event_id, conflict_id,
                              arbiter_action, reason, affected_agents, decision_latency_ms)
    VALUES (design_decision, run_design, design_msg_b, conflict_design, 'escalate',
            'Requirements cannot both hold for identical requests', ARRAY['agent_a','agent_b'], 10000);
    INSERT INTO decision_labels (checkpoint_event_id, expected_actions, label_source, rationale)
    VALUES (design_msg_b, ARRAY['escalate'], 'synthetic', 'Provisional fixture label');

    -- Overall results are run-level, and goal results reference the task's original goals.
    INSERT INTO evaluations (run_id, evaluator_version, outcome_correct, shared_goal_satisfied,
                             critical_constraints_passed, runtime_seconds, input_tokens,
                             output_tokens, token_cost_usd, evaluation_notes)
    VALUES (run_baseline, 'fixture-v1', false, false, false, 120, 5000, 1000, 0.050000,
            'Illustrative failure, not measured'),
           (run_align, 'fixture-v1', true, true, true, 140, 5800, 1300, 0.065000,
            'Illustrative success, not measured'),
           (run_design, 'fixture-v1', true, false, true, 70, 2000, 500, 0.020000,
            'Correct escalation, not completed goals');

    INSERT INTO goal_results (run_id, task_id, goal_id, result)
    VALUES (run_baseline, task_inventory, goal_reserve, 'passed'),
           (run_baseline, task_inventory, goal_cancel, 'passed'),
           (run_baseline, task_inventory, goal_consistency, 'failed'),
           (run_align, task_inventory, goal_reserve, 'passed'),
           (run_align, task_inventory, goal_cancel, 'passed'),
           (run_align, task_inventory, goal_consistency, 'passed'),
           (run_design, task_design, goal_reject, 'not_evaluated'),
           (run_design, task_design, goal_accept, 'not_evaluated');
END $$;
COMMIT;
