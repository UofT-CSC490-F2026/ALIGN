-- Flyway versioned migration. Do not edit after deployment; add V4, V5, ... instead.
-- Derived metrics; keep expensive counting out of the write path.
CREATE VIEW v_run_summary AS
WITH goal_stats AS (
    SELECT
        run_id,
        count(*) AS goals_recorded,
        count(*) FILTER (WHERE result IN ('passed','failed')) AS goals_evaluated,
        count(*) FILTER (WHERE result = 'passed') AS goals_passed
    FROM goal_results GROUP BY run_id
),
decision_stats AS (
    SELECT
        d.run_id,
        count(*) AS arbiter_checks,
        count(*) FILTER (WHERE d.arbiter_action <> 'continue') AS interventions
    FROM decision_log d GROUP BY d.run_id
),
conflict_stats AS (
    SELECT
        run_id,
        count(*) AS conflicts,
        count(*) FILTER (WHERE status = 'resolved') AS conflicts_resolved,
        count(*) FILTER (WHERE status = 'escalated') AS conflicts_escalated
    FROM conflict_episodes GROUP BY run_id
)
SELECT
    r.run_id, r.task_id, t.task_key, t.conflict_type,
    r.trial_key, r.seed, r.arbiter_condition, r.status,
    e.outcome_correct, e.shared_goal_satisfied, e.critical_constraints_passed,
    coalesce(g.goals_recorded, 0) AS goals_recorded,
    coalesce(g.goals_evaluated, 0) AS goals_evaluated,
    coalesce(g.goals_passed, 0) AS goals_passed,
    CASE WHEN coalesce(g.goals_evaluated, 0) > 0
         THEN g.goals_passed::numeric / g.goals_evaluated
         ELSE NULL END AS goal_completion_rate,
    coalesce(d.arbiter_checks, 0) AS arbiter_checks,
    coalesce(d.interventions, 0) AS interventions,
    coalesce(c.conflicts, 0) AS num_conflicts,
    coalesce(c.conflicts_resolved, 0) AS conflicts_resolved,
    coalesce(c.conflicts_escalated, 0) AS conflicts_escalated,
    e.runtime_seconds, e.input_tokens, e.output_tokens, e.token_cost_usd
FROM runs r
JOIN tasks t ON t.task_id = r.task_id
LEFT JOIN evaluations e ON e.run_id = r.run_id
LEFT JOIN goal_stats g ON g.run_id = r.run_id
LEFT JOIN decision_stats d ON d.run_id = r.run_id
LEFT JOIN conflict_stats c ON c.run_id = r.run_id;

-- A descriptive grouped comparison; formal causal analysis should match runs by
-- task, trial_key, seed, and agent configuration rather than treating means as pairs.
CREATE VIEW v_task_comparison AS
SELECT
    task_id, task_key, conflict_type, arbiter_condition,
    count(*) AS evaluated_runs,
    count(outcome_correct) AS runs_with_outcome_label,
    avg(outcome_correct::integer) AS outcome_success_rate,
    avg(goal_completion_rate) AS avg_goal_completion_rate,
    avg(runtime_seconds) AS avg_runtime_seconds,
    avg(token_cost_usd) AS avg_token_cost_usd,
    sum(interventions) AS total_interventions,
    sum(num_conflicts) AS total_conflicts
FROM v_run_summary
WHERE status = 'completed' AND goals_recorded > 0
GROUP BY task_id, task_key, conflict_type, arbiter_condition;
