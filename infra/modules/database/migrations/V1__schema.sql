-- Initial PostgreSQL 16 schema (10 tables); use Flyway for later schema changes.
-- 1. Benchmark scenarios.
CREATE TABLE tasks (
    task_id          INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    task_key         TEXT NOT NULL UNIQUE,
    dataset_source   TEXT NOT NULL,
    external_task_id TEXT,
    version          INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
    repo_url         TEXT NOT NULL,
    base_commit      TEXT NOT NULL,
    shared_goal      TEXT NOT NULL,
    conflict_type    TEXT NOT NULL CHECK (
        conflict_type IN ('none', 'execution_conflict', 'design_conflict', 'adversarial')
    ),
    expected_outcome TEXT NOT NULL CHECK (
        expected_outcome IN ('complete_all_goals', 'pick_one_goal', 'escalate')
    ),
    constraints      JSONB NOT NULL DEFAULT '{}'::jsonb
                     CHECK (jsonb_typeof(constraints) = 'object'),
    setup_spec       JSONB NOT NULL DEFAULT '{}'::jsonb
                     CHECK (jsonb_typeof(setup_spec) = 'object'),
    metadata         JSONB NOT NULL DEFAULT '{}'::jsonb
                     CHECK (jsonb_typeof(metadata) = 'object'),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX uq_tasks_external_source_version
    ON tasks (dataset_source, external_task_id, version)
    WHERE external_task_id IS NOT NULL;

-- 2. Individually testable agent or shared goals.
CREATE TABLE task_goals (
    goal_id          INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    task_id          INTEGER NOT NULL REFERENCES tasks(task_id) ON DELETE RESTRICT,
    goal_key         TEXT NOT NULL,
    owner_type       TEXT NOT NULL CHECK (owner_type IN ('agent', 'shared')),
    owner_agent_id   TEXT,
    description      TEXT NOT NULL,
    success_criteria JSONB NOT NULL DEFAULT '{}'::jsonb
                     CHECK (jsonb_typeof(success_criteria) = 'object'),
    is_critical      BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (task_id, goal_key),
    UNIQUE (goal_id, task_id),
    CHECK (
        (owner_type = 'agent' AND owner_agent_id IS NOT NULL AND length(trim(owner_agent_id)) > 0)
        OR (owner_type = 'shared' AND owner_agent_id IS NULL)
    )
);

-- 3. Repeated executions of task scenarios.
CREATE TABLE runs (
    run_id            INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    task_id           INTEGER NOT NULL REFERENCES tasks(task_id) ON DELETE RESTRICT,
    trial_key         TEXT,
    arbiter_condition TEXT NOT NULL CHECK (arbiter_condition IN ('none','monitor','rule_based','llm')),
    agent_config      JSONB NOT NULL DEFAULT '{}'::jsonb
                      CHECK (jsonb_typeof(agent_config) = 'object'),
    arbiter_config    JSONB NOT NULL DEFAULT '{}'::jsonb
                      CHECK (jsonb_typeof(arbiter_config) = 'object'),
    seed              INTEGER,
    status            TEXT NOT NULL DEFAULT 'pending'
                      CHECK (status IN ('pending','running','completed','failed','cancelled')),
    started_at        TIMESTAMPTZ,
    finished_at       TIMESTAMPTZ,
    artifact_uri      TEXT,
    failure_category   TEXT CHECK (failure_category IN ('task_failure','infrastructure_error','agent_error','timeout','other')),
    failure_reason     TEXT,
    UNIQUE (run_id, task_id),
    CHECK (finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at)
);
CREATE INDEX idx_runs_task_condition ON runs (task_id, arbiter_condition);
CREATE INDEX idx_runs_trial ON runs (task_id, trial_key, seed);

-- 4. Common event header for all meaningful observations and actions.
CREATE TABLE events (
    event_id        INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
    sequence_number INTEGER NOT NULL CHECK (sequence_number > 0),
    actor_type      TEXT NOT NULL CHECK (actor_type IN ('agent','arbiter','system')),
    actor_id        TEXT,
    event_type      TEXT NOT NULL,
    occurred_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    observed_at     TIMESTAMPTZ,
    payload         JSONB NOT NULL DEFAULT '{}'::jsonb
                    CHECK (jsonb_typeof(payload) = 'object'),
    artifact_uri    TEXT,
    UNIQUE (run_id, sequence_number),
    UNIQUE (event_id, run_id),
    CHECK (event_type <> 'arbiter_decision' OR actor_type = 'arbiter')
);

-- 5. Actual conflict episodes, including those ALIGN missed or had no chance to act on.
-- Each referenced boundary event is verified to belong to the episode's run.
CREATE TABLE conflict_episodes (
    conflict_id         INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id              INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
    conflict_type       TEXT NOT NULL CHECK (
        conflict_type IN ('execution_conflict','design_conflict','adversarial')
    ),
    onset_event_id      INTEGER,
    detected_event_id   INTEGER,
    resolution_event_id INTEGER,
    status              TEXT NOT NULL CHECK (
        status IN ('open','resolved','escalated','unresolved')
    ),
    label_source        TEXT NOT NULL CHECK (
        label_source IN ('human','rules','llm','synthetic')
    ),
    review_status       TEXT NOT NULL DEFAULT 'provisional'
                        CHECK (review_status IN ('provisional','reviewed')),
    description         TEXT NOT NULL,
    UNIQUE (conflict_id, run_id),
    FOREIGN KEY (onset_event_id, run_id) REFERENCES events(event_id, run_id),
    FOREIGN KEY (detected_event_id, run_id) REFERENCES events(event_id, run_id),
    FOREIGN KEY (resolution_event_id, run_id) REFERENCES events(event_id, run_id)
);
CREATE INDEX idx_conflict_episodes_run ON conflict_episodes (run_id);

-- 6. Arbiter decision details. All such decisions are also events, not additional timeline events.
-- run_id is intentionally repeated here solely to enforce same-run foreign keys.
CREATE TABLE decision_log (
    event_id            INTEGER PRIMARY KEY,
    run_id              INTEGER NOT NULL,
    trigger_event_id    INTEGER,
    conflict_id         INTEGER,
    arbiter_action      TEXT NOT NULL CHECK (
        arbiter_action IN ('continue','query','mediate','restructure','escalate','block')
    ),
    reason              TEXT,
    affected_agents     TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
    evidence            JSONB NOT NULL DEFAULT '{}'::jsonb
                        CHECK (jsonb_typeof(evidence) = 'object'),
    decision_latency_ms INTEGER CHECK (decision_latency_ms IS NULL OR decision_latency_ms >= 0),
    FOREIGN KEY (event_id, run_id) REFERENCES events(event_id, run_id),
    FOREIGN KEY (trigger_event_id, run_id) REFERENCES events(event_id, run_id),
    FOREIGN KEY (conflict_id, run_id) REFERENCES conflict_episodes(conflict_id, run_id)
);
CREATE INDEX idx_decisions_conflict ON decision_log (conflict_id) WHERE conflict_id IS NOT NULL;
CREATE INDEX idx_decisions_trigger ON decision_log (trigger_event_id) WHERE trigger_event_id IS NOT NULL;

-- Ensure the event referenced by decision_log is actually an arbiter decision.
CREATE FUNCTION enforce_decision_event_type() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM events e
        WHERE e.event_id = NEW.event_id AND e.run_id = NEW.run_id
          AND e.event_type = 'arbiter_decision' AND e.actor_type = 'arbiter'
    ) THEN
        RAISE EXCEPTION 'decision_log.event_id % must reference an arbiter_decision event in run %',
            NEW.event_id, NEW.run_id;
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER trg_decision_event_type
BEFORE INSERT OR UPDATE OF event_id, run_id ON decision_log
FOR EACH ROW EXECUTE FUNCTION enforce_decision_event_type();

-- An event's classification cannot be changed once its arbiter subtype exists.
CREATE FUNCTION protect_decision_event_type() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF EXISTS (SELECT 1 FROM decision_log d WHERE d.event_id = OLD.event_id)
       AND (NEW.event_type <> 'arbiter_decision' OR NEW.actor_type <> 'arbiter') THEN
        RAISE EXCEPTION 'Cannot change type/actor of event % with decision_log details', OLD.event_id;
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER trg_protect_decision_event_type
BEFORE UPDATE OF event_type, actor_type ON events
FOR EACH ROW EXECUTE FUNCTION protect_decision_event_type();

-- 7. One overall evaluation per run (v1 contract).
CREATE TABLE evaluations (
    run_id                      INTEGER PRIMARY KEY REFERENCES runs(run_id) ON DELETE RESTRICT,
    evaluator_version           TEXT NOT NULL,
    outcome_correct             BOOLEAN,
    shared_goal_satisfied       BOOLEAN,
    critical_constraints_passed BOOLEAN,
    runtime_seconds             NUMERIC CHECK (runtime_seconds IS NULL OR runtime_seconds >= 0),
    input_tokens                INTEGER CHECK (input_tokens IS NULL OR input_tokens >= 0),
    output_tokens               INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0),
    token_cost_usd              NUMERIC(12,6) CHECK (token_cost_usd IS NULL OR token_cost_usd >= 0),
    evaluation_notes            TEXT,
    evaluated_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 8. One result per task goal per run.
-- Extra task_id column enforces that the goal belongs to the task being executed.
CREATE TABLE goal_results (
    run_id  INTEGER NOT NULL REFERENCES evaluations(run_id) ON DELETE RESTRICT,
    task_id INTEGER NOT NULL,
    goal_id INTEGER NOT NULL,
    result  TEXT NOT NULL CHECK (result IN ('passed','failed','not_evaluated')),
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb
             CHECK (jsonb_typeof(evidence) = 'object'),
    notes   TEXT,
    PRIMARY KEY (run_id, goal_id),
    FOREIGN KEY (run_id, task_id) REFERENCES runs(run_id, task_id),
    FOREIGN KEY (goal_id, task_id) REFERENCES task_goals(goal_id, task_id)
);
CREATE INDEX idx_goal_results_goal ON goal_results (goal_id);

-- 9. Independent judgments about monitoring opportunities, even in no-arbiter runs.
CREATE TABLE decision_labels (
    label_id             INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    checkpoint_event_id  INTEGER NOT NULL REFERENCES events(event_id) ON DELETE RESTRICT,
    expected_actions     TEXT[] NOT NULL,
    label_source         TEXT NOT NULL CHECK (label_source IN ('human','rules','llm','synthetic')),
    review_status        TEXT NOT NULL DEFAULT 'provisional'
                         CHECK (review_status IN ('provisional','reviewed')),
    rationale            TEXT,
    is_reference       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (NOT is_reference OR review_status = 'reviewed'),
    CHECK (cardinality(expected_actions) > 0 AND expected_actions <@
           ARRAY['continue','query','mediate','restructure','escalate','block']::text[])
);
CREATE INDEX idx_decision_labels_checkpoint ON decision_labels (checkpoint_event_id);

-- Auditable agent action proposals and safety judgments.
CREATE TABLE action_attempts (
    attempt_id          INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id              INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
    proposal_event_id   INTEGER NOT NULL UNIQUE,
    decision_event_id   INTEGER,
    proposed_action     JSONB NOT NULL DEFAULT '{}'::jsonb
                        CHECK (jsonb_typeof(proposed_action) = 'object'),
    files_touched       TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
    is_unsafe           BOOLEAN,
    violated_rules      TEXT[] NOT NULL DEFAULT ARRAY[]::text[],
    label_source        TEXT CHECK (label_source IN ('human','rules','llm','synthetic')),
    review_status       TEXT NOT NULL DEFAULT 'provisional'
                        CHECK (review_status IN ('provisional','reviewed')),
    outcome             TEXT NOT NULL DEFAULT 'unknown'
                        CHECK (outcome IN ('blocked','dispatched','cancelled','unknown')),
    FOREIGN KEY (proposal_event_id, run_id) REFERENCES events(event_id, run_id),
    FOREIGN KEY (decision_event_id, run_id) REFERENCES events(event_id, run_id),
    CHECK (is_unsafe IS NULL OR label_source IS NOT NULL),
    CHECK (review_status <> 'reviewed' OR is_unsafe IS NOT NULL)
);
CREATE INDEX idx_action_attempts_run ON action_attempts(run_id);

CREATE UNIQUE INDEX uq_one_reference_label_per_checkpoint
    ON decision_labels(checkpoint_event_id) WHERE is_reference;

CREATE INDEX idx_decisions_checkpoint ON decision_log(run_id, trigger_event_id)
    WHERE trigger_event_id IS NOT NULL;

CREATE INDEX idx_events_run_event_type ON events (run_id, event_type);
COMMENT ON COLUMN events.observed_at IS
'When the passive monitor first received the event, which may differ from occurred_at.';
