-- Additive, non-destructive post-v3 enhancement.
-- Helps monitoring queries filter one run's observations by event type.
-- Demonstrates an upgrade applied to an already-populated database.
CREATE INDEX idx_events_run_event_type ON events (run_id, event_type);
COMMENT ON COLUMN events.observed_at IS
'When the passive monitor first received the event, which may differ from occurred_at.';
