"""Single-process event recorder compatible with the existing A2 v1 contract.

Only sanitized payloads may be passed by callers. This is not a redaction layer.
A future concurrent adapter must share one recorder and serialize action decisions.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock
from uuid import uuid4

from scripts.run_artifacts import validate_events, stage


class EventWriter:
    def __init__(self, *, task_key, arbiter_condition, agent_config,
                 arbiter_config, seed=None, run_key=None):
        self.run_key = run_key if run_key is not None else str(uuid4())
        self._events = []
        self._lock = RLock()
        self.emit("system", "run_start", {
            "task_key": task_key, "arbiter_condition": arbiter_condition,
            "agent_config": agent_config, "arbiter_config": arbiter_config,
            "seed": seed,
        })

    def emit(self, actor_type, event_type, payload, *, actor_id=None,
             trigger_sequence=None):
        """Append a JSON snapshot and return its sequence number."""
        with self._lock:
            if self._events and self._events[-1]["event_type"] == "run_end":
                raise ValueError("Run already ended")
            if event_type == "run_start" and self._events:
                raise ValueError("Duplicate run_start")
            seq = len(self._events) + 1
            event = {
                "schema_version": 1, "event_uuid": str(uuid4()),
                "run_key": self.run_key, "sequence_number": seq,
                "occurred_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "actor_type": actor_type, "actor_id": actor_id,
                "event_type": event_type, "payload": payload,
            }
            if trigger_sequence is not None:
                event["trigger_sequence"] = trigger_sequence
            snapshot = json.loads(json.dumps(event, allow_nan=False))
            # A2 validates shape; add the proposal association required by its SQL importer.
            if event_type in {"arbiter_decision", "action_result"}:
                if (type(trigger_sequence) is not int or not 0 < trigger_sequence < seq
                        or self._events[trigger_sequence - 1]["event_type"] != "proposed_action"):
                    raise ValueError("Trigger must refer to an earlier proposed_action")
                if any(e["event_type"] == event_type and e.get("trigger_sequence") == trigger_sequence
                       for e in self._events):
                    raise ValueError("Duplicate decision/result for proposal")
                if event_type == "action_result" and payload.get("outcome") not in {
                    "dispatched", "cancelled", "unknown"
                }:
                    raise ValueError("Invalid action_result outcome")
            # This simple core revalidates the prefix; profile before scaling.
            validate_events([*self._events, snapshot])
            self._events.append(snapshot)
            return seq

    def snapshot(self):
        """Return an independent copy; callers cannot mutate recorded history."""
        with self._lock:
            return json.loads(json.dumps(self._events))

    def finish(self, status, **metrics):
        return self.emit("system", "run_end", {**metrics, "status": status})

    def write_bundle(self, events_path: Path, bundle_dir: Path):
        """Write a new local JSONL file and stage it using A2.

        Incomplete traces are supported by A2. No AWS or database writes.
        Existing event files are never overwritten.
        """
        events_path = Path(events_path)
        bundle_dir = Path(bundle_dir)
        with self._lock:
            validate_events(self._events)
            raw = "".join(json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n"
                          for e in self._events)
            events_path.parent.mkdir(parents=True, exist_ok=True)
            with events_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(raw)
            return stage(events_path, bundle_dir)
