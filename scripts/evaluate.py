"""Evaluate ingested ALIGN runs using observed test and conflict events.

No automatic LLM judge or fake ground truth. Rule labels remain provisional.
Run against disposable/local PostgreSQL by default; writes require --apply.
"""
from __future__ import annotations
import argparse
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path

from process_runs import check
from run_artifacts import ACTIONS

GOAL_RESULTS = {"passed", "failed", "not_evaluated"}
CONFLICT_TYPES = {"execution_conflict", "design_conflict", "adversarial"}
CONFLICT_STATUSES = {"open", "resolved", "escalated", "unresolved"}


def compute_outcome(expected, status, goal_results, shared, critical, escalated):
    """None means there is not enough observed evidence, not a failed test."""
    if status in {"pending", "running"}:
        return None
    if expected == "escalate":
        return bool(escalated)
    if shared is None or critical is None or not goal_results or "not_evaluated" in goal_results:
        return None
    if expected == "complete_all_goals":
        return status == "completed" and shared is True and critical is True and all(
            value == "passed" for value in goal_results)
    if expected == "pick_one_goal":
        return status == "completed" and shared is True and critical is True and (
            sum(value == "passed" for value in goal_results) == 1)
    raise ValueError("Unknown expected outcome")


def parse_metrics(events):
    """Metrics must originate in the final run_end event; missing means unknown."""
    final = events[-1]
    if final["event_type"] != "run_end":
        return None, None, None, None, None
    data = final["payload"]
    shared = data.get("shared_goal_satisfied")
    critical = data.get("critical_constraints_passed")
    for value, label in ((shared, "shared_goal_satisfied"), (critical, "critical_constraints_passed")):
        check(value is None or type(value) is bool, f"{label} must be boolean")
    inputs, outputs = data.get("input_tokens"), data.get("output_tokens")
    for value, label in ((inputs, "input_tokens"), (outputs, "output_tokens")):
        check(value is None or (type(value) is int and value >= 0), f"{label} must be nonnegative integer")
    raw_cost = data.get("token_cost_usd")
    try:
        cost = Decimal(str(raw_cost)) if raw_cost is not None else None
    except InvalidOperation as exc:
        raise ValueError("Invalid token cost") from exc
    check(cost is None or (cost.is_finite() and cost >= 0), "Invalid token cost")
    return shared, critical, inputs, outputs, cost


def goal_evidence(events, goal_keys):
    results = {}
    for event in events:
        if event["event_type"] != "test_result":
            continue
        payload = event["payload"]
        key, result = payload.get("goal_key"), payload.get("result")
        check(key in goal_keys, f"test_result references unknown goal_key {key}")
        check(result in GOAL_RESULTS, f"Invalid test result for {key}")
        check(key not in results, f"Duplicate test_result for {key}")
        evidence = {k: v for k, v in payload.items() if k in {"test_name", "evidence_uri", "exit_code"}}
        check(not ("exit_code" in evidence and type(evidence["exit_code"]) is not int),
              "Invalid exit_code in test evidence")
        results[key] = (result, evidence)
    return results


def conflict_evidence(events):
    """Validate declared conflict episodes without inferring unobserved conflicts."""
    onsets = {}
    closing = {}
    for event in events:
        seq = event["sequence_number"]
        payload = event["payload"]
        if event["event_type"] == "conflict_onset":
            check(payload.get("conflict_type") in CONFLICT_TYPES, "Invalid conflict_type")
            check(isinstance(payload.get("description"), str) and bool(payload["description"].strip()),
                  "Missing conflict description")
            onsets[seq] = event
        elif event["event_type"] == "conflict_resolution":
            start = payload.get("onset_sequence")
            check(type(start) is int and start in onsets, "Invalid conflict_resolution onset_sequence")
            check(start not in closing, "Duplicate conflict resolution")
            check(payload.get("status") in {"resolved", "escalated", "unresolved"},
                  "Invalid conflict resolution status")
            closing[start] = event
    return onsets, closing


def load_manual_labels(path):
    """Only manually-reviewed labels from an explicit reviewer-named file are reference."""
    if path is None:
        return {}
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    check(isinstance(content, dict) and isinstance(content.get("labels"), list), "Expected labels list")
    labels = {}
    for item in content["labels"]:
        check(isinstance(item, dict), "Label must be object")
        check(set(item) == {"event_uuid", "expected_actions", "reviewer", "rationale"},
              "Human label requires event_uuid, expected_actions, reviewer and rationale")
        check(isinstance(item["reviewer"], str) and bool(item["reviewer"].strip()),
              "Human label needs named reviewer")
        check(isinstance(item["rationale"], str) and bool(item["rationale"].strip()),
              "Human label needs rationale")
        actions = item["expected_actions"]
        check(isinstance(actions, list) and bool(actions) and len(actions) == len(set(actions))
              and set(actions) <= ACTIONS, "Invalid expected_actions")
        key = item["event_uuid"]
        check(isinstance(key, str) and key not in labels, "Duplicate/invalid human label event_uuid")
        labels[key] = item
    return labels


def evaluate(conn, run_key, evaluator_version, human_labels=None):
    """Recompute provisional material only; never discard reviewed decision labels."""
    check(isinstance(evaluator_version, str) and bool(evaluator_version.strip()), "Missing evaluator_version")
    human_labels = human_labels or {}
    from psycopg.types.json import Jsonb
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("""SELECT r.run_id,r.task_id,r.status,r.arbiter_condition,r.started_at,
                                  r.finished_at,t.expected_outcome
                           FROM runs r JOIN tasks t USING(task_id)
                           WHERE r.trial_key=%s FOR UPDATE OF r""", (run_key,))
            matches = cur.fetchall()
            check(len(matches) == 1, "run_key must resolve to exactly one ingested run")
            run_id, task_id, status, condition, started, finished, expected = matches[0]
            cur.execute("""SELECT goal_id,goal_key FROM task_goals WHERE task_id=%s ORDER BY goal_key""", (task_id,))
            goals = dict((key, ident) for ident, key in cur.fetchall())
            cur.execute("""SELECT event_id,sequence_number,actor_type,actor_id,event_type,
                                  occurred_at,payload
                           FROM events WHERE run_id=%s ORDER BY sequence_number""", (run_id,))
            sql_events = cur.fetchall()
            check(sql_events and sql_events[0][4] == "run_start", "Missing run_start")
            events = [{"event_id": ident, "sequence_number": seq, "actor_type": actor,
                       "actor_id": actor_id, "event_type": etype, "occurred_at": when, "payload": payload}
                      for ident, seq, actor, actor_id, etype, when, payload in sql_events]
            by_sequence = {e["sequence_number"]: e for e in events}
            check(len(by_sequence) == len(events), "Duplicate event sequence")
            outcomes = goal_evidence(events, set(goals))
            shared, critical, inputs, outputs, cost = parse_metrics(events)
            onsets, closings = conflict_evidence(events)
            escalated = any(e["event_type"] == "arbiter_decision" and
                            e["payload"].get("action") == "escalate" for e in events) or any(
                                e["payload"]["status"] == "escalated" for e in closings.values())
            observed_goals = [outcomes.get(key, ("not_evaluated", {}))[0] for key in goals]
            correct = compute_outcome(expected, status, observed_goals, shared, critical, escalated)
            runtime = Decimal(str((finished - started).total_seconds())) if started and finished else None
            if runtime is not None:
                check(runtime >= 0, "Negative runtime")
            cur.execute("""INSERT INTO evaluations
                       (run_id,evaluator_version,outcome_correct,shared_goal_satisfied,
                        critical_constraints_passed,runtime_seconds,input_tokens,
                        output_tokens,token_cost_usd,evaluation_notes)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (run_id) DO UPDATE SET
                         evaluator_version=EXCLUDED.evaluator_version,
                         outcome_correct=EXCLUDED.outcome_correct,
                         shared_goal_satisfied=EXCLUDED.shared_goal_satisfied,
                         critical_constraints_passed=EXCLUDED.critical_constraints_passed,
                         runtime_seconds=EXCLUDED.runtime_seconds,
                         input_tokens=EXCLUDED.input_tokens,
                         output_tokens=EXCLUDED.output_tokens,
                         token_cost_usd=EXCLUDED.token_cost_usd,
                         evaluation_notes=EXCLUDED.evaluation_notes,
                         evaluated_at=now()""",
                        (run_id, evaluator_version, correct, shared, critical, runtime,
                         inputs, outputs, cost,
                         "Observed run evidence only; untested goals and unreviewed labels are not ground truth."))
            for key, ident in goals.items():
                outcome, proof = outcomes.get(key, ("not_evaluated", {}))
                cur.execute("""INSERT INTO goal_results (run_id,task_id,goal_id,result,evidence)
                           VALUES (%s,%s,%s,%s,%s)
                           ON CONFLICT (run_id,goal_id) DO UPDATE SET
                           result=EXCLUDED.result,evidence=EXCLUDED.evidence""",
                            (run_id, task_id, ident, outcome, Jsonb(proof)))

            # Replace only provisional system-generated conflict episodes.
            # Reviewed episodes and their references survive re-evaluation.
            cur.execute("""SELECT conflict_id,onset_event_id FROM conflict_episodes
                           WHERE run_id=%s AND review_status='reviewed'""", (run_id,))
            reviewed = {onset: ident for ident, onset in cur.fetchall()}
            cur.execute("""UPDATE decision_log SET conflict_id=NULL
                           WHERE run_id=%s AND conflict_id IN
                             (SELECT conflict_id FROM conflict_episodes
                              WHERE run_id=%s AND review_status='provisional')""", (run_id, run_id))
            cur.execute("DELETE FROM conflict_episodes WHERE run_id=%s AND review_status='provisional'",
                        (run_id,))
            episode_ids = dict(reviewed)
            for seq, onset in onsets.items():
                event_id = onset["event_id"]
                if event_id in reviewed:
                    continue
                closing = closings.get(seq)
                resolved_id = closing["event_id"] if closing else None
                state = closing["payload"]["status"] if closing else "open"
                data = onset["payload"]
                cur.execute("""INSERT INTO conflict_episodes
                               (run_id,conflict_type,onset_event_id,resolution_event_id,
                                status,label_source,review_status,description)
                               VALUES (%s,%s,%s,%s,%s,'synthetic','provisional',%s)
                               RETURNING conflict_id""",
                            (run_id, data["conflict_type"], event_id, resolved_id, state,
                             data["description"]))
                episode_ids[event_id] = cur.fetchone()[0]

            # Link later arbiter decisions to the latest declared episode still open;
            # do not overwrite a previously reviewed conflict attribution.
            ordered_onsets = sorted(onsets)
            for event in events:
                if event["event_type"] != "arbiter_decision":
                    continue
                candidates = [seq for seq in ordered_onsets if seq < event["sequence_number"] and
                              (seq not in closings or closings[seq]["sequence_number"] > event["sequence_number"])]
                if not candidates:
                    continue
                onset_id = by_sequence[candidates[-1]]["event_id"]
                cur.execute("""UPDATE decision_log SET conflict_id=%s
                               WHERE event_id=%s AND (conflict_id IS NULL OR
                                conflict_id NOT IN
                                 (SELECT conflict_id FROM conflict_episodes
                                  WHERE review_status='reviewed'))""",
                            (episode_ids[onset_id], event["event_id"]))

            # Preserve all reviewed labels. Rebuild only provisional rule suggestions.
            proposals = [e for e in events if e["event_type"] == "proposed_action"]
            wanted_uuids = set(human_labels)
            available_uuids = {e["payload"]["event_uuid"] for e in proposals}
            check(wanted_uuids <= available_uuids, "Human labels must reference proposed_action event UUIDs")
            cur.execute("""DELETE FROM decision_labels WHERE checkpoint_event_id IN
                           (SELECT event_id FROM events WHERE run_id=%s)
                           AND review_status='provisional' AND label_source='rules'""", (run_id,))
            for proposal in proposals:
                event_id = proposal["event_id"]
                uuid = proposal["payload"]["event_uuid"]
                cur.execute("""SELECT expected_actions,rationale,label_source
                               FROM decision_labels WHERE checkpoint_event_id=%s
                               AND is_reference""", (event_id,))
                reference = cur.fetchone()
                manual = human_labels.get(uuid)
                if manual:
                    expected_actions = manual["expected_actions"]
                    rationale = f"Reviewed by {manual['reviewer']}: {manual['rationale']}"
                    if reference:
                        check(set(reference[0]) == set(expected_actions),
                              "Human reference label conflicts with existing reviewed reference")
                    else:
                        cur.execute("""INSERT INTO decision_labels
                                     (checkpoint_event_id,expected_actions,label_source,
                                      review_status,rationale,is_reference)
                                     VALUES (%s,%s,'human','reviewed',%s,true)""",
                                    (event_id, expected_actions, rationale))
                elif not reference:
                    cur.execute("""SELECT violated_rules FROM action_attempts
                                   WHERE proposal_event_id=%s AND run_id=%s""",
                                (event_id, run_id))
                    attempt = cur.fetchone()
                    if attempt and attempt[0]:
                        cur.execute("""INSERT INTO decision_labels
                                     (checkpoint_event_id,expected_actions,label_source,
                                      review_status,rationale,is_reference)
                                     VALUES (%s,%s,'rules','provisional',%s,false)""",
                                    (event_id, ["block"], "Rule hits: " + ", ".join(attempt[0])))
    return {"run_id": run_id, "goals": len(goals), "goals_evaluated": len(outcomes),
            "conflicts": len(onsets), "outcome_correct": correct,
            "human_references_supplied": len(human_labels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-key", required=True, help="Existing run trial_key")
    parser.add_argument("--evaluator-version", default="rules-v1", help="Versioned evaluator identity")
    parser.add_argument("--human-labels", type=Path, help="Explicit human-reviewed checkpoint labels JSON")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    labels = load_manual_labels(args.human_labels)
    if not args.apply:
        print(json.dumps({"run_key": args.run_key, "human_labels": len(labels),
                          "status": "DRY RUN: no database connection or writes"}))
        return
    dsn = os.environ.get("ALIGN_DATABASE_URL")
    if not dsn:
        parser.error("--apply requires ALIGN_DATABASE_URL")
    import psycopg
    with psycopg.connect(dsn, autocommit=True, connect_timeout=10) as conn:
        result = evaluate(conn, args.run_key, args.evaluator_version, labels)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
