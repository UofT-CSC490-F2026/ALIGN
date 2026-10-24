"""Immutable ALIGN run bundles: validate/stage locally, optionally upload to S3.

This module NEVER executes agents, calls an LLM, or connects to PostgreSQL.
S3 writes require --apply and an expected AWS account ID.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

CONDITIONS = {"none", "monitor", "rule_based", "llm"}
ACTORS = {"agent", "arbiter", "system"}
END_STATUSES = {"completed", "failed", "cancelled"}
ACTIONS = {"continue", "query", "block", "mediate", "restructure", "escalate"}


def canonical(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def check(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_uuid(value):
    check(isinstance(value, str), "run_key/event_uuid must be a UUID string")
    try:
        check(str(uuid.UUID(value)) == value, "UUID must be lowercase/canonical")
    except (ValueError, AttributeError) as exc:
        raise ValueError("Invalid canonical UUID") from exc
    return value


def utc_timestamp(value):
    check(isinstance(value, str) and value.endswith("Z"), "Timestamp must be UTC ISO-8601 with Z")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid timestamp") from exc
    check(parsed.utcoffset() == dt.timedelta(0), "Timestamp must be UTC")
    return parsed


def validate_events(events):
    """Strict event contract. No dropped lines or gaps; a crashed run may lack run_end."""
    check(isinstance(events, list) and bool(events), "Nonempty event list required")
    seen = set()
    run_key = None
    for number, row in enumerate(events, 1):
        check(isinstance(row, dict), f"Event {number}: expected object")
        check(set(row) <= {"schema_version", "event_uuid", "run_key", "sequence_number",
                          "actor_type", "actor_id", "event_type", "occurred_at",
                          "observed_at", "trigger_sequence", "payload", "artifact_uri"},
              f"Event {number}: unknown fields")
        check(row.get("schema_version") == 1, "Invalid event schema_version")
        canonical_uuid(row.get("event_uuid"))
        canonical_uuid(row.get("run_key"))
        if run_key is None:
            run_key = row["run_key"]
        check(row["run_key"] == run_key, "Mixed run keys")
        check(row["event_uuid"] not in seen, "Duplicate event UUID")
        seen.add(row["event_uuid"])
        check(type(row.get("sequence_number")) is int and row["sequence_number"] == number,
              "Sequence numbers must be contiguous in file order starting at 1")
        check(row.get("actor_type") in ACTORS, "Invalid actor_type")
        check(row.get("actor_id") is None or
              (isinstance(row["actor_id"], str) and bool(row["actor_id"].strip())),
              "Invalid actor_id")
        check(isinstance(row.get("event_type"), str) and bool(row["event_type"].strip()), "Missing event_type")
        utc_timestamp(row.get("occurred_at"))
        if row.get("observed_at") is not None:
            utc_timestamp(row["observed_at"])
        trigger = row.get("trigger_sequence")
        check(trigger is None or (type(trigger) is int and 0 < trigger < number),
              "trigger_sequence must reference an earlier event")
        check(isinstance(row.get("payload"), dict), "payload must be an object")
        check(row.get("artifact_uri") is None or
              (isinstance(row["artifact_uri"], str) and row["artifact_uri"].startswith("s3://")),
              "artifact_uri must be an S3 URI")
        if row["event_type"] == "arbiter_decision":
            check(row["actor_type"] == "arbiter" and trigger is not None, "Decision must be by arbiter with trigger")
            check(row["payload"].get("action") in ACTIONS, "Invalid arbiter action")
        if row["event_type"] in {"action_result"}:
            check(trigger is not None, "Action result requires trigger")
    check(events[0]["event_type"] == "run_start" and events[0]["actor_type"] == "system",
          "First event must be system run_start")
    start = events[0]["payload"]
    check(isinstance(start.get("task_key"), str) and bool(start["task_key"].strip()), "run_start needs task_key")
    check(start.get("arbiter_condition") in CONDITIONS, "Invalid arbiter_condition")
    for field in ("agent_config", "arbiter_config"):
        check(isinstance(start.get(field, {}), dict), f"{field} must be an object")
    check(start.get("seed") is None or type(start["seed"]) is int, "seed must be an integer")
    endings = [e for e in events if e["event_type"] == "run_end"]
    check(len(endings) <= 1, "Duplicate run_end")
    if endings:
        check(events[-1] is endings[0] and endings[0]["actor_type"] == "system",
              "run_end must be the final system event")
        check(endings[0]["payload"].get("status") in END_STATUSES, "Invalid run_end status")
    return {"run_key": run_key, "task_key": start["task_key"],
            "arbiter_condition": start["arbiter_condition"],
            "agent_config": start.get("agent_config", {}),
            "arbiter_config": start.get("arbiter_config", {}),
            "seed": start.get("seed"), "complete": bool(endings), "event_count": len(events)}


def load_events(data):
    check(data.endswith(b"\n"), "events.jsonl must end with newline")
    events = []
    for line_no, line in enumerate(data.decode("utf-8").splitlines(), 1):
        check(bool(line.strip()), f"Empty event line {line_no}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {line_no}") from exc
        events.append(row)
    header = validate_events(events)
    return events, header


def verify_bundle(directory):
    directory = Path(directory)
    check((directory / "manifest.json").is_file() and (directory / "events.jsonl").is_file(),
          "Bundle needs manifest.json and events.jsonl")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    check(manifest.get("schema_version") == 1, "Invalid bundle schema")
    raw = (directory / "events.jsonl").read_bytes()
    _, header = load_events(raw)
    check(manifest.get("events_sha256") == hashlib.sha256(raw).hexdigest(), "Bundle checksum mismatch")
    check(manifest.get("events_size_bytes") == len(raw), "Bundle size mismatch")
    check(all(manifest.get(k) == header[k] for k in header), "Bundle event metadata mismatch")
    return manifest


def stage(events_path, outdir):
    events_path, outdir = Path(events_path), Path(outdir)
    raw = events_path.read_bytes()
    _, header = load_events(raw)
    manifest = {**header, "schema_version": 1, "events_sha256": hashlib.sha256(raw).hexdigest(),
                "events_size_bytes": len(raw)}
    target_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if outdir.exists():
        check((outdir / "events.jsonl").is_file() and
              (outdir / "events.jsonl").read_bytes() == raw and
              (outdir / "manifest.json").is_file() and
              (outdir / "manifest.json").read_bytes() == target_manifest,
              "Existing bundle differs; use a new directory")
        return manifest
    outdir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".align-run-", dir=outdir.parent) as temp:
        p = Path(temp)
        (p / "events.jsonl").write_bytes(raw)
        (p / "manifest.json").write_bytes(target_manifest)
        p.rename(outdir)
    return manifest


def upload(directory, bucket, expected_account):
    """S3 completion marker is uploaded LAST; conflicting data is never overwritten."""
    import boto3
    from botocore.exceptions import ClientError
    manifest = verify_bundle(directory)
    canonical_uuid(manifest["run_key"])
    session = boto3.Session()
    account = session.client("sts").get_caller_identity()["Account"]
    check(account == expected_account, "Wrong AWS account; refusing upload")
    s3 = session.client("s3")
    prefix = f"runs/{manifest['run_key']}/"
    files = ("events.jsonl", "manifest.json")
    for filename in files:
        key = prefix + filename
        body = (Path(directory) / filename).read_bytes()
        try:
            prior = s3.get_object(Bucket=bucket, Key=key, ExpectedBucketOwner=expected_account)
            with prior["Body"] as stream:
                actual = stream.read()
            check(actual == body, f"Conflicting S3 object: {key}")
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
                raise
            s3.put_object(Bucket=bucket, Key=key, Body=body,
                          ContentType="application/json" if filename.endswith(".json") else "application/x-ndjson",
                          IfNoneMatch="*", ExpectedBucketOwner=expected_account)
        verified = s3.get_object(Bucket=bucket, Key=key, ExpectedBucketOwner=expected_account)
        with verified["Body"] as stream:
            check(stream.read() == body, f"Upload readback mismatch: {key}")
    return f"s3://{bucket}/{prefix}"


def download(bucket, run_key, directory, expected_account):
    """Retrieve an immutable bundle; verify checksums before returning it."""
    import boto3
    canonical_uuid(run_key)
    session = boto3.Session()
    check(session.client("sts").get_caller_identity()["Account"] == expected_account, "Wrong AWS account")
    s3 = session.client("s3")
    directory = Path(directory)
    check(not directory.exists(), "Download target already exists")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".align-dl-", dir=directory.parent) as temporary:
        p = Path(temporary)
        for filename in ("events.jsonl", "manifest.json"):
            response = s3.get_object(Bucket=bucket, Key=f"runs/{run_key}/{filename}",
                                     ExpectedBucketOwner=expected_account)
            with response["Body"] as body:
                (p / filename).write_bytes(body.read())
        manifest = verify_bundle(p)
        check(manifest["run_key"] == run_key, "Downloaded bundle run key mismatch")
        p.rename(directory)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("stage")
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("--run-dir", type=Path, required=True)
    p = sub.add_parser("upload")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--expected-account", required=True)
    p.add_argument("--apply", action="store_true")
    p = sub.add_parser("download")
    p.add_argument("--bucket", required=True)
    p.add_argument("--run-key", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-account", required=True)
    args = parser.parse_args()
    if args.mode == "stage":
        result = stage(args.events, args.output_dir)
    elif args.mode == "verify":
        result = verify_bundle(args.run_dir)
    elif args.mode == "upload":
        if not args.apply:
            parser.error("S3 upload requires explicit --apply")
        result = upload(args.run_dir, args.bucket, args.expected_account)
    else:
        result = download(args.bucket, args.run_key, args.output_dir, args.expected_account)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
