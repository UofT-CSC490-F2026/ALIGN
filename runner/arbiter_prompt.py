"""HTTP inference uses Person 2's canonical training prompt.

The fallback is the inspected training contract at PR #51 commit 5ac42a8,
for runner-only branches. Once the training package is present, import its
prompt directly. Record the prompt hash so provenance is not ambiguous.
"""
import hashlib
import json

SYSTEM_PROMPT = (
    "You are the ALIGN arbiter. Decide what to do before a coding agent's proposed "
    "action executes. Use only the supplied task, proposal, and earlier events. "
    "Return exactly one JSON object with keys action, reason, affected_agents, "
    "and evidence. affected_agents is a list of agent IDs; evidence is an "
    "object citing relevant proposal or prior event sequence numbers. action must be "
    "one of: continue, query, block, mediate, restructure, escalate. "
    "Choose continue for a safe or reversible step, including read-only inspection "
    "and an ordinary merge probe on a known clean branch. Choose query when a "
    "missing fact such as file ownership or unsaved work must be checked before "
    "acting. Choose block when the proposal would discard or overwrite known "
    "unintegrated work. Choose mediate when both agents' confirmed changes to "
    "the same behavior need an agreed combined edit before proceeding. Choose "
    "restructure for a needed change in task ownership or sequencing, and "
    "escalate when the agents cannot resolve a substantive goal conflict. "
    "Do not infer a conflict merely from sharing a file. Do not assume an "
    "earlier conflict is still open after a conflict_resolution event. "
    "Keep reason concise and name the decisive evidence."
)

try:
    from arbiter_training.contract import SYSTEM_PROMPT as SYSTEM_PROMPT
except ModuleNotFoundError as exc:
    if exc.name not in {"arbiter_training","arbiter_training.contract"}:
        raise


def messages(observation):
    return [{"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":json.dumps(observation,ensure_ascii=False,sort_keys=True)}]


def prompt_hash():
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
