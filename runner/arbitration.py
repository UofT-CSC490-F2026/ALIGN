"""Pre-dispatch arbiter gate; monitor recommendations never enter enforced decision_log."""
from copy import deepcopy
import json
from time import monotonic

ACTIONS={"continue","query","block","mediate","restructure","escalate"}
ENFORCED={"continue","block","mediate","escalate"}

def validate_decision(d, actors):
    if not isinstance(d,dict) or set(d)!={"action","reason","affected_agents","evidence"}:
        raise ValueError("Invalid arbiter decision schema")
    if (not isinstance(d["action"],str) or d["action"] not in ACTIONS or not isinstance(d["reason"],str)
        or not isinstance(d["affected_agents"],list)
        or not all(isinstance(a,str) and a in actors for a in d["affected_agents"])
        or not isinstance(d["evidence"],dict)):
        raise ValueError("Invalid arbiter decision values")
    json.dumps(d, allow_nan=False, ensure_ascii=False).encode("utf-8")
    return deepcopy(d)

class ArbitrationGate:
    def __init__(self,condition,arbiter=None):
        if condition not in {"none","monitor","rule_based","llm"}:
            raise ValueError("Invalid condition")
        if condition!="none" and arbiter is None:
            raise ValueError("Arbiter required")
        self.condition,self.arbiter=condition,arbiter

    def __call__(self,writer,obs,execute):
        seq=obs["proposal_sequence"]
        action="continue"
        decision=None
        if self.condition!="none":
            started=monotonic()
            try:
                decision=validate_decision(self.arbiter.decide(deepcopy(obs)),obs["agent_goals"])
                action=decision["action"]
                if self.condition!="monitor" and action not in ENFORCED:
                    raise ValueError("Unsupported enforced action")
            except Exception as exc:
                writer.emit("system","arbiter_error",{"error_type":type(exc).__name__,
                            "decision_latency_ms":int((monotonic()-started)*1000)},trigger_sequence=seq)
                if self.condition!="monitor":
                    writer.emit("system","action_result",{"outcome":"cancelled"},trigger_sequence=seq)
                    raise
                # Advisory failures cannot affect baseline execution.
                action="continue"
            else:
                event="arbiter_advice" if self.condition=="monitor" else "arbiter_decision"
                writer.emit("arbiter",event,{**decision,"decision_latency_ms":int((monotonic()-started)*1000)},
                            actor_id="arbiter",trigger_sequence=seq)
        if self.condition=="monitor" or action=="continue":
            try:
                result=execute()
            except Exception as exc:
                writer.emit("system","action_result",{"outcome":"unknown","error_type":type(exc).__name__},trigger_sequence=seq)
                raise
            writer.emit("system","action_result",{"outcome":"dispatched",**result},trigger_sequence=seq)
            return result,False
        result={"output":f"Arbiter {action}: {decision['reason']}. Revise the proposed action.",
                "exit_code":1}
        writer.emit("system","action_result",{"outcome":"cancelled",**result},trigger_sequence=seq)
        return result,action=="escalate"

class ConstantArbiter:
    def __init__(self,action="continue",reason="Deterministic fixture"):
        self.action,self.reason=action,reason
    def decide(self,obs):
        return {"action":self.action,"reason":self.reason,"affected_agents":[obs["actor_id"]],"evidence":{}}
