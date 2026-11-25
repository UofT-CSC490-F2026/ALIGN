"""Pre-dispatch arbiter gate; monitor recommendations never enter enforced decision_log."""
from copy import deepcopy
import json
from time import monotonic

ACTIONS={"continue","query","block","mediate","restructure","escalate"}
ENFORCED={"continue","query","block","mediate","escalate"}

def validate_decision(d, actors):
    if not isinstance(d,dict) or set(d)!={"action","reason","affected_agents","evidence"}:
        raise ValueError("Invalid arbiter decision schema")
    if (not isinstance(d["action"],str) or d["action"] not in ACTIONS or not isinstance(d["reason"],str) or not d["reason"].strip()
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

    def evaluate(self, obs):
        """No writer or environment side effects; callers can cancel a late response."""
        if self.condition=="none":
            return "continue",None,0,None
        started=monotonic()
        try:
            decision=validate_decision(self.arbiter.decide(deepcopy(obs)),obs["agent_goals"])
            action=decision["action"]
            if self.condition!="monitor" and action not in ENFORCED:
                raise ValueError("Unsupported enforced action")
            return action,decision,int((monotonic()-started)*1000),None
        except Exception as exc:
            return "continue" if self.condition=="monitor" else None,None,int((monotonic()-started)*1000),exc

    def record(self,writer,obs,evaluation):
        action,decision,latency,error=evaluation
        seq=obs["proposal_sequence"]
        if error is not None:
            writer.emit("system","arbiter_error",{"error_type":type(error).__name__,
                        "decision_latency_ms":latency},trigger_sequence=seq)
            if self.condition!="monitor":
                writer.emit("system","action_result",{"outcome":"cancelled"},trigger_sequence=seq)
                raise error
        elif decision is not None:
            event="arbiter_advice" if self.condition=="monitor" else "arbiter_decision"
            writer.emit("arbiter",event,{**decision,"decision_latency_ms":latency},
                        actor_id="arbiter",trigger_sequence=seq)
        return "continue" if self.condition=="monitor" else action

    @staticmethod
    def feedback(action,decision):
        instruction = ("Inspect the missing facts, ask the peer if necessary, and submit a new proposal."
                       if action=="query" else
                       "Revise the proposed action to reconcile competing objectives or changes."
                       if action=="mediate" else "Revise the proposed action.")
        return {"output":f"Arbiter {action}: {decision['reason']}. {instruction}","exit_code":1}

    def __call__(self,writer,obs,execute):
        evaluation=self.evaluate(obs)
        action=self.record(writer,obs,evaluation)
        seq=obs["proposal_sequence"]
        if action=="continue":
            try:
                result=execute()
            except Exception as exc:
                writer.emit("system","action_result",{"outcome":"unknown","error_type":type(exc).__name__},trigger_sequence=seq)
                raise
            writer.emit("system","action_result",{"outcome":"dispatched",**result},trigger_sequence=seq)
            return result,False
        result=self.feedback(action,evaluation[1])
        writer.emit("system","action_result",{"outcome":"cancelled",**result},trigger_sequence=seq)
        return result,action=="escalate"

class ConstantArbiter:
    def __init__(self,action="continue",reason="Deterministic fixture"):
        self.action,self.reason=action,reason
    def decide(self,obs):
        return {"action":self.action,"reason":self.reason,"affected_agents":[obs["actor_id"]],"evidence":{}}
