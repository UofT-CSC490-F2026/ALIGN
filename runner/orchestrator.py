"""Bounded alternating two-agent runner; same execution path for mocks and live agents."""
from copy import deepcopy
import json
from .arbitration import ArbitrationGate
from .task_loader import agent_view

def validate_proposal(p):
    if not isinstance(p,dict) or set(p)!={"action","files_touched"}:
        raise ValueError("Invalid proposal")
    a,paths=p["action"],p["files_touched"]
    if not isinstance(a,dict) or not isinstance(paths,list) or not all(isinstance(x,str) for x in paths):
        raise ValueError("Invalid action/paths")
    if a.get("tool")=="bash":
        if set(a)!={"tool","command"} or not isinstance(a.get("command"),str) or not a["command"].strip() or len(a["command"])>16000 or "\x00" in a["command"]:
            raise ValueError("Invalid bash command")
    elif a.get("tool")=="message":
        if set(a)!={"tool","recipient","content"} or a.get("recipient") not in ("agent1","agent2") or not isinstance(a.get("content"),str) or not a["content"].strip() or len(a["content"])>4000 or paths:
            raise ValueError("Invalid message")
    elif a.get("tool")=="git_publish":
        if set(a)!={"tool"} or paths:
            raise ValueError("Invalid Git publication")
    elif a.get("tool")=="git_fetch":
        if set(a)!={"tool","peer"} or a.get("peer") not in ("agent1","agent2") or paths:
            raise ValueError("Invalid Git fetch")
    else:
        raise ValueError("Unsupported tool")
    json.dumps(p, ensure_ascii=False).encode("utf-8")
    return deepcopy(p)

def observation(view,actor,seq,proposal,history,state):
    # Exclude earlier embedded observations to prevent recursive growth.
    history=[e for e in history if e["event_type"] not in {"arbiter_observation","run_start"}]
    recent=[{k:e[k] for k in ("sequence_number","actor_type","actor_id","event_type","payload")} for e in history[-8:]]
    return {"schema_version":1,**deepcopy(view),"actor_id":actor,"proposal_sequence":seq,
            "proposed_action":deepcopy(proposal),"recent_events":recent,"repository_state":deepcopy(state)}

def dispatch(env,agents,actor,action,finished=()):
    if action["tool"]=="message":
        peer=action["recipient"]
        if peer==actor:
            raise ValueError("Cannot message self")
        if peer in finished:
            return {"output":f"{peer} has submitted; message not delivered", "exit_code":1}
        agents[peer].receive({"output":f"Message from {actor}: {action['content']}","exit_code":0})
        return {"output":"Message delivered","exit_code":0}
    return env.execute(actor,action)

def run_task(task,agents,env,writer,*,max_steps=20,gate=None,evaluator=None,usage=None):
    """Own environment lifecycle; evaluator receives patches outside agent sandboxes."""
    metrics={}
    status="failed"
    failure={}
    phase="agent_error"
    try:
        if gate is None:
            gate=ArbitrationGate("none")
        view=agent_view(task)
        if set(agents)!=set(view["agent_goals"]) or type(max_steps) is not int or max_steps<1:
            raise ValueError("Invalid agents or step limit")
        finished=set()
        escalated=False
        for _ in range(max_steps):
            for actor in sorted(agents):
                if actor in finished:
                    continue
                phase="infrastructure_error"
                state=env.state()
                ctx={"goal":view["agent_goals"][actor],"shared_goal":view["shared_goal"],
                     "constraints":view["constraints"],"actor_id":actor,
                     "workspace":state.get(actor,{})}
                phase="agent_error"
                proposed=agents[actor].propose(deepcopy(ctx))
                if proposed is None:
                    finished.add(actor)
                    writer.emit("agent","agent_submitted",{},actor_id=actor)
                    continue
                p=validate_proposal(proposed)
                if writer.sanitize(p)!=p:
                    raise ValueError("Proposal contains configured secret")
                if p["action"].get("recipient")==actor:
                    raise ValueError("Cannot message self")
                seq=writer.emit("agent","proposed_action",p,actor_id=actor)
                obs=writer.sanitize(observation(view,actor,seq,p,writer.snapshot()[:-1],state))
                writer.emit("system","arbiter_observation",obs,trigger_sequence=seq)
                attempted=False
                def execute():
                    nonlocal attempted,phase
                    if attempted:
                        raise RuntimeError("Proposal already dispatched")
                    attempted=True
                    phase="infrastructure_error"
                    return writer.sanitize(dispatch(env,agents,actor,p["action"],finished))
                result,stop=gate(writer,obs,execute)
                if stop:
                    escalated=True
                    break
                phase="agent_error"
                agents[actor].receive(writer.sanitize(result))
            if escalated or len(finished)==len(agents):
                break
        if evaluator is not None and not escalated:
            phase="infrastructure_error"
            evidence=evaluator(task,env.patches())
            for goal in task["goals"]:
                key=goal["goal_key"]
                result=evidence.get("goal_results",{}).get(key,"not_evaluated")
                if result not in {"passed","failed","not_evaluated"}:
                    raise ValueError("Invalid evaluator evidence")
                writer.emit("system","test_result",{"goal_key":key,"result":result})
            metrics={k:evidence.get(k) for k in ("shared_goal_satisfied","critical_constraints_passed","merge_status")}
            if evidence.get("evaluation_error"):
                failure={"failure_category":"infrastructure_error","failure_reason":"Evaluator failed"}
        if failure:
            status="failed"
        elif escalated or len(finished)==len(agents):
            status="completed"
            metrics["termination_reason"]="escalated" if escalated else "agents_submitted"
        else:
            failure={"failure_category":"timeout","failure_reason":"Step limit reached"}
    except Exception as exc:
        # Exception strings can contain credentials or model payloads.
        failure={"failure_category":"timeout" if isinstance(exc,TimeoutError) else phase,"failure_reason":type(exc).__name__}
    finally:
        try:
            env.close()
        except Exception as exc:
            status="failed"
            failure={"failure_category":"infrastructure_error","failure_reason":"Environment cleanup failed: "+type(exc).__name__}
        if usage is not None:
            try:
                metrics.update(usage())
            except Exception:
                metrics["usage_error"]=True
    writer.finish(status,**failure,**metrics)
    return writer.snapshot()
