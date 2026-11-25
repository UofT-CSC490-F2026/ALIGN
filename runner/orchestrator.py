"""Bounded alternating two-agent runner; same execution path for mocks and live agents."""
from copy import deepcopy
from .task_loader import agent_view

def validate_proposal(p):
    if not isinstance(p,dict) or set(p)!={"action","files_touched"}:
        raise ValueError("Invalid proposal")
    a,paths=p["action"],p["files_touched"]
    if not isinstance(a,dict) or not isinstance(paths,list) or not all(isinstance(x,str) for x in paths):
        raise ValueError("Invalid action/paths")
    if a.get("tool")=="bash":
        if not isinstance(a.get("command"),str) or not a["command"] or len(a["command"])>16000:
            raise ValueError("Invalid bash command")
    elif a.get("tool")=="message":
        if a.get("recipient") not in {"agent1","agent2"} or not isinstance(a.get("content"),str) or len(a["content"])>4000:
            raise ValueError("Invalid message")
    else:
        raise ValueError("Unsupported tool")
    return deepcopy(p)

def observation(view,actor,seq,proposal,history,state):
    # Exclude earlier embedded observations to prevent recursive growth.
    history=[e for e in history if e["event_type"] not in {"arbiter_observation","run_start"}]
    recent=[{k:e[k] for k in ("sequence_number","actor_type","actor_id","event_type","payload")} for e in history[-8:]]
    return {"schema_version":1,**deepcopy(view),"actor_id":actor,"proposal_sequence":seq,
            "proposed_action":deepcopy(proposal),"recent_events":recent,"repository_state":deepcopy(state)}

def dispatch(env,agents,actor,action):
    if action["tool"]=="message":
        peer=action["recipient"]
        if peer==actor:
            raise ValueError("Cannot message self")
        agents[peer].receive({"output":f"Message from {actor}: {action['content']}","exit_code":0})
        return {"output":"Message delivered","exit_code":0}
    return env.execute(actor,action)

def passthrough(writer,obs,execute):
    result=execute()
    writer.emit("system","action_result",{"outcome":"dispatched",**result},trigger_sequence=obs["proposal_sequence"])
    return result,False

def run_task(task,agents,env,writer,*,max_steps=20,gate=passthrough,evaluator=None,usage=None):
    """Own environment lifecycle; evaluator receives patches outside agent sandboxes."""
    try:
        view=agent_view(task)
        if set(agents)!=set(view["agent_goals"]) or type(max_steps) is not int or max_steps<1:
            raise ValueError("Invalid agents or step limit")
        finished=set()
        escalated=False
        for _ in range(max_steps):
            for actor in sorted(agents):
                if actor in finished:
                    continue
                state=env.state()
                ctx={"goal":view["agent_goals"][actor],"shared_goal":view["shared_goal"],
                     "constraints":view["constraints"],"actor_id":actor,
                     "workspace":state.get(actor,{})}
                proposed=agents[actor].propose(deepcopy(ctx))
                if proposed is None:
                    finished.add(actor)
                    writer.emit("agent","agent_submitted",{},actor_id=actor)
                    continue
                p=validate_proposal(proposed)
                seq=writer.emit("agent","proposed_action",p,actor_id=actor)
                obs=observation(view,actor,seq,p,writer.snapshot()[:-1],state)
                writer.emit("system","arbiter_observation",obs,trigger_sequence=seq)
                result,stop=gate(writer,obs,lambda:dispatch(env,agents,actor,p["action"]))
                agents[actor].receive(result)
                if stop:
                    escalated=True
                    break
            if escalated or len(finished)==len(agents):
                break
        metrics={}
        if evaluator is not None and not escalated:
            evidence=evaluator(task,env.patches())
            for goal in task["goals"]:
                key=goal["goal_key"]
                result=evidence.get("goal_results",{}).get(key,"not_evaluated")
                if result not in {"passed","failed","not_evaluated"}:
                    raise ValueError("Invalid evaluator evidence")
                writer.emit("system","test_result",{"goal_key":key,"result":result})
            metrics={k:evidence.get(k) for k in ("shared_goal_satisfied","critical_constraints_passed","merge_status")}
            if evidence.get("evaluation_error"):
                writer.finish("failed",failure_category="infrastructure_error",failure_reason="Evaluator failed",**metrics)
                return writer.snapshot()
        if usage is not None:
            metrics.update(usage())
        if escalated or len(finished)==len(agents):
            writer.finish("completed",**metrics)
        else:
            writer.finish("failed",failure_category="timeout",failure_reason="Step limit reached",**metrics)
    except Exception as exc:
        # Exception strings can contain credentials or model payloads.
        writer.finish("failed",failure_category="agent_error",failure_reason=type(exc).__name__)
    finally:
        env.close()
    return writer.snapshot()
