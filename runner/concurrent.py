"""Concurrent two-agent trials with serialized authorization, not serialized tools.

Each worker owns its agent history and local workspace. The coordination lock
protects snapshots, mailboxes and causal event order; no model or tool runs
while it is held. The arbiter lock serializes model decisions only.
"""
from copy import deepcopy
from queue import Queue, Empty
from threading import Event, Lock, Thread
from time import monotonic

from .arbitration import ArbitrationGate
from .deadline import TrialDeadline
from .orchestrator import observation, validate_proposal
from .task_loader import agent_view


def run_concurrent(task, agents, env, writer, *, max_steps=20, gate=None,
                   evaluator=None, usage=None, deadline=None):
    """Run independent agent loops; retain a bounded cleanup window on abort."""
    deadline = deadline or TrialDeadline()
    gate = gate or ArbitrationGate("none")
    lock, arbiter_lock = Lock(), Lock()
    stop = Event()
    done = Queue()
    pending, inflight, workspaces = {}, {}, {}
    mail = {a: [] for a in agents}
    finished = set()
    version = 0
    failure = None
    escalated = False
    metrics = {}
    threads = []
    status = "failed"

    def check():
        if stop.is_set():
            raise TimeoutError("Trial stopped")
        deadline.remaining()

    def state_for(actor):
        check()
        if hasattr(env, "state_for"):
            return env.state_for(actor)
        return env.state().get(actor, {})

    def shared():
        return env.shared_state() if hasattr(env, "shared_state") else {}

    def state():
        return {"workspaces": deepcopy(workspaces), "shared_git": shared(),
                "pending_proposals": deepcopy(pending),
                "in_flight_actions": deepcopy(inflight),
                "submitted_agents": sorted(finished),
                "coordination_version": version}

    def cancelled(seq, reason):
        writer.emit("system","action_result",{"outcome":"cancelled","output":reason,"exit_code":1},
                    trigger_sequence=seq)
        return {"output":reason,"exit_code":1}

    def worker(actor, view):
        nonlocal version, escalated, failure
        agent = agents[actor]
        phase = "infrastructure_error"
        try:
            initial = state_for(actor)
            with lock:
                check()
                workspaces[actor] = writer.sanitize(initial)
            for _ in range(max_steps):
                check()
                with lock:
                    check()
                    inbox, mail[actor] = mail[actor], []
                    context = {"goal":view["agent_goals"][actor],"shared_goal":view["shared_goal"],
                               "constraints":view["constraints"],"actor_id":actor,
                               "workspace":deepcopy(workspaces[actor]),"shared_git":shared()}
                phase = "agent_error"
                for message in inbox:
                    agent.receive(message)
                proposal = agent.propose(writer.sanitize(context))
                with lock:
                    check()
                    if proposal is None:
                        finished.add(actor)
                        version += 1
                        writer.emit("agent","agent_submitted",{},actor_id=actor)
                        return
                    p = validate_proposal(proposal)
                    if p["action"].get("recipient")==actor or p["action"].get("peer")==actor:
                        raise ValueError("Cannot target self")
                    # Never dispatch a command containing a configured secret.
                    if writer.sanitize(p) != p:
                        raise ValueError("Proposal contains configured secret")
                    pending[actor] = {"proposed_action":deepcopy(p),"phase":"queued"}
                    writer.emit("agent","proposal_queued",p,actor_id=actor)
                # Only arbitration is serialized. Waiting workers expose pending proposals.
                with arbiter_lock:
                    with lock:
                        check()
                        p = pending.pop(actor)["proposed_action"]
                        seq = writer.emit("agent","proposed_action",p,actor_id=actor)
                        snap = state()
                        observed_version = version
                        obs = writer.sanitize(observation(view,actor,seq,p,writer.snapshot()[:-1],snap))
                        writer.emit("system","arbiter_observation",obs,trigger_sequence=seq)
                        pending[actor] = {"proposal_sequence":seq, "proposed_action":p,"phase":"deciding"}
                    evaluation = gate.evaluate(obs)
                    with lock:
                        check()
                        pending.pop(actor)
                        action = gate.record(writer,obs,evaluation)
                        if action != "continue":
                            result = writer.sanitize(gate.feedback(action,evaluation[1]))
                            writer.emit("system","action_result",{"outcome":"cancelled",**result},trigger_sequence=seq)
                            if action=="mediate":
                                for peer in evaluation[1]["affected_agents"]:
                                    if peer!=actor and peer not in finished:
                                        mail[peer].append({"output":f"Arbiter mediation concerning {actor}: {result['output']}",
                                                           "exit_code":1})
                            if action=="escalate":
                                escalated=True
                                stop.set()
                                return
                        elif gate.condition in {"llm","rule_based"} and (version!=observed_version or shared()!=snap["shared_git"]):
                            # A completion/publication during reasoning invalidates this
                            # approval. Do not silently dispatch against stale state.
                            result=cancelled(seq,"Coordination changed during arbitration; inspect and repropose.")
                        else:
                            inflight[actor]={"proposal_sequence":seq,"proposed_action":p,"started":False}
                            result=None
                # Release both locks before executing tools.
                if result is None:
                    phase="infrastructure_error"
                    try:
                        with lock:
                            check()
                            inflight[actor]["started"]=True
                            writer.emit("system","action_started",{"actor_id":actor},trigger_sequence=seq)
                        a=p["action"]
                        if a["tool"]=="message":
                            with lock:
                                check()
                                peer=a["recipient"]
                                if peer in finished:
                                    result={"output":f"{peer} has submitted; message not delivered","exit_code":1}
                                else:
                                    mail[peer].append({"output":f"Message from {actor}: {a['content']}","exit_code":0})
                                    result={"output":"Message delivered","exit_code":0}
                        elif a["tool"] in {"git_publish","git_fetch"}:
                            result=env.execute(actor,a,expected_refs=snap["shared_git"].get("published",{}))
                        else:
                            result=env.execute(actor,a)
                        after=state_for(actor)
                    except Exception as exc:
                        with lock:
                            if not stop.is_set() and actor in inflight:
                                inflight.pop(actor)
                                version+=1
                                writer.emit("system","action_result",{"outcome":"unknown","error_type":type(exc).__name__},
                                            trigger_sequence=seq)
                        raise
                    with lock:
                        check()
                        inflight.pop(actor)
                        version+=1
                        workspaces[actor]=writer.sanitize(after)
                        result=writer.sanitize(result)
                        writer.emit("system","action_result",{"outcome":"dispatched",**result},trigger_sequence=seq)
                phase="agent_error"
                check()
                agent.receive(result)
            raise TimeoutError("Agent step limit reached")
        except Exception as exc:
            with lock:
                if not stop.is_set():
                    failure={"failure_category":"timeout" if isinstance(exc,TimeoutError) else phase,
                             "failure_reason":type(exc).__name__}
                    stop.set()
        else:
            done.put((actor,None,None))
        finally:
            done.put((actor,"finished_worker",None))

    try:
        view=agent_view(task)
        if set(agents)!=set(view["agent_goals"]) or type(max_steps) is not int or max_steps<1:
            raise ValueError("Invalid agents or step limit")
        for actor in sorted(agents):
            thread=Thread(target=worker,args=(actor,view),name=f"align-{actor}",daemon=True)
            threads.append(thread)
            thread.start()
        exits=set()
        while len(exits)<len(threads) and not stop.is_set():
            try:
                actor,kind,reason=done.get(timeout=min(.1,deadline.remaining()))
            except Empty:
                continue
            if kind=="finished_worker":
                exits.add(actor)
            elif kind:
                failure={"failure_category":kind,"failure_reason":reason}
                stop.set()
        with lock:
            if escalated:
                metrics["termination_reason"]="escalated"
                status="completed"
            elif failure is None and len(finished)==len(agents):
                status="completed"
                metrics["termination_reason"]="agents_submitted"
        if status=="completed" and not escalated and evaluator is not None:
            # No workers can mutate workspaces after all have submitted.
            deadline.remaining()
            evidence=evaluator(task,env.patches())
            deadline.remaining()
            for goal in task["goals"]:
                key=goal["goal_key"]
                result=evidence.get("goal_results",{}).get(key,"not_evaluated")
                if result not in {"passed","failed","not_evaluated"}:
                    raise ValueError("Invalid evaluator evidence")
                writer.emit("system","test_result",{"goal_key":key,"result":result})
            metrics.update({k:evidence.get(k) for k in ("shared_goal_satisfied","critical_constraints_passed","merge_status")})
            if evidence.get("evaluation_error"):
                status="failed"
                failure={"failure_category":"infrastructure_error","failure_reason":"Evaluator failed"}
    except Exception as exc:
        status="failed"
        failure={"failure_category":"timeout" if isinstance(exc,TimeoutError) else "infrastructure_error",
                 "failure_reason":type(exc).__name__}
    finally:
        # Stop all authorization and prevent late writes before removing resources.
        with lock:
            stop.set()
        deadline.cancel()
        try:
            env.close()
        except Exception as exc:
            status="failed"
            metrics["cleanup_failed"]=True
            if failure is None:
                failure={"failure_category":"infrastructure_error","failure_reason":"Environment cleanup failed: "+type(exc).__name__}
        # A remote model call cannot necessarily be killed. It may return later,
        # but cannot authorize or dispatch after stop, or append to the bundle.
        grace=monotonic()+1
        for thread in threads:
            thread.join(max(0,grace-monotonic()))
        with lock:
            for item in inflight.values():
                writer.emit("system","action_result",{"outcome":"unknown" if item["started"] else "cancelled",
                                                       "output":"Trial stopped during execution" if item["started"] else "Trial stopped before execution"},
                            trigger_sequence=item["proposal_sequence"])
            for item in pending.values():
                if "proposal_sequence" in item:
                    writer.emit("system","action_result",{"outcome":"cancelled","output":"Trial stopped before authorization"},
                                trigger_sequence=item["proposal_sequence"])
            if usage is not None:
                try:
                    if all(not t.is_alive() for t in threads):
                        metrics.update(usage())
                    else:
                        metrics["usage_incomplete"]=True
                except Exception:
                    metrics["usage_error"]=True
            if failure:
                status="failed"
            if status=="failed" and failure is None:
                failure={"failure_category":"agent_error","failure_reason":"Trial stopped before submission"}
            writer.finish(status,**(failure or {}),**metrics)
    return writer.snapshot()
