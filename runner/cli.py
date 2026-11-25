"""Run one or more normalized ALIGN tasks. Defaults to free deterministic mocks."""
import argparse
import json
from pathlib import Path
from .arbitration import ArbitrationGate,ConstantArbiter
from .cooperbench_adapter import prepare_environment,evaluate_patches,installed_revision
from .environment import MockEnvironment,ScriptedAgent
from .event_writer import EventWriter
from .model_adapter import ChatAgent,ChatArbiter,ChatClient,SafetyRuleArbiter
from .orchestrator import run_task
from .concurrent import run_concurrent
from .deadline import TrialDeadline
from .arbiter_prompt import prompt_hash
from .task_loader import load_tasks,cooperbench_identity

def run(args, *, arbiter_factory=None):
    tasks=load_tasks(args.tasks)
    if args.task_key:
        tasks=[task for task in tasks if task["task_key"] in args.task_key]
        if {task["task_key"] for task in tasks}!=set(args.task_key):
            raise ValueError("Unknown task key requested")
    if args.limit<1 or args.max_steps<1:
        raise ValueError("Positive limits required")
    if args.live and not all((args.agent_endpoint,args.agent_model,args.images,args.dataset_dir)):
        raise ValueError("Live mode requires agent endpoint/model, trusted image map and dataset dir")
    if args.live and args.protocol=="alternating":
        raise ValueError("Live experiments require the concurrent protocol")
    concurrent=args.live or args.protocol=="concurrent"
    if args.live and args.condition in {"llm","monitor"}:
        if not args.arbiter_model or (arbiter_factory is None and not args.arbiter_endpoint):
            raise ValueError("Live LLM/monitor requires an arbiter model and endpoint or callable factory")
    TrialDeadline(args.trial_timeout)  # validate before starting trials
    upstream_revision=installed_revision() if args.live else None
    images=json.loads(Path(args.images).read_text()) if args.live else {}
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    failures=0
    for task in tasks[:args.limit]:
        deadline=TrialDeadline(args.trial_timeout)
        agent_config={"backend":"docker" if args.live else "mock","model":args.agent_model if args.live else "scripted",
                      "max_steps":args.max_steps,"protocol":"align-concurrent-shared-git-v1" if concurrent else "align-turn-based-v1","upstream_revision":upstream_revision,
                      "model_timeout_s":args.model_timeout,"tool_timeout_s":args.tool_timeout,"trial_timeout_s":args.trial_timeout}
        arbiter_config={}
        if args.condition in {"llm","monitor"}:
            if args.live:
                arbiter_config={"model":args.arbiter_model,"training":args.training,
                                "checkpoint":args.checkpoint,"prompt_sha256":prompt_hash(),
                                "backend":"callable" if arbiter_factory is not None else "http"}
            else:
                arbiter_config={"model":"fixture","action":args.mock_action}
        elif args.condition=="rule_based":
            arbiter_config={"rules":"a2-provisional-v1"}
        writer=EventWriter(task_key=task["task_key"],arbiter_condition=args.condition,
                           agent_config=agent_config,arbiter_config=arbiter_config,seed=args.seed)
        folder=out/writer.run_key
        folder.mkdir()
        env=None
        try:
            if args.live:
                repo,number,_=cooperbench_identity(task)
                image=images[f"{repo}:task{number}"]
                agents={a:ChatAgent(ChatClient(args.agent_endpoint,args.agent_model,key_env="ALIGN_AGENT_API_KEY",
                                                timeout=args.model_timeout,deadline=deadline)) for a in ("agent1","agent2")}
                evaluator=lambda t,p:evaluate_patches(t,p,args.dataset_dir,deadline=deadline,
                                                          expected_image=getattr(env,"image_id",None))
            else:
                env=MockEnvironment()
                agents={a:ScriptedAgent([{"action":{"tool":"bash","command":"echo fixture"},"files_touched":[]}])
                        for a in ("agent1","agent2")}
                evaluator=None
            if args.condition=="none":
                arbiter=None
            elif args.condition=="rule_based":
                arbiter=SafetyRuleArbiter()
            elif args.live:
                arbiter=(arbiter_factory(args) if arbiter_factory is not None else
                         ChatArbiter(ChatClient(args.arbiter_endpoint,args.arbiter_model,
                                               key_env="ALIGN_ARBITER_API_KEY",timeout=args.model_timeout,
                                               deadline=deadline)))
            else:
                arbiter=ConstantArbiter(args.mock_action)
            if args.live:
                env=prepare_environment(task,image,timeout=args.tool_timeout,deadline=deadline,concurrent=True)
                writer.emit("system","environment_ready",{"image_id":getattr(env,"image_id",None),
                            "base_commit":task.get("base_commit"),"protocol":"align-concurrent-shared-git-v1"})
            def usage():
                if not args.live:
                    return {}
                clients=[a.client for a in agents.values()]
                if isinstance(arbiter,ChatArbiter):
                    clients.append(arbiter.client)
                elif args.condition in {"monitor","llm"}:
                    # The callable API currently exposes no complete usage totals.
                    return {"usage_incomplete":True}
                if not all(c.usage_known for c in clients):
                    return {"usage_incomplete":True}
                return {"input_tokens":sum(c.input_tokens for c in clients),
                        "output_tokens":sum(c.output_tokens for c in clients)}
            runner=run_concurrent if concurrent else run_task
            events=runner(task,agents,env,writer,max_steps=args.max_steps,
                          gate=ArbitrationGate(args.condition,arbiter),evaluator=evaluator,usage=usage,
                          **({"deadline":deadline} if concurrent else {}))
        except Exception as exc:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            events=writer.snapshot()
            if events[-1]["event_type"]!="run_end":
                writer.finish("failed",failure_category="timeout" if isinstance(exc,TimeoutError) else "infrastructure_error",failure_reason=type(exc).__name__)
                events=writer.snapshot()
        manifest=writer.write_bundle(folder/"events.jsonl",folder/"bundle")
        status=events[-1]["payload"].get("status")
        failures+=status!="completed"
        print(json.dumps(writer.sanitize({"task_key":task["task_key"],"run_key":writer.run_key,"status":status,
                          "bundle":str(folder/"bundle"),"event_count":manifest["event_count"]})))
    return int(failures>0)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--condition",choices=["none","monitor","rule_based","llm"],default="none")
    p.add_argument("--live",action="store_true",help="Explicitly allow Docker and paid model calls")
    p.add_argument("--limit",type=int,default=1)
    p.add_argument("--task-key",action="append",help="Select a normalized task key; repeat to select several")
    p.add_argument("--max-steps",type=int,default=20)
    p.add_argument("--seed",type=int,default=42,help="Trial identifier; endpoints may not support deterministic seeds")
    p.add_argument("--images",help="Trusted JSON image map keyed repo_directory:taskN")
    p.add_argument("--dataset-dir",help="Trusted CooperBench dataset directory, never mounted in agent containers")
    p.add_argument("--agent-endpoint")
    p.add_argument("--agent-model")
    p.add_argument("--arbiter-endpoint")
    p.add_argument("--arbiter-model")
    p.add_argument("--training",choices=["none","sft"],default="none")
    p.add_argument("--checkpoint")
    p.add_argument("--tool-timeout",type=int,default=30)
    p.add_argument("--model-timeout",type=int,default=60)
    p.add_argument("--trial-timeout",type=float,default=600)
    p.add_argument("--protocol",choices=["auto","alternating","concurrent"],default="auto",
                   help="Live always uses concurrent; mocks default to deterministic alternating")
    p.add_argument("--arbiter-backend",choices=["http","modal"],default="http",
                   help="Modal is explicit opt-in and requires Person 2's training package")
    p.add_argument("--mock-action",choices=["continue","query","block","mediate","escalate"],default="continue")
    args=p.parse_args(argv)
    if args.arbiter_backend=="modal":
        if not args.live or args.condition not in {"llm","monitor"}:
            p.error("Modal backend requires --live and an LLM/monitor condition")
        if args.arbiter_model not in {"qwen","gpt-oss"}:
            p.error("Modal backend requires --arbiter-model qwen or gpt-oss")
        if args.training=="sft" and not args.checkpoint:
            p.error("SFT Modal backend requires --checkpoint adapter path")
        from arbiter_training.inference import ModalArbiter
        from arbiter_training.modal_app import app
        with app.run():
            return run(args,arbiter_factory=lambda cfg:ModalArbiter(
                model=cfg.arbiter_model,adapter_path=cfg.checkpoint if cfg.training=="sft" else "",
                invalid_policy="raise"))
    return run(args)

if __name__=="__main__":
    raise SystemExit(main())
