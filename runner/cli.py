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
from .task_loader import load_tasks,cooperbench_identity

def run(args):
    tasks=load_tasks(args.tasks)
    if args.task_key:
        tasks=[task for task in tasks if task["task_key"] in args.task_key]
        if {task["task_key"] for task in tasks}!=set(args.task_key):
            raise ValueError("Unknown task key requested")
    if args.limit<1 or args.max_steps<1:
        raise ValueError("Positive limits required")
    if args.live and not all((args.agent_endpoint,args.agent_model,args.images,args.dataset_dir)):
        raise ValueError("Live mode requires agent endpoint/model, trusted image map and dataset dir")
    if args.live and args.condition in {"llm","monitor"} and not (args.arbiter_endpoint and args.arbiter_model):
        raise ValueError("Live LLM/monitor requires an arbiter endpoint and model")
    upstream_revision=installed_revision() if args.live else None
    images=json.loads(Path(args.images).read_text()) if args.live else {}
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    failures=0
    for task in tasks[:args.limit]:
        agent_config={"backend":"docker" if args.live else "mock","model":args.agent_model if args.live else "scripted",
                      "max_steps":args.max_steps,"protocol":"align-turn-based-v1","upstream_revision":upstream_revision,
                      "model_timeout_s":args.model_timeout,"tool_timeout_s":args.tool_timeout}
        arbiter_config={}
        if args.condition in {"llm","monitor"}:
            if args.live:
                arbiter_config={"model":args.arbiter_model,"training":args.training,
                                "checkpoint":args.checkpoint,"prompt_version":"v1"}
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
                                                timeout=args.model_timeout)) for a in ("agent1","agent2")}
                evaluator=lambda t,p:evaluate_patches(t,p,args.dataset_dir)
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
                arbiter=ChatArbiter(ChatClient(args.arbiter_endpoint,args.arbiter_model,key_env="ALIGN_ARBITER_API_KEY",
                                               timeout=args.model_timeout))
            else:
                arbiter=ConstantArbiter(args.mock_action)
            if args.live:
                env=prepare_environment(task,image,timeout=args.tool_timeout)
            def usage():
                if not args.live:
                    return {}
                clients=[a.client for a in agents.values()]
                if isinstance(arbiter,ChatArbiter):
                    clients.append(arbiter.client)
                if not all(c.usage_known for c in clients):
                    return {}
                return {"input_tokens":sum(c.input_tokens for c in clients),
                        "output_tokens":sum(c.output_tokens for c in clients)}
            events=run_task(task,agents,env,writer,max_steps=args.max_steps,
                            gate=ArbitrationGate(args.condition,arbiter),evaluator=evaluator,usage=usage)
        except Exception as exc:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            events=writer.snapshot()
            if events[-1]["event_type"]!="run_end":
                writer.finish("failed",failure_category="infrastructure_error",failure_reason=type(exc).__name__)
                events=writer.snapshot()
        manifest=writer.write_bundle(folder/"events.jsonl",folder/"bundle")
        status=events[-1]["payload"].get("status")
        failures+=status!="completed"
        print(json.dumps({"task_key":task["task_key"],"run_key":writer.run_key,"status":status,
                          "bundle":str(folder/"bundle"),"event_count":manifest["event_count"]}))
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
    p.add_argument("--mock-action",choices=["continue","block","mediate","escalate"],default="continue")
    return run(p.parse_args(argv))

if __name__=="__main__":
    raise SystemExit(main())
