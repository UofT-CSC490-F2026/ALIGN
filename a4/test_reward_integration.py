"""Local integration contracts, no torch, CUDA, Modal, weights or datasets."""
import argparse
import ast
import copy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from reward_integration import (add_reward_arguments, validate_reward_arguments,
                                optimizer_steps, reference_text, reward_components)
from launcher_config import build_command, merge_move

HERE=Path(__file__).parent
SOURCE=HERE/'nanochat/scripts/chat_rl.py'


def test_default_parser_and_baseline_bypass(monkeypatch):
    args=add_reward_arguments(argparse.ArgumentParser()).parse_args([])
    validate_reward_arguments(args)
    assert args.reward_mode=='baseline' and args.max_steps is None
    monkeypatch.setattr('reward_integration.completion_reward',lambda *a: pytest.fail('baseline parsed completion'))
    monkeypatch.setattr('reward_integration.arithmetic_reward',lambda *a: pytest.fail('baseline parsed arithmetic'))
    for correct in (0.,1.):
        result=reward_components(correct,None,None)
        assert result['total'] is correct
        assert result['completion']==result['arithmetic']==0


@pytest.mark.parametrize('mode,total', [('baseline',0),('completion',.15),('arithmetic',.20),('combined',.35)])
def test_modes_and_reference_serialization(mode,total):
    conv={'messages':[{'role':'assistant','content':[
        {'type':'text','text':'Calculate '},{'type':'python','text':'3*4'},
        {'type':'python_output','text':'12'},{'type':'text','text':'12\n#### 12'}]}]}
    reference=reference_text(conv)
    assert reference=='Calculate <<3*4=12>>12\n#### 12'
    result=reward_components(0,reference,'3*4=12\n#### 13',mode)
    assert result['correctness']==0
    assert result['total']==pytest.approx(total)


@pytest.mark.parametrize('mode', ['completion','arithmetic','combined'])
def test_shaping_produces_group_relative_advantages(mode):
    scores=[reward_components(0,'<<3*4=12>>',t,mode)['total'] for t in ('3*4=12\n#### 13','unfinished')]
    mu=sum(scores)/len(scores)
    advantages=[r-mu for r in scores]
    assert advantages[0]>0>advantages[1]
    assert sum(advantages)==pytest.approx(0)


@pytest.mark.parametrize('resume,cap,expected', [(None,1,[0]),(None,3,[0,1,2]),(60,3,[61,62,63]),(465,3,[466]),(466,1,[]),(None,None,list(range(467)))])
def test_optimizer_step_boundaries(resume,cap,expected):
    assert list(optimizer_steps(467,resume,cap))==expected


def test_actual_training_loop_cap_and_final_checkpoint():
    """Execute the production loop's iterator/update/save condition with stub compute.

    Loss, sampling and distributed evaluation are excluded by AST selection;
    the actual optimizer call and checkpoint condition execute unchanged.
    """
    tree=ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
    loop=next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
    update=next(n for n in loop.body if isinstance(n,ast.Expr) and ast.unparse(n)=='optimizer.step()')
    save=next(n for n in loop.body if isinstance(n,ast.If) and 'args.save_every' in ast.unparse(n.test))
    test_loop=copy.deepcopy(loop)
    checkpoint=ast.If(test=copy.deepcopy(save.test),body=[ast.parse('saved.append(step)').body[0]],orelse=[])
    test_loop.body=[copy.deepcopy(update),checkpoint]
    module=ast.fix_missing_locations(ast.Module(body=[test_loop],type_ignores=[]))
    for resume, cap, expected in [(None,1,1),(None,3,3),(60,3,3),(465,3,1)]:
        optimizer=SimpleNamespace(step=Mock())
        namespace=dict(optimizer=optimizer,training_steps=optimizer_steps(467,resume,cap),
                       master_process=True,args=SimpleNamespace(save_every=60),saved=[])
        exec(compile(module,str(SOURCE),'exec'),namespace)
        assert optimizer.step.call_count==expected
        assert namespace['saved'][-1]==namespace['training_steps'].stop-1


def test_training_and_eval_correctness_paths_unchanged():
    tree=ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
    get_batch=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_batch')
    source=ast.unparse(get_batch)
    assert 'train_task.reward(conversation, generated_text)' in source
    assert "rewards.append(component['total'])" in source
    assert 'advantages = rewards - mu' in source
    assert "'reward': c['correctness']" in source
    evaluate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_gsm8k_eval')
    assert 'task.evaluate(conversation, generated_text)' in ast.unparse(evaluate)
    assert not any(isinstance(n,ast.Name) and n.id=='reward_components' for n in ast.walk(evaluate))


@pytest.mark.parametrize('field,value', [('max_steps',0),('max_steps',-1),('completion_weight',float('nan')),('arithmetic_weight',.21),('output_tag',None)])
def test_configuration_rejected(field,value):
    args=SimpleNamespace(reward_mode='combined',completion_weight=.15,arithmetic_weight=.20,max_steps=1,output_tag='unique')
    setattr(args,field,value)
    with pytest.raises(ValueError): validate_reward_arguments(args)


@pytest.mark.parametrize('mode,steps,processes', [('dry','1',1),('smoke','3',8),('full',None,8)])
def test_launcher_caps_identity_and_rewards(mode,steps,processes):
    command=build_command('rl',mode,'new-run',[],start='a4-sft-default',reward_mode='combined')
    assert command[command.index('--output-tag')+1]=='new-run'
    assert command[command.index('--model-tag')+1]=='a4-sft-default'
    assert command[command.index('--rollouts-dir')+1]=='/vol/rollouts/new-run'
    assert command[command.index('--reward-mode')+1]=='combined'
    if steps: assert command[command.index('--max-steps')+1]==steps
    else: assert '--max-steps' not in command
    assert any('torchrun' in arg for arg in command)==(processes==8)


@pytest.mark.parametrize('extra', [['--max-steps','100'],['--max-steps=100'],['--output-tag','old'],['--run=old'],['--resume-from','old']])
def test_no_smoke_cap_or_artifact_identity_override(extra):
    with pytest.raises(ValueError): build_command('rl','smoke','new',extra,start='a4-sft-default')


def test_local_preview_imports_no_cloud_or_model_modules():
    code="import sys; sys.path.insert(0,sys.argv[1]); import preview_launch; assert not any(m in sys.modules for m in ('modal','torch','wandb')); sys.argv=['preview','--run','preview','--reward-mode','combined']; preview_launch.main()"
    result=subprocess.run([sys.executable,'-c',code,str(HERE)],capture_output=True,text=True,check=True)
    assert '--max-steps 3' in result.stdout


def test_merge_move_refuses_collisions(tmp_path):
    source=tmp_path/'source'; target=tmp_path/'target'
    source.mkdir(); target.mkdir()
    (source/'model.pt').write_text('new'); (target/'model.pt').write_text('old')
    with pytest.raises(FileExistsError): merge_move(source,target)
    assert (target/'model.pt').read_text()=='old'
    assert (source/'model.pt').read_text()=='new'


def test_image_packages_all_reward_dependencies():
    tree=ast.parse((HERE/'modal_launch.py').read_text(encoding='utf-8-sig'))
    image=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='image' for t in n.targets))
    paths=[n.value for n in ast.walk(image) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
    assert all('/root/a4/'+f in paths for f in ('rewards.py','arithmetic_reward.py','reward_integration.py'))


@pytest.mark.parametrize('mode', ['baseline','completion','arithmetic','combined'])
def test_production_rollout_pipeline_with_stub_sampling(tmp_path,mode):
    """Execute get_batch itself using tiny in-memory samples and tensor stubs."""
    import itertools
    import json
    class Tensor:
        def __init__(self,values): self.values=values
        def __getitem__(self,key): return self
        def __setitem__(self,key,value): pass
        def __eq__(self,other): return False
        def clone(self): return self
        def mean(self): return sum(self.values)/len(self.values)
        def __sub__(self,value): return [v-value for v in self.values]
    text=['3*4=12\n#### 13','unfinished']
    conv={'messages':[{'role':'user','content':'Calculate 3*4'}, {'role':'assistant','content':[
        {'type':'python','text':'3*4'},{'type':'python_output','text':'12'},
        {'type':'text','text':'12\n#### 12'}]}]}
    class Task:
        def __len__(self): return 1
        def __getitem__(self,i): return conv
        def reward(self,conversation,generated_text):
            assert conversation is conv
            return 0.0
    tokenizer=SimpleNamespace(encode_special=lambda x: 0, render_for_completion=lambda x:['prefix'],
                              decode=lambda tokens: ''.join(tokens))
    engine=SimpleNamespace(generate_batch=lambda *a,**kw: ([['prefix',*t] for t in text],[[1]*(len(t)+1) for t in text]))
    tree=ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
    function=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_batch'))
    function.decorator_list=[]
    namespace=dict(torch=SimpleNamespace(tensor=lambda values,**kw:Tensor(values),long='long',float='float'),
                   tokenizer=tokenizer,engine=engine,model=SimpleNamespace(eval=lambda:None),
                   ddp_rank=0,ddp_world_size=1,train_task=Task(),itertools=itertools,step=0,device='stub',
                   args=SimpleNamespace(resume_from=None,examples_per_step=1,num_samples=2,device_batch_size=2,
                                        max_new_tokens=32,temperature=1.,top_k=50,reward_mode=mode,
                                        completion_weight=.15,arithmetic_weight=.20),
                   reference_text=reference_text,reward_components=reward_components,json=json,
                   rollouts_path=str(tmp_path/'rank0.jsonl'))
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    exec(compile(module,str(SOURCE),'exec'),namespace)
    batch=next(namespace['get_batch']())
    row=json.loads((tmp_path/'rank0.jsonl').read_text())
    assert row['reference']=='<<3*4=12>>12\n#### 12'
    assert row['reward_mode']==mode
    assert all(sample['reward']==sample['correctness']==0 for sample in row['samples'])
    expected={'baseline':0,'completion':.15,'arithmetic':.2,'combined':.35}[mode]
    assert batch[3].values==pytest.approx([expected,0])
    assert batch[4]==pytest.approx([expected/2,-expected/2])
    assert row['samples'][0]['total']==pytest.approx(expected)


def test_isolated_launcher_refuses_existing_artifacts_before_process(tmp_path):
    tree=ast.parse((HERE/'modal_launch.py').read_text(encoding='utf-8-sig'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute_isolated_rl')
    def mapped_path(value):
        return tmp_path/'rollouts' if str(value)=='/vol/rollouts' else Path(value)
    process=Mock()
    namespace=dict(Path=mapped_path,subprocess=SimpleNamespace(Popen=process),volume=SimpleNamespace(commit=Mock()),mark_run=Mock())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'isolated','exec'),namespace)
    root=tmp_path/'checkpoints'
    (root/'old').mkdir(parents=True)
    (root/'old'/'model.pt').write_text('keep')
    with pytest.raises(FileExistsError): namespace['execute_isolated_rl']([],str(root),'old')
    (tmp_path/'rollouts'/'logged').mkdir(parents=True)
    with pytest.raises(FileExistsError): namespace['execute_isolated_rl']([],str(root),'logged')
    process.assert_not_called()
    assert (root/'old'/'model.pt').read_text()=='keep'


def test_manifest_and_command_plan_match_original_defaults():
    import json
    from plan_experiments import experiment_commands
    manifest=json.loads((HERE/'part4_experiments.json').read_text())
    tree=ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
    parser=argparse.ArgumentParser()
    parser_nodes=[]
    for node in tree.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and ast.unparse(node.value.func)=='parser.add_argument':
            parser_nodes.append(node)
    exec(compile(ast.Module(body=parser_nodes,type_ignores=[]),'parser','exec'),{'parser':parser})
    defaults=vars(parser.parse_args([]))
    assert all(defaults[k]==value for k,value in manifest['training'].items())
    planned=[e for e in manifest['experiments'] if e['status']=='planned']
    assert {e['reward_mode'] for e in planned}=={'baseline','completion','arithmetic','combined'}
    assert all(e['start']=='a4-sft-default' for e in planned)
    assert len({e['checkpoint_dir'] for e in planned})==4
    assert len({e['wandb_run'] for e in planned})==4
    for mode in ('dry','smoke','full'):
        commands=experiment_commands(manifest,mode)
        assert len(commands)==4
        assert all('--start a4-sft-default' in c['command'] for c in commands)
        assert all(f'--mode {mode}' in c['command'] for c in commands)


def test_launcher_gpu_mode_selection_without_remote_calls():
    tree=ast.parse((HERE/'modal_launch.py').read_text(encoding='utf-8-sig'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    selections=[n for n in main.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('functions','function') for t in n.targets)]
    code=compile(ast.fix_missing_locations(ast.Module(body=selections,type_ignores=[])),'selection','exec')
    for script in ('rl','mcdiag'):
        for mode,expected in [('dry','one'),('smoke','smoke-eight'),('full','full-eight')]:
            namespace=dict(one_gpu='one',eight_gpus_smoke='smoke-eight',eight_gpus='full-eight',
                           SINGLE_PROCESS={'mcdiag'},script=script,mode=mode)
            exec(code,namespace)
            assert namespace['function']==('one' if script=='mcdiag' else expected)


def test_eval_loader_accepts_rl_checkpoints():
    tree=ast.parse((HERE/'eval_models.py').read_text(encoding='utf-8-sig'))
    folders=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FOLDERS' for t in n.targets))
    assert ast.literal_eval(folders.value)['rl']=='chatrl_checkpoints'



def test_real_gsm8k_reference_with_tool_calculations():
    # Weng's babysitting example from the pinned tasks/gsm8k.py documentation.
    reference='Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10'
    text=('<|python_start|>12/60<|python_end|><|output_start|>0.2<|output_end|>\n'
          '<|python_start|>0.2*50<|python_end|><|output_start|>10.0<|output_end|>\n#### 10')
    result=reward_components(1,reference,text,'combined')
    assert result==pytest.approx(dict(correctness=1,completion=.15,arithmetic=.2,total=1.35))
