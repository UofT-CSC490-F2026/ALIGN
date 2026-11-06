"""Offline recovery checks. No model packages, network access or GPU required."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from state import (record, latest_checkpoint, bpb_rows, saved_answers,
                   VALIDATION_TOKENS, BATCH_SIZE, RESULT_FOLDER)

HERE = Path(__file__).parent


def load_function(filename, name, env):
    tree = ast.parse((HERE / filename).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]),filename,'exec'),env)
    return env[name]


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def row(self, model='pico', context=512):
        return dict(model=model, checkpoint_step=500 if model=='pico' else 5571,
                    context=context, batch_size=8, nominal_positions=VALIDATION_TOKENS,
                    batches=VALIDATION_TOKENS//(8*context), bpb=1.2)

    def checkpoint(self, step, ranks=8):
        folder=self.root/'base_checkpoints/fox-d24';folder.mkdir(parents=True,exist_ok=True)
        record(folder/f'meta_{step:06d}.json',{'step':step})
        (folder/f'model_{step:06d}.pt').write_bytes(b'model')
        for rank in range(ranks):(folder/f'optim_{step:06d}_rank{rank}.pt').write_bytes(b'optimizer')
        return folder

    def test_incomplete_checkpoint_skipped(self):
        folder=self.checkpoint(500);self.checkpoint(1000,ranks=7)
        self.assertEqual(latest_checkpoint(folder,optimizers=True)[0],500)
        self.assertEqual(latest_checkpoint(folder)[0],1000)

    def test_truncated_metadata_skipped(self):
        folder=self.checkpoint(500);self.checkpoint(1000)
        (folder/'meta_001000.json').write_text('{')
        self.assertEqual(latest_checkpoint(folder,optimizers=True)[0],500)

    def test_no_checkpoint_raises(self):
        with self.assertRaises(FileNotFoundError):latest_checkpoint(self.root)

    def test_atomic_write_preserves_valid_file(self):
        path=self.root/'x.json';record(path,{'saved':1});before=path.read_bytes()
        with self.assertRaises(ValueError):record(path,{'bad':float('nan')})
        self.assertEqual(path.read_bytes(),before)

    def test_result_validation(self):
        path=self.root/'bpb.json';steps={'pico':500,'nano':5571}
        record(path,[self.row()]);self.assertEqual(len(bpb_rows(path,steps)),1)
        for row in [dict(self.row(),nominal_positions=262144),dict(self.row(),checkpoint_step=400),dict(self.row(),bpb=-1)]:
            record(path,[row])
            with self.assertRaises(ValueError):bpb_rows(path,steps)
        record(path,[self.row(),self.row()])
        with self.assertRaises(ValueError):bpb_rows(path,steps)

    def test_answer_resume_and_changed_prompt(self):
        path=self.root/'answers.json';q={'id':21,'prompt':'water?','criterion':'H2O'}
        record(path,[dict(q,completion='H2O',grade=None)])
        self.assertEqual(len(saved_answers(path,{'questions':[q]})),1)
        with self.assertRaises(ValueError):saved_answers(path,{'questions':[dict(q,prompt='changed')]})

    def test_train_command_uses_native_budget_and_resume(self):
        command=load_function('modal_app.py','train_command',{})
        fresh=command('test');resume=command('test',500)
        self.assertFalse(any(s.startswith('--eval-tokens=') for s in fresh))
        self.assertIn('--resume-from-step=500',resume)
        self.assertNotIn('--resume-from-step=500',fresh)
        self.assertNotIn('--num-iterations=12',fresh)

    def test_complete_validation_does_not_load_models(self):
        folder=self.root/RESULT_FOLDER;folder.mkdir()
        rows=[self.row(m,c) for m in ('pico','nano') for c in (512,2048)]
        record(folder/'bpb.json',rows);summaries=[]
        def forbidden(*a,**k):raise AssertionError('Model should not be loaded')
        env=dict(Path=Path,RESULT_FOLDER=RESULT_FOLDER,checkpoint_steps=lambda *a,**k:{'pico':500,'nano':5571},
                 bpb_rows=bpb_rows,CONTEXTS=(512,2048),load_saved=forbidden,
                 summarize=lambda *a:summaries.append(a))
        evaluate=load_function('evaluate.py','default_validation',env)
        self.assertEqual(evaluate(self.root),rows);self.assertEqual(len(summaries),1)

    def training_case(self, checkpoint_step=None, started=False):
        from types import SimpleNamespace as NS,ModuleType
        import sys
        config={'planned_updates':5571,'total_batch_size':1048576}
        record(self.root/'configuration.json',config)
        if started:(self.root/'console.log').write_text('started')
        module=ModuleType('A3part5.state')
        def latest(*a,**k):
            if checkpoint_step is None:raise FileNotFoundError('No checkpoint')
            return checkpoint_step, {'total_batch_size':1048576}
        module.latest_checkpoint=latest
        old=sys.modules.get('A3part5.state');sys.modules['A3part5.state']=module
        calls=[]
        env=dict(Path=lambda _:self.root.parent, json=json,record=record,
                 volume=NS(reload=lambda:None,commit=lambda:None),
                 train_command=load_function('modal_app.py','train_command',{}),
                 train_environment=lambda p:{},execute=lambda *a:calls.append(a))
        try:load_function('modal_app.py','training',env)(self.root.name)
        finally:
            if old is None:sys.modules.pop('A3part5.state',None)
            else:sys.modules['A3part5.state']=old
        return calls

    def test_training_resumes_existing_checkpoint(self):
        calls=self.training_case(500,started=True)
        self.assertIn('--resume-from-step=500',calls[0][0])

    def test_training_complete_does_not_restart(self):
        self.assertEqual(self.training_case(5571,started=True),[])

    def test_interrupted_training_without_checkpoint_stops(self):
        with self.assertRaises(RuntimeError):self.training_case(started=True)

    def test_core_is_measured_not_constant(self):
        from types import SimpleNamespace as NS,ModuleType
        import sys
        calls=[];module=ModuleType('scripts.base_eval')
        def scorer(*a,**k):calls.append(k);return {'core_metric':.123,'results':{'test':.5},'centered_results':{'test':.123}}
        module.evaluate_core=scorer
        old=sys.modules.get('scripts.base_eval');sys.modules['scripts.base_eval']=module
        try:
            env=dict(Path=Path,json=json,checkpoint_steps=lambda *a:{'nano':5571},record=record,
                     load_saved=lambda *a:(NS(eval=lambda:None,config=NS(sequence_len=2048)),None),
                     torch=NS(device=lambda x:x))
            core=load_function('evaluate.py','core',env);core(self.root);core(self.root)
            result=json.loads((self.root/'core-final.json').read_text())
            self.assertEqual(result['core_metric'],.123);self.assertEqual(calls,[{'max_per_task':-1}])
        finally:
            if old is None:sys.modules.pop('scripts.base_eval',None)
            else:sys.modules['scripts.base_eval']=old


if __name__=='__main__':unittest.main()
