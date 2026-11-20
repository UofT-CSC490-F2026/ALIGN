"""Print future billable commands as documentation; never import or invoke Modal."""
import argparse
import json
import shlex
from pathlib import Path
from launcher_config import build_command


def experiment_commands(manifest, mode='full'):
    if mode not in ('dry','smoke','full'):
        raise ValueError('invalid mode')
    result=[]
    for experiment in manifest['experiments']:
        if experiment['status'] != 'planned':
            continue
        name=experiment['run'] + (f'-{mode}' if mode != 'full' else '')
        settings=dict(manifest['training'])
        if mode != 'full':
            settings['eval_examples']=8
            settings['save_every']=100000
        if manifest['sft_model_step'] is not None:
            settings['model_step']=manifest['sft_model_step']
        extra=[str(value) for key,value in settings.items() for value in ('--'+key.replace('_','-'),value)]
        # Validate the same local command constructor used by the real launcher.
        build_command('rl',mode,name,extra,start=experiment['start'],reward_mode=experiment['reward_mode'],
                      completion_weight=experiment['completion_weight'],arithmetic_weight=experiment['arithmetic_weight'])
        # Windows/PowerShell-friendly quoting: flags and numeric values are fixed
        # by this checked-in manifest. --extra is one string argument.
        command=(f'modal run a4/modal_launch.py --script rl --mode {mode} --run {name} '
                 f"--start {experiment['start']} --reward-mode {experiment['reward_mode']} "
                 f"--completion-weight {experiment['completion_weight']} --arithmetic-weight {experiment['arithmetic_weight']} "
                 f'--extra "{shlex.join(extra)}"')
        result.append(dict(id=experiment['id'],optional=experiment.get('optional',False),command=command))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path(__file__).with_name('part4_experiments.json'))
    parser.add_argument('--mode',choices=('dry','smoke','full'),default='smoke')
    args=parser.parse_args()
    for entry in experiment_commands(json.loads(args.manifest.read_text(encoding='utf-8')),args.mode):
        print(f"# DOCUMENTATION ONLY - BILLABLE IF EXECUTED ({entry['id']}{', optional' if entry['optional'] else ''})")
        print(entry['command'])


if __name__=='__main__':
    main()
