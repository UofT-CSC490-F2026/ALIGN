"""Local/free command preview. Does not import Modal, torch, or model code."""
import argparse
import shlex
from launcher_config import build_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script', default='rl')
    parser.add_argument('--mode', choices=('dry', 'smoke', 'full'), default='smoke')
    parser.add_argument('--run', required=True)
    parser.add_argument('--start', default='a4-sft-default')
    parser.add_argument('--reward-mode', default='baseline')
    parser.add_argument('--completion-weight', type=float, default=.15)
    parser.add_argument('--arithmetic-weight', type=float, default=.20)
    parser.add_argument('--extra', default='')
    args = parser.parse_args()
    print(shlex.join(build_command(args.script, args.mode, args.run, shlex.split(args.extra), start=args.start,
                                  reward_mode=args.reward_mode, completion_weight=args.completion_weight,
                                  arithmetic_weight=args.arithmetic_weight)))


if __name__ == '__main__':
    main()
