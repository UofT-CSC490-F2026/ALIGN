"""Dependency-free RL shaping, reference serialization, and step-limit helpers."""
import math
import re
from arithmetic_reward import arithmetic_reward
from rewards import completion_reward

MODES = ('baseline', 'completion', 'arithmetic', 'combined')


def add_reward_arguments(parser):
    parser.add_argument('--reward-mode', choices=MODES, default='baseline')
    parser.add_argument('--completion-weight', type=float, default=.15)
    parser.add_argument('--arithmetic-weight', type=float, default=.20)
    parser.add_argument('--max-steps', type=int, default=None,
                        help='maximum optimizer updates in this invocation, including resumes; default unlimited')
    parser.add_argument('--output-tag', default=None, help='isolated checkpoint folder; defaults to model tag')
    return parser


def validate_reward_arguments(args):
    if not math.isfinite(args.completion_weight) or args.completion_weight < 0:
        raise ValueError('completion weight must be finite and nonnegative')
    if not math.isfinite(args.arithmetic_weight) or not 0 <= args.arithmetic_weight <= .20:
        raise ValueError('arithmetic weight must be in [0, .20]')
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError('max steps must be positive')
    if args.output_tag and not re.fullmatch(r'[A-Za-z0-9_-]+', args.output_tag):
        raise ValueError('output tag must be a folder name containing letters, digits, hyphens and underscores')
    if args.reward_mode != 'baseline' and not args.output_tag:
        raise ValueError('shaped runs require an isolated --output-tag')


def optimizer_steps(total_steps, resume_step=None, max_steps=None):
    """Cap updates without compressing the original learning-rate horizon."""
    if max_steps is not None and max_steps < 1:
        raise ValueError('max steps must be positive')
    first = 0 if resume_step is None else resume_step + 1
    stop = total_steps if max_steps is None else min(total_steps, first + max_steps)
    return range(first, stop)


def reference_text(conversation):
    """Reconstruct original GSM8K <<expr=result>> from trusted reference parts."""
    content = conversation['messages'][-1]['content']
    if isinstance(content, str):
        return content
    result = []
    i = 0
    while i < len(content):
        part = content[i]
        if part['type'] == 'python' and i + 1 < len(content) and content[i + 1]['type'] == 'python_output':
            result.append(f"<<{part['text']}={content[i + 1]['text']}>>")
            i += 2
        else:
            if part['type'] == 'text':
                result.append(part['text'])
            i += 1
    return ''.join(result)


def reward_components(correctness, reference, text, mode='baseline', completion_weight=.15, arithmetic_weight=.20):
    if mode not in MODES:
        raise ValueError('unknown reward mode')
    # Baseline bypasses both parsers and floating point addition entirely.
    completion = completion_reward(text, completion_weight) if mode in ('completion', 'combined') else 0.0
    arithmetic = arithmetic_reward(reference, text, arithmetic_weight) if mode in ('arithmetic', 'combined') else 0.0
    total = correctness if mode == 'baseline' else correctness + completion + arithmetic
    return dict(correctness=correctness, completion=completion, arithmetic=arithmetic, total=total)
