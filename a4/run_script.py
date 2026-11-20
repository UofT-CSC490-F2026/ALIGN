"""Run a nanochat script with the FoX model and its KV cache installed.

    python a4/run_script.py scripts.chat_sft --model-tag fox-d24 ...
    torchrun --standalone --nproc_per_node=8 a4/run_script.py scripts.chat_sft ...

nanochat's loader builds the stock GPT, so the A3 FoX checkpoint only loads after
fox_cache.install() has swapped in the FoX model. Under torchrun every process
runs this file, so every process installs it before the script starts.
"""
import os
import random
import runpy
import sys

import fox_cache


def patch_mixture(extra_path, mix_seed):
    """Change the first TaskMixture a script builds, which is its training mix: append the conversations in a
    CustomJSON file, and/or shuffle it again with a different seed (nanochat always uses seed 42)."""
    from tasks.common import TaskMixture
    from tasks.customjson import CustomJSON
    original, built = TaskMixture.__init__, []

    def init(self, tasks, **kwargs):
        first = not built
        if first and extra_path:
            tasks = [*tasks, CustomJSON(filepath=extra_path)]
        built.append(self)
        original(self, tasks, **kwargs)
        if first and mix_seed is not None:
            random.Random(mix_seed).shuffle(self.index_map)

    TaskMixture.__init__ = init


def main(argv):
    module, *arguments = argv
    if arguments[:1] == ["--"]:  # nanochat's own usage line puts "--" before script arguments
        arguments = arguments[1:]
    fox_cache.install()
    extra, seed = os.environ.get("EXTRA_DATA"), os.environ.get("MIX_SEED")  # set by modal_launch.py
    if extra or seed:
        patch_mixture(extra, int(seed) if seed else None)
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main(sys.argv[1:])
