"""Run a nanochat script with the FoX model and its KV cache installed.

    python a4/run_script.py scripts.chat_sft --model-tag fox-d24 ...
    torchrun --standalone --nproc_per_node=8 a4/run_script.py scripts.chat_sft ...

nanochat's loader builds the stock GPT, so the A3 FoX checkpoint only loads after
fox_cache.install() has swapped in the FoX model. Under torchrun every process
runs this file, so every process installs it before the script starts.
"""
import os
import runpy
import sys

import fox_cache


def add_extra_data(path):
    """Append the conversations in a CustomJSON file to the first TaskMixture a script builds (its training mix)."""
    from tasks.common import TaskMixture
    from tasks.customjson import CustomJSON
    original, built = TaskMixture.__init__, []

    def init(self, tasks, **kwargs):
        if not built:
            tasks = [*tasks, CustomJSON(filepath=path)]
        built.append(self)
        original(self, tasks, **kwargs)

    TaskMixture.__init__ = init


def main(argv):
    module, *arguments = argv
    if arguments[:1] == ["--"]:  # nanochat's own usage line puts "--" before script arguments
        arguments = arguments[1:]
    fox_cache.install()
    if os.environ.get("EXTRA_DATA"):  # path to a CustomJSON file, set by modal_launch.py --extra-data
        add_extra_data(os.environ["EXTRA_DATA"])
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main(sys.argv[1:])
