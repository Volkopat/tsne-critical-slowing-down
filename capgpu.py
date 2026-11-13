"""Run a script with a cap on this process's GPU memory, so several jobs can share one GPU safely.
Usage: python capgpu.py FRACTION path/to/script.py [args...]
"""

import os
import runpy
import sys

import torch

frac = float(sys.argv[1])
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(frac)
    print(f"capgpu: GPU memory capped at {frac:.2f} of the device", flush=True)
script = sys.argv[2]
sys.argv = sys.argv[2:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
