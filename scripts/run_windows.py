"""Reuse the previous Windows interpreter with dependencies scoped to ProductSearch."""
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME",str(ROOT/"runtime/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.environ.setdefault("TORCH_HOME",str(ROOT/"runtime/torch-cache"))
os.environ.setdefault("CUDA_CACHE_PATH",str(ROOT/"runtime/cuda-cache"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",str(ROOT/"runtime/torch-compiler-cache"))
task_tmp=ROOT/"runtime/tmp"
task_tmp.mkdir(parents=True,exist_ok=True)
for name in ("TEMP","TMP","TMPDIR"):
    os.environ[name]=str(task_tmp)
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"runtime/matplotlib"))
sys.pycache_prefix=str(ROOT/"runtime/pycache-windows")
sys.path.insert(0,str(ROOT/"runtime/windows-packages"))
sys.path.insert(0,str(ROOT/"src"))
os.chdir(ROOT)
if len(sys.argv)<2:
    raise SystemExit("Usage: run_windows.py scripts/train_dense.py OR -m productsearch")
if sys.argv[1]=="-m":
    module=sys.argv[2]
    sys.argv=sys.argv[2:]
    runpy.run_module(module,run_name="__main__",alter_sys=True)
else:
    script=(ROOT/sys.argv[1]).resolve()
    if not script.is_relative_to(ROOT):
        raise SystemExit("Only ProductSearch scripts may be run")
    sys.argv=[str(script),*sys.argv[2:]]
    runpy.run_path(str(script),run_name="__main__")
