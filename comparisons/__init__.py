"""
comparisons — portfolio optimisation benchmark package.

This __init__.py automatically adds the parent directory (the folder that
contains fd_core.py, backtest_core.py, real_data_loader.py, nn_core.py)
to sys.path so those modules are importable from anywhere, including from
a Jupyter notebook kernel whose working directory might differ.
"""
import sys
from pathlib import Path

# Parent of comparisons/ = the "Claude Code" project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
