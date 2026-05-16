"""comparisons.core — re-exports the public API of the benchmark framework."""
# Ensure project root is on sys.path (in case core is imported before the
# top-level comparisons package has had a chance to run its __init__)
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
