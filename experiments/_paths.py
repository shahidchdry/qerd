"""
Bootstrap `sys.path` so scripts in `experiments/` can be run either directly
(`python experiments/<script>.py`) or as modules (`python -m experiments.<script>`)
without requiring prior installation.
"""

import os
import sys

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EXPERIMENTS_DIR)

for _path in (REPO_ROOT, EXPERIMENTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)
