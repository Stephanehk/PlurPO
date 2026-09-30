"""Filesystem anchors for the PlurPO release.

Every path in the package and in `scripts/` is derived from `REPO_ROOT` (the
repository root), so all code runs identically from any working
directory. Nothing here touches the network or a GPU.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
TUNING_DIR = os.path.join(DATA_DIR, "tuning")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
VENDOR_DIR = os.path.join(REPO_ROOT, "vendor")
